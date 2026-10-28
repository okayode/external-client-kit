import argparse
import json
import os
from pathlib import Path

import torch
import torch.optim as optim
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

from model import SybilNLSTClassifier
from train_utils import (
    set_seed,
    get_class_weights,
    make_criterion,
    run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVFlare client for the UPMC NLST experiment."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    config_path = args.config or os.environ.get("NLST_CLIENT_CONFIG")
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for key in (
            "data_root",
            "batch_size",
            "val_fraction",
            "seed",
            "num_workers",
            "local_epochs",
            "lr",
            "weight_decay",
            "threshold",
        ):
            if key in config:
                current = getattr(args, key)
                default = parser.get_default(key)
                if current == default:
                    setattr(args, key, config[key])

        if config.get("amp") is False:
            args.no_amp = True

    if args.data_root:
        os.environ["NLST_DATA_ROOT"] = str(
            Path(args.data_root).expanduser().resolve()
        )

    if args.local_epochs < 1:
        parser.error("--local_epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val_fraction must be between 0 and 1.")

    return args


def main(args):
    # Import after applying NLST_DATA_ROOT because data.py reads this
    # environment variable when it is imported.
    from data import make_loaders

    set_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    amp = not args.no_amp and device.type == "cuda"

    print(f"Device: {device}")
    print(f"Mixed precision: {amp}")
    print(f"Local epochs per round: {args.local_epochs}")

    train_loader, val_loader, _test_loader = make_loaders(
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Derive class weights from the local training labels only.
    train_labels = train_loader.dataset.df["label"].tolist()
    class_weights = get_class_weights(train_labels, device)
    criterion = make_criterion(class_weights)

    model = SybilNLSTClassifier().to(device)

    flare.init()
    site_name = flare.get_site_name()
    print(f"NVFlare site: {site_name}")

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            continue

        round_num = input_model.current_round
        print(f"\n[{site_name}] Starting round {round_num}")

        # The server must provide a complete, compatible global model.
        model.load_state_dict(input_model.params, strict=True)
        model.to(device)

        # Preserve the received global state on CPU for delta computation.
        global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        # Initialize a fresh local optimizer for this round.
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        steps = 0
        for epoch in range(args.local_epochs):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                amp=amp,
            )
            steps += len(train_loader)

            print(
                f"[{site_name}] Round {round_num}, "
                f"epoch {epoch + 1}/{args.local_epochs}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"accuracy={train_metrics['accuracy']:.4f}"
            )

        # Validation only: the official test loader is not used here.
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            amp=amp,
            threshold=args.threshold,
        )

        print(
            f"[{site_name}] Round {round_num} validation: "
            f"loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"ROC-AUC={val_metrics['roc_auc']:.4f}, "
            f"PR-AUC={val_metrics['pr_auc']:.4f}"
        )

        # Return local-minus-global deltas for floating-point state only.
        # Integer buffers, such as num_batches_tracked, are omitted.
        local_state = model.state_dict()
        model_diff = {}

        for name, global_tensor in global_state.items():
            local_tensor = local_state[name].detach().cpu()

            if torch.is_floating_point(global_tensor):
                delta = local_tensor - global_tensor
                if not torch.isfinite(delta).all():
                    raise ValueError(
                        f"Non-finite model delta detected: {name}"
                    )
                model_diff[name] = delta

        if not model_diff:
            raise RuntimeError("No floating-point model deltas were produced.")

        metrics_to_send = {
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_sensitivity": float(val_metrics["sensitivity"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_f1": float(val_metrics["f1"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
            "val_pr_auc": float(val_metrics["pr_auc"]),
        }

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics=metrics_to_send,
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        flare.send(output_model)
        print(f"[{site_name}] Sent update for round {round_num}")


if __name__ == "__main__":
    main(parse_args())
