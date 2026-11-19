
import argparse
import json
import os
from pathlib import Path

import torch
import nvflare.client as flare

from nvflare.client.api import get_task_name
from nvflare.app_common.abstract.fl_model import ParamsType

from model import SybilNLSTClassifier

from train_utils import (
    set_seed,
    get_class_weights,
    make_criterion,
    make_optimizer,
    get_round_learning_rates,
    run_epoch,
    predict_with_pids,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    WARMUP_EPOCHS,
    MAX_GRAD_NORM,
)


# ============================================================
# SITE CONFIGURATION
# ============================================================

DEFAULT_DATA_ROOT = str(
    Path.home() / "NLST_UPMC"
)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="NVFlare client for the UPMC NLST experiment."
    )

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)

    # Central-aligned data loading.
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # Federated training.
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--total_rounds", type=int, default=12)

    # Centralized-reference optimizer settings.
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=LR_BACKBONE,
    )

    parser.add_argument(
        "--lr_head",
        type=float,
        default=LR_HEAD,
    )

    # Legacy option for compatibility with older configuration files.
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=WARMUP_EPOCHS,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=MAX_GRAD_NORM,
    )

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    # Preserve UPMC's existing configuration fallback.
    config_path = (
        args.config
        or os.environ.get("NLST_CLIENT_CONFIG")
        or os.environ.get("CIFAR10_CLIENT_CONFIG")
    )

    if config_path:

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config_keys = (
            "data_root",
            "batch_size",
            "val_fraction",
            "seed",
            "num_workers",
            "local_epochs",
            "total_rounds",
            "lr_backbone",
            "lr_head",
            "lr",
            "weight_decay",
            "warmup_epochs",
            "max_grad_norm",
            "threshold",
        )

        for key in config_keys:

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

    # Backward compatibility with the previous single-LR setting.
    if args.lr is not None:

        print(
            "WARNING: Legacy --lr supplied. "
            "It overrides both backbone and head learning rates."
        )

        args.lr_backbone = args.lr
        args.lr_head = args.lr

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    if args.local_epochs < 1:
        parser.error("--local_epochs must be at least 1.")

    if args.total_rounds < 1:
        parser.error("--total_rounds must be at least 1.")

    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative.")

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val_fraction must be between 0 and 1.")

    if args.warmup_epochs < 0:
        parser.error("--warmup_epochs cannot be negative.")

    total_training_epochs = (
        args.total_rounds * args.local_epochs
    )

    if args.warmup_epochs >= total_training_epochs:
        parser.error(
            "--warmup_epochs must be less than total training epochs."
        )

    if args.lr_backbone <= 0 or args.lr_head <= 0:
        parser.error("Learning rates must be positive.")

    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative.")

    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive.")

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    return args


# ============================================================
# GLOBAL EVALUATION CONFIGURATION
# ============================================================

GLOBAL_EVAL_THRESHOLD = 0.17259


# ============================================================
# RECEIVED MODEL LOADING
# ============================================================

def _load_received_model(model, params):
    """
    Load global model parameters received from NVFlare.

    Restore original tensor shapes using the local model state_dict.
    This is necessary because HE encryption flattens model parameters.
    """

    if params is None:
        raise RuntimeError(
            "Received model parameters are None."
        )

    if not isinstance(params, dict):
        raise TypeError(
            "Expected received model parameters to be a dictionary."
        )

    expected_state_dict = model.state_dict()

    expected_keys = set(expected_state_dict.keys())
    received_keys = set(params.keys())

    missing_keys = expected_keys - received_keys
    unexpected_keys = received_keys - expected_keys

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Received model parameter keys do not match the local model. "
            f"Missing keys: {sorted(missing_keys)}. "
            f"Unexpected keys: {sorted(unexpected_keys)}."
        )

    global_state_dict = {}

    for name, value in params.items():

        expected_tensor = expected_state_dict[name]

        if torch.is_tensor(value):
            tensor = value.detach()
        else:
            tensor = torch.as_tensor(value)

        if tensor.numel() != expected_tensor.numel():
            raise RuntimeError(
                f"Parameter size mismatch for {name}: "
                f"received {tensor.numel()} values, "
                f"expected {expected_tensor.numel()} values. "
                f"Received shape: {tuple(tensor.shape)}, "
                f"expected shape: {tuple(expected_tensor.shape)}."
            )

        if tensor.shape != expected_tensor.shape:
            tensor = tensor.reshape(expected_tensor.shape)

        global_state_dict[name] = tensor.to(
            device=expected_tensor.device,
            dtype=expected_tensor.dtype,
        )

    model.load_state_dict(
        global_state_dict,
        strict=True,
    )

# def _load_received_model(model, params):
#     """
#     Load global model parameters received from NVFlare.

#     Convert NumPy arrays to PyTorch tensors when necessary.
#     """

#     if params is None:
#         raise RuntimeError(
#             "Received model parameters are None."
#         )

#     if not isinstance(params, dict):
#         raise TypeError(
#             "Expected received model parameters to be a dictionary."
#         )

#     global_state_dict = {
#         name: torch.as_tensor(value.copy())
#         if not torch.is_tensor(value)
#         else value
#         for name, value in params.items()
#     }

#     model.load_state_dict(
#         global_state_dict,
#         strict=True,
#     )


# ============================================================
# LOCAL GLOBAL-MODEL EVALUATION OUTPUT
# ============================================================

def _save_global_evaluation(
    output_dir,
    site_name,
    round_num,
    metrics,
    prediction_rows,
):
    """
    Write held-out metrics and patient-level predictions locally only.
    """

    import csv

    output_dir = Path(output_dir).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_fields = [
        "site_name",
        "round",
        "threshold",
        "n",
        "positive_count",
        "negative_count",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
    ]

    metrics_path = output_dir / "round_metrics.csv"

    metric_row = {
        "site_name": site_name,
        "round": int(round_num),
        **metrics,
    }

    write_header = not metrics_path.exists()

    with metrics_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=metric_fields,
        )

        if write_header:
            writer.writeheader()

        writer.writerow({
            key: metric_row.get(key)
            for key in metric_fields
        })

    predictions_path = (
        output_dir
        / f"predictions_round_{int(round_num)}.csv"
    )

    prediction_fields = [
        "pid",
        "y_true",
        "p_positive",
        "y_pred",
    ]

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=prediction_fields,
        )

        writer.writeheader()
        writer.writerows(prediction_rows)

    print(
        f"[{site_name}] Local metrics: {metrics_path}",
        flush=True,
    )

    print(
        f"[{site_name}] Local predictions: {predictions_path}",
        flush=True,
    )


# ============================================================
# MAIN NVFLARE CLIENT
# ============================================================

def main(args):

    # Import after applying NLST_DATA_ROOT.
    from data import make_loaders

    set_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    amp = not args.no_amp and device.type == "cuda"

    print(f"Device: {device}", flush=True)

    print(
        f"Mixed precision: {'bfloat16' if amp else 'disabled'}",
        flush=True,
    )

    print(
        f"Batch size: {args.batch_size}",
        flush=True,
    )

    print(
        f"Workers: {args.num_workers}",
        flush=True,
    )

    print(
        f"Local epochs per round: {args.local_epochs}",
        flush=True,
    )

    print(
        f"Total federated rounds: {args.total_rounds}",
        flush=True,
    )

    print(
        f"Backbone LR: {args.lr_backbone:.3e}",
        flush=True,
    )

    print(
        f"Head LR: {args.lr_head:.3e}",
        flush=True,
    )

    print(
        f"Weight decay: {args.weight_decay}",
        flush=True,
    )

    print(
        f"Label smoothing: {LABEL_SMOOTHING}",
        flush=True,
    )

    print(
        f"Warmup epochs: {args.warmup_epochs}",
        flush=True,
    )

    print(
        f"Gradient clipping: {args.max_grad_norm}",
        flush=True,
    )

    # Preserve UPMC's data root and validation checks.
    data_root = Path(
        os.environ.get(
            "NLST_DATA_ROOT",
            DEFAULT_DATA_ROOT,
        )
    ).expanduser().resolve()

    print(
        f"NLST data root: {data_root}",
        flush=True,
    )

    missing_csvs = [
        name
        for name in (
            "NLST-T0-pids-train.csv",
            "NLST-T0-pids-test.csv",
        )
        if not (data_root / name).is_file()
    ]

    if missing_csvs:
        raise FileNotFoundError(
            f"NLST data root {data_root} is missing "
            f"required CSV files: {missing_csvs}"
        )

    train_loader, val_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Class weights are computed from local training labels only.
    train_labels = (
        train_loader.dataset.df["label"].tolist()
    )

    class_weights = get_class_weights(
        train_labels,
        device,
    )

    criterion = make_criterion(
        class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    print(
        f"Local class weights: {class_weights.detach().cpu().tolist()}",
        flush=True,
    )

    model = SybilNLSTClassifier().to(device)

    # Preserve NVFlare initialization.
    flare.init()

    site_name = flare.get_site_name()

    print(
        f"NVFlare site: {site_name}",
        flush=True,
    )

    default_eval_dir = (
        data_root / "global_model_evaluation"
    )

    eval_output_dir = Path(
        os.environ.get(
            "NLST_EVAL_OUTPUT_DIR",
            str(default_eval_dir),
        )
    ).expanduser().resolve()

    # The first observed training round anchors the schedule.
    # This accommodates either zero-based or one-based NVFlare rounds.
    first_training_round = None

    while flare.is_running():

        # Preserve NVFlare model reception.
        input_model = flare.receive()

        if input_model is None:
            continue

        task_name = get_task_name()

        round_num = input_model.current_round

        # ----------------------------------------------------
        # GLOBAL MODEL EVALUATION TASK
        # ----------------------------------------------------

        if task_name == "evaluate_global":

            if round_num is None or input_model.params is None:
                raise RuntimeError(
                    "Global evaluation task is missing round "
                    "or model parameters."
                )

            print(
                f"\n[{site_name}] Evaluating aggregated global model "
                f"for round {round_num}",
                flush=True,
            )

            _load_received_model(
                model,
                input_model.params,
            )

            model.to(device)

            eval_metrics, prediction_rows = predict_with_pids(
                model=model,
                loader=test_loader,
                device=device,
                amp=amp,
                threshold=GLOBAL_EVAL_THRESHOLD,
                bootstrap_samples=1000,
                bootstrap_seed=0,
            )

            _save_global_evaluation(
                eval_output_dir,
                site_name,
                round_num,
                eval_metrics,
                prediction_rows,
            )

            print(
                "[{}] Round {} held-out test: "
                "n={}, ROC-AUC={:.4f}, PR-AUC={:.4f}, "
                "balanced accuracy={:.4f}, "
                "ROC-AUC 95% CI=({:.4f}, {:.4f}), "
                "sensitivity={:.4f}, specificity={:.4f}".format(
                    site_name,
                    round_num,
                    eval_metrics["n"],
                    eval_metrics["roc_auc"],
                    eval_metrics["pr_auc"],
                    eval_metrics["balanced_accuracy"],
                    eval_metrics["roc_auc_ci_low"],
                    eval_metrics["roc_auc_ci_high"],
                    eval_metrics["sensitivity"],
                    eval_metrics["specificity"],
                ),
                flush=True,
            )

            # print(
            #     f"[{site_name}] Round {round_num} held-out test: "
            #     f"n={eval_metrics['n']}, "
            #     f"ROC-AUC={eval_metrics['roc_auc']:.4f}, "
            #     f"PR-AUC={eval_metrics['pr_auc']:.4f}, "
            #     f"balanced accuracy="
            #     f"{eval_metrics['balanced_accuracy']:.4f}, "
            #     f"ROC-AUC 95% CI="
            #     f"({eval_metrics['roc_auc_ci_low']:.4f}, "
            #     f"{eval_metrics['roc_auc_ci_high']:.4f}),"
            #     f"sensitivity={eval_metrics['sensitivity']:.4f}, "
            #     f"specificity={eval_metrics['specificity']:.4f}, "
            #     flush=True,
            # )

            # Only acknowledgement and site identifier leave this client.
            flare.send(
                flare.FLModel(
                    params=None,
                    metrics={
                        "evaluation_complete": 1.0
                    },
                    meta={
                        "client_name": site_name
                    },
                )
            )

            print(
                f"[{site_name}] Evaluation acknowledgement sent.",
                flush=True,
            )

            continue

        # ----------------------------------------------------
        # NORMAL FEDERATED TRAINING TASK
        # ----------------------------------------------------

        if round_num is None:
            raise RuntimeError(
                "Normal FL training task has no current_round."
            )

        round_num = int(round_num)

        print(
            f"\n[{site_name}] Starting round {round_num}",
            flush=True,
        )

        # Load the complete global model received from the server.
        _load_received_model(
            model,
            input_model.params,
        )

        model.to(device)

        # Preserve the global model state on CPU for delta computation.
        global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        # Establish the first actual training round.
        if first_training_round is None:
            first_training_round = round_num

        schedule_round = (
            round_num - first_training_round + 1
        )

        if not 1 <= schedule_round <= args.total_rounds:
            raise RuntimeError(
                f"Invalid schedule round {schedule_round}; "
                f"expected 1 through {args.total_rounds}."
            )

        total_training_epochs = (
            args.total_rounds * args.local_epochs
        )

        first_training_epoch = (
            (schedule_round - 1) * args.local_epochs + 1
        )

        # Calculate the starting learning rates for this FL round.
        initial_rates = get_round_learning_rates(
            first_training_epoch,
            total_rounds=total_training_epochs,
            warmup_rounds=args.warmup_epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
        )

        # Preserve the existing federated behavior:
        # create a fresh optimizer for every FL round.
        optimizer = make_optimizer(
            model,
            lr_backbone=initial_rates["backbone"],
            lr_head=initial_rates["head"],
            weight_decay=args.weight_decay,
        )

        steps = 0

        for epoch in range(args.local_epochs):

            global_epoch = (
                first_training_epoch + epoch
            )

            # Apply the centralized-reference cosine LR schedule.
            epoch_rates = get_round_learning_rates(
                global_epoch,
                total_rounds=total_training_epochs,
                warmup_rounds=args.warmup_epochs,
                lr_backbone=args.lr_backbone,
                lr_head=args.lr_head,
            )

            for group in optimizer.param_groups:
                group["lr"] = epoch_rates[group["name"]]

            # Freeze backbone during the first two global epochs.
            backbone_frozen = (
                global_epoch <= args.warmup_epochs
            )

            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                amp=amp,
                max_grad_norm=args.max_grad_norm,
                backbone_frozen=backbone_frozen,
            )

            steps += len(train_loader)

            print(
                f"[{site_name}] Round {round_num}, "
                f"local epoch {epoch + 1}/{args.local_epochs}, "
                f"global epoch {global_epoch}/"
                f"{total_training_epochs}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"accuracy={train_metrics['accuracy']:.4f}, "
                f"backbone_frozen={backbone_frozen}, "
                f"backbone_lr={epoch_rates['backbone']:.3e}, "
                f"head_lr={epoch_rates['head']:.3e}",
                flush=True,
            )

        # ----------------------------------------------------
        # LOCAL VALIDATION
        # ----------------------------------------------------

        # The official held-out test loader is not used here.
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
            "[{}] Round {} validation: "
            "loss={:.4f}, accuracy={:.4f}, ROC-AUC={:.4f}, "
            "PR-AUC={:.4f}, sensitivity={:.4f}, "
            "specificity={:.4f}".format(
                site_name,
                round_num,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["roc_auc"],
                val_metrics["pr_auc"],
                val_metrics["sensitivity"],
                val_metrics["specificity"],
            ),
            flush=True,
        )

        # print(
        #     f"[{site_name}] Round {round_num} validation: "
        #     f"loss={val_metrics['loss']:.4f}, "
        #     f"accuracy={val_metrics['accuracy']:.4f}, "
        #     f"ROC-AUC={val_metrics['roc_auc']:.4f}, "
        #     f"PR-AUC={val_metrics['pr_auc']:.4f},"
        #     f"sensitivity={val_metrics['sensitivity']:.4f}, "
        #     f"specificity={val_metrics['specificity']:.4f}",
        #     flush=True,
        # )

        # ----------------------------------------------------
        # MODEL DELTA CALCULATION
        # ----------------------------------------------------

        # Return local-minus-global floating-point deltas.
        # Integer buffers remain omitted.
        local_state = model.state_dict()

        model_diff = {}

        for name, global_tensor in global_state.items():

            local_tensor = (
                local_state[name].detach().cpu()
            )

            if torch.is_floating_point(global_tensor):

                delta = local_tensor - global_tensor

                if not torch.isfinite(delta).all():
                    raise ValueError(
                        f"Non-finite model delta detected: {name}"
                    )

                model_diff[name] = delta

        if not model_diff:
            raise RuntimeError(
                "No floating-point model deltas were produced."
            )

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

        # Preserve NVFlare DIFF communication.
        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics=metrics_to_send,
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps
            },
        )

        flare.send(output_model)

        print(
            f"[{site_name}] Sent update for round {round_num}",
            flush=True,
        )


if __name__ == "__main__":
    main(parse_args())
