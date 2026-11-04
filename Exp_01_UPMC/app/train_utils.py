
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CENTRALIZED-REFERENCE TRAINING SETTINGS
# ============================================================

LR_BACKBONE = 2e-6
LR_HEAD = 5e-4
WEIGHT_DECAY = 0.005
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 2
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 4
MAX_GRAD_NORM = 5.0
COSINE_ETA_MIN = 1e-7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CLASS WEIGHTS AND LOSS
# ============================================================

def get_class_weights(labels, device):
    """
    Compute inverse-frequency class weights from local training labels only.

    This is equivalent to the centralized reference:
        n_train / (2 * class_count)

    Returns [weight_for_class_0, weight_for_class_1].
    """
    labels = torch.as_tensor(
        labels,
        dtype=torch.long
    ).view(-1)

    if labels.numel() == 0:
        raise ValueError(
            "Cannot compute class weights from empty labels."
        )

    if torch.any((labels < 0) | (labels > 1)):
        raise ValueError(
            "Labels must contain only binary values 0 and 1."
        )

    counts = torch.bincount(
        labels,
        minlength=2
    ).float()

    if torch.any(counts == 0):
        raise ValueError(
            "Both classes must be present in training labels; "
            f"counts={counts.tolist()}"
        )

    weights = labels.numel() / (2.0 * counts)

    return weights.to(device)


def make_criterion(
    class_weights=None,
    label_smoothing: float = LABEL_SMOOTHING,
):
    """Create the centralized-reference weighted cross-entropy loss."""

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            "label_smoothing must be in [0, 1)."
        )

    if class_weights is None:
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    class_weights = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
    )

    if class_weights.numel() != 2:
        raise ValueError(
            "class_weights must contain exactly two values."
        )

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


# ============================================================
# MODEL MODE AND OPTIMIZER HELPERS
# ============================================================

def set_backbone_trainable(model, trainable: bool) -> None:
    """Enable or disable gradients for the Sybil backbone."""

    if not hasattr(model, "backbone"):
        raise AttributeError(
            "Model must expose a 'backbone' module."
        )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = bool(trainable)


def set_training_mode(
    model,
    backbone_frozen: bool = False,
) -> None:
    """
    Apply the centralized reference's training-mode behavior.

    During warmup, the backbone stays in eval mode while the head trains.
    After warmup, the full model trains but every BatchNorm3d stays in eval.
    """

    model.train()

    if backbone_frozen:
        model.backbone.eval()
        model.head.train()
        return

    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()


def make_optimizer(
    model,
    lr_backbone: float = LR_BACKBONE,
    lr_head: float = LR_HEAD,
    weight_decay: float = WEIGHT_DECAY,
):
    """Create AdamW with separate backbone and classification-head groups."""

    if not hasattr(model, "backbone") or not hasattr(model, "head"):
        raise AttributeError(
            "Model must expose both 'backbone' and 'head' modules."
        )

    backbone_parameters = list(
        model.backbone.parameters()
    )

    head_parameters = list(
        model.head.parameters()
    )

    if not backbone_parameters or not head_parameters:
        raise ValueError(
            "Backbone and head must both contain parameters."
        )

    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(lr_backbone),
                "name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": float(lr_head),
                "name": "head",
            },
        ],
        weight_decay=float(weight_decay),
    )


def get_round_learning_rates(
    round_number: int,
    total_rounds: int = MAX_EPOCHS,
    warmup_rounds: int = WARMUP_EPOCHS,
    lr_backbone: float = LR_BACKBONE,
    lr_head: float = LR_HEAD,
    eta_min: float = COSINE_ETA_MIN,
):
    """
    Return per-group learning rates for a 1-based federated round.

    The first two rounds use the base rates. Cosine decay begins at round 3.
    This reproduces the centralized schedule's round-level LR progression
    when the client creates a fresh optimizer at each server round.
    """

    round_number = int(round_number)
    total_rounds = int(total_rounds)
    warmup_rounds = int(warmup_rounds)

    if total_rounds < 1:
        raise ValueError(
            "total_rounds must be >= 1."
        )

    if warmup_rounds < 0 or warmup_rounds >= total_rounds:
        raise ValueError(
            "warmup_rounds must be >= 0 and < total_rounds."
        )

    if round_number < 1 or round_number > total_rounds:
        raise ValueError(
            f"round_number must be between 1 and {total_rounds}; "
            f"received {round_number}."
        )

    if round_number <= warmup_rounds + 1:
        # In the reference, scheduler.step() occurs after each post-warmup
        # epoch. Therefore the first post-warmup epoch starts at the base LR.
        factor = 1.0

    else:
        steps = round_number - warmup_rounds - 1
        t_max = total_rounds - warmup_rounds

        factor = 0.5 * (
            1.0 + math.cos(math.pi * steps / t_max)
        )

    return {
        "backbone": float(
            eta_min + (lr_backbone - eta_min) * factor
        ),
        "head": float(
            eta_min + (lr_head - eta_min) * factor
        ),
    }


# ============================================================
# MIXED PRECISION AND BATCH VALIDATION
# ============================================================

def _autocast_context(device, enabled=True):
    """Use bfloat16 autocast on CUDA, matching the centralized reference."""

    device = torch.device(device)

    use_amp = bool(
        enabled and device.type == "cuda"
    )

    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    )


def _batch_to_device(batch, device):

    x, y = batch

    x = x.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    y = y.to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    ).view(-1)

    if x.ndim != 5:
        raise ValueError(
            f"Expected input [B,C,D,H,W], got {tuple(x.shape)}"
        )

    if x.shape[1] != 3:
        raise ValueError(
            f"Expected 3 input channels, got {x.shape[1]}"
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Input batch contains non-finite values."
        )

    if torch.any((y < 0) | (y > 1)):
        raise ValueError(
            "Labels must be binary values 0 or 1."
        )

    return x, y


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """Calculate threshold-based metrics and ranking metrics."""

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Labels and probabilities have different lengths."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics for an empty dataset."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp)
        else 0.0
    )

    sensitivity = (
        float(tp / (tp + fn))
        if (tp + fn)
        else 0.0
    )

    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "n": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "sensitivity": sensitivity,

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "specificity": specificity,

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": (
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            )
            if has_both_classes
            else float("nan")
        ),

        "pr_auc": (
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            if has_both_classes
            else float("nan")
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# ONE TRAINING OR EVALUATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    amp=True,
    threshold=0.5,
    max_grad_norm: Optional[float] = MAX_GRAD_NORM,
    backbone_frozen: bool = False,
):
    """
    Run one training or evaluation epoch.

    Training uses CUDA bfloat16 autocast (without GradScaler), gradient
    clipping at 5.0 by default, and centralized BatchNorm3d handling.

    Set backbone_frozen=True during the first two federated rounds.
    """

    device = torch.device(device)

    training = optimizer is not None

    if training:

        set_backbone_trainable(
            model,
            trainable=not backbone_frozen,
        )

        set_training_mode(
            model,
            backbone_frozen=backbone_frozen,
        )

    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0

    all_labels = []
    all_probabilities = []

    for batch in loader:

        x, y = _batch_to_device(
            batch,
            device,
        )

        batch_size = int(y.shape[0])

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        grad_context = (
            torch.enable_grad()
            if training
            else torch.no_grad()
        )

        with grad_context:

            with _autocast_context(
                device,
                enabled=amp,
            ):

                logits = model(x)

                if (
                    logits.ndim != 2
                    or logits.shape != (batch_size, 2)
                ):
                    raise ValueError(
                        "Model must return logits of shape [B, 2]; "
                        f"received {tuple(logits.shape)}"
                    )

                loss = criterion(
                    logits,
                    y,
                )

            if training:

                loss.backward()

                if max_grad_norm is not None:

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )

                optimizer.step()

        probabilities = torch.softmax(
            logits.detach().float(),
            dim=1,
        )[:, 1]

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )

        total_examples += batch_size

        all_labels.extend(
            y.detach().cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

    if total_examples == 0:
        raise ValueError(
            "DataLoader produced no batches."
        )

    metrics = compute_metrics(
        all_labels,
        all_probabilities,
        threshold=threshold,
    )

    metrics["loss"] = float(
        total_loss / total_examples
    )

    return metrics


# ============================================================
# CHECKPOINTING AND RESUME SUPPORT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    patience_counter,
    best_val_metrics=None,
    seed=42,
    scheduler=None,
    best_val_auc=None,
):
    """Save model, optimizer, scheduler, and RNG state for resuming."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": int(epoch),

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "best_val_loss": float(best_val_loss),

        "best_val_auc": (
            float(best_val_auc)
            if best_val_auc is not None
            else None
        ),

        "patience_counter": int(patience_counter),

        "best_val_metrics": best_val_metrics,

        "seed": int(seed),

        "torch_rng_state": torch.get_rng_state(),

        "numpy_rng_state": np.random.get_state(),

        "python_rng_state": random.getstate(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
    scheduler=None,
):
    """Restore model, optimizer, scheduler, and RNG states."""

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if "torch_rng_state" in checkpoint:

        torch.set_rng_state(
            checkpoint["torch_rng_state"]
            .detach()
            .cpu()
            .to(torch.uint8)
        )

    if "numpy_rng_state" in checkpoint:

        np.random.set_state(
            checkpoint["numpy_rng_state"]
        )

    if "python_rng_state" in checkpoint:

        random.setstate(
            checkpoint["python_rng_state"]
        )

    if (
        torch.cuda.is_available()
        and "cuda_rng_state_all" in checkpoint
    ):

        cuda_states = [
            state.detach().cpu().to(torch.uint8)
            for state in checkpoint["cuda_rng_state_all"]
        ]

        for index, state in enumerate(
            cuda_states[:torch.cuda.device_count()]
        ):

            torch.cuda.set_rng_state(
                state,
                device=index,
            )

    return checkpoint


# ============================================================
# OPTIONAL CENTRALIZED-STYLE FIT ROUTINE
# ============================================================

def fit(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    epochs=MAX_EPOCHS,
    learning_rate=None,
    weight_decay=WEIGHT_DECAY,
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=0.0,
    class_weights=None,
    amp=True,
    threshold=0.5,
    seed=42,
    resume_from=None,
    lr_backbone=LR_BACKBONE,
    lr_head=LR_HEAD,
    warmup_epochs=WARMUP_EPOCHS,
    label_smoothing=LABEL_SMOOTHING,
    max_grad_norm=MAX_GRAD_NORM,
):
    """
    Optional centralized-style local fit routine.

    The actual NVFlare client performs its own server-round training loop;
    this helper is retained for standalone experiments and testing.

    Selects the best checkpoint using validation ROC-AUC, freezes the
    backbone for the first two epochs, and applies cosine decay after warmup.
    """

    if epochs < 1:
        raise ValueError(
            "epochs must be >= 1."
        )

    if patience < 1:
        raise ValueError(
            "patience must be >= 1."
        )

    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError(
            "warmup_epochs must be >= 0 and < epochs."
        )

    # Backward compatibility with callers that supplied one learning rate.
    if learning_rate is not None:
        lr_backbone = float(learning_rate)
        lr_head = float(learning_rate)

    set_seed(seed)

    device = torch.device(device)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.to(device)

    criterion = make_criterion(
        class_weights.to(device)
        if isinstance(class_weights, torch.Tensor)
        else class_weights,
        label_smoothing=label_smoothing,
    )

    optimizer = make_optimizer(
        model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=COSINE_ETA_MIN,
    )

    best_path = output_dir / "best_model.pt"

    last_path = output_dir / "last_checkpoint.pt"

    history_path = output_dir / "history.json"

    history = []

    start_epoch = 1

    best_val_auc = float("-inf")

    best_val_loss = float("inf")

    patience_counter = 0

    best_val_metrics = None

    if resume_from is not None:

        state = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
        )

        start_epoch = int(state["epoch"]) + 1

        best_val_loss = float(
            state.get(
                "best_val_loss",
                float("inf"),
            )
        )

        patience_counter = int(
            state.get(
                "patience_counter",
                0,
            )
        )

        best_val_metrics = state.get(
            "best_val_metrics"
        )

        saved_auc = state.get(
            "best_val_auc"
        )

        if saved_auc is None and best_val_metrics is not None:
            saved_auc = best_val_metrics.get("roc_auc")

        if saved_auc is not None:
            best_val_auc = float(saved_auc)

        if history_path.is_file():

            history = json.loads(
                history_path.read_text()
            )

        print(
            f"Resuming at epoch {start_epoch}"
        )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        backbone_frozen = (
            epoch <= warmup_epochs
        )

        set_backbone_trainable(
            model,
            trainable=not backbone_frozen,
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp=amp,
            threshold=threshold,
            max_grad_norm=max_grad_norm,
            backbone_frozen=backbone_frozen,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            amp=amp,
            threshold=threshold,
        )

        record = {
            "epoch": epoch,

            "backbone_frozen": backbone_frozen,

            "lr_backbone": float(
                optimizer.param_groups[0]["lr"]
            ),

            "lr_head": float(
                optimizer.param_groups[1]["lr"]
            ),

            "train": train_metrics,

            "validation": val_metrics,
        }

        history.append(record)

        val_auc = float(
            val_metrics["roc_auc"]
        )

        improved = (
            math.isfinite(val_auc)
            and val_auc > best_val_auc + min_delta
        )

        if improved:

            best_val_auc = val_auc

            best_val_loss = float(
                val_metrics["loss"]
            )

            patience_counter = 0

            best_val_metrics = val_metrics

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict": model.state_dict(),

                    "best_val_loss": best_val_loss,

                    "best_val_auc": best_val_auc,

                    "best_val_metrics": best_val_metrics,

                    "seed": seed,
                },
                best_path,
            )

        else:

            patience_counter += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
            patience_counter,
            best_val_metrics=best_val_metrics,
            seed=seed,
            scheduler=scheduler,
            best_val_auc=(
                best_val_auc
                if math.isfinite(best_val_auc)
                else None
            ),
        )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
                allow_nan=True,
            )
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val ROC-AUC {val_metrics['roc_auc']:.4f} | "
            f"Val PR-AUC {val_metrics['pr_auc']:.4f} | "
            f"Val sensitivity {val_metrics['sensitivity']:.4f} | "
            f"Val specificity {val_metrics['specificity']:.4f} | "
            f"LR backbone {optimizer.param_groups[0]['lr']:.3e} | "
            f"LR head {optimizer.param_groups[1]['lr']:.3e}"
        )

        # Match the reference: no scheduler step during warmup.
        if epoch > warmup_epochs:
            scheduler.step()

        if patience_counter >= patience:

            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation ROC-AUC={best_val_auc:.6f}"
            )

            break

    if not best_path.is_file():

        raise RuntimeError(
            "No best-model checkpoint was produced. Validation ROC-AUC "
            "was never finite; check the validation split and metrics."
        )

    best_state = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_state["model_state_dict"]
    )

    return {
        "model": model,

        "history": history,

        "best_epoch": int(
            best_state["epoch"]
        ),

        "best_val_loss": float(
            best_state["best_val_loss"]
        ),

        "best_val_auc": float(
            best_state["best_val_auc"]
        ),

        "best_val_metrics": best_state["best_val_metrics"],

        "best_checkpoint": str(best_path),

        "last_checkpoint": str(last_path),

        "history_path": str(history_path),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
    amp=True,
    threshold=0.5,
):
    """Evaluate a selected model on a supplied loader."""

    device = torch.device(device)

    model = model.to(device)

    return run_epoch(
        model,
        loader,
        criterion=make_criterion(),
        device=device,
        optimizer=None,
        amp=amp,
        threshold=threshold,
    )


def predict_with_pids(
    model,
    loader,
    device,
    amp=True,
    threshold=0.17259,
    bootstrap_samples=1000,
    bootstrap_seed=0,
):
    """
    Evaluate a sequential held-out loader and retain local patient IDs.

    Returns metrics and local-only prediction rows containing PID, label,
    positive-class probability, and thresholded prediction.
    """

    from torch.utils.data import SequentialSampler

    device = torch.device(device)

    if not isinstance(
        loader.sampler,
        SequentialSampler,
    ):
        raise ValueError(
            "Held-out evaluation requires a sequential DataLoader."
        )

    dataset_frame = getattr(
        loader.dataset,
        "df",
        None,
    )

    if (
        dataset_frame is None
        or "pid" not in dataset_frame.columns
    ):
        raise ValueError(
            "Held-out dataset must expose dataset.df['pid']."
        )

    pids = dataset_frame["pid"].tolist()

    if len(pids) != len(loader.dataset):
        raise ValueError(
            "PID count does not match held-out dataset length."
        )

    model = model.to(device)

    model.eval()

    all_labels = []

    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            x, y = _batch_to_device(
                batch,
                device,
            )

            with _autocast_context(
                device,
                enabled=amp,
            ):

                logits = model(x)

            if (
                logits.ndim != 2
                or logits.shape != (y.shape[0], 2)
            ):
                raise ValueError(
                    "Model must return logits [B, 2]; "
                    f"received {tuple(logits.shape)}"
                )

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )[:, 1]

            all_labels.extend(
                y.detach().cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.detach().cpu().tolist()
            )

    y_true = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64,
    )

    if len(y_true) != len(pids):
        raise ValueError(
            "Prediction count does not match PID count."
        )

    metrics = compute_metrics(
        y_true,
        probabilities,
        threshold=threshold,
    )

    # Seeded patient-level bootstrap confidence interval for ROC-AUC.
    rng = np.random.default_rng(
        bootstrap_seed
    )

    bootstrap_auc = []

    if len(np.unique(y_true)) == 2:

        for _ in range(int(bootstrap_samples)):

            indices = rng.choice(
                len(y_true),
                size=len(y_true),
                replace=True,
            )

            sampled_labels = y_true[indices]

            if len(np.unique(sampled_labels)) < 2:
                continue

            bootstrap_auc.append(
                roc_auc_score(
                    sampled_labels,
                    probabilities[indices],
                )
            )

    if bootstrap_auc:

        ci_low, ci_high = np.percentile(
            bootstrap_auc,
            [2.5, 97.5],
        )

        metrics["roc_auc_ci_low"] = float(ci_low)

        metrics["roc_auc_ci_high"] = float(ci_high)

    else:

        metrics["roc_auc_ci_low"] = float("nan")

        metrics["roc_auc_ci_high"] = float("nan")

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    rows = [
        {
            "pid": pid,
            "y_true": int(label),
            "p_positive": float(probability),
            "y_pred": int(prediction),
        }
        for pid, label, probability, prediction in zip(
            pids,
            y_true,
            probabilities,
            predictions,
        )
    ]

    return metrics, rows
