from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_weights(labels, device):
    """
    Compute inverse-frequency class weights from training labels only.

    Returns a tensor [weight_for_class_0, weight_for_class_1].
    """
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)

    counts = torch.bincount(labels, minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError(
            f"Both classes must be present in training labels; counts={counts.tolist()}"
        )

    weights = counts.sum() / (2.0 * counts)
    return weights.to(device)


def make_criterion(class_weights=None):
    """
    Construct a binary cross-entropy loss.

    The model must return two logits in class order [0, 1].
    """
    if class_weights is None:
        return nn.CrossEntropyLoss()

    class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
    if class_weights.numel() != 2:
        raise ValueError("class_weights must contain exactly two values.")

    return nn.CrossEntropyLoss(weight=class_weights)


def _autocast_context(device, enabled=True):
    """Use CUDA mixed precision when requested and available."""
    use_amp = bool(enabled and device.type == "cuda")
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    )


def _batch_to_device(batch, device):
    x, y = batch
    x = x.to(device=device, dtype=torch.float32, non_blocking=True)
    y = y.to(device=device, dtype=torch.long, non_blocking=True).view(-1)

    if x.ndim != 5:
        raise ValueError(f"Expected input [B,C,D,H,W], got {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")
    if not torch.isfinite(x).all():
        raise ValueError("Input batch contains non-finite values.")
    if not torch.isin(y, torch.tensor([0, 1], device=device)).all():
        raise ValueError("Labels must be binary values 0 or 1.")

    return x, y


def compute_metrics(y_true, probabilities, threshold=0.5):
    """Calculate threshold-based metrics and ranking metrics."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    predictions = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0

    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "n": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "sensitivity": sensitivity,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if has_both_classes else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, probabilities))
            if has_both_classes else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    amp=True,
    threshold=0.5,
    max_grad_norm=1.0,
):
    """
    Run one training or evaluation epoch.

    If optimizer is supplied, perform parameter updates.
    Otherwise, run evaluation without gradients.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_probabilities = []

    scaler_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    for batch in loader:
        x, y = _batch_to_device(batch, device)
        batch_size = y.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            with _autocast_context(device, enabled=amp):
                logits = model(x)

                if logits.ndim != 2 or logits.shape != (batch_size, 2):
                    raise ValueError(
                        "Model must return logits of shape [B, 2]; "
                        f"received {tuple(logits.shape)}"
                    )

                loss = criterion(logits, y)

            if training:
                scaler.scale(loss).backward()

                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )

                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.softmax(logits.detach().float(), dim=1)[:, 1]

        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        all_labels.extend(y.detach().cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    if total_examples == 0:
        raise ValueError("DataLoader produced no batches.")

    metrics = compute_metrics(
        all_labels, all_probabilities, threshold=threshold
    )
    metrics["loss"] = float(total_loss / total_examples)

    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    patience_counter,
    best_val_metrics=None,
    seed=42,
):
    """Save model and optimizer state for resuming training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "best_val_loss": float(best_val_loss),
        "patience_counter": int(patience_counter),
        "best_val_metrics": best_val_metrics,
        "seed": int(seed),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Restore model, optimizer, and random states from a saved checkpoint."""
    checkpoint = torch.load(
        Path(path), map_location=device, weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"].detach().cpu().to(torch.uint8)
        )

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        cuda_states = [
            state.detach().cpu().to(torch.uint8)
            for state in checkpoint["cuda_rng_state_all"]
        ]
        # Restore only states corresponding to available CUDA devices.
        device_count = torch.cuda.device_count()
        for index, state in enumerate(cuda_states[:device_count]):
            torch.cuda.set_rng_state(state, device=index)

    return checkpoint


def fit(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    epochs=30,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=7,
    min_delta=1e-4,
    class_weights=None,
    amp=True,
    threshold=0.5,
    seed=42,
    resume_from=None,
):
    """
    Train with validation-loss early stopping.

    Saves:
      best_model.pt       Best validation-loss model weights.
      last_checkpoint.pt  Most recent resumable training state.
      history.json        Per-epoch train and validation metrics.

    The held-out test loader is intentionally not accepted here.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if patience < 1:
        raise ValueError("patience must be >= 1.")

    set_seed(seed)
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = make_criterion(
        class_weights.to(device)
        if isinstance(class_weights, torch.Tensor)
        else class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_checkpoint.pt"
    history_path = output_dir / "history.json"

    history = []
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    best_val_metrics = None

    if resume_from is not None:
        state = load_checkpoint(
            resume_from, model, optimizer=optimizer, device=device
        )
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
        patience_counter = int(state["patience_counter"])
        best_val_metrics = state.get("best_val_metrics")

        if history_path.is_file():
            history = json.loads(history_path.read_text())

        print(f"Resuming at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, amp=amp, threshold=threshold,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, amp=amp, threshold=threshold,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)

        improved = val_metrics["loss"] < best_val_loss - min_delta

        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_val_metrics = val_metrics

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
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
        )
        history_path.write_text(json.dumps(history, indent=2, allow_nan=True))

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val ROC-AUC {val_metrics['roc_auc']:.4f} | "
            f"val PR-AUC {val_metrics['pr_auc']:.4f} | "
            f"val sensitivity {val_metrics['sensitivity']:.4f} | "
            f"val specificity {val_metrics['specificity']:.4f}"
        )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation loss={best_val_loss:.6f}"
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            "No best-model checkpoint was produced. "
            "Check the training logs and validation loss."
        )

    best_state = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(best_state["model_state_dict"])

    return {
        "model": model,
        "history": history,
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_val_metrics": best_state["best_val_metrics"],
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(history_path),
    }


def evaluate(model, loader, device, amp=True, threshold=0.5):
    """Evaluate a selected model on a supplied loader."""
    device = torch.device(device)
    model = model.to(device)

    metrics = run_epoch(
        model,
        loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        optimizer=None,
        amp=amp,
        threshold=threshold,
    )
    return metrics
