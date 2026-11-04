#!/usr/bin/env python3
"""
Plot validation and held-out test metrics from an NVFlare NLST client log.

Usage:
    python FL_NLST_metrics.py RUN_DIRECTORY

Example:
    python FL_NLST_metrics.py 1df6f954-7f53-4239-b4b7-95517c27b4ec

The input log is expected at RUN_DIRECTORY/log.txt. Output PNG files and a
metrics CSV are written into RUN_DIRECTORY.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)

TRAIN_TASK_RE = re.compile(r"Received from server\.\s*getTask:\s*train\b")
EVAL_TASK_RE = re.compile(r"Received from server\.\s*getTask:\s*evaluate_global\b")

ROUND_START_RE = re.compile(r"\[.*?\]\s*Starting round\s+(?P<round>\d+)")
VALIDATION_RE = re.compile(
    r"Round\s+(?P<round>\d+)\s+validation:\s*"
    r"loss=(?P<loss>[-+0-9.eE]+),\s*"
    r"accuracy=(?P<accuracy>[-+0-9.eE]+),\s*"
    r"ROC-AUC=(?P<roc_auc>[-+0-9.eE]+),\s*"
    r"PR-AUC=(?P<pr_auc>[-+0-9.eE]+)"
)
EVAL_START_RE = re.compile(
    r"Evaluating aggregated global model for round\s+(?P<round>\d+)"
)
HELDOUT_RE = re.compile(
    r"Round\s+(?P<round>\d+)\s+held-out test:\s*"
    r"n=(?P<n>\d+),\s*"
    r"ROC-AUC=(?P<roc_auc>[-+0-9.eE]+),\s*"
    r"PR-AUC=(?P<pr_auc>[-+0-9.eE]+),\s*"
    r"balanced accuracy=(?P<balanced_accuracy>[-+0-9.eE]+),\s*"
    r"ROC-AUC 95% CI=\((?P<ci_low>[-+0-9.eE]+),\s*"
    r"(?P<ci_high>[-+0-9.eE]+)\)"
)


def parse_timestamp(line: str):
    """Return a datetime for timestamped NVFlare log lines, or None."""
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")


def parse_log(log_path: Path) -> pd.DataFrame:
    """Extract one row per round from the supplied NVFlare log."""
    records: dict[int, dict] = {}

    latest_train_task_time = None
    latest_eval_task_time = None
    round_train_start: dict[int, datetime] = {}
    round_eval_start: dict[int, datetime] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = parse_timestamp(line)
            message = line[TIMESTAMP_RE.match(line).end():].strip() if timestamp else line.strip()

            if timestamp and TRAIN_TASK_RE.search(message):
                latest_train_task_time = timestamp

            if timestamp and EVAL_TASK_RE.search(message):
                latest_eval_task_time = timestamp

            match = ROUND_START_RE.search(message)
            if match:
                round_number = int(match.group("round"))
                round_train_start[round_number] = latest_train_task_time
                records.setdefault(round_number, {"round": round_number})

            match = VALIDATION_RE.search(message)
            if match:
                round_number = int(match.group("round"))
                row = records.setdefault(round_number, {"round": round_number})
                for key in ("loss", "accuracy", "roc_auc", "pr_auc"):
                    row[key] = float(match.group(key))
                if timestamp and round_train_start.get(round_number):
                    row["validation_elapsed_min"] = (
                        timestamp - round_train_start[round_number]
                    ).total_seconds() / 60.0
                else:
                    row["validation_elapsed_min"] = np.nan

            match = EVAL_START_RE.search(message)
            if match:
                round_number = int(match.group("round"))
                round_eval_start[round_number] = latest_eval_task_time
                records.setdefault(round_number, {"round": round_number})

            match = HELDOUT_RE.search(message)
            if match:
                round_number = int(match.group("round"))
                row = records.setdefault(round_number, {"round": round_number})
                for key in ("n", "roc_auc", "pr_auc", "balanced_accuracy",
                            "ci_low", "ci_high"):
                    row[f"test_{key}"] = (
                        int(match.group(key)) if key == "n"
                        else float(match.group(key))
                    )
                if timestamp and round_eval_start.get(round_number):
                    row["test_elapsed_min"] = (
                        timestamp - round_eval_start[round_number]
                    ).total_seconds() / 60.0
                else:
                    row["test_elapsed_min"] = np.nan

    if not records:
        raise ValueError("No round metrics were found in the log.")

    df = pd.DataFrame(records.values()).sort_values("round").reset_index(drop=True)

    required = ["loss", "accuracy", "roc_auc", "pr_auc",
                "test_roc_auc", "test_pr_auc", "test_balanced_accuracy",
                "test_ci_low", "test_ci_high"]
    found = [column for column in required if column in df.columns]
    if not found:
        raise ValueError("The log was read, but no expected metric fields were parsed.")

    return df


def plot_validation(df: pd.DataFrame, output_path: Path) -> None:
    """Create the five-panel validation figure."""
    metrics = [
        ("loss", "Validation loss", "Loss"),
        ("accuracy", "Validation accuracy", "Accuracy"),
        ("roc_auc", "Validation ROC-AUC", "ROC-AUC"),
        ("pr_auc", "Validation PR-AUC", "PR-AUC"),
        ("validation_elapsed_min", "Round time through validation", "Elapsed time (minutes)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()

    for ax, (column, title, ylabel) in zip(axes, metrics):
        if column not in df.columns or df[column].notna().sum() == 0:
            ax.text(0.5, 0.5, "Metric not found in log",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            valid = df.dropna(subset=[column])
            ax.plot(valid["round"], valid[column], marker="o", linewidth=2)
            ax.set_xticks(df["round"].dropna().astype(int).unique())
            ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Federated round")
        ax.set_ylabel(ylabel)

    axes[-1].set_visible(False)
    fig.suptitle("NLST Federated Learning — Validation Metrics", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heldout(df: pd.DataFrame, output_path: Path) -> None:
    """Create the five-panel held-out test figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()

    metrics = [
        ("test_roc_auc", "Held-out test ROC-AUC", "ROC-AUC"),
        ("test_pr_auc", "Held-out test PR-AUC", "PR-AUC"),
        ("test_balanced_accuracy", "Held-out test balanced accuracy",
         "Balanced accuracy"),
    ]

    for ax, (column, title, ylabel) in zip(axes[:3], metrics):
        if column not in df.columns or df[column].notna().sum() == 0:
            ax.text(0.5, 0.5, "Metric not found in log",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            valid = df.dropna(subset=[column])
            ax.plot(valid["round"], valid[column], marker="o", linewidth=2)
            ax.set_xticks(df["round"].dropna().astype(int).unique())
            ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Federated round")
        ax.set_ylabel(ylabel)

    # Display the ROC-AUC confidence interval as the fourth panel.
    ax = axes[3]
    ci = df.dropna(subset=["test_ci_low", "test_ci_high"])
    if ci.empty:
        ax.text(0.5, 0.5, "ROC-AUC confidence intervals not found",
                ha="center", va="center", transform=ax.transAxes)
    else:
        center = (ci["test_ci_low"] + ci["test_ci_high"]) / 2.0
        ax.fill_between(
            ci["round"].to_numpy(),
            ci["test_ci_low"].to_numpy(),
            ci["test_ci_high"].to_numpy(),
            alpha=0.2,
            label="95% CI",
        )
        ax.plot(ci["round"], center, marker="o", linewidth=2,
                label="CI midpoint")
        ax.plot(ci["round"], ci["test_ci_low"], linestyle="--",
                linewidth=1, label="Lower bound")
        ax.plot(ci["round"], ci["test_ci_high"], linestyle="--",
                linewidth=1, label="Upper bound")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df["round"].dropna().astype(int).unique())
    ax.set_title("Held-out test ROC-AUC 95% CI")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("ROC-AUC confidence interval")

    # Fifth panel: time required for the global-model evaluation task.
    ax = axes[4]
    if "test_elapsed_min" not in df.columns or df["test_elapsed_min"].notna().sum() == 0:
        ax.text(0.5, 0.5, "Evaluation timing not found in log",
                ha="center", va="center", transform=ax.transAxes)
    else:
        valid = df.dropna(subset=["test_elapsed_min"])
        ax.plot(valid["round"], valid["test_elapsed_min"],
                marker="o", linewidth=2)
        ax.set_xticks(df["round"].dropna().astype(int).unique())
        ax.grid(True, alpha=0.3)
    ax.set_title("Global-model test evaluation time")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Elapsed time (minutes)")

    # The sixth panel is unused.
    axes[5].set_visible(False)

    fig.suptitle("NLST Federated Learning — Held-out Test Metrics", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot validation and held-out test metrics from an NVFlare run."
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Run directory containing log.txt (for example, a run UUID).",
    )
    args = parser.parse_args()

    run_directory = args.run_directory.expanduser()
    log_path = run_directory / "log.txt"

    if not run_directory.is_dir():
        parser.error(f"Run directory does not exist: {run_directory}")
    if not log_path.is_file():
        parser.error(f"Could not find log.txt in: {run_directory}")

    try:
        df = parse_log(log_path)
        validation_path = run_directory / "validation_metrics.png"
        heldout_path = run_directory / "heldout_test_metrics.png"
        csv_path = run_directory / "metrics_summary.csv"

        plot_validation(df, validation_path)
        plot_heldout(df, heldout_path)
        df.to_csv(csv_path, index=False)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Parsed {len(df)} federated round(s).")
    print(f"Validation figure: {validation_path}")
    print(f"Held-out test figure: {heldout_path}")
    print(f"Metrics table: {csv_path}")
    print("\nMetrics parsed by round:")
    display_columns = [
        column for column in (
            "round", "loss", "accuracy", "roc_auc", "pr_auc",
            "validation_elapsed_min", "test_roc_auc", "test_pr_auc",
            "test_balanced_accuracy", "test_ci_low", "test_ci_high",
            "test_elapsed_min",
        ) if column in df.columns
    ]
    print(df[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
