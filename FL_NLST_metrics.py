

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


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.titlesize": 12,
    "savefig.facecolor": "white",
})


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

SENSITIVITY_RE = re.compile(
    r"\bsensitivity\s*[=:]\s*(?P<value>[-+0-9.eE]+)",
    re.IGNORECASE,
)

SPECIFICITY_RE = re.compile(
    r"\bspecificity\s*[=:]\s*(?P<value>[-+0-9.eE]+)",
    re.IGNORECASE,
)

# SENSITIVITY_RE = re.compile(
#     r"\\bsensitivity\\s*[=:]\\s*(?P<value>[-+0-9.eE]+)", re.IGNORECASE
# )
# SPECIFICITY_RE = re.compile(
#     r"\\bspecificity\\s*[=:]\\s*(?P<value>[-+0-9.eE]+)", re.IGNORECASE
# )



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
                sensitivity_match = SENSITIVITY_RE.search(message)
                specificity_match = SPECIFICITY_RE.search(message)
                if sensitivity_match:
                    row["sensitivity"] = float(sensitivity_match.group("value"))
                if specificity_match:
                    row["specificity"] = float(specificity_match.group("value"))
                if timestamp and round_train_start.get(round_number):
                    row["validation_elapsed_secs"] = (
                        timestamp - round_train_start[round_number]
                    ).total_seconds()
                else:
                    row["validation_elapsed_secs"] = np.nan

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
                sensitivity_match = SENSITIVITY_RE.search(message)
                specificity_match = SPECIFICITY_RE.search(message)
                if sensitivity_match:
                    row["test_sensitivity"] = float(sensitivity_match.group("value"))
                if specificity_match:
                    row["test_specificity"] = float(specificity_match.group("value"))
                if timestamp and round_eval_start.get(round_number):
                    row["test_elapsed_secs"] = (
                        timestamp - round_eval_start[round_number]
                    ).total_seconds()
                else:
                    row["test_elapsed_secs"] = np.nan

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


def style_axis(ax, ylabel: str, *, time_axis: bool = False) -> None:
    """Apply consistent axis formatting."""
    ax.set_xlabel("Federated rounds")
    ax.set_ylabel(ylabel)
    if not time_axis:
        ax.set_ylim(0.45, 1)
    else:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(np.floor(y_min), np.ceil(y_max))

    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def plot_validation(df: pd.DataFrame, output_path: Path) -> None:
    """Create validation plots, including sensitivity/specificity when logged."""
    metrics = [
        ("loss", "Validation loss", "Loss", False),
        ("accuracy", "Validation accuracy", "Accuracy", False),
        ("roc_auc", "Validation ROC-AUC", "ROC-AUC", False),
        ("pr_auc", "Validation PR-AUC", "PR-AUC", False),
        ("sensitivity", "Validation sensitivity", "Sensitivity", False),
        ("specificity", "Validation specificity", "Specificity", False),
        ("validation_elapsed_secs", "Round time through validation",
         "Elapsed time (seconds)", True),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    axes = axes.ravel()

    for ax, (column, title, ylabel, is_time) in zip(axes, metrics):
        if column not in df.columns or df[column].notna().sum() == 0:
            ax.text(0.5, 0.5, "Metric unavailable", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
        else:
            valid = df.dropna(subset=[column])
            ax.plot(valid["round"], valid[column], marker=".",
                    linewidth=1.8, markersize=4)
            ax.set_xticks(df["round"].dropna().astype(int).unique())
        ax.set_title(title, fontsize=10, pad=6)
        style_axis(ax, ylabel, time_axis=is_time)
        if column in ("sensitivity", "specificity"):
            ax.set_ylim(0, 1)

    axes[7].set_visible(False)
    fig.suptitle("NLST Non-HE Federated Learning: Validation Metrics (Moffitt)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_heldout(df: pd.DataFrame, output_path: Path) -> None:
    """Create held-out test plots, including sensitivity/specificity when logged."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    axes = axes.ravel()
    rounds = df["round"].dropna().astype(int).unique()

    metrics = [
        ("test_roc_auc", "Held-out test ROC-AUC", "ROC-AUC"),
        ("test_pr_auc", "Held-out test PR-AUC", "PR-AUC"),
        ("test_balanced_accuracy", "Held-out test balanced accuracy",
         "Balanced accuracy"),
    ]

    for ax, (column, title, ylabel) in zip(axes[:3], metrics):
        if column not in df.columns or df[column].notna().sum() == 0:
            ax.text(0.5, 0.5, "Metric unavailable", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
        else:
            valid = df.dropna(subset=[column])
            ax.plot(valid["round"], valid[column], marker=".",
                    linewidth=1.8, markersize=4)
            ax.set_xticks(rounds)
        ax.set_title(title, fontsize=10, pad=6)
        style_axis(ax, ylabel)

    ax = axes[3]
    ci = df.dropna(subset=["test_ci_low", "test_ci_high"])
    if ci.empty:
        ax.text(0.5, 0.5, "Confidence intervals unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        center = (ci["test_ci_low"] + ci["test_ci_high"]) / 2
        ax.fill_between(ci["round"].to_numpy(), ci["test_ci_low"].to_numpy(),
                        ci["test_ci_high"].to_numpy(), alpha=0.18,
                        label="95% CI")
        ax.plot(ci["round"], center, marker=".", linewidth=1.8,
                markersize=4, label="Midpoint")
        ax.plot(ci["round"], ci["test_ci_low"], "--", linewidth=1,
                color="none") #, label="Lower bound")
        ax.plot(ci["round"], ci["test_ci_high"], "--", linewidth=1,
                color="none") #, label="Upper bound")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(rounds)
    ax.set_title("Held-out test ROC-AUC 95% CI", fontsize=10, pad=6)
    style_axis(ax, "ROC-AUC")

    ax = axes[4]
    if "test_elapsed_secs" not in df.columns or df["test_elapsed_secs"].notna().sum() == 0:
        ax.text(0.5, 0.5, "Evaluation timing unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        valid = df.dropna(subset=["test_elapsed_secs"])
        ax.plot(valid["round"], valid["test_elapsed_secs"], marker=".",
                linewidth=1.8, markersize=4)
        ax.set_xticks(rounds)
    ax.set_title("Global-model test evaluation time", fontsize=10, pad=6)
    style_axis(ax, "Elapsed time (seconds)", time_axis=True)

    for ax, column, title, ylabel in (
        (axes[5], "test_sensitivity", "Held-out test sensitivity", "Sensitivity"),
        (axes[6], "test_specificity", "Held-out test specificity", "Specificity"),
    ):
        if column not in df.columns or df[column].notna().sum() == 0:
            ax.text(0.5, 0.5, "Metric unavailable", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
        else:
            valid = df.dropna(subset=[column])
            ax.plot(valid["round"], valid[column], marker=".",
                    linewidth=1.8, markersize=4)
            ax.set_xticks(rounds)
        ax.set_title(title, fontsize=10, pad=6)
        style_axis(ax, ylabel)
        ax.set_ylim(0, 1)

    axes[7].set_visible(False)
    fig.suptitle("NLST Non-HE Federated Learning: Held-out Test Metrics (Moffitt)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
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
            "sensitivity", "specificity", "validation_elapsed_secs",
            "test_roc_auc", "test_pr_auc", "test_balanced_accuracy",
            "test_sensitivity", "test_specificity", "test_ci_low",
            "test_ci_high", "test_elapsed_secs",
        ) if column in df.columns
    ]
    print(df[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
