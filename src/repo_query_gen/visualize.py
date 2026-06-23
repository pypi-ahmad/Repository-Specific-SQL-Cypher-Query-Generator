"""Visualization helpers for training and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from repo_query_gen.utils import ensure_dir


def plot_training_summary(train_result_path: Path, eval_result_path: Path, output_dir: Path) -> dict[str, Path]:
    """Plot high-level training and validation metrics."""

    ensure_dir(output_dir)
    train = json.loads(train_result_path.read_text(encoding="utf-8"))
    eval_ = json.loads(eval_result_path.read_text(encoding="utf-8"))

    df = pd.DataFrame(
        [
            {"metric": "train_loss", "value": train.get("train_loss", 0.0)},
            {"metric": "eval_loss", "value": eval_.get("eval_loss", 0.0)},
            {"metric": "train_runtime", "value": train.get("train_runtime", 0.0)},
        ]
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="metric", y="value")
    plt.title("Training Summary Metrics")
    plt.tight_layout()
    out = output_dir / "training_summary.png"
    plt.savefig(out)
    plt.close()

    return {"training_summary": out}


def plot_error_distribution(metrics_csv: Path, output_dir: Path) -> dict[str, Path]:
    """Plot error/failure distributions from per-example metrics."""

    ensure_dir(output_dir)
    df = pd.read_csv(metrics_csv)

    # Convert correctness into coarse error signals.
    df["error_type"] = "correct"
    df.loc[df["syntax_success"] < 1.0, "error_type"] = "syntax_failure"
    df.loc[(df["syntax_success"] == 1.0) & (df["exact_match"] < 1.0), "error_type"] = "semantic_mismatch"

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="error_type", hue="task")
    plt.title("Error Distribution by Task")
    plt.tight_layout()
    out = output_dir / "error_distribution.png"
    plt.savefig(out)
    plt.close()

    return {"error_distribution": out}


__all__ = ["plot_training_summary", "plot_error_distribution"]
