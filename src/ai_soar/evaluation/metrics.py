"""Evaluation metrics for AI SOAR.

Every number this project reports goes through this module so that:
- metrics are consistent between experiments
- confusion matrices are always saved as PNGs for the report/thesis
- every JSON report carries the standing caveat that CICIDS2017 stratified
  scores are an UPPER BOUND (near-duplicate flows across splits), and that
  temporal / cross-dataset results are the honest generalization numbers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")          # headless: no GUI needed, server-safe

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai_soar.config import get_settings
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

STANDING_CAVEAT = (
    "CICIDS2017 stratified-split scores are an UPPER BOUND: near-duplicate "
    "flows from the same attack burst appear across splits. Temporal "
    "(Mon-Thu -> Friday) and cross-dataset (CSE-CIC-IDS2018) results are the "
    "honest generalization numbers."
)


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    """The headline metric. Never accuracy - the dataset is ~83% BENIGN."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_table(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> dict:
    """Per-class precision/recall/f1/support as a dict (JSON-friendly)."""
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    counts = np.array([(np.asarray(y_true) == lab).sum() for lab in labels])
    return {
        str(lab): {
            "precision": round(float(pp), 4),
            "recall": round(float(rr), 4),
            "f1": round(float(ff), 4),
            "support": int(cc),
        }
        for lab, pp, rr, ff, cc in zip(labels, p, r, f, counts)
    }


def binary_metrics(y_true: Sequence, proba: Sequence, threshold: float = 0.5) -> dict:
    """Stage-1 metrics from P(malicious): AUC, F1, and FP/FN counts."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "false_positives": int(((pred == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((pred == 0) & (y_true == 1)).sum()),
        "threshold": threshold,
    }


def confusion_matrix_plot(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
    filename: str,
    normalize: bool = True,
) -> Path:
    """Save a confusion-matrix heatmap PNG under artifacts/reports/."""
    out = Path(get_settings().paths.reports) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("confusion matrix saved to %s", out)
    return out


def save_report(payload: dict, filename: str) -> Path:
    """Write a JSON report with timestamp + standing caveat attached."""
    out = Path(get_settings().paths.reports) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("caveat", STANDING_CAVEAT)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("report saved to %s", out)
    return out


def print_classification_report(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> str:
    """Console-friendly full report (also returns the string for logging)."""
    text = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0)
    print(text)
    return text