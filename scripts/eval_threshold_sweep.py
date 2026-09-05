"""Threshold operating-point sweep for the stage-1 gate on unseen-day data.

A gate's usefulness is not one number - it is a curve. This script trains the
gate on Mon-Thu, scores Friday, and prints recall / false-positive count for
a range of thresholds, then picks the operating point that reaches
recall >= TARGET_RECALL with the fewest false positives.

Security trade-off made explicit:
  low threshold  -> few misses (FN down), more false alarms (FP up)
  high threshold -> few alarms, more misses

Run:  python scripts/eval_threshold_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_soar.config import get_settings
from ai_soar.data.features import load_split, to_binary_target, X_y
from ai_soar.evaluation.metrics import save_report
from ai_soar.models import binary as stage1
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("eval_threshold_sweep")

TARGET_RECALL = 0.95
THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)


def _carve_val_time(df, frac: float = 0.1):
    import pandas as pd
    from ai_soar.data.labels import FAMILY_COLUMN
    tr, va = [], []
    for fam, grp in df.groupby(FAMILY_COLUMN, sort=False):
        n = len(grp)
        k = int(round(n * frac)) if n >= 10 else 0
        if k == 0:
            tr.append(grp)
            continue
        tr.append(grp.iloc[: n - k])
        va.append(grp.iloc[n - k :])
    return pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)


def main() -> int:
    settings = get_settings()
    configure_from_settings(settings)

    day_train = load_split("train", stratified=False)
    friday = load_split("test", stratified=False)
    tr, va = _carve_val_time(day_train)
    X_tr, y_tr = X_y(tr)
    X_va, y_va = X_y(va)
    X_fr, y_fr = X_y(friday)
    b_tr, b_va, b_fr = to_binary_target(y_tr), to_binary_target(y_va), to_binary_target(y_fr)

    log.info("training gate on Mon-Thu...")
    model = stage1.train_binary(X_tr, b_tr, X_va, b_va)
    proba = stage1.malicious_probability(model, X_fr)

    n_benign = int((b_fr == 0).sum())
    n_mal = int((b_fr == 1).sum())
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (b_fr == 1)).sum())
        fp = int(((pred == 1) & (b_fr == 0)).sum())
        rows.append(
            {
                "threshold": t,
                "recall": round(tp / max(n_mal, 1), 4),
                "misses": n_mal - tp,
                "false_positives": fp,
                "fp_rate_on_benign": round(fp / max(n_benign, 1), 5),
            }
        )

    chosen = next((r for r in rows if r["recall"] >= TARGET_RECALL), rows[-1])
    report = {"target_recall": TARGET_RECALL, "sweep": rows, "chosen": chosen}
    save_report(report, "threshold_sweep_report.json")

    print("\n================ THRESHOLD SWEEP (Friday) ================")
    print(f"{'threshold':>10} {'recall':>8} {'misses':>10} {'FP':>9} {'FP-rate':>9}")
    for r in rows:
        mark = "  <-- chosen" if r is chosen else ""
        print(f"{r['threshold']:>10} {r['recall']:>8} {r['misses']:>10,} "
              f"{r['false_positives']:>9,} {r['fp_rate_on_benign']:>9}{mark}")
    print(f"\nbenign rows Friday: {n_benign:,}   malicious: {n_mal:,}")
    print(f"report: {Path(settings.paths.reports) / 'threshold_sweep_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())