"""Leakage audit: quantify near-duplicate inflation and measure honest scores.

WHY
---
Stratified splits put flows from the same attack burst (near-identical rows)
into both train and test, inflating scores. This script:

1. MEASURES the inflation directly: for each split design, what fraction of
   test rows have an (int-rounded) feature signature that also appears in
   train? High overlap == sibling contamination.
2. BUILDS a burst-hardened split: within every source file, for every
   family, the FIRST 80% of rows (capture order) go to train and the LAST
   20% to test. Bursts are contiguous in file order, so siblings cannot
   straddle the boundary.
3. TRAINS both stages on the hardened split and reports the honest metrics,
   side by side with the stratified (upper-bound) numbers.

The gap between the two tiers IS the leakage inflation - a number worth
putting in the thesis/resume because almost nobody reports it.

Run:  python scripts/eval_leakage_audit.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ai_soar.config import get_settings
from ai_soar.data.features import FEATURE_COLUMNS, to_binary_target, X_y
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.evaluation.metrics import (
    confusion_matrix_plot,
    macro_f1,
    per_class_table,
    print_classification_report,
    save_report,
)
from ai_soar.models import binary as stage1
from ai_soar.models import multiclass as stage2
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("eval_leakage_audit")

TEST_FRAC = 0.2          # last 20% of each family's rows (capture order)
VAL_FRAC_OF_TRAIN = 0.1  # carved from the END of hardened train, still burst-safe


def _source_files() -> list[Path]:
    processed = Path(get_settings().paths.processed)
    files = sorted((processed / "train").glob("*.parquet")) + sorted(
        (processed / "test").glob("*.parquet")
    )
    if not files:
        raise SystemExit("Run scripts/build_dataset.py first.")
    return files


def _signatures(df: pd.DataFrame) -> set[int]:
    """Int-rounded feature signature hashes (approximate near-dup indicator)."""
    rounded = df[list(FEATURE_COLUMNS)].round(0)
    return set(pd.util.hash_pandas_object(rounded, index=False).tolist())


def _overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    train_sigs = _signatures(train_df)
    test_sigs = pd.util.hash_pandas_object(
        test_df[list(FEATURE_COLUMNS)].round(0), index=False
    )
    hits = sum(1 for h in test_sigs if h in train_sigs)
    return round(hits / max(len(test_df), 1), 4)


def _build_hardened() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered 80/20 per source per family (bursts stay together)."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for src in _source_files():
        df = ds.dataset(src, format="parquet").to_table(
            columns=list(FEATURE_COLUMNS) + [FAMILY_COLUMN]
        ).to_pandas()
        for fam, grp in df.groupby(FAMILY_COLUMN, sort=False):
            n = len(grp)
            n_test = int(round(n * TEST_FRAC)) if n >= 5 else 0
            if n_test == 0:
                train_parts.append(grp)
                continue
            train_parts.append(grp.iloc[: n - n_test])
            test_parts.append(grp.iloc[n - n_test :])
        log.info("hardened-split %s", src.name)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    for part in (train_df, test_df):
        part[list(FEATURE_COLUMNS)] = part[list(FEATURE_COLUMNS)].astype(np.float32)
    return train_df, test_df


def _carve_val(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last 10% of each family's train rows -> val (still burst-safe)."""
    tr_parts, va_parts = [], []
    for fam, grp in train_df.groupby(FAMILY_COLUMN, sort=False):
        n = len(grp)
        n_val = int(round(n * VAL_FRAC_OF_TRAIN)) if n >= 10 else 0
        if n_val == 0:
            tr_parts.append(grp)
            continue
        tr_parts.append(grp.iloc[: n - n_val])
        va_parts.append(grp.iloc[n - n_val :])
    return (
        pd.concat(tr_parts, ignore_index=True),
        pd.concat(va_parts, ignore_index=True),
    )


def main() -> int:
    settings = get_settings()
    configure_from_settings(settings)

    # ---- 1) leakage overlap on the existing stratified splits -------------
    from ai_soar.data.features import load_split

    strat_train = load_split("train")
    strat_test = load_split("test")
    strat_overlap = _overlap(strat_train, strat_test)
    log.info("stratified near-dup overlap: %s", strat_overlap)
    del strat_train, strat_test

    # ---- 2) hardened split + its overlap ----------------------------------
    hard_train, hard_test = _build_hardened()
    hard_overlap = _overlap(hard_train, hard_test)
    log.info("hardened near-dup overlap: %s", hard_overlap)

    hard_tr, hard_va = _carve_val(hard_train)
    log.info(
        "hardened rows: train=%d val=%d test=%d", len(hard_tr), len(hard_va), len(hard_test)
    )

    # ---- 3) train both stages on hardened data ----------------------------
    X_tr, y_tr = X_y(hard_tr)
    X_va, y_va = X_y(hard_va)
    X_te, y_te = X_y(hard_test)

    log.info("training stage-1 on hardened split...")
    bin_model = stage1.train_binary(X_tr, to_binary_target(y_tr), X_va, to_binary_target(y_va))
    log.info("training stage-2 on hardened split...")
    m_tr = stage2.filter_malicious(hard_tr)
    m_va = stage2.filter_malicious(hard_va)
    Xm_tr, ym_tr = X_y(m_tr)
    Xm_va, ym_va = X_y(m_va)
    mc_model = stage2.train_multiclass(Xm_tr, ym_tr, Xm_va, ym_va)

    labels7 = list(stage2.MALICIOUS_FAMILIES)
    m_test = stage2.filter_malicious(hard_test)
    Xm_te, ym_te = X_y(m_test)
    pred = stage2.predict_family(mc_model, Xm_te)
    print("\n--- stage-2 HARDENED test report ---")
    print_classification_report(ym_te, pred, labels7)
    hard_macro7 = macro_f1(ym_te, pred, labels7)
    cm7 = confusion_matrix_plot(ym_te, pred, labels7, "hardened_multiclass_cm.png")

    # cascaded 8-family system metric on hardened test
    proba = stage1.malicious_probability(bin_model, X_te)
    is_mal = proba >= 0.5
    sys_pred = np.full(len(X_te), BENIGN_LABEL, dtype=object)
    sys_pred[is_mal] = stage2.predict_family(mc_model, X_te[is_mal])
    hard_macro8 = macro_f1(y_te, sys_pred, list(FAMILY_ORDER))
    cm8 = confusion_matrix_plot(y_te, sys_pred, list(FAMILY_ORDER), "hardened_system_cm.png")

    # ---- 4) compare with stratified upper bound ---------------------------
    reports = Path(get_settings().paths.reports)
    strat_macro7 = json.loads((reports / "multiclass_test_metrics.json").read_text())["test_macro_f1"]
    strat_macro8 = json.loads((reports / "system_test_metrics.json").read_text())[
        "test_macro_f1_8families"
    ]

    report = {
        "near_dup_overlap": {"stratified": strat_overlap, "hardened": hard_overlap},
        "rows": {"train": int(len(hard_tr)), "val": int(len(hard_va)), "test": int(len(hard_test))},
        "macro_f1_7families": {"stratified": strat_macro7, "hardened": round(hard_macro7, 4)},
        "macro_f1_8families_system": {"stratified": strat_macro8, "hardened": round(hard_macro8, 4)},
        "hardened_per_class": per_class_table(ym_te, pred, labels7),
    }
    save_report(report, "leakage_audit_report.json")

    print("\n================ LEAKAGE AUDIT ================")
    print(f"near-dup overlap   stratified={strat_overlap}   hardened={hard_overlap}")
    print(f"macro-F1 (7 fam)   stratified={strat_macro7}   hardened={round(hard_macro7, 4)}")
    print(f"macro-F1 (8 system)stratified={strat_macro8}   hardened={round(hard_macro8, 4)}")
    print(f"inflation          7fam={round(strat_macro7 - hard_macro7, 4)}  "
          f"8fam={round(strat_macro8 - hard_macro8, 4)}")
    print(f"\nreport: {reports / 'leakage_audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())