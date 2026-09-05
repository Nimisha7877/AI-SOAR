"""Cleaning rules for CICIDS2017 chunks.

Applied to EVERY chunk, in this exact order:

1. ``Inf`` / ``-Inf``  ->  ``NaN``      (IAT columns contain Inf; it breaks
                                         scalers and corrupts metrics)
2. drop rows with any ``NaN`` feature  (we drop, never impute - imputation
                                         would invent synthetic rows)
3. drop exact duplicate rows           (documented CICIDS2017 defect; MUST
                                         happen before any train/test split
                                         or the same flow leaks into both)

:func:`clean_dataframe` returns ``(cleaned_df, stats)`` so the pipeline can
log exactly how much data was removed and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_soar.data.schema import LABEL_COLUMN
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != LABEL_COLUMN]


def clean_dataframe(
    df: pd.DataFrame, drop_duplicates: bool = True
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one chunk. Returns the cleaned frame plus removal statistics."""
    stats = {
        "rows_in": int(len(df)),
        "inf_cells": 0,
        "rows_with_nan_dropped": 0,
        "duplicate_rows_dropped": 0,
        "rows_out": 0,
    }

    out = df.copy()
    feats = _feature_columns(out)

    # 1) Inf -> NaN
    inf_count = int(np.isinf(out[feats].to_numpy(dtype=float)).sum())
    stats["inf_cells"] = inf_count
    if inf_count:
        out[feats] = out[feats].replace([np.inf, -np.inf], np.nan)

    # 2) drop rows carrying any NaN feature
    before = len(out)
    out = out.dropna(subset=feats)
    stats["rows_with_nan_dropped"] = before - len(out)

    # 3) drop exact duplicates
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        stats["duplicate_rows_dropped"] = before - len(out)

    stats["rows_out"] = int(len(out))
    if stats["rows_in"] != stats["rows_out"]:
        log.info(
            "cleaned %d -> %d rows (inf=%d, nan_rows=%d, dups=%d)",
            stats["rows_in"],
            stats["rows_out"],
            stats["inf_cells"],
            stats["rows_with_nan_dropped"],
            stats["duplicate_rows_dropped"],
        )
    return out.reset_index(drop=True), stats


def merge_stats(all_stats: list[dict[str, int]]) -> dict[str, int]:
    """Sum per-chunk stats into one pipeline-level report.

    Only integer values are summed; any metadata keys that callers attach
    to the same dict (``split``, ``parquet``, ``families``, ...) are ignored.
    """
    total: dict[str, int] = {}
    for s in all_stats:
        for k, v in s.items():
            if isinstance(v, int):
                total[k] = total.get(k, 0) + v
    return total