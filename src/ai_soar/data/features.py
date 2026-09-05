"""Feature matrix construction from processed parquet splits.

Responsibilities:
- drop the 8 constant columns identified by ``scripts/profile_dataset.py``
  (zero information; they only add noise and memory)
- read ONLY the needed columns from parquet (features + Family)
- cast to float32 to halve RAM without meaningful precision loss
- expose X/y extraction for both stages:
    stage 1 binary     : BENIGN vs MALICIOUS
    stage 2 multiclass : the 8 attack families

All models and the inference service MUST get their feature list from
``FEATURE_COLUMNS`` so train/serve feature spaces can never drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ai_soar.config import get_settings
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN
from ai_soar.data.schema import CANONICAL_FEATURES

# Zero-variance columns per artifacts/reports/profile_report.json.
DROP_FEATURES: tuple[str, ...] = (
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
)

# The single source of truth for the model feature space (70 columns).
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in CANONICAL_FEATURES if c not in DROP_FEATURES
)

_READ_COLUMNS: list[str] = list(FEATURE_COLUMNS) + [FAMILY_COLUMN]


def split_dir(split: str, stratified: bool = True) -> Path:
    """Resolve a split name to its parquet directory.

    stratified=True  -> data/processed/strat/<split>   (primary)
    stratified=False -> data/processed/<split>         (day split, temporal test)
    """
    processed = Path(get_settings().paths.processed)
    base = processed / "strat" if stratified else processed
    out = base / split
    if not out.exists():
        raise FileNotFoundError(f"Split directory missing: {out}")
    return out


def load_split(
    split: str, stratified: bool = True, max_rows: int | None = None
) -> pd.DataFrame:
    """Load one split as a float32 DataFrame (features + Family only)."""
    dataset = ds.dataset(split_dir(split, stratified), format="parquet")
    table = dataset.to_table(columns=_READ_COLUMNS)
    df = table.to_pandas()
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    feats = list(FEATURE_COLUMNS)          # pandas needs a list, not a tuple
    df[feats] = df[feats].astype(np.float32)
    return df


def X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a frame into (feature matrix, family labels)."""
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y = df[FAMILY_COLUMN].to_numpy(dtype=object)
    return X, y


def to_binary_target(y: np.ndarray) -> np.ndarray:
    """Stage-1 targets: 0 = BENIGN, 1 = MALICIOUS (anything else)."""
    return (y != BENIGN_LABEL).astype(np.int8)


def feature_names() -> list[str]:
    """Ordered feature names, exactly as the model expects them."""
    return list(FEATURE_COLUMNS)