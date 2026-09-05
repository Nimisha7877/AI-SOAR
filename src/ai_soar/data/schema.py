"""CICIDS2017 schema: canonical column names, dtypes and normalization.

Built against the ACTUAL mirror present in ``data/raw``: 78 features + Label
= 79 columns. This mirror omits ``Protocol`` and ``Timestamp`` and ships
column names with inconsistent leading spaces plus a few typos.

Every downstream module (cleaner, loader, models, inference) works ONLY with
the canonical names produced by :func:`normalize_dataframe`. That guarantees
the training feature space and the real-time feature space can never drift.
"""

from __future__ import annotations

import pandas as pd

LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"

# raw (stripped) name -> canonical name, for mirror-specific typos
COLUMN_ALIASES: dict[str, str] = {
    "FlowIAT Min": "Flow IAT Min",
    "Min Packet Length": "Packet Length Min",
    "Max Packet Length": "Packet Length Max",
    "Avg Fwd Segment Size": "Fwd Segment Size Avg",
    "Avg Bwd Segment Size": "Bwd Segment Size Avg",
}

# The 78 canonical feature columns, in dataset order.
CANONICAL_FEATURES: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Bwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
)

# Conceptually integer-valued columns (counts, flags, ports). Kept as float64
# in memory for NaN-safety; used for reporting and future feature engineering.
INTEGER_FEATURES: frozenset[str] = frozenset(
    {
        "Destination Port",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "Fwd Header Length",
        "Bwd Header Length",
        "Fwd Header Length.1",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWE Flag Count",
        "ECE Flag Count",
        "Subflow Fwd Packets",
        "Subflow Bwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Bytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
    }
)

EXPECTED_FEATURE_COUNT = len(CANONICAL_FEATURES)  # 78


def normalize_column_name(name: str) -> str:
    """Strip whitespace and apply alias fixes to a single raw column name."""
    stripped = name.strip()
    return COLUMN_ALIASES.get(stripped, stripped)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical column names, numeric features, fixed order.

    - strips/aliases every column name
    - coerces all features to numeric (``errors='coerce'`` turns junk into NaN)
    - strips whitespace from Label values
    - reorders to CANONICAL_FEATURES + Label, dropping anything unexpected
    """
    out = df.rename(columns=normalize_column_name).copy()

    for col in CANONICAL_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if LABEL_COLUMN in out.columns:
        out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(str).str.strip()

    cols = [c for c in CANONICAL_FEATURES if c in out.columns]
    if LABEL_COLUMN in out.columns:
        cols.append(LABEL_COLUMN)
    return out[cols]


def validate_dataframe(df: pd.DataFrame) -> dict[str, list[str]]:
    """Health-check a normalized frame.

    Returns ``{'missing': [...], 'extra': [...], 'non_numeric': [...]}``.
    All three empty == the frame matches the expected schema exactly.
    """
    missing = [c for c in CANONICAL_FEATURES if c not in df.columns]
    known = set(CANONICAL_FEATURES) | {LABEL_COLUMN}
    extra = [c for c in df.columns if c not in known]
    non_numeric = [
        c
        for c in CANONICAL_FEATURES
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    return {"missing": missing, "extra": extra, "non_numeric": non_numeric}