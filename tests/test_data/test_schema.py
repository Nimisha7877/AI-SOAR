"""Tests for schema normalisation and validation.

The CICIDS2017 mirror this project is built on does not have clean column
names: several arrive with leading/trailing spaces, five are spelled
differently from the canonical CICFlowMeter names, and the mirror omits
``Protocol`` and ``Timestamp`` entirely.

Nothing downstream is allowed to see that mess. Every module works on the
canonical frame produced by :func:`normalize_dataframe`, because LightGBM
stores features **positionally**: if the column order drifts between training
and serving, the model keeps returning confident numbers for the wrong
features, and no metric in the world will tell you.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_soar.data.features import DROP_FEATURES, FEATURE_COLUMNS
from ai_soar.data.schema import (
    CANONICAL_FEATURES,
    COLUMN_ALIASES,
    EXPECTED_FEATURE_COUNT,
    LABEL_COLUMN,
    normalize_column_name,
    normalize_dataframe,
    validate_dataframe,
)


def raw_frame(
    *,
    padded: bool = False,
    rows: int = 3,
    alias_names: bool = False,
) -> pd.DataFrame:
    """A raw mirror-style frame: 78 features + Label.

    ``padded`` reproduces the leading/trailing spaces the real download has.
    ``alias_names`` swaps in the five mirror-specific misspellings.
    """
    names = list(CANONICAL_FEATURES)
    if alias_names:
        rev = {v: k for k, v in COLUMN_ALIASES.items()}
        names = [rev.get(n, n) for n in names]
    if padded:
        names = [f" {n} " for n in names]

    data: dict[str, list] = {
        name: [float(i + j + 1) for i in range(rows)] for j, name in enumerate(names)
    }
    data[LABEL_COLUMN] = ["BENIGN", "DoS Hulk", "DDoS"][:rows]
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# normalize_column_name
# --------------------------------------------------------------------------


def test_strips_surrounding_whitespace() -> None:
    assert normalize_column_name("  Flow Duration ") == "Flow Duration"
    assert normalize_column_name("\tDestination Port\n") == "Destination Port"


def test_applies_every_documented_alias() -> None:
    assert normalize_column_name("FlowIAT Min") == "Flow IAT Min"
    assert normalize_column_name("Min Packet Length") == "Packet Length Min"
    assert normalize_column_name("Max Packet Length") == "Packet Length Max"
    assert normalize_column_name("Avg Fwd Segment Size") == "Fwd Segment Size Avg"
    assert normalize_column_name("Avg Bwd Segment Size") == "Bwd Segment Size Avg"


def test_alias_lookup_happens_after_stripping() -> None:
    """The real file pads the typo too: ' FlowIAT Min ' must still resolve."""
    assert normalize_column_name(" FlowIAT Min ") == "Flow IAT Min"


def test_unknown_names_pass_through_unchanged() -> None:
    assert normalize_column_name("Something New") == "Something New"
    assert normalize_column_name("Fwd Header Length.1") == "Fwd Header Length.1"


def test_alias_targets_are_all_canonical() -> None:
    """An alias that maps to a non-canonical name would silently drop a column."""
    assert set(COLUMN_ALIASES.values()) <= set(CANONICAL_FEATURES)
    assert len(COLUMN_ALIASES) == 5


# --------------------------------------------------------------------------
# normalize_dataframe - the full mirror shape
# --------------------------------------------------------------------------


def test_padded_mirror_columns_normalise_to_the_canonical_set() -> None:
    out = normalize_dataframe(raw_frame(padded=True))

    assert list(out.columns) == [*CANONICAL_FEATURES, LABEL_COLUMN]
    assert len(out.columns) == 79


def test_aliased_mirror_columns_normalise_to_the_canonical_set() -> None:
    out = normalize_dataframe(raw_frame(alias_names=True))

    assert list(out.columns) == [*CANONICAL_FEATURES, LABEL_COLUMN]


def test_padded_and_aliased_together() -> None:
    out = normalize_dataframe(raw_frame(padded=True, alias_names=True))

    assert list(out.columns) == [*CANONICAL_FEATURES, LABEL_COLUMN]
    assert validate_dataframe(out) == {"missing": [], "extra": [], "non_numeric": []}


def test_column_order_is_exactly_canonical() -> None:
    """Positional model contract: order is part of the schema, not cosmetics."""
    shuffled = raw_frame()[list(reversed([*CANONICAL_FEATURES, LABEL_COLUMN]))]
    out = normalize_dataframe(shuffled)

    assert list(out.columns) == [*CANONICAL_FEATURES, LABEL_COLUMN]


def test_label_moves_to_the_end_even_when_it_comes_first() -> None:
    df = raw_frame()
    df = df[[LABEL_COLUMN, *CANONICAL_FEATURES]]
    out = normalize_dataframe(df)

    assert out.columns[-1] == LABEL_COLUMN
    assert list(out.columns[:-1]) == list(CANONICAL_FEATURES)


def test_unexpected_columns_are_dropped() -> None:
    """The mirror omits Protocol/Timestamp; other mirrors add junk columns."""
    df = raw_frame()
    df["Protocol"] = 6
    df[" Timestamp "] = "05/07/2017 03:00:01 PM"
    df[" Flow ID"] = 1
    out = normalize_dataframe(df)

    assert "Protocol" not in out.columns
    assert "Timestamp" not in out.columns
    assert "Flow ID" not in out.columns
    assert list(out.columns) == [*CANONICAL_FEATURES, LABEL_COLUMN]


def test_junk_values_become_nan_rather_than_raising() -> None:
    """``errors='coerce'`` hands the problem to the cleaner's NaN-drop rule."""
    df = raw_frame()
    junk = df["Flow Duration"].astype(object).tolist()
    junk[0] = "not-a-number"
    df["Flow Duration"] = junk
    out = normalize_dataframe(df)

    assert np.isnan(out.loc[0, "Flow Duration"])
    # helper builds value = row + column_index + 1, and Flow Duration is
    # column index 1 -> row 1 is 3.0
    assert out.loc[1, "Flow Duration"] == 3.0
    assert out.loc[2, "Flow Duration"] == 4.0


def test_numeric_strings_are_coerced_to_float() -> None:
    df = raw_frame()
    df["Destination Port"] = ["443", "80", "22"]
    out = normalize_dataframe(df)

    assert list(out["Destination Port"]) == [443.0, 80.0, 22.0]
    assert pd.api.types.is_numeric_dtype(out["Destination Port"])


def test_label_values_are_stripped() -> None:
    """A padded label would fail the family mapping with a ValueError."""
    df = raw_frame()
    df[LABEL_COLUMN] = ["  BENIGN ", " DoS Hulk", "DDoS\n"]
    out = normalize_dataframe(df)

    assert list(out[LABEL_COLUMN]) == ["BENIGN", "DoS Hulk", "DDoS"]


def test_input_frame_is_not_mutated() -> None:
    df = raw_frame(padded=True)
    before = list(df.columns)
    out = normalize_dataframe(df)

    assert out is not df
    assert list(df.columns) == before
    assert " Flow Duration " in df.columns


def test_missing_features_are_tolerated_and_reported() -> None:
    """Normalise does not invent columns; validate is what complains."""
    df = raw_frame().drop(columns=["Idle Min", "Active Mean"])
    out = normalize_dataframe(df)

    assert "Idle Min" not in out.columns
    assert len(out.columns) == 77
    report = validate_dataframe(out)
    assert report["missing"] == ["Active Mean", "Idle Min"]


# --------------------------------------------------------------------------
# validate_dataframe
# --------------------------------------------------------------------------


def test_clean_frame_passes_with_all_three_lists_empty() -> None:
    report = validate_dataframe(normalize_dataframe(raw_frame()))

    assert report == {"missing": [], "extra": [], "non_numeric": []}
    assert set(report) == {"missing", "extra", "non_numeric"}


def test_reports_missing_and_extra_columns() -> None:
    df = raw_frame()
    df["Random Extra"] = 1.0
    df = df.drop(columns=["Destination Port"])
    report = validate_dataframe(df)

    assert report["missing"] == ["Destination Port"]
    assert report["extra"] == ["Random Extra"]


def test_reports_non_numeric_feature_columns() -> None:
    df = raw_frame()
    df["Flow Bytes/s"] = ["a", "b", "c"]
    report = validate_dataframe(df)

    assert report["non_numeric"] == ["Flow Bytes/s"]
    assert report["missing"] == []


def test_label_column_is_never_reported_as_extra() -> None:
    report = validate_dataframe(raw_frame())

    assert LABEL_COLUMN not in report["extra"]
    assert report["extra"] == []


# --------------------------------------------------------------------------
# The 78 -> 70 feature-space contract
# --------------------------------------------------------------------------


def test_feature_counts_are_the_documented_numbers() -> None:
    """README, reports and model_metadata all quote 78 canonical / 70 served."""
    assert EXPECTED_FEATURE_COUNT == 78
    assert len(CANONICAL_FEATURES) == 78
    assert len(DROP_FEATURES) == 8
    assert len(FEATURE_COLUMNS) == 70


def test_dropped_columns_are_the_zero_variance_bulk_columns() -> None:
    """All eight are Avg Bytes/Packets/Bulk Rate columns - constant in CICIDS2017."""
    assert all(
        ("Bulk" in name) or name in {"Bwd PSH Flags", "Bwd URG Flags"}
        for name in DROP_FEATURES
    )
    assert set(DROP_FEATURES) <= set(CANONICAL_FEATURES)


def test_served_features_preserve_dataset_order() -> None:
    """FEATURE_COLUMNS must be a subsequence of CANONICAL_FEATURES."""
    canonical = [c for c in CANONICAL_FEATURES if c not in DROP_FEATURES]
    assert list(FEATURE_COLUMNS) == canonical


def test_duplicate_canonical_name_would_be_caught() -> None:
    assert len(set(CANONICAL_FEATURES)) == len(CANONICAL_FEATURES)
    assert "Fwd Header Length" in CANONICAL_FEATURES
    assert "Fwd Header Length.1" in CANONICAL_FEATURES  # pandas dupe suffix, kept
