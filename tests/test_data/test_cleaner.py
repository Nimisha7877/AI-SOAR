"""Tests for the chunk cleaner: Inf -> NaN, drop NaN rows, drop duplicates.

These three rules are the reason the trained models can be trusted at all.
CICIDS2017 ships documented defects - ``Inf`` in the inter-arrival-time
columns and a large block of exact duplicate flows - and each one breaks a
different thing:

* ``Inf`` survives ``dropna``, reaches the scaler, and turns every scaled
  feature into ``nan`` or a meaningless constant.
* Exact duplicates that survive into the split put **the same flow in train and
  test** - that is textbook leakage, and it is why the headline stratified score
  is reported as an upper bound and never as "accuracy".

So the order matters as much as the rules: Inf must become NaN *before* the NaN
drop, and the NaN drop must happen *before* dedup, otherwise a row that is both
infinite and duplicated would be counted twice or survive as one clean-looking
duplicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_soar.data.cleaner import clean_dataframe, merge_stats
from ai_soar.data.schema import LABEL_COLUMN

STATS_KEYS = {
    "rows_in",
    "inf_cells",
    "rows_with_nan_dropped",
    "duplicate_rows_dropped",
    "rows_out",
}

# Three columns are enough to exercise every rule; the cleaner treats all
# non-Label columns as features, so the logic is identical at 78 columns.
FEATS = ["Flow IAT Mean", "Flow Duration", "Destination Port"]


def frame(rows: list[dict], *, unique_by_default: bool = True) -> pd.DataFrame:
    """Small frame with the canonical float dtype the pipeline expects.

    Values default to 1.0 so a test only has to state what is *interesting*
    about a row (an Inf, a NaN) instead of writing 78 numbers.

    ``unique_by_default`` gives every row a distinct ``Destination Port``
    (1000+i). Without it, two "ordinary" rows would be *exact* duplicates and
    the dedup rule would silently eat them - correct cleaner behaviour, but the
    wrong thing for an Inf/NaN test to be asserting on. Tests that specifically
    want duplicates pass ``unique_by_default=False``.
    """
    data: dict[str, list] = {}
    for c in FEATS:
        vals = []
        for i, r in enumerate(rows):
            if c in r:
                vals.append(float(r[c]))
            elif unique_by_default and c == "Destination Port":
                vals.append(1000.0 + i)
            else:
                vals.append(1.0)
        data[c] = vals
    data[LABEL_COLUMN] = [str(r.get(LABEL_COLUMN, "BENIGN")) for r in rows]
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Rule 1 - Inf / -Inf must never survive
# --------------------------------------------------------------------------


def test_inf_cells_are_converted_and_their_rows_dropped() -> None:
    df = frame([{}, {"Flow IAT Mean": np.inf}, {}])
    out, stats = clean_dataframe(df)

    assert stats["inf_cells"] == 1
    assert stats["rows_with_nan_dropped"] == 1
    assert stats["rows_out"] == 2
    assert np.isfinite(out[FEATS].to_numpy(dtype=float)).all()


def test_negative_inf_is_handled_the_same_way() -> None:
    df = frame([{}, {"Flow Duration": -np.inf}])
    out, stats = clean_dataframe(df)

    assert stats["inf_cells"] == 1
    assert stats["rows_out"] == 1
    assert np.isfinite(out["Flow Duration"].to_numpy(dtype=float)).all()


def test_inf_in_multiple_columns_of_one_row_is_counted_per_cell() -> None:
    """``inf_cells`` counts cells, not rows - the report must not understate it."""
    df = frame(
        [{"Flow IAT Mean": np.inf, "Flow Duration": -np.inf, "Destination Port": np.inf}]
    )
    _, stats = clean_dataframe(df)

    assert stats["rows_in"] == 1
    assert stats["inf_cells"] == 3
    assert stats["rows_with_nan_dropped"] == 1
    assert stats["rows_out"] == 0


# --------------------------------------------------------------------------
# Rule 2 - drop rows with NaN features, never impute
# --------------------------------------------------------------------------


def test_rows_with_nan_features_are_dropped_not_imputed() -> None:
    df = frame([{}, {"Flow IAT Mean": np.nan}, {}])
    out, stats = clean_dataframe(df)

    assert stats["rows_with_nan_dropped"] == 1
    assert stats["rows_out"] == 2
    assert not out[FEATS].isna().any().any()


def test_all_nan_input_yields_empty_frame_with_columns_intact() -> None:
    df = frame([{"Flow IAT Mean": np.nan}, {"Flow Duration": np.nan}])
    out, stats = clean_dataframe(df)

    assert len(out) == 0
    assert stats["rows_out"] == 0
    assert list(out.columns) == [*FEATS, LABEL_COLUMN]


def test_label_is_never_treated_as_a_feature() -> None:
    """A NaN/empty Label must not delete the row.

    ``_feature_columns`` excludes ``Label`` on purpose: labels are cleaned and
    mapped later, and dropping rows here would silently bias the family counts.
    """
    df = frame([{}, {}])
    df.loc[0, LABEL_COLUMN] = np.nan
    out, stats = clean_dataframe(df)

    assert stats["rows_with_nan_dropped"] == 0
    assert len(out) == 2


def test_inf_then_nan_order_matters() -> None:
    """An Inf row is reported as an Inf cell AND as a dropped NaN row.

    Inf -> NaN happens first, so the row is removed by the NaN rule. If the
    order were reversed, the Inf row would survive the dropna and poison the
    scaler downstream.
    """
    df = frame([{}, {"Flow IAT Mean": np.inf}])
    _, stats = clean_dataframe(df)

    assert stats["inf_cells"] == 1
    assert stats["rows_with_nan_dropped"] == 1
    assert stats["rows_in"] - stats["rows_out"] == 1


# --------------------------------------------------------------------------
# Rule 3 - exact duplicates (the leakage rule)
# --------------------------------------------------------------------------


def test_exact_duplicate_rows_are_dropped() -> None:
    df = frame([{}, {}, {}], unique_by_default=False)
    out, stats = clean_dataframe(df)

    assert stats["duplicate_rows_dropped"] == 2
    assert stats["rows_out"] == 1
    assert len(out) == 1


def test_drop_duplicates_false_keeps_them() -> None:
    df = frame([{}, {}, {}], unique_by_default=False)
    _, stats = clean_dataframe(df, drop_duplicates=False)

    assert stats["duplicate_rows_dropped"] == 0
    assert stats["rows_out"] == 3


def test_exact_duplicates_are_counted_separately_from_nan_drops() -> None:
    """Both removal reasons are reported independently in the same run."""
    df = frame([{}, {}, {"Flow IAT Mean": np.nan}], unique_by_default=False)
    out, stats = clean_dataframe(df)

    assert stats["rows_in"] == 3
    assert stats["rows_with_nan_dropped"] == 1
    assert stats["duplicate_rows_dropped"] == 1
    assert stats["rows_out"] == 1
    assert len(out) == 1


def test_same_features_different_label_is_not_a_duplicate() -> None:
    """Dedup is on the WHOLE row: a label conflict is data, not a duplicate."""
    df = frame(
        [{LABEL_COLUMN: "DoS Hulk"}, {LABEL_COLUMN: "BENIGN"}], unique_by_default=False
    )
    out, stats = clean_dataframe(df)

    assert stats["duplicate_rows_dropped"] == 0
    assert stats["rows_out"] == 2
    assert set(out[LABEL_COLUMN]) == {"DoS Hulk", "BENIGN"}


def test_duplicate_with_different_inf_placements_drops_both() -> None:
    df = frame([{"Flow IAT Mean": np.inf}, {"Flow Duration": np.inf}])
    _, stats = clean_dataframe(df)

    assert stats["inf_cells"] == 2
    assert stats["rows_out"] == 0  # both removed by the NaN rule


def test_dedup_runs_after_the_nan_drop_so_dropped_rows_cannot_shadow() -> None:
    """One clean row plus its Inf twin must leave exactly one row.

    If dedup ran first the two rows would differ (Inf vs 1.0) and both would
    survive to the NaN drop - the same final frame, but the *stats* would
    attribute the removal to the wrong rule, and the audit log is the evidence
    trail for the leakage report.
    """
    df = frame([{}, {"Flow IAT Mean": np.inf}])
    _, stats = clean_dataframe(df)

    assert stats["duplicate_rows_dropped"] == 0
    assert stats["rows_with_nan_dropped"] == 1
    assert stats["rows_out"] == 1


# --------------------------------------------------------------------------
# Contract: stats, index, no mutation
# --------------------------------------------------------------------------


def test_stats_contract_and_row_arithmetic() -> None:
    df = frame(
        [
            {},                                                 # kept
            {},                                                 # kept (distinct port)
            {"Flow IAT Mean": np.inf},                          # inf -> nan -> dropped
            {"Flow Duration": np.nan},                          # dropped
            {"Flow IAT Mean": 5.0, "Destination Port": 443.0},  # kept
        ]
    )
    out, stats = clean_dataframe(df)

    assert set(stats) == STATS_KEYS
    assert stats["rows_in"] == 5
    assert stats["inf_cells"] == 1
    assert stats["rows_with_nan_dropped"] == 2
    assert stats["duplicate_rows_dropped"] == 0
    assert stats["rows_out"] == 3
    assert len(out) == stats["rows_out"]
    assert all(isinstance(v, int) for v in stats.values())


def test_row_arithmetic_holds_for_a_mixed_chunk() -> None:
    """rows_out == rows_in - nan_dropped - duplicate_dropped, always."""
    df = frame(
        [
            {},
            {"Flow IAT Mean": np.inf},
            {"Flow Duration": np.nan},
        ]
        + [{}] * 3,
        unique_by_default=False,
    )
    _, stats = clean_dataframe(df)

    assert stats["rows_out"] == (
        stats["rows_in"]
        - stats["rows_with_nan_dropped"]
        - stats["duplicate_rows_dropped"]
    )


def test_input_frame_is_not_mutated() -> None:
    df = frame([{}, {"Flow IAT Mean": np.inf}, {}])
    original = df.copy(deep=True)
    out, _ = clean_dataframe(df)

    assert out is not df
    pd.testing.assert_frame_equal(df, original)


def test_index_is_reset_to_a_clean_range_index() -> None:
    """Downstream splits are positional; a sparse index would misalign them."""
    df = frame([{}, {"Flow IAT Mean": np.inf}, {}, {}, {}])
    out, stats = clean_dataframe(df)

    assert list(out.index) == list(range(len(out)))
    assert stats["rows_out"] == 4
    assert list(out.index) == [0, 1, 2, 3]


def test_empty_frame_is_handled_without_error() -> None:
    out, stats = clean_dataframe(frame([]))

    assert len(out) == 0
    assert stats == {
        "rows_in": 0,
        "inf_cells": 0,
        "rows_with_nan_dropped": 0,
        "duplicate_rows_dropped": 0,
        "rows_out": 0,
    }


# --------------------------------------------------------------------------
# merge_stats - per-chunk stats into one pipeline report
# --------------------------------------------------------------------------


def test_merge_stats_sums_only_integer_keys() -> None:
    chunks = [
        {"rows_in": 100, "inf_cells": 3, "rows_out": 90, "split": "train"},
        {"rows_in": 50, "inf_cells": 0, "rows_out": 48, "split": "test"},
    ]
    total = merge_stats(chunks)

    assert total["rows_in"] == 150
    assert total["inf_cells"] == 3
    assert total["rows_out"] == 138
    assert "split" not in total  # metadata strings must not be summed


def test_merge_stats_of_empty_list_is_empty() -> None:
    assert merge_stats([]) == {}


def test_merge_stats_reproduces_the_pipeline_arithmetic() -> None:
    """Summed chunk stats must still satisfy rows_in - dropped == rows_out."""
    frames = [
        frame([{}, {}, {"Flow IAT Mean": np.inf}]),
        frame([{"Flow Duration": np.nan}, {"Destination Port": 80.0}]),
    ]
    total = merge_stats([clean_dataframe(f)[1] for f in frames])

    assert total["rows_in"] == 5
    assert total["rows_out"] == (
        total["rows_in"]
        - total["rows_with_nan_dropped"]
        - total["duplicate_rows_dropped"]
    )
