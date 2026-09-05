"""Temporal train/test split for CICIDS2017.

Rule: train on Monday-Thursday, test on Friday. Never split randomly -
a random split places the same attack burst in both train and test, which
is leakage and produces fake accuracy.

Deliberate NON-decision: no whole day is reserved for validation. Some
attack families exist in only one day (Infiltration only in
Thursday-Afternoon), so holding out a day would delete that class from
training entirely. Validation is instead carved *inside* the training set
(stratified) at model-training time - see ai_soar/models/.
"""

from __future__ import annotations

from ai_soar.data.loader import SOURCE_ORDER
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

TRAIN_SOURCES: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday_morning",
    "thursday_afternoon",
)

TEST_SOURCES: tuple[str, ...] = (
    "friday_morning",
    "friday_afternoon_portscan",
    "friday_afternoon_ddos",
)


def split_for_key(source_key: str) -> str:
    """Return 'train' or 'test' for a source key. Raises on unknown keys."""
    if source_key in TRAIN_SOURCES:
        return "train"
    if source_key in TEST_SOURCES:
        return "test"
    raise ValueError(
        f"Source key '{source_key}' is not assigned to any split. "
        "Update TRAIN_SOURCES / TEST_SOURCES in ai_soar/data/splitter.py."
    )


def describe_split() -> dict[str, tuple[str, ...]]:
    """Mapping of split name -> source keys, for logs and reports."""
    covered = set(TRAIN_SOURCES) | set(TEST_SOURCES)
    uncovered = [k for k in SOURCE_ORDER if k not in covered]
    if uncovered:
        raise ValueError(f"Sources missing from split definition: {uncovered}")
    return {"train": TRAIN_SOURCES, "test": TEST_SOURCES}


def is_train(source_key: str) -> bool:
    return split_for_key(source_key) == "train"