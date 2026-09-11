"""Tests for the 15-raw-class -> 8-family label collapse.

This is the first domino in the whole pipeline: every split, every metric and
every incident family name downstream comes from ``Family``. If the mapping is
wrong here, nothing after it can be right - and it fails *silently*, because a
mis-mapped label still trains a model that looks healthy.

The tests below pin down three things that are decisions, not accidents:

1. ``Heartbleed`` (11 rows in 2.8M) is folded into ``DoS`` rather than kept as
   an unlearnable 11-example class.
2. ``Web Attack - Brute Force`` is a **WebAttack**, not a BruteForce - the word
   "brute force" in a web-login attack is not a credential-stuffing flow.
3. Real mirror downloads contain en dashes, em dashes, mangled bytes and stray
   whitespace in labels. Normalisation must absorb all of them, and anything
   genuinely unknown must raise instead of becoming a quiet ``BENIGN``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_soar.data.labels import (
    FAMILY_COLUMN,
    FAMILY_ORDER,
    add_family_column,
    family_counts,
    family_from_label,
    normalize_label,
)
from ai_soar.data.schema import BENIGN_LABEL, LABEL_COLUMN

# The 15 classes actually present in CICIDS2017 (mirror spelling).
ALL_RAW_LABELS: tuple[str, ...] = (
    "BENIGN",
    "FTP-Patator",
    "SSH-Patator",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Bot",
    "Infiltration",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Heartbleed",
)


# --------------------------------------------------------------------------
# The mapping itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BENIGN", BENIGN_LABEL),
        ("FTP-Patator", "BruteForce"),
        ("SSH-Patator", "BruteForce"),
        ("DoS Hulk", "DoS"),
        ("DoS GoldenEye", "DoS"),
        ("DoS slowloris", "DoS"),
        ("DoS Slowhttptest", "DoS"),
        ("Heartbleed", "DoS"),
        ("DDoS", "DDoS"),
        ("PortScan", "PortScan"),
        ("Web Attack - XSS", "WebAttack"),
        ("Web Attack - Sql Injection", "WebAttack"),
        ("Web Attack - Brute Force", "WebAttack"),
        ("Bot", "Botnet"),
        ("Infiltration", "Infiltration"),
    ],
)
def test_raw_label_maps_to_expected_family(raw: str, expected: str) -> None:
    """Explicit table, independent of the conftest fixture.

    Deliberately duplicated with ``RAW_LABEL_CASES``: this is the contract a
    reviewer reads, and a test suite that only encodes its expectations in a
    fixture is harder to audit than the code it protects.
    """
    assert family_from_label(raw) == expected


def test_dataset_uses_exactly_15_raw_labels(raw_label_cases) -> None:
    """Guard against a dataset revision silently adding a 16th class.

    Compared after ``normalize_label`` because the fixture deliberately carries
    dash/encoding *variants* of two web-attack labels (that is the point of the
    fixture), so the raw spellings are not identical on both sides.
    """
    assert len(ALL_RAW_LABELS) == 15
    assert len(set(ALL_RAW_LABELS)) == 15
    assert {normalize_label(raw) for raw in ALL_RAW_LABELS} == {
        normalize_label(case[0]) for case in raw_label_cases
    }


def test_every_raw_label_lands_in_a_known_family() -> None:
    """All 15 raw labels must map, and only into the 8 declared families."""
    mapped = {family_from_label(raw) for raw in ALL_RAW_LABELS}
    assert mapped == set(FAMILY_ORDER)
    assert len(FAMILY_ORDER) == 8
    assert BENIGN_LABEL in mapped


def test_web_attack_brute_force_is_not_a_credential_attack() -> None:
    """Regression: substring matching would file this under BruteForce.

    BruteForce is on the auto-response allowlist (block the source IP). Acting
    automatically on a web-login attack because its label contains the words
    "brute force" is exactly the kind of mistake an allowlist must not inherit.
    """
    assert family_from_label("Web Attack - Brute Force") == "WebAttack"
    assert family_from_label("Web Attack - Brute Force") != "BruteForce"


def test_heartbleed_is_folded_into_dos_documented_decision() -> None:
    """11 rows against 2,273,097 BENIGN - unlearnable as its own class."""
    assert family_from_label("Heartbleed") == "DoS"


def test_unknown_label_raises_instead_of_defaulting_to_benign() -> None:
    """The most dangerous possible bug: a new attack class read as benign.

    If mapping ever returned ``BENIGN`` for unknown input, an unseen attack
    family would be *labelled* as good traffic and the model would be trained
    to ignore it. Raising turns that into a loud, immediate failure.
    """
    for bad in ("Totally New Attack", "MALWARE", "", "BENIGNN", "dos"):
        with pytest.raises(ValueError, match="Unknown CICIDS2017 label"):
            family_from_label(bad)


def test_error_message_names_the_file_to_edit() -> None:
    """The exception should tell the next person exactly where to add a label."""
    with pytest.raises(ValueError) as excinfo:
        family_from_label("SomeFutureAttack")
    assert "_RAW_TO_FAMILY" in str(excinfo.value)
    assert "labels.py" in str(excinfo.value)


# --------------------------------------------------------------------------
# Normalisation: dashes, spacing, case, mangled bytes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("Web Attack - XSS", "WebAttack"),       # plain hyphen
        ("Web Attack \u2010 XSS", "WebAttack"),  # U+2010 hyphen
        ("Web Attack \u2013 XSS", "WebAttack"),  # en dash
        ("Web Attack \u2014 XSS", "WebAttack"),  # em dash
        ("Web Attack \u2212 XSS", "WebAttack"),  # minus sign
        ("Web Attack \ufffd XSS", "WebAttack"),  # replacement char (bad decode)
        ("Web Attack \u0096 XSS", "WebAttack"),  # cp1252 leftover
        ("web attack - xss", "WebAttack"),       # lowercased
        ("WEB ATTACK - XSS", "WebAttack"),       # uppercased
        ("  Web Attack - XSS  ", "WebAttack"),   # padded
        ("Web   Attack   -   XSS", "WebAttack"),  # repeated inner spaces
    ],
)
def test_dash_spacing_case_and_encoding_variants_all_map(variant: str, expected: str) -> None:
    assert family_from_label(variant) == expected


def test_known_limitation_space_separated_repeated_dashes_are_not_unified() -> None:
    """Pinned current behaviour, not an endorsement of it.

    ``_DASH_LIKE`` collapses *adjacent* dash characters, but ``"Web Attack - -
    XSS"`` has a space between the dashes, so it normalises to
    ``"web attack - - xss"`` and raises. No CICIDS2017 mirror produces that
    spelling, so this is documented rather than fixed: if someone later widens
    normalisation, this test fails and tells them the mapping surface changed.
    """
    assert normalize_label("Web Attack - - XSS") == "web attack - - xss"
    with pytest.raises(ValueError, match="Unknown CICIDS2017 label"):
        family_from_label("Web Attack - - XSS")


def test_normalize_label_output_shape() -> None:
    assert normalize_label("  DoS\u2014HULK ") == "dos-hulk"
    assert normalize_label("Web   Attack - XSS") == "web attack - xss"
    assert normalize_label("BENIGN") == "benign"


# --------------------------------------------------------------------------
# DataFrame-level helpers
# --------------------------------------------------------------------------


def test_add_family_column_does_not_mutate_input(labelled_frame: pd.DataFrame) -> None:
    """Callers keep the raw frame; the pipeline must not edit data in place."""
    before = list(labelled_frame.columns)
    out = add_family_column(labelled_frame)

    assert out is not labelled_frame
    assert FAMILY_COLUMN in out.columns
    assert FAMILY_COLUMN not in labelled_frame.columns
    assert list(labelled_frame.columns) == before
    assert len(out) == len(labelled_frame)


def test_add_family_column_values_match_row_labels(labelled_frame: pd.DataFrame) -> None:
    out = add_family_column(labelled_frame)
    expected = [family_from_label(raw) for raw in labelled_frame[LABEL_COLUMN]]
    assert out[FAMILY_COLUMN].tolist() == expected


def test_family_counts_covers_all_eight_families_in_order(labelled_frame: pd.DataFrame) -> None:
    """Counts are reindexed to FAMILY_ORDER, so absent families show as 0.

    Reports iterate families positionally; a missing key would silently shift
    every number after it.
    """
    out = add_family_column(labelled_frame)
    counts = family_counts(out)

    assert list(counts.index) == list(FAMILY_ORDER)
    assert len(counts) == 8
    assert int(counts.sum()) == len(out)
    # The fixture has one of each family except DoS (5) and BruteForce/WebAttack (2 and 3).
    assert int(counts["DoS"]) == 5
    assert int(counts["WebAttack"]) == 3
    assert int(counts["BruteForce"]) == 2
    assert int(counts[BENIGN_LABEL]) == 1


def test_family_counts_zero_fills_absent_families() -> None:
    frame = pd.DataFrame(
        {LABEL_COLUMN: ["BENIGN", "DoS Hulk"], FAMILY_COLUMN: [BENIGN_LABEL, "DoS"]}
    )
    counts = family_counts(frame)

    assert list(counts.index) == list(FAMILY_ORDER)
    assert int(counts["DDoS"]) == 0
    assert int(counts["Infiltration"]) == 0
    assert int(counts.sum()) == 2
