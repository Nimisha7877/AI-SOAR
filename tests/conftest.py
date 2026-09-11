"""Shared pytest fixtures for the AI SOAR test suite.

Two kinds of fixtures live here, on purpose:

* **Synthetic** (``synthetic_features``, ``labelled_frame``) - tiny, fast and
  deterministic. They never touch the 2.8M-row dataset, so the suite runs in
  seconds on any laptop and in CI.
* **Real model** (``predictor``) - loads the committed
  ``artifacts/models/*.joblib`` once per session. Tests that use it verify the
  *shipped* artefacts, not a toy stand-in. If the models are absent (bare
  clone) those tests SKIP instead of failing, so the suite stays green.

Nothing in here imports pandas read paths for ``data/raw``: tests must not
depend on a multi-GB local dataset.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_soar.config import PROJECT_ROOT
from ai_soar.data.features import DROP_FEATURES, FEATURE_COLUMNS
from ai_soar.data.labels import FAMILY_ORDER
from ai_soar.data.schema import CANONICAL_FEATURES, LABEL_COLUMN

#: Kept as literals (not imported from ``ai_soar.inference.predictor``) so that
#: collecting this conftest never drags in lightgbm/scikit-learn. The
#: ``predictor`` fixture imports those lazily and cross-checks these names.
STAGE1_FILENAME = "binary_stage1.joblib"
STAGE2_FILENAME = "multiclass_stage2.joblib"

# --------------------------------------------------------------------------
# Environment hygiene
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_stray_env(monkeypatch: pytest.MonkeyPatch):
    """Strip ``AI_SOAR_*`` overrides so tests see documented defaults.

    A developer's shell may export ``AI_SOAR_GATE_THRESHOLD`` or
    ``AI_SOAR_AUTO_FAMILIES`` while experimenting; if that leaked into the
    suite, policy tests would pass or fail depending on the terminal they were
    run from. Autouse + ``reset_settings_cache`` keeps every run identical.
    """
    import os

    from ai_soar.config import reset_settings_cache

    for key in list(os.environ):
        if key.startswith("AI_SOAR_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


# --------------------------------------------------------------------------
# Feature-space facts (the contract every other test leans on)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def feature_columns() -> tuple[str, ...]:
    """The 70 canonical model features, in serving order."""
    return FEATURE_COLUMNS


@pytest.fixture(scope="session")
def synthetic_features() -> dict[str, float]:
    """One fully-formed, finite flow: exactly the 70 expected features.

    Values are deterministic (index-derived) and deliberately ordinary, so the
    stage-1 gate returns a *benign-looking* score by default. Tests that need a
    malicious verdict monkeypatch the predictor instead of hand-crafting a
    flow - inventing feature values that fool a real LightGBM model is
    guesswork, and guesswork in a test suite is a false sense of safety.
    """
    return {name: float((i % 17) + 1) for i, name in enumerate(FEATURE_COLUMNS)}


@pytest.fixture(scope="session")
def nonfinite_features(synthetic_features: dict[str, float]) -> dict[str, float]:
    """Same flow with one ``inf`` and one ``nan`` injected.

    Real taps emit these (CICIDS2017 itself shipped 4,376 ``Inf`` cells in
    ``Flow IAT``), so the API must refuse to produce a confident verdict.
    """
    out = dict(synthetic_features)
    names = list(FEATURE_COLUMNS)
    out[names[0]] = math.inf
    out[names[1]] = math.nan
    return out


# --------------------------------------------------------------------------
# Label / family fixtures (tiny DataFrames, no dataset needed)
# --------------------------------------------------------------------------

#: raw CICIDS2017 label -> expected family. Covers every documented mapping
#: plus the awkward cases: the 11-row ``Heartbleed`` class, ``Web Attack -
#: Brute Force`` (which is a WebAttack, NOT a BruteForce) and dash/encoding
#: variants that appear in real mirror downloads.
RAW_LABEL_CASES: tuple[tuple[str, str], ...] = (
    ("BENIGN", "BENIGN"),
    ("FTP-Patator", "BruteForce"),
    ("SSH-Patator", "BruteForce"),
    ("DoS Hulk", "DoS"),
    ("DoS GoldenEye", "DoS"),
    ("DoS slowloris", "DoS"),
    ("DoS Slowhttptest", "DoS"),
    ("Heartbleed", "DoS"),
    ("DDoS", "DDoS"),
    ("PortScan", "PortScan"),
    ("Web Attack \u2013 XSS", "WebAttack"),          # en dash, not hyphen
    ("Web Attack \u2014 Sql Injection", "WebAttack"),  # em dash
    ("Web Attack - Brute Force", "WebAttack"),
    ("Bot", "Botnet"),
    ("Infiltration", "Infiltration"),
)


@pytest.fixture(scope="session")
def raw_label_cases() -> tuple[tuple[str, str], ...]:
    return RAW_LABEL_CASES


@pytest.fixture
def labelled_frame(raw_label_cases: tuple[tuple[str, str], ...]):
    """Small DataFrame with ``Label`` + the minimum columns the cleaner needs.

    Rows are ordered as in :data:`RAW_LABEL_CASES` and every numeric feature is
    finite, so a test can assert on ``Family`` without worrying about
    inf/nan handling at the same time.
    """
    import pandas as pd

    n = len(raw_label_cases)
    data: dict[str, object] = {LABEL_COLUMN: [case[0] for case in raw_label_cases]}
    for i, name in enumerate(CANONICAL_FEATURES):
        data[name] = [float((row + i) % 13) + 1.0 for row in range(n)]
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def family_order() -> tuple[str, ...]:
    """The 8 families in canonical report order (BENIGN first)."""
    return FAMILY_ORDER


# --------------------------------------------------------------------------
# Real shipped models (session-scoped: joblib load happens once)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "models"


@pytest.fixture(scope="session")
def predictor(models_dir: Path):
    """Predictor built on the committed models, or SKIP if they are absent.

    Session scope matters: deserialising two LightGBM boosters takes seconds,
    and no test mutates the predictor's thresholds (tests that need different
    policy values construct their own ``Predictor`` with explicit arguments).

    The import is lazy on purpose - a machine without lightgbm installed should
    see "skipped", not "collection error".
    """
    try:
        from ai_soar.inference import predictor as predictor_module
        from ai_soar.inference.predictor import Predictor
    except ImportError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"model runtime not installed ({exc})")

    # Fail loudly if this file's literals ever drift from the shipped names.
    assert predictor_module.STAGE1_FILENAME == STAGE1_FILENAME
    assert predictor_module.STAGE2_FILENAME == STAGE2_FILENAME

    missing = [n for n in (STAGE1_FILENAME, STAGE2_FILENAME) if not (models_dir / n).exists()]
    if missing:
        pytest.skip(f"trained models not committed/present: {missing}")

    return Predictor(models_dir=models_dir)


# --------------------------------------------------------------------------
# Sanity checks on the fixtures themselves (cheap, catches drift early)
# --------------------------------------------------------------------------


def test_feature_space_is_70_columns(feature_columns: tuple[str, ...]) -> None:
    assert len(feature_columns) == 70
    assert len(set(feature_columns)) == 70, "duplicate feature names would silently misalign"
    assert len(CANONICAL_FEATURES) == 78
    assert len(DROP_FEATURES) == 8
    assert not set(feature_columns) & set(DROP_FEATURES)


def test_synthetic_features_match_contract(
    synthetic_features: dict[str, float], feature_columns: tuple[str, ...]
) -> None:
    assert set(synthetic_features) == set(feature_columns)
    assert all(math.isfinite(v) for v in synthetic_features.values())


def test_nonfinite_fixture_really_is_broken(
    nonfinite_features: dict[str, float], feature_columns: tuple[str, ...]
) -> None:
    assert set(nonfinite_features) == set(feature_columns)
    assert not all(math.isfinite(v) for v in nonfinite_features.values())
