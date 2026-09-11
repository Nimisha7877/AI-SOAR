"""Tests for the cascade policy layer: gate -> family -> confidence -> decision.

The predictor's maths belongs to LightGBM; what this project owns - and what a
SOC actually audits - is the **routing**: which predictions may trigger an
automatic response, which need a human, and which must be parked as UNKNOWN.

So these tests deliberately do *not* load the shipped models. They inject
stub stage-1/stage-2 objects with known probabilities, which makes every
branch of the policy reachable and deterministic. A test that tried to
hand-craft 70 feature values to make a real booster say "DDoS at 0.93" would
be guesswork, and guesswork that happens to pass is worse than no test.

The shipped models get their own smoke tests in ``test_predictor_models.py``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ai_soar.data.features import FEATURE_COLUMNS
from ai_soar.data.schema import BENIGN_LABEL
from ai_soar.inference.predictor import (
    AUTO_RESPONSE_FAMILIES,
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_GATE_THRESHOLD,
    METADATA_FILENAME,
    STAGE1_FILENAME,
    STAGE2_FILENAME,
    Predictor,
)
from ai_soar.inference.schemas import FlowFeaturesRequest

MALICIOUS_FAMILIES = (
    "Botnet",
    "BruteForce",
    "DDoS",
    "DoS",
    "Infiltration",
    "PortScan",
    "WebAttack",
)


# --------------------------------------------------------------------------
# Stubs and builders
# --------------------------------------------------------------------------


class StubBinary:
    """Stand-in for stage 1: returns a fixed P(malicious) and records calls."""

    def __init__(self, p: float) -> None:
        self.p = float(p)
        self.calls: list[np.ndarray] = []

    def predict_proba(self, X):  # noqa: ANN001 - mirrors LGBMClassifier
        self.calls.append(np.asarray(X))
        return np.array([[1.0 - self.p, self.p]])


class StubFamily:
    """Stand-in for stage 2: fixed per-family probabilities."""

    classes_ = list(MALICIOUS_FAMILIES)

    def __init__(self, probs: dict[str, float]) -> None:
        unknown = set(probs) - set(self.classes_)
        if unknown:
            raise AssertionError(f"stub got non-family classes: {sorted(unknown)}")
        self.probs = probs
        self.calls: list[np.ndarray] = []

    def predict_proba(self, X):  # noqa: ANN001
        self.calls.append(np.asarray(X))
        return np.array([[self.probs.get(c, 0.0) for c in self.classes_]])


def make_predictor(
    gate_p: float = 0.99,
    probs: dict[str, float] | None = None,
    **kwargs,
) -> tuple[Predictor, StubBinary, StubFamily]:
    """A Predictor with stub models, built without touching disk."""
    pred = object.__new__(Predictor)  # skip __init__ (which loads joblib models)
    pred.models_dir = Path(".")
    pred.gate_threshold = float(kwargs.pop("gate_threshold", DEFAULT_GATE_THRESHOLD))
    pred.confidence_floor = float(kwargs.pop("confidence_floor", DEFAULT_CONFIDENCE_FLOOR))
    pred.auto_families = frozenset(kwargs.pop("auto_families", AUTO_RESPONSE_FAMILIES))
    pred._stage1 = StubBinary(gate_p)
    pred._stage2 = StubFamily(probs if probs is not None else {"DoS": 0.99})
    pred._metadata = kwargs.pop("metadata", {})
    assert not kwargs, f"unexpected kwargs: {sorted(kwargs)}"
    return pred, pred._stage1, pred._stage2


def write_fake_models(models_dir: Path, feature_columns=None) -> None:
    """Minimal on-disk artefacts so ``Predictor.load()`` reaches its guards."""
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / STAGE1_FILENAME).write_bytes(pickle.dumps(StubBinary(0.5)))
    (models_dir / STAGE2_FILENAME).write_bytes(pickle.dumps(StubFamily({"DoS": 1.0})))
    meta = {"trained_at": "2026-01-01T00:00:00+00:00", "rows": {"train": 42}}
    if feature_columns is not None:
        meta["feature_columns"] = list(feature_columns)
    (models_dir / METADATA_FILENAME).write_text(json.dumps(meta), encoding="utf-8")


def patch_models_dir(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> None:
    """Point ``get_settings().paths.models`` at a temp dir.

    ``Predictor.from_env()`` resolves the models directory from settings, so a
    test that only wants to read env overrides still needs *some* loadable
    artefact on disk. Patching the predictor module's own ``get_settings``
    reference keeps the real config (and the developer's machine) untouched.
    """
    from types import SimpleNamespace

    from ai_soar.inference import predictor as predictor_module

    fake = SimpleNamespace(paths=SimpleNamespace(models=models_dir))
    monkeypatch.setattr(predictor_module, "get_settings", lambda: fake)


# --------------------------------------------------------------------------
# The policy constants are the deployed banner
# --------------------------------------------------------------------------


def test_deployed_policy_constants() -> None:
    """These four numbers are what the API banner and README promise."""
    assert DEFAULT_GATE_THRESHOLD == 0.5
    assert DEFAULT_CONFIDENCE_FLOOR == 0.60
    assert AUTO_RESPONSE_FAMILIES == frozenset({"BruteForce", "DDoS", "DoS", "PortScan"})


def test_allowlist_excludes_the_weak_families() -> None:
    """Botnet (hardened F1 0.000) and Infiltration (0.034) must never automate."""
    for weak in ("Botnet", "Infiltration", "WebAttack"):
        assert weak not in AUTO_RESPONSE_FAMILIES
    assert AUTO_RESPONSE_FAMILIES <= set(MALICIOUS_FAMILIES)
    assert BENIGN_LABEL not in AUTO_RESPONSE_FAMILIES


def test_defaults_are_used_when_no_overrides_given() -> None:
    pred, _, _ = make_predictor()

    assert pred.gate_threshold == 0.5
    assert pred.confidence_floor == 0.60
    assert pred.auto_families == AUTO_RESPONSE_FAMILIES


# --------------------------------------------------------------------------
# _policy: the (family, confidence) -> decision table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "confidence", "expected"),
    [
        ("BruteForce", 0.99, "auto_response"),
        ("DDoS", 0.99, "auto_response"),
        ("DoS", 0.985, "auto_response"),
        ("PortScan", 0.98, "auto_response"),
        ("WebAttack", 0.79, "human_approval"),
        ("Botnet", 1.0, "human_approval"),
        ("Infiltration", 1.0, "human_approval"),
        ("DoS", 0.60, "auto_response"),
        ("DoS", 0.5999, "unknown_queue"),
        ("Botnet", 0.10, "unknown_queue"),
        ("WebAttack", 0.0, "unknown_queue"),
    ],
)
def test_policy_routing_table(family: str, confidence: float, expected: str) -> None:
    pred, _, _ = make_predictor()
    decision, reason = pred._policy(family, confidence)

    assert decision == expected
    assert reason  # every decision must carry a human-readable justification


def test_confidence_floor_is_checked_before_the_allowlist() -> None:
    """An allowlisted family with weak confidence must NOT auto-respond.

    This ordering is the whole point of the second gate: without it, a 0.55
    "DoS" would block an IP on a coin flip.
    """
    pred, _, _ = make_predictor()
    decision, reason = pred._policy("DoS", 0.55)

    assert decision == "unknown_queue"
    assert "floor" in reason


def test_high_confidence_non_allowlisted_family_needs_a_human() -> None:
    """Confidence is not permission. Botnet at 1.0 still waits for approval."""
    pred, _, _ = make_predictor()
    decision, reason = pred._policy("Botnet", 1.0)

    assert decision == "human_approval"
    assert "allowlist" in reason


def test_reason_strings_quote_the_numbers() -> None:
    pred, _, _ = make_predictor()

    _, auto_reason = pred._policy("DDoS", 0.931)
    assert "0.931" in auto_reason

    _, unknown_reason = pred._policy("DoS", 0.42)
    assert "0.420" in unknown_reason and "0.60" in unknown_reason


# --------------------------------------------------------------------------
# predict(): the full cascade
# --------------------------------------------------------------------------


def test_benign_flow_short_circuits_at_the_gate(synthetic_features) -> None:
    pred, stage1, stage2 = make_predictor(gate_p=0.02)
    res = pred.predict(FlowFeaturesRequest(request_id="r1", features=synthetic_features))

    assert res.is_malicious is False
    assert res.family == BENIGN_LABEL
    assert res.decision == "log_only"
    assert res.confidence == 0.0
    assert res.family_probabilities == {}
    assert res.gate_probability == pytest.approx(0.02)
    assert "threshold" in res.decision_reason
    assert len(stage1.calls) == 1
    assert stage2.calls == []  # stage 2 must never run for a benign flow


def test_gate_boundary_is_strict_less_than(synthetic_features) -> None:
    """gate_p == threshold is malicious (``<`` in code, not ``<=``)."""
    at_threshold, _, s2 = make_predictor(gate_p=0.5)
    res = at_threshold.predict(FlowFeaturesRequest(features=synthetic_features))
    assert res.is_malicious is True
    assert len(s2.calls) == 1

    just_under, _, s2b = make_predictor(gate_p=0.4999)
    res2 = just_under.predict(FlowFeaturesRequest(features=synthetic_features))
    assert res2.is_malicious is False
    assert s2b.calls == []


def test_allowlisted_family_auto_responds(synthetic_features) -> None:
    pred, _, _ = make_predictor(gate_p=0.97, probs={"DDoS": 0.93, "DoS": 0.07})
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert res.is_malicious is True
    assert res.family == "DDoS"
    assert res.confidence == pytest.approx(0.93)
    assert res.decision == "auto_response"


def test_low_confidence_goes_to_unknown_queue(synthetic_features) -> None:
    """A split vote across families is exactly what the floor exists for."""
    pred, _, _ = make_predictor(
        gate_p=0.97,
        probs={"DDoS": 0.30, "DoS": 0.28, "PortScan": 0.22, "BruteForce": 0.20},
    )
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert res.family == "DDoS"          # argmax still reported
    assert res.confidence == pytest.approx(0.30)
    assert res.decision == "unknown_queue"
    assert "floor" in res.decision_reason


def test_non_allowlisted_family_goes_to_human_approval(synthetic_features) -> None:
    pred, _, _ = make_predictor(gate_p=0.97, probs={"Infiltration": 0.99})
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert res.family == "Infiltration"
    assert res.decision == "human_approval"


def test_nonfinite_features_never_reach_a_model(nonfinite_features) -> None:
    """Inf/NaN in, confident verdict out = the worst possible failure mode."""
    pred, stage1, stage2 = make_predictor(gate_p=0.99, probs={"DoS": 0.99})
    res = pred.predict(FlowFeaturesRequest(features=nonfinite_features))

    assert res.decision == "unknown_queue"
    assert res.is_malicious is False
    assert res.family == BENIGN_LABEL
    assert res.confidence == 0.0
    assert res.gate_probability == 0.0
    assert "non-finite" in res.decision_reason
    assert stage1.calls == []   # rejected before any scoring
    assert stage2.calls == []


def test_nonfinite_reason_is_distinguishable_from_low_confidence(
    synthetic_features, nonfinite_features
) -> None:
    """Both land in unknown_queue, but an analyst must be able to tell why."""
    broken, _, _ = make_predictor()
    split, _, _ = make_predictor(probs={"DoS": 0.4, "DDoS": 0.3, "PortScan": 0.3})

    r1 = broken.predict(FlowFeaturesRequest(features=nonfinite_features))
    r2 = split.predict(FlowFeaturesRequest(features=synthetic_features))

    assert r1.decision == r2.decision == "unknown_queue"
    assert r1.decision_reason != r2.decision_reason


# --------------------------------------------------------------------------
# Response contract
# --------------------------------------------------------------------------


def test_response_carries_request_id_and_timing(synthetic_features) -> None:
    pred, _, _ = make_predictor()
    res = pred.predict(FlowFeaturesRequest(request_id="flow-42", features=synthetic_features))

    assert res.request_id == "flow-42"
    assert res.latency_ms >= 0.0
    assert res.gate_threshold == 0.5
    assert res.served_at is not None


def test_request_id_is_optional(synthetic_features) -> None:
    pred, _, _ = make_predictor()
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert res.request_id is None


def test_family_probabilities_cover_every_family_and_sum_to_one(
    synthetic_features,
) -> None:
    pred, _, _ = make_predictor(
        probs={f: 1.0 / len(MALICIOUS_FAMILIES) for f in MALICIOUS_FAMILIES}
    )
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert set(res.family_probabilities) == set(MALICIOUS_FAMILIES)
    assert sum(res.family_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_features_wrapper_matches_predict(synthetic_features) -> None:
    """The dict convenience wrapper must not change the verdict."""
    pred, _, _ = make_predictor(gate_p=0.97, probs={"PortScan": 0.95})
    a = pred.predict(FlowFeaturesRequest(request_id="x", features=synthetic_features))
    b = pred.predict_features(synthetic_features, request_id="x")

    assert (a.family, a.decision, a.confidence) == (b.family, b.decision, b.confidence)


def test_feature_vector_uses_canonical_order_regardless_of_dict_order(
    synthetic_features,
) -> None:
    """LightGBM is positional: the request dict order must be irrelevant."""
    pred, stage1, _ = make_predictor()
    shuffled = {k: synthetic_features[k] for k in reversed(list(FEATURE_COLUMNS))}

    pred.predict(FlowFeaturesRequest(features=shuffled))

    sent = stage1.calls[0]
    assert sent.shape == (1, 70)
    expected = np.array([[synthetic_features[n] for n in FEATURE_COLUMNS]], dtype=np.float64)
    np.testing.assert_allclose(sent, expected)


# --------------------------------------------------------------------------
# Per-deployment overrides
# --------------------------------------------------------------------------


def test_gate_threshold_override_flips_the_verdict(synthetic_features) -> None:
    """Same flow, same stub score: a looser gate calls it malicious.

    gate_p is fixed at 0.40, so the default 0.5 threshold says benign while a
    0.3 threshold says malicious - the knob, not the model, decides.
    """
    strict, _, _ = make_predictor(gate_p=0.40)
    loose, _, _ = make_predictor(gate_p=0.40, gate_threshold=0.3)

    assert strict.predict(FlowFeaturesRequest(features=synthetic_features)).decision == "log_only"
    assert loose.predict(FlowFeaturesRequest(features=synthetic_features)).decision != "log_only"


def test_widening_the_allowlist_changes_the_decision(synthetic_features) -> None:
    pred, _, _ = make_predictor(
        gate_p=0.97,
        probs={"WebAttack": 0.95},
        auto_families=AUTO_RESPONSE_FAMILIES | {"WebAttack"},
    )
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert res.decision == "auto_response"


def test_raising_the_confidence_floor_demotes_a_previously_auto_response(
    synthetic_features,
) -> None:
    pred, _, _ = make_predictor(
        gate_p=0.97, probs={"DoS": 0.75}, confidence_floor=0.9
    )
    res = pred.predict(FlowFeaturesRequest(features=synthetic_features))

    assert res.decision == "unknown_queue"


def test_from_env_reads_the_documented_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Operators tune the policy with env vars, not code edits."""
    write_fake_models(tmp_path)
    patch_models_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("AI_SOAR_GATE_THRESHOLD", "0.3")
    monkeypatch.setenv("AI_SOAR_CONFIDENCE_FLOOR", "0.75")
    monkeypatch.setenv("AI_SOAR_AUTO_FAMILIES", "DoS, DDoS ,WebAttack")

    pred = Predictor.from_env()

    assert pred.gate_threshold == 0.3
    assert pred.confidence_floor == 0.75
    assert pred.auto_families == frozenset({"DoS", "DDoS", "WebAttack"})


def test_from_env_without_overrides_uses_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_fake_models(tmp_path)
    patch_models_dir(monkeypatch, tmp_path)
    for var in ("AI_SOAR_GATE_THRESHOLD", "AI_SOAR_CONFIDENCE_FLOOR", "AI_SOAR_AUTO_FAMILIES"):
        monkeypatch.delenv(var, raising=False)

    pred = Predictor.from_env()

    assert pred.gate_threshold == DEFAULT_GATE_THRESHOLD
    assert pred.confidence_floor == DEFAULT_CONFIDENCE_FLOOR
    assert pred.auto_families == AUTO_RESPONSE_FAMILIES


def test_from_env_ignores_a_blank_auto_families_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A blank allowlist must fall back, not silently disable automation."""
    write_fake_models(tmp_path)
    patch_models_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("AI_SOAR_AUTO_FAMILIES", " , , ")

    pred = Predictor.from_env()

    assert pred.auto_families == AUTO_RESPONSE_FAMILIES


# --------------------------------------------------------------------------
# health(), version, repr
# --------------------------------------------------------------------------


def test_health_reports_the_serving_contract() -> None:
    pred, _, _ = make_predictor()
    h = pred.health()

    assert h.status == "ok"
    assert h.model_loaded is True
    assert h.n_features_expected == 70
    assert h.gate_threshold == 0.5
    assert h.auto_response_families == ["BruteForce", "DDoS", "DoS", "PortScan"]


def test_health_is_degraded_when_models_are_absent() -> None:
    pred, _, _ = make_predictor()
    pred._stage1 = None

    h = pred.health()
    assert h.status == "degraded"
    assert h.model_loaded is False
    assert pred.loaded is False


def test_version_string_includes_train_row_count() -> None:
    pred, _, _ = make_predictor(
        metadata={"trained_at": "2026-09-04T18:11:27+00:00", "rows": {"train": 2093105}}
    )

    assert pred.version == "2026-09-04T18:11:27+00:00 (train_rows=2093105)"


def test_version_survives_missing_metadata() -> None:
    pred, _, _ = make_predictor(metadata={})

    assert pred.version == "unknown (train_rows=?)"


def test_repr_shows_the_policy_not_internals() -> None:
    pred, _, _ = make_predictor()
    text = repr(pred)

    assert "loaded=True" in text
    assert "families=7" in text
    assert "gate_threshold=0.5" in text


# --------------------------------------------------------------------------
# Load-time guards (real files on disk, stub payloads inside)
# --------------------------------------------------------------------------


def test_missing_models_raise_with_the_fix_in_the_message(tmp_path: Path) -> None:
    """A bare clone must fail loudly, not serve 500s."""
    with pytest.raises(FileNotFoundError) as excinfo:
        Predictor(models_dir=tmp_path)

    msg = str(excinfo.value)
    assert STAGE1_FILENAME in msg and STAGE2_FILENAME in msg
    assert "train_models.py" in msg


def test_partially_missing_models_also_raise(tmp_path: Path) -> None:
    (tmp_path / STAGE1_FILENAME).write_bytes(pickle.dumps(StubBinary(0.5)))

    with pytest.raises(FileNotFoundError) as excinfo:
        Predictor(models_dir=tmp_path)

    assert STAGE2_FILENAME in str(excinfo.value)


def test_feature_space_mismatch_is_refused(tmp_path: Path) -> None:
    """The strongest guard here: metadata vs FEATURE_COLUMNS drift."""
    write_fake_models(tmp_path, feature_columns=["Wrong", "Columns", "Here"])

    with pytest.raises(ValueError, match="feature space mismatch"):
        Predictor(models_dir=tmp_path)


def test_matching_feature_space_in_metadata_loads(tmp_path: Path) -> None:
    write_fake_models(tmp_path, feature_columns=list(FEATURE_COLUMNS))
    pred = Predictor(models_dir=tmp_path)

    assert pred.loaded is True
    assert pred.families == list(MALICIOUS_FAMILIES)


def test_metadata_is_optional(tmp_path: Path) -> None:
    """No metadata file == no feature-space check, but the models still load."""
    (tmp_path / STAGE1_FILENAME).write_bytes(pickle.dumps(StubBinary(0.5)))
    (tmp_path / STAGE2_FILENAME).write_bytes(pickle.dumps(StubFamily({"DoS": 1.0})))
    pred = Predictor(models_dir=tmp_path)

    assert pred.loaded is True
    assert pred.version == "unknown (train_rows=?)"
