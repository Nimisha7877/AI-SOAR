"""Tests for the HTTP contract of the inference API.

Three disciplines are pinned here, all of them driven by how this service is
actually consumed:

1. **503 vs 500.** n8n polls ``/health`` and posts alerts. "Not ready yet"
   (503) and "crashed" (500) must stay distinguishable or the workflow either
   retries forever or pages a human about a healthy service. A ``try/except
   Exception`` around a call that itself raises ``HTTPException(503)`` silently
   converts 503 into 500 - these tests are the regression guard for that.
2. **422 before inference.** The request schema demands exactly the 70
   canonical features. A caller sending the CSE-CIC-IDS2018 column set, or
   dropping one feature, is rejected at the door: a misaligned vector still
   produces a confident number, just a meaningless one.
3. **400 before 503.** A malformed query filter is the caller's mistake and
   must be reported as such, even while the response engine is down.

Real models are not loaded: ``api._predictor`` / ``api._engine`` are replaced
with stubs so failure branches are reachable in milliseconds.
"""

from __future__ import annotations

import math

import pytest

from ai_soar.data.features import FEATURE_COLUMNS
from ai_soar.data.schema import BENIGN_LABEL
from ai_soar.inference import api as api_module
from ai_soar.inference.schemas import HealthResponse, PredictionResponse

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi/httpx not installed"
)
TestClient = fastapi_testclient.TestClient

ENDPOINT_LIST = [
    "/health",
    "/predict",
    "/predict/batch",
    "/ingest",
    "/incidents",
    "/incidents/summary",
    "/incidents/{incident_id}",
    "/incidents/{incident_id}/approve",
    "/incidents/{incident_id}/dismiss",
]

ONE_FLOW = dict.fromkeys(FEATURE_COLUMNS, 1.0)


class StubPredictor:
    """Minimal stand-in exposing exactly what the routes touch."""

    def __init__(
        self,
        decision: str = "log_only",
        family: str = BENIGN_LABEL,
        confidence: float = 0.0,
        gate_probability: float = 0.01,
        raise_on_predict: bool = False,
        raise_on_health: bool = False,
    ) -> None:
        self.decision = decision
        self.family = family
        self.confidence = confidence
        self.gate_probability = gate_probability
        self.raise_on_predict = raise_on_predict
        self.raise_on_health = raise_on_health
        self.seen: list = []

    def health(self) -> HealthResponse:
        if self.raise_on_health:
            raise RuntimeError("boom in health")
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_features_expected=len(FEATURE_COLUMNS),
            model_version="test-version",
            gate_threshold=0.5,
            auto_response_families=["BruteForce", "DDoS", "DoS", "PortScan"],
        )

    def predict(self, request) -> PredictionResponse:
        self.seen.append(request)
        if self.raise_on_predict:
            raise RuntimeError("boom in predict")
        return PredictionResponse(
            request_id=request.request_id,
            is_malicious=self.decision != "log_only",
            gate_probability=self.gate_probability,
            family=self.family,
            family_probabilities={} if self.family == BENIGN_LABEL else {self.family: 1.0},
            confidence=self.confidence,
            decision=self.decision,
            decision_reason="stub",
            gate_threshold=0.5,
            model_version="test-version",
            latency_ms=0.1,
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """TestClient with both service globals under our control.

    ``monkeypatch.setattr`` restores the originals after each test, so a stub
    can never leak into the next one (the app object is a module-level
    singleton created at import time).
    """
    monkeypatch.setattr(api_module, "_predictor", None, raising=False)
    monkeypatch.setattr(api_module, "_engine", None, raising=False)
    return TestClient(api_module.app)


def use_predictor(monkeypatch: pytest.MonkeyPatch, **kwargs) -> StubPredictor:
    stub = StubPredictor(**kwargs)
    monkeypatch.setattr(api_module, "_predictor", stub, raising=False)
    return stub


def detail_text(response) -> str:
    """Flatten FastAPI's 422 detail (list of errors) or a plain string."""
    detail = response.json().get("detail")
    return detail if isinstance(detail, str) else str(detail)


# --------------------------------------------------------------------------
# Root / service descriptor
# --------------------------------------------------------------------------


def test_root_lists_all_nine_public_endpoints(client) -> None:
    body = client.get("/").json()

    assert body["endpoints"] == ENDPOINT_LIST
    assert len(body["endpoints"]) == 9
    assert body["service"] == "ai-soar-inference"
    assert body["docs"] == "/docs"


def test_root_reports_response_engine_unavailable_when_not_loaded(client) -> None:
    assert client.get("/").json()["response_engine"] == "unavailable"


def test_root_reports_response_engine_ready(client, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_engine", object(), raising=False)

    assert client.get("/").json()["response_engine"] == "ready"


# --------------------------------------------------------------------------
# 503 discipline: not-ready must never look like crashed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/health", None),
        ("POST", "/predict", {"features": ONE_FLOW}),
        ("POST", "/predict/batch", {"flows": [{"features": ONE_FLOW}]}),
        ("POST", "/ingest", {"features": ONE_FLOW}),
        ("GET", "/incidents", None),
        ("GET", "/incidents/summary", None),
        ("GET", "/incidents/does-not-exist", None),
    ],
)
def test_routes_return_503_not_500_when_services_are_unloaded(
    client, method: str, path: str, payload
) -> None:
    """Regression guard: ``except Exception`` must not swallow HTTPException.

    ``/incidents/summary`` and ``/incidents/{id}`` used to answer 500 here
    because ``engine()`` raises 503 *inside* their try block.
    """
    response = (
        client.request(method, path, json=payload) if method == "POST" else client.get(path)
    )

    assert response.status_code == 503, response.text


def test_summary_and_detail_keep_the_engine_detail_message(client) -> None:
    """503 must still explain *which* dependency is missing."""
    for path in ("/incidents/summary", "/incidents/whatever"):
        body = client.get(path).json()
        assert "response engine unavailable" in body["detail"], path


def test_health_probe_failure_is_503_even_when_the_predictor_raises(
    client, monkeypatch
) -> None:
    """The probe must stay a probe: an exception inside is still 'not ready'."""
    use_predictor(monkeypatch, raise_on_health=True)
    response = client.get("/health")

    assert response.status_code == 503
    assert "health check failed" in detail_text(response)


def test_health_returns_the_serving_contract(client, monkeypatch) -> None:
    use_predictor(monkeypatch)
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features_expected"] == 70
    assert body["gate_threshold"] == 0.5
    assert body["auto_response_families"] == ["BruteForce", "DDoS", "DoS", "PortScan"]


# --------------------------------------------------------------------------
# The 70-feature guard (422 before inference)
# --------------------------------------------------------------------------


def test_predict_rejects_a_missing_feature_with_422(client, monkeypatch) -> None:
    stub = use_predictor(monkeypatch)
    features = {name: 1.0 for name in FEATURE_COLUMNS}
    del features["Flow Duration"]

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert "feature space mismatch" in detail_text(response)
    assert "missing=" in detail_text(response)
    assert stub.seen == []  # never reached the model


def test_predict_rejects_an_extra_feature_with_422(client, monkeypatch) -> None:
    """The 2018 dataset has columns 2017 does not - they must bounce."""
    stub = use_predictor(monkeypatch)
    features = {name: 1.0 for name in FEATURE_COLUMNS}
    features["Protocol"] = 6.0

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert "extra=" in detail_text(response)
    assert stub.seen == []


def test_predict_rejects_missing_and_extra_together(client, monkeypatch) -> None:
    use_predictor(monkeypatch)
    features = {name: 1.0 for name in FEATURE_COLUMNS[:-1]}
    features["Timestamp"] = 0.0

    text = detail_text(client.post("/predict", json={"features": features}))

    assert "1 missing" in text and "1 extra" in text


def test_predict_requires_the_features_field(client, monkeypatch) -> None:
    use_predictor(monkeypatch)

    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"features": "not-a-dict"}).status_code == 422


def test_predict_rejects_non_numeric_feature_values(client, monkeypatch) -> None:
    use_predictor(monkeypatch)
    features = {name: 1.0 for name in FEATURE_COLUMNS}
    features["Flow Duration"] = "twelve"

    assert client.post("/predict", json={"features": features}).status_code == 422


# --------------------------------------------------------------------------
# Non-finite input: stays up, answers "unknown"
# --------------------------------------------------------------------------


def test_nonfinite_features_are_accepted_and_scored_as_unknown(
    client, monkeypatch, nonfinite_features
) -> None:
    """Python's json accepts NaN/Infinity tokens, and a real tap will send them.

    The contract: HTTP 200, the request reaches the predictor intact, and the
    predictor's own non-finite guard (tested in test_predictor_policy) decides
    ``unknown_queue``. What must never happen is a 500 or an invented family.
    """
    stub = use_predictor(monkeypatch, decision="unknown_queue")

    response = client.post("/predict", json={"features": nonfinite_features})

    assert response.status_code == 200, response.text
    assert len(stub.seen) == 1
    assert len(stub.seen[0].features) == 70
    assert not all(math.isfinite(v) for v in stub.seen[0].features.values())
    assert response.json()["decision"] == "unknown_queue"


def test_predict_failure_is_500_and_never_a_partial_verdict(client, monkeypatch) -> None:
    use_predictor(monkeypatch, raise_on_predict=True)
    response = client.post("/predict", json={"features": ONE_FLOW})

    assert response.status_code == 500
    assert "inference failed" in detail_text(response)
    assert "family" not in response.json()


# --------------------------------------------------------------------------
# /predict/batch
# --------------------------------------------------------------------------


def test_batch_preserves_request_order(client, monkeypatch) -> None:
    stub = use_predictor(monkeypatch)
    flows = [
        {"request_id": f"f{i}", "features": dict.fromkeys(FEATURE_COLUMNS, float(i))}
        for i in range(5)
    ]

    response = client.post("/predict/batch", json={"flows": flows})

    assert response.status_code == 200
    assert [item["request_id"] for item in response.json()] == ["f0", "f1", "f2", "f3", "f4"]
    assert len(stub.seen) == 5


def test_batch_rejects_an_empty_list(client, monkeypatch) -> None:
    use_predictor(monkeypatch)

    assert client.post("/predict/batch", json={"flows": []}).status_code == 422


def test_batch_rejects_more_than_the_500_flow_limit(client, monkeypatch) -> None:
    use_predictor(monkeypatch)
    flows = [{"features": ONE_FLOW} for _ in range(501)]

    assert client.post("/predict/batch", json={"flows": flows}).status_code == 422


def test_batch_accepts_exactly_500_flows(client, monkeypatch) -> None:
    """Boundary is inclusive: MAX_BATCH == 500 must work, not 499."""
    use_predictor(monkeypatch)
    flows = [{"features": ONE_FLOW} for _ in range(500)]

    response = client.post("/predict/batch", json={"flows": flows})

    assert response.status_code == 200
    assert len(response.json()) == 500


def test_batch_validates_every_flow_not_just_the_first(client, monkeypatch) -> None:
    """One bad flow in a batch must reject the whole call."""
    stub = use_predictor(monkeypatch)
    bad = dict(ONE_FLOW)
    del bad["Destination Port"]

    response = client.post(
        "/predict/batch", json={"flows": [{"features": ONE_FLOW}, {"features": bad}]}
    )

    assert response.status_code == 422
    assert stub.seen == []


def test_batch_failure_is_500(client, monkeypatch) -> None:
    use_predictor(monkeypatch, raise_on_predict=True)

    response = client.post("/predict/batch", json={"flows": [{"features": ONE_FLOW}]})

    assert response.status_code == 500
    assert "batch inference failed" in detail_text(response)


# --------------------------------------------------------------------------
# Query-parameter validation: 400/422 before 503
# --------------------------------------------------------------------------


def test_unknown_status_filter_is_400_even_without_an_engine(client) -> None:
    """A malformed filter is the caller's error and must be reported as such."""
    response = client.get("/incidents", params={"status": "bogus"})

    assert response.status_code == 400
    assert "unknown status" in detail_text(response)
    assert "pending_approval" in detail_text(response)


def test_valid_status_filter_still_needs_the_engine(client) -> None:
    """...but once the filter is valid, the missing engine is the real problem."""
    assert client.get("/incidents", params={"status": "pending_approval"}).status_code == 503


def test_limit_is_bounded(client) -> None:
    assert client.get("/incidents", params={"limit": 0}).status_code == 422
    assert client.get("/incidents", params={"limit": 501}).status_code == 422


# --------------------------------------------------------------------------
# CORS helper
# --------------------------------------------------------------------------


def test_cors_defaults_cover_the_local_dashboard_origins(monkeypatch) -> None:
    monkeypatch.delenv("AI_SOAR_CORS_ORIGINS", raising=False)
    origins = api_module._cors_origins()

    assert "http://localhost:8000" in origins
    assert "http://127.0.0.1:8000" in origins
    assert "*" not in origins  # never wildcard by default


def test_cors_env_override_is_parsed(monkeypatch) -> None:
    monkeypatch.setenv(
        "AI_SOAR_CORS_ORIGINS", "https://soc.example.com, https://dash.example.com"
    )
    assert api_module._cors_origins() == [
        "https://soc.example.com",
        "https://dash.example.com",
    ]

    monkeypatch.setenv("AI_SOAR_CORS_ORIGINS", "*")
    assert api_module._cors_origins() == ["*"]


def test_cors_blank_env_falls_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AI_SOAR_CORS_ORIGINS", "   ")

    assert "*" not in api_module._cors_origins()
