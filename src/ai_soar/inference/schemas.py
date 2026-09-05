"""Pydantic contracts for the inference API.

The request schema is the guardrail against train/serve feature drift:
a prediction request MUST carry exactly the 70 canonical features (names and
count) the models were trained on. Anything else is rejected with a 422
before it can corrupt a prediction.

The response schema carries not just the prediction but the DECISION the
SOAR policy layer took (auto_response / human_approval / unknown_queue),
because in AI SOAR a prediction without a governed action is incomplete.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_soar.data.features import FEATURE_COLUMNS
from ai_soar.data.labels import BENIGN_LABEL


class FlowFeaturesRequest(BaseModel):
    """One network flow, as 70 canonical CICFlowMeter features."""

    request_id: Optional[str] = Field(default=None, description="caller-supplied id")
    source: Optional[str] = Field(default=None, description="e.g. tap0, replay, cicflowmeter")
    features: dict[str, float] = Field(
        description="mapping of canonical feature name -> value"
    )

    @field_validator("features")
    @classmethod
    def _exact_feature_space(cls, v: dict[str, float]) -> dict[str, float]:
        expected = set(FEATURE_COLUMNS)
        got = set(v)
        missing = expected - got
        extra = got - expected
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing={sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
            if extra:
                detail.append(f"extra={sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
            raise ValueError(
                f"feature space mismatch ({len(missing)} missing, {len(extra)} extra): "
                + "; ".join(detail)
            )
        return v


class PredictionResponse(BaseModel):
    """Prediction + governed decision for one flow."""

    # 'model_version' would otherwise collide with pydantic's protected 'model_' namespace
    model_config = ConfigDict(protected_namespaces=())

    request_id: Optional[str] = None
    is_malicious: bool
    gate_probability: float = Field(description="P(malicious) from stage-1")
    family: str = Field(description=f"predicted family, or {BENIGN_LABEL} if gate says benign")
    family_probabilities: dict[str, float] = Field(
        default_factory=dict, description="stage-2 probabilities per family"
    )
    confidence: float = Field(description="max stage-2 probability (0 if benign)")
    decision: str = Field(
        description="log_only | auto_response | human_approval | unknown_queue"
    )
    decision_reason: str = Field(default="")
    gate_threshold: float
    model_version: str = Field(default="unknown")
    latency_ms: float = Field(default=0.0)
    served_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    n_features_expected: int = len(FEATURE_COLUMNS)
    model_version: str = "unknown"
    gate_threshold: float = 0.5
    auto_response_families: list[str] = Field(default_factory=list)