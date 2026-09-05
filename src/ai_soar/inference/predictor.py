"""Cascade predictor: stage-1 gate -> stage-2 family -> confidence gate -> decision.

This module is the "brain" behind ``POST /predict``. It owns three things that
must never drift apart:

1. **Feature ordering** - LightGBM stores features positionally, so the request
   dict is re-ordered into ``FEATURE_COLUMNS`` here. A caller cannot silently
   shuffle columns and get wrong-but-plausible predictions.
2. **The cascade** - a flow only reaches the multiclass stage if the binary
   gate says malicious. Benign flows are logged and dropped (cheap, and it
   matches how a SOC treats 90% of traffic).
3. **The response policy** - which families may be auto-responded to, and which
   need a human. The allowlist below is *evidence*, not taste: it comes from
   the burst-hardened leakage audit
   (``artifacts/reports/leakage_audit_report.json``)::

       BruteForce 0.999 | DDoS 0.999 | DoS 0.985 | PortScan 0.980  -> automation
       WebAttack  0.790 | Botnet 0.000 | Infiltration 0.034        -> human approval

   Honest caveat, from the temporal evaluation
   (``artifacts/reports/temporal_eval_report.json``): an attack family the model
   never saw in training gets classified into a *known* family with confidence
   ~1.0. So the confidence floor catches ambiguity inside the seen class set
   only - it is NOT a novelty detector. What actually contains that risk is the
   human_approval route plus per-family response allowlisting (Step 7).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ai_soar.config import get_settings
from ai_soar.data.features import FEATURE_COLUMNS
from ai_soar.data.schema import BENIGN_LABEL
from ai_soar.inference.schemas import FlowFeaturesRequest, HealthResponse, PredictionResponse
from ai_soar.models.binary import load_model as load_binary_model
from ai_soar.models.binary import malicious_probability
from ai_soar.models.multiclass import load_model as load_multiclass_model
from ai_soar.models.multiclass import predict_proba_families
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Policy constants (evidence-backed; overridable per deployment)
# --------------------------------------------------------------------------
#: Families allowed to trigger automatic response. Hardened F1 >= 0.90.
AUTO_RESPONSE_FAMILIES: frozenset[str] = frozenset(
    {"BruteForce", "DDoS", "DoS", "PortScan"}
)

#: Default P(malicious) cut for the stage-1 gate.
DEFAULT_GATE_THRESHOLD: float = 0.5

#: Below this stage-2 confidence a malicious flow goes to the human queue.
DEFAULT_CONFIDENCE_FLOOR: float = 0.60

STAGE1_FILENAME = "binary_stage1.joblib"
STAGE2_FILENAME = "multiclass_stage2.joblib"
METADATA_FILENAME = "model_metadata.json"


class Predictor:
    """Loads both models once and scores single flows in microseconds."""

    def __init__(
        self,
        models_dir: Optional[Path | str] = None,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        auto_families: Optional[frozenset[str]] = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path(get_settings().paths.models)
        self.gate_threshold = float(gate_threshold)
        self.confidence_floor = float(confidence_floor)
        self.auto_families = frozenset(auto_families) if auto_families else AUTO_RESPONSE_FAMILIES
        self._stage1 = None
        self._stage2 = None
        self._metadata: dict = {}
        self.load()

    # -- construction ------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Predictor":
        """Build a predictor with optional environment overrides.

        AI_SOAR_GATE_THRESHOLD      e.g. 0.3
        AI_SOAR_CONFIDENCE_FLOOR    e.g. 0.7
        AI_SOAR_AUTO_FAMILIES       e.g. "BruteForce,DDoS,DoS,PortScan,WebAttack"
        """
        gate = os.getenv("AI_SOAR_GATE_THRESHOLD")
        floor = os.getenv("AI_SOAR_CONFIDENCE_FLOOR")
        families = os.getenv("AI_SOAR_AUTO_FAMILIES")
        return cls(
            gate_threshold=float(gate) if gate else DEFAULT_GATE_THRESHOLD,
            confidence_floor=float(floor) if floor else DEFAULT_CONFIDENCE_FLOOR,
            auto_families=(
                frozenset(f.strip() for f in families.split(",") if f.strip())
                if families
                else None
            ),
        )

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        stage1_path = self.models_dir / STAGE1_FILENAME
        stage2_path = self.models_dir / STAGE2_FILENAME
        missing = [str(p) for p in (stage1_path, stage2_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"trained model(s) not found: {missing}. Run scripts/train_models.py first."
            )
        self._stage1 = load_binary_model(stage1_path)
        self._stage2 = load_multiclass_model(stage2_path)

        meta_path = self.models_dir / METADATA_FILENAME
        self._metadata = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )

        # Guard: the on-disk feature space must be the one this code assumes.
        meta_features = self._metadata.get("feature_columns")
        if meta_features and list(meta_features) != list(FEATURE_COLUMNS):
            raise ValueError(
                "feature space mismatch between model_metadata.json and FEATURE_COLUMNS - "
                "the served models were trained on a different feature set."
            )
        log.info(
            "predictor ready: stage1=%s stage2=%s classes=%s",
            stage1_path.name,
            stage2_path.name,
            list(self._stage2.classes_),
        )

    @property
    def loaded(self) -> bool:
        return self._stage1 is not None and self._stage2 is not None

    @property
    def families(self) -> list[str]:
        """Attack families the stage-2 model can output."""
        return [str(c) for c in self._stage2.classes_] if self._stage2 else []

    @property
    def version(self) -> str:
        trained_at = self._metadata.get("trained_at", "unknown")
        rows = self._metadata.get("rows", {})
        n_train = rows.get("train", "?")
        return f"{trained_at} (train_rows={n_train})"

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok" if self.loaded else "degraded",
            model_loaded=self.loaded,
            n_features_expected=len(FEATURE_COLUMNS),
            model_version=self.version,
            gate_threshold=self.gate_threshold,
            auto_response_families=sorted(self.auto_families),
        )

    # -- scoring -----------------------------------------------------------
    def _feature_vector(self, features: dict[str, float]) -> np.ndarray:
        """Re-order the request into the canonical column order (1 x 70)."""
        return np.array(
            [[float(features[name]) for name in FEATURE_COLUMNS]], dtype=np.float64
        )

    def predict(self, request: FlowFeaturesRequest) -> PredictionResponse:
        """Score one flow and attach the governed decision."""
        started = time.perf_counter()
        X = self._feature_vector(request.features)

        # Real taps emit inf/nan (CICIDS2017 itself had 4,376 inf cells).
        # Never let one bad flow produce a confident-looking wrong verdict.
        if not np.isfinite(X).all():
            return self._response(
                request,
                is_malicious=False,
                gate_probability=0.0,
                family=BENIGN_LABEL,
                family_probabilities={},
                confidence=0.0,
                decision="unknown_queue",
                reason="non-finite feature values (inf/nan) in request",
                started=started,
            )

        gate_p = float(malicious_probability(self._stage1, X)[0])

        if gate_p < self.gate_threshold:
            return self._response(
                request,
                is_malicious=False,
                gate_probability=gate_p,
                family=BENIGN_LABEL,
                family_probabilities={},
                confidence=0.0,
                decision="log_only",
                reason=f"gate probability {gate_p:.4f} < threshold {self.gate_threshold}",
                started=started,
            )

        probs = predict_proba_families(self._stage2, X).iloc[0]
        family = str(probs.idxmax())
        confidence = float(probs.max())
        family_probabilities = {str(k): round(float(v), 6) for k, v in probs.items()}
        decision, reason = self._policy(family, confidence)

        return self._response(
            request,
            is_malicious=True,
            gate_probability=gate_p,
            family=family,
            family_probabilities=family_probabilities,
            confidence=round(confidence, 6),
            decision=decision,
            reason=reason,
            started=started,
        )

    def predict_features(
        self, features: dict[str, float], request_id: Optional[str] = None
    ) -> PredictionResponse:
        """Convenience wrapper for callers that already hold a plain dict."""
        return self.predict(FlowFeaturesRequest(request_id=request_id, features=features))

    def _policy(self, family: str, confidence: float) -> tuple[str, str]:
        """Map (family, confidence) -> (decision, human-readable reason)."""
        if confidence < self.confidence_floor:
            return (
                "unknown_queue",
                f"confidence {confidence:.3f} < floor {self.confidence_floor:.2f}: "
                f"{family} not reliable enough to act on",
            )
        if family in self.auto_families:
            return (
                "auto_response",
                f"{family} cleared the automation allowlist (hardened F1 >= 0.90, "
                f"confidence {confidence:.3f})",
            )
        return (
            "human_approval",
            f"{family} is not in the automation allowlist (weak hardened performance); "
            f"human approval required",
        )

    def _response(
        self,
        request: FlowFeaturesRequest,
        started: float,
        reason: str = "",
        **kwargs,
    ) -> PredictionResponse:
        return PredictionResponse(
            request_id=request.request_id,
            gate_threshold=self.gate_threshold,
            model_version=self.version,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            decision_reason=reason,
            **kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Predictor(loaded={self.loaded}, families={len(self.families)}, "
            f"gate_threshold={self.gate_threshold}, confidence_floor={self.confidence_floor})"
        )