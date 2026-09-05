"""FastAPI application: the AI SOAR inference service.

Endpoints
---------
GET  /health         model readiness + policy in force (n8n probes this first)
POST /predict        score ONE flow -> prediction + governed decision
POST /predict/batch  score up to ``MAX_BATCH`` flows in one round trip

Design notes
------------
- The predictor is loaded ONCE at startup (lifespan) and reused for every
  request. Loading LightGBM models per request would cost seconds; scoring a
  flow costs well under a millisecond.
- Endpoints are plain ``def`` (not ``async def``) so FastAPI runs them in a
  threadpool: CPU-bound numpy/LightGBM work never blocks the event loop.
- Pydantic validation failures (wrong feature space, non-numeric values) are
  answered with HTTP 422 by FastAPI before the model ever sees the payload.
- Anything unexpected inside inference becomes HTTP 500 with a logged traceback,
  never a partial prediction - a wrong verdict delivered confidently is worse
  than an explicit failure.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_soar.config import get_settings
from ai_soar.inference.predictor import Predictor
from ai_soar.inference.schemas import FlowFeaturesRequest, HealthResponse, PredictionResponse
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

MAX_BATCH = 500

_predictor: Optional[Predictor] = None


class BatchRequest(BaseModel):
    """Several flows in one call (e.g. a CICFlowMeter flush or a pcap window)."""

    flows: list[FlowFeaturesRequest] = Field(min_length=1, max_length=MAX_BATCH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global _predictor
    _predictor = Predictor.from_env()
    log.info("inference service ready: %s", _predictor)
    yield
    _predictor = None
    log.info("inference service stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="AI SOAR Inference",
        description=(
            "Two-stage network-flow classifier (binary gate + family classifier) "
            "with an evidence-based response policy: auto_response / human_approval / "
            "unknown_queue. Metrics context: stratified scores are an UPPER BOUND; see "
            "artifacts/reports/leakage_audit_report.json."
        ),
        version=settings.version,
        lifespan=lifespan,
    )

    def predictor() -> Predictor:
        if _predictor is None:
            raise HTTPException(status_code=503, detail="predictor not loaded")
        return _predictor

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Readiness probe: are models loaded, and what policy is in force?"""
        return predictor().health()

    @application.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(request: FlowFeaturesRequest) -> PredictionResponse:
        """Score one flow. Requires exactly the 70 canonical features."""
        try:
            return predictor().predict(request)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a partial verdict
            log.exception("inference failed for request_id=%s", request.request_id)
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    @application.post(
        "/predict/batch", response_model=list[PredictionResponse], tags=["inference"]
    )
    def predict_batch(request: BatchRequest) -> list[PredictionResponse]:
        """Score up to 500 flows. Responses come back in request order."""
        engine = predictor()
        out: list[PredictionResponse] = []
        try:
            for flow in request.flows:
                out.append(engine.predict(flow))
        except Exception as exc:  # noqa: BLE001
            log.exception("batch inference failed after %d flows", len(out))
            raise HTTPException(status_code=500, detail=f"batch inference failed: {exc}") from exc
        return out

    @application.get("/", tags=["ops"])
    def root() -> dict:
        return {
            "service": "ai-soar-inference",
            "docs": "/docs",
            "endpoints": ["/health", "/predict", "/predict/batch"],
        }

    return application


app = create_app()