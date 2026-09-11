"""FastAPI application: the AI SOAR inference + response service.

Endpoints
---------
GET  /health                       model readiness + policy in force (n8n probes this first)
POST /predict                      score ONE flow -> prediction + governed decision
POST /predict/batch                score up to ``MAX_BATCH`` flows in one round trip
POST /ingest                       score ONE flow AND run the response engine -> incident
GET  /incidents                    current incident states (filter by status/family/pending)
GET  /incidents/summary            tallies + ids awaiting human approval
GET  /incidents/{incident_id}      one incident, full response trail
POST /incidents/{id}/approve       human releases the held actions (event-sourced)
POST /incidents/{id}/dismiss       human rejects the incident (held actions skipped)

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
- ``/predict`` stays scoring-only (backwards compatible). ``/ingest`` is the
  operational entry point: it scores, then hands the prediction to the response
  engine, which is what actually creates and audits an incident. A live tap or a
  replay driver should call ``/ingest``; a benchmarking harness should call
  ``/predict``.
- The response engine is built LAZILY and guarded: if actuator config or the
  incident store cannot be created, ``/health`` and ``/predict`` keep working and
  only the incident endpoints answer 503. Inference must not die because the
  response layer is misconfigured.
- ``GET /incidents`` folds the append-only JSONL log (last line wins per
  incident id), which is an O(n) scan. Fine at demo volume; see PROJECT_BRIEF §7
  for the staged upgrade (rotation + in-memory index, then SQLite WAL).
- ``/incidents/summary`` is declared BEFORE ``/incidents/{incident_id}``:
  FastAPI matches routes in order, so otherwise "summary" would be parsed as an
  incident id.
- Explanations are NOT generated here on purpose: an LLM call costs seconds and
  would turn a millisecond decision path into a timeout risk. Run
  ``scripts/explain_incidents.py`` against the incident log instead.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ai_soar.config import get_settings
from ai_soar.inference.predictor import Predictor
from ai_soar.inference.schemas import FlowFeaturesRequest, HealthResponse, PredictionResponse
from ai_soar.response.engine import ResponseEngine
from ai_soar.response.schemas import Incident, IncidentStatus
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

MAX_BATCH = 500
MAX_LIMIT = 500

_predictor: Optional[Predictor] = None
_engine: Optional[ResponseEngine] = None


class BatchRequest(BaseModel):
    """Several flows in one call (e.g. a CICFlowMeter flush or a pcap window)."""

    flows: list[FlowFeaturesRequest] = Field(min_length=1, max_length=MAX_BATCH)


class IngestResponse(BaseModel):
    """Prediction + the incident the response engine raised (None for benign)."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: PredictionResponse
    incident: Optional[Incident] = Field(
        default=None, description="None when the gate called the flow benign"
    )
    notified: bool = Field(
        default=False, description="True when the n8n webhook accepted the alert"
    )


class ApproveRequest(BaseModel):
    """Who released the held actions. The approver id is written to the audit log."""

    approver: str = Field(default="api-user", min_length=1, max_length=64)


class DismissRequest(BaseModel):
    """Who rejected the incident, and why. Both are written to the audit log."""

    approver: str = Field(default="api-user", min_length=1, max_length=64)
    reason: str = Field(default="", max_length=500)


class IncidentListResponse(BaseModel):
    count: int
    total_in_log: int = Field(description="distinct incident ids in the store")
    filters: dict[str, str] = Field(default_factory=dict)
    incidents: list[Incident] = Field(default_factory=list)


def _notify(incident: Incident) -> bool:
    """Best-effort n8n webhook call. NEVER raises into the response path.

    The orchestration module is optional: if it is not installed, or n8n is
    disabled in config, or the webhook is unreachable, the SOAR still works end
    to end. Fail-open by design - losing a notification must not lose an incident.
    """
    try:
        from ai_soar.orchestration.n8n import notify_incident  # lazy: optional module
    except Exception:  # noqa: BLE001 - module absent or not importable
        return False
    try:
        return bool(notify_incident(incident))
    except Exception as exc:  # noqa: BLE001 - orchestration is best-effort
        log.warning("n8n notification failed for %s (non-fatal): %s", incident.incident_id, exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global _predictor, _engine
    _predictor = Predictor.from_env()
    log.info("inference service ready: %s", _predictor)

    # Response engine is optional at startup: a misconfigured actuator layer must
    # not take the inference endpoints down with it.
    try:
        _engine = ResponseEngine()
        log.info("response engine ready: backend=%s", _engine.backend.name)
    except Exception as exc:  # noqa: BLE001
        _engine = None
        log.warning("response engine unavailable, incident endpoints disabled: %s", exc)

    yield
    _predictor = None
    _engine = None
    log.info("inference service stopped")


def _cors_origins() -> list[str]:
    """Demo-grade default, overridable via AI_SOAR_CORS_ORIGINS (comma-separated).

    Server-to-server callers (n8n, a replay driver) do not need CORS at all; this
    only matters if a browser page calls the API. Set AI_SOAR_CORS_ORIGINS=* only
    behind an authenticated reverse proxy.
    """
    raw = os.environ.get("AI_SOAR_CORS_ORIGINS", "").strip()
    if raw:
        return ["*"] if raw == "*" else [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


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
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def predictor() -> Predictor:
        if _predictor is None:
            raise HTTPException(status_code=503, detail="predictor not loaded")
        return _predictor

    def engine() -> ResponseEngine:
        if _engine is None:
            raise HTTPException(
                status_code=503,
                detail="response engine unavailable (check actuator/incident-store config)",
            )
        return _engine

    # -- ops -----------------------------------------------------------------
    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Readiness probe: are models loaded, and what policy is in force?

        Any failure here is a 503, never a 500: an orchestrator (n8n) polls this
        endpoint and must be able to tell "not ready" from "crashed".
        """
        try:
            return predictor().health()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("health probe failed")
            raise HTTPException(status_code=503, detail=f"health check failed: {exc}") from exc

    # -- inference -----------------------------------------------------------
    @application.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(request: FlowFeaturesRequest) -> PredictionResponse:
        """Score one flow. Requires exactly the 70 canonical features.

        Scoring only - no incident is created. Use ``/ingest`` for that.
        """
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
        engine_ref = predictor()
        out: list[PredictionResponse] = []
        try:
            for flow in request.flows:
                out.append(engine_ref.predict(flow))
        except Exception as exc:  # noqa: BLE001
            log.exception("batch inference failed after %d flows", len(out))
            raise HTTPException(status_code=500, detail=f"batch inference failed: {exc}") from exc
        return out

    # -- detection + response (the operational entry point) -------------------
    @application.post("/ingest", response_model=IngestResponse, tags=["response"])
    def ingest(request: FlowFeaturesRequest) -> IngestResponse:
        """Score one flow AND run the governed response.

        Returns ``incident=None`` for benign flows: the response engine stores
        nothing for them, which is why the incident log grows at incident rate
        rather than flow rate. For a held decision (``human_approval`` or
        ``unknown_queue``) the incident comes back with status
        ``pending_approval`` and its destructive actions unexecuted until
        ``/incidents/{id}/approve`` is called.
        """
        response_engine = engine()
        try:
            prediction = predictor().predict(request)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest: inference failed for request_id=%s", request.request_id)
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

        try:
            incident = response_engine.handle(prediction)
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest: response engine failed for request_id=%s", request.request_id)
            raise HTTPException(status_code=500, detail=f"response failed: {exc}") from exc

        notified = _notify(incident) if incident is not None else False
        return IngestResponse(prediction=prediction, incident=incident, notified=notified)

    # -- incident log ---------------------------------------------------------
    @application.get("/incidents", response_model=IncidentListResponse, tags=["incidents"])
    def list_incidents(
        status: Optional[str] = Query(
            default=None, description="filter by incident status, e.g. pending_approval"
        ),
        family: Optional[str] = Query(default=None, description="filter by attack family"),
        pending_only: bool = Query(
            default=False, description="only incidents with actions awaiting approval"
        ),
        limit: int = Query(default=50, ge=1, le=MAX_LIMIT, description="most recent first"),
    ) -> IncidentListResponse:
        """Current state per incident, newest first.

        Valid ``status`` values are the ``IncidentStatus`` enum members. The log is
        append-only and event-sourced, so this folds it down to one record per
        incident id (last write wins) before filtering.
        """
        valid = {s.value for s in IncidentStatus}
        if status is not None and status not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"unknown status '{status}'; valid: {sorted(valid)}",
            )
        response_engine = engine()

        try:
            states = response_engine.latest()
        except Exception as exc:  # noqa: BLE001
            log.exception("incident listing failed")
            raise HTTPException(status_code=500, detail=f"incident store read failed: {exc}") from exc

        items = list(states.values())
        if status is not None:
            items = [i for i in items if i.status.value == status]
        if family is not None:
            items = [i for i in items if i.family == family]
        if pending_only:
            items = [
                i
                for i in items
                if i.status == IncidentStatus.PENDING_APPROVAL or i.pending_actions()
            ]
        items.sort(key=lambda i: i.created_at, reverse=True)

        return IncidentListResponse(
            count=len(items[:limit]),
            total_in_log=len(states),
            filters={
                k: v
                for k, v in {
                    "status": status or "",
                    "family": family or "",
                    "pending_only": "true" if pending_only else "",
                }.items()
                if v
            },
            incidents=items[:limit],
        )

    @application.get("/incidents/summary", tags=["incidents"])
    def incidents_summary() -> dict:
        """Tallies by status and family + the ids awaiting human approval.

        Declared before ``/incidents/{incident_id}`` on purpose (route order).
        """
        response_engine = engine()
        try:
            return response_engine.summary()
        except Exception as exc:  # noqa: BLE001
            log.exception("incident summary failed")
            raise HTTPException(status_code=500, detail=f"incident store read failed: {exc}") from exc

    @application.get("/incidents/{incident_id}", response_model=Incident, tags=["incidents"])
    def get_incident(incident_id: str) -> Incident:
        """One incident with its full response trail (actions, approvals, notes)."""
        response_engine = engine()
        try:
            state = response_engine.latest().get(incident_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("incident fetch failed for %s", incident_id)
            raise HTTPException(status_code=500, detail=f"incident store read failed: {exc}") from exc
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown incident '{incident_id}'")
        return state

    @application.post("/incidents/{incident_id}/approve", response_model=Incident, tags=["incidents"])
    def approve_incident(
        incident_id: str,
        request: Optional[ApproveRequest] = Body(
            default=None, description="optional: approver id (defaults to 'api-user')"
        ),
    ) -> Incident:
        """Human releases the held actions of one incident.

        The body is OPTIONAL: ``POST`` with no payload works and records the
        approver as ``api-user``. n8n's HTTP Request node can therefore call this
        without building a JSON body, while a real UI can send the analyst's id.

        Event-sourced: the engine re-appends the updated incident rather than
        editing history, so the audit log keeps every intermediate state.
        404 if the id is unknown, 409 if nothing is pending approval.
        """
        body = request or ApproveRequest()
        response_engine = engine()
        try:
            current = response_engine.latest().get(incident_id)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("approve: incident store read failed for %s", incident_id)
            raise HTTPException(
                status_code=500, detail=f"incident store read failed: {exc}"
            ) from exc
        if current is None:
            raise HTTPException(status_code=404, detail=f"unknown incident '{incident_id}'")
        if not current.pending_actions() and current.status != IncidentStatus.PENDING_APPROVAL:
            raise HTTPException(
                status_code=409,
                detail=f"incident '{incident_id}' has nothing pending (status={current.status.value})",
            )

        try:
            updated = response_engine.approve(incident_id, body.approver)
        except Exception as exc:  # noqa: BLE001
            log.exception("approve failed for %s", incident_id)
            raise HTTPException(status_code=500, detail=f"approve failed: {exc}") from exc
        if updated is None:
            raise HTTPException(status_code=409, detail="no held actions to approve")
        notified = _notify(updated)
        log.info("incident %s approved by %s (n8n notified=%s)", incident_id, body.approver, notified)
        return updated

    @application.post("/incidents/{incident_id}/dismiss", response_model=Incident, tags=["incidents"])
    def dismiss_incident(
        incident_id: str,
        request: Optional[DismissRequest] = Body(
            default=None, description="optional: approver id + reason"
        ),
    ) -> Incident:
        """Human rejects the incident: held actions are skipped, case closed.

        Body is OPTIONAL (see ``/approve``); an empty ``POST`` dismisses with
        approver ``api-user`` and no reason.

        404 if the id is unknown. Dismissing an already-resolved incident is
        allowed and recorded - an analyst closing a false positive after the fact
        is a legitimate audit event.
        """
        body = request or DismissRequest()
        response_engine = engine()
        try:
            known = response_engine.latest().get(incident_id) is not None
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("dismiss: incident store read failed for %s", incident_id)
            raise HTTPException(
                status_code=500, detail=f"incident store read failed: {exc}"
            ) from exc
        if not known:
            raise HTTPException(status_code=404, detail=f"unknown incident '{incident_id}'")

        try:
            updated = response_engine.dismiss(incident_id, body.approver, body.reason)
        except Exception as exc:  # noqa: BLE001
            log.exception("dismiss failed for %s", incident_id)
            raise HTTPException(status_code=500, detail=f"dismiss failed: {exc}") from exc
        if updated is None:
            raise HTTPException(status_code=404, detail=f"unknown incident '{incident_id}'")
        notified = _notify(updated)
        log.info("incident %s dismissed by %s (n8n notified=%s)", incident_id, body.approver, notified)
        return updated

    @application.get("/", tags=["ops"])
    def root() -> dict:
        return {
            "service": "ai-soar-inference",
            "docs": "/docs",
            "response_engine": "ready" if _engine is not None else "unavailable",
            "endpoints": [
                "/health",
                "/predict",
                "/predict/batch",
                "/ingest",
                "/incidents",
                "/incidents/summary",
                "/incidents/{incident_id}",
                "/incidents/{incident_id}/approve",
                "/incidents/{incident_id}/dismiss",
            ],
        }

    return application


app = create_app()
