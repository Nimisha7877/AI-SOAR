"""n8n orchestration client — pushes SOAR incident events to an n8n webhook.

Why this module exists
----------------------
Detection, policy and response stay in Python. n8n owns the *human* side of the
loop: notify the SOC, collect an approval decision, and call the SOAR API back.
This module is the one-way door from Python into n8n.

Contract used by ``ai_soar.inference.api``::

    from ai_soar.orchestration.n8n import notify_incident
    notify_incident(incident) -> bool      # never raises

Design rules
------------
1. **Fail-open.** Nothing here may raise into the response path. If n8n is
   disabled, unreachable, slow, or answers 5xx, ``notify_incident`` returns
   ``False`` and the incident is still created, audited and acted upon. Losing a
   notification must never lose an incident.
2. **No retry storm.** Exactly one attempt with a short timeout (default 3 s,
   override with ``AI_SOAR_N8N_TIMEOUT``). The caller is a millisecond decision
   path; a notification must not turn it into a timeout risk. Retries belong in
   the n8n workflow, not here.
3. **Disabled by default.** ``n8n.enabled`` is ``false`` in ``config/settings.yaml``,
   so clone-and-run works with no n8n installed. Flip it once n8n is up.
4. **Self-describing payload.** Every event carries the SOAR API callback URLs
   (detail / approve / dismiss), so a workflow's approval button can POST straight
   back without hardcoding a host. Override the advertised base with
   ``AI_SOAR_API_BASE_URL`` when the API is reached through a reverse proxy.
5. **Stable keys.** The payload is hand-built rather than dumping the pydantic
   model: workflow expressions like ``{{ $json.severity }}`` must keep working
   when the incident schema grows a field.

Event types (``event`` field, used by the workflow's Switch node)
----------------------------------------------------------------
``approval_required`` a human must release held actions
``approved``          held actions were released by a human
``dismissed``         a human rejected the incident
``resolved``          auto-response finished with nothing pending
``opened``            anything else (defensive default)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Optional

from ai_soar.config import get_settings
from ai_soar.response.schemas import (
    ActionStatus,
    Incident,
    IncidentStatus,
)
from ai_soar.utils.logging import get_logger

try:  # httpx is a pinned dependency, but orchestration is optional by design
    import httpx
except Exception:  # noqa: BLE001 - degrade to "notifications unavailable"
    httpx = None  # type: ignore[assignment]

log = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 3.0
ENV_TIMEOUT = "AI_SOAR_N8N_TIMEOUT"
ENV_API_BASE = "AI_SOAR_API_BASE_URL"

EVENT_APPROVAL_REQUIRED = "approval_required"
EVENT_APPROVED = "approved"
EVENT_DISMISSED = "dismissed"
EVENT_RESOLVED = "resolved"
EVENT_OPENED = "opened"


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------
def infer_event(incident: Incident) -> str:
    """Derive the event type from the incident's current state.

    Order matters: an incident that was just approved still has status
    ``auto_resolved``, so the presence of an ``approved_by`` actor is checked
    before the generic resolved case.
    """
    if incident.status == IncidentStatus.DISMISSED:
        return EVENT_DISMISSED
    if incident.status == IncidentStatus.PENDING_APPROVAL or incident.pending_actions():
        return EVENT_APPROVAL_REQUIRED
    if any(a.approved_by for a in incident.actions):
        return EVENT_APPROVED
    if incident.status == IncidentStatus.AUTO_RESOLVED:
        return EVENT_RESOLVED
    return EVENT_OPENED


def api_base_url(settings: Any = None) -> str:
    """Advertised SOAR API base, used to build callback URLs for n8n.

    ``settings.api.host`` is ``0.0.0.0`` (bind-all), which is not callable, so the
    port is reused with ``localhost`` unless ``AI_SOAR_API_BASE_URL`` says otherwise.
    """
    override = os.environ.get(ENV_API_BASE, "").strip()
    if override:
        return override.rstrip("/")
    settings = settings or get_settings()
    return f"http://localhost:{settings.api.port}"


def build_payload(
    incident: Incident,
    event: Optional[str] = None,
    api_base: Optional[str] = None,
) -> dict:
    """Flat, workflow-friendly view of one incident.

    Only JSON-native types are emitted (no enums, no datetimes), so n8n
    expressions work without coercion.
    """
    event = event or infer_event(incident)
    base = (api_base or api_base_url()).rstrip("/")
    pending = incident.pending_actions()

    return {
        "event": event,
        "incident_id": incident.incident_id,
        "family": incident.family,
        "severity": incident.severity.value,
        "status": incident.status.value,
        "decision": incident.decision,
        "decision_reason": incident.decision_reason,
        "confidence": round(float(incident.confidence), 6),
        "gate_probability": round(float(incident.gate_probability), 6),
        "model_version": incident.model_version,
        "request_id": incident.request_id,
        "created_at": incident.created_at.isoformat(),
        "needs_human": bool(pending) or incident.status == IncidentStatus.PENDING_APPROVAL,
        "pending_actions": [a.action for a in pending],
        "playbook": list(incident.playbook),
        "actions": [
            {
                "action": a.action,
                "actuator": a.actuator,
                "status": a.status.value,
                "simulated": bool(a.simulated),
                "approved_by": a.approved_by,
                "message": a.message,
            }
            for a in incident.actions
        ],
        "notes": list(incident.notes)[-5:],
        "callback": {
            "detail": f"{base}/incidents/{incident.incident_id}",
            "approve": f"{base}/incidents/{incident.incident_id}/approve",
            "dismiss": f"{base}/incidents/{incident.incident_id}/dismiss",
        },
        "sent_at": datetime.now().isoformat(),
        "source": "ai-soar",
    }


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
class N8nClient:
    """Posts incident events to an n8n webhook. One attempt, short timeout.

    Constructed lazily so importing this module never opens a socket. A single
    ``httpx.Client`` is reused (connection pooling) and is thread-safe, which
    matters because FastAPI runs the endpoints in a threadpool.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        cfg = settings.n8n
        self.enabled = cfg.enabled if enabled is None else bool(enabled)
        self.webhook_url = (webhook_url or cfg.webhook_url).rstrip("/")
        self.timeout = timeout if timeout is not None else _timeout_from_env()

        secret = api_key
        if secret is None and cfg.api_key is not None:
            secret = cfg.api_key.get_secret_value()
        self._api_key = secret or None
        self._http: Optional[Any] = None

    # -- internals ---------------------------------------------------------
    def _client(self) -> Optional[Any]:
        if httpx is None:
            log.warning("httpx not installed - n8n notifications unavailable")
            return None
        if self._http is None:
            headers = {"Content-Type": "application/json", "User-Agent": "ai-soar/0.1"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._http = httpx.Client(timeout=httpx.Timeout(self.timeout), headers=headers)
        return self._http

    def close(self) -> None:
        if self._http is not None:
            try:
                self._http.close()
            finally:
                self._http = None

    # -- public ------------------------------------------------------------
    def post(self, payload: dict) -> bool:
        """Send one event. Returns True only on a 2xx answer. Never raises."""
        if not self.enabled:
            log.debug("n8n disabled (n8n.enabled=false); event '%s' not sent", payload.get("event"))
            return False
        client = self._client()
        if client is None:
            return False
        try:
            response = client.post(self.webhook_url, json=payload)
        except Exception as exc:  # noqa: BLE001 - network failure is not fatal
            log.warning(
                "n8n webhook unreachable at %s (%s): %s", self.webhook_url, type(exc).__name__, exc
            )
            return False

        if 200 <= response.status_code < 300:
            log.info(
                "n8n notified: event=%s incident=%s status=%d",
                payload.get("event"),
                payload.get("incident_id"),
                response.status_code,
            )
            return True
        log.warning(
            "n8n webhook answered %d for incident %s: %s",
            response.status_code,
            payload.get("incident_id"),
            response.text[:200],
        )
        return False

    def notify(self, incident: Incident, event: Optional[str] = None) -> bool:
        """Build the payload for one incident and post it."""
        try:
            payload = build_payload(incident, event=event)
        except Exception as exc:  # noqa: BLE001 - a malformed incident must not break response
            log.warning("could not build n8n payload for %s: %s", getattr(incident, "incident_id", "?"), exc)
            return False
        return self.post(payload)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"N8nClient(enabled={self.enabled}, webhook_url={self.webhook_url!r}, "
            f"timeout={self.timeout}s, api_key={'set' if self._api_key else 'none'})"
        )


def _timeout_from_env() -> float:
    raw = os.environ.get(ENV_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %.1fs", ENV_TIMEOUT, raw, DEFAULT_TIMEOUT_SECONDS)
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


_client: Optional[N8nClient] = None


def get_client(reset: bool = False) -> N8nClient:
    """Process-wide client (settings are read once, at first use)."""
    global _client
    if _client is None or reset:
        if _client is not None:
            _client.close()
        _client = N8nClient()
    return _client


def notify_incident(incident: Incident, event: Optional[str] = None) -> bool:
    """Module-level entry point used by the API layer. Never raises."""
    if incident is None:
        return False
    return get_client().notify(incident, event=event)


# ---------------------------------------------------------------------------
# self-test: python -m ai_soar.orchestration.n8n --test
# ---------------------------------------------------------------------------
def _synthetic_incident() -> Incident:
    """A fake incident for wiring tests, so no model or dataset is needed."""
    from ai_soar.response.schemas import ActionResult, Severity

    return Incident(
        incident_id="INC-SELFTEST",
        family="DDoS",
        confidence=0.999,
        gate_probability=0.998,
        decision="human_approval",
        decision_reason="self-test event from ai_soar.orchestration.n8n",
        severity=Severity.CRITICAL,
        status=IncidentStatus.PENDING_APPROVAL,
        playbook=["notify_soc", "firewall_block_source"],
        actions=[
            ActionResult(
                action="firewall_block_source",
                actuator="simulated",
                status=ActionStatus.PENDING_APPROVAL,
                simulated=True,
                message="destructive action held for human approval (self-test)",
            )
        ],
        model_version="self-test",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a test event to the n8n webhook.")
    parser.add_argument("--test", action="store_true", help="post a synthetic incident event")
    parser.add_argument("--url", default=None, help="override the webhook URL for this call")
    parser.add_argument("--force", action="store_true", help="send even if n8n.enabled is false")
    parser.add_argument("--show", action="store_true", help="print the payload instead of sending")
    args = parser.parse_args()

    client = N8nClient(webhook_url=args.url, enabled=True if args.force else None)
    incident = _synthetic_incident()

    if args.show or not args.test:
        import json

        print(json.dumps(build_payload(incident), indent=2))
        print(f"\n{client}")
        if not args.test:
            print("\n(nothing sent - pass --test to post it)")
        return 0

    ok = client.notify(incident)
    print(f"webhook: {client.webhook_url}")
    print(f"delivered: {ok}")
    if not ok:
        print(
            "\nNot delivered. Check, in order:\n"
            "  1. n8n is running and the workflow is ACTIVE (a disabled workflow's\n"
            "     /webhook/... path returns 404; only /webhook-test/... works while editing)\n"
            f"  2. n8n.webhook_path in config/settings.yaml matches your Webhook node\n"
            "  3. n8n.enabled=true in config/settings.yaml (or pass --force)\n"
            "  4. AI_SOAR_N8N_BASE_URL points at the right host/port"
        )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
