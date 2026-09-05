"""Simulated actuators: the "hands" of the SOAR (no real infrastructure).

An actuator turns a playbook step into an effect on the world. This backend
performs NO real change; it records exactly what a real backend would have
done, using the same status vocabulary, so demos, thesis screenshots and CI
runs stay honest and safe.

Package layout (why two folders):
- ``actuators/``  = things that ACT on the world (this module; real vendor
  integrations will land here later as e.g. ``live_firewall.py``)
- ``backends/``   = things that STORE/FORWARD incident records (JSONL store,
  n8n webhook) - added in a later file of this step

Swapping to real infrastructure means implementing the same two-method contract
(:class:`ResponseBackend`) against a firewall API or EDR SDK; the response
engine above it does not change at all.

Every handler returns a plain operator-facing sentence; those sentences end up
in ``incidents.jsonl`` and in the LLM explainer's context (Step 8).
"""

from __future__ import annotations

from typing import Callable, Optional

from ai_soar.config import get_settings
from ai_soar.response.schemas import ActionResult, ActionStatus, Incident
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Simulated action handlers. Key = playbook action name.
# TODO(real infra): move each handler into its own vendor module under this
# package and keep only the dispatch table here.
# --------------------------------------------------------------------------
SimHandler = Callable[[Incident], str]


def _rate_limit(inc: Incident) -> str:
    return (
        f"[SIMULATED] rate-limited the offending source to 10 req/s for incident "
        f"{inc.incident_id} ({inc.family}, confidence {inc.confidence:.2f})"
    )


def _firewall_block(inc: Incident) -> str:
    return (
        f"[SIMULATED] firewall rule created: DROP traffic from the attacking source "
        f"for 60 minutes (incident {inc.incident_id}, {inc.family})"
    )


def _null_route(inc: Incident) -> str:
    return (
        f"[SIMULATED] null-routed the flooded destination for 30 minutes to protect "
        f"upstream capacity (incident {inc.incident_id}, {inc.family})"
    )


def _host_isolation(inc: Incident) -> str:
    return (
        f"[SIMULATED] EDR asked to isolate the implicated host into a quarantine VLAN "
        f"(incident {inc.incident_id}, {inc.family})"
    )


def _waf_block(inc: Incident) -> str:
    return (
        f"[SIMULATED] WAF virtual patch enabled for the targeted web application "
        f"(incident {inc.incident_id}, {inc.family})"
    )


def _watchlist(inc: Incident) -> str:
    return (
        f"[SIMULATED] source added to a 24-hour SOC watchlist with elevated logging "
        f"(incident {inc.incident_id}, {inc.family})"
    )


def _forensics(inc: Incident) -> str:
    return (
        f"[SIMULATED] packet + process capture scheduled on the implicated host for "
        f"forensic review (incident {inc.incident_id}, {inc.family})"
    )


def _ticket(inc: Incident) -> str:
    return (
        f"[SIMULATED] ticket created: '{inc.family} detected by AI SOAR' severity="
        f"{inc.severity.value} decision={inc.decision} (incident {inc.incident_id})"
    )


def _notify_soc(inc: Incident) -> str:
    return (
        f"[SIMULATED] SOC channel notified: {inc.family} @ confidence {inc.confidence:.2f}, "
        f"decision={inc.decision} (incident {inc.incident_id})"
    )


SIMULATED_ACTIONS: dict[str, SimHandler] = {
    "rate_limit_source": _rate_limit,
    "firewall_block_source": _firewall_block,
    "null_route_target": _null_route,
    "host_isolation": _host_isolation,
    "waf_block": _waf_block,
    "add_to_watchlist": _watchlist,
    "capture_forensics": _forensics,
    "create_ticket": _ticket,
    "notify_soc": _notify_soc,
}


class ResponseBackend:
    """Contract every backend (simulated or live) must satisfy."""

    name: str = "abstract"

    def supports(self, action: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def execute(self, incident: Incident, action: str) -> ActionResult:  # pragma: no cover
        raise NotImplementedError


class SimulatedResponseBackend(ResponseBackend):
    """Records effects without touching any real system."""

    name = "simulated"

    def supports(self, action: str) -> bool:
        return action in SIMULATED_ACTIONS

    def execute(self, incident: Incident, action: str) -> ActionResult:
        handler = SIMULATED_ACTIONS.get(action)
        if handler is None:
            log.warning("unknown action '%s' for incident %s", action, incident.incident_id)
            return ActionResult(
                action=action,
                actuator=self.name,
                status=ActionStatus.FAILED,
                simulated=True,
                message=f"unknown action '{action}' - playbook/backend mismatch",
            )
        try:
            message = handler(incident)
            log.info("%s", message)
            return ActionResult(
                action=action,
                actuator=self.name,
                status=ActionStatus.SIMULATED,
                simulated=True,
                message=message,
            )
        except Exception as exc:  # noqa: BLE001 - an actuator crash must not kill the incident
            log.exception("actuator '%s' raised", action)
            return ActionResult(
                action=action,
                actuator=self.name,
                status=ActionStatus.FAILED,
                simulated=True,
                message=f"actuator error: {exc}",
            )


class LiveResponseBackend(ResponseBackend):
    """Placeholder for real infrastructure. Refuses to run until implemented.

    Deliberate: silently pretending to be live would be dangerous. When real
    integrations land, implement ``supports``/``execute`` per action against
    the vendor API and return ``ActionStatus.EXECUTED`` on success.
    """

    name = "live"

    def supports(self, action: str) -> bool:
        return False

    def execute(self, incident: Incident, action: str) -> ActionResult:
        return ActionResult(
            action=action,
            actuator=self.name,
            status=ActionStatus.FAILED,
            simulated=False,
            message=(
                "live backend selected but no real integration is implemented for "
                f"'{action}'; set response.backend=simulated in config/settings.yaml"
            ),
        )


def get_backend(backend_name: Optional[str] = None) -> ResponseBackend:
    """Factory honouring ``settings.response.backend`` (simulated | live)."""
    name = backend_name or get_settings().response.backend
    if name == "live":
        log.warning("LIVE response backend selected - no real integrations exist yet")
        return LiveResponseBackend()
    if name != "simulated":
        log.warning("unknown backend '%s'; falling back to simulated", name)
    return SimulatedResponseBackend()