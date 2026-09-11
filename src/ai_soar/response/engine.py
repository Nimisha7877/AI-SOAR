"""Response engine: prediction -> incident -> playbook -> governed actions.

The engine is the SOAR's decision desk. Given a :class:`PredictionResponse` it:

1. Creates an :class:`Incident` (benign flows create NOTHING - they are logged
   by the API and dropped, like a real SOC ignores 90% of traffic).
2. Looks up the family's declarative playbook.
3. Applies TWO approval gates:
   - if the predictor's decision was ``human_approval`` or ``unknown_queue``,
     the WHOLE playbook is held: nothing executes until a human approves;
   - inside an ``auto_response`` incident, steps flagged destructive AND listed
     in ``settings.response.require_approval_for`` are still held individually.
   Automation therefore never performs a destructive act alone - which is the
   entire point of the leakage-audit-derived allowlist.
4. Executes the rest through the configured backend (simulated by default) and
   appends the incident to the JSONL store.

Event-sourced updates: ``approve``/``dismiss`` do NOT rewrite history. They
append a NEW line with the updated incident; readers take the latest line per
incident id. Every state change stays visible in the audit trail.
"""

from __future__ import annotations

import uuid
from typing import Optional

from ai_soar.config import get_settings
from ai_soar.data.schema import BENIGN_LABEL
from ai_soar.inference.schemas import PredictionResponse
from ai_soar.response.actuators.simulated import ResponseBackend, get_backend
from ai_soar.response.backends.incident_store import IncidentStore
from ai_soar.response.schemas import (
    ActionResult,
    ActionStatus,
    Incident,
    IncidentStatus,
    PlaybookStep,
    Severity,
)
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Declarative family playbooks (order = execution order)
# --------------------------------------------------------------------------
PLAYBOOKS: dict[str, list[PlaybookStep]] = {
    "BruteForce": [
        PlaybookStep(action="rate_limit_source", destructive=False,
                     rationale="slow the credential guessing immediately"),
        PlaybookStep(action="firewall_block_source", destructive=True,
                     rationale="stop the source after rate-limit evidence"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="record for credential-reset follow-up"),
        PlaybookStep(action="notify_soc", destructive=False,
                     rationale="keep analysts aware of active guessing"),
    ],
    "DoS": [
        PlaybookStep(action="rate_limit_source", destructive=False,
                     rationale="shed single-source flood volume"),
        PlaybookStep(action="null_route_target", destructive=True,
                     rationale="protect upstream if flood persists"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="capacity follow-up"),
        PlaybookStep(action="notify_soc", destructive=False,
                     rationale="awareness"),
    ],
    "DDoS": [
        PlaybookStep(action="null_route_target", destructive=True,
                     rationale="multi-source flood needs upstream relief"),
        PlaybookStep(action="rate_limit_source", destructive=False,
                     rationale="limit residual volume"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="post-incident capacity review"),
        PlaybookStep(action="notify_soc", destructive=False,
                     rationale="awareness"),
    ],
    "PortScan": [
        PlaybookStep(action="add_to_watchlist", destructive=False,
                     rationale="reconnaissance: watch, do not disrupt"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="track recon source"),
    ],
    "WebAttack": [
        PlaybookStep(action="waf_block", destructive=True,
                     rationale="virtual-patch the attacked application"),
        PlaybookStep(action="capture_forensics", destructive=False,
                     rationale="preserve evidence of attempted injection"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="app-owner follow-up"),
        PlaybookStep(action="notify_soc", destructive=False,
                     rationale="awareness"),
    ],
    "Botnet": [
        PlaybookStep(action="host_isolation", destructive=True,
                     rationale="C2 traffic: contain the compromised host"),
        PlaybookStep(action="capture_forensics", destructive=False,
                     rationale="preserve C2 evidence before cleanup"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="IR case"),
        PlaybookStep(action="notify_soc", destructive=False,
                     rationale="awareness"),
    ],
    "Infiltration": [
        PlaybookStep(action="host_isolation", destructive=True,
                     rationale="internal pivot: contain first"),
        PlaybookStep(action="capture_forensics", destructive=False,
                     rationale="lateral-movement evidence"),
        PlaybookStep(action="create_ticket", destructive=False,
                     rationale="IR case"),
        PlaybookStep(action="notify_soc", destructive=False,
                     rationale="awareness"),
    ],
}

FALLBACK_PLAYBOOK: list[PlaybookStep] = [
    PlaybookStep(action="create_ticket", destructive=False,
                 rationale="unknown family: record for human triage"),
    PlaybookStep(action="notify_soc", destructive=False,
                 rationale="unknown family: page analysts"),
]

SEVERITY_BY_FAMILY: dict[str, Severity] = {
    "BruteForce": Severity.HIGH,
    "DoS": Severity.HIGH,
    "DDoS": Severity.CRITICAL,
    "PortScan": Severity.LOW,
    "WebAttack": Severity.HIGH,
    "Botnet": Severity.CRITICAL,
    "Infiltration": Severity.CRITICAL,
    BENIGN_LABEL: Severity.LOW,
}

HELD_DECISIONS = {"human_approval", "unknown_queue"}


class ResponseEngine:
    """Turns predictions into governed, audited incidents."""

    def __init__(
        self,
        backend: Optional[ResponseBackend] = None,
        store: Optional[IncidentStore] = None,
        require_approval_for: Optional[list[str]] = None,
    ) -> None:
        settings = get_settings()
        self.backend = backend or get_backend()
        self.store = store or IncidentStore()
        self.require_approval = list(
            require_approval_for if require_approval_for is not None
            else settings.response.require_approval_for
        )

    # -- ids & state ---------------------------------------------------------
    @staticmethod
    def _new_id() -> str:
        return f"INC-{uuid.uuid4().hex[:8].upper()}"

    def latest(self) -> dict[str, Incident]:
        """Current state per incident id (last JSONL line wins)."""
        state: dict[str, Incident] = {}
        for incident in self.store.read_all():
            state[incident.incident_id] = incident
        return state

    def summary(self) -> dict:
        states = list(self.latest().values())
        return {
            "total_incidents": len(states),
            "by_status": _tally(i.status.value for i in states),
            "by_family": _tally(i.family for i in states),
            # Status, not pending_actions(): a decision-level hold
            # (human_approval / unknown_queue) has NO action records yet, so
            # selecting on actions would hide exactly the incidents a human
            # must review.
            "pending_approval": [
                i.incident_id
                for i in states
                if i.status == IncidentStatus.PENDING_APPROVAL or i.pending_actions()
            ],
        }

    # -- main entry ----------------------------------------------------------
    def handle(
        self,
        prediction: PredictionResponse,
        ground_truth: Optional[str] = None,
    ) -> Optional[Incident]:
        """Create + process an incident. Returns None for benign flows.

        ``ground_truth`` is ONLY for offline demo/replay modes (the dashboard's
        TP/FP triangle). Live traffic never carries it.
        """
        if not prediction.is_malicious:
            return None

        family = prediction.family
        steps = PLAYBOOKS.get(family, FALLBACK_PLAYBOOK)
        incident = Incident(
            incident_id=self._new_id(),
            request_id=prediction.request_id,
            family=family,
            confidence=prediction.confidence,
            gate_probability=prediction.gate_probability,
            decision=prediction.decision,
            decision_reason=prediction.decision_reason,
            severity=SEVERITY_BY_FAMILY.get(family, Severity.MEDIUM),
            playbook=[s.action for s in steps],
            model_version=prediction.model_version,
            ground_truth=ground_truth,
        )
        if family not in PLAYBOOKS:
            incident.add_note(f"no playbook for family '{family}'; fallback playbook used")

        # Gate 1: the predictor's own decision holds the whole playbook.
        if prediction.decision in HELD_DECISIONS:
            incident.status = IncidentStatus.PENDING_APPROVAL
            incident.add_note(
                f"decision={prediction.decision}: playbook held until a human approves "
                f"({prediction.decision_reason})"
            )
            self.store.append(incident)
            log.info("incident %s (%s) held for human approval", incident.incident_id, family)
            return incident

        # Gate 2: per-action approval for destructive steps inside auto_response.
        pending = False
        executed = False
        for step in steps:
            if step.destructive and step.action in self.require_approval:
                incident.actions.append(
                    ActionResult(
                        action=step.action,
                        actuator=self.backend.name,
                        status=ActionStatus.PENDING_APPROVAL,
                        simulated=True,
                        message=f"destructive action held for human approval: {step.rationale}",
                    )
                )
                pending = True
                continue
            result = self.backend.execute(incident, step.action)
            incident.actions.append(result)
            if result.status in (ActionStatus.SIMULATED, ActionStatus.EXECUTED):
                executed = True

        incident.status = (
            IncidentStatus.PENDING_APPROVAL if pending else IncidentStatus.AUTO_RESOLVED
        )
        if executed and pending:
            incident.add_note("non-destructive steps executed; destructive steps await approval")
        self.store.append(incident)
        log.info(
            "incident %s (%s) status=%s actions=%d",
            incident.incident_id, family, incident.status.value, len(incident.actions),
        )
        return incident

    # -- human-in-the-loop -----------------------------------------------------
    def approve(self, incident_id: str, approver: str) -> Optional[Incident]:
        """A human releases the held actions of one incident (event-sourced)."""
        state = self.latest().get(incident_id)
        if state is None:
            log.warning("approve: unknown incident %s", incident_id)
            return None
        held = state.pending_actions()

        # Decision-level hold (gate 1): the incident has no action records yet,
        # so approving it releases the ENTIRE playbook.
        if not held and state.status == IncidentStatus.PENDING_APPROVAL and not state.actions:
            steps = PLAYBOOKS.get(state.family, FALLBACK_PLAYBOOK)
            for step in steps:
                result = self.backend.execute(state, step.action)
                result.approved_by = approver
                result.message = f"[approved by {approver}] {result.message}"
                state.actions.append(result)
            state.status = IncidentStatus.AUTO_RESOLVED
            state.add_note(f"approved by {approver}: full playbook released ({len(steps)} actions)")
            self.store.append(state)
            log.info("incident %s approved by %s (full playbook)", incident_id, approver)
            return state

        if not held:
            log.info("approve: incident %s has nothing pending", incident_id)
            return None

        for held_action in held:
            result = self.backend.execute(state, held_action.action)
            result.approved_by = approver
            result.message = f"[approved by {approver}] {result.message}"
            _replace_action(state, held_action.action, result)

        state.status = IncidentStatus.AUTO_RESOLVED
        state.add_note(f"approved by {approver}: {len(held)} action(s) released")
        self.store.append(state)
        log.info("incident %s approved by %s", incident_id, approver)
        return state

    def dismiss(self, incident_id: str, approver: str, reason: str = "") -> Optional[Incident]:
        """A human rejects the incident: held actions are skipped, case closed."""
        state = self.latest().get(incident_id)
        if state is None:
            log.warning("dismiss: unknown incident %s", incident_id)
            return None
        for held_action in state.pending_actions():
            _replace_action(
                state,
                held_action.action,
                ActionResult(
                    action=held_action.action,
                    actuator=self.backend.name,
                    status=ActionStatus.SKIPPED,
                    simulated=True,
                    message=f"dismissed by {approver}: {reason or 'no reason given'}",
                ),
            )
        state.status = IncidentStatus.DISMISSED
        state.add_note(f"dismissed by {approver}: {reason or 'no reason given'}")
        self.store.append(state)
        log.info("incident %s dismissed by %s", incident_id, approver)
        return state


# --------------------------------------------------------------------------
def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _replace_action(incident: Incident, action_name: str, new_result: ActionResult) -> None:
    """Swap the pending record for the final one, keeping list order."""
    for idx, existing in enumerate(incident.actions):
        if existing.action == action_name and existing.status == ActionStatus.PENDING_APPROVAL:
            incident.actions[idx] = new_result
            return
    incident.actions.append(new_result)