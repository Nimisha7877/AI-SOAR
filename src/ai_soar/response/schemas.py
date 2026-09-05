"""Data contracts for the response layer (incidents, actions, playbooks).

Why these exist: a SOAR is judged by its AUDIT TRAIL, not by how fast it blocks
something. Every action taken (or refused, or queued for a human) becomes an
:class:`ActionResult` attached to an :class:`Incident`, and every incident is
appended to ``artifacts/incidents/incidents.jsonl`` so a reviewer can replay
exactly what the system decided and why.

Status vocabulary is deliberately small and explicit:

- ``executed``            ran for real (only possible with a live backend)
- ``simulated``           backend is simulated; action recorded, nothing touched
- ``pending_approval``    destructive action held back until a human approves
- ``skipped``             playbook step not applicable to this incident
- ``failed``              actuator raised; incident keeps going, error recorded
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActionStatus(str, Enum):
    EXECUTED = "executed"
    SIMULATED = "simulated"
    PENDING_APPROVAL = "pending_approval"
    SKIPPED = "skipped"
    FAILED = "failed"


class IncidentStatus(str, Enum):
    OPEN = "open"
    AUTO_RESOLVED = "auto_resolved"
    PENDING_APPROVAL = "pending_approval"
    CLOSED = "closed"
    DISMISSED = "dismissed"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionResult(BaseModel):
    """One playbook step's outcome."""

    action: str = Field(description="playbook step name, e.g. firewall_block_source")
    actuator: str = Field(default="simulated", description="which backend performed it")
    status: ActionStatus
    simulated: bool = Field(default=True, description="True unless a live backend ran it")
    message: str = Field(default="", description="human-readable outcome")
    executed_at: datetime = Field(default_factory=utcnow)
    approved_by: Optional[str] = Field(default=None, description="human id when approved")


class PlaybookStep(BaseModel):
    """A declarative step inside a family playbook."""

    action: str
    destructive: bool = Field(
        default=False,
        description="destructive steps obey settings.response.require_approval_for",
    )
    rationale: str = Field(default="", description="why this step exists for this family")


class Incident(BaseModel):
    """Everything known about one detected event and how it was handled."""

    # 'model_version' collides with pydantic's protected namespace otherwise
    model_config = ConfigDict(protected_namespaces=())

    incident_id: str
    created_at: datetime = Field(default_factory=utcnow)
    request_id: Optional[str] = None
    family: str
    confidence: float
    gate_probability: float
    decision: str
    decision_reason: str = ""
    severity: Severity = Severity.MEDIUM
    status: IncidentStatus = IncidentStatus.OPEN
    playbook: list[str] = Field(default_factory=list, description="planned action names")
    actions: list[ActionResult] = Field(default_factory=list)
    model_version: str = ""
    notes: list[str] = Field(default_factory=list)
    ground_truth: Optional[str] = Field(
        default=None,
        description="only set in offline demo/replay mode; never in live traffic",
    )

    # -- helpers -----------------------------------------------------------
    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def pending_actions(self) -> list[ActionResult]:
        return [a for a in self.actions if a.status == ActionStatus.PENDING_APPROVAL]

    def to_log_record(self) -> dict:
        """Flat JSON-safe record for incidents.jsonl."""
        return self.model_dump(mode="json")