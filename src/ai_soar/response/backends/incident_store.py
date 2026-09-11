"""Incident persistence backend: an append-only JSONL store.

Why JSONL and not SQLite: an incident log is append-only, reviewed by humans,
and grep-able; one JSON object per line survives partial writes and needs no
migration tooling. If query needs grow later, swap this class for a DB-backed
one - the response engine only calls ``append`` / ``read_all`` / ``latest`` /
``pending_approval``.

Thread safety: the API may score flows from several worker threads, so appends
take a lock. A torn line can never be written, and a corrupt line can never
kill ``read_all`` (it is skipped with a warning).

Reading: the log is EVENT-SOURCED. ``approve``/``dismiss`` append a new line
rather than editing the old one, so "what is true now" is always the LAST line
for an incident id. Anything that answers a *current state* question
(``pending_approval``) must go through :meth:`latest`; reading raw lines would
report an incident as pending forever.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from ai_soar.config import get_settings
from ai_soar.response.schemas import Incident, IncidentStatus
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

FILENAME = "incidents.jsonl"


class IncidentStore:
    """Append-only store at ``<paths.incidents>/incidents.jsonl``."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = (
            Path(path)
            if path
            else Path(get_settings().paths.incidents) / FILENAME
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- writes ------------------------------------------------------------
    def append(self, incident: Incident) -> Path:
        """Persist one incident as a single JSON line."""
        record = json.dumps(incident.to_log_record(), ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record + "\n")
        log.info("incident %s appended to %s", incident.incident_id, self.path)
        return self.path

    def clear(self) -> None:
        """Delete the log. Intended for demos/tests ONLY - audit data is precious."""
        if self.path.exists():
            self.path.unlink()
            log.warning("incident store cleared: %s", self.path)

    # -- reads -------------------------------------------------------------
    def read_all(self, limit: Optional[int] = None) -> list[Incident]:
        """Replay the log oldest-first; ``limit`` returns the newest N."""
        if not self.path.exists():
            return []
        out: list[Incident] = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Incident.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001 - never let one bad line kill replay
                log.warning("skipping corrupt incident line %d: %s", lineno, exc)
        return out[-limit:] if limit else out

    def latest(self) -> dict[str, Incident]:
        """Current state per incident id: the LAST line for an id is the truth.

        Earlier lines are history and are kept on purpose (that is what makes
        the log an audit trail), but they must never be read as current state.
        """
        state: dict[str, Incident] = {}
        for incident in self.read_all():
            state[incident.incident_id] = incident
        return state

    def pending_approval(self) -> list[Incident]:
        """The human's work queue: incidents CURRENTLY waiting on an approver."""
        return [i for i in self.latest().values() if i.status == IncidentStatus.PENDING_APPROVAL]

    def counts(self) -> dict[str, int]:
        """Incidents per status (latest state per id) - a one-glance SOC summary."""
        tally: dict[str, int] = {}
        for incident in self.latest().values():
            tally[incident.status.value] = tally.get(incident.status.value, 0) + 1
        return tally

    # -- object protocol ----------------------------------------------------
    def __bool__(self) -> bool:
        """ALWAYS truthy.

        ``__len__`` below makes an empty log falsy, and ``store or
        IncidentStore()`` in a caller would then silently throw away the store
        it was given and fall back to the configured (production) path. Pinning
        truthiness removes that trap for every current and future caller.
        """
        return True

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.read_all())
