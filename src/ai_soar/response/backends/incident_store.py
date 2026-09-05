"""Incident persistence backend: an append-only JSONL store.

Why JSONL and not SQLite: an incident log is append-only, reviewed by humans,
and grep-able; one JSON object per line survives partial writes and needs no
migration tooling. If query needs grow later, swap this class for a DB-backed
one - the response engine only calls ``append`` / ``read_all`` / ``pending_approval``.

Thread safety: the API may score flows from several worker threads, so appends
take a lock. A torn line can never be written, and a corrupt line can never
kill ``read_all`` (it is skipped with a warning).
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

    def pending_approval(self) -> list[Incident]:
        """The human's work queue: incidents waiting on an approver."""
        return [i for i in self.read_all() if i.status == IncidentStatus.PENDING_APPROVAL]

    def counts(self) -> dict[str, int]:
        """Incidents per status - a one-glance SOC summary."""
        tally: dict[str, int] = {}
        for incident in self.read_all():
            tally[incident.status.value] = tally.get(incident.status.value, 0) + 1
        return tally

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.read_all())