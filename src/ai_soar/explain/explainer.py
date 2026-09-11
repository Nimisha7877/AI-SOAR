"""Incident explainer: incident + RAG context -> justified, cited explanation.

Two guarantees drive the design:

1. **Grounded, not hallucinated.** The prompt hands the LLM the retrieved
   knowledge chunks and forbids facts outside them. Every explanation carries
   its citations (``source#section``) so a reviewer can check each claim.
2. **Never empty.** If the LLM is unreachable (or --skip-llm is requested), a
   deterministic template built from the SAME knowledge chunks is used instead,
   honestly labelled ``provider="offline-fallback"``. A SOC report that silently
   vanishes (or silently invents) is worse than a plain one.

Explanations are appended to ``artifacts/incidents/explanations.jsonl`` so the
audit trail includes not just what was done but why it was justified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ai_soar.config import get_settings
from ai_soar.explain.llm import LLMClient, LLMReply, OFFLINE
from ai_soar.explain.rag.store import Chunk, KnowledgeStore
from ai_soar.response.schemas import Incident
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

EXPLANATIONS_FILENAME = "explanations.jsonl"

#: CPU-hosted local models (e.g. llama3 / qwen2.5 via Ollama) can take minutes
#: to cold-load and to generate ~250 grounded words.
LLM_TIMEOUT_SECONDS = 240.0

SYSTEM_PROMPT = (
    "You are a senior SOC analyst writing an incident explanation for a human "
    "reviewer. Use ONLY the knowledge context provided; never invent tools, "
    "metrics, MITRE ids or dataset facts. Write in plain English, at most 250 "
    "words, no markdown headers, in four short paragraphs: "
    "(1) what this attack family is, with its MITRE ATT&CK id from context; "
    "(2) why the classifier's verdict is credible here, quoting the model note "
    "(hardened F1 / limitations) from context; "
    "(3) why each response action was executed, held for human approval, or "
    "skipped, quoting the playbook rationale from context; "
    "(4) what the system cannot guarantee (closed-set limitation, weakly "
    "measured families), quoting the policy section from context."
)


@dataclass
class Explanation:
    """One justification record."""

    incident_id: str
    family: str
    decision: str
    status: str
    text: str
    citations: list[str] = field(default_factory=list)
    provider: str = OFFLINE
    model: str = "none"
    is_fallback: bool = True
    context_chunks: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_log_record(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "family": self.family,
            "decision": self.decision,
            "status": self.status,
            "text": self.text,
            "citations": self.citations,
            "provider": self.provider,
            "model": self.model,
            "is_fallback": self.is_fallback,
            "context_chunks": self.context_chunks,
            "generated_at": self.generated_at.isoformat(),
        }


class IncidentExplainer:
    def __init__(
        self,
        store: Optional[KnowledgeStore] = None,
        llm: Optional[LLMClient] = None,
        out_path: Optional[Path | str] = None,
        use_llm: bool = True,
    ) -> None:
        # ``is None``, not ``or``: KnowledgeStore defines __len__, so an empty
        # injected store is falsy and ``or`` would discard it and re-ingest
        # the real knowledge_base/ directory.
        self.kb = KnowledgeStore() if store is None else store
        if not self.kb.chunks:
            self.kb.ingest()
        self.llm = llm or LLMClient()
        self.use_llm = use_llm
        settings = get_settings()
        self.out_path = (
            Path(out_path)
            if out_path
            else Path(settings.paths.incidents) / EXPLANATIONS_FILENAME
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- public ------------------------------------------------------------
    def explain(self, incident: Incident, top_k: Optional[int] = None) -> Explanation:
        """Produce (and persist) a justified explanation for one incident."""
        hits = self.kb.retrieve_for_incident(incident.family, incident.decision, top_k=top_k)
        context = self._context_block(hits)
        citations = [chunk.cite() for chunk, _ in hits]

        if self.use_llm:
            reply = self.llm.complete(
                SYSTEM_PROMPT,
                self._user_prompt(incident, context),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        else:
            reply = LLMReply("", OFFLINE, self.llm.cfg.model, "--skip-llm requested")

        if reply.is_fallback or not reply.text.strip():
            text = self._template_explanation(incident, hits)
            explanation = Explanation(
                incident_id=incident.incident_id,
                family=incident.family,
                decision=incident.decision,
                status=incident.status.value,
                text=text,
                citations=citations,
                provider=OFFLINE,
                model=self.llm.cfg.model,
                is_fallback=True,
                context_chunks=len(hits),
            )
            log.warning(
                "explanation for %s used offline template (%s)",
                incident.incident_id, reply.error or "empty reply",
            )
        else:
            explanation = Explanation(
                incident_id=incident.incident_id,
                family=incident.family,
                decision=incident.decision,
                status=incident.status.value,
                text=reply.text.strip(),
                citations=citations,
                provider=reply.provider,
                model=reply.model,
                is_fallback=False,
                context_chunks=len(hits),
            )

        with self.out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(explanation.to_log_record(), ensure_ascii=False) + "\n")
        log.info("explanation for %s via %s", incident.incident_id, explanation.provider)
        return explanation

    # -- prompt building -----------------------------------------------------
    @staticmethod
    def _context_block(hits: list[tuple[Chunk, float]]) -> str:
        parts = []
        for chunk, score in hits:
            parts.append(f"[{chunk.cite()} | relevance {score:.2f}]\n{chunk.text}")
        return "\n\n".join(parts)

    @staticmethod
    def _user_prompt(incident: Incident, context: str) -> str:
        incident_json = json.dumps(incident.to_log_record(), ensure_ascii=False, indent=2)
        return (
            f"INCIDENT RECORD:\n{incident_json}\n\n"
            f"KNOWLEDGE CONTEXT (the only facts you may use):\n{context}\n\n"
            "Now write the explanation."
        )

    # -- deterministic fallback ------------------------------------------------
    @staticmethod
    def _template_explanation(incident: Incident, hits: list[tuple[Chunk, float]]) -> str:
        family_chunk = next((c for c, _ in hits if c.section == incident.family), None)
        policy_chunk = next((c for c, _ in hits if c.section == "Global response policy"), None)

        def bullets(chunk: Optional[Chunk], prefix: str) -> str:
            if chunk is None:
                return "(no knowledge available)"
            marker = f"- {prefix}"
            for line in chunk.text.splitlines():
                if line.strip().startswith(marker):
                    return line.strip()[len(marker):].strip()
            return "(no knowledge available)"

        lines = [
            f"OFFLINE TEMPLATE EXPLANATION (LLM unavailable) - incident {incident.incident_id}.",
            "",
            f"What it is: {bullets(family_chunk, 'What it is:')}",
            f"MITRE ATT&CK: {bullets(family_chunk, 'MITRE ATT&CK:')}",
            "",
            f"The two-stage classifier (binary gate then family model) returned "
            f"{incident.family} with confidence {incident.confidence:.3f} "
            f"(gate probability {incident.gate_probability:.3f}). "
            f"Model note from the knowledge base: {bullets(family_chunk, 'Model note:')}",
            "",
            "Actions and why:",
        ]
        if incident.actions:
            for action in incident.actions:
                lines.append(
                    f"  - {action.action}: {action.status.value} - {action.message}"
                    + (f" (approved by {action.approved_by})" if action.approved_by else "")
                )
        else:
            lines.append(
                "  - playbook held in full: the decision policy required human "
                "approval before any action"
            )
        lines += [
            "",
            f"Playbook rationale: {bullets(family_chunk, 'Rationale:')}",
            f"Policy context: {bullets(policy_chunk, 'Known systemic limitation:')}",
            "",
            f"Decision: {incident.decision} ({incident.decision_reason}). "
            f"Status: {incident.status.value}.",
        ]
        return "\n".join(lines)