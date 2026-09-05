"""Retrieval store for the explainer: heading-aware chunks + TF-IDF search.

Why sparse retrieval (TF-IDF) instead of a vector database:
- the knowledge base is small and LEXICAL - queries and docs share exact tokens
  (family names, MITRE ids, action names), which is exactly what TF-IDF nails;
- zero external services or API keys, so explanations work offline and in CI;
- fully deterministic, which matters when an explanation is quoted in a report.

Why heading-aware chunking: each ``## Section`` of a knowledge document is one
semantic unit (a whole family, or the global policy). Fixed-size windows would
split "Botnet" mid-sentence and retrieve half-truths. Oversized sections fall
back to paragraph splits.

Upgrade path: swap :class:`KnowledgeStore._vectorize` for an embedding model
and keep every caller unchanged - ``search`` and ``retrieve_for_incident``
are the only two entry points the explainer uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from ai_soar.config import get_settings
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

MAX_CHUNK_CHARS = 3000
DOC_GLOBS = ("*.md", "*.txt")
GLOBAL_POLICY_SECTION = "Global response policy"


@dataclass
class Chunk:
    """One retrievable unit of knowledge."""

    chunk_id: str
    source: str          # file name
    section: str         # heading the chunk belongs to
    text: str
    metadata: dict = field(default_factory=dict)

    def cite(self) -> str:
        return f"{self.source}#{self.section}"


def _split_markdown(text: str, source: str) -> list[Chunk]:
    """Split on '## ' headings; oversized sections split on blank lines."""
    chunks: list[Chunk] = []
    current_section = "preamble"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::{current_section}",
                    source=source,
                    section=current_section,
                    text=body,
                )
            )
            return
        # oversized: paragraph-level fallback
        piece: list[str] = []
        for para in body.split("\n\n"):
            if sum(len(p) for p in piece) + len(para) > MAX_CHUNK_CHARS and piece:
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}::{current_section}#{len(chunks)}",
                        source=source,
                        section=current_section,
                        text="\n\n".join(piece).strip(),
                    )
                )
                piece = [para]
            else:
                piece.append(para)
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::{current_section}#{len(chunks)}",
                    source=source,
                    section=current_section,
                    text="\n\n".join(piece).strip(),
                )
            )

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current_section = line[3:].strip()
        else:
            buffer.append(line)
    flush()
    return chunks


class KnowledgeStore:
    """Ingests knowledge_base/ and answers relevance queries."""

    def __init__(self, kb_dir: Optional[Path | str] = None) -> None:
        self.kb_dir = Path(kb_dir) if kb_dir else Path(get_settings().paths.knowledge_base)
        self.chunks: list[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    # -- ingest ------------------------------------------------------------
    def ingest(self) -> int:
        """(Re)build the index from every .md/.txt under the knowledge base."""
        self.chunks = []
        for pattern in DOC_GLOBS:
            for path in sorted(self.kb_dir.glob(pattern)):
                text = path.read_text(encoding="utf-8")
                found = _split_markdown(text, path.name)
                log.info("ingested %s: %d chunk(s)", path.name, len(found))
                self.chunks.extend(found)
        if not self.chunks:
            raise FileNotFoundError(f"no knowledge documents found in {self.kb_dir}")
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])
        log.info("knowledge store ready: %d chunks", len(self.chunks))
        return len(self.chunks)

    # -- retrieval -----------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[Chunk, float]]:
        """Cosine-similar chunks for a free-text query, best first."""
        if self._matrix is None:
            self.ingest()
        k = top_k or get_settings().llm.rag_top_k
        vec = self._vectorizer.transform([query])
        scores = (self._matrix @ vec.T).toarray().ravel()
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0.0]
        return out[:k]

    def retrieve_for_incident(
        self,
        family: str,
        decision: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        """Targeted retrieval for an incident: family + policy, then lexical.

        Two guaranteed anchors ground every explanation:
        1. the incident's family section (exact heading match), and
        2. the global response-policy section (allowlist, approval gates,
           known limitations).
        Lexical search then adds any extra relevant context.
        """
        hits: list[tuple[Chunk, float]] = []
        seen: set[str] = set()

        for chunk in self.chunks:
            if chunk.section in (family, GLOBAL_POLICY_SECTION) and chunk.chunk_id not in seen:
                hits.append((chunk, 1.0))
                seen.add(chunk.chunk_id)

        query = (
            f"{family} attack response policy approval rationale MITRE "
            f"decision={decision} limitations confidence"
        )
        for chunk, score in self.search(query, top_k=(top_k or 4) + 2):
            if chunk.chunk_id not in seen:
                hits.append((chunk, score))
                seen.add(chunk.chunk_id)
        return hits[: top_k or get_settings().llm.rag_top_k]

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.chunks)