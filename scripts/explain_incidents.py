#!/usr/bin/env python
"""Batch-explain incidents from the audit log (LLM + RAG, with template fallback).

Run:
    python scripts/explain_incidents.py                    # newest 3 incidents
    python scripts/explain_incidents.py --limit 10
    python scripts/explain_incidents.py --family DDoS --limit 2
    python scripts/explain_incidents.py --skip-llm --all   # instant template mode

Cost warning: with a CPU-hosted local LLM each explanation can take 1-3 minutes
(the model reads the incident record plus retrieved knowledge chunks and writes
~250 grounded words). --skip-llm produces the deterministic template instantly.

Every explanation is appended to artifacts/incidents/explanations.jsonl and a
run summary lands in artifacts/reports/explanations_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_soar.config import get_settings  # noqa: E402
from ai_soar.explain.explainer import IncidentExplainer  # noqa: E402
from ai_soar.explain.llm import LLMClient, OFFLINE  # noqa: E402
from ai_soar.explain.rag.store import KnowledgeStore  # noqa: E402
from ai_soar.response.backends.incident_store import IncidentStore  # noqa: E402
from ai_soar.response.engine import ResponseEngine  # noqa: E402
from ai_soar.utils.logging import configure_from_settings, get_logger  # noqa: E402

log = get_logger("explain_incidents")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain incidents with LLM + RAG")
    parser.add_argument("--limit", type=int, default=3, help="newest N incidents (default 3)")
    parser.add_argument("--family", default=None, help="only this family (e.g. DDoS)")
    parser.add_argument("--all", action="store_true", help="ignore --limit, explain everything selected")
    parser.add_argument("--skip-llm", action="store_true", help="deterministic template only (fast)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_from_settings(settings)
    settings.ensure_directories()

    engine = ResponseEngine(store=IncidentStore())
    incidents = sorted(engine.latest().values(), key=lambda i: i.created_at)
    if args.family:
        incidents = [i for i in incidents if i.family == args.family]
    if not args.all:
        incidents = incidents[-args.limit:]
    if not incidents:
        print("no incidents selected - run scripts/demo_response.py first")
        return 1

    kb = KnowledgeStore()
    kb.ingest()
    llm = LLMClient()
    if args.skip_llm:
        print("--skip-llm: deterministic template explanations (no model calls)")
    elif not llm.available():
        print("WARNING: LLM provider not reachable; explanations will use the offline template")
    explainer = IncidentExplainer(store=kb, llm=llm, use_llm=not args.skip_llm)

    print(f"\nexplaining {len(incidents)} incident(s) ...\n")
    rows = []
    for idx, incident in enumerate(incidents, 1):
        started = time.perf_counter()
        explanation = explainer.explain(incident)
        secs = time.perf_counter() - started
        rows.append(
            {
                "incident_id": explanation.incident_id,
                "family": explanation.family,
                "decision": explanation.decision,
                "status": explanation.status,
                "provider": explanation.provider,
                "is_fallback": explanation.is_fallback,
                "context_chunks": explanation.context_chunks,
                "citations": explanation.citations,
                "words": len(explanation.text.split()),
                "seconds": round(secs, 1),
            }
        )
        print(f"[{idx}/{len(incidents)}] {explanation.incident_id} {explanation.family:11s} "
              f"via {explanation.provider} in {secs:.0f}s")
        print("-" * 70)
        print(explanation.text[:700])
        print("-" * 70 + "\n")

    report = {
        "generated_by": "scripts/explain_incidents.py",
        "skip_llm": bool(args.skip_llm),
        "n_explained": len(rows),
        "providers": {p: sum(1 for r in rows if r["provider"] == p) for p in {r["provider"] for r in rows}},
        "fallbacks": sum(1 for r in rows if r["is_fallback"]),
        "total_seconds": round(sum(r["seconds"] for r in rows), 1),
        "explanations": rows,
        "explanations_log": str(explainer.out_path),
        "STANDING_CAVEAT": (
            "Explanations are grounded in knowledge_base/ chunks and cite them; "
            "they describe model behaviour, not independent verification of it."
        ),
    }
    report_path = Path(settings.paths.reports) / "explanations_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"summary : {report_path}")
    print(f"log     : {explainer.out_path}")
    print("\ndone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())