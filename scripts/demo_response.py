#!/usr/bin/env python
"""End-to-end response demo: real test flows -> predictor -> engine -> incidents.

Run:
    python scripts/demo_response.py                     # 500 flows, append to incident log
    python scripts/demo_response.py --rows 2000 --fresh # wipe log first, bigger sample
    python scripts/demo_response.py --auto-approve      # also demo the human-approval path

What it proves: the FULL chain works on real data - binary gate, family
classifier, evidence-based decision policy, family playbooks, approval gates,
simulated actuators and the append-only audit log. The JSON summary lands in
artifacts/reports/response_demo_report.json for the thesis.

Demo-only ground truth: because this script replays a labelled test split, it
tags every incident with ``ground_truth`` so the dashboard can draw a TP/FP
triangle. Live traffic never carries ground truth.

Honest caveat printed in the report: these flows come from the STRATIFIED test
split, so incident quality here is an UPPER BOUND (near-duplicate flows). The
honest generalization numbers live in the leakage/temporal/cross-dataset tiers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_soar.config import get_settings  # noqa: E402
from ai_soar.data.features import FEATURE_COLUMNS, load_split  # noqa: E402
from ai_soar.inference.predictor import Predictor  # noqa: E402
from ai_soar.inference.schemas import FlowFeaturesRequest  # noqa: E402
from ai_soar.response.backends.incident_store import IncidentStore  # noqa: E402
from ai_soar.response.engine import ResponseEngine  # noqa: E402
from ai_soar.utils.logging import configure_from_settings, get_logger  # noqa: E402

log = get_logger("demo_response")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI SOAR end-to-end response demo")
    parser.add_argument("--rows", type=int, default=500, help="flows to score (default 500)")
    parser.add_argument("--split", default="test", help="stratified split name (default test)")
    parser.add_argument("--fresh", action="store_true", help="clear the incident log first")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="approve every pending incident as demo-script@soc (shows the human path)",
    )
    return parser.parse_args(argv)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_from_settings(settings)
    settings.ensure_directories()

    predictor = Predictor.from_env()
    store = IncidentStore()
    if args.fresh:
        log.warning("--fresh: clearing incident log %s", store.path)
        store.clear()
    engine = ResponseEngine(store=store)

    print(f"\nloading {args.rows} flows from stratified split '{args.split}' ...")
    df = load_split(args.split, max_rows=args.rows)
    records = df[list(FEATURE_COLUMNS)].to_dict("records")
    truth = df["Family"].tolist()

    decisions: Counter = Counter()
    latencies: list[float] = []
    family_ok = 0
    created = 0

    started = time.perf_counter()
    for feats, true_family in zip(records, truth):
        prediction = predictor.predict(
            FlowFeaturesRequest(request_id=None, source="demo", features={k: float(v) for k, v in feats.items()})
        )
        decisions[prediction.decision] += 1
        latencies.append(prediction.latency_ms)
        family_ok += prediction.family == true_family
        if engine.handle(prediction, ground_truth=true_family) is not None:
            created += 1
    wall = time.perf_counter() - started

    print(f"\nscored {len(records)} flows in {wall:.2f}s "
          f"({len(records) / wall:.0f} flows/s, p95 { _percentile(latencies, 95):.2f} ms/flow)")
    print("decisions      :", dict(decisions))
    print("family accuracy:", f"{family_ok}/{len(records)} = {family_ok / len(records):.4f} (stratified = upper bound)")

    if args.auto_approve:
        pending = [i.incident_id for i in engine.latest().values() if i.pending_actions()
                   or (i.status.value == "pending_approval" and not i.actions)]
        print(f"\nauto-approving {len(pending)} pending incident(s) as demo-script@soc ...")
        for incident_id in pending:
            engine.approve(incident_id, "demo-script@soc")

    summary = engine.summary()
    print("\n================ INCIDENT SUMMARY ================")
    print("total incidents :", summary["total_incidents"])
    print("by status       :", summary["by_status"])
    print("by family       :", summary["by_family"])
    print("pending approval:", summary["pending_approval"] or "none")

    # One representative incident per status, with its action trail.
    print("\n================ SAMPLE INCIDENTS ================")
    shown: set[str] = set()
    for incident in engine.latest().values():
        if incident.status.value in shown:
            continue
        shown.add(incident.status.value)
        print(f"\n[{incident.incident_id}] {incident.family} | severity={incident.severity.value} "
              f"| status={incident.status.value} | decision={incident.decision}")
        print(f"  reason   : {incident.decision_reason}")
        for action in incident.actions:
            print(f"  - {action.action:22s} {action.status.value:17s} {action.message[:70]}")
        for note in incident.notes:
            print(f"  note     : {note}")

    report = {
        "generated_by": "scripts/demo_response.py",
        "split": args.split,
        "rows_scored": len(records),
        "flows_per_second": round(len(records) / wall, 1),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
        },
        "decisions": dict(decisions),
        "family_accuracy_upper_bound": round(family_ok / len(records), 4),
        "incidents_created_this_run": created,
        "incident_summary": summary,
        "auto_approved": bool(args.auto_approve),
        "incident_log": str(store.path),
        "STANDING_CAVEAT": (
            "Flows come from the stratified test split: incident quality here is an UPPER "
            "BOUND. Honest generalization numbers live in leakage_audit_report.json, "
            "temporal_eval_report.json and (pending) the CSE-CIC-IDS2018 cross-dataset tier."
        ),
    }
    report_path = Path(settings.paths.reports) / "response_demo_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport  : {report_path}")
    print(f"audit   : {store.path}")
    print("\ndemo complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())