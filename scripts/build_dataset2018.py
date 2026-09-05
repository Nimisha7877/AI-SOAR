"""Build the CSE-CIC-IDS2018 evaluation set (tier D) from the downloaded CSVs.

Why this script exists
----------------------
Tier D is the honest generalisation test: models trained on CICIDS2017 are run
against a *different year, a different network (AWS instead of the UNB lab), a
different CICFlowMeter version and a different label vocabulary*. Nothing here
trains anything - this only converts the raw 2018 CSVs into a parquet set that
presents the SAME 70-feature space the deployed models expect.

Design decisions (all deliberate, all reported):

* **One parquet per source CSV.** Days stay separable, so the tier-D report can
  give a per-day false-positive rate, and a newly downloaded day can be added
  without rebuilding the others.
* **Chunked streaming (default 100k rows).** The Tuesday file alone is 4 GB /
  7.9M rows; the whole set is ~13.2M rows. Nothing is ever held in RAM at once.
* **float32 features.** ``load_split()`` in the 2017 pipeline already casts to
  float32, and LightGBM bins in float32 - so this is lossless w.r.t. the model
  and halves the parquet size.
* **Timestamp is dropped.** Calendar time is leakage: a model must not learn
  "attacks happen on Tuesday". It is only read here for the truncation probe.
* **Sliding-window deduplication** (``--dedup window``, default). DDoS bursts
  produce long runs of byte-identical flows. Each chunk is deduplicated against
  the previous chunk, which catches burst runs across chunk boundaries without
  holding 13M rows in RAM. ``--dedup none`` keeps every flow (defensible too:
  2018 is evaluation-only, so duplicates cannot leak between train and test, and
  each duplicated flow is a genuine extra alert opportunity). The count is
  reported either way, and the window limitation is recorded in the report.
* **Non-finite rows are dropped, not imputed.** ``Flow Byts/s`` and
  ``Flow Pkts/s`` contain Inf/NaN (zero-duration flows). This mirrors exactly
  what the 2017 build did, so both datasets are cleaned by the same rule.
* **Junk header rows are dropped and counted.** The 2018 CSVs contain repeated
  header lines whose Label cell literally reads "Label".
* **Unknown labels raise.** Silently discarding an attack class would corrupt
  tier D; extend ``FAMILY_RULES_2018`` instead.

Run:
    python scripts/build_dataset2018.py --dry-run                 # audit only
    python scripts/build_dataset2018.py --file Wednesday-21 --limit-rows 200000
    python scripts/build_dataset2018.py                           # full build
    python scripts/build_dataset2018.py --dedup none              # keep every flow
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ai_soar.config import get_settings
from ai_soar.data.features import FEATURE_COLUMNS
from ai_soar.data.labels import FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.data.schema import CANONICAL_FEATURES, LABEL_COLUMN
from ai_soar.data.schema2018 import (
    IDENTITY_COLUMNS_2018,
    add_family_column as add_family_column_2018,
    apply_2018_schema,
    detect_label_column,
    map_columns_2018,
)
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("build_dataset2018")

FILE_GLOB = "*TrafficForML_CICFlowMeter.csv"
OUT_SUBDIR = "_processed"
SOURCE_COLUMN = "SourceFile"
DEFAULT_CHUNKSIZE = 100_000
OUT_COLUMNS: list[str] = list(CANONICAL_FEATURES) + [LABEL_COLUMN, FAMILY_COLUMN, SOURCE_COLUMN]
FEATS: list[str] = list(CANONICAL_FEATURES)

# The processed 2018 CSVs are widely suspected of being capped at 2**20 lines
# (Excel's row limit). Five of the six downloaded files have exactly this many
# data rows, which cannot be coincidence - flag it instead of hiding it.
ROW_CAP_SUSPECT = 1_048_575

STANDING_CAVEAT = (
    "Tier-D (cross-dataset) numbers describe models trained on CICIDS2017 and "
    "evaluated on CSE-CIC-IDS2018 flows. They are the most honest generalisation "
    "estimate in this project: no stratified sampling, no shared-day leakage. "
    "Attack families absent from the downloaded 2018 files are reported as "
    "'not covered' and are NOT scored as zero."
)


# --------------------------------------------------------------------------- #
# discovery + probes
# --------------------------------------------------------------------------- #
def discover(external: Path, only: Optional[str]) -> list[Path]:
    """Find the 2018 CSVs anywhere under the external dir (recursive)."""
    if not external.exists():
        return []
    hits = sorted(p for p in external.rglob(FILE_GLOB) if p.is_file())
    if not hits:  # tolerate a differently-named mirror (e.g. the fixed release)
        hits = sorted(p for p in external.rglob("*.csv") if p.is_file())
    if only:
        needle = only.lower()
        hits = [p for p in hits if needle in p.name.lower()]
    return hits


def _timestamp_of(line: str) -> str:
    for tok in line.split(","):
        if "/" in tok and ":" in tok:
            return tok.strip()
    return "?"


def probe(path: Path, external: Path) -> dict[str, Any]:
    """Read head/middle/tail bytes only - instant even on the 4 GB file."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        header = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
        first = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
        fh.seek(size // 2)
        fh.readline()  # discard the partial line we landed on
        middle = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
        fh.seek(max(0, size - 16384))
        tail = fh.read().decode("utf-8", "replace")

    lines = [ln for ln in tail.split("\n") if ln.strip()]
    last = lines[-1].rstrip("\r") if lines else ""
    rel = path.relative_to(external) if external in path.parents else Path(path.name)
    return {
        "rel": str(rel),
        "depth": len(rel.parts) - 1,
        "size_mb": round(size / 1e6, 1),
        "header_fields": len(header.split(",")),
        "first_ts": _timestamp_of(first),
        "middle_ts": _timestamp_of(middle),
        "last_ts": _timestamp_of(last),
        "last_fields": len(last.split(",")) if last else 0,
        "ends_with_newline": tail.endswith("\n"),
        "last_row_tail": last[-64:],
    }


def audit_header(path: Path) -> dict[str, Any]:
    """Header-only schema audit against the 2017 canonical space."""
    cols = [str(c) for c in pd.read_csv(path, nrows=0).columns]
    canon = map_columns_2018(cols, CANONICAL_FEATURES)
    model = map_columns_2018(cols, FEATURE_COLUMNS)
    return {
        "columns": len(cols),
        "canonical_covered": canon["covered_count"],
        "canonical_required": canon["required_count"],
        "model_covered": model["covered_count"],
        "model_required": model["required_count"],
        "aliases_applied": canon["alias_count"],
        "duplicated": canon["duplicated"],
        "unresolved": model["unresolved"],
        "unknown_2018_columns": canon["unknown"],
        "identity_columns_present": [c for c in cols if c in IDENTITY_COLUMNS_2018],
        "complete": canon["complete"] and not model["unresolved"],
    }


def load_prior_audit(reports: Path) -> dict[str, Any]:
    """Reuse the full-scan row/label counts already produced by the verifier.

    Key names come from scripts/verify_cicids2018.py: per file ``rows`` (full
    scan) or ``rows_sampled``, ``family_counts``, ``verdict``; top level
    ``family_counts_combined``.
    """
    path = reports / "cicids2018_verify_report.json"
    if not path.exists():
        return {"found": False, "files": {}, "combined": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corrupt report is non-fatal
        log.warning("could not read verify report: %s", exc)
        return {"found": False, "files": {}, "combined": {}}
    files = {e["file"]: e for e in data.get("files", []) if e.get("file")}
    return {
        "found": True,
        "files": files,
        "combined": data.get("family_counts_combined") or {},
    }


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def clean_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """2018 raw chunk -> canonical 78 features + Label + Family, cleaned.

    Same rules as the 2017 build: junk header rows out, non-finite rows out,
    byte-identical duplicates out. Every removal is counted and reported.
    """
    df = apply_2018_schema(chunk, CANONICAL_FEATURES)
    df = add_family_column_2018(df, LABEL_COLUMN)
    junk = int(df.attrs.get("dropped_junk_rows", 0))

    for col in FEATS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    arr = df[FEATS].to_numpy(dtype=np.float64)
    inf_cells = int(np.isinf(arr).sum())
    nan_cells = int(np.isnan(arr).sum())
    # NOTE the parentheses: "~np.isfinite(arr).any(axis=1)" would negate the
    # per-row "has a finite value" flag and delete every healthy row.
    nonfinite_row = ~np.isfinite(arr).all(axis=1)
    bad_rows = int(nonfinite_row.sum())
    if bad_rows:
        df = df.loc[~nonfinite_row]

    df[FEATS] = df[FEATS].astype(np.float32)
    stats = {
        "junk_rows": junk,
        "inf_cells": inf_cells,
        "nan_cells": nan_cells,
        "rows_dropped_nonfinite": bad_rows,
    }
    return df[FEATS + [LABEL_COLUMN, FAMILY_COLUMN]], stats


def dedup_window(
    prev: Optional[pd.DataFrame], cur: pd.DataFrame, key_cols: list[str]
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], int, dict[str, int]]:
    """Drop exact duplicates using a two-chunk sliding window.

    A global dedup would need all ~13M rows in RAM. Duplicates in these files are
    burst-adjacent (a DDoS tool emitting byte-identical flows back to back - the
    smoke run measured 38% of a HOIC-heavy prefix), so comparing each chunk with
    the previous one catches runs across the chunk boundary without holding the
    whole file.

    Returns ``(kept, new_prev, n_dups, dups_per_family)``. ``new_prev`` is the
    current chunk even when every one of its rows was a duplicate, so a fully
    duplicated chunk cannot reset the window and let the next chunk through.
    """
    if len(cur) == 0:
        return cur, prev, 0, {}

    if prev is None:
        mask = cur.duplicated(subset=key_cols, keep="first")
    else:
        combined = pd.concat([prev, cur[key_cols]], ignore_index=True)
        keep = ~combined.duplicated(subset=key_cols, keep="first")
        mask = pd.Series(~keep.to_numpy()[len(prev):], index=cur.index)

    dup_families: dict[str, int] = {}
    if FAMILY_COLUMN in cur.columns:
        dup_families = {
            str(k): int(v) for k, v in cur.loc[mask, FAMILY_COLUMN].value_counts().items()
        }
    return cur.loc[~mask], cur[key_cols].copy(), int(mask.sum()), dup_families


def build_file(
    path: Path,
    out_dir: Path,
    chunksize: int,
    limit_rows: Optional[int],
    dedup: str = "window",
) -> dict[str, Any]:
    """Stream one 2018 CSV into one parquet. Returns per-file statistics."""
    started = time.time()
    out_path = out_dir / f"{path.stem}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "rows_read": 0,
        "rows_out": 0,
        "chunks": 0,
        "junk_rows": 0,
        "inf_cells": 0,
        "nan_cells": 0,
        "rows_dropped_nonfinite": 0,
        "duplicates_dropped": 0,
    }
    families: dict[str, int] = {}
    dup_families: dict[str, int] = {}
    raw_labels: dict[str, int] = {}
    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    prev: Optional[pd.DataFrame] = None
    key_cols = FEATS + [LABEL_COLUMN]

    try:
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=True)
        for chunk in reader:
            if limit_rows is not None:
                budget = limit_rows - totals["rows_read"]
                if budget <= 0:
                    break
                if len(chunk) > budget:
                    chunk = chunk.iloc[:budget]
            totals["rows_read"] += int(len(chunk))
            totals["chunks"] += 1

            label_col = detect_label_column([str(c) for c in chunk.columns])
            if label_col:
                for value, count in chunk[label_col].astype(str).value_counts().items():
                    raw_labels[value] = raw_labels.get(value, 0) + int(count)

            df, stats = clean_chunk(chunk)
            for key, value in stats.items():
                totals[key] += int(value)

            if dedup == "window":
                df, prev, dups, dup_fams = dedup_window(prev, df, key_cols)
                totals["duplicates_dropped"] += dups
                for fam, n in dup_fams.items():
                    dup_families[fam] = dup_families.get(fam, 0) + n

            if len(df):
                df[SOURCE_COLUMN] = path.name
                df = df[OUT_COLUMNS]
                for fam, count in df[FAMILY_COLUMN].value_counts().items():
                    families[fam] = families.get(fam, 0) + int(count)
                totals["rows_out"] += int(len(df))

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out_path, schema, compression="snappy")
                else:
                    table = table.cast(schema)
                writer.write_table(table)

            if limit_rows is not None and totals["rows_read"] >= limit_rows:
                log.info("%s: --limit-rows reached (%d)", path.name, totals["rows_read"])
                break
            if totals["chunks"] % 10 == 0:
                log.info("  %s: %s rows read, %s written", path.name,
                         f"{totals['rows_read']:,}", f"{totals['rows_out']:,}")
    finally:
        if writer is not None:
            writer.close()

    if totals["rows_out"] == 0:
        raise RuntimeError(f"{path.name}: every row was removed by cleaning - refusing to write an empty parquet")

    return {
        "file": path.name,
        "out": str(out_path),
        "out_size_mb": round(out_path.stat().st_size / 1e6, 1) if out_path.exists() else 0.0,
        "seconds": round(time.time() - started, 1),
        "families": families,
        "duplicates_by_family": dup_families,
        "raw_labels": raw_labels,
        **totals,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def prior_rows(entry: dict[str, Any]) -> tuple[Optional[int], bool]:
    """(row count, was it only a sample?) from a verifier report entry."""
    if entry.get("rows"):
        return int(entry["rows"]), False
    if entry.get("rows_sampled"):
        return int(entry["rows_sampled"]), True
    return None, False


def _print_dry_run(files: list[Path], external: Path, prior: dict[str, Any]) -> bool:
    print("=" * 78)
    print("CSE-CIC-IDS2018 build - DRY RUN (nothing is written)")
    print("=" * 78)
    print(f"external dir : {external}")
    print(f"csv found    : {len(files)}  (pattern '{FILE_GLOB}', searched recursively)")
    if not files:
        print(f"\n[xx] no CSV under {external}. Put the 2018 files anywhere inside it.")
        return False

    ok = True
    combined: dict[str, int] = {}
    for path in files:
        pr = probe(path, external)
        au = audit_header(path)
        entry = prior.get("files", {}).get(path.name, {})
        rows, sampled = prior_rows(entry)
        print()
        print(f"=== {path.name} ===")
        print(f"  location         : {pr['rel']}  (depth {pr['depth']}, {pr['size_mb']} MB)")
        print(f"  header fields    : {pr['header_fields']}   last row fields: {pr['last_fields']}"
              f"   ends_with_newline={pr['ends_with_newline']}")
        print(f"  timestamps       : first={pr['first_ts']}  middle={pr['middle_ts']}  last={pr['last_ts']}")
        print(f"  schema           : canonical={au['canonical_covered']}/{au['canonical_required']}"
              f"  model={au['model_covered']}/{au['model_required']}"
              f"  aliases={au['aliases_applied']}  dup={bool(au['duplicated'])}")
        print(f"  identity dropped : {au['identity_columns_present']}")
        if au["unresolved"]:
            ok = False
            print(f"  [xx] UNRESOLVED  : {au['unresolved']}")
        if au["unknown_2018_columns"]:
            print(f"  [!!] unknown cols: {au['unknown_2018_columns']}")
        if rows:
            cap = "   <-- 2**20-1 ROW CAP SUSPECTED (truncated day)" if rows == ROW_CAP_SUSPECT else ""
            kind = "sampled only" if sampled else "full scan"
            print(f"  rows (verifier)  : {rows:,}  ({kind}){cap}")
        for fam, count in (entry.get("family_counts") or entry.get("families") or {}).items():
            combined[fam] = combined.get(fam, 0) + int(count)
        if entry.get("verdict"):
            print(f"  prior verdict    : {entry['verdict']}  (before schema2018.py existed)")
        print(f"  last row ends    : ...{pr['last_row_tail']}")
        if pr["last_fields"] != pr["header_fields"]:
            ok = False
            print(f"  [xx] MALFORMED   : last row has {pr['last_fields']} fields, header has {pr['header_fields']}")

    print()
    print("=" * 78)
    print("COVERAGE (from the verifier's full label scan, if available)")
    print("=" * 78)
    if prior.get("combined"):
        combined = {k: int(v) for k, v in prior["combined"].items()}
    if combined:
        total = sum(combined.values())
        for fam in FAMILY_ORDER:
            n = combined.get(fam, 0)
            mark = "     " if n else " [!!]"
            print(f" {mark} {fam:<14} {n:>12,}  ({n / total:6.2%})" if total else f" {mark} {fam}")
        missing = [f for f in FAMILY_ORDER[1:] if not combined.get(f)]
        print(f"\n  attack families NOT covered : {missing or 'none'}")
        print("  -> these are reported as 'not covered' in tier D, never as score 0.")
    elif not prior.get("found"):
        print("  no verifier report found - run scripts/verify_cicids2018.py --full first")
    else:
        print("  verifier report has no label counts - re-run it with --full")

    print()
    print("VERDICT           :", "READY TO BUILD" if ok else "BLOCKED - fix the [xx] lines above")
    return ok


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the CSE-CIC-IDS2018 tier-D evaluation set.")
    parser.add_argument("--dry-run", action="store_true", help="discover + audit only, write nothing")
    parser.add_argument("--file", default=None, help="substring filter, e.g. Wednesday-21")
    parser.add_argument("--limit-rows", type=int, default=None, help="stop after N rows per file (smoke test)")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="rows per chunk (default 100000)")
    parser.add_argument("--dedup", choices=("window", "none"), default="window",
                        help="window = drop exact duplicates vs previous chunk (default); none = keep every flow")
    parser.add_argument("--out", default=None, help="output dir (default data/external/_processed)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_from_settings(settings)
    settings.ensure_directories()

    external = Path(settings.paths.external)
    out_dir = Path(args.out) if args.out else external / OUT_SUBDIR
    files = discover(external, args.file)
    prior = load_prior_audit(Path(settings.paths.reports))

    if args.dry_run:
        return 0 if _print_dry_run(files, external, prior) else 1

    if not files:
        print(f"[xx] no CSV under {external} - nothing to build.")
        return 1

    print("=" * 78)
    print("CSE-CIC-IDS2018 build (tier D evaluation set)")
    print("=" * 78)
    print(f"external dir : {external}")
    print(f"output dir   : {out_dir}")
    print(f"files        : {len(files)}   chunksize: {args.chunksize:,}   dedup: {args.dedup}   limit-rows: {args.limit_rows}")
    print(f"feature space: {len(CANONICAL_FEATURES)} canonical -> {len(FEATURE_COLUMNS)} model features (float32)")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    for path in files:
        au = audit_header(path)
        if not au["complete"]:
            print(f"[xx] {path.name}: unresolved {au['unresolved']} - skipping (fix schema2018.py)")
            continue
        log.info("building %s ...", path.name)
        try:
            res = build_file(path, out_dir, args.chunksize, args.limit_rows, args.dedup)
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"\n[xx] {path.name}: {exc}")
            print("    -> fix FAMILY_RULES_2018 / COLUMN_ALIASES_2018 in src/ai_soar/data/schema2018.py,")
            print("       then re-run. Refusing to write a silently-incomplete evaluation set.")
            return 1
        res["probe"] = probe(path, external)
        res["audit"] = au
        results.append(res)
        print(f"  {path.name[:40]:<40} read={res['rows_read']:>10,} out={res['rows_out']:>10,} "
              f"nonfinite={res['rows_dropped_nonfinite']:>7,} dups={res['duplicates_dropped']:>8,} "
              f"junk={res['junk_rows']:>3} {res['seconds']}s -> {res['out_size_mb']} MB")

    if not results:
        print("[xx] nothing was built.")
        return 1

    combined: dict[str, int] = {}
    dups_by_family: dict[str, int] = {}
    for res in results:
        for fam, count in res["families"].items():
            combined[fam] = combined.get(fam, 0) + int(count)
        for fam, count in res.get("duplicates_by_family", {}).items():
            dups_by_family[fam] = dups_by_family.get(fam, 0) + int(count)
    total_out = sum(r["rows_out"] for r in results)
    total_read = sum(r["rows_read"] for r in results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_dataset2018.py",
        "dataset": "CSE-CIC-IDS2018",
        "purpose": "tier-D cross-dataset evaluation set (no training)",
        "external_dir": str(external),
        "output_dir": str(out_dir),
        "chunksize": args.chunksize,
        "limit_rows": args.limit_rows,
        "feature_space": {
            "canonical": len(CANONICAL_FEATURES),
            "model_features": len(FEATURE_COLUMNS),
            "dtype": "float32",
            "dropped_constant_features": 8,
        },
        "cleaning_rules": [
            "junk repeated-header rows dropped (Label == 'Label')",
            "rows with any Inf/NaN across the 78 canonical features dropped",
            f"duplicates: mode={args.dedup} (window = exact duplicates vs the previous chunk)",
            "Timestamp/Protocol/IP/FlowID identity columns dropped (calendar + identity leakage)",
            "unmapped 2018 labels raise instead of being silently discarded",
            "features stored as float32, matching the 2017 pipeline's load_split() cast",
        ],
        "dedup_mode": args.dedup,
        "dedup_limitation": (
            "Window dedup only compares each chunk with the previous one, so a run of "
            "identical flows longer than 2 x chunksize can leave duplicates in the output. "
            "This inflates per-flow counts proportionally; it cannot cause train/test leakage "
            "because the 2018 set is used for evaluation only."
        ),
        "totals": {
            "files": len(results),
            "rows_read": total_read,
            "rows_out": total_out,
            "rows_dropped_nonfinite": sum(r["rows_dropped_nonfinite"] for r in results),
            "duplicates_dropped": sum(r["duplicates_dropped"] for r in results),
            "junk_rows": sum(r["junk_rows"] for r in results),
            "inf_cells": sum(r["inf_cells"] for r in results),
            "nan_cells": sum(r["nan_cells"] for r in results),
            "seconds": round(time.time() - started, 1),
        },
        "families": {fam: combined.get(fam, 0) for fam in FAMILY_ORDER},
        "duplicates_by_family": {fam: dups_by_family.get(fam, 0) for fam in FAMILY_ORDER},
        "families_not_covered": [f for f in FAMILY_ORDER[1:] if not combined.get(f)],
        "row_cap_suspect_files": [
            r["file"] for r in results
            if prior_rows(prior.get("files", {}).get(r["file"], {}))[0] == ROW_CAP_SUSPECT
        ],
        "prior_audit": {
            "found": prior.get("found", False),
            "family_counts_combined": prior.get("combined", {}),
        },
        "files": results,
        "caveat": STANDING_CAVEAT,
    }
    report_path = settings.report_path("cicids2018_build_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 78)
    print("BUILT")
    print("=" * 78)
    print(f"files written      : {len(results)} -> {out_dir}")
    print(f"rows in / out      : {total_read:,} / {total_out:,}")
    print(f"dropped non-finite : {report['totals']['rows_dropped_nonfinite']:,}")
    print(f"duplicates dropped : {report['totals']['duplicates_dropped']:,}")
    print(f"junk header rows   : {report['totals']['junk_rows']:,}")
    print(f"elapsed            : {report['totals']['seconds']}s")
    print()
    for fam in FAMILY_ORDER:
        n = combined.get(fam, 0)
        mark = "    " if n else "[!!]"
        pct = f"{n / total_out:6.2%}" if total_out else "   n/a"
        print(f" {mark} {fam:<14} {n:>12,}  {pct}")
    if args.dedup == "window" and dups_by_family:
        print()
        print("per-family duplication (byte-identical flows removed):")
        print(f"  {'family':<14}{'kept':>12}{'dups':>12}{'dup rate':>11}")
        for fam in FAMILY_ORDER:
            kept_n = combined.get(fam, 0)
            dup_n = dups_by_family.get(fam, 0)
            if kept_n or dup_n:
                rate = dup_n / (kept_n + dup_n) if (kept_n + dup_n) else 0.0
                print(f"  {fam:<14}{kept_n:>12,}{dup_n:>12,}{rate:>11.2%}")
        print("  (window-local: runs longer than 2 x chunksize can survive)")

    missing = report["families_not_covered"]
    print(f"\nattack families NOT covered : {missing or 'none'}")
    if report["row_cap_suspect_files"]:
        print(f"row-cap suspect files       : {report['row_cap_suspect_files']}")
        print("  (exactly 1,048,575 rows = 2**20-1 = Excel's row limit: these days are")
        print("   truncated subsets, recorded in the report as a tier-D limitation)")
    print(f"\nreport             : {report_path}")
    print(f"\nnext               : python scripts/eval_cross_dataset.py   (File 4/4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())