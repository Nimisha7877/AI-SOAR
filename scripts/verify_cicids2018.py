#!/usr/bin/env python
"""Verify the CSE-CIC-IDS2018 download before any modelling touches it.

Why this file exists
--------------------
Step 9 (cross-dataset evaluation) only works if the 2018 data can be mapped onto
the SAME feature space and the SAME 8 families that the deployed models use.
The two datasets were produced by different CICFlowMeter versions, so column
names drift (e.g. 2017 "Fwd Packet Length Max" vs 2018 "Fwd Packet Size Max"),
2018 ships extra identity columns (Flow ID, IPs, Protocol, Timestamp) and its
raw label vocabulary is different ("DDOS attack-HOIC", "Brute Force -XSS",
"FTP-BruteForce", ...).

This script answers, per file, before we write any loader:
  1. does the header match what we expect, and what is extra/missing?
  2. which 2017 canonical features are absent, and which can be recovered by a
     mechanical rename (Length<->Size, whitespace, case)?
  3. what raw labels exist, how do they map to our 8 families, and which labels
     are UNMAPPED (a hard blocker)?
  4. hygiene on a sample: Inf / NaN / negative-duration counts
  5. is the file usable for tier-D evaluation?  (verdict per file + overall)

Nothing here trains, writes parquet or mutates the CSVs. It only reads and
reports to ``artifacts/reports/cicids2018_verify_report.json``.

Run (fast, samples each file):
    python scripts/verify_cicids2018.py

Run (exhaustive label + row counts; slow on the 4 GB Tuesday file):
    python scripts/verify_cicids2018.py --full

One file only:
    python scripts/verify_cicids2018.py --file Friday-16-02-2018.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai_soar.config import get_settings  # noqa: E402
from ai_soar.data.features import DROP_FEATURES, FEATURE_COLUMNS  # noqa: E402
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_ORDER, normalize_label  # noqa: E402
from ai_soar.data.schema import CANONICAL_FEATURES, LABEL_COLUMN  # noqa: E402

REPORT_NAME = "cicids2018_verify_report.json"

# ---------------------------------------------------------------------------
# 2018 raw label -> family. RULE-BASED and ORDER-SENSITIVE: the first matching
# rule wins, so specific web attacks are tested before the generic
# "brute force" rule (2018 has "Brute Force -XSS", which is a web attack, not
# a credential attack). Every label that matches nothing is reported as
# UNMAPPED - that is a hard blocker for tier-D evaluation.
# ---------------------------------------------------------------------------
FAMILY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (BENIGN_LABEL, re.compile(r"^benign$")),
    ("DDoS", re.compile(r"ddos|loit|hoic")),
    ("DoS", re.compile(r"\bdos\b|hulk|goldeneye|slowloris|slowhttptest|heartbleed")),
    ("WebAttack", re.compile(r"xss|sql\s*injection|injection")),
    ("PortScan", re.compile(r"portscan|port\s*scan")),
    ("Infiltration", re.compile(r"infiltration")),
    ("Botnet", re.compile(r"^bot$|botnet|ares")),
    ("BruteForce", re.compile(r"brute|patator|weblogin")),
)

# CICFlowMeter v3 (2018) renamed columns with word swaps and abbreviations:
# "Total Backward Packets" -> "Tot Bwd Packets", "Fwd Packet Length Max" ->
# "Fwd Packet Size Max". Substring rules cannot handle two swaps at once, so
# instead we canonicalise every column name into synonym-normalised tokens and
# match on that key. This is a PROPOSAL only - nothing is renamed until the
# Step 9 loader does it explicitly and the eval proves it.
_SYNONYMS: dict[str, str] = {
    "tot": "total", "fwd": "forward", "bwd": "backward", "pkt": "packet",
    "pkts": "packets", "size": "length", "len": "length", "byts": "bytes",
    "avg": "average", "max": "maximum", "min": "minimum", "std": "stdev",
    "dest": "destination", "src": "source", "dst": "destination",
    "flw": "flow", "iat": "iat", "sy": "system", "win": "window",
}
_NON_WORD = re.compile(r"[^a-z0-9]+")


def canonical_key(name: str) -> str:
    """'Tot Bwd Pkts' and 'Total Backward Packets' -> the same key."""
    tokens = _NON_WORD.sub(" ", strip_name(name).lower()).split()
    out = []
    for tok in tokens:
        tok = _SYNONYMS.get(tok, tok)
        if tok.endswith("s") and tok not in ("bps", "pps") and len(tok) > 3:
            tok = tok[:-1]          # packets -> packet, bytes -> byte
        out.append(tok)
    return " ".join(out)


def family_from_2018_label(raw: str) -> Optional[str]:
    key = normalize_label(raw)
    for family, pattern in FAMILY_RULES:
        if pattern.search(key):
            return family
    return None


def strip_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def propose_rename(target: str, by_key: dict[str, str]) -> Optional[str]:
    """Find the 2018 column whose canonical key equals the 2017 name's key."""
    return by_key.get(canonical_key(target))


# ---------------------------------------------------------------------------
def find_csvs(external_dir: Path, only: Optional[str]) -> list[Path]:
    if not external_dir.exists():
        return []
    files = sorted(p for p in external_dir.rglob("*.csv") if p.is_file())
    if only:
        files = [p for p in files if p.name.lower() == only.lower() or only.lower() in p.name.lower()]
    return files


def read_sample(path: Path, rows: int) -> pd.DataFrame:
    return pd.read_csv(
        path,
        nrows=rows,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )


def label_counts_chunked(path: Path, label_col: str, chunksize: int = 400_000) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for chunk in pd.read_csv(
        path,
        usecols=[label_col],
        chunksize=chunksize,
        encoding="utf-8",
        encoding_errors="replace",
    ):
        total += len(chunk)
        counts.update(str(v).strip() for v in chunk[label_col].tolist())
        print(f"      ... {total:,} rows scanned", flush=True)
    return counts, total


def check_file(path: Path, args) -> dict[str, Any]:
    print(f"\n=== {path.name}  ({path.stat().st_size / 1e6:.0f} MB) ===", flush=True)
    result: dict[str, Any] = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "problems": [],
        "warnings": [],
    }

    # ---- 1. header -------------------------------------------------------
    sample_rows = args.sample_rows
    try:
        df = read_sample(path, sample_rows)
    except Exception as exc:  # noqa: BLE001 - report, never crash the audit
        result["problems"].append(f"unreadable: {exc}")
        print(f"  [xx] unreadable: {exc}")
        return result

    columns = [strip_name(c) for c in df.columns]
    result["n_columns"] = len(columns)
    result["columns"] = columns
    print(f"  columns            : {len(columns)}")

    label_candidates = [c for c in columns if c.lower() in ("label", "labels", "class", "attack")]
    label_col = LABEL_COLUMN if LABEL_COLUMN in columns else (label_candidates[0] if label_candidates else None)
    result["label_column"] = label_col
    if label_col is None:
        result["problems"].append("no Label column found")
        print("  [xx] no Label column")
    else:
        print(f"  label column       : {label_col}")

    # ---- 2. extra identity columns (expected in 2018, must be dropped) ---
    known = set(CANONICAL_FEATURES) | {LABEL_COLUMN}
    extras = [c for c in columns if c not in known]
    result["extra_columns"] = extras
    print(f"  extra (non-2017)   : {len(extras)} -> {extras[:8]}{'...' if len(extras) > 8 else ''}")

    # ---- 3. feature coverage vs the deployed 70-feature space ------------
    available = set(columns)
    exact = [c for c in FEATURE_COLUMNS if c in available]
    missing = [c for c in FEATURE_COLUMNS if c not in available]
    # index the not-already-matched 2018 columns by canonical key
    by_key: dict[str, str] = {}
    for col in columns:
        if col in exact:
            continue
        by_key.setdefault(canonical_key(col), col)
    renames: dict[str, str] = {}
    unresolved: list[str] = []
    for name in missing:
        proposal = propose_rename(name, by_key)
        if proposal and proposal not in renames.values():
            renames[name] = proposal
        else:
            unresolved.append(name)

    result["feature_space"] = {
        "model_features_expected": len(FEATURE_COLUMNS),
        "exact_match": len(exact),
        "recoverable_by_rename": renames,
        "unresolved": unresolved,
    }
    print(f"  model features     : {len(exact)}/{len(FEATURE_COLUMNS)} exact")
    if renames:
        print(f"  recoverable renames: {len(renames)}")
        for k, v in list(renames.items())[:6]:
            print(f"      2017 '{k}'  <-  2018 '{v}'")
    if unresolved:
        result["problems"].append(f"{len(unresolved)} model features cannot be resolved: {unresolved}")
        print(f"  [xx] UNRESOLVED    : {unresolved}")

    # constant columns we drop anyway - harmless if 2018 lacks them
    drop_missing = [c for c in DROP_FEATURES if c not in available]
    if drop_missing:
        result["warnings"].append(f"dropped-constant columns absent (harmless): {drop_missing}")

    # ---- 4. labels on the sample ----------------------------------------
    if label_col is not None:
        raw_counts = Counter(str(v).strip() for v in df[label_col].tolist())
        if args.full:
            print("  full label scan (this file may be large)...", flush=True)
            raw_counts, total_rows = label_counts_chunked(path, label_col)
            result["rows"] = total_rows
            print(f"  rows (full)        : {total_rows:,}")
        else:
            result["rows_sampled"] = int(len(df))
            print(f"  rows sampled       : {len(df):,}  (use --full for the whole file)")

        mapped: Counter = Counter()
        unmapped: Counter = Counter()
        for raw, n in raw_counts.items():
            fam = family_from_2018_label(raw)
            if fam is None:
                unmapped[raw] += n
            else:
                mapped[fam] += n
        result["raw_labels"] = dict(raw_counts.most_common())
        result["family_counts"] = dict(mapped.most_common())
        result["unmapped_labels"] = dict(unmapped.most_common())

        print(f"  raw label types    : {len(raw_counts)}")
        for raw, n in raw_counts.most_common(12):
            fam = family_from_2018_label(raw)
            flag = "" if fam else "   <-- UNMAPPED"
            print(f"      {n:>10,}  {raw:<32} -> {fam or '???'}{flag}")
        print(f"  families present   : {dict(mapped.most_common())}")
        if unmapped:
            result["problems"].append(f"unmapped labels: {dict(unmapped.most_common())}")

        for fam in ("Botnet", "Infiltration"):
            n = mapped.get(fam, 0)
            if n > 0:
                print(f"  [!!] {fam}: {n:,} rows (2017 had {'1' if fam == 'Botnet' else '36'} usable)")

    # ---- 5. hygiene on the sample ---------------------------------------
    numeric = df.select_dtypes(include="number").columns.tolist()
    num = df[numeric].apply(pd.to_numeric, errors="coerce")
    arr = num.to_numpy(dtype="float64")
    inf_cells = int(np.isinf(arr).sum())
    nan_cells = int(np.isnan(arr).sum())
    inf_by_col = pd.Series(np.isinf(arr).sum(axis=0), index=num.columns)
    inf_cols = {str(c): int(n) for c, n in inf_by_col[inf_by_col > 0].items()}
    nan_by_col = num.isna().sum()
    nan_cols = {str(c): int(n) for c, n in nan_by_col[nan_by_col > 0].items()}
    neg_duration = 0
    if "Flow Duration" in df.columns:
        neg_duration = int((pd.to_numeric(df["Flow Duration"], errors="coerce") < 0).sum())
    result["hygiene_sample"] = {
        "numeric_columns": len(numeric),
        "inf_cells": inf_cells,
        "nan_cells": nan_cells,
        "inf_columns": inf_cols,
        "nan_columns": nan_cols,
        "negative_flow_duration": neg_duration,
    }
    print(f"  hygiene (sample)   : numeric cols {len(numeric)}, Inf cells {inf_cells:,}, "
          f"NaN cells {nan_cells:,}, negative Flow Duration {neg_duration:,}")
    if inf_cols:
        print(f"      Inf in         : {list(inf_cols)[:5]}")
    if nan_cols:
        print(f"      NaN in         : {list(nan_cols)[:5]}")

    # ---- 6. timestamp (needed for temporal ordering later) --------------
    ts_col = next((c for c in columns if c.lower() in ("timestamp", "time", "date")), None)
    if ts_col:
        result["timestamp_column"] = ts_col
        result["timestamp_samples"] = [str(v) for v in df[ts_col].head(3).tolist()]
        print(f"  timestamp column   : {ts_col} e.g. {result['timestamp_samples'][0]}")
    else:
        result["warnings"].append("no Timestamp column - temporal ordering inside 2018 impossible")
        print("  [!!] no Timestamp column")

    # ---- 7. verdict -----------------------------------------------------
    if result["problems"]:
        result["verdict"] = "BLOCKED"
    elif result["warnings"]:
        result["verdict"] = "USABLE_WITH_CARE"
    else:
        result["verdict"] = "READY"
    print(f"  verdict            : {result['verdict']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the CSE-CIC-IDS2018 download.")
    parser.add_argument("--full", action="store_true",
                        help="count every row and every label (slow on multi-GB files)")
    parser.add_argument("--file", default=None, help="only check files whose name contains this")
    parser.add_argument("--sample-rows", type=int, default=100_000,
                        help="rows sampled per file for schema/hygiene checks (default 100000)")
    args = parser.parse_args()

    settings = get_settings()
    external = Path(settings.paths.external)
    files = find_csvs(external, args.file)

    print("=" * 74)
    print("CSE-CIC-IDS2018 verification (Step 9a prerequisite)")
    print("=" * 74)
    print(f"external dir         : {external}")
    print(f"files found          : {len(files)}")
    print(f"2017 canonical feats : {len(CANONICAL_FEATURES)} (+ Label)")
    print(f"deployed feature set : {len(FEATURE_COLUMNS)} (after dropping {len(DROP_FEATURES)} constant)")
    print(f"families expected    : {', '.join(FAMILY_ORDER[1:])}")
    if not files:
        print(f"\n[xx] no CSVs under {external} - put the 2018 files there (or fix paths.external).")
        return 1

    results = [check_file(p, args) for p in files]

    # ---------------- cross-file summary ----------------
    all_columns: set[str] = set()
    for r in results:
        all_columns.update(r.get("columns", []))
    exact_all = [c for c in FEATURE_COLUMNS if c in all_columns]
    renames_all: dict[str, str] = {}
    for r in results:
        renames_all.update(r.get("feature_space", {}).get("recoverable_by_rename", {}))
    unresolved_all = sorted(
        {u for r in results for u in r.get("feature_space", {}).get("unresolved", [])}
    )
    unmapped_all = sorted({k for r in results for k in r.get("unmapped_labels", {})})
    families_all: Counter = Counter()
    for r in results:
        families_all.update(r.get("family_counts", {}))
    verdicts = [r["verdict"] for r in results]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "full" if args.full else "sampled",
        "sample_rows_per_file": args.sample_rows,
        "files_checked": len(results),
        "union_of_columns": len(all_columns),
        "feature_coverage": {
            "exact": len(exact_all),
            "expected": len(FEATURE_COLUMNS),
            "recoverable_by_rename": renames_all,
            "unresolved": unresolved_all,
        },
        "family_counts_combined": dict(families_all.most_common()),
        "unmapped_labels": unmapped_all,
        "verdicts": {r["file"]: r["verdict"] for r in results},
        "overall": "BLOCKED" if "BLOCKED" in verdicts else
                   ("USABLE_WITH_CARE" if "USABLE_WITH_CARE" in verdicts else "READY"),
        "files": results,
    }

    report_path = Path(settings.paths.reports) / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"files checked        : {len(results)}")
    print(f"feature coverage     : {len(exact_all)}/{len(FEATURE_COLUMNS)} exact "
          f"+ {len(renames_all)} recoverable by rename")
    if unresolved_all:
        print(f"[xx] unresolved feats : {unresolved_all}")
    print(f"families (combined)  : {dict(families_all.most_common())}")
    if unmapped_all:
        print(f"[xx] UNMAPPED labels   : {unmapped_all}")
    for name, verdict in summary["verdicts"].items():
        print(f"  {verdict:<18} {name}")
    print(f"\nOVERALL              : {summary['overall']}")
    print(f"report               : {report_path}")

    if summary["overall"] == "READY":
        print("\nNext: Step 9 File 2 - schema mapper + 2018 parquet builder.")
    elif summary["overall"] == "USABLE_WITH_CARE":
        print("\nUsable, but read the warnings above before Step 9 File 2.")
    else:
        print("\nBLOCKED - fix the problems above before building the 2018 loader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())