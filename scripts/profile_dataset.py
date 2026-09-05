"""Profile the processed dataset (data/processed) and write a JSON report.

Answers, with real numbers, the questions the model design depends on:

- how many rows per family in train vs test (and which families are absent
  from a split - Friday contains no WebAttack/Infiltration/etc.)
- imbalance ratio of every family against BENIGN
- which feature columns are constant or low-cardinality
- post-clean sanity: zero NaN / zero Inf must hold

Memory-safe: streams parquet in batches; per-column uniqueness is tracked
with a capped set so continuous columns cannot blow up RAM.

Run:  python scripts/profile_dataset.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from ai_soar.config import get_settings
from ai_soar.data.labels import FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.data.schema import LABEL_COLUMN
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("profile_dataset")

BATCH_SIZE = 200_000
UNIQUE_CAP = 20_000          # beyond this a column is marked high-cardinality
LOW_CARDINALITY = 20         # <= this many uniques == candidate for encoding


def _family_counts(split_dir: Path) -> dict[str, int]:
    dataset = ds.dataset(split_dir, format="parquet")
    counts: dict[str, int] = {}
    for batch in dataset.to_batches(columns=[FAMILY_COLUMN], batch_size=BATCH_SIZE):
        col = batch.to_pandas()[FAMILY_COLUMN]
        for fam, n in col.value_counts().items():
            counts[fam] = counts.get(fam, 0) + int(n)
    return {f: counts.get(f, 0) for f in FAMILY_ORDER}


def _profile_features(split_dir: Path) -> dict[str, dict]:
    dataset = ds.dataset(split_dir, format="parquet")
    acc: dict[str, dict] = {}

    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        for col in df.columns:
            if col in (LABEL_COLUMN, FAMILY_COLUMN):
                continue
            arr = df[col].to_numpy(dtype=float)
            st = acc.setdefault(
                col,
                {
                    "nan": 0,
                    "inf": 0,
                    "n": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": math.inf,
                    "max": -math.inf,
                    "uniq": set(),
                    "uniq_capped": False,
                },
            )
            st["nan"] += int(np.isnan(arr).sum())
            st["inf"] += int(np.isinf(arr).sum())
            finite = arr[np.isfinite(arr)]
            st["n"] += int(finite.size)
            if finite.size:
                st["sum"] += float(finite.sum())
                st["sumsq"] += float((finite * finite).sum())
                st["min"] = min(st["min"], float(finite.min()))
                st["max"] = max(st["max"], float(finite.max()))
                if not st["uniq_capped"]:
                    st["uniq"].update(np.unique(finite).tolist())
                    if len(st["uniq"]) > UNIQUE_CAP:
                        st["uniq_capped"] = True
                        st["uniq"] = set()   # free the memory
        log.debug("profiled batch of %d rows", len(df))

    report: dict[str, dict] = {}
    for col, st in acc.items():
        n = max(st["n"], 1)
        mean = st["sum"] / n
        var = max(st["sumsq"] / n - mean * mean, 0.0)
        n_uniq = -1 if st["uniq_capped"] else len(st["uniq"])
        report[col] = {
            "nan": st["nan"],
            "inf": st["inf"],
            "min": st["min"] if st["min"] != math.inf else None,
            "max": st["max"] if st["max"] != -math.inf else None,
            "mean": mean,
            "std": math.sqrt(var),
            "n_unique": n_uniq,
            "constant": (not st["uniq_capped"]) and len(st["uniq"]) <= 1,
            "low_cardinality": (not st["uniq_capped"]) and 1 < len(st["uniq"]) <= LOW_CARDINALITY,
        }
    return report


def main() -> int:
    settings = get_settings()
    configure_from_settings(settings)

    processed = Path(settings.paths.processed)
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "splits": {}}

    families: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        families[split] = _family_counts(processed / split)
        out["splits"][split] = {"family_counts": families[split]}
        log.info("counted families for %s", split)

    # imbalance ratios against BENIGN, per split
    for split, counts in families.items():
        benign = max(counts.get("BENIGN", 0), 1)
        out["splits"][split]["imbalance_ratio_vs_benign"] = {
            f: round(counts[f] and benign / counts[f] or 0, 1) for f in FAMILY_ORDER
        }
        out["splits"][split]["families_absent"] = [
            f for f in FAMILY_ORDER if counts[f] == 0
        ]

    feats = _profile_features(processed / "train")
    out["train_features"] = feats

    constant = [c for c, v in feats.items() if v["constant"]]
    low_card = [c for c, v in feats.items() if v["low_cardinality"]]
    dirty = [c for c, v in feats.items() if v["nan"] or v["inf"]]
    out["summary"] = {
        "constant_columns": constant,
        "low_cardinality_columns": low_card,
        "columns_with_nan_or_inf": dirty,
    }

    report_path = settings.report_path("profile_report.json")
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- human summary ----------------------------------------------------
    print("\n================ DATASET PROFILE ================")
    for split in ("train", "test"):
        counts = families[split]
        print(f"\n[{split}]")
        for f in FAMILY_ORDER:
            print(f"  {f:<14} {counts[f]:>10,}")
        absent = out["splits"][split]["families_absent"]
        if absent:
            print(f"  absent families : {', '.join(absent)}")

    print("\nconstant columns      :", constant or "none")
    print("low-cardinality cols  :", low_card or "none")
    print("cols with NaN/Inf     :", dirty or "none  (cleaning verified)")
    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())