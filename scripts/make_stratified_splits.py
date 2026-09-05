"""Create stratified 80/10/10 train/val/test splits by Family.

WHY THIS EXISTS
---------------
Profiling showed CICIDS2017 attack classes are day-segregated: Mon-Thu holds
{BruteForce, DoS, WebAttack, Infiltration}, Friday holds {DDoS, PortScan,
Botnet}. A pure day split cannot evaluate multiclass performance. So:

- PRIMARY split (this script): stratified by Family, seed 42, so every
  family appears in train, val and test. Used for model development and
  all per-class metrics.
- SECONDARY split: the day split (processed/train vs processed/test) is
  kept as a temporal stress test for BINARY detection only.

Reads every parquet under processed/train and processed/test (together they
contain all 8 sources), assigns each row a split by sampling within its
Family, and writes data/processed/strat/{train,val,test}/<source>.parquet.

Run:  python scripts/make_stratified_splits.py
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ai_soar.config import get_settings
from ai_soar.data.labels import FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("make_stratified_splits")

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
SEED = 42
BATCH_SIZE = 200_000
SPLITS = ("train", "val", "test")


def _assign_splits(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified split labels for n rows of ONE family, shuffled."""
    n_val = int(round(n * VAL_FRAC))
    n_test = int(round(n * (1.0 - TRAIN_FRAC - VAL_FRAC)))
    if n >= 3:
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)
    n_train = n - n_val - n_test
    if n_train < 1:                      # 1-2 row family: keep all in train
        n_val, n_test, n_train = 0, 0, n
    labels = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test, dtype=object
    )
    rng.shuffle(labels)
    return labels


def _split_source(
    src_file: Path, out_root: Path, rng: np.random.Generator, counts: dict
) -> None:
    dataset = ds.dataset(src_file, format="parquet")
    writers: dict[str, pq.ParquetWriter] = {}

    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        split_col = np.empty(len(df), dtype=object)
        for fam in df[FAMILY_COLUMN].unique():
            mask = (df[FAMILY_COLUMN] == fam).to_numpy()
            labels = _assign_splits(int(mask.sum()), rng)
            split_col[mask] = labels
            for split in SPLITS:
                counts[split][fam] += int((labels == split).sum())
        df["_split"] = split_col

        for split in SPLITS:
            part = df[df["_split"] == split].drop(columns="_split")
            if len(part) == 0:
                continue
            table = pa.Table.from_pandas(part, preserve_index=False)
            writer = writers.get(split)
            if writer is None:
                out_dir = out_root / split
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / src_file.name
                writer = pq.ParquetWriter(out_file, table.schema)
                writers[split] = writer
            writer.write_table(table)

    for writer in writers.values():
        writer.close()
    log.info("split %s", src_file.name)


def main() -> int:
    settings = get_settings()
    configure_from_settings(settings)

    processed = Path(settings.paths.processed)
    out_root = processed / "strat"
    if out_root.exists():
        shutil.rmtree(out_root)          # idempotent reruns

    sources = sorted((processed / "train").glob("*.parquet")) + sorted(
        (processed / "test").glob("*.parquet")
    )
    if not sources:
        raise SystemExit("No source parquets found - run scripts/build_dataset.py first.")

    rng = np.random.default_rng(SEED)
    counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in SPLITS}

    for src in sources:
        _split_source(src, out_root, rng, counts)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "fracs": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": 1 - TRAIN_FRAC - VAL_FRAC},
        "counts": {s: dict(counts[s]) for s in SPLITS},
    }
    report_path = settings.report_path("strat_split_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n================ STRATIFIED SPLITS ================")
    header = f"{'family':<14}" + "".join(f"{s:>12}" for s in SPLITS)
    print(header)
    for fam in FAMILY_ORDER:
        row = f"{fam:<14}" + "".join(f"{counts[s][fam]:>12,}" for s in SPLITS)
        print(row)
    totals = {s: sum(counts[s].values()) for s in SPLITS}
    print(f"{'TOTAL':<14}" + "".join(f"{totals[s]:>12,}" for s in SPLITS))
    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())