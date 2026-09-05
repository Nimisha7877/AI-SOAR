"""Build the processed dataset: raw CSVs -> cleaned, labeled, split parquet.

Pipeline per source file:
    chunked load -> schema normalize -> clean (Inf/NaN/dups) -> family label
    -> write parquet into data/processed/<split>/<source>.parquet

Also writes artifacts/reports/dataset_report.json with per-source removal
stats and family counts, so every future decision can cite real numbers.

Run:  python scripts/build_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ai_soar.config import PROJECT_ROOT, get_settings
from ai_soar.data.cleaner import clean_dataframe, merge_stats
from ai_soar.data.labels import FAMILY_COLUMN, add_family_column
from ai_soar.data.loader import SOURCE_ORDER, discover_sources, iter_normalized_chunks
from ai_soar.data.splitter import split_for_key
from ai_soar.utils.logging import configure_from_settings, get_logger


def _process_source(source_key: str, csv_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{source_key}.parquet"
    if out_file.exists():
        out_file.unlink()  # idempotent reruns

    chunk_stats: list[dict] = []
    families: Counter = Counter()
    rows_out = 0
    writer: pq.ParquetWriter | None = None

    for i, chunk in enumerate(iter_normalized_chunks(csv_path)):
        cleaned, stats = clean_dataframe(chunk)
        chunk_stats.append(stats)
        if len(cleaned) == 0:
            continue
        df = add_family_column(cleaned)
        families.update(df[FAMILY_COLUMN].value_counts().to_dict())

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)
        rows_out += len(df)
        log.info("%s chunk %d: %d -> %d rows", source_key, i, stats["rows_in"], stats["rows_out"])

    if writer is not None:
        writer.close()

    merged = merge_stats(chunk_stats)
    merged["split"] = split_for_key(source_key)
    merged["parquet"] = str(out_file.relative_to(PROJECT_ROOT))
    merged["families"] = dict(families)
    log.info("%s DONE: %d -> %d rows", source_key, merged["rows_in"], rows_out)
    return merged


def main() -> int:
    global log
    settings = get_settings()
    configure_from_settings(settings)
    log = get_logger("build_dataset")

    sources = discover_sources(settings.paths.raw)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }

    for key in SOURCE_ORDER:
        split = split_for_key(key)
        out_dir = Path(settings.paths.processed) / split
        report["sources"][key] = _process_source(key, sources[key], out_dir)

    # ---- totals -----------------------------------------------------------
    totals = merge_stats([report["sources"][k] for k in SOURCE_ORDER])
    family_totals: Counter = Counter()
    split_rows: Counter = Counter()
    for key in SOURCE_ORDER:
        entry = report["sources"][key]
        family_totals.update(entry["families"])
        split_rows[entry["split"]] += entry["rows_out"]
    report["totals"] = totals
    report["family_totals"] = dict(family_totals)
    report["split_rows"] = dict(split_rows)

    report_path = settings.report_path("dataset_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- human summary ----------------------------------------------------
    print("\n================ DATASET BUILD COMPLETE ================")
    print(f"rows in   : {totals['rows_in']:,}")
    print(f"rows out  : {totals['rows_out']:,}")
    print(f"inf cells : {totals['inf_cells']:,}")
    print(f"nan rows  : {totals['rows_with_nan_dropped']:,}")
    print(f"dup rows  : {totals['duplicate_rows_dropped']:,}")
    print(f"train rows: {split_rows['train']:,}   test rows: {split_rows['test']:,}")
    print("\nfamily totals:")
    for fam, n in family_totals.most_common():
        print(f"  {fam:<14} {n:>10,}")
    print(f"\nreport: {report_path}")
    return 0


log = get_logger("build_dataset")

if __name__ == "__main__":
    raise SystemExit(main())