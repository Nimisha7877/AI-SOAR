"""Chunked loader for the CICIDS2017 CSVs in data/raw.

Never loads the whole dataset into memory. Every source file is streamed in
chunks of ``CHUNK_SIZE`` rows, each chunk normalized through the schema so
downstream code only ever sees canonical columns.

Source keys (used later for the temporal train/test split)::

    monday, tuesday, wednesday,
    thursday_morning, thursday_afternoon,
    friday_morning, friday_afternoon_portscan, friday_afternoon_ddos
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ai_soar.data.schema import normalize_dataframe, validate_dataframe
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

CHUNK_SIZE = 200_000

# ordered: earlier days first (matters for temporal splitting)
SOURCE_ORDER = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday_morning",
    "thursday_afternoon",
    "friday_morning",
    "friday_afternoon_portscan",
    "friday_afternoon_ddos",
)


def source_key_from_name(filename: str) -> str:
    """Map a CICIDS2017 filename (any mirror casing) to a canonical source key."""
    name = filename.lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if name.startswith(day):
            break
    else:
        raise ValueError(f"Unrecognised CICIDS2017 filename: {filename}")

    if day in ("monday", "tuesday", "wednesday"):
        return day

    part = "morning" if "morning" in name else "afternoon"
    if "portscan" in name:
        return f"{day}_{part}_portscan"
    if "ddos" in name or "ddo" in name:
        return f"{day}_{part}_ddos"
    return f"{day}_{part}"


def discover_sources(raw_dir: Path | str) -> dict[str, Path]:
    """Find the 8 source CSVs. Raises if anything is missing or unexpected."""
    raw_dir = Path(raw_dir)
    found: dict[str, Path] = {}
    for csv in sorted(raw_dir.glob("*.csv")):
        key = source_key_from_name(csv.name)
        if key in found:
            raise ValueError(f"Duplicate source key '{key}': {csv} vs {found[key]}")
        found[key] = csv

    missing = [k for k in SOURCE_ORDER if k not in found]
    if missing:
        raise ValueError(f"Missing source files for: {missing}")
    extra = [k for k in found if k not in SOURCE_ORDER]
    if extra:
        raise ValueError(f"Unexpected source files: {extra}")

    log.info("Discovered %d source files in %s", len(found), raw_dir)
    return {k: found[k] for k in SOURCE_ORDER}


def iter_normalized_chunks(
    path: Path | str, chunk_size: int = CHUNK_SIZE
) -> Iterator[pd.DataFrame]:
    """Stream one CSV as normalized, schema-validated chunks."""
    path = Path(path)
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    for i, chunk in enumerate(reader):
        df = normalize_dataframe(chunk)
        problems = validate_dataframe(df)
        if any(problems.values()):
            log.warning("Schema problems in %s chunk %d: %s", path.name, i, problems)
        yield df
    log.debug("Finished streaming %s", path.name)


def load_source(path: Path | str, max_rows: int | None = None) -> pd.DataFrame:
    """Load ONE source file fully into memory (tests / small experiments only)."""
    frames = []
    rows = 0
    for chunk in iter_normalized_chunks(path):
        frames.append(chunk)
        rows += len(chunk)
        if max_rows is not None and rows >= max_rows:
            break
    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        df = df.head(max_rows)
    log.info("Loaded %s rows from %s", len(df), Path(path).name)
    return df