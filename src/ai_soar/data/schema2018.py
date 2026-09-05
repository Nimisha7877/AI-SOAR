"""CSE-CIC-IDS2018 -> CICIDS2017 canonical schema mapping.

Why this module exists
----------------------
Tier-D evaluation (cross-dataset) is only meaningful if the 2018 flows are fed
to the deployed models through the SAME 70-feature space. The two datasets were
produced by different CICFlowMeter versions, and v3 (2018) abbreviates column
names::

    2017 "Total Length of Fwd Packets"  ->  2018 "TotLen Fwd Pkts"
    2017 "FIN Flag Count"               ->  2018 "FIN Flag Cnt"
    2017 "Packet Length Variance"       ->  2018 "Pkt Len Var"
    2017 "Average Packet Size"          ->  2018 "Pkt Size Avg"
    2017 "Init_Win_bytes_forward"       ->  2018 "Init Fwd Win Byts"
    2017 "act_data_pkt_fwd"             ->  2018 "Fwd Act Data Pkts"
    2017 "min_seg_size_forward"         ->  2018 "Fwd Seg Size Min"

2018 also ships identity/meta columns that 2017's mirror does not (``Flow ID``,
``Src IP``, ``Dst IP``, ``Src Port``, ``Protocol``, ``Timestamp``) and a
different raw label vocabulary (``DDOS attack-HOIC``, ``Brute Force -XSS``,
``FTP-BruteForce``, ``DoS attacks-SlowHTTPTest``, ...).

Everything here is an EXPLICIT, reviewable table: no fuzzy guessing at runtime.
:func:`map_columns_2018` reports anything it cannot resolve so a schema change
shows up as a hard error instead of silently mis-training a model.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from ai_soar.data.labels import BENIGN_LABEL, normalize_label
from ai_soar.data.schema import CANONICAL_FEATURES, LABEL_COLUMN

# ---------------------------------------------------------------------------
# 1. identity / meta columns present in 2018 but NOT part of the feature space.
#    Timestamp is dropped deliberately: it is calendar leakage (the models must
#    not learn "attacks happen on Tuesday"), exactly as in the 2017 pipeline.
#    NOTE: "Dst Port" is NOT here - it maps to the canonical "Destination Port".
# ---------------------------------------------------------------------------
IDENTITY_COLUMNS_2018: tuple[str, ...] = (
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Protocol",
    "Timestamp",
)

# ---------------------------------------------------------------------------
# 2. 2018 raw column name -> 2017 canonical feature name.
# ---------------------------------------------------------------------------
COLUMN_ALIASES_2018: dict[str, str] = {
    # ports / packet & byte totals
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    # per-direction packet length stats
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    # rates
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    # inter-arrival totals
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    # header lengths (the second occurrence arrives as ".1" from pandas,
    # mirroring the duplicate column in the 2017 mirror)
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Header Len.1": "Fwd Header Length.1",
    # overall packet length stats
    "Pkt Len Min": "Packet Length Min",
    "Pkt Len Max": "Packet Length Max",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "Pkt Size Avg": "Average Packet Size",
    # TCP flag counts
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    # segment sizes
    "Fwd Seg Size Avg": "Fwd Segment Size Avg",
    "Bwd Seg Size Avg": "Bwd Segment Size Avg",
    "Fwd Seg Size Min": "min_seg_size_forward",
    # bulk statistics. Two spellings exist across 2018 mirrors: the downloaded
    # "TrafficForML_CICFlowMeter" files use "Fwd Byts/b Avg" style, other
    # exports use "Fwd Avg Pkts/Bulk" style. All six are constant in 2017 and
    # therefore dropped from the model space - mapping them anyway keeps the
    # audit honest instead of reporting them as unknown columns.
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Fwd Avg Pkts/Bulk": "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulks Rate": "Fwd Avg Bulk Rate",
    "Bwd Avg Pkts/Bulk": "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulks Rate": "Bwd Avg Bulk Rate",
    # subflows
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    # window / data-packet counters (2017 uses snake_case here)
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
}

# ---------------------------------------------------------------------------
# 2b. columns the 2017 mirror only has because its CSV header repeats a name.
#     CICIDS2017 lists "Fwd Header Length" twice, so pandas exposes the second
#     copy as "Fwd Header Length.1" - and that copy is part of the deployed
#     70-feature space. CSE-CIC-IDS2018 lists it ONCE. Before duplicating we
#     must know the two 2017 copies hold identical values (scripts check this);
#     if they do, copying is exact and lossless, not an approximation.
# ---------------------------------------------------------------------------
DUPLICATE_COLUMNS_2018: dict[str, str] = {
    "Fwd Header Length.1": "Fwd Header Length",
}

# ---------------------------------------------------------------------------
# 3. 2018 raw label -> family. ORDER-SENSITIVE: the first matching rule wins,
#    so "Brute Force -XSS" is a WebAttack (it is a DVWA web-app attack) while
#    "Brute Force -Web" / "FTP-BruteForce" / "SSH-BruteForce" are BruteForce.
# ---------------------------------------------------------------------------
FAMILY_RULES_2018: tuple[tuple[str, re.Pattern[str]], ...] = (
    (BENIGN_LABEL, re.compile(r"^benign$")),
    ("DDoS", re.compile(r"ddos|loit|hoic")),
    ("DoS", re.compile(r"\bdos\b|hulk|goldeneye|slowloris|slowhttptest|heartbleed")),
    ("WebAttack", re.compile(r"xss|sql\s*injection|injection")),
    ("PortScan", re.compile(r"portscan|port\s*scan")),
    ("Infiltration", re.compile(r"infiltration")),
    ("Botnet", re.compile(r"^bot$|botnet|ares")),
    ("BruteForce", re.compile(r"brute|patator|weblogin")),
)

# 2018 label vocabulary seen in the six downloaded files (kept for reporting).
KNOWN_2018_LABELS: tuple[str, ...] = (
    "Benign",
    "DoS attacks-Hulk",
    "DoS attacks-GoldenEye",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIT-HTTP",
    "Brute Force -Web",
    "Brute Force -WebLogin",
    "Brute Force -XSS",
    "FTP-BruteForce",
    "SSH-BruteForce",
    "SQL Injection",
    "XSS",
    "Infiltration",
    "Bot",
    "PortScan",
    "Label",  # header artefact rows occasionally appear in the 2018 CSVs
)

_LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "attack")

# The 2018 CSVs are known to contain repeated header rows mid-file (the label
# cell literally reads "Label"). Those are junk rows, not a new class: drop
# them and report the count instead of failing the whole build.
INVALID_LABEL_VALUES = frozenset({"label", "nan", "", "none", "null"})


def family_from_2018_label(raw: str) -> Optional[str]:
    """Map one 2018 raw label onto our 8 families. ``None`` = unmapped."""
    key = normalize_label(raw)
    for family, pattern in FAMILY_RULES_2018:
        if pattern.search(key):
            return family
    return None


def strip_name(name: Any) -> str:
    """Collapse whitespace so ' Flow  Duration ' matches the canonical name."""
    return re.sub(r"\s+", " ", str(name)).strip()


def detect_label_column(columns: list[str]) -> Optional[str]:
    for col in columns:
        if col.lower() in _LABEL_COLUMN_CANDIDATES:
            return col
    return None


def map_columns_2018(columns: list[str], required: tuple[str, ...] = CANONICAL_FEATURES) -> dict[str, Any]:
    """Audit a 2018 header against the canonical 2017 feature space.

    Returns a dict with:
      ``exact``      - canonical names present verbatim
      ``aliases``    - {2018 name: canonical name} needed to complete the space
      ``unresolved`` - canonical names with no 2018 counterpart (hard blocker)
      ``identity``   - meta columns to drop
      ``unknown``    - 2018 columns that are neither canonical, aliased nor meta
    """
    clean = [strip_name(c) for c in columns]
    available = set(clean)
    exact = [name for name in required if name in available]

    aliases: dict[str, str] = {}
    for raw, canonical in COLUMN_ALIASES_2018.items():
        if raw in available and canonical in required and canonical not in exact:
            aliases[raw] = canonical

    # the source column may itself only exist after aliasing ("Fwd Header Len"
    # -> "Fwd Header Length"), so test coverage, not raw presence
    covered_by_name = set(exact) | set(aliases.values())
    duplicated = {
        target: source
        for target, source in DUPLICATE_COLUMNS_2018.items()
        if target in required and source in covered_by_name and target not in available
    }

    covered = covered_by_name | set(duplicated)
    unresolved = [name for name in required if name not in covered]

    identity = [c for c in clean if c in IDENTITY_COLUMNS_2018]
    unknown = [c for c in clean
               if c not in available.intersection(set(required))
               and c not in aliases
               and c not in identity
               and c != LABEL_COLUMN
               and strip_name(c) not in COLUMN_ALIASES_2018]

    return {
        "exact": exact,
        "exact_count": len(exact),
        "aliases": aliases,
        "alias_count": len(aliases),
        "covered_count": len(covered),
        "required_count": len(required),
        "unresolved": unresolved,
        "duplicated": duplicated,
        "identity": identity,
        "unknown": sorted(set(unknown)),
        "complete": not unresolved,
    }


def apply_2018_schema(df: pd.DataFrame, required: tuple[str, ...] = CANONICAL_FEATURES) -> pd.DataFrame:
    """Rename 2018 columns to canonical names and select the feature space + Label.

    Raises ``KeyError`` listing every missing feature - a silent partial feature
    vector would poison tier-D evaluation, so this must fail loudly.
    """
    out = df.copy()
    out.columns = [strip_name(c) for c in out.columns]

    rename = {raw: canonical for raw, canonical in COLUMN_ALIASES_2018.items() if raw in out.columns}
    if rename:
        out = out.rename(columns=rename)

    for target, source in DUPLICATE_COLUMNS_2018.items():
        if target in required and target not in out.columns and source in out.columns:
            out[target] = out[source]

    label_col = detect_label_column(list(out.columns)) or LABEL_COLUMN
    missing = [name for name in required if name not in out.columns]
    if missing:
        raise KeyError(
            f"2018 file is missing {len(missing)} canonical features: {missing}. "
            "Update COLUMN_ALIASES_2018 in src/ai_soar/data/schema2018.py."
        )

    keep = list(required) + ([label_col] if label_col in out.columns else [])
    return out[keep]


def add_family_column(df: pd.DataFrame, label_col: str = LABEL_COLUMN) -> pd.DataFrame:
    """Attach the 8-family ``Family`` column.

    Junk rows (repeated headers) are dropped and reported on ``df.attrs``;
    genuinely unknown labels raise, because silently discarding a real attack
    class would corrupt tier-D evaluation.
    """
    out = df.copy()
    normalized = [normalize_label(v) for v in out[label_col].tolist()]
    junk = [i for i, key in enumerate(normalized) if key in INVALID_LABEL_VALUES]
    if junk:
        out = out.drop(index=[out.index[i] for i in junk])
        normalized = [normalize_label(v) for v in out[label_col].tolist()]

    families = [family_from_2018_label(v) for v in out[label_col].tolist()]
    unknown = sorted({str(out[label_col].iloc[i]) for i, f in enumerate(families) if f is None})
    if unknown:
        raise ValueError(
            f"unmapped 2018 labels: {unknown}. Extend FAMILY_RULES_2018 "
            "in src/ai_soar/data/schema2018.py (do not silently drop rows)."
        )
    out["Family"] = families
    out.attrs["dropped_junk_rows"] = len(junk)
    return out


__all__ = [
    "COLUMN_ALIASES_2018",
    "DUPLICATE_COLUMNS_2018",
    "INVALID_LABEL_VALUES",
    "FAMILY_RULES_2018",
    "IDENTITY_COLUMNS_2018",
    "KNOWN_2018_LABELS",
    "add_family_column",
    "apply_2018_schema",
    "detect_label_column",
    "family_from_2018_label",
    "map_columns_2018",
    "strip_name",
]