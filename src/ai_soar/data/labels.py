"""Label handling: collapse CICIDS2017's 15 raw classes into 8 families.

Why collapse: ``Heartbleed`` has ~11 rows in the whole dataset against
2.27M BENIGN. No classifier can learn 11 examples. Collapsing to attack
families makes every class learnable and matches how a SOC actually
responds ("brute force", not "FTP-Patator specifically").

Families (8)::

    BENIGN, BruteForce, DoS, DDoS, PortScan, WebAttack, Botnet, Infiltration
"""

from __future__ import annotations

import re

import pandas as pd

from ai_soar.data.schema import BENIGN_LABEL, LABEL_COLUMN
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

FAMILY_COLUMN = "Family"

FAMILY_ORDER: tuple[str, ...] = (
    BENIGN_LABEL,
    "BruteForce",
    "DoS",
    "DDoS",
    "PortScan",
    "WebAttack",
    "Botnet",
    "Infiltration",
)

# normalized raw label -> family. Keys are lowercased, dashes unified.
_RAW_TO_FAMILY: dict[str, str] = {
    "benign": BENIGN_LABEL,
    # BruteForce (credential attacks)
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    # DoS (single-source denial of service; heartbleed behaves as DoS here)
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "DoS",
    # DDoS (multi-source)
    "ddos": "DDoS",
    # Reconnaissance
    "portscan": "PortScan",
    # Web attacks
    "web attack - xss": "WebAttack",
    "web attack - sql injection": "WebAttack",
    "web attack - brute force": "WebAttack",
    # Botnet
    "bot": "Botnet",
    # Infiltration
    "infiltration": "Infiltration",
}

# Any hyphen/dash variant PLUS mangled-byte artifacts (U+FFFD replacement
# char, cp1252 leftovers) collapse to a single '-'. This makes label mapping
# encoding-agnostic: en dash, em dash, or a corrupted byte all map the same.
_DASH_LIKE = re.compile(r"[\-\u2010-\u2015\u2212\u0096\u0097\ufffd]+")
_SPACES = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    """Unify dash styles, mangled bytes and spacing so mirror variants map."""
    s = str(label).strip()
    s = _DASH_LIKE.sub("-", s)
    s = _SPACES.sub(" ", s)
    return s.lower()


def family_from_label(label: str) -> str:
    """Map one raw label to its family. Raises on anything unknown."""
    key = normalize_label(label)
    try:
        return _RAW_TO_FAMILY[key]
    except KeyError:
        raise ValueError(
            f"Unknown CICIDS2017 label {label!r}. "
            "Add it to _RAW_TO_FAMILY in ai_soar/data/labels.py."
        ) from None


def add_family_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``Family`` column derived from ``Label``."""
    out = df.copy()
    out[FAMILY_COLUMN] = out[LABEL_COLUMN].map(family_from_label)
    return out


def family_counts(df: pd.DataFrame) -> pd.Series:
    """Counts per family, in FAMILY_ORDER (useful for profiling reports)."""
    counts = df[FAMILY_COLUMN].value_counts()
    return counts.reindex(FAMILY_ORDER, fill_value=0)