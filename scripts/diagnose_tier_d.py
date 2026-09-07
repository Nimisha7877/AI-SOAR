"""Diagnose why the stage-1 gate misses CSE-CIC-IDS2018 attacks.

WHY THIS EXISTS
---------------
The tier-D smoke run reported two facts that cannot both be true by accident::

    stage-1 AUC on 2018  = 0.8561     (the gate RANKS attacks above benign)
    detection recall@0.5 = 0.0000%    (but not ONE attack crosses the cut)
    false-positive rate  = 0.0109%    (benign scores are also very low)

High AUC + zero recall means the whole probability scale has slid downwards:
2018 flows score in a band that sits under the 0.5 cut, while still sitting
above benign. There are only a few causes, and they need opposite responses:

  (a) SCORE DEFLATION under distribution shift - a different lab network,
      different victims, different throughput. Ranking survives, the threshold
      does not transfer. -> an HONEST tier-D finding. Report it, do not "fix" it.
  (b) A FEATURE UNIT/SCALE MISMATCH introduced by the 2018 adapter (a column
      that is microseconds in one dataset and milliseconds in the other, or a
      byte/packet count off by 1000x). -> a BUG. Fix before the full run.
  (c) POSITIONAL MISALIGNMENT - the booster expects a different column order
      than FEATURE_COLUMNS, so every score is meaningless. -> a BUG, fatal.

This script separates them in about a minute by measuring four things:

  1. alignment   - feature count, booster names and model metadata. Note that
                   LightGBM stores placeholders (Column_0..) when fitted on a
                   numpy array, which is what features.X_y() hands it, so the
                   metadata written at train time is the guard that matters.
  2. control     - the SAME gate on the 2017 stratified test split. If the
                   control recalls ~99% at 0.5, the model is not broken; the
                   problem is in the 2018 inputs.
  3. scores      - percentile distribution of P(malicious) for attack and
                   benign rows in BOTH datasets, plus a threshold sweep that
                   asks "is there ANY operating point that transfers?"
  4. features    - for the gate's highest-gain features, 2017 vs 2018 on
                   BENIGN traffic (the cleanest unit test) and on attack
                   traffic (noisier, informational only). Cause (b) needs three
                   things together, because any one alone is ordinary drift:
                     * the IQR moved by more than ~5x,
                     * the p5-p95 windows barely overlap (intersection/union),
                     * ONE factor explains every quantile (a unit change is a
                       pure multiply; a wrong-column mapping is not).
                   Zero-centred columns (flag counts) are exempt: they cannot
                   have a microseconds-vs-milliseconds bug.
                   This is also checked FILE BY FILE, since pooling six days
                   can dilute a single-day mapping error into a bimodal blob
                   that no longer looks uniform - and the six 2018 CSVs do not
                   all share one header layout.

It reads a SPREAD sample of each 2018 parquet (evenly spaced row groups plus a
random sample inside each, never head-only) because those CSVs are grouped by
victim machine: the first 300k rows describe one machine's attack, not the day.

OUTPUTS
-------
Console report, plus ``artifacts/reports/tier_d_diagnostic.json``. It never
writes to ``tier_d_cross_dataset.json`` and never touches the deployed models,
so it is safe to run as often as you like. Nothing here changes any threshold:
re-tuning on 2018 would leak the evaluation set back into the model.

Run:
    venv\\Scripts\\python.exe scripts\\diagnose_tier_d.py

Flags:
    --dir PATH          2018 parquet dir (default data/external/_processed)
    --file SUBSTR       restrict to parquets whose name contains SUBSTR
    --per-file N        sampled rows per 2018 file (default 150000)
    --ref-rows N        sampled rows from the 2017 test split (default 200000)
    --ref-split NAME    2017 split to use as the control (default test)
    --thr F             deployed gate threshold to judge against (default 0.5)
    --top-features N    how many high-gain features to compare (default 15)
    --models-dir PATH   default artifacts/models
    --no-json           console only, skip the report file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from ai_soar.config import get_settings
from ai_soar.data.features import FEATURE_COLUMNS, load_split
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.evaluation.metrics import save_report
from ai_soar.inference.predictor import (
    DEFAULT_GATE_THRESHOLD,
    METADATA_FILENAME,
    STAGE1_FILENAME,
)
from ai_soar.models.binary import load_model as load_binary_model
from ai_soar.models.binary import malicious_probability
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("diagnose_tier_d")

IN_SUBDIR = "_processed"
SOURCE_COLUMN = "SourceFile"
SWEEP: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001)
PCTS: tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)
SCALE_LOG_TOL = 0.7        # |log10(IQR ratio)| above this = ~5x scale move
OVERLAP_TOL = 0.10         # ...and below this much p5-p95 overlap = unit suspect
QQ_SPREAD_TOL = 0.30       # log10 spread of quantile ratios below this = uniform scaling
SPREAD_GROUPS = 6          # evenly spaced row groups sampled per 2018 file
RANDOM_STATE = 42
REPORT_NAME = "tier_d_diagnostic.json"
FEATS: list[str] = list(FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _f(x: Any) -> Optional[float]:
    """JSON-safe float: never NaN/inf, which json.dumps would write illegally."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if np.isfinite(v) else None


def cell(v: Optional[float], width: int = 9, digits: int = 4, pct: bool = False) -> str:
    """Right-aligned, None-safe numeric cell - an empty slice must not crash a run."""
    if v is None:
        return "n/a".rjust(width)
    txt = f"{round(float(v), digits):,.{digits}f}"
    return (txt + "%").rjust(width) if pct else txt.rjust(width)


def gcell(v: Optional[float], width: int = 12, sig: int = 4) -> str:
    """Same, but 4 significant digits - feature magnitudes span many orders."""
    if v is None:
        return "n/a".rjust(width)
    return f"{float(v):>{width},.{sig}g}"


def pctl(a: np.ndarray) -> dict[str, Any]:
    """Percentile summary of a score/feature vector (empty-safe)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        out: dict[str, Any] = {"n": 0, "min": None, "max": None}
        out.update({f"p{p}": None for p in PCTS})
        return out
    vals = np.percentile(a, list(PCTS))
    out = {"n": int(a.size), "min": _f(a.min()), "max": _f(a.max())}
    out.update({f"p{p}": _f(v) for p, v in zip(PCTS, vals)})
    return out


def auc_safe(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    return _f(roc_auc_score(y_true, score))


class ColumnStats:
    """Streaming per-column n / mean / min / max / zero% / non-finite count."""

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        k = len(self.names)
        self.n = np.zeros(k, dtype=np.int64)
        self.s = np.zeros(k, dtype=np.float64)
        self.mn = np.full(k, np.inf, dtype=np.float64)
        self.mx = np.full(k, -np.inf, dtype=np.float64)
        self.zero = np.zeros(k, dtype=np.int64)
        self.bad = np.zeros(k, dtype=np.int64)

    def update(self, X: np.ndarray) -> None:
        if X.shape[0] == 0:
            return
        X = np.asarray(X, dtype=np.float64)
        finite = np.isfinite(X)
        self.n += finite.sum(axis=0)
        Xz = np.where(finite, X, 0.0)
        self.s += Xz.sum(axis=0)
        self.bad += (~finite).sum(axis=0)
        self.zero += ((Xz == 0.0) & finite).sum(axis=0)
        self.mn = np.minimum(self.mn, np.where(finite, X, np.inf).min(axis=0))
        self.mx = np.maximum(self.mx, np.where(finite, X, -np.inf).max(axis=0))

    def row(self, i: int) -> dict[str, Any]:
        n = int(self.n[i])
        return {
            "feature": self.names[i],
            "n": n,
            "mean": _f(self.s[i] / n) if n else None,
            "min": _f(self.mn[i]) if np.isfinite(self.mn[i]) else None,
            "max": _f(self.mx[i]) if np.isfinite(self.mx[i]) else None,
            "zero_pct": _f(100.0 * self.zero[i] / n) if n else None,
            "non_finite": int(self.bad[i]),
            "constant": bool(n and np.isfinite(self.mn[i]) and self.mn[i] == self.mx[i]),
        }


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def sample_parquet(path: Path, cap: int, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Read a SPREAD sample of one parquet: evenly spaced row groups.

    Head-only sampling would be wrong here - the 2018 CSVs are grouped by
    victim machine, so the first N rows describe one machine's traffic.
    """
    pf = pq.ParquetFile(path)
    names = list(pf.schema.names)
    missing = [c for c in cols if c not in names]
    if missing:
        raise SystemExit(f"{path.name}: missing columns {missing}")
    total = int(pf.metadata.num_rows)
    groups = max(1, int(pf.metadata.num_row_groups))
    if total == 0:
        return pd.DataFrame(columns=cols), 0
    if total <= cap:
        df = pf.read(columns=cols).to_pandas()
    else:
        per_group = max(1, total // groups)
        parts: list[pd.DataFrame] = []
        if per_group > cap:
            # one oversized row group: stride over batches, never read it whole
            batch = max(1, min(cap, 65_536))
            got = 0
            for i, chunk in enumerate(pf.iter_batches(batch_size=batch, columns=cols)):
                if i % 2 == 0:
                    part = chunk.to_pandas()
                    parts.append(part)
                    got += len(part)
                if got >= cap:
                    break
        else:
            # evenly spaced row groups, plus a random sample INSIDE each one, so
            # the result spans the whole file rather than describing the first
            # victim machine it happens to start with
            k = max(1, min(groups, SPREAD_GROUPS))
            picks = sorted({int(v) for v in np.linspace(0, groups - 1, k).round()})
            per_pick = max(1, cap // max(1, len(picks)))
            for i in picks:
                part = pf.read_row_group(i, columns=cols).to_pandas()
                if len(part) > per_pick:
                    part = part.sample(n=per_pick, random_state=RANDOM_STATE)
                parts.append(part)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
        if len(df) > cap:
            df = df.sample(n=cap, random_state=RANDOM_STATE).reset_index(drop=True)
    df[FEATS] = df[FEATS].astype(np.float32)
    return df, total


def load_reference(split: str, ref_rows: int) -> tuple[pd.DataFrame, str]:
    """2017 control frame: stratified test first, day split as fallback."""
    for strat in (True, False):
        where = f"data/processed/strat/{split}" if strat else f"data/processed/{split}"
        try:
            return load_split(split, stratified=strat, max_rows=ref_rows), where
        except FileNotFoundError:
            continue
    raise SystemExit(
        f"no 2017 '{split}' split found - run scripts/build_dataset.py then "
        "scripts/make_stratified_splits.py"
    )


def find_parquets(base: Path, only: Optional[str]) -> list[Path]:
    if not base.exists():
        raise SystemExit(f"2018 parquet dir missing: {base} - run scripts/build_dataset2018.py")
    hits = sorted(base.glob("*.parquet"))
    if only:
        needle = only.lower()
        hits = [p for p in hits if needle in p.name.lower()]
    if not hits:
        raise SystemExit(f"no parquets matched under {base}")
    return hits


def gate_feature_importance(model: Any) -> tuple[list[str], np.ndarray, str]:
    """(names, gain, kind) - gain if the booster exposes it, else split counts."""
    try:
        names = list(model.booster_.feature_name())
        gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
        return names, gain, "gain"
    except Exception:                                   # pragma: no cover
        gain = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
        return list(FEATURE_COLUMNS), gain, "split"


#: LightGBM invents these when it is fitted on a plain numpy array, which is
#: exactly what ``features.X_y()`` hands it - so the deployed booster almost
#: certainly carries placeholders, NOT the real column names. Treating that as
#: misalignment would be a false alarm.
AUTO_NAME_RE = re.compile(r"^(?:Column_|column_|col_|f|F|x|X|feature_|Feature_)\d+$")


def classify_feature_names(names: list[str]) -> str:
    """'real' = they match FEATURE_COLUMNS, 'auto' = placeholders, else 'mismatch'."""
    if list(names) == FEATS:
        return "real"
    if names and all(AUTO_NAME_RE.match(str(n)) for n in names):
        return "auto"
    return "mismatch"


# --------------------------------------------------------------------------
# scoring pass over one dataframe
# --------------------------------------------------------------------------
def score_frame(model: Any, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(P(malicious), is_attack) for a frame that carries FEATURE_COLUMNS + Family."""
    X = df[FEATS].to_numpy(dtype=np.float32)
    y = df[FAMILY_COLUMN].to_numpy(dtype=object)
    return malicious_probability(model, X).astype(np.float32), (y != BENIGN_LABEL)


# --------------------------------------------------------------------------
# report sections
# --------------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_scores(label: str, atk: np.ndarray, ben: np.ndarray, thr: float) -> dict[str, Any]:
    print(f"\n  {label}")
    print(f"    {'pop':<8}{'n':>10}   " + "".join(f"{'p' + str(p):>9}" for p in PCTS)
          + f"{'max':>9}{'>=' + str(thr):>9}")
    for name, arr in (("attack", atk), ("benign", ben)):
        d = pctl(arr)
        frac = _f(100.0 * float((arr >= thr).mean())) if arr.size else None
        cells = "".join(cell(d[f"p{p}"]) for p in PCTS)
        print(f"    {name:<8}{d['n']:>10,}   {cells}{cell(d['max'])}"
              f"{cell(frac, 9, 3, pct=True)}")
    return {
        "attack": pctl(atk),
        "benign": pctl(ben),
        "auc": auc_safe(np.concatenate([np.ones(atk.size, np.int8), np.zeros(ben.size, np.int8)]),
                        np.concatenate([atk, ben])) if (atk.size and ben.size) else None,
        f"pct_attack_ge_{thr}": _f(100.0 * float((atk >= thr).mean())) if atk.size else None,
        f"pct_benign_ge_{thr}": _f(100.0 * float((ben >= thr).mean())) if ben.size else None,
    }


def sweep_table(thrs: tuple[float, ...], new_a: np.ndarray, new_b: np.ndarray,
                ref_a: np.ndarray, ref_b: np.ndarray, thr: float) -> list[dict[str, Any]]:
    print(f"\n  {'threshold':>10} | {'2018 recall':>12} {'2018 FPR':>10} "
          f"| {'2017 recall':>12} {'2017 FPR':>10}")
    print("  " + "-" * 68)
    rows: list[dict[str, Any]] = []
    for t in thrs:
        rec_n = _f(100.0 * float((new_a >= t).mean())) if new_a.size else None
        fpr_n = _f(100.0 * float((new_b >= t).mean())) if new_b.size else None
        rec_r = _f(100.0 * float((ref_a >= t).mean())) if ref_a.size else None
        fpr_r = _f(100.0 * float((ref_b >= t).mean())) if ref_b.size else None
        mark = "   <-- deployed" if abs(t - thr) < 1e-9 else ""
        print(f"  {t:>10.3f} | {cell(rec_n, 12, pct=True)} {cell(fpr_n, 10, pct=True)}"
              f" | {cell(rec_r, 12, pct=True)} {cell(fpr_r, 10, pct=True)}{mark}")
        rows.append({"threshold": t, "recall_2018_pct": rec_n, "fpr_2018_pct": fpr_n,
                     "recall_2017_pct": rec_r, "fpr_2017_pct": fpr_r})
    return rows


def _const(stats: ColumnStats, name: str) -> Optional[bool]:
    """Is this feature constant in the sample? None if the name is not tracked."""
    try:
        return bool(stats.row(stats.names.index(name))["constant"])
    except ValueError:
        return None


def _iqr(pa: dict[str, Any]) -> Optional[float]:
    if pa.get("p25") is None or pa.get("p75") is None:
        return None
    return float(pa["p75"]) - float(pa["p25"])


def _overlap(pa: dict[str, Any], pb: dict[str, Any]) -> Optional[float]:
    """Intersection-over-union of the two p5-p95 windows (1.0 = same range).

    A unit change (microseconds vs milliseconds, bytes vs packets) moves or
    rescales the ENTIRE distribution, so the windows stop overlapping. Ordinary
    drift keeps them largely on top of each other. This is one of the three
    things that separate a bug from a finding.
    """
    lo_a, hi_a = pa.get("p5"), pa.get("p95")
    lo_b, hi_b = pb.get("p5"), pb.get("p95")
    if None in (lo_a, hi_a, lo_b, hi_b):
        return None
    width = float(hi_a) - float(lo_a)
    if width <= 0:
        return None
    inter = min(float(hi_a), float(hi_b)) - max(float(lo_a), float(lo_b))
    union = max(float(hi_a), float(hi_b)) - min(float(lo_a), float(lo_b))
    if union <= 0:
        return None
    # Intersection over UNION, not over the 2017 width: a zero-centred column
    # that is simply scaled up swallows the 2017 window entirely, which would
    # read as "100% overlap" and hide the very thing we are looking for.
    return _f(max(0.0, inter) / union)


def qq_scale(pa: dict[str, Any], pb: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """(median quantile ratio, log10 spread of the quantile ratios).

    This is the discriminator between a unit bug and ordinary drift. A unit
    change multiplies EVERY quantile by the same factor, so the ratios agree
    and the spread stays near zero. Drift (or a wrong-column mapping) moves
    different parts of the distribution by different amounts, so the spread
    blows up. Quantiles whose 2017 value is within 5% of the 2017 IQR of zero
    are skipped, and so are sign flips - a ratio there is meaningless.
    """
    iqr_a = _iqr(pa)
    if not iqr_a or iqr_a <= 0:
        return None, None
    tol = 0.05 * iqr_a
    ratios: list[float] = []
    for q in PCTS:
        a, b = pa.get(f"p{q}"), pb.get(f"p{q}")
        if a is None or b is None or abs(a) < tol or a * b <= 0:
            continue
        ratios.append(float(b) / float(a))
    if len(ratios) < 3:
        return None, None
    lr = np.log10(np.abs(np.asarray(ratios, dtype=np.float64)))
    return _f(float(np.median(ratios))), _f(float(lr.max() - lr.min()))


#: multipliers that mean a unit changed rather than the network did
UNIT_HINTS: tuple[tuple[float, str], ...] = (
    (1_000_000.0, "micro-units vs base units (1e6) - e.g. microseconds vs seconds"),
    (1_000.0, "milli-units vs base units (1e3) - e.g. microseconds vs milliseconds"),
    (1_024.0, "KiB/MiB vs bytes"),
    (8.0, "bits vs bytes"),
)


def unit_hint(ratio: Optional[float]) -> Optional[str]:
    if not ratio or ratio <= 0:
        return None
    for factor, text in UNIT_HINTS:
        for r in (ratio, 1.0 / ratio):
            if abs(np.log10(r) - np.log10(factor)) < 0.05:
                return text
    return None


def unit_flag(pa: dict[str, Any], pb: dict[str, Any]) -> Optional[tuple[str, Optional[float]]]:
    """(flag text, uniform qq ratio) when the unit-change fingerprint is present.

    All four conditions must hold, because any one alone is ordinary drift:

      1. the IQR moved past SCALE_LOG_TOL (~5x),
      2. the p5-p95 windows barely overlap (IoU below OVERLAP_TOL),
      3. one factor explains every quantile (qq spread below QQ_SPREAD_TOL) -
         this is what a unit change does and a wrong-column mapping does not,
      4. neither distribution is centred on zero. A flag count cannot have a
         microseconds-vs-milliseconds bug, and a zero-centred column that merely
         widened is not identifiable as a unit change at all.
    """
    iqr_a, iqr_b = _iqr(pa), _iqr(pb)
    if not pa.get("n") or not pb.get("n") or not iqr_a or iqr_b is None or iqr_a <= 0:
        return None
    ratio = iqr_b / iqr_a
    if ratio <= 0 or abs(np.log10(ratio)) < SCALE_LOG_TOL:
        return None
    ov = _overlap(pa, pb)
    if ov is None or ov >= OVERLAP_TOL:
        return None
    qq_r, qq_s = qq_scale(pa, pb)
    if qq_s is None or qq_s > QQ_SPREAD_TOL or qq_r is None:
        return None
    ma, mb = pa.get("p50"), pb.get("p50")
    if ma is None or mb is None or abs(ma) < 0.1 * iqr_a or abs(mb) < 0.1 * iqr_b:
        return None
    txt = f"UNIT? uniform x{qq_r:,.4g} (qq spread {qq_s:.2f})"
    hint = unit_hint(qq_r)
    if hint:
        txt += f" - {hint}"
    return txt, qq_r


def unit_suspects(cols_file: dict[str, np.ndarray], ref_ben: dict[str, dict[str, Any]],
                  top: list[str], min_rows: int = 200) -> list[str]:
    """Top-gain features whose BENIGN spread moved >5x, stopped overlapping, and
    did so UNIFORMLY across quantiles - checked one file at a time.

    Per-file matters: pooling all six days together can hide a mismatch that
    affects only one file, and these CSVs do not share a single header layout
    (Thuesday-20-02 carries four extra identity columns), so a per-day mapping
    error is a real possibility rather than a hypothetical.
    """
    out: list[str] = []
    for c in top:
        v = cols_file.get(c)
        pa = ref_ben.get(c) or {}
        if v is None or v.size < min_rows or not pa.get("n"):
            continue
        hit = unit_flag(pa, pctl(v))
        if hit:
            out.append(f"{c} (x{hit[1]:,.4g})" if hit[1] else c)
    return out


def scale_table(stats_new: ColumnStats, stats_ref: ColumnStats, cols_new: dict[str, np.ndarray],
                cols_ref: dict[str, np.ndarray], top: list[str], kind: str) -> list[dict[str, Any]]:
    """Robust scale comparison for the gate's highest-gain features.

    The ratio is IQR-based, NOT median-based: many flow features are centred on
    or near zero, where a median ratio is pure noise (0.001 / 0.02 reads as
    "20x" while both columns are indistinguishable from zero). A feature is
    only called a unit suspect when three things hold together: the IQR moved
    past SCALE_LOG_TOL, the two 90% windows barely overlap, and the quantile
    ratios are uniform (see qq_scale). Any one of those alone is ordinary drift.
    """
    print(f"\n  {kind} - top-{len(top)} gate features by importance")
    print(f"    {'feature':<26}{'2017 med':>12}{'2018 med':>12}{'IQR ratio':>11}"
          f"{'overlap':>9}{'qq ratio':>11}  flag")
    print("    " + "-" * 100)
    rows: list[dict[str, Any]] = []
    for name in top:
        a, b = cols_ref.get(name), cols_new.get(name)
        pa = pctl(a) if a is not None else {}
        pb = pctl(b) if b is not None else {}
        ma, mb = pa.get("p50"), pb.get("p50")
        iqr_a, iqr_b = _iqr(pa), _iqr(pb)
        ratio = _f(iqr_b / iqr_a) if (iqr_a and iqr_b is not None and iqr_a > 0) else None
        ov = _overlap(pa, pb)
        qq_r, qq_s = qq_scale(pa, pb)
        flag = ""
        if not pa.get("n") or not pb.get("n"):
            flag = "NO DATA"
        elif ratio is None:
            flag = "2017 IQR = 0" if iqr_a == 0 else "IQR undefined"
        else:
            hit = unit_flag(pa, pb)
            if hit:
                flag = hit[0]
            elif abs(np.log10(ratio)) >= SCALE_LOG_TOL:
                if ov is not None and ov < OVERLAP_TOL:
                    sp = f"{qq_s:.2f}" if qq_s is not None else "n/a"
                    flag = f"shifted {ratio:,.3g}x, non-uniform (qq spread {sp}) - drift-like"
                else:
                    flag = f"shifted {ratio:,.3g}x, windows still overlap"
        if (ma == 0.0) != (mb == 0.0):
            flag = (flag + " " if flag else "") + "zero-drift"
        c_new, c_ref = _const(stats_new, name), _const(stats_ref, name)
        if c_ref and c_new is False:
            flag = (flag + " " if flag else "") + "2017-CONST"
        if c_new:
            flag = (flag + " " if flag else "") + "2018-CONST"
        ovs = cell(_f(100 * ov) if ov is not None else None, 9, 1, pct=True)
        print(f"    {name:<26}{gcell(ma)}{gcell(mb)}{gcell(ratio, 11, 3)}{ovs}"
              f"{gcell(qq_r, 11)}  {flag}")
        rows.append({
            "feature": name, "median_2017": ma, "median_2018": mb,
            "iqr_2017": _f(iqr_a) if iqr_a is not None else None,
            "iqr_2018": _f(iqr_b) if iqr_b is not None else None,
            "iqr_ratio": ratio, "overlap_pct": _f(100 * ov) if ov is not None else None,
            "qq_ratio": qq_r, "qq_spread_log10": qq_s,
            "p5_2017": pa.get("p5"), "p95_2017": pa.get("p95"),
            "p5_2018": pb.get("p5"), "p95_2018": pb.get("p95"),
            "unit_hint": unit_hint(qq_r) if qq_r else None,
            "flag": flag or None,
        })
    return rows


def suspect_detail(sc: list[dict[str, Any]], which: str) -> None:
    """Print raw quantiles for flagged features so their units can be eyeballed."""
    hit = [r for r in sc if r["flag"] and "UNIT?" in r["flag"]]
    if not hit:
        return
    g = lambda v: "n/a" if v is None else f"{v:,.6g}"                              # noqa: E731
    print(f"\n  {which} - raw quantiles for the flagged feature(s):")
    for r in hit:
        print(f"    {r['feature']}")
        print(f"      2017 benign  p5 / p50 / p95 : {g(r['p5_2017'])} / "
              f"{g(r['median_2017'])} / {g(r['p95_2017'])}")
        print(f"      2018 benign  p5 / p50 / p95 : {g(r['p5_2018'])} / "
              f"{g(r['median_2018'])} / {g(r['p95_2018'])}")
        print(f"      IQR ratio {g(r['iqr_ratio'])}, uniform quantile ratio {g(r['qq_ratio'])}")
        if r["unit_hint"]:
            print(f"      -> consistent with {r['unit_hint']}")
        else:
            print("      -> no standard unit factor matches; check the source documentation")


def family_table(fam_new: dict[str, np.ndarray], thr: float) -> list[dict[str, Any]]:
    print(f"\n  per attack family in the 2018 sample (gate score percentiles)")
    print(f"    {'family':<15}{'n':>9}{'p5':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p95':>9}"
          f"{'max':>9}{'>=' + str(thr):>9}")
    print("    " + "-" * 87)
    rows: list[dict[str, Any]] = []
    for fam in FAMILY_ORDER:
        if fam == BENIGN_LABEL:
            continue
        arr = fam_new.get(fam, np.empty(0, dtype=np.float32))
        d = pctl(arr)
        frac = _f(100.0 * float((arr >= thr).mean())) if arr.size else None
        c9 = lambda k: cell(d[k])                                                  # noqa: E731
        print(f"    {fam:<15}{d['n']:>9,}" + "".join(c9(f"p{p}") for p in (5, 25, 50, 75, 95))
              + c9("max") + cell(frac, 9, 3, pct=True))
        rows.append({"family": fam, **d, f"pct_ge_{thr}": frac})
    return rows


def file_table(per_file: dict[str, dict[str, Any]],
               thr: float) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    print(f"\n  per source file in the 2018 sample")
    print(f"    {'file':<46}{'atk n':>9}{'atk med':>10}{'>=thr':>8}"
          f"{'ben n':>10}{'ben med':>10}{'>=thr':>8}{'AUC':>8}")
    print("    " + "-" * 109)
    rows: list[dict[str, Any]] = []
    for name, d in per_file.items():
        print(f"    {name[:44]:<46}{d['n_attack']:>9,}{cell(d['attack_median'], 10)}"
              f"{cell(d['attack_pct_ge'], 8, 2, pct=True)}{d['n_benign']:>10,}"
              f"{cell(d['benign_median'], 10)}{cell(d['benign_pct_ge'], 8, 2, pct=True)}"
              f"{cell(d.get('auc'), 8)}")
        for sus in (d.get("unit_suspects") or []):
            print(f"        [!] unit suspect here: {sus}")
        rows.append({"file": name, **d})

    # systematic (every file) vs file-specific (that day's header variant)
    names = list(per_file)
    sus_by_file = {n: {x.split(" (")[0] for x in (per_file[n].get("unit_suspects") or [])}
                   for n in names}
    systematic = sorted(set.intersection(*sus_by_file.values())) if names and all(
        sus_by_file.values()) else []
    specific = sorted({x for v in sus_by_file.values() for x in v} - set(systematic))
    if systematic:
        print(f"\n  [!] flagged in EVERY file - systematic, suspect the adapter/schema mapping:")
        for f in systematic:
            print(f"        {f}")
    if specific:
        print(f"\n  [!] flagged in only SOME files - suspect that day's header variant:")
        for f in specific:
            where = [n[:28] for n in names if f in sus_by_file[n]]
            print(f"        {f}  ({', '.join(where)})")
    if not systematic and not specific:
        print("\n  [ok] no per-file unit suspects: every day's columns scale like the 2017 ones.")
    return rows, systematic, specific

    

def verdict(alignment_ok: bool, ctrl_rec: Optional[float], new_rec: Optional[float],
            auc_new: Optional[float], sweep: list[dict[str, Any]], scale_flags: list[str],
            thr: float, n_atk: int, n_ben: int) -> tuple[str, list[str]]:
    """Pick the single most likely diagnosis and say what to do about it.

    Two probes come out of the sweep and drive the decision:
      * t50  - the lowest cut that would recall half the 2018 attacks, and the
               false-positive rate it costs.
      * t1   - the lowest cut that still keeps the FP rate under 1%, and the
               recall it buys.
    A gate that only lost its SCALE answers t50 cheaply; a gate that lost its
    RANKING does not answer it at any price.
    """
    t50 = next((r for r in reversed(sweep) if (r["recall_2018_pct"] or 0) >= 50.0), None)
    t1 = next((r for r in reversed(sweep) if (r["fpr_2018_pct"] or 100.0) <= 1.0), None)
    lines: list[str] = []

    if n_atk == 0 or n_ben == 0:
        cause = ("NOT ENOUGH DATA - this sample holds "
                 f"{n_atk:,} attack and {n_ben:,} benign rows; both must be non-zero to judge "
                 "anything.")
        lines.append("Widen the sample (--per-file), drop the --file filter, or check the label "
                     "mapping for that day.")
    elif not alignment_ok:
        cause = ("(c) POSITIONAL MISALIGNMENT - the deployed booster's feature space is not "
                 "the FEATURE_COLUMNS this code sends it.")
        lines.append("This is a BUG, not a finding: every score produced so far is meaningless.")
        lines.append("Fix the load/adapter path before running the full tier-D evaluation.")
    elif scale_flags and (new_rec is None or new_rec < 50.0):
        cause = ("(b) SUSPECT FEATURE UNIT/SCALE MISMATCH on high-gain gate features - the "
                 "numbers above are NOT trustworthy until this is cleared.")
        lines.append("Flagged (IQR moved >5x, windows barely overlap, and one factor explains "
                     "every quantile): " + ", ".join(scale_flags))
        lines.append("Verify those columns by eye in src/ai_soar/data/schema2018.py against the "
                     "CSE-CIC-IDS2018 documentation, then re-run this diagnostic.")
        lines.append("If the shift is systematic across all six days it is an adapter bug; if it "
                     "is one day only, check that day's header variant (Thuesday-20-02 carries "
                     "four extra identity columns).")
    elif new_rec is not None and new_rec >= 50.0:
        cause = "NONE - the gate transfers to this sample at the deployed threshold."
        lines.append("The smoke slice was unrepresentative; run the full evaluation.")
    elif auc_new is not None and auc_new >= 0.70 and t50 and (t50["fpr_2018_pct"] or 100) <= 10.0:
        cause = ("(a) SCORE DEFLATION under distribution shift - the ranking survived, the 0.5 "
                 "threshold did not transfer.")
        lines.append(f"A cut of {t50['threshold']} would recall "
                     f"{round(t50['recall_2018_pct'] or 0, 1)}% of 2018 attacks at "
                     f"{round(t50['fpr_2018_pct'] or 0, 3)}% FP.")
        lines.append("Report tier D as recall@0.5 AND AUC, and state plainly that 0.5 was "
                     "chosen on 2017 data only.")
        lines.append("Do NOT re-tune the threshold on 2018 - that would leak the evaluation "
                     "set back into the model. Re-tuning belongs to Step 9b, with a new split.")
    elif auc_new is not None and auc_new >= 0.70:
        cause = ("(a) SCORE DEFLATION with heavy overlap - ranking partly intact, but no clean "
                 "operating point exists on this sample.")
        if t50:
            lines.append(f"Recall 50% would cost {round(t50['fpr_2018_pct'] or 0, 2)}% FP "
                         f"(cut {t50['threshold']}).")
        if t1:
            lines.append(f"Holding FP under 1% (cut {t1['threshold']}) buys only "
                         f"{round(t1['recall_2018_pct'] or 0, 1)}% recall.")
        lines.append("Still a legitimate tier-D finding: report AUC alongside recall@0.5.")
    elif auc_new is not None and auc_new < 0.65:
        cause = ("RANKING DEGRADED - the 2017 decision surface barely separates 2018 attack "
                 "from 2018 benign.")
        lines.append("Weaker than pure deflation: this argues for the merged-training A/B "
                     "(Step 9b) rather than a threshold change.")
    else:
        cause = "WEAK SEPARATION (AUC 0.65-0.70) - partial transfer, no usable cut."
        lines.append("Treat tier D as evidence that 2017-only training does not generalise to "
                     "a different lab network.")

    if cause.startswith("(a)") or cause.startswith("RANKING") or cause.startswith("WEAK"):
        lines.append("Next: run the FULL evaluation - this diagnostic samples ~150k rows per "
                     "file, and the 2018 CSVs are grouped by victim machine.")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    fmt = lambda v, sfx="%": "n/a" if v is None else f"{round(v, 4)}{sfx}"           # noqa: E731
    print(f"  control (2017 test) recall @{thr} : {fmt(ctrl_rec)}")
    print(f"  2018 sample         recall @{thr} : {fmt(new_rec)}")
    print(f"  2018 sample                    AUC : {fmt(auc_new, '')}")
    print(f"  probe t50  (recall>=50%)           : "
          + ("none" if not t50 else
             f"cut {t50['threshold']} -> FP {fmt(t50['fpr_2018_pct'])}"))
    print(f"  probe t1   (FPR<=1%)               : "
          + ("none" if not t1 else
             f"cut {t1['threshold']} -> recall {fmt(t1['recall_2018_pct'])}"))
    print(f"\n  DIAGNOSIS: {cause}")
    for ln in lines:
        print(f"    - {ln}")
    return cause, lines


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose tier-D gate behaviour on CSE-CIC-IDS2018.")
    settings = get_settings()
    ap.add_argument("--dir", type=Path, default=Path(settings.paths.external) / IN_SUBDIR)
    ap.add_argument("--file", default=None, help="substring filter on parquet filename")
    ap.add_argument("--per-file", type=int, default=150_000)
    ap.add_argument("--ref-rows", type=int, default=200_000)
    ap.add_argument("--ref-split", default="test")
    ap.add_argument("--thr", type=float, default=DEFAULT_GATE_THRESHOLD)
    ap.add_argument("--top-features", type=int, default=15)
    ap.add_argument("--models-dir", type=Path, default=Path(settings.paths.models))
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args(argv)

    configure_from_settings(settings)
    t0 = time.time()

    stage1_path = args.models_dir / STAGE1_FILENAME
    if not stage1_path.exists():
        print(f"[xx] trained gate not found: {stage1_path}", file=sys.stderr)
        return 1
    stage1 = load_binary_model(stage1_path)
    meta_path = args.models_dir / METADATA_FILENAME
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    boost_names, gain, gain_kind = gate_feature_importance(stage1)
    name_kind = classify_feature_names(boost_names)
    count_ok = len(boost_names) == len(FEATS)
    meta_feats = meta.get("feature_columns")
    meta_ok = (not meta_feats) or list(meta_feats) == FEATS
    # Auto-generated placeholders carry no order information, so the real guard
    # is the metadata written at train time: it records the exact column order
    # the booster was fitted on, and predictor.load() re-checks it at startup.
    alignment_ok = count_ok and meta_ok and name_kind in ("real", "auto")

    files = find_parquets(Path(args.dir), args.file)
    ref_df, ref_where = load_reference(args.ref_split, args.ref_rows)

    print("=" * 78)
    print("TIER-D DIAGNOSTIC  -  is the 0% recall a finding or a bug?")
    print("=" * 78)
    print(f"  gate model       : {stage1_path.name}")
    print(f"  deployed thr     : {args.thr}   sweep: {', '.join(str(s) for s in SWEEP)}")
    print(f"  2018 parquets    : {len(files)} file(s) from {args.dir}  "
          f"(<= {args.per_file:,} spread rows each)")
    print(f"  2017 control     : {ref_where}  ({len(ref_df):,} rows)")
    print(f"  importance basis : {gain_kind}")

    # ---- section 1: alignment -------------------------------------------
    section("1. ALIGNMENT  (must pass before any score means anything)")
    print(f"  booster feature count == {len(FEATS)}                : {count_ok}"
          f"   (booster reports {len(boost_names)})")
    print(f"  booster feature names                  : {name_kind}"
          + ("  <- LightGBM placeholders, order is positional" if name_kind == "auto" else ""))
    print(f"  metadata feature_columns == FEATURE_COLUMNS : {meta_ok}"
          + ("   <- the order guard that matters" if name_kind == "auto" else ""))
    print(f"  => alignment trustworthy                 : {alignment_ok}")
    if name_kind == "mismatch":
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(boost_names, FEATS)) if a != b]
        print(f"  [xx] {len(diff)} position(s) differ; first 10:")
        for i, a, b in diff[:10]:
            print(f"       [{i:>2}] booster={a!r}  code={b!r}")
    if not meta_ok and meta_feats:
        print(f"  [xx] metadata lists {len(list(meta_feats))} features, expected {len(FEATS)}")

    # ---- score both populations -----------------------------------------
    # `gain` is positional in the model's own feature order, which the metadata
    # check above has just confirmed to be FEATURE_COLUMNS.
    top_idx = (list(np.argsort(-gain)[: args.top_features]) if gain.size
               else list(range(min(15, len(FEATS)))))
    top_names = [FEATS[i] for i in top_idx]
    print(f"\n  top-{len(top_names)} gate features by {gain_kind}: "
          + ", ".join(top_names[:6]) + (", ..." if len(top_names) > 6 else ""))

    stats_new = ColumnStats(FEATS)
    stats_ref = ColumnStats(FEATS)
    # Retained per-feature values, split by truth. Kept separate because the two
    # datasets have very different class mixes (a 2018 slice can be 70% attack,
    # the 2017 test split ~10%), so pooled medians would compare apples to pears.
    kept_new_atk: dict[str, list[np.ndarray]] = {c: [] for c in top_names}
    kept_new_ben: dict[str, list[np.ndarray]] = {c: [] for c in top_names}
    kept_ref_atk: dict[str, list[np.ndarray]] = {c: [] for c in top_names}
    kept_ref_ben: dict[str, list[np.ndarray]] = {c: [] for c in top_names}
    new_atk: list[np.ndarray] = []
    new_ben: list[np.ndarray] = []
    new_fam: dict[str, list[np.ndarray]] = {f: [] for f in FAMILY_ORDER}
    per_file: dict[str, dict[str, Any]] = {}
    nonfinite_cells = 0
    sampled_rows = 0

    cat = lambda parts: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)  # noqa: E731

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # The 2017 control goes FIRST: its benign quantiles are the yardstick
        # every 2018 file is then compared against, one file at a time.
        Xr = ref_df[FEATS].to_numpy(dtype=np.float64)
        stats_ref.update(Xr)
        ref_score, ref_atk = score_frame(stage1, ref_df)
        for c in top_names:
            v = ref_df[c].to_numpy(dtype=np.float32)
            kept_ref_atk[c].append(v[ref_atk])
            kept_ref_ben[c].append(v[~ref_atk])
        del ref_df, Xr
        ref_ben_pctl = {c: pctl(cat(v)) for c, v in kept_ref_ben.items()}

        for path in files:
            df, total = sample_parquet(path, args.per_file, FEATS + [FAMILY_COLUMN, SOURCE_COLUMN])
            if df.empty:
                continue
            X = df[FEATS].to_numpy(dtype=np.float64)
            nonfinite_cells += int((~np.isfinite(X)).sum())
            stats_new.update(X)
            score, is_atk = score_frame(stage1, df)
            fam = df[FAMILY_COLUMN].to_numpy(dtype=object)
            sampled_rows += len(df)

            new_atk.append(score[is_atk])
            new_ben.append(score[~is_atk])
            for f in FAMILY_ORDER:
                m = fam == f
                if m.any():
                    new_fam[f].append(score[m])

            file_ben: dict[str, np.ndarray] = {}
            for c in top_names:
                v = df[c].to_numpy(dtype=np.float32)
                kept_new_atk[c].append(v[is_atk])
                kept_new_ben[c].append(v[~is_atk])
                file_ben[c] = v[~is_atk]

            a, b = score[is_atk], score[~is_atk]
            per_file[path.name.replace(".parquet", ".csv")] = {
                "rows_total_in_file": total,
                "rows_sampled": int(len(df)),
                "n_attack": int(is_atk.sum()),
                "n_benign": int((~is_atk).sum()),
                "attack_median": _f(np.median(a)) if a.size else None,
                "benign_median": _f(np.median(b)) if b.size else None,
                "attack_pct_ge": _f(100.0 * float((a >= args.thr).mean())) if a.size else None,
                "benign_pct_ge": _f(100.0 * float((b >= args.thr).mean())) if b.size else None,
                "auc": auc_safe(is_atk.astype(np.int8), score),
                "unit_suspects": unit_suspects(file_ben, ref_ben_pctl, top_names),
            }
            log.info("%s: sampled %s of %s rows", path.name, f"{len(df):,}", f"{total:,}")
            del df, X

    new_a, new_b = cat(new_atk), cat(new_ben)
    ref_a, ref_b = ref_score[ref_atk], ref_score[~ref_atk]
    fam_scores = {f: cat(v) for f, v in new_fam.items()}
    cols_new_atk = {c: cat(v) for c, v in kept_new_atk.items()}
    cols_new_ben = {c: cat(v) for c, v in kept_new_ben.items()}
    cols_ref_atk = {c: cat(v) for c, v in kept_ref_atk.items()}
    cols_ref_ben = {c: cat(v) for c, v in kept_ref_ben.items()}
    y_new = np.concatenate([np.ones(new_a.size, np.int8), np.zeros(new_b.size, np.int8)])
    s_new = np.concatenate([new_a, new_b])
    auc_new = auc_safe(y_new, s_new)
    auc_ref = auc_safe(ref_atk.astype(np.int8), ref_score)

    # ---- section 2: score distributions ---------------------------------
    section("2. GATE SCORE DISTRIBUTION  (2018 sample vs the 2017 control)")
    d_new = print_scores(f"CSE-CIC-IDS2018 sample ({new_a.size + new_b.size:,} rows, AUC "
                         f"{'n/a' if auc_new is None else round(auc_new, 4)})",
                         new_a, new_b, args.thr)
    d_ref = print_scores(f"CICIDS2017 control [{ref_where}] ({ref_a.size + ref_b.size:,} rows, AUC "
                         f"{'n/a' if auc_ref is None else round(auc_ref, 4)})",
                         ref_a, ref_b, args.thr)
    print("\n  read this as: if the control sits near 1.0 and the 2018 attack scores sit")
    print("  in a low band, the SCALE moved. If both sit high, the slice was the problem.")

    # ---- section 3: threshold sweep -------------------------------------
    section("3. THRESHOLD SWEEP  (is there ANY operating point that transfers?)")
    sweep = sweep_table(SWEEP, new_a, new_b, ref_a, ref_b, args.thr)

    # ---- section 4: feature scale ---------------------------------------
    section("4. FEATURE SCALE COMPARISON  (fingerprint of a unit mismatch)")
    print("  IQR ratio = 2018 interquartile spread / 2017 spread. Near 1.0 = same units.")
    print("  overlap   = p5-p95 window intersection / union (1.0 = same range, 0 = disjoint).")
    print("  qq ratio  = the single factor that best explains ALL quantiles at once.")
    print("  'UNIT?' needs all three: >5x IQR move, <10% overlap, uniform qq ratio.")
    sc_ben = scale_table(stats_new, stats_ref, cols_new_ben, cols_ref_ben,
                         top_names, "BENIGN rows (cleanest unit test: normal traffic in both labs)")
    sc_atk = scale_table(stats_new, stats_ref, cols_new_atk, cols_ref_atk,
                         top_names, "ATTACK rows (noisier: the two datasets attack differently)")
    flags = sorted({r["feature"] for r in sc_ben if r["flag"] and "UNIT?" in r["flag"]})
    flags_pooled = list(flags)
    atk_flags = sorted({r["feature"] for r in sc_atk if r["flag"] and "UNIT?" in r["flag"]})
    suspect_detail(sc_ben, "BENIGN side")
    print("\n  only the BENIGN table drives the verdict: the two datasets attack differently,")
    print("  so an attack-side shift is expected and is not evidence of a unit bug.")
    if flags:
        print(f"  [!] benign-side unit suspects: {', '.join(flags)}")
    else:
        print("  [ok] no top-gain feature moved >5x with non-overlapping windows on benign traffic.")
    if atk_flags:
        print(f"      (attack-side only, informational: {', '.join(atk_flags)})")

    # ---- section 5: per family / per file -------------------------------
    section("5. WHERE THE DEFLATION LIVES")
    fam_rows = family_table(fam_scores, args.thr)
    file_rows, systematic, specific = file_table(per_file, args.thr)
    # A unit suspect found file-by-file counts just as much as one found in the
    # pooled table - pooling six days can dilute a single-day mapping error into
    # a bimodal blob that no longer looks uniform.
    flags = sorted(set(flags) | set(systematic) | set(specific))

    # ---- section 6: column health ---------------------------------------
    section("6. COLUMN HEALTH OVER THE 2018 SAMPLE  (all 70 features)")
    bad = []
    for i, name in enumerate(FEATS):
        rn, rr = stats_new.row(i), stats_ref.row(i)
        if rn["constant"] or (rn["zero_pct"] or 0) >= 99.0 or rn["non_finite"] > 0:
            bad.append({"feature": name, "in_2018": rn, "in_2017": rr})
    if bad:
        print(f"  [!] {len(bad)} feature(s) look degenerate in the 2018 sample:")
        for b in bad[:20]:
            r = b["in_2018"]
            print(f"      {r['feature']:<28} constant={r['constant']} "
                  f"zero%={r['zero_pct']} non_finite={r['non_finite']:,}")
    else:
        print("  all 70 features vary and are finite in the 2018 sample.")
    print(f"\n  non-finite cells seen while sampling: {nonfinite_cells:,} (expect 0 - the builder drops them)")

    # ---- verdict ---------------------------------------------------------
    ctrl_rec = d_ref.get(f"pct_attack_ge_{args.thr}")
    new_rec = d_new.get(f"pct_attack_ge_{args.thr}")
    cause, actions = verdict(alignment_ok, ctrl_rec, new_rec, auc_new, sweep, flags,
                             args.thr, int(new_a.size), int(new_b.size))

    if not args.no_json:
        payload = {
            "tier": "D-diagnostic",
            "purpose": "separate distribution shift from a feature/unit or alignment bug",
            "gate_model": stage1_path.name,
            "deployed_threshold": args.thr,
            "alignment": {
                "booster_feature_count": len(boost_names),
                "expected_feature_count": len(FEATS),
                "booster_name_kind": name_kind,
                "metadata_match": meta_ok,
                "trustworthy": alignment_ok,
                "importance_basis": gain_kind,
            },
            "inputs": {
                "parquet_dir": str(args.dir),
                "files": [p.name for p in files],
                "per_file_cap": args.per_file,
                "rows_sampled_2018": int(sampled_rows),
                "reference_split": ref_where,
                "rows_reference": int(ref_a.size + ref_b.size),
            },
            "scores_2018": d_new,
            "scores_2017_control": d_ref,
            "threshold_sweep": sweep,
            "feature_scale_benign": sc_ben,
            "feature_scale_attack": sc_atk,
            "scale_suspects_benign_pooled": flags_pooled,
            "scale_suspects_systematic_all_files": systematic,
            "scale_suspects_file_specific": specific,
            "scale_suspects_combined": flags,
            "scale_suspects_attack_side_only": atk_flags,
            "families_2018": fam_rows,
            "per_file_2018": file_rows,
            "degenerate_columns": bad,
            "non_finite_cells_seen": int(nonfinite_cells),
            "diagnosis": cause,
            "recommended_actions": actions,
            "sampling_note": "Evenly spaced row groups per file, never head-only: the 2018 "
                             "CSVs are grouped by victim machine, so a prefix describes one "
                             "machine rather than the day.",
        }
        save_report(payload, REPORT_NAME)
        print(f"\nreport           : {Path(settings.paths.reports) / REPORT_NAME}")

    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())