"""Tier D - cross-dataset evaluation of the DEPLOYED models on CSE-CIC-IDS2018.

Why this script exists
----------------------
Every other number in this project is measured on CICIDS2017, the dataset the
models were trained on. Tier A (stratified) is an upper bound, tier B
(burst-hardened) and tier C (temporal) reduce leakage but still draw train and
test from the same week of the same lab network.

Tier D is different: the models see flows from **another year, another network
(AWS instead of the UNB lab), another CICFlowMeter version and another label
vocabulary**. Nothing from 2018 was used for training, validation, threshold
selection or feature choice. This is the honest generalisation number, and it is
the one that produces a realistic false-positive rate - measured here on
**9.6M benign flows**, not on a stratified 10% slice.

What is evaluated
-----------------
The deployed cascade, exactly as ``ai_soar.inference.predictor.Predictor`` runs
it in production (same thresholds, same allowlist, imported from that module so
the two cannot drift apart):

1. **stage-1 gate** - ``P(malicious) >= gate_threshold`` (default 0.50).
   Below it the flow is BENIGN and only logged.
2. **stage-2 multiclass** - argmax over the attack families.
3. **confidence floor** - below ``confidence_floor`` (default 0.60) the verdict
   becomes ``UNKNOWN`` and the flow goes to the human queue instead of
   triggering a playbook.
4. **policy** - a confident verdict in ``AUTO_RESPONSE_FAMILIES`` auto-responds;
   anything else needs human approval.

Memory safety
-------------
10.7M rows x 70 float32 features is ~3 GB, so nothing is loaded at once. Each
parquet is streamed in batches; only fixed-size accumulators (confusion matrix,
counters, histograms) plus two 1-D arrays for the AUC are kept.

Run:
    python scripts/eval_cross_dataset.py --file Wednesday-21 --limit-rows 300000
    python scripts/eval_cross_dataset.py                  # full tier-D run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.metrics import roc_auc_score

from ai_soar.config import get_settings
from ai_soar.data.features import FEATURE_COLUMNS
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.evaluation.metrics import save_report
from ai_soar.inference.predictor import (
    AUTO_RESPONSE_FAMILIES,
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_GATE_THRESHOLD,
    METADATA_FILENAME,
    STAGE1_FILENAME,
    STAGE2_FILENAME,
)
from ai_soar.models.binary import load_model as load_binary_model
from ai_soar.models.binary import malicious_probability
from ai_soar.models.multiclass import load_model as load_multiclass_model
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("eval_cross_dataset")

IN_SUBDIR = "_processed"
SOURCE_COLUMN = "SourceFile"
BATCH_SIZE = 200_000
UNKNOWN = "UNKNOWN"
DECISIONS: tuple[str, ...] = ("log_only", "auto_response", "human_approval", "unknown_queue")
PRED_LABELS: list[str] = list(FAMILY_ORDER) + [UNKNOWN]
LOW_SUPPORT = 1_000          # below this, recall is reported with a wide CI + a flag
REPORT_NAME = "tier_d_cross_dataset.json"
CONFUSION_PNG = "tier_d_confusion_2018.png"
CONF_BINS = np.linspace(0.0, 1.0, 21)

LIMITATIONS = [
    "5 of the 6 source CSVs contain exactly 1,048,575 rows (2**20-1 = Excel's row "
    "limit), so those days are capped subsets of the original capture.",
    "The 2018 CSVs are grouped by victim machine/pcap, NOT ordered by time (verified: "
    "the middle row's timestamp precedes the first row's in all six files). The cap "
    "therefore drops whole machines rather than a slice of the day, and no temporal "
    "analysis is possible on this dataset.",
    "PortScan, Botnet and Infiltration do not occur in the six downloaded files. They "
    "are reported as 'not covered' and excluded from macro-F1 - never scored as zero.",
    "CSE-CIC-IDS2018 has no PortScan label at all: the nmap port scan is part of the "
    "Infiltration scenario, so PortScan coverage would require the 01-03/02-03 files.",
    "WebAttack support is very small (n=314), so its recall carries a wide interval and "
    "is flagged LOW SUPPORT.",
    "74,595 rows (0.57%) were dropped for Inf/NaN features, mirroring the 2017 cleaning "
    "rule; their per-family distribution was not recorded.",
    "Deduplication is window-local (each chunk vs the previous one), so runs of identical "
    "flows longer than 2 x chunksize can survive. Per-family duplicate rates are reported "
    "by scripts/build_dataset2018.py.",
    "No 2018 data was used for training, validation, threshold selection or feature "
    "choice - which is what makes this tier an honest generalisation estimate.",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - behaves at p near 0 and at small n (WebAttack)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
    return (round(max(0.0, (centre - margin) / denom), 6),
            round(min(1.0, (centre + margin) / denom), 6))


def find_parquets(base: Path, only: Optional[str]) -> list[Path]:
    hits = sorted(base.glob("*.parquet"))
    if only:
        needle = only.lower()
        hits = [p for p in hits if needle in p.name.lower()]
    return hits


def preflight(files: list[Path]) -> dict[str, Any]:
    """Metadata-only validation before spending minutes on inference."""
    required = list(FEATURE_COLUMNS) + [FAMILY_COLUMN, SOURCE_COLUMN]
    info: dict[str, Any] = {}
    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    total = 0
    for p in files:
        pf = pq.ParquetFile(p)
        names = list(pf.schema.names)
        missing = [c for c in required if c not in names]
        rows = int(pf.metadata.num_rows)
        total += rows
        info[p.name] = {"rows": rows, "row_groups": int(pf.metadata.num_row_groups),
                        "columns": len(names), "missing": missing}
        flag = "  [xx] MISSING " + str(missing) if missing else ""
        print(f"  {p.name[:52]:<52} rows={rows:>10,} groups={pf.metadata.num_row_groups:>3}"
              f" cols={len(names)}{flag}")
        if missing:
            raise SystemExit(f"{p.name}: missing {missing} - re-run scripts/build_dataset2018.py")
    print(f"  {'TOTAL':<52} rows={total:>10,}")
    return {"files": info, "total_rows": total}


def load_models(models_dir: Path) -> tuple[Any, Any, list[str], dict]:
    stage1_path = models_dir / STAGE1_FILENAME
    stage2_path = models_dir / STAGE2_FILENAME
    for p in (stage1_path, stage2_path):
        if not p.exists():
            raise SystemExit(f"model not found: {p} - run scripts/train_models.py first")
    stage1 = load_binary_model(stage1_path)
    stage2 = load_multiclass_model(stage2_path)
    classes = [str(c) for c in stage2.classes_]

    meta_path = models_dir / METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta_feats = meta.get("feature_columns")
    if meta_feats and list(meta_feats) != list(FEATURE_COLUMNS):
        raise SystemExit(
            "model metadata feature_columns != FEATURE_COLUMNS - the deployed models were "
            "trained on a different feature space; refusing to produce misleading numbers."
        )
    print(f"\nmodels           : {stage1_path.name} + {stage2_path.name}")
    print(f"stage-2 classes  : {classes}")
    print(f"auto-response    : {sorted(AUTO_RESPONSE_FAMILIES)}")
    return stage1, stage2, classes, meta


def save_confusion_png(cm: np.ndarray, labels: list[str], filename: str) -> Optional[Path]:
    """Row-normalised heatmap of the accumulated matrix (all-zero lines dropped)."""
    keep_row = cm.sum(axis=1) > 0
    keep_col = cm.sum(axis=0) > 0
    if not keep_row.any():
        return None
    sub = cm[np.ix_(keep_row, keep_col)]
    sub_labels = [l for l, k in zip(labels, keep_row) if k]
    col_labels = [l for l, k in zip(labels, keep_col) if k]
    norm = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)), max(6, len(sub_labels) - 1)))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=col_labels, yticklabels=sub_labels, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_xlabel("predicted (cascade output)")
    ax.set_ylabel("true family (CSE-CIC-IDS2018)")
    ax.set_title("Tier D - CICIDS2017 models on CSE-CIC-IDS2018 (row-normalised)")
    fig.tight_layout()
    out = Path(get_settings().paths.reports) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def run_eval(
    files: list[Path],
    stage1: Any,
    stage2: Any,
    classes: list[str],
    gate_threshold: float,
    confidence_floor: float,
    batch_size: int,
    limit_rows: Optional[int],
) -> dict[str, Any]:
    auto = np.array([lab in AUTO_RESPONSE_FAMILIES for lab in classes])

    cm = np.zeros((len(PRED_LABELS), len(PRED_LABELS)), dtype=np.int64)
    per_family: dict[str, dict[str, int]] = {
        fam: {"support": 0, "gate_fired": 0, "stage2_correct": 0,
              "cascade_correct": 0, "unknown_routed": 0}
        for fam in FAMILY_ORDER
    }
    per_day: dict[str, dict[str, int]] = {}
    decisions: dict[str, int] = {}
    decision_truth: dict[str, int] = {}
    conf_hist = np.zeros(len(CONF_BINS) - 1, dtype=np.int64)

    gate_bin = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    proba_true: list[np.ndarray] = []
    proba_pred: list[np.ndarray] = []
    nonfinite_cells = 0
    rows_seen = 0
    started = time.time()

    for path in files:
        pf = pq.ParquetFile(path)
        cols = list(FEATURE_COLUMNS) + [FAMILY_COLUMN, SOURCE_COLUMN]
        file_rows = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()
            if limit_rows is not None and file_rows + len(df) > limit_rows:
                df = df.iloc[: limit_rows - file_rows]
            if df.empty:
                break
            file_rows += len(df)
            rows_seen += len(df)

            X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
            nonfinite_cells += int((~np.isfinite(X)).sum())
            y = df[FAMILY_COLUMN].astype(str).to_numpy()
            day = df[SOURCE_COLUMN].astype(str).to_numpy()
            y_bin = (y != BENIGN_LABEL)

            # ---- stage 1: the gate -------------------------------------
            gate_p = malicious_probability(stage1, X).astype(np.float32)
            mal = gate_p >= gate_threshold
            proba_true.append(y_bin.astype(np.uint8))
            proba_pred.append(gate_p)

            gate_bin["tp"] += int((mal & y_bin).sum())
            gate_bin["fp"] += int((mal & ~y_bin).sum())
            gate_bin["tn"] += int((~mal & ~y_bin).sum())
            gate_bin["fn"] += int((~mal & y_bin).sum())

            # ---- stage 2 + confidence floor + policy --------------------
            final = np.full(len(df), BENIGN_LABEL, dtype=object)
            conf = np.zeros(len(df), dtype=np.float32)
            decision = np.full(len(df), "log_only", dtype=object)
            idx = np.flatnonzero(mal)
            stage2_correct = np.zeros(len(df), dtype=bool)
            if idx.size:
                probs = stage2.predict_proba(X[idx])
                j = probs.argmax(axis=1)
                c = probs.max(axis=1).astype(np.float32)
                fam = np.asarray(classes, dtype=object)[j]
                unk = c < confidence_floor
                final[idx] = np.where(unk, UNKNOWN, fam).astype(object)
                conf[idx] = c
                stage2_correct[idx] = (fam == y[idx])
                decision[idx] = np.where(
                    unk, "unknown_queue",
                    np.where(auto[j], "auto_response", "human_approval"),
                ).astype(object)

            # ---- accumulate --------------------------------------------
            t_i = pd.Categorical(y, categories=PRED_LABELS).codes
            p_i = pd.Categorical(final.astype(str), categories=PRED_LABELS).codes
            ok = (t_i >= 0) & (p_i >= 0)
            np.add.at(cm, (t_i[ok], p_i[ok]), 1)

            cascade_correct = mal & stage2_correct & (conf >= confidence_floor)
            for fam in FAMILY_ORDER:
                m = y == fam
                if not m.any():
                    continue
                slot = per_family[fam]
                slot["support"] += int(m.sum())
                slot["gate_fired"] += int((m & mal).sum())
                slot["stage2_correct"] += int((m & mal & stage2_correct).sum())
                slot["cascade_correct"] += int((m & cascade_correct).sum())
                slot["unknown_routed"] += int((m & mal & (conf < confidence_floor)).sum())

            for d in pd.unique(day):
                m = day == d
                slot = per_day.setdefault(
                    d, {"rows": 0, "benign": 0, "benign_flagged": 0, "attack": 0, "attack_missed": 0}
                )
                benign_m = m & ~y_bin
                attack_m = m & y_bin
                slot["rows"] += int(m.sum())
                slot["benign"] += int(benign_m.sum())
                slot["benign_flagged"] += int((benign_m & mal).sum())
                slot["attack"] += int(attack_m.sum())
                slot["attack_missed"] += int((attack_m & ~mal).sum())

            # vectorised masks (4 per batch) instead of per-row string building
            for name in DECISIONS:
                m = decision == name
                n_dec = int(m.sum())
                if not n_dec:
                    continue
                decisions[name] = decisions.get(name, 0) + n_dec
                n_atk = int((m & y_bin).sum())
                decision_truth[f"{name}|attack"] = decision_truth.get(f"{name}|attack", 0) + n_atk
                decision_truth[f"{name}|benign"] = (
                    decision_truth.get(f"{name}|benign", 0) + n_dec - n_atk
                )

            attack_gate = mal & y_bin
            if attack_gate.any():
                conf_hist += np.histogram(conf[attack_gate], bins=CONF_BINS)[0]

            if rows_seen % 1_000_000 < batch_size:
                log.info("  %s rows evaluated (%s)", f"{rows_seen:,}", path.name)

        log.info("%s: %s rows", path.name, f"{file_rows:,}")
        if limit_rows is not None and file_rows >= limit_rows:
            log.info("%s: --limit-rows reached", path.name)

    y_true = np.concatenate(proba_true) if proba_true else np.zeros(0, dtype=np.uint8)
    y_prob = np.concatenate(proba_pred) if proba_pred else np.zeros(0, dtype=np.float32)
    # AUC is undefined when only one class is present (e.g. a --file/--limit-rows
    # smoke run that happens to contain nothing but benign flows). None, never NaN:
    # json.dumps would write the invalid literal `NaN`.
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None

    return {
        "rows_evaluated": rows_seen,
        "seconds": round(time.time() - started, 1),
        "nonfinite_cells": nonfinite_cells,
        "gate": {**gate_bin, "threshold": gate_threshold,
                 "auc": round(auc, 6) if auc is not None else None},
        "confidence_floor": confidence_floor,
        "confusion": cm,
        "per_family": per_family,
        "per_day": per_day,
        "decisions": decisions,
        "decision_truth": decision_truth,
        "conf_hist": conf_hist,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _prf(cm: np.ndarray, i: int) -> dict[str, Any]:
    """Precision / recall / F1 / support for one label from the cascade matrix."""
    tp = int(cm[i, i])
    support = int(cm[i].sum())
    predicted = int(cm[:, i].sum())
    prec = tp / predicted if predicted else 0.0
    rec = tp / support if support else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"true_positives": tp, "support": support, "predicted": predicted,
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6)}


def summarise(res: dict[str, Any]) -> dict[str, Any]:
    gate = res["gate"]
    cm = res["confusion"]
    idx = {lab: i for i, lab in enumerate(PRED_LABELS)}

    tp, fp, tn, fn = gate["tp"], gate["fp"], gate["tn"], gate["fn"]
    benign_n = fp + tn
    attack_n = tp + fn
    fpr = fp / benign_n if benign_n else 0.0

    benign = _prf(cm, idx[BENIGN_LABEL])
    specificity = benign["recall"]          # BENIGN "recall" IS the true-negative rate

    families: dict[str, Any] = {}
    covered: list[str] = []
    for fam in FAMILY_ORDER[1:]:            # attack families only
        m = _prf(cm, idx[fam])
        slot = res["per_family"][fam]
        n = m["support"]
        if n == 0:
            families[fam] = {"status": "not covered", "support": 0,
                             "predicted": m["predicted"]}
            continue
        covered.append(fam)
        lo, hi = wilson_ci(m["true_positives"], n)
        families[fam] = {
            "status": "covered",
            "support": n,
            "low_support": n < LOW_SUPPORT,
            "precision": m["precision"],
            "recall": m["recall"],
            "recall_ci95": [lo, hi],
            "f1": m["f1"],
            "predicted_as": m["predicted"],
            # where the loss happens: gate, stage-2, or the confidence floor
            "gate_recall": round(slot["gate_fired"] / n, 6),
            "stage2_recall_given_gate": round(
                slot["stage2_correct"] / slot["gate_fired"], 6) if slot["gate_fired"] else 0.0,
            "unknown_routed": slot["unknown_routed"],
            "unknown_routed_pct": round(slot["unknown_routed"] / n, 6),
        }

    def macro(key: str) -> Optional[float]:
        return round(float(np.mean([families[f][key] for f in covered])), 6) if covered else None

    dt = res["decision_truth"]
    false_auto = dt.get("auto_response|benign", 0)
    return {
        "gate": {
            "threshold": gate["threshold"],
            "auc": gate["auc"],
            "detection_recall": round(tp / attack_n, 6) if attack_n else 0.0,
            "detection_recall_ci95": list(wilson_ci(tp, attack_n)),
            "false_positives": fp,
            "benign_flows": benign_n,
            "false_positive_rate": round(fpr, 8),
            "false_positive_rate_ci95": list(wilson_ci(fp, benign_n)),
            "specificity": round(specificity, 8),
            "specificity_ci95": list(wilson_ci(benign["true_positives"], benign_n)),
            "false_negatives": fn,
            "attack_flows": attack_n,
        },
        "operational_impact": {
            "benign_rows_auto_responded": false_auto,
            "benign_auto_response_rate": round(false_auto / benign_n, 8) if benign_n else 0.0,
            "benign_rows_human_approval": dt.get("human_approval|benign", 0),
            "benign_rows_unknown_queue": dt.get("unknown_queue|benign", 0),
            "attack_rows_auto_responded": dt.get("auto_response|attack", 0),
            "attack_rows_human_approval": dt.get("human_approval|attack", 0),
            "attack_rows_unknown_queue": dt.get("unknown_queue|attack", 0),
            "attack_rows_missed_by_gate": fn,
        },
        "families": families,
        "families_covered": covered,
        "families_not_covered": [f for f in FAMILY_ORDER[1:] if f not in covered],
        "macro_over_covered_attack_families": {
            "precision": macro("precision"), "recall": macro("recall"), "f1": macro("f1"),
            "n_families": len(covered), "n_families_total": len(FAMILY_ORDER) - 1,
        },
        "predicted_labels": PRED_LABELS,
    }


def print_summary(res: dict[str, Any], summary: dict[str, Any]) -> None:
    g = summary["gate"]
    oi = summary["operational_impact"]
    print()
    print("=" * 78)
    print("TIER D - CROSS-DATASET (CICIDS2017 models -> CSE-CIC-IDS2018)")
    print("=" * 78)
    print(f"rows evaluated   : {res['rows_evaluated']:,}   ({res['seconds']}s)")
    print(f"non-finite cells : {res['nonfinite_cells']}  (expect 0)")

    print()
    print("--- stage-1 gate: is this flow malicious at all? ---")
    auc_txt = f"{g['auc']:.4f}" if g["auc"] is not None else "n/a (single class in this selection)"
    print(f"  AUC                     : {auc_txt}")
    print(f"  detection recall        : {g['detection_recall']:.4%}  "
          f"CI95 [{g['detection_recall_ci95'][0]:.4%}, {g['detection_recall_ci95'][1]:.4%}]"
          f"   ({g['false_negatives']:,} missed of {g['attack_flows']:,} attack flows)")
    print(f"  FALSE POSITIVE RATE     : {g['false_positive_rate']:.4%}  "
          f"CI95 [{g['false_positive_rate_ci95'][0]:.4%}, {g['false_positive_rate_ci95'][1]:.4%}]"
          f"   ({g['false_positives']:,} of {g['benign_flows']:,} benign flows)")
    print(f"  specificity (benign)    : {g['specificity']:.4%}")

    print()
    print("--- operational impact: what the SOAR would actually DO ---")
    print(f"  benign -> auto response : {oi['benign_rows_auto_responded']:,}"
          f"   ({oi['benign_auto_response_rate']:.4%} of benign)   <-- the number that matters")
    print(f"  benign -> human queue   : {oi['benign_rows_human_approval']:,} approval"
          f" + {oi['benign_rows_unknown_queue']:,} unknown")
    print(f"  attack -> auto response : {oi['attack_rows_auto_responded']:,}")
    print(f"  attack -> human queue   : {oi['attack_rows_human_approval']:,} approval"
          f" + {oi['attack_rows_unknown_queue']:,} unknown")
    print(f"  attack -> missed at gate: {oi['attack_rows_missed_by_gate']:,}")

    print()
    print("--- cascade per attack family (gate -> multiclass -> confidence floor) ---")
    print(f"  {'family':<14}{'support':>10}{'recall':>9}{'CI95':>20}{'prec':>8}{'F1':>8}"
          f"{'gate':>8}{'st2|gate':>10}{'UNKWN':>8}")
    for fam in FAMILY_ORDER[1:]:
        f = summary["families"][fam]
        if f["status"] != "covered":
            extra = f"  (predicted anyway: {f['predicted']:,})" if f.get("predicted") else ""
            print(f"  {fam:<14}{0:>10,}{'-':>9}{'NOT COVERED':>20}{extra}")
            continue
        lo, hi = f["recall_ci95"]
        tag = " LOW-SUPPORT" if f["low_support"] else ""
        print(f"  {fam:<14}{f['support']:>10,}{f['recall']:>9.2%}"
              f"{('[' + format(lo, '.2%') + ', ' + format(hi, '.2%') + ']'):>20}"
              f"{f['precision']:>8.2%}{f['f1']:>8.2%}{f['gate_recall']:>8.2%}"
              f"{f['stage2_recall_given_gate']:>10.2%}{f['unknown_routed_pct']:>8.2%}{tag}")
    mac = summary["macro_over_covered_attack_families"]
    if mac["n_families"] == 0:
        print("\n  MACRO: n/a - no attack family had any support in this selection")
    else:
        print(f"\n  MACRO over {mac['n_families']} covered attack families"
              f" (of {mac['n_families_total']}):  P={mac['precision']:.4f}"
              f"  R={mac['recall']:.4f}  F1={mac['f1']:.4f}")
    if summary["families_not_covered"]:
        print(f"  not covered by the downloaded 2018 subset: {summary['families_not_covered']}")
        print("  -> excluded from the macro average, NOT scored as zero.")

    print()
    print("--- routing decisions ---")
    total = sum(res["decisions"].values()) or 1
    for name in DECISIONS:
        n = res["decisions"].get(name, 0)
        atk = res["decision_truth"].get(f"{name}|attack", 0)
        ben = res["decision_truth"].get(f"{name}|benign", 0)
        print(f"  {name:<16}{n:>12,}  ({n / total:6.2%})   attack={atk:>10,}  benign={ben:>10,}")

    print()
    print("--- per day (source file) ---")
    print(f"  {'source file':<50}{'rows':>10}{'benign':>10}{'FP rate':>10}{'attack':>9}{'missed':>9}")
    for day, sl in sorted(res["per_day"].items()):
        fpr = sl["benign_flagged"] / sl["benign"] if sl["benign"] else 0.0
        print(f"  {day[:50]:<50}{sl['rows']:>10,}{sl['benign']:>10,}{fpr:>10.3%}"
              f"{sl['attack']:>9,}{sl['attack_missed']:>9,}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier-D cross-dataset evaluation on CSE-CIC-IDS2018.")
    parser.add_argument("--dir", default=None, help="parquet dir (default data/external/_processed)")
    parser.add_argument("--file", default=None, help="substring filter, e.g. Wednesday-21")
    parser.add_argument("--limit-rows", type=int, default=None, help="max rows per file (smoke test)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="rows per inference batch")
    parser.add_argument("--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    parser.add_argument("--confidence-floor", type=float, default=DEFAULT_CONFIDENCE_FLOOR)
    parser.add_argument("--no-plot", action="store_true", help="skip the confusion-matrix PNG")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_from_settings(settings)

    in_dir = Path(args.dir) if args.dir else Path(settings.paths.external) / IN_SUBDIR
    files = find_parquets(in_dir, args.file)
    if not files:
        print(f"[xx] no parquet under {in_dir} - run scripts/build_dataset2018.py first.")
        return 1

    pre = preflight(files)
    models_dir = Path(settings.paths.models)
    stage1, stage2, classes, meta = load_models(models_dir)
    print(f"gate threshold   : {args.gate_threshold}   confidence floor: {args.confidence_floor}")
    print(f"batch size       : {args.batch_size:,}   limit-rows: {args.limit_rows}")

    res = run_eval(files, stage1, stage2, classes, args.gate_threshold,
                   args.confidence_floor, args.batch_size, args.limit_rows)
    summary = summarise(res)
    print_summary(res, summary)

    png = None
    if not args.no_plot:
        png = save_confusion_png(res["confusion"], PRED_LABELS, CONFUSION_PNG)
        if png:
            print(f"\nconfusion matrix : {png}")

    payload = {
        "tier": "D - cross-dataset",
        "train_dataset": "CICIDS2017",
        "eval_dataset": "CSE-CIC-IDS2018",
        "script": "scripts/eval_cross_dataset.py",
        "models_dir": str(models_dir),
        "model_metadata": meta,
        "stage2_classes": classes,
        "auto_response_families": sorted(AUTO_RESPONSE_FAMILIES),
        "policy": {
            "gate_threshold": args.gate_threshold,
            "confidence_floor": args.confidence_floor,
            "cascade": "gate -> multiclass argmax -> confidence floor -> UNKNOWN/human queue",
        },
        "inputs": pre,
        "rows_evaluated": res["rows_evaluated"],
        "seconds": res["seconds"],
        "nonfinite_cells": res["nonfinite_cells"],
        "results": summary,
        "routing_decisions": res["decisions"],
        "routing_decisions_by_truth": res["decision_truth"],
        "per_day": res["per_day"],
        "confidence_histogram_attack_rows": {
            "bin_edges": [round(float(b), 3) for b in CONF_BINS],
            "counts": [int(c) for c in res["conf_hist"]],
        },
        "confusion_matrix": {
            "labels": PRED_LABELS,
            "rows_true": [[int(v) for v in row] for row in res["confusion"]],
        },
        "confusion_png": str(png) if png else None,
        "limitations": LIMITATIONS,
        "partial_run": bool(args.limit_rows or args.file),
    }
    if args.limit_rows or args.file:
        payload["note"] = (
            "PARTIAL RUN (--file/--limit-rows): numbers describe a subset only and must "
            "not be quoted as tier-D results. Re-run without filters for the real numbers."
        )
        print("\n[!!] PARTIAL RUN - these numbers are a smoke test, not tier-D results.")

    report_path = save_report(payload, REPORT_NAME)
    print(f"report           : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())