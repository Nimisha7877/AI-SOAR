"""Temporal generalization test: train Mon-Thu, evaluate on Friday.

Answers three honest questions:

1. Does the stage-1 gate hold up on a day the model never saw?
   (binary AUC / F1 / false negatives on Friday)
2. What does the stage-2 classifier do with families that exist on Friday
   but were NEVER in training (DDoS, PortScan, Botnet)? Expect them to be
   forced into known families - this is the cost of a closed label set.
3. Can the confidence gate catch those forced-wrong predictions?
   We compare max-probability of Friday's unseen-family rows against a
   high-confidence threshold. If unseen families still score >0.9, the
   gate alone is NOT enough and the per-family response allowlist
   (PROJECT_BRIEF 8) becomes mandatory.

Run:  python scripts/eval_temporal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_soar.config import get_settings
from ai_soar.data.features import load_split, to_binary_target, X_y
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.evaluation.metrics import (
    binary_metrics,
    confusion_matrix_plot,
    macro_f1,
    save_report,
)
from ai_soar.models import binary as stage1
from ai_soar.models import multiclass as stage2
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("eval_temporal")

TRAIN_FAMILIES_SEEN = ("BENIGN", "BruteForce", "DoS", "WebAttack", "Infiltration")


def _carve_val_time(df: pd.DataFrame, frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `frac` of each family's rows (file order) -> val. Burst-safe."""
    tr, va = [], []
    for fam, grp in df.groupby(FAMILY_COLUMN, sort=False):
        n = len(grp)
        k = int(round(n * frac)) if n >= 10 else 0
        if k == 0:
            tr.append(grp)
            continue
        tr.append(grp.iloc[: n - k])
        va.append(grp.iloc[n - k :])
    return pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)


def main() -> int:
    settings = get_settings()
    configure_from_settings(settings)

    # ---- day split: train = Mon-Thu, test = Friday -------------------------
    day_train = load_split("train", stratified=False)
    friday = load_split("test", stratified=False)
    log.info("day-train=%d friday=%d rows", len(day_train), len(friday))

    tr, va = _carve_val_time(day_train)
    X_tr, y_tr = X_y(tr)
    X_va, y_va = X_y(va)
    X_fr, y_fr = X_y(friday)

    # ---- 1) gate on unseen day --------------------------------------------
    log.info("training stage-1 on Mon-Thu...")
    bin_model = stage1.train_binary(X_tr, to_binary_target(y_tr), X_va, to_binary_target(y_va))
    proba_fr = stage1.malicious_probability(bin_model, X_fr)
    gate = binary_metrics(to_binary_target(y_fr), proba_fr)
    log.info("friday gate metrics: %s", gate)

    # ---- 2) stage-2 on families it never saw -------------------------------
    log.info("training stage-2 on Mon-Thu malicious rows...")
    m_tr, m_va = stage2.filter_malicious(tr), stage2.filter_malicious(va)
    Xm_tr, ym_tr = X_y(m_tr)
    Xm_va, ym_va = X_y(m_va)
    mc_model = stage2.train_multiclass(Xm_tr, ym_tr, Xm_va, ym_va)

    fr_mal = friday[friday[FAMILY_COLUMN] != BENIGN_LABEL].reset_index(drop=True)
    Xm_fr, ym_fr = X_y(fr_mal)
    pred_fr = stage2.predict_family(mc_model, Xm_fr)
    proba_mc = stage2.predict_proba_families(mc_model, Xm_fr)

    unseen = [f for f in stage2.MALICIOUS_FAMILIES if f not in TRAIN_FAMILIES_SEEN]
    per_unseen = {}
    for fam in unseen:
        mask = ym_fr == fam
        if not mask.any():
            continue
        got = pd.Series(pred_fr[mask]).value_counts().to_dict()
        per_unseen[fam] = {
            "rows": int(mask.sum()),
            "predicted_as": {str(k): int(v) for k, v in got.items()},
        }
        conf = proba_mc.max(axis=1)[mask]
        per_unseen[fam]["confidence"] = {
            "mean": round(float(conf.mean()), 4),
            "median": round(float(conf.median()), 4),
            "frac_above_0.9": round(float((conf > 0.9).mean()), 4),
        }
    log.info("unseen-family behaviour: %s", json.dumps(per_unseen))

    # cascaded system on Friday (only families present there are scored)
    is_mal = proba_fr >= 0.5
    sys_pred = np.full(len(X_fr), BENIGN_LABEL, dtype=object)
    sys_pred[is_mal] = stage2.predict_family(mc_model, X_fr[is_mal])
    fams_present = [f for f in FAMILY_ORDER if (y_fr == f).any()]
    sys_macro = macro_f1(y_fr, sys_pred, fams_present)
    cm = confusion_matrix_plot(y_fr, sys_pred, fams_present, "temporal_system_cm.png")

    report = {
        "split": "train=Mon-Thu, test=Friday (unseen day)",
        "gate_on_friday": gate,
        "unseen_family_behaviour": per_unseen,
        "system_macro_f1_present_families": round(sys_macro, 4),
        "families_present_on_friday": fams_present,
        "confusion_matrix_png": cm.name,
    }
    save_report(report, "temporal_eval_report.json")

    print("\n================ TEMPORAL (FRIDAY) EVAL ================")
    print(f"gate  : AUC={gate['auc']} F1={gate['f1']} FP={gate['false_positives']} "
          f"FN={gate['false_negatives']}")
    for fam, info in per_unseen.items():
        print(f"\n{fam} ({info['rows']} rows, never in train) predicted as:")
        for k, v in info["predicted_as"].items():
            print(f"    {k:<12} {v:>8,}")
        c = info["confidence"]
        print(f"    confidence mean={c['mean']} median={c['median']} "
              f"frac>0.9={c['frac_above_0.9']}")
    print(f"\nsystem macro-F1 (families present Friday): {round(sys_macro, 4)}")
    print(f"report: {Path(settings.paths.reports) / 'temporal_eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())