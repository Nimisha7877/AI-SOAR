"""Train both AI SOAR detector stages and evaluate on the stratified test split.

Pipeline:
  1. load stratified train/val/test (float32, 70 features)
  2. stage 1: binary BENIGN vs MALICIOUS  (early stopping on val)
  3. stage 2: 7-family multiclass on malicious rows only
  4. evaluate on TEST: binary metrics, per-class table, confusion matrices,
     and the CASCATED system metric (gate + classifier together, 8 families)
  5. save models + metadata + JSON reports under artifacts/

Run:
    python scripts/train_models.py                # full run (~3-6 min)
    python scripts/train_models.py --sample 500000  # quick low-RAM smoke run

NOTE ON NUMBERS: stratified-test scores are an UPPER BOUND (near-duplicate
flows). Honest generalization numbers come later from the temporal (Friday)
and cross-dataset (CSE-CIC-IDS2018) evaluations - see PROJECT_BRIEF 5A.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai_soar.config import get_settings
from ai_soar.data.features import FEATURE_COLUMNS, load_split, to_binary_target, X_y
from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.evaluation.metrics import (
    binary_metrics,
    confusion_matrix_plot,
    macro_f1,
    per_class_table,
    print_classification_report,
    save_report,
)
from ai_soar.models import binary as stage1
from ai_soar.models import multiclass as stage2
from ai_soar.utils.logging import configure_from_settings, get_logger

log = get_logger("train_models")


def main(sample: int | None) -> int:
    settings = get_settings()
    configure_from_settings(settings)
    models_dir = Path(settings.paths.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- load -------------------------------------------------------------
    log.info("loading splits (sample=%s)...", sample)
    train_df = load_split("train", max_rows=sample)
    val_df = load_split("val", max_rows=sample)
    test_df = load_split("test", max_rows=sample)
    X_tr, y_tr = X_y(train_df)
    X_va, y_va = X_y(val_df)
    X_te, y_te = X_y(test_df)
    log.info("train=%d val=%d test=%d rows", len(X_tr), len(X_va), len(X_te))

    # ---- stage 1: binary ---------------------------------------------------
    log.info("training stage-1 binary detector...")
    b_tr, b_va, b_te = to_binary_target(y_tr), to_binary_target(y_va), to_binary_target(y_te)
    bin_model = stage1.train_binary(X_tr, b_tr, X_va, b_va)
    bin_report = binary_metrics(b_te, stage1.malicious_probability(bin_model, X_te))
    save_report({"stage": "binary", "test": bin_report}, "binary_test_metrics.json")
    log.info("stage-1 test metrics: %s", bin_report)

    # ---- stage 2: multiclass ----------------------------------------------
    log.info("training stage-2 multiclass classifier...")
    m_tr, m_va, m_te = (
        stage2.filter_malicious(train_df),
        stage2.filter_malicious(val_df),
        stage2.filter_malicious(test_df),
    )
    Xm_tr, ym_tr = X_y(m_tr)
    Xm_va, ym_va = X_y(m_va)
    Xm_te, ym_te = X_y(m_te)
    mc_model = stage2.train_multiclass(Xm_tr, ym_tr, Xm_va, ym_va)

    pred_te = stage2.predict_family(mc_model, Xm_te)
    labels7 = list(stage2.MALICIOUS_FAMILIES)
    print("\n--- stage-2 test classification report ---")
    print_classification_report(ym_te, pred_te, labels7)
    cm_path = confusion_matrix_plot(ym_te, pred_te, labels7, "multiclass_test_cm.png")
    mc_report = {
        "stage": "multiclass",
        "test_macro_f1": round(macro_f1(ym_te, pred_te, labels7), 4),
        "per_class": per_class_table(ym_te, pred_te, labels7),
        "confusion_matrix_png": cm_path.name,
    }
    save_report(mc_report, "multiclass_test_metrics.json")

    # ---- cascaded system metric (what deployment actually does) -----------
    log.info("evaluating cascaded system (gate + classifier)...")
    proba = stage1.malicious_probability(bin_model, X_te)
    is_mal = proba >= 0.5
    system_pred = np.full(len(X_te), BENIGN_LABEL, dtype=object)
    system_pred[is_mal] = stage2.predict_family(mc_model, X_te[is_mal])
    sys_macro = macro_f1(y_te, system_pred, list(FAMILY_ORDER))
    sys_report = {
        "stage": "cascaded_system",
        "test_macro_f1_8families": round(sys_macro, 4),
        "per_class": per_class_table(y_te, system_pred, list(FAMILY_ORDER)),
    }
    save_report(sys_report, "system_test_metrics.json")
    cm8 = confusion_matrix_plot(y_te, system_pred, list(FAMILY_ORDER), "system_test_cm.png")

    # ---- persist models + metadata ----------------------------------------
    stage1.save_model(bin_model, models_dir / "binary_stage1.joblib")
    stage2.save_model(mc_model, models_dir / "multiclass_stage2.joblib")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "rows": {"train": int(len(X_tr)), "val": int(len(X_va)), "test": int(len(X_te))},
        "sample": sample,
        "stage1": {"best_iteration": bin_model.best_iteration_, "test": bin_report},
        "stage2": {
            "best_iteration": mc_model.best_iteration_,
            "classes": list(mc_model.classes_),
            "test_macro_f1": mc_report["test_macro_f1"],
        },
        "system": {"test_macro_f1_8families": round(sys_macro, 4)},
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n================ TRAINING COMPLETE ================")
    print(f"stage-1 binary  : AUC={bin_report['auc']}  F1={bin_report['f1']}  "
          f"FP={bin_report['false_positives']} FN={bin_report['false_negatives']}")
    print(f"stage-2 multi   : macro-F1={mc_report['test_macro_f1']}")
    print(f"cascaded system : macro-F1(8 families)={round(sys_macro, 4)}")
    print(f"models          : {models_dir}")
    print(f"confusion plots : {cm_path.name}, {cm8.name}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None,
                    help="cap rows per split (for low-RAM smoke runs)")
    args = ap.parse_args()
    raise SystemExit(main(args.sample))