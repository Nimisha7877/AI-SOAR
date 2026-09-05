"""Stage-2 multiclass classifier: which attack family?

Trained ONLY on malicious rows (stage-1 already filtered BENIGN out).
Classes = the 7 malicious families:

    BruteForce, DoS, DDoS, PortScan, WebAttack, Botnet, Infiltration

Design notes:
- ``min_child_samples=10`` because Infiltration has only ~28 training rows;
  the default (20) would make it impossible for any leaf to represent it.
- ``class_weight='balanced'`` so WebAttack/Botnet/Infiltration are not
  drowned by DoS/DDoS/PortScan.
- early stopping on the stratified val split.
- probability output feeds the confidence gate: low-confidence predictions
  are routed to the UNKNOWN/human queue instead of auto-response.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ai_soar.data.labels import BENIGN_LABEL, FAMILY_COLUMN, FAMILY_ORDER
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

MALICIOUS_FAMILIES: tuple[str, ...] = tuple(f for f in FAMILY_ORDER if f != BENIGN_LABEL)

MULTICLASS_PARAMS: dict = {
    "objective": "multiclass",
    "n_estimators": 1000,          # capped by early stopping
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 10,       # rare families must be learnable
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 50


def filter_malicious(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only malicious rows (stage-2 training/eval input)."""
    return df[df[FAMILY_COLUMN] != BENIGN_LABEL].reset_index(drop=True)


def train_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[dict] = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> lgb.LGBMClassifier:
    """Train the stage-2 family classifier with early stopping on val."""
    model = lgb.LGBMClassifier(**(params or MULTICLASS_PARAMS))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    log.info(
        "multiclass model trained: classes=%s best_iteration=%s",
        list(model.classes_),
        model.best_iteration_ or "n/a",
    )
    return model


def predict_family(model: lgb.LGBMClassifier, X: np.ndarray) -> np.ndarray:
    """Predicted family label per row."""
    return model.predict(X)


def predict_proba_families(model: lgb.LGBMClassifier, X: np.ndarray) -> pd.DataFrame:
    """Per-family probabilities, columns = model.classes_ (confidence gate input)."""
    return pd.DataFrame(model.predict_proba(X), columns=list(model.classes_))


def save_model(model: lgb.LGBMClassifier, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("multiclass model saved to %s", path)
    return path


def load_model(path: Path | str) -> lgb.LGBMClassifier:
    return joblib.load(path)