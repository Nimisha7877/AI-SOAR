"""Stage-1 binary detector: BENIGN (0) vs MALICIOUS (1).

Purpose in the AI SOAR pipeline: a fast, high-recall gate. Anything the
binary stage calls benign is logged and dropped; anything malicious goes to
the stage-2 multiclass classifier for family identification.

Design notes:
- LightGBM: histogram boosting on tabular flow features; trains in minutes
  on ~2M rows and predicts in microseconds (real-time requirement).
- ``class_weight='balanced'`` instead of SMOTE: we never invent synthetic
  rows, we just make the loss care about the minority class.
- early stopping on the stratified val split prevents overfitting and
  picks the iteration count automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np

from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

BINARY_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 1000,          # capped by early stopping
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 100,
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


def train_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[dict] = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> lgb.LGBMClassifier:
    """Train the stage-1 detector with early stopping on val."""
    model = lgb.LGBMClassifier(**(params or BINARY_PARAMS))
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
        "binary model trained: best_iteration=%s", model.best_iteration_ or "n/a"
    )
    return model


def malicious_probability(model: lgb.LGBMClassifier, X: np.ndarray) -> np.ndarray:
    """P(malicious) for each row - used by the confidence gate later."""
    return model.predict_proba(X)[:, 1]


def save_model(model: lgb.LGBMClassifier, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("binary model saved to %s", path)
    return path


def load_model(path: Path | str) -> lgb.LGBMClassifier:
    return joblib.load(path)