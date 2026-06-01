"""Build candidate models for cardiovascular disease prediction."""

from __future__ import annotations

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


RANDOM_STATE = 42

def build_scaled_model(model: object) -> Pipeline:
    """Pipeline that applies standard scaling before the model."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def build_model_candidates() -> dict[str, object]:
    """Define the set of models to be tuned and compared."""

    return {
        "logistic_regression": build_scaled_model(
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                C=1.0,
            )
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=RANDOM_STATE,
            reg_lambda=1.5,
            n_jobs=1,
            eval_metric="logloss",
        ),
    }


if __name__ == "__main__":
    candidates = build_model_candidates()
    LOGGER.info("Available models:")
    for model_name in candidates:
        LOGGER.info(" - %s", model_name)
