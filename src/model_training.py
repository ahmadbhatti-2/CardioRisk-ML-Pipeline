"""
Pipeline for training and tuning cardiovascular disease prediction models.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from model_building import build_model_candidates


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Training configuration and directory paths."""

    project_root: Path = Path(__file__).resolve().parents[1]

    # Data inputs
    preprocessing_artifact_dir: Path = project_root / "artifacts" / "data_preprocessing"
    x_train_path: Path = preprocessing_artifact_dir / "X_train.csv"
    y_train_path: Path = preprocessing_artifact_dir / "y_train.csv"

    # Training outputs
    training_artifact_dir: Path = project_root / "artifacts" / "model_training"
    trained_models_dir: Path    = training_artifact_dir / "trained_models"
    summary_path: Path          = training_artifact_dir / "training_summary.json"

    cv_splits: int = 3
    tuning_scoring: str = "f1"


@dataclass(frozen=True)
class TrainedModelInfo:
    model_name: str
    model_path: str
    training_time_seconds: float
    best_cv_score: float
    best_params: dict[str, object]


@dataclass(frozen=True)
class ModelTrainingArtifacts:
    trained_models_dir: Path
    summary_path: Path
    trained_models: list[TrainedModelInfo]


class ModelTraining:
    """Handles model hyperparameter tuning and persistence."""

    def __init__(self, config: ModelTrainingConfig | None = None) -> None:
        self.config = config or ModelTrainingConfig()

    def run(self) -> ModelTrainingArtifacts:
        LOGGER.info("Starting model training stage.")

        x_train, y_train  = self._load_training_data()
        model_candidates   = build_model_candidates()

        trained_models = self._train_and_save_models(
            model_candidates=model_candidates,
            x_train=x_train,
            y_train=y_train,
        )

        artifacts = self._save_training_summary(
            trained_models=trained_models,
            rows_count=len(x_train),
            features_count=x_train.shape[1],
        )

        LOGGER.info("Model training stage completed successfully.")
        return artifacts

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        if not self.config.x_train_path.exists():
            raise FileNotFoundError(f"Training features not found: {self.config.x_train_path}")
        if not self.config.y_train_path.exists():
            raise FileNotFoundError(f"Training target not found: {self.config.y_train_path}")

        LOGGER.info("Loading training data from %s", self.config.preprocessing_artifact_dir)

        x_train = pd.read_csv(self.config.x_train_path)
        y_train = pd.read_csv(self.config.y_train_path).squeeze("columns") 

        LOGGER.info("Loaded %d rows, %d features.", len(x_train), x_train.shape[1])
        return x_train, y_train

    def _train_and_save_models(
        self,
        model_candidates: dict[str, object],
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> list[TrainedModelInfo]:
        
        self.config.trained_models_dir.mkdir(parents=True, exist_ok=True)
        trained_models: list[TrainedModelInfo] = []
        parameter_grids = self._get_parameter_grids()

        cv_strategy = StratifiedKFold(
            n_splits=self.config.cv_splits,
            shuffle=True,
            random_state=42,
        )

        for model_name, model in model_candidates.items():
            LOGGER.info("Tuning '%s'...", model_name)

            start_time = time.perf_counter()

            search = GridSearchCV(
                estimator=model,
                param_grid=parameter_grids[model_name],
                cv=cv_strategy,
                scoring=self.config.tuning_scoring,
                n_jobs=1,
                refit=True,
            )
            search.fit(x_train, y_train)

            training_time = round(time.perf_counter() - start_time, 3)
            best_cv_score = round(float(search.best_score_), 4)
            best_params   = self._clean_parameter_names(search.best_params_)

            LOGGER.info("'%s' best CV %s: %.4f", model_name, self.config.tuning_scoring, best_cv_score)

            model_path = self.config.trained_models_dir / f"{self._make_file_name(model_name)}.joblib"
            joblib.dump(search.best_estimator_, model_path)

            trained_models.append(
                TrainedModelInfo(
                    model_name=model_name,
                    model_path=str(model_path),
                    training_time_seconds=training_time,
                    best_cv_score=best_cv_score,
                    best_params=best_params,
                )
            )

            LOGGER.info("Saved '%s' -> %s (%.3fs)", model_name, model_path, training_time)

        return trained_models

    @staticmethod
    def _get_parameter_grids() -> dict[str, dict[str, list[object]]]:
        # Defined search spaces for tuning
        return {
            "logistic_regression": {
                "model__C": [0.5, 1.0, 2.0],
            },
            "random_forest": {
                "n_estimators":      [100, 200],
                "max_depth":         [10, 12, 14],
                "min_samples_split": [8, 10],
                "min_samples_leaf":  [2, 4],
                "max_features":      ["sqrt"],
            },
            "xgboost": {
                "n_estimators":     [200, 300],
                "learning_rate":    [0.03, 0.05],
                "max_depth":        [3, 4],
                "min_child_weight": [3, 5],
                "subsample":        [0.8],
                "colsample_bytree": [0.8],
                "reg_lambda":       [1.0, 1.5],
            },
        }

    @staticmethod
    def _clean_parameter_names(parameters: dict[str, object]) -> dict[str, object]:
        """Clean pipeline prefix from param names."""
        return {name.replace("model__", ""): value for name, value in parameters.items()}

    def _save_training_summary(
        self,
        trained_models: list[TrainedModelInfo],
        rows_count: int,
        features_count: int,
    ) -> ModelTrainingArtifacts:
        
        self.config.training_artifact_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "rows_count":             rows_count,
            "features_count":         features_count,
            "cross_validation_folds": self.config.cv_splits,
            "tuning_scoring":         self.config.tuning_scoring,
            "models_trained":         len(trained_models),
            "trained_models":         [asdict(m) for m in trained_models],
        }

        self.config.summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
        LOGGER.info("Training summary saved to %s", self.config.summary_path)

        return ModelTrainingArtifacts(
            trained_models_dir=self.config.trained_models_dir,
            summary_path=self.config.summary_path,
            trained_models=trained_models,
        )

    @staticmethod
    def _make_file_name(model_name: str) -> str:
        return model_name.lower().replace(" ", "_")


if __name__ == "__main__":
    ModelTraining().run()
