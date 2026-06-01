"""Evaluate trained models for cardiovascular disease prediction."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Paths for test data and evaluation results."""

    project_root: Path = Path(__file__).resolve().parents[1]

    # Input paths
    preprocessing_artifact_dir: Path = project_root / "artifacts" / "data_preprocessing"
    x_test_path: Path = preprocessing_artifact_dir / "X_test.csv"
    y_test_path: Path = preprocessing_artifact_dir / "y_test.csv"

    training_artifact_dir: Path = project_root / "artifacts" / "model_training"
    trained_models_dir: Path = training_artifact_dir / "trained_models"

    # Output paths
    evaluation_artifact_dir: Path = project_root / "artifacts" / "model_evaluation"
    summary_path: Path = evaluation_artifact_dir / "evaluation_summary.json"
    best_model_path: Path = evaluation_artifact_dir / "best_model.joblib"


@dataclass(frozen=True)
class ModelMetrics:
    model_name: str
    model_path: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: dict[str, int]


@dataclass(frozen=True)
class ModelEvaluationArtifacts:
    summary_path: Path
    best_model_path: Path
    best_model_name: str
    model_metrics: list[ModelMetrics]


class ModelEvaluation:
    """Evaluates trained models and selects the best performer based on F1-score."""

    def __init__(self, config: ModelEvaluationConfig | None = None) -> None:
        self.config = config or ModelEvaluationConfig()

    def run(self) -> ModelEvaluationArtifacts:
        LOGGER.info("Starting model evaluation stage.")

        x_test, y_test = self._load_test_data()
        trained_models = self._load_trained_models()

        model_metrics = self._evaluate_models(
            trained_models=trained_models,
            x_test=x_test,
            y_test=y_test,
        )

        best_model_name, best_model_path = self._select_best_model(
            model_metrics=model_metrics,
        )

        artifacts = self._save_evaluation_summary(
            model_metrics=model_metrics,
            best_model_name=best_model_name,
            best_model_path=best_model_path,
            rows_count=len(x_test),
            features_count=x_test.shape[1],
        )

        LOGGER.info("Model evaluation stage completed successfully.")
        return artifacts

    def _load_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        if not self.config.x_test_path.exists():
            raise FileNotFoundError(f"Test features not found: {self.config.x_test_path}")

        if not self.config.y_test_path.exists():
            raise FileNotFoundError(f"Test target not found: {self.config.y_test_path}")

        LOGGER.info("Loading test data from %s", self.config.preprocessing_artifact_dir)

        x_test = pd.read_csv(self.config.x_test_path)
        y_test = pd.read_csv(self.config.y_test_path).squeeze("columns")

        LOGGER.info("Test data loaded: %d rows, %d features.", len(x_test), x_test.shape[1])
        return x_test, y_test

    def _load_trained_models(self) -> dict[str, object]:
        if not self.config.trained_models_dir.exists():
            raise FileNotFoundError(f"Trained models directory not found: {self.config.trained_models_dir}")

        model_paths = sorted(self.config.trained_models_dir.glob("*.joblib"))

        if not model_paths:
            raise FileNotFoundError(f"No trained model files found in: {self.config.trained_models_dir}")

        trained_models = {
            model_path.stem: joblib.load(model_path)
            for model_path in model_paths
        }

        LOGGER.info("Loaded %d trained models.", len(trained_models))
        return trained_models

    def _evaluate_models(
        self,
        trained_models: dict[str, object],
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> list[ModelMetrics]:
        model_metrics: list[ModelMetrics] = []

        for model_name, model in trained_models.items():
            LOGGER.info("Evaluating '%s'...", model_name)

            y_pred = model.predict(x_test)
            y_score = self._get_prediction_scores(model, x_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            metrics = ModelMetrics(
                model_name=model_name,
                model_path=str(self.config.trained_models_dir / f"{model_name}.joblib"),
                accuracy=round(accuracy_score(y_test, y_pred), 4),
                precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
                recall=round(recall_score(y_test, y_pred, zero_division=0), 4),
                f1_score=round(f1_score(y_test, y_pred, zero_division=0), 4),
                roc_auc=round(roc_auc_score(y_test, y_score), 4),
                confusion_matrix={
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                },
            )

            model_metrics.append(metrics)
            LOGGER.info("'%s' ROC-AUC: %.4f | F1: %.4f", model_name, metrics.roc_auc, metrics.f1_score)

        return model_metrics

    @staticmethod
    def _get_prediction_scores(model: object, x_test: pd.DataFrame) -> pd.Series | list[float]:
        """Extract probability scores for ROC-AUC calculation."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_test)[:, 1]

        if hasattr(model, "decision_function"):
            return model.decision_function(x_test)

        return model.predict(x_test)

    def _select_best_model(self, model_metrics: list[ModelMetrics]) -> tuple[str, Path]:
        best_model = max(model_metrics, key=lambda m: m.f1_score)
        best_model_source_path = Path(best_model.model_path)

        LOGGER.info("Best model: '%s' (F1: %.4f)", best_model.model_name, best_model.f1_score)
        return best_model.model_name, best_model_source_path

    def _save_evaluation_summary(
        self,
        model_metrics: list[ModelMetrics],
        best_model_name: str,
        best_model_path: Path,
        rows_count: int,
        features_count: int,
    ) -> ModelEvaluationArtifacts:
        
        self.config.evaluation_artifact_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(best_model_path, self.config.best_model_path)

        summary = {
            "rows_count": rows_count,
            "features_count": features_count,
            "selection_metric": "f1_score",
            "best_model_name": best_model_name,
            "best_model_path": str(self.config.best_model_path),
            "models_evaluated": len(model_metrics),
            "model_metrics": [asdict(m) for m in model_metrics],
        }

        self.config.summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
        LOGGER.info("Evaluation summary saved to %s", self.config.summary_path)

        return ModelEvaluationArtifacts(
            summary_path=self.config.summary_path,
            best_model_path=self.config.best_model_path,
            best_model_name=best_model_name,
            model_metrics=model_metrics,
        )


if __name__ == "__main__":
    evaluation = ModelEvaluation()
    evaluation.run()
