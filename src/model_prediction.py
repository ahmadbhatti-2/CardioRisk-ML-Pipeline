"""
Model prediction pipeline for cardiovascular disease prediction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPredictionConfig:
    """Configuration for the prediction stage."""

    project_root: Path = Path(__file__).resolve().parents[1]

    # Input and Output paths
    evaluation_artifact_dir: Path = project_root / "artifacts" / "model_evaluation"
    best_model_path: Path = evaluation_artifact_dir / "best_model.joblib"
    prediction_artifact_dir: Path = project_root / "artifacts" / "model_prediction"
    summary_path: Path = prediction_artifact_dir / "prediction_summary.json"

    selected_features: tuple[str, ...] = (
        "age_years", "gender", "ap_hi", "ap_lo", "cholesterol",
        "gluc", "active", "bmi", "pulse_pressure",
    )


@dataclass(frozen=True)
class PatientInput:
    """Raw patient input data."""
    age_years: float
    gender: int
    height_cm: float
    weight_kg: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    active: int


@dataclass(frozen=True)
class PredictionResult:
    """Prediction output for a single patient."""
    prediction: int
    risk_label: str
    probability: float | None
    model_path: str
    input_features: dict[str, float | int]


@dataclass(frozen=True)
class ModelPredictionArtifacts:
    summary_path: Path
    prediction_result: PredictionResult


class ModelPrediction:
    """Handles model loading and inference for new patient data."""

    def __init__(self, config: ModelPredictionConfig | None = None) -> None:
        self.config = config or ModelPredictionConfig()
        self.model: object | None = None

    def run(self) -> ModelPredictionArtifacts:
        LOGGER.info("Starting model prediction stage.")

        # Sample patient for pipeline testing
        sample_patient = PatientInput(
            age_years=52, gender=2, height_cm=170, weight_kg=82,
            ap_hi=135, ap_lo=88, cholesterol=2, gluc=1, active=1,
        )

        prediction_result = self.predict_patient(sample_patient)
        artifacts = self._save_prediction_summary(prediction_result)

        LOGGER.info("Model prediction stage completed successfully.")
        return artifacts

    def predict_patient(self, patient: PatientInput) -> PredictionResult:
        """Predict risk for a single patient."""
        self._validate_patient_input(patient)

        input_features = self._build_features(patient)
        input_dataframe = self._to_model_dataframe(input_features)

        model = self._load_best_model()

        prediction = int(model.predict(input_dataframe)[0])
        probability = self._get_positive_probability(model, input_dataframe)
        risk_label = "High Risk" if prediction == 1 else "Low Risk"

        LOGGER.info("Prediction: %s%s", risk_label, f" ({probability:.2f})" if probability else "")

        return PredictionResult(
            prediction=prediction,
            risk_label=risk_label,
            probability=probability,
            model_path=str(self.config.best_model_path),
            input_features=input_features,
        )

    def predict_batch(self, patients: list[PatientInput]) -> list[PredictionResult]:
        return [self.predict_patient(p) for p in patients]

    def _load_best_model(self) -> object:
        if self.model is not None:
            return self.model

        if not self.config.best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at: {self.config.best_model_path}")

        LOGGER.info("Loading best model from %s", self.config.best_model_path)
        self.model = joblib.load(self.config.best_model_path)
        return self.model

    def _build_features(self, patient: PatientInput) -> dict[str, float | int]:
        height_in_meters = patient.height_cm / 100
        bmi = round(patient.weight_kg / (height_in_meters ** 2), 1)
        pulse_pressure = patient.ap_hi - patient.ap_lo

        return {
            "age_years": patient.age_years,
            "gender": patient.gender,
            "ap_hi": patient.ap_hi,
            "ap_lo": patient.ap_lo,
            "cholesterol": patient.cholesterol,
            "gluc": patient.gluc,
            "active": patient.active,
            "bmi": bmi,
            "pulse_pressure": pulse_pressure,
        }

    def _to_model_dataframe(self, input_features: dict[str, float | int]) -> pd.DataFrame:
        missing_features = set(self.config.selected_features).difference(input_features)
        if missing_features:
            raise ValueError(f"Missing prediction features: {', '.join(sorted(missing_features))}")

        return pd.DataFrame([input_features], columns=list(self.config.selected_features))

    @staticmethod
    def _get_positive_probability(model: object, input_dataframe: pd.DataFrame) -> float | None:
        if not hasattr(model, "predict_proba"):
            return None

        probability = float(model.predict_proba(input_dataframe)[0][1])
        return round(probability, 4)

    @staticmethod
    def _validate_patient_input(patient: PatientInput) -> None:
        """Basic input validation."""
        if patient.height_cm <= 0 or patient.weight_kg <= 0:
            raise ValueError("Height and weight must be positive values.")

        if patient.ap_hi <= patient.ap_lo:
            raise ValueError("Systolic BP (ap_hi) must be greater than diastolic BP (ap_lo).")

        if patient.cholesterol not in {1, 2, 3} or patient.gluc not in {1, 2, 3}:
            raise ValueError("Cholesterol and Glucose must be in range [1, 2, 3].")

        if patient.active not in {0, 1}:
            raise ValueError("Active status must be 0 or 1.")

    def _save_prediction_summary(self, prediction_result: PredictionResult) -> ModelPredictionArtifacts:
        self.config.prediction_artifact_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "prediction_type": "single_patient",
            "selected_features": list(self.config.selected_features),
            "result": asdict(prediction_result),
        }

        self.config.summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
        LOGGER.info("Prediction summary saved to %s", self.config.summary_path)

        return ModelPredictionArtifacts(
            summary_path=self.config.summary_path,
            prediction_result=prediction_result,
        )


if __name__ == "__main__":
    prediction = ModelPrediction()
    prediction.run()
