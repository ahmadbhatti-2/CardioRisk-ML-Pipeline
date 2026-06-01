"""
Shared serving utilities for Streamlit and FastAPI prediction flows.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingConfig:
    """Configuration for model inference and input validation."""

    project_root: Path = Path(__file__).resolve().parents[1]

    # Model path from evaluation stage
    evaluation_artifact_dir: Path = project_root / "artifacts" / "model_evaluation"
    best_model_path: Path = evaluation_artifact_dir / "best_model.joblib"

    selected_features: tuple[str, ...] = (
        "age_years",
        "gender",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "active",
        "bmi",
        "pulse_pressure",
    )

    # Input validation ranges
    min_age_years, max_age_years = 18, 100
    min_height_cm, max_height_cm = 120, 220
    min_weight_kg, max_weight_kg = 30, 250
    min_systolic_bp, max_systolic_bp = 90, 250
    min_diastolic_bp, max_diastolic_bp = 30, 150
    min_bmi, max_bmi = 12.0, 60.0


@dataclass(frozen=True)
class ServingInput:
    """User-provided patient data."""
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
class ModelFeatures:
    """Processed features for model input."""
    age_years: float
    gender: int
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    active: int
    bmi: float
    pulse_pressure: int


@dataclass(frozen=True)
class ServingPrediction:
    """Formatted prediction response."""
    prediction: int
    risk_label: str
    probability: float | None
    probability_percent: str | None
    model_features: dict[str, float | int]


GENDER_MAP = {"Female": 1, "Male": 2}
LEVEL_MAP = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
ACTIVE_MAP = {"No": 0, "Yes": 1}


class ServingUtils:
    """Helper class for real-time model inference."""

    def __init__(self, config: ServingConfig | None = None) -> None:
        self.config = config or ServingConfig()
        self.model: object | None = None

    def predict(self, patient_input: ServingInput) -> ServingPrediction:
        """Full pipeline: validate -> engineer -> predict -> format."""
        self.validate_input(patient_input)

        model_features = self.build_model_features(patient_input)
        model_dataframe = self.to_model_dataframe(model_features)
        model = self.load_model()

        prediction = int(model.predict(model_dataframe)[0])
        probability = self.get_positive_probability(model, model_dataframe)

        return ServingPrediction(
            prediction=prediction,
            risk_label=self.get_risk_label(prediction),
            probability=probability,
            probability_percent=self.format_probability(probability),
            model_features=asdict(model_features),
        )

    def build_model_features(self, patient_input: ServingInput) -> ModelFeatures:
        bmi = calculate_bmi(patient_input.height_cm, patient_input.weight_kg)
        pulse_pressure = calculate_pulse_pressure(patient_input.ap_hi, patient_input.ap_lo)

        return ModelFeatures(
            age_years=patient_input.age_years,
            gender=patient_input.gender,
            ap_hi=patient_input.ap_hi,
            ap_lo=patient_input.ap_lo,
            cholesterol=patient_input.cholesterol,
            gluc=patient_input.gluc,
            active=patient_input.active,
            bmi=bmi,
            pulse_pressure=pulse_pressure,
        )

    def to_model_dataframe(self, model_features: ModelFeatures) -> pd.DataFrame:
        feature_dict = asdict(model_features)
        
        missing_features = set(self.config.selected_features).difference(feature_dict)
        if missing_features:
            raise ValueError(f"Missing model features: {', '.join(sorted(missing_features))}")

        return pd.DataFrame(
            [feature_dict],
            columns=list(self.config.selected_features),
        )

    def load_model(self) -> object:
        if self.model is not None:
            return self.model

        if not self.config.best_model_path.exists():
            raise FileNotFoundError(f"Best model not found: {self.config.best_model_path}")

        LOGGER.info("Loading model from %s", self.config.best_model_path)
        self.model = joblib.load(self.config.best_model_path)
        return self.model

    def validate_input(self, patient_input: ServingInput) -> None:
        """Basic medical range validation for user input."""
        if not self.config.min_age_years <= patient_input.age_years <= self.config.max_age_years:
            raise ValueError(f"Age must be between {self.config.min_age_years} and {self.config.max_age_years}.")

        if patient_input.gender not in {1, 2}:
            raise ValueError("Gender must be 1 or 2.")

        if not self.config.min_height_cm <= patient_input.height_cm <= self.config.max_height_cm:
            raise ValueError(f"Height must be between {self.config.min_height_cm} and {self.config.max_height_cm}.")

        if not self.config.min_weight_kg <= patient_input.weight_kg <= self.config.max_weight_kg:
            raise ValueError(f"Weight must be between {self.config.min_weight_kg} and {self.config.max_weight_kg}.")

        if not self.config.min_systolic_bp <= patient_input.ap_hi <= self.config.max_systolic_bp:
            raise ValueError(f"Systolic BP must be between {self.config.min_systolic_bp} and {self.config.max_systolic_bp}.")

        if not self.config.min_diastolic_bp <= patient_input.ap_lo <= self.config.max_diastolic_bp:
            raise ValueError(f"Diastolic BP must be between {self.config.min_diastolic_bp} and {self.config.max_diastolic_bp}.")

        if patient_input.ap_hi <= patient_input.ap_lo:
            raise ValueError("Systolic BP must be greater than diastolic BP.")

        if patient_input.cholesterol not in {1, 2, 3}:
            raise ValueError("Cholesterol must be 1, 2, or 3.")

        if patient_input.gluc not in {1, 2, 3}:
            raise ValueError("Glucose must be 1, 2, or 3.")

        if patient_input.active not in {0, 1}:
            raise ValueError("Active status must be 0 or 1.")

        bmi = calculate_bmi(patient_input.height_cm, patient_input.weight_kg)
        if not self.config.min_bmi <= bmi <= self.config.max_bmi:
            raise ValueError(f"Calculated BMI must be between {self.config.min_bmi} and {self.config.max_bmi}.")

    @staticmethod
    def get_positive_probability(model: object, model_dataframe: pd.DataFrame) -> float | None:
        if not hasattr(model, "predict_proba"):
            return None

        probability = float(model.predict_proba(model_dataframe)[0][1])
        return round(probability, 4)

    @staticmethod
    def get_risk_label(prediction: int) -> str:
        return "High Risk" if prediction == 1 else "Low Risk"

    @staticmethod
    def format_probability(probability: float | None) -> str | None:
        return f"{probability * 100:.2f}%" if probability is not None else None


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    if height_cm <= 0:
        raise ValueError("Height must be greater than 0.")
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def calculate_pulse_pressure(ap_hi: int, ap_lo: int) -> int:
    return ap_hi - ap_lo


def map_form_value(label: str, mapping: dict[str, int]) -> int:
    if label not in mapping:
        raise ValueError(f"Invalid label '{label}'. Expected one of: {', '.join(mapping)}")
    return mapping[label]


def build_serving_input_from_form(
    age_years: float,
    gender_label: str,
    height_cm: float,
    weight_kg: float,
    ap_hi: int,
    ap_lo: int,
    cholesterol_label: str,
    glucose_label: str,
    active_label: str,
) -> ServingInput:
    return ServingInput(
        age_years=age_years,
        gender=map_form_value(gender_label, GENDER_MAP),
        height_cm=height_cm,
        weight_kg=weight_kg,
        ap_hi=ap_hi,
        ap_lo=ap_lo,
        cholesterol=map_form_value(cholesterol_label, LEVEL_MAP),
        gluc=map_form_value(glucose_label, LEVEL_MAP),
        active=map_form_value(active_label, ACTIVE_MAP),
    )


def prediction_to_dict(prediction: ServingPrediction) -> dict[str, Any]:
    return asdict(prediction)


if __name__ == "__main__":
    utils = ServingUtils()
    sample_input = ServingInput(
        age_years=52, gender=2, height_cm=170, weight_kg=82,
        ap_hi=135, ap_lo=88, cholesterol=2, gluc=1, active=1,
    )
    result = utils.predict(sample_input)
    LOGGER.info("Sample prediction: %s", prediction_to_dict(result))
