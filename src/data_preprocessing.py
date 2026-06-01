"""
Data preprocessing for the cardiovascular disease prediction project.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataPreprocessingConfig:
    """Preprocessing configuration and domain constraints."""

    project_root: Path = Path(__file__).resolve().parents[1]

    # Input paths
    ingestion_artifact_dir: Path = project_root / "artifacts" / "data_ingestion"
    train_split_path: Path = ingestion_artifact_dir / "train.csv"
    test_split_path: Path = ingestion_artifact_dir / "test.csv"

    # Output paths
    preprocessing_artifact_dir: Path = project_root / "artifacts" / "data_preprocessing"
    cleaned_train_path: Path = preprocessing_artifact_dir / "train_preprocessed.csv"
    cleaned_test_path: Path = preprocessing_artifact_dir / "test_preprocessed.csv"
    x_train_path: Path = preprocessing_artifact_dir / "X_train.csv"
    y_train_path: Path = preprocessing_artifact_dir / "y_train.csv"
    x_test_path: Path = preprocessing_artifact_dir / "X_test.csv"
    y_test_path: Path = preprocessing_artifact_dir / "y_test.csv"
    summary_path: Path = preprocessing_artifact_dir / "preprocessing_summary.json"

    target_column: str = "cardio"
    id_column: str = "id"

    # Domain constants for medical validation
    days_in_year: float = 365.25
    min_systolic_bp, max_systolic_bp = 90, 250
    min_diastolic_bp, max_diastolic_bp = 30, 150
    min_height_cm, max_height_cm = 120, 220
    min_weight_kg, max_weight_kg = 30, 250
    min_bmi, max_bmi = 12.0, 60.0

    selected_features: tuple[str, ...] = (
        "age_years", "gender", "ap_hi", "ap_lo", "cholesterol",
        "gluc", "active", "bmi", "pulse_pressure",
    )


@dataclass(frozen=True)
class CleaningSummary:
    input_rows: int
    duplicate_rows_removed: int
    missing_values_removed: int
    invalid_medical_rows_removed: int
    output_rows: int


@dataclass(frozen=True)
class DataPreprocessingArtifacts:
    cleaned_train_path: Path
    cleaned_test_path: Path
    x_train_path: Path
    y_train_path: Path
    x_test_path: Path
    y_test_path: Path
    summary_path: Path


class DataPreprocessing:
    """Handles data cleaning, domain validation, and feature engineering."""

    required_columns = {
        "age", "height", "weight", "ap_hi", "ap_lo", 
        "cholesterol", "gluc", "cardio",
    }

    def __init__(self, config: DataPreprocessingConfig | None = None) -> None:
        self.config = config or DataPreprocessingConfig()

    def run(self) -> DataPreprocessingArtifacts:
        LOGGER.info("Starting data preprocessing stage.")

        train_data = self._read_split(self.config.train_split_path, "train")
        test_data = self._read_split(self.config.test_split_path, "test")

        train_clean, train_summary = self._prepare_split(train_data, "train")
        test_clean, test_summary = self._prepare_split(test_data, "test")

        x_train, y_train = self._split_xy(train_clean)
        x_test, y_test = self._split_xy(test_clean)

        artifacts = self._save_artifacts(
            train_clean=train_clean, test_clean=test_clean,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            train_summary=train_summary, test_summary=test_summary,
        )

        LOGGER.info("Data preprocessing completed successfully.")
        return artifacts

    def _read_split(self, path: Path, split_name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{split_name.capitalize()} split not found at: {path}")
        return pd.read_csv(path)

    def _check_required_columns(self, dataframe: pd.DataFrame, split_name: str) -> None:
        missing_columns = self.required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"{split_name.capitalize()} split missing columns: {', '.join(sorted(missing_columns))}")

    def _prepare_split(self, dataframe: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, CleaningSummary]:
        self._check_required_columns(dataframe, split_name)
        
        input_rows = len(dataframe)
        deduplicated = dataframe.drop_duplicates()
        duplicate_rows_removed = input_rows - len(deduplicated)

        missing_values_removed = int(deduplicated.isna().any(axis=1).sum())
        non_missing = deduplicated.dropna().copy()

        valid_rows = self._validate_medical_rows(non_missing)
        clean_data = non_missing.loc[valid_rows].copy()
        invalid_medical_rows_removed = len(non_missing) - len(clean_data)

        if clean_data.empty:
            raise ValueError(f"No rows left in the {split_name} split after cleaning.")

        clean_data = clean_data.drop(columns=[self.config.id_column], errors="ignore")
        clean_data = self._add_derived_features(clean_data)
        clean_data = self._keep_selected_columns(clean_data)

        summary = CleaningSummary(
            input_rows=input_rows,
            duplicate_rows_removed=duplicate_rows_removed,
            missing_values_removed=missing_values_removed,
            invalid_medical_rows_removed=invalid_medical_rows_removed,
            output_rows=len(clean_data),
        )

        LOGGER.info("%s split: %d rows removed, %d remaining.", split_name.capitalize(), input_rows - len(clean_data), len(clean_data))
        return clean_data.reset_index(drop=True), summary

    def _validate_medical_rows(self, dataframe: pd.DataFrame) -> pd.Series:
        """Filter out physiologically implausible records."""
        height_in_meters = dataframe["height"] / 100
        bmi = dataframe["weight"] / (height_in_meters ** 2)

        return (
            dataframe["ap_hi"].between(self.config.min_systolic_bp, self.config.max_systolic_bp) &
            dataframe["ap_lo"].between(self.config.min_diastolic_bp, self.config.max_diastolic_bp) &
            (dataframe["ap_hi"] > dataframe["ap_lo"]) &
            dataframe["height"].between(self.config.min_height_cm, self.config.max_height_cm) &
            dataframe["weight"].between(self.config.min_weight_kg, self.config.max_weight_kg) &
            bmi.between(self.config.min_bmi, self.config.max_bmi)
        )

    def _add_derived_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        prepared = dataframe.copy()
        height_in_meters = prepared["height"] / 100

        prepared["age_years"] = (prepared["age"] / self.config.days_in_year).round(1)
        prepared["bmi"] = (prepared["weight"] / (height_in_meters ** 2)).round(1)
        prepared["pulse_pressure"] = prepared["ap_hi"] - prepared["ap_lo"]

        return prepared.drop(columns="age")

    def _keep_selected_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        final_columns = list(self.config.selected_features) + [self.config.target_column]
        return dataframe.loc[:, final_columns].copy()

    def _split_xy(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x_data = dataframe.drop(columns=self.config.target_column)
        y_data = dataframe[self.config.target_column]
        return x_data, y_data

    def _save_artifacts(
        self, 
        train_clean: pd.DataFrame, test_clean: pd.DataFrame, 
        x_train: pd.DataFrame, y_train: pd.Series, 
        x_test: pd.DataFrame, y_test: pd.Series, 
        train_summary: CleaningSummary, test_summary: CleaningSummary
    ) -> DataPreprocessingArtifacts:
        
        self.config.preprocessing_artifact_dir.mkdir(parents=True, exist_ok=True)

        train_clean.to_csv(self.config.cleaned_train_path, index=False)
        test_clean.to_csv(self.config.cleaned_test_path, index=False)
        x_train.to_csv(self.config.x_train_path, index=False)
        y_train.to_csv(self.config.y_train_path, index=False)
        x_test.to_csv(self.config.x_test_path, index=False)
        y_test.to_csv(self.config.y_test_path, index=False)

        summary = {
            "train": asdict(train_summary),
            "test": asdict(test_summary),
            "derived_features": ["age_years", "bmi", "pulse_pressure"],
            "selected_features": list(self.config.selected_features),
            "features_count": x_train.shape[1],
        }

        self.config.summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
        LOGGER.info("Saved preprocessing artifacts to %s", self.config.preprocessing_artifact_dir)

        return DataPreprocessingArtifacts(
            cleaned_train_path=self.config.cleaned_train_path,
            cleaned_test_path=self.config.cleaned_test_path,
            x_train_path=self.config.x_train_path,
            y_train_path=self.config.y_train_path,
            x_test_path=self.config.x_test_path,
            y_test_path=self.config.y_test_path,
            summary_path=self.config.summary_path,
        )


if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    preprocessing.run()
