"""
Data ingestion pipeline for the cardiovascular disease dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration and paths for data ingestion."""

    project_root: Path = Path(__file__).resolve().parents[1]

    # Input source
    source_path: Path = project_root / "Data" / "cardio_train.csv"

    # Output paths
    artifacts_dir: Path = project_root / "artifacts" / "data_ingestion"
    raw_data_path: Path = artifacts_dir / "raw_dataset.csv"
    train_data_path: Path = artifacts_dir / "train.csv"
    test_data_path: Path = artifacts_dir / "test.csv"

    target_column: str = "cardio"
    source_separator: str = ";"  # Source file uses semicolons
    test_size: float = 0.20
    random_state: int = 42


@dataclass(frozen=True)
class DataIngestionArtifacts:
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path


class DataIngestion:
    """Handles loading, validation, and splitting of the dataset."""

    required_columns = {
        "id", "age", "gender", "height", "weight", 
        "ap_hi", "ap_lo", "cholesterol", "gluc", 
        "smoke", "alco", "active", "cardio",
    }

    def __init__(self, config: DataIngestionConfig | None = None) -> None:
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """Run the full ingestion workflow."""
        LOGGER.info("Starting data ingestion pipeline.")

        dataframe = self._read_dataset()
        self._validate_dataset(dataframe)

        train_data, test_data = self._split_dataset(dataframe)

        artifacts = self._save_dataset_files(
            dataframe,
            train_data,
            test_data,
        )

        LOGGER.info("Data ingestion pipeline completed successfully.")
        return artifacts

    def _read_dataset(self) -> pd.DataFrame:
        if not self.config.source_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.config.source_path}")

        LOGGER.info("Reading dataset from %s", self.config.source_path)

        return pd.read_csv(
            self.config.source_path,
            sep=self.config.source_separator,
        )

    def _validate_dataset(self, dataframe: pd.DataFrame) -> None:
        """Basic checks for dataset integrity."""
        if dataframe.empty:
            raise ValueError("Loaded dataset is empty.")

        missing_columns = self.required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

        unique_classes = dataframe[self.config.target_column].nunique()
        if unique_classes < 2:
            raise ValueError(f"Target column '{self.config.target_column}' must have at least 2 classes.")

    def _split_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        LOGGER.info("Splitting dataset into train and test sets.")

        train_data, test_data = train_test_split(
            dataframe,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=dataframe[self.config.target_column],
        )

        return (
            train_data.reset_index(drop=True),
            test_data.reset_index(drop=True),
        )

    def _save_dataset_files(
        self,
        raw_data: pd.DataFrame,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> DataIngestionArtifacts:
        
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        raw_data.to_csv(self.config.raw_data_path, index=False)
        train_data.to_csv(self.config.train_data_path, index=False)
        test_data.to_csv(self.config.test_data_path, index=False)

        LOGGER.info("Saved raw, train, and test datasets to %s", self.config.artifacts_dir)

        return DataIngestionArtifacts(
            raw_data_path=self.config.raw_data_path,
            train_data_path=self.config.train_data_path,
            test_data_path=self.config.test_data_path,
        )


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
