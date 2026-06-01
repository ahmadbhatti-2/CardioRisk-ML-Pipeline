"""Entry point for the CardioRisk ML pipeline and services."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src and root are in path for sibling imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROJECT_NAME = "CardioRisk"
VERSION = "1.0.0"

ARTIFACTS = {
    "raw data": PROJECT_ROOT / "artifacts" / "data_ingestion" / "raw_dataset.csv",
    "train split": PROJECT_ROOT / "artifacts" / "data_ingestion" / "train.csv",
    "test split": PROJECT_ROOT / "artifacts" / "data_ingestion" / "test.csv",
    "X train": PROJECT_ROOT / "artifacts" / "data_preprocessing" / "X_train.csv",
    "y train": PROJECT_ROOT / "artifacts" / "data_preprocessing" / "y_train.csv",
    "X test": PROJECT_ROOT / "artifacts" / "data_preprocessing" / "X_test.csv",
    "y test": PROJECT_ROOT / "artifacts" / "data_preprocessing" / "y_test.csv",
    "preprocessing summary": PROJECT_ROOT / "artifacts" / "data_preprocessing" / "preprocessing_summary.json",
    "training summary": PROJECT_ROOT / "artifacts" / "model_training" / "training_summary.json",
    "best model": PROJECT_ROOT / "artifacts" / "model_evaluation" / "best_model.joblib",
    "evaluation summary": PROJECT_ROOT / "artifacts" / "model_evaluation" / "evaluation_summary.json",
    "prediction summary": PROJECT_ROOT / "artifacts" / "model_prediction" / "prediction_summary.json",
}

STAGE_REQUIREMENTS = {
    "preprocess": ["train split", "test split"],
    "train": ["X train", "y train"],
    "evaluate": ["X test", "y test", "training summary"],
    "predict": ["best model"],
    "api": ["best model"],
    "app": ["best model"],
}

PACKAGE_REQUIREMENTS = {
    "ingest": ["pandas", "sklearn"],
    "preprocess": ["pandas"],
    "train": ["pandas", "joblib", "sklearn", "xgboost"],
    "evaluate": ["pandas", "joblib", "sklearn"],
    "predict": ["pandas", "joblib"],
    "api": ["fastapi", "uvicorn", "pydantic", "pydantic_settings"],
    "app": ["streamlit"],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(PROJECT_NAME)


def print_header() -> None:
    print("\n" + "=" * 70)
    print(f"{PROJECT_NAME} v{VERSION}")
    print("Cardiovascular disease prediction pipeline and local app launcher")
    print("=" * 70)


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")


def check_packages(group: str) -> bool:
    missing = [
        package
        for package in PACKAGE_REQUIREMENTS.get(group, [])
        if importlib.util.find_spec(package) is None
    ]

    if not missing:
        return True

    LOGGER.error("Missing package(s) for '%s': %s", group, ", ".join(missing))
    LOGGER.info("Run: pip install -r requirements.txt")
    return False


def show_status() -> None:
    groups = {
        "Data ingestion": ["raw data", "train split", "test split"],
        "Preprocessing": ["X train", "y train", "X test", "y test", "preprocessing summary"],
        "Training": ["training summary"],
        "Evaluation": ["best model", "evaluation summary"],
        "Prediction": ["prediction summary"],
    }

    print("\nArtifact status")
    print("-" * 70)
    for group_name, keys in groups.items():
        print(f"\n{group_name}")
        for key in keys:
            path = ARTIFACTS[key]
            status = "OK" if path.exists() else "MISSING"
            print(f"  [{status:7}] {path.relative_to(PROJECT_ROOT)}")
    print()


def require_artifacts(stage: str) -> bool:
    missing = [key for key in STAGE_REQUIREMENTS.get(stage, []) if not ARTIFACTS[key].exists()]
    if not missing:
        return True

    LOGGER.error("Missing required artifact(s) for '%s':", stage)
    for key in missing:
        LOGGER.error("  - %s: %s", key, ARTIFACTS[key].relative_to(PROJECT_ROOT))
    return False


def run_stage(stage: str, runner: Callable[[], object]) -> bool:
    if not check_packages(stage) or not require_artifacts(stage):
        return False

    try:
        LOGGER.info("Running stage: %s", stage)
        runner()
        LOGGER.info("Finished stage: %s", stage)
        return True
    except Exception:
        LOGGER.exception("Stage failed: %s", stage)
        return False


def run_data_ingestion() -> bool:
    def runner():
        from src.data_ingestion import DataIngestion
        return DataIngestion().initiate_data_ingestion()
    return run_stage("ingest", runner)


def run_data_preprocessing() -> bool:
    def runner():
        from src.data_preprocessing import DataPreprocessing
        return DataPreprocessing().run()
    return run_stage("preprocess", runner)


def run_model_training() -> bool:
    def runner():
        from src.model_training import ModelTraining
        return ModelTraining().run()
    return run_stage("train", runner)


def run_model_evaluation() -> bool:
    def runner():
        from src.model_evaluation import ModelEvaluation
        return ModelEvaluation().run()
    return run_stage("evaluate", runner)


def run_sample_prediction() -> bool:
    def runner():
        from src.model_prediction import ModelPrediction
        return ModelPrediction().run()
    return run_stage("predict", runner)


def run_full_pipeline() -> bool:
    LOGGER.info("Starting full ML pipeline.")
    stages = [
        run_data_ingestion,
        run_data_preprocessing,
        run_model_training,
        run_model_evaluation,
        run_sample_prediction,
    ]

    for stage in stages:
        if not stage():
            LOGGER.error("Pipeline interrupted due to failure in stage.")
            return False

    LOGGER.info("Full pipeline completed successfully.")
    show_status()
    return True


def start_api() -> bool:
    if not check_packages("api") or not require_artifacts("api"):
        return False

    LOGGER.info("Starting FastAPI on http://127.0.0.1:8000")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return True


def start_app() -> bool:
    if not check_packages("app") or not require_artifacts("app"):
        return False

    LOGGER.info("Starting Streamlit on http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return True


def start_services() -> bool:
    if not check_packages("api") or not check_packages("app") or not require_artifacts("api"):
        return False

    LOGGER.info("Launching FastAPI in background...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=PROJECT_ROOT,
    )

    try:
        LOGGER.info("Launching Streamlit frontend...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
            cwd=PROJECT_ROOT,
            check=False,
        )
    finally:
        api_process.terminate()
    return True


def show_project_overview() -> None:
    print(
        """
Project structure:
- src/     ML pipeline (ingestion, preprocessing, training, eval, predict)
- api/     FastAPI backend
- app/     Streamlit frontend
- Data/    Source cardiovascular dataset
- artifacts Generated results and model files
        """
    )


COMMANDS: dict[str, Callable[[], bool | None]] = {
    "ingest": run_data_ingestion,
    "preprocess": run_data_preprocessing,
    "train": run_model_training,
    "evaluate": run_model_evaluation,
    "predict": run_sample_prediction,
    "pipeline": run_full_pipeline,
    "api": start_api,
    "app": start_app,
    "services": start_services,
    "status": show_status,
    "overview": show_project_overview,
}

MENU = """
Choose an option:
----------------
1  Data ingestion
2  Data preprocessing
3  Model training
4  Model evaluation
5  Sample prediction
6  Full pipeline
7  FastAPI backend
8  Streamlit frontend
9  Start all services
S  Artifact status
O  Project overview
0  Exit
"""

MENU_ACTIONS = {
    "1": run_data_ingestion, "2": run_data_preprocessing, "3": run_model_training,
    "4": run_model_evaluation, "5": run_sample_prediction, "6": run_full_pipeline,
    "7": start_api, "8": start_app, "9": start_services, "s": show_status, "o": show_project_overview,
}


def run_menu() -> int:
    print_header()
    show_status()

    while True:
        print(MENU)
        choice = input("Selection: ").strip().lower()
        print()

        if choice == "0":
            LOGGER.info("Exiting.")
            return 0

        action = MENU_ACTIONS.get(choice)
        if action is None:
            LOGGER.warning("Invalid option: %s", choice)
        else:
            if action() is False:
                LOGGER.error("Action failed.")

        input("\nPress Enter to continue...")
        print_header()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioRisk pipeline and services manager.")
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS), help="Command to run.")
    return parser.parse_args()


def main() -> int:
    check_python_version()
    args = parse_args()

    if args.command is None:
        return run_menu()

    print_header()
    result = COMMANDS[args.command]()
    return 0 if result is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
