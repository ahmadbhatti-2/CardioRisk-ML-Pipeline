# CardioRisk: Cardiovascular Disease Risk Prediction

CardioRisk is an end-to-end machine learning project for cardiovascular disease risk screening. It includes a reproducible ML pipeline, saved model artifacts, a FastAPI prediction service, and a Streamlit user interface.

This project is structured like a real-world ML application: data processing, training, evaluation, serving, and UI code are separated into clear modules instead of being kept in one notebook.

## Project Goals

- Build a complete ML workflow from raw dataset to deployable prediction service.
- Keep pipeline stages modular and independently runnable.
- Save artifacts at every stage for reproducibility and debugging.
- Provide both API and UI layers for real-time prediction.
- Make the project easy to explain in interviews and portfolio reviews.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Data and ML | pandas, scikit-learn, XGBoost, joblib |
| Model evaluation | accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Project control | Python CLI through `main.py` |

## Live Deployment

The project is deployed on Render with separate services for the API and UI.

| Service | URL |
| --- | --- |
| Streamlit frontend | https://cardiorisk-ml-pipline-1.onrender.com |
| FastAPI backend | https://cardiorisk-ml-pipline.onrender.com/ |
| API health check | https://cardiorisk-ml-pipline.onrender.com/health |
| API docs | https://cardiorisk-ml-pipline.onrender.com/docs |

Open the Streamlit frontend, confirm the FastAPI URL is set to:

```text
https://cardiorisk-ml-pipline.onrender.com
```

Then submit the patient form to receive a live prediction from the deployed API.

## Architecture

```text
Raw CSV dataset
    |
    v
Data ingestion
    - read source CSV
    - validate required columns
    - create train/test split
    |
    v
Data preprocessing
    - remove invalid medical records
    - derive age_years, bmi, pulse_pressure
    - select model features
    |
    v
Model training
    - train Logistic Regression
    - train Random Forest
    - train XGBoost
    - tune with cross-validation
    |
    v
Model evaluation
    - compare models on test data
    - select best model by F1-score
    - save best_model.joblib
    |
    v
Serving
    - FastAPI backend exposes /predict
    - Streamlit frontend calls the API
```

## Project Structure

```text
api/
  main.py                 FastAPI backend and prediction endpoint

app/
  streamlit_app.py        Streamlit frontend for patient input and result display

src/
  data_ingestion.py       Loads raw data and creates train/test splits
  data_preprocessing.py   Cleans data and creates model-ready features
  model_building.py       Defines model candidates
  model_training.py       Trains and tunes models
  model_evaluation.py     Evaluates models and saves the best model
  model_prediction.py     Runs sample/single-patient prediction
  serving_utils.py        Shared inference logic for API and UI

artifacts/
  data_ingestion/         Raw, train, and test CSV files
  data_preprocessing/     Cleaned features, targets, and preprocessing summary
  model_training/         Trained model files and training summary
  model_evaluation/       Evaluation metrics and best_model.joblib
  model_prediction/       Prediction summary

Data/
  cardio_train.csv        Source dataset

Notebook/
  cardiovascular.ipynb    Exploration and experimentation notebook

main.py                   CLI controller for pipeline and services
requirements.txt          Python dependencies
README.md                 Project documentation
```

## Setup

Run these commands from the project root.

```powershell
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Training uses XGBoost. If your environment does not already have it, install it:

```powershell
pip install xgboost
```

## Main Commands

The recommended way to run the project is through `main.py`.

```powershell
python main.py --help
python main.py status
python main.py overview
```

Run the full pipeline:

```powershell
python main.py pipeline
```

Run individual stages:

```powershell
python main.py ingest
python main.py preprocess
python main.py train
python main.py evaluate
python main.py predict
```

Start the backend and frontend:

```powershell
python main.py api
python main.py app
```

You can also start both services from one command:

```powershell
python main.py services
```

## API Usage

Start the FastAPI backend:

```powershell
python main.py api
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Live API docs:

```text
https://cardiorisk-ml-pipline.onrender.com/docs
```

Health endpoint:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Example request body:

```json
{
  "age_years": 52,
  "gender": 2,
  "height_cm": 170,
  "weight_kg": 82,
  "ap_hi": 135,
  "ap_lo": 88,
  "cholesterol": 2,
  "gluc": 1,
  "active": 1
}
```

Example PowerShell request:

```powershell
$body = @{
  age_years = 52
  gender = 2
  height_cm = 170
  weight_kg = 82
  ap_hi = 135
  ap_lo = 88
  cholesterol = 2
  gluc = 1
  active = 1
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

For the deployed API, use:

```powershell
Invoke-RestMethod `
  -Uri "https://cardiorisk-ml-pipline.onrender.com/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Streamlit App

Start the Streamlit frontend:

```powershell
python main.py app
```

Open:

```text
http://localhost:8501
```

Live Streamlit app:

```text
https://cardiorisk-ml-pipline-1.onrender.com
```

The Streamlit app collects patient input, sends it to the FastAPI backend, and displays the returned prediction, probability, BMI, and pulse pressure.

## Features Used by the Model

The final model uses 9 features:

```text
age_years
gender
ap_hi
ap_lo
cholesterol
gluc
active
bmi
pulse_pressure
```

Derived features:

- `age_years`: converted from age in days
- `bmi`: calculated from height and weight
- `pulse_pressure`: systolic BP minus diastolic BP

## Data Cleaning

Preprocessing removes medically invalid records using realistic constraints for:

- systolic and diastolic blood pressure
- height
- weight
- BMI
- cases where systolic BP is not greater than diastolic BP

Current preprocessing summary:

| Split | Input rows | Invalid rows removed | Output rows |
| --- | ---: | ---: | ---: |
| Train | 56,000 | 1,188 | 54,812 |
| Test | 14,000 | 294 | 13,706 |

## Model Results

The project trains three models and selects the best model using F1-score.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7254 | 0.7497 | 0.6684 | 0.7067 | 0.7873 |
| Random Forest | 0.7283 | 0.7476 | 0.6811 | 0.7128 | 0.7955 |
| XGBoost | 0.7322 | 0.7486 | 0.6911 | 0.7187 | 0.7977 |

These scores are moderate rather than high; XGBoost reaches an F1-score of about 0.71 and ROC-AUC of about 0.79, so the model should be treated as a learning/portfolio risk-screening demo rather than a production clinical diagnosis system.

Best model:

```text
XGBoost
Selection metric: F1-score
Saved artifact: artifacts/model_evaluation/best_model.joblib
```

## Configuration Design

The project uses module-level configuration classes. Each pipeline stage keeps its own focused config dataclass close to the code that uses it:

- `DataIngestionConfig` controls source data, train/test split, and ingestion artifact paths.
- `DataPreprocessingConfig` controls cleaning rules, selected features, and preprocessing artifact paths.
- `ModelTrainingConfig` controls training inputs, model output paths, cross-validation, and scoring.
- `ModelEvaluationConfig` controls test data paths, trained model loading, metrics output, and best model storage.
- `ModelPredictionConfig` and `ServingConfig` control inference features and model loading.

This keeps every module independently runnable and easy to test, while avoiding a large global configuration file that every part of the project depends on.

## Why `main.py` Exists

`main.py` is the project controller. It does not contain the ML logic itself. Instead, it gives one clean entry point to:

- run pipeline stages
- check artifact status
- run sample prediction
- start the FastAPI backend
- start the Streamlit frontend

This makes the project easier to use, demo, and explain.

## Notes for Recruiters

- The project is organized as an end-to-end ML application, not only a notebook.
- Pipeline code is separated from serving code.
- API and UI share inference utilities to avoid duplicate prediction logic.
- Model artifacts and metric summaries are saved for reproducibility.
- The selected model is based on measured test performance, not a hardcoded choice.

## Medical Disclaimer

This project is for learning, portfolio, and demonstration purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.

## Future Improvements

- Add unit tests for preprocessing and serving utilities.
- Add Docker support for API deployment.
- Add CI checks for linting and tests.
- Add model versioning and experiment tracking.
- Improve dependency pinning for reproducible environments.
