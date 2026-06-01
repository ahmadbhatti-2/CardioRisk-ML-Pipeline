# Deployment Guide

This project has two runtime services:

- FastAPI backend: serves prediction endpoints.
- Streamlit frontend: collects user input and calls the backend.

## Local Production-Style Run

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirm the trained model artifact exists:

```powershell
python main.py status
```

Start the API:

```powershell
python main.py api
```

Start the Streamlit app in a second terminal:

```powershell
python main.py app
```

## Required Artifact

The API and UI need this file:

```text
artifacts/model_evaluation/best_model.joblib
```

If it is missing, rebuild the pipeline:

```powershell
python main.py pipeline
```

## Deployment Notes

- Keep `requirements.txt` updated with every direct dependency.
- Do not deploy virtual environment folders such as `.venv/`.
- Use environment variables for deployment-specific API host, port, or model path.
- For cloud deployment, deploy the FastAPI backend and Streamlit frontend as separate services unless the platform supports running both processes cleanly.
