# Deployment Guide

This project implements a two-service deployment architecture on Render.

| Runtime | Responsibility |
| --- | --- |
| FastAPI backend | Serves health checks, API docs, and prediction endpoints |
| Streamlit frontend | Collects patient inputs and calls the deployed FastAPI service |

## Live Render URLs

| Service | Production URL |
| --- | --- |
| Streamlit frontend | https://cardiorisk-ml-pipline-1.onrender.com |
| FastAPI backend | https://cardiorisk-ml-pipline.onrender.com/ |
| API health check | https://cardiorisk-ml-pipline.onrender.com/health |
| API docs | https://cardiorisk-ml-pipline.onrender.com/docs |

## Local Production-Style Run

Create the environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify model artifacts:

```powershell
python main.py status
```

Run backend:

```powershell
python main.py api
```

Run frontend:

```powershell
python main.py app
```

## Required Artifact

The deployed services depend on the trained model artifact:

```text
artifacts/model_evaluation/best_model.joblib
```

If it is missing, rebuild the pipeline:

```powershell
python main.py pipeline
```

## Render Deployment

Deploy the backend and frontend as separate Render Web Services from the same GitHub repository.

### FastAPI Backend

| Setting | Value |
| --- | --- |
| Service type | Web Service |
| Name | `cardiorisk-ml-pipline` |
| Language | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |

Environment variable:

| Key | Value |
| --- | --- |
| `API_CORS_ORIGINS` | `https://cardiorisk-ml-pipline-1.onrender.com` |

Validation endpoints:

```text
https://cardiorisk-ml-pipline.onrender.com/health
https://cardiorisk-ml-pipline.onrender.com/docs
```

### Streamlit Frontend

| Setting | Value |
| --- | --- |
| Service type | Web Service |
| Name | `cardiorisk-ml-pipline-1` |
| Language | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true` |

Frontend API target:

```text
https://cardiorisk-ml-pipline.onrender.com
```

## Deployment Verification

Quality assurance checklist:

- [ ] Frontend URL opens the Streamlit dashboard.
- [ ] Backend `/health` endpoint returns a successful response.
- [ ] Backend `/docs` endpoint loads the FastAPI Swagger UI.
- [ ] Streamlit reports that the CardioRisk Prediction API is reachable.
- [ ] Patient form submission returns a `Low Risk` or `High Risk` prediction.
- [ ] Frontend is configured with the deployed backend URL, not a localhost URL.
- [ ] `artifacts/model_evaluation/best_model.joblib` is available in the deployed repository.

## Deployment Notes

- Keep `requirements.txt` aligned with direct runtime dependencies.
- Exclude virtual environments, caches, local `.env` files, and notebook checkpoints from Git.
- Use environment variables for deployment-specific values such as CORS origins and model paths.
- Render free services can sleep after inactivity. The first request after idle time may take longer while the service spins up.
