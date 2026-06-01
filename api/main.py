"""FastAPI backend for cardiovascular disease risk prediction."""

from __future__ import annotations

import logging
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyHttpUrl, BaseModel, Field, validator
from pydantic_settings import BaseSettings

from src.serving_utils import ServingConfig, ServingInput, ServingUtils


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


class Settings(BaseSettings):
    """API service settings."""

    title: str = "CardioRisk Prediction API"
    description: str = "FastAPI service for cardiovascular disease screening."
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    model_path: Path = Path(__file__).resolve().parents[1] / "artifacts" / "model_evaluation" / "best_model.joblib"
    cors_origins: list[AnyHttpUrl] = [
        AnyHttpUrl("http://localhost:8501"),
        AnyHttpUrl("http://127.0.0.1:8501"),
    ]

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, value: str | list[str] | list[AnyHttpUrl]) -> list[AnyHttpUrl]:
        if isinstance(value, str):
            return [AnyHttpUrl(item.strip()) for item in value.split(",") if item.strip()]
        return value

    class Config:
        env_prefix = "API_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


class PredictionRequest(BaseModel):
    """Patient data for prediction."""

    age_years: float = Field(..., ge=18, le=100)
    gender: int = Field(..., ge=1, le=2) # 1: Female, 2: Male
    height_cm: float = Field(..., ge=120, le=220)
    weight_kg: float = Field(..., ge=30, le=250)
    ap_hi: int = Field(..., ge=90, le=250)
    ap_lo: int = Field(..., ge=30, le=150)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    active: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    probability: float | None
    probability_percent: str | None
    model_features: dict[str, float | int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    service: str


def _build_serving_input(request: PredictionRequest) -> ServingInput:
    return ServingInput(**request.model_dump())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    serving_utils = ServingUtils(
        config=ServingConfig(best_model_path=settings.model_path)
    )

    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.serving_utils = serving_utils

    @app.on_event("startup")
    def startup_event() -> None:
        LOGGER.info("Starting %s on %s:%s", settings.title, settings.host, settings.port)
        try:
            serving_utils.load_model()
            LOGGER.info("Model loaded from %s", settings.model_path)
        except FileNotFoundError:
            LOGGER.warning("Model artifact not found at %s.", settings.model_path)

    @app.get("/", tags=["System"])
    def root() -> dict[str, str]:
        return {"service": settings.title, "status": "running", "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health() -> HealthResponse:
        model_loaded = settings.model_path.is_file()
        return HealthResponse(
            status="healthy" if model_loaded else "model_not_found",
            model_loaded=model_loaded,
            service=settings.title,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Prediction"],
    )
    def predict(request: PredictionRequest, http_request: Request) -> PredictionResponse:
        serving_utils = http_request.app.state.serving_utils

        try:
            patient_input = _build_serving_input(request)
            prediction = serving_utils.predict(patient_input)
            
            LOGGER.info("Prediction: %s (Prob: %s)", prediction.risk_label, prediction.probability_percent)
            return PredictionResponse(**asdict(prediction))

        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
        except FileNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))
        except Exception:
            LOGGER.exception("Unexpected prediction failure.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during prediction.",
            )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
