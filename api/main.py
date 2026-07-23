# api/main.py

"""FastAPI app. Run: uvicorn api.main:app --reload --port 8000"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from api.model_loader import ModelArtifacts, load_artifacts
from api.predictor import predict, predict_with_explanation
from api.schemas import (
    ExplainResponse,
    HealthResponse,
    LoanApplicationRequest,
    ModelInfoResponse,
    PredictionResponse,
)
from src.models.explain import plot_waterfall

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_artifacts: ModelArtifacts | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts
    Path("models").mkdir(exist_ok=True)
    try:
        _artifacts = load_artifacts(
            model_name=os.getenv("MODEL_NAME", "home-credit-lgbm"),
            stage=os.getenv("MODEL_STAGE", "Production"),
        )
        logger.info(
            f"Model ready: {_artifacts.model_version}, {len(_artifacts.feature_cols)} features"
        )
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        _artifacts = None
    yield


app = FastAPI(title="Home Credit Default Risk API", version="1.0.0", lifespan=lifespan)


def get_artifacts() -> ModelArtifacts:
    if _artifacts is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return _artifacts


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _artifacts is None:
        return HealthResponse(status="degraded", model_loaded=False, model_version="none")
    return HealthResponse(status="ok", model_loaded=True, model_version=_artifacts.model_version)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    arts = get_artifacts()
    return ModelInfoResponse(
        model_name="home-credit-lgbm",
        model_version=arts.model_version,
        cv_auc=arts.cv_auc,
        n_features=len(arts.feature_cols),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_default(request: LoanApplicationRequest) -> PredictionResponse:
    arts = get_artifacts()
    try:
        return predict(request, arts)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain_prediction(request: LoanApplicationRequest, n_top: int = 10) -> ExplainResponse:
    arts = get_artifacts()
    if arts.explainer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SHAP explainer not available")
    try:
        return predict_with_explanation(request, arts, n_top=min(n_top, 50))
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@app.post("/explain/plot")
def explain_plot(request: LoanApplicationRequest, n_top: int = 10) -> FileResponse:
    arts = get_artifacts()
    if arts.explainer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SHAP explainer not available")
    try:
        from api.predictor import _preprocess

        X = _preprocess(request, arts)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plot_waterfall(
            arts.explainer,
            X.values,
            arts.feature_cols,
            output_path=Path(tmp.name),
            max_display=min(n_top, 20),
        )
        return FileResponse(tmp.name, media_type="image/png", filename="shap_waterfall.png")
    except Exception as e:
        logger.error(f"Plot failed: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
