# api/model_loader.py

"""Loads model artifacts once at API startup. MLflow-first, local-pkl fallback."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


@dataclass
class ModelArtifacts:
    model: object
    encoders: dict
    feature_cols: list[str]
    explainer: object | None = None
    model_version: str = "unknown"
    cv_auc: float | None = None


def _load_from_mlflow(model_name: str, stage: str = "Production") -> ModelArtifacts | None:
    """Try MLflow Registry. Returns None (never raises) if unreachable — caller falls back."""
    try:
        import mlflow
        import mlflow.lightgbm

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        model = mlflow.lightgbm.load_model(f"models:/{model_name}/{stage}")

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            return None

        run = client.get_run(versions[0].run_id)
        cv_auc = float(run.data.tags.get("cv_auc", 0)) or None

        logger.info(f"Loaded model from MLflow: {model_name}/{stage} v{versions[0].version}")
        return ModelArtifacts(
            model=model,
            encoders={},  # encoders/explainer aren't logged to MLflow, fall back to local for those
            feature_cols=[],
            model_version=f"v{versions[0].version}",
            cv_auc=cv_auc,
        )
    except Exception as e:
        logger.warning(f"MLflow load failed ({e}) — falling back to local files")
        return None


def _load_from_local() -> ModelArtifacts:
    """Fallback when MLflow is unreachable. Raises if local files are also missing."""
    paths = {
        "model": MODELS_DIR / "lgbm_best.pkl",
        "encoders": MODELS_DIR / "target_encoders.pkl",
        "feature_cols": MODELS_DIR / "feature_cols.pkl",
        "explainer": MODELS_DIR / "shap_explainer.pkl",
    }
    for name, path in paths.items():
        if name != "explainer" and not path.exists():  # explainer is optional
            raise FileNotFoundError(f"Missing artifact: {path}. Run dvc pull or dvc repro train.")

    explainer = joblib.load(paths["explainer"]) if paths["explainer"].exists() else None

    logger.info(f"Loaded model from local files: {MODELS_DIR}/")
    return ModelArtifacts(
        model=joblib.load(paths["model"]),
        encoders=joblib.load(paths["encoders"]),
        feature_cols=joblib.load(paths["feature_cols"]),
        explainer=explainer,
        model_version="local",
    )


def load_artifacts(
    model_name: str = "home-credit-lgbm", stage: str = "Production"
) -> ModelArtifacts:
    """MLflow-first, local-fallback. Always merges local encoders/explainer since
    MLflow registry doesn't store those alongside the model."""
    artifacts = _load_from_mlflow(model_name, stage)
    local = _load_from_local()

    if artifacts is None:
        return local

    # Use the MLflow model, but encoders/feature_cols/explainer always come from local
    artifacts.encoders = local.encoders
    artifacts.feature_cols = local.feature_cols
    artifacts.explainer = local.explainer
    return artifacts
