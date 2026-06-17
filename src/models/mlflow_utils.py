# src/models/mlflow_utils.py

"""
src/models/mlflow_utils.py

Thin MLflow helpers so train.py stays focused on ML logic, not tracking plumbing.
"""

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def setup_mlflow(cfg) -> None:
    """Point MLflow at DagsHub (or env var override) and set the experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    logger.info(f"MLflow tracking URI: {tracking_uri}")


def promote_if_better(model_name: str, model_uri: str, new_auc: float, threshold: float) -> bool:
    """
    Register model_uri under model_name. Promote to Production if it beats
    the current Production model's AUC by more than `threshold`.

    First-ever registration is always promoted (nothing to compare against).
    """
    client = MlflowClient()
    version = mlflow.register_model(model_uri, model_name).version

    prod = client.get_latest_versions(model_name, stages=["Production"])
    if not prod:
        client.transition_model_version_stage(model_name, version, "Production")
        logger.info(f"No prior Production model — promoted v{version} (AUC {new_auc:.5f})")
        return True

    prod_auc_tag = client.get_run(prod[0].run_id).data.tags.get("cv_auc")
    prod_auc = float(prod_auc_tag) if prod_auc_tag else 0.0

    if new_auc - prod_auc > threshold:
        client.transition_model_version_stage(model_name, prod[0].version, "Archived")
        client.transition_model_version_stage(model_name, version, "Production")
        logger.info(f"Promoted v{version}: {new_auc:.5f} beats {prod_auc:.5f}")
        return True

    client.transition_model_version_stage(model_name, version, "Staging")
    logger.info(f"v{version} kept in Staging: {new_auc:.5f} vs {prod_auc:.5f} (below threshold)")
    return False