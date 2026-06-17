# tests/test_mlflow_utils.py

"""Minimal smoke test for promote_if_better — catches crashes in the promotion logic."""

import mlflow
import mlflow.sklearn
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def local_mlflow(tmp_path):
    # MLflow's plain file:// backend is deprecated for new setups —
    # use a local SQLite database instead, the recommended lightweight
    # backend for testing and local development.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")
    yield


def test_first_registration_is_promoted(local_mlflow):
    from src.models.mlflow_utils import promote_if_better

    # Log a real (trivial) model so there's an artifact at "model"
    # for register_model to find — promote_if_better expects a
    # model to already be logged in the run, exactly like train.py does.
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        dummy_model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(dummy_model, artifact_path="model")

    promoted = promote_if_better(
        model_name="test-model-first",
        model_uri=f"runs:/{run_id}/model",
        new_auc=0.78,
        threshold=0.001,
    )
    assert promoted is True
