# tests/test_placeholder.py
# This file exists so pytest has something to collect during Phase 1.
# Replace with real tests in Phase 2 onward.


def test_project_imports() -> None:
    """Verify that core packages installed correctly."""
    import hydra  # noqa: F401
    import lightgbm  # noqa: F401
    import mlflow  # noqa: F401
    import pandas  # noqa: F401
    import pydantic  # noqa: F401
    import shap  # noqa: F401
    import optuna  # noqa: F401
