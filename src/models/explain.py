# src/models/explain.py

"""
src/models/explain.py

SHAP explanations for LightGBM — local (per-prediction) and global (summary plots).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

def _extract_positive_class_shap(shap_values):
    """
    Normalize SHAP output across versions.
    Some SHAP versions return a list [class0, class1] for binary classifiers;
    others return a single 2D array that's already the class-1 contribution.
    Confirmed via diagnostic: this project's SHAP version returns 2D directly.
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def build_explainer(model) -> shap.TreeExplainer:
    """Build once, reuse for the lifetime of the process — parsing the tree is the expensive part."""
    return shap.TreeExplainer(model)


# Explain a single prediction with SHAP values, returning a list of dicts for JSON serialization
def explain_single(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_cols: list[str],
    n_top: int = 10,
) -> list[dict]:
    """
    Top N features driving one prediction, sorted by |SHAP value|.

    Returns dicts (not raw arrays) so this can be returned directly as
    JSON from the future FastAPI /explain endpoint.
    """
    shap_values = _extract_positive_class_shap(explainer.shap_values(X_row))

    shap_row = shap_values[0]
    feat_row = X_row[0]
    top_idx = np.argsort(np.abs(shap_row))[::-1][:n_top]

    return [
        {
            "feature": feature_cols[i],
            "shap_value": round(float(shap_row[i]), 6),
            "direction": "increases_risk" if shap_row[i] > 0 else "decreases_risk",
            "feature_value": None if np.isnan(feat_row[i]) else round(float(feat_row[i]), 4),
        }
        for i in top_idx
    ]

def log_shap_to_mlflow(
    explainer: shap.TreeExplainer,
    X_sample: np.ndarray,
    feature_cols: list[str],
    output_dir: Path = Path("reports"),
) -> None:
    """Generate a beeswarm + bar plot and log both as MLflow artifacts on the active run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = _extract_positive_class_shap(explainer.shap_values(X_sample))

    # Beeswarm — distribution of SHAP values per feature
    explanation = shap.Explanation(values=shap_values, feature_names=feature_cols)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False, color="#1E88E5")
    plt.tight_layout()
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(beeswarm_path), "shap_plots")

    # Bar — simpler mean absolute SHAP, easier to read at a glance
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_cols[i] for i in reversed(top_idx)], [mean_abs[i] for i in reversed(top_idx)])
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    bar_path = output_dir / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(bar_path), "shap_plots")

    logger.info("SHAP plots logged to MLflow")


# Explain a single prediction with SHAP waterfall plot, saving the image to disk for later retrieval by FastAPI
def plot_waterfall(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_cols: list[str],
    output_path: Path,
    max_display: int = 10,
) -> None:
    """
    Waterfall plot for a single prediction — shows how each feature pushes
    the prediction from the baseline (expected value) to the final output.

    Called per-request from the FastAPI /explain endpoint (Phase 4),
    not from train.py — this explains one application, not the whole dataset.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap_values = _extract_positive_class_shap(explainer.shap_values(X_row))
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if hasattr(base_value, "__len__") and len(base_value) > 1 else base_value

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_row[0],
        feature_names=feature_cols,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Waterfall plot saved -> {output_path}")
