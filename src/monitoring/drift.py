# src/monitoring/drift.py

"""
Evidently-based data drift monitoring.

Compares a reference dataset (training features) against current data
(simulated or real incoming requests) and generates:
  - An HTML drift report saved to reports/drift_report.html
  - Drift metrics logged to MLflow under a 'monitoring' experiment

Usage:
    python src/monitoring/drift.py                  # local dry run
    uv run python src/monitoring/drift.py           # via uv
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH = Path("data/processed/features_train.parquet")
REPORTS_DIR = Path("reports")
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_METRICS_PATH = REPORTS_DIR / "drift_metrics.json"

# Features to monitor — exclude ID and target columns
EXCLUDE_COLS = {"SK_ID_CURR", "TARGET"}

# Number of rows to sample for drift detection
SAMPLE_SIZE = 1000


def load_reference_data() -> pd.DataFrame:
    """Load training features as the reference distribution."""
    logger.info(f"Loading reference data from {REFERENCE_DATA_PATH}")
    df = pd.read_parquet(REFERENCE_DATA_PATH)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    df = df[feature_cols].sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    logger.info(f"Reference data shape: {df.shape}")
    return df


def simulate_current_data(reference: pd.DataFrame, drift_intensity: float = 0.15) -> pd.DataFrame:
    """
    Simulate current/production data by adding noise to reference data.

    In a real deployment, this would be replaced by actual logged
    inference requests from a feature store or request log database.

    drift_intensity: fraction of features that will have injected drift
    """
    logger.info("Simulating current data with injected drift...")
    current = reference.copy()

    numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()

    # Inject drift into a random subset of numeric features
    rng = np.random.default_rng(seed=0)
    n_drift_cols = max(1, int(len(numeric_cols) * drift_intensity))
    drift_cols = rng.choice(numeric_cols, size=n_drift_cols, replace=False)

    for col in drift_cols:
        col_std = current[col].std()
        if col_std > 0:
            # Shift the mean by 1.5 standard deviations to trigger drift
            current[col] = current[col] + 1.5 * col_std
            logger.info(f"Injected drift into: {col}")

    logger.info(f"Current data shape: {current.shape}")
    return current


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[dict, object]:
    """
    Run Evidently drift detection using Evidently 0.7.x API.
    """
    from evidently.core.report import Report
    from evidently.metrics import DatasetMissingValueCount, DriftedColumnsCount
    from evidently.presets.drift import DataDriftPreset

    logger.info("Running Evidently drift detection...")

    report = Report(
        [
            DataDriftPreset(),
            DriftedColumnsCount(),
            DatasetMissingValueCount(),
        ]
    )

    snapshot = report.run(reference_data=reference, current_data=current)

    # Extract metrics from snapshot.metric_results
    drift_result = None
    missing_result = None

    for _key, result in snapshot.metric_results.items():
        # Match by display_name which is stable
        display = getattr(result, "display_name", "")
        if display == "Count of Drifted Columns":
            drift_result = result
        if "Missing" in display and hasattr(result, "current"):
            missing_result = result

    n_drifted = int(drift_result.count.value) if drift_result else 0
    share_drifted = float(drift_result.share.value) if drift_result else 0.0
    n_total = float(reference.shape[1])

    missing_current = 0.0
    missing_reference = 0.0
    if missing_result:
        try:
            missing_current = float(missing_result.current.share.value)
            missing_reference = float(missing_result.reference.share.value)
        except Exception:
            pass

    metrics = {
        "share_drifted_features": round(share_drifted, 4),
        "n_drifted_features": int(n_drifted),
        "n_total_features": int(n_total),
        "dataset_drift_detected": int(share_drifted > 0.5),
        "share_missing_values_current": round(missing_current, 4),
        "share_missing_values_reference": round(missing_reference, 4),
    }

    logger.info(f"Drift detected: {bool(metrics['dataset_drift_detected'])}")
    logger.info(f"Drifted features: {metrics['n_drifted_features']}/{metrics['n_total_features']}")

    return metrics, snapshot


def generate_html_report(snapshot, output_path: Path = DRIFT_REPORT_PATH) -> None:
    """Save the Evidently snapshot as an interactive HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    logger.info(f"Drift report saved -> {output_path}")


def save_metrics_json(metrics: dict, output_path: Path = DRIFT_METRICS_PATH) -> None:
    """Save drift metrics as JSON for CI artifact upload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Drift metrics saved -> {output_path}")


def log_drift_to_mlflow(metrics: dict, report_path: Path = DRIFT_REPORT_PATH) -> None:
    try:
        import mlflow

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "https://dagshub.com/ahmaad.alii.213/home-credit-default-risk.mlflow",
        )
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("monitoring")

        with mlflow.start_run(run_name="drift_detection"):
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
            if report_path.exists():
                mlflow.log_artifact(str(report_path), "drift_reports")

        logger.info("Drift metrics logged to MLflow.")

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e} — continuing without MLflow.")


def run_monitoring_pipeline(log_to_mlflow: bool = True) -> dict:
    """
    Full monitoring pipeline:
    1. Load reference data
    2. Load/simulate current data
    3. Run drift detection
    4. Generate HTML report
    5. Save metrics JSON
    6. Log to MLflow
    """
    reference = load_reference_data()
    current = simulate_current_data(reference)

    metrics, report = detect_drift(reference, current)

    generate_html_report(report)
    save_metrics_json(metrics)

    if log_to_mlflow:
        log_drift_to_mlflow(metrics)

    return metrics


if __name__ == "__main__":
    metrics = run_monitoring_pipeline()
    logger.info("Monitoring pipeline complete.")
    logger.info(f"Results: {json.dumps(metrics, indent=2)}")
