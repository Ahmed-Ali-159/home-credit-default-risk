"""
src/models/train.py

DVC pipeline entry point for the 'train' stage.
Reads processed parquets → runs 5-fold CV with LightGBM → saves model + metrics.

Run directly:   python src/models/train.py
Run via DVC:    dvc repro train

Key design decisions:
  - Target encoding for high-cardinality columns is done INSIDE the CV loop
    to prevent leakage. Encoding uses smoothing so rare categories are
    pulled toward the global mean.
  - For inference, we save the encoders fitted on the FULL training set
    (not just one fold) — those are what the FastAPI service will use.
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# This MUST come before any 'from src...' imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Now this import will work, because the project root is on sys.path
from src.models.mlflow_utils import promote_if_better, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Columns to target-encode inside each CV fold
HIGH_CARD_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
SMOOTHING = 10  # pulls rare categories toward the global mean


def target_encode(
    train_col: pd.Series,
    val_col: pd.Series,
    test_col: pd.Series,
    y_train: np.ndarray,
    smoothing: int = SMOOTHING,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """
    Smoothed target encoding fitted on training fold only.

    The formula:
        encoded = (group_mean * group_count + global_mean * smoothing)
                  / (group_count + smoothing)
    Effect: large groups trust their own mean; small groups are pulled toward
    the global mean. This prevents wild encodings from categories with 1-2 rows.

    Returns:
        Encoded train, val, test columns + the encoding map for reuse.
    """
    global_mean = y_train.mean()
    stats = (
        pd.DataFrame({"cat": train_col.values, "target": y_train})
        .groupby("cat")["target"]
        .agg(["mean", "count"])
    )
    stats["encoded"] = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
        stats["count"] + smoothing
    )
    encode_map = stats["encoded"].to_dict()

    train_enc = train_col.map(encode_map).fillna(global_mean)
    val_enc = val_col.map(encode_map).fillna(global_mean)
    test_enc = test_col.map(encode_map).fillna(global_mean)
    return train_enc, val_enc, test_enc, encode_map


def run_cv_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lgb_params: dict,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run stratified K-fold cross-validation with LightGBM.

    Returns a dict containing:
      - oof_preds:     out-of-fold predictions on training set
      - test_preds:    averaged predictions on test set
      - fold_aucs:     per-fold validation AUC
      - mean_auc:      mean validation AUC across folds
      - feat_imp:      averaged feature importance
      - feature_cols:  list of feature names (in column order)
    """
    feature_cols = [c for c in train_df.columns if c not in ["TARGET", "SK_ID_CURR"]]
    X_df = train_df[feature_cols].copy()
    y = train_df["TARGET"].values
    X_test_df = test_df[feature_cols].copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_df))
    test_preds = np.zeros(len(X_test_df))
    feat_imp = np.zeros(len(feature_cols))
    fold_aucs = []

    high_card_present = [c for c in HIGH_CARD_COLS if c in X_df.columns]
    logger.info(f"Target encoding columns: {high_card_present}")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_df, y), 1):
        X_trn = X_df.iloc[trn_idx].copy()
        X_val = X_df.iloc[val_idx].copy()
        X_tst = X_test_df.copy()
        y_trn = y[trn_idx]
        y_val = y[val_idx]

        # Target encode high-cardinality columns inside this fold
        for col in high_card_present:
            X_trn[col], X_val[col], X_tst[col], _ = target_encode(
                X_trn[col], X_val[col], X_tst[col], y_trn
            )

        # ── Train ─────────────────────────────────────────────
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_trn,
            y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_tst)[:, 1] / n_splits
        feat_imp += model.feature_importances_ / n_splits

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(fold_auc)
        logger.info(f"Fold {fold}: AUC = {fold_auc:.5f}")

    mean_auc = float(np.mean(fold_aucs))
    overall_auc = float(roc_auc_score(y, oof_preds))
    logger.info(f"\nMean fold AUC:  {mean_auc:.5f}")
    logger.info(f"OOF AUC:        {overall_auc:.5f}")
    logger.info(f"AUC std:        {np.std(fold_aucs):.5f}")

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "overall_auc": overall_auc,
        "feat_imp": feat_imp,
        "feature_cols": feature_cols,
    }


def fit_target_encoders_full(
    X: pd.DataFrame, y: np.ndarray, high_card_cols: list[str]
) -> dict[str, dict]:
    """
    Fit target encoders on the FULL training set for inference.

    Why separate from CV: CV encoders are fold-specific and exist only to
    measure honest validation AUC. For prediction on truly new data
    (the FastAPI service), we want encoders trained on every available
    label, which is the full training set.
    """
    encoders = {}
    global_mean = y.mean()
    for col in high_card_cols:
        if col not in X.columns:
            continue
        stats = (
            pd.DataFrame({"cat": X[col].values, "target": y})
            .groupby("cat")["target"]
            .agg(["mean", "count"])
        )
        stats["encoded"] = (stats["mean"] * stats["count"] + global_mean * SMOOTHING) / (
            stats["count"] + SMOOTHING
        )
        encoders[col] = {
            "map": stats["encoded"].to_dict(),
            "global_mean": float(global_mean),
        }
    return encoders


@hydra.main(config_path="../../configs", config_name="model", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("=" * 60)
    logger.info("STAGE: train")
    logger.info("=" * 60)

    # Hydra resolves paths from data.yaml too — we load it manually here
    # because @hydra.main can only load one config_name at a time
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    # @hydra.main already initialized GlobalHydra for model.yaml.
    # We must clear it before initializing again for data.yaml,
    # otherwise Hydra raises "GlobalHydra is already initialized".
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        data_cfg = compose(config_name="data")

    # ── MLflow setup ──────────────────────────────────────────
    # Must happen before any mlflow.log_* calls
    setup_mlflow(cfg)

    # ── Load features ─────────────────────────────────────────
    logger.info("\nLoading processed feature parquets")
    train_df = pd.read_parquet(data_cfg.paths.features_train)
    test_df = pd.read_parquet(data_cfg.paths.features_test)
    logger.info(f"  train: {train_df.shape} | test: {test_df.shape}")

    # ── Convert Hydra LightGBM params to plain dict ───────────
    lgb_params = dict(cfg.lightgbm)
    # Remove keys LightGBM doesn't accept directly
    lgb_params.pop("early_stopping_rounds", None)
    lgb_params.pop("scale_pos_weight", None)  # null in config = skip

    # ── Everything from here logs to one MLflow run ───────────
    with mlflow.start_run(run_name="lgbm-train"):
        mlflow.log_params(lgb_params)
        mlflow.log_param("n_splits", cfg.cv.n_splits)
        mlflow.log_param("random_seed", cfg.cv.random_seed)
        mlflow.log_param("train_rows", len(train_df))

        # ── Run CV training ───────────────────────────────────────
        results = run_cv_training(
            train_df=train_df,
            test_df=test_df,
            lgb_params=lgb_params,
            n_splits=cfg.cv.n_splits,
            seed=cfg.cv.random_seed,
        )

        # Log per-fold AUCs + overall OOF AUC
        for i, auc in enumerate(results["fold_aucs"], 1):
            mlflow.log_metric(f"fold_{i}_auc", auc)
        mlflow.log_metric("cv_mean_auc", results["mean_auc"])
        mlflow.log_metric("cv_overall_auc", results["overall_auc"])
        mlflow.log_metric("cv_std", float(np.std(results["fold_aucs"])))

        # Tag used by promote_if_better() to compare against Production
        mlflow.set_tag("cv_auc", str(results["overall_auc"]))

        # ── Save model + artifacts ────────────────────────────────
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Fit one final model on full training set for inference + SHAP
        logger.info("\nFitting final model on full training set for inference")
        feature_cols = results["feature_cols"]
        X_full = train_df[feature_cols].copy()
        y_full = train_df["TARGET"].values

        # Fit encoders on full training set
        encoders = fit_target_encoders_full(X_full, y_full, HIGH_CARD_COLS)
        for col in HIGH_CARD_COLS:
            if col in X_full.columns:
                X_full[col] = (
                    X_full[col].map(encoders[col]["map"]).fillna(encoders[col]["global_mean"])
                )

        final_model = lgb.LGBMClassifier(**lgb_params)
        final_model.fit(X_full.values, y_full)

        # Log model to MLflow + register, then promote if it beats Production
        logged_model = mlflow.lightgbm.log_model(final_model, name="model")
        model_uri = logged_model.model_uri
        promote_if_better(
            model_name=cfg.mlflow.model_name,
            model_uri=model_uri,
            new_auc=results["overall_auc"],
            threshold=cfg.mlflow.promotion_threshold,
        )

        # Save local artifacts (DVC tracks these, FastAPI fallback)
        joblib.dump(final_model, models_dir / "lgbm_best.pkl")
        joblib.dump(encoders, models_dir / "target_encoders.pkl")
        joblib.dump(feature_cols, models_dir / "feature_cols.pkl")
        logger.info(f"Saved model + encoders + feature list to {models_dir}/")

        # Save metrics (DVC tracks this file)
        metrics = {
            "cv_mean_auc": results["mean_auc"],
            "cv_overall_auc": results["overall_auc"],
            "cv_fold_aucs": [float(x) for x in results["fold_aucs"]],
            "cv_std": float(np.std(results["fold_aucs"])),
            "n_features": len(feature_cols),
        }
        with open(models_dir / "cv_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(models_dir / "cv_metrics.json"))
        logger.info(f"\nFinal CV AUC: {metrics['cv_overall_auc']:.5f}")

        # Save OOF predictions for ensembling / analysis later
        oof_df = pd.DataFrame(
            {"SK_ID_CURR": train_df["SK_ID_CURR"], "oof_pred": results["oof_preds"]}
        )
        oof_df.to_parquet(models_dir / "oof_predictions.parquet", index=False)

        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    logger.info("\n" + "=" * 60)
    logger.info("train stage complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
