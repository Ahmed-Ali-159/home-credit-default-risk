"""
src/features/build.py

DVC pipeline entry point for the 'build_features' stage.
Reads raw CSVs directly → builds 6 feature groups → writes processed parquets.

Run directly:   python src/features/build.py
Run via DVC:    dvc repro build_features
"""

import gc
import json
import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import (
    load_application,
    load_bureau,
    load_bureau_balance,
    load_credit_card,
    load_installments,
    load_pos_cash,
    load_previous_application,
)
from src.features.application import (
    align_train_test,
    preprocess_application,
)
from src.features.bureau import build_bureau_features
from src.features.credit_card import build_credit_card_features
from src.features.cross_table import build_cross_table_features
from src.features.installments import build_installments_features
from src.features.pos_cash import build_pos_features
from src.features.previous_app import build_prev_app_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: str | Path, name: str) -> None:
    """Save DataFrame to Parquet and log the file size."""

    # Ensure the directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Save to Parquet with pyarrow and snappy compression
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    # Log the file size in MB
    size_mb = Path(path).stat().st_size / 1e6
    logger.info(f"Saved {name}: {df.shape} → {path} ({size_mb:.1f} MB)")


def drop_high_missing(
    train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:

    # drop high-missing features
    # Features with >80% missing rate
    feature_cols = [c for c in train.columns if c not in ["TARGET", "SK_ID_CURR"]]
    missing_rates = train[feature_cols].isna().mean()
    high_missing = missing_rates[missing_rates > threshold].index.tolist()
    logger.info(f"Dropping {len(high_missing)} features with >{threshold:.0%} missing rate")
    train = train.drop(columns=high_missing, errors="ignore")
    test = test.drop(columns=high_missing, errors="ignore")
    return train, test, high_missing


@hydra.main(config_path="../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("=" * 60)
    logger.info("STAGE: build_features")
    logger.info("=" * 60)

    # ── 1. Load raw CSVs ────────────────────────────────────────────────
    logger.info("\n[1/8] Loading raw CSVs")
    app_train = load_application(cfg.paths.application_train, is_train=True)
    app_test = load_application(cfg.paths.application_test, is_train=False)
    bureau = load_bureau(cfg.paths.bureau)
    bureau_balance = load_bureau_balance(cfg.paths.bureau_balance)
    prev_app = load_previous_application(cfg.paths.previous_app)
    pos_cash = load_pos_cash(cfg.paths.pos_cash)
    installments = load_installments(cfg.paths.installments)
    credit_card = load_credit_card(cfg.paths.credit_card)

    # ── 2. Application preprocessing ────────────────────────────────────
    logger.info("\n[2/8] Preprocessing application table")
    app_train = preprocess_application(app_train)
    app_test = preprocess_application(app_test)
    app_train, app_test = align_train_test(app_train, app_test)
    logger.info(f"  train: {app_train.shape}, test: {app_test.shape}")

    # ── 3. Bureau features ──────────────────────────────────────────────
    logger.info("\n[3/8] Building bureau features")
    bureau_feats = build_bureau_features(bureau, bureau_balance)
    app_train = app_train.merge(bureau_feats, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(bureau_feats, on="SK_ID_CURR", how="left")
    del bureau, bureau_balance, bureau_feats
    gc.collect()
    logger.info(f"  after bureau merge — train: {app_train.shape}")

    # ── 4. Previous application features ────────────────────────────────
    logger.info("\n[4/8] Building previous_application features")
    prev_feats = build_prev_app_features(prev_app)
    app_train = app_train.merge(prev_feats, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(prev_feats, on="SK_ID_CURR", how="left")
    del prev_feats
    gc.collect()
    logger.info(f"  after prev_app merge — train: {app_train.shape}")

    # ── 5. POS_CASH features ────────────────────────────────────────────
    logger.info("\n[5/8] Building POS_CASH features")
    pos_feats = build_pos_features(pos_cash, prev_app)
    app_train = app_train.merge(pos_feats, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(pos_feats, on="SK_ID_CURR", how="left")
    del pos_cash, pos_feats
    gc.collect()
    logger.info(f"  after POS_CASH merge — train: {app_train.shape}")

    # ── 6. Installments features ────────────────────────────────────────
    logger.info("\n[6/8] Building installments features")
    inst_feats = build_installments_features(installments, prev_app)
    app_train = app_train.merge(inst_feats, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(inst_feats, on="SK_ID_CURR", how="left")
    del installments, inst_feats
    gc.collect()
    logger.info(f"  after installments merge — train: {app_train.shape}")

    # ── 7. Credit card features ─────────────────────────────────────────
    logger.info("\n[7/8] Building credit_card features")
    cc_feats = build_credit_card_features(credit_card, prev_app)
    app_train = app_train.merge(cc_feats, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(cc_feats, on="SK_ID_CURR", how="left")
    del credit_card, cc_feats, prev_app
    gc.collect()
    logger.info(f"  after credit_card merge — train: {app_train.shape}")

    # ── 8. Cross-table features + final save ────────────────────────────
    logger.info("\n[8/8] Cross-table features + cleanup")
    app_train = build_cross_table_features(app_train)
    app_test = build_cross_table_features(app_test)
    app_train, app_test, dropped = drop_high_missing(app_train, app_test, threshold=0.8)
    logger.info(f"  final train shape: {app_train.shape}")
    logger.info(f"  final test shape:  {app_test.shape}")

    save_parquet(app_train, cfg.paths.features_train, "features_train")
    save_parquet(app_test, cfg.paths.features_test, "features_test")

    feature_count = app_train.shape[1] - 2
    report = {
        "train_rows": int(app_train.shape[0]),
        "test_rows": int(app_test.shape[0]),
        "feature_count": int(feature_count),
        "dropped_high_missing": dropped,
        "target_mean": float(app_train["TARGET"].mean()),
    }
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.paths.processed_dir) / "feature_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"build_features complete — {feature_count} features ready for modelling")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
