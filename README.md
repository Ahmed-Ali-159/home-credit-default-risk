# Home Credit Default Risk — Production MLOps Pipeline

> End-to-end machine learning system for loan default prediction, built with production-grade MLOps practices across the full lifecycle: data ingestion, feature engineering, model training, explainability, serving, containerisation, CI/CD, and drift monitoring.

[![CI](https://github.com/Ahmed-Ali-159/home-credit-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmed-Ali-159/home-credit-default-risk/actions/workflows/ci.yml)
[![CD](https://github.com/Ahmed-Ali-159/home-credit-default-risk/actions/workflows/cd.yml/badge.svg)](https://github.com/Ahmed-Ali-159/home-credit-default-risk/actions/workflows/cd.yml)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/Ahmed-Ali-159/home-credit-default-risk/pkgs/container/home-credit-default-risk)

Built on the [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition dataset (binary classification, ROC-AUC metric).

---

## What this project demonstrates

- **Feature engineering at scale** — 262 features engineered from 7 relational tables (300k+ applications, 27M+ bureau balance rows)
- **Leak-free training** — target encoding done inside each CV fold, preventing data leakage
- **Experiment tracking** — every training run logs params, per-fold AUCs, OOF AUC, SHAP plots, and model artifacts to MLflow on DagsHub
- **Model registry** — automatic champion/challenger promotion: new model enters Production only if it beats the current champion by a configurable AUC threshold
- **Explainability** — global SHAP beeswarm/bar plots logged to MLflow; per-prediction waterfall plots served via the API
- **LLM narrative** — SHAP output translated into plain-English explanations for non-technical loan officers via Groq (Llama 3.3 70B)
- **Production serving** — FastAPI with `/predict`, `/explain`, and `/explain/plot` endpoints; MLflow registry lookup with local artifact fallback
- **Containerisation** — multi-stage Docker build (non-root user, OpenMP support, stdlib healthcheck); image published to GHCR on every merge to main
- **CI/CD** — GitHub Actions: lint (ruff) + tests on every PR; Docker build and push on merge to main
- **Drift monitoring** — weekly Evidently drift detection comparing training distribution vs incoming data; HTML report + MLflow metrics logged to DagsHub

---

## Project structure

```
home-credit-default-risk/
├── .github/workflows/
│   ├── ci.yml              # Lint + tests on every PR
│   ├── cd.yml              # Docker build + push to GHCR on merge to main
│   └── monitoring.yml      # Weekly drift detection cron job
├── configs/
│   ├── data.yaml           # File paths (raw CSVs, processed parquets)
│   ├── features.yaml       # Feature group toggles
│   ├── model.yaml          # LightGBM params, CV settings, MLflow settings
│   └── model_best.yaml     # Written by Optuna tuning (optional)
├── src/
│   ├── data/
│   │   └── loader.py       # Reads 7 CSVs with correct dtypes, fixes DAYS_EMPLOYED sentinel
│   ├── features/
│   │   ├── application.py  # Anomaly fixes + engineered features + encoding
│   │   ├── bureau.py       # Two-level aggregation to client level
│   │   ├── previous_app.py # Approval/refusal counts, credit ask growth
│   │   ├── pos_cash.py     # Two-hop SK_ID_PREV -> SK_ID_CURR
│   │   ├── installments.py # Days late, payment ratio per installment
│   │   ├── credit_card.py  # Utilization, ATM advances, balance trends
│   │   ├── cross_table.py  # 6 interaction features across modules
│   │   └── build.py        # DVC entry point, orchestrates all 6 modules
│   ├── models/
│   │   ├── train.py        # 5-fold CV + final model + MLflow logging + SHAP
│   │   ├── tune.py         # Standalone Optuna HPO script (not in DVC pipeline)
│   │   ├── mlflow_utils.py # setup_mlflow(), promote_if_better()
│   │   ├── explain.py      # build_explainer(), explain_single(), plot_waterfall()
│   │   └── narrate.py      # LLM narrative generation via Groq
│   └── monitoring/
│       └── drift.py        # Evidently drift detection pipeline
├── api/
│   ├── schemas.py          # Pydantic request/response models (262 features)
│   ├── model_loader.py     # MLflow registry lookup with local fallback
│   ├── predictor.py        # Preprocessing + prediction (mirrors application.py)
│   └── main.py             # FastAPI app: /health, /model-info, /predict, /explain, /explain/plot
├── tests/                  # pytest smoke tests for features, loader, explain, api, mlflow_utils
├── docker/
│   ├── Dockerfile          # Multi-stage build, non-root user, libgomp1, healthcheck
│   └── docker-compose.yml  # api + local mlflow services
├── dvc.yaml                # DVC pipeline: build features -> train
├── data/
│   ├── raw/                # Original Kaggle CSVs (DVC-tracked, stored on DagsHub S3)
│   └── processed/          # Feature-engineered parquets (DVC output)
├── models/                 # Local model artifacts (joblib fallback for FastAPI)
│   ├── lgbm_best.pkl
│   ├── target_encoders.pkl
│   ├── feature_cols.pkl
│   ├── shap_explainer.pkl
│   └── oof_predictions.parquet
└── pyproject.toml          # uv dependencies + ruff + pytest config
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model load status and version |
| GET | `/model-info` | Model name, CV AUC, feature count |
| POST | `/predict` | Default probability + risk tier (LOW/MEDIUM/HIGH/VERY_HIGH) |
| POST | `/explain` | SHAP top-10 features + LLM plain-English narrative |
| POST | `/explain/plot` | SHAP waterfall PNG (downloadable) |

### Risk tiers

| Tier | Default probability |
|---|---|
| LOW | < 5% |
| MEDIUM | 5% – 15% |
| HIGH | 15% – 30% |
| VERY_HIGH | > 30% |

### Example: `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000.0,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "EXT_SOURCE_1": 0.68,
    "EXT_SOURCE_2": 0.72,
    "EXT_SOURCE_3": 0.65,
    "REGION_RATING_CLIENT": 2
  }'
```

Response:

```json
{
  "default_probability": 0.000276,
  "risk_tier": "LOW",
  "model_version": "v3"
}
```

### Example: `/explain`

Same payload as `/predict`. Returns SHAP top features + LLM narrative:

```json
{
  "default_probability": 0.000276,
  "risk_tier": "LOW",
  "model_version": "v3",
  "baseline_probability": 0.000475,
  "top_features": [
    {"feature": "EXT_SOURCE_2_3", "shap_value": -0.703807, "direction": "decreases_risk", "feature_value": 0.468},
    {"feature": "EXT_SOURCE_MEAN", "shap_value": -0.360745, "direction": "decreases_risk", "feature_value": 0.683}
  ],
  "narrative_explanation": "This loan application is considered low risk, with a default probability lower than the average population. The applicant's strong external credit scores are the primary driver of this assessment..."
}
```

> **Note:** In a production deployment, bureau and credit history features (`buro_*`, `prev_*`, `pos_*`, `inst_*`, `cc_*`) would be looked up from an internal data warehouse using `SK_ID_CURR`. For this demo, all features can be provided directly in the request body. The model handles missing bureau/history features gracefully via LightGBM's native NaN splits.

---

## How to run locally

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for containerised run)
- A [DagsHub](https://dagshub.com) account (for DVC remote + MLflow tracking)
- A [Groq](https://console.groq.com) API key (for LLM narrative in `/explain`)

### 1. Clone and install

```bash
git clone https://github.com/Ahmed-Ali-159/home-credit-default-risk.git
cd home-credit-default-risk
uv sync
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```plaintext
GROQ_API_KEY=your-groq-api-key

MLFLOW_TRACKING_URI=https://dagshub.com/ahmaad.alii.213/home-credit-default-risk.mlflow
MLFLOW_TRACKING_USERNAME=ahmaad.alii.213
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

### 3. Pull model artifacts

```bash
dvc remote modify origin --local access_key_id YOUR_DAGSHUB_ACCESS_KEY
dvc remote modify origin --local secret_access_key YOUR_DAGSHUB_SECRET_KEY
dvc pull
```

This pulls the processed feature parquets and trained model artifacts from DagsHub S3.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Run tests

```bash
uv run pytest
```

### 6. Run drift monitoring

```bash
uv run python src/monitoring/drift.py
start reports/drift_report.html   # Windows
open reports/drift_report.html    # macOS
```

---

## Run with Docker

### Pull the pre-built image

```bash
docker pull ghcr.io/ahmed-ali-159/home-credit-default-risk:latest
```

### Run the container

```bash
docker run -d \
  --name home-credit-api \
  -p 8000:8000 \
  -v "./models:/app/models:ro" \
  --env-file .env \
  ghcr.io/ahmed-ali-159/home-credit-default-risk:latest
```

The `models/` directory is mounted read-only — the image contains only code, not model artifacts. Pull artifacts first with `dvc pull`.

### Run with docker-compose (API + local MLflow)

```bash
docker compose -f docker/docker-compose.yml up
```

This starts two services: the FastAPI app on port 8000 and a local MLflow server on port 5000.

---

## Reproduce the full pipeline

> Requires the raw Kaggle CSVs in `data/raw/`. Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all CSVs there.

```bash
# Run feature engineering + training
dvc repro

# Or run on Kaggle (recommended for full 300k-row dataset — 30GB RAM, ~1-2 hours)
# Clone the repo on Kaggle, set DagsHub secrets, then:
uv run python src/features/build.py
uv run python src/models/train.py

# Push artifacts to DagsHub
dvc push
```

### Hyperparameter tuning (optional)

Optuna tuning is intentionally kept outside the DVC pipeline:

```bash
uv run python src/models/tune.py
# Writes best params to configs/model_best.yaml
# train.py reads this file automatically if present
```

---

## MLflow experiment tracking

All training runs are tracked on DagsHub:

**Tracking server:** `https://dagshub.com/ahmaad.alii.213/home-credit-default-risk.mlflow`

Each run logs:
- LightGBM hyperparameters
- Per-fold AUC scores (5-fold CV)
- OOF AUC (honest out-of-fold estimate)
- SHAP beeswarm and bar plots
- `cv_metrics.json` artifact
- Model artifact (registered in MLflow Model Registry)

The `promote_if_better()` function automatically promotes a new model to Production in the MLflow registry if its OOF AUC exceeds the current Production model by more than a configurable threshold.

---

## Drift monitoring

The monitoring pipeline compares the training feature distribution (reference) against incoming data (current) using Evidently:

```bash
uv run python src/monitoring/drift.py
```

Outputs:
- `reports/drift_report.html` — interactive per-feature drift report with distribution comparisons
- `reports/drift_metrics.json` — summary metrics (share of drifted features, dataset drift verdict)
- MLflow run under the `monitoring` experiment on DagsHub

The monitoring workflow runs automatically every Monday at 08:00 UTC via GitHub Actions, with the HTML report uploaded as a downloadable artifact.

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Package management | uv |
| Configuration | Hydra + OmegaConf |
| Data versioning | DVC + DagsHub S3 |
| Feature engineering | pandas, numpy |
| ML | LightGBM |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow + DagsHub |
| LLM narrative | Groq (Llama 3.3 70B) |
| Serving | FastAPI + uvicorn |
| Drift monitoring | Evidently |
| Containerisation | Docker (multi-stage) |
| Container registry | GitHub Container Registry (GHCR) |
| CI/CD | GitHub Actions |
| Linting | ruff |
| Testing | pytest |

---

## Dataset

7 relational tables from Home Credit Group, joined on `SK_ID_CURR`:

| Table | Rows (approx) | Description |
|---|---|---|
| application_train | 307,511 | Main table — one row per loan application, includes TARGET |
| application_test | 48,744 | Test set (no TARGET) |
| bureau | 1,716,428 | External credit bureau history |
| bureau_balance | 27,299,925 | Monthly credit statuses from bureau |
| previous_application | 1,670,214 | Past Home Credit loan applications |
| POS_CASH_balance | 10,001,358 | Monthly POS and cash loan snapshots |
| installments_payments | 13,605,401 | Repayment history for previous loans |
| credit_card_balance | 3,840,312 | Monthly credit card snapshots |

Raw data is tracked by DVC and stored on DagsHub S3 — never committed to git.

---

## Key design decisions

**Training-serving consistency:** `api/predictor.py` mirrors `src/features/application.py` exactly. Binary columns are label-encoded with the same alphabetical mapping used by `sklearn.LabelEncoder` during training. One-hot encoded columns are reproduced deterministically. Unknown categories produce all-zero dummies — same behaviour as unseen categories at train time.

**Leak-free target encoding:** `ORGANIZATION_TYPE` and `OCCUPATION_TYPE` are high-cardinality categoricals. Their target encoding is fitted inside each CV fold on the training split only, preventing the test split's label distribution from influencing the encoding.

**NaN handling at serving time:** Bureau, POS, installment, and credit card features are not available from the application form — they come from internal databases. The API accepts these as optional fields. When missing, LightGBM handles them natively via its NaN-aware split logic without imputation.

**Optuna outside DVC:** Hyperparameter tuning is a one-off, compute-intensive step run manually on Kaggle. Keeping it outside the DVC pipeline avoids re-running it on every `dvc repro`. The best params are written to `configs/model_best.yaml` and picked up by `train.py` automatically if the file exists.
