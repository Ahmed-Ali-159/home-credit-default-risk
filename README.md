# Home Credit Default Risk

Production-grade binary classification system predicting loan repayment probability for underbanked borrowers.
Built on the [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition dataset.

**Evaluation metric:** ROC-AUC  
**Current CV score:** —  
**Kaggle leaderboard score:** —

---

## Project architecture

```
home-credit-default-risk/
├── .github/workflows/      # CI (lint + test on PR) and CD (build + deploy on merge)
├── configs/                # Hydra YAML configs — paths, features, model hyperparameters
├── data/
│   ├── raw/                # Original Kaggle CSVs — tracked by DVC, never by git
│   ├── interim/            # Validated, typed parquets — DVC output
│   └── processed/          # Feature-engineered parquets — DVC output
├── notebooks/              # EDA and modelling exploration notebooks
├── src/
│   ├── data/               # Pydantic schemas + data loading/validation scripts
│   ├── features/           # Feature engineering modules (one per source table)
│   ├── models/             # Training, evaluation, prediction scripts
│   └── monitoring/         # Evidently drift detection
├── api/                    # FastAPI prediction service
├── tests/                  # pytest unit and integration tests
├── docker/                 # Dockerfile and docker-compose
├── dvc.yaml                # Full DVC pipeline definition
└── pyproject.toml          # uv dependencies + ruff + pytest config
```

---

## Quickstart

### 1. Clone and create environment

```bash
git clone https://github.com/<your-username>/home-credit-default-risk.git
cd home-credit-default-risk

# Install uv (if not already installed)
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync --all-extras
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your Kaggle credentials
```

### 3. Download the data

```bash
# Activate the environment
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows PowerShell

# Download from Kaggle (requires KAGGLE_USERNAME and KAGGLE_KEY in .env)
kaggle competitions download -c home-credit-default-risk -p data/raw/
unzip data/raw/home-credit-default-risk.zip -d data/raw/

# Tell DVC to track the raw data
dvc add data/raw/
git add data/raw.dvc .gitignore
git commit -m "feat: track raw data with DVC"
```

### 4. Run the full pipeline

```bash
dvc repro
```

This runs all 4 stages in order: validate → build features → train → predict.

### 5. Launch the prediction API

```bash
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

### 6. Run tests

```bash
pytest
```

---

## MLflow experiment tracking

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Environment | Python 3.11, uv |
| Config | Hydra, OmegaConf |
| Data versioning | DVC |
| Validation | Pydantic v2 |
| ML | LightGBM, XGBoost, scikit-learn |
| Explainability | SHAP |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Serving | FastAPI, uvicorn |
| Monitoring | Evidently |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## Dataset

7 relational tables from Home Credit Group. Key join column: `SK_ID_CURR`.

| Table | Rows (approx) | Description |
|---|---|---|
| application_train | 307,511 | Main table with TARGET label |
| application_test | 48,744 | Test set (no TARGET) |
| bureau | 1,716,428 | External credit history |
| bureau_balance | 27,299,925 | Monthly bureau credit statuses |
| previous_application | 1,670,214 | Past HC loan applications |
| POS_CASH_balance | 10,001,358 | Monthly POS/cash loan snapshots |
| installments_payments | 13,605,401 | Payment history |
| credit_card_balance | 3,840,312 | Monthly CC snapshots |
