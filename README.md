# Customer Churn Prediction – Zero‑Config ML Pipeline + Streamlit App

## What this repo delivers

An end‑to‑end, production‑shaped customer churn prediction system that:
- Auto‑discovers your CSV in `data/raw/` and auto‑detects the target column
- Infers schema (numerical/categorical), builds preprocessing, and trains multiple ML models
- Handles class imbalance (SMOTE) and optimizes the classification threshold per dataset
- Saves models per dataset (so you can switch datasets from the app without conflicts)
- Provides a Streamlit web app for data exploration, model comparison, row‑level and full‑dataset predictions
- Ships with GitHub CI, contribution templates, and a clean `.gitignore`

## Key features

- **Zero‑config data ingestion**: Drop a CSV in `data/raw/`, pick it in the sidebar. The pipeline auto‑detects the schema and trains if needed.
- **Target detection**: Heuristic search for common churn/exit labels (e.g., `churn`, `Exited`, `label`, etc.) or any binary column.
- **Preprocessing**:
  - Numerical: median imputation + StandardScaler (configurable)
  - Categorical: most‑frequent imputation + One‑Hot Encoding (`handle_unknown='ignore'`)
  - Optional interaction features and missing value handling utilities
- **Modeling**: Logistic Regression, Random Forest, XGBoost, SVM (extensible)
- **Class imbalance**: SMOTE (configurable)
- **Threshold optimization**: Finds the best threshold (F1 by default; can optimize accuracy/precision/recall)
- **Per‑dataset artifacts**: Models and pipelines saved under `models/<dataset-stem>/` to avoid clashes when you switch datasets
- **App pages**:
  - Home: overview, quick stats, class distribution
  - Data Exploration: types, correlations, dynamic filters for categorical columns
  - Model Performance: metrics, comparison bars, confusion matrix, ROC curve
  - Predictions: row‑by‑row prediction and whole‑dataset scoring, CSV download, dataset‑level summary
  - Feature Analysis: feature importance and correlation with churn

## Project structure

```
<project>/
├── config.yaml
├── requirements.txt
├── requirements-ci.txt
├── run_pipeline.py
├── data/
│   ├── raw/                # put your CSVs here
│   ├── processed/          # generated train/test splits
│   └── external/
├── models/                  # per-dataset artifacts (auto-created)
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── model_trainer.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── webapp/
│   └── streamlit_app.py
├── tests/
│   └── test_imports.py
└── .github/ (workflows, templates)
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt  # or requirements-ci.txt for a lighter setup
```

## Data expectations

- Place your dataset in `data/raw/` as a CSV.
- The system detects the target column automatically. It prefers columns named like `churn`, `Exited`, `label`, `target` (case-insensitive) or any binary column. If not found, you’ll get a helpful error.
- Target values can be `Yes/No`, `True/False`, or `0/1`.

## Run the app (zero‑config)

```bash
streamlit run webapp/streamlit_app.py
```
- Pick the dataset from the sidebar dropdown. First run may take a few minutes while training models.
- You’ll see loading bars/toasts for dataset load, preprocessing, and model training.

## Optional: Run the pipeline via CLI

```bash
# full pipeline (data → features → models)
python run_pipeline.py --step all

# or individual steps
python run_pipeline.py --step data
python run_pipeline.py --step features
python run_pipeline.py --step models
```

## Configuration (config.yaml)

- `auto_mode: true` enables dataset discovery and schema inference.
- `features` controls encoders/scalers (default: One‑Hot + StandardScaler).
- `models` controls SMOTE, CV folds, seeds, and per‑algorithm params.
- `evaluation` controls threshold optimization (`optimize_threshold`, `optimize_for`, `threshold_range`).
- `output` sets paths for models and reports.

Example:
```yaml
auto_mode: true
features:
  encoding_method: onehot
  scaling_method: standard
models:
  use_smote: true
  random_state: 42
  cv_folds: 5
  test_size: 0.2
  random_forest:
    n_estimators: 100
    max_depth: 10
evaluation:
  optimize_threshold: true
  optimize_for: f1   # accuracy|precision|recall|f1
  threshold_range: [0.1, 0.9]
  threshold_step: 0.05
```

## How predictions work

- The best model is selected based on F1 score (configurable by changing the code/criteria) and saved as `models/<dataset>/best_model.joblib`.
- The app uses the saved threshold for classifying probabilities into churn/non‑churn.
- Predictions page:
  - Row prediction: select a row index; the app aligns its columns to the pipeline’s expected inputs.
  - Full dataset: scores every row and offers CSV download; shows aggregate churn rate and, if ground truth exists, evaluation metrics.

## Extending the system

- Add models: extend `ModelTrainer.initialize_models()` and the parameter grids.
- Add feature engineering: extend `FeatureEngineer` to include interaction features, target encoding, or custom transforms.
- Change optimization metric: switch `optimize_for` to `accuracy`, `precision`, or `recall` (or implement a custom utility/cost function).

## CI / Contribution

- GitHub Actions workflow runs flake8 and pytest using `requirements-ci.txt`.
- Use issue and PR templates under `.github/`.
- See `CONTRIBUTING.md` for dev workflow and `CODE_OF_CONDUCT.md` for community standards.

## Limitations & notes

- It’s not statistically sound to guarantee 95% accuracy on arbitrary datasets. We optimize threshold and include SMOTE to improve performance, but actual results depend on dataset quality and class balance. Consider F1/recall for churn use cases.
- First run per dataset will be slower due to training; subsequent runs reuse saved artifacts.

## License

MIT – see `LICENSE`.