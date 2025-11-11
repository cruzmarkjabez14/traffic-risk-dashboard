# Running the MMDA Traffic Risk Pipeline

This guide explains how to execute the pipeline and notebook for predicting high-risk traffic periods, both locally and in Google Colab.

## Directory Layout

- `data/raw/`: Place `data_mmda_traffic_spatial.csv` (or symbolic link) here.
- `data/processed/`: Populated automatically with cleaned artifacts.
- `scripts/traffic_pipeline.py`: Reusable pipeline module.
- `notebooks/traffic_pipeline_prototype.ipynb`: Colab-ready walkthrough.
- `artifacts/`: Generated outputs (`models/`, `reports/`, `figures/`, `outputs/`).

## Local Workflow

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt  # or install pandas, numpy, scikit-learn, joblib, seaborn, matplotlib
   ```
2. Ensure the dataset is available at `Ahontek/data/raw/data_mmda_traffic_spatial.csv`.
3. Run the end-to-end pipeline:
   ```bash
   python scripts/traffic_pipeline.py
   ```
4. Inspect outputs under `artifacts/`:
   - `models/traffic_classifier.joblib`
   - `models/traffic_regressor.joblib`
   - `reports/evaluation_metrics.json`
   - `reports/classification_eval.csv` / `reports/regression_eval.csv`
   - `reports/classification_feature_importance.csv` / `reports/regression_feature_importance.csv`
   - `outputs/incident_windows.csv`

## Google Colab Workflow

1. Upload the repository (or clone from version control) into `/content/Ahontek`.
2. Upload `data_mmda_traffic_spatial.csv` into `/content/Ahontek/data/raw/`.
3. Open `notebooks/traffic_pipeline_prototype.ipynb`.
4. Run the setup cell to install dependencies and configure paths.
5. Execute the remaining cells to explore the dataset, engineer features, train models, and export artifacts.

Artifacts produced in Colab appear under `/content/Ahontek/artifacts/` and can be downloaded for integration with the application.

## Integrating with the Web App

- Use the serialized models for inference endpoints that accept `(timestamp, city, optional context)` inputs.
- Feed `incident_windows.csv` into the app to display upcoming high-risk periods and historical trends.
- Refresh the pipeline periodically (e.g., weekly) as new incident data is ingested.

## Extending the Pipeline

- Add weather or road construction data via additional preprocessing steps.
- Tune hyperparameters by extending `PipelineConfig` or integrating Optuna/Hyperopt.
- Introduce alert thresholds tailored to specific stakeholders (commuters vs. traffic management).


