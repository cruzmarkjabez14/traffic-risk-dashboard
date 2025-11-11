<!-- Traffic pipeline design for MMDA incidents -->
# MMDA Traffic Risk Pipeline Design

## Overview

This document outlines the end-to-end machine learning pipeline for predicting high-risk
traffic periods in Metro Manila using the `data_mmda_traffic_spatial.csv` dataset. The
pipeline mirrors the structure of our previous forecasting blueprint—staging, feature
engineering, dual-task modelling, evaluation, and artifact hand-off—while tailoring
specifics to MMDA incident data and the forthcoming web application.

## Objectives

- Provide re-usable components for both **classification** (flag high-risk time windows)
  and **regression** (estimate incident intensities).
- Maintain Colab-run compatibility for rapid experimentation.
- Produce artifacts consumable by downstream app modules (e.g., serialized models,
  feature importance tables, and period-level risk summaries).

## Data Understanding

Source: MMDA incident feed curated from Twitter posts (`Date`, `Time`, `City`,
`Location`, `Latitude`, `Longitude`, `High_Accuracy`, `Direction`, `Type`,
`Lanes_Blocked`, `Involved`, `Tweet`, `Source`).

Key considerations:

- Combine `Date` and `Time` into a proper timestamp for temporal analysis.
- Validate geographic coordinates and optionally snap to known road segments.
- Account for textual leakage (e.g., `Tweet` containing redundant labels).
- Address duplicates caused by re-posted alerts or updated tweets.

## Pipeline Stages

1. **Ingestion**
   - Load CSV into a schema-validated `pandas` DataFrame.
   - Enforce column typing (dates, categorical features, numeric fields).
   - Optional: persist preprocessed parquet under `data/processed/`.

2. **Preprocessing**
   - Timestamp assembly and timezone normalization.
   - Filter invalid coordinates or impute missing values (drop or replace).
   - Standardize categorical labels (e.g., uppercase & trimmed).
   - Encode `Direction` into canonical categories (NB/SB/EB/WB/UNK).
   - Create `incident_id` for deduplication (`Date` + `Time` + `Location` + `Type`).
   - Train/validation/test splitting using temporal folds (e.g., expanding window).

3. **Feature Engineering**
   - Temporal: hour-of-day, day-of-week, month, holiday/weekend indicators,
     lag counts (rolling incident densities), seasonal decomposition attributes.
   - Spatial: city-level and grid-based aggregations, proximity clusters, lane
     blockage statistics.
   - Incident semantics: type embeddings (Count or TF-IDF vectors from `Tweet`,
     simplified categories), involved vehicle groupings.
   - Weather/external hooks (placeholder for future enrichment).

4. **Label Construction**
   - **Classification**: define high-risk period when incidents per defined window
     exceed configurable threshold (e.g., 75th percentile per hour/city).
   - **Regression**: target is incident count per window or probability of incident.
   - Aggregate to consistent time slices (e.g., hourly or 30-minute bins) prior to
     modelling.

5. **Modelling**
   - Baseline: logistic/linear regression with regularization.
   - Tree-based: Gradient Boosting (LightGBM/XGBoost) or Random Forest for both tasks.
   - Optional neural or sequence models (LSTM/Temporal Fusion) for future iterations.
   - Hyperparameter tuning via cross-validation respecting temporal ordering.

6. **Evaluation**
   - Classification metrics: ROC-AUC, PR-AUC, F1 at relevant thresholds, expected
     cost/benefit curves.
   - Regression metrics: RMSE/MAE, coverage of prediction intervals.
   - Calibration analysis to support decision thresholds (e.g., reliability curves).
   - Feature importance and SHAP to support explainability in the app.

7. **Artifact Management**
   - Serialize trained models (`.pkl` or `.joblib`) under `artifacts/models/`.
   - Export evaluation reports (`evaluation_metrics.json`, per-fold prediction CSVs, feature importance tables) and plots under `artifacts/reports/` and `artifacts/figures/`.
   - Persist aggregated risk tables for UI consumption (`artifacts/outputs/high_risk_windows.csv`).

8. **Integration Hooks**
   - Provide `predict_proba` and scoring helper functions for API usage.
   - Document expected inputs for real-time scoring (timestamp, city, optional context).
   - Outline retraining cadence and data refresh steps.

## Implementation Deliverables

- `scripts/traffic_pipeline.py`: modular implementation with reusable preprocessing,
  feature, and modelling classes.
- `notebooks/traffic_pipeline_prototype.ipynb`: Colab-ready notebook demonstrating
  the pipeline on sample data, including quick EDA and visualization outputs.
- `docs/README.md` (or existing project overview) updated with instructions for
  running the pipeline and integrating results with the application.

## Next Steps

1. Implement modular code built around `PipelineConfig` to keep thresholds, window
   sizes, and model hyperparameters configurable.
2. Develop dataset versioning strategy (local folder now, extendable to DVC).
3. Integrate the pipeline outputs into the app's data service layer once models
   reach baseline performance targets.


