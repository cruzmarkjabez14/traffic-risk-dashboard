"""
Reusable ML pipeline components for MMDA traffic incident risk prediction.

The module encapsulates data ingestion, preprocessing, feature engineering,
dual-task modelling (classification & regression), and artifact persistence.

Designed to be imported into notebooks (e.g., Colab) and command-line scripts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar  # Placeholder; replace with PH calendar if available
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    """Configuration envelope for the traffic risk pipeline."""

    raw_data_path: Path = REPO_ROOT / "data" / "raw" / "data_mmda_traffic_spatial.csv"
    processed_data_dir: Path = REPO_ROOT / "data" / "processed"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    resample_freq: str = "1H"
    classification_threshold: float = 0.75  # percentile for high-risk label
    feature_lag_hours: List[int] = field(default_factory=lambda: [1, 3, 6, 24])
    test_size: int = 4  # number of temporal folds for evaluation
    seed: int = 42

    def ensure_directories(self) -> None:
        """Ensure output directories exist."""
        for directory in [
            self.processed_data_dir,
            self.artifacts_dir / "models",
            self.artifacts_dir / "reports",
            self.artifacts_dir / "figures",
            self.artifacts_dir / "outputs",
        ]:
            directory.mkdir(parents=True, exist_ok=True)


class TrafficDataLoader:
    """Load and validate raw incident data."""

    REQUIRED_COLUMNS = [
        "Date",
        "Time",
        "City",
        "Location",
        "Latitude",
        "Longitude",
        "High_Accuracy",
        "Direction",
        "Type",
        "Lanes_Blocked",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        """Load CSV data into a DataFrame with initial typing."""
        LOGGER.info("Loading raw data from %s", self.config.raw_data_path)
        raw_df = pd.read_csv(self.config.raw_data_path)
        missing_cols = set(self.REQUIRED_COLUMNS).difference(raw_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
        return raw_df


class TrafficPreprocessor:
    """Preprocess raw incident data."""

    @staticmethod
    def combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime")
        return df

    @staticmethod
    def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cat_cols = ["City", "Location", "Direction", "Type"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        df["Direction"] = df["Direction"].replace(
            {"N/B": "NB", "S/B": "SB", "E/B": "EB", "W/B": "WB"}
        )
        df["Direction"] = df["Direction"].where(
            df["Direction"].isin(["NB", "SB", "EB", "WB"]), other="UNK"
        )
        return df

    @staticmethod
    def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["incident_id"] = (
            df["Date"].astype(str)
            + "_"
            + df["Time"].astype(str)
            + "_"
            + df["Location"].astype(str)
            + "_"
            + df["Type"].astype(str)
        )
        df = df.drop_duplicates(subset=["incident_id"])
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Preprocessing data...")
        df = self.combine_datetime(df)
        df = self.clean_categories(df)
        df = self.deduplicate(df)
        return df


class FeatureEngineer:
    """Create temporal and categorical features for modelling."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Engineering temporal features with %s resampling", self.config.resample_freq)
        df = df.copy()
        df["timestamp"] = df["datetime"].dt.floor(self.config.resample_freq)
        grouped = df.groupby(["timestamp", "City"])

        agg = grouped.agg(
            incidents=("incident_id", "count"),
            lanes_blocked=("Lanes_Blocked", "sum"),
            unique_types=("Type", "nunique"),
        )
        agg = agg.reset_index()

        # Add time-based features
        agg["hour"] = agg["timestamp"].dt.hour
        agg["day_of_week"] = agg["timestamp"].dt.dayofweek
        agg["is_weekend"] = agg["day_of_week"].isin([5, 6]).astype(int)

        # Holiday flag (placeholder using US calendar; replace with PH-specific logic as needed)
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=agg["timestamp"].min(), end=agg["timestamp"].max())
        agg["is_holiday"] = agg["timestamp"].isin(holidays).astype(int)

        # Rolling/lags per city
        lag_features = []
        agg = agg.set_index("timestamp")
        for lag in self.config.feature_lag_hours:
            feature_name = f"incidents_lag_{lag}h"
            lag_features.append(feature_name)
            agg[feature_name] = (
                agg.groupby("City")["incidents"].shift(1).rolling(window=lag, min_periods=1).mean()
            )
        agg = agg.reset_index()
        agg[lag_features] = agg[lag_features].fillna(0.0)

        return agg


class RiskLabeler:
    """Construct classification and regression targets."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def label(self, features: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Labelling data for classification and regression tasks")
        df = features.copy()
        window_thresholds = (
            df.groupby("City")["incidents"].transform(
                lambda x: np.nanpercentile(x, self.config.classification_threshold * 100)
            )
        )
        df["high_risk"] = (df["incidents"] >= window_thresholds).astype(int)
        df["target_incidents"] = df["incidents"]
        return df


def make_preprocessing_pipeline(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """Build column transformer for downstream models."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ]
    )
    return preprocessor


def train_classification_model(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[BaseEstimator, Dict[str, float], pd.DataFrame]:
    """Train and evaluate classification model."""
    categorical_cols = ["City"]
    numeric_cols = [
        "incidents",
        "lanes_blocked",
        "unique_types",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
    ] + [col for col in df.columns if col.startswith("incidents_lag_")]

    preprocessor = make_preprocessing_pipeline(categorical_cols, numeric_cols)
    classifier = GradientBoostingClassifier(random_state=config.seed)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    tscv = TimeSeriesSplit(n_splits=config.test_size)
    X = df[categorical_cols + numeric_cols]
    y = df["high_risk"]

    roc_scores: List[float] = []
    pr_scores: List[float] = []
    accuracy_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []
    fold_predictions: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_score = roc_auc_score(y_test, y_pred_proba)
        pr_score = average_precision_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        roc_scores.append(roc_score)
        pr_scores.append(pr_score)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        eval_slice = df.iloc[test_idx][["timestamp", "City", "incidents"]].copy()
        eval_slice["fold"] = fold
        eval_slice["actual_high_risk"] = y_test.values
        eval_slice["pred_high_risk"] = y_pred
        eval_slice["pred_high_risk_proba"] = y_pred_proba
        fold_predictions.append(eval_slice)

        LOGGER.info(
            "Fold %s ROC-AUC: %.3f | PR-AUC: %.3f | Acc: %.3f | F1: %.3f",
            fold,
            roc_score,
            pr_score,
            accuracy,
            f1,
        )

    eval_df = pd.concat(fold_predictions).reset_index(drop=True)
    metrics = {
        "roc_auc_mean": float(np.mean(roc_scores)),
        "roc_auc_std": float(np.std(roc_scores)),
        "average_precision_mean": float(np.mean(pr_scores)),
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "precision_mean": float(np.mean(precision_scores)),
        "recall_mean": float(np.mean(recall_scores)),
        "f1_mean": float(np.mean(f1_scores)),
    }

    # Confusion matrix on stacked predictions at 0.5 threshold
    conf_matrix = pd.crosstab(
        eval_df["actual_high_risk"],
        (eval_df["pred_high_risk_proba"] >= 0.5).astype(int),
        rownames=["actual"],
        colnames=["predicted"],
    )
    metrics["confusion_matrix"] = conf_matrix.to_dict()

    # Fit on the full dataset before returning
    model.fit(X, y)
    return model, metrics, eval_df


def train_regression_model(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[BaseEstimator, Dict[str, float], pd.DataFrame]:
    """Train and evaluate regression model."""
    categorical_cols = ["City"]
    numeric_cols = [
        "incidents",
        "lanes_blocked",
        "unique_types",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
    ] + [col for col in df.columns if col.startswith("incidents_lag_")]

    preprocessor = make_preprocessing_pipeline(categorical_cols, numeric_cols)
    regressor = GradientBoostingRegressor(random_state=config.seed)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

    tscv = TimeSeriesSplit(n_splits=config.test_size)
    X = df[categorical_cols + numeric_cols]
    y = df["target_incidents"]

    mae_scores: List[float] = []
    rmse_scores: List[float] = []
    r2_scores: List[float] = []
    fold_predictions: List[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_mae = mean_absolute_error(y_test, y_pred)
        # Support older scikit-learn versions without the `squared` kwarg
        mse = mean_squared_error(y_test, y_pred)
        fold_rmse = float(np.sqrt(mse))
        mae_scores.append(fold_mae)
        rmse_scores.append(fold_rmse)
        fold_r2 = r2_score(y_test, y_pred)
        r2_scores.append(fold_r2)

        eval_slice = df.iloc[test_idx][["timestamp", "City", "incidents"]].copy()
        eval_slice["fold"] = fold
        eval_slice["actual_incidents"] = y_test.values
        eval_slice["pred_incidents"] = y_pred
        fold_predictions.append(eval_slice)

        LOGGER.info("Fold %s MAE: %.3f | RMSE: %.3f | R2: %.3f", fold, fold_mae, fold_rmse, fold_r2)

    metrics = {
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "r2_mean": float(np.mean(r2_scores)),
    }

    model.fit(X, y)
    eval_df = pd.concat(fold_predictions).reset_index(drop=True)
    return model, metrics, eval_df


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def persist_model(model: BaseEstimator, path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to persist models") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def extract_feature_importances(model: Pipeline, estimator_step: str) -> pd.DataFrame:
    """Return feature importances for models that expose them."""
    if "preprocessor" not in model.named_steps or estimator_step not in model.named_steps:
        return pd.DataFrame()

    estimator = model.named_steps[estimator_step]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    preprocessor = model.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        return pd.DataFrame()

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_pipeline(config: Optional[PipelineConfig] = None) -> Dict[str, Dict[str, float]]:
    """Execute the end-to-end pipeline returning evaluation metrics."""
    config = config or PipelineConfig()
    config.ensure_directories()

    loader = TrafficDataLoader(config)
    preprocessor = TrafficPreprocessor()
    engineer = FeatureEngineer(config)
    labeler = RiskLabeler(config)

    raw_df = loader.load()
    processed_df = preprocessor.preprocess(raw_df)
    features_df = engineer.engineer(processed_df)
    labelled_df = labeler.label(features_df)

    metrics: Dict[str, Dict[str, float]] = {}

    classification_model, cls_metrics, cls_eval_df = train_classification_model(labelled_df, config)
    metrics["classification"] = cls_metrics
    persist_model(classification_model, config.artifacts_dir / "models" / "traffic_classifier.joblib")
    cls_eval_path = config.artifacts_dir / "reports" / "classification_eval.csv"
    cls_eval_df.to_csv(cls_eval_path, index=False)

    cls_importances = extract_feature_importances(classification_model, "classifier")
    if not cls_importances.empty:
        cls_importances.to_csv(
            config.artifacts_dir / "reports" / "classification_feature_importance.csv", index=False
        )

    regression_model, reg_metrics, reg_eval_df = train_regression_model(labelled_df, config)
    metrics["regression"] = reg_metrics
    persist_model(regression_model, config.artifacts_dir / "models" / "traffic_regressor.joblib")
    reg_eval_path = config.artifacts_dir / "reports" / "regression_eval.csv"
    reg_eval_df.to_csv(reg_eval_path, index=False)

    reg_importances = extract_feature_importances(regression_model, "regressor")
    if not reg_importances.empty:
        reg_importances.to_csv(
            config.artifacts_dir / "reports" / "regression_feature_importance.csv", index=False
        )

    # Save metrics
    save_json(metrics, config.artifacts_dir / "reports" / "evaluation_metrics.json")

    # Persist aggregated outputs for app usage
    labelled_df.to_csv(config.artifacts_dir / "outputs" / "incident_windows.csv", index=False)

    LOGGER.info("Pipeline completed successfully")
    return metrics


if __name__ == "__main__":
    pipeline_metrics = run_pipeline()
    LOGGER.info("Final metrics: %s", pipeline_metrics)


