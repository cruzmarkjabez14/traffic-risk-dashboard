import json
from pathlib import Path
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# Optional export helpers (CSV/ZIP)
try:  # pragma: no cover
    from app.utils_export import dataframe_to_csv_bytes, make_zip_bytes  # type: ignore
except Exception:  # pragma: no cover
    try:
        from utils_export import dataframe_to_csv_bytes, make_zip_bytes  # type: ignore
    except Exception:  # pragma: no cover
        dataframe_to_csv_bytes = None  # type: ignore
        make_zip_bytes = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
# Handle both running from the repo root and from a nested project directory.
PROJECT_ROOT = REPO_ROOT
if not (PROJECT_ROOT / "artifacts").exists():
    candidate = PROJECT_ROOT / "Ahontek"
    if (candidate / "artifacts").exists():
        PROJECT_ROOT = candidate
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data_mmda_traffic_spatial.csv"
DAY_NAMES: List[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _missing_message(path: Path) -> str:
    return f"Missing expected artifact: `{path}`. Run the traffic pipeline notebook to regenerate."


@st.cache_data(show_spinner=False)
def load_incident_windows() -> pd.DataFrame:
    path = OUTPUTS_DIR / "incident_windows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    return df


@st.cache_data(show_spinner=False)
def load_raw_incidents() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df = df.dropna(subset=["datetime", "Latitude", "Longitude"])
    df["timestamp"] = df["datetime"]
    df["hour"] = df["datetime"].dt.hour
    df["day_name"] = df["datetime"].dt.day_name()
    df["date"] = df["datetime"].dt.date
    df["lanes_blocked"] = pd.to_numeric(df["Lanes_Blocked"], errors="coerce").fillna(0.0)
    df["severity"] = np.select(
        [
            df["lanes_blocked"] >= 3,
            df["lanes_blocked"] >= 2,
            df["lanes_blocked"] >= 1,
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    return df


@st.cache_data(show_spinner=False)
def load_classification_eval() -> pd.DataFrame:
    path = REPORTS_DIR / "classification_eval.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    df["date"] = df["timestamp"].dt.date
    return df


@st.cache_data(show_spinner=False)
def load_regression_eval() -> pd.DataFrame:
    path = REPORTS_DIR / "regression_eval.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    df["date"] = df["timestamp"].dt.date
    return df


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> dict:
    path = REPORTS_DIR / "evaluation_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_feature_importance(task: str) -> pd.DataFrame | None:
    filename = f"{task}_feature_importance.csv"
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def filter_incidents(
    df: pd.DataFrame,
    cities: List[str],
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    days_selected: List[str],
    risk_filter: str,
    types_selected: List[str] | None = None,
    lanes_range: Tuple[float, float] | None = None,
    location_query: str | None = None,
) -> pd.DataFrame:
    if cities:
        df = df[df["City"].isin(cities)]
    if types_selected:
        type_col = "Type" if "Type" in df.columns else None
        if type_col is None and "incident_type" in df.columns:
            type_col = "incident_type"
        if type_col:
            df = df[df[type_col].isin(types_selected)]
    if lanes_range and "lanes_blocked" in df.columns:
        low, high = lanes_range
        df = df[(df["lanes_blocked"] >= low) & (df["lanes_blocked"] <= high)]
    if days_selected:
        if "day_name" in df.columns:
            df = df[df["day_name"].isin(days_selected)]
    if date_range:
        start, end = date_range
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if risk_filter == "High-risk windows" and "high_risk" in df.columns:
        df = df[df["high_risk"] == 1]
    elif risk_filter == "Non-high windows" and "high_risk" in df.columns:
        df = df[df["high_risk"] == 0]
    if location_query:
        location_query = location_query.strip().upper()
        for col in ["Location", "City"]:
            if col in df.columns and location_query:
                df = df[df[col].astype(str).str.upper().str.contains(location_query, na=False)]
                break
    return df


def build_daily_chart(df: pd.DataFrame) -> alt.Chart:
    daily = df.groupby("date", as_index=False)["target_incidents"].sum()
    return (
        alt.Chart(daily)
        .mark_area(color="#2c7fb8", opacity=0.8)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("target_incidents:Q", title="Incident count"),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("target_incidents:Q", title="Incidents")],
        )
        .properties(height=260)
    )


def build_heatmap(df: pd.DataFrame) -> alt.Chart:
    heat = (
        df.groupby(["day_name", "hour"], as_index=False)["target_incidents"]
        .sum()
        .rename(columns={"target_incidents": "incident_count"})
    )
    heat["day_name"] = pd.Categorical(heat["day_name"], categories=DAY_NAMES, ordered=True)
    return (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("day_name:O", title="Day of week"),
            color=alt.Color("incident_count:Q", title="Incidents", scale=alt.Scale(scheme="inferno")),
            tooltip=[
                alt.Tooltip("day_name:O", title="Day"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("incident_count:Q", title="Incidents"),
            ],
        )
        .properties(height=220)
    )


def build_city_bar(df: pd.DataFrame) -> alt.Chart:
    top = (
        df.groupby("City", as_index=False)["target_incidents"]
        .sum()
        .sort_values("target_incidents", ascending=False)
        .head(12)
    )
    return (
        alt.Chart(top)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("target_incidents:Q", title="Incident count"),
            y=alt.Y("City:N", sort="-x", title="City"),
            color=alt.value("#f97316"),
            tooltip=["City", alt.Tooltip("target_incidents:Q", title="Incidents")],
        )
        .properties(height=280)
    )


def build_severity_chart(df: pd.DataFrame) -> alt.Chart:
    severity = (
        df.groupby(["date", "severity"], as_index=False)["target_incidents"]
        .sum()
    )
    return (
        alt.Chart(severity)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("target_incidents:Q", title="Incidents"),
            color=alt.Color("severity:N", title="Severity",
                            scale=alt.Scale(domain=["Critical", "High", "Medium", "Low"],
                                            range=["#ef4444", "#f97316", "#facc15", "#4ade80"])),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("target_incidents:Q", title="Incidents"),
            ],
        )
        .properties(height=240)
    )


def build_hourly_chart(df: pd.DataFrame) -> alt.Chart:
    hourly = (
        df.groupby("hour", as_index=False)["target_incidents"]
        .sum()
    )
    return (
        alt.Chart(hourly)
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("target_incidents:Q", title="Incidents"),
            tooltip=[alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("target_incidents:Q", title="Incidents")],
        )
        .properties(height=220)
    )


def build_day_chart(df: pd.DataFrame) -> alt.Chart:
    by_day = (
        df.groupby("day_name", as_index=False)["target_incidents"]
        .sum()
    )
    by_day["day_name"] = pd.Categorical(by_day["day_name"], categories=DAY_NAMES, ordered=True)
    return (
        alt.Chart(by_day)
        .mark_bar(color="#1f9d55")
        .encode(
            x=alt.X("day_name:O", title="Day of week"),
            y=alt.Y("target_incidents:Q", title="Incidents"),
            tooltip=[alt.Tooltip("day_name:O", title="Day"), alt.Tooltip("target_incidents:Q", title="Incidents")],
        )
        .properties(height=220)
    )


def build_type_chart(raw_df: pd.DataFrame) -> alt.Chart:
    by_type = (
        raw_df.groupby("Type", as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
        .sort_values("incident_count", ascending=False)
        .head(12)
    )
    return (
        alt.Chart(by_type)
        .mark_bar(color="#0ea5e9")
        .encode(
            x=alt.X("incident_count:Q", title="Incidents"),
            y=alt.Y("Type:N", sort="-x", title="Incident type"),
            tooltip=["Type", alt.Tooltip("incident_count:Q", title="Incidents")],
        )
        .properties(height=280)
    )


def as_date_range(values) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(values, (list, tuple)) and len(values) == 2:
        return pd.to_datetime(values[0]), pd.to_datetime(values[1])
    raise ValueError("Date input must contain a start and end value.")


st.set_page_config(page_title="Traffic_risk Dashboard", layout="wide")

st.title("Traffic_risk: Metro Manila High-Risk Incident Dashboard")
st.caption(
    "Aggregates produced from MMDA incident reports (2018–2020). "
    "Use the filters to surface emerging congestion windows and review model diagnostics."
)

try:
    incidents_df = load_incident_windows()
except FileNotFoundError as exc:
    st.error(_missing_message(Path(exc.args[0])))
    st.stop()

try:
    classification_eval = load_classification_eval()
except FileNotFoundError:
    classification_eval = None

try:
    regression_eval = load_regression_eval()
except FileNotFoundError:
    regression_eval = None

try:
    evaluation_metrics = load_evaluation_metrics()
except FileNotFoundError:
    evaluation_metrics = {}

try:
    raw_incidents = load_raw_incidents()
except FileNotFoundError:
    raw_incidents = None

classification_importance = load_feature_importance("classification") if evaluation_metrics else None
regression_importance = load_feature_importance("regression") if evaluation_metrics else None

type_options: List[str] = []
if raw_incidents is not None and "Type" in raw_incidents.columns:
    type_options = sorted(raw_incidents["Type"].dropna().unique().tolist())

min_ts = incidents_df["timestamp"].min()
max_ts = incidents_df["timestamp"].max()
cities = sorted(incidents_df["City"].unique())
lanes_min = float(incidents_df["lanes_blocked"].min())
lanes_max = float(incidents_df["lanes_blocked"].max())

with st.sidebar:
    st.header("Filters")
    selected_cities = st.multiselect("Cities", cities, default=cities)
    if type_options:
        selected_types = st.multiselect("Incident types", type_options, default=type_options)
    else:
        selected_types = []
    lane_min_value = float(lanes_min)
    lane_max_value = float(lanes_max) if lanes_max > lanes_min else lane_min_value + 1.0
    lanes_selection = st.slider(
        "Lanes blocked",
        lane_min_value,
        lane_max_value,
        (lane_min_value, lane_max_value),
        step=0.5,
        help="Filter windows by the number of lanes affected.",
    )
    default_range = (min_ts.date(), max_ts.date())
    date_selection = st.date_input(
        "Date window",
        value=default_range,
        min_value=min_ts.date(),
        max_value=max_ts.date(),
        help="Applies to both overview metrics and evaluation tables.",
    )
    risk_filter = st.selectbox("Window type", ["All windows", "High-risk windows", "Non-high windows"])
    selected_days = st.multiselect("Days of week", DAY_NAMES, default=DAY_NAMES)
    location_query = st.text_input("Search road/intersection", "")
    st.markdown("---")
    st.caption("Artifacts directory: `Ahontek/artifacts/`")

selected_range = as_date_range(date_selection)
filtered_incidents = filter_incidents(
    incidents_df,
    selected_cities,
    selected_range,
    selected_days,
    risk_filter,
    types_selected=selected_types,
    lanes_range=lanes_selection,
    location_query=location_query,
)
filtered_incidents = filtered_incidents.copy()
filtered_incidents["severity"] = np.select(
    [
        filtered_incidents["high_risk"] == 1,
        filtered_incidents["lanes_blocked"] >= 2,
        filtered_incidents["lanes_blocked"] >= 1,
    ],
    ["Critical", "High", "Medium"],
    default="Low",
)

if raw_incidents is not None:
    high_risk_lookup = incidents_df[["timestamp", "City", "high_risk"]].drop_duplicates()
    raw_with_risk = raw_incidents.merge(high_risk_lookup, on=["timestamp", "City"], how="left")
    raw_filtered = filter_incidents(
        raw_with_risk,
        selected_cities,
        selected_range,
        selected_days,
        risk_filter,
        types_selected=selected_types,
        lanes_range=lanes_selection,
        location_query=location_query,
    )
    raw_filtered = raw_filtered.copy()
    raw_filtered["severity"] = np.select(
        [
            raw_filtered["high_risk"] == 1,
            raw_filtered["lanes_blocked"] >= 3,
            raw_filtered["lanes_blocked"] >= 2,
            raw_filtered["lanes_blocked"] >= 1,
        ],
        ["Critical", "Critical", "High", "Medium"],
        default="Low",
    )
else:
    raw_filtered = None

if filtered_incidents.empty:
    st.warning("No windows match the chosen filters. Adjust the selections to see insights.")
    st.stop()

total_incidents = int(filtered_incidents["target_incidents"].sum())
window_count = int(len(filtered_incidents))
high_share = float(filtered_incidents["high_risk"].mean()) if "high_risk" in filtered_incidents else float("nan")
top_city_row = (
    filtered_incidents.groupby("City")["target_incidents"].sum().sort_values(ascending=False).head(1)
)
top_city_label = top_city_row.index[0] if not top_city_row.empty else "—"
top_city_value = int(top_city_row.iloc[0]) if not top_city_row.empty else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Incident windows", f"{window_count:,}")
metric_cols[1].metric("Incidents (sum)", f"{total_incidents:,}")
metric_cols[2].metric("High-risk share", f"{high_share:.2%}")
metric_cols[3].metric("Top city (by incidents)", f"{top_city_label} • {top_city_value:,}")

overview_tab, insights_tab, evaluation_tab, downloads_tab, figures_tab, about_tab = st.tabs(
    ["Overview", "Insights", "Evaluation", "Downloads", "Figures", "About"]
)

with overview_tab:
    st.subheader("Dashboard overview")
    charts_row = st.columns((1.7, 1.7, 1.6))
    with charts_row[0]:
        st.markdown("#### Daily incidents")
        st.altair_chart(build_daily_chart(filtered_incidents), use_container_width=True)
    with charts_row[1]:
        st.markdown("#### Severity trend")
        st.altair_chart(build_severity_chart(filtered_incidents), use_container_width=True)
    with charts_row[2]:
        st.markdown("#### Top cities (incidents)")
        st.altair_chart(build_city_bar(filtered_incidents), use_container_width=True)

    st.markdown("### Incident type outlook")
    type_cols = st.columns((2, 1))
    if raw_filtered is not None and not raw_filtered.empty:
        with type_cols[0]:
            st.altair_chart(build_type_chart(raw_filtered), use_container_width=True)
        top_types = (
            raw_filtered.groupby("Type")
            .size()
            .sort_values(ascending=False)
            .head(5)
            .rename("Incidents")
            .astype(int)
        )
        with type_cols[1]:
            st.markdown("**Most frequent types**")
            st.table(top_types)
    else:
        st.info("Incident type breakdown appears after the raw MMDA dataset is available.")

    st.markdown("### High-risk window highlights")
    hour_selection = st.slider("Select hour range", 0, 23, (6, 9), key="hour_range_overview")
    range_subset = filtered_incidents[
        (filtered_incidents["hour"] >= hour_selection[0]) & (filtered_incidents["hour"] <= hour_selection[1])
    ]
    if classification_eval is not None:
        class_filtered = filter_incidents(
            classification_eval,
            selected_cities,
            selected_range,
            selected_days,
            risk_filter,
            types_selected=selected_types,
            lanes_range=lanes_selection,
            location_query=location_query,
        )
        if not class_filtered.empty:
            highlight = (
                class_filtered.sort_values("pred_high_risk_proba", ascending=False)
                .head(15)[["timestamp", "City", "incidents", "pred_high_risk_proba", "actual_high_risk"]]
            )
            highlight = highlight.rename(
                columns={
                    "timestamp": "Timestamp",
                    "City": "City",
                    "incidents": "Incidents",
                    "pred_high_risk_proba": "Predicted risk",
                    "actual_high_risk": "Actual high-risk",
                }
            )
            highlight["Predicted risk"] = highlight["Predicted risk"].map(lambda x: f"{x:.3f}")
            highlight["Actual high-risk"] = highlight["Actual high-risk"].map(lambda x: "Yes" if x == 1 else "No")
            st.dataframe(highlight, use_container_width=True, hide_index=True)
        else:
            st.info("No evaluation folds overlap the current filter selections.")
    else:
        st.info("Classification evaluation artifacts not found. Run the pipeline to generate them.")

    if not range_subset.empty:
        risky = range_subset[range_subset["high_risk"] == 1]
        if not risky.empty:
            top_combo = (
                risky.groupby(["City", "hour"])["target_incidents"]
                .sum()
                .sort_values(ascending=False)
                .head(1)
            )
            city_name, peak_hour = top_combo.index[0]
            st.success(
                f"Recommendation: Expect congestion spikes around **{city_name}** between "
                f"**{peak_hour}:00–{(peak_hour + 1) % 24}:00**."
            )

with insights_tab:
    st.subheader("Insights")
    insight_cols = st.columns((1.5, 1.5, 1.5))
    with insight_cols[0]:
        st.markdown("#### Incidents by hour")
        st.altair_chart(build_hourly_chart(filtered_incidents), use_container_width=True)
    with insight_cols[1]:
        st.markdown("#### Incidents by weekday")
        st.altair_chart(build_day_chart(filtered_incidents), use_container_width=True)
    with insight_cols[2]:
        st.markdown("#### Heatmap (day/hour)")
        st.altair_chart(build_heatmap(filtered_incidents), use_container_width=True)

    if raw_filtered is not None and not raw_filtered.empty:
        top_city = (
            raw_filtered.groupby("City")
            .size()
            .sort_values(ascending=False)
            .head(1)
        )
        top_type = (
            raw_filtered.groupby("Type")
            .size()
            .sort_values(ascending=False)
            .head(1)
        )
        if not top_city.empty and not top_type.empty:
            st.info(
                f"Operational insight: **{top_city.index[0]}** registers {int(top_city.iloc[0])} incidents "
                f"dominated by **{top_type.index[0]}** reports."
            )

with evaluation_tab:
    st.subheader("Model summary")
    if evaluation_metrics:
        metrics_df = pd.DataFrame(evaluation_metrics).T.reset_index().rename(columns={"index": "task"})
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning(_missing_message(REPORTS_DIR / "evaluation_metrics.json"))

    st.markdown("### Classification diagnostics")
    if classification_eval is not None and not classification_eval.empty:
        class_filtered = filter_incidents(classification_eval, selected_cities, selected_range, selected_days, "All windows")
        if not class_filtered.empty:
            y_true = class_filtered["actual_high_risk"].astype(int)
            y_prob = class_filtered["pred_high_risk_proba"].astype(float)
            y_pred = class_filtered["pred_high_risk"].astype(int)

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            cm = confusion_matrix(y_true, y_pred)

            roc_chart = (
                alt.Chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
                .mark_line(color="#1b9e77", size=3)
                .encode(x=alt.X("FPR", title="False positive rate"), y=alt.Y("TPR", title="True positive rate"))
                .properties(height=240)
            )
            pr_chart = (
                alt.Chart(pd.DataFrame({"Recall": recall, "Precision": precision}))
                .mark_line(color="#d95f02", size=3)
                .encode(x=alt.X("Recall", title="Recall"), y=alt.Y("Precision", title="Precision"))
                .properties(height=240)
            )
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.altair_chart(roc_chart, use_container_width=True)
            with chart_cols[1]:
                st.altair_chart(pr_chart, use_container_width=True)

            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.caption("Confusion matrix (evaluation folds)")
            st.dataframe(cm_df, use_container_width=True)
        else:
            st.info("No classification evaluation rows match the current filters.")

        if classification_importance is not None and not classification_importance.empty:
            st.markdown("#### Top classification feature importances")
            st.dataframe(
                classification_importance.head(20),
                use_container_width=True,
                hide_index=True,
                    )
    else:
        st.warning(_missing_message(REPORTS_DIR / "classification_eval.csv"))

    st.markdown("### Regression diagnostics")
    if regression_eval is not None and not regression_eval.empty:
        reg_filtered = filter_incidents(regression_eval, selected_cities, selected_range, selected_days, "All windows")
        if not reg_filtered.empty:
            reg_filtered = reg_filtered.assign(residual=lambda df: df["actual_incidents"] - df["pred_incidents"])
            scatter_chart = (
                alt.Chart(reg_filtered)
                .mark_circle(color="#4daf4a", opacity=0.5)
                .encode(
                    x=alt.X("actual_incidents:Q", title="Actual incidents"),
                    y=alt.Y("pred_incidents:Q", title="Predicted incidents"),
                    tooltip=["timestamp", "City", "actual_incidents", "pred_incidents"],
                )
                .properties(height=260)
            )
            hist_chart = (
                alt.Chart(reg_filtered)
                .mark_bar(color="#984ea3", opacity=0.8)
                .encode(
                    x=alt.X("residual:Q", bin=alt.Bin(maxbins=40), title="Actual - predicted"),
                    y=alt.Y("count()", title="Frequency"),
                )
                .properties(height=260)
            )
            reg_cols = st.columns(2)
            with reg_cols[0]:
                st.altair_chart(scatter_chart, use_container_width=True)
            with reg_cols[1]:
                st.altair_chart(hist_chart, use_container_width=True)
        else:
            st.info("No regression evaluation rows match the current filters.")

        if regression_importance is not None and not regression_importance.empty:
            st.markdown("#### Top regression feature importances")
            st.dataframe(regression_importance.head(20), use_container_width=True, hide_index=True)
    else:
        st.warning(_missing_message(REPORTS_DIR / "regression_eval.csv"))

with downloads_tab:
    st.subheader("Exports")
    download_cols = st.columns(3)

    with download_cols[0]:
        if dataframe_to_csv_bytes is not None:
            st.download_button(
                "Download filtered windows (CSV)",
                data=dataframe_to_csv_bytes(filtered_incidents),
                file_name="traffic_windows_filtered.csv",
            )
        else:
            st.caption("CSV export helper unavailable.")

    with download_cols[1]:
        if classification_eval is not None and dataframe_to_csv_bytes is not None:
            class_filtered = filter_incidents(
                classification_eval,
                selected_cities,
                selected_range,
                selected_days,
                risk_filter,
                types_selected=selected_types,
                lanes_range=lanes_selection,
                location_query=location_query,
            )
            st.download_button(
                "Download classification eval (CSV)",
                data=dataframe_to_csv_bytes(class_filtered),
                file_name="classification_eval_filtered.csv",
            )
        else:
            st.caption("Classification evaluation not available.")

    with download_cols[2]:
        if make_zip_bytes is not None:
            desired_files = [
                REPORTS_DIR / "evaluation_metrics.json",
                REPORTS_DIR / "classification_eval.csv",
                REPORTS_DIR / "regression_eval.csv",
                OUTPUTS_DIR / "incident_windows.csv",
            ]
            files_present = [p for p in desired_files if p.exists()]
            if files_present:
                st.download_button(
                    "Download key artifacts (ZIP)",
                    data=make_zip_bytes(files_present),
                    file_name="traffic_risk_artifacts.zip",
                )
            else:
                st.caption("Artifacts missing. Run the pipeline to generate them.")
with figures_tab:
    st.subheader("Figures")
    if FIGURES_DIR.exists():
        figures = sorted(FIGURES_DIR.glob("*.png"))
        if figures:
            cols_per_row = 3
            for i in range(0, len(figures), cols_per_row):
                row = st.columns(min(cols_per_row, len(figures) - i))
                for col, img_path in zip(row, figures[i : i + cols_per_row]):
                    with col:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
        else:
            st.caption("No figures saved yet.")
    else:
        st.caption("Figures directory not found. Run the notebook to generate visuals.")

with about_tab:
    st.subheader("About Traffic_risk")
    st.markdown(
        """
        **Traffic_risk** is a decision-support dashboard powered by historical MMDA incident alerts.
        It advances **SDG 11 – Sustainable Cities and Communities** by exposing recurring congestion windows
        and critical corridors across Metro Manila.

        ### How the system works
        - MMDA tweets are ingested, cleaned, and aggregated hourly per city.
        - Classification and regression models flag high-risk hours and estimate incident volume (see the notebook + pipeline).
        - Outputs feed this dashboard: overview metrics, hotspot heatmaps, interactive map, and downloadable reports.

        ### Data source
        - Dataset: *Manila Traffic Incident Data* (Kaggle, by user **esparko**).
        - Fields: timestamp, city, location, incident type, lanes blocked, tweet metadata, lat/long coordinates.

        ### Future extensions
        - Automate daily refreshes or streaming updates.
        - Enrich with weather, construction, or event schedules.
        - Deliver API endpoints or alerts for commuter and LGU applications.
        """
    )

st.divider()
st.caption(
    "Ethics: Aggregated insights only. Raw MMDA incident data stays local; exports contain summaries for planning."
)