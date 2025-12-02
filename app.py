"""
Toll Traffic & Revenue Prediction App (fixed for newer pandas versions)

This is an improved version where:
- DataFrame.append is fully replaced with pd.concat
- Recursive forecasting uses safer DataFrame updates
- Works on pandas versions where DataFrame.append has been removed
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from datetime import timedelta

# -------------------------- #
# Streamlit page config
# -------------------------- #
st.set_page_config(page_title="Toll Traffic & Revenue Forecast", layout="wide")

# -------------------------- #
# Utility functions
# -------------------------- #
def safe_to_int(x):
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:
        return np.nan


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names (common variants)
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()

        if "date" in lc:
            col_map[c] = "Date"

        if "car/jeep" in lc and "count" in lc:
            col_map[c] = "Car/Jeep Count"
        if "car/jeep" in lc and "amount" in lc:
            col_map[c] = "Total Car/Jeep Amount (Rs)"

        if "bus/truck" in lc and "count" in lc:
            col_map[c] = "Bus/Truck Count"
        if "bus/truck" in lc and "amount" in lc:
            col_map[c] = "Total Bus/Truck Amount (Rs)"

        if "lcv" in lc and "count" in lc:
            col_map[c] = "LCV Count"
        if "lcv" in lc and "amount" in lc:
            col_map[c] = "Total LCV Amount (Rs)"

        if "mav" in lc and "count" in lc:
            col_map[c] = "MAV Count"
        if "mav" in lc and "amount" in lc:
            col_map[c] = "Total MAV Amount (Rs)"

    df = df.rename(columns=col_map)

    required_counts = [
        "Car/Jeep Count",
        "Bus/Truck Count",
        "LCV Count",
        "MAV Count",
        "Total Car/Jeep Amount (Rs)",
        "Total Bus/Truck Amount (Rs)",
        "Total LCV Amount (Rs)",
        "Total MAV Amount (Rs)",
    ]

    # Clean numeric columns
    for c in df.columns:
        if c != "Date":
            df[c] = df[c].apply(lambda x: safe_to_int(x) if pd.notna(x) else np.nan)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Create missing columns (if any) as 0
    for rc in required_counts:
        if rc not in df.columns:
            df[rc] = 0

    # Aggregate totals
    df["Total_Vehicles"] = (
        df["Car/Jeep Count"].fillna(0)
        + df["Bus/Truck Count"].fillna(0)
        + df["LCV Count"].fillna(0)
        + df["MAV Count"].fillna(0)
    )

    df["Total_Revenue"] = (
        df["Total Car/Jeep Amount (Rs)"].fillna(0)
        + df["Total Bus/Truck Amount (Rs)"].fillna(0)
        + df["Total LCV Amount (Rs)"].fillna(0)
        + df["Total MAV Amount (Rs)"].fillna(0)
    )

    # Calendar features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Lag features
    for lag in (1, 7):
        df[f"lag_v_{lag}"] = df["Total_Vehicles"].shift(lag)
        df[f"lag_r_{lag}"] = df["Total_Revenue"].shift(lag)

    # Rolling mean features
    df["rm7_v"] = (
        df["Total_Vehicles"].rolling(window=7, min_periods=1).mean().shift(1)
    )
    df["rm7_r"] = (
        df["Total_Revenue"].rolling(window=7, min_periods=1).mean().shift(1)
    )

    # Drop initial rows where lags are not available
    df = df.dropna(
        subset=["lag_v_1", "lag_v_7", "rm7_v", "lag_r_1", "lag_r_7", "rm7_r"]
    ).reset_index(drop=True)

    return df


def time_series_cv_eval(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    r2s = []
    preds = np.zeros(len(y))

    for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        m = model
        m.fit(X_train, y_train)
        y_hat = m.predict(X_test)
        preds[test_index] = y_hat

        maes.append(mean_absolute_error(y_test, y_hat))
        r2s.append(r2_score(y_test, y_hat))

    return {
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s),
        "preds": preds,
    }


def train_xgb_with_cv(X, y, param_search=False, random_state=42):
    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )

    if param_search:
        param_dist = {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        rs = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=12,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=random_state,
            n_jobs=-1,
        )
        rs.fit(X, y)
        best = rs.best_estimator_
    else:
        best = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
        best.fit(X, y)

    cv_res = time_series_cv_eval(X, y, best, n_splits=5)
    return best, cv_res


def make_future_input(history_df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    Create future rows for next n_days using recursive approach
    but without using DataFrame.append (uses pd.concat instead).
    """
    hist = history_df.copy().sort_values("Date").reset_index(drop=True)
    last_date = hist["Date"].max()
    rows = []
    temp = hist.copy()  # used to compute lags/rolling

    for i in range(1, n_days + 1):
        dt = last_date + timedelta(days=i)
        dayofweek = dt.dayofweek
        isweekend = 1 if dayofweek in (5, 6) else 0
        month = dt.month
        dayofyear = dt.timetuple().tm_yday

        lag_v_1 = temp.iloc[-1]["Total_Vehicles"]
        lag_r_1 = temp.iloc[-1]["Total_Revenue"]

        if len(temp) >= 7:
            lag_v_7 = temp.iloc[-7]["Total_Vehicles"]
            lag_r_7 = temp.iloc[-7]["Total_Revenue"]
        else:
            lag_v_7 = temp["Total_Vehicles"].iloc[0]
            lag_r_7 = temp["Total_Revenue"].iloc[0]

        rm7_v = temp["Total_Vehicles"].rolling(window=7, min_periods=1).mean().iloc[-1]
        rm7_r = temp["Total_Revenue"].rolling(window=7, min_periods=1).mean().iloc[-1]

        row = {
            "Date": dt,
            "DayOfWeek": dayofweek,
            "IsWeekend": isweekend,
            "Month": month,
            "DayOfYear": dayofyear,
            "lag_v_1": lag_v_1,
            "lag_v_7": lag_v_7,
            "rm7_v": rm7_v,
            "lag_r_1": lag_r_1,
            "lag_r_7": lag_r_7,
            "rm7_r": rm7_r,
        }
        rows.append(row)

        # Instead of append(), use pd.concat to add a small dataframe row to temp.
        new_row_df = pd.DataFrame(
            [{"Date": dt, "Total_Vehicles": lag_v_1, "Total_Revenue": lag_r_1}]
        )
        temp = pd.concat([temp, new_row_df], ignore_index=True)

    fut_df = pd.DataFrame(rows)
    return fut_df


# -------------------------- #
# Streamlit UI
# -------------------------- #
st.title("🚦 Somatne Phata — Traffic & Revenue Forecast (Fixed)")

uploaded_file = st.file_uploader(
    "Upload your CSV / Excel (Somatne Phata data)", type=["csv", "xlsx", "xls"]
)
use_example = st.checkbox("Use example (if you don't upload)", value=False)

if uploaded_file is None and not use_example:
    st.info(
        "Upload CSV/Excel with Date and toll columns, or tick 'Use example' to try demo data."
    )
    st.stop()

# -------------------------- #
# Load data
# -------------------------- #
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw = pd.read_csv(uploaded_file)
        else:
            raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    # Example synthetic data
    dates = pd.date_range(start="2025-01-01", periods=120)
    np.random.seed(42)
    raw = pd.DataFrame(
        {
            "Date": dates,
            "Car/Jeep Count": (
                8000 + np.random.randint(-1000, 1000, len(dates))
            ).astype(int),
            "Bus/Truck Count": (
                1000 + np.random.randint(-300, 300, len(dates))
            ).astype(int),
            "LCV Count": (
                600 + np.random.randint(-200, 200, len(dates))
            ).astype(int),
            "MAV Count": (
                1800 + np.random.randint(-400, 400, len(dates))
            ).astype(int),
        }
    )

    raw["Total Car/Jeep Amount (Rs)"] = raw["Car/Jeep Count"] * 40
    raw["Total Bus/Truck Amount (Rs)"] = raw["Bus/Truck Count"] * 150
    raw["Total LCV Amount (Rs)"] = raw["LCV Count"] * 80
    raw["Total MAV Amount (Rs)"] = raw["MAV Count"] * 350

st.subheader("Raw data preview")
st.write(raw.head())

# -------------------------- #
# Preprocess & feature engineering
# -------------------------- #
with st.spinner("Preprocessing & feature engineering..."):
    df = preprocess(raw)

st.success(f"Preprocessing done — {len(df)} usable rows after lag/rolling features.")

# -------------------------- #
# Trend charts
# -------------------------- #
st.subheader("Traffic & Revenue trend (input data)")
c1, c2 = st.columns(2)

with c1:
    st.line_chart(df.set_index("Date")["Total_Vehicles"])

with c2:
    st.line_chart(df.set_index("Date")["Total_Revenue"])

# -------------------------- #
# Features & targets
# -------------------------- #
feature_cols = [
    "DayOfWeek",
    "IsWeekend",
    "Month",
    "DayOfYear",
    "lag_v_1",
    "lag_v_7",
    "rm7_v",
    "lag_r_1",
    "lag_r_7",
    "rm7_r",
]

X = df[feature_cols]
y_v = df["Total_Vehicles"]
y_r = df["Total_Revenue"]

# -------------------------- #
# Model training
# -------------------------- #
st.subheader("Model training (XGBoost)")
param_search = st.checkbox(
    "Perform hyperparameter randomized search (slower)", value=False
)

if st.button("Train models now"):
    with st.spinner("Training traffic model..."):
        traffic_model, traffic_cv = train_xgb_with_cv(
            X, y_v, param_search=param_search
        )

    with st.spinner("Training revenue model..."):
        revenue_model, revenue_cv = train_xgb_with_cv(
            X, y_r, param_search=param_search
        )

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(traffic_model, model_dir / "traffic_model.pkl")
    joblib.dump(revenue_model, model_dir / "revenue_model.pkl")

    st.success("Models trained and saved to /models/*.pkl")

    st.write(
        "Traffic CV MAE (mean ± std):",
        f"{traffic_cv['mae_mean']:.2f} ± {traffic_cv['mae_std']:.2f}",
    )
    st.write("Traffic CV R2 mean:", f"{traffic_cv['r2_mean']:.3f}")

    st.write(
        "Revenue CV MAE (mean ± std):",
        f"{revenue_cv['mae_mean']:.2f} ± {revenue_cv['mae_std']:.2f}",
    )
    st.write("Revenue CV R2 mean:", f"{revenue_cv['r2_mean']:.3f}")

    # Feature importance
    st.subheader("Feature importance — Traffic model")
    try:
        fi = (
            pd.Series(traffic_model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )
        st.bar_chart(fi)
    except Exception:
        st.info("Could not compute feature importance for traffic model.")

    st.subheader("Feature importance — Revenue model")
    try:
        fi2 = (
            pd.Series(revenue_model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )
        st.bar_chart(fi2)
    except Exception:
        st.info("Could not compute feature importance for revenue model.")

    # Backtest
    st.subheader("Backtest (last 14 days)")
    preds_v = traffic_model.predict(X)
    preds_r = revenue_model.predict(X)

    backtest = df[["Date", "Total_Vehicles", "Total_Revenue"]].copy()
    backtest["pred_v"] = preds_v
    backtest["pred_r"] = preds_r

    st.write(backtest.tail(14))
    st.line_chart(backtest.set_index("Date")[["Total_Vehicles", "pred_v"]])
    st.line_chart(backtest.set_index("Date")[["Total_Revenue", "pred_r"]])

# -------------------------- #
# Load models if already trained
# -------------------------- #
model_dir = Path("models")
traffic_model = None
revenue_model = None

if (model_dir / "traffic_model.pkl").exists() and (
    model_dir / "revenue_model.pkl"
).exists():
    try:
        traffic_model = joblib.load(model_dir / "traffic_model.pkl")
        revenue_model = joblib.load(model_dir / "revenue_model.pkl")
    except Exception:
        traffic_model = None
        revenue_model = None

# -------------------------- #
# Forecast future days
# -------------------------- #
st.subheader("Predict next N days")
n_days = st.number_input(
    "Number of days to forecast", min_value=1, max_value=90, value=7
)

if st.button("Forecast next days"):
    if traffic_model is None or revenue_model is None:
        st.error("Models not found. Train models first (click 'Train models now').")
    else:
        future_input = make_future_input(df, n_days)
        X_future = future_input[feature_cols].copy()

        # iterative prediction while updating next-day lags using pd.concat
        fut_preds_v = []
        fut_preds_r = []

        temp_df = df.copy()
        X_future = X_future.reset_index(drop=True)

        for idx in range(len(X_future)):
            row = X_future.iloc[idx : idx + 1]
            pred_v = int(round(traffic_model.predict(row)[0]))
            pred_r = int(round(revenue_model.predict(row)[0]))

            fut_preds_v.append(pred_v)
            fut_preds_r.append(pred_r)

            # append predicted row to temp_df using pd.concat
            new_row_df = pd.DataFrame(
                [
                    {
                        "Date": future_input["Date"].iloc[idx],
                        "Total_Vehicles": pred_v,
                        "Total_Revenue": pred_r,
                    }
                ]
            )
            temp_df = pd.concat([temp_df, new_row_df], ignore_index=True)

            # update next row's lag features if exists
            if idx + 1 < len(X_future):
                # update lag_1
                X_future.at[idx + 1, "lag_v_1"] = pred_v
                X_future.at[idx + 1, "lag_r_1"] = pred_r

                # update lag_7
                if len(temp_df) >= 7:
                    X_future.at[idx + 1, "lag_v_7"] = temp_df.iloc[-7][
                        "Total_Vehicles"
                    ]
                    X_future.at[idx + 1, "lag_r_7"] = temp_df.iloc[-7][
                        "Total_Revenue"
                    ]
                else:
                    X_future.at[idx + 1, "lag_v_7"] = temp_df["Total_Vehicles"].iloc[0]
                    X_future.at[idx + 1, "lag_r_7"] = temp_df["Total_Revenue"].iloc[0]

                # update rolling mean using temp_df's last rows
                X_future.at[idx + 1, "rm7_v"] = (
                    temp_df["Total_Vehicles"]
                    .rolling(window=7, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                X_future.at[idx + 1, "rm7_r"] = (
                    temp_df["Total_Revenue"]
                    .rolling(window=7, min_periods=1)
                    .mean()
                    .iloc[-1]
                )

        fut_dates = df["Date"].max() + pd.to_timedelta(
            np.arange(1, n_days + 1), unit="D"
        )

        result = pd.DataFrame(
            {
                "Date": fut_dates,
                "Predicted_Total_Vehicles": fut_preds_v,
                "Predicted_Total_Revenue": fut_preds_r,
            }
        )

        st.write(result)
        st.line_chart(result.set_index("Date")["Predicted_Total_Vehicles"])
        st.line_chart(result.set_index("Date")["Predicted_Total_Revenue"])

st.markdown("---")
st.caption(
    "Fixed append issues. Model uses XGBoost with time-series features (lags & rolling means)."
)
