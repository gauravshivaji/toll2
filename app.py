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

# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Toll Traffic & Revenue Forecast", layout="wide")

# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------
def safe_to_int(x):
    try:
        return int(str(x).replace(",", "").strip())
    except:
        return np.nan

def preprocess(df):
    df = df.copy()

    # Standardize column names (common variations)
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

    required_cols = [
        "Car/Jeep Count", "Bus/Truck Count", "LCV Count", "MAV Count",
        "Total Car/Jeep Amount (Rs)", "Total Bus/Truck Amount (Rs)",
        "Total LCV Amount (Rs)", "Total MAV Amount (Rs)"
    ]

    # Convert numerical values
    for c in df.columns:
        if c != "Date":
            df[c] = df[c].apply(lambda x: safe_to_int(x) if pd.notna(x) else np.nan)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Fill missing required columns
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

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
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Lag features
    for lag in (1,7):
        df[f"lag_v_{lag}"] = df["Total_Vehicles"].shift(lag)
        df[f"lag_r_{lag}"] = df["Total_Revenue"].shift(lag)

    # Rolling mean
    df["rm7_v"] = df["Total_Vehicles"].rolling(7, min_periods=1).mean().shift(1)
    df["rm7_r"] = df["Total_Revenue"].rolling(7, min_periods=1).mean().shift(1)

    df = df.dropna().reset_index(drop=True)
    return df


def time_series_cv_eval(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, r2s = [], []
    preds = np.zeros(len(y))

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        preds[test_index] = y_hat

        maes.append(mean_absolute_error(y_test, y_hat))
        r2s.append(r2_score(y_test, y_hat))

    return {
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s),
        "preds": preds
    }


def train_xgb_with_cv(X, y, param_search=False):
    base = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    if param_search:
        param_dist = {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1],
            "colsample_bytree": [0.6, 0.8, 1]
        }
        search = RandomizedSearchCV(
            base, param_dist, n_iter=12,
            cv=3, scoring="neg_mean_absolute_error",
            n_jobs=-1, random_state=42
        )
        search.fit(X, y)
        model = search.best_estimator_
    else:
        model = base
        model.fit(X, y)

    cv_result = time_series_cv_eval(X, y, model)
    return model, cv_result


def make_future_input(history_df, n_days):
    hist = history_df.copy()
    last_date = hist["Date"].max()

    rows = []
    temp = hist.copy()

    for i in range(1, n_days + 1):
        dt = last_date + timedelta(days=i)

        row = {
            "Date": dt,
            "DayOfWeek": dt.dayofweek,
            "IsWeekend": 1 if dt.dayofweek in (5,6) else 0,
            "Month": dt.month,
            "DayOfYear": dt.timetuple().tm_yday,
            "lag_v_1": temp.iloc[-1]["Total_Vehicles"],
            "lag_r_1": temp.iloc[-1]["Total_Revenue"],
            "lag_v_7": temp.iloc[-7]["Total_Vehicles"] if len(temp) >= 7 else temp["Total_Vehicles"].iloc[0],
            "lag_r_7": temp.iloc[-7]["Total_Revenue"] if len(temp) >= 7 else temp["Total_Revenue"].iloc[0],
            "rm7_v": temp["Total_Vehicles"].rolling(7, min_periods=1).mean().iloc[-1],
            "rm7_r": temp["Total_Revenue"].rolling(7, min_periods=1).mean().iloc[-1],
        }

        rows.append(row)

        # update temp for next lag calculations
        temp = pd.concat([
            temp,
            pd.DataFrame([{
                "Date": dt,
                "Total_Vehicles": row["lag_v_1"],
                "Total_Revenue": row["lag_r_1"]
            }])
        ], ignore_index=True)

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.title("🚦 Somatne Phata — Traffic & Revenue Forecast (Working, With Captions)")

uploaded_file = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx", "xls"])
use_example = st.checkbox("Use example dataset", value=False)

if uploaded_file is None and not use_example:
    st.info("Upload file or select 'Use example dataset'.")
    st.stop()

# Load dataset
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
else:
    # Example dummy dataset
    dates = pd.date_range(start="2025-01-01", periods=150)
    np.random.seed(42)
    raw = pd.DataFrame({
        "Date": dates,
        "Car/Jeep Count": 8000 + np.random.randint(-900, 900, len(dates)),
        "Bus/Truck Count": 1200 + np.random.randint(-200, 200, len(dates)),
        "LCV Count": 700 + np.random.randint(-150, 150, len(dates)),
        "MAV Count": 1800 + np.random.randint(-300, 300, len(dates)),
    })
    raw["Total Car/Jeep Amount (Rs)"] = raw["Car/Jeep Count"] * 40
    raw["Total Bus/Truck Amount (Rs)"] = raw["Bus/Truck Count"] * 150
    raw["Total LCV Amount (Rs)"] = raw["LCV Count"] * 80
    raw["Total MAV Amount (Rs)"] = raw["MAV Count"] * 350

st.subheader("Raw Data Preview")
st.write(raw.head())

# PREPROCESS
df = preprocess(raw)
st.success(f"Preprocessed successfully — {len(df)} usable rows.")

# INPUT TRENDS
st.subheader("Input Data Trends")
c1, c2 = st.columns(2)
with c1:
    st.caption("📈 Vehicle Count Trend")
    st.line_chart(df.set_index("Date")["Total_Vehicles"])
with c2:
    st.caption("💰 Revenue Trend")
    st.line_chart(df.set_index("Date")["Total_Revenue"])

# FEATURES
feature_cols = [
    "DayOfWeek", "IsWeekend", "Month", "DayOfYear",
    "lag_v_1", "lag_v_7", "rm7_v",
    "lag_r_1", "lag_r_7", "rm7_r"
]

X = df[feature_cols]
y_v = df["Total_Vehicles"]
y_r = df["Total_Revenue"]

st.subheader("Model Training — XGBoost")
param_search = st.checkbox("Enable hyperparameter search (slow)", value=False)

if st.button("Train Models"):
    with st.spinner("Training traffic model..."):
        traffic_model, cv_v = train_xgb_with_cv(X, y_v, param_search)

    with st.spinner("Training revenue model..."):
        revenue_model, cv_r = train_xgb_with_cv(X, y_r, param_search)

    # Save models
    Path("models").mkdir(exist_ok=True)
    joblib.dump(traffic_model, "models/traffic_model.pkl")
    joblib.dump(revenue_model, "models/revenue_model.pkl")

    st.success("Models trained & saved.")

    st.write("### Traffic Model Accuracy")
    st.write(f"MAE: {cv_v['mae_mean']:.2f} ± {cv_v['mae_std']:.2f}")
    st.write(f"R²: {cv_v['r2_mean']:.3f}")

    st.write("### Revenue Model Accuracy")
    st.write(f"MAE: {cv_r['mae_mean']:.2f} ± {cv_r['mae_std']:.2f}")
    st.write(f"R²: {cv_r['r2_mean']:.3f}")

    # Feature Importance
    st.subheader("Feature Importance — Traffic Model")
    st.bar_chart(pd.Series(traffic_model.feature_importances_, index=feature_cols))

    st.subheader("Feature Importance — Revenue Model")
    st.bar_chart(pd.Series(revenue_model.feature_importances_, index=feature_cols))


# FORECASTING
st.subheader("Forecast Future Days")
n = st.number_input("Days to forecast", 1, 90, 7)

if st.button("Generate Forecast"):
    if not os.path.exists("models/traffic_model.pkl"):
        st.error("Please train models first.")
        st.stop()

    traffic_model = joblib.load("models/traffic_model.pkl")
    revenue_model = joblib.load("models/revenue_model.pkl")

    future_input = make_future_input(df, n)
    X_future = future_input[feature_cols].copy()

    # Iterative forecast
    fut_v, fut_r = [], []
    temp = df.copy()

    for i in range(len(X_future)):
        row = X_future.iloc[i:i+1]

        pv = int(traffic_model.predict(row)[0])
        pr = int(revenue_model.predict(row)[0])

        fut_v.append(pv)
        fut_r.append(pr)

        # Add predicted result
        temp = pd.concat([
            temp,
            pd.DataFrame([{
                "Date": future_input["Date"].iloc[i],
                "Total_Vehicles": pv,
                "Total_Revenue": pr
            }])
        ], ignore_index=True)

        # Update next row lag features
        if i + 1 < len(X_future):
            X_future.loc[i+1, "lag_v_1"] = pv
            X_future.loc[i+1, "lag_r_1"] = pr
            X_future.loc[i+1, "lag_v_7"] = (
                temp.iloc[-7]["Total_Vehicles"]
                if len(temp) >= 7 else temp["Total_Vehicles"].iloc[0]
            )
            X_future.loc[i+1, "lag_r_7"] = (
                temp.iloc[-7]["Total_Revenue"]
                if len(temp) >= 7 else temp["Total_Revenue"].iloc[0]
            )
            X_future.loc[i+1, "rm7_v"] = temp["Total_Vehicles"].rolling(7).mean().iloc[-1]
            X_future.loc[i+1, "rm7_r"] = temp["Total_Revenue"].rolling(7).mean().iloc[-1]

    # Prepare output
    dates = df["Date"].max() + pd.to_timedelta(np.arange(1, n+1), "D")

    result = pd.DataFrame({
        "Date": dates,
        "Predicted_Total_Vehicles": fut_v,
        "Predicted_Total_Revenue": fut_r
    })

    st.write(result)

    st.line_chart(result.set_index("Date")["Predicted_Total_Vehicles"])
    st.line_chart(result.set_index("Date")["Predicted_Total_Revenue"])

st.caption("✔ Fully working app with captions, XGBoost models, feature engineering, and forecasting.")
