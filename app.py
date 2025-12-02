"""
Toll Traffic & Revenue Prediction App (with captions added for clarity)
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

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Toll Traffic & Revenue Forecast", layout="wide")

# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

# Clean number & convert to int
def safe_to_int(x):
    try:
        return int(str(x).replace(",", "").strip())
    except:
        return np.nan

# ----------------------------------------
# PREPROCESSING: clean columns, create lags
# ----------------------------------------
def preprocess(df):
    df = df.copy()

    # Standardize column names
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

    # Convert numeric columns safely
    for c in df.columns:
        if c != "Date":
            df[c] = df[c].apply(lambda x: safe_to_int(x) if pd.notna(x) else np.nan)

    # Fix invalid dates & sort
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Fill missing required columns
    required_counts = [
        "Car/Jeep Count","Bus/Truck Count","LCV Count","MAV Count",
        "Total Car/Jeep Amount (Rs)","Total Bus/Truck Amount (Rs)",
        "Total LCV Amount (Rs)","Total MAV Amount (Rs)"
    ]
    for rc in required_counts:
        if rc not in df.columns:
            df[rc] = 0

    # Create totals
    df['Total_Vehicles'] = df['Car/Jeep Count'] + df['Bus/Truck Count'] + df['LCV Count'] + df['MAV Count']
    df['Total_Revenue'] = df['Total Car/Jeep Amount (Rs)'] + df['Total Bus/Truck Amount (Rs)'] + df['Total LCV Amount (Rs)'] + df['Total MAV Amount (Rs)']

    # Create time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Create lag & rolling features
    for lag in (1,7):
        df[f'lag_v_{lag}'] = df['Total_Vehicles'].shift(lag)
        df[f'lag_r_{lag}'] = df['Total_Revenue'].shift(lag)

    df['rm7_v'] = df['Total_Vehicles'].rolling(window=7, min_periods=1).mean().shift(1)
    df['rm7_r'] = df['Total_Revenue'].rolling(window=7, min_periods=1).mean().shift(1)

    df = df.dropna().reset_index(drop=True)
    return df

# ----------------------------------------
# TIME SERIES CROSS VALIDATION FOR ACCURACY
# ----------------------------------------
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
        'mae_mean': np.mean(maes),
        'mae_std': np.std(maes),
        'r2_mean': np.mean(r2s),
        'preds': preds
    }

# ----------------------------------------
# TRAIN XGBOOST MODEL WITH CV
# ----------------------------------------
def train_xgb_with_cv(X, y, param_search=False, random_state=42):
    base = XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=-1)

    # Optional hyperparameter search
    if param_search:
        param_dist = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1]
        }
        rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=12,
                                cv=3, scoring='neg_mean_absolute_error',
                                random_state=random_state, n_jobs=-1)
        rs.fit(X, y)
        best = rs.best_estimator_
    else:
        best = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        )
        best.fit(X, y)

    # Perform time-series CV
    cv_res = time_series_cv_eval(X, y, best)
    return best, cv_res

# ----------------------------------------
# CREATE FUTURE INPUTS FOR PREDICTION
# ----------------------------------------
def make_future_input(history_df, n_days):
    hist = history_df.copy().sort_values('Date').reset_index(drop=True)
    last_date = hist['Date'].max()
    rows = []
    temp = hist.copy()

    for i in range(1, n_days+1):
        dt = last_date + timedelta(days=i)

        # Prepare next-day features
        row = {
            'Date': dt,
            'DayOfWeek': dt.dayofweek,
            'IsWeekend': int(dt.dayofweek in (5,6)),
            'Month': dt.month,
            'DayOfYear': dt.timetuple().tm_yday,
            'lag_v_1': temp.iloc[-1]['Total_Vehicles'],
            'lag_r_1': temp.iloc[-1]['Total_Revenue'],
            'lag_v_7': temp.iloc[-7]['Total_Vehicles'] if len(temp)>=7 else temp['Total_Vehicles'].iloc[0],
            'lag_r_7': temp.iloc[-7]['Total_Revenue'] if len(temp)>=7 else temp['Total_Revenue'].iloc[0],
            'rm7_v': temp['Total_Vehicles'].rolling(window=7,min_periods=1).mean().iloc[-1],
            'rm7_r': temp['Total_Revenue'].rolling(window=7,min_periods=1).mean().iloc[-1]
        }
        rows.append(row)

        # Add placeholder for next rolling values
        new_row_df = pd.DataFrame([{
            'Date': dt,
            'Total_Vehicles': row['lag_v_1'],
            'Total_Revenue': row['lag_r_1']
        }])
        temp = pd.concat([temp, new_row_df], ignore_index=True)

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# STREAMLIT APP UI
# ---------------------------------------------------------

st.title("🚦 Somatne Phata — Traffic & Revenue Forecast (Captioned Version)")

# ------------------ File Upload -------------------------
uploaded_file = st.file_uploader("Upload CSV / Excel", type=['csv','xlsx','xls'])
use_example = st.checkbox("Use example dataset")

if uploaded_file is None and not use_example:
    st.info("Upload data or use the example to continue.")
    st.stop()

# ------------------ Load Data ---------------------------
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
else:
    # Example dataset
    dates = pd.date_range(start="2025-01-01", periods=120)
    np.random.seed(42)
    raw = pd.DataFrame({
        "Date": dates,
        "Car/Jeep Count": 8000 + np.random.randint(-1000,1000,len(dates)),
        "Bus/Truck Count": 1000 + np.random.randint(-300,300,len(dates)),
        "LCV Count": 600 + np.random.randint(-200,200,len(dates)),
        "MAV Count": 1800 + np.random.randint(-400,400,len(dates)),
    })
    raw["Total Car/Jeep Amount (Rs)"] = raw["Car/Jeep Count"] * 40
    raw["Total Bus/Truck Amount (Rs)"] = raw["Bus/Truck Count"] * 150
    raw["Total LCV Amount (Rs)"] = raw["LCV Count"] * 80
    raw["Total MAV Amount (Rs)"] = raw["MAV Count"] * 350

st.subheader("📌 Raw Data Preview")
st.write(raw.head())

# ------------------ Preprocessing -----------------------
with st.spinner("Running preprocessing..."):
    df = preprocess(raw)

st.success(f"Preprocessing completed — {len(df)} rows ready for modeling.")

# ------------------ Trend Plots -------------------------
st.subheader("📈 Input Data Trends")
c1, c2 = st.columns(2)
c1.line_chart(df.set_index('Date')['Total_Vehicles'])
c2.line_chart(df.set_index('Date')['Total_Revenue'])

# ------------------ Prepare Features ---------------------
feature_cols = ['DayOfWeek','IsWeekend','Month','DayOfYear','lag_v_1','lag_v_7','rm7_v','lag_r_1','lag_r_7','rm7_r']
X = df[feature_cols]
y_v = df['Total_Vehicles']
y_r = df['Total_Revenue']

# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------
st.subheader("⚙️ Model Training (XGBoost)")

param_search = st.checkbox("Enable hyperparameter search")

if st.button("Train models now"):
    with st.spinner("Training vehicle model..."):
        traffic_model, traffic_cv = train_xgb_with_cv(X, y_v, param_search)

    with st.spinner("Training revenue model..."):
        revenue_model, revenue_cv = train_xgb_with_cv(X, y_r, param_search)

    # Save models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(traffic_model, model_dir / "traffic_model.pkl")
    joblib.dump(revenue_model, model_dir / "revenue_model.pkl")

    st.success("Training completed and models saved.")

    # Accuracy display
    st.write("MAE (Vehicles):", traffic_cv['mae_mean'])
    st.write("R2 (Vehicles):", traffic_cv['r2_mean'])
    st.write("MAE (Revenue):", revenue_cv['mae_mean'])
    st.write("R2 (Revenue):", revenue_cv['r2_mean'])

    # Feature Importance
    st.subheader("🔍 Feature Importance — Vehicles")
    st.bar_chart(pd.Series(traffic_model.feature_importances_, index=feature_cols))

    st.subheader("🔍 Feature Importance — Revenue")
    st.bar_chart(pd.Series(revenue_model.feature_importances_, index=feature_cols))

    # Backtest predictions
    st.subheader("📌 Backtest Results (Last 14 Days)")
    preds_v = traffic_model.predict(X)
    preds_r = revenue_model.predict(X)

    backtest = df[['Date','Total_Vehicles','Total_Revenue']].copy()
    backtest['pred_v'] = preds_v
    backtest['pred_r'] = preds_r

    st.write(backtest.tail(14))
    st.line_chart(backtest.set_index('Date')[['Total_Vehicles','pred_v']])
    st.line_chart(backtest.set_index('Date')[['Total_Revenue','pred_r']])

# ---------------------------------------------------------
# FORECASTING FUTURE DAYS
# ---------------------------------------------------------
model_dir = Path("models")
traffic_model = joblib.load(model_dir / "traffic_model.pkl") if (model_dir/"traffic_model.pkl").exists() else None
revenue_model = joblib.load(model_dir / "revenue_model.pkl") if (model_dir/"revenue_model.pkl").exists() else None

st.subheader("🔮 Predict Next N Days")

n_days = st.number_input("Forecast days:", min_value=1, max_value=90, value=7)

if st.button("Generate Forecast"):
    if traffic_model is None:
        st.error("Train the model first!")
    else:
        future_input = make_future_input(df, n_days)
        X_future = future_input[feature_cols]

        fut_preds_v, fut_preds_r = [], []
        temp_df = df.copy()
        X_future = X_future.reset_index(drop=True)

        for idx in range(len(X_future)):
            row = X_future.iloc[idx:idx+1]
            pred_v = int(round(traffic_model.predict(row)[0]))
            pred_r = int(round(revenue_model.predict(row)[0]))
            fut_preds_v.append(pred_v)
            fut_preds_r.append(pred_r)

            # Update temp history for next-day lags
            new_row_df = pd.DataFrame([{
                'Date': future_input['Date'].iloc[idx],
                'Total_Vehicles': pred_v,
                'Total_Revenue': pred_r
            }])
            temp_df = pd.concat([temp_df, new_row_df], ignore_index=True)

            if idx+1 < len(X_future):
                X_future.at[idx+1, 'lag_v_1'] = pred_v
                X_future.at[idx+1, 'lag_r_1'] = pred_r

                if len(temp_df) >= 7:
                    X_future.at[idx+1, 'lag_v_7'] = temp_df.iloc[-7]['Total_Vehicles']
                    X_future.at[idx+1, 'lag_r_7'] = temp_df.iloc[-7]['Total_Revenue']

                X_future.at[idx+1, 'rm7_v'] = temp_df['Total_Vehicles'].rolling(7,min_periods=1).mean().iloc[-1]
                X_future.at[idx+1, 'rm7_r'] = temp_df['Total_Revenue'].rolling(7,min_periods=1).mean().iloc[-1]

        # Show Results
        fut_dates = df['Date'].max() + pd.to_timedelta(np.arange(1, n_days+1), unit='D')

        result = pd.DataFrame({
            'Date': fut_dates,
            'Predicted_Total_Vehicles': fut_preds_v,
            'Predicted_Total_Revenue': fut_preds_r
        })

        st.write(result)
        st.line_chart(result.set_index('Date')['Predicted_Total_Vehicles'])
        st.line_chart(result.set_index('Date')['Predicted_Total_Revenue'])

st.caption("Model uses XGBoost + time-series features (lags, rolling averages).")
