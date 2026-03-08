import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
HIST_PATH = BASE_DIR / "app/models/historical.json"
MACRO_DATA_PATH = BASE_DIR / "app/models/economic_indicators.json"
MODEL_DIR = BASE_DIR / "app/models"
 
 
def train():
 
    # -----------------------------
    # Load FDI historical data
    # -----------------------------
    with open(HIST_PATH) as f:
        hist_data = json.load(f)
 
    fdi_df = pd.DataFrame(hist_data)
 
    # -----------------------------
    # Load macroeconomic indicators
    # -----------------------------
    with open(MACRO_DATA_PATH) as f:
        macro_data = json.load(f)
 
    macro_df = pd.DataFrame(macro_data)
 
    # -----------------------------
    # Merge datasets by quarter
    # -----------------------------
    df = pd.merge(fdi_df, macro_df, on="quarter", how="inner")
 
    # Ensure correct chronological order
    df = df.sort_values("quarter").reset_index(drop=True)
 
    # -----------------------------
    # Train Holt-Winters model
    # -----------------------------
    hw_model = ExponentialSmoothing(
        df["fdi"],
        trend="add",
        seasonal="add",
        seasonal_periods=4
    ).fit()
 
    # Baseline predictions
    df["baseline"] = hw_model.fittedvalues
 
    # Residuals
    df["residual"] = df["fdi"] - df["baseline"]
 
    # -----------------------------
    # Create lag-1 macro features
    # -----------------------------
    df["gdp_growth_lag1"] = df["gdp_growth"].shift(1)
    df["inflation_lag1"] = df["inflation"].shift(1)
    df["exchange_rate_lag1"] = df["exchange_rate"].shift(1)
    df["interest_rate_lag1"] = df["interest_rate"].shift(1)
    df["private_credit_lag1"] = df["private_credit"].shift(1)
 
    # Drop first row (no lag values)
    df = df.dropna().reset_index(drop=True)
 
    # -----------------------------
    # Train SVR on residuals
    # -----------------------------
    feature_cols = [
        "gdp_growth_lag1",
        "inflation_lag1",
        "exchange_rate_lag1",
        "interest_rate_lag1",
        "private_credit_lag1"
    ]
 
    X = df[feature_cols]
    y = df["residual"]
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    svr = SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1)
   
    svr.fit(X_scaled, y)
 
    # -----------------------------
    # Hybrid predictions
    # -----------------------------
 
    predicted_residuals = svr.predict(X_scaled)
    final_predictions = df["baseline"] + predicted_residuals
 
    # -----------------------------
    # Calculate metrics
    # -----------------------------
    mae = float(np.mean(np.abs(df["fdi"] - final_predictions)))
    rmse = float(np.sqrt(np.mean((df["fdi"] - final_predictions) ** 2)))
    residual_std = float(np.std(df["fdi"] - final_predictions))
 
    # -----------------------------
    # Save models
    # -----------------------------
    joblib.dump(hw_model, MODEL_DIR / "hw_model.pkl")
    joblib.dump(svr, MODEL_DIR / "svr_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
 
    # -----------------------------
    # Update metadata
    # -----------------------------
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
 
    meta["mae"] = round(mae, 2)
    meta["rmse"] = round(rmse, 2)
    meta["residual_std"] = residual_std
 
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
 
    print("Models retrained successfully")