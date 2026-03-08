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
MODEL_DIR = BASE_DIR / "app/models"
 
 
def train():
 
    with open(HIST_PATH) as f:
        data = json.load(f)
 
    df = pd.DataFrame(data)
 
    # ---- HW MODEL ----
    hw_model = ExponentialSmoothing(
        df["fdi"],
        trend="add",
        seasonal="add",
        seasonal_periods=4
    ).fit()
 
    baseline = hw_model.fittedvalues
 
    residuals = df["fdi"] - baseline
 
    # ---- Dummy lag features (replace with real macro dataset if available)
    X = np.random.randn(len(residuals), 5)
    y = residuals
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    svr = SVR(kernel="rbf")
    svr.fit(X_scaled, y)
 
    # ---- Metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
 
    # ---- Save models
    joblib.dump(hw_model, MODEL_DIR / "hw_model.pkl")
    joblib.dump(svr, MODEL_DIR / "svr_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
 
    # ---- Update metadata
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
 
    meta["mae"] = round(mae, 2)
    meta["rmse"] = round(rmse, 2)
    meta["residual_std"] = float(np.std(residuals))
 
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
 
    print("Models retrained")
 