import joblib
import numpy as np
import json
from pathlib import Path
from services.shap_service import compute_shap_values
from services.insight_service import generate_executive_insight
from utils.time_utils import next_quarter
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
# Load models once
hw_model = joblib.load(BASE_DIR / "models/hw_model.pkl")
svr_model = joblib.load(BASE_DIR / "models/svr_model.pkl")
scaler = joblib.load(BASE_DIR / "models/scaler.pkl")
 
with open(BASE_DIR / "models/metadata.json") as f:
    metadata = json.load(f)
 
 
 
# -----------------------------
# Reload models after retraining
# -----------------------------
def reload_models():
 
    global hw_model, svr_model, scaler, metadata
 
    hw_model = joblib.load(BASE_DIR / "models/hw_model.pkl")
    svr_model = joblib.load(BASE_DIR / "models/svr_model.pkl")
    scaler = joblib.load(BASE_DIR / "models/scaler.pkl")
 
    with open(BASE_DIR / "models/metadata.json") as f:
        metadata = json.load(f)
 
 
 
# -----------------------------
# Prediction function
# -----------------------------
def predict_fdi(input_data: dict):
 
    # 1️⃣ Base HW forecast (1 step ahead)
    hw_forecast = float(hw_model.forecast(1).iloc[0])
 
    # 2️⃣ Prepare SVR input
    features = np.array([[
        input_data["gdp_growth_lag1"],
        input_data["inflation_lag1"],
        input_data["exchange_rate_lag1"],
        input_data["interest_rate_lag1"],
        input_data["private_credit_lag1"]
    ]])
 
    scaled_features = scaler.transform(features)
 
    # 3️⃣ Predict residual
    residual_pred = float(svr_model.predict(scaled_features)[0])
 
    # SHAP explanations
    drivers = compute_shap_values(scaled_features)
 
    residual_std = metadata["residual_std"]
 
    # 4️⃣ Final forecast
    final_forecast = hw_forecast + residual_pred
 
    # 5️⃣ % Change
    last_actual = float(metadata["last_actual_fdi"])
    percent_change_qoq = ((final_forecast - last_actual) / last_actual) * 100
 
    # 95% confidence interval
    lower_bound = final_forecast - 1.96 * residual_std
    upper_bound = final_forecast + 1.96 * residual_std
 
    # 8️⃣ Forecast period
    forecast_period = next_quarter(metadata["last_observed_period"])
 
   
 
    # -----------------------------
    # Final response
    # -----------------------------
    result = {
        "forecast": round(final_forecast, 2),
        "period": forecast_period,
        "confidence_interval": {
            "lower": round(lower_bound, 2),
            "upper": round(upper_bound, 2)
        },
        "percent_change_qoq": round(percent_change_qoq, 2),
        "model_metrics": {
            "mae": metadata["mae"],
            "rmse": metadata["rmse"],
            "model_type": metadata["model_type"]
        },
        "drivers": drivers
    }
 
    result["executive_insight"] = generate_executive_insight(result)
 
    return result
 