import json
from pathlib import Path
from services.prediction_service import reload_models
from training.train_models import train
from services.drift_service import detect_drift
from services.prediction_service import predict_fdi
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
HIST_PATH = BASE_DIR / "models/historical.json"
META_PATH = BASE_DIR / "models/metadata.json"
MACRO_PATH = BASE_DIR / "models/last_macro_inputs.json"
PRED_HISTORY_PATH = BASE_DIR / "models/prediction_history.json"
MACRO_DATASET_PATH = BASE_DIR / "models/economic_indicators.json"
 
 
def update_actual_data(data):
 
    quarter = data.quarter
    actual_fdi  = float(data.fdi)
 
    print(f"Updating actual FDI for {quarter}: {actual_fdi}")
 
    # New macro inputs (current quarter actuals)
    new_macros = {
        "gdp_growth_lag1": float(data.gdp_growth_lag1),
        "inflation_lag1": float(data.inflation_lag1),
        "exchange_rate_lag1": float(data.exchange_rate_lag1),
        "interest_rate_lag1": float(data.interest_rate_lag1),
        "private_credit_lag1": float(data.private_credit_lag1)
    }
 
    print(f"New macro inputs for {quarter}:", new_macros)
 
    # STEP 1 — Load previous macro inputs
    with open(MACRO_PATH) as f:
        last_macros = json.load(f)
 
 
     # STEP 2 — Predict using OLD macros
    prediction = predict_fdi(last_macros)
    predicted_fdi = float(prediction["forecast"])
 
    print(f"Predicted FDI for {quarter}: {predicted_fdi}")
 
    # STEP 3 — Load metadata
    with open(META_PATH) as f:
        meta = json.load(f)
 
    # STEP 4 — Drift detection
    drift = detect_drift(actual_fdi, predicted_fdi, meta["residual_std"])
 
    print(f"Drift analysis for {quarter}: {drift}")
 
    # STEP 5 — Store prediction history
    if PRED_HISTORY_PATH.exists():
        with open(PRED_HISTORY_PATH) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
 
    history.append({
        "quarter": quarter,
        "predicted": predicted_fdi,
        "actual": actual_fdi,
        "error": float(round(abs(actual_fdi - predicted_fdi), 2))
    })
 
    print(f"Updated prediction history for {quarter}:", history[-1])
 
    with open(PRED_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)
 
    # STEP 6 — Update historical data
    with open(HIST_PATH) as f:
        historical = json.load(f)
 
    historical.append({
        "quarter": quarter,
        "fdi": actual_fdi
    })
 
    print(f"Updated historical data with {quarter}:", historical[-1])
 
    with open(HIST_PATH, "w") as f:
        json.dump(historical, f, indent=4)
 
    # STEP 7 — Update metadata
    meta["last_actual_fdi"] = actual_fdi
    meta["last_observed_period"] = f"{quarter[:4]} {quarter[4:]}"  # "2026 Q1"
 
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
 
    print(f"Updated metadata with last actual FDI and period for {quarter}:", {
        "last_actual_fdi": meta["last_actual_fdi"],
        "last_observed_period": meta["last_observed_period"]
    })
 
    # STEP 8 — Replace macro inputs with new ones
    with open(MACRO_PATH, "w") as f:
        json.dump(new_macros, f, indent=4)
 
    # STEP 9 — Update macro dataset
    if MACRO_DATASET_PATH.exists():
        with open(MACRO_DATASET_PATH) as f:
            try:
                macro_dataset = json.load(f)
            except json.JSONDecodeError:
                macro_dataset = []
    else:
        macro_dataset = []
 
    macro_dataset.append({
        "quarter": quarter,
        "gdp_growth": new_macros["gdp_growth_lag1"],
        "inflation": new_macros["inflation_lag1"],
        "exchange_rate": new_macros["exchange_rate_lag1"],
        "interest_rate": new_macros["interest_rate_lag1"],
        "private_credit": new_macros["private_credit_lag1"]
    })
 
    with open(MACRO_DATASET_PATH, "w") as f:
        json.dump(macro_dataset, f, indent=4)
 
    print(f"Macro dataset updated for {quarter}")
 
 
    # STEP 10 — Retrain models
    train()
 
    # STEP 11 — Reload models
    reload_models()
 
    result = {
        "message": "Data updated and model retrained",
        "prediction_before_update": predicted_fdi,
        "actual": actual_fdi,
        "drift_analysis": {
            "drift_detected": drift["drift_detected"],
            "error": drift["error"],
            "threshold": float(drift["threshold"])
        }
    }
 
    print(f"Final update result for {quarter}:", result)
 
    return result