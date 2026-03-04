import shap
import numpy as np
import joblib
from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
svr_model = joblib.load(BASE_DIR / "models/svr_model.pkl")
scaler = joblib.load(BASE_DIR / "models/scaler.pkl")
background = joblib.load(BASE_DIR / "models/background.pkl")
 
# Background dataset for KernelExplainer
# Use small sample from training distribution
background = np.zeros((1, 5))
 
explainer = shap.KernelExplainer(
    svr_model.predict,
    background
)
 
feature_names = [
    "GDP Growth",
    "Inflation",
    "Exchange Rate",
    "Interest Rate",
    "Private Credit"
]
 
 
def compute_shap_values(input_array):
 
    shap_values = explainer.shap_values(input_array)
 
    shap_values = shap_values[0]
 
    drivers = []
 
    for i in range(len(feature_names)):
 
        impact = float(shap_values[i])
 
        direction = "positive" if impact > 0 else "negative"
 
        drivers.append({
            "feature": feature_names[i],
            "impact_mn_usd": round(impact, 2),
            "direction": direction
        })
 
    # Sort by absolute impact
    drivers = sorted(drivers, key=lambda x: abs(x["impact_mn_usd"]), reverse=True)
 
    return drivers
 