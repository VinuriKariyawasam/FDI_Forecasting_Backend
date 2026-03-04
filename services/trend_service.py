import json
from pathlib import Path
from services.prediction_service import predict_fdi
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
with open(BASE_DIR / "models/historical.json") as f:
    historical_data = json.load(f)
 
 
def get_trend(input_data: dict):
 
    forecast_result = predict_fdi(input_data)
 
    # Get last 12 quarters
    last_12 = historical_data[-12:]
 
     # Calculate QoQ percent change
    last_actual = last_12[-1]["fdi"]
    forecast_value = forecast_result["forecast"]
 
    percent_change = (
        (forecast_value - last_actual) / last_actual
    ) * 100
 
 
    return {
        "historical": last_12,
        "forecast": {
            "quarter": "Next Quarter",
            "value": forecast_result["forecast"],
            "lower": forecast_result["confidence_interval"]["lower"],
            "upper": forecast_result["confidence_interval"]["upper"]
        },
        "percent_change_qoq": percent_change
    }
 