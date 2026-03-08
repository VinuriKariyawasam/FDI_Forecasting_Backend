from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ForecastRequest
from services.prediction_service import predict_fdi
from services.trend_service import get_trend
from services.prediction_service import reload_models
from schemas import ActualFDIUpdate
from services.update_service import update_actual_data
from pathlib import Path
import json
 
app = FastAPI(title="FDI Forecast API")
 
BASE_DIR = Path(__file__).resolve().parent
 
 
# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.post("/forecast")
def forecast(data: ForecastRequest):
    result = predict_fdi(data.dict())
    return result
 
 
@app.post("/trend")
def trend(data: ForecastRequest):
    return get_trend(data.dict())
 
 
@app.post("/reload-models")
def reload():
 
    reload_models()
 
    return {"message": "Models reloaded"}
 
 
@app.post("/update-actual")
def update_actual(data: ActualFDIUpdate):
    return update_actual_data(data)
 
 
@app.get("/last-macros")
def get_last_macros():
 
    with open(BASE_DIR / "models/last_macro_inputs.json") as f:
        data = json.load(f)
 
        print("Last macro inputs:", data)
 
    return data