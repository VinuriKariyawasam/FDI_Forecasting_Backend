from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ForecastRequest
from services.prediction_service import predict_fdi
from services.trend_service import get_trend
from services.prediction_service import reload_models
from schemas import ActualFDIUpdate
from services.update_service import update_actual_data
 
app = FastAPI(title="FDI Forecast API")
 
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
 
    return update_actual_data(data.quarter, data.fdi)
 