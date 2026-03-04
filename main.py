from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ForecastRequest
from services.prediction_service import predict_fdi
from services.trend_service import get_trend
 
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