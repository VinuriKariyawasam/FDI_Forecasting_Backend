from pydantic import BaseModel
 
class ForecastRequest(BaseModel):
    gdp_growth_lag1: float
    inflation_lag1: float
    exchange_rate_lag1: float
    interest_rate_lag1: float
    private_credit_lag1: float
 
 
class ActualFDIUpdate(BaseModel):
    quarter: str
    fdi: float