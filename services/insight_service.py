def generate_executive_insight(forecast_result: dict):
 
    forecast = forecast_result["forecast"]
    pct_change = forecast_result["percent_change_qoq"]
    drivers = forecast_result["drivers"]
 
    positive = [d for d in drivers if d["direction"] == "positive"]
    negative = [d for d in drivers if d["direction"] == "negative"]
 
    top_pos = sorted(positive, key=lambda x: x["impact_mn_usd"], reverse=True)[:2]
    top_neg = sorted(negative, key=lambda x: x["impact_mn_usd"])[:1]
 
    pos_text = ", ".join([d["feature"].replace("_lag1", "").replace("_", " ") for d in top_pos])
    neg_text = ", ".join([d["feature"].replace("_lag1", "").replace("_", " ") for d in top_neg])
 
    insight = f"""
FDI is forecasted to reach approximately ${forecast:.2f} Mn next quarter,
representing a {pct_change:.2f}% change compared to the previous quarter.
 
Economic drivers such as {pos_text} are expected to positively influence investment inflows,
while pressures from {neg_text} may slightly constrain growth.
"""
 
    return insight.strip()