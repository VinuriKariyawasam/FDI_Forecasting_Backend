import json
from pathlib import Path
from services.prediction_service import reload_models
from training.train_models import train
 
BASE_DIR = Path(__file__).resolve().parent.parent
HIST_PATH = BASE_DIR / "models/historical.json"
META_PATH = BASE_DIR / "models/metadata.json"
 
 
def update_actual_data(quarter, fdi):
 
    # update historical
    with open(HIST_PATH) as f:
        data = json.load(f)
 
    data.append({
        "quarter": quarter,
        "fdi": fdi
    })
 
    with open(HIST_PATH, "w") as f:
        json.dump(data, f, indent=4)
 
    # update metadata
    with open(META_PATH) as f:
        meta = json.load(f)
 
    meta["last_actual_fdi"] = fdi
    meta["last_observed_period"] = quarter
 
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
 
    # retrain models
    train()
 
    # reload models in API
    reload_models()
 
    return {"message": "Data updated and model retrained"}
 