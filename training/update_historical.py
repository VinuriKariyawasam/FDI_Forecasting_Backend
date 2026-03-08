import json
from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent.parent
HIST_PATH = BASE_DIR / "app/models/historical.json"
META_PATH = BASE_DIR / "app/models/metadata.json"
 
 
def append_new_actual(quarter, fdi):
 
    with open(HIST_PATH) as f:
        data = json.load(f)
 
    data.append({
        "quarter": quarter,
        "fdi": fdi
    })
 
    with open(HIST_PATH, "w") as f:
        json.dump(data, f, indent=4)
 
    with open(META_PATH) as f:
        meta = json.load(f)
 
    meta["last_actual_fdi"] = fdi
    meta["last_observed_period"] = quarter
 
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
 
    print("Historical data updated")