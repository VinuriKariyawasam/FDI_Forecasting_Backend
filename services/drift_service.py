def detect_drift(actual, predicted, residual_std):
 
    error = abs(actual - predicted)
 
    threshold = 2 * residual_std
 
    if error > threshold:
        return {
            "drift_detected": True,
            "error": error,
            "threshold": threshold
        }
 
    return {
        "drift_detected": False,
        "error": error,
        "threshold": threshold
    }