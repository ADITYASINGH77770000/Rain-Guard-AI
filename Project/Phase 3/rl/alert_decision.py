from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "alert_policy.pkl"
model = joblib.load(MODEL_PATH)


def decide_alert(risk_score: float) -> str:
    prob = model.predict_proba([[risk_score]])[0][1]
    return "SEND ALERT" if prob > 0.5 else "NO ALERT"


# Demo decisions
print("Risk 0.85 ->", decide_alert(0.85))
print("Risk 0.40 ->", decide_alert(0.40))
