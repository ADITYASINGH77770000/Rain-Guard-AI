from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "rl_data" / "alert_rewards.csv"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "alert_policy.pkl"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# State (from ANN)
X = df[["risk_score"]]

# Action taken (alert sent or not)
y = df["alert_sent"]

# -----------------------------
# Train policy model
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Save model
# -----------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("RL alert policy trained and saved at:", MODEL_PATH)
