"""
RainGuardAI — ANN Flood Risk Model
===================================
Trains a binary flood risk classifier using Mumbai weather data.

Key improvements over original:
  1. Merges real soil moisture data by date (not a fixed average)
  2. Adds humidity as a 4th feature (strong flood predictor)
  3. Physics-based labeling: flood = high intensity + high humidity + saturated soil
  4. Class-weight balancing instead of hard synthetic labels
  5. Saves both the model AND the scaler so the dashboard can use them
  6. Outputs a 4-tier risk level (LOW / MEDIUM / HIGH / CRITICAL)

Run from the model/ directory:
    python train_ann.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
MODEL_DIR   = Path(__file__).resolve().parent
DATA_DIR    = MODEL_DIR.parent / "data"

CORE_PATH   = DATA_DIR / "processed"      / "Core_features.csv"
SOIL_PATH   = DATA_DIR / "Soil_Moisture"  / "Processed" / "soil_moisture_index.csv"
OUTPUT_CSV  = DATA_DIR / "processed"      / "model_input.csv"

MODEL_PATH  = MODEL_DIR / "ann_flood_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# MUST match dashboard/app.py exactly
FEATURE_COLS = ["Intensity", "soil_moisture_index", "elevation_risk", "humidity"]
LABEL_COL    = "flood_label"


# ==================================================================
# STEP 1 — LOAD CORE WEATHER DATA
# ==================================================================
print("\n" + "="*55)
print("STEP 1 — Loading core weather data")
print("="*55)

if not CORE_PATH.exists():
    raise FileNotFoundError(f"Core features not found: {CORE_PATH}")

df = pd.read_csv(CORE_PATH)
df["datetime_parsed"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["datetime_parsed"])

print(f"  Loaded {len(df):,} rows")
print(f"  Date range: {df['datetime_parsed'].min().date()} to {df['datetime_parsed'].max().date()}")


# ==================================================================
# STEP 2 — MERGE REAL SOIL MOISTURE
# ==================================================================
print("\n" + "="*55)
print("STEP 2 — Merging soil moisture")
print("="*55)

if not SOIL_PATH.exists():
    raise FileNotFoundError(f"Soil moisture not found: {SOIL_PATH}")

soil = pd.read_csv(SOIL_PATH)
soil["date_parsed"] = pd.to_datetime(soil["date"], errors="coerce")
soil = soil.dropna(subset=["date_parsed"])

df = df.merge(
    soil[["date_parsed", "soil_moisture_index"]].rename(
        columns={"date_parsed": "datetime_parsed"}),
    on="datetime_parsed", how="left"
)

# Fill gaps outside soil data range with seasonal monsoon defaults
def seasonal_soil_default(date):
    m = date.month
    if m in [6, 7, 8, 9]:    return 0.68   # monsoon
    elif m in [10, 11]:       return 0.45   # post-monsoon
    else:                     return 0.22   # dry

mask = df["soil_moisture_index"].isna()
df.loc[mask, "soil_moisture_index"] = df.loc[mask, "datetime_parsed"].apply(seasonal_soil_default)

print(f"  Merged from CSV: {(~mask).sum():,} rows")
print(f"  Seasonal default: {mask.sum():,} rows")
print(f"  Range: {df['soil_moisture_index'].min():.3f} – {df['soil_moisture_index'].max():.3f}")


# ==================================================================
# STEP 3 — ELEVATION RISK
# ==================================================================
print("\n" + "="*55)
print("STEP 3 — Elevation risk (Mumbai coastal baseline = 0.65)")
print("="*55)

df["elevation_risk"] = 0.65
print("  Set elevation_risk = 0.65 for all rows")


# ==================================================================
# STEP 4 — PHYSICS-BASED FLOOD LABELING
# ==================================================================
print("\n" + "="*55)
print("STEP 4 — Physics-based flood labeling")
print("="*55)

# Real Mumbai floods (2016, 2018): intensity > 150mm + humidity > 88%
# HIGH risk: intensity > 80mm + humidity > 85%  
# These thresholds are calibrated to the two known flood events in the dataset

df[LABEL_COL] = 0

critical_mask = (
    (df["Intensity"]           > 150) &
    (df["humidity"]            > 88)  &
    (df["soil_moisture_index"] > 0.55)
)
high_mask = (
    (df["Intensity"]           > 80) &
    (df["humidity"]            > 85) &
    (df["soil_moisture_index"] > 0.35)
)

df.loc[high_mask | critical_mask, LABEL_COL] = 1

flood_n    = int(df[LABEL_COL].sum())
no_flood_n = len(df) - flood_n
print(f"  Flood (1):    {flood_n:,}  ({flood_n/len(df)*100:.1f}%)")
print(f"  No-Flood (0): {no_flood_n:,}  ({no_flood_n/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: 1 : {no_flood_n//max(flood_n,1)}")

for d in ["01-08-2016", "08-07-2018"]:
    row = df[df["datetime"] == d]
    if not row.empty:
        lbl = int(row[LABEL_COL].values[0])
        ity = row["Intensity"].values[0]
        hum = row["humidity"].values[0]
        print(f"  Real flood {d} → I={ity:.0f}mm H={hum:.0f}% → Label={lbl} {'OK' if lbl==1 else 'CHECK THRESHOLD'}")


# ==================================================================
# STEP 5 — SAVE MODEL INPUT CSV
# ==================================================================
print("\n" + "="*55)
print("STEP 5 — Saving model_input.csv")
print("="*55)

required = FEATURE_COLS + [LABEL_COL]
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

model_input = df[required].dropna()
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
model_input.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved {len(model_input):,} rows to {OUTPUT_CSV}")


# ==================================================================
# STEP 6 — TRAIN / TEST SPLIT
# ==================================================================
print("\n" + "="*55)
print("STEP 6 — Train / test split (80 / 20, stratified)")
print("="*55)

X = model_input[FEATURE_COLS].values
y = model_input[LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_train):,} rows  (flood: {int(y_train.sum())})")
print(f"  Test:  {len(X_test):,} rows  (flood: {int(y_test.sum())})")


# ==================================================================
# STEP 7 — FEATURE SCALING + SAVE SCALER
# ==================================================================
print("\n" + "="*55)
print("STEP 7 — StandardScaler (saving scaler.pkl for dashboard)")
print("="*55)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)
print(f"  Scaler saved to {SCALER_PATH}")
print(f"  Feature means : {dict(zip(FEATURE_COLS, scaler.mean_.round(3)))}")
print(f"  Feature stds  : {dict(zip(FEATURE_COLS, scaler.scale_.round(3)))}")


# ==================================================================
# STEP 8 — CLASS WEIGHTS
# ==================================================================
print("\n" + "="*55)
print("STEP 8 — Class weights (handles 1:54 imbalance)")
print("="*55)

classes     = np.unique(y_train)
class_wts   = compute_class_weight("balanced", classes=classes, y=y_train)
cw_dict     = {int(c): float(w) for c, w in zip(classes, class_wts)}
print(f"  Class weights: {cw_dict}")
print(f"  Flood events weighted {class_wts[1]/class_wts[0]:.0f}x heavier")


# ==================================================================
# STEP 9 — BUILD ANN
# ==================================================================
print("\n" + "="*55)
print("STEP 9 — Building ANN (4 features → flood probability)")
print("="*55)

n_features = X_train_s.shape[1]

model = Sequential([
    Dense(32, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.20),

    Dense(16, activation="relu"),
    BatchNormalization(),
    Dropout(0.15),

    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
], name="RainGuardANN")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print(f"\n  Input features ({n_features}): {FEATURE_COLS}")
print("  Output: flood probability 0.0 – 1.0")


# ==================================================================
# STEP 10 — TRAIN
# ==================================================================
print("\n" + "="*55)
print("STEP 10 — Training (max 100 epochs with early stopping)")
print("="*55)

callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train_s, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1
)

best_epoch = int(np.argmin(history.history["val_loss"])) + 1
print(f"\n  Best epoch    : {best_epoch}")
print(f"  Best val_loss : {min(history.history['val_loss']):.4f}")
print(f"  Best val_acc  : {max(history.history['val_accuracy']):.4f}")


# ==================================================================
# STEP 11 — EVALUATE
# ==================================================================
print("\n" + "="*55)
print("STEP 11 — Evaluation on held-out test set")
print("="*55)

y_pred_prob = model.predict(X_test_s, verbose=0).flatten()
y_pred      = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"  Accuracy : {acc:.4f}")

try:
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"  ROC-AUC  : {auc:.4f}  (1.0 = perfect)")
except ValueError:
    print("  ROC-AUC  : cannot compute (single class in test set)")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion matrix:")
print(f"                  Predicted")
print(f"                  No-Flood   Flood")
print(f"  Actual No-Flood   {cm[0][0]:5d}    {cm[0][1]:5d}")
print(f"  Actual Flood      {cm[1][0]:5d}    {cm[1][1]:5d}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))


# ==================================================================
# STEP 12 — SAVE MODEL
# ==================================================================
print("\n" + "="*55)
print("STEP 12 — Saving model")
print("="*55)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
print(f"  Model saved  → {MODEL_PATH}")
print(f"  Scaler saved → {SCALER_PATH}")


# ==================================================================
# STEP 13 — 4-TIER RISK PREDICTION DEMO
# ==================================================================
print("\n" + "="*55)
print("STEP 13 — 4-Tier risk demo predictions")
print("="*55)

RISK_LEVELS = [
    (0.90, "CRITICAL", "🔴", "Immediate evacuation — activate all emergency teams"),
    (0.75, "HIGH",     "🟠", "Send alerts to authorities — prepare shelters"),
    (0.50, "MEDIUM",   "🟡", "Monitor closely — keep response teams on standby"),
    (0.00, "LOW",      "🟢", "Routine monitoring — no immediate action"),
]

def predict_risk(rainfall_mm, soil_saturation, elevation_risk, humidity_pct):
    """
    Use the trained model to predict flood risk.

    Parameters match FEATURE_COLS exactly:
      rainfall_mm      : mm/hr
      soil_saturation  : 0.0 – 1.0
      elevation_risk   : 0.0 – 1.0
      humidity_pct     : 0 – 100

    Returns dict: probability, risk_level, color, action
    """
    X_in = np.array([[rainfall_mm, soil_saturation, elevation_risk, humidity_pct]])
    X_sc = scaler.transform(X_in)
    prob = float(model.predict(X_sc, verbose=0)[0][0])

    for threshold, level, color, action in RISK_LEVELS:
        if prob >= threshold:
            return {"probability": round(prob, 4),
                    "risk_level": level,
                    "color": color,
                    "action": action}

scenarios = [
    (250, 0.95, 0.65, 93, "Extreme monsoon (Aug 2016 profile)"),
    (120, 0.75, 0.65, 90, "Heavy rain, saturated soil, humid"),
    (80,  0.65, 0.65, 87, "Moderate rain, high humidity"),
    (40,  0.45, 0.65, 75, "Light rain, normal conditions"),
    (5,   0.22, 0.65, 55, "Dry season, sunny day"),
]

print(f"\n  {'Scenario':<44} {'Prob':>6}  {'Level':>8}  Signal")
print(f"  {'-'*44} {'-'*6}  {'-'*8}  ------")
for rainfall, soil, elev, hum, label in scenarios:
    r = predict_risk(rainfall, soil, elev, hum)
    print(f"  {label:<44} {r['probability']:>6.3f}  {r['risk_level']:>8}  {r['color']}")

print("\n" + "="*55)
print("TRAINING COMPLETE")
print("="*55)
print(f"\n  Model  : {MODEL_PATH}")
print(f"  Scaler : {SCALER_PATH}")
print(f"\n  Dashboard usage:")
print(f"    from tensorflow.keras.models import load_model")
print(f"    import joblib, numpy as np")
print(f"    model  = load_model('model/ann_flood_model.keras')")
print(f"    scaler = joblib.load('model/scaler.pkl')")
print(f"    # 4 features: rainfall, soil_moisture, elevation_risk, humidity")
print(f"    X    = np.array([[120, 0.75, 0.65, 90]])")
print(f"    prob = float(model.predict(scaler.transform(X))[0][0])")
print("="*55 + "\n")