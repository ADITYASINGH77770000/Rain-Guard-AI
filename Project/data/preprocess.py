import pandas as pd
from pathlib import Path

# =========================
# SET BASE PATH (VERY IMPORTANT)
# =========================

BASE_PATH = Path("C:/Users/Admin/Desktop/Riverthon/data")

core_path = BASE_PATH / "processed/Core_features.csv"
soil_path = BASE_PATH / "Soil_Moisture/Processed/soil_moisture_index.csv"
output_path = BASE_PATH / "processed/model_input.csv"


# =========================
# 1. LOAD CORE FEATURES
# =========================

if not core_path.exists():
    raise FileNotFoundError(f"Core file not found: {core_path}")

df = pd.read_csv(core_path)


# =========================
# 2. LOAD SOIL MOISTURE
# =========================

if not soil_path.exists():
    raise FileNotFoundError(f"Soil moisture file not found: {soil_path}")

soil = pd.read_csv(soil_path)

# Merge by date if available
if "date" in df.columns and "date" in soil.columns:
    df = df.merge(soil, on="date", how="inner")
else:
    print("⚠️ 'date' column missing — using average soil moisture.")
    df["soil_moisture_index"] = soil["soil_moisture_index"].mean()


# =========================
# 3. ADD DEM ELEVATION RISK
# =========================

# Derived from DEM (low-lying Mumbai assumption)
df["elevation_risk"] = 0.65


# =========================
# 4. CREATE MODEL INPUT
# =========================

required_cols = [
    "Intensity",
    "soil_moisture_index",
    "elevation_risk",
    "Flood Risk"
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

model_input = df[required_cols]


# =========================
# 5. CLEAN DATA
# =========================

model_input = model_input.dropna()


# =========================
# 6. SAVE FILE
# =========================

output_path.parent.mkdir(parents=True, exist_ok=True)
model_input.to_csv(output_path, index=False)

print("✅ model_input.csv created successfully")
print(model_input.head())