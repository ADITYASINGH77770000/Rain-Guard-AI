from pathlib import Path

import geopandas as gpd
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1] / "data"
FLOOD_ZONES = BASE_DIR / "processed" / "low_lying_zones.geojson"
INFRA_IMPACT = BASE_DIR / "processed" / "flood_impact_zones.geojson"
POPULATION_CANDIDATES = [
    BASE_DIR / "raw" / "population" / "mumbai_wardwise_population_data.csv",
    BASE_DIR / "raw" / "Demographics" / "mumbai_wardwise_population_data.csv",
]

OUTPUT = BASE_DIR / "processed" / "flood_exposure_summary.csv"

# -----------------------------
# Load data
# -----------------------------
flood_zones = gpd.read_file(FLOOD_ZONES)
infra = gpd.read_file(INFRA_IMPACT)
population_path = next((p for p in POPULATION_CANDIDATES if p.exists()), None)
if population_path is None:
    expected = ", ".join(str(p) for p in POPULATION_CANDIDATES)
    raise FileNotFoundError(f"Population CSV not found. Expected one of: {expected}")
population = pd.read_csv(population_path)

# -----------------------------
# Population proxy
# -----------------------------
population_column_candidates = [
    "population",
    "Population",
    "Population_2025",
    "Population_2024",
]
population_column = next(
    (col for col in population_column_candidates if col in population.columns),
    None,
)
if population_column is None:
    raise KeyError(
        "Population column not found. Expected one of: "
        + ", ".join(population_column_candidates)
    )
avg_population = pd.to_numeric(population[population_column], errors="coerce").mean()

# -----------------------------
# Exposure scoring
# -----------------------------
summary = {
    "flood_zones_count": len(flood_zones),
    "critical_assets_at_risk": len(infra),
    "avg_population_proxy": int(avg_population),
}

df = pd.DataFrame([summary])
df["exposure_level"] = df["avg_population_proxy"].apply(
    lambda x: "HIGH" if x > 500000 else "MEDIUM"
)

# -----------------------------
# Save
# -----------------------------
df.to_csv(OUTPUT, index=False)

print("Flood exposure proxy analysis completed")
