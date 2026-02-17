import geopandas as gpd
import pandas as pd
from pathlib import Path

# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = Path(r"C:\Users\Admin\Desktop\Riverthon\data")

LOW_LYING_PATH = BASE_DIR / "processed/low_lying_zones.geojson"

# ✅ Correct Infrastructure Paths
INFRASTRUCTURE_LAYERS = {
    "Waterways": BASE_DIR / r"raw\hydrology\mah_waterways.gpkg",
    "Fire_Station": BASE_DIR / r"Critical Centers\mah_firestat.gpkg",
    "Healthcare": BASE_DIR / r"Critical Centers\Mah_healthcare.gpkg",
    "Police": BASE_DIR / r"Critical Centers\mah_police.gpkg",
    "Shelter": BASE_DIR / r"Critical Centers\mah_shelter.gpkg"
}

OUTPUT_PATH = BASE_DIR / "processed/flood_impact_zones.geojson"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD LOW-LYING ZONES
# =====================================================

print("📥 Loading flood zones...")

if not LOW_LYING_PATH.exists():
    raise FileNotFoundError(f"Low-lying zones not found: {LOW_LYING_PATH}")

low_lying = gpd.read_file(LOW_LYING_PATH)

# Convert to projected CRS (meters)
low_lying = low_lying.to_crs(epsg=32643)

print(f"✅ Flood zones loaded: {len(low_lying)}")

impact_results = []

# =====================================================
# PROCESS INFRASTRUCTURE
# =====================================================

for infra_type, path in INFRASTRUCTURE_LAYERS.items():

    print(f"\n🔍 Processing {infra_type}...")

    if not path.exists():
        print(f"⚠️ File missing → Skipping: {path}")
        continue

    infra = gpd.read_file(path)

    if infra.empty:
        print("⚠️ Empty file — skipping.")
        continue

    # Fix invalid geometries (VERY COMMON)
    infra["geometry"] = infra.buffer(0)

    # Match CRS
    infra = infra.to_crs(low_lying.crs)

    # -----------------------------
    # FAST spatial intersection
    # -----------------------------
    try:
        intersected = gpd.overlay(infra, low_lying, how="intersection")
    except Exception:
        print("⚠️ Geometry issue detected — attempting repair...")
        infra = infra.explode(index_parts=False)
        intersected = gpd.overlay(infra, low_lying, how="intersection")

    if not intersected.empty:
        intersected["infrastructure_type"] = infra_type
        impact_results.append(intersected)

        print(f"✅ Impacted {infra_type}: {len(intersected)}")
    else:
        print("No intersections found.")

# =====================================================
# MERGE RESULTS
# =====================================================

if impact_results:

    final_gdf = gpd.GeoDataFrame(
        pd.concat(impact_results, ignore_index=True),
        crs=low_lying.crs
    )

    final_gdf = final_gdf.explode(index_parts=False)

    # Convert back to lat/long for GeoJSON compatibility
    final_gdf = final_gdf.to_crs(epsg=4326)

    final_gdf.to_file(OUTPUT_PATH, driver="GeoJSON")

    print("\n🎉 Flood impact zones created successfully!")
    print(f"📍 Output saved at: {OUTPUT_PATH}")

else:
    print("\n⚠️ No infrastructure intersects flood zones.")