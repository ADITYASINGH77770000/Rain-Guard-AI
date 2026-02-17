import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from pathlib import Path

# =====================================================
# BASE DIRECTORY
# Change ONLY this if project moves
# =====================================================

BASE_DIR = Path("C:/Users/Admin/Desktop/Riverthon/data")

DEM_PATH = BASE_DIR / "raw/terrain/dem_mumbai.tif"
OUTPUT_PATH = BASE_DIR / "processed/low_lying_zones.geojson"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =====================================================
# READ DEM
# =====================================================

print("📥 Reading DEM...")

if not DEM_PATH.exists():
    raise FileNotFoundError(f"DEM file not found at: {DEM_PATH}")

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

print("✅ DEM loaded successfully")

# =====================================================
# HANDLE NODATA + BAD VALUES
# =====================================================

if nodata is not None:
    dem = np.where(dem == nodata, np.nan, dem)

# Remove unrealistic garbage values
dem = np.where(dem < -100, np.nan, dem)

print("Elevation Stats:")
print("Min:", np.nanmin(dem))
print("Max:", np.nanmax(dem))

# =====================================================
# DEFINE LOW-LYING THRESHOLD
# =====================================================

# Using hybrid logic:
# If percentile becomes unrealistic (like 0),
# fallback to scientific flood threshold (5m)

percentile_threshold = np.nanpercentile(dem, 20)

if percentile_threshold <= 1:
    print("⚠️ Percentile too low — likely ocean influence.")
    threshold = 5   # scientifically defendable flood threshold
else:
    threshold = percentile_threshold

print(f"📉 Low-lying threshold elevation: {threshold:.2f} meters")

low_lying_mask = dem <= threshold

# =====================================================
# CONVERT MASK → POLYGONS
# =====================================================

print("🧠 Extracting low-lying polygons...")

polygon_generator = shapes(
    low_lying_mask.astype(np.uint8),
    mask=low_lying_mask,
    transform=transform
)

geoms = [shape(geom) for geom, val in polygon_generator if val == 1]

if len(geoms) == 0:
    raise ValueError("No low-lying zones detected. Check DEM.")

# =====================================================
# CREATE GEODATAFRAME
# =====================================================

gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

print(f"✅ Raw zones detected: {len(gdf)}")

# =====================================================
# FIX AREA CALCULATION (PROJECT TO METERS)
# =====================================================

# Mumbai → UTM Zone 43N
gdf_projected = gdf.to_crs(epsg=32643)

# Calculate TRUE land area
gdf_projected["area_m2"] = gdf_projected.area

# Remove tiny noise polygons (< 5,000 sq meters)
gdf_projected = gdf_projected[gdf_projected["area_m2"] > 5000]

print(f"✅ Zones after noise removal: {len(gdf_projected)}")

# =====================================================
# MERGE TOUCHING POLYGONS (Cleaner Flood Regions)
# =====================================================

gdf_projected = gdf_projected.dissolve()
gdf_projected = gdf_projected.explode(index_parts=False)

print(f"✅ Final merged zones: {len(gdf_projected)}")

# Convert back to original CRS for GeoJSON
gdf_final = gdf_projected.to_crs(crs)

# =====================================================
# SAVE FILE
# =====================================================

gdf_final.to_file(OUTPUT_PATH, driver="GeoJSON")

print("\n🎉 Low-lying zones extracted successfully!")
print(f"📍 Output file: {OUTPUT_PATH}")