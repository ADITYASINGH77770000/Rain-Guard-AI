import os
import glob
import shutil
import rasterio
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DATA_DIR = os.environ.get(
    "RIVERTHON_AFTER_IMG_DATA_DIR",
    os.path.join(BASE_DIR, "../data/Satellite data/IMG_DATA_AFTER"),
)
OUTPUT_DIR = os.environ.get("RIVERTHON_OUTPUT_DIR", r"D:\Riverthon\output")
MIN_FREE_BYTES = 50 * 1024 * 1024
MAX_SIDE = 2048


def find_band_path(data_dir, band_code):
    matches = sorted(glob.glob(os.path.join(data_dir, f"*_{band_code}_*.jp2")))
    if not matches:
        raise FileNotFoundError(f"No JP2 file found for {band_code} in: {data_dir}")
    return matches[0]


def make_rgb_image(b2_path, b3_path, b4_path, out_path):
    for band_path in (b2_path, b3_path, b4_path):
        if not os.path.exists(band_path):
            raise FileNotFoundError(f"Band file not found: {band_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(b2_path) as src_b2:
        b2 = src_b2.read(1)
    with rasterio.open(b3_path) as src_b3:
        b3 = src_b3.read(1)
    with rasterio.open(b4_path) as src_b4:
        b4 = src_b4.read(1)

    rgb = np.dstack((b4, b3, b2))

    max_value = np.max(rgb)
    if max_value == 0:
        raise ValueError("All pixel values are zero; cannot normalize image.")
    rgb_norm = ((rgb / max_value) * 255).astype(np.uint8)

    img = Image.fromarray(rgb_norm)
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    img.save(out_path, format="JPEG", quality=90, optimize=True)
    print(f"Saved after flood image: {out_path}")


def generate_all_after_images():
    free_bytes = shutil.disk_usage(OUTPUT_DIR).free if os.path.exists(OUTPUT_DIR) else shutil.disk_usage(BASE_DIR).free
    if free_bytes < MIN_FREE_BYTES:
        free_mb = round(free_bytes / (1024 * 1024), 2)
        raise OSError(
            f"Not enough free disk space ({free_mb} MB). "
            f"Please free at least {int(MIN_FREE_BYTES / (1024 * 1024))} MB, then rerun."
        )

    resolution_dirs = sorted(glob.glob(os.path.join(IMG_DATA_DIR, "R*m")))
    if not resolution_dirs:
        raise FileNotFoundError(f"No resolution folders found in: {IMG_DATA_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for resolution_dir in resolution_dirs:
        resolution_name = os.path.basename(resolution_dir).lower()
        b2_path = find_band_path(resolution_dir, "B02")
        b3_path = find_band_path(resolution_dir, "B03")
        b4_path = find_band_path(resolution_dir, "B04")
        out_path = os.path.join(OUTPUT_DIR, f"after_flood_{resolution_name}.jpg")
        make_rgb_image(b2_path, b3_path, b4_path, out_path)


if __name__ == "__main__":
    generate_all_after_images()
