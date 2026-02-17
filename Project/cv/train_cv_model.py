import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from water_detection import _compute_ndwi_proxy, _enhance, _load_image

PROJECT_DIR = Path(__file__).resolve().parents[1]
CV_DIR = Path(__file__).resolve().parent
IMG_DIR = CV_DIR / "sample_images"
OUTPUT_DIR = CV_DIR / "output"
CALIB_PATH = CV_DIR / "calibration.json"
RIVERTHON_OUTPUT_DIR = Path(os.environ.get("RIVERTHON_OUTPUT_DIR", str(PROJECT_DIR / "output")))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OCEAN_FRACTION = 0.40


def apply_water_mask(ndwi: np.ndarray, threshold: float, ocean_fraction: float) -> np.ndarray:
    h, w = ndwi.shape
    mask = (ndwi > threshold).astype(np.uint8)
    mask[:, : int(w * ocean_fraction)] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def calibrate_threshold(before_ndwi: np.ndarray, after_ndwi: np.ndarray, ocean_fraction: float = OCEAN_FRACTION) -> float:
    h, w = before_ndwi.shape
    land_px = int(w * (1 - ocean_fraction)) * h
    ocean_cols = int(w * ocean_fraction)

    best_threshold = 0.02
    best_score = -1e9

    for threshold in np.arange(0.00, 0.21, 0.01):
        b_mask = (before_ndwi > threshold).astype(np.uint8)
        a_mask = (after_ndwi > threshold).astype(np.uint8)
        b_mask[:, :ocean_cols] = 0
        a_mask[:, :ocean_cols] = 0

        b_pct = b_mask.sum() / max(land_px, 1) * 100
        a_pct = a_mask.sum() / max(land_px, 1) * 100
        change = a_pct - b_pct

        score = change - (2.0 * b_pct)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return round(best_threshold, 2)


def create_overlay(img_rgb: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = vis.copy()
    water = mask > 0
    overlay[water] = (0.5 * overlay[water] + 0.5 * np.array([255, 80, 0], dtype=np.float32)).astype(np.uint8)
    out = cv2.addWeighted(overlay, 0.85, vis, 0.15, 0)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def find_images(resolution: str = "r10m"):
    before = RIVERTHON_OUTPUT_DIR / f"before_flood_{resolution}.jpg"
    after = RIVERTHON_OUTPUT_DIR / f"after_flood_{resolution}.jpg"

    if before.exists() and after.exists():
        return before, after

    b2 = IMG_DIR / f"before_flood_{resolution}.jpg"
    a2 = IMG_DIR / f"after_flood_{resolution}.jpg"
    if b2.exists() and a2.exists():
        return b2, a2

    return None, None


def train(resolution: str = "r10m") -> dict:
    before_path, after_path = find_images(resolution=resolution)
    if before_path is None or after_path is None:
        raise FileNotFoundError(
            f"Missing images for {resolution}. Expected in {RIVERTHON_OUTPUT_DIR} or {IMG_DIR}"
        )

    before_rgb = _load_image(before_path)
    after_rgb = _load_image(after_path)

    before_ndwi = _compute_ndwi_proxy(_enhance(before_rgb))
    after_ndwi = _compute_ndwi_proxy(_enhance(after_rgb))

    threshold = calibrate_threshold(before_ndwi, after_ndwi)
    before_mask = apply_water_mask(before_ndwi, threshold, OCEAN_FRACTION)
    after_mask = apply_water_mask(after_ndwi, threshold, OCEAN_FRACTION)

    h, w = before_mask.shape
    land_px = int(w * (1 - OCEAN_FRACTION)) * h

    before_pct = float(before_mask.sum()) / max(land_px, 1) * 100
    after_pct = float(after_mask.sum()) / max(land_px, 1) * 100
    change_pct = after_pct - before_pct

    new_flood = int(((after_mask > 0) & (before_mask == 0)).sum())
    flood_area_km2 = round(new_flood * 0.0001, 2)

    cv2.imwrite(str(OUTPUT_DIR / "annotated_before.jpg"), create_overlay(before_rgb, before_mask, "Before Flood"))
    cv2.imwrite(str(OUTPUT_DIR / "annotated_after.jpg"), create_overlay(after_rgb, after_mask, "After Flood"))
    cv2.imwrite(str(OUTPUT_DIR / "water_mask_after.jpg"), (after_mask * 255).astype(np.uint8))

    diff = cv2.absdiff((before_mask * 255).astype(np.uint8), (after_mask * 255).astype(np.uint8))
    cv2.imwrite(str(OUTPUT_DIR / "flood_comparison.jpg"), diff)

    payload = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": resolution,
        "before_image": str(before_path),
        "after_image": str(after_path),
        "optimal_ndwi_threshold": threshold,
        "ocean_exclusion_frac": OCEAN_FRACTION,
        "metrics": {
            "before_water_pct": round(before_pct, 3),
            "after_water_pct": round(after_pct, 3),
            "water_change_pct": round(change_pct, 3),
            "estimated_flood_area_km2": flood_area_km2,
            "flood_confirmed": (change_pct > 1.5 and after_pct > 2.0),
        },
        "outputs": {
            "annotated_before": str(OUTPUT_DIR / "annotated_before.jpg"),
            "annotated_after": str(OUTPUT_DIR / "annotated_after.jpg"),
            "water_mask_after": str(OUTPUT_DIR / "water_mask_after.jpg"),
            "flood_comparison": str(OUTPUT_DIR / "flood_comparison.jpg"),
        },
    }

    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    result = train("r10m")
    print("CV training complete")
    print(f"Before: {result['metrics']['before_water_pct']:.2f}%")
    print(f"After : {result['metrics']['after_water_pct']:.2f}%")
    print(f"Change: {result['metrics']['water_change_pct']:+.2f}%")
    print(f"Saved calibration: {CALIB_PATH}")
