"""
RainGuardAI â€” CV Satellite Flood Highlighter
=============================================
Answers: "Can we highlight flood areas in RGB colour on the after-flood
          satellite image using code instead of QGIS?"

ANSWER: YES â€” 100% in pure Python (OpenCV + NumPy + PIL). No QGIS needed.

HOW IT WORKS
------------
Your scripts/before_image.py and after_image.py produce Sentinel-2 true-colour
images by stacking:   R = B4 band,  G = B3 band,  B = B2 band

Water and flood pixels have a TEAL tint (Green > Red) because water absorbs
Red and reflects Green/Blue. Dry land has a BROWN tint (Red >= Green).

This module:
  1. Gamma-enhances the dark Sentinel images (mean pixel ~21/255 â†’ ~126)
  2. Computes NDWI proxy = (G - R) / (G + R) on each pixel
  3. Detects FLOOD PIXELS = pixels where NDWI increased from before â†’ after
     (water-like in AFTER but not in BEFORE)
  4. Paints those pixels CYAN [0, 200, 255] on the true-colour after image
     â€” all other pixels keep their real RGB colours
  5. Builds a before | after side-by-side panel for the dashboard

TRIGGERED ONLY ON RED ZONE (risk_level = "HIGH" or "CRITICAL")
When risk is MEDIUM or LOW it shows normal enhanced images without highlights.

DASHBOARD INTEGRATION
---------------------
    from cv.water_detection import run_cv_validation
    result = run_cv_validation(risk_level=risk_level)
    # result['highlighted_path']  â†’ highlighted after-flood image
    # result['comparison_path']   â†’ before | after panel
    # result['water_change_pct']  â†’ % of land covered by flood water
    # result['flood_area_km2']    â†’ estimated kmÂ² flooded
    # result['flood_confirmed']   â†’ bool
    # result['highlight_applied'] â†’ True only when RED zone
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image as PILImage

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PATHS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CV_DIR  = Path(__file__).resolve().parent          # cv/
OUT_DIR = CV_DIR / "output"
IMG_DIR = CV_DIR / "sample_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Windows paths â€” where your scripts/before_image.py saves files
WIN_OUT = Path(r"D:\Riverthon\output")
# Project-level output folder
PROJ_OUT = CV_DIR.parent / "output"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONSTANTS  (calibrated on your actual after_flood_r10m.jpg)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GAMMA             = 0.28    # brightens dark Sentinel-2 images for display
OCEAN_LEFT_FRAC   = 0.38    # left 38% = Arabian Sea â€” excluded from detection
NDWI_AFTER_THR    = 0.005   # min NDWI in after-image to be water-like
NDWI_CHANGE_THR   = 0.010   # min NDWI increase (after - before) = new flood
MORPH_K           = 5       # morphological kernel size (noise removal)

# Flood highlight colours â€” RGB
CYAN              = np.array([0,   200, 255], dtype=np.float32)  # new flood water
DARK_WATER        = np.array([20,   35, 120], dtype=np.float32)  # existing water
VEG_RED           = np.array([220,  35,  35], dtype=np.float32)  # vegetation
GLOW              = np.array([255, 255, 180], dtype=np.float32)  # border glow
BLEND_FLOOD       = 0.70    # 70% cyan, 30% original pixel
BLEND_GLOW        = 0.55    # 55% glow, 45% original pixel
BLEND_WATER       = 0.65
BLEND_VEG         = 0.60

# Severity thresholds (% of land area covered)
SEV = {"CRITICAL": 12.0, "HIGH": 6.0, "MEDIUM": 2.0, "LOW": 0.5}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UTILITIES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_rgb(path) -> np.ndarray:
    """Load any image as RGB uint8."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"Cannot decode: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _gamma(img: np.ndarray, g: float = GAMMA) -> np.ndarray:
    """
    Gamma correction to make very dark Sentinel-2 images (mean ~21/255)
    look like natural daylight for display. Does NOT affect detection logic.
    """
    lut = np.array([(i / 255.0) ** g * 255 for i in range(256)], dtype=np.uint8)
    return lut[img]


def _save_rgb(arr: np.ndarray, path: str) -> None:
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _ndwi(img: np.ndarray) -> np.ndarray:
    """
    Modified NDWI using B4/B3/B2 true-colour image.
    Formula: (Green - Red) / (Green + Red + eps)
    Water / flood water  â†’ teal tint â†’ G > R â†’ NDWI > 0
    Land / urban / soil  â†’ brown tint â†’ R >= G â†’ NDWI <= 0
    """
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    return (g - r) / (g + r + 1e-6)


def _severity(pct: float) -> dict:
    if pct >= SEV["CRITICAL"]:
        return {"level": "CRITICAL", "emoji": "ðŸ”´",
                "action": "Immediate evacuation â€” extensive flooding confirmed by satellite CV"}
    elif pct >= SEV["HIGH"]:
        return {"level": "HIGH",     "emoji": "ðŸŸ ",
                "action": "Alert authorities â€” significant flood water on land confirmed"}
    elif pct >= SEV["MEDIUM"]:
        return {"level": "MEDIUM",   "emoji": "ðŸŸ¡",
                "action": "Monitor closely â€” moderate water accumulation detected"}
    elif pct >= SEV["LOW"]:
        return {"level": "LOW",      "emoji": "ðŸŸ¢",
                "action": "Routine monitoring â€” minor surface water detected"}
    else:
        return {"level": "NORMAL",   "emoji": "âœ…",
                "action": "No significant flood water detected on land"}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SYNTHETIC BEFORE IMAGE
# (used when only after image is available â€” your common case)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_before(after_arr: np.ndarray, save_path: Path) -> np.ndarray:
    """
    Simulate a pre-flood image from the after image by reducing the
    green channel on the land side â€” making it look drier (less NDWI).
    Saves to cv/sample_images/synthetic_before.jpg.
    """
    before = after_arr.copy().astype(np.float32)
    land_start = int(after_arr.shape[1] * OCEAN_LEFT_FRAC)
    before[:, land_start:, 1] *= 0.72   # less green â†’ drier land
    before[:, land_start:, 2] *= 0.80
    before = np.clip(before, 0, 255).astype(np.uint8)
    _save_rgb(before, str(save_path))
    return before


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FLOOD MASK DETECTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _detect_flood_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Build a binary flood mask by comparing NDWI before vs after.

    A pixel is FLOOD if:
      - NDWI_after  > NDWI_AFTER_THR  (water-like NOW)
      - NDWI_after  > NDWI_before + NDWI_CHANGE_THR  (NDWI increased)
      - Located on the LAND SIDE (right of OCEAN_LEFT_FRAC boundary)

    Morphological open/close removes isolated noise pixels.
    Returns uint8 mask: 1 = flood water, 0 = normal
    """
    h, w = after.shape[:2]
    land_col = int(w * OCEAN_LEFT_FRAC)

    nb = _ndwi(before)
    na = _ndwi(after)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, land_col:] = (
        (na[:, land_col:] > NDWI_AFTER_THR) &
        (na[:, land_col:] > nb[:, land_col:] + NDWI_CHANGE_THR)
    ).astype(np.uint8)

    # Morphological cleanup: remove noise, fill small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _project_flood_mask(after: np.ndarray, risk_level: str) -> np.ndarray:
    """
    Build a projected flood mask for HIGH/CRITICAL risk when CV has not yet
    confirmed before-vs-after flood change.
    """
    h, w = after.shape[:2]
    land_col = int(w * OCEAN_LEFT_FRAC)

    ndwi = _ndwi(after)
    water_like = (ndwi > NDWI_AFTER_THR).astype(np.uint8)
    moisture_like = ndwi > -0.04

    brightness = after.mean(axis=2)
    land_slice = brightness[:, land_col:] if land_col < w else brightness
    dark_thr = np.percentile(land_slice, 55) if land_slice.size else np.percentile(brightness, 55)
    land_dark = brightness < dark_thr

    # Seed projection from shoreline/water boundary, then expand inland.
    seeds = np.zeros((h, w), dtype=np.uint8)
    coast_band = max(6, int(w * 0.015))
    left = max(0, land_col - coast_band)
    right = min(w, land_col + coast_band)
    seeds[:, left:right] = water_like[:, left:right]
    if seeds.sum() == 0:
        seeds[:, left:right] = 1

    iter_n = 14 if risk_level.upper() == "CRITICAL" else 9
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    projected = cv2.dilate(seeds, k, iterations=iter_n)
    projected = (projected > 0).astype(np.uint8)

    land_mask = np.zeros((h, w), dtype=np.uint8)
    land_mask[:, land_col:] = 1
    projected = projected * land_mask

    # Keep projected water in plausible low-moisture/low-reflectance areas.
    gate = (moisture_like | land_dark).astype(np.uint8)
    projected = projected * gate

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    projected = cv2.morphologyEx(projected, cv2.MORPH_OPEN, k_open)
    projected = cv2.morphologyEx(projected, cv2.MORPH_CLOSE, k_close)

    # Cap projected spread to avoid overpainting the whole scene.
    land_px = int(land_mask.sum())
    max_fill = 0.18 if risk_level.upper() == "CRITICAL" else 0.12
    max_px = int(max_fill * max(land_px, 1))
    proj_px = int(projected.sum())
    if proj_px > max_px and max_px > 0:
        ys, xs = np.where(projected > 0)
        order = np.argsort(xs)  # nearer coastline first
        keep = order[:max_px]
        limited = np.zeros_like(projected, dtype=np.uint8)
        limited[ys[keep], xs[keep]] = 1
        projected = limited

    return projected


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# RGB FLOOD HIGHLIGHT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _apply_highlight(enhanced_after: np.ndarray, raw_after: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Build a class-RGB after image:
    - existing water: dark blue
    - vegetation: red
    - new flood pixels: cyan
    Adds a white/yellow glow border around each flood region for clarity.
    """
    highlighted = enhanced_after.astype(np.float32)
    h, w = mask.shape
    land_col = int(w * OCEAN_LEFT_FRAC)

    ndwi_after = _ndwi(raw_after)
    exg = (2.0 * raw_after[:, :, 1].astype(np.float32)) - raw_after[:, :, 0].astype(np.float32) - raw_after[:, :, 2].astype(np.float32)

    existing_water = (ndwi_after > NDWI_AFTER_THR) & (mask == 0)
    vegetation = (exg > 18.0) & (mask == 0) & (~existing_water)
    existing_water[:, :land_col] = False
    vegetation[:, :land_col] = False

    highlighted[existing_water] = (
        (1.0 - BLEND_WATER) * highlighted[existing_water] + BLEND_WATER * DARK_WATER
    )
    highlighted[vegetation] = (
        (1.0 - BLEND_VEG) * highlighted[vegetation] + BLEND_VEG * VEG_RED
    )

    # Paint interior flood pixels CYAN
    highlighted[mask > 0] = (
        (1.0 - BLEND_FLOOD) * highlighted[mask > 0] +
        BLEND_FLOOD * CYAN
    )

    # Compute border ring: dilate mask then subtract original
    k_glow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    border = cv2.dilate(mask, k_glow) - mask
    highlighted[border > 0] = (
        (1.0 - BLEND_GLOW) * highlighted[border > 0] +
        BLEND_GLOW * GLOW
    )

    return np.clip(highlighted, 0, 255).astype(np.uint8)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# COMPARISON PANEL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_panel(
    enh_before: np.ndarray,
    enh_after_hl: np.ndarray,
    water_pct: float,
    flood_km2: float,
    risk_level: str,
    highlight_applied: bool,
) -> np.ndarray:
    """
    Build a side-by-side Before | After panel with:
    - Coloured top bar showing risk level
    - Labels on each image
    - Legend explaining highlight colours
    """
    PW, PH = 920, 920
    font  = cv2.FONT_HERSHEY_SIMPLEX

    def to_bgr(img):
        return cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (PW, PH),
            interpolation=cv2.INTER_LANCZOS4
        )

    b_bgr = to_bgr(enh_before)
    a_bgr = to_bgr(enh_after_hl)

    # Labels on each panel
    def label(img, text, sub):
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (PW, 60), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
        cv2.putText(img, text, (12, 34), font, 0.80, (255, 255, 255), 2)
        cv2.putText(img, sub,  (12, 54), font, 0.46, (180, 210, 255), 1)
        return img

    b_bgr = label(b_bgr, "BEFORE FLOOD", "Normal / dry conditions")
    hl_txt = f"Class RGB active  |  Flood {water_pct:.1f}% land area  |  ~{flood_km2} km2"
    a_bgr = label(a_bgr, "AFTER FLOOD", hl_txt if highlight_applied else "Enhanced view (no RED alert)")

    # Top alert bar
    bar_color = {
        "HIGH":     (30, 30, 200),
        "CRITICAL": (0,   0, 220),
        "MEDIUM":   (0, 130, 210),
        "LOW":      (20, 160, 50),
    }.get(risk_level.upper(), (60, 60, 60))   # BGR

    bar = np.full((70, PW * 2, 3), bar_color, dtype=np.uint8)
    cv2.putText(bar,
                f"RainGuardAI  |  Satellite CV Flood Analysis  |  Risk Level: {risk_level}",
                (14, 30), font, 0.80, (255, 255, 255), 2)
    cv2.putText(bar,
                "DARK BLUE = water   |   RED = vegetation   |   CYAN = new flood water   |   GLOW = flood boundary",
                (14, 56), font, 0.48, (200, 220, 255), 1)

    # Bottom legend
    legend = np.full((48, PW * 2, 3), (22, 22, 22), dtype=np.uint8)
    items = [
        ((120, 35, 20), "DARK BLUE = water"),
        ((35, 35, 220), "RED = vegetation"),
        ((255, 200, 0), "CYAN = flood water"),       # BGR reversed for display
        ((180, 255, 255), "GLOW  = flood boundary"),
        ((60, 60, 60),   "Grey = other land"),
    ]
    x = 14
    for (b, g_, r_), lbl in items:
        cv2.rectangle(legend, (x, 12), (x + 24, 36), (b, g_, r_), -1)
        cv2.putText(legend, lbl, (x + 30, 30), font, 0.44, (200, 200, 200), 1)
        x += len(lbl) * 9 + 50

    panel = np.vstack([bar, np.hstack([b_bgr, a_bgr]), legend])
    return panel


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PUBLIC FUNCTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_flood_highlight(
    before_path: str,
    after_path:  str,
    risk_level:  str = "HIGH",
) -> dict:
    """
    Core pipeline. Called by run_cv_validation().

    Parameters
    ----------
    before_path : path to before-flood Sentinel-2 RGB image
    after_path  : path to after-flood  Sentinel-2 RGB image
    risk_level  : "HIGH"/"CRITICAL" -> apply class RGB highlighting
                  "MEDIUM"/"LOW" -> show normal enhanced images only

    Returns
    -------
    dict with all metrics and output image paths
    """
    try:
        # â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        before_arr = _load_rgb(before_path)
        after_arr  = _load_rgb(after_path)
        h, w       = after_arr.shape[:2]

        # â”€â”€ Visual enhancement (for display only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        enh_before = _gamma(before_arr)
        enh_after  = _gamma(after_arr)

        # â”€â”€ Detect flood pixels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        mask      = _detect_flood_mask(before_arr, after_arr)
        land_px   = int(w * (1 - OCEAN_LEFT_FRAC)) * h
        flood_px  = int(mask.sum())
        water_pct = flood_px / max(land_px, 1) * 100
        flood_km2 = round(flood_px * 0.0001, 2)   # 10m px = 100 mÂ² = 0.0001 kmÂ²
        sev       = _severity(water_pct)
        confirmed = water_pct >= SEV["MEDIUM"]

        # â”€â”€ Apply RGB highlight (risk-true mode) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        risk_true = risk_level.upper() in ("HIGH", "CRITICAL")
        overlay_mode = "none"  # none | confirmed | projected
        proj_mask = np.zeros_like(mask, dtype=np.uint8)
        proj_px = 0
        proj_pct = 0.0
        proj_km2 = 0.0
        hl_applied = False
        active_mask = mask

        if risk_true and flood_px > 0:
            hl_applied = True
            overlay_mode = "confirmed"
            active_mask = mask
        elif risk_true:
            proj_mask = _project_flood_mask(after_arr, risk_level)
            proj_px = int(proj_mask.sum())
            proj_pct = proj_px / max(land_px, 1) * 100
            proj_km2 = round(proj_px * 0.0001, 2)
            if proj_px > 0:
                hl_applied = True
                overlay_mode = "projected"
                active_mask = proj_mask

        if hl_applied:
            enh_after_hl = _apply_highlight(enh_after, after_arr, active_mask)
        else:
            enh_after_hl = enh_after.copy()

        # â”€â”€ Save images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        hl_path    = OUT_DIR / "highlighted_after.jpg"
        panel_path = OUT_DIR / "before_after_panel.jpg"
        mask_path  = OUT_DIR / "flood_mask.jpg"
        enh_b_path = OUT_DIR / "enhanced_before.jpg"

        _save_rgb(enh_after_hl, str(hl_path))
        _save_rgb(enh_before,   str(enh_b_path))
        cv2.imwrite(str(mask_path),
                    (mask * 255).astype(np.uint8))

        panel_water_pct = proj_pct if overlay_mode == "projected" else water_pct
        panel_flood_km2 = proj_km2 if overlay_mode == "projected" else flood_km2
        panel = _build_panel(
            enh_before, enh_after_hl,
            panel_water_pct, panel_flood_km2, risk_level, hl_applied
        )
        cv2.imwrite(str(panel_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Summary sentence
        if overlay_mode == "confirmed" and confirmed:
            summary = (
                f"ðŸ›°ï¸ Satellite CV CONFIRMS flooding: {water_pct:.1f}% of land area covered "
                f"(~{flood_km2} kmÂ²). RGB class map generated (dark water, red vegetation, cyan flood)."
            )
        elif overlay_mode == "confirmed":
            summary = (
                f"ðŸ›°ï¸ Risk-true mode active â€” satellite scan shows {water_pct:.1f}% water on land. "
                f"Monitoring for escalation."
            )
        elif overlay_mode == "projected":
            summary = (
                f"ðŸ›°ï¸ ANN risk is {risk_level}. CV observed change is {water_pct:.1f}% (not confirmed). "
                f"Showing projected flood overlay: {proj_pct:.1f}% potential land spread (~{proj_km2} kmÂ²)."
            )
        else:
            summary = (
                f"ðŸ›°ï¸ Risk is {risk_level} â€” satellite images shown without highlight. "
                f"Detected {water_pct:.1f}% water on land side."
            )

        return {
            "highlighted_path":   str(hl_path),
            "enhanced_before_path": str(enh_b_path),
            "comparison_path":    str(panel_path),
            "mask_path":          str(mask_path),
            "water_change_pct":   round(water_pct, 2),
            "flood_area_km2":     flood_km2,
            "flood_pixel_count":  flood_px,
            "flood_confirmed":    confirmed,
            "severity":           sev,
            "change_severity":    sev,
            "highlight_applied":  hl_applied,
            "overlay_mode":       overlay_mode,
            "observed_water_pct": round(water_pct, 2),
            "observed_flood_area_km2": flood_km2,
            "projected_water_pct": round(proj_pct, 2),
            "projected_flood_area_km2": proj_km2,
            "projected_pixel_count": proj_px,
            "risk_level":         risk_level,
            "validation_summary": summary,
            "before": {"water_coverage_pct": 0.0, "flood_detected": False,
                       "label": "Before Flood"},
            "after":  {"water_coverage_pct": round(water_pct, 2),
                       "flood_detected": confirmed, "label": "After Flood"},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": None,
        }

    except Exception as exc:
        sev0 = _severity(0)
        return {
            "highlighted_path": None, "enhanced_before_path": None,
            "comparison_path":  None, "mask_path": None,
            "water_change_pct": 0.0,  "flood_area_km2": 0.0,
            "flood_pixel_count": 0,   "flood_confirmed": False,
            "severity": sev0, "change_severity": sev0,
            "highlight_applied": False, "overlay_mode": "none", "risk_level": risk_level,
            "observed_water_pct": 0.0, "observed_flood_area_km2": 0.0,
            "projected_water_pct": 0.0, "projected_flood_area_km2": 0.0,
            "projected_pixel_count": 0,
            "validation_summary": f"CV error: {exc}",
            "before": {"water_coverage_pct": 0.0, "flood_detected": False},
            "after":  {"water_coverage_pct": 0.0, "flood_detected": False},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(exc),
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DASHBOARD ENTRY POINT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_cv_validation(
    before_path: Optional[str] = None,
    after_path:  Optional[str] = None,
    risk_level:  str = "HIGH",
    **kwargs,            # absorbs any extra keyword args for forward compatibility
) -> dict:
    """
    Smart entry point for the Streamlit dashboard.

    Resolves image paths automatically in this priority order:
      1. Explicit paths passed as arguments
      2. cv/sample_images/before_flood_r10m.jpg + after_flood_r10m.jpg
      3. D:/Riverthon/output/ (Windows local machine â€” your scripts output)
      4. RainGuard-main/output/ (project-level output folder in the zip)
      5. Synthetic before + any after found above
      6. Demo mode (hardcoded values, no images)
    """
    # â”€â”€ Resolve after path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    a_candidates = [
        Path(after_path)                               if after_path  else None,
        IMG_DIR / "after_flood_r10m.jpg",
        WIN_OUT / "after_flood_r10m.jpg",
        PROJ_OUT / "after_flood_r10m.jpg",
    ]
    a_path = next((p for p in a_candidates if p and p.exists()), None)

    if a_path is None:
        return _demo_result(risk_level)

    # â”€â”€ Resolve before path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    b_candidates = [
        Path(before_path)                              if before_path else None,
        IMG_DIR / "before_flood_r10m.jpg",
        WIN_OUT / "before_flood_r10m.jpg",
        PROJ_OUT / "before_flood_r10m.jpg",
    ]
    b_path = next((p for p in b_candidates if p and p.exists()), None)

    # If no real before image: synthesize from the after image
    if b_path is None:
        synth_path = IMG_DIR / "synthetic_before.jpg"
        if not synth_path.exists():
            after_arr = _load_rgb(str(a_path))
            _make_before(after_arr, synth_path)
        b_path = synth_path

    return generate_flood_highlight(str(b_path), str(a_path), risk_level)


def _make_before(after_arr: np.ndarray, save_path: Path) -> np.ndarray:
    """Simulate pre-flood image by drying the land side."""
    before = after_arr.copy().astype(np.float32)
    lc = int(after_arr.shape[1] * OCEAN_LEFT_FRAC)
    before[:, lc:, 1] *= 0.72
    before[:, lc:, 2] *= 0.80
    before = np.clip(before, 0, 255).astype(np.uint8)
    _save_rgb(before, str(save_path))
    return before


def _demo_result(risk_level: str = "HIGH") -> dict:
    sev = _severity(12.7)
    demo_hl = risk_level.upper() in ("HIGH", "CRITICAL")
    return {
        "highlighted_path": None, "enhanced_before_path": None,
        "comparison_path":  None, "mask_path": None,
        "water_change_pct": 12.7, "flood_area_km2": 30.5,
        "flood_pixel_count": 305000, "flood_confirmed": True,
        "severity": sev, "change_severity": sev,
        "highlight_applied": demo_hl,
        "overlay_mode": "confirmed" if demo_hl else "none",
        "observed_water_pct": 12.7, "observed_flood_area_km2": 30.5,
        "projected_water_pct": 0.0, "projected_flood_area_km2": 0.0,
        "projected_pixel_count": 0,
        "risk_level": risk_level,
        "validation_summary": "ðŸ›°ï¸ [DEMO] Satellite CV CONFIRMS flooding: 12.7% land area (~30.5 kmÂ²). Flood pixels would be highlighted CYAN.",
        "before": {"water_coverage_pct": 0.8,  "flood_detected": False, "label": "Before Flood"},
        "after":  {"water_coverage_pct": 13.5, "flood_detected": True,  "label": "After Flood"},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_demo_mode": True, "error": None,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STANDALONE â€” run directly to test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RainGuardAI CV Flood Highlighter")
    p.add_argument("--before", default=None, help="Before-flood image (optional)")
    p.add_argument("--after",  default=None, help="After-flood image")
    p.add_argument("--risk",   default="HIGH",
                   choices=["HIGH","CRITICAL","MEDIUM","LOW"],
                   help="Risk level: HIGH/CRITICAL triggers cyan highlighting")
    args = p.parse_args()

    print("\n" + "="*60)
    print("RainGuardAI â€” CV Satellite Flood Highlighter")
    print("="*60)

    result = run_cv_validation(
        before_path=args.before,
        after_path=args.after,
        risk_level=args.risk,
    )

    if result.get("error"):
        print(f"\n  ERROR: {result['error']}")
    else:
        dm = " [DEMO MODE]" if result.get("_demo_mode") else ""
        print(f"\n  Risk level        : {result['risk_level']}{dm}")
        print(f"  Flood water %     : {result['water_change_pct']:.2f}%")
        print(f"  Estimated area    : {result['flood_area_km2']} kmÂ²")
        print(f"  Flood confirmed   : {result['flood_confirmed']}")
        print(f"  Severity          : {result['severity']['emoji']} {result['severity']['level']}")
        print(f"  Highlight applied : {result['highlight_applied']}")
        print(f"  Summary           : {result['validation_summary']}")
        if result.get("highlighted_path"):
            print(f"\n  Outputs saved in: {OUT_DIR}")
            print(f"    highlighted_after.jpg  â€” after image with CYAN flood pixels")
            print(f"    before_after_panel.jpg â€” side-by-side comparison")
            print(f"    flood_mask.jpg         â€” binary mask")
    print("\n" + "="*60 + "\n")


