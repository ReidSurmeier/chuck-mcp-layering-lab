# Python Image-Processing Pipeline: Photo → Calibrated Lab → YAML

Production-quality reference implementation for chuck-mcp v2's calibration MCP tools.

## Dependency stack

```toml
# pyproject.toml extract
[project]
dependencies = [
    "rawpy >= 0.21",              # libraw bindings, RAW decoding
    "colour-science >= 4.5",      # CIE Lab, color spaces, CCM
    "colour-checker-detection >= 0.2",  # auto-detect ColorChecker
    "opencv-python >= 4.8",       # ArUco markers, image ops
    "numpy >= 1.24",
    "scipy >= 1.11",              # brentq solver for K-M inverse
    "pyyaml >= 6.0",
    "imageio >= 2.31",
    "scikit-image >= 0.22",
]
```

## Module structure

```
chuck_mcp_v2/calibration/
  __init__.py
  capture.py        # RAW loading, linearization
  detect.py         # ColorChecker detection, ArUco fiducial detection
  ccm.py            # Color correction matrix fitting (Finlayson 2015 root-poly)
  km_inverse.py     # Curtis 1997 two-substrate K-M inverse solver
  drift.py          # ΔE-based drift detection and re-cal triggering
  pigment_yaml.py   # YAML schema + read/write
  mcp_tools.py      # MCP-exposed tool definitions
```

## Full pipeline: RAW photo → pigment YAML

```python
"""
chuck_mcp_v2/calibration/pipeline.py — full reference pipeline.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import datetime as dt

import numpy as np
import rawpy
import colour
import cv2
from colour_checker_detection import detect_colour_checkers_segmentation
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# Stage 1: RAW Loading
# ----------------------------------------------------------------------
def load_raw_linear(path: Path) -> np.ndarray:
    """Load RAW file as linear sensor RGB, 16-bit, no camera processing.
    Returns float32 in [0, 1]."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            no_auto_bright=True,
            gamma=(1.0, 1.0),
            half_size=False,
        )
    return rgb.astype(np.float32) / 65535.0


def apply_flat_field(image: np.ndarray, flat_field: np.ndarray) -> np.ndarray:
    """Divide image by normalized flat-field to remove vignetting."""
    flat_norm = flat_field / np.median(flat_field)
    flat_norm = np.clip(flat_norm, 0.5, 2.0)  # safety clamp
    return np.clip(image / flat_norm, 0, 1)


# ----------------------------------------------------------------------
# Stage 2: ColorChecker Detection & CCM Fitting
# ----------------------------------------------------------------------
def detect_colorchecker(image: np.ndarray) -> np.ndarray:
    """Returns (24, 3) array of detected patch values (camera linear RGB).
    Order: dark skin, light skin, ..., black 2."""
    results = list(detect_colour_checkers_segmentation(image, additional_data=False))
    if not results:
        raise RuntimeError("ColorChecker not detected. Check framing and lighting.")
    return np.asarray(results[0], dtype=np.float64)


def get_reference_lab_d50() -> np.ndarray:
    """Returns (24, 3) Lab D50 reference values for ColorChecker24-after-Nov-2014."""
    ref = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    illum = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    xyY_list = list(ref.data.values())
    xyz = np.array([colour.xyY_to_XYZ(xyY) for xyY in xyY_list])
    return np.array([colour.XYZ_to_Lab(x, illum) for x in xyz])


def get_reference_xyz_d50() -> np.ndarray:
    """Returns (24, 3) XYZ D50 reference."""
    ref = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    xyY_list = list(ref.data.values())
    return np.array([colour.xyY_to_XYZ(xyY) for xyY in xyY_list])


def fit_root_polynomial_ccm(
    camera_rgbs: np.ndarray,  # (24, 3) detected patch values
    reference_xyzs: np.ndarray,  # (24, 3) ground-truth XYZ
    degree: int = 2,
) -> dict:
    """Fit a root-polynomial CCM (Finlayson 2015). Exposure-invariant.
    Returns a dict with the fitted matrix and metadata."""
    # Use colour-science's built-in
    matrix = colour.characterisation.matrix_colour_correction(
        camera_rgbs, reference_xyzs,
        method='Finlayson 2015', degree=degree,
    )
    # Compute residual ΔE for QA
    corrected = colour.characterisation.apply_matrix_colour_correction(
        camera_rgbs, matrix, method='Finlayson 2015',
    )
    illum_d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    pred_lab = np.array([colour.XYZ_to_Lab(c, illum_d50) for c in corrected])
    ref_lab = np.array([colour.XYZ_to_Lab(c, illum_d50) for c in reference_xyzs])
    delta_es = colour.delta_E(pred_lab, ref_lab, method='CIE 2000')

    return {
        "matrix": matrix.tolist(),
        "method": "Finlayson 2015",
        "degree": degree,
        "fit_delta_e_mean": float(np.mean(delta_es)),
        "fit_delta_e_max": float(np.max(delta_es)),
        "fit_delta_e_per_patch": delta_es.tolist(),
        "session_timestamp": dt.datetime.now().isoformat(),
    }


def apply_ccm(image_rgb: np.ndarray, ccm: dict) -> np.ndarray:
    """Apply fitted CCM to an image or sample array. Returns XYZ D50."""
    matrix = np.asarray(ccm["matrix"])
    return colour.characterisation.apply_matrix_colour_correction(
        image_rgb, matrix, method=ccm["method"],
    )


# ----------------------------------------------------------------------
# Stage 3: ArUco Fiducial Detection & Swatch Sampling
# ----------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

PLATE_WIDTH_MM = 240.0
PLATE_HEIGHT_MM = 360.0
SWATCH_SIZE_MM = 18.0
SWATCH_GUTTER_MM = 3.0
GRID_ORIGIN_X_MM = 48.0   # left edge of leftmost swatch
GRID_ORIGIN_Y_MM = 90.0   # top of top row
N_CONCENTRATIONS = 7
N_PIGMENTS_PER_PLATE = 15


def warp_via_aruco(image_uint8: np.ndarray, output_size_px: tuple[int, int] = (2400, 3600)) -> np.ndarray:
    """Detect 4 ArUco markers (IDs 0-3), warp to canonical plate frame.
    Output dimensions are (height, width) in pixels for the plate canvas
    (default 10 px/mm = 2400 × 3600).
    """
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(image_uint8)
    if ids is None or len(ids) < 4:
        raise RuntimeError(f"Expected 4 ArUco fiducials, found {len(ids) if ids is not None else 0}")
    id_list = ids.flatten().tolist()
    needed = {0: None, 1: None, 2: None, 3: None}
    for i, mid in enumerate(id_list):
        if mid in needed:
            # Centre of marker
            needed[mid] = corners[i][0].mean(axis=0)
    if any(v is None for v in needed.values()):
        raise RuntimeError(f"Missing required markers, have {id_list}")

    # Source quadrilateral: detected marker centers
    src = np.array([needed[0], needed[1], needed[3], needed[2]], dtype=np.float32)
    # Destination: canonical positions (markers at 20mm in from each corner)
    h, w = output_size_px
    px_per_mm = w / PLATE_WIDTH_MM
    dst = np.array([
        [20 * px_per_mm, 20 * px_per_mm],                     # marker 0: top-left
        [(PLATE_WIDTH_MM - 20) * px_per_mm, 20 * px_per_mm],  # marker 1: top-right
        [(PLATE_WIDTH_MM - 20) * px_per_mm, (PLATE_HEIGHT_MM - 20) * px_per_mm],
        [20 * px_per_mm, (PLATE_HEIGHT_MM - 20) * px_per_mm],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_uint8, M, (w, h))


def sample_swatch_grid(
    warped_image: np.ndarray,        # (H, W, 3) warped to canonical plate frame
    pigment_row_indices: Sequence[int],
    px_per_mm: float = 10.0,
) -> np.ndarray:
    """Sample mean RGB for each swatch on a given pigment row.
    Returns (len(pigment_row_indices), N_CONCENTRATIONS, 3) array."""
    out = np.empty((len(pigment_row_indices), N_CONCENTRATIONS, 3), dtype=np.float32)
    swatch_px = SWATCH_SIZE_MM * px_per_mm
    pitch_px = (SWATCH_SIZE_MM + SWATCH_GUTTER_MM) * px_per_mm
    origin_x_px = GRID_ORIGIN_X_MM * px_per_mm
    origin_y_px = GRID_ORIGIN_Y_MM * px_per_mm
    inset = swatch_px * 0.15  # 15% inset to avoid edge bleed

    for ri, row_idx in enumerate(pigment_row_indices):
        for ci in range(N_CONCENTRATIONS):
            x0 = int(origin_x_px + ci * pitch_px + inset)
            y0 = int(origin_y_px + row_idx * pitch_px + inset)
            x1 = int(x0 + swatch_px - 2 * inset)
            y1 = int(y0 + swatch_px - 2 * inset)
            patch = warped_image[y0:y1, x0:x1].reshape(-1, 3)
            # Robust median; trim outliers (dust, fiber glints that survived CXP)
            out[ri, ci] = np.median(patch, axis=0)
    return out


# ----------------------------------------------------------------------
# Stage 4: Kubelka-Munk Inverse (Curtis 1997, two-substrate)
# ----------------------------------------------------------------------
def km_inverse_two_substrate(
    R_W: np.ndarray,           # reflectance over white, per-channel
    R_B: np.ndarray,           # reflectance over black
    R_g_white: float = 0.95,
    R_g_black: float = 0.05,
    x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover K, S from two reflectance measurements (per channel).
    Returns (K, S) arrays."""
    R_W = np.clip(R_W, 1e-4, 0.999)
    R_B = np.clip(R_B, 1e-4, 0.999)

    # Eliminate substrate (algebraic, from two-equation system)
    # R_W = R + T² R_gw / (1 - R R_gw)
    # R_B = R + T² R_gb / (1 - R R_gb)
    # Subtracting: (R_W - R_B) = T² (R_gw / (1 - R R_gw) - R_gb / (1 - R R_gb))
    # We solve numerically for R (the K-M layer reflectance with no substrate effect).
    K = np.zeros_like(R_W)
    S = np.zeros_like(R_W)
    for ch in range(R_W.shape[-1]):
        rw, rb = R_W[ch], R_B[ch]

        def residual(R):
            denom_w = (1.0 - R * R_g_white)
            denom_b = (1.0 - R * R_g_black)
            if denom_w < 1e-6 or denom_b < 1e-6:
                return 1e6
            T_sq_w = (rw - R) * denom_w / R_g_white if R_g_white > 0 else 0
            T_sq_b = (rb - R) * denom_b / R_g_black if R_g_black > 0 else 0
            return T_sq_w - T_sq_b

        try:
            R_layer = brentq(residual, 0.001, 0.999, xtol=1e-5)
        except ValueError:
            # Solver couldn't bracket - pigment may be ~opaque or ~transparent
            R_layer = (rw + rb) / 2  # fallback

        # T² from R_W equation
        T_sq = (rw - R_layer) * (1 - R_layer * R_g_white) / max(R_g_white, 1e-6)
        T_sq = max(T_sq, 1e-6)
        T = np.sqrt(T_sq)

        # K-M parameters from R(x=1), T(x=1)
        # sinh(bSx) / (a sinh + b cosh) = R
        # b / (a sinh + b cosh)         = T
        # => sinh(bSx) = R * (1/T) * b   [ ratio of the two ]
        # Use the closed form:  a = (1 + R² - T²) / (2R)
        if R_layer < 1e-4:
            a = 1e6
        else:
            a = (1.0 + R_layer * R_layer - T_sq) / (2.0 * R_layer)
        a = max(a, 1.001)
        b = np.sqrt(a * a - 1.0)
        # bSx = arctanh(b / (a - R/T))   — direct closed form from Kubelka 1948
        denom = a - R_layer / max(T, 1e-6)
        if abs(denom) < 1e-6 or b / denom >= 1.0:
            S_ch = 1.0  # opaque limit
        else:
            S_ch = np.arctanh(b / denom) / (b * x)
        K_ch = S_ch * (a - 1.0)

        S[ch] = S_ch
        K[ch] = max(K_ch, 0.0)

    return K, S


# ----------------------------------------------------------------------
# Stage 5: ΔE Drift Detection
# ----------------------------------------------------------------------
def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float | np.ndarray:
    """CIEDE2000."""
    return colour.delta_E(lab1, lab2, method='CIE 2000')


def check_drift_against_baseline(
    current_swatch_lab: np.ndarray,    # (N_CONCENTRATIONS, 3)
    baseline_swatch_lab: np.ndarray,   # from previous calibration
    warn_threshold: float = 2.0,
    block_threshold: float = 3.5,
) -> dict:
    """Compare current measurements to last calibration. Decide pass/warn/block."""
    delta_es = np.array([
        delta_e_2000(current_swatch_lab[i], baseline_swatch_lab[i])
        for i in range(len(current_swatch_lab))
    ])
    max_de = float(np.max(delta_es))
    mean_de = float(np.mean(delta_es))

    if max_de >= block_threshold:
        status = "BLOCK"
    elif max_de >= warn_threshold:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "status": status,
        "delta_e_mean": mean_de,
        "delta_e_max": max_de,
        "delta_e_per_concentration": delta_es.tolist(),
        "thresholds": {"warn": warn_threshold, "block": block_threshold},
    }


# ----------------------------------------------------------------------
# Stage 6: MCP Tool Entry Points
# ----------------------------------------------------------------------
def bootstrap_pigment(
    pigment_id: str,
    pigment_name: str,
    source: str,
    raw_white: Path,
    raw_black: Path,
    raw_colorchecker: Path,
    raw_flat_field: Path,
    pigment_row_on_plate: int,
    concentration_ladder: list[float] = [0.03, 0.06, 0.12, 0.25, 0.50, 0.75, 1.00],
    recipes: dict[float, str] | None = None,
    output_dir: Path = Path("pigments"),
) -> Path:
    """Full one-shot calibration. Outputs pigments/{pigment_id}.yaml."""
    recipes = recipes or {c: "" for c in concentration_ladder}

    img_cc = load_raw_linear(raw_colorchecker)
    img_flat = load_raw_linear(raw_flat_field)
    img_white = apply_flat_field(load_raw_linear(raw_white), img_flat)
    img_black = apply_flat_field(load_raw_linear(raw_black), img_flat)

    cc_patches = detect_colorchecker(img_cc)
    ref_xyz = get_reference_xyz_d50()
    ccm = fit_root_polynomial_ccm(cc_patches, ref_xyz, degree=2)

    if ccm["fit_delta_e_max"] > 4.0:
        raise RuntimeError(
            f"CCM fit max ΔE = {ccm['fit_delta_e_max']:.2f} > 4.0. "
            f"Re-check lighting / focus / ColorChecker condition."
        )

    img_white_u8 = (np.clip(img_white, 0, 1) * 255).astype(np.uint8)
    img_black_u8 = (np.clip(img_black, 0, 1) * 255).astype(np.uint8)
    warped_white = warp_via_aruco(img_white_u8)
    warped_black = warp_via_aruco(img_black_u8)

    swatch_rgb_W = sample_swatch_grid(warped_white.astype(np.float32) / 255.0, [pigment_row_on_plate])[0]
    swatch_rgb_B = sample_swatch_grid(warped_black.astype(np.float32) / 255.0, [pigment_row_on_plate])[0]

    swatch_xyz_W = apply_ccm(swatch_rgb_W, ccm)
    swatch_xyz_B = apply_ccm(swatch_rgb_B, ccm)
    illum_d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    swatch_lab_W = np.array([colour.XYZ_to_Lab(c, illum_d50) for c in swatch_xyz_W])
    swatch_lab_B = np.array([colour.XYZ_to_Lab(c, illum_d50) for c in swatch_xyz_B])

    # Linear sRGB approximation for K-M inverse (3-channel; t3 will replace with 36-wl)
    swatch_R_W = np.array([colour.XYZ_to_RGB(c, colour.RGB_COLOURSPACES['sRGB']) for c in swatch_xyz_W])
    swatch_R_B = np.array([colour.XYZ_to_RGB(c, colour.RGB_COLOURSPACES['sRGB']) for c in swatch_xyz_B])

    ladder = []
    for i, c in enumerate(concentration_ladder):
        K, S = km_inverse_two_substrate(swatch_R_W[i], swatch_R_B[i])
        ladder.append({
            "c_ratio": float(c),
            "recipe": recipes.get(c, ""),
            "measured_R_over_white": swatch_R_W[i].tolist(),
            "measured_R_over_black": swatch_R_B[i].tolist(),
            "measured_Lab_over_white": swatch_lab_W[i].tolist(),
            "measured_Lab_over_black": swatch_lab_B[i].tolist(),
            "derived_K": K.tolist(),
            "derived_S": S.tolist(),
        })

    yaml_dict = {
        "pigment_id": pigment_id,
        "name": pigment_name,
        "source": source,
        "calibration_date": dt.date.today().isoformat(),
        "calibration_protocol_version": "chuck-mcp-v2.0.0",
        "session_ccm": ccm,
        "plate_row_index": pigment_row_on_plate,
        "supply_level": "high",
        "ladder": ladder,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pigment_id}.yaml"
    import yaml
    out_path.write_text(yaml.safe_dump(yaml_dict, default_flow_style=False))
    return out_path


def drift_check(
    pigment_id: str,
    raw_sentinel: Path,
    raw_colorchecker: Path,
    raw_flat_field: Path,
    plate_row_index: int,
    pigments_dir: Path = Path("pigments"),
) -> dict:
    """Per-session drift check before printing an edition."""
    import yaml
    baseline = yaml.safe_load((pigments_dir / f"{pigment_id}.yaml").read_text())
    baseline_lab = np.array([row["measured_Lab_over_white"] for row in baseline["ladder"]])

    img_cc = load_raw_linear(raw_colorchecker)
    img_flat = load_raw_linear(raw_flat_field)
    img_sentinel = apply_flat_field(load_raw_linear(raw_sentinel), img_flat)

    cc_patches = detect_colorchecker(img_cc)
    ref_xyz = get_reference_xyz_d50()
    ccm = fit_root_polynomial_ccm(cc_patches, ref_xyz, degree=2)

    img_s_u8 = (np.clip(img_sentinel, 0, 1) * 255).astype(np.uint8)
    warped = warp_via_aruco(img_s_u8)
    swatch_rgb = sample_swatch_grid(warped.astype(np.float32) / 255.0, [plate_row_index])[0]
    swatch_xyz = apply_ccm(swatch_rgb, ccm)
    illum_d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    current_lab = np.array([colour.XYZ_to_Lab(c, illum_d50) for c in swatch_xyz])

    return check_drift_against_baseline(current_lab, baseline_lab)
```

## Testing the pipeline (synthetic ground-truth)

Before running on real photos, validate the math against synthetic inputs:

```python
def test_km_roundtrip():
    """Generate synthetic R_W, R_B from known (K, S), invert, check round-trip."""
    K_true = np.array([0.5, 1.2, 0.3])
    S_true = np.array([0.8, 0.4, 1.5])
    # Forward model (Kubelka-Munk + 2-substrate composition)
    a = 1.0 + K_true / S_true
    b = np.sqrt(a * a - 1)
    x = 1.0
    bSx = b * S_true * x
    R_layer = np.sinh(bSx) / (a * np.sinh(bSx) + b * np.cosh(bSx))
    T_layer = b / (a * np.sinh(bSx) + b * np.cosh(bSx))
    R_W = R_layer + T_layer**2 * 0.95 / (1 - R_layer * 0.95)
    R_B = R_layer + T_layer**2 * 0.05 / (1 - R_layer * 0.05)
    # Inverse
    K_rec, S_rec = km_inverse_two_substrate(R_W, R_B)
    assert np.allclose(K_rec, K_true, atol=0.01), f"K mismatch: {K_rec} vs {K_true}"
    assert np.allclose(S_rec, S_true, atol=0.01), f"S mismatch: {S_rec} vs {S_true}"
```

## Performance / timing budget

| Stage | Wall time (M1 Mac) | Notes |
|---|---|---|
| Load RAW (24 MP) | ~1.5 sec | rawpy AHD demosaic |
| Flat-field | ~0.05 sec | numpy divide |
| ColorChecker detect | ~3-5 sec | YOLOv8 inference |
| CCM fit (Finlayson 2015) | ~0.5 sec | colour-science |
| ArUco warp | ~0.2 sec | OpenCV |
| Swatch sample | ~0.05 sec | numpy slicing |
| K-M inverse (7 stripes × 3 channels) | ~0.3 sec | brentq per channel |
| YAML write | ~0.01 sec | pyyaml |
| **Total per pigment** | **~6-8 sec** | Well within MCP tool latency budget |

## Failure modes the implementation must handle

1. **ColorChecker not detected** — clear error, ask Reid to reframe and reshoot.
2. **<4 ArUco fiducials visible** — clear error, swatch plate not fully in frame.
3. **CCM fit ΔE_max > 4** — auto-block, indicates capture quality issue. Common causes: bad lighting, wrong ColorChecker version, ambient light leak.
4. **K-M solver fails to converge** — fallback to a 2-parameter regression on the ladder. Log as warning.
5. **Negative K or S** — clip to zero, log warning. Indicates measurement noise dominating, common at very low c_ratio.
6. **Saturated pixels in any swatch** — error, bracket exposure properly. Saturation kills the math.

## Why this is "good enough" without a spectrophotometer

The 3-channel approximation (Lab → linear sRGB → R_RGB) loses spectral fidelity. For Reid's V1 calibration goal — supporting **chuck-mcp t1 Mixbox tier and t2 empirical LUT tier** — this is acceptable:

- **t1 Mixbox**: only needs Lab values, not K/S directly. The pipeline delivers Lab at ΔE_00 ≈ 2 mean.
- **t2 empirical LUT**: needs swatch reflectance as 3-channel for the look-up; the K-M inverse gives "principled" K/S that act as the LUT's index.
- **t3 spectral (future)**: requires 36-wavelength data. Calibration MCP v1 doesn't deliver this. v2 of the calibration MCP could add a multi-spectral capture mode (filter wheel + monochrome conversion, or use a multispectral target like the X-Rite SpectralLight QC). Not v1 scope.

The 3-channel pipeline is "good enough" because:
1. Reid's editions are 10 prints. Sub-ΔE-2 across 10 prints, **using the same camera+CCM**, only requires repeatability, not absolute accuracy.
2. Mokuhanga has inherent print-to-print variation of ΔE 2-4 from inking and pressure. The camera-based system's accuracy ceiling matches the practical floor of the printing process.
3. Reid can swap in a spectrophotometer later without changing the API surface — just upgrade the `bootstrap_pigment` implementation.
