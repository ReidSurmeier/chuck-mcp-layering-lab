# Capture Protocol: Cross-Polarized Copy-Stand Photography for Pigment Swatches

Sources:
- Cultural Heritage Science Open Source [polarized light photography for art documentation](https://chsopensource.org/polarized-light-photography-for-art-documentation/)
- BetterLight technical paper [Cross Polarization for Art Reproduction](https://www.betterlight.com/downloads/conference10_speakers/guyer_Polarization.pdf)
- npj Heritage Science 2021 [Exploring the limits of color accuracy in technical photography](https://www.nature.com/articles/s40494-021-00536-x)
- DPMag [Photographic Copy Work tip](https://www.dpmag.com/how-to/tip-of-the-week/photographic-copy-work-1-20-14/)

## Why cross-polarization (CXP) is non-negotiable for mokuhanga swatches

Mokuhanga pigment + nori on washi has three optical layers that all reflect specularly to varying degrees:
1. **Surface glaze** of dried starch + binder (mirror-shine in fresh pulls)
2. **Pigment particle facets** — especially flake pigments like gunjo (azurite) and gofun (oyster shell)
3. **Washi fiber surface** — visible glints from kozo strands

Without polarization, these specular components ride on top of the diffuse reflectance signal we actually want. The camera measures `R_measured ≈ R_diffuse + R_specular`, but the Kubelka-Munk and Saunderson math operates strictly on `R_diffuse`. Specular contamination = systematic L* error of 2-8 units, easily ΔE 3-10 contribution that's invisible to eyeballing.

Cross-polarization extinguishes the specular component because surface reflection preserves polarization while diffuse reflection scrambles it. Two crossed linear polarizers (one on each light, one on the lens) block the specular path entirely.

## Equipment list (one-time investment)

| Item | Cost | Notes |
|---|---|---|
| Copy stand (Kaiser RS-1 or Smith-Victor 36") | $150-300 | Bigger > smaller; you want 20" of pillar travel for full-sheet swatches |
| DSLR or mirrorless with RAW output | (Reid has) | Any modern body. Sony A7-class or Fuji X-T4 are typical; iPhone 14 Pro+ in ProRAW also viable (see error budget) |
| Macro or normal lens, f/8-f/11 sharpest aperture | (Reid has) | 50mm or 60mm macro is ideal |
| **Linear polarizing sheets, 17"×17" or larger** | $40-80 each (×2) | Edmund Optics #34-475 or B+H linear polarizer film. NOT photography sheet polarizers — laminated sheet stock |
| **Circular polarizer for lens** | $30-80 | "CPL" filter. **Must be circular**, not linear, to keep autofocus working |
| Two continuous LED panels (≥95 CRI, 5000K) | $80-200 each (×2) | Aputure Amaran 200d, GodoxSL150, or similar. **High CRI critical**; cheap LEDs have spectral holes that wreck color rendering |
| ColorChecker Classic (Calibrite) | $119 | The cornerstone reference |
| 18% gray card + 99% white reference | $20 | Acts as bright/dark anchors; PTFE alternative works |
| Black velvet or matte black foam-core | $15 | Background — kills the bottom-substrate ambiguity |
| Bubble level (small, for camera shoe) | $5 | Ensures swatch plane is perpendicular to lens axis |

**Total one-time: ~$700-1000.** Compare to a hand-held spectrophotometer (X-Rite i1Pro 3 ≈ $1500-2500) and you get a setup that handles 24 swatches at a time vs one patch at a time.

## Geometry: the magic 45/0 configuration

This is the spectroscopy-standard "45/0" geometry adapted for a flat copy-stand setup.

```
                Camera (lens axis = surface normal, looking straight down)
                          |
                          |  CPL on lens (rotation θ)
                          |
                          v
   Light A  ──>  ╱45°   [swatch plate on flat black bg]   45°╲  <── Light B
   (polarizer P)                                              (polarizer P)
```

- **Both lights at 45° to surface normal**, equidistant from the subject (~24-36" away), aimed at the swatch center.
- **Both light polarizers oriented identically** (e.g., horizontal axis = 0°).
- **Camera CPL rotated to extinguish reflection** → find rotation θ where a glossy test surface (a glass coverslip or laminated photo) goes darkest. This is the "cross" position, typically 90° off the light polarizers.

### Verification (do this every session)

1. Place a glossy laminated print under the lights.
2. Rotate the CPL on the lens slowly while watching the live view.
3. The bright reflection should darken to near-black at the cross position. If it doesn't, one of your light polarizers is misaligned.
4. Mark the correct rotation on the CPL ring with a sharpie — once aligned, the position is fixed for that light setup.

### Common errors

- **Wrong polarizer type on the lens:** linear polarizers break phase-detect AF on most cameras. Always use a CPL on the lens.
- **Polarizer not flush with light source:** even 5° tilt leaks polarization. Tape the sheet polarizer flat against the LED panel diffuser, or use a frame.
- **Tilted swatch:** if the print isn't flat, parts of it deviate from the 45° geometry and pick up specular. Use a sheet-glass press or vacuum hold-down.
- **Sneaking ambient light:** room lights / windows / the laptop monitor leak unpolarized light into the scene. Shoot in a dark room, or build a tent with the black velvet.

## Per-session capture sequence

1. **Set up the lights** — both polarizers parallel, CPL on lens crossed. Verify with glossy test object.
2. **Camera settings (manual everything):**
   - ISO: 100 (native, lowest noise)
   - Aperture: f/8 (sharpest, deepest DOF — swatch plate may have slight warp)
   - Shutter: long enough for proper exposure. Use a tripod / 2-sec self-timer to kill shake.
   - White balance: irrelevant for RAW capture but set to ~5000K daylight so previews don't lie
   - Color space: doesn't matter for RAW (the matrix lives in the file)
   - File format: **RAW only** (not RAW+JPEG, RAW only). JPEG already applied a profile = unrecoverable bias.
   - Mirror lockup: ON (DSLR)
   - Image stabilization: OFF (camera is locked to copy stand)
3. **Reference frame:** lay ColorChecker on the copy stand surface, frame it to fill ~30-50% of the image. Take 3 RAW exposures, bracket ±0.7 stop. Pick the one with the white patch at ~85-95% RGB max (no clipping).
4. **Flat-field frame:** photograph a uniformly-lit blank sheet of the same washi Reid prints on, same exposure. Used to correct lens vignetting and lighting falloff.
5. **Swatch frames:** place each swatch plate in the same position as the ColorChecker was. Shoot 3 RAW exposures, pick the cleanest.
6. **Dark frame:** lens cap on, same shutter speed, same ISO. One frame is enough. Used to subtract sensor bias on long exposures (only matters for shutter >1s).

## RAW processing pipeline (Python, headless)

```python
import rawpy
import numpy as np

def load_raw_linear(path):
    """Load a RAW file as linear sensor RGB. Bypass camera WB and tone mapping."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],   # neutral WB; we'll apply ours later
            output_color=rawpy.ColorSpace.raw,  # stay in linear camera RGB
            output_bps=16,                  # full bit depth
            no_auto_bright=True,            # disable auto exposure
            gamma=(1.0, 1.0),               # linear gamma
            half_size=False,
        )
    return rgb.astype(np.float32) / 65535.0

# Flat-field correction
swatch_linear = load_raw_linear("swatch.cr3")
flat_linear   = load_raw_linear("flat_field.cr3")
flat_norm     = flat_linear / np.median(flat_linear)
swatch_corrected = swatch_linear / flat_norm
# Optional: subtract dark frame for very long exposures
```

## Color correction with `colour-science` + `colour-checker-detection`

```python
import colour
from colour_checker_detection import detect_colour_checkers_segmentation

# Detect ColorChecker in reference image
reference_image_linear = load_raw_linear("colorchecker.cr3")
swatches_list = list(
    detect_colour_checkers_segmentation(reference_image_linear, additional_data=False)
)
camera_rgbs = swatches_list[0]  # shape (24, 3), linear camera RGB

# Get reference Lab D50 values
ref = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
illum_d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
ref_xyzs = np.array([colour.xyY_to_XYZ(xyY) for xyY in ref.data.values()])
ref_labs = np.array([colour.XYZ_to_Lab(xyz, illum_d50) for xyz in ref_xyzs])

# Fit a CCM from camera RGB -> CIE XYZ D50 using root-polynomial regression
from colour.characterisation import matrix_colour_correction_Cheung2004
# OR root-polynomial (Finlayson 2015) — chuck-mcp should prefer root-poly for exposure invariance
ccm = matrix_colour_correction_Finlayson2015(camera_rgbs, ref_xyzs, degree=2)

# Apply to a swatch image
swatch_xyz = colour.colour_correction(
    swatch_corrected, camera_rgbs, ref_xyzs,
    method='Finlayson 2015', degree=2,
)
swatch_lab = colour.XYZ_to_Lab(swatch_xyz, illum_d50)
```

## What ΔE accuracy can you actually achieve?

Synthesizing the npj Heritage Science 2021 results, Imatest documentation, and Finlayson 2015:

| Setup | Achievable ΔE_00 mean | Worst-case ΔE_00 | Notes |
|---|---|---|---|
| iPhone 14 Pro ProRAW + CPL + ColorChecker | 3-5 | 8-10 | Spectral sensitivity narrower than DSLR; some colors clip |
| Mid-tier DSLR + CPL + ColorChecker (linear CCM) | 2-3 | 4-6 | The "standard" published result |
| DSLR + CPL + ColorChecker + **root-polynomial CCM (Finlayson 2015)** | **1-2** | **3-4** | This is the target for chuck-mcp v2 |
| DSLR + CPL + custom DCP profile + chart | 1-1.5 | 2-3 | Requires X-Rite ColorChecker Camera Calibration software |
| i1Pro 3 spectrophotometer (handheld, 8mm aperture) | 0.3-0.8 | 1.5 | Reference baseline, not feasible for 24-up swatches |

**Target for chuck-mcp v2 calibration MCP:** Mean ΔE_00 ≤ 2, max ΔE_00 ≤ 4 against the ColorChecker, using root-polynomial CCM (degree 2). This is achievable with a DSLR + $700 setup.

## Drift-detection: when to recalibrate

The CCM is **scene-specific**: same camera, same lens, same lights → reusable. Recalibrate the CCM when ANY of these change:

| Change | Recalibrate? | Why |
|---|---|---|
| Same session, took a break <2 hours | No | Lights stable, no drift |
| Same day, lights turned off and on | Yes | LED phosphor warms up; can shift CCT by 100-300K |
| Different lens | Yes | Lens IR/UV filtering differs |
| Different lights | Yes (obvious) | |
| Different RAW developer or version | Yes | Demosaicing changes |
| Same setup, >7 days later | Yes | LED phosphor degrades over hours of use |
| Same setup, ambient temp Δ > 10°F | Maybe | LED CCT shifts with temperature |

**Practical rule:** photograph the ColorChecker AT THE START of every photo session as part of the standard workflow. Computing a fresh CCM takes ~2 seconds. The 30-second cost is insurance against ΔE 5-10 mistakes.

For drift detection of **pigments** (not camera), see `web_pigment_drift_lightfastness.md`.

## Why not just use Lightroom / Photoshop?

Reid has both. Why a custom Python pipeline?

Lightroom's "click-on-gray" white balance does ONE linear-channel scale (3 multipliers). That's worth maybe ΔE 6-8 of correction. Custom CCM (3×3 matrix or 6×9 root-polynomial) is worth ΔE 1-2. Photoshop's "match color" is unbounded perceptual nonsense.

**The MCP runs headless** as part of the chuck-mcp v2 calibration tool. Reid takes photos → MCP processes → updates YAML. No human-in-the-loop except for the photo capture itself.

## Sources

- [Cultural Heritage Science Open Source — Polarized Light Photography](https://chsopensource.org/polarized-light-photography-for-art-documentation/)
- [BetterLight — Cross Polarization for Art Reproduction](https://www.betterlight.com/downloads/conference10_speakers/guyer_Polarization.pdf)
- [npj Heritage Science 2021 — limits of color accuracy](https://www.nature.com/articles/s40494-021-00536-x)
- [Fstoppers — Cross Polarization Tutorial](https://fstoppers.com/science/capturing-colors-unseen-simple-tutorial-cross-polarization-photography-649027)
- [DPMag — Photographic Copy Work](https://www.dpmag.com/how-to/tip-of-the-week/photographic-copy-work-1-20-14/)
- [Adobe — Color Managed RAW Workflow PDF](https://www.adobe.com/digitalimag/pdfs/color_managed_raw_workflow.pdf)
- [Imatest — Color Check Documentation](https://www.imatest.com/docs/colorcheck/)
