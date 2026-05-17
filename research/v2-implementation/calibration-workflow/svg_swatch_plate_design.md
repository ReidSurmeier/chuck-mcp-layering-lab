# Swatch Plate Carving Design

## Goal

A single carved block (or pair: white-substrate + black-substrate prints) that, in one pull, exercises **all of Reid's pigments at multiple concentrations against the ColorChecker reference, with built-in registration and fiducials.**

## Why a single multi-pigment plate

Carving 15-25 individual blocks (one per pigment) is impractical and introduces uncontrolled pigment-to-pigment substrate variation. A SINGLE block with N stripe regions has:
- Same washi sheet → same substrate baseline.
- Same lighting in the photo → same CCM.
- Same registration fiducials → automatic patch detection.
- Same pull pressure → consistent across pigments.

Reid prints THIS block once per pigment-set update (e.g., when he adds a new pigment or grinds a new batch). Each print uses ONE pigment at a time, manually swapped between stripes. This is a "tedious but bounded" workflow — maybe 4 hours every few months.

## Plate dimensions (sized for Reid's typical washi)

- **Total plate size: 24 cm × 36 cm** (fits on a standard A3-equivalent sheet of Iwano washi with margins for registration kento)
- **Stripe count: 7 concentrations × 15 pigments = 105 swatches** — split into 5 horizontal bands of 21 swatches each for easier inking
- **Swatch size: each swatch 30mm × 30mm**
- **Spacing: 4mm gutter between swatches**
- **ColorChecker reference region: a 90mm × 60mm reserved blank area in one corner where the physical ColorChecker is placed during photography**

## Layout (top-down view of the printed sheet)

```
+--------------------------------------------+
| KENTO                                      |
| L-shape    [ColorChecker placed here in   ]|
|            [physical chart during photo  ]|
| (L-mark)                                   |
|--------------------------------------------|
|  P1c1  P1c2  P1c3  P1c4  P1c5  P1c6  P1c7  |
|  P2c1  P2c2  P2c3  P2c4  P2c5  P2c6  P2c7  |
|  P3c1  P3c2  P3c3  P3c4  P3c5  P3c6  P3c7  |
|  ...                                       |
|  P15c1 P15c2 P15c3 P15c4 P15c5 P15c6 P15c7 |
|--------------------------------------------|
| F1   F2   F3   F4    [Fiducial corners]   |
|              KENTO (registration mark)     |
+--------------------------------------------+
```

Where:
- `Pn` = pigment number (1-15, expanding to 25 with two rows of plates)
- `cm` = concentration index 1-7 (3%, 6%, 12%, 25%, 50%, 75%, 100%)
- `Fn` = ArUco fiducial markers at corners (essential for automatic detection)
- `KENTO` = traditional Japanese registration marks (L-shape + straight bar)

## The 7-step concentration ladder

Geometric ladder, base 2 with slight stretch at low end:
```
c_ratio: [0.03, 0.06, 0.12, 0.25, 0.50, 0.75, 1.00]
```

Reid mixes 7 inkings per pigment. Each inking is `c_ratio × (pigment paste) + (1 - c_ratio) × nori`. The block is brushed and printed once per inking — so 7 separate pulls per pigment. He inks ONE stripe at a time per concentration band.

Actually — more practical alternative: **carve 7 separate stripes per pigment as 7 separate blocks**, but print all 7 in one session on one washi sheet, swapping the inked block between pulls into the same kento registration. This is closer to actual mokuhanga workflow (one block = one ink).

## ArUco fiducial markers — why they're critical

The photo capture pipeline needs to find each swatch automatically. Without fiducials:
- Patch positions drift between prints (paper distortion ± 1-2mm).
- Reid would have to manually click on every swatch in software.
- Re-calibration of one pigment is impossible without manual labor.

With four ArUco markers in the corners:
- Image perspective is corrected via 4-point homography.
- Each swatch position is computed from the warped image coordinates.
- The whole pipeline is hands-off after photo capture.

ArUco markers are **carved into the block** as black-on-white patterns. Use the 4x4_50 dictionary (16 unique markers, 4x4 grid) so we have room for plate-version metadata too:

```
Marker 0: top-left corner anchor
Marker 1: top-right corner anchor
Marker 2: bottom-left corner anchor
Marker 3: bottom-right corner anchor
Marker 10-15: optional encoding of plate version / pigment batch
```

Marker size: **20mm × 20mm** in printed form (large enough for reliable detection from a 24MP DSLR image at copy-stand distance).

## SVG specification

The actual SVG file lives at `assets/calibration_plate_v1.svg` in chuck-mcp v2. Per-element spec:

```xml
<!-- Page setup: 240mm × 360mm, A3-ish, mokuhanga-friendly proportions -->
<svg xmlns="http://www.w3.org/2000/svg"
     width="240mm" height="360mm"
     viewBox="0 0 240 360">

  <!-- Outline (for carving guide) -->
  <rect x="0" y="0" width="240" height="360" fill="white" stroke="black" stroke-width="0.5"/>

  <!-- ArUco fiducials, 20mm × 20mm, one per corner, 10mm from edge -->
  <image href="aruco_4x4_50_0.svg" x="10"  y="10"  width="20" height="20"/>
  <image href="aruco_4x4_50_1.svg" x="210" y="10"  width="20" height="20"/>
  <image href="aruco_4x4_50_2.svg" x="10"  y="330" width="20" height="20"/>
  <image href="aruco_4x4_50_3.svg" x="210" y="330" width="20" height="20"/>

  <!-- Reserved area for physical ColorChecker (top center, 90mm × 60mm) -->
  <rect x="75" y="15" width="90" height="60" fill="none"
        stroke="black" stroke-width="0.3" stroke-dasharray="2,2"/>
  <text x="120" y="50" text-anchor="middle" font-size="6">
    Place ColorChecker here
  </text>

  <!-- 15 pigment rows × 7 concentration columns, swatch grid starting at y=90 -->
  <!-- Each swatch is 30mm × 30mm with 4mm gutter -->
  <!-- Row labels left side (pigment IDs); column labels top (concentrations) -->

  <!-- Programmatically generated: -->
  <g id="swatch-grid">
    <!-- For row i in 0..14: -->
    <!--   For col j in 0..6: -->
    <!--     <rect x="40 + j*34" y="90 + i*17" width="30" height="14" fill="black" /> -->
    <!--   (Note: thinner rows than swatches to fit 15 rows; or split across 2 plates) -->
  </g>

  <!-- Pigment ID labels (left margin) - carved as raised text -->
  <text x="35" y="97" text-anchor="end" font-size="3">P1</text>
  <!-- ... etc -->

  <!-- Concentration labels (top of grid) -->
  <text x="55" y="86" text-anchor="middle" font-size="3">3%</text>
  <!-- ... 6%, 12%, 25%, 50%, 75%, 100% -->

  <!-- Traditional kento registration marks (bottom edge + bottom-left corner) -->
  <!-- L-shape kento, ~15mm legs -->
  <path d="M 10,355 L 10,340 L 25,340" fill="none" stroke="black" stroke-width="0.5"/>
  <!-- Straight bar kento on right -->
  <line x1="200" y1="355" x2="230" y2="355" stroke="black" stroke-width="0.5"/>

  <!-- Version marker (carved as text, machine-readable) -->
  <text x="120" y="355" text-anchor="middle" font-size="4">CHUCK-MCP CAL v1.0</text>
</svg>
```

## Practical adjustment: 15 pigments at 7 concentrations = 105 swatches per plate

With swatches at 30×30mm + 4mm gutter, the grid needs:
- Width: 7 × (30+4) - 4 = 234mm ✓ fits in 240mm
- Height: 15 × (30+4) - 4 = 506mm ✗ doesn't fit in 360mm

**Two solutions:**

### Option A: Smaller swatches (recommended)

Reduce swatch to **18mm × 18mm with 3mm gutter**:
- Width: 7 × 21 - 3 = 144mm
- Height: 15 × 21 - 3 = 312mm
- Fits in 240×360 with margins for ColorChecker + fiducials + kento

18×18mm is still well above the 8mm aperture of an i1Pro spectrophotometer and well above the per-patch pixel count needed for averaging (a 24MP image at 240mm field-of-view gives ~12 pixels/mm, so 18×18mm = ~216×216 pixels per swatch = plenty).

### Option B: Two plates (one for pigments 1-7, one for 8-15)

Carve and print twice. Maintains 30×30 swatches but doubles the work.

**Recommendation: Option A.** Smaller swatches, one plate, single workflow.

## Re-carving doesn't happen often

The block is a one-time carve. When Reid adds a new pigment (say, P16), he can either:
1. **Carve a second mini-plate** with just that pigment's 7 stripes + fiducials, and run the calibration MCP against it.
2. **Re-carve the main plate** if the pigment inventory churns enough.

For supply-level YAML updates (pigment running low), no re-carving needed — just edit the YAML field.

## Bootstrap procedure (the one-time MCP tool)

This is **a separate MCP tool from the per-print calibration tool**, called `calibrate_pigment_bootstrap`. Invoked once per pigment (or once per pigment-batch when Reid grinds a new batch).

Pseudocode:

```python
@mcp.tool("calibrate_pigment_bootstrap")
def bootstrap(
    pigment_id: str,
    pigment_name: str,
    source: str,
    swatch_photo_over_white: Path,   # RAW file
    swatch_photo_over_black: Path,   # RAW file
    colorchecker_photo: Path,         # RAW file from same session
    concentration_ladder: list[float] = [0.03, 0.06, 0.12, 0.25, 0.50, 0.75, 1.00],
    recipes: dict[float, str] = ...,  # human-readable recipe per c_ratio
) -> Path:
    """One-time setup for a new pigment. Photographs swatch plate, extracts Lab per stripe,
    runs Curtis inverse, writes pigments/{pigment_id}.yaml."""

    # 1. Load and demosaic RAW files (linear sensor RGB)
    img_cc = load_raw_linear(colorchecker_photo)
    img_white = load_raw_linear(swatch_photo_over_white)
    img_black = load_raw_linear(swatch_photo_over_black)

    # 2. Detect ColorChecker, fit root-polynomial CCM
    ccm = fit_ccm_root_polynomial(img_cc)
    save_ccm(ccm, session_id=...)  # for drift tracking

    # 3. Detect ArUco fiducials in swatch images, warp to canonical grid
    img_white_warp = warp_via_aruco(img_white)
    img_black_warp = warp_via_aruco(img_black)

    # 4. Sample each swatch position (averaging 80% of patch area, robust median)
    swatch_RGB_W = sample_swatches(img_white_warp, pigment_row=...)  # (7, 3)
    swatch_RGB_B = sample_swatches(img_black_warp, pigment_row=...)  # (7, 3)

    # 5. Apply CCM to get CIE XYZ -> Lab
    swatch_XYZ_W = apply_ccm(swatch_RGB_W, ccm)
    swatch_Lab_W = XYZ_to_Lab(swatch_XYZ_W, illum_D50)
    swatch_R_W = XYZ_to_linear_sRGB(swatch_XYZ_W)  # for K-M inverse (3-channel approximation)
    swatch_R_B = XYZ_to_linear_sRGB(apply_ccm(swatch_RGB_B, ccm))

    # 6. Run Curtis inverse per concentration
    K_array, S_array = [], []
    for i, c in enumerate(concentration_ladder):
        K, S = km_inverse_two_substrate(swatch_R_W[i], swatch_R_B[i],
                                         R_g_white=0.95, R_g_black=0.05)
        K_array.append(K.tolist())
        S_array.append(S.tolist())

    # 7. Write YAML
    yaml_out = {
        "pigment_id": pigment_id,
        "name": pigment_name,
        "source": source,
        "batch_date": datetime.today().isoformat(),
        "calibration_protocol_version": "chuck-mcp-v2.0.0",
        "camera_ccm_id": ccm.id,
        "supply_level": "high",
        "ladder": [
            {
                "c_ratio": c,
                "recipe": recipes.get(c, ""),
                "measured_R_over_white": swatch_R_W[i].tolist(),
                "measured_R_over_black": swatch_R_B[i].tolist(),
                "measured_Lab_over_white": swatch_Lab_W[i].tolist(),
                "derived_K": K_array[i],
                "derived_S": S_array[i],
            }
            for i, c in enumerate(concentration_ladder)
        ],
    }
    output_path = Path(f"pigments/{pigment_id}.yaml")
    output_path.write_text(yaml.dump(yaml_out))
    return output_path
```

## Per-edition drift check tool

Separate MCP tool: `calibrate_drift_check`. Called before printing an edition of 10. Photographs the standard ColorChecker + a single "drift sentinel" patch (e.g., a printed swatch of Reid's most-frequently-used pigment), compares to last calibration, and either:
- **Passes** if ΔE_00 of sentinel < 2.0 — proceed with edition.
- **Warns** if ΔE_00 < 3.5 — flag in log but proceed.
- **Blocks** if ΔE_00 >= 3.5 — re-bootstrap the affected pigment.

See `web_drift_detection_strategy.md` for the full drift logic.
