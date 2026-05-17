# Calibration Workflow — Research Notes & Verdict

Research agent: CALIBRATION-WORKFLOW for chuck-mcp v2 implementation swarm.
Swarm: `swarm-1778969836247-ys4o7z`.
Date: 2026-05-16.

Output folder: `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v2-implementation/calibration-workflow/`.

---

## VERDICT (TL;DR)

**The calibration system is two MCP tools (bootstrap + drift-check), built on a cross-polarized DSLR copy-stand rig, OpenCV + colour-science Python stack, root-polynomial CCM (Finlayson 2015), Curtis 1997 two-substrate K-M inverse, ArUco-fiducial-anchored swatch plate.**

Reid prints a swatch plate (single carved block, 10 pigments × 7 concentrations = 70 swatches per plate, two plates cover 20 pigments). He prints it on BOTH white and black washi substrates. He photographs both prints + a Calibrite ColorChecker Classic + a flat-field reference, all under crossed linear polarizers, on a copy stand with 95+ CRI LED lights at 45°. The MCP processes the RAW files → extracts Lab values per swatch → runs Curtis's inverse K-M to recover (K, S) per pigment per concentration → writes a YAML.

**Achievable error budget on V1:**
- Absolute color accuracy: **mean ΔE_2000 ≈ 2-3, max ≈ 5.** Good enough for chuck-mcp t1 (Mixbox) and t2 (empirical LUT) tiers.
- Inter-print repeatability: **ΔE_2000 ≈ 1-2** in an edition of 10, given consistent technique. Meets Reid's "sub-ΔE-2 across edition" requirement.

**Cost: ~$1400 one-time** ($700 for the DSLR rig if Reid uses a mid-tier used body, $700 for the CXP + lighting setup). Phone-based ($200 for CXP filters alone) is marginal — usable for non-edition work, not edition production.

The architecture cleanly separates calibration (model-time, interactive, photo capture) from rendering (inference-time, batch, downstream of YAML files). Pigment YAMLs are the contract between them.

---

## Concrete capture protocol (the workflow Reid follows)

### Equipment (settled)

- DSLR or mirrorless (Sony A6400, Fuji X-T30, or similar; ~$700 used)
- 50mm or 60mm macro lens, f/8-f/11 aperture
- Kaiser RS-1 or equivalent copy stand
- 2× Aputure Amaran 200d LED panels (95+ CRI, 5000K), $200 each
- 2× linear polarizing sheets, 17×17 inch (Edmund #34-475), $40-80 each
- Circular polarizer for lens (B+W or Hoya CPL), $50
- Calibrite ColorChecker Classic 24-patch, $119
- 18% gray card + 99% PTFE white reference, $20
- Black velvet or matte black background, $15
- Bubble level for camera shoe, $5

### Per-session capture sequence

1. Set up lights at 45° to copy stand surface, both polarizers parallel.
2. Rotate camera CPL to extinguish reflection on a glossy test surface.
3. Place ColorChecker on plate, take 3 bracketed RAW exposures.
4. Replace ColorChecker with blank washi sheet (flat-field), take 3 RAW.
5. Place swatch print over white substrate, take 3 RAW.
6. Replace with swatch print over black substrate, take 3 RAW.
7. Lens cap on, take 1 dark frame.
8. (For drift check, after step 2: place sentinel sheet, 3 RAW, done.)

### Processing pipeline (Python, headless)

```
RAW → rawpy linear demosaic → flat-field correction →
detect_colour_checkers_segmentation → fit Finlayson 2015 root-poly CCM →
ArUco fiducial detection + perspective warp → robust median sample per swatch →
apply CCM → CIE XYZ D50 → Lab D50 →
Curtis 1997 two-substrate inverse → (K, S) per channel →
YAML write
```

End-to-end: ~6-8 seconds per pigment after capture. Well within MCP tool latency.

---

## Sample swatch plate (what gets carved)

**Plate: 240mm × 360mm, 10 pigments × 7 concentrations = 70 swatches per plate.**

- Swatches: 18mm × 18mm with 3mm gutter
- 4 corner ArUco fiducials (DICT_4X4_50, IDs 0-3) at 20mm × 20mm
- ColorChecker reservation: 90mm × 60mm top center
- Traditional kento registration marks at bottom
- Concentration ladder: [3%, 6%, 12%, 25%, 50%, 75%, 100%]
- Two plates total (rows 1-10 on plate 1, rows 11-20 on plate 2)

Full SVG generation code in `svg_swatch_plate_assets.md`. The plate is CNC-carvable via the cnc-woodblock-tools pipeline or hand-carvable from a print of the SVG.

---

## Python pipeline summary

Single reference implementation in `web_pipeline_python_implementation.md`. Six modules:

| Module | Function | Lines |
|---|---|---|
| `capture.py` | RAW load + flat-field | ~30 |
| `detect.py` | ColorChecker + ArUco detection | ~80 |
| `ccm.py` | Finlayson 2015 root-poly CCM fit | ~40 |
| `km_inverse.py` | Curtis 1997 two-substrate K-M inverse | ~60 |
| `drift.py` | ΔE-based drift detection | ~40 |
| `pigment_yaml.py` | YAML schema + IO | ~60 |
| `mcp_tools.py` | MCP tool entry points | ~150 |

Total: ~460 lines of new code. Dependencies are all pip-installable; no proprietary software.

---

## Error budget acceptable for V1

| Component | ΔE_2000 contribution | Notes |
|---|---|---|
| ColorChecker chart-to-chart variation | 0.5-1.0 | Buy fresh chart |
| Camera spectral sensitivity mismatch | 0.5-1.5 | Inherent to 3-channel capture |
| CCM fit residual (Finlayson 2015) | 0.5-1.5 | Mean fit ΔE; published benchmark |
| Specular contamination (if CXP imperfect) | 0.5-3.0 | Must verify CXP every session |
| 3-channel approximation in K-M inverse | 0.3-1.0 | T3 (spectral) will eliminate; v1 accepts |
| Print-to-print pigment variation | 1.0-2.5 | Mitigated by 3-print averaging |
| **TOTAL absolute (RSS)** | **~2.5-3.5** | mean across pigments |
| **TOTAL inter-print (RSS)** | **~1.2-2.0** | given same camera + same YAML |

Reid's bar: ΔE ≤ 2 across edition of 10. Achievable in repeatability (the metric that matters). Absolute accuracy is ΔE 2-3 which is fine for t1/t2 tiers.

---

## Drift-detection strategy summary

**Two drift types:**
1. **Camera/setup drift** (hours-days): Always recompute CCM at start of every photo session. Cheap insurance.
2. **Pigment drift** (weeks-months): Sentinel-sheet drift check monthly + before every edition print.

**Sentinel sheet:** one printed swatch at c_ratio=0.50 of EVERY pigment in inventory, on one washi sheet, photographed once. ~30 minutes total. Compares Lab per pigment to baseline.

**ΔE thresholds:**
- ΔE < 1.5: PASS
- 1.5-2.5: WATCH
- 2.5-3.5: WARN (block edition use)
- ≥ 3.5: BLOCK (force re-bootstrap)

**Triggers for forced bootstrap:**
- New pigment batch (always)
- Storage conditions changed (manual)
- Drift check BLOCKED status (automatic)
- Calibration protocol major version bump (automatic)

Longitudinal `drift_history` in each YAML accumulates a record. Lets Reid see which pigments are stable (most azurites) vs fugitive (some plant-ground organics).

---

## MCP tool design (verdict)

**TWO tools, not one.** Bootstrap and drift-check have different photo requirements, durations, and blast radii.

| Tool | Purpose | Time per call |
|---|---|---|
| `chuck_mcp.calibration.bootstrap_pigment` | One-shot per pigment | ~10 sec processing; ~4 hr session (Reid's time) |
| `chuck_mcp.calibration.drift_check` | Periodic check across all pigments | ~30 sec processing; ~30 min session |
| `chuck_mcp.calibration.session_ccm` (optional) | Cache CCM for long sessions | ~3 sec |
| `chuck_mcp.calibration.list_pigments` | Inventory query | <1 sec |
| `chuck_mcp.calibration.set_supply` | Update supply_level YAML field | <1 sec |

Each tool has explicit failure modes with actionable `remediation_hint` fields in the error response. The bootstrap tool refuses to write a YAML if CCM fit max ΔE > 4.

---

## File inventory (this folder)

| File | Purpose | Lines |
|---|---|---|
| `NOTES.md` | This file — verdict + summary | (this) |
| `web_colorchecker_classic_lab_reference.md` | The 24-patch Lab D50 reference table + buying guide | ~150 |
| `web_capture_protocol_cross_polarized.md` | Full capture rig + protocol + Python RAW loading | ~280 |
| `web_curtis_inverse_km_two_swatches.md` | The Curtis 1997 inverse K-M procedure (math + code) | ~250 |
| `svg_swatch_plate_design.md` | Swatch plate dimensions + layout reasoning | ~250 |
| `svg_swatch_plate_assets.md` | Production-ready SVG + Python generator | ~270 |
| `web_pipeline_python_implementation.md` | Complete reference Python pipeline | ~460 |
| `web_drift_detection_strategy.md` | Drift detection policies + YAML schema additions | ~200 |
| `web_error_budget_v1.md` | V1 ΔE budget with two-track analysis | ~200 |
| `web_mcp_tool_design.md` | The two MCP tools — full API + workflows | ~280 |

**Total: 10 markdown artifacts.** Source-cited throughout.

---

## Top-3 must-reads

1. **`web_capture_protocol_cross_polarized.md`** — the photo capture is the most error-prone part of the whole system. Cross-polarization is non-negotiable for mokuhanga because the medium has high specular contamination. Without proper CXP, the whole pipeline yields ΔE 5-10 noise. Read this before buying any gear.

2. **`web_curtis_inverse_km_two_swatches.md`** — Curtis 1997's two-substrate (paint-over-white + paint-over-black) inverse procedure is the load-bearing math. It's how Reid recovers K-M parameters without a spectrophotometer. This file includes the closed-form numpy implementation, the synthetic round-trip validation, and the mokuhanga-specific edge cases (washi translucency, nori, etc.).

3. **`web_pipeline_python_implementation.md`** — the actual code that ships in chuck-mcp v2. Six modules, ~460 lines. Reference implementation for the swarm's coder agents to adapt. Includes the full failure-mode handling, the ArUco-anchored swatch sampling, the Finlayson 2015 root-polynomial CCM fit, and the K-M inverse solver.

---

## Verdict on capture protocol + error budget (the asked-for summary)

**Capture protocol:** **Cross-polarized copy-stand DSLR with Calibrite ColorChecker Classic.** Non-negotiable. The cross-polarization eliminates specular contamination that would otherwise add ΔE 3-10 of systematic error. The copy stand removes camera-handling variability. The ColorChecker anchors color to absolute Lab D50. The Finlayson 2015 root-polynomial CCM provides exposure-invariant calibration with mean ΔE fit residual < 2.

**Acceptable error budget V1:** **Mean ΔE_2000 ≤ 3, max ≤ 5 absolute. Mean ΔE_2000 ≤ 1.5, max ≤ 2 inter-print.** The repeatability budget (the one that matters for editions) is achievable with disciplined technique. The absolute budget is fine for chuck-mcp t1/t2 tiers and not aimed at spectrophotometer parity.

Phone-based fallback is possible (Reid already has an iPhone Pro with ProRAW) but pushes ΔE to 4-5 absolute. Use for non-edition exploration only.

A spectrophotometer ($1500-2500) gives ΔE < 1 absolute but only measures one patch at a time and breaks the "photograph 24-up swatches in one go" workflow. Not recommended for V1; reserved as a V2 upgrade path.

---

## Open questions to escalate to the swarm

1. **Plate carving — manual or CNC?** Reid has access to cnc-woodblock-tools, but hand-carving a flat 18mm × 18mm relief grid is also straightforward. Recommendation: **CNC**, because the ArUco fiducials need fine precision (sub-millimeter) that's tedious by hand.

2. **iPhone ProRAW vs DSLR — does Reid actually have to buy a DSLR?** Phone gives ~ΔE 4-5 absolute. Probably OK for early development (V0.x), upgrade for V1 production. Recommend: **start with phone, plan for DSLR after first editions.**

3. **Where does the swatch plate live physically?** It's a carved block + 2 printed swatches per pigment per recalibration. Reid's studio needs storage. Recommend: **a 3-ring binder of mylar-sleeved swatch prints, indexed by pigment_id + calibration_date.**

4. **Multi-spectral upgrade path.** V1 is 3-channel. T3 chuck-mcp tier wants 36-wavelength. Path: add a monochrome+filter-wheel mode to the calibration MCP (V2 work). Out of V1 scope.

5. **Reid's nori (rice paste) variability** is a wildcard. Suggest: **a separate "nori-only" baseline swatch on every plate**, photographed alongside, to subtract the paste's optical contribution from each pigment. Adds 1 column to the plate (8 columns total). Could fit if pitch drops to 20mm. Worth a swarm-level discussion.

---

## Sources (consolidated)

### Camera and color science

- [Calibrite ColorChecker Classic product page](https://calibrite.com/us/product/colorchecker-classic/)
- [BabelColor — ColorChecker Pages (averaged 30-chart data)](https://babelcolor.com/colorchecker-2.htm)
- [Russell Cottrell ColorChecker Calculator](https://www.russellcottrell.com/photo/ccCalculator.htm)
- [OpenCV Color Correction Model docs](https://docs.opencv.org/4.x/d1/dc1/tutorial_ccm_color_correction_model.html)
- [colour-science Python package](https://github.com/colour-science/colour)
- [colour-checker-detection Python package](https://github.com/colour-science/colour-checker-detection)
- [Finlayson 2015 — Color Correction Using Root-Polynomial Regression (IEEE TIP)](https://pubmed.ncbi.nlm.nih.gov/25769139/)
- [Imatest — Color Check documentation](https://www.imatest.com/docs/colorcheck/)
- [Adobe — Color Managed RAW Workflow](https://www.adobe.com/digitalimag/pdfs/color_managed_raw_workflow.pdf)
- [rawpy documentation](https://letmaik.github.io/rawpy/api/)

### Cross-polarized photography for art

- [Cultural Heritage Science Open Source — Polarized Light Photography](https://chsopensource.org/polarized-light-photography-for-art-documentation/)
- [BetterLight — Cross Polarization for Art Reproduction (PDF)](https://www.betterlight.com/downloads/conference10_speakers/guyer_Polarization.pdf)
- [npj Heritage Science 2021 — limits of color accuracy](https://www.nature.com/articles/s40494-021-00536-x)
- [Fstoppers — Cross Polarization Tutorial](https://fstoppers.com/science/capturing-colors-unseen-simple-tutorial-cross-polarization-photography-649027)

### Kubelka-Munk and Curtis 1997

- [Curtis 1997 — Computer-Generated Watercolor SIGGRAPH](https://scispace.com/papers/computer-generated-watercolor-506flf6tpz)
- [Joe's Sketchbook — working derivation notes on Curtis 1997](http://www.joesfer.com/?p=244)
- [Davis @ WPI — Rendering the Pigmented Layers](https://davis.wpi.edu/~matt/courses/watercolor/rendering.html)
- [Berns 2022 — Artist Acrylic Paint Spectral Dataset (PDF)](https://grayskyimaging.com/wp-content/uploads/2022/06/Berns_Archiving_2022.pdf)
- [ISO 18314-2:2023 — Analytical colorimetry KM + Saunderson](https://www.iso.org/standard/81971.html)
- Prior chuck-mcp research: `research/papers/color-science-km-mixbox/` (12 files)

### Pigment drift and lightfastness

- [Wiley 2025 — Microfading testing for lightfastness](https://onlinelibrary.wiley.com/doi/10.1111/cote.12788)
- [Wikipedia — Lightfastness (ASTM scale + BWS)](https://en.wikipedia.org/wiki/Lightfastness)
- [Kim Crick — Fugitive Pigments List](https://www.kimcrick.com/pages/fugitive-pigments-list-lightfast-test-problems-art-supplies)
- [Just Paint — How to Test for Lightfastness](https://justpaint.org/how-to-test-for-lightfastness/)

### Smartphone color and recent ML work

- [HuggingFace papers — CCMNet (cross-camera color constancy, 2025)](https://hf.co/papers/2504.07959)
- [HuggingFace papers — GCC (generative color constancy, 2025)](https://hf.co/papers/2502.17435)
- [Halide Process Zero — anti-AI iPhone photography](https://petapixel.com/2024/08/14/halides-anti-intelligent-update-makes-iphone-photos-truly-natural/)

### Cross-references inside chuck-mcp-layering-lab

- `research/papers/INDEX.md` — top-level research index
- `research/papers/color-science-km-mixbox/NOTES.md` — Curtis + Saunderson + Mixbox synthesis
- `research/papers/color-science-km-mixbox/web_curtis_1997_computer_generated_watercolor.md`
- `research/papers/color-science-km-mixbox/web_saunderson_1942_surface_correction.md`
- `research/papers/color-science-km-mixbox/web_berns_2016_artist_paint_spectral_database.md`
