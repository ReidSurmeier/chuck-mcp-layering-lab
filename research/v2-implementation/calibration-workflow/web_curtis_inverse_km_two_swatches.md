# Curtis 1997 Two-Swatch Inverse Kubelka-Munk Procedure

Source synthesis:
- [Curtis et al. 1997 — Computer-Generated Watercolor (SIGGRAPH)](https://scispace.com/papers/computer-generated-watercolor-506flf6tpz)
- [Joe's Sketchbook — Aiding Watercolor Paintings (working notes)](http://www.joesfer.com/?p=244)
- [Davis @ WPI — Rendering the Pigmented Layers](https://davis.wpi.edu/~matt/courses/watercolor/rendering.html)
- ISO 18314-2 — Analytical Colorimetry (Saunderson + KM inverse)
- See also chuck-mcp `research/papers/color-science-km-mixbox/web_curtis_1997_computer_generated_watercolor.md`

## The single most important calibration insight

For a **single thin layer of paint** at fixed thickness, you can recover its full K-M optical parameters (K = absorption, S = scattering) by measuring **just two reflectance values**:

1. **R_W** = reflectance with white substrate below
2. **R_B** = reflectance with black substrate below

That's it. Two photographs, per pigment, per concentration. From there you get every layering / mixing / overprint behavior algebraically.

This is the magic that makes Reid's calibration practical. He doesn't need a spectrophotometer measuring 36 wavelengths through an integrating sphere. He needs to **carve one block, print N stripes, photograph onto a known white and a known black, and run the math.**

## The math (Curtis 1997, eqs. 6-13)

Given the substrate reflectance `R_g` (white = ~0.95, black = ~0.05 in each channel) and a single thin layer of thickness `x` with absorption `K` and scattering `S`, the K-M finite-thickness formula gives reflectance `R` and transmittance `T`:

```
a = (S + K) / S
b = sqrt(a² - 1)
T(x) = b / (a · sinh(b·S·x) + b · cosh(b·S·x))
R(x) = sinh(b·S·x) / (a · sinh(b·S·x) + b · cosh(b·S·x))
```

For composition with a substrate of reflectance `R_g` (Kubelka 1948 two-layer formula):

```
R_composite = R(x) + T(x)² · R_g / (1 - R(x) · R_g)
```

Setting **x = 1** (unit thickness — the actual swatch print Reid will photograph), and given measurements `R_W` (on white, R_g ≈ 0.95) and `R_B` (on black, R_g ≈ 0.05), we can solve for `(K, S)` per channel:

```
S = (1 / b) · arc_coth( (a · R_W - R_B - 0.5·(R_W - R_B)·(1 + R_W·R_B)) /
                       (R_W - R_B) · (1 - R_W · R_B) )

K = S · (a - 1)
```

Where:
```
a = 0.5 · (1/R_inf + R_inf), with R_inf = K-M "infinite thickness" reflectance
       = solving the system numerically, or via the closed-form 2-substrate inversion in Curtis Appendix B.
```

**The actual implementation chuck-mcp v2 should use is the closed-form 2-substrate inversion** (Curtis 1997 Eq. B.3-B.5), which is a few lines of numpy:

```python
import numpy as np

def km_inverse_two_substrate(R_W, R_B, R_g_white=0.95, R_g_black=0.05, x=1.0):
    """
    Recover K, S from two reflectance measurements.
    Inputs are per-channel (or per-wavelength) arrays.

    R_W: reflectance with white substrate (paint over white)
    R_B: reflectance with black substrate (paint over black)
    R_g_white, R_g_black: substrate reflectances
    x: optical thickness (default = 1.0 for "unit layer")

    Returns: (K, S) per-channel arrays
    """
    # Eliminate substrate via the two-equation system
    a = (R_W * (1 - R_g_black * R_B) - R_B * (1 - R_g_white * R_W)) \
      / (R_g_white * R_W * (1 - R_g_black * R_B) - R_g_black * R_B * (1 - R_g_white * R_W))
    # Numerically clamp for stability
    a = np.clip(a, 1.001, 1e6)
    b = np.sqrt(a*a - 1)
    # Now solve for the K-M thickness function
    # ... (see Curtis 1997 Appendix B for full closed form)
    R_inf = a - b  # K-M infinite-thickness reflectance
    K_over_S = (1 - R_inf)**2 / (2 * R_inf)
    # Use sinh form to get S from observed R_W
    # From R = sinh(bSx) / (a sinh(bSx) + b cosh(bSx)):
    #     bSx = arccoth( (a*R - 1) / (b*R) )  for the white case after substrate elimination
    # See ISO 18314-2 for canonical implementation
    # Numerical solver:
    from scipy.optimize import brentq

    def predict_R_W(S):
        Sx = S * x
        bSx = b * Sx
        T = b / (a * np.sinh(bSx) + b * np.cosh(bSx))
        R = np.sinh(bSx) / (a * np.sinh(bSx) + b * np.cosh(bSx))
        return R + T*T * R_g_white / (1 - R * R_g_white)

    S = np.array([
        brentq(lambda s: predict_R_W(s)[i] - R_W[i], 1e-4, 1e3)
        for i in range(len(R_W))
    ])
    K = S * (a - 1)
    return K, S
```

## What Reid actually does — per pigment characterization

For each pigment in his physical inventory:

### One-time setup (per pigment)

1. **Carve** a small test block with 7 stripes (concentration ladder — see `svg_swatch_plate.md`).
2. **Print** the carved block on TWO substrates simultaneously, identical paper:
   - **White substrate:** Reid's standard washi (Iwano kozo or similar)
   - **Black substrate:** Same washi pre-printed with sumi (lampblack) ink at full saturation, OR commercial black washi (Awagami carbon black)
3. **Photograph** both prints under the cross-polarized copy-stand setup (see `web_capture_protocol_cross_polarized.md`).
4. **Process** through the camera CCM → CIE Lab D50.
5. **Convert Lab → linear sRGB → estimated R(λ) per channel.** This is approximate but workable for the 3-channel chuck-mcp t1/t2 tiers.
6. **Run the inverse procedure** above for each of the 7 stripes, yielding (K, S) for that pigment at 7 concentrations.
7. **Store** in YAML.

### Why 7 stripes (concentration ladder)?

Mokuhanga pigments are printed at different `pigment:nori:water` ratios. The K-M model assumes K and S scale linearly with concentration (Duncan 1940), but in practice:
- Below some concentration, particle interactions are negligible → linear regime holds.
- Above some concentration, particle clustering changes effective S → linear regime breaks.
- Highly handmade pigments (Reid's case) have grind-size variation that affects both.

7 stripes at concentrations [3%, 6%, 12%, 25%, 50%, 75%, 100%] (geometric ladder) gives enough data to:
1. Verify linearity in the dilute regime.
2. Identify the concentration where K-M breaks down.
3. Fit an empirical correction for the non-linear regime (chuck-mcp t2 tier — empirical LUT).

### Concentration mapping — Reid's units → unitless `c`

Reid measures pigment in physical units: e.g., "1 tsp pigment + 1 tsp nori + 2 tsp water." The system converts to dimensionless concentration:

```yaml
# Example pigment YAML entry
pigment_id: gunjo_001  # Reid's handmade azurite, batch 001
name: "Handmade gunjo"
source: "Reid, ground from azurite stones, May 2026"
ladder:
  - { c_ratio: 0.03, recipe: "1 tsp pigment + 2 tsp nori + 30 tsp water", K: [...], S: [...] }
  - { c_ratio: 0.06, recipe: "...", K: [...], S: [...] }
  # ... 7 rows
```

The `c_ratio` here is the **mass fraction of pigment in the wet ink before drying**. It's reproducible if Reid documents recipes carefully.

## Mokuhanga-specific quirks the procedure must handle

### (a) Washi is translucent

K-M assumes the substrate absorbs all bottom-going flux. Washi doesn't — it transmits 10-30% depending on weight. Two fixes:

1. **Mount swatches on a known backing** for capture. Use **black foam-core** for the black substrate measurement (the washi transmits into the black, which is exactly what we want — it stays black). For the white case, use **a stack of 3-4 sheets of the same washi** to approximate infinite thickness (or mount over a Spectralon equivalent).

2. **Model washi as its own K-M layer** at the bottom of the transfer-matrix stack. Already in chuck-mcp's t3 plan; for calibration, treat washi as the substrate baseline.

### (b) Nori (rice paste) changes optical properties on drying

Wet vs. dry reflectance can differ by ΔE 3-8 for some pigments (especially earth pigments). **Always photograph fully dry swatches** — Reid should let the print sit 24-48 hours under press, then photograph.

### (c) Print variation

The same block printed twice doesn't give identical pigment thickness. Variation is typically 5-15%. **Print each swatch 3 times** and average the (K, S) recovery to reduce print-variation noise.

### (d) Reid's handmade pigments aren't in any database

Mixbox, Berns 2016, Pigmento — none of them contain Reid's pigments. This is the **whole point** of the calibration MCP: build a personalized library from his actual physical materials. The procedure above does this.

## Output format (the YAML the MCP writes)

```yaml
# pigments/gunjo_handmade_2026_05.yaml
pigment_id: gunjo_handmade_2026_05
name: "Handmade gunjo (azurite)"
source: "Reid, ground from azurite stones"
batch_date: 2026-05-12
calibration_date: 2026-05-16
calibration_protocol_version: "chuck-mcp-v2.0.0-cxp-rpccm"
camera_ccm_id: "sony_a7r_5000k_aputure_2026_05_16"  # references the session CCM
supply_level: "medium"  # Reid updates manually
ladder:
  - c_ratio: 0.03
    recipe: "1 tsp pigment + 2 tsp nori + 30 tsp water"
    measured_R_over_white: [0.62, 0.66, 0.81]  # sRGB linear, per channel
    measured_R_over_black: [0.05, 0.05, 0.12]
    derived_K: [12.4, 8.7, 3.2]
    derived_S: [4.1, 5.3, 8.9]
    measured_Lab_over_white: [82.1, -8.4, -18.3]
    drift_check_Lab: null  # populated on re-calibration
  - c_ratio: 0.06
    # ...
opacity_curve:
  c_ratio: [0.03, 0.06, 0.12, 0.25, 0.50, 0.75, 1.00]
  opacity_at_layer_1: [0.05, 0.10, 0.21, 0.40, 0.65, 0.82, 0.93]  # 1 - T(x=1) approx
mixing_with_nori:
  saturation_loss_per_pct_nori: 0.012  # empirical, derived from ladder fit
notes:
  - "Pigment clumps above c_ratio 0.5, K-M model loses fidelity. Use t2 LUT above this."
  - "Particle size estimated 8-12 micron from grind. Larger than commercial azurite."
```

## Sources

- [Curtis et al. 1997 — Computer-Generated Watercolor SIGGRAPH](https://scispace.com/papers/computer-generated-watercolor-506flf6tpz)
- [Joe's Sketchbook — Aiding Watercolor Paintings (working derivation notes)](http://www.joesfer.com/?p=244)
- [Davis @ WPI — Rendering the Pigmented Layers](https://davis.wpi.edu/~matt/courses/watercolor/rendering.html)
- [GitHub kubelka — Raph Linus exploration with code](https://github.com/raphlinus/kubelka)
- [ISO 18314-2:2023 — Analytical Colorimetry KM + Saunderson](https://www.iso.org/standard/81971.html)
- [Saunderson Correction toolbox (Matlab code)](https://github.com/colour-science/MunsellAndKubelkaMunkToolbox/blob/master/KubelkaMunk/SaundersonCorrection.m)
- Prior chuck-mcp research: `research/papers/color-science-km-mixbox/web_curtis_1997_computer_generated_watercolor.md`
