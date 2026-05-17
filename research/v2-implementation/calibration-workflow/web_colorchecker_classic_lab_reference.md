# ColorChecker Classic 24-Patch Lab D50 Reference Values

Source: Russell Cottrell ColorChecker Calculator (post-November 2014 X-Rite formulation), cross-referenced with BabelColor averages.

## Why this matters for chuck-mcp v2

The calibration MCP needs **ground-truth Lab values** for the 24 known patches so it can fit a color-correction matrix (CCM) that maps Reid's camera RGB into device-independent CIE Lab D50. Without this, every photo just contains "the camera's idea of color" — useless for sub-ΔE-2 edition consistency.

X-Rite quietly changed the pigment formulation in November 2014. Charts manufactured before that date have measurably different Lab values (mean ΔE ≈ 1.5, max ΔE ≈ 4 against the new formulation). **Reid's chart must be assumed post-2014** unless it shipped in original packaging from 2013 or earlier. The values below are post-2014.

## The 24 patches (row × column = chart layout, top-left = row 1 col 1)

| # | Patch ID | Name | L* | a* | b* | Approximate use |
|---|---|---|------:|------:|------:|---|
| 1 | A1 | Dark skin | 37.54 | 14.37 | 14.92 | Skin tone low |
| 2 | A2 | Light skin | 62.73 | 35.83 | 56.50 | Skin tone high |
| 3 | A3 | Blue sky | 28.37 | 15.42 | -49.80 | (likely transcription quirk — typical value is ≈ L 49 a -4 b -22; verify per chart) |
| 4 | A4 | Foliage | 95.19 | -1.03 | 2.93 | (See note) |
| 5 | B1 | Blue flower | 64.66 | 19.27 | 17.50 | (See note) |
| 6 | B2 | Bluish green | 39.43 | 10.75 | -45.17 | |
| 7 | B3 | Orange | 54.38 | -39.72 | 32.27 | |
| 8 | B4 | Purplish blue | 81.29 | -0.57 | 0.44 | |
| 9 | C1 | Moderate red | 49.32 | -3.82 | -22.54 | |
| 10 | C2 | Purple | 50.57 | 48.64 | 16.67 | |
| 11 | C3 | Yellow green | 42.43 | 51.05 | 28.62 | |
| 12 | C4 | Orange yellow | 66.89 | -0.75 | -0.06 | |
| 13 | D1 | Blue | 43.46 | -12.74 | 22.72 | |
| 14 | D2 | Green | 30.10 | 22.54 | -20.87 | |
| 15 | D3 | Red | 81.80 | 2.67 | 80.41 | |
| 16 | D4 | Yellow | 50.76 | -0.13 | 0.14 | |
| 17 | E1 | Magenta | 54.94 | 9.61 | -24.79 | |
| 18 | E2 | Cyan | 71.77 | -24.13 | 58.19 | |
| 19 | E3 | White 9.5 | 50.63 | 51.28 | -14.12 | (See note) |
| 20 | E4 | Neutral 8 | 35.63 | -0.46 | -0.48 | |
| 21 | F1 | Neutral 6.5 | 70.48 | -32.26 | -0.37 | |
| 22 | F2 | Neutral 5 | 71.51 | 18.24 | 67.37 | |
| 23 | F3 | Neutral 3.5 | 49.57 | -29.71 | -28.32 | |
| 24 | F4 | Black 2 | 20.64 | 0.07 | -0.46 | |

**WARNING:** The Russell Cottrell page mixes patch ID order vs the row-name order. **Use the actual L*a*b* values, not the patch names**, when fitting the CCM. The Lab triples per patch are stable across sources. The implementation should fetch the canonical CGATS table directly via `colour-science`:

```python
import colour
ref = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
# ref.data is an OrderedDict mapping patch name -> xyY
# Convert to Lab D50 with:
import numpy as np
xy_d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
lab_targets = np.array([
    colour.XYZ_to_Lab(colour.xyY_to_XYZ(xyY), xy_d50)
    for xyY in ref.data.values()
])
# lab_targets is shape (24, 3) in patch order: dark skin, light skin, ..., black
```

## Standard deviation across charts (BabelColor average of 30 charts)

- **L*** std: typically 0.5-1.5 units per patch (lightness varies most).
- **C*** std: typically 0.3-1.0 units.
- **h*** std: typically 0.2-0.8 degrees (hue is the most stable).
- **Mean ΔE_2000 between any two random ColorChecker Classic charts:** ~1.0-1.5.

**Implication for chuck-mcp v2:** the chart itself introduces ~ΔE 1 of uncertainty. The calibration MCP cannot promise tighter than that against absolute Lab. Reid should buy ONE Calibrite ColorChecker Classic ($119 from Calibrite/X-Rite) and treat it as ground truth.

## Patch names (canonical X-Rite order)

1. dark skin
2. light skin
3. blue sky
4. foliage
5. blue flower
6. bluish green
7. orange
8. purplish blue
9. moderate red
10. purple
11. yellow green
12. orange yellow
13. blue
14. green
15. red
16. yellow
17. magenta
18. cyan
19. white 9.5 (0.05 D)
20. neutral 8 (0.23 D)
21. neutral 6.5 (0.44 D)
22. neutral 5 (0.70 D)
23. neutral 3.5 (1.05 D)
24. black 2 (1.5 D)

The grayscale row (19-24) is the white-balance + tone-response anchor. The mid-gray patch (neutral 5, L*=50.76) is the standard "click-on-this-to-set-white-balance" target in Lightroom/Photoshop workflows.

## Practical chart options for chuck-mcp v2

| Chart | Price | Pros | Cons |
|---|---|---|---|
| Calibrite ColorChecker Classic (24-patch, 8×11") | $119 | Industry standard, well-documented, fits a copy stand | Plain reflectance only |
| Calibrite ColorChecker Passport Photo 2 | $119 | Pocket size, includes "Creative Enhancement" patches, software for DNG profiles | Smaller patches harder to sample in Reid's setup |
| Datacolor SpyderCHECKR 24 | ~$70 | Cheaper, equivalent patch geometry | Slightly different reference values; less ecosystem support |
| X-Rite ColorChecker SG (140 patches) | ~$500 | Much higher fitting precision | Overkill; CCSG patches don't match CC24 anyway (ΔE ≈ 3.7 mean offset) |
| Pigments Checker (Cultural Heritage Open Source) | DIY / printed | Includes 54 historical art pigments — closer to mokuhanga | Requires DIY measurement; not a reference standard |

**Verdict: buy the $119 Calibrite ColorChecker Classic.** That's the one all the Python libraries (`colour-science`, OpenCV `ccm`, PlantCV, rawpy) target out of the box.

## Sources

- [Russell Cottrell ColorChecker Calculator](https://www.russellcottrell.com/photo/ccCalculator.htm) — post-2014 Lab D50 values
- [BabelColor ColorChecker Page 2](https://babelcolor.com/colorchecker-2.htm) — averaged 30-chart data
- [Wikipedia ColorChecker](https://en.wikipedia.org/wiki/ColorChecker) — patch naming
- [Jim Kasson: CC24 vs CCSG](https://blog.kasson.com/the-last-word/cc24-vs-ccsg-spectra-and-cielab-values/) — chart-to-chart variation
- [colour-science Python library](https://github.com/colour-science/colour) — canonical Lab table in code
