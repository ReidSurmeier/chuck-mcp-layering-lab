# V1 Error Budget — Where the ΔE Comes From

## Reid's hard requirement

> Edition of 10 requires sub-ΔE-2 consistency across the 10 prints.

This is the **intra-edition** consistency requirement. It's a TIGHTER requirement than absolute color accuracy because it only needs the system to be **repeatable**, not **truthful**.

## Two separate budgets

### Budget A: Absolute accuracy of pigment characterization (lenient)

How well does the YAML's predicted color match the actual color of a printed swatch?

This is forgiving because:
- The downstream forward-render system (chuck-mcp t1/t2/t3) is itself approximate.
- Reid will visually verify and tune.
- The pigment YAML is a starting point, not a finished product.

**Target: mean ΔE_2000 ≤ 3, max ΔE_2000 ≤ 5.**

### Budget B: Inter-print repeatability of an edition (strict)

When Reid prints the same edition target 10 times using the SAME pigment YAML, how much do the 10 prints differ from each other?

This is strict because:
- Reid's whole pitch is "edition of 10 with sub-ΔE-2 consistency."
- Visible to a careful eye at ΔE ≈ 1-2.
- ΔE 2 is the threshold below which most viewers can't see differences side-by-side.

**Target: max ΔE_2000 ≤ 2 between any two prints in the edition.**

The two budgets are **independent**. Repeatability comes from the system NEVER moving the pigment YAML between prints (deterministic forward render + locked pigment library). Absolute accuracy comes from the calibration math being correct.

## Error decomposition for Budget A (absolute)

| Source | Typical contribution (ΔE_2000) | Notes |
|---|---|---|
| **ColorChecker chart-to-chart variation** | 0.5 - 1.0 | BabelColor's average across 30 charts. Inherent. |
| **Camera spectral sensitivity mismatch** | 0.5 - 1.5 | Camera RGB filters can't capture spectra perfectly. Worst for narrow-spectrum pigments. |
| **CCM fit residual (root-poly degree 2)** | 0.5 - 1.5 | Mean fit ΔE; published Finlayson 2015 benchmark. |
| **Quantization & demosaic noise** | 0.1 - 0.3 | Negligible if shooting RAW and not saturating. |
| **Flat-field / lighting non-uniformity** | 0.2 - 0.5 | Cross-polarization adds 2-stop loss, increasing this. |
| **Specular contamination (if CXP imperfect)** | 0.5 - 3.0 | Hopefully near zero with good CXP setup. |
| **3-channel approximation in K-M inverse** | 0.3 - 1.0 | Lab → linear sRGB loses spectral. Worse for "weird" pigments outside sRGB gamut. |
| **Reid's print-to-print pigment variation** | 1.0 - 2.5 | Hard to control without highly disciplined inking. The averaging of 3 prints mitigates. |
| **Round-trip to YAML and back** | 0.0 | Numerical, negligible if using float64. |
| **TOTAL (RSS, optimistic)** | **~1.8** | Sqrt of sum of squares, no correlation |
| **TOTAL (linear, pessimistic)** | **~5-7** | If all errors align |

Practically, expect mean ΔE_2000 around 2.5-3.5 with careful execution. **Achievable: ≤ 3 mean, ≤ 5 max.**

## Error decomposition for Budget B (repeatability)

This is the budget Reid cares about most.

| Source | Typical contribution (ΔE_2000) | Notes |
|---|---|---|
| **Same camera, same session, same CCM, same YAML** | 0 | Deterministic |
| **Inking variation across prints** | 0.5 - 1.5 | Reid's hand. Mitigated by his consistent technique. |
| **Pressure variation across pulls** | 0.3 - 0.8 | Baren technique, paper stretching. |
| **Sheet-to-sheet washi variation** | 0.2 - 0.6 | Within a single washi batch. |
| **Nori paste consistency over a single session** | 0.1 - 0.5 | Easy to control. |
| **Pigment settling in container during session** | 0.2 - 0.8 | Stir before each pull. |
| **TOTAL (RSS)** | **~1.2** | |
| **TOTAL (linear)** | **~2.5** | |

**Practical: ΔE_2000 inter-print 1.0-2.0 is achievable with good technique.** Reid's "sub-ΔE-2" goal is at the edge of what's physically possible for a hand-pulled mokuhanga edition.

## The phone-vs-DSLR question

Reid asked: can he use his phone or does he need a DSLR?

| Setup | Cost | Budget A (absolute ΔE) | Budget B (repeatability ΔE) | Verdict |
|---|---|---|---|---|
| **iPhone 14+/15 Pro, ProRAW, no CXP** | $0 (already owns) | 5-8 mean, 12+ max | 1.5-3 | Calibration too poor for v1 |
| **iPhone Pro, ProRAW, with CXP setup** | $200 (filters) | 3-5 mean, 8 max | 1-2 | Marginal; might work for non-edition use |
| **Mid-tier DSLR (Sony A6400 / Fuji X-T30 used)** | $500-800 used | 2-3 mean, 5 max | 1-2 | **Recommended for v1** |
| **Full-frame DSLR + sharp prime + CXP** | $1500+ | 1.5-2.5 mean, 4 max | 1-2 | Overkill; same budget B as mid-tier |

**Recommendation: mid-tier DSLR.** Used Sony A6400 + 30mm macro is ~$700 total. Combined with the $700 CXP setup, **total v1 calibration rig costs ~$1400.** This is a fraction of a single edition's value at gallery prices.

If Reid REALLY wants to start with the phone:
- Accept Budget A of ΔE 4-5 mean.
- The Budget B repeatability is fine (phone repeatability is good).
- Edition-print absolute color may be off from Reid's intent, but each of the 10 prints will be consistent.
- Upgrade path is clean — same MCP API surface, just better data when he gets the DSLR.

## What ΔE 2 looks like

Reference points for calibration:
- **ΔE 1**: not perceivable by most observers in side-by-side comparison.
- **ΔE 2**: perceivable in side-by-side by ~50% of observers; not perceivable when separated.
- **ΔE 3**: perceivable by most observers side-by-side.
- **ΔE 5**: clearly different colors.
- **ΔE 10**: obviously different.

For a fine-art edition: ΔE < 2 between prints means **the prints look identical**. ΔE 2-4 means **a careful collector might notice** if they had two prints next to each other. ΔE > 5 means the edition is inconsistent.

## Why "ΔE 2 across edition of 10" is the right bar

Reid set this number from his fine-art instincts, but it's also defensible mathematically:
1. **Below the perceptibility threshold for non-adjacent prints** — collectors typically own one print, not the whole edition.
2. **Within published manufacturer specs for fine-art reproduction** — Crane Museo papers + Epson Signature Worthy targets ΔE < 2 for ICC profile compliance.
3. **At the edge of what hand-pulled mokuhanga can physically achieve** — pushing harder would require mechanized printing, which violates the medium.

## What V1 cannot promise

- **Absolute ΔE < 2 against a spectrophotometer reading.** This needs an i1Pro 3 ($1500+) or equivalent. V1 is a camera-based approximation.
- **Wide-gamut pigments accurately captured.** Cadmium reds, ultramarine blues at full saturation may be outside the sRGB gamut a 3-channel camera can capture. Visible to the eye, lost in the YAML.
- **Spectral accuracy for tier-3 K-M.** V1 is 3-channel. T3 in chuck-mcp needs 36-wavelength. Calibration MCP v2 will add this; v1 explicitly does not.
- **Aging / fading prediction.** V1 captures current state; doesn't model future drift. The drift detection strategy is reactive, not predictive.

## What V1 CAN promise (the contract)

1. **Per-pigment YAML with Lab values that match printed swatches to mean ΔE 2-3.** Good enough for t1 Mixbox and t2 LUT tiers.
2. **Inter-print repeatability of ΔE 1-2** on a printed edition, given consistent technique.
3. **Drift detection** that catches material changes within ΔE 2-3 of occurrence.
4. **Re-calibration workflow** that completes in ~30 minutes per pigment.
5. **MCP API** that lets the rest of chuck-mcp v2 use pigment data without knowing how it was measured.

## Validation milestones for v1 acceptance

Before shipping, the calibration MCP must demonstrate:

1. **Synthetic round-trip test** (in code): `km_inverse_two_substrate` recovers K, S from synthetic R_W, R_B with error < 1%.
2. **ColorChecker re-prediction**: photograph the ColorChecker, fit CCM, predict each patch's Lab from RGB. Mean ΔE_00 fit residual < 2, max < 4.
3. **Real-pigment cross-validation**: Reid picks ONE of his stable pigments. Bootstrap it from photos (no spectro). Then independently measure the same pigment with a borrowed i1Pro at RISD. Compare YAML predictions to spectro readings. Mean ΔE_00 < 4, max < 7. This is the "sanity check" against a real reference.
4. **Edition print test**: print 10 copies using a 3-pigment forward render from the YAML. Measure inter-print ΔE. Max should be < 2.5. Mean should be < 1.5.
5. **Drift simulation**: artificially edit a baseline YAML (shift Lab by ΔE 3.5). Drift-check should BLOCK. Edit by ΔE 1.0, should PASS.

## Sources

- [Mirjalili 2019 — Color difference for non-separated samples](https://arxiv.org/abs/1904.11293) — adjacent-swatch ΔE perception
- [BabelColor — 30-chart averaging](https://babelcolor.com/colorchecker-2.htm) — chart-to-chart variation baseline
- [npj Heritage Science 2021 — limits of color accuracy](https://www.nature.com/articles/s40494-021-00536-x)
- [Finlayson 2015 — Root-polynomial color correction](https://pubmed.ncbi.nlm.nih.gov/25769139/) — CCM fit residuals
- Imatest documentation on JND thresholds
