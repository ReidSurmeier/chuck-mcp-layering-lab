# Prediction Model for Semitransparent Watercolor Pigment Mixtures Using Deep Learning with a Dataset of Transmittance and Reflectance

authors: Mei-Yun Chen, Ya-Bo Huang, Sheng-Ping Chang, Ming Ouhyoung (National Taiwan University)
arxiv: 1904.00275v1 [cs.LG] 30 Mar 2019
relevance: Measures BOTH transmittance AND reflectance of 13 primary watercolor pigments at 12 different concentrations each on Canson Ca grain paper, plus 780 two-pigment mixture spectra. This is the canonical example of the empirical-LUT (t2) data-collection protocol chuck-mcp needs to replicate for mokuhanga pigments — but with transmittance included, which is crucial for stacked layers (T appears in Kubelka two-layer composition formula).

## Abstract

Learning color mixing is difficult for novice painters. In order to support novice painters in learning color mixing, we propose a prediction model for semitransparent pigment mixtures and use its prediction results to create a Smart Palette system. Such a system is constructed by first building a watercolor dataset with two types of color mixing data, indicated by transmittance and reflectance: incrementation of the same primary pigment and a mixture of two different pigments. Next, we apply the collected data to a deep neural network to train a model for predicting the results of semitransparent pigment mixtures. Finally, we constructed a Smart Palette that provides easily-followable instructions on mixing a target color with two primary pigments in real life.

When evaluating the pigment mixtures produced by the aforementioned model against ground truth, 83% of the test set registered a color distance of dE*_ab < 5. dE*_ab above 5 is where average observers start determining that the colors in comparison are two different colors.

## Why this matters for chuck-mcp

1. **Transmittance-measured dataset (NTU WPSM)**: They explicitly measure T(lambda) — not just R(lambda) — for each pigment at multiple concentrations. T is required for the 2-layer Kubelka composition rule (Curtis 1997) and is what Mixbox/Pigmento omit. Chuck-mcp's t3 tier MUST measure T as well as R for each pigment, on paper held above black for R + held to light for T.
2. **Equipment**: OTO SD1220 spectrometer, K1 light source. Practical bill-of-materials for the chuck-mcp data collection rig.
3. **Substrate is Canson Ca grain** (specific watercolor paper). Chuck-mcp should swap in Awagami or Iwano kozo washi (typical Pace Editions paper for mokuhanga) and measure substrate reflectance R_paper(lambda) and transmittance T_paper(lambda) on that exact paper.
4. **DNN as surrogate**: Inputs are (T_A, R_A, T_B, R_B, R_substrate, mix ratio, ...) — DNN learns the mixing function. This is one valid t2 design (alongside arxiv 2409.04558's gated-residual MLP) for chuck-mcp's empirical LUT tier.
5. **Validation metric**: dE*_ab (CIE 1976) with threshold 5 for "different color". A coarser metric than CIEDE2000 — chuck-mcp should use CIEDE2000 for validation (current best practice; see arxiv 1904.11293).

## Key methodology

NTU WPSM (Watercolor Pigment Spectral Mixture) dataset:
- 13 primary pigments
- 12 concentration levels per pigment
- 780 two-pigment mixture combinations
- Spectral T and R measured by OTO SD1220 spectrometer
- All on Canson Ca grain paper substrate

DNN inputs (per-mixture sample):
1. T of pigment A (per wavelength)
2. R of pigment A (per wavelength)
3. T of pigment B (per wavelength)
4. R of pigment B (per wavelength)
5. R of substrate (white paper)
6. concentrations / mixing ratios

DNN output: T and R of resulting mixture, which can then be converted to RGB via XYZ -> sRGB.

## Caveats / limitations for chuck-mcp adaptation

- They model only TWO co-mixed pigments per layer (2-pigment palette dish). Mokuhanga uses one pigment per block. Chuck-mcp needs SINGLE-pigment-per-layer measurements + a stacking model on top.
- They co-mix in the dish (i.e., wet binder mixing) rather than overprint. The dataset does NOT include layered measurements (block A printed, dried, then block B printed on top). This is the gap chuck-mcp must fill with custom measurements of mokuhanga overprints.

## Citations to add to ADR-0002

- This paper (arxiv 1904.00275) for the T+R measurement protocol
- Xu et al. (ref [18] in this paper) — earlier neural mixing using single-constant K-M with S=1 assumption (a chuck-mcp alternative for first-pass calibration)
- Haase & Meyer (ref [5]) — first K-M for computer graphics
