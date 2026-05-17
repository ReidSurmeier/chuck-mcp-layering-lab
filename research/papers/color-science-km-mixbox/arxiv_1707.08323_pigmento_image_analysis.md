# Pigmento: Pigment-Based Image Analysis and Editing

authors: Jianchao Tan, Stephen DiVerdi, Jingwan Lu, Yotam Gingold (George Mason University + Adobe Research)
arxiv: 1707.08323v3 [cs.GR] 19 Jul 2018 (final TVCG version)
relevance: Inverse problem solver that recovers per-pixel pigment concentrations from a single RGB image of a painting, using multispectral K-M absorption AND scattering parameters. Per-pixel mixture of N pigments, with Kubelka-Munk + Duncan 1940 mixing + layering. This is the closest existing system to chuck-mcp's "given Pace Editions proof scan, recover the per-block pigment recipe" inverse problem.

## Abstract

The colorful appearance of a physical painting is determined by the distribution of paint pigments across the canvas, which we model as a per-pixel mixture of a small number of pigments with multispectral absorption and scattering coefficients. We present an algorithm to efficiently recover this structure from an RGB image, yielding a plausible set of pigments and a low RGB reconstruction error. Using our decomposition, we repose standard digital image editing operations as operations in pigment space rather than RGB, with interestingly novel results: tonal adjustments, selection masking, cut-copy-paste, recoloring, palette summarization, and edge enhancement.

## Why this matters for chuck-mcp

1. **Inverse problem template**: Pigmento solves exactly the inverse of chuck-mcp's forward render — given the appearance of a finished painting in RGB, recover the K and S coefficients of N pigments and their per-pixel mixture weights. Chuck-mcp needs this when ingesting Pace Editions plate proofs to calibrate the spectral t3 model.
2. **Multispectral, not RGB**: Pigmento explicitly works at 33 wavelengths (380-700 nm, 10 nm step) — matches Mixbox's 36-wavelength grid and the Berns 2016 database sampling. This is the right resolution for chuck-mcp's spectral fit.
3. **Single-layer-of-mixture vs. multi-layer-of-pure**: Pigmento models a single layer containing a mixture of N pigments. Chuck-mcp needs N stacked layers of (potentially) pure pigments. Different model — but the forward render uses the SAME single-layer K-M equation, just composed via Kubelka layer-compositing for the stack.
4. **Validated against ground truth**: Authors demonstrate ability to recover near-ground-truth pigments when they exist, while always producing plausible reconstructions even when input is noisy.

## Key equations (verbatim)

The forward model per-pixel is a single layer of homogeneous pigment atop a substrate:

```
r = km(k, xi, t)
```

where:
- k is the vector (over wavelengths lambda) of the pigment's K-M parameters (a = absorption per unit thickness, s = scattering per unit thickness)
- xi is the substrate reflectance (per wavelength)
- t is the thickness
- r is the resulting layer reflectance (per wavelength)

For a mixture with concentrations c = [c_1, ..., c_N], the layer's K-M parameters are linear combinations (Duncan 1940):
```
K_layer(lambda) = sum_i c_i K_i(lambda)
S_layer(lambda) = sum_i c_i S_i(lambda)
```

Then layer reflectance follows from the canonical Kubelka closed-form (same as Curtis 1997) with the (K_layer, S_layer, t, xi) inputs.

To convert spectral reflectance to displayed sRGB:
- multiply by D65 illuminant
- integrate against CIE 1931 standard observer XYZ
- transform XYZ to linear RGB, gamma-correct to sRGB

## Wavelength sampling

33 wavelengths from 380 to 700 nm at 10 nm step. (Note: Mixbox uses 380-750 nm at 10 nm = 36 wavelengths. Berns 2016 uses 380-750 nm at 10 nm = 38 wavelengths. Chuck-mcp should standardize on the 380-750 nm range for compatibility.)

## Substrate reflectance handling

Pigmento explicitly takes xi (substrate spectral reflectance) as an input parameter. For chuck-mcp this is the washi paper reflectance R_paper(lambda), which is the bottom boundary of the t3 K-M stack and needs to be measured once per paper batch (or per Emma proof if multiple paper types are used).

## Citations to add to ADR-0002

- Pigmento itself (arxiv 1707.08323) — for inverse pigment-extraction algorithm
- IMPaSTo (Baxter et al. ref [5] in Pigmento) — earlier real-time K-M paint system
- Tan et al. layer extraction (ref [10]) — alpha-style translucent layers
- Lin et al. layer extraction (ref [11]) — additive layer extraction
- Aksoy layer extraction — additive mixing layer recovery
- Berns et al. (ref [24-26]) — multispectral painting measurements + reduced-dimension reflectance fitting
- Zhao et al. (ref [27]) — improved reflectance reconstructions
- Pelagotti / Cosentino (refs [28, 29]) — multispectral pigment identification
