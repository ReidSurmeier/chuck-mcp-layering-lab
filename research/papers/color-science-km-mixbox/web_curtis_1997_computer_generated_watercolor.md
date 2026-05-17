# Computer-Generated Watercolor (Curtis et al. 1997)

authors: Cassidy J. Curtis, Sean E. Anderson, Joshua E. Seims (U. Washington); Kurt W. Fleischer (Pixar); David H. Salesin (U. Washington)
venue: SIGGRAPH '97 Proceedings, pp. 421-430
url: https://grail.cs.washington.edu/projects/watercolor/
DOI: 10.1145/258734.258896
relevance: THE foundational computer graphics paper on K-M optical compositing of stacked translucent layers — exactly the mokuhanga overprint case. Curtis et al. use Kubelka's layer-compositing equations (NOT Mixbox's binder mixing) to stack N glazes. This is the closest existing prior art to chuck-mcp's t3 tier and the algorithm chuck-mcp should adapt.

## Technical summary (3 paragraphs)

Curtis et al. simulate watercolor as an ordered set of translucent glazes, each created independently by a shallow-water fluid simulation, then optically composited via the Kubelka-Munk model. For each glaze layer they model two pigment quantities: g_k (concentration of pigment in the shallow-water layer above the paper) and d_k (concentration deposited on the paper). The total thickness x_k of a layer is the sum of pigment thicknesses. Each pigment has 3-channel RGB K (absorption) and S (scattering) coefficients. Critically: when MULTIPLE pigments are co-mixed in a single layer, K and S of the layer are weighted sums of K_k and S_k in proportion to relative thickness x_k — this is Duncan 1940 / Mixbox-style per-layer mixing. When SEPARATE layers (glazes) are stacked, the layers compose via Kubelka's transfer-style equations (this is what chuck-mcp needs).

The single-layer reflectance R and transmittance T from K-M theory are computed via:
- `R = sinh(b S x) / c`
- `T = b / c`
- where `c = a sinh(b S x) + b cosh(b S x)`
- and `a = 1 + K/S`, `b = sqrt(a^2 - 1)`

For inverse problem: given user-painted "appearance over white" R_w and "appearance over black" R_b for a unit-thickness swatch, they solve `a = (1/2)(R_w + (R_b - R_w + 1)/R_b)`, `b = sqrt(a^2 - 1)`, `S = (1/b) coth^-1( (b^2 - (a - R_w)(a - 1)) / (b (1 - R_w)) )`, `K = S(a - 1)`. Requirement `0 < R_b < R_w < 1` per channel. This is the practical UI shortcut chuck-mcp could adopt to bootstrap K and S without spectrophotometry.

Two abutting layers with reflectances R1, R2 and transmittances T1, T2 compose as:
- `R = R1 + (T1^2 * R2) / (1 - R1 * R2)`
- `T = (T1 * T2) / (1 - R1 * R2)`

This is the canonical 2-layer Kubelka compositing rule, derived from summing the infinite geometric series of internal bounces (a special case of Hebert-Hersch transfer-matrix algebra). It is associative — N glazes compose by repeated application. This is the algorithm chuck-mcp's t3 tier should implement per wavelength.

## Key equations (verbatim from paper)

```
Single-layer K-M (with thickness x, coefficients K, S):
    a = 1 + K/S
    b = sqrt(a^2 - 1)
    c = a sinh(b S x) + b cosh(b S x)
    R = sinh(b S x) / c              (single-layer reflectance)
    T = b / c                        (single-layer transmittance)

Two-layer optical compositing (layer 1 ABOVE layer 2):
    R_total = R1 + (T1^2 * R2) / (1 - R1 * R2)
    T_total = (T1 * T2) / (1 - R1 * R2)

Inverse (given user-painted R_w over white, R_b over black at unit thickness):
    a = (1/2) * (R_w + (R_b - R_w + 1)/R_b)
    b = sqrt(a^2 - 1)
    S = (1/b) * coth^-1( (b^2 - (a - R_w)(a - 1)) / (b * (1 - R_w)) )
    K = S * (a - 1)

Multi-pigment-per-layer (Duncan 1940):
    K_layer = sum_k (x_k / x_total) * K_k
    S_layer = sum_k (x_k / x_total) * S_k
    x_total = sum_k x_k
```

## Pigment table (verbatim from Figure 5)

Each pigment has 3-channel K_r, K_g, K_b and S_r, S_g, S_b. Examples from the paper:

| Pigment | K_r | K_g | K_b | S_r | S_g | S_b |
|---|---|---|---|---|---|---|
| Quinacridone Rose | 0.22 | 1.47 | 0.57 | 0.05 | 0.003 | 0.03 |
| Indian Red | 0.46 | 1.07 | 1.50 | 1.28 | 0.38 | 0.21 |
| Cadmium Yellow | 0.10 | 0.36 | 3.45 | 0.97 | 0.65 | 0.007 |
| Hookers Green | 1.62 | 0.61 | 1.64 | 0.01 | 0.012 | 0.003 |
| Cerulean Blue | 1.52 | 0.32 | 0.25 | 0.06 | 0.26 | 0.40 |
| Burnt Umber | 0.74 | 1.54 | 2.10 | 0.09 | 0.09 | 0.004 |
| Cadmium Red | 0.14 | 1.08 | 1.68 | 0.77 | 0.015 | 0.018 |
| Brilliant Orange | 0.13 | 0.81 | 3.45 | 0.005 | 0.009 | 0.007 |

These are RGB-quantized — not spectral. For chuck-mcp's 8-channel spectral t3, we want per-wavelength K(lambda) and S(lambda) sampled at 10 nm over 380-750 nm (36 wavelengths, matching Mixbox/Berns).

## Explicit assumptions Curtis et al. call out

1. All colorant layers are immersed in mediums of the same refractive index — Saunderson correction is NOT applied. Curtis notes "a fairly simple correction term has been proposed [18]" — that's Saunderson 1942.
2. Pigments have nearly horizontal orientation (i.e., scattering is approximately isotropic about the vertical axis). Caveat: metallic / mica pigments violate this, and so do some mokuhanga pigments (mica, gold leaf usage in Pace Editions work).
3. Lighting and viewing are approximately diffuse — same K-M assumption.

## Why this is the right starting algorithm for chuck-mcp t3

- It is exactly the multilayer-translucent-on-paper case.
- It handles N layers via recursive 2-layer compositing.
- Per-layer K and S can be either Duncan-summed (Mixbox-style) for co-mixed bokashi or used standalone for single-pigment plates.
- Already validated on watercolor — visually equivalent regime to mokuhanga.
- Missing pieces chuck-mcp must add: (a) Saunderson correction at the top air/sizing interface, (b) per-wavelength spectral evaluation (36 lambdas not 3 RGB channels), (c) measured rather than user-inverted K, S from Pace Editions plate proofs, (d) explicit washi substrate reflectance R_paper(lambda) as the bottom boundary of the stack.

## Citations to add to ADR-0002

- Kubelka 1948 (Curtis ref [20]) — original layer compositing equations follow-up paper
- Kubelka 1954 (Curtis ref [21]) — the 1954 layering equations
- Haase & Meyer 1992 (Curtis ref [14]) — first K-M in computer graphics
- Dorsey & Hanrahan — patina layer modeling (Curtis ref [7]) — same K-M layering trick applied to weathered metal
- Curtis et al. 1997 itself — directly cited
