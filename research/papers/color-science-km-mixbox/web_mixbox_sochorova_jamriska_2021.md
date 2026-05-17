# Practical Pigment Mixing for Digital Painting (Mixbox)

authors: Sarka Sochorova, Ondrej Jamriska (Czech Technical University in Prague + Secret Weapons)
venue: ACM Transactions on Graphics 40(6), Article 234 (December 2021). Proceedings of SIGGRAPH Asia 2021.
DOI: 10.1145/3478513.3480549
url: https://scrtwpns.com/mixbox.pdf
project: https://scrtwpns.com/mixbox/ ; https://github.com/scrtwpns/mixbox ; https://github.com/scrtwpns/pigment-mixing
relevance: This IS the t1 tier of chuck-mcp. Exact reference for what the project currently does. The full technical method below is what we are explicitly *not* using for the spectral t3 tier — we keep this as the fallback / fast path. Critically: Mixbox itself is a K-M *mixing-in-binder* model, not a layering model. Mokuhanga is layering. So even though Mixbox uses K-M coefficients, it answers the wrong physical question for our use case.

## Technical summary (3 paragraphs)

Mixbox implements full Kubelka-Munk pigment mixing inside painting software while preserving the RGB-in/RGB-out interface that every painting program assumes. Their key trick is a 7-dimensional latent representation `z = [c1 c2 c3 c4 r_R r_G r_B]^T`: four pigment concentrations summing to 1 (Phthalo Blue PB15:4, Hansa Yellow PY73, Quinacridone Magenta PR122, Titanium White PW6 — all Golden Artist acrylic paints) plus a 3-channel additive RGB residual that captures colors outside the achievable pigment-mixture gamut. The encoder `F(RGB) = [unmix(RGB); RGB - mix(unmix(RGB))]` solves a constrained nonlinear least-squares problem via L-BFGS-B with automatic differentiation; the decoder `G(z) = mix(c) + r` runs the forward K-M pipeline on the concentrations and adds back the residual. Linear interpolation in latent space, decoded back to RGB, is the `kmerp` operator that replaces RGB `lerp` everywhere in a paint program.

The forward `mix(c)` pipeline goes: K_mix(c, lambda) = sum_i c_i K_i(lambda) and S_mix(c, lambda) = sum_i c_i S_i(lambda) (Duncan 1940 linear K and S in concentration); then K-M infinite-thickness reflectance R_mix = 1 + K/S - sqrt((K/S)^2 + 2 K/S) per wavelength; Saunderson 1942 surface correction R'_mix = (1-k1)(1-k2) R_mix / (1 - k2 R_mix); CIE 1931 standard observer integration with D65 illuminant over 380-750 nm at 36 sample wavelengths (10 nm spacing); finally XYZ-to-sRGB matrix and normalization by Y_D65. To stop the four-pigment gamut from poking outside the RGB cube (which would break invertibility), they solve a secondary optimization (eq. 15) that perturbs the K and S of each primary to find surrogate pigments Q* whose mixtures stay inside sRGB while being perceptually close (Oklab distance) to the original P*.

Runtime is made fast by precomputing two 256^3 lookup tables: one for `unmix` (the encoder) and one for `mix` (the decoder). Each table stores 3 of the 4 concentrations (the fourth is implicit because they sum to 1), 8-bit-quantized, total 48 MB per table = 96 MB in memory. On disk the tables are packed as two 4096x4096 PNGs lossless-compressed, totaling 7 MB — that's the famous `mixbox_lut.png`. At runtime, F and G reduce to two/one trilinear texture lookups plus a subtraction/addition, fast enough for real-time painting. The K and S coefficients of the four primaries are sourced from the Berns 2016 Artist Paint Spectral Database; Saunderson constants k1, k2 are from Okumura 2005.

## Key equations (verbatim from paper)

```
Eq. 1 (Duncan 1940, linear K and S in c):
    K_mix(c, lambda) = sum_{i=1..N} c_i K_i(lambda)
    S_mix(c, lambda) = sum_{i=1..N} c_i S_i(lambda)

Eq. 2 (K-M reflectance at infinite thickness):
    R_mix(c, lambda) = 1 + K_mix/S_mix - sqrt( (K_mix/S_mix)^2 + 2 K_mix/S_mix )

Eq. 6 (Saunderson 1942 surface correction):
    R'_mix(c, lambda) = (1 - k1)(1 - k2) R_mix(c, lambda) / (1 - k2 R_mix(c, lambda))

Eq. 7 (sRGB from XYZ with D65 normalization):
    [R G B]^T = (1/Y_D65) * M_sRGB * [X(c) Y(c) Z(c)]^T
    where M_sRGB has rows (3.2406, -1.5372, -0.4986), (-0.9689, 1.8758, 0.0415), (0.0557, -0.2040, 1.0570)

Latent encoder / decoder (Eqns. 12-13):
    F(RGB) = [unmix(RGB); RGB - mix(unmix(RGB))]
    G([c; r]) = mix(c) + r

Mixing operator (Eq. 14):
    kmerp(RGB1, RGB2, t) = G( (1-t) F(RGB1) + t F(RGB2) )

Surrogate pigment optimization (Eq. 15):
    argmin_Q E_push(Q) + alpha * E_pull(Q, P*)
    s.t. K(lambda) > 0 and S(lambda) > 0 for all (K, S) in Q
```

## Why this is the WRONG model for mokuhanga (per ADR-0002)

Mixbox's forward model is `mix one batch of paint with N pigments dispersed in a single binder`. The K and S coefficients are summed linearly (Duncan 1940) because all pigments share the same wet binder and are co-mixed before drying. Then K-M reflectance is computed *once* on the resulting K_mix and S_mix.

Mokuhanga is not this. Mokuhanga is: pigment 1 + starch paste -> brush -> woodblock -> press onto washi -> DRY. Then pigment 2 + paste -> brush -> separate block -> press onto same washi -> DRY. Each impression is a thin translucent film with its own K, S, and thickness X. The optical model is RECURSIVE: light enters the top film, partially absorbs/scatters, hits the next film down, repeats, eventually hits the washi substrate, reflects back, repeats the climb. This is exactly the multilayer K-M / Hebert-Hersch transfer matrix case from arxiv 2603.09139 — NOT the in-binder mixing case Mixbox solves.

## Implication for chuck-mcp tier 3

- Keep Mixbox as the t1 fast path for previewing palette mixes within a single block (e.g., bokashi gradients where two pigments ARE co-mixed on the block).
- For overprint stacks: implement multilayer K-M two-flux recursion per wavelength. Each impression is its own layer with measured (K_i, S_i, X_i). Compose with Hebert-Hersch transfer matrices.
- Borrow Mixbox's measurement/database approach: 36 wavelengths over 380-750 nm at 10 nm spacing is the right grid for an 8-channel spectral fit. (Berns 2016 Artist Paint Spectral Database is a direct source.)
- Borrow Saunderson correction eq. 6 verbatim — applies to mokuhanga top surface (washi sizing -> air interface) too.
- DO NOT borrow Mixbox's `K_mix = sum c_i K_i` linear-in-concentration model for inter-layer interactions. Layers don't mix; they stack.

## Citations to add to ADR-0002

- Duncan 1940 — linear K and S in concentration (the assumption Mixbox depends on)
- Saunderson 1942 — surface reflection correction
- Berns 2016 — Artist Paint Spectral Database (source of K, S for 4 Golden acrylics; relevant for getting baseline mineral pigment K, S)
- Okumura 2005 — Saunderson reflectance constants k1, k2
- Haase & Meyer 1992 — first K-M in computer graphics
- Curtis et al. 1997 — Watercolor simulation (CG classic)
- Baxter et al. 2004 — K-M painting system
- Hebert & Hersch (referenced from arxiv 2603.09139) — transfer matrix multilayer compositional framework
