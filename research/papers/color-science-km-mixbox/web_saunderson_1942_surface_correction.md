# Calculation of the Color of Pigmented Plastics (Saunderson 1942)

authors: J. L. Saunderson (Dow Chemical Company)
venue: Journal of the Optical Society of America 32(12), 727-736 (December 1942)
DOI: 10.1364/JOSA.32.000727
url: https://opg.optica.org/josa/abstract.cfm?uri=josa-32-12-727
also archived: https://annas-archive.gd/scidb/10.1364/JOSA.32.000727 (downloaded as ehtml shell, paper body not extractable from this source — see NOTES.md)
relevance: Introduces the Saunderson correction — the surface-reflection patch that converts K-M's "internal" reflectance R to externally measured / observed reflectance R_e at a refractive index discontinuity (paint-to-air interface). Required for any predictive K-M color rendering on real-world artifacts. Chuck-mcp's t3 tier MUST apply this between the topmost mokuhanga impression and air, and arguably between washi sizing and air if the substrate-air interface is not handled by the substrate reflectance measurement.

## Technical summary (3 paragraphs)

K-M theory derives reflectance from absorption K and scattering S inside a coating, but the derivation assumes a refractive-index-matched medium throughout — no interface effects. In reality, a paint film (or any pigmented translucent layer) sits on a substrate with the top surface against air. The refractive index jump from air (n approx 1.0) to paint binder (n approx 1.5) creates two distinct Fresnel reflection effects: a specular front-surface reflection of incident light (parameter k1, the externally observable gloss), and an internal Fresnel reflection of diffuse light trying to escape from the inside back to air (parameter k2, which is typically much larger — about 0.4-0.6 for n=1.5 because internally diffuse light hits the interface at all angles, with much more total-internal-reflection than the collimated incident case).

The Saunderson 1942 correction equation relates externally measured reflectance R_T (or R_e, R_obs) to the "internal" K-M reflectance R_i:

```
R_T = k1 + (1 - k1)(1 - k2) R_i / (1 - k2 R_i)
```

The first term k1 is the directly reflected gloss off the top surface (typically 0.04 for diffuse illumination on a matte film, up to ~0.08 for collimated; absent for matte / integrating-sphere geometry). The numerator (1 - k1)(1 - k2) R_i represents light that successfully refracted INTO the film, scattered around inside per K-M, and refracted back OUT. The denominator (1 - k2 R_i) is the geometric series for multiple internal bounces off the top interface before escape. As Curtis et al. 1997 note, this correction was deferred in early CG K-M work; Mixbox 2021 explicitly includes it.

The inverse operation is required when you have a measured external R_T and want the internal R_i to feed into K-M layer composition. Solving Eq. (1) for R_i:

```
R_i = (R_T - k1) / ( (1 - k1)(1 - k2) + k2 (R_T - k1) )
```

For chuck-mcp this matters in two specific places. First, the topmost mokuhanga impression must apply Saunderson when rendering to screen, because we are predicting what the print looks like to the viewer in air. Second, when ingesting measured plate proofs (spectrophotometer or color-calibrated camera), the measured R_T must be back-converted to R_i before the t3 K-M stack solver fits per-layer K and S.

## Saunderson equation forms

```
Forward (internal -> external):
    R_e = k1 + (1 - k1)(1 - k2) R_i / (1 - k2 R_i)         (Saunderson 1942 eq.)

Inverse (external -> internal):
    R_i = (R_e - k1) / ((1 - k1)(1 - k2) + k2 (R_e - k1))

Typical constants for paint-on-air (n_paint = 1.5, n_air = 1.0):
    k1 ~ 0.04 (diffuse illumination + diffuse view, matte surface)
    k1 ~ 0.08 (collimated illumination, diffuse view)
    k1 ~ 0.0  (integrating-sphere geometry that excludes specular)
    k2 ~ 0.4 - 0.6 (diffuse internal light at n=1.5, computed by Fresnel
                    averaged over hemisphere; typically taken as 0.4 or 0.6
                    depending on whether one uses the simple 0.6 estimate
                    from Saunderson's paper or the more rigorous integral)
```

The Mixbox paper uses Okumura 2005 as the source for k1 and k2 values (the actual values used in `mixbox_lut.png` are not published directly but can be reverse-engineered).

## Why this equation has the geometric-series form

Light incident on the top of the paint film:
1. Fraction k1 reflects directly off the air-paint surface (gloss).
2. Fraction (1 - k1) refracts into the paint.
3. Inside the film, K-M gives diffuse reflectance R_i back upward.
4. Of this returning diffuse light, fraction (1 - k2) escapes through the top into air (and is observed), fraction k2 bounces back DOWN into the film, then back up at R_i again, then escapes at (1 - k2), etc.

Summing the infinite geometric series:
```
Observed light = k1 + (1-k1) R_i (1-k2) * sum_{n=0..inf} (k2 R_i)^n
              = k1 + (1-k1)(1-k2) R_i / (1 - k2 R_i)
```

This is identical in structure to the Kubelka 2-layer compositing rule (`R1 + T1^2 R2 / (1 - R1 R2)`), confirming the Hebert-Hersch view that Saunderson IS just another layer in the transfer-matrix algebra (one with R=k1, T=(1-k1), and back-reflectance k2). This is the unification observed in arxiv 2603.09139.

## Caveats

- For matte / rough surfaces, k1 is reduced and the gloss is spread; full BRDF modeling needed for highly glossy varnish layers (not relevant for mokuhanga which is matte).
- k2 depends on the angular distribution of internally diffuse light; the value 0.6 in the original paper is a simplification.
- For washi paper substrates with significant sizing variability, the effective k2 may be position-dependent. This is a calibration issue for chuck-mcp.

## Citations to add to ADR-0002

- Saunderson 1942 directly
- Okumura 2005 (for measured k1, k2 values used in Mixbox)
- Walsh ~1920s (origin of the surface correction concept per the search results — predates Saunderson by 20 years)
- Hebert & Hersch transfer-matrix framework (referenced from 2603.09139) — generalizes Saunderson as just another layer in the stack
