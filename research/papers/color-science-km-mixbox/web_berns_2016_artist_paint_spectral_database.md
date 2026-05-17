# Artist Paint Spectral Database (Berns 2016)

author: Roy S. Berns (Rochester Institute of Technology, Munsell Color Science Lab / Studio for Scientific Imaging and Archiving of Cultural Heritage)
venue: CIC24 (Color and Imaging Conference 24), 2016
url: https://www.rit.edu/science/sites/rit.edu.science/files/2019-03/ArtistSpectralDatabase.pdf
update: 2022 expansion paper "Artist Acrylic Paint Spectral, Colorimetric, and Image Dataset" (Archiving 2022)
relevance: The spectral K and S coefficient database that Mixbox draws its 4 primary pigments from. 19 Golden Heavy Body acrylic pigments characterized via masstone-tint method with Saunderson correction. Gives the exact protocol chuck-mcp should follow to characterize mokuhanga pigments. Provides published Saunderson constants k1=0.03 (collimated), k2=0.65 (diffuse).

## Technical summary (3 paragraphs)

Berns measured 19 Golden Heavy Body acrylic dispersion paints with a Macbeth MS7000 integrating-sphere spectrophotometer (SPIN = specular-included) at 380-750 nm in 10 nm increments — yielding 38 wavelength samples per spectrum. Each paint was drawn down on Leneta Form 3B Opacity Charts (a standardized opacity test paper with black and white areas side-by-side) using a 0.006-inch drawdown bar. Each paint was measured in two forms: as a masstone (paint straight from the tube, opaque) and as a tint (paint mixed with Titanium White at a known concentration). This masstone-tint pair lets you back out both K(lambda) and S(lambda) for the pigment — a single masstone measurement alone is degenerate because K-M's K/S ratio determines R_inf but not the absolute thicknesses.

The opaque form of K-M was used for the forward model, with Saunderson surface correction applied to convert measured external reflectance R_T to internal R_i for fitting. The Saunderson constants are stated explicitly: k1 = 0.03 for collimated incidence, k2 = 0.65 for diffuse internal reflection, and k_instrument = 1.0 for SPIN geometry. (k_instrument = 0 for SPEX geometry, equivalent to "varnished" or glossy externalization.) These values are the consensus-best parameters for paint-on-air interfaces with n_paint ~ 1.5 and are derived from earlier work cited as ref [16].

To produce additional chromatic samples that fill in CIELAB hue gaps, Berns COMPUTATIONALLY mixed extra pairs (Phthalo Green Yellow Shade + Bismuth Vanadate Yellow, Phthalo Blue Green Shade + Phthalo Green Blue Shade, Quinacridone Magenta + Dioxazine Purple) at three ratios each. Final database contains 770 spectra spanning 23 hues + 1 gray scale, plus principal component eigenvectors (first 5 reconstruct the dataset with reasonable accuracy; first 3 are useful as approximation). Spectra include "tints" (mixed with white) and "tones" (mixed with bone black) at concentrations chosen to approximately uniformly sample CIELAB.

## Pigments characterized (19 Golden Heavy Body)

| Paint | C.I. # |
|---|---|
| Titanium White | PW 6 |
| Bone Black | PBk 9 |
| Bismuth Vanadate Yellow | PY 184 |
| Hansa Yellow Opaque | PY 74 |
| Diarylide Yellow | PY 83 |
| C.P. Cadmium Orange | PO 20 |
| Pyrrole Orange | PO 73 |
| C.P. Cadmium Red Light | PR 108 |
| Pyrrole Red | PR 254 |
| Quinacridone Red | PV 19 |
| Quinacridone Magenta | PR 122 |
| Dioxazine Purple | PV 23 |
| Ultramarine Blue | PB 29 |
| Cobalt Blue | PB 28 |
| Cerulean Blue, Chromium | PB 36:1 |
| Phthalo Blue (Red Shade) | PB 15:1 |
| Phthalo Blue (Green Shade) | PB 15:4 |
| Phthalo Green (Blue Shade) | PG 7 |
| Phthalo Green (Yellow Shade) | PG 36 |

Note: Mixbox uses 4 of these (PB 15:4, PY 73, PR 122, PW 6) — except PY 73 isn't in the Berns list (Hansa Yellow OPAQUE PY 74 is); Mixbox sources its specific PY 73 values from elsewhere in the Berns 2022 expansion, or measured them separately.

## Saunderson constants (canonical values from Berns)

```
k1 = 0.03   (collimated front-surface specular)
k2 = 0.65   (internal diffuse Fresnel for n=1.5 paint/air)
k_instrument = 1.0  (SPIN: specular-included integrating sphere)
k_instrument = 0    (SPEX or "varnished" mode)
```

## Wavelength grid

380-750 nm at 10 nm step = 38 samples. (Or 39 if 750 inclusive — depends on convention.)

## Measurement protocol for chuck-mcp adaptation

For each mokuhanga pigment:
1. Procure pigment + traditional starch paste at typical concentrations.
2. Print masstone on a black-and-white Leneta-equivalent test paper (or directly on Awagami/Iwano kozo washi held alternately over black and white substrates).
3. Print tint (50% pigment + 50% white pigment from same source, e.g., gofun shell white) on same substrate.
4. Measure spectral reflectance with an integrating-sphere spectrophotometer at 380-750 nm / 10 nm. (X-Rite eXact or similar; calibration to NIST standards required.)
5. Fit K(lambda), S(lambda) per pigment using the masstone-tint inversion (procedure cited in Berns ref [14]).
6. Apply Saunderson correction with k1 = 0.03, k2 = 0.65 (verify these are right for washi-air vs paint-air; the n ~ 1.5 assumption may need adjustment for paste-binder washi systems).

## Citations to add to ADR-0002

- Berns 2016 CIC24 — primary reference
- Berns 2022 Archiving — expanded dataset
- Mohammadi et al. — the masstone-tint method (Berns ref [14])
- Golden Artist Colors — paint manufacturer, source of pigment samples
