# Color Science Research Notes — Chuck MCP Layering Lab

Research compiled by agent COLOR-SCIENCE for swarm-1778962124344-s4cm4l on 2026-05-16.

Output folder contains 11 markdown artifacts (6 arxiv summaries + 5 canonical web references) covering Kubelka-Munk foundations, Saunderson correction, Mixbox, multilayer compositing, multispectral pigment analysis, CIE color difference, and spectral databases.

---

## (a) Top 5 papers ranked by relevance to t3_spectral K-M tier

| # | Artifact | Title | What it gives chuck-mcp |
|---|---|---|---|
| 1 | `web_curtis_1997_computer_generated_watercolor.md` | Curtis et al. 1997, SIGGRAPH | The canonical multilayer K-M optical compositing algorithm. Direct algorithm chuck-mcp's t3 should adopt: per-layer (K, S, x) -> (R, T) via K-M single-layer formulas, then stack via `R = R1 + T1^2 R2 / (1 - R1 R2)` and `T = T1 T2 / (1 - R1 R2)`. Validated on watercolor — same regime as mokuhanga. Includes pigment table with K_r/K_g/K_b/S_r/S_g/S_b values. |
| 2 | `arxiv_2603.09139_km_galerkin_rte.md` | Zeller 2026, arxiv 2603.09139 | Rigorous proof that multilayer K-M is a rank-2 Galerkin projection of the RTE. Establishes accuracy bound: error grows with reduced optical thickness `tau* = tau(1-g)`. Confirms that washi (highly diffuse, low g) + thin pigment films (low tau) is the regime where K-M is most accurate. Endorses Hebert-Hersch transfer-matrix framework as the rigorous extension. |
| 3 | `web_mixbox_sochorova_jamriska_2021.md` | Sochorova & Jamriska 2021, SIGGRAPH Asia | Mixbox is the current t1 tier. Full method here clarifies WHY it is the wrong model for mokuhanga overprint (it's a binder-mixing model, not a layering model). Saunderson correction equation (eq. 6) is reusable verbatim. Documents the 36-wavelength / 380-750 nm / 10 nm grid that chuck-mcp should standardize on. |
| 4 | `web_kubelka_munk_1931_westin_translation.md` | Kubelka & Munk 1931, original | The bedrock. Two-flux differential equations, infinite-thickness formula `R_inf = 1 + K/S - sqrt((K/S)^2 + 2 K/S)`, finite-thickness closed form. Glaze-limit equation `R = R_g exp(-2 s X)` is the closest analytic match to thin translucent mokuhanga films. The notation map (1931 s/r -> modern K/S) is documented. |
| 5 | `web_saunderson_1942_surface_correction.md` | Saunderson 1942, JOSA | The surface-reflection patch K-M needs at the paint/air boundary. Equation `R_e = k1 + (1 - k1)(1 - k2) R_i / (1 - k2 R_i)` with k1=0.03 (collimated specular) and k2=0.65 (diffuse internal Fresnel) for n=1.5. Required at the top of the mokuhanga stack for any predictive render. Inverse equation given for ingesting measured proofs. |

Honorable mentions:
- `web_berns_2016_artist_paint_spectral_database.md` — protocol + Saunderson constants + 19 measured pigments (Golden Heavy Body). Direct template for the chuck-mcp pigment characterization rig.
- `arxiv_1707.08323_pigmento_image_analysis.md` — inverse problem solver (recover pigment K, S, c from RGB) at 33 wavelengths. Complementary to the forward t3 tier.

---

## (b) Citations the project's `docs/adr/0002-overlay-not-mixing-3-tier-render.md` should add

### Foundational (must cite)

1. **Kubelka & Munk 1931** — `Zeitschrift fur Technische Physik` 12:593-601. English translation by Westin available at Cornell graphics group. This is the source of the K-M two-flux model that the entire spectral tier rests on.
2. **Saunderson 1942** — JOSA 32(12):727-736. DOI: 10.1364/JOSA.32.000727. The surface-reflection correction. Mandatory at any boundary with refractive-index discontinuity.
3. **Kubelka 1948, 1954** — the layer-compositing extensions. The 2-layer formula `R = R1 + T1^2 R2 / (1 - R1 R2)` is from these papers (Curtis 1997 refs [20, 21]).
4. **Duncan 1940** — the linearity-of-K-and-S-in-concentration result that lets `K_mix = sum c_i K_i`. Required when ANY layer contains co-mixed pigments (e.g., bokashi gradients).

### Algorithm references (must cite)

5. **Curtis et al. 1997** — `Computer-Generated Watercolor`, SIGGRAPH 97. The multilayer K-M optical compositing algorithm chuck-mcp's t3 should directly adapt.
6. **Sochorova & Jamriska 2021** — `Practical Pigment Mixing for Digital Painting`, ACM TOG 40(6) art. 234. The t1 reference; explains the latent space + LUT trick.
7. **Hebert & Hersch and collaborators (refs [4-8] in arxiv 2603.09139)** — transfer-matrix framework that unifies K-M, Saunderson, Williams-Clapper, and 4-flux models into one compositional algebra. The "right" production framework for the spectral tier.
8. **Zeller 2026** — `Geometric Realism Without Angular Resolution`, arxiv 2603.09139. The Galerkin-projection error bound that justifies trusting K-M for the chuck-mcp regime.

### Data and calibration references (must cite)

9. **Berns 2016 CIC24** — `Artist Paint Spectral Database`. Source of canonical Saunderson constants (k1=0.03, k2=0.65) and the masstone-tint K/S characterization protocol.
10. **Mudgett & Richards 1971** — RTE-to-K-M derivation; establishes `K = 2 sigma_a`. Required if any "principled" derivation of K, S from BRDF measurements is attempted.
11. **Sharma et al. 2005** — CIEDE2000 implementation notes. For perceptual color-difference validation.
12. **Mirjalili et al. 2019** — arxiv 1904.11293, dE_NS for adjacent-swatch perceptual color difference. Chuck-mcp's validation metric.

### Adjacent / "see also" references (useful but not strictly required)

13. **Pigmento (Tan et al. 2018)** — arxiv 1707.08323, the inverse problem solver. Future direction for auto-calibrating chuck-mcp's t3 from photos of Pace Editions proofs.
14. **Taufique & Messinger 2021** — arxiv 2104.04884, K/S-space linear unmixing on the Selden Map. Real historical-paper-substrate case study.
15. **NTU WPSM dataset (Chen et al. 2019)** — arxiv 1904.00275, T+R measurement protocol for the empirical t2 tier.
16. **Haase & Meyer 1992** — first K-M in computer graphics. Historical reference.
17. **Baxter et al. 2004 IMPaSTo** — real-time K-M paint system; alternative architecture for t2.

---

## (c) Open questions about applying K-M to washi paper substrate

### Q1: Does Saunderson correction with k1=0.03, k2=0.65 apply to washi-air, or is the effective interface paste-air?

The standard k1 and k2 are derived for paint binder with n ~ 1.5 against air. Mokuhanga uses starch paste (nori) as binder, dries thin, and the topmost surface a viewer sees is part paste, part exposed pigment particles, part washi fibers. The effective refractive index at the top boundary is probably lower than 1.5 — closer to 1.35-1.4 for a partially-fibrous starch interface. This would reduce both k1 and k2, with k2 affected more (it scales roughly as `((n-1)/(n+1))^2` integrated over angle). NEEDS EMPIRICAL VERIFICATION with a glossmeter and refractometer on Iwano/Awagami washi.

### Q2: Is washi opaque enough for the K-M model to assume a substrate reflectance R_paper(lambda) at the bottom boundary?

Washi is translucent — you can see through a single sheet of kozo against bright light. This means K-M's assumption that R_paper(lambda) absorbs the bottom-going flux is partially violated. There are three options: (a) measure washi spectral transmittance and explicitly model it as a SECOND substrate layer in the transfer-matrix stack, (b) for the Pace Editions Emma, mount the print on white backing and measure the COMBINED (washi + backing) reflectance as R_substrate, (c) model washi as a Kubelka-Munk layer itself with its own K_paper and S_paper, then attach a backing reflectance below. Option (c) is most rigorous and matches Hebert-Hersch transfer-matrix philosophy.

### Q3: Do mokuhanga pigments satisfy K-M's "small-particle isotropic-scattering" assumption?

Mineral pigments used in traditional mokuhanga — gunjo (azurite), shoenoogu (vermillion), gofun (oyster shell white), sumi (lampblack) — have widely varying particle sizes and morphologies. Some (sumi) are sub-micron and approximate isotropic scattering well. Others (gunjo, gofun) are flaky / plate-like and have strong forward-scattering anisotropy (high g, where Zeller 2026 says K-M error grows). Reid's HANDMADE pigments (per memory `project_woodblock_handmade_pigments.md`) are doubly off-baseline since they're not in any standard database. NEEDS MEASUREMENT — every Reid pigment requires its own (K, S) characterization on his actual paste and on his actual washi.

### Q4: How does the wet-into-dry overprint timing affect the optical stack?

If block A is printed and the washi is allowed to FULLY dry before block B is printed, the layers are optically distinct (refractive-index discontinuities between them are smoothed only by paste). If printed wet-into-wet (or with insufficient drying), pigments physically intermix at the interlayer interface, partially violating the discrete-layer assumption. Pace Editions documentation should be consulted for the actual drying-between-pulls protocol used on the Close Emma print. (132 pulls / 27 blocks suggests deliberate drying time.)

### Q5: Should we measure R AND T for chuck-mcp pigments, or just R?

Curtis 1997 needs T to do layer composition. Berns 2016 measures only R (masstone + tint over opaque substrate). For an OPAQUE single layer, T is implicit (=0). But for the THIN translucent mokuhanga films we care about, T is large and non-trivial. Chuck-mcp MUST measure T separately, either by transmittance through a swatch printed on clear acrylic, or by inverting the masstone-tint Berns method with the additional constraint of measured R over multiple known substrates. NTU WPSM (arxiv 1904.00275) measures T directly with a SD1220 spectrometer in transmission geometry — this is the protocol to copy.

### Q6: How many wavelength channels does chuck-mcp need? 8? 36? Spectral?

The task brief mentions "8-channel multispectral" as the t3 target. But all four reference systems (Mixbox, Pigmento, Berns, NTU WPSM) use 33-38 wavelengths at 10 nm. The reason: pigments have narrow spectral features (e.g., the rare-earth-pigment dip at ~590 nm, or cadmium yellows' sharp transitions) that are ALIASED by 8-channel sampling. RECOMMENDATION: use 36 wavelengths internally (380-750 nm at 10 nm, matching Mixbox), but expose 8-channel output for downstream consumers via PCA projection (Berns 2016 demonstrates 5 PCs reconstruct paint spectra well; 8 is comfortably enough). The "8-channel" in the brief is probably about display / file-format channels, not internal computation.

### Q7: Should the chuck-mcp validation metric be dE_NS or dE_2000?

Standard practice is CIEDE2000, but Mirjalili et al. 2019 (arxiv 1904.11293) shows that for ADJACENT printed swatches (the Close Emma case) the lightness weighting in CIEDE2000 over-penalizes low-dE cases and under-penalizes high-dE cases. Chuck-mcp's QA loop should report BOTH dE_2000 (for compatibility with prior literature) AND dE_NS (for the actually-correct adjacent-swatch judgment). Target a dE_NS < 2 per swatch for "production ready" t3 output.

### Q8: Is there a paper-fiber-driven scattering effect we're missing?

Washi is not a uniform diffuser — it has visible fibers with anisotropic scattering. Standard K-M models the substrate as a Lambertian diffuser with a single R_paper(lambda). For fine-art reproduction at the level Pace Editions targets, fiber-level scattering anisotropy may be visible at edges of pigment regions. This is in the kernel of Zeller 2026's projection error and would require going to 4-flux or higher-order methods. Probably overkill for the chuck-mcp MVP but flagged for the v2 spec.

---

## Anna's Archive download fingerprints (for manual fetch)

- **Saunderson 1942** — DOI: 10.1364/JOSA.32.000727, Anna's Archive hash: `b8dd57cc9238bd7f874dc400a570d1a3`. Annas-archive download succeeded but returned a SciDB JS shell (.ehtml), not the actual PDF body. To get the real Saunderson PDF: query SciDB directly at https://annas-archive.gd/scidb/10.1364/JOSA.32.000727 in a browser session with the donation key, or use Opt-Out-of-paywall channels.

## Artifact file count

11 markdown files written to `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/papers/color-science-km-mixbox/`:

- `arxiv_1707.08323_pigmento_image_analysis.md`
- `arxiv_1904.00275_watercolor_dnn_transmittance.md`
- `arxiv_1904.11293_color_difference_no_separation.md`
- `arxiv_2104.04884_km_selden_map_hyperspectral.md`
- `arxiv_2409.04558_km_spray_paint_nn.md`
- `arxiv_2603.09139_km_galerkin_rte.md`
- `web_berns_2016_artist_paint_spectral_database.md`
- `web_curtis_1997_computer_generated_watercolor.md`
- `web_kubelka_munk_1931_westin_translation.md`
- `web_mixbox_sochorova_jamriska_2021.md`
- `web_saunderson_1942_surface_correction.md`
- `NOTES.md` (this file)

Plus this NOTES.md = 12 files total.
