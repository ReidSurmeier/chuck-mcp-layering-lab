# Geometric Realism Without Angular Resolution: Structural Classification of Multilayer Kubelka-Munk Theory within Radiative Transport

authors: Claude Zeller (Claude Zeller Consulting LLC)
arxiv: 2603.09139v1 [physics.optics] 10 Mar 2026
license: CC BY 4.0
relevance: Places K-M two-flux on rigorous footing as a rank-2 Galerkin projection of the full RTE — and proves multilayer K-M composition cannot recover discarded angular information. This is the theoretical justification (and limit-of-accuracy criterion) for the t3 spectral K-M tier used to render stacked mokuhanga impressions.

## Abstract

Kubelka-Munk (KM) theory provides a two-flux description of radiative transport in layered scattering and absorbing media. Despite its wide use in the coatings, paper, paint, and textile industries, the theory has often been regarded as a phenomenological model whose connection to the full radiative transfer equation (RTE) remains unclear. Under the standard steady-state, plane-parallel, azimuthally symmetric assumptions, we show that multilayer KM theory is exactly a rank-2 Galerkin projection of the RTE onto hemispherical basis functions. The projection is idempotent with an infinite-dimensional kernel, and its rank is preserved under multilayer composition — so no amount of layer stacking can recover angular information discarded by the projection. We derive the KM coefficients as hemispherical moments of the transport operator and compute the projection error for representative scattering media (g from 0 to 0.85), finding that the reduced optical thickness tau* = tau(1-g) governs KM accuracy. The projection-error framework explains the well-documented accuracy of compositional multilayer models in printed media and shows where higher-order methods become necessary.

## Key contributions relevant to chuck-mcp t3 tier

1. **Rank preservation under composition**: Multilayer K-M is the exact rank-2 projection of the RTE; stacking N translucent layers (mokuhanga impressions) on washi keeps the same rank. The compositional algebra used by Hebert/Hersch transfer-matrix work is a valid algebra over the projected operator.
2. **Projection-error bound**: epsilon(tau*) where tau* = tau(1 - g) is the reduced optical thickness. For forward-peaked phase functions (g near 1), K-M error grows; for diffuse media (g near 0), the projection is near-exact. Washi paper is highly diffuse (g << 0.5) and translucent pigment films are thin (tau small), so the reduced optical thickness stays in the regime where K-M is accurate.
3. **Saunderson interpretation**: Surface reflection correction is itself a separate layer in the transfer-matrix algebra — not an ad-hoc patch. Confirms the t3 architecture (interface layer + N pigment films + washi backing).
4. **Identification of K and S from RTE**: K = 2 sigma_a (absorption moment), S relates to backscatter fraction p_bar_{-+}. Gives a principled path from measured BRDF / transmittance pairs to physical (K, S) coefficients for the 8-channel fit.
5. **Hebert/Hersch transfer matrix framework** is referenced as the validated machinery for ink-on-substrate composition — this is the production algorithm the chuck-mcp t3 tier should adopt.

## Key formulas extracted

- Reduced optical thickness governs accuracy: tau* = tau(1 - g)
- K-M absorption: K = 2 sigma_a where sigma_a is the absorption coefficient of the underlying RTE
- The KM scattering coefficient S relates to the backscatter fraction of the phase function: S ~ sigma_s * p_bar_{-+}, where p_bar_{-+} is the hemispherical-mean back-scattering probability
- Multilayer composition: reflectance/transmittance of stack equals the closed-form sum of an infinite geometric series of internal bounces

## What chuck-mcp should reference

- This paper's projection-error analysis as theoretical anchor for "why t3 works at all"
- Hebert & Hersch transfer-matrix references for the actual implementation
- Mudgett & Richards (1971) and Star et al. derivation of K, S from RTE — cited here as the foundation upon which the Galerkin classification is built

## Related references inside the paper

- [1, 2] Kubelka & Munk 1931 / Kubelka 1948 originals
- [9] Saunderson surface correction 1942
- [10] Mudgett & Richards — RTE-to-KM derivation
- [11, 12] Star et al., van Gemert & Star — anisotropic scattering
- [4-8] Hebert, Hersch and collaborators — compositional transfer-matrix multilayer

## Body (verbatim excerpt)

Two-flux models collapse the full angular distribution of radiance into a pair of scalar fluxes: one going forward, one going backward. It is about the crudest simplification imaginable, and it has been extraordinarily successful. The Kubelka-Munk (KM) model, introduced in 1931, is the most widely used instance. It describes a homogeneous scattering-absorbing slab using two parameters — an absorption coefficient K and a scattering coefficient S — and yields closed-form expressions for reflectance and transmittance. For decades it has been the workhorse of color science in paints, paper, textiles, and coatings.

The simplicity that makes KM theory useful also makes it suspect. The model cannot distinguish an isotropic scatterer from one with a sharp forward peak. It treats every photon within a hemisphere as interchangeable. Many authors have regarded it as "merely phenomenological" and have proposed corrections — Saunderson surface corrections, path-length multipliers, modified scattering coefficients — that try to put back what the two-flux reduction threw away.

The connection between KM theory and the radiative transfer equation is not itself new. Mudgett and Richards derived KM from the RTE by hemispherical averaging and established that K = 2*sigma_a. Star et al. and van Gemert and Star refined the treatment for anisotropic scattering, identifying the backscatter fraction p_bar_{-+} as the bridge between transport and KM scattering coefficients. These derivations are correct and this paper relies on them. What they did not do is identify the hemispherical average as an orthogonal Galerkin projection with a specific rank, kernel, and idempotence structure.

A substantial body of work by Hebert, Hersch, and collaborators has developed a powerful compositional framework for predicting the reflectance and transmittance of multilayer specimens using transfer matrices. Each optical element — a scattering layer, an ink film, a refractive interface — is characterized by its reflectances and transmittances. Stacking two elements produces a composite element whose properties follow from summing the infinite geometric series of internal bounces in closed form. They showed that Kubelka's layering model, the Saunderson correction, and the Williams-Clapper model are all special cases of a single compositional formalism, and that the two-flux matrix approach extends naturally to four-flux and multiflux models when one needs to distinguish collimated from diffuse light. The framework has been validated to high accuracy in printed paper, ink-on-substrate systems, and stacked transparencies — application domains where KM theory is known to work well.
