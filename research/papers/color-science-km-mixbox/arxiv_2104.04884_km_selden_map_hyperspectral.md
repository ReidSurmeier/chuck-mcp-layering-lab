# Hyperspectral Pigment Analysis of Cultural Heritage Artifacts Using the Opaque Form of Kubelka-Munk Theory

authors: Abu Md Niamul Taufique, David W. Messinger (Rochester Institute of Technology)
arxiv: 2104.04884v1 [cs.CV, eess.IV] 11 Apr 2021 (originally Proc. SPIE 10986, 2019)
relevance: Demonstrates that linear unmixing in K/S space (single-constant K-M) outperforms reflectance-domain unmixing for pigment classification on a real historical paper artifact (Selden Map). Directly applicable to chuck-mcp because Chuck Close's Emma is also paper substrate + translucent layered pigment, and 113 distinct color pulls means our t3 needs robust K/S-space unmixing to back out per-block effective pigment concentrations from measured plate reflectances.

## Abstract

Kubelka-Munk (K-M) theory has been successfully used to estimate pigment concentrations in the pigment mixtures of modern paintings in spectral imagery. In this study the single-constant K-M theory has been utilized for the classification of green pigments in the Selden Map of China, a navigational map of the South China Sea likely created in the early seventeenth century. Hyperspectral data of the map was collected at the Bodleian Library, University of Oxford, and can be used to estimate the pigment diversity, and spatial distribution, within the map. This work seeks to assess the utility of analyzing the data in the K/S space from Kubelka-Munk theory, as opposed to the traditional reflectance domain. We estimate the dimensionality of the data and extract endmembers in the reflectance domain. Then we perform linear unmixing to estimate abundances in the K/S space, and following Bai, et al. (2017), we perform a classification in the abundance space. Finally, due to the lack of ground truth labels, the classification accuracy was estimated by computing the mean spectrum of each class as the representative signature of that class, and calculating the root mean squared error with all the pixels in that class to create a spatial representation of the error.

## Technical methodology

1. Dimensionality estimation and endmember extraction in reflectance domain.
2. Convert reflectance R to K/S using single-constant K-M: K/S = (1 - R)^2 / (2R).
3. Linear unmixing in K/S space to estimate abundance fractions of each pigment endmember.
4. Classification in abundance space (per Bai et al. 2017).
5. Per-class spatial RMSE error map for accuracy assessment.

## Why this matters for chuck-mcp

- The single-constant K/S = (1-R)^2 / (2R) formula is the simplest workable K-M ingest for measured Pace Editions plate proofs. It assumes the substrate is opaque enough that S can be normalized out — fine for fully-printed regions but the washi paper is NOT opaque, so chuck-mcp will need the two-constant form (recover K and S separately) eventually.
- The linearity argument: K/S is roughly additive under pigment mixing — `(K/S)_mix ~ sum c_i * (K/S)_i` — but only when concentrations are weight-fractions in a common binder. For mokuhanga the "binder" is starch paste and ratio is per-block, not co-mixed, so the K/S linearity in our case is per-layer not within-layer.
- The use of CIELAB / RMSE for ground-truth-free validation is a good template for our t3 calibration when we can't measure every Emma plate directly.

## Citations to add to ADR-0002

- Bai et al. 2017 — classification in K/S abundance space (cited by this paper)
- Berns, R.S. — Billmeyer & Saltzman's Principles of Color Technology — the standard reference for two-constant K-M paint mixing
