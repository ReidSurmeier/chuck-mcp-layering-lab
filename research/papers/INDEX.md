# chuck-mcp-layering-lab — Research Index

Created: 2026-05-16
Swarm: `swarm-1778962124344-s4cm4l` (hierarchical, 6 specialized researchers)
Total artifacts: 71 papers + 6 NOTES.md syntheses across 6 domains

## Domain folders

| Folder | Files | NOTES summary |
|---|---:|---|
| `color-science-km-mixbox/` | 12 | t3_spectral = Curtis 1997 + Saunderson 1942 + Berns 2016 K/S table |
| `inverse-rendering-diff/` | 15 | Staged 3-batch outer loop + progressive-simplified targets + topo-derivative spawning + shrinking trust region |
| `segmentation-cellgraph/` | 12 | SNIC drop-in for SLIC (deterministic, polygons native, 2–4× faster) |
| `mokuhanga-methodology/` | 14 | Shibata uses reduction technique — `(block, pass_idx, mask)` not `(block, mask)`. 132 pulls / 26 blocks = ~5 inkings/block |
| `vectorization-cnc/` | 12 | Mill-sized area-opening + opening-by-reconstruction before Potrace |
| `graph-coloring-block-packing/` | 12 | DSATUR already optimal on chordal graphs at chuck-mcp scale. MaxRects for face assignment downstream |

## Unified pipeline (from all 6 domains)

```
S3.b  segmentation        SLIC                 → SNIC (segmentation agent)
S4    warm start          nearest-to-mean      → max-yellow-lift over paper (mokuhanga + user annotation)
S5    solver              flat 12-plane LBFGS  → staged 3-batch outer loop (inverse-rendering)
                                                   - progressive-simplified targets (Wang 2024)
                                                   - topo-derivative plate spawn (Mehta 2023)
                                                   - shrinking trust region (Worchel 2023)
                                                   - L-BFGS-B inner (JAXopt, unchanged)
S5    impression key      (block, mask)        → (block, pass_idx, mask) w/ monotone reduction (mokuhanga)
S6.b  jigsaw              re-segments target   → consumes SNIC polygons (segmentation)
S6.c  printability repair turdsize heuristic   → mill-sized area-opening + opening-by-reconstruction
                                                   + ΔE guard (vectorization)
S7    plate packing       DSATUR               → DSATUR + chordality cert + MaxRects face packer (graph-coloring)
S7    SVG export          (current)            → + horizontal flip (user annotation)
forward render t1         Mixbox lerp          → keep as baseline only; wrong physics for overprint
forward render t2         (unbuilt)            → Curtis 1997 recursion + Saunderson 1942 + 2-strip inverse from one swatch print
forward render t3         (unbuilt)            → t2 + 36-wavelength Berns 2016 K/S table
HITL fixture              none                 → pin_region against user-annotated 9-underlayer reference
```

Every move has a published reference. Nothing requires inventing new algorithms.

## Top must-reads per domain

**color-science-km-mixbox/**
1. Curtis et al. 1997 "Computer-Generated Watercolor" (SIGGRAPH) — `web_curtis_watercolor_1997.md`
2. Zeller 2026 "Geometric Realism Without Angular Resolution" — `arxiv_2603_09139_*.md`
3. Sochorová & Jamriška 2021 Mixbox (DOI 10.1145/3478513.3480549)

**inverse-rendering-diff/**
1. Jiang et al. 2025 "Birth of a Painting" — `arxiv_2511_13191_birth_of_a_painting.md`
2. Wang et al. 2024 "Layered Vectorization via Semantic Simplification" — `arxiv_2406_05404_*.md`
3. Mehta et al. 2023 "Topological Derivatives for Inverse Rendering" — `arxiv_2308_09865_*.md`

**segmentation-cellgraph/**
1. Achanta & Süsstrunk 2017 SNIC — `web_achanta_snic_2017.md`
2. Giraud & Clément 2024 "Ill-Posed Problem" — `arxiv_2411_06478_*.md`
3. Barcelos et al. 2024 Superpixel Survey — `arxiv_2409_19179_*.md`

**mokuhanga-methodology/**
1. Sultan/Shiff "Chuck Close Prints: Process & Collaboration" (Princeton UP, 2003) — annas MD5 `be57d6df27782b9d4240c6b5a005abf6`
2. Salter "Japanese Woodblock Printing" (UH Press, 2002) — annas MD5 `bcb210ef60b44caba138bced4db5e78f`
3. Vollmer "Japanese Woodblock Print Workshop" (Watson-Guptill, 2015) — annas MD5 `33b961fffee2de3d08b3de7d9aaa1f2f`

**vectorization-cnc/**
1. Selinger 2003 Potrace — `web_potrace_selinger_2003.md`
2. Vincent 1993 area-opening + opening-by-reconstruction — `web_morphological_area_opening_vincent_1993.md`
3. CNC min-feature-size for relief carving — `web_cnc_min_feature_size_endmill_relief.md`

**graph-coloring-block-packing/**
1. Yekezare et al. 2024 "Optimality of DSATUR on Chordal Graphs" — `web_yekezare_dsatur_chordal_2024.md`
2. Brélaz 1979 — `web_brelaz_dsatur_1979.md`
3. Furini et al. 2015 improved DSATUR B&B — `web_furini_dsatur_bb_2015.md`

## Cross-cutting findings

**1. Mixbox is binder-mixing, not layering** (color-science). ADR-0002's 3-tier premise is correct. Curtis 1997 already solved multilayer K-M for watercolor 29 years ago. The "new research" needed is integration, not invention.

**2. Shibata uses reduction technique** (mokuhanga). Block surface evolves between passes. Current `(block_id, mask)` model is structurally wrong. Promote to `(block_id, pass_index, mask)` with `mask[t+1] ⊆ mask[t]`.

**3. The v12→v13 dE plateau is structural, not parametric** (inverse-rendering). Flat 12-plane simultaneous L-BFGS-B cannot escape this floor. Need staged outer loop with progressive targets + principled plate spawning.

**4. SLIC nondeterminism is masking gains** (segmentation). SNIC drop-in restores reproducibility so v14+ improvements are attributable.

**5. v13 raster proofs → CNC plates is one transform** (vectorization). Area-opening + opening-by-reconstruction sized by physical end-mill diameter, then Potrace. ~20 lines of skimage.

**6. DSATUR is already correct for the scale** (graph-coloring). Don't replace; just certify and add MaxRects for face assignment.

## Suggested implementation order

1. Wire v13 methodology output → S6.c (mill-sized morphology + flip + Potrace). Closes the v09/v13 pipeline fork. Single coherent shippable artifact.
2. SNIC drop-in for SLIC in S3.b. Pure win, no downstream changes.
3. Ingest user-annotated 9-underlayer reference via `pin_region`. Regression fixture for every future iteration.
4. Staged 3-batch outer loop in S5 (Wang + Mehta + Worchel). Structural break out of dE plateau.
5. t2_empirical LUT — print one calibration swatch, fit via Curtis 1997's white+black inverse procedure.
6. `(block, pass_idx, mask)` impression key with reduction monotonicity. Enables scaling to Shibata's 132 pulls / 26 blocks = ~5 inkings/block.
7. Certify DSATUR chordality + MaxRects face packer.
8. t3_spectral — 36-wavelength K/S from Berns 2016 + Curtis recursion + Saunderson correction.
