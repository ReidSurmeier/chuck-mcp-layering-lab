# Color Separation Methodology: From Maquette to Block in Fine Art Mokuhanga

**Primary URLs**:  
- https://mlyon.com/2012/post-digital-printmaking/  
- https://www.moreofmyjapanesehanga.com/home/artist-index/kurosaki-akira-%E9%BB%92%E5%B4%8E%E5%BD%B0-1937-2019  
- https://www.numberanalytics.com/blog/art-color-separation-printmakers-guide  
- https://paceprints.com/2021/jonas-wood  
**Fetch date**: 2026-05-16  
**Evidence grade**: Medium — synthesis from multiple practitioner sources; no single primary document describes Shibata's specific separation methodology

---

## The Color Separation Problem in Mokuhanga

Starting from a continuous-tone painting (as with Close's Emma), the master printer must decompose the image into a sequence of discrete woodblock passes. This is fundamentally different from commercial CMYK color separation: there is no fixed ink set. Every pass can use a unique mixed color, and the color seen in the final print is the optical result of all stacked layers — not the color on any single block.

This overprint-based color system means separation planning is not a decomposition problem (split one image into components that sum to the original) but a forward-stacking problem (plan a sequence of layers whose cumulative optical interaction approximates the target).

## Kurosaki's Contribution: The High-Density Pass Model

Akira Kurosaki (1937–2019), Shibata's teacher at Kyoto Seika University, pioneered the approach of very high pass counts per relatively small block sets. His "Red Darkness" series (1970) used up to 8 blocks, 15 colors, 70 passes. This establishes the formal precedent: block count constrains spatial forms, pass count drives color density. Kurosaki's innovation was treating the block not as a color carrier but as a form library used repeatedly with different ink.

## Mike Lyon's CNC Digital Separation

Lyon's documented workflow for digital mokuhanga separation:
1. Photograph or scan the source image at high resolution
2. In Photoshop, separate the image into discrete luminosity bands using threshold or posterization
3. Each luminosity band becomes a separate layer/bitmap
4. Kento registration marks are added at fixed positions to each bitmap
5. CNC router cuts each bitmap as a relief block
6. Printing proceeds light-to-dark, building luminosity

Lyon's approach is strictly luminosity-based (each block carries one tonal value). Shibata's approach for Close's work is more complex: the separation follows Close's existing color-grid structure — each cell in the painting's grid carries a specific hue, so separation must account for hue identity, not just luminosity. The grid structure means adjacent spatial zones carry unrelated colors, which is the source of the high block-use complexity.

## The Shibata Tracing/Separation Process (Documented)

From gallery documentation of his collaborative process, Shibata's workflow involves:
1. Making a precise tracing of the artist's maquette
2. Analyzing which color zones require individual block areas vs. which can share a block with different ink on separate passes
3. Assigning spatial forms to blocks — trying to maximize re-use of each block surface (to minimize carving labor while maximizing color flexibility)
4. Carving blocks in sequence, proofing each against the maquette
5. Test-printing sequences to verify overprint color interactions
6. Iterating ink mixtures until the cumulative impression matches the artist's approval

The key planning decision — shared extensively across large woodblock traditions — is which colors are spatially proximate (requiring separate blocks to avoid contamination) vs. which are spatially isolated (allowing block sharing on separate passes). Close's grid structure, where each cell is spatially isolated from its neighbor by the grid structure itself, enables substantial block reuse because adjacent zones rarely touch.

## Implications for Chuck-MCP Planner

The current chuck-mcp assumption — that block count is approximately equal to color count — does not match Shibata's practice. The actual planning rule is:

**Blocks encode spatial forms. Passes encode color decisions. Colors = passes × (unique ink mixtures per pass).**

A 27-block system producing 113 colors over 132 passes implies an average of ~4.9 colors per block (different ink on different passes) and ~1.17 passes per color (most colors used once, but some pass-to-pass re-inking of the same block produces a color used across multiple spatial zones).

The planner should model this as a (block, pass, ink) three-dimensional space, not a (block = color) one-dimensional mapping.

