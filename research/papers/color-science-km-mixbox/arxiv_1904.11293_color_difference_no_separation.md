# A color-difference formula for evaluating color pairs with no separation - Delta E_NS

authors: Fereshteh Mirjalili, Ming Ronnier Luo, Guihua Cui, Jan Morovic
arxiv: 1904.11293v1 [eess.IV] 25 Apr 2019
relevance: When validating chuck-mcp's three rendering tiers against measured Pace Editions proofs, a pixel-by-pixel CIEDE2000 difference is NOT the right metric — that formula was developed for color pairs viewed with hair-line separation, which exaggerates lightness differences. Adjacent (no-separation) printed swatches need this corrected formula dE_NS. Critically applicable to chuck-mcp's QA loop because all 113 colors on the Emma print are adjacent.

## Abstract

All color-difference formulas are developed to evaluate color differences for pairs of stimuli with hair-line separation. In printing applications, however, color differences are frequently judged between a pair of samples with no-separation because they are printed adjacent on the same piece of paper. A new formula, dE_NS has been developed for pairs of stimuli with no-separation (NS). An experiment was conducted to investigate the effect of different color-difference magnitudes using sample pairs with NS. 1,012 printed pairs with NS were prepared around 11 CIE recommended color centers. The pairs, representing four color-difference magnitudes of 1, 2, 4 and 8 CIELAB units were visually evaluated by a panel of 19 observers using the gray-scale method. Comparison of the present data based on pairs with NS, and previously generated data using pairs with hair-line separation, showed a clear separation effect.

A new color-difference equation for the NS viewing condition (dE_NS) is proposed by modifying the CIEDE2000 formula. The separation effect can be well described by the new formula. For a sample pair with NS, when the CIEDE2000 color difference is less than 9.1, a larger color difference leads to a larger lightness difference, and thus the total color difference increases. When the CIEDE2000 color difference is greater than 9.1, the effect is opposite, i.e. the lightness difference decreases, and thus the total color difference also decreases. The new formula is recommended for future research to evaluate its performance in appropriate applications.

## Why this matters for chuck-mcp

1. **Adjacent-swatch case**: The Chuck Close Emma print is 113 colors physically adjacent on the same sheet — exactly the "no separation" condition. Standard CIEDE2000 will OVER-estimate perceived color difference in this regime for dE < 9.1 and UNDER-estimate it for dE > 9.1.
2. **Quality threshold**: When chuck-mcp's t3 renderer is validated against a measured proof, the QA threshold should use dE_NS, not dE_ab or even dE_2000. A target of dE_NS < 2 is "imperceptible to most observers" for adjacent swatches.
3. **Tier comparison**: When comparing t1 (Mixbox) vs t2 (empirical LUT) vs t3 (spectral K-M) outputs, ALL three should be scored against ground truth using dE_NS, not dE_ab. This gives a fair perceptual comparison.

## Key takeaway

The formula `dE_NS` is a CIEDE2000 variant tuned for adjacent samples. CIEDE2000 itself is already the right starting point — but the L (lightness) weighting needs to be adjusted for the NS condition. Implementation is a small patch to a standard CIEDE2000 library.

## Citations to add to ADR-0002 (color-difference choice)

- CIE Pub. 142 (CIEDE2000 standard) — baseline color-difference
- This paper (dE_NS) — adjacent-swatch correction
- Sharma et al. 2005 "The CIEDE2000 color-difference formula: Implementation notes" — for the actual code path
