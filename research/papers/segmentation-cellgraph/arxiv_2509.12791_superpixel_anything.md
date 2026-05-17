---
title: "Superpixel Anything: A general object-based framework for accurate yet regular superpixel segmentation"
authors: ["Julien Walther", "Rémi Giraud", "Michaël Clément"]
arxiv_id: "2509.12791"
year: 2025
venue: "BMVC 2025"
links:
  - "https://arxiv.org/abs/2509.12791"
  - "https://arxiv.org/html/2509.12791v1"
  - "https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1035/paper.pdf"
  - "https://github.com/waldo-j/spam"
relevance: 9
relevance_reason: "Current SOTA superpixel method as of late 2025. Uses SAM as object-mask prior, fills with regular superpixels via differentiable clustering. Best-of-both-worlds: deep features for object boundaries, regularity by construction. Relevant for mokuhanga because it respects object structure (line art) while keeping cells carve-able."
---

# Superpixel Anything (SPAM): A General Object-Based Framework

## Algorithmic Core

SPAM (SuperPixel Anything Model) is a two-stage system:

1. **High-level prior segmentation.** Run a pretrained foundation segmentation model (the paper uses SAM, Kirillov et al. 2023) to produce a coarse object decomposition. This step is *not* trained for the superpixel task — SAM generates whatever object masks it generates, possibly overlapping, possibly with uncertain regions between objects.

2. **Differentiable superpixel filling.** A trainable CNN encoder produces per-pixel deep features. A differentiable clustering algorithm (extended from SSN / Jampani 2018) then clusters pixels into superpixels under a *constraint*: every superpixel must be strictly contained within one of the SAM masks. The clustering loss combines feature similarity (for accuracy) with a spatial regularization term (for grid-like regularity). The CNN is trained end-to-end on a small annotated dataset (BSDS500 + Pascal Context).

The clever part is the constraint handling. SAM's mask output has *uncertainty regions* (pixels assigned to no object or to overlapping objects). SPAM handles these by treating the union of all SAM masks as a soft prior: pixels in unambiguous mask regions are constrained to that mask; pixels in uncertain regions are clustered without constraint, allowing the SLIC-style locality to take over. This means SPAM gracefully degrades to SLIC-quality output in regions where SAM is uncertain, but uses SAM's object boundaries where SAM is confident.

Two adaptive inference modes are proposed:

- **Visual-attention mode.** Use a saliency map (computed from the CNN encoder's attention) to allocate *more* superpixels to high-attention regions and *fewer* to background. Produces a hierarchical decomposition: ~50 large cells for background, ~500 small cells for the salient object.
- **User-interaction mode.** A user can click on objects to request finer cells inside that object only. Useful for annotation pipelines.

## Benchmark Performance

On BSDS500 the paper claims SOTA across all four metric axes the authors' previous "ill-posed" paper proposed: Object (ASA, UE), Contour (BR), Regularity (CO, GR, EV), and Color (ICV). Notably, SPAM is the first deep-learning superpixel method that doesn't tank on regularity — earlier methods like LNS-Net and AINet outperform on ASA but produce skinny irregular cells that don't satisfy CO. SPAM matches AINet on ASA (~0.95 at N=600 superpixels) while keeping CO ~ 0.6 (vs AINet's ~0.35).

Speed: SPAM is slower than classical SLIC (10-50× SAM cost dominates) but achievable in real-time on a single GPU. The paper reports ~2 seconds per 512×512 image on RTX 4090.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

SPAM is genuinely interesting for chuck-mcp but with one important caveat: SAM is trained on natural photographs. SAM's object understanding may or may not transfer to mokuhanga subject matter — pre-process scanned line art, then SAM may treat the entire image as one "object" because it's a single scanned page.

That said, SPAM's architectural pattern is exactly right for mokuhanga jigsaw planning:

- The **prior segmentation** step (replace SAM with whatever) defines *which boundaries are semantically meaningful*. For mokuhanga, the right prior is: contours of the line-art carved into the keyblock. These are the boundaries the printer absolutely wants superpixel cells to respect.
- The **differentiable filling** step then tiles each line-art-bounded region with regular cells. Each cell is a candidate plate. The regularity ensures cells are carve-able; the SAM-style boundary respect ensures cells don't cross line work.

This is precisely the workflow a printmaker would describe in natural language: "carve the keyblock first, then carve color plates that fit within the keyblock outlines." SPAM gives that workflow algorithmic form.

**Concrete proposal for chuck-mcp:**

1. Replace SAM with a mokuhanga-specific prior: edge detection (Canny / Sobel) on the line-art channel, with morphological closing to make regions. Or: use the inverse-stack-solver's keyblock-alpha output as the prior directly.
2. Adopt SPAM's differentiable filling as the S3.b primary algorithm. Train the CNN encoder on a small set of hand-annotated mokuhanga prints (5-10 should suffice given the constraint structure).
3. Keep SLIC as the fallback for prints with no useful prior.

The visual-attention mode also gives a natural plate-budget knob: use the importance map (e.g., printer-marked priority regions) to allocate plates non-uniformly.

Code is open-source MIT-licensed at github.com/waldo-j/spam. The differentiable clustering module is the reusable piece — it's a PyTorch-native SSN extension that could plug into chuck-mcp's existing JAX inverse-solver as a forward step.

Sources:
- [arXiv:2509.12791 (HTML)](https://arxiv.org/html/2509.12791v1)
- [arXiv abstract](https://arxiv.org/abs/2509.12791)
- [BMVC 2025 paper PDF](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1035/paper.pdf)
- [SPAM reference code](https://github.com/waldo-j/spam)
