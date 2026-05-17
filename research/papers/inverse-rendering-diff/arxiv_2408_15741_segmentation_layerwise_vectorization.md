# Segmentation-guided Layer-wise Image Vectorization with Gradient Fills

authors: Hengyu Zhou, Hui Zhang, Bin Wang
arxiv: 2408.15741
relevance: Progressive layer-wise vectorization that segments the residual error each epoch, picks the largest unfit region, initializes a new path there, then re-optimizes ALL paths. This is exactly the staged solver pattern chuck-mcp wants: solve broad supports first, take the residual, segment it, append a chroma/accent plate driven by the largest error region. Direct algorithmic blueprint.

Key algorithmic moves to steal:
1. After each "batch" solve, run a Laplacian-on-residual + Otsu threshold + watershed to find the next region to address.
2. Initialize the new plate at the centroid of the worst-residual segment.
3. Continue optimizing earlier plates while later plates are added (bounded trust-region equivalent via UDF-style weight that emphasizes contour pixels).
4. Use a `segmentation-guided weight` that's high for the segment-owning plate and low elsewhere - this is the "bounded feedback to batch 1 only on low-frequency residual" idea written directly.

---

(Full text from arXiv 2408.15741 follows.)

Segmentation-guided Layer-wise Image Vectorization with Gradient Fills

Authors: Hengyu Zhou, Hui Zhang, Bin Wang
School of Software, Tsinghua University

Abstract:
The widespread use of vector graphics creates a significant demand for vectorization methods. While recent learning-based techniques have shown their capability to create vector images of clear topology, filling these primitives with gradients remains a challenge. In this paper, we propose a segmentation-guided vectorization framework to convert raster images into concise vector graphics with radial gradient fills. With the guidance of an embedded gradient-aware segmentation subroutine, our approach progressively appends gradient-filled Bezier paths to the output, where primitive parameters are initiated with our newly designed initialization technique and are optimized to minimize our novel loss function. We build our method on a differentiable renderer with traditional segmentation algorithms to develop it as a model-free tool for raster-to-vector conversion.

Keywords: Vectorization, Segmentation, Differentiable rendering

1 Introduction

Vector graphics offer great flexibility in digital design as they can be easily edited and arbitrarily scaled. Vectorization, the procedure of converting raster images to vector ones, serves as a second way to creating vector graphics other than designing from scratch.

LIVE (Ma et al. 2022) utilizes DiffVG (Li et al. 2020), a differentiable renderer, to present a model-free vectorization framework. It progressively translates a raster image into an SVG in a layer-wise hierarchy, through which the topology of the input is preserved within the order of geometric primitives. However, its lack of support for gradients results in excessive primitives being added in case of images with rich gradient effects.

In this paper, we propose a novel segmentation-guided vectorization framework that extends the capability of LIVE to support radial gradients. The additional parameters of a radial gradient pose an increased challenge to optimization, where an effective method to determine whether a pixel contributes to a path's gradient fill is necessary.

Contributions:
- A segmentation-guided vectorization framework to create vector graphics automatically with layer-wise hierarchy and radial gradients.
- A gradient-aware segmentation method to evaluate the pixel-wise contribution to the geometric and gradient parameters of a path.
- The segmentation as guidance for a new initialization technique and as a part of a novel segmentation-guided loss.

2 Related Work

2.1 Image Vectorization
Most traditional vectorization methods aim to create vector images of high fidelity with different representations:
- Mesh-based: triangular, rectangular, or irregular patches (bezigons).
- Curve-based: diffusion curves with colors on either side.

Recent learning-based vectorization mainly uses ordered primitives as a sequence of drawing operations. Models include RNNs (SketchRNN), transformers (DeepSVG), often combined with variational autoencoders.

DiffVG fills the gap between raster and vector graphics: loss functions on raster images directly optimize the vector images.

2.2 Image Topology
A related problem is layer decomposition, where images are decomposed into semi-transparent layers. Photo2ClipArt and similar work replace heavy manual interaction with a user-provided segmentation input. However, these methods require concise segmentation for efficient and effective vectorization.

LIVE is the most similar work: a progressive framework converting raster images into layered vector paths utilizing DiffVG. Its missing support for color gradients is not trivially achievable by simply optimizing gradient parameters.

3 Method

3.1 Method Overview

The framework works progressively. At each epoch i, single or multiple Bezier paths are added and optimized.

At the beginning of each epoch i, we calculate the difference between the input raster image and the previous output. We segment it with our gradient-aware segmentation. n_i segmented regions are selected for initialization of new paths. All added paths, including those from previous epochs, are optimized to minimize the vectorization loss. The parameters consist of geometric parameters (positions of curve control points) and gradient parameters (center, radius, color stops).

3.2 Gradient-aware Segmentation

When paths are filled with a solid color, a connected component of similar colors has a good chance of being covered by one path. LIVE designed a component-wise initialization where pixels are clustered into buckets based on their L2-length over RGB channels.

As gradients are considered, the clustering algorithm used by LIVE may result in excessive segmentation. Other clustering methods including Mean-shift suffer from over-segmentation since colors within a path may vary more than colors between paths.

Our approach is designed to detect edges of gradients. Colors should derive smoothly inside a region filled with the same gradient fill and change abruptly at its boundary. We calculate the secondary spatial gradient with a discrete Laplacian filter L = [[1,1,1],[1,-8,1],[1,1,1]] to identify rapid changes at gradient boundaries.

Steps:
1. Compute S_0 = correlate((I - I_hat) * 1_{||I_hat - I||_2 > epsilon}, L). Pixels with error below epsilon=0.1 are excluded.
2. Sum S_0 over RGB channels to obtain grayscale S_1.
3. Convert S_1 to binary S_2 = 1_{S_1 > Otsu(S_1)}.
4. Apply morphological closing and watershed for final segmentation S.

Since we segment the difference between output and target, already-fitted pixels are ignored. Otsu's threshold decreases in response to descending overall difference. This dynamic threshold avoids hyperparameters.

3.3 Segmentation-guided Initialization

We add one or more paths at each epoch. For each path, we select the segmentation region with the largest accumulated square error:
  w_i = sum_{p in S_tilde[i]} ||I_p - I_hat_p||^2

This prioritizes larger regions to encourage a hierarchical initialization order and prevents regions that are almost properly filled from being chosen.

A circle path of four cubic Bezier curves is added at the selected region's center of mass p_m. We fill the path with a radial gradient, centered at p_m, with diameter equal to the geometric mean of width/height of the region's bounding box, clipped to [0.2, 1.0]. The two stop colors at offsets 0% and 100% are both initialized to the color of the input image at p_m.

3.4 Loss Function with Segmentation-guided weight

LIVE's UDF loss focuses on pixels on the contour, but for radial gradients, correct colors on the contour do not mean correctness inside. We draw from LIVE's insight emphasizing pixels on edges and extend this concept to include all pixels within a path, except those occluded by other paths.

For each added path p_i, given pixels covered by the path and pixels within the segment from which the path is initialized, we take their intersection and mark these as focused. The union forms set F.

  w_SG(i) = max(d_i', alpha_s)            if i in F
          = d_i' * (1 - alpha_s)            otherwise

with alpha_s = 0.6 empirically, and d_i' = ReLU(tau - |d_i|) / sum_j ReLU(tau - |d_j|), where d_i is distance to nearest path contour.

  L_SG = (1/3) sum_i w_SG(i) sum_c (I_{c,i} - I_hat_{c,i})^2

Xing loss penalizes self-intersection. Final objective L = L_SG + lambda * L_Xing, lambda = 0.05.

4 Experiments

4.1 Implementation Details

Implemented with DiffVG renderer based on PyTorch. Adam optimizer with learning rate 10^-2 for gradient parameters and 1 for path points. scikit-image for morphological operations and watershed.

4.2 Datasets
- Noto Emoji (256 emojis, Unicode 15.0): newer resigned emojis filled with gradients.
- Fluent Emoji (256 emojis from Microsoft): rich gradient details.
- Iconfont (128 vector arts, ~1020 paths/image, no gradients).

4.3 Qualitative Comparison
LIVE struggles with gradient-filled facial features. Our approach reconstructs accurately with concise paths. 'w/o guidance' attempts to optimize gradient parameters without segmentation guidance result in degraded outcomes.

4.4 Quantitative Comparison
PSNR comparison: our method achieves generally faster convergence than LIVE, especially when a small number of paths are added. With excessive paths added, both methods yield the same level of quality.

4.5 Layer Decomposition
The framework captures layer-wise structure during progressive vectorization. Segmentation-guided initialization prioritizes segments more significant in accumulated error, while details with relatively small errors are captured later by adaptive gradient-aware segmentation. Given input with clear hierarchy, progressively added ordered paths resemble handcrafted vector graphics, creating an easy-to-edit output.

4.6 User Study
223 participants, 4460 votes. Clear preference for our method, especially Fluent Emoji (65.3% overall) and Iconfont (57.9% overall).

4.7 Ablation Study
Combinations of {with/without gradients} x {with/without segmentation guidance}. The combination of gradients + segmentation yields superior results on both Noto and Fluent datasets.

4.8 Limitations and Future Work
Initial shape (loop of four Bezier curves) encounters challenges in fitting intricate shapes. Radial gradients with two color stops for simplicity. A more comprehensive implementation could dynamically determine gradient types and color stop counts.

5 Conclusion
A segmentation-guided layer-wise vectorization framework synthesizing vector images by progressively adding paths with radial gradients. A gradient-aware segmentation method out of traditional algorithms guides a novel initialization approach and newly designed loss function.

References (selected, relevant to chuck-mcp):
- LIVE (Ma et al. 2022): Towards layer-wise image vectorization. CVPR.
- DiffVG (Li et al. 2020): Differentiable vector graphics rasterization for editing and learning. ACM TOG.
- Photo2ClipArt (Favreau et al. 2017): Image abstraction and vectorization using layered linear gradients. ACM TOG.
- Diffusion Curves (Orzan et al. 2008): a vector representation for smooth-shaded images. ACM TOG.
- Im2Vec (Reddy et al. 2021): Synthesizing vector graphics without vector supervision. CVPR.
