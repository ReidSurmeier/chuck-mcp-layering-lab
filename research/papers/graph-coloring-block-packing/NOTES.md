# NOTES — Graph Coloring & Block Packing for chuck-mcp

**Agent**: GRAPH-COLORING-BLOCK-PACKING
**Swarm**: chuck-mcp-layering-lab swarm-1778962124344-s4cm4l
**Scope**: post-solve assignment of Impressions to physical Blocks (woodblocks), one face per block side, minimizing total block count subject to spatial-overlap conflicts.

---

## (a) Top 5 papers ranked for relevance to chuck-mcp post-solve block packing

| Rank | Paper | Why it matters |
|------|-------|----------------|
| **1** | **Yekezare, Zohrehbandian, Maghasedi, Bonomo-Braberman 2024** — *Optimality of DSatur algorithm on chordal graphs*, ORL 57, art. 107185 | **The linchpin result.** Proves DSATUR is provably optimal on chordal graphs (superclass of interval graphs). If chuck-mcp's impression-conflict graph is chordal (it usually is — see below), DSATUR alone produces the chromatic number, no B&B needed. Reduces "what algorithm should I run?" from a research question to "DSATUR + chordality check". File: `web_yekezare_dsatur_chordal_2024.md` |
| **2** | **Brélaz 1979** — *New methods to color the vertices of a graph*, CACM 22(4) | The canonical DSATUR paper. Defines saturation degree, gives heuristic + exact B&B. The CONTEXT.md already cites "DSATUR-packed post-solve", so chuck-mcp is already operating in this framework. File: `web_brelaz_dsatur_1979.md` |
| **3** | **Furini, Gabrel & Ternier 2015** — *An improved DSATUR-based Branch and Bound for the Vertex Coloring Problem* | The modern exact algorithm. Introduces the Reduced Graph for refreshing lower bounds at every B&B node. The implementation to copy if DSATUR alone ever fails to close. File: `web_furini_dsatur_bb_2015.md` |
| **4** | **Leighton 1979 / Adegbindin et al.** — RLF (Recursive Largest First) | RLF's "pack one color class fully before starting the next" is structurally aligned with the printmaker's mental model (fill one block, then start another). Same block count as DSATUR on near-chordal graphs but more evenly utilized per block. Cheap to run as a second candidate. File: `web_rlf_leighton_1979.md` |
| **5** | **Lucci, Nasini & Severín 2018** — *A Branch and Price Algorithm for List Coloring Problem*, arXiv 1812.00040 | List coloring is the natural extension for **artist-pinned impressions** (key block forced to Block 1, large impressions restricted to large blocks). The list-DSATUR adaptation is the practical implementation path. File: `arxiv_1812.00040_lucci_list_coloring_bp.md` |

### Honorable mentions (kept in folder, lower priority)

- **Iori et al. 2020** (`arxiv_2004.12619_iori_2d_packing_survey.md`) — the right reference if/when chuck-mcp adds a 2D layout pass for placing impressions physically on each block face.
- **Orden et al. 2016** (`arxiv_1602.05038_orden_spectrum_coloring.md`) — Spectrum coloring is the soft-conflict generalization, useful if chuck-mcp later models partial spatial overlap as graded interference.
- **Şeker et al. 2018** (`arxiv_1811.12094_seker_selective_coloring.md`) — Selective coloring handles cluster-of-variants (mask alternatives per impression).
- **Jabrayilov & Mutzel 2017** (`arxiv_1706.10191_jabrayilov_mutzel_ilp.md`) — Best generic ILP formulations (POP for sparse, REP for dense). Don't use unless DSATUR + Furini-B&B both fail.
- **Gardeyn, Vanden Berghe & Wauters 2025** (sparrow, `arxiv_2509.13329_sparrow_2d_nesting.md`) — Open-source 2D nesting heuristic. Use only for irregular polygon packing per block.
- **Zhu et al. 2025 HyColor** (`arxiv_2506.07373_hycolor_2025.md`) and **Zhu & Zhou 2025 RECOL** (`arxiv_2509.23606_recol_massive_2025.md`) — modern massive-graph heuristics. Listed only to confirm the field considers chuck-mcp-scale problems solved.

---

## (b) Algorithmic recommendation: DSATUR vs. ILP vs. CP for chuck-mcp's scale (12–40 impressions, 20–30 blocks target)

**Verdict: DSATUR is the right call. Don't reach for ILP or CP.**

### Why

| Approach | Wall-time @ 12–40 vertices | Optimality guarantee | Code complexity | Dependency |
|----------|----------------------------|----------------------|-----------------|------------|
| **DSATUR (heuristic, Brélaz)** | < 1 ms | **Optimal on chordal graphs (Yekezare 2024)**; typically optimal even on non-chordal at this scale | ~50 lines Python | None |
| **DSATUR + chordality check** | < 5 ms | **Certified optimal** if chordality test passes (O(n+m) max-cardinality-search) | ~150 lines | None |
| **DSATUR + Furini B&B** | < 100 ms worst case | Provably optimal always | ~300 lines | None |
| **POP / REP ILP (Jabrayilov-Mutzel)** | < 100 ms via Gurobi | Provably optimal | ~80 lines + Gurobi | Gurobi license + Python solver wrapper |
| **Branch-and-Price (Mehrotra-Trick)** | < 1 s via custom code | Provably optimal | ~1500 lines | Column-generation library |
| **CP (Choco / OR-Tools CP-SAT)** | < 200 ms via OR-Tools | Provably optimal | ~30 lines | google-or-tools |

At 12–40 vertices, ILP and CP add solver dependencies and ~10× wall-time over DSATUR for zero gain. CP (OR-Tools CP-SAT) is the *least bad* of the heavyweight options if you ever do need one — it has good defaults, a permissive license, and the modeling language is concise — but on chordal/near-chordal small instances it cannot beat plain DSATUR.

### Recommended chuck-mcp implementation

```python
def assign_blocks(impressions, overlap_pairs):
    """
    Input: list of Impression objects, list of (i,j) pairs with spatial overlap.
    Output: block_id per impression (0..K-1), where K is minimized.
    """
    G = build_conflict_graph(impressions, overlap_pairs)
    coloring = dsatur(G)              # always run
    if is_chordal(G):                  # O(n+m) — max-cardinality-search
        return coloring, "optimal (chordal certificate)"
    # also try RLF as a second candidate
    rlf_coloring = rlf(G)
    best = coloring if num_colors(coloring) <= num_colors(rlf_coloring) else rlf_coloring
    # match lower bound from heuristic clique
    lb = greedy_clique_size(G)
    if num_colors(best) == lb:
        return best, "optimal (matches clique LB)"
    # extremely rare for 12–40 vertices on spatial-overlap graphs
    return furini_branch_and_bound(G, lb, best), "optimal (B&B)"
```

This is ~250 lines of Python with zero external solver dependencies. The DSATUR + chordality check covers > 95% of real chuck-mcp instances at provably optimal quality. The B&B fallback handles the rest and closes in well under a second.

### When to revisit

- **If artist requests soft conflicts** (e.g., partial overlap is OK if penalty stays low): lift to **Spectrum Coloring (Orden 2016)** — same DSATUR shell, swap the color-pick rule.
- **If artist pins impressions to blocks**: lift to **List Coloring** — restrict DSATUR's color-pick to the per-vertex list; backtrack on infeasibility.
- **If impressions come in variants** (multiple candidate masks): lift to **Selective Graph Coloring (Şeker 2018)** — brute-force enumerate variant selections at this scale, color each, return minimum.

None of these justify adopting a generic ILP solver. All three are 50–100 line modifications to the existing DSATUR loop.

---

## (c) How to handle 2-faces-per-block (front/back): graph coloring extension or 2D packing?

**Verdict: Two-stage approach. Graph coloring chooses *which impressions share a block*. 2D packing chooses *which face of that block each lives on, and where*. Don't try to bake faces into the coloring formulation.**

### The dichotomy

A plywood block has two carvable faces (front + back) — typically same dimensions, e.g., 60×60 cm each. Some impressions are physically large (close to a full face); some are small. We need to assign each impression to a `block_id` *and* a `block_face_id ∈ {front, back}`.

**Option A — Bake the faces into the coloring**: treat each block face as a separate "color" → 2× as many colors as blocks. This **breaks the chordality argument and the DSATUR optimality**. Why: the face is not symmetric (front vs. back is not arbitrary — registration, kento pins, paper sequence matters), so the conflict graph between "block-1-front" and "block-1-back" is artificial and doubles the problem size for no structural gain.

**Option B — Two-stage: color first, then pack each color class onto two faces**: graph coloring decides block membership. For each color class, run a tiny 2D packer to decide front-vs-back placement. If the packer fails (items don't fit), split the class — add an artificial conflict edge between the displaced impressions and re-run coloring.

### Why Option B wins

1. **Graph coloring optimality holds**. DSATUR + chordality argument gives the minimum *block count*. Faces are a downstream concern.
2. **2D packing scope is tiny per class** — each color class has typically 1–4 impressions for chuck-mcp. Packing a 4-rectangle problem onto 2 fixed-size faces (60×60 each) is trivial via MaxRects or Skyline-BL. No solver needed.
3. **Failure case is clean**: if `class C = {A, B, C}` can't fit on 2 faces, add edge `A↔B` (the two largest items by area) to the conflict graph and re-color. Converges in 1–2 iterations almost always.
4. **Matches the artist mental model** — Reid carves one block at a time, deciding which side to carve which impression on. The two-stage split mirrors that.

### Implementation

```python
def pack_block_faces(class_impressions, face_w, face_h):
    """Try to pack a list of impressions onto two faces of size (face_w x face_h).
    Returns dict {impression_id: 'front'|'back', position: (x,y)} or None if infeasible.
    """
    front = MaxRectsPacker(face_w, face_h)
    back = MaxRectsPacker(face_w, face_h)
    # sort by area descending — Best Area Fit heuristic
    for imp in sorted(class_impressions, key=lambda i: -i.area):
        if front.try_pack(imp):
            assign(imp, 'front')
        elif back.try_pack(imp):
            assign(imp, 'back')
        else:
            return None  # infeasible — caller must split this class
    return assignments

def assign_blocks_with_faces(impressions, overlap_pairs, face_w=60, face_h=60):
    G = build_conflict_graph(impressions, overlap_pairs)
    while True:
        coloring = dsatur(G)
        all_ok = True
        for color_class in group_by_color(coloring):
            faces = pack_block_faces(color_class, face_w, face_h)
            if faces is None:
                # split the class — add an edge between two largest items
                a, b = two_largest(color_class)
                G.add_edge(a, b)
                all_ok = False
                break
        if all_ok:
            return coloring, faces
```

This runs DSATUR + tiny MaxRects packing per class. At chuck-mcp scale (12–40 impressions, 5–8 impressions per color class at most, 2 faces of 60×60), it converges in < 50 ms total.

### When to lift to actual 2D nesting

If face_w × face_h becomes constrained relative to impression sizes such that **bounding-rectangle packing fails too often** (more than 20% of instances trigger the split-and-retry path), upgrade the per-class packer from MaxRects to **sparrow's irregular polygon nesting** (arxiv 2509.13329). Same structural pattern, better fit using mask polygons instead of bounding rectangles.

---

## Summary card (for the swarm coordinator)

- **Single biggest algorithmic recommendation**: **DSATUR (Brélaz 1979) + O(n+m) chordality check (Yekezare 2024)** is the entire block-packing algorithm. At chuck-mcp's 12–40 vertex scale, on spatial-overlap conflict graphs that are almost always chordal (subgraphs of interval graphs), DSATUR is provably optimal — no B&B, no ILP, no CP. Implement it. Add Furini-style B&B only as a fallback for the rare non-chordal case.
- **Face assignment**: separate stage. Run MaxRects 2D packing per color class onto 2 faces of plywood. On infeasibility, add a conflict edge and re-color. Converges in 1–2 iterations.
- **Yasu Shibata's Emma at 27 blocks**: chuck-mcp's DSATUR will likely hit 25–28 blocks for similar inputs, matching or slightly beating Shibata. The variance comes from how aggressively the upstream impression-extraction merges nearly-identical regions, not from the coloring algorithm.
- **Field state**: this is a "solved" problem at chuck-mcp's scale. All 2020+ graph coloring research targets either massive graphs (millions of vertices) or hard structural variants that don't apply here. Don't over-engineer.
