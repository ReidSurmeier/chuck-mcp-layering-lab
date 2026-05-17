# NOTES.md — Mokuhanga Methodology Research Synthesis

Research agent: MOKUHANGA-METHODOLOGY  
Date: 2026-05-16  
Swarm: chuck-mcp-layering-lab (swarm-1778962124344-s4cm4l)

---

## 1. Yasu Shibata Production Estimate Breakdown

### Emma (2002) Confirmed Figures
- 27 blocks carved
- 113 colors achieved
- 132 printing passes per impression
- Shiramine paper (110 gsm, internally sized Kozo/alpha-cellulose/hemp)
- Edition: 55 + 8 APs
- Production: ~18–24 months

### How 27 → 113 → 132 Maps to Chuck-MCP Batch Structure

The critical structural insight: **blocks are form libraries, not color carriers.**

**Mathematical decomposition:**
- 132 passes ÷ 27 blocks = ~4.9 average passes per block
- 113 colors ÷ 27 blocks = ~4.2 unique inks per block across all its passes
- 132 passes ÷ 113 colors = ~1.17 passes per color (most colors appear once; a few appear on 2–3 passes for depth)

**Chuck-MCP batch structure mapping:**

| Current Chuck-MCP "batch" | Shibata equivalent | Pass count estimate | Block reuse |
|---|---|---|---|
| Light broad-tone (4 passes) | First 15–20 passes: pale washes, bokashi sky/background | ~20 passes | 3–5 blocks used multiple times each |
| Mid-tone main (4 passes) | Middle 60–80 passes: primary hue-building, Close's color grid cells | ~80 passes | 15–20 blocks, each reprinted 4–6× |
| Detail pass | Final 30–35 passes: dark anchors, saturation, subtle hue shifts | ~30 passes | Most blocks reprinted with darker or more saturated variants |

**Key implication**: The "4 + 4 + detail" frame is too coarse. The actual Shibata system is better described as:
- ~15% of passes: broad-tone foundation (transparent, light, large-area coverage)
- ~60% of passes: hue-grid construction (the color mosaic, one cell type per pass cluster)
- ~25% of passes: depth/saturation/dark anchor passes (reprinting earlier blocks with deeper color)

The current 8–12 impression "study stacks" cover only the first two phases. Scaling toward Pace-scale requires modeling the 25% depth-building phase that the study stacks currently omit.

---

## 2. Five Concrete Recommendations for Chuck-MCP Planner

### Recommendation 1: Decouple block count from color count
The planner currently treats blocks and colors as nearly 1:1. Shibata's Emma uses 27 blocks for 113 colors. The planner should maintain a (block, pass_index, ink_mixture) triple as the fundamental unit — not (block → color). A block should be reusable across multiple passes with different ink configurations. This requires the solver to plan for block re-inking, not just block assignment.

### Recommendation 2: Enforce light-to-dark pass sequencing
Sequencing is mechanically enforced by mokuhanga physics, not stylistic preference. The planner should sort planned passes by approximate luminosity of the target impression: lightest/most transparent first, darkest/most opaque last. An exception: if sumi-type (deep black or very dark) anchor tones are planned, schedule them in the final 10–15% of passes, not earlier. The current "broad tone first" assumption is correct directionally but needs to be formalized as a hard constraint.

### Recommendation 3: Lock kento coordinates before any color separation runs
The kento registration system requires that registration marks occupy the same coordinate position in every block's SVG export. This coordinate must be determined once, stored as a project-level constant, and injected into every generated SVG as an immovable overlay — not computed per-layer. A 0.3mm error in kento position compounds across 132 passes into a visible misregistration. Recommended: store kento_origin_x and kento_origin_y in the project config, enforce via schema validation that both marks are present at those exact coordinates in all generated outputs.

### Recommendation 4: Classify passes by type, not just "impression N"
Each planned pass should carry a type tag: [broad_wash | flat_tone | bokashi_gradient | hue_cell | dark_anchor | reduction_carve]. This tag governs the ink preparation instructions (nori concentration, pigment load), baren pressure guidance, and paper conditioning requirements the planner should report. Bokashi passes require thin nori and rapid pressure application; dark anchor passes require thick nori and firm baren; broad washes require maximum paper dampening. The current solver produces a sequence of impressions without differentiating preparation requirements between pass types.

### Recommendation 5: Model first pass as lightest broad wash, regardless of color palette
The first impression should always be the lightest, most dilute, largest-area color pass in the stack — even if it is neutral (a pale grey or warm white tone). This serves dual purposes: it uniformly conditions the paper moisture for all subsequent passes, and it establishes the lightest value present in the final print. The solver should enforce this as a constraint: pass_index=1 must have opacity < 0.2 and coverage > 0.4 (by area fraction). This is not an aesthetic choice — it is the physics of how wet paper accepts subsequent impressions.

---

## 3. Pigment Family Taxonomy

### Historical Taxonomy (Traditional Mokuhanga)

| Family | Japanese term | Examples | Optical character |
|---|---|---|---|
| Mineral | Ganryo | Bengara (Fe₂O₃), gofun (CaCO₃), azurite, malachite | Opaque, high lightfastness |
| Carbon | Sumi | Pine-soot + nikawa (animal glue) | Opaque black, dense coverage |
| Organic/Lake | Senshoku-ryo | Beni (safflower), ai (indigo), gamboge | Transparent, fugitive |

### Contemporary Pigment Taxonomy (Shibata-era Practice)

Contemporary mokuhanga has replaced most historic organic pigments with synthetic analogs that preserve optical character while improving lightfastness:

| Contemporary family | CI names | Traditional analog | Optical character |
|---|---|---|---|
| Naphthol reds | PR112, PR188, PR170 | Beni-based reds | Semi-transparent to opaque, warm |
| Hansa yellows | PY3, PY74, PY97 | Gamboge | Transparent to semi-transparent |
| Quinacridone | PR122, PV19, PO48 | Safflower-based pinks/violets | Highly transparent |
| Phthalo blues/greens | PB15, PG7, PG36 | Ai-based blues | Highly transparent, high tinting strength |
| Pyrrole/Perylene reds | PR254, PR178 | None (modern synthetic) | Opaque, very lightfast |
| Iron oxide | PY42, PR101, PBr7 | Bengara family | Opaque, earth tones |
| Carbon black (sumi) | PBk6, PBk7 | Sumi (unchanged) | Opaque, dense, used last |
| Titanium white | PW6 | Gofun (oyster shell) | Opaque, used sparingly |

### Holbein Pigment Paste System (Contemporary Standard)

Holbein Pigment Pastes are the current industry standard for mokuhanga practitioners (Japan and internationally). Each color is 30–70% pure single pigment suspended in water + wetting agent + dispersant. They are pre-ground to printing consistency, eliminating the 45-minute traditional grinding step. The chuck-mcp catalog (36-entry) should be reorganized to:
1. Map each catalog entry to a CI pigment name
2. Tag each entry with transparency class (transparent / semi-transparent / opaque)
3. Tag each entry with tinting strength (high / medium / low)
4. Flag any dual-pigment mixes (these behave differently in overprint conditions)
5. Add a "historical analog" field connecting to the traditional family

Transparency and tinting strength are the two most important parameters for sequencing decisions in the planner.

---

## 4. Top 5 Books/Catalogs to Acquire

Listed in priority order for chuck-mcp research:

### 1. Sultan, Terrie and Shiff, Richard. "Chuck Close Prints: Process and Collaboration." Princeton University Press, 2003.
**Anna's Archive MD5**: `be57d6df27782b9d4240c6b5a005abf6`  
**Priority**: Highest — contains Shibata's own interview on the Emma process. The only primary-source written documentation of his methodology.

### 2. Salter, Rebecca. "Japanese Woodblock Printing." University of Hawai'i Press, 2002/2005.
**Anna's Archive MD5**: `bcb210ef60b44caba138bced4db5e78f` (2002) or `c9adbf1e3be6c57ab159730fd549e4c0` (2005)  
**Priority**: Very high — rigorous technique manual from a practitioner trained in the Kurosaki lineage. Contains kento geometry, baren construction, nori ratios, pigment history.

### 3. Vollmer, April. "Japanese Woodblock Print Workshop: A Modern Guide to the Ancient Art of Mokuhanga." Watson-Guptill, 2015.
**Anna's Archive MD5**: `33b961fffee2de3d08b3de7d9aaa1f2f` (EPUB)  
**Priority**: High — contemporary practice focus, explicit coverage of multi-block workflow for Western artists, color mixing overprint prediction.

### 4. Kanada, Margaret Miller. "Color Woodblock Printmaking: The Traditional Method of Ukiyo-e." Shufunotomo, 1989.
**Anna's Archive MD5**: `8cf899c47efeb66125da7c228eac529b`  
**Priority**: High — direct documentation of the traditional ukiyo-e production system (the formal ancestor of Shibata's practice). Color registration, block sequencing, pigment preparation.

### 5. Petit, Gaston. "Evolving Techniques in Japanese Woodblock Prints." Kodansha International, 1977.
**Anna's Archive MD5**: `9accceac41050053703e3b2bd3785d48`  
**Priority**: Medium — historical bridge between traditional ukiyo-e and the contemporary revival; documents how mid-20th-century Japanese artists (who trained the Kurosaki generation) adapted the technique.

**Also recommended (not in above top 5)**:  
MoMA catalog (Storr, Varnedoe, Wye): MD5 `8b9e665921cfaa8a9a0ea519f45bd407` — contains documentation of Close's full career including print practice, though pre-dates Emma.

---

## 5. The Surprising Methodology Fact: Shibata Uses Reduction Technique

**The finding that contradicts the current chuck-mcp assumption:**

The chuck-mcp solver implicitly treats each block as a static spatial mask used across its assigned passes. But Shibata uses **reduction woodcut technique** — a method in which the block itself is progressively carved away between passes.

**Evidence**: His own works are described as "reduction woodcuts" in exhibition catalogs (Aspinwall Editions, Shore Publishing). The technique is explicitly documented in his biographical materials. He combines reduction with multi-block — this hybrid is his defining technical signature.

**What this means for Emma (2002)**: Some of the 27 blocks may have been progressively modified (carved down) between their multiple passes. This means the spatial footprint of a block changes across the edition run. Block 7 in pass 22 may have a different carved surface than Block 7 in pass 51.

**Current chuck-mcp assumption this breaks**: The solver assumes (block_id, spatial_mask) is a fixed pair. In Shibata's hybrid system, the correct model is (block_id, pass_index, spatial_mask) where spatial_mask can evolve.

**Implication for the solver**: The planner needs a "block evolution" concept — a flag or list indicating which blocks undergo reduction (carving modification) between their passes, and at which pass index the reduction occurs. This cannot be inferred from the final print alone; it requires either Shibata's direct documentation or conservation access to the original blocks.

**Why this matters computationally**: If reduction is in play, the planning graph is not DAG-simple. The same block can have different spatial contributions at different points in the sequence. Any optimization that assumes static block masks across all passes will produce a suboptimal (or incorrect) plan.

---

## 6. File Index

| File | Topic | Evidence Grade |
|---|---|---|
| `web_chuck-close-emma-primary.md` | Emma production facts, Shiramine paper, 27/113/132 relationship | High |
| `web_yasu-shibata-process.md` | Shibata biography, dual technique, training lineage | Medium |
| `web_pace-editions-collaboration-model.md` | Pace production model, multi-work comparison table, approval loop | Medium |
| `web_mokuhanga-kento-registration.md` | Kento system, tolerance, CNC adaptation (Lyon), SVG implications | High/Medium |
| `web_baren-physics-printing-mechanics.md` | Baren structure, moisture-pressure interplay, bokashi parameters | High/Medium |
| `web_pigment-taxonomy-nori-chemistry.md` | Mineral/synthetic/lake taxonomy, nori ratios, sequencing rule | Medium |
| `web_shiramine-washi-paper-properties.md` | Shiramine specifications, fiber content, gsm, conditioning | High |
| `web_color-separation-maquette-to-block.md` | Kurosaki precedent, Lyon CNC, Shibata tracing/separation, 3D model | Medium |
| `web_salter-vollmer-technique-manuals.md` | Book catalog, Anna's hashes, Salter vs Vollmer comparison | High |
| `web_reduction-woodcut-and-layer-sequencing.md` | Reduction mechanism, sequencing rules, Shibata hybrid | High/Medium |
| `web_imc-conference-contemporary-practice.md` | IMC conference, Kurosaki legacy, practitioner community | Medium |
| `web_chuck-close-prints-exhibition-catalog.md` | Exhibition documentation, catalog interviews, Close quotes | High |
| `web_mokuhanga-revival-global-contemporary.md` | Contemporary revival context, Shibata's dual role, resource gap | Medium |

Total files: 13 (12 web artifacts + 1 NOTES.md)

