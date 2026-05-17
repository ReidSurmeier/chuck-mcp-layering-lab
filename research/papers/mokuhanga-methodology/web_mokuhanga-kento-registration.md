# Kento Registration: Traditional System and CNC Adaptation

**Primary URLs**:  
- https://mlyon.com/2012/post-digital-printmaking/  
- https://1000woodcuts.blogspot.com/2010/04/registration-simplified.html  
- http://www.aprilvollmer.com/mokuhanga-beyond-the-basics/  
- https://www.elizabethforrest.ca/what-is-mokuhanga  
**Fetch date**: 2026-05-16  
**Evidence grade**: High for traditional system; Medium for CNC tolerance figures

---

## The Kento System

Kento is the registration system that makes multi-block mokuhanga possible. Two notches are carved directly into the woodblock surface — typically in the lower-right corner of the printing area. The standard configuration is one L-shaped notch (kagi-kento) at the corner and one straight notch (hikitsuke-kento) along the bottom edge. The paper is registered by pressing its corner into the L-notch and its edge against the straight notch before each impression.

Critically, kento notches are carved into the block itself, not added as external fixtures. This means registration is block-specific and must be established identically on every block in the set. In traditional ukiyo-e production, the han-giri (block cutter) was responsible for precise kento placement; in contemporary fine art practice, this falls to the master printer.

## Tolerance in Practice

Salter's "Japanese Woodblock Printing" (2002/2005) identifies kento registration as one of the three defining elements of quality mokuhanga (alongside washi and baren). Working practitioners cite achievable registration tolerance of approximately 0.2–0.5mm under good conditions (dry studio, stable paper moisture, consistent baren pressure). Print scholar Rebecca Salter states the system allows for "precise" color-on-color registration when all variables are controlled.

The primary degradation mechanisms are:
1. Paper expansion/contraction from moisture uptake between passes
2. Wood dimensional change from repeated wetting during printing
3. Wear on the kento notches over a long edition run
4. Inconsistent paper placement if the printer is fatigued

For a 132-pass edition like Emma, paper dimensional stability is the primary challenge — Shiramine's internal sizing addresses this by reducing moisture absorption variance between sheets.

## CNC Adaptation: Mike Lyon's System

Artist Mike Lyon (documented in "Post-Digital Printmaking," 2012) developed a CNC-to-mokuhanga workflow. He uses digital Photoshop luminosity layer separation to generate separate bitmap files for each carving pass, then uses a CNC router to cut the blocks. Critical to this workflow: kento notches are added manually to the bitmap files after image separation and before CNC cutting. The notches must be placed in identical positions in all bitmap files; the CNC cuts them as part of the block carving.

Lyon's practice spans works with 12–17 blocks in tight registration, including prints up to 195.5 × 107 cm. He describes "islands of wood that are not visually represented in the final print" serving dual registration and paper-support roles.

The implication for chuck-mcp: when generating SVG exports for laser-cut or CNC blocks, kento notch geometry must be added as a fixed overlay — ideally as a non-printing registration layer in the SVG file — at a position that is pixel-identical across all block exports. The notch coordinate must be locked before any color separation is run.

## Vector Export Precision Requirement

Given that achievable kento tolerance is ~0.3mm and that contemporary fine art prints operate at the upper end of this tolerance range over many passes, the chuck-mcp SVG export pipeline needs to:
- Snap all block registration mark coordinates to a fixed grid
- Export registration marks at the same document-coordinate position across all SVG layers
- Not allow any layer-specific transform or offset to affect the registration mark position

A 0.5mm misregistration across 132 passes becomes visible as a halo or soft edge — unacceptable at Pace Editions production quality.

