# Woodblock (v22.1 mokuhanga)

A tool that ingests a single PNG/JPG and produces a CNC-ready multi-impression carving + printing plan for mokuhanga (Japanese woodblock with water-based pigments on washi). v22.1 reframes the deliverable as **inverse stack solving** — proposing a plausible impression sequence whose forward-render matches the target image — not flat color separation.

## Language

**Block**:
A physical woodblock — one piece of maple plywood that the artist carves.
_Avoid_: plate, sheet, panel.

**Impression**:
One atomic (**Pigment**, **Mask**) printing application at one position in the **Order**, assigned to one physical **Block**. Many Impressions may share the same `(block_id, order_step, pull_group)` triple — they are inked together and printed in a single physical rub-and-pull (Pace-Editions process).
_Avoid_: layer, plate-pull, pass (ambiguous).

**Pull group**:
A set of **Impressions** that share a physical printing pull — same **Block**, same **Order** step, inked simultaneously, transferred to paper in one rub. Identified by `pull_group` integer in the manifest. Display-time grouping only; the solver treats each Impression as an independent atom.
_Avoid_: pull, pass.

**Mask**:
The printed area for one **Impression**. Stored as a three-state grid: `visible` (still showing in the final image), `covered` (printed but fully hidden by later impressions), `support` (printed and partially showing through later overprints). Optional soft α companion in `[0,1]`.
_Avoid_: layer, alpha, region.

**Pigment**:
A hue + density + opacity recipe applied during one **Impression**. Catalog seed: 13 Mixbox-anchored pigments mapped to mokuhanga water-based equivalents.
_Avoid_: color, ink, paint.

**Order**:
The sequence in which **Impressions** are printed (light → dark by default, mokuhanga convention).
_Avoid_: stack order (ambiguous), pass order.

**Underprint**:
A **Mask** region intentionally printed below later **Impressions** to shift their final visible color. The mechanism v21 failed to model.
_Avoid_: underbase (screen-print-specific), underlayer (overloaded).

**Stack**:
The ordered list of **Impressions** for one print. Output of the inverse solver.
_Avoid_: separation, plate set.

**Plan**:
The full machine-readable output: `Stack` + per-**Impression** **Mask** + **Block** assignments (DSATUR-packed) + **Order** + per-region confidence labels. Exported as a ZIP.
_Avoid_: recipe, instructions.

**Strategy library**:
A corpus of registered `(final image, block scan stack)` tuples used as inductive bias when solving new images. Emma is the seed entry. Library queries return hue-family pattern, typical impression count, and grouping heuristics.

## Relationships

- A **Plan** is one **Stack** + supporting metadata.
- A **Stack** is an ordered list of **Impressions**.
- An **Impression** has exactly one **Mask**, exactly one **Pigment**, exactly one **Block** assignment, and one position in **Order**.
- A **Block** carries one-or-more **Impressions** (`block_face_id` distinguishes them when on the same physical block).
- A **Mask** classifies every pixel as `visible | covered | support | none` relative to the rest of the **Stack**.
- An **Underprint** is an **Impression** whose **Mask** is mostly `covered` or `support`, never mostly `visible`.

## Confidence labels (per Mask region)

- **confirmed-from-scan** — derived from a registered block scan in the **Strategy library**
- **visible-in-final** — directly observed pigment cluster in the target image
- **inferred-underprint** — solver-proposed, forward-render gate passed
- **ambiguous** — multiple recipes within ΔE budget; top-N alternatives returned

## Example dialogue

> **Dev:** "Block 7 carries the cyan pigment, right?"
> **Artist:** "No — Block 7 is the physical piece of maple plywood. It carries *two* Impressions: a pale cyan first-pass at Order=3 (mostly support pixels under the later detail work) and the dark teal shadow second-pass at Order=11. Same physical Block, two Impressions, two Masks, two Pigments."

## Flagged ambiguities

- v21 called the DSATUR-coloring output "blocks" but they were really one-Impression-per-block partitions — closer to "single-Impression Block sets." v22 separates the concepts.
- "Layer" was banned: in v21 docs it meant Mask, in v22 user docs it sometimes meant Impression, and in Qwen-Image-Layered it means RGBA channels. Use **Impression** for the print pass and **Mask** for the printed region.
- "Underlayer" was banned: heritage science uses it for physical analytical findings; v22 outputs "inferred-underprint" candidates only, never claimed-recovered "underlayers."
