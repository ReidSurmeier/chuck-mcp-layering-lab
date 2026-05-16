# Reconstruction Plan After Example-Sheet Review

Created: 2026-05-16

Reference folder:

`/srv/woodblock-share/Examples`

Latest failed sheet:

`/srv/woodblock-share/LATEST_CHUCK_MCP_ALL_BLOCKS_CONTACT_SHEET.png`

## Blunt Diagnosis

The v13 all-block sheet is not close enough. It is not just visually weak; it is using the wrong representation.

The current "blocks" look like faded residual/cumulative render states. Several late blocks are basically partial final portraits. That is not what the example sheets show. A real block plate should be a physically inkable, reversed plate containing assigned color zones. A proof sheet should be a cumulative print after one or more of those plates are pulled. v13 confuses those two surfaces.

## Specific Comparison To The Example Folder

### `Screenshot 2026-05-14 at 6.13.05 PM.png`

This is the closest target for proof-state behavior.

What the example does:

- Shows cumulative proofs, not individual blocks.
- The first proof is pale and incomplete, but already has the portrait grid/cell logic.
- Each following proof adds a visible family of new marks.
- The image gains density in stages: warm face, reddish/brown structure, cool/dark contour, then background/key density.
- The proof states do not look like independent plates.

What v13 gets wrong:

- The all-block sheet shows "block" images that look like cumulative states.
- Blocks 24-26 are essentially finished composite images, which is not a believable physical plate.
- The growth rhythm exists in the pull preview, but it is disconnected from plate logic.
- There is no clean distinction between "this is a block" and "this is the print after this block."

### `Screenshot 2026-05-10 at 5.49.23 PM.png`

This is the strongest example for individual Emma-style plates.

What the example does:

- Shows the finished print beside separate underlayer plates.
- The plates are reversed relative to the final print, because blocks flip when printed.
- Each block has a wood/base field and clear painted/inked regions.
- The first layer can be light yellow and still detailed, but it is an underlayer role, not a full final-image reconstruction.
- Multiple colors can live on one block, but they are organized as discrete jigsaw zones.

What v13 gets wrong:

- Blocks are not shown as reversed print plates.
- Many blocks repeat the whole face silhouette instead of isolating a printable role.
- Plate zones are not jigsaw-organized; they read as low-opacity whole-image residue.
- The yellow/light layers are too ghosted and too globally derived, not discrete inkable shapes.
- Color zones do not have the hard production boundaries seen in the reference plates.

### `12341243.png`

This is the user's annotated methodology correction.

What the example says:

- The first visible layer is light yellow.
- These are underlayers.
- The underlayer plates are reversed.
- The goal is not broad-only. Early plates can contain detailed geometry, but their role is still supportive and local.

What v13 gets wrong:

- It corrected "broad only" too far into "everything is a faint copy of the final image."
- It does not enforce plate reversal as a first-class output.
- It lacks a real underlayer classifier: yellow, blue, pink, orange, green, and black are not being assigned production roles before optimization.

### `Untitled-1.png`

This shows painting/print relation.

What the example does:

- The painting/source has dense cell color.
- The finished print is a translated construction, not a pixel-level reconstruction.
- Color marks are discrete, bounded, and layered.
- The image works because many small colored decisions are grouped into larger printmaking decisions.

What v13 gets wrong:

- It uses cell-like texture, but the grouping is not printmaking-aware.
- It does not decide which cells belong together on one physical block.
- It does not preserve the feeling that each pull is an intentional color decision.

### `Woodblock-print-process.png`

This is a clean didactic model.

What the example does:

- Shows a plate/color at the top and the cumulative print below.
- Every step has a specific color role.
- Print order is explicit.
- The user can understand which plate caused which visible change.

What v13 gets wrong:

- The output surfaces are not paired this way.
- There is no per-block "plate above / cumulative result below" artifact.
- Because the blocks are not cleanly role-separated, the user cannot inspect why a given pull exists.

## Reconstruction Principles

1. Separate model surfaces:
   - `Plate`: reversed, physical, inkable block map.
   - `Pull`: one impression from one plate with one or more color zones.
   - `ProofState`: cumulative rendered print after a group of pulls.

2. Generate plates first, not faded cumulative images:
   - A plate is a sparse set of assigned cells/zones.
   - It may be detailed.
   - It must not look like the final print unless it is a key/detail plate with a specific role.

3. Use proof states as validation, not as plate assets:
   - Proof 01 might be after pulls 1-4.
   - Proof 02 after pulls 5-8.
   - Proof sheets should mimic the 8-up Chuck Close example.

4. Treat pigment/color names as flexible:
   - The planner can synthesize target batch colors.
   - Pigment names are mixing guidance, not a fixed palette bottleneck.

5. Make jigsaw grouping upstream:
   - Cells must be assigned to blocks before final rendering.
   - Adjacent or visually related cells should be grouped into inkable zones.
   - Zones on one block must be separated enough to brush/ink intentionally.

6. Enforce reversal:
   - Block previews must be mirrored relative to the print.
   - Pull/proof previews must be normal orientation.

## New Pipeline Plan

### Stage 1: Analyze Source Into Print Cells

Inputs:

- Target image.
- Optional grid/cell prior.
- Superpixel or cell graph segmentation.

Outputs:

- `CellGraph`: nodes are print cells/regions; edges are adjacency.
- Per-cell attributes: mean color, luminance, chroma, local contrast, target role, edge/detail score.

Required change:

- Stop solving free alpha maps as the primary representation.
- Use cells/zones as the primary object for block planning.

### Stage 2: Assign Roles Before Solving

Roles:

- `underlayer_light`: yellow, pale pink, pale blue, light orange support.
- `local_chroma`: red, orange, green, blue, magenta accents.
- `regional_mass`: hair, cheek, neck, clothing, background fields.
- `key_detail`: dark hair, eyes, mouth, contour, final density.

Rules:

- Early underlayers may contain detailed geometry, but each one has a limited color/opacity role.
- Red gets its own high-chroma role when present.
- Dark/key roles are held until late proof states.
- A role can create several blocks if there are too many disconnected or incompatible zones.

### Stage 3: Group Cells Into Physical Blocks

Use graph partitioning, not pixel masks.

Objective:

- Group same-role cells with compatible color batches.
- Keep physically separated jigsaw areas brushable.
- Limit each block to a small number of color batches unless the geometry is clearly separated.
- Penalize tiny orphan islands, repeated whole-face silhouettes, and blocks that resemble cumulative prints.

Hard checks:

- Every block has a plate role.
- Every block has a mirrored plate preview.
- No block can explain more than a capped fraction of final-image color error alone.
- No late block may be a full composite clone.

### Stage 4: Render Pulls From Plates

Each block produces:

- `plate_preview_mirrored.png`
- `plate_zones_overlay.png`
- `pull_after_normal_orientation.png`
- `zone_manifest.json`

Rendering:

- Convert mirrored plate zones back into normal orientation for the cumulative pull.
- Simulate transparent/semi-opaque overprint.
- Track pigment load/opacity per zone.

### Stage 5: Make Example-Matched Review Sheets

Required outputs:

- `all_blocks_plate_contact_sheet.png`: all reversed plates only.
- `plate_and_pull_contact_sheet.png`: each plate above its cumulative result, like `Woodblock-print-process.png`.
- `proof_state_sheet.png`: 8-up cumulative proofs like `Screenshot 2026-05-14`.
- `target_vs_final.png`.
- `first_four_underlayers_sheet.png`: compare against the annotated light-yellow/blue/pink/orange underlayer examples.

### Stage 6: Score Against The Example Methodology

Validation metrics:

- `plate_not_composite_score`: penalizes a block that looks like the final image.
- `role_purity_score`: each block must have a clear print role.
- `jigsaw_separation_score`: zones on a block need distinct, brushable boundaries.
- `proof_progression_score`: proof states must add visible families over time.
- `underlayer_reversal_check`: plate previews are mirrored; pulls are not.
- `final_match_score`: color/structure match after all pulls.

Gate before accepting a run:

- First proof must be incomplete but coherent.
- Individual plates must be readable as plates, not faded final images.
- Last proof must carry the strongest dark/key work.
- The all-block contact sheet must resemble the Emma plate examples more than a stack of residual renders.

## Immediate Implementation Tasks

1. Add explicit `Plate`, `Pull`, and `ProofState` domain objects to the v23 methodology path.
2. Replace v13 block image generation with mirrored plate previews built from assigned cell groups.
3. Add a cell-graph role assignment pass before adaptive ink stack generation.
4. Add a block grouping pass that partitions cells by role, color compatibility, adjacency, and brushability.
5. Add the five required review sheets listed above.
6. Add validators that fail if plate previews look like cumulative prints.
7. Re-run against Emma input and compare only after these sheet types exist.

## Acceptance Target For The Next Run

The next run does not need to be final CNC geometry. It does need to prove the representation is fixed:

- Blocks must be reversed plates.
- Proofs must be cumulative states.
- Early yellow/pink/blue/orange plates must be separate underlayer roles.
- A block must contain grouped jigsaw zones, not global ghost texture.
- The final proof can still be imperfect, but the process must look like the example sheets.
