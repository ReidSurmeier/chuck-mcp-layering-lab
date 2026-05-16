# Chuck MCP Output Archive

Created 2026-05-16.

The shared output tree is organized at:

`/srv/woodblock-share/chuck-mcp-iterations`

This replaces ambiguous pointers like `latest-emma` with dated iteration folders and standardized contact sheet names.

## Structure

- `current-review/`: copied snapshots of the most recent useful variations for visual review.
- `contact-sheets/`: dated copies of pull previews, proof previews, contact sheets, and target-vs-final images.
- `archived-source-folders/`: original output folders moved from the share root.
- `references/`: user annotations and example references copied from the share root.
- `manifests/`: folder map and contact-sheet index.

## Current Review Iterations

- `2026-05-14_iter-v13_methodology-adaptive-proofs-emma`
- `2026-05-14_iter-v12_methodology-adaptive-proofs-emma`
- `2026-05-14_iter-v11_methodology-proof-v2-emma`
- `2026-05-14_iter-v10_methodology-proof-v1-emma`
- `2026-05-14_iter-v12_clean-handoff`
- `2026-05-14_iter-v09_batch-production-carousel-v2`

## Naming Scheme

Contact sheet files use:

`YYYY-MM-DD[_HHMM]_<iteration-folder>_<artifact-kind>.<ext>`

Examples:

- `2026-05-14_methodology-adaptive-proofs-emma-v13-20260514_proof-preview.png`
- `2026-05-14_1735_chuck-batch-production-carousel-v2-fast-m10-main-20260514-1735_production-pull-block-grid.png`

The generated index is:

`/srv/woodblock-share/chuck-mcp-iterations/manifests/2026-05-16_contact-sheet-index.tsv`
