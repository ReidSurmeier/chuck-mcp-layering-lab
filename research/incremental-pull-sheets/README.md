# Incremental Pull Sheet Research Base

Created 2026-05-16 for Chuck MCP methodology validation.

The images in this folder are intentionally tracked because this repo is being
used as a private research/build archive for Chuck MCP.

This folder separates three different evidence types that were getting mixed together during solver review:

- **Cumulative state/progressive proofs**: rows of prints showing the image after groups of blocks have been pulled.
- **Individual underlayer/block plates**: carved or inked blocks that may already contain detailed geometry, but are not themselves proof states.
- **Chuck MCP output checks**: generated previews to compare against the real reference rhythm.

The Pugnet/Windows printmaking folder is reachable in this WSL session as:

`/mnt/c/Users/reidsurmeier2/Books/printmaking`

I do not see a separate mounted SMB path named `pugnet` in `mount`; the Windows folder above appears to be the accessible copy.

## Local Assets

| Folder | File | Source | Use |
|---|---|---|---|
| `local-progressive-proofs/` | `chuck-close-progressive-proofs-aiap.jpg` | `/mnt/c/.../_chuck-close-blocks/exhibitions/aiap_dart_progressive_proofs.jpg` | Clean eight-state Chuck Close progressive proof sheet. The primary visual target for proof-preview layout. |
| `local-progressive-proofs/` | `chuck-close-progressive-proofs-screenshot-2026-05-14.png` | `/srv/woodblock-share/Screenshot 2026-05-14 at 6.13.05 PM.png` | Higher-resolution screenshot of the same cumulative proof logic. |
| `local-progressive-proofs/` | `user-annotated-emma-underlayer-blocks.png` | `/srv/woodblock-share/Examples/Screenshot 2026-05-10 at 5.49.23 PM.png` | User annotation showing finished Emma beside individual underlayer plates. |
| `local-progressive-proofs/` | `user-annotated-emma-underlayer-methodology.png` | `/srv/woodblock-share/12341243.png` | User annotation emphasizing that early yellow/light plates are underlayers and are reversed because blocks flip in printing. |
| `emma-underlayer-plates/` | `plate-emma-blocks-130.jpg` | `/mnt/c/.../_chuck-close-blocks/sultan-natives-renamed/plate-emma-blocks-130.jpg` | Native Sultan page: Emma blocks 1, 4, 8, 12. |
| `emma-underlayer-plates/` | `plate-emma-blocks-131.jpg` | `/mnt/c/.../_chuck-close-blocks/sultan-natives-renamed/plate-emma-blocks-131.jpg` | Native Sultan page: Emma blocks 15, 16, 20, 23. |
| `pace-self-portrait-2015/` | `2017-03-Close_01.jpg` through `2017-03-Close_16.jpg` | `/mnt/c/.../_chuck-close-blocks/exhibitions/paceprints_2017_selfportrait/` | Pace 2017 installation/process photos for Self-Portrait 2015, including woodblocks and state proofs. |
| `chuck-mcp-output-checks/` | `latest-methodology-full-pull-preview.png` | `/srv/woodblock-share/chuck-methodology-proofs/latest-emma/` | Latest generated full pull preview to compare against references. |
| `chuck-mcp-output-checks/` | `latest-methodology-proof-preview.png` | `/srv/woodblock-share/chuck-methodology-proofs/latest-emma/` | Latest generated grouped proof preview to compare against references. |

## Grounded Observations

1. A proof state is not a block. In the Chuck Close 2015 Self-Portrait reference, each proof is a cumulative image after several colors/blocks have been added.
2. A block can contain multiple color zones. The Pace/Artsy process notes describe blocks carrying multiple variations of color and printer markings that delineate islands of like color.
3. The first block can look detailed without being a finished proof. The Emma underlayer plates show real carved detail and local color decisions even on early/light blocks.
4. Light-first does not mean broad-only. Yellow, pink, pale blue, and light orange often come first because they preserve luminance and provide transparent support for later darker pulls.
5. Later proof states should progressively resolve dark/key structure. The real progressive sheet gains chroma and contour in visible batches instead of jumping from pale scaffold to full reconstruction.
6. Count is adaptive. Emma's 27 blocks and Self-Portrait 2015's 24 blocks are evidence of complexity scale, not constants for every input image.

## Web Sources

- Pace Prints, *Chuck Close: Self-Portrait, 2015, Print & Process*: documents the 2017 exhibition, 24 blocks, progressive proofs, and a published print. Source says 84 colors.  
  https://paceprints.com/2017/chuck-close-self-portrait-2015-print-process
- Artsy/Pace Prints, *Behind-the-Scenes: Chuck Close's "Self-Portrait" (2015)*: documents the state proofs, 24 blocks, color islands, multiple color variations per block, and the role of Hecksher, Israels, and Shibata. Source says 86 colors.  
  https://www.artsy.net/article/pace-prints-behind-the-scenes-chuck-close-s-self-portrait-2015
- AI-AP, *Chuck Close at Pace Prints*: explicitly describes progressive proofs showing how layers of ink build up, plus eight of the twenty-four blocks.  
  https://www.ai-ap.com/publications/article/21218/chuck-close-at-pace-prints.html
- Metropolitan Museum of Art, *Chuck Close Prints: Process and Collaboration*: distinguishes progressive proofs and state proofs, and documents Emma as a 113-color Japanese-style woodblock made from 27 blocks over about two years.  
  https://www.metmuseum.org/exhibitions/listings/2004/chuck-close
- National Gallery of Art, *Yes, No, Maybe: Artists Working at Crown Point Press*: frames working proofs as decision records used to steer a print's evolution.  
  https://www.nga.gov/exhibitions/yes-no-maybe-artists-working-crown-point-press
- Crown Point Press bookstore, *Yes, No, Maybe*: corroborates the exhibition/catalog focus on working proofs and process decisions.  
  https://store.crownpoint.com/products/yes-no-maybe-artists-working-at-crown-point-press
- Helen Frankenthaler Foundation, *No Rules: Helen Frankenthaler Woodcuts*: useful non-Close precedent for jigsaw color-block thinking, overprinting support layers, and painterly woodcut translation.  
  https://www.frankenthalerfoundation.org/exhibitions/no-rules-helen-frankenthaler-woodcuts
- Pace Prints, *Helen Frankenthaler: Woodcuts, 1998-2009*: documents Frankenthaler/Shibata ukiyo-e collaborations and the high block/color counts of painterly woodcuts.  
  https://paceprints.com/2019/helen-frankenthaler-woodcuts-1998-2009
- Pace Prints, *Jonas Wood: Five Bonsais*: modern Shibata precedent where adaptive block counts and many hand-applied colors are normal.  
  https://paceprints.com/2024/jonas-wood

## Implications For Chuck MCP

- The renderer should expose both **block plates** and **proof states**. The proof state sequence should group several blocks at a time and show cumulative development.
- The solver should optimize against a proof rhythm, not only a final image. A plausible rhythm starts with pale support scaffolds, then regional color masses, then dark/key structure.
- A block planner should allow detailed carved geometry on early blocks while still penalizing full final-image responsibility in the first proof state.
- Color grouping should be based on printable zones and jigsaw adjacency, not a fixed pigment list. Pigment names are output guidance; the internal palette can expand to mixed target colors.
- Validation should score: final match, state progression, light-to-dark overlap plausibility, multi-zone block clarity, and whether color islands are grouped into print-safe separations.
