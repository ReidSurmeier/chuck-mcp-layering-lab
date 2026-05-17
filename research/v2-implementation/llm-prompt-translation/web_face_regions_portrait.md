# Face Region Vocabulary for Portrait Prompts

For chuck-mcp v2's portrait use case, the LLM must resolve natural-language region references ("cheek", "under the eye", "temple") to specific labeled masks in Reid's Photoshop overlay.

## Authoritative source — the overlay is canonical, not the LLM

**Critical design decision:** the **Photoshop overlay is the ground truth for spatial reference**, not the LLM. Reid hand-labels 4–9 underlayer masks per portrait. The LLM's job is to match natural-language descriptors to the **labels Reid already wrote** on those layers.

This is much simpler and more robust than running a face-landmark detector. It also handles non-portrait inputs (still lifes, abstract compositions) with zero changes.

## Recommended overlay-label vocabulary (chuck-mcp v2 convention)

Reid's labels should follow this controlled list when possible. If user invents a label, the LLM accepts it and uses it verbatim.

### Portraits — primary regions

| Canonical label | Synonyms the LLM accepts | Notes |
|---|---|---|
| `skin_forehead` | forehead, brow plane, T-zone upper | Yellower zone (Bailey) |
| `skin_cheek_l`, `skin_cheek_r` | cheek, cheekbone, zygomatic | Pinkest zone, most blood flow |
| `skin_temple_l`, `skin_temple_r` | temple | Often shadow side |
| `skin_nose_bridge` | nose bridge | Middle (pinker) zone |
| `skin_nose_tip` | nose, nose tip | Often most red |
| `skin_chin` | chin, jaw | Greenish for women/children, bluer for men (Bailey's third zone) |
| `skin_neck` | neck | Cools toward chest |
| `lip_outer` | lip, mouth, lips | Highest chroma red region |
| `lip_inner` | inner lip, lip line | Darker, often deepest hue |
| `eye_white_l`, `eye_white_r` | eye white, sclera | Almost-white, slight warm tint |
| `eye_iris_l`, `eye_iris_r` | iris, eye color | Highly variable |
| `eye_socket_l`, `eye_socket_r` | under eye, eye socket, eye bag | "Under the eye" in user prompt |
| `eyebrow_l`, `eyebrow_r` | eyebrow, brow | Hair color territory |
| `hair_main` | hair | Bulk hair |
| `hair_under` | hair underlayer, hair shadow | When user says "deep umber under indigo" hair |
| `background` | bg, background, behind |  |

### Three-zone shortcut (Bailey 2018)

User-facing concept: "the three color zones of the face." The LLM should recognize:

- **Zone 1 — Yellow Zone:** maps to `skin_forehead`
- **Zone 2 — Red Zone:** maps to `skin_cheek_l`, `skin_cheek_r`, `skin_nose_tip`, `skin_nose_bridge`, ears
- **Zone 3 — Cool Zone:** maps to `skin_chin`, `skin_neck`

So when user says "warm-tonal Emma. Skin between #f7d8c2 and #e0a888", the LLM should:
1. Default `skin` to mean **all skin labels** (forehead/cheek/temple/etc), not one specific region.
2. Unless qualifying language present ("cheeks specifically pinker", "shadow side cooler"), apply the tone range uniformly across all skin masks.
3. Surface this default in `ambiguities_flagged`.

## When overlay has unlabeled masks

If the user says "under the eye" and there is no `skin_under_eye` mask in the overlay, the LLM should:
1. Emit `ambiguities_flagged: ["region 'under the eye' has no matching overlay mask; nearest matches: skin_cheek, eye_socket"]`
2. Pick the most likely match (`eye_socket_l`/`eye_socket_r`) and proceed with `confidence: "low"`.

User clarifies on next iteration. **Do not silently fail.**

## Fallback: MediaPipe FaceMesh for unlabeled portraits

If the input is a portrait but Reid hasn't done the overlay yet, fall back to MediaPipe FaceMesh:

- 468 surface landmarks + 10 iris landmarks = 478 total
- Sparse coverage on forehead (mostly hairline-edge points)
- Dense coverage on lips, eyes, irises
- Use Google's published landmark connectivity to derive a rough mask for each canonical label

This is a **degraded mode**, not the primary path. Reid's hand overlay is always better. Document MediaPipe purely as the "no-overlay smoke-test" backup.

## Non-portrait inputs

For still lifes, abstract work, landscapes: the user's overlay labels are the entire vocabulary. The LLM operates without any anatomical priors. This is fine — the structured-extraction problem is unchanged, just smaller vocabulary.

## Sources

- [Heather Bailey — Color Zones of the Face](https://heatherbailey.com/blogs/fine-art-4/%F0%9F%92%A5-c-o-l-o-r-z-o-n-e-s-of-the-face-%F0%9F%92%A5)
- [Renso Art — Anatomy of the Face for Artists](https://www.rensoart.com/the-anatomy-of-the-face-a-guide-for-artists/)
- [Portrait Society — Anatomy of the Head for Artists](https://www.portraitsociety.org/single-post/anatomy-of-the-head-for-artists)
- [MediaPipe FaceMesh documentation](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh)
- [MediaPipe 478 Landmark Diagram](https://www.sanderdesnaijer.com/blog/mediapipe-face-mesh-landmarks)
