# `corpus/` — Validation-System Eval Fixtures

This directory holds the **committed** eval corpus for the woodblock plate-separator validation system. Every fixture is one image plus a JSON ground-truth annotation. The runner at `tests/eval/run_corpus.py` (V2+) walks this tree, runs the separator on each `original.*`, then scores against `annotations.json`.

See `~/Prompts/validation-system-v1.md` sections 3 + 12 + 13 for the full spec.

---

## Directory layout

```
corpus/
├── .gitattributes           # routes *.jpg, *.jpeg, *.png, *.tif, *.tiff through Git LFS
├── README.md                # this file
└── <image_id>/
    ├── original.{jpg,jpeg,png}
    └── annotations.json
```

`<image_id>` is a stable lowercase-snake-case slug. Once committed, **never rename** — golden-image regression keys against it.

---

## Tiering (per validation-system-v1.md §3)

| Tier | Definition | Commit? | This corpus |
|---|---|---|---|
| **A** | Public-domain or user-cleared-for-commit. Live test fixtures. | Yes (Git LFS) | All current fixtures |
| **B** | Synthetic (regeneratable from `tools/synth_corpus.py`). Generator + seeds committed; outputs not. | Generator only | not yet — V2 work |
| **C** | License-restricted. Env-gated via `EMMA_FIXTURE_PATH`. | Never | not used here per user directive |
| **D** | Pending classification. | tbd | not used |

**Per user directive 2026-05-10:** "dont worry about copyright" — every current fixture is tiered **A** regardless of provenance. No Tier-C env-gating is in effect for V0.

---

## Annotation schema

Every `annotations.json` conforms to this shape. Unknown values are `null`; never omit a field.

```jsonc
{
  "image_id": "hiroshige_edo_116",         // matches directory name
  "source_filename": "100-views-of-edo-8-by-ando-hiroshige-116.jpeg",
  "dimensions": [640, 930],                 // [width, height] in pixels (PIL convention)
  "tier": "A",
  "license_note": "User-provided test image, personal use",
  "category": "ukiyo-e",                    // ukiyo-e | mokuhanga | reference | user-work
  "block_count": 8,                         // integer or null
  "color_count": 12,                        // integer or null
  "print_order": null,                      // null | "light_to_dark" | ["yellow","pink",...]
  "block_masks": [],                        // [] | [{ "block_id": 1, "mask_svg": "block_01.svg", "colors": ["#a8c4d6"] }, ...]
  "difficulty": "medium",                   // easy | medium | hard | unknown
  "notes": "100 Views of Edo #116 ..."      // free-form provenance + ground-truth source
}
```

### Extended fields for museum-sourced fixtures

Fixtures pulled from a museum Open Access API may include additional verified fields:

- `source` — short label, e.g. `"Met OA"`, `"Brooklyn OA"`, `"LoC"`
- `source_url` — human-facing object page
- `image_url` — direct image download
- `accession_number`, `object_id` — provenance identifiers
- `title`, `artist`, `date`, `physical_dimensions_cm`
- `license` — verbatim license string (e.g. `"CC0 / Public Domain (Met OA)"`)

See `met_hokusai_great_wave_jp1847/annotations.json` for a worked example.

### Unicode-in-filename note

If the source filename contains non-ASCII (e.g. macOS time-format U+202F narrow no-break space in `Screenshot 2025-11-23 at 9.09.18 PM.png`), record the substitution in a `source_filename_unicode_note` field and keep `source_filename` ASCII-normalized for greppability.

---

## Categories

| Category | Meaning |
|---|---|
| `ukiyo-e` | Classical Japanese woodblock print (Hokusai, Hiroshige, etc.) |
| `mokuhanga` | Contemporary/Western mokuhanga collaborations (Frankenthaler-Shibata, Chuck Close-Pace, etc.) |
| `reference` | Catalog scan / auction-listing image, used for sanity-check or comparison |
| `user-work` | Reid Surmeier's own prints/designs |

---

## Adding a new fixture

1. Choose a stable slug — `<artist>_<title>_<year>` or `<museum>_<accession>`. Keep it lowercase-snake-case. Never reuse a slug.
2. `mkdir corpus/<image_id>`
3. Copy the image as `corpus/<image_id>/original.<ext>`. The `.gitattributes` rule auto-routes `.jpg|.jpeg|.png|.tif|.tiff` through Git LFS.
4. Write `annotations.json` matching the schema above. Use `null` for unknowns — never omit a field.
5. `git add corpus/<image_id>/` + commit.
6. Verify LFS pickup: `git lfs ls-files | grep <image_id>` should show the image.

---

## Git LFS requirement

This directory **requires Git LFS** to clone the images correctly. Without it, `original.*` files will appear as ~130-byte text pointers.

```bash
# one-time host setup
sudo apt-get install -y git-lfs
git lfs install

# inside a fresh clone of the repo
git lfs pull
```

---

## Roadmap

- **V0 (this commit):** 14 user test images + 3 Met OA CC0 ukiyo-e seeded as Tier A.
- **V2:** `tests/eval/run_corpus.py` walks this tree, runs `forward_render_km()` against each fixture, emits `EvalResult`.
- **V4+:** `corpus/<id>/goldens/` populated by `make update-goldens` after the algorithm stabilizes.
- **Phase 3+:** Publish Tier-A subset as `huggingface.co/datasets/reidsurmeier/mokuhanga-plate-separation`. No public mokuhanga plate-separation dataset exists yet — this would be a first.

---

## References

- `~/Prompts/validation-system-v1.md` — full spec, sections 3, 12, 13
- `~/Prompts/woodblock-build-plan-v2.md` — master architecture
- Met Open Access API: <https://metmuseum.github.io/>
- Tyler Graphics catalogue raisonné 1974-1985 (book 06 in `_research-dossier`) — ground-truth source for Frankenthaler-Shibata trio
- Sultan, *Chuck Close Prints*, pp. 125-169 — ground-truth source for `close_emma_2002`
