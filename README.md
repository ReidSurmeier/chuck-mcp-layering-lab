# woodblock_stack — v23-MCP

Mokuhanga **plausible plan synthesis** from one image, delivered as a local MCP server. Opus 4.7 drives a JAX inverse-stack solver, an optional pigment calibration path, and CNC-ready SVG output over filesystem handoffs. Single mode: one-image-in, plausible Emma-style print plan + plain-language recipe out.

This repo is the **research/production assistant** form of the workflow. The public web demo (v20, image → color separation → CNC SVGs, no underprint solver) lives at <https://color.reidsurmeier.wtf>. v23 is not a web app — it is an MCP server you wire into Claude Code.

## Posture (locked)

> **These are plausible underprint candidates that reduce reconstruction error under this pigment/printing model. Never recovered true underlayers.**

Underprints are **designed by printmaking rules** (skin-tone support, cool-under-shadow, warm-under-warm, support-larger-than-visible, detail-covers-support), not detected from the input image. The system never claims to have recovered the artist's true block sequence. `WB-LANG-01` lints the codebase for `plate`, `separator`, `layer`, `detect underlayer`, `recover underprint`, `true hidden block`. `WB-LANG-02` lints any string that presents Mixbox output as the print without the `as if pre-mixed` qualifier — mokuhanga is overprint glazing, not palette mixing, and v23 ships Tier-1 Mixbox with directional honesty (see ADR-0002).

## Quickstart

Tooling assumes Linux dev box (Opus 4.7 host) + Windows GPU box over Tailscale, WSL2 Ubuntu serves the MCP subprocess.

```bash
# 1. Install package + solver extras
uv tool install --extra solver --extra mcp --extra io woodblock-mcp

# 2. Register with Claude Code (stdio over SSH, see ADR-0003)
claude mcp add woodblock_stack --scope user -- \
  ssh reidsurmeier2@100.67.23.102 \
  "wsl -d Ubuntu -- /home/reidsurmeier2/.venv-v23/bin/woodblock-mcp"

# 3. Verify
claude mcp list | grep woodblock_stack
```

Smoke test from Claude Code:

```
/mcp call woodblock_stack ingest_reference_image '{"path": "corpus/portraits/emma_01.png"}'
/mcp call woodblock_stack analyze_image '{"path": "corpus/portraits/emma_01.png"}'
/mcp call woodblock_stack propose_stack '{"image_path": "...", "solve_profile": "fast"}'
```

## Architecture

```
Chat (Opus 4.7, Linux dev box)
  | MCP JSON-RPC over stdio (FastMCP 2.x)
  v
woodblock_stack subprocess (WSL2 on Windows GPU box)
  | in-process Python imports
  v
backend/services/v23/
  S1 ingest → S2 SAM (HTTP gateway to v20) → S3 palette (13-Mixbox)
  → S4 adjacency + strategy template pick
  → S5 JAX L-BFGS-B 4-level pyramid (8 base + 3 underprint-rule loss terms)
  → S6 DSATUR block packing (post-solve)
  → S7 three-state mask classifier
  → S8 carveability morphology + post-solve topology repair (ADR-0005)
  → S9 SVG vectorize + kento marks
  → S10 manifest v23.0 + ZIP (always bundles recipe.md)
  |
  v
~/.woodblock/v23/ filesystem artifacts (manifests, masks, plans, sessions)
  |
  v (opt-in)
Next.js /v23/review/{plan_id} viewer — read-only, <= 800 LOC
```

Day-1 ships **11 MCP tools** (ADR-0004); the remaining 22 land as v23.1 once Tier-1 5/5 is green on the corpus. See `CONTEXT.md` for locked glossary (Block / Impression / Mask / Pigment / Order / Underprint / Stack / Plan / Pull group / Strategy template / Solve profile / Render tier / Overprint / Mixing / Glazing).

## Render tiers

| Tier | Engine | Status | When |
|---|---|---|---|
| T1 | Mixbox 7-D latent lerp | ships v23 | default, generic 13-pigment palette, stacks ≤ 3 |
| T2 | Empirical 2-layer LUT | v23.1 | once artist uploads swatch-sheet calibration |
| T3 | K-M two-flux recursion (8λ K,S fit) | v24 | spectral pigment fit available + stack > 3 |

T1 is directionally accurate, ΔE 4–8 absolute shift vs reality on deep stacks. Every T1 recipe ships the honest qualifier. See ADR-0002 for why overlay (overprint) is not mixing.

## Documentation

| Path | Contents |
|---|---|
| `CONTEXT.md` | locked glossary, banned terms, example dialogue |
| `docs/adr/` | architecture decision records (5 ADRs) |
| `docs/architecture/README.md` | index pointing to master plan + addendums on SMB share |
| `CONTRIBUTING.md` | TDD discipline, 400 LOC cap, branch + commit format |
| `CHANGELOG.md` | Keep a Changelog format |

Master plan + binding addendums live on the SMB share at `/mnt/c/Users/reidsurmeier2/Books/printmaking/v23/` — see `docs/architecture/README.md` for the file index and content hashes.

## Repo layout

```
backend/
├── mcp/             # FastMCP server, error envelope, paths, ULID
├── services/v23/    # types, session, stages/ (S1..S10), forward_render
└── tests/v23/       # scaffold/, unit/, stages/, transport/, conversation/
corpus/              # 17 fixtures (14 user + 3 Met OA CC0 ukiyo-e) via Git LFS
docs/
├── adr/             # 0001..0005
└── architecture/    # plan index
.github/
├── workflows/v23.yml        # CI: lint + WB-LANG-01/02 + pytest
├── ISSUE_TEMPLATE/          # blocker.yml, build-step.yml
├── PULL_REQUEST_TEMPLATE.md # TDD evidence + CONTEXT.md compliance
└── CODEOWNERS
```

## Contributing

Read `CONTRIBUTING.md` before opening a PR. TDD only (red → green → refactor), 400 LOC cap per file, branches named `pipeline/v23-D<N>-<slug>`, commits formatted `v23-MCP D<N>.<n> <slug> — <count>/<count> green`. `WB-LANG-01` + `WB-LANG-02` must pass.

## License

MIT — see `LICENSE`. Copyright 2026 Reid Surmeier.
