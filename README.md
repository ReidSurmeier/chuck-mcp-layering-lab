# woodblock_stack — v23-MCP

Mokuhanga **plausible plan synthesis** from one image, delivered as a local MCP server. Opus 4.7 drives a JAX inverse-stack solver, an optional pigment calibration path, and CNC-ready SVG output over filesystem handoffs. Single mode: one-image-in, plausible Emma-style print plan + plain-language recipe out.

This repo is the **research/production assistant** form of the workflow. The public web demo (v20, image → color separation → CNC SVGs, no underprint solver) lives at <https://color.reidsurmeier.wtf>. v23 is not a web app — it is an MCP server you wire into Claude Code.

## Posture (locked)

> **These are plausible underprint candidates that reduce reconstruction error under this pigment/printing model. Never recovered true underlayers.**

Underprints are **designed by printmaking rules** (skin-tone support, cool-under-shadow, warm-under-warm, support-larger-than-visible, detail-covers-support), not detected from the input image. The system never claims to have recovered the artist's true block sequence. `WB-LANG-01` lints the codebase for `plate`, `separator`, `layer`, `detect underlayer`, `recover underprint`, `true hidden block`. `WB-LANG-02` lints any string that presents Mixbox output as the print without the `as if pre-mixed` qualifier — mokuhanga is overprint glazing, not palette mixing, and v23 ships Tier-1 Mixbox with directional honesty (see ADR-0002).

## Quickstart

Tooling assumes a Linux/WSL2 host with an NVIDIA GPU. The current tested host
uses an RTX 4070 SUPER, driver CUDA 13.1, and JAX 0.10.0 with the CUDA 13
plugin.

```bash
# 1. Create the local venv
python -m venv .venv-v23
. .venv-v23/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[solver,mcp,io,dev]"

# 2. Verify JAX is on CUDA
python - <<'PY'
import jax
print(jax.devices())
print(jax.default_backend())
PY

# 3. Register with Claude Code (stdio over SSH, see ADR-0003)
claude mcp add woodblock_stack --scope user -- \
  ssh reidsurmeier2@100.67.23.102 \
  "wsl -d Ubuntu -- /home/reidsurmeier/src/woodblock-reidsurmeier-wtf/.venv-v23/bin/woodblock-mcp"

# 4. Verify the registration
claude mcp list | grep woodblock_stack
```

For direct local smoke tests:

```bash
WOODBLOCK_HOME=/tmp/woodblock-smoke \
WOODBLOCK_DISABLE_SAM=1 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python - <<'PY'
from backend.mcp.registry import call_mcp_tool
image = "corpus/reid_untitled_01/original.png"
print(call_mcp_tool("ingest_reference_image", {"path": image}))
print(call_mcp_tool("analyze_image", {"path": image}))
print(call_mcp_tool("propose_stack", {"path": image, "solve_profile": "fast"}))
PY
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
  → S5 JAX L-BFGS inverse solve over alpha masks
  → S6 three-state mask classifier
  → S7 DSATUR block packing
  → S8 topology diagnostics / repair hooks
  → S9 SVG vectorize + kento marks
  → S10 manifest v23.0 + ZIP (always bundles recipe.md)
  |
  v
~/.woodblock/v23/ filesystem artifacts (manifests, masks, plans, sessions)
  |
  v (opt-in)
Next.js API proxy routes for preview/result access
```

The registered MCP surface is generated from `backend/mcp/registry.py` and
includes core flow, HITL, calibration, introspection, session, carve, and overlay
tools. Call `tools/list` through MCP or `backend.mcp.registry.list_mcp_tools()`
to inspect the exact surface in a checkout.

Large source images are solved on a bounded internal grid and then upscaled back
to full source resolution before render/vector/export. This avoids exhausting
12 GB GPUs during JAX L-BFGS while still producing full-size masks and SVGs.
Override the internal budget with `WOODBLOCK_SOLVER_MAX_PIXELS`; the current
defaults are 256k fast, 512k default, and 768k thorough. `solver_telemetry`
reports `optimized_shape` and `downsample_scale` for each plan.

## Render tiers

| Tier | Engine | Status | When |
|---|---|---|---|
| T1 | Mixbox 7-D latent lerp | ships v23 | default, generic 13-pigment palette |
| T2 | Empirical 2-layer LUT bias | ships v23 local path | after `upload_swatch_overprint_matrix` |
| T3 | K-M two-flux recursion (8λ K,S fit) | v24 | spectral pigment fit available + stack > 3 |

T1 is directionally useful, not physically final: it renders pigments as if
pre-mixed, while mokuhanga is cumulative overprint glazing. T2 applies the
active swatch-matrix correction when present. T3 remains deferred.

## Current Limits

The current optimizer is still primarily an RGB reconstruction objective through
the forward renderer. It emits overlapping alpha impressions and cumulative pull
artifacts, but broad underlayer structure, brushed-zone grouping, non-machinable
geometry penalties, and island/disjointness penalties are not fully inside the
loss yet. Treat outputs as testable v23 plans, not final CNC-ready artistic
recommendations without review.

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
