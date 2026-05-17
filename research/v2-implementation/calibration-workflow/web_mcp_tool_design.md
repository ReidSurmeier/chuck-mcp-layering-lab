# MCP Tool Design — Separate Bootstrap and Drift-Check Tools

## Decision: TWO MCP tools, not one

The brief says: "system should provide a SEPARATE MCP tool for calibration bootstrap (one-time-per-pigment setup)."

This is the right call. Bootstrap and routine drift check have:
- Different photo requirements (bootstrap = 2 photos per pigment; drift = 1 photo for all)
- Different durations (bootstrap = 4 hours per pigment; drift = 30 min total)
- Different blast radius (bootstrap = writes new YAML; drift = read-only check)
- Different error tolerances (bootstrap can fail loudly; drift must not block accidentally)

Splitting them keeps each tool single-purpose and easy to reason about.

## Tool 1: `chuck_mcp.calibrate_pigment_bootstrap`

### Inputs

```python
@mcp.tool()
async def calibrate_pigment_bootstrap(
    pigment_id: str,                    # e.g. "gunjo_handmade_2026_05"
    pigment_name: str,                  # e.g. "Handmade gunjo (azurite)"
    source: str,                        # e.g. "Reid, ground from azurite stones"
    raw_swatch_over_white: str,         # absolute path to RAW file
    raw_swatch_over_black: str,         # absolute path to RAW file
    raw_colorchecker: str,              # absolute path to RAW file (same session)
    raw_flat_field: str,                # absolute path to RAW file
    pigment_row_on_plate: int,          # which row 0-14 the pigment occupies
    concentration_ladder: list[float] | None = None,  # default [0.03, 0.06, 0.12, 0.25, 0.50, 0.75, 1.00]
    recipes: dict[str, str] | None = None,  # human-readable per-c_ratio recipes
    pigments_dir: str = "pigments",     # output directory
    overwrite_existing: bool = False,   # safety
) -> dict:
    """
    One-time calibration of a single pigment. Photographs of swatch plate
    over white and black substrates are processed against a ColorChecker
    reference to extract Lab values, then K-M inverse runs the Curtis 1997
    two-substrate procedure to derive (K, S) per concentration.

    OUTPUTS:
        pigments/{pigment_id}.yaml — full pigment characterization

    Returns:
        {
            "status": "success" | "fail",
            "output_path": "/abs/path/to/pigments/{pigment_id}.yaml",
            "ccm_fit_quality": { "delta_e_mean": float, "delta_e_max": float },
            "ladder_summary": [ {"c": 0.03, "lab": [...], "K": [...], "S": [...]}, ... ],
            "warnings": [ ... ],
            "errors": [ ... ],
        }
    """
```

### Error cases

The tool MUST fail loudly when:
- ColorChecker detection fails → error code `CC_NOT_DETECTED`
- ArUco fiducial detection fails → error code `FIDUCIALS_NOT_DETECTED`
- CCM fit residual ΔE_max > 4 → error code `CCM_QUALITY_FAIL`
- Pigment ID already exists and `overwrite_existing=False` → error code `EXISTS`
- K-M inverse fails to converge on >2 stripes → error code `KM_INVERSE_FAIL`
- Saturated pixels detected in any swatch → error code `EXPOSURE_SATURATED`

Each error includes a `remediation_hint` field telling Reid what to do:
```json
{
  "errors": [{
    "code": "CCM_QUALITY_FAIL",
    "message": "CCM fit max ΔE = 5.8, exceeds 4.0 limit.",
    "remediation_hint": "Re-shoot with: (1) Verify cross-polarization (rotate CPL until glossy test surface goes black). (2) Increase exposure so white patch sits 85-95% (no clipping). (3) Confirm ColorChecker is the post-2014 version."
  }]
}
```

### Side effects

- Writes `pigments/{pigment_id}.yaml`.
- Caches raw files into `.calibration_sessions/{timestamp}/` for traceability.
- Logs to `.calibration_sessions/{timestamp}/session_log.md`.
- Updates `pigments/_inventory.yaml` index file.

## Tool 2: `chuck_mcp.calibrate_drift_check`

### Inputs

```python
@mcp.tool()
async def calibrate_drift_check(
    raw_sentinel_sheet: str,            # path to RAW photo of sentinel sheet (all pigments)
    raw_colorchecker: str,              # path to RAW of ColorChecker (same session)
    raw_flat_field: str,                # path to RAW of flat field
    pigment_ids: list[str] | None = None,  # which pigments to check; None = all in inventory
    pigments_dir: str = "pigments",
    sentinel_c_ratio: float = 0.50,     # which concentration to use as sentinel
    warn_threshold: float = 2.5,
    block_threshold: float = 3.5,
) -> dict:
    """
    Periodic drift check using a 'sentinel sheet' — one printed swatch at
    c_ratio=0.5 for each pigment, all on one washi sheet, photographed once.

    Returns:
        {
            "status": "pass" | "warn" | "block",
            "per_pigment": [
                {
                    "pigment_id": "...",
                    "delta_e": 0.5,
                    "status": "PASS",
                    "lab_baseline": [78.3, -8.1, -17.5],
                    "lab_observed": [78.4, -8.0, -17.4],
                    "remediation": null,
                },
                ...
            ],
            "summary": {
                "n_pass": 14,
                "n_warn": 0,
                "n_block": 1,
                "blocked_pigments": ["..."],
            },
            "next_action": "Re-bootstrap blocked pigments before printing editions."
        }
    """
```

### Side effects

- READ-ONLY for pigment YAMLs (does NOT modify them). Updates only an in-place `drift_history` list in each YAML.
- Caches the drift session into `.calibration_sessions/drift_{timestamp}/`.
- Outputs `drift_report_{timestamp}.md` to the session folder for Reid to review.

## Tool 3 (optional v1.x): `chuck_mcp.calibrate_session_ccm`

For long shooting sessions where Reid takes the ColorChecker shot at the start, this tool computes the CCM and returns a session ID. Subsequent capture tools reference the session ID rather than re-detecting the chart each time. This is an optimization for the printing workflow (where Reid might photograph 5 prints in a row).

```python
@mcp.tool()
async def calibrate_session_ccm(
    raw_colorchecker: str,
    raw_flat_field: str,
    session_label: str = "",
) -> dict:
    """
    Compute and cache a CCM for the current photo session. Returns a session_id
    that downstream tools can reference to avoid recomputing.

    Returns:
        {
            "status": "success",
            "session_id": "ccm_2026_05_16_14_22_00",
            "ccm_fit_quality": {...},
            "expires_at": "ISO timestamp (4 hours from now)",
        }
    """
```

Session expiry after 4 hours forces re-calibration if Reid takes a long break.

## Inventory tool: `chuck_mcp.list_pigments`

```python
@mcp.tool()
async def list_pigments(
    pigments_dir: str = "pigments",
    include_supply_level: bool = True,
    include_last_drift_check: bool = True,
) -> dict:
    """
    List all pigments in the library with their metadata.

    Returns:
        {
            "pigments": [
                {
                    "id": "gunjo_handmade_2026_05",
                    "name": "Handmade gunjo (azurite)",
                    "source": "...",
                    "calibration_date": "2026-05-16",
                    "supply_level": "medium",
                    "last_drift_check_date": "2026-06-15",
                    "last_drift_status": "PASS",
                    "days_since_drift_check": 30,
                },
                ...
            ],
            "recommendations": [
                "Pigment 'oyster_white' has not been drift-checked in 65 days.",
                "Pigment 'organic_red_42' is supply_level=low; order replacement.",
            ]
        }
```

## Supply level update tool: `chuck_mcp.set_pigment_supply`

```python
@mcp.tool()
async def set_pigment_supply(
    pigment_id: str,
    supply_level: str,    # "high" | "medium" | "low" | "out"
    note: str = "",
    pigments_dir: str = "pigments",
) -> dict:
    """Manually update Reid's pigment inventory."""
```

This is the only mutation Reid does by hand to the YAML in routine use.

## Putting it together: Reid's workflows

### Workflow A: First-time setup (one-time)

```
1. Reid carves the swatch plate (or downloads SVG, sends to CNC).
2. Reid sets up the copy-stand cross-polarized lighting rig.
3. Reid takes a flat-field photo (blank washi).
4. For each pigment in inventory:
   a. Mix 7 concentrations of ink.
   b. Print the row of the swatch plate corresponding to this pigment, once on white washi.
   c. Print same row once on black washi.
   d. Photograph ColorChecker, swatch_white, swatch_black.
   e. Call calibrate_pigment_bootstrap(...).
   f. Review the warnings + ΔE summary.
5. Result: pigments/ directory with N YAML files.
```

Time: ~4 hours per pigment × 15 pigments = ~60 hours over 1-2 weeks of work.

### Workflow B: Per-edition production (routine, every print run)

```
1. Reid sets up lights (same setup each time, hopefully).
2. Photographs ColorChecker + flat-field.
3. Calls calibrate_session_ccm to compute session CCM.
4. Prints sentinel sheet (single sheet, c=0.5 stripe of each pigment).
5. Calls calibrate_drift_check.
6. If any pigment is blocked → re-bootstrap that pigment first.
7. If all pass → proceed with edition print.
```

Time: ~30 min before each edition.

### Workflow C: New pigment added mid-life

```
1. Reid grinds a new pigment, gives it a name.
2. Adds a new row to the plate (carves separate small block).
3. Single-pigment bootstrap (workflow A step 4).
4. From here on, drift-check covers it like any other pigment.
```

Time: ~4 hours for new pigment.

### Workflow D: Pigment failed drift check

```
1. Drift check reports BLOCK on pigment X.
2. Reid investigates (visual inspection, maybe a fresh print).
3. If pigment is permanently changed (oxidation, contamination):
   - Mark old YAML as deprecated.
   - Bootstrap fresh (workflow A step 4) with a new pigment_id (e.g., "_v2" suffix).
4. If transient (e.g., humidity event):
   - Wait, re-test.
```

## Tool naming convention

All calibration tools live in the `chuck_mcp.calibration.*` MCP namespace:

| Full name | Purpose | Reid uses |
|---|---|---|
| `chuck_mcp.calibration.bootstrap_pigment` | One-shot setup per pigment | Workflow A, C |
| `chuck_mcp.calibration.drift_check` | Periodic check across all pigments | Workflow B |
| `chuck_mcp.calibration.session_ccm` | Cache session CCM (optimization) | Workflow B (optional) |
| `chuck_mcp.calibration.list_pigments` | Inventory query | Anytime |
| `chuck_mcp.calibration.set_supply` | Mark supply level | Anytime |
| `chuck_mcp.calibration.validate_yaml` | Schema check on a YAML file | CI / debugging |

## Idempotency and dry-run modes

`bootstrap_pigment` accepts `dry_run: bool = False`. When true, runs the full pipeline but doesn't write any YAML — just returns the computed values for Reid to review.

`drift_check` is naturally read-only-ish (only appends to drift_history).

`session_ccm` is idempotent given the same input (session ID becomes a deterministic hash of inputs + timestamp).

## Versioning of the calibration protocol

Every YAML stamps `calibration_protocol_version`. The MCP refuses to mix YAMLs across protocol versions in a single forward render (chuck-mcp's t1/t2/t3 pipeline). When a major version bump happens (e.g., we add multispectral support in v2), older YAMLs go through an explicit `migrate_pigment_yaml` tool.

## Why not put calibration inside the main render MCP?

Could fold the calibration tools into `chuck_mcp.render.*`. But:
- Calibration needs different dependencies (rawpy, colour-checker-detection) than rendering (jax, mixbox).
- Calibration is "model-time," render is "inference-time." Same separation as ML training vs deployment.
- Calibration is interactive (Reid in the studio); render is batch (production).
- Different failure modes shouldn't pollute each other.

**Verdict: keep calibration as a separate MCP server / module, communicating via the pigment YAML files on disk.** This is the same architectural pattern as ICC profile creation (a totally separate workflow from print production).
