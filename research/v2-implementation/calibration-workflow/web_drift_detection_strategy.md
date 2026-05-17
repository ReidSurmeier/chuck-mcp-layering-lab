# Drift Detection Strategy

Sources:
- [Kim Crick — Fugitive Pigments List](https://www.kimcrick.com/pages/fugitive-pigments-list-lightfast-test-problems-art-supplies)
- [Wiley 2025 — Microfading testing for light fastness](https://onlinelibrary.wiley.com/doi/10.1111/cote.12788)
- [Just Paint — How to Test for Lightfastness](https://justpaint.org/how-to-test-for-lightfastness/)
- [Jackson's Art — Fugitive Pigments](https://www.jacksonsart.com/blog/2024/04/22/fugitive-pigments-why-do-they-fade-and-does-it-matter/)
- [Wikipedia — Lightfastness (ASTM scale + BWS)](https://en.wikipedia.org/wiki/Lightfastness)

## Two distinct drift problems

The calibration MCP has to detect **two different kinds of drift**, with different timescales and remedies:

### (1) Camera/setup drift (HOURS to DAYS)

LED phosphor warm-up, ambient temperature shifts, lens fingerprints, ColorChecker fading.

- **Timescale:** Real changes within a single shooting session up to weeks.
- **Detection:** Re-shoot the ColorChecker; compare new CCM fit residual to last fit's residual.
- **Remedy:** Recompute CCM. Cheap (seconds).
- **Threshold:** If new CCM fit ΔE_max > 4, OR new CCM differs from last-session CCM by Frobenius norm > 0.1, **always recompute** and use the fresh one.

### (2) Pigment drift (WEEKS to MONTHS)

Reid's handmade pigments age. Some fade in light. Particle morphology shifts as the pigment ages. Nori (rice paste) eventually goes bad. Storage humidity changes pigment hydration.

- **Timescale:** Weeks to months.
- **Detection:** Re-shoot the **swatch plate** for a target pigment; compare new Lab values per-concentration to baseline.
- **Remedy:** Re-bootstrap that pigment. Tedious (4 hours for one pigment).
- **Threshold:** See policy below.

## Pigment drift policy

| ΔE_2000 (max across ladder) vs baseline | Action |
|---|---|
| < 1.5 | PASS — pigment is stable, use cached YAML |
| 1.5 - 2.5 | WATCH — log warning, proceed for non-edition prints only |
| 2.5 - 3.5 | WARN — block edition prints, force user acknowledgment for one-offs |
| ≥ 3.5 | BLOCK — refuse to use the YAML for any printing. Force re-bootstrap. |

These thresholds are tuned for Reid's "sub-ΔE-2 across edition of 10" requirement. The drift threshold is intentionally tighter than the absolute ΔE budget because:
- Edition consistency requires *intra-edition* repeatability of <ΔE 2 between prints.
- The same pigment must give the same color from print 1 to print 10.
- A ΔE 2 drift in the pigment YAML directly contributes to inter-print variation.
- Therefore: warn at half-budget, block at full-budget.

## When to TRIGGER a drift check (not just when to react to one)

The MCP enforces these triggers proactively:

| Trigger | Type | Notes |
|---|---|---|
| First use of a pigment after >30 days | Auto check | "When did you last calibrate?" gate |
| Reid edits the pigment YAML supply_level field | Auto check | He's handling the pigment anyway, easy to add a photo |
| Starting an edition print run (>3 prints planned) | Auto check | Edition consistency = highest stakes |
| Pigment ground from a new batch | **MUST RE-BOOTSTRAP** | New batch = new material, K-M parameters change |
| Storage conditions changed (moved studios, humidity event) | Manual check | Reid's call |
| Visible pigment color change to the eye | Manual check | Eye is sensitive to ΔE ~2, so this is a high-confidence drift signal |
| Calibration protocol version bumped | Auto check | All YAMLs must be re-validated against new protocol |

## The drift-sentinel approach

Instead of re-bootstrapping all 7 concentrations of all 15 pigments every time (impractical), the system uses a **drift sentinel**: a single representative swatch per pigment that gets photographed regularly.

Specifically, the c_ratio = 0.50 (middle of the ladder) swatch is the sentinel because:
- It's near the inflection point of the K-M opacity curve, sensitive to both K and S shifts.
- It's mid-luminance, less likely to clip.
- It's the most common concentration used in actual printing.

The drift-check tool reads ONE swatch per pigment in a quick photo session:

```
Reid prints a single "sentinel sheet" with c=0.5 stripes of every pigment in inventory.
Photographs it ONCE (with ColorChecker).
MCP compares each stripe's Lab to the cached baseline for that pigment.
Outputs a per-pigment PASS/WATCH/WARN/BLOCK report.
```

This takes Reid ~30 minutes (one print session) once a month, vs ~60 hours to fully re-bootstrap all pigments.

## ColorChecker drift (the reference itself fades)

The chart isn't immortal. X-Rite specs the ColorChecker for "5+ years stable color under indoor storage." In practice:
- Storage in original cardstock binder + low light: stable for 7-10 years.
- Frequent shoot use with bright LEDs: noticeable fade in 3-5 years.
- The blue patches (B1, A3, A4) fade fastest because of the pigments used.

**Mitigation:** Reid keeps the ColorChecker in its closing flap when not in active use. When the chart is 5+ years old, buy a fresh one and cross-calibrate (photograph both side-by-side, save the ΔE offset). Treat the old chart as a historical anchor.

## Schema additions to pigment YAML

```yaml
# pigments/gunjo_handmade_2026_05.yaml
pigment_id: gunjo_handmade_2026_05
# ... (existing fields)
drift_history:
  - check_date: 2026-05-16
    sentinel_c_ratio: 0.50
    sentinel_lab_baseline: [78.3, -8.1, -17.5]
    sentinel_lab_observed: [78.3, -8.1, -17.5]
    delta_e: 0.0
    status: BASELINE
    note: "Initial bootstrap"
  - check_date: 2026-06-15
    sentinel_c_ratio: 0.50
    sentinel_lab_baseline: [78.3, -8.1, -17.5]
    sentinel_lab_observed: [78.5, -7.9, -17.2]
    delta_e: 0.45
    status: PASS
    note: "Monthly check, well within tolerance"
  - check_date: 2026-09-15
    sentinel_c_ratio: 0.50
    sentinel_lab_baseline: [78.3, -8.1, -17.5]
    sentinel_lab_observed: [76.1, -6.4, -15.8]
    delta_e: 3.2
    status: WARN
    note: "Block humidity changed; re-bootstrap scheduled"
```

This gives Reid (and the system) a longitudinal record of pigment behavior. Useful for:
- Identifying which of his handmade pigments are stable (most azurites) vs fugitive (some organic dyes ground from plants).
- Calibrating expectations for batch-to-batch consistency.
- Knowing when to throw out a pigment that's drifted past usable.

## Calibration history caching

The MCP keeps a per-session record in `.calibration_sessions/`:

```
.calibration_sessions/
  2026-05-16T14-22-00/
    raw_colorchecker.cr3
    raw_swatch_white.cr3
    raw_swatch_black.cr3
    raw_flat_field.cr3
    ccm.yaml
    detections.json
    session_log.md
```

Sessions older than 6 months get pruned to just `ccm.yaml + session_log.md` to save disk.

## Validation: the "ten prints" verification

Once a quarter, Reid does a verification print: 10 copies of a single edition target. The MCP measures each one and reports:
- Inter-print ΔE_2000 max
- Inter-print ΔE_2000 mean
- Worst single pigment / region

If inter-print ΔE_max > 2 on this verification print, **something is broken** in the calibration. Likely candidates:
1. The CCM is fitting noisily — recompute with a better-framed ColorChecker shot.
2. A pigment YAML has drifted (drift-check should have caught it but didn't).
3. The printing process itself is varying more than expected (mechanical, not optical).

The verification print closes the loop on the whole system.

## Light exposure tracking

For the rigorous version: log cumulative lux-hours of every photo session on each pigment. Reid's LED panels at ~5000K with ~10,000 lux for 60 seconds per pigment per session = 167 lux-hours. At that rate, ASTM 1 lightfast pigments (Reid's mineral pigments, probably) are good for 100+ years. ASTM 3+ pigments (some organics, dyes) start fading at 10-50 lux-hours.

The MCP can warn: "this pigment has accumulated 47 lux-hours of light exposure during calibration sessions; ASTM 4 fade expected within 100 hours."

V1 skips this; V2 adds a `lightfastness_class: ASTM_I/II/III` field to the YAML and tracks cumulative exposure.

## Quick summary

- **2 drift types:** camera/setup (fast) and pigment (slow).
- **Always recompute CCM** at the start of every photo session (cheap).
- **Sentinel-based pigment drift check** monthly + before every edition print run.
- **ΔE thresholds:** PASS <1.5, WATCH 1.5-2.5, WARN 2.5-3.5, BLOCK ≥3.5.
- **Full re-bootstrap** when blocked, when batch changes, when protocol version bumps.
- **Longitudinal history in YAML** so Reid (and the system) learn which pigments are stable.
