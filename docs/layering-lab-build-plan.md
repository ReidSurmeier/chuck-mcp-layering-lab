# Chuck MCP Layering Lab Build Plan

This repo is a separate experiment fork of Chuck MCP. It starts from
`woodblock-reidsurmeier-wtf` commit `6db6f11` and must not share code or tool
state with `emma-mokuhanga-mcp`.

## Goal

Move the solver away from pixel-level reconstruction plates and toward a
printable mokuhanga construction:

1. Broad underlayers establish light and low-frequency color support.
2. Middle plates form separated regional hue-shift or shadow masses.
3. Later plates carry small detail/key marks.
4. Pigment choices can include premixed recipes when no single catalog pigment
   matches a target color.
5. CNC/vector output is accepted only after the mask topology is plausible.

## Reference Observations

The annotated files in `/srv/woodblock-share/Examples` show the key pattern.

- The first block in the reference is light yellow because the image needs a
  high-luminance warm support field. More generally, the first role should be
  the lightest broad underlayer that moves the paper toward the low-frequency
  target without destroying later color range.
- Base yellow, blue, pink, brown, and green underlayers use large diffuse zones.
- Hair and shadow areas work when brown/blue regions overlap as larger support
  shapes rather than pixel details.
- Later hue shifts are often jigsawed on the block: adjacent color regions have
  clear separations, then the print stack and opacity create the final optical
  mixture.
- Red should be a separate plate role in this image because the red areas are
  high-chroma regional accents with clear boundaries. More generally, a hue
  gets its own plate when it cannot be cleanly explained by the broad base plus
  later dark/detail pulls, or when it needs a crisp jigsaw boundary.

## Systematic Role Logic

The role choice should be image-driven, not hard-coded to this print.

- Pick the first/base pull from low-frequency target structure:
  - high luminance before low luminance
  - broad coverage before isolated marks
  - low opacity/transparency before opaque/key detail
  - pigment hue chosen by the weighted mean of broad support pixels

- Pick separate chroma/accent pulls when a color has a real regional job:
  - high chroma relative to neighboring tones
  - clean spatial boundary or jigsaw-like adjacency
  - poor approximation from existing broad underlayers
  - enough area to be worth carving as a plate

- Pick jigsaw regions when overlap would be impractical:
  - adjacent colors require different premixes or opacity recipes
  - the boundary should stay crisp after printing
  - stacking translucent gradients would muddy the hue or make registration
    impossible to reason about

- Pick late key/detail pulls from residual structure:
  - dark, edge-like, high-frequency, or contour-dominant information
  - low area but high visual importance
  - printed last because it defines form and should not be buried

## Implemented In This Fork

- S4 now uses `layering_lab_warmstart`.
  - Broad underlayer seed from low-frequency color/tonal structure.
  - Underlayer pigment is inferred from weighted broad-support color, preferring
    high-luminance pigments for early pulls.
  - Separate high-chroma accent seed when the image has a regional hue shift.
  - Base-hue Tan seeds are blurred and de-emphasized so the first role cannot
    start as a skinny detail mask.

- S5 role geometry is stricter.
  - Underlayers use a 12x coarser control grid.
  - Mid layers use a 4x coarser control grid.
  - Underlayer seeds are blurred before logit parameterization.
  - Underlayers carry stronger high-frequency and speckle penalties.
  - Middle color plates carry a pairwise overlap penalty to prefer jigsawed
    regions over every color fading into every other color.

- The pigment catalog is expanded to 24 pigments.
  - Added ochres/siennas/umbers, alizarin, vermilion, naphthol red,
    prussian/phthalo/cerulean blues, phthalo green, and sap green.
  - Cadmium red already existed in the parent catalog; the lab change is making
    it an explicit role seed, not merely an optional nearest pigment.

- The MCP surface now includes `suggest_pigment_mix`.
  - Input: target hex color.
  - Output: ranked premix ratios using one to three catalog pigments.
  - The output is bench guidance, not an overprint physics claim.

## Next Experiments

1. Replace mid-layer grids with superpixel or brushed-zone regions.
   The mid layer should optimize region opacities, not free pixels.

2. Add a hard jigsaw assignment mode.
   Adjacent hue-shift regions should be mutually exclusive inside a plate group,
   except for deliberate overprint support layers.

3. Add base-role topology gates.
   Reject any run where the first broad underlayer has too many connected
   components, too much high-frequency energy, or too little broad coverage.

4. Add a separate chroma-role acceptance check.
   Red should appear as its own region plate when the input has meaningful red
   signal, but the same rule should apply to blue/green/pink/orange accents in
   other images.

5. Integrate mix recipes into plate metadata.
   When a generated pigment is a premix, downstream recipe/export should report
   ratios and keep the plate identity as a mixed color role.

6. Run the full shared-example validation loop.
   Use `/srv/woodblock-share/Examples` and the current Emma input. Compare
   cumulative pulls against the reference construction, not only final Delta E.

## Acceptance Criteria

- First/base pull looks like a broad underlayer, not final-image detail.
- Pull order remains light to dark, with key/detail last.
- High-chroma regional colors appear as distinct plates when warranted by the
  input, including red in this reference.
- Mid color plates show separated regional boundaries where the reference uses
  jigsaw blocks.
- SVG export is blocked or flagged when topology is not machinable.
- Cumulative pull contact sheets remain the primary review artifact.
