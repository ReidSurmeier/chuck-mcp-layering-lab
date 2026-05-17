# An Article on Optics of Paint Layers (English translation by S. Westin)

original: Paul Kubelka, Franz Munk. "Ein Beitrag zur Optik der Farbanstriche." Zeitschrift fur Technische Physik 12, 593-601 (August 1931).
translation: Stephen H. Westin, Cornell University Program of Computer Graphics
url: http://www.graphics.cornell.edu/~westin/pubs/kubelka.pdf
DOI: none (1931 journal paper; no DOI exists)
relevance: The foundational 1931 paper. Defines the two-flux differential equations, absorption constant s, scattering constant r (modern usage: K and S), the infinite-thickness albedo H_inf = a - sqrt(a^2 - 1) where a = 1 + s/r, and the finite-thickness reflectance formula. This is the bedrock of every K-M variant chuck-mcp will implement.

## Technical summary (3 paragraphs)

The 1931 Kubelka-Munk paper sets up a planar parallel paint coating of thickness X illuminated by diffuse light. At depth x they define i (downward intensity) and j (upward intensity). An infinitesimal layer dx absorbs and scatters a fraction s*dx + r*dx of the passing light, where s is the absorption constant and r the scattering constant. They write the coupled ODEs `-di = -(r+s)i dx + r j dx` and `dj = -(r+s)j dx + r i dx`. By dividing through, defining h = j/i, and integrating across the full thickness using the boundary conditions h(0) = H' (substrate albedo) and h(X) = H (coating top albedo), they obtain the master equation (3). The key substitution is `a = (r+s)/r = 1 + s/r`, which compresses the problem into a single dimensionless parameter.

For an infinitely thick coating, X -> infinity in (3) yields `H_inf = a - sqrt(a^2 - 1) = 1 + s/r - sqrt((s/r)^2 + 2 s/r)`. This is the canonical infinite-thickness K-M formula. Note that H_inf depends only on the ratio s/r, not on the absolute magnitudes — a crucial result because it means thinning a paint with a clear binder leaves the infinite-thickness color unchanged (the binder dilutes s and r proportionally and thickens X compensatingly, so s/r and rX both stay invariant). For finite thickness X the closed-form is equation (5), reducible on a black substrate to `H = (e^(rX(1/H_inf - H_inf)) - 1) / ((1/H_inf) e^(rX(1/H_inf - H_inf)) - H_inf)`. Special cases s = 0 (ideal white, equations 7-9) and r = 0 (pure glaze, exponential decay equation 11) drop out cleanly.

For chuck-mcp this matters in two ways. First: r=0 is the "glaze" limit and is the closest analytic match to thin mokuhanga pigment films on washi (the binder is starch paste, the pigment particles are dispersed but the substrate scatters most of the light back through). The exponential transmission `H = H' * e^(-2sX)` then composes simply across N stacked layers as `H_final = H_substrate * exp(-2 * sum_n s_n * X_n)`. Second: the s/r-invariance under thinning means the "concentration" parameter in our t2 LUT can be defined as `c = X * rho` (rho = pigment mass per unit binder volume) rather than as a fraction, and the model will be physically self-consistent.

## Key equations extracted (verbatim)

```
Differential equations (1):
    -di = -(r+s) i dx + r j dx
     dj = -(r+s) j dx + r i dx

Substitution (2):
    a = (r+s)/r = 1 + s/r

Master equation (3):
    ln[ (H - a - sqrt(a^2-1))(H' - a + sqrt(a^2-1)) / 
        (H' - a - sqrt(a^2-1))(H  - a + sqrt(a^2-1)) ] = 2rX sqrt(a^2-1)

Infinite-thickness albedo (4):
    H_inf = a - sqrt(a^2 - 1) = 1 + s/r - sqrt((s/r)^2 + 2 s/r)

Finite-thickness albedo, general substrate (5):
    H = [H_inf*(H' - H_inf) - H_inf*(H' - 1/H_inf)*e^(rX(1/H_inf - H_inf))] / 
        [(H' - H_inf) - (H' - 1/H_inf)*e^(rX(1/H_inf - H_inf))]

Finite-thickness on black substrate (H' = 0), eq. 6:
    H = (e^(rX(1/H_inf - H_inf)) - 1) / 
        ((1/H_inf) e^(rX(1/H_inf - H_inf)) - H_inf)

Glaze limit (r = 0), eq. 11:
    H = H' * e^(-2 s X)

Ideal white limit (s = 0), eq. 9:
    H = r X / (r X + 1)   on black substrate
```

## Notation map (1931 -> modern)

- 1931 `s` (absorption constant) = modern `K`
- 1931 `r` (scattering constant) = modern `S`
- 1931 `H_inf` (intrinsic albedo) = modern `R_inf` (infinite-thickness reflectance)
- 1931 `H` = modern `R` (coating top reflectance)
- 1931 `H'` = modern `R_g` (substrate / ground reflectance)
- 1931 `s/r` = modern `K/S`

So the canonical modern restatement is `R_inf = 1 + K/S - sqrt((K/S)^2 + 2*K/S)` and the inverse `K/S = (1 - R_inf)^2 / (2 R_inf)`.

## Limitations called out by Kubelka & Munk themselves

- Only two spatial directions (up/down) — error grows as the illumination becomes less diffuse and the medium less matte. They cite Gurevic 1930 as making the same simplification.
- Glossy coatings are not considered. (Saunderson 1942 addresses this.)
- Achromatic (gray/white) treatment is primary; colored case is "only touched on."
- No internal/external surface reflection (i.e., no n != 1 correction). That's Saunderson's 1942 contribution.
