export function rgbToHex(rgb: [number, number, number]): string {
  return "#" + rgb.map((c) => c.toString(16).padStart(2, "0")).join("");
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// Coarse color-name buckets — mirror backend/color_names.py so exported
// filenames are scannable without decoding hex. Perceptually-weighted
// Y'CbCr-ish distance keeps hue wrap and near-gray pitfalls in check.
const COLOR_PALETTE: Array<[string, [number, number, number]]> = [
  ["black", [0, 0, 0]],
  ["white", [255, 255, 255]],
  ["gray", [128, 128, 128]],
  ["red", [220, 40, 40]],
  ["orange", [240, 140, 40]],
  ["yellow", [240, 220, 40]],
  ["olive", [150, 150, 50]],
  ["green", [60, 180, 60]],
  ["teal", [40, 170, 170]],
  ["cyan", [60, 220, 240]],
  ["blue", [50, 90, 210]],
  ["navy", [20, 30, 110]],
  ["purple", [140, 70, 200]],
  ["magenta", [220, 70, 200]],
  ["pink", [240, 160, 190]],
  ["brown", [130, 80, 40]],
  ["tan", [210, 180, 140]],
  ["cream", [250, 240, 210]],
];

export function rgbToColorName(rgb: [number, number, number]): string {
  const [r, g, b] = rgb;
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const spread = Math.max(r, g, b) - Math.min(r, g, b);
  if (spread <= 12) {
    if (lum < 32) return "black";
    if (lum > 232) return "white";
    return "gray";
  }
  let best = COLOR_PALETTE[0][0];
  let bestDist = Infinity;
  for (const [name, [pr, pg, pb]] of COLOR_PALETTE) {
    const d =
      Math.pow(0.3 * (r - pr), 2) +
      Math.pow(0.59 * (g - pg), 2) +
      Math.pow(0.11 * (b - pb), 2);
    if (d < bestDist) {
      bestDist = d;
      best = name;
    }
  }
  return best;
}

// Filename-safe slug used in ZIP entries: "<name>-<hex>" (no leading #, lowercase).
export function colorSlug(rgb: [number, number, number]): string {
  return `${rgbToColorName(rgb)}-${rgbToHex(rgb).replace("#", "").toLowerCase()}`;
}

// ZIP entry stem: "<plateNumber>_<hex>" — leading number is the plate's
// print order so an unzipped folder sorts naturally (1_, 2_, 3_ …); hex
// is the plate color at a glance. Shared with backend
// (color_names.py#plate_filename_stem) so both ZIP paths emit identical names.
export function plateFilenameStem(
  rgb: [number, number, number],
  plateNumber: number
): string {
  const hex = rgbToHex(rgb).replace("#", "").toLowerCase();
  return `${Math.trunc(plateNumber)}_${hex}`;
}

// Sanitize a filename into something safe for a ZIP directory entry.
// Collapses path separators, control chars, and platform-reserved characters
// to hyphens; trims edge whitespace. Empty / all-stripped input falls back
// to the caller's default.
export function safeZipFolderName(name: string, fallback: string): string {
  const cleaned = name
    .replace(/[\\/\x00-\x1f<>:"|?*]+/g, "-")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || fallback;
}
