import type { SeparationParams } from "@/lib/types";

export type VersionId = SeparationParams["version"];

export const VERSIONS: { id: VersionId; label: string }[] = [
  { id: "v15", label: "v15 (SAM)" },
  { id: "v16", label: "v16 (SAM+)" },
  { id: "v17", label: "v17 (SAM+lines)" },
  { id: "v18", label: "v18 (SAM best)" },
  { id: "v20", label: "v20 (best)" },
  { id: "v19", label: "v19 (guided)" },
  { id: "v14", label: "v14 (hybrid)" },
  { id: "v13", label: "v13" },
  { id: "v12", label: "v12" },
  { id: "v11", label: "v11 (merge+cache)" },
  { id: "v10", label: "v10 (smooth)" },
  { id: "v9", label: "v9 (clean)" },
  { id: "v8", label: "v8 (bilateral+crf)" },
  { id: "v7", label: "v7 (crf)" },
  { id: "v6", label: "v6 (superpixel)" },
  { id: "v5", label: "v5 (clean)" },
  { id: "v4", label: "v4 (ai)" },
  { id: "v3", label: "v3 (paper)" },
  { id: "v2", label: "v2" },
];

export interface PaletteColor {
  rgb: [number, number, number];
  locked: boolean;
}

export interface PlateImage {
  name: string;
  url: string;
  color: [number, number, number];
  coverage: number;
}

export interface AppError {
  message: string;
  retryable: boolean;
}
