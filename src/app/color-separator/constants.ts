import type { SeparationParams } from "@/lib/types";

export type VersionId = SeparationParams["version"];

export const VERSIONS: { id: VersionId; label: string }[] = [
  { id: "v20", label: "v20 (production)" },
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
  svg?: string;
}

export interface AppError {
  message: string;
  retryable: boolean;
}
