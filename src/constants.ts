export type VersionId = "v20";

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
  manifestIndex: number;
  svg?: string;
}

export interface AppError {
  message: string;
  retryable: boolean;
}
