import { test as base, expect, type Page, type Locator } from "@playwright/test";
import * as path from "path";
import * as fs from "fs";

// Test image paths — these exist in the project root
const TEST_IMAGES = {
  small: "test_red.png", // smallest, fastest processing
  standard: "test_colorsep.png", // standard test image
  large: "test_large.png", // large image for stress tests
};

export interface TimingResult {
  compositeVisibleAt: number;
  lastPlateVisibleAt: number;
  plateCount: number;
}

/**
 * Upload an image file via the hidden file input and click process.
 */
export async function uploadAndProcess(
  page: Page,
  imageName: keyof typeof TEST_IMAGES = "small",
): Promise<void> {
  const filePath = path.resolve(__dirname, "..", TEST_IMAGES[imageName]);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Test image not found: ${filePath}`);
  }

  // Set file on the hidden input
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);

  // Click process button
  const processBtn = page.locator("button.process-btn");
  await processBtn.click();
}

/**
 * Wait for composite image to appear in the main canvas area.
 */
export async function waitForComposite(page: Page): Promise<Locator> {
  const composite = page.locator(".canvas-wrapper img");
  await composite.waitFor({ state: "visible", timeout: 90_000 });
  return composite;
}

/**
 * Wait for plate cards to appear and return them.
 */
export async function waitForPlates(
  page: Page,
  minCount: number = 1,
): Promise<Locator> {
  // Wait for at least one real plate card (not skeleton)
  const plateCard = page.locator(".plate-card:not(.plate-skeleton)");
  await expect(plateCard.first()).toBeVisible({ timeout: 90_000 });

  // Wait until we have at least minCount plates
  await page.waitForFunction(
    ({ selector, min }) => document.querySelectorAll(selector).length >= min,
    { selector: ".plate-card:not(.plate-skeleton)", min: minCount },
    { timeout: 90_000 },
  );
  return plateCard;
}

/**
 * Measure timing between composite appearing and all plates loaded.
 */
export async function measureLoadTiming(page: Page): Promise<TimingResult> {
  const t0 = Date.now();

  // Wait for composite
  await waitForComposite(page);
  const compositeVisibleAt = Date.now() - t0;

  // Wait for plates section title to show final count (no "of" loading indicator)
  // e.g. "plates (4)" not "plates (2 of 4)"
  const platesTitle = page.locator(".plates-section-title");
  await expect(platesTitle).toBeVisible({ timeout: 90_000 });

  // Wait for loading to stop — no more skeletons
  await expect(page.locator(".plate-skeleton")).toHaveCount(0, {
    timeout: 90_000,
  });

  const lastPlateVisibleAt = Date.now() - t0;
  const plateCount = await page
    .locator(".plate-card:not(.plate-skeleton)")
    .count();

  return { compositeVisibleAt, lastPlateVisibleAt, plateCount };
}

/**
 * Collect console errors during test.
 */
export function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      const text = msg.text();
      // Filter out known benign errors
      if (
        text.includes("favicon") ||
        text.includes("_next/static") ||
        text.includes("net::ERR_")
      ) {
        return;
      }
      errors.push(text);
    }
  });
  return errors;
}

export { TEST_IMAGES };
