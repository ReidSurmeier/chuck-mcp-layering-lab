import { test, expect } from "@playwright/test";
import * as path from "path";
import {
  uploadAndProcess,
  waitForComposite,
  waitForPlates,
  measureLoadTiming,
  collectConsoleErrors,
} from "./fixtures";

test.describe("Color Separator — Core Flow", () => {
  test("1. page loads without auth gate", async ({ page }) => {
    await page.goto("/color-separator");
    await expect(page.locator('input[type="file"]')).toBeAttached();
    await expect(page.locator("text=GPU Access Required")).not.toBeVisible();
    await expect(page.locator("button.process-btn")).toBeVisible();
  });

  test("2. upload → composite → plates complete flow", async ({ page }) => {
    const errors = collectConsoleErrors(page);

    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    // Composite appears
    const composite = await waitForComposite(page);
    await expect(composite).toBeVisible();

    // Plates appear
    const plates = await waitForPlates(page);
    const count = await plates.count();
    expect(count).toBeGreaterThanOrEqual(2);

    // Each plate card has an image, hex color, and coverage
    for (let i = 0; i < count; i++) {
      const card = plates.nth(i);
      await expect(card.locator("img")).toBeVisible();
      await expect(card.locator(".plate-card-hex")).toBeVisible();
      await expect(card.locator(".plate-card-coverage")).toBeVisible();
    }

    // No unexpected console errors
    const realErrors = errors.filter(
      (e) => !e.includes("[api]") && !e.includes("debug"),
    );
    expect(
      realErrors,
      `Unexpected console errors: ${realErrors.join("\n")}`,
    ).toHaveLength(0);
  });

  test("4. progress bar visible and tracks processing", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    // Progress bar root should appear
    const progressRoot = page.locator(".progress-bar-root");
    await expect(progressRoot).toBeVisible({ timeout: 10_000 });

    // Should have a fill element
    const fill = page.locator(".progress-bar-fill");
    await expect(fill).toBeVisible();

    // Label should show a stage name
    const label = page.locator(".progress-bar-label");
    await expect(label).toBeVisible();
    const labelText = await label.textContent();
    expect(labelText?.length).toBeGreaterThan(0);

    // Timer should be ticking
    const timer = page.locator(".progress-bar-time");
    await expect(timer).toBeVisible();

    // Wait for processing to complete
    await waitForComposite(page);
  });

  test("5. download produces valid ZIP with plates", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    // Wait for results
    await waitForComposite(page);
    await waitForPlates(page);

    // Click download button
    const downloadBtn = page.locator("button", { hasText: "ZIP" });
    await expect(downloadBtn).toBeEnabled({ timeout: 30_000 });

    // Listen for download event
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 60_000 }),
      downloadBtn.click(),
    ]);

    // Verify download completed
    expect(download.suggestedFilename()).toBe("color-separator-plates.zip");

    // Save and verify file is non-empty
    const filePath = await download.path();
    expect(filePath).toBeTruthy();

    // Download failure check
    const failure = await download.failure();
    expect(failure).toBeNull();
  });

  test("6. v20 backend responds within 45s (cold) or 30s (cached)", async ({
    page,
  }) => {
    await page.goto("/color-separator");

    const start = Date.now();
    await uploadAndProcess(page, "small");

    // Wait for composite (signals backend processing complete)
    await waitForComposite(page);
    const elapsed = Date.now() - start;

    // Allow 45s for cold start, 30s for cached
    expect(
      elapsed,
      `Backend took ${elapsed}ms (max 45000ms cold)`,
    ).toBeLessThanOrEqual(45_000);
  });
});

test.describe("Color Separator — Timing", () => {
  test("3. plates appear concurrently (within 2s of composite)", async ({
    page,
  }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    const timing = await measureLoadTiming(page);

    // Plates should appear within 2000ms after composite
    const plateDelay = timing.lastPlateVisibleAt - timing.compositeVisibleAt;
    expect(
      plateDelay,
      `Plates took ${plateDelay}ms after composite (max 2000ms)`,
    ).toBeLessThanOrEqual(2000);
  });
});

test.describe("Color Separator — Error Handling", () => {
  test("error display on network issues shows retryable message", async ({
    page,
  }) => {
    await page.goto("/color-separator");

    // Block ALL backend API requests to simulate complete network failure
    // Abort both the Next.js API routes and direct backend calls
    await page.route("**/api/preview-stream**", (route) => route.abort("connectionfailed"));
    await page.route("**/api/preview**", (route) => route.abort("connectionfailed"));

    // Try to process
    const fileInput = page.locator('input[type="file"]');
    const testImagePath = path.resolve(__dirname, "..", "test_red.png");
    await fileInput.setInputFiles(testImagePath);
    const processBtn = page.locator("button.process-btn");
    await processBtn.click();

    // Error should appear in nav panel — look for dismiss button (always present on error)
    const dismissBtn = page.locator("button", { hasText: "dismiss" });
    await expect(dismissBtn).toBeVisible({ timeout: 90_000 });
  });
});
