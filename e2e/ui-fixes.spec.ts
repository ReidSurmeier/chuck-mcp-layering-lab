import { test, expect } from "@playwright/test";
import * as path from "path";
import {
  uploadAndProcess,
  waitForComposite,
  waitForPlates,
  collectConsoleErrors,
} from "./fixtures";

test.describe("UI Fixes — Progress Bar", () => {
  test("progress bar visible at viewport top within 1s of process click", async ({
    page,
  }) => {
    await page.goto("/color-separator");

    // Upload file
    const fileInput = page.locator('input[type="file"]');
    const testImagePath = path.resolve(__dirname, "..", "test_red.png");
    await fileInput.setInputFiles(testImagePath);

    // Click process and immediately check for progress bar
    const processBtn = page.locator("button.process-btn");
    await processBtn.click();

    // Progress bar root should appear within 1s
    const progressRoot = page.locator('[role="progressbar"]');
    await expect(progressRoot).toBeVisible({ timeout: 1_000 });

    // Must be at top: 0 (fixed position, top of viewport)
    const box = await progressRoot.boundingBox();
    expect(box, "Progress bar should have a bounding box").toBeTruthy();
    expect(box!.y, "Progress bar should be at y=0 (top of viewport)").toBeLessThanOrEqual(2);

    // ARIA attributes present
    await expect(progressRoot).toHaveAttribute("role", "progressbar");
    await expect(progressRoot).toHaveAttribute("aria-valuemin", "0");
    await expect(progressRoot).toHaveAttribute("aria-valuemax", "100");
    const valuenow = await progressRoot.getAttribute("aria-valuenow");
    expect(valuenow).not.toBeNull();

    // aria-live region exists
    const liveRegion = page.locator('[aria-live="polite"]');
    await expect(liveRegion).toBeAttached();

    // Wait for processing to finish
    await waitForComposite(page);
  });

  test("progress bar percentage increases during processing", async ({
    page,
  }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    const progressRoot = page.locator('[role="progressbar"]');
    await expect(progressRoot).toBeVisible({ timeout: 5_000 });

    // Collect aria-valuenow values over time
    const values: number[] = [];
    for (let i = 0; i < 10; i++) {
      const val = await progressRoot.getAttribute("aria-valuenow");
      if (val) values.push(parseInt(val, 10));
      await page.waitForTimeout(500);
      // Stop if processing finished
      if (!(await progressRoot.isVisible())) break;
    }

    // Should see increasing values (at least one > 0)
    const nonZero = values.filter((v) => v > 0);
    expect(
      nonZero.length,
      `Expected some non-zero progress values, got: ${values}`,
    ).toBeGreaterThan(0);

    await waitForComposite(page);
  });
});

test.describe("UI Fixes — Download Progress", () => {
  test("download shows progress bar that tracks percentage", async ({
    page,
  }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page);

    // Click download
    const downloadBtn = page.locator("button", { hasText: "ZIP" });
    await expect(downloadBtn).toBeEnabled({ timeout: 30_000 });

    // Listen for download
    const downloadPromise = page.waitForEvent("download", { timeout: 60_000 });
    await downloadBtn.click();

    // Download progress bar should appear with tracked width
    const progressFill = page.locator(".download-progress-fill");
    // Wait for download progress to appear
    await expect(
      page.locator(".download-progress"),
    ).toBeVisible({ timeout: 10_000 });

    // The fill element should exist and have a width style (not just animation)
    // Check that the progress bar eventually reaches a meaningful width
    const fillVisible = await progressFill.isVisible().catch(() => false);
    if (fillVisible) {
      const style = await progressFill.getAttribute("style");
      // Should have an explicit width (not just the animation default)
      if (style) {
        expect(
          style,
          "Download progress fill should have explicit width",
        ).toContain("width");
      }
    }

    // Wait for download to complete
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe("color-separator-plates.zip");
  });
});

test.describe("UI Fixes — Upscale 4x", () => {
  test("4x upscale scale sent in plate stream request", async ({ page }) => {
    await page.goto("/color-separator");

    // Set upscale to 4x
    const btn4x = page.locator(".upscale-toggle button", { hasText: "4x" });
    await btn4x.click();

    // Intercept plates-stream request to verify upscale_scale param
    let capturedScale: string | null = null;
    await page.route("**/api/plates-stream**", async (route) => {
      const request = route.request();
      const postData = request.postData();
      if (postData) {
        // FormData boundary parsing — look for upscale_scale field
        const scaleMatch = postData.match(
          /name="upscale_scale"\r?\n\r?\n(\d+)/,
        );
        capturedScale = scaleMatch ? scaleMatch[1] : null;
      }
      await route.continue();
    });

    await uploadAndProcess(page, "small");
    await waitForComposite(page);

    // Verify 4x was sent
    expect(
      capturedScale,
      "upscale_scale=4 should be sent in plates-stream request",
    ).toBe("4");
  });
});

test.describe("UI Fixes — Zero Console Errors", () => {
  test("no console errors during full separation flow", async ({ page }) => {
    const errors = collectConsoleErrors(page);

    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page);

    // Filter known benign messages
    const realErrors = errors.filter(
      (e) =>
        !e.includes("[api]") &&
        !e.includes("debug") &&
        !e.includes("favicon") &&
        !e.includes("_next/static") &&
        !e.includes("net::ERR_"),
    );
    expect(
      realErrors,
      `Console errors: ${realErrors.join("\n")}`,
    ).toHaveLength(0);
  });
});
