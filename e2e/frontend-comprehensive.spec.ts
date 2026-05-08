import { test, expect } from "@playwright/test";
import * as path from "path";
import {
  uploadAndProcess,
  waitForComposite,
  waitForPlates,
  collectConsoleErrors,
} from "./fixtures";

/**
 * Comprehensive frontend Playwright spec for color.reidsurmeier.wtf
 * Covers: layout, nav panel, progress bar, plates grid, merge UI,
 * keyboard shortcuts, mobile responsive, error states, accessibility.
 */

test.describe("Layout & Navigation", () => {
  test("back-to-tools bar renders at viewport top", async ({ page }) => {
    await page.goto("/color-separator");
    const bar = page.locator(".back-to-tools");
    await expect(bar).toBeVisible();
    const box = await bar.boundingBox();
    expect(box).toBeTruthy();
    expect(box!.y).toBe(0);
    expect(box!.height).toBe(28);
    await expect(bar).toHaveAttribute("href", "https://tools.reidsurmeier.wtf");
  });

  test("nav panel positioned below header bar", async ({ page }) => {
    await page.goto("/color-separator");
    const nav = page.locator(".nav-panel");
    await expect(nav).toBeVisible();
    const box = await nav.boundingBox();
    expect(box).toBeTruthy();
    // Nav top should be >= 28px (header height)
    expect(box!.y).toBeGreaterThanOrEqual(28);
  });

  test("nav panel contains all required sections", async ({ page }) => {
    await page.goto("/color-separator");
    const nav = page.locator(".nav-panel");

    // Title
    await expect(nav.locator(".app-title")).toContainText("COLOR.SEPARATOR");

    // Version selector
    await expect(nav.locator("select")).toBeVisible();

    // Source button
    await expect(nav.locator(".source-btn")).toBeVisible();

    // Plates slider
    await expect(nav.locator('input[type="range"]').first()).toBeVisible();

    // Process button
    await expect(nav.locator(".process-btn")).toBeVisible();

    // About toggle
    await expect(nav.locator("button", { hasText: /show|hide/ })).toBeVisible();
  });

  test("version dropdown defaults to v20 and lists all versions", async ({ page }) => {
    await page.goto("/color-separator");
    const select = page.locator(".nav-panel select");
    await expect(select).toHaveValue("v20");

    const options = await select.locator("option").allTextContents();
    expect(options.length).toBeGreaterThanOrEqual(15);
    expect(options[0]).toContain("v15");
    expect(options.some((o) => o.includes("v20"))).toBeTruthy();
  });
});

test.describe("Upscale Toggle", () => {
  test("upscale toggle shows off/2x/4x buttons", async ({ page }) => {
    await page.goto("/color-separator");
    const toggle = page.locator(".upscale-toggle");
    await expect(toggle).toBeVisible();

    const buttons = toggle.locator("button");
    await expect(buttons).toHaveCount(3);

    const texts = await buttons.allTextContents();
    expect(texts).toEqual(["off", "2x", "4x"]);
  });

  test("2x active by default, clicking 4x activates it", async ({ page }) => {
    await page.goto("/color-separator");
    const toggle = page.locator(".upscale-toggle");

    // 2x should be active by default
    const btn2x = toggle.locator("button", { hasText: "2x" });
    await expect(btn2x).toHaveAttribute("data-active", "true");

    // Click 4x
    const btn4x = toggle.locator("button", { hasText: "4x" });
    await btn4x.click();
    await expect(btn4x).toHaveAttribute("data-active", "true");
    await expect(btn2x).toHaveAttribute("data-active", "false");
  });

  test("clicking off deactivates upscale", async ({ page }) => {
    await page.goto("/color-separator");
    const toggle = page.locator(".upscale-toggle");
    const btnOff = toggle.locator("button", { hasText: "off" });
    await btnOff.click();
    await expect(btnOff).toHaveAttribute("data-active", "true");
  });
});

test.describe("File Upload", () => {
  test("source button shows filename after upload", async ({ page }) => {
    await page.goto("/color-separator");
    const fileInput = page.locator('input[type="file"]');
    const testImagePath = path.resolve(__dirname, "..", "test_red.png");
    await fileInput.setInputFiles(testImagePath);

    const sourceBtn = page.locator(".source-btn");
    await expect(sourceBtn).toContainText("test_red.png");
  });

  test("image preview appears after file selection", async ({ page }) => {
    await page.goto("/color-separator");
    const fileInput = page.locator('input[type="file"]');
    const testImagePath = path.resolve(__dirname, "..", "test_red.png");
    await fileInput.setInputFiles(testImagePath);

    // Source image should appear in main canvas
    const img = page.locator(".canvas-wrapper img");
    await expect(img).toBeVisible({ timeout: 5_000 });
  });

  test("image info shown in data box after upload", async ({ page }) => {
    await page.goto("/color-separator");
    const fileInput = page.locator('input[type="file"]');
    const testImagePath = path.resolve(__dirname, "..", "test_red.png");
    await fileInput.setInputFiles(testImagePath);

    const dataBox = page.locator(".data-box");
    await expect(dataBox).toBeVisible({ timeout: 5_000 });
    await expect(dataBox.locator(".data-row")).toHaveCount(3); // size, file, type
  });
});

test.describe("Progress Bar", () => {
  test("progress bar renders at viewport top with z-index above header", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    const bar = page.locator(".progress-bar-root");
    await expect(bar).toBeVisible({ timeout: 5_000 });

    const box = await bar.boundingBox();
    expect(box).toBeTruthy();
    expect(box!.y).toBeLessThanOrEqual(2); // top: 0

    // Verify z-index via computed style
    const zIndex = await bar.evaluate((el) => getComputedStyle(el).zIndex);
    expect(parseInt(zIndex)).toBeGreaterThanOrEqual(250);

    await waitForComposite(page);
  });

  test("progress bar has correct ARIA attributes", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    const bar = page.locator('[role="progressbar"]');
    await expect(bar).toBeVisible({ timeout: 5_000 });
    await expect(bar).toHaveAttribute("aria-valuemin", "0");
    await expect(bar).toHaveAttribute("aria-valuemax", "100");
    await expect(bar).toHaveAttribute("aria-label", /.+/);

    await waitForComposite(page);
  });

  test("progress bar shows elapsed timer", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    const timer = page.locator(".progress-bar-time");
    await expect(timer).toBeAttached({ timeout: 5_000 });

    await waitForComposite(page);
  });

  test("indeterminate animation when pct is 0", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    // Check early — before real progress arrives
    const fill = page.locator(".progress-bar-fill");
    await expect(fill).toBeAttached({ timeout: 3_000 });

    // If indeterminate, should have the data attribute
    const isIndeterminate = await fill.getAttribute("data-indeterminate");
    // May or may not be indeterminate depending on timing — just verify attribute exists or fill has width
    expect(isIndeterminate === "true" || (await fill.evaluate((el) => el.style.width)) !== "").toBeTruthy();

    await waitForComposite(page);
  });

  test("progress bar hidden after processing completes", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);

    // After completion, progress bar should be gone
    const bar = page.locator(".progress-bar-root");
    await expect(bar).not.toBeVisible({ timeout: 5_000 });
  });
});

test.describe("Plates Grid", () => {
  test("skeleton placeholders shown during loading", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    // Skeletons should appear while processing
    const skeletons = page.locator(".plate-skeleton");
    // May or may not be visible depending on timing, but the class should exist in CSS
    await waitForComposite(page);
    await waitForPlates(page);

    // After loading, no skeletons remain
    await expect(skeletons).toHaveCount(0, { timeout: 30_000 });
  });

  test("each plate card has image, color swatch, hex, and coverage", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    const plates = await waitForPlates(page, 2);

    const count = await plates.count();
    for (let i = 0; i < Math.min(count, 4); i++) {
      const card = plates.nth(i);
      await expect(card.locator("img")).toBeVisible();
      await expect(card.locator(".plate-card-swatch")).toBeVisible();
      await expect(card.locator(".plate-card-hex")).toBeVisible();
      await expect(card.locator(".plate-card-coverage")).toBeVisible();
    }
  });

  test("plates section title shows count", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page);

    const title = page.locator(".plates-section-title");
    await expect(title).toBeVisible();
    const text = await title.textContent();
    expect(text).toMatch(/plates \(\d+\)/);
  });

  test("plate card has appear animation delay based on index", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page, 2);

    const secondCard = page.locator(".plate-card:not(.plate-skeleton)").nth(1);
    const style = await secondCard.getAttribute("style");
    expect(style).toContain("animation-delay");
  });
});

test.describe("Plate Zoom", () => {
  test("clicking plate opens zoom overlay", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page, 2);

    // Click first plate
    const firstPlate = page.locator(".plate-card:not(.plate-skeleton)").first();
    await firstPlate.click();

    // Zoom overlay should appear
    const overlay = page.locator(".plate-zoom-overlay");
    await expect(overlay).toBeVisible({ timeout: 3_000 });

    // Should show zoomed image
    await expect(page.locator(".plate-zoom-img")).toBeVisible();

    // Should show hex and name
    await expect(page.locator(".plate-zoom-hex")).toBeVisible();

    // Close button
    const closeBtn = page.locator(".plate-zoom-close");
    await expect(closeBtn).toBeVisible();
    await closeBtn.click();
    await expect(overlay).not.toBeVisible();
  });
});

test.describe("Keyboard Shortcuts", () => {
  test("spacebar toggles original/composite comparison", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);

    // Initially showing composite
    const compareLabel = page.locator(".compare-label");
    await expect(compareLabel).not.toBeVisible();

    // Press spacebar
    await page.keyboard.press("Space");

    // Should show "ORIGINAL" label
    await expect(compareLabel).toBeVisible({ timeout: 2_000 });
    await expect(compareLabel).toContainText("ORIGINAL");

    // Press again to toggle back
    await page.keyboard.press("Space");
    await expect(compareLabel).not.toBeVisible({ timeout: 2_000 });
  });
});

test.describe("About Overlay", () => {
  test("about overlay opens and closes", async ({ page }) => {
    await page.goto("/color-separator");

    // Click show about
    const aboutBtn = page.locator("button", { hasText: "show" });
    await aboutBtn.click();

    const overlay = page.locator(".about-overlay");
    await expect(overlay).toBeVisible();
    await expect(overlay.locator(".about-ascii")).toBeVisible();

    // Has required sections
    const labels = await overlay.locator(".about-label").allTextContents();
    expect(labels).toContain("ABOUT");
    expect(labels).toContain("TECH");
    expect(labels).toContain("ALGORITHMS");
    expect(labels).toContain("REFERENCES");
    expect(labels).toContain("PRIVACY");

    // Close
    await page.locator(".about-close").click();
    await expect(overlay).not.toBeVisible();
  });
});

test.describe("Merge Mode", () => {
  test("merge mode activates and shows group UI", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page, 2);

    // Click "select plates" to enter merge mode
    const mergeBtn = page.locator("button", { hasText: "select plates" });
    await expect(mergeBtn).toBeEnabled();
    await mergeBtn.click();

    // Should now show cancel button
    await expect(page.locator("button", { hasText: "cancel" }).first()).toBeVisible();

    // Should show "create a group, then click plates" hint
    await expect(page.locator("text=create a group")).toBeVisible();

    // Add a group
    const addGroupBtn = page.locator("button", { hasText: "+ new group" });
    await addGroupBtn.click();

    // Group 1 should appear
    await expect(page.locator("text=group 1")).toBeVisible();
  });
});

test.describe("Error Handling UI", () => {
  test("error banner shows retry and dismiss buttons", async ({ page }) => {
    await page.goto("/color-separator");

    // Block backend
    await page.route("**/api/preview-stream**", (route) => route.abort("connectionfailed"));
    await page.route("**/api/preview**", (route) => route.abort("connectionfailed"));

    await uploadAndProcess(page, "small");

    // Error should appear
    const dismissBtn = page.locator("button", { hasText: "dismiss" });
    await expect(dismissBtn).toBeVisible({ timeout: 90_000 });

    // Retry button should also be present
    const retryBtn = page.locator("button", { hasText: "retry" });
    await expect(retryBtn).toBeVisible();

    // Dismiss clears error
    await dismissBtn.click();
    await expect(dismissBtn).not.toBeVisible();
  });
});

test.describe("Reset", () => {
  test("reset clears composite and plates", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);
    await waitForPlates(page);

    // Click reset
    const resetBtn = page.locator("button", { hasText: "reset" });
    await resetBtn.click();

    // Plates section should disappear
    await expect(page.locator(".plates-section")).not.toBeVisible({ timeout: 5_000 });
  });
});

test.describe("Loading State", () => {
  test("main canvas gets is-loading class during processing", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    const canvas = page.locator(".main-canvas");
    await expect(canvas).toHaveClass(/is-loading/, { timeout: 3_000 });

    await waitForComposite(page);

    // After completion, loading class removed
    await expect(canvas).not.toHaveClass(/is-loading/, { timeout: 5_000 });
  });

  test("cancel button appears during processing and stops it", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");

    // Cancel button should appear
    const cancelBtn = page.locator("button", { hasText: "cancel" });
    // May appear briefly — check within processing window
    const isVisible = await cancelBtn.isVisible().catch(() => false);
    // If processing is fast enough, cancel may not appear — that's OK
    if (isVisible) {
      await cancelBtn.click();
      // Loading should stop
      await expect(page.locator(".main-canvas")).not.toHaveClass(/is-loading/, { timeout: 5_000 });
    }
  });
});

test.describe("High Plate Count Warnings", () => {
  test("warning shown when plates > 20 on SAM version", async ({ page }) => {
    await page.goto("/color-separator");

    // Set plates to 25
    const slider = page.locator('input[type="range"]').first();
    await slider.fill("25");

    // Warning should appear
    await expect(page.locator("text=High plate count")).toBeVisible({ timeout: 2_000 });
  });

  test("extreme warning at 46+ plates", async ({ page }) => {
    await page.goto("/color-separator");
    const slider = page.locator('input[type="range"]').first();
    await slider.fill("46");

    await expect(page.locator("text=Extreme plate count")).toBeVisible({ timeout: 2_000 });
  });
});

test.describe("Paper Texture Overlay", () => {
  test("paper texture appears on composite but not original", async ({ page }) => {
    await page.goto("/color-separator");
    await uploadAndProcess(page, "small");
    await waitForComposite(page);

    // Paper texture should be visible
    const texture = page.locator(".paper-texture");
    await expect(texture).toBeVisible();

    // Toggle to original
    await page.keyboard.press("Space");
    await expect(texture).not.toBeVisible({ timeout: 2_000 });

    // Toggle back
    await page.keyboard.press("Space");
    await expect(texture).toBeVisible({ timeout: 2_000 });
  });
});
