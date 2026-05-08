import { test, expect } from '@playwright/test';

test('color separator has no auth gate', async ({ page }) => {
  await page.goto('/color-separator');
  // Auth gate should NOT be present
  await expect(page.locator('text=GPU Access Required')).not.toBeVisible();
  await expect(page.locator('text=Enter the password')).not.toBeVisible();
  // Upload area SHOULD be present (file input is hidden, but source button visible)
  await expect(page.locator('input[type="file"]')).toBeAttached({ timeout: 10000 });
  await expect(page.locator('button.source-btn')).toBeVisible({ timeout: 10000 });
});
