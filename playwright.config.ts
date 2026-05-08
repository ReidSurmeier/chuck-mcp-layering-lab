import { defineConfig, devices } from "@playwright/test";

const BASE_URL = process.env.BASE_URL || "http://localhost:8004";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // sequential — shared GPU backend
  retries: 0, // no retries during red/green TDD phase
  timeout: 120_000, // 2min per test (SAM processing is slow)
  expect: { timeout: 60_000 },
  reporter: [["html", { open: "never" }], ["list"]],
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
