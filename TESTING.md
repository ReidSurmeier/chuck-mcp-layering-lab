# Color Separator — Test Suite

## Structure

```
e2e/
  fixtures.ts          — shared helpers (upload, wait, timing)
  color-separator.spec.ts — 6 core criteria + error handling
  no-auth-gate.spec.ts — auth gate regression test
backend/tests/
  test_performance.py  — backend API performance + correctness
playwright.config.ts   — Playwright configuration
TESTING.md             — this file
```

## Prerequisites

### Frontend (Playwright)
```bash
# From project root (C:\colorsep on Windows)
npm install
npx playwright install chromium
```

### Backend (pytest)
```bash
# Inside backend container or with backend virtualenv
pip install pytest requests
```

## Running Tests

### E2E Tests (Playwright)
```bash
# Run all E2E tests against live site
npx playwright test

# Run specific spec
npx playwright test e2e/color-separator.spec.ts

# Run with headed browser (debug)
npx playwright test --headed

# Run against local dev
BASE_URL=http://localhost:3000 npx playwright test
```

### Backend Tests (pytest)
```bash
# From project root, hitting local backend
pytest backend/tests/ -v

# Hit different backend URL
BACKEND_URL=http://localhost:8001 pytest backend/tests/ -v
```

### Full Suite
```bash
npx playwright test && pytest backend/tests/ -v
```

## Success Criteria Map

| # | Criterion | Test |
|---|-----------|------|
| 1 | Upload → composite → plates → zip download | `color-separator.spec.ts` test 2 + 5 |
| 2 | Plates appear concurrently (≤2s after composite) | `color-separator.spec.ts` test 3 |
| 3 | Progress bar visible and tracks stages | `color-separator.spec.ts` test 4 |
| 4 | v20 backend ≤30s cached / ≤45s cold | `color-separator.spec.ts` test 6 + `test_performance.py` |
| 5 | ZIP download contains plates | `color-separator.spec.ts` test 5 + `test_performance.py` |
| 6 | Zero console errors | `color-separator.spec.ts` test 2 |

## Expected Initial State (Red)

All tests should initially FAIL to validate they're testing real behavior:
- Timing tests may fail if backend is cold or overloaded
- Download test may fail if processing doesn't complete
- Progress bar test depends on actual SSE streaming working

After fixes, re-run 3x to confirm zero flakiness:
```bash
for i in 1 2 3; do npx playwright test && echo "PASS $i" || echo "FAIL $i"; done
```
