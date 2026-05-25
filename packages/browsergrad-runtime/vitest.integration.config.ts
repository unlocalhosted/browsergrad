import { defineConfig } from "vitest/config";

/**
 * Integration tests run with a real Pyodide in node, exercising the
 * runtime's Python-side protocol (PY_PREAMBLE + browsergrad module) and
 * — via a FakeWorker — the client.ts message routing.
 *
 * Separate config so `pnpm test` stays snappy. Run via `pnpm test:integration`.
 */
export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["tests-integration/**/*.test.ts"],
    testTimeout: 60_000,
    hookTimeout: 120_000,
    pool: "forks",
    fileParallelism: false,
  },
});
