import { defineConfig } from "vitest/config";

/**
 * Integration tests boot Pyodide and run real Python.
 * Slower than surface tests (5-15s for Pyodide boot + numpy load) — kept
 * in a separate config so `pnpm test` stays snappy.
 *
 * Run with `pnpm test:integration`.
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
