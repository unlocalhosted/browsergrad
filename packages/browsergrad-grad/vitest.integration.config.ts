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
    // Pyodide boot + numpy preload can take 30+s on a cold cache.
    testTimeout: 60_000,
    hookTimeout: 120_000,
    // Pyodide and the installed grad module are single-threaded global state;
    // run sequentially in a single fork.
    pool: "forks",
    fileParallelism: false,
  },
});
