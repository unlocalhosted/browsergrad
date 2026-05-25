import { defineConfig } from "vitest/config";

/**
 * Node-mode tests only for now.
 *
 * The worker + Pyodide path requires a real browser env (Vitest browser mode
 * with playwright or @vitest/browser) — added in v0.2. For v0, tests cover
 * the type surface, error paths, and pure-JS helpers.
 */
export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["tests/**/*.test.ts"],
  },
});
