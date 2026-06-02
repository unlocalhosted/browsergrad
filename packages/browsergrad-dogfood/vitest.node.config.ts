/**
 * Node-mode vitest config — Pyodide-in-Node for grad / jit / runtime.
 *
 * Pyodide in Node is significantly faster to bootstrap than Pyodide in
 * headed Chromium, and the Python source is identical, so these tests gate
 * the published Python tarball just as faithfully as the browser path
 * would. Browser-specific tests (WebGPU, WGSL kernels) live in
 * `vitest.config.ts`.
 */

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests-node/**/*.test.ts"],
    testTimeout: 180_000,
    hookTimeout: 180_000,
    environment: "node",
    // grad-0.5.1 and jit-0.8.1 add `with { type: "json" }` to their
    // package.json ESM imports, so the inline workaround that prior
    // versions needed is no longer required. The compat test in
    // tests-node/published_module_compat.test.ts now asserts the raw
    // Node import succeeds.
  },
});
