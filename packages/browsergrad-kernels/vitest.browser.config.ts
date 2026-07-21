/**
 * Browser-mode vitest config for the WebGPU realizer + WGSL kernel tests.
 *
 * Launches Chromium via Playwright with WebGPU enabled. The headless
 * The minimal `--enable-unsafe-webgpu` launch profile matches the required
 * real-world browser gate and lets Chromium select its available adapter.
 *
 * Tests that need a real GPUDevice import navigator.gpu directly. When
 * the adapter request returns null (no GPU even via SwiftShader), the
 * test skips with a clear message instead of failing — the bench is
 * data collection, not a regression gate.
 *
 * Run with: `pnpm test:browser`
 */

import { defineConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

function packageVersion(url: URL): string {
  const parsed = JSON.parse(readFileSync(url, "utf8")) as { readonly version?: unknown };
  if (typeof parsed.version !== "string" || parsed.version.length === 0) {
    throw new Error(`missing package version in ${url.pathname}`);
  }
  return parsed.version;
}

const kernelsVersion = packageVersion(new URL("./package.json", import.meta.url));
const semanticCoreVersion = packageVersion(new URL("../browsergrad-semantic-core/package.json", import.meta.url));

export default defineConfig(({ mode }) => ({
  define: {
    __BG_REQUIRE_WEBGPU__: JSON.stringify(mode === "webgpu-required"),
    __BG_KERNELS_VERSION__: JSON.stringify(kernelsVersion),
    __BG_SEMANTIC_CORE_VERSION__: JSON.stringify(semanticCoreVersion),
  },
  test: {
    include: ["tests-browser/**/*.test.ts"],
    testTimeout: 60_000,
    hookTimeout: 60_000,
    browser: {
      enabled: true,
      provider: playwright(),
      // Headless Chromium on macOS doesn't expose a GPU adapter for
      // WebGPU. Run headed locally to get the real Metal driver; CI on
      // Linux needs xvfb. The CI workflow flips this via an env var.
      headless: process.env.BG_BROWSER_HEADLESS === "1",
      instances: [
        {
          browser: "chromium",
          launch: { args: ["--enable-unsafe-webgpu"] },
        },
      ],
    },
  },
  server: {
    fs: { allow: [fileURLToPath(new URL("../..", import.meta.url))] },
  },
}));
