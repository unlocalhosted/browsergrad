/**
 * Browser-mode vitest config for the WebGPU realizer + WGSL kernel tests.
 *
 * Launches Chromium via Playwright with WebGPU enabled. The headless
 * `--enable-unsafe-swiftshader` flag is required for Apple Silicon and
 * for any environment that doesn't have a hardware-accelerated GPU
 * visible to headless Chromium (CI runners often fall into this).
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

export default defineConfig({
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
          launch: {
            // Headless Chromium on macOS ARM64 needs a specific cocktail
            // for WebGPU: software-rendered Vulkan via SwiftShader is the
            // reliable cross-platform answer (slow but functional).
            //
            // The hierarchy of supported headless adapters varies by
            // platform; we throw the kitchen sink and fall back gracefully
            // if none materialise.
            args: [
              "--enable-unsafe-webgpu",
              "--enable-features=Vulkan,UseSkiaRenderer",
              "--enable-unsafe-swiftshader",
              "--use-vulkan=swiftshader",
              "--use-angle=swiftshader",
              "--disable-gpu-sandbox",
              "--ignore-gpu-blocklist",
              "--no-sandbox",
            ],
          },
        },
      ],
    },
  },
});
