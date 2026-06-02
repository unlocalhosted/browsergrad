/**
 * Browser-mode vitest config. Real Chromium + Playwright + WebGPU.
 * Mirrors the kernels package's own browser config, with the difference
 * that imports come from `@unlocalhosted/browsergrad-kernels` (npm)
 * rather than `../src/*` (source tree).
 */

import { defineConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    testTimeout: 60_000,
    hookTimeout: 60_000,
    browser: {
      enabled: true,
      provider: playwright(),
      // Headed on macOS to get the real Metal driver. CI on Linux can flip
      // via BG_BROWSER_HEADLESS=1 (relies on SwiftShader Vulkan).
      headless: process.env.BG_BROWSER_HEADLESS === "1",
      instances: [
        {
          browser: "chromium",
          launch: {
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
