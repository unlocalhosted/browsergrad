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
