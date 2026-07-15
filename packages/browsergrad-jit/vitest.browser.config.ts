import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";
import { captureJitSemanticPermuteSubmissions } from "./tests-integration/semantic-permute-emission-capture";

function packageVersion(url: URL): string {
  const parsed = JSON.parse(readFileSync(url, "utf8")) as { readonly version?: unknown };
  if (typeof parsed.version !== "string" || parsed.version.length === 0) {
    throw new Error(`missing package version in ${url.pathname}`);
  }
  return parsed.version;
}

const jitVersion = packageVersion(new URL("./package.json", import.meta.url));
const kernelsVersion = packageVersion(new URL("../browsergrad-kernels/package.json", import.meta.url));
const semanticCoreVersion = packageVersion(new URL("../browsergrad-semantic-core/package.json", import.meta.url));
const workspaceRoot = fileURLToPath(new URL("../..", import.meta.url));
const sourceRevision = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: workspaceRoot,
  encoding: "utf8",
}).trim();
if (!/^[0-9a-f]{40}$/u.test(sourceRevision)) {
  throw new Error(`git HEAD is not a full lowercase commit SHA: ${sourceRevision}`);
}

let jitEmissionCapture: string;
try {
  jitEmissionCapture = await captureJitSemanticPermuteSubmissions();
} catch (error) {
  jitEmissionCapture = JSON.stringify({
    schema: "browsergrad.jit.semantic-permute-emission-capture-failure",
    error: error instanceof Error
      ? { name: error.name, message: error.message }
      : { name: "UnknownError", message: String(error) },
  });
}

export default defineConfig(({ mode }) => ({
  define: {
    __BG_REQUIRE_WEBGPU__: JSON.stringify(mode === "webgpu-required"),
    __BG_JIT_VERSION__: JSON.stringify(jitVersion),
    __BG_KERNELS_VERSION__: JSON.stringify(kernelsVersion),
    __BG_SEMANTIC_CORE_VERSION__: JSON.stringify(semanticCoreVersion),
    __BG_SOURCE_REVISION__: JSON.stringify(sourceRevision),
    __BG_JIT_SEMANTIC_PERMUTE_CAPTURE_JSON__: JSON.stringify(jitEmissionCapture),
  },
  test: {
    include: ["tests-browser/**/*.test.ts"],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    browser: {
      enabled: true,
      provider: playwright(),
      headless: process.env.BG_BROWSER_HEADLESS === "1",
      instances: [{
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
      }],
    },
  },
  server: {
    fs: { allow: [fileURLToPath(new URL("../..", import.meta.url))] },
  },
}));
