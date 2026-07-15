import { defineConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

function packageVersion(url: URL): string {
  const parsed = JSON.parse(readFileSync(url, "utf8")) as { readonly version?: unknown };
  if (typeof parsed.version !== "string" || parsed.version.length === 0) {
    throw new Error(`missing package version in ${url.pathname}`);
  }
  return parsed.version;
}

const compilerVersion = packageVersion(new URL("./package.json", import.meta.url));
const kernelsVersion = packageVersion(new URL("../browsergrad-kernels/package.json", import.meta.url));
const semanticCoreVersion = packageVersion(new URL("../browsergrad-semantic-core/package.json", import.meta.url));
const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const sourceRevision = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: repositoryRoot,
  encoding: "utf8",
}).trim();
if (!/^[0-9a-f]{40}$/u.test(sourceRevision)) {
  throw new Error(`git HEAD is not a full lowercase commit SHA: ${sourceRevision}`);
}

export default defineConfig(({ mode }) => ({
  define: {
    __BG_REQUIRE_WEBGPU__: JSON.stringify(mode === "webgpu-required"),
    __BG_COMPILER_VERSION__: JSON.stringify(compilerVersion),
    __BG_KERNELS_VERSION__: JSON.stringify(kernelsVersion),
    __BG_SEMANTIC_CORE_VERSION__: JSON.stringify(semanticCoreVersion),
    __BG_SOURCE_REVISION__: JSON.stringify(sourceRevision),
  },
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
  server: {
    fs: { allow: [fileURLToPath(new URL("../..", import.meta.url))] },
  },
}));
