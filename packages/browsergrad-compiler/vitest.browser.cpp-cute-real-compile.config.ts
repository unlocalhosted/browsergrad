import { createReadStream } from "node:fs";
import { statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig, type Plugin } from "vitest/config";

interface ExternalAsset {
  readonly assetId: string;
  readonly path: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: number;
}

interface ExternalInputs {
  readonly schema: "browsergrad.compiler.cpp-cute.browser-real-compile-inputs";
  readonly version: 3;
  readonly authority: "local-exact-byte-preflight-only";
  readonly caseId:
    | "rank2"
    | "rank3"
    | "rank1"
    | "rank4"
    | "strided-slice"
    | "broadcast"
    | "i32-rank2"
    | "u32-broadcast";
  readonly assets: readonly ExternalAsset[];
  readonly wasmAuthority:
    | "package-pinned-two-clean-build-output"
    | "untrusted-diagnostic-local-byte-observation-only";
  readonly pinnedReproducibleWasmMatched: boolean;
  readonly untrustedDiagnosticWasm: boolean;
  readonly headerDistributionReproducibilityId: string;
  readonly headerDistributionOutputVerificationId: string;
  readonly packagePinnedHeaderPacksMatched: true;
  readonly headerDistributionLicenseApproved: false;
  readonly producerTrusted: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

const inputs = parseInputs(process.env.BG_CPP_CUTE_REAL_COMPILE_INPUTS);
const routePrefix = "/__browsergrad_cpp_cute_real_compile__/";
const assetsByRoute = new Map(inputs.assets.map((asset) => [
  `${routePrefix}${encodeURIComponent(asset.assetId)}`,
  asset,
]));
const browserInputs = {
  schema: inputs.schema,
  version: inputs.version,
  authority: inputs.authority,
  caseId: inputs.caseId,
  assets: inputs.assets.map(({ path: _path, ...asset }) => asset),
  wasmAuthority: inputs.wasmAuthority,
  pinnedReproducibleWasmMatched: inputs.pinnedReproducibleWasmMatched,
  untrustedDiagnosticWasm: inputs.untrustedDiagnosticWasm,
  headerDistributionReproducibilityId:
    inputs.headerDistributionReproducibilityId,
  headerDistributionOutputVerificationId:
    inputs.headerDistributionOutputVerificationId,
  packagePinnedHeaderPacksMatched: inputs.packagePinnedHeaderPacksMatched,
  headerDistributionLicenseApproved: inputs.headerDistributionLicenseApproved,
  producerTrusted: inputs.producerTrusted,
  workerExecutionObserved: inputs.workerExecutionObserved,
  releaseReady: inputs.releaseReady,
};
const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

function externalAssetServer(): Plugin {
  return {
    name: "browsergrad-cpp-cute-real-compile-assets",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = request.url?.split("?", 1)[0];
        if (path === undefined || !path.startsWith(routePrefix)) {
          next();
          return;
        }
        const asset = assetsByRoute.get(path);
        if (request.method !== "GET" || asset === undefined ||
            request.headers.range !== undefined) {
          response.statusCode = asset === undefined ? 404 : 405;
          response.end();
          return;
        }
        const observed = statSync(asset.path);
        if (!observed.isFile() || observed.size !== asset.byteLength) {
          response.statusCode = 409;
          response.end();
          return;
        }
        response.statusCode = 200;
        response.setHeader("Content-Type", asset.mediaType);
        response.setHeader("Content-Length", String(asset.byteLength));
        response.setHeader("Cache-Control", "no-store");
        response.setHeader("X-Content-Type-Options", "nosniff");
        const stream = createReadStream(asset.path);
        stream.once("error", () => {
          if (!response.headersSent) response.statusCode = 500;
          response.destroy();
        });
        stream.pipe(response);
      });
    },
  };
}

function parseInputs(value: string | undefined): ExternalInputs {
  if (value === undefined) throw new Error("BG_CPP_CUTE_REAL_COMPILE_INPUTS is required");
  const parsed = JSON.parse(value) as Partial<ExternalInputs>;
  const pinned =
    parsed.pinnedReproducibleWasmMatched === true &&
    parsed.untrustedDiagnosticWasm === false &&
    parsed.wasmAuthority === "package-pinned-two-clean-build-output";
  const diagnostic =
    parsed.pinnedReproducibleWasmMatched === false &&
    parsed.untrustedDiagnosticWasm === true &&
    parsed.wasmAuthority === "untrusted-diagnostic-local-byte-observation-only";
  if (parsed.schema !== "browsergrad.compiler.cpp-cute.browser-real-compile-inputs" ||
      parsed.version !== 3 ||
      parsed.authority !== "local-exact-byte-preflight-only" ||
      (parsed.caseId !== "rank2" &&
       parsed.caseId !== "rank3" &&
       parsed.caseId !== "rank1" &&
       parsed.caseId !== "rank4" &&
       parsed.caseId !== "strided-slice" &&
       parsed.caseId !== "broadcast" &&
       parsed.caseId !== "i32-rank2" &&
       parsed.caseId !== "u32-broadcast") ||
      (!pinned && !diagnostic) ||
      !/^bg\.cpp\.browser-header-distribution-reproducibility\.sha256\.[0-9a-f]{64}$/u
        .test(parsed.headerDistributionReproducibilityId ?? "") ||
      !/^bg\.cpp\.distribution-output-file-verification\.sha256\.[0-9a-f]{64}$/u
        .test(parsed.headerDistributionOutputVerificationId ?? "") ||
      parsed.packagePinnedHeaderPacksMatched !== true ||
      parsed.headerDistributionLicenseApproved !== false ||
      parsed.producerTrusted !== false ||
      parsed.workerExecutionObserved !== false ||
      parsed.releaseReady !== false ||
      !Array.isArray(parsed.assets) ||
      parsed.assets.length !== 6) {
    throw new Error("BG_CPP_CUTE_REAL_COMPILE_INPUTS has an invalid authority boundary");
  }
  const seen = new Set<string>();
  for (const asset of parsed.assets) {
    if (typeof asset !== "object" || asset === null ||
        typeof asset.assetId !== "string" ||
        typeof asset.path !== "string" ||
        typeof asset.mediaType !== "string" ||
        !/^[0-9a-f]{64}$/u.test(asset.sha256) ||
        !Number.isSafeInteger(asset.byteLength) ||
        asset.byteLength <= 0 ||
        seen.has(asset.assetId)) {
      throw new Error("BG_CPP_CUTE_REAL_COMPILE_INPUTS contains an invalid asset");
    }
    seen.add(asset.assetId);
  }
  return parsed as ExternalInputs;
}

export default defineConfig({
  define: {
    __BG_CPP_CUTE_REAL_COMPILE_INPUTS__: JSON.stringify(browserInputs),
    __BG_CPP_CUTE_REAL_COMPILE_ROUTE_PREFIX__: JSON.stringify(routePrefix),
  },
  plugins: [externalAssetServer()],
  test: {
    include: ["tests-browser/cpp_cute_browser_real_compile.test.ts"],
    testTimeout: 240_000,
    hookTimeout: 240_000,
    browser: {
      enabled: true,
      provider: playwright(),
      headless: process.env.BG_BROWSER_HEADLESS !== "0",
      instances: [{ browser: "chromium" }],
    },
  },
  server: {
    fs: { allow: [repositoryRoot] },
  },
});
