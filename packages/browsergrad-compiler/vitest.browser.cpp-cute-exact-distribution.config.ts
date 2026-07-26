import { createReadStream } from "node:fs";
import {
  closeSync,
  constants,
  fstatSync,
  openSync,
} from "node:fs";
import { fileURLToPath } from "node:url";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig, type Plugin } from "vitest/config";

interface ServedInput {
  readonly route: string;
  readonly path: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: number;
}

interface ExternalInputs {
  readonly schema:
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-inputs";
  readonly version: 1;
  readonly authority: "host-preflight-exact-private-distribution-only";
  readonly caseId:
    | "rank2"
    | "rank3"
    | "rank1"
    | "rank4"
    | "strided-slice"
    | "broadcast"
    | "i32-rank2"
    | "u32-broadcast";
  readonly sourceRevision: string;
  readonly controls: Readonly<Record<string, ServedInput>>;
  readonly assets: readonly (ServedInput & {
    readonly assetId: string;
  })[];
  readonly distribution: Readonly<Record<string, string | number>>;
  readonly producer: Readonly<Record<string, string>>;
  readonly claims: Readonly<Record<string, boolean>>;
}

const inputs = parseInputs(
  process.env.BG_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_INPUTS,
);
const served = [
  ...Object.values(inputs.controls),
  ...inputs.assets,
];
const byRoute = new Map(served.map((entry) => [entry.route, entry]));
if (byRoute.size !== served.length) {
  throw new Error("exact distribution inputs contain duplicate HTTP routes");
}
const browserInputs = {
  schema: inputs.schema,
  version: inputs.version,
  authority: inputs.authority,
  caseId: inputs.caseId,
  sourceRevision: inputs.sourceRevision,
  controls: Object.fromEntries(Object.entries(inputs.controls).map(
    ([name, { path: _path, ...control }]) => [name, control],
  )),
  assets: inputs.assets.map(({ path: _path, ...asset }) => asset),
  distribution: inputs.distribution,
  producer: inputs.producer,
  claims: inputs.claims,
};
const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

function exactDistributionServer(): Plugin {
  return {
    name: "browsergrad-cpp-cute-exact-distribution-assets",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const route = request.url?.split("?", 1)[0];
        if (route === undefined || !byRoute.has(route)) {
          next();
          return;
        }
        const input = byRoute.get(route);
        if (request.method !== "GET" || input === undefined ||
            request.headers.range !== undefined) {
          response.statusCode = input === undefined ? 404 : 405;
          response.end();
          return;
        }
        let descriptor: number | undefined;
        try {
          descriptor = openSync(
            input.path,
            constants.O_RDONLY | constants.O_NOFOLLOW,
          );
          const observed = fstatSync(descriptor);
          if (!observed.isFile() || observed.size !== input.byteLength) {
            throw new Error("served input identity changed after preflight");
          }
          response.statusCode = 200;
          response.setHeader("Content-Type", input.mediaType);
          response.setHeader("Content-Length", String(input.byteLength));
          response.setHeader("Cache-Control", "no-store");
          response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
          response.setHeader("X-Content-Type-Options", "nosniff");
          const stream = createReadStream(input.path, {
            fd: descriptor,
            autoClose: true,
          });
          descriptor = undefined;
          stream.once("error", () => response.destroy());
          stream.pipe(response);
        } catch {
          if (descriptor !== undefined) {
            closeSync(descriptor);
          }
          response.statusCode = 409;
          response.end();
        }
      });
    },
  };
}

function parseInputs(value: string | undefined): ExternalInputs {
  if (value === undefined) {
    throw new Error(
      "BG_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_INPUTS is required",
    );
  }
  const parsed = JSON.parse(value) as Partial<ExternalInputs>;
  const cases = new Set([
    "rank2",
    "rank3",
    "rank1",
    "rank4",
    "strided-slice",
    "broadcast",
    "i32-rank2",
    "u32-broadcast",
  ]);
  if (parsed.schema !==
        "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-inputs" ||
      parsed.version !== 1 ||
      parsed.authority !==
        "host-preflight-exact-private-distribution-only" ||
      typeof parsed.caseId !== "string" ||
      !cases.has(parsed.caseId) ||
      typeof parsed.sourceRevision !== "string" ||
      !/^[0-9a-f]{40}$/u.test(parsed.sourceRevision) ||
      typeof parsed.controls !== "object" ||
      parsed.controls === null ||
      Object.keys(parsed.controls).sort().join(",") !==
        "assetManifest,buildInputLock,envelope,producerPolicy,producerTrustStore,profile" ||
      !Array.isArray(parsed.assets) ||
      parsed.assets.length !== 9 ||
      typeof parsed.distribution !== "object" ||
      parsed.distribution === null ||
      typeof parsed.producer !== "object" ||
      parsed.producer === null ||
      typeof parsed.claims !== "object" ||
      parsed.claims === null ||
      parsed.claims.exactPrivateDistributionTreeVerified !== true ||
      parsed.claims.packagePinnedFullDistributionReproducibilityMatched !==
        true ||
      parsed.claims.localEngineeringProducerAuthenticated !== true ||
      parsed.claims.externalProducerTrusted !== false ||
      parsed.claims.licenseReviewComplete !== false ||
      parsed.claims.distributionAuthorized !== false ||
      parsed.claims.workerExecutionObserved !== false ||
      parsed.claims.loweringAuthorityMinted !== false ||
      parsed.claims.backendExecutionObserved !== false ||
      parsed.claims.releaseReady !== false) {
    throw new Error(
      "exact distribution inputs have an invalid authority boundary",
    );
  }
  for (const input of [
    ...Object.values(parsed.controls),
    ...parsed.assets,
  ]) {
    if (!validServedInput(input)) {
      throw new Error("exact distribution inputs contain an invalid file");
    }
  }
  const assetIds = new Set(parsed.assets.map((asset) => asset.assetId));
  if (assetIds.size !== parsed.assets.length ||
      [...assetIds].some((assetId) =>
        typeof assetId !== "string" || assetId.length === 0)) {
    throw new Error("exact distribution inputs contain duplicate assets");
  }
  return parsed as ExternalInputs;
}

function validServedInput(value: unknown): value is ServedInput {
  if (typeof value !== "object" || value === null) return false;
  const input = value as Partial<ServedInput>;
  return typeof input.route === "string" &&
    input.route.startsWith("/") &&
    !input.route.includes("?") &&
    !input.route.includes("#") &&
    typeof input.path === "string" &&
    input.path.startsWith("/") &&
    typeof input.mediaType === "string" &&
    input.mediaType.length > 0 &&
    typeof input.sha256 === "string" &&
    /^[0-9a-f]{64}$/u.test(input.sha256) &&
    typeof input.byteLength === "number" &&
    Number.isSafeInteger(input.byteLength) &&
    input.byteLength > 0;
}

export default defineConfig({
  define: {
    __BG_REQUIRE_WEBGPU__: JSON.stringify(true),
    __BG_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_INPUTS__:
      JSON.stringify(browserInputs),
  },
  plugins: [exactDistributionServer()],
  test: {
    include: [
      "tests-browser/cpp_cute_browser_exact_distribution_convergence.test.ts",
    ],
    testTimeout: 300_000,
    hookTimeout: 300_000,
    browser: {
      enabled: true,
      provider: playwright({
        launchOptions: { args: ["--enable-unsafe-webgpu"] },
      }),
      headless: process.env.BG_BROWSER_HEADLESS !== "0",
      instances: [{ browser: "chromium" }],
    },
  },
  server: {
    fs: { allow: [repositoryRoot] },
  },
});
