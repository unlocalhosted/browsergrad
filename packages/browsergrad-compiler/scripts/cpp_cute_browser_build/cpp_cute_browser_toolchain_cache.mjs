import { pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

const CACHE_KEY_DOMAIN =
  "browsergrad.compiler.cpp-cute.clang-wasm-toolchain-cache-key.v1";
const CACHE_KEY_PREFIX = "bg.cpp.clang-wasm-toolchain-cache.sha256.";
const COMPATIBLE_LEGACY_EXTRACTOR_CONFIGURATION = Object.freeze({
  path: "CMakeLists.txt",
  sha256: "8e68e5347dac776bb4a83aec73adc4d5c5dd83d7bc69faf88e12085aa41f8ecc",
  byteLength: "4645",
});

/**
 * Selects only inputs that can change the reusable LLVM/Clang build layer.
 * Ordinary extractor implementation files are deliberately excluded; the
 * external project's CMake configuration and final-executable linker flags are
 * also excluded. CMake reapplies both during mandatory exact reconfiguration,
 * the configured-target reviewer checks the resulting flags before compile,
 * and cached BrowserGrad objects are invalidated. The independently selected
 * Clang libraries and recipe compile inputs remain key-bound because they can
 * change reusable static-library objects.
 *
 * The resulting cache is an untrusted diagnostic accelerator. It is never
 * build, reproducibility, provenance, distribution, or release evidence.
 *
 * @param {ReturnType<typeof unwrapPreparedCppCuteBrowserBuildInputLock>["lock"]["body"]} body
 */
export function selectCppCuteBrowserToolchainCacheInputs(body) {
  const llvmSource = body.sources.find((source) => source.sourceId === "llvm-project");
  if (llvmSource === undefined) throw new Error("locked LLVM source selection is missing");
  return Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.clang-wasm-toolchain-cache-inputs",
    version: 1,
    llvmSource: Object.freeze({ ...llvmSource }),
    builder: Object.freeze({
      platform: body.builder.platform,
      platformManifestDigest: body.builder.platformManifestDigest,
      imageConfigDigest: body.builder.imageConfigDigest,
      emsdk: Object.freeze({ ...body.builder.emsdk }),
    }),
    recipe: Object.freeze({
      recipeId: body.recipe.recipeId,
      sourceDateEpoch: body.recipe.sourceDateEpoch,
      environment: body.recipe.environment,
      parallelJobs: body.recipe.parallelJobs,
      prefixMapKinds: body.recipe.prefixMapKinds,
      stages: body.recipe.stages.map(selectReusableStageInputs),
    }),
    selectedClangLibraries: body.recipe.extractorLinkPolicy.selectedClangLibraries,
  });
}

/**
 * @template T
 * @param {T & {readonly linkerFlags?: readonly string[]}} stage
 * @returns {Readonly<Omit<T, "linkerFlags">>}
 */
function selectReusableStageInputs(stage) {
  const selected = /** @type {Omit<T, "linkerFlags">} */ (Object.fromEntries(
    Object.entries(stage).filter(([name]) => name !== "linkerFlags"),
  ));
  return Object.freeze(selected);
}

/** @param {ReturnType<typeof selectCppCuteBrowserToolchainCacheInputs>} inputs */
export async function deriveCppCuteBrowserToolchainCacheKey(inputs) {
  const sha256 = await hashCanonicalJson({
    domain: CACHE_KEY_DOMAIN,
    inputs,
  });
  return `${CACHE_KEY_PREFIX}${sha256}`;
}

export async function projectCppCuteBrowserToolchainCache() {
  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
  const inputs = selectCppCuteBrowserToolchainCacheInputs(body);
  const compatibleLegacyCacheKey = await deriveCppCuteBrowserToolchainCacheKey(
    Object.freeze({
      ...inputs,
      extractorConfiguration: COMPATIBLE_LEGACY_EXTRACTOR_CONFIGURATION,
    }),
  );
  return Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.clang-wasm-toolchain-cache-projection",
    version: 1,
    authority: "untrusted-diagnostic-cache-selection-only",
    cacheKey: await deriveCppCuteBrowserToolchainCacheKey(inputs),
    compatibleLegacyCacheKey,
    inputs,
    claims: Object.freeze({
      cacheContentsTrusted: false,
      cleanBuild: false,
      buildExecuted: false,
      reproducibilityVerified: false,
      releaseReady: false,
    }),
  });
}

async function main() {
  const arguments_ = process.argv.slice(2);
  const report = await projectCppCuteBrowserToolchainCache();
  if (arguments_.length === 1 && arguments_[0] === "--github-output") {
    process.stdout.write(`cache-key=${report.cacheKey}\n`);
    process.stdout.write(`compatible-legacy-cache-key=${report.compatibleLegacyCacheKey}\n`);
    return;
  }
  if (arguments_.length !== 0) {
    throw new Error("expected no arguments or exactly --github-output");
  }
  process.stdout.write(canonicalJsonBytes(report));
  process.stdout.write("\n");
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
