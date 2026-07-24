import createBrowserGradCppCuteExtractor from "./resources/clang-extractor.mjs";
import {
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
} from "./resources/cpp_cute_browser_factory_identity.js";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_SHA256,
} from "./resources/cpp_cute_browser_reproducibility_identity_v1.js";
import type {
  CppCuteBrowserGeneratedEmscriptenFactory,
} from "./cpp_cute_browser_emscripten_factory.js";

export const CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256 =
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH =
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_ID =
  `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY = Object.freeze({
  source: "reviewed-two-clean-build-reproducible-output",
  packageOwned: true,
  exactSourcePinned: true,
  cleanBuildRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
  cleanBuildSourceRevision:
    CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
  reproducibilityVerifierRunId:
    CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
  reproducibilityVerifierSourceRevision:
    CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
  reproducibilityResourceSha256:
    CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256,
  cleanBuildWasmSha256: CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_SHA256,
  cleanBuildWasmByteLength:
    CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_BYTE_LENGTH,
  exactReproducibleFactoryMatch: true,
  exactInterfaceConformance: true,
  cleanBuildVerified: true,
  reproducibilityVerified: true,
  workerBundleVerified: false,
  workerExecutionObserved: false,
  releaseReady: false,
} as const);

/** Internal package factory; callers cannot supply or replace this function. */
export const CPP_CUTE_BROWSER_GENERATED_FACTORY:
  CppCuteBrowserGeneratedEmscriptenFactory = createBrowserGradCppCuteExtractor;
