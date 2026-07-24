import createBrowserGradCppCuteExtractor from "./resources/clang-extractor.mjs";
import {
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
} from "./resources/cpp_cute_browser_factory_identity.js";
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
  cleanBuildRunId: "30047077419",
  cleanBuildSourceRevision:
    "c41ab6a84750fde6b3059459dece3df837903ae7",
  reproducibilityVerifierRunId: "30049923259",
  reproducibilityVerifierSourceRevision:
    "2d3cd52b4b5a5ea7f8ebc9fe37851539537547f1",
  reproducibilityResourceSha256:
    "974bcaae92e88522f2a8ed91874c50269fbe0a84ec00823508495e3f034ac047",
  cleanBuildWasmSha256:
    "7950c52270fdac4ea8cae36fbaafbde56cb61720242e10ea5881becf2fe4cfd4",
  cleanBuildWasmByteLength: 31_653_752,
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
