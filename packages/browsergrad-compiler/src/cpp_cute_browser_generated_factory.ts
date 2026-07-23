import createBrowserGradCppCuteExtractor from "./resources/clang-extractor.mjs";
import type {
  CppCuteBrowserGeneratedEmscriptenFactory,
} from "./cpp_cute_browser_emscripten_factory.js";

export const CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256 =
  "f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534";
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH = 27_285;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_ID =
  `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY = Object.freeze({
  source: "reviewed-cached-diagnostic-build-output",
  packageOwned: true,
  exactSourcePinned: true,
  diagnosticBuildRunId: "30036413179",
  diagnosticBuildSourceRevision:
    "8d784a5a6f10194cf10621f690ff522cd050396e",
  diagnosticBuildWasmSha256:
    "9645fc7cfba18132d4cd32f2285c9b8b3ea71ea0015e4071e511240d79668f38",
  diagnosticBuildWasmByteLength: 31_651_665,
  diagnosticBuildProjectionSha256:
    "dadb54ca1239780e4833dafc50b4e6277644d26a7c1277e4b9d7316ffbdcbd46",
  exactDiagnosticFactoryMatch: true,
  exactInterfaceConformance: true,
  cleanBuildVerified: false,
  reproducibilityVerified: false,
  workerBundleVerified: false,
  workerExecutionObserved: false,
  releaseReady: false,
} as const);

/** Internal package factory; callers cannot supply or replace this function. */
export const CPP_CUTE_BROWSER_GENERATED_FACTORY:
  CppCuteBrowserGeneratedEmscriptenFactory = createBrowserGradCppCuteExtractor;
