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
  diagnosticBuildRunId: "30037750529",
  diagnosticBuildSourceRevision:
    "6a9def86c81051b38650359b6f0ff1e33376f407",
  diagnosticBuildWasmSha256:
    "318a5063284b368538afbf620906297c0ff7876aca837ea98c6b510baf477775",
  diagnosticBuildWasmByteLength: 31_651_668,
  diagnosticBuildProjectionSha256:
    "167025594d83ee17a932f27b5a0c3f9880869f0f4500437323bce0d2fd7629d0",
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
