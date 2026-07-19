import createBrowserGradCppCuteExtractor from "./resources/clang-extractor.mjs";
import type {
  CppCuteBrowserGeneratedEmscriptenFactory,
} from "./cpp_cute_browser_emscripten_factory.js";

export const CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256 =
  "796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12";
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH = 27_125;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_ID =
  `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY = Object.freeze({
  source: "reviewed-clean-validation-build-output",
  packageOwned: true,
  exactSourcePinned: true,
  cleanValidationRunId: "29681845216",
  cleanValidationSourceRevision: "aca7ee4ea799357b2c0ee3a57f6687e2139e7b7b",
  cleanValidationWasmSha256:
    "5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758",
  cleanValidationWasmByteLength: 31_641_377,
  exactCleanFactoryMatch: true,
  cleanBuildVerified: true,
  reproducibilityBuildRunId: "29683677087",
  reproducibilityBuildSourceRevision:
    "96ad7b16ec57b3634f14c3eaff45da4d959fc117",
  reproducibilityVerifierRunId: "29685632925",
  reproducibilityVerifierSourceRevision:
    "c1a79c0dc487a50787d0a16377d18c910b7b950b",
  reproducibilityEvidenceSha256:
    "6c7aebe1376edf0f9a526b55bacd930e7e9e0fd454a95213d97931117548f31a",
  reproducibilityEvidenceByteLength: 3_470,
  reproducibilityVerified: true,
  workerBundleVerified: false,
  workerExecutionObserved: false,
  releaseReady: false,
} as const);

/** Internal package factory; callers cannot supply or replace this function. */
export const CPP_CUTE_BROWSER_GENERATED_FACTORY:
  CppCuteBrowserGeneratedEmscriptenFactory = createBrowserGradCppCuteExtractor;
