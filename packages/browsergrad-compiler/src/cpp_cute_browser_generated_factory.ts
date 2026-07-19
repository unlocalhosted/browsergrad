import createBrowserGradCppCuteExtractor from "./resources/clang-extractor.mjs";
import type {
  CppCuteBrowserGeneratedEmscriptenFactory,
} from "./cpp_cute_browser_emscripten_factory.js";

export const CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256 =
  "ef6b757b053fe6ce232b7f32bb6d7e747211ad65102940607e8346fd027f63c3";
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH = 23_916;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_ID =
  `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`;
export const CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY = Object.freeze({
  source: "reviewed-cached-diagnostic-build-output",
  packageOwned: true,
  exactSourcePinned: true,
  cleanBuildVerified: false,
  reproducibilityVerified: false,
  workerBundleVerified: false,
  workerExecutionObserved: false,
  releaseReady: false,
} as const);

/** Internal package factory; callers cannot supply or replace this function. */
export const CPP_CUTE_BROWSER_GENERATED_FACTORY:
  CppCuteBrowserGeneratedEmscriptenFactory = createBrowserGradCppCuteExtractor;
