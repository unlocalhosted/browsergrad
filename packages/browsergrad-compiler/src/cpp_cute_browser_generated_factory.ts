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
  cleanBuildRunId: "30055588624",
  cleanBuildSourceRevision:
    "15f26e5d9191320fbc29216f02dec6042df902aa",
  reproducibilityVerifierRunId: "30057685177",
  reproducibilityVerifierSourceRevision:
    "de6d0f98fc354ed200cb5d5353a76b876e4274fb",
  reproducibilityResourceSha256:
    "f8e7fd51ec5122f40cf03d7ab53d1674f6482f5000cc6b2b81493243dc880ac9",
  cleanBuildWasmSha256:
    "c789fb45a2a849f82b0bce6bfaf3c501722764a0ecc3f0015efaeb2770c3a5cf",
  cleanBuildWasmByteLength: 31_835_141,
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
