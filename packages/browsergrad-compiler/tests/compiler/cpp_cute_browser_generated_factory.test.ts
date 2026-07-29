import { expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_GENERATED_FACTORY,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_ID,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
} from "../../src/cpp_cute_browser_generated_factory.js";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_SHA256,
  cppCuteBrowserReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserReproducibility,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../src/cpp_cute_browser_reproducibility.js";

it("pins the exact package-owned generated factory without widening its authority", () => {
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_ID).toBe(
    `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`,
  );
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256)
    .toBe("2eaa4ce31951cd5eff989679fd8d63c4ae74df0293f8f727209a3ce0f681764d");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH).toBe(27_884);
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY.constructor.name).toBe("AsyncFunction");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY).toEqual({
    source: "reviewed-two-clean-build-reproducible-output",
    packageOwned: true,
    exactSourcePinned: true,
    cleanBuildRunId: "30475350224",
    cleanBuildSourceRevision:
      "ce2c1fc833683493a3fe804a0f33a230398866ba",
    reproducibilityVerifierRunId: "30475350224",
    reproducibilityVerifierSourceRevision:
      "ce2c1fc833683493a3fe804a0f33a230398866ba",
    reproducibilityResourceSha256:
      "870662f1a2b62a278a5826817f2dbdb6f5a8fa0af4ea0be87ce1799bf07a9c4f",
    cleanBuildWasmSha256:
      "937ffe559aa5bd57db7eeecd26a590d90344457e6a6e0b06151afb6cfcc3fb6d",
    cleanBuildWasmByteLength: 31_841_971,
    exactReproducibleFactoryMatch: true,
    exactInterfaceConformance: true,
    cleanBuildVerified: true,
    reproducibilityVerified: true,
    workerBundleVerified: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  expect(Object.isFrozen(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY)).toBe(true);
});

it("admits only the exact package-pinned reproducibility evidence", async () => {
  const bytes = cppCuteBrowserReproducibilityResourceBytes();
  expect(bytes.byteLength).toBe(CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH);
  const authority = await verifyCppCuteBrowserReproducibilityResource(bytes);
  expect(authority).toMatchObject({
    schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
    version: 3,
    authority: "package-pinned-extractor-reproducibility-only",
    resourceSha256: CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256,
    resourceByteLength: CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
    buildRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
    buildSourceRevision: CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
    verifierRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
    verifierSourceRevision: CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
    factoryModuleSha256:
      "2eaa4ce31951cd5eff989679fd8d63c4ae74df0293f8f727209a3ce0f681764d",
    factoryModuleByteLength: 27_884,
    wasmSha256:
      "937ffe559aa5bd57db7eeecd26a590d90344457e6a6e0b06151afb6cfcc3fb6d",
    wasmByteLength: 31_841_971,
    extractorOutputsReproducible: true,
    fullDistributedOutputSetReproducible: false,
    producerAttested: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  expect(authority.factoryModuleSha256).toBe(
    CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
  );
  expect(authority.wasmSha256).toBe(
    CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_SHA256,
  );
  expect(authority.wasmByteLength).toBe(
    CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_BYTE_LENGTH,
  );
  expect(() => requireVerifiedCppCuteBrowserReproducibility(authority)).not.toThrow();
  expect(() => requireVerifiedCppCuteBrowserReproducibility({ ...authority })).toThrowError(
    /REPRODUCIBILITY-EVIDENCE-UNVERIFIED/u,
  );

  const mutated = cppCuteBrowserReproducibilityResourceBytes();
  mutated[mutated.byteLength - 1] = (mutated[mutated.byteLength - 1] ?? 0) ^ 1;
  await expect(verifyCppCuteBrowserReproducibilityResource(mutated)).rejects.toMatchObject({
    code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-HASH-MISMATCH",
    path: "$bytes",
  });
  await expect(verifyCppCuteBrowserReproducibilityResource(bytes.subarray(1))).rejects.toMatchObject({
    code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-RESOURCE-LIMIT",
    path: "$bytes.byteLength",
  });
});
