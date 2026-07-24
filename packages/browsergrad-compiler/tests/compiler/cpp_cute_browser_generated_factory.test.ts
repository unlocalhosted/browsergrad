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
      "c789fb45a2a849f82b0bce6bfaf3c501722764a0ecc3f0015efaeb2770c3a5cf",
    wasmByteLength: 31_835_141,
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
