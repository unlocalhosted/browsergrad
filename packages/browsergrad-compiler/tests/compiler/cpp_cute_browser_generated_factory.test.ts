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
  cppCuteBrowserReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserReproducibility,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../src/cpp_cute_browser_reproducibility.js";

it("pins the exact package-owned generated factory without widening its authority", () => {
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_ID).toBe(
    `bg.cpp.browser-emscripten-factory.sha256.${CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256}`,
  );
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256)
    .toBe("796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH).toBe(27_125);
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY.constructor.name).toBe("AsyncFunction");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY).toEqual({
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
    reproducibilityBuildRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
    reproducibilityBuildSourceRevision:
      CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
    reproducibilityVerifierRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
    reproducibilityVerifierSourceRevision:
      CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
    reproducibilityEvidenceSha256:
      CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256,
    reproducibilityEvidenceByteLength:
      CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
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
    factoryModuleSha256: CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
    factoryModuleByteLength: CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
    wasmSha256: CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY.cleanValidationWasmSha256,
    wasmByteLength: CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY.cleanValidationWasmByteLength,
    extractorOutputsReproducible: true,
    fullDistributedOutputSetReproducible: false,
    producerAttested: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
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
