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
    .toBe("f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH).toBe(27_285);
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY.constructor.name).toBe("AsyncFunction");
  expect(CPP_CUTE_BROWSER_GENERATED_FACTORY_AUTHORITY).toEqual({
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
      "796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12",
    factoryModuleByteLength: 27_125,
    wasmSha256:
      "5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758",
    wasmByteLength: 31_641_377,
    extractorOutputsReproducible: true,
    fullDistributedOutputSetReproducible: false,
    producerAttested: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  expect(authority.factoryModuleSha256).not.toBe(
    CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
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
