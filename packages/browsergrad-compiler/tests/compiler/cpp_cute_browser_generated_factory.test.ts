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
      "f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534",
    factoryModuleByteLength: 27_285,
    wasmSha256:
      "7950c52270fdac4ea8cae36fbaafbde56cb61720242e10ea5881becf2fe4cfd4",
    wasmByteLength: 31_653_752,
    extractorOutputsReproducible: true,
    fullDistributedOutputSetReproducible: false,
    producerAttested: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  expect(authority.factoryModuleSha256).toBe(
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
