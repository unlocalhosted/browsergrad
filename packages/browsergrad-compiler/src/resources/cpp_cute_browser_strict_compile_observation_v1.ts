import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

/**
 * Exact strict Chromium observation produced after package admission of the
 * current two-clean-build Clang-Wasm and the package-pinned header packs.
 *
 * This is execution evidence only. Its own claims deliberately retain false
 * producer-trust, license, lowering, backend, distribution, and release state.
 */
const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE = {
  schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
  version: 1,
  outcome: "compiled",
  authority: "local-real-browser-worker-execution-observation-only",
  source: {
    virtualPath: "/workspace/src/real-layout.cu",
    sourceSha256:
      "6bfc03003baeaf1c5a8386ab6de9c818b8f27ca36d75716ce38576df0e6caeb1",
    syntax: "unchanged-cpp17-cute",
    selectedDeclaration: "layout",
  },
  inputs: {
    externalAssetCount: 6,
    totalExternalByteLength: 101_333_752,
    wasmSha256:
      "7950c52270fdac4ea8cae36fbaafbde56cb61720242e10ea5881becf2fe4cfd4",
    wasmAuthority: "package-pinned-two-clean-build-output",
    pinnedReproducibleWasmMatched: true,
    untrustedDiagnosticWasm: false,
    headerDistributionReproducibilityId:
      "bg.cpp.browser-header-distribution-reproducibility.sha256.43f703672ddbeaf1e6e6d544e3ed50721a2585e947b5d0a1e624293cac80d449",
    headerDistributionOutputVerificationId:
      "bg.cpp.distribution-output-file-verification.sha256.1cc298cf70ed624df258a14b0eb687c6a0666a14cdd4e5d208674f6c0f7fb3df",
    packagePinnedHeaderPacksMatched: true,
    assetSetSha256:
      "68e2b6a0233f4e2eb5bc87270820c1cbab2040502768f8cd2f8fe04160daef5e",
    packCount: 5,
    installedFileCount: 5_788,
  },
  execution: {
    evidenceId:
      "bg.cpp.browser-worker-execution.sha256.fbff539a3f5a3ad532e21d24ab07665f1f7b5b434b8aec74d57b7ba5e3b69019",
    invocationId:
      "bg.cpp.browser-worker-invocation.sha256.4fbd447626617802e135fbbd1dc175da9c89c0a912d544181167b38e9875188b",
    requestId:
      "bg.cpp.frontend-request.sha256.79918d9df7be2b11bf6cba59c10668640f138192c5c363255b6ca2ee9ec0410f",
    artifactId:
      "bg.artifact.cpp-cute-frontend.sha256.4489656ea0da6faef2a37164fd73e36e201e15f4fba640fa88395a46deb81991",
    artifactHash:
      "4489656ea0da6faef2a37164fd73e36e201e15f4fba640fa88395a46deb81991",
    artifactOutcome: "accepted",
    acceptedTerminalMessages: "1",
    hostElapsedMicroseconds: "14577901",
    compileElapsedMilliseconds: 21_133,
    totalElapsedMilliseconds: 24_331,
    openedSourceFiles: "1",
    openedHeaderFiles: "1104",
    verifierEvidenceId:
      "bg.cpp.browser-wasm-verifier-conformance.sha256.87de3d5482e4a57028e34128f1fe1d41a3b953b3c2f9d29d6e2a96cf0d133929",
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    verifierWorkerExecutionObserved: true,
    workerExecutionObserved: true,
  },
  semanticCandidate: {
    candidateId:
      "bg.cpp.browser-worker-layout-candidate.sha256.72f3b5933de96569359767f19fdaaed3eaadadc56ca5c297238abcc149c0d34d",
    entryId:
      "bg.cpp.entry.sha256.17b80d466abf3ce9fdfffcea29e65c2263c9d59e3a421d08d0945b77396202c0",
    layoutSemanticHash:
      "9c4ad2f7a3f05e21511c98873155d689ae2e6f253ef29ab1022945f0e2198be0",
    indexMapId:
      "bg.entity.index-map.scope-sha256.78b57b6f3d4aaee90d854b41b0764431f000e338c2e22432501019e9dcde4f09.ordinal.0",
    coordinateRank: 2,
    sharedLayoutSemanticsPrepared: true,
  },
  headerDistributionLicenseApproved: false,
  producerTrusted: false,
  loweringAuthorityMinted: false,
  backendExecutionAuthorized: false,
  releaseReady: false,
  workerExecutionObserved: true,
} as const satisfies JsonObject;

export type CppCuteBrowserStrictCompileObservationV1Resource =
  typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE;

export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE =
  deepFreezeJson(
    CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE,
  ) as unknown as CppCuteBrowserStrictCompileObservationV1Resource;
