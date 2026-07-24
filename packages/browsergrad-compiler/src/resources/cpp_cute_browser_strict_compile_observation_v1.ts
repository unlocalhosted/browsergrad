import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

/**
 * Exact strict Chromium matrix produced from the package-pinned two-clean-build
 * extractor and package-pinned header packs. This records execution only;
 * producer trust, license approval, lowering, backend, and release remain
 * separate false authorities.
 */
const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE = {
  authority: "local-real-browser-worker-matrix-observation-only",
  caseCount: 2,
  cases: [
    {
      authority: "local-real-browser-worker-execution-observation-only",
      backendExecutionAuthorized: false,
      execution: {
        acceptedTerminalMessages: "1",
        artifactHash:
          "0fb6011500c18de77c143cd5dc6e4e70caf2d15c6b51f2771913c99ab5d56abc",
        artifactId:
          "bg.artifact.cpp-cute-frontend.sha256.0fb6011500c18de77c143cd5dc6e4e70caf2d15c6b51f2771913c99ab5d56abc",
        artifactOutcome: "accepted",
        compileElapsedMilliseconds: 18_829,
        evidenceId:
          "bg.cpp.browser-worker-execution.sha256.a259346f658386bf36c202545306ec436890b4b0d6059c9bd9f293d17bebef6d",
        exactInterfaceConformanceObserved: true,
        hostElapsedMicroseconds: "13403801",
        invocationId:
          "bg.cpp.browser-worker-invocation.sha256.9cfe9a0f536e0ddb180d7f192a9d9de177bea0e5264fe4c103bcddcc1a815b38",
        openedHeaderFiles: "1168",
        openedSourceFiles: "1",
        rawWasmVerified: true,
        requestId:
          "bg.cpp.frontend-request.sha256.21399c68fbcb88257ad315009180fd6171d8d938146460c3053d7b2168492018",
        totalElapsedMilliseconds: 21_370,
        verifierEvidenceId:
          "bg.cpp.browser-wasm-verifier-conformance.sha256.85c11139590049a0c20d2862f692f18b0e4a61f849c1d0c4466fa5ff7b4cd436",
        verifierWorkerExecutionObserved: true,
        workerExecutionObserved: true,
      },
      headerDistributionLicenseApproved: false,
      inputs: {
        assetSetSha256:
          "5f2903219bcf8b2f412466576f0db3c4ed2cd5182bf2e2815cf3e0391a5fa2c9",
        externalAssetCount: 6,
        headerDistributionOutputVerificationId:
          "bg.cpp.distribution-output-file-verification.sha256.5bc2231523c4537b30dac139a40c515b725815eec34d81cd0af79f759b31a441",
        headerDistributionReproducibilityId:
          "bg.cpp.browser-header-distribution-reproducibility.sha256.4d4c054fd4c93dbdbdef9581eeac52b037af3425e6a1c7eff8acc585abce1e55",
        installedFileCount: 5_788,
        packCount: 5,
        packagePinnedHeaderPacksMatched: true,
        pinnedReproducibleWasmMatched: true,
        totalExternalByteLength: 101_519_835,
        untrustedDiagnosticWasm: false,
        wasmAuthority: "package-pinned-two-clean-build-output",
        wasmSha256:
          "c5e40d131c4ab004a1b70fb7a0ba56c2f0379afd65da5add47ec4330e5bc6ae8",
      },
      loweringAuthorityMinted: false,
      outcome: "compiled",
      producerTrusted: false,
      releaseReady: false,
      schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
      semanticCandidate: {
        candidateId:
          "bg.cpp.browser-worker-view-copy-candidate.sha256.6ed233f0001d3616021468126fa5cdf5be0782dee172325e0c158b17c9a2c4f3",
        destinationCoordinateRank: 2,
        destinationLayoutFactId:
          "bg.cpp.fact.sha256.c31e96594236dc862fc362d719b6f640123f831a804fbdff15833d8dc635c11f",
        destinationSpanElements: "6",
        entryId:
          "bg.cpp.entry.sha256.b8962535f22cf2d7ce707e819ac95e520e9d598cfdbeb0415512f1f7c7b0b3b7",
        entrySubjectHash:
          "21387da2dd028258032a6ef817686325c247dd4afe17c7398b20dcb399851f57",
        sharedViewCopySemanticsPrepared: true,
        sourceCoordinateRank: 2,
        sourceLayoutFactId:
          "bg.cpp.fact.sha256.1b7d5f61b711ebf43bb5b1654b5f1966a0fed7fcea080b554bbe5ef31fc569db",
        sourceSpanElements: "6",
      },
      source: {
        caseId: "rank2",
        selectedDeclaration: "copy_views",
        sourceSha256:
          "4134804a9892ed1f0a2778fae305e957b5a981afccf2a096f1585f3b1d4e6f06",
        syntax: "unchanged-cpp17-cute",
        virtualPath: "/workspace/src/real-view-copy-rank2.cu",
      },
      version: 2,
      workerExecutionObserved: true,
    },
    {
      authority: "local-real-browser-worker-execution-observation-only",
      backendExecutionAuthorized: false,
      execution: {
        acceptedTerminalMessages: "1",
        artifactHash:
          "ec353b8bcc7d7e49edcf7ea9c27d8a748a80e950c5ebd325bbad33440fb5c11c",
        artifactId:
          "bg.artifact.cpp-cute-frontend.sha256.ec353b8bcc7d7e49edcf7ea9c27d8a748a80e950c5ebd325bbad33440fb5c11c",
        artifactOutcome: "accepted",
        compileElapsedMilliseconds: 18_864,
        evidenceId:
          "bg.cpp.browser-worker-execution.sha256.93493ef46b6768cb4c916c71f20794232bc19bcef3b9cb509752e57474584c24",
        exactInterfaceConformanceObserved: true,
        hostElapsedMicroseconds: "13568500",
        invocationId:
          "bg.cpp.browser-worker-invocation.sha256.97413fd7060ff84153e751d5efd756e546c84374e8629af5834c289a9b15d78a",
        openedHeaderFiles: "1168",
        openedSourceFiles: "1",
        rawWasmVerified: true,
        requestId:
          "bg.cpp.frontend-request.sha256.1c10bdc8007e6dc9e01bff9298a8a1a9bb9ec8afe689c97cd9c132840be1771f",
        totalElapsedMilliseconds: 21_277,
        verifierEvidenceId:
          "bg.cpp.browser-wasm-verifier-conformance.sha256.da5e16ed42cba803812735faa03f13f0fb92d9d41642f791d98aeb585965c30e",
        verifierWorkerExecutionObserved: true,
        workerExecutionObserved: true,
      },
      headerDistributionLicenseApproved: false,
      inputs: {
        assetSetSha256:
          "5f2903219bcf8b2f412466576f0db3c4ed2cd5182bf2e2815cf3e0391a5fa2c9",
        externalAssetCount: 6,
        headerDistributionOutputVerificationId:
          "bg.cpp.distribution-output-file-verification.sha256.5bc2231523c4537b30dac139a40c515b725815eec34d81cd0af79f759b31a441",
        headerDistributionReproducibilityId:
          "bg.cpp.browser-header-distribution-reproducibility.sha256.4d4c054fd4c93dbdbdef9581eeac52b037af3425e6a1c7eff8acc585abce1e55",
        installedFileCount: 5_788,
        packCount: 5,
        packagePinnedHeaderPacksMatched: true,
        pinnedReproducibleWasmMatched: true,
        totalExternalByteLength: 101_519_835,
        untrustedDiagnosticWasm: false,
        wasmAuthority: "package-pinned-two-clean-build-output",
        wasmSha256:
          "c5e40d131c4ab004a1b70fb7a0ba56c2f0379afd65da5add47ec4330e5bc6ae8",
      },
      loweringAuthorityMinted: false,
      outcome: "compiled",
      producerTrusted: false,
      releaseReady: false,
      schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
      semanticCandidate: {
        candidateId:
          "bg.cpp.browser-worker-view-copy-candidate.sha256.df19826ea88bf6bff42042243b112b5ccb4c412cd027d0ce1dc2452ed52f51e8",
        destinationCoordinateRank: 3,
        destinationLayoutFactId:
          "bg.cpp.fact.sha256.3fe260f7d90c8a302062f42dd2594ace4f4554bd524cbd494c9de85275975bac",
        destinationSpanElements: "24",
        entryId:
          "bg.cpp.entry.sha256.1cec20aa9cfd5b1ea339b4409dc29febfc411cfe83985a0bb517049dc5be3f3c",
        entrySubjectHash:
          "fea40eacb5526beb1884688b862c95f8670e4c74675f365a27170d90ba5e7066",
        sharedViewCopySemanticsPrepared: true,
        sourceCoordinateRank: 3,
        sourceLayoutFactId:
          "bg.cpp.fact.sha256.fe4eb695a37566b6edd0df898cbaf90a60090185ffc0f1a2ab101e856a3d1a57",
        sourceSpanElements: "24",
      },
      source: {
        caseId: "rank3",
        selectedDeclaration: "copy_views",
        sourceSha256:
          "6a7beae44e88d7fe8749cb5b485dc7d51d30ed285d33314895be461d428550dd",
        syntax: "unchanged-cpp17-cute",
        virtualPath: "/workspace/src/real-view-copy-rank3.cu",
      },
      version: 2,
      workerExecutionObserved: true,
    },
  ],
  claims: {
    backendExecutionAuthorized: false,
    canonicalGate2LayoutFixturesMatched: true,
    headerDistributionLicenseApproved: false,
    loweringAuthorityMinted: false,
    packagePinnedHeaderPacksMatched: true,
    pinnedReproducibleWasmMatched: true,
    producerTrusted: false,
    releaseReady: false,
    unchangedCpp17CuteRank2Compiled: true,
    unchangedCpp17CuteRank3Compiled: true,
    untrustedDiagnosticWasm: false,
    workerExecutionObserved: true,
  },
  schema:
    "browsergrad.compiler.cpp-cute.browser-real-compile-matrix-observation",
  version: 1,
} as const satisfies JsonObject;

export type CppCuteBrowserStrictCompileObservationV1Resource =
  typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE;

export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE =
  deepFreezeJson(
    CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE,
  ) as unknown as CppCuteBrowserStrictCompileObservationV1Resource;
