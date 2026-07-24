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
  caseCount: 4,
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
        compileElapsedMilliseconds: 18_981,
        evidenceId:
          "bg.cpp.browser-worker-execution.sha256.ed5e221470d53b73b7acf1671d62cdcce8d8ba2e8a572e16a2986f4b967ecc6b",
        exactInterfaceConformanceObserved: true,
        hostElapsedMicroseconds: "13553901",
        invocationId:
          "bg.cpp.browser-worker-invocation.sha256.8b0db62cd9f83be1fa13dd55054b16f6496cf95bd95c45f602d05511bf3feee9",
        openedHeaderFiles: "1168",
        openedSourceFiles: "1",
        rawWasmVerified: true,
        requestId:
          "bg.cpp.frontend-request.sha256.21399c68fbcb88257ad315009180fd6171d8d938146460c3053d7b2168492018",
        totalElapsedMilliseconds: 21_516,
        verifierEvidenceId:
          "bg.cpp.browser-wasm-verifier-conformance.sha256.2fcc774856238289f8aa45e29ff6d6d87a5c7ee49abecad60d001135fff8910e",
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
          "bg.cpp.browser-worker-view-copy-candidate.sha256.83b5716b26c91c95a69d57d4786ec085bb8f722313cc2082ef1074e64a7050a1",
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
        compileElapsedMilliseconds: 19_716,
        evidenceId:
          "bg.cpp.browser-worker-execution.sha256.6c9478a4eeea1f1aa718a681220177d5241793ecd1ccd267afcefb55a5b643f5",
        exactInterfaceConformanceObserved: true,
        hostElapsedMicroseconds: "14338401",
        invocationId:
          "bg.cpp.browser-worker-invocation.sha256.99a2db8f6d20e68011a8335d2168844f3e0d803f9e29ddd63284891469ff2481",
        openedHeaderFiles: "1168",
        openedSourceFiles: "1",
        rawWasmVerified: true,
        requestId:
          "bg.cpp.frontend-request.sha256.1c10bdc8007e6dc9e01bff9298a8a1a9bb9ec8afe689c97cd9c132840be1771f",
        totalElapsedMilliseconds: 22_148,
        verifierEvidenceId:
          "bg.cpp.browser-wasm-verifier-conformance.sha256.8e08d3b126c9d8bd71ff1f1684cb4de1f8eed5d1fa7c6506987772a8c67c579f",
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
          "bg.cpp.browser-worker-view-copy-candidate.sha256.878cfa9123ca0c0c3a4d8ec6d2ede2efbe4a2783fbcebba21f7d9f0afb056ca8",
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
    {
      authority: "local-real-browser-worker-execution-observation-only",
      backendExecutionAuthorized: false,
      execution: {
        acceptedTerminalMessages: "1",
        artifactHash:
          "a1d2eb268960144184e7710269918da69e2d2235ac8ae3e6a34b889f054abc2b",
        artifactId:
          "bg.artifact.cpp-cute-frontend.sha256.a1d2eb268960144184e7710269918da69e2d2235ac8ae3e6a34b889f054abc2b",
        artifactOutcome: "accepted",
        compileElapsedMilliseconds: 18_835,
        evidenceId:
          "bg.cpp.browser-worker-execution.sha256.f4a843ada7f3b2727743bf00804bc39e9459614f0627e421101ea0b2b3cc33ad",
        exactInterfaceConformanceObserved: true,
        hostElapsedMicroseconds: "13463200",
        invocationId:
          "bg.cpp.browser-worker-invocation.sha256.321f99fe6562badcf8dc5151c84ed54cbfc31cf7c579658b392e4265bd057238",
        openedHeaderFiles: "1168",
        openedSourceFiles: "1",
        rawWasmVerified: true,
        requestId:
          "bg.cpp.frontend-request.sha256.aefd0e4a5b69e7083b522447b2fec0c223c443285cfded0ca2392f9853e5c98c",
        totalElapsedMilliseconds: 21_236,
        verifierEvidenceId:
          "bg.cpp.browser-wasm-verifier-conformance.sha256.469399add0dbb8c7d0634715e597ef35f3992ba76b12eb6855eb0ae16dc01896",
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
          "bg.cpp.browser-worker-view-copy-candidate.sha256.75afd89cdb230f1cc7eb94cbbbf5700e45cf035acc3ff47ff1fbef2d036f50a8",
        destinationCoordinateRank: 2,
        destinationLayoutFactId:
          "bg.cpp.fact.sha256.0b769fc2f8835494f9d310917b0d3e5a18d44edfbb85f66ed8e0b0cc45f290ae",
        destinationSpanElements: "6",
        entryId:
          "bg.cpp.entry.sha256.1374abd70e6f02372f5e8999f38132e1e9a8a90da2e017f0a935453ae6d12b15",
        entrySubjectHash:
          "d96fe21be4cd9b42d6bd6074abafb52e9db96c4c1e90f7981d102d55923ccf8f",
        sharedViewCopySemanticsPrepared: true,
        sourceCoordinateRank: 2,
        sourceLayoutFactId:
          "bg.cpp.fact.sha256.ea98c6439b586baa0fd71c518f3d6d67fe3cb79f368562ce8597724d6673ca92",
        sourceSpanElements: "12",
      },
      source: {
        caseId: "strided-slice",
        selectedDeclaration: "copy_views",
        sourceSha256:
          "55f4f5fcf55093a05cb977e3b83479098f6ddc42b830ec63f44b97f27fe3264a",
        syntax: "unchanged-cpp17-cute",
        virtualPath: "/workspace/src/real-view-copy-strided-slice.cu",
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
          "2f4d178ecb9c9413da8ca670e036995e200d64e98093f3a8cb6ed5d2a1e28a52",
        artifactId:
          "bg.artifact.cpp-cute-frontend.sha256.2f4d178ecb9c9413da8ca670e036995e200d64e98093f3a8cb6ed5d2a1e28a52",
        artifactOutcome: "accepted",
        compileElapsedMilliseconds: 19_192,
        evidenceId:
          "bg.cpp.browser-worker-execution.sha256.7fab79cdcacf5891b373bb64710bb206006e9ca03bcc731679a6aeb510e54db4",
        exactInterfaceConformanceObserved: true,
        hostElapsedMicroseconds: "13743500",
        invocationId:
          "bg.cpp.browser-worker-invocation.sha256.1ba536b9c383c7fc82db94b5eb916826efc9a90ac7c5ed8e337448845e86a2bc",
        openedHeaderFiles: "1168",
        openedSourceFiles: "1",
        rawWasmVerified: true,
        requestId:
          "bg.cpp.frontend-request.sha256.c820b0f33075f71fcf3e735defca3c9c5f35422b49f0adb021a6e38ba1c53208",
        totalElapsedMilliseconds: 21_631,
        verifierEvidenceId:
          "bg.cpp.browser-wasm-verifier-conformance.sha256.b7b3b8c0500ee9eae1cbdcf9fa35935db0849648ca8c7b5e6c598af5c38da048",
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
          "bg.cpp.browser-worker-view-copy-candidate.sha256.259f785995d8d3fa4a68c1214ccb80c0eb367098c853cab83569a1b613eb369b",
        destinationCoordinateRank: 2,
        destinationLayoutFactId:
          "bg.cpp.fact.sha256.d59a39de8ce5b34cc248307e5376f9667d63ade3d3ad442ede4f93c6037eba99",
        destinationSpanElements: "6",
        entryId:
          "bg.cpp.entry.sha256.b1197381642cc6c30788674aa09aa73346b6dc317c49b790925475edea704712",
        entrySubjectHash:
          "3082f5c086b5002c2e6b844813b51ba39a0d46027c0902973ea56fcd92420f89",
        sharedViewCopySemanticsPrepared: true,
        sourceCoordinateRank: 2,
        sourceLayoutFactId:
          "bg.cpp.fact.sha256.bd9f95125c54f7f97f198a57219bd5090bc7982a14f4bc387776b5fb1b7af9c5",
        sourceSpanElements: "2",
      },
      source: {
        caseId: "broadcast",
        selectedDeclaration: "copy_views",
        sourceSha256:
          "bfd91bdaac57ef7314570a8de56f26165a7b263593f319d728c53c13ef7c6376",
        syntax: "unchanged-cpp17-cute",
        virtualPath: "/workspace/src/real-view-copy-broadcast.cu",
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
    unchangedCpp17CuteBroadcastCompiled: true,
    unchangedCpp17CuteRank2Compiled: true,
    unchangedCpp17CuteRank3Compiled: true,
    unchangedCpp17CuteStridedSliceCompiled: true,
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
