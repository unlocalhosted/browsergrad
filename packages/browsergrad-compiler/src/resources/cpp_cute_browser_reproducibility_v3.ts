import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

/**
 * Canonical verifier output for the two clean Clang-Wasm builds produced by
 * workflow run 30047077419. The package separately records the source and
 * verifier workflow identities; this value contains only the path-independent
 * build observation emitted by the v3 verifier.
 */
const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE = {
  authority: "clang-wasm-extractor-reproducibility-observation-only",
  builds: [
    {
      buildExecutionEvidenceByteLength: 32_532,
      buildExecutionEvidenceSha256:
        "0ea352100bc6fd35b2d4a7cd4ed85c5eb22ff179a435fbbf3ee92a2a2e3adb6e",
      factoryModuleByteLength: 27_285,
      factoryModuleSha256:
        "f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534",
      linkMapByteLength: 24_476_842,
      linkMapCanonicalByteLength: 24_427_906,
      linkMapCanonicalSha256:
        "ab2bb08b9d1c2fa1cc864e5493b1c0ccc15a83372c530a42f9a81986c06121c4",
      linkMapSha256:
        "cdb3bd1a89d0c5ffcb235b2b353fbb128c6442100758a6dcdb96d23322f07780",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 1,
      runtimeAbiReviewByteLength: 1_678_675,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "4fc2ac5437059c89f7c15de1f653a13fcb56bb91f5ceae2c4ed6c6fe2f5deef4",
      runtimeClosureObservationByteLength: 15_594,
      runtimeClosureObservationSha256:
        "71e8fffad2eb706067f03679cf914d71615f27d0fc5e30dc15051af8a1395965",
      runtimeClosureSha256:
        "0c3a0903948daa296b3c48bace809d737651ea0090c79b5f15f01570eab37cec",
      wasmByteLength: 31_653_752,
      wasmSha256:
        "7950c52270fdac4ea8cae36fbaafbde56cb61720242e10ea5881becf2fe4cfd4",
    },
    {
      buildExecutionEvidenceByteLength: 32_532,
      buildExecutionEvidenceSha256:
        "66af1562e0661cb11e0ef8cdaea2fa175da35153969d0385955f4eda9a4e129e",
      factoryModuleByteLength: 27_285,
      factoryModuleSha256:
        "f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534",
      linkMapByteLength: 24_476_842,
      linkMapCanonicalByteLength: 24_427_906,
      linkMapCanonicalSha256:
        "ab2bb08b9d1c2fa1cc864e5493b1c0ccc15a83372c530a42f9a81986c06121c4",
      linkMapSha256:
        "424251d0da70f43f436dc63ed57eeb1473794c33fbd0d531a5ed71183b26de50",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 2,
      runtimeAbiReviewByteLength: 1_678_675,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "4fc2ac5437059c89f7c15de1f653a13fcb56bb91f5ceae2c4ed6c6fe2f5deef4",
      runtimeClosureObservationByteLength: 15_594,
      runtimeClosureObservationSha256:
        "71e8fffad2eb706067f03679cf914d71615f27d0fc5e30dc15051af8a1395965",
      runtimeClosureSha256:
        "0c3a0903948daa296b3c48bace809d737651ea0090c79b5f15f01570eab37cec",
      wasmByteLength: 31_653_752,
      wasmSha256:
        "7950c52270fdac4ea8cae36fbaafbde56cb61720242e10ea5881becf2fe4cfd4",
    },
  ],
  claims: {
    abiConformanceVerified: false,
    extractorOutputsReproducible: true,
    fullDistributedOutputSetReproducible: false,
    outputIdentityAuthorized: false,
    producerAttested: false,
    releaseReady: false,
  },
  cleanBuildCount: 2,
  comparison: {
    canonicalCommandsAndEnvironmentMatched: true,
    factoryModuleBytesMatched: true,
    linkMapCanonicalProjectionMatched: true,
    nativeTablegenIdentitiesMatched: true,
    runtimeAbiReviewBytesMatched: true,
    runtimeClosureMatched: true,
    sourceAndBuildPathsDistinct: true,
    wasmBytesMatched: true,
  },
  lockId:
    "bg.cpp.browser-build-input-lock.sha256.489aa5b8657d2b0a4309869dc4c18e2e32f58be03d25a4c7cf1c0c2b981d28a4",
  schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
  sourceSetSha256:
    "6a8ec5cfa519cb3e6b8399662c8e2664921845a0dd6e900a4b0ddcd4e28b3c93",
  version: 3,
} as const satisfies JsonObject;

export type CppCuteBrowserReproducibilityV3Resource =
  typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE;

export const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE = deepFreezeJson(
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE,
) as unknown as CppCuteBrowserReproducibilityV3Resource;
