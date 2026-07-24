import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
} from "./cpp_cute_browser_factory_identity.js";

/**
 * Canonical verifier output for the two clean Clang-Wasm builds produced by
 * workflow run 30069614333. The package separately records the source and
 * verifier workflow identities; this value contains only the path-independent
 * build observation emitted by the v3 verifier.
 */
const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE = {
  authority: "clang-wasm-extractor-reproducibility-observation-only",
  builds: [
    {
      buildExecutionEvidenceByteLength: 33_928,
      buildExecutionEvidenceSha256:
        "75abf8fd81aa2885e32140566da489a3867fe024c85ca313f4261053c2383d78",
      factoryModuleByteLength: CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
      factoryModuleSha256: CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
      linkMapByteLength: 24_635_354,
      linkMapCanonicalByteLength: 24_586_418,
      linkMapCanonicalSha256:
        "059857f0e49d7091e57e5dd3032a7b8f148e1d58e5212d4c720542d0187713b6",
      linkMapSha256:
        "7eaa8f4c1f06ab60e4990ed8c65d9d06bb719263473b733a0f65e113d02744d6",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 1,
      runtimeAbiReviewByteLength: 1_681_553,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "7a00df98b475f15d7bcf0e44a6874315478f91cfd18758f5f1de29bfe7b69e42",
      runtimeClosureObservationByteLength: 16_990,
      runtimeClosureObservationSha256:
        "f0b9ca747d1d287f94a95c39057b7111648215af312a8c79d2f4a930d0582198",
      runtimeClosureSha256:
        "923dac08515f373a0b3ed5ad2da3ad5644d8d11e0211458c1cc2f68fabd776ee",
      wasmByteLength: 31_841_008,
      wasmSha256:
        "19edd5622461b2308e83f10fb90f9f029241a5ba706e4c1741b194cb52a82138",
    },
    {
      buildExecutionEvidenceByteLength: 33_928,
      buildExecutionEvidenceSha256:
        "cd9c8899199799ce4fee44811753cd6706d4cc20cb3c83a70c8aab536885dd42",
      factoryModuleByteLength: CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
      factoryModuleSha256: CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
      linkMapByteLength: 24_635_354,
      linkMapCanonicalByteLength: 24_586_418,
      linkMapCanonicalSha256:
        "059857f0e49d7091e57e5dd3032a7b8f148e1d58e5212d4c720542d0187713b6",
      linkMapSha256:
        "6bd80734b30230e6c9af7412d4a808fd2694442892d4c0b3697fc4fb3e667382",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 2,
      runtimeAbiReviewByteLength: 1_681_553,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "7a00df98b475f15d7bcf0e44a6874315478f91cfd18758f5f1de29bfe7b69e42",
      runtimeClosureObservationByteLength: 16_990,
      runtimeClosureObservationSha256:
        "f0b9ca747d1d287f94a95c39057b7111648215af312a8c79d2f4a930d0582198",
      runtimeClosureSha256:
        "923dac08515f373a0b3ed5ad2da3ad5644d8d11e0211458c1cc2f68fabd776ee",
      wasmByteLength: 31_841_008,
      wasmSha256:
        "19edd5622461b2308e83f10fb90f9f029241a5ba706e4c1741b194cb52a82138",
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
    "bg.cpp.browser-build-input-lock.sha256.5a96def9bac1db052108142dfe4c82e729f4b41f450d459406a4f3c5227daad7",
  schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
  sourceSetSha256:
    "25ec50f4e2e6300978db67692e1fa7a033b0fb77b66fec79cfe933625c61b298",
  version: 3,
} as const satisfies JsonObject;

export type CppCuteBrowserReproducibilityV3Resource =
  typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE;

export const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE = deepFreezeJson(
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE,
) as unknown as CppCuteBrowserReproducibilityV3Resource;
