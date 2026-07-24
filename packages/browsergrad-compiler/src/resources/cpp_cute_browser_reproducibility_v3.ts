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
 * workflow run 30055588624. The package separately records the source and
 * verifier workflow identities; this value contains only the path-independent
 * build observation emitted by the v3 verifier.
 */
const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE = {
  authority: "clang-wasm-extractor-reproducibility-observation-only",
  builds: [
    {
      buildExecutionEvidenceByteLength: 33_457,
      buildExecutionEvidenceSha256:
        "4c1dfba9cabd53462bebb610bae479af80dc1316ad644ba301d9ea2fc484d17d",
      factoryModuleByteLength: CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
      factoryModuleSha256: CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
      linkMapByteLength: 24_635_399,
      linkMapCanonicalByteLength: 24_586_463,
      linkMapCanonicalSha256:
        "133f93b24b864dfd241890f939997b98717b103ea2b6214f5caace3aaed3d718",
      linkMapSha256:
        "c41dfded72d2cd23fbf495d6bc7af4bd5d4f5c2b6228ff5efbdfb66e84e32133",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 1,
      runtimeAbiReviewByteLength: 1_681_572,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "0b238c442061e07f864e30d8d021832489855847b028e302ffe1994fda0311c3",
      runtimeClosureObservationByteLength: 16_519,
      runtimeClosureObservationSha256:
        "b3db6b21466c685d74e5dc88b494b100b04fd8c2b9ac61d6e352d01f2df370c6",
      runtimeClosureSha256:
        "b87e19c8d2528156ff340d114ac06cbcc1f36d7678d25204525b9873e5c7ff35",
      wasmByteLength: 31_835_141,
      wasmSha256:
        "c789fb45a2a849f82b0bce6bfaf3c501722764a0ecc3f0015efaeb2770c3a5cf",
    },
    {
      buildExecutionEvidenceByteLength: 33_457,
      buildExecutionEvidenceSha256:
        "8720506325798459b4ff766bbbf876d977482fa7c2dd51d9a4a94977c83a8174",
      factoryModuleByteLength: CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
      factoryModuleSha256: CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
      linkMapByteLength: 24_635_399,
      linkMapCanonicalByteLength: 24_586_463,
      linkMapCanonicalSha256:
        "133f93b24b864dfd241890f939997b98717b103ea2b6214f5caace3aaed3d718",
      linkMapSha256:
        "5840dfea59563e528589579dcf011c72571011d0a461044d8397c817d095906f",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 2,
      runtimeAbiReviewByteLength: 1_681_572,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "0b238c442061e07f864e30d8d021832489855847b028e302ffe1994fda0311c3",
      runtimeClosureObservationByteLength: 16_519,
      runtimeClosureObservationSha256:
        "b3db6b21466c685d74e5dc88b494b100b04fd8c2b9ac61d6e352d01f2df370c6",
      runtimeClosureSha256:
        "b87e19c8d2528156ff340d114ac06cbcc1f36d7678d25204525b9873e5c7ff35",
      wasmByteLength: 31_835_141,
      wasmSha256:
        "c789fb45a2a849f82b0bce6bfaf3c501722764a0ecc3f0015efaeb2770c3a5cf",
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
    "bg.cpp.browser-build-input-lock.sha256.2ba2736bfd84e2243f6f019bbabc0c7b9ce486df714cef9b6b2ba9035ee48b22",
  schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
  sourceSetSha256:
    "4e2c72a7e84299493d7627b52c1ef2aa1fa1137552b33a352aae31dc45086d62",
  version: 3,
} as const satisfies JsonObject;

export type CppCuteBrowserReproducibilityV3Resource =
  typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE;

export const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE = deepFreezeJson(
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE,
) as unknown as CppCuteBrowserReproducibilityV3Resource;
