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
 * workflow run 30062294974. The package separately records the source and
 * verifier workflow identities; this value contains only the path-independent
 * build observation emitted by the v3 verifier.
 */
const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE = {
  authority: "clang-wasm-extractor-reproducibility-observation-only",
  builds: [
    {
      buildExecutionEvidenceByteLength: 33_457,
      buildExecutionEvidenceSha256:
        "cde07a9f012e210badebe4099be13e3917037eb8ba5e30ec0e188398ea72c158",
      factoryModuleByteLength: CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
      factoryModuleSha256: CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
      linkMapByteLength: 24_636_625,
      linkMapCanonicalByteLength: 24_587_689,
      linkMapCanonicalSha256:
        "78dbe3bb9093fe39bb4cc0d20b7273b35368252c2f50359b06b541a5049205d3",
      linkMapSha256:
        "3d854834ca1da5ed616bdc9c1f026da49bfdab2798a24deed7918b3cfb8aa812",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 1,
      runtimeAbiReviewByteLength: 1_681_605,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "6497d851782bd59b7efab0966e8baee57bcfef93ce3c1dfcdc6709bcdc735234",
      runtimeClosureObservationByteLength: 16_519,
      runtimeClosureObservationSha256:
        "05312e331fe6a4d78195345c854efd88d85ba98d3e6e21c36e10782e88ff0af8",
      runtimeClosureSha256:
        "37dc03f8bb360029133171c4b8c4cbe625f2ba71a5113ece4c91513d1354e33e",
      wasmByteLength: 31_839_835,
      wasmSha256:
        "c5e40d131c4ab004a1b70fb7a0ba56c2f0379afd65da5add47ec4330e5bc6ae8",
    },
    {
      buildExecutionEvidenceByteLength: 33_457,
      buildExecutionEvidenceSha256:
        "82a12463884e757d0c28cd2270bdd6b76e682730ce758a1b7dd5e57a8f7cfb90",
      factoryModuleByteLength: CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
      factoryModuleSha256: CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
      linkMapByteLength: 24_636_625,
      linkMapCanonicalByteLength: 24_587_689,
      linkMapCanonicalSha256:
        "78dbe3bb9093fe39bb4cc0d20b7273b35368252c2f50359b06b541a5049205d3",
      linkMapSha256:
        "2c43bb4b9c623448e08fce97f9da1b16b69d63abea6ae58ea90bb873e910de1a",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 2,
      runtimeAbiReviewByteLength: 1_681_605,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "6497d851782bd59b7efab0966e8baee57bcfef93ce3c1dfcdc6709bcdc735234",
      runtimeClosureObservationByteLength: 16_519,
      runtimeClosureObservationSha256:
        "05312e331fe6a4d78195345c854efd88d85ba98d3e6e21c36e10782e88ff0af8",
      runtimeClosureSha256:
        "37dc03f8bb360029133171c4b8c4cbe625f2ba71a5113ece4c91513d1354e33e",
      wasmByteLength: 31_839_835,
      wasmSha256:
        "c5e40d131c4ab004a1b70fb7a0ba56c2f0379afd65da5add47ec4330e5bc6ae8",
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
    "bg.cpp.browser-build-input-lock.sha256.3e6742cde1fd6fa984d70af6b6b818e901696e666d4490475cd65f00dd53ca9c",
  schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
  sourceSetSha256:
    "00eeade403122cbec5213429df2311d5968db4f55d0d861956e195dff51d422e",
  version: 3,
} as const satisfies JsonObject;

export type CppCuteBrowserReproducibilityV3Resource =
  typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE;

export const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE = deepFreezeJson(
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE,
) as unknown as CppCuteBrowserReproducibilityV3Resource;
