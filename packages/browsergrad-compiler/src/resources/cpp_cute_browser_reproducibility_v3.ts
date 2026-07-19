import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

/**
 * Canonical verifier output for the two clean Clang-Wasm builds produced by
 * workflow run 29683677087. The package separately records the source and
 * verifier workflow identities; this value contains only the path-independent
 * build observation emitted by the v3 verifier.
 */
const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE = {
  authority: "clang-wasm-extractor-reproducibility-observation-only",
  builds: [
    {
      buildExecutionEvidenceByteLength: 32_053,
      buildExecutionEvidenceSha256:
        "5d03605b64655314287e747ca488f8ed480ca991944d205c69d9ea1c08da711a",
      factoryModuleByteLength: 27_125,
      factoryModuleSha256:
        "796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12",
      linkMapByteLength: 24_471_036,
      linkMapCanonicalByteLength: 24_422_100,
      linkMapCanonicalSha256:
        "9ffb3d30fff81d05a607d5b910748f6204886870d7c24db5b6bfba5e1ff46559",
      linkMapSha256:
        "bb76bd467b86d24de683f5c14af19fc7f4336310860453d3cca0efa84b9792be",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 1,
      runtimeAbiReviewByteLength: 1_678_025,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "2da68fceb93f95c840b7974482b03ec6426f2339198b77e449140d8c51198884",
      runtimeClosureObservationByteLength: 15_151,
      runtimeClosureObservationSha256:
        "7b25fc57c4731b65e33daff201592a4939917687fc99b6ba89e1991478e0a779",
      runtimeClosureSha256:
        "7436e64d2ce335aa02a6f3f39017282b93cb0e92538d41c85f32df2ef7d7ea7e",
      wasmByteLength: 31_641_377,
      wasmSha256:
        "5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758",
    },
    {
      buildExecutionEvidenceByteLength: 32_053,
      buildExecutionEvidenceSha256:
        "6cf88d7f979fad913f8f39d02efe6942df485ccffaeb5535b25c2529912d65b8",
      factoryModuleByteLength: 27_125,
      factoryModuleSha256:
        "796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12",
      linkMapByteLength: 24_471_036,
      linkMapCanonicalByteLength: 24_422_100,
      linkMapCanonicalSha256:
        "9ffb3d30fff81d05a607d5b910748f6204886870d7c24db5b6bfba5e1ff46559",
      linkMapSha256:
        "6db88236de0dc370bceaba4867fe6e9a375dfb9f9aae3f6f868a6ec4b3e36744",
      nativeTools: {
        clangTablegenByteLength: 2_571_848,
        clangTablegenSha256:
          "aeb5987e0f3523b62f423ccc7f2bf281e990a944f21e405b175eca1b05b80a0b",
        llvmTablegenByteLength: 5_057_944,
        llvmTablegenSha256:
          "b13c84826686062917c3fbb5bb7165c7a5512c7c35eae4128d07b960f8fba163",
      },
      ordinal: 2,
      runtimeAbiReviewByteLength: 1_678_025,
      runtimeAbiReviewExactInterfaceConformance: true,
      runtimeAbiReviewSha256:
        "2da68fceb93f95c840b7974482b03ec6426f2339198b77e449140d8c51198884",
      runtimeClosureObservationByteLength: 15_151,
      runtimeClosureObservationSha256:
        "7b25fc57c4731b65e33daff201592a4939917687fc99b6ba89e1991478e0a779",
      runtimeClosureSha256:
        "7436e64d2ce335aa02a6f3f39017282b93cb0e92538d41c85f32df2ef7d7ea7e",
      wasmByteLength: 31_641_377,
      wasmSha256:
        "5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758",
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
    "bg.cpp.browser-build-input-lock.sha256.bf62353c9421b955cd1a07e14e13c5e3417b5431e2be4555283acdacc0ee7def",
  schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
  sourceSetSha256:
    "1555fb6e5d48793442fbd767f8225cecaabbf0ede37c7a0ab9db7b4140363d34",
  version: 3,
} as const satisfies JsonObject;

export type CppCuteBrowserReproducibilityV3Resource =
  typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE;

export const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE = deepFreezeJson(
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE,
) as unknown as CppCuteBrowserReproducibilityV3Resource;
