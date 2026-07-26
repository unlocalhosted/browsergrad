import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

export interface CppCuteBrowserFullDistributionReproducibilityOutputV1
  extends JsonObject {
  readonly outputPath: string;
  readonly role: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserFullDistributionReproducibilityResourceV1
  extends JsonObject {
  readonly schema:
    "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility-observation";
  readonly version: 1;
  readonly authority:
    "two-root-exact-full-distribution-reproducibility-observation-only";
  readonly verifierSourceRevision: string;
  readonly materializerSourceRevision: string;
  readonly producerPolicyScope:
    "local-engineering-reproducibility-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly reproducibilityId: string;
  readonly deterministicMetadata: JsonObject & {
    readonly metadataId: string;
    readonly profileId: string;
    readonly profileHash: string;
    readonly compilationContractHash: string;
    readonly profileSha256: string;
    readonly profileByteLength: string;
    readonly assetManifestId: string;
    readonly assetManifestSha256: string;
    readonly assetManifestByteLength: string;
    readonly assetSetSha256: string;
    readonly buildSubjectId: string;
    readonly buildSubjectSha256: string;
    readonly wasmSha256: string;
    readonly wasmByteLength: string;
    readonly workerBundleSha256: string;
    readonly headerDistributionReproducibilityId: string;
    readonly headerDistributionOutputVerificationId: string;
  };
  readonly deterministicOutputs:
    readonly CppCuteBrowserFullDistributionReproducibilityOutputV1[];
  readonly detachedEvidence: JsonObject & {
    readonly outputPath:
      "assets/browsergrad-cpp-cute/build-provenance.dsse.json";
    readonly role: "detached-build-provenance";
    readonly mediaType: "application/vnd.dsse.envelope.v1+json";
    readonly firstSha256: string;
    readonly firstByteLength: string;
    readonly secondSha256: string;
    readonly secondByteLength: string;
    readonly buildSubjectId: string;
    readonly buildSubjectSha256: string;
  };
  readonly totals: JsonObject & {
    readonly outputCount: 25;
    readonly deterministicSubjectCount: 24;
    readonly detachedEvidenceCount: 1;
    readonly firstByteLength: "103637695";
    readonly secondByteLength: "103637695";
  };
  readonly claims: JsonObject & {
    readonly twoDistinctPrivateOutputRootsVerified: true;
    readonly exactBuildLockOutputPlanMatched: true;
    readonly exactOutputsRehashedInBothRoots: true;
    readonly deterministicSubjectsByteIdentical: true;
    readonly detachedEvidenceBuildSubjectMatched: true;
    readonly fullDistributedOutputSetReproducible: true;
    readonly detachedSignatureVerified: false;
    readonly externallyRootedProducerTrusted: false;
    readonly licenseReviewComplete: false;
    readonly distributionAuthorized: false;
    readonly workerExecutionObserved: false;
    readonly loweringAuthorityMinted: false;
    readonly backendExecutionObserved: false;
    readonly releaseReady: false;
  };
}

const VALUE: CppCuteBrowserFullDistributionReproducibilityResourceV1 = {
  schema:
    "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility-observation",
  version: 1,
  authority:
    "two-root-exact-full-distribution-reproducibility-observation-only",
  verifierSourceRevision:
    "8d7f27eb9a249d8277def3b401377c42e961b6c7",
  materializerSourceRevision:
    "8d7f27eb9a249d8277def3b401377c42e961b6c7",
  producerPolicyScope: "local-engineering-reproducibility-only",
  buildInputLockId:
    "bg.cpp.browser-build-input-lock.sha256.fa21cfe45dec6b4869662cd613a7a300848657518f375c04f7f2193f3a874ad4",
  buildInputLockResourceSha256:
    "fd0f4f978399c6e52ebdb0489f35ce6b0a88e289dce8cfdfa112e52d6217cf3c",
  reproducibilityId:
    "bg.cpp.browser-full-distribution-reproducibility.sha256.64cc7401523b6026aba9430e2f081d708bc62cfcbe6fc343bf58ec0798aeec7b",
  deterministicMetadata: {
    metadataId:
      "bg.cpp.browser-distribution-metadata.sha256.79ebaa98ce989d822aad21c925bd062e1850d7877f72e4c904719fa5a0663cfd",
    profileId: "browsergrad.compiler.cpp-cute.browser-clang@1",
    profileHash:
      "4f4b7416ec509ea97b612cc5b6c6c01596624ef63b8badc4f2a21ffd6b2e1003",
    compilationContractHash:
      "bc909fc7331ccac6c781ed6c11363ce58b68b1d5cf1bc04f9f7275bdd68cc965",
    profileSha256:
      "16d47a72abe1851ce51810898cfd7d4223eae8e114c1f7ea300858486b30c6a8",
    profileByteLength: "7148",
    assetManifestId:
      "bg.cpp.browser-assets.sha256.9db5c28897a9d9fd512056a767ade5446e0c188e3dc3f12946929f6d59d01c25",
    assetManifestSha256:
      "40085018f1266909688a6aadead0e1d0dbeea60274bf4113d966cddebcbca10c",
    assetManifestByteLength: "11501",
    assetSetSha256:
      "2a3a9256bcc1501ee8fc1bddde10b51a5e7eb59048c5f34e5285cba6383510bd",
    buildSubjectId:
      "bg.cpp.browser-build-subject.sha256.ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
    buildSubjectSha256:
      "ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
    wasmSha256:
      "19edd5622461b2308e83f10fb90f9f029241a5ba706e4c1741b194cb52a82138",
    wasmByteLength: "31841008",
    workerBundleSha256:
      "9c9591e725fca512d10a366bdec38b0067366f3d8ebdef50c29a5ebb0134def5",
    headerDistributionReproducibilityId:
      "bg.cpp.browser-header-distribution-reproducibility.sha256.4d4c054fd4c93dbdbdef9581eeac52b037af3425e6a1c7eff8acc585abce1e55",
    headerDistributionOutputVerificationId:
      "bg.cpp.distribution-output-file-verification.sha256.5bc2231523c4537b30dac139a40c515b725815eec34d81cd0af79f759b31a441",
  },
  deterministicOutputs: [
    {
      outputPath: "assets/browsergrad-cpp-cute/asset-manifest.json",
      role: "asset-manifest",
      mediaType: "application/json",
      sha256:
        "40085018f1266909688a6aadead0e1d0dbeea60274bf4113d966cddebcbca10c",
      byteLength: "11501",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/build-input-lock.json",
      role: "build-input-lock",
      mediaType: "application/json",
      sha256:
        "fd0f4f978399c6e52ebdb0489f35ce6b0a88e289dce8cfdfa112e52d6217cf3c",
      byteLength: "26662",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/clang-extractor.wasm",
      role: "clang-extractor",
      mediaType: "application/wasm",
      sha256:
        "19edd5622461b2308e83f10fb90f9f029241a5ba706e4c1741b194cb52a82138",
      byteLength: "31841008",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
      role: "clang-resource-header-vfs",
      mediaType: "application/octet-stream",
      sha256:
        "037acb8aaae9a437ed8275ca608dd92c31a142aa8c882b7ac238e80b3343805e",
      byteLength: "7704705",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/cpp-cute-browser-worker.mjs",
      role: "browser-worker-module",
      mediaType: "text/javascript",
      sha256:
        "9c9591e725fca512d10a366bdec38b0067366f3d8ebdef50c29a5ebb0134def5",
      byteLength: "584894",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
      role: "cuda-header-vfs",
      mediaType: "application/octet-stream",
      sha256:
        "1917ba19e65d1e3be9dfe23b80c693ba8de5ce8e44538a7d16715cd61ece2cbd",
      byteLength: "18954596",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
      role: "cutlass-header-vfs",
      mediaType: "application/octet-stream",
      sha256:
        "4f1c39b73f2fa7252628a253f7bb5b1411bdfdada872c5ff733b1b9008d89555",
      byteLength: "21403975",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/diagnostic-normalization.json",
      role: "diagnostic-normalization-manifest",
      mediaType:
        "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      sha256:
        "6de153792fb09711a9f71ee470433f58ce4183a09fe4e8eb987c9d9bf6a46997",
      byteLength: "8799",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
      role: "libcxx-header-vfs",
      mediaType: "application/octet-stream",
      sha256:
        "1f2c5a1e86b04c29b6af33cc3fba0487b9bdcb87affaa44fceb32bec424e7dba",
      byteLength: "12689654",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/license-inventory.json",
      role: "license-inventory",
      mediaType: "application/json",
      sha256:
        "13ace4a0c861dfec04feb42a3526fa3fe491334f2ec9096e01df3c1fc1cdbd20",
      byteLength: "1208403",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
      role: "linux-sysroot-header-vfs",
      mediaType: "application/octet-stream",
      sha256:
        "d04a460dc605703b8e8a104cc5c043e6a7020ca7201991470c615273d43e7ae4",
      byteLength: "8927070",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
      role: "runtime-abi-manifest",
      mediaType:
        "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      sha256:
        "7370bace059b3790f868a9802197a6b2baed6a9b99c76b4f4f09ecb72b14591a",
      byteLength: "44478",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/semantic-adapter-manifest.json",
      role: "semantic-adapter-manifest",
      mediaType:
        "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      sha256:
        "e5aa795c4feebd523ed72b95be03b102d497f2e0313ee9c99fadf1309cde6150",
      byteLength: "2575",
    },
    {
      outputPath:
        "assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt",
      role: "third-party-notices",
      mediaType: "text/plain",
      sha256:
        "5086675598ef771d8909ddb599786b4307534152af379463777b538546440472",
      byteLength: "115330",
    },
    {
      outputPath: "licenses/browsergrad-compiler.LICENSE",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "84996b33d49ed1a25655a12449c504089e2ac164f3f27a2c893711010bef1620",
      byteLength: "1070",
    },
    {
      outputPath: "licenses/clang.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47",
      byteLength: "15140",
    },
    {
      outputPath: "licenses/compiler-rt.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "1a8f1058753f1ba890de984e48f0242a3a5c29a6a8f2ed9fd813f36985387e8d",
      byteLength: "16708",
    },
    {
      outputPath: "licenses/cutlass.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "42fec630f410aa308f70a51a89fadcd19586fa620f9831a32bee528a9a10000e",
      byteLength: "1547",
    },
    {
      outputPath: "licenses/emscripten.LICENSE",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "620a78084fc7ca97c0b5dea9abf891f3ffcadfdbf305276f099c9c4e12fc1d86",
      byteLength: "5093",
    },
    {
      outputPath: "licenses/emscripten-musl.COPYRIGHT",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "b870108ec5e7790e9f9919064f1b9421d62d5f9b0e6c230c6adf7ea2da62e97b",
      byteLength: "6196",
    },
    {
      outputPath: "licenses/libcxx.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "539dd7aed86e8a4f12cbdd0e6c50c189c7d74847e4fecc64ce2c6ee3a01da38b",
      byteLength: "16703",
    },
    {
      outputPath: "licenses/libcxxabi.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "e2b35be49f7284a45b7baca8fc7b3ab7440e7902392b2528a457816b5bb2a15c",
      byteLength: "16706",
    },
    {
      outputPath: "licenses/libunwind.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "b5efebcaca80879234098e52d1725e6d9eb8fb96a19fce625d39184b705f7b6d",
      byteLength: "16706",
    },
    {
      outputPath: "licenses/llvm.LICENSE.txt",
      role: "component-license",
      mediaType: "text/plain",
      sha256:
        "8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee",
      byteLength: "15141",
    },
  ],
  detachedEvidence: {
    outputPath:
      "assets/browsergrad-cpp-cute/build-provenance.dsse.json",
    role: "detached-build-provenance",
    mediaType: "application/vnd.dsse.envelope.v1+json",
    firstSha256:
      "e90daa7c78c4ba6870045ddb40f53a148ef1ec5416b3084ca8056a80a18ad378",
    firstByteLength: "3035",
    secondSha256:
      "e90daa7c78c4ba6870045ddb40f53a148ef1ec5416b3084ca8056a80a18ad378",
    secondByteLength: "3035",
    buildSubjectId:
      "bg.cpp.browser-build-subject.sha256.ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
    buildSubjectSha256:
      "ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
  },
  totals: {
    outputCount: 25,
    deterministicSubjectCount: 24,
    detachedEvidenceCount: 1,
    firstByteLength: "103637695",
    secondByteLength: "103637695",
  },
  claims: {
    twoDistinctPrivateOutputRootsVerified: true,
    exactBuildLockOutputPlanMatched: true,
    exactOutputsRehashedInBothRoots: true,
    deterministicSubjectsByteIdentical: true,
    detachedEvidenceBuildSubjectMatched: true,
    fullDistributedOutputSetReproducible: true,
    detachedSignatureVerified: false,
    externallyRootedProducerTrusted: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
    backendExecutionObserved: false,
    releaseReady: false,
  },
};

export const
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE =
    deepFreezeJson(VALUE) as
      CppCuteBrowserFullDistributionReproducibilityResourceV1;
