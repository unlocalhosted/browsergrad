import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

export interface CppCuteBrowserHeaderDistributionReproducibilityOutputV1 extends JsonObject {
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserHeaderDistributionReproducibilityResourceV1
  extends JsonObject {
  readonly schema: "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility";
  readonly version: 1;
  readonly authority: "two-root-exact-header-distribution-reproducibility-observation-only";
  readonly scope: "five-header-packs-license-inventory-and-notice-outputs-only";
  readonly verifierSourceRevision: string;
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly pipelineId: string;
  readonly outputVerificationId: string;
  readonly reproducibilityId: string;
  readonly outputs: readonly CppCuteBrowserHeaderDistributionReproducibilityOutputV1[];
  readonly totals: JsonObject & {
    readonly outputCount: 17;
    readonly byteLength: "71114813";
  };
  readonly claims: JsonObject & {
    readonly twoDistinctPrivateOutputRootsVerified: true;
    readonly exactOutputsRehashedInBothRoots: true;
    readonly exactHeaderDistributionOutputSetReproducible: true;
    readonly fullDistributedOutputSetReproducible: false;
    readonly externalDistributedFileLicenseMapReviewed: false;
    readonly licenseReviewComplete: false;
    readonly distributionAuthorized: false;
    readonly signedProvenanceVerified: false;
    readonly workerExecutionObserved: false;
    readonly releaseReady: false;
  };
}

const VALUE: CppCuteBrowserHeaderDistributionReproducibilityResourceV1 = {
  schema: "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility",
  version: 1,
  authority: "two-root-exact-header-distribution-reproducibility-observation-only",
  scope: "five-header-packs-license-inventory-and-notice-outputs-only",
  verifierSourceRevision: "7476fd4819af82ad5b1283b82083831ab77c5d86",
  buildInputLockId:
    "bg.cpp.browser-build-input-lock.sha256.489aa5b8657d2b0a4309869dc4c18e2e32f58be03d25a4c7cf1c0c2b981d28a4",
  buildInputLockResourceSha256:
    "3a705905182a763a9cee693e270d5c7a737b264aad41cf5ca0ef913f4ad370ed",
  pipelineId:
    "bg.cpp.browser-header-pack-pipeline.sha256.80a29abc734fcf3183c98fbd3bce5c23005a045f06e6837b80231845fdf09b71",
  outputVerificationId:
    "bg.cpp.distribution-output-file-verification.sha256.1cc298cf70ed624df258a14b0eb687c6a0666a14cdd4e5d208674f6c0f7fb3df",
  reproducibilityId:
    "bg.cpp.browser-header-distribution-reproducibility.sha256.43f703672ddbeaf1e6e6d544e3ed50721a2585e947b5d0a1e624293cac80d449",
  outputs: [
    {
      outputPath: "assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt",
      sha256: "372372e03447024a930606c7530785b7cf67c2f595140651882ee04adcd0bc07",
      byteLength: "115316",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
      sha256: "037acb8aaae9a437ed8275ca608dd92c31a142aa8c882b7ac238e80b3343805e",
      byteLength: "7704705",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
      sha256: "1917ba19e65d1e3be9dfe23b80c693ba8de5ce8e44538a7d16715cd61ece2cbd",
      byteLength: "18954596",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
      sha256: "4f1c39b73f2fa7252628a253f7bb5b1411bdfdada872c5ff733b1b9008d89555",
      byteLength: "21403975",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
      sha256: "1f2c5a1e86b04c29b6af33cc3fba0487b9bdcb87affaa44fceb32bec424e7dba",
      byteLength: "12689654",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/license-inventory.json",
      sha256: "2f1a493b0fdd5eb73a46fb7c16714e716709b4ab68f968d9de9f7c9ec3c63740",
      byteLength: "1208487",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
      sha256: "d04a460dc605703b8e8a104cc5c043e6a7020ca7201991470c615273d43e7ae4",
      byteLength: "8927070",
    },
    {
      outputPath: "licenses/browsergrad-compiler.LICENSE",
      sha256: "84996b33d49ed1a25655a12449c504089e2ac164f3f27a2c893711010bef1620",
      byteLength: "1070",
    },
    {
      outputPath: "licenses/clang.LICENSE.txt",
      sha256: "ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47",
      byteLength: "15140",
    },
    {
      outputPath: "licenses/compiler-rt.LICENSE.txt",
      sha256: "1a8f1058753f1ba890de984e48f0242a3a5c29a6a8f2ed9fd813f36985387e8d",
      byteLength: "16708",
    },
    {
      outputPath: "licenses/cutlass.LICENSE.txt",
      sha256: "42fec630f410aa308f70a51a89fadcd19586fa620f9831a32bee528a9a10000e",
      byteLength: "1547",
    },
    {
      outputPath: "licenses/emscripten-musl.COPYRIGHT",
      sha256: "b870108ec5e7790e9f9919064f1b9421d62d5f9b0e6c230c6adf7ea2da62e97b",
      byteLength: "6196",
    },
    {
      outputPath: "licenses/emscripten.LICENSE",
      sha256: "620a78084fc7ca97c0b5dea9abf891f3ffcadfdbf305276f099c9c4e12fc1d86",
      byteLength: "5093",
    },
    {
      outputPath: "licenses/libcxx.LICENSE.txt",
      sha256: "539dd7aed86e8a4f12cbdd0e6c50c189c7d74847e4fecc64ce2c6ee3a01da38b",
      byteLength: "16703",
    },
    {
      outputPath: "licenses/libcxxabi.LICENSE.txt",
      sha256: "e2b35be49f7284a45b7baca8fc7b3ab7440e7902392b2528a457816b5bb2a15c",
      byteLength: "16706",
    },
    {
      outputPath: "licenses/libunwind.LICENSE.txt",
      sha256: "b5efebcaca80879234098e52d1725e6d9eb8fb96a19fce625d39184b705f7b6d",
      byteLength: "16706",
    },
    {
      outputPath: "licenses/llvm.LICENSE.txt",
      sha256: "8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee",
      byteLength: "15141",
    },
  ],
  totals: {
    outputCount: 17,
    byteLength: "71114813",
  },
  claims: {
    twoDistinctPrivateOutputRootsVerified: true,
    exactOutputsRehashedInBothRoots: true,
    exactHeaderDistributionOutputSetReproducible: true,
    fullDistributedOutputSetReproducible: false,
    externalDistributedFileLicenseMapReviewed: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    signedProvenanceVerified: false,
    workerExecutionObserved: false,
    releaseReady: false,
  },
};

export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE =
  deepFreezeJson(VALUE) as CppCuteBrowserHeaderDistributionReproducibilityResourceV1;
