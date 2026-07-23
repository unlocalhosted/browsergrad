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
    readonly byteLength: "69004028";
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
  verifierSourceRevision: "00e1f91121045be08a045fe6636ec97ab20c109b",
  buildInputLockId:
    "bg.cpp.browser-build-input-lock.sha256.5b6ded62849e8ecaa02fd5d00f27cd55f5bec4ec86ed3c11cf74fef467025542",
  buildInputLockResourceSha256:
    "1bcfbd031400c6527c8ce4a27a319b79839d1713269db39a3583ae3c7e8ccde1",
  pipelineId:
    "bg.cpp.browser-header-pack-pipeline.sha256.45d53638ee008fb36f8f5304431c8f975b6bcbc946823277b79f9303cb1f8153",
  outputVerificationId:
    "bg.cpp.distribution-output-file-verification.sha256.dbd720f0602ab6844becba52f5badfc59d73940d2e2d600204dfec183061f87b",
  reproducibilityId:
    "bg.cpp.browser-header-distribution-reproducibility.sha256.a0bbb1b0ced6201edbd4cac78a148786b2b213e3d1853e5121c8ca1cfe0c6127",
  outputs: [
    {
      outputPath: "assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt",
      sha256: "9c06e791a6229d4d5bf929c6b0b470a9004e505b73dea9598ffb178a0f54d530",
      byteLength: "115316",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
      sha256: "fd7fb977130d1181c5ce0e038472a45e30623b54e0249b167f5f2ed228b51977",
      byteLength: "7704399",
    },
    {
      outputPath: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
      sha256: "f795494ab3d97cbed3e8dab374daeb90574bbd52f0d462b3466bd89e2aa11a77",
      byteLength: "16848942",
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
      sha256: "8e01f50c5a036cb5da142cb8f404f735d1d77c4f7934f609de38a5bd73172c70",
      byteLength: "1203662",
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
    byteLength: "69004028",
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
