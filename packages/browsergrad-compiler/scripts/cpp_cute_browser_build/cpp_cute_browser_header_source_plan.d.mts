export const CPP_CUTE_BROWSER_HEADER_SOURCE_PLAN_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-source-plan";

export class CppCuteBrowserHeaderSourcePlanError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-SOURCE-PLAN";
  readonly path: string;
}

export interface CppCuteBrowserHeaderSourceSelection {
  readonly includeRootId: string;
  readonly archiveSubtree: string;
  readonly virtualPrefix: "";
  readonly intendedAsset: string;
  readonly licenseComponentIds: readonly string[];
  readonly contribution:
    | "complete-selected-header-subtree"
    | "complete-configured-resource-header-output"
    | "complete-configured-libcxx-header-output";
  readonly configuredResourceOutput?: Readonly<{
    upstreamBuildManifest: Readonly<{
      virtualPath: "CMakeLists.txt";
      sha256: string;
      byteLength: string;
    }>;
    buildStageId: "clang-extractor-wasm";
    llvmTargetsToBuild: "WebAssembly";
    clangEnableHlsl: "OFF";
    generatedVirtualPaths: readonly [];
    omittedSourceVirtualPaths: readonly ["CMakeLists.txt"];
  }>;
}

export interface CppCuteBrowserHeaderSourceSupplementalFile {
  readonly supplementalFileId: string;
  readonly includeRootId: string;
  readonly archivePath: string;
  readonly virtualPath: string;
  readonly intendedAsset: string;
  readonly licenseComponentIds: readonly string[];
  readonly contribution: "copy-configured-libcxx-header";
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserHeaderSourceArchive {
  readonly sourceId: string;
  readonly sourceKind:
    | "git-source-archive"
    | "nvidia-cuda-redist-component"
    | "ubuntu-debian-binary-package";
  readonly provider: string;
  readonly version: string;
  readonly repository?: string;
  readonly acquisitionUrl: string;
  readonly archiveFormat: "tar.gz" | "tar.xz" | "deb-data-tar-zstd";
  readonly archiveSha256: string;
  readonly archiveByteLength: string;
  readonly commit?: string;
  readonly treeSha1?: string;
  readonly attestationUrl?: string;
  readonly attestationSha256?: string;
  readonly attestationByteLength?: string;
  readonly index?: Readonly<Record<string, string>>;
  readonly licenseComponentId: string;
  readonly licensePolicy: string;
  readonly licenseEvidence: readonly Readonly<{
    evidenceId: string;
    archivePath: string;
    componentId: string;
    evidenceRole: "upstream-license-text" | "source-package-copyright";
    sha256: string;
    byteLength: string;
  }>[];
  readonly selections: readonly CppCuteBrowserHeaderSourceSelection[];
  readonly supplementalFiles: readonly CppCuteBrowserHeaderSourceSupplementalFile[];
}

export interface CppCuteBrowserHeaderSourceContributor
  extends CppCuteBrowserHeaderSourceSelection {
  readonly sourceId: string;
}

export interface CppCuteBrowserHeaderSourceIncludeRoot {
  readonly includeRootId: string;
  readonly contributors: readonly CppCuteBrowserHeaderSourceContributor[];
  readonly generatedInputsComplete: boolean;
  readonly licenseReviewComplete: false;
  readonly readyForPackAssembly: false;
}

export interface CppCuteBrowserHeaderSourceBlocker {
  readonly blockerId: string;
  readonly requirement: string;
}

export interface CppCuteBrowserHeaderSourcePlan {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_SOURCE_PLAN_SCHEMA;
  readonly version: 4;
  readonly planId: string;
  readonly authority: "exact-header-source-selection-policy-only";
  readonly body: Readonly<{
    buildInputLockId: string;
    buildInputLockResourceSha256: string;
    headerInputProjectionId: string;
    archives: readonly CppCuteBrowserHeaderSourceArchive[];
    includeRoots: readonly CppCuteBrowserHeaderSourceIncludeRoot[];
    unresolvedBlockers: readonly CppCuteBrowserHeaderSourceBlocker[];
  }>;
  readonly totals: Readonly<{
    archiveCount: number;
    archiveByteLength: string;
    includeRootCount: number;
    selectedSubtreeCount: number;
    supplementalFileCount: number;
    supplementalFileByteLength: string;
    licenseEvidenceFileCount: number;
    licenseEvidenceByteLength: string;
  }>;
  readonly claims: Readonly<{
    exactBuildInputLockBound: true;
    exactHeaderInputProjectionBound: true;
    exactArchiveSelectionPinned: true;
    exactSourceSubtreesPinned: true;
    exactHeaderPackLicensePolicyBound: true;
    exactUpstreamLicenseEvidencePinned: true;
    allFiveIncludeRootsSelected: true;
    archiveBytesVerified: false;
    archiveAttestationsVerified: false;
    sourceSubtreesExtracted: false;
    generatedClangResourceHeadersComplete: false;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    headerUniverseComplete: false;
    headerPacksAssembled: false;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export function prepareCppCuteBrowserHeaderSourcePlan():
Promise<Readonly<CppCuteBrowserHeaderSourcePlan>>;

export function requireCppCuteBrowserHeaderSourcePlanAuthority(
  plan: CppCuteBrowserHeaderSourcePlan,
): void;

export function canonicalCppCuteBrowserHeaderSourcePlanBytes(
  plan: CppCuteBrowserHeaderSourcePlan,
): Uint8Array;
