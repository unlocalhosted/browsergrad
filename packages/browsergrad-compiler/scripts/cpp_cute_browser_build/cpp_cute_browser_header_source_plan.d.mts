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
    | "source-templates-requires-generated-resource-headers";
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
  readonly selections: readonly CppCuteBrowserHeaderSourceSelection[];
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
  readonly version: 1;
  readonly planId: string;
  readonly authority: "exact-header-source-selection-policy-only";
  readonly body: Readonly<{
    buildInputLockId: string;
    buildInputLockResourceSha256: string;
    archives: readonly CppCuteBrowserHeaderSourceArchive[];
    includeRoots: readonly CppCuteBrowserHeaderSourceIncludeRoot[];
    unresolvedBlockers: readonly CppCuteBrowserHeaderSourceBlocker[];
  }>;
  readonly totals: Readonly<{
    archiveCount: number;
    archiveByteLength: string;
    includeRootCount: number;
    selectedSubtreeCount: number;
  }>;
  readonly claims: Readonly<{
    exactBuildInputLockBound: true;
    exactArchiveSelectionPinned: true;
    exactSourceSubtreesPinned: true;
    exactHeaderPackLicensePolicyBound: true;
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
