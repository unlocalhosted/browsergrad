import type {
  CppCuteBrowserBsdtarToolAdmission,
} from "./cpp_cute_browser_archive_normalization.mjs";
import type {
  CppCuteBrowserHeaderSourceArchiveAdmission,
} from "./cpp_cute_browser_header_source_archive_admission.mjs";
import type {
  CppCuteBrowserHeaderSourceBlocker,
  CppCuteBrowserHeaderSourceSelection,
  CppCuteBrowserHeaderSourceSupplementalFile,
} from "./cpp_cute_browser_header_source_plan.mjs";
import type {
  CppCuteBrowserSourceArchiveInput,
} from "./cpp_cute_browser_source_archive_admission.mjs";

export const CPP_CUTE_BROWSER_HEADER_SOURCE_EXTRACTION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-source-extraction";

export class CppCuteBrowserHeaderSourceExtractionError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-SOURCE-EXTRACTION";
  readonly path: string;
}

export interface CppCuteBrowserExtractedHeaderSourceSelection
  extends CppCuteBrowserHeaderSourceSelection {
  readonly sourceTreeId: string;
  readonly fileCount: number;
  readonly fileContentByteLength: string;
}

export interface CppCuteBrowserExtractedHeaderSourceArchive {
  readonly sourceId: string;
  readonly archiveFormat: "tar.gz" | "tar.xz" | "deb-data-tar-zstd";
  readonly archiveSha256: string;
  readonly archiveByteLength: string;
  readonly licenseComponentId: string;
  readonly licensePolicy: string;
  readonly normalizationId: string;
  readonly selections: readonly CppCuteBrowserExtractedHeaderSourceSelection[];
  readonly supplementalFiles: readonly (CppCuteBrowserHeaderSourceSupplementalFile & Readonly<{
    sourceTreeId: string;
  }>)[];
  readonly licenseEvidence: readonly Readonly<{
    evidenceId: string;
    archivePath: string;
    componentId: string;
    evidenceRole: "upstream-license-text" | "source-package-copyright";
    sha256: string;
    byteLength: string;
    sourceTreeId: string;
  }>[];
}

export interface CppCuteBrowserHeaderSourceExtraction {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_SOURCE_EXTRACTION_SCHEMA;
  readonly version: 4;
  readonly extractionId: string;
  readonly authority: "exact-plan-host-tool-source-materialization-observation-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly headerSourcePlanId: string;
  readonly archiveAdmissionId: string;
  readonly bsdtarTool: Readonly<{
    toolAdmissionId: string;
    executableSha256: string;
    executableByteLength: string;
    observedVersion: string;
    packageToolIdentityPinned: boolean;
    nodeZstdRuntime?: NonNullable<CppCuteBrowserBsdtarToolAdmission["nodeZstdRuntime"]>;
  }>;
  readonly archives: readonly CppCuteBrowserExtractedHeaderSourceArchive[];
  readonly totals: Readonly<{
    archiveCount: number;
    selectedSubtreeCount: number;
    supplementalFileCount: number;
    supplementalFileByteLength: string;
    fileCount: number;
    fileContentByteLength: string;
    licenseEvidenceFileCount: number;
    licenseEvidenceByteLength: string;
  }>;
  readonly unresolvedBlockers: readonly CppCuteBrowserHeaderSourceBlocker[];
  readonly claims: Readonly<{
    exactCurrentHeaderSourcePlanArchiveBytesVerified: true;
    exactBuildInputLockBound: true;
    exactHeaderInputProjectionBound: true;
    exactHeaderSourcePlanBound: true;
    sourceSubtreeMaterializationsObserved: true;
    exactSelectedSourceSubtreesComplete: boolean;
    collisionFreePortableStorageMaterialized: true;
    allFiveIncludeRootsRepresented: true;
    copiedSourceArchivesRemoved: true;
    hostToolImplementationAttested: false;
    hostToolPackageIdentityPinned: boolean;
    nodeZstdDecompressorPackageIdentityPinned: boolean;
    generatedClangResourceHeadersComplete: true;
    exactUpstreamLicenseEvidenceExtracted: true;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    headerUniverseComplete: boolean;
    headerPacksAssembled: false;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserHeaderSourceExtractionRoot {
  readonly sourceId: string;
  readonly includeRootId: string;
  readonly storageRoot: string;
  readonly sourceTreeId: string;
}

export interface CppCuteBrowserHeaderSourceExtractionArguments {
  readonly archives: readonly CppCuteBrowserSourceArchiveInput[];
  readonly bsdtarPath: string;
  readonly outputRoot: string;
}

export function extractCppCuteBrowserHeaderSourcePlan(input: Readonly<{
  archiveAdmission: CppCuteBrowserHeaderSourceArchiveAdmission;
  bsdtarTool: CppCuteBrowserBsdtarToolAdmission;
  outputRoot: string;
}>): Promise<Readonly<CppCuteBrowserHeaderSourceExtraction>>;

export function requireCppCuteBrowserHeaderSourceExtractionAuthority(
  extraction: CppCuteBrowserHeaderSourceExtraction,
): void;

export function canonicalCppCuteBrowserHeaderSourceExtractionBytes(
  extraction: CppCuteBrowserHeaderSourceExtraction,
): Uint8Array;

export function cppCuteBrowserHeaderSourceExtractionRoots(
  extraction: CppCuteBrowserHeaderSourceExtraction,
): readonly Readonly<CppCuteBrowserHeaderSourceExtractionRoot>[];

export function copyCppCuteBrowserExtractedHeaderSourceFile(
  extraction: CppCuteBrowserHeaderSourceExtraction,
  sourceId: string,
  includeRootId: string,
  relativePath: string,
): Promise<Uint8Array>;

export function copyCppCuteBrowserExtractedHeaderLicenseEvidence(
  extraction: CppCuteBrowserHeaderSourceExtraction,
  sourceId: string,
  evidenceId: string,
): Promise<Uint8Array>;

export function copyCppCuteBrowserExtractedHeaderSupplementalFile(
  extraction: CppCuteBrowserHeaderSourceExtraction,
  sourceId: string,
  supplementalFileId: string,
): Promise<Uint8Array>;

export function cppCuteBrowserExtractedHeaderSourceFiles(
  extraction: CppCuteBrowserHeaderSourceExtraction,
  sourceId: string,
  includeRootId: string,
): readonly Readonly<{
  relativePath: string;
  contentSha256: string;
  byteLength: string;
}>[];

export function parseCppCuteBrowserHeaderSourceExtractionArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserHeaderSourceExtractionArguments>;
