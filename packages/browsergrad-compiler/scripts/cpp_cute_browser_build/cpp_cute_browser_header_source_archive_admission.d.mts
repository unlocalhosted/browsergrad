import type {
  CppCuteBrowserHeaderSourceArchive,
  CppCuteBrowserHeaderSourceBlocker,
  CppCuteBrowserHeaderSourceSelection,
} from "./cpp_cute_browser_header_source_plan.mjs";
import type {
  CppCuteBrowserSourceArchiveCopyReport,
  CppCuteBrowserSourceArchiveInput,
  CppCuteBrowserSourceArchiveTotals,
} from "./cpp_cute_browser_source_archive_admission.mjs";

export const CPP_CUTE_BROWSER_HEADER_SOURCE_ARCHIVE_ADMISSION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-source-archive-admission";

export class CppCuteBrowserHeaderSourceArchiveAdmissionError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-SOURCE-ARCHIVE-ADMISSION";
  readonly path: string;
}

export interface CppCuteBrowserHeaderSourceArchiveObservation {
  readonly sourceId: string;
  readonly sourceKind: CppCuteBrowserHeaderSourceArchive["sourceKind"];
  readonly provider: string;
  readonly version: string;
  readonly acquisitionUrl: string;
  readonly archiveFormat: CppCuteBrowserHeaderSourceArchive["archiveFormat"];
  readonly archiveSha256: string;
  readonly archiveByteLength: string;
  readonly observedArchiveSha256: string;
  readonly observedArchiveByteLength: string;
  readonly licenseComponentId: string;
  readonly licensePolicy: string;
  readonly selections: readonly CppCuteBrowserHeaderSourceSelection[];
}

export interface CppCuteBrowserHeaderSourceArchiveAdmission {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_SOURCE_ARCHIVE_ADMISSION_SCHEMA;
  readonly version: 1;
  readonly admissionId: string;
  readonly authority: "exact-current-header-source-plan-archive-admission-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerSourcePlanId: string;
  readonly headerSourcePlanSha256: string;
  readonly headerSourcePlanByteLength: number;
  readonly expectedArchiveSetSha256: string;
  readonly archives: readonly CppCuteBrowserHeaderSourceArchiveObservation[];
  readonly totals: Readonly<CppCuteBrowserSourceArchiveTotals>;
  readonly unresolvedBlockers: readonly CppCuteBrowserHeaderSourceBlocker[];
  readonly claims: Readonly<{
    exactCurrentHeaderSourcePlanArchiveBytesVerified: true;
    exactBuildInputLockBound: true;
    exactHeaderSourcePlanBound: true;
    localArchivePathsRetainedOpaquely: true;
    networkAccessed: false;
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

export function admitCppCuteBrowserHeaderSourcePlanArchives(input: Readonly<{
  archives: readonly CppCuteBrowserSourceArchiveInput[];
}>): Promise<Readonly<CppCuteBrowserHeaderSourceArchiveAdmission>>;

export function requireCppCuteBrowserHeaderSourceArchiveAuthority(
  admission: CppCuteBrowserHeaderSourceArchiveAdmission,
): void;

export function canonicalCppCuteBrowserHeaderSourceArchiveAdmissionBytes(
  admission: CppCuteBrowserHeaderSourceArchiveAdmission,
): Uint8Array;

export function copyCppCuteBrowserHeaderSourceArchive(
  admission: CppCuteBrowserHeaderSourceArchiveAdmission,
  sourceId: string,
  outputPath: string,
): Promise<Readonly<CppCuteBrowserSourceArchiveCopyReport>>;

export function parseCppCuteBrowserHeaderSourceArchiveArguments(
  argv: readonly string[],
): Readonly<{ readonly archives: readonly CppCuteBrowserSourceArchiveInput[] }>;
