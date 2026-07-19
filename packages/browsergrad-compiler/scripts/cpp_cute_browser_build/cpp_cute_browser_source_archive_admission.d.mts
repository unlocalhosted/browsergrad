export const CPP_CUTE_BROWSER_SOURCE_ARCHIVE_INSPECTION_SCHEMA:
"browsergrad.compiler.cpp-cute.source-archive-inspection";
export const CPP_CUTE_BROWSER_REGULAR_ARCHIVE_INSPECTION_SCHEMA:
"browsergrad.compiler.cpp-cute.regular-archive-inspection";
export const CPP_CUTE_BROWSER_SOURCE_ARCHIVE_ADMISSION_SCHEMA:
"browsergrad.compiler.cpp-cute.current-source-archive-admission";

export class CppCuteBrowserSourceArchiveAdmissionError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-SOURCE-ARCHIVE-ADMISSION";
  readonly path: string;
}

export interface CppCuteBrowserSourceArchiveExpectation {
  readonly sourceId: string;
  readonly repository: string;
  readonly acquisitionUrl: string;
  readonly tag: string;
  readonly commit: string;
  readonly treeSha1: string;
  readonly archiveSha256: string;
  readonly archiveByteLength: string;
  readonly attestationUrl?: string;
  readonly attestationSha256?: string;
  readonly attestationByteLength?: string;
}

export interface CppCuteBrowserSourceArchiveInput {
  readonly sourceId: string;
  readonly archivePath: string;
}

export interface CppCuteBrowserRegularArchiveExpectation {
  readonly sourceId: string;
  readonly archiveSha256: string;
  readonly archiveByteLength: string;
}

export interface CppCuteBrowserRegularArchiveObservation
  extends CppCuteBrowserRegularArchiveExpectation {
  readonly observedArchiveSha256: string;
  readonly observedArchiveByteLength: string;
}

export interface CppCuteBrowserRegularArchiveInspection {
  readonly schema: typeof CPP_CUTE_BROWSER_REGULAR_ARCHIVE_INSPECTION_SCHEMA;
  readonly version: 1;
  readonly inspectionId: string;
  readonly authority: "caller-supplied-regular-archive-byte-expectations-only";
  readonly expectedArchiveSetSha256: string;
  readonly archives: readonly CppCuteBrowserRegularArchiveObservation[];
  readonly totals: Readonly<CppCuteBrowserSourceArchiveTotals>;
  readonly claims: Readonly<{
    exactCallerExpectedArchiveBytesVerified: true;
    packagePolicyBound: false;
    networkAccessed: false;
    archivesExtracted: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserSourceArchiveObservation
  extends CppCuteBrowserSourceArchiveExpectation {
  readonly observedArchiveSha256: string;
  readonly observedArchiveByteLength: string;
}

export interface CppCuteBrowserSourceArchiveTotals {
  readonly archiveCount: number;
  readonly archiveByteLength: string;
}

export interface CppCuteBrowserSourceArchiveInspection {
  readonly schema: typeof CPP_CUTE_BROWSER_SOURCE_ARCHIVE_INSPECTION_SCHEMA;
  readonly version: 1;
  readonly inspectionId: string;
  readonly authority: "caller-supplied-archive-expectations-only";
  readonly expectedSourceSetSha256: string;
  readonly archives: readonly CppCuteBrowserSourceArchiveObservation[];
  readonly totals: Readonly<CppCuteBrowserSourceArchiveTotals>;
  readonly claims: Readonly<{
    exactCallerExpectedArchiveBytesVerified: true;
    currentBuildInputLockBound: false;
    networkAccessed: false;
    archivesExtracted: false;
    sourceTreesVerified: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserCurrentSourceArchiveAdmission {
  readonly schema: typeof CPP_CUTE_BROWSER_SOURCE_ARCHIVE_ADMISSION_SCHEMA;
  readonly version: 1;
  readonly admissionId: string;
  readonly authority: "current-build-lock-local-source-archive-admission-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly expectedSourceSetSha256: string;
  readonly archives: readonly CppCuteBrowserSourceArchiveObservation[];
  readonly totals: Readonly<CppCuteBrowserSourceArchiveTotals>;
  readonly claims: Readonly<{
    exactCurrentBuildInputLockArchiveBytesVerified: true;
    currentBuildInputLockBound: true;
    localArchivePathsRetainedOpaquely: true;
    networkAccessed: false;
    archivesExtracted: false;
    sourceTreesVerified: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserCurrentSourceArchiveExpectations {
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly sources: readonly CppCuteBrowserSourceArchiveExpectation[];
}

export interface CppCuteBrowserSourceArchiveArguments {
  readonly archives: readonly CppCuteBrowserSourceArchiveInput[];
}

export interface CppCuteBrowserSourceArchiveCopyReport {
  readonly sourceId: string;
  readonly outputPath: string;
  readonly archiveSha256: string;
  readonly archiveByteLength: string;
  readonly releaseReady: false;
}

export function cppCuteBrowserCurrentSourceArchiveExpectations():
Promise<Readonly<CppCuteBrowserCurrentSourceArchiveExpectations>>;

export function inspectCppCuteBrowserSourceArchives(input: Readonly<{
  archives: readonly CppCuteBrowserSourceArchiveInput[];
  expectedSources: readonly CppCuteBrowserSourceArchiveExpectation[];
}>): Promise<Readonly<CppCuteBrowserSourceArchiveInspection>>;

export function inspectCppCuteBrowserRegularArchiveFiles(input: Readonly<{
  archives: readonly CppCuteBrowserSourceArchiveInput[];
  expectedArchives: readonly CppCuteBrowserRegularArchiveExpectation[];
}>): Promise<Readonly<CppCuteBrowserRegularArchiveInspection>>;

export function copyCppCuteBrowserInspectedRegularArchive(
  inspection: CppCuteBrowserRegularArchiveInspection,
  sourceId: string,
  outputPath: string,
): Promise<Readonly<CppCuteBrowserSourceArchiveCopyReport>>;

export function admitCppCuteBrowserCurrentSourceArchives(input: Readonly<{
  archives: readonly CppCuteBrowserSourceArchiveInput[];
}>): Promise<Readonly<CppCuteBrowserCurrentSourceArchiveAdmission>>;

export function requireCppCuteBrowserCurrentSourceArchiveAuthority(
  admission: CppCuteBrowserCurrentSourceArchiveAdmission,
): void;

export function canonicalCppCuteBrowserCurrentSourceArchiveAdmissionBytes(
  admission: CppCuteBrowserCurrentSourceArchiveAdmission,
): Uint8Array;

export function copyCppCuteBrowserCurrentSourceArchive(
  admission: CppCuteBrowserCurrentSourceArchiveAdmission,
  sourceId: string,
  outputPath: string,
): Promise<Readonly<CppCuteBrowserSourceArchiveCopyReport>>;

export function parseCppCuteBrowserSourceArchiveArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserSourceArchiveArguments>;
