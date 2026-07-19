export const CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_FILES_SCHEMA:
"browsergrad.compiler.cpp-cute.distribution-output-file-materialization";
export const CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_VERIFICATION_SCHEMA:
"browsergrad.compiler.cpp-cute.distribution-output-file-verification";

export class CppCuteBrowserDistributionOutputFilesError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-OUTPUT-FILES";
  readonly path: string;
}

export interface CppCuteBrowserDistributionExistingOutput {
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserDistributionNewOutput {
  readonly outputPath: string;
  readonly bytes: Uint8Array;
}

export interface CppCuteBrowserDistributionOutputFileVerification {
  readonly schema: typeof CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_VERIFICATION_SCHEMA;
  readonly version: 1;
  readonly verificationId: string;
  readonly authority: "caller-expected-private-distribution-output-verification-only";
  readonly outputRoot: string;
  readonly outputs: readonly CppCuteBrowserDistributionExistingOutput[];
  readonly totals: Readonly<{
    fileCount: number;
    byteLength: string;
  }>;
  readonly claims: Readonly<{
    exactTreeVerifiedBeforeAndAfter: true;
    exactFileBytesReverified: true;
    callerPolicyBound: false;
    reproducibilityVerified: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserDistributionOutputFileMaterialization {
  readonly schema: typeof CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_FILES_SCHEMA;
  readonly version: 1;
  readonly materializationId: string;
  readonly authority: "caller-expected-private-distribution-output-materialization-only";
  readonly outputRoot: string;
  readonly existingOutputs: readonly CppCuteBrowserDistributionExistingOutput[];
  readonly outputs: readonly Readonly<{
    ordinal: number;
    outputPath: string;
    sha256: string;
    byteLength: string;
  }>[];
  readonly totals: Readonly<{
    existingFileCount: number;
    existingByteLength: string;
    materializedFileCount: number;
    materializedByteLength: string;
  }>;
  readonly claims: Readonly<{
    exactInitialTreeVerified: true;
    exactExistingFileBytesReverified: true;
    newFilesWrittenWithoutClobber: true;
    newFilesIndependentlyReread: true;
    exactFinalTreeVerified: true;
    callerPolicyBound: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export function materializeCppCuteBrowserDistributionOutputFiles(input: Readonly<{
  outputRoot: string;
  existingOutputs: readonly CppCuteBrowserDistributionExistingOutput[];
  outputs: readonly CppCuteBrowserDistributionNewOutput[];
}>): Promise<Readonly<CppCuteBrowserDistributionOutputFileMaterialization>>;

export function verifyCppCuteBrowserDistributionOutputFiles(input: Readonly<{
  outputRoot: string;
  expectedOutputs: readonly CppCuteBrowserDistributionExistingOutput[];
}>): Promise<Readonly<CppCuteBrowserDistributionOutputFileVerification>>;
