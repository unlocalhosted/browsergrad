import type {
  CppCuteBrowserDistributionExistingOutput,
} from "./cpp_cute_browser_distribution_output_files.mjs";
import type {
  CppCuteBrowserHeaderPackPipeline, CppCuteBrowserHeaderPackPipelineInput,
} from "./cpp_cute_browser_header_pack_pipeline.mjs";

export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility";

export class CppCuteBrowserHeaderDistributionReproducibilityError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-DISTRIBUTION-REPRODUCIBILITY";
  readonly path: string;
}

export interface CppCuteBrowserHeaderDistributionReproducibilityReport {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_SCHEMA;
  readonly version: 2;
  readonly reproducibilityId: string;
  readonly authority: "two-root-exact-header-distribution-reproducibility-only";
  readonly scope: "five-header-packs-license-inventory-and-notice-outputs-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly pipelineId: string;
  readonly outputVerificationId: string;
  readonly firstOutputRoot: string;
  readonly secondOutputRoot: string;
  readonly outputs: readonly CppCuteBrowserDistributionExistingOutput[];
  readonly totals: Readonly<{
    outputCount: 17;
    byteLength: string;
  }>;
  readonly claims: Readonly<{
    twoDistinctPrivateOutputRootsVerified: true;
    exactOutputsRehashedInBothRoots: true;
    exactHeaderDistributionOutputSetReproducible: true;
    fullDistributedOutputSetReproducible: false;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    signedProvenanceVerified: false;
    workerExecutionObserved: false;
    releaseReady: false;
  }>;
}

declare const cppCuteBrowserHeaderDistributionReproducibilityInputBrand: unique symbol;

export interface CppCuteBrowserHeaderDistributionReproducibilityInput {
  readonly [cppCuteBrowserHeaderDistributionReproducibilityInputBrand]: true;
  readonly first: Readonly<CppCuteBrowserHeaderPackPipelineInput>;
  readonly second: Readonly<CppCuteBrowserHeaderPackPipelineInput>;
}

export function verifyCppCuteBrowserHeaderDistributionReproducibility(input: Readonly<{
  first: CppCuteBrowserHeaderPackPipeline;
  second: CppCuteBrowserHeaderPackPipeline;
}>): Promise<Readonly<CppCuteBrowserHeaderDistributionReproducibilityReport>>;

export function materializeAndVerifyCppCuteBrowserHeaderDistributionReproducibility(
  input: CppCuteBrowserHeaderDistributionReproducibilityInput,
): Promise<Readonly<CppCuteBrowserHeaderDistributionReproducibilityReport>>;

export function parseCppCuteBrowserHeaderDistributionReproducibilityArguments(
  argv: readonly string[],
): CppCuteBrowserHeaderDistributionReproducibilityInput;

export function requireCppCuteBrowserHeaderDistributionReproducibilityAuthority(
  report: CppCuteBrowserHeaderDistributionReproducibilityReport,
): void;

export function canonicalCppCuteBrowserHeaderDistributionReproducibilityBytes(
  report: CppCuteBrowserHeaderDistributionReproducibilityReport,
): Uint8Array;
