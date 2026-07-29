import type {
  PreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import type {
  CppCuteBrowserDistributionExistingOutput,
} from "./cpp_cute_browser_distribution_output_files.mjs";

export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility";

export class CppCuteBrowserFullDistributionReproducibilityError extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-REPRODUCIBILITY";
  readonly path: string;
}

export interface CppCuteBrowserFullDistributionReproducibilityArguments {
  readonly firstOutputRoot: string;
  readonly secondOutputRoot: string;
  readonly evidenceOutput: string;
}

export interface CppCuteBrowserFullDistributionReproducibilityInputTree {
  readonly outputRoot: string;
  readonly expectedOutputs:
    readonly CppCuteBrowserDistributionExistingOutput[];
}

export interface CppCuteBrowserFullDistributionReproducibility {
  readonly schema:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_SCHEMA;
  readonly version: 1;
  readonly reproducibilityId: string;
  readonly authority:
    "two-root-complete-distribution-output-reproducibility-only";
  readonly scope: "complete-build-input-lock-distributed-output-plan";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly firstOutputRoot: string;
  readonly secondOutputRoot: string;
  readonly firstOutputVerificationId: string;
  readonly secondOutputVerificationId: string;
  readonly deterministicOutputs: readonly Readonly<{
    outputPath: string;
    role: string;
    mediaType: string;
    sha256: string;
    byteLength: string;
  }>[];
  readonly detachedEvidence: Readonly<{
    outputPath:
      "assets/browsergrad-cpp-cute/build-provenance.dsse.json";
    role: string;
    mediaType: string;
    firstSha256: string;
    firstByteLength: string;
    secondSha256: string;
    secondByteLength: string;
    buildSubjectId: string;
    buildSubjectSha256: string;
  }>;
  readonly totals: Readonly<{
    outputCount: 25;
    deterministicSubjectCount: 24;
    detachedEvidenceCount: 1;
    firstByteLength: string;
    secondByteLength: string;
  }>;
  readonly claims: Readonly<{
    twoDistinctPrivateOutputRootsVerified: true;
    exactBuildLockOutputPlanMatched: true;
    exactOutputsRehashedInBothRoots: true;
    deterministicSubjectsByteIdentical: true;
    detachedEvidenceBuildSubjectMatched: true;
    fullDistributedOutputSetReproducible: true;
    detachedSignatureVerified: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    producerTrusted: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export function verifyCppCuteBrowserFullDistributionReproducibility(
  input: Readonly<{
    buildInputLock: PreparedCppCuteBrowserBuildInputLock;
    first: CppCuteBrowserFullDistributionReproducibilityInputTree;
    second: CppCuteBrowserFullDistributionReproducibilityInputTree;
  }>,
): Promise<Readonly<CppCuteBrowserFullDistributionReproducibility>>;

export function parseCppCuteBrowserFullDistributionReproducibilityArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserFullDistributionReproducibilityArguments>;

export function observeCppCuteBrowserFullDistributionReproducibility(
  input: Readonly<{
    firstOutputRoot: string;
    secondOutputRoot: string;
  }>,
): Promise<Readonly<CppCuteBrowserFullDistributionReproducibility>>;

export function requireCppCuteBrowserFullDistributionReproducibilityAuthority(
  report: CppCuteBrowserFullDistributionReproducibility,
): void;

export function canonicalCppCuteBrowserFullDistributionReproducibilityBytes(
  report: CppCuteBrowserFullDistributionReproducibility,
): Uint8Array;
