import type {
  AdmittedCppCuteBrowserProducerTrustPolicy,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import type {
  VerifiedCppCuteBrowserBuildProducer,
} from "../../dist/cpp_cute_browser_producer_trust.js";
import type {
  CppCuteBrowserDistributionExistingOutput,
} from "./cpp_cute_browser_distribution_output_files.mjs";

export const
  CPP_CUTE_BROWSER_DETERMINISTIC_DISTRIBUTION_MATERIALIZATION_SCHEMA:
    "browsergrad.compiler.cpp-cute.browser-deterministic-distribution-materialization";
export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_MATERIALIZATION_SCHEMA:
  "browsergrad.compiler.cpp-cute.browser-full-distribution-materialization";

export class CppCuteBrowserFullDistributionMaterializationError
  extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-MATERIALIZATION";
  readonly path: string;
}

export interface CppCuteBrowserDeterministicDistributionMaterialization {
  readonly schema:
    typeof CPP_CUTE_BROWSER_DETERMINISTIC_DISTRIBUTION_MATERIALIZATION_SCHEMA;
  readonly version: 1;
  readonly materializationId: string;
  readonly authority:
    "exact-current-private-deterministic-distribution-materialization-only";
  readonly outputRoot: string;
  readonly metadataId: string;
  readonly profileId: string;
  readonly profileHash: string;
  readonly compilationContractHash: string;
  readonly profileSha256: string;
  readonly profileByteLength: string;
  readonly assetManifestId: string;
  readonly assetManifestSha256: string;
  readonly assetManifestByteLength: string;
  readonly assetSetSha256: string;
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly workerBundleSha256: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: string;
  readonly headerDistributionReproducibilityId: string;
  readonly headerDistributionOutputVerificationId: string;
  readonly outputVerificationId: string;
  readonly outputs: readonly CppCuteBrowserDistributionExistingOutput[];
  readonly totals: Readonly<{
    fileCount: number;
    byteLength: string;
  }>;
  readonly claims: Readonly<{
    exactCurrentHeaderDistributionVerified: true;
    exactPackageWasmVerified: true;
    exactPackageWorkerVerified: true;
    exactPackagePolicyAssetsVerified: true;
    exactCurrentBuildInputLockVerified: true;
    exactBuildInputLockDeterministicOutputPlanMatched: true;
    exactDeterministicOutputTreeVerified: true;
    newFilesWrittenWithoutClobber: boolean;
    detachedEnvelopePresent: false;
    signatureVerified: false;
    producerTrusted: false;
    fullDistributedOutputSetMaterialized: false;
    fullDistributedOutputSetReproducible: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserFullDistributionMaterialization {
  readonly schema:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_MATERIALIZATION_SCHEMA;
  readonly version: 1;
  readonly materializationId: string;
  readonly authority:
    "producer-authenticated-exact-private-distribution-materialization-only";
  readonly outputRoot: string;
  readonly deterministicMaterializationId: string;
  readonly metadataId: string;
  readonly producerEvidenceId: string;
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly outputVerificationId: string;
  readonly outputs: readonly CppCuteBrowserDistributionExistingOutput[];
  readonly totals: Readonly<{
    fileCount: number;
    byteLength: string;
  }>;
  readonly claims: Readonly<{
    exactCurrentHeaderDistributionVerified: true;
    exactPackageWasmVerified: true;
    exactPackagePolicyAssetsVerified: true;
    exactBuildInputLockOutputPlanMatched: true;
    deterministicSubjectSetVerified: true;
    detachedEnvelopeMaterializedWithoutClobber: true;
    signatureVerified: true;
    producerTrusted: true;
    fullDistributedOutputSetMaterialized: true;
    fullDistributedOutputSetReproducible: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export function materializeCppCuteBrowserDeterministicDistribution(
  input: Readonly<{
    outputRoot: string;
    producerTrustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
    wasmBytes: Uint8Array;
  }>,
): Promise<Readonly<CppCuteBrowserDeterministicDistributionMaterialization>>;

export function admitCppCuteBrowserDeterministicDistribution(
  input: Readonly<{
    outputRoot: string;
    producerTrustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
    profileBytes: Uint8Array;
  }>,
): Promise<Readonly<CppCuteBrowserDeterministicDistributionMaterialization>>;

export function finalizeCppCuteBrowserFullDistribution(
  input: Readonly<{
    deterministicDistribution:
      CppCuteBrowserDeterministicDistributionMaterialization;
    producer: VerifiedCppCuteBrowserBuildProducer;
  }>,
): Promise<Readonly<CppCuteBrowserFullDistributionMaterialization>>;

export function requireCppCuteBrowserDeterministicDistributionAuthority(
  authority: CppCuteBrowserDeterministicDistributionMaterialization,
): void;

export function requireCppCuteBrowserFullDistributionMaterializationAuthority(
  authority: CppCuteBrowserFullDistributionMaterialization,
): void;

export function copyCppCuteBrowserDeterministicDistributionProfileBytes(
  authority: CppCuteBrowserDeterministicDistributionMaterialization,
): Uint8Array;

export type CppCuteBrowserFullDistributionMaterializationArguments =
  | Readonly<{
      operation: "materialize-deterministic";
      "output-root": string;
      "producer-policy": string;
      "profile-output": string;
      wasm: string;
    }>
  | Readonly<{
      operation: "finalize";
      "output-root": string;
      "producer-policy": string;
      profile: string;
      "trust-store": string;
      envelope: string;
    }>;

export function parseCppCuteBrowserFullDistributionMaterializationArguments(
  argv: readonly string[],
): CppCuteBrowserFullDistributionMaterializationArguments;

export function runCppCuteBrowserFullDistributionMaterialization(
  argv: readonly string[],
): Promise<Readonly<{
  operation: "materialize-deterministic" | "finalize";
  report:
    | CppCuteBrowserDeterministicDistributionMaterialization
    | CppCuteBrowserFullDistributionMaterialization;
  profileOutputPath?: string;
  profileSha256?: string;
  profileByteLength?: string;
}>>;
