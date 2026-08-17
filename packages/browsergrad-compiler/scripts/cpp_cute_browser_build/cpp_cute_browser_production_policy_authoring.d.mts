import type {
  CppCuteBrowserDistributionOutputFileMaterialization,
} from "./cpp_cute_browser_distribution_output_files.mjs";

export const CPP_CUTE_BROWSER_PRODUCTION_POLICY_HANDOFF_SCHEMA:
  "browsergrad.compiler.cpp-cute.browser-production-policy-handoff";

export class CppCuteBrowserProductionPolicyAuthoringError extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-POLICY-AUTHORING";
  readonly path: string;
}

export interface CppCuteBrowserProductionPolicyAuthoringArguments {
  readonly outputRoot: string;
  readonly producerId: string;
  readonly reviewerId: string;
}

export interface CppCuteBrowserProductionPolicyOutputIdentity {
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserProductionPolicyTrustStoreIdentity extends
  CppCuteBrowserProductionPolicyOutputIdentity {
  readonly trustStoreHash: string;
}

export interface CppCuteBrowserProductionPolicyIdentity extends
  CppCuteBrowserProductionPolicyOutputIdentity {
  readonly policyId: string;
}

export interface CppCuteBrowserProductionPolicyPrincipal {
  readonly identity: string;
  readonly keyId: string;
  readonly publicKey: CppCuteBrowserProductionPolicyOutputIdentity;
  readonly trustStore: CppCuteBrowserProductionPolicyTrustStoreIdentity;
  readonly policy: CppCuteBrowserProductionPolicyIdentity;
}

export interface CppCuteBrowserProductionPolicyHandoffRecord {
  readonly schema:
    typeof CPP_CUTE_BROWSER_PRODUCTION_POLICY_HANDOFF_SCHEMA;
  readonly version: 1;
  readonly authority: "package-authored-public-policy-handoff-only";
  readonly handoffId: string;
  readonly producer: CppCuteBrowserProductionPolicyPrincipal;
  readonly reviewer: CppCuteBrowserProductionPolicyPrincipal;
  readonly separation: Readonly<{
    producerReviewerIdentitySeparated: true;
    producerReviewerKeySeparated: true;
  }>;
  readonly authoredOutputs:
    readonly CppCuteBrowserProductionPolicyOutputIdentity[];
  readonly claims: Readonly<{
    exactPublicKeysReverified: true;
    canonicalTrustStoresPrepared: true;
    canonicalPoliciesAdmitted: true;
    privateKeyAccepted: false;
    signatureCreated: false;
    externalKeyControlVerified: false;
    producerTrusted: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserProductionPolicyAuthoringResult {
  readonly operation: "author-production-policies";
  readonly record: CppCuteBrowserProductionPolicyHandoffRecord;
  readonly recordSha256: string;
  readonly recordByteLength: string;
  readonly materialization:
    CppCuteBrowserDistributionOutputFileMaterialization;
}

export function authorCppCuteBrowserProductionPolicies(
  input: CppCuteBrowserProductionPolicyAuthoringArguments,
): Promise<Readonly<CppCuteBrowserProductionPolicyAuthoringResult>>;

export function parseCppCuteBrowserProductionPolicyAuthoringArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserProductionPolicyAuthoringArguments>;

export function runCppCuteBrowserProductionPolicyAuthoring(
  argv: readonly string[],
): Promise<Readonly<CppCuteBrowserProductionPolicyAuthoringResult>>;
