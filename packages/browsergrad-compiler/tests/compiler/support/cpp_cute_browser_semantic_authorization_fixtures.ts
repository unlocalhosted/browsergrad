import type { VerifiedCppCuteFrontendArtifact } from
  "../../../src/cpp_cute_frontend_artifact.js";
import type { PreparedCppCuteFrontendProfile } from
  "../../../src/cpp_cute_frontend_profile.js";
import type { PreparedCppCuteFrontendRequestBinding } from
  "../../../src/cpp_cute_frontend_request_binding.js";

export interface CppCuteBrowserSemanticAuthorityFixtureInput {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
}

export interface CppCuteBrowserSemanticAuthorityFixture {
  readonly execution: Readonly<Record<string, unknown>>;
  readonly executionRecord: Readonly<Record<string, unknown>>;
  readonly frame: Readonly<Record<string, unknown>>;
  readonly frameRecord: Readonly<Record<string, unknown>>;
  readonly conformance: Readonly<Record<string, unknown>>;
  readonly conformanceInspection: Readonly<Record<string, unknown>>;
  readonly conformanceRecord: Readonly<Record<string, unknown>>;
  readonly invocation: Readonly<Record<string, unknown>>;
  readonly workerBundle: Readonly<Record<string, unknown>>;
  readonly workerInspection: Readonly<Record<string, unknown>>;
  readonly signatureBinding: Readonly<Record<string, unknown>>;
  readonly signatureRecord: Readonly<Record<string, unknown>>;
  readonly producer: Readonly<Record<string, unknown>>;
  readonly producerRecord: Readonly<Record<string, unknown>>;
}

/** Pure synthetic authority graph for unit tests; it carries no storage facts. */
export function createCppCuteBrowserSemanticAuthorityFixture(
  input: CppCuteBrowserSemanticAuthorityFixtureInput,
): CppCuteBrowserSemanticAuthorityFixture {
  const manifestId = `bg.cpp.browser-assets.sha256.${"1".repeat(64)}`;
  const assetSetSha256 = "2".repeat(64);
  const workerBundleSha256 = "3".repeat(64);
  const executionEvidenceId = `bg.cpp.browser-worker-execution.sha256.${"4".repeat(64)}`;
  const invocationId = `bg.cpp.browser-worker-invocation.sha256.${"5".repeat(64)}`;
  const producerEvidenceId = `bg.cpp.browser-build-producer.sha256.${"7".repeat(64)}`;
  const invocationNonceSha256 = "8".repeat(64);
  const verifierEvidenceId =
    `bg.cpp.browser-wasm-verifier-conformance.sha256.${"9".repeat(64)}`;
  const verifierEvidenceRegionSha256 = "a".repeat(64);
  const validationId = `bg.cpp.browser-worker-caller-frame.sha256.${"b".repeat(64)}`;
  const manifest = Object.freeze({ manifestId, assetSetSha256 });
  const workerInspection = Object.freeze({ sha256: workerBundleSha256 });
  const workerBundle = Object.freeze({ worker: true });
  const conformance = Object.freeze({ conformance: true });
  const frame = Object.freeze({
    validationId,
    invocationId,
    requestId: input.requestBinding.requestId,
    requestBindingId: input.requestBinding.bindingId,
    artifactId: input.artifact.artifactId,
    artifactBytesSha256: input.artifact.artifactBytesSha256,
    outcome: "accepted",
  });
  const invocation = Object.freeze({
    invocationId,
    invocationNonceSha256,
    profileHash: input.profile.profileHash,
    requestId: input.requestBinding.requestId,
    assetManifestId: manifestId,
    assetSetSha256,
    verifierEvidenceId,
    verifierEvidenceRegionSha256,
    worker: Object.freeze({ moduleSha256: workerBundleSha256 }),
  });
  const lineage = Object.freeze({
    invocation,
    workerBundle: workerInspection,
    observedWasmConformance: conformance,
    verifierEvidenceId,
    verifierEvidenceRegionSha256,
  });
  const execution = Object.freeze({
    authority: "host-owned-browser-worker-execution",
    evidenceId: executionEvidenceId,
    invocationId,
    profileHash: input.profile.profileHash,
    requestId: input.requestBinding.requestId,
    workerModuleSha256: workerBundleSha256,
    invocationNonceSha256,
    verifierEvidenceRegionSha256,
    acceptedTerminalMessages: "1",
    workerExecutionObserved: true,
    loweringAuthorityMinted: false,
    releaseReady: false,
  });
  const signatureBinding = Object.freeze({
    buildSubjectId: `bg.cpp.browser-build-subject.sha256.${"c".repeat(64)}`,
    buildSubjectSha256: "d".repeat(64),
    statementSha256: "e".repeat(64),
    evidenceSha256: "f".repeat(64),
    builderId: "https://builder.browsergrad.dev/compiler",
    keyId: "browsergrad-test-key",
    trustStoreSha256: "0".repeat(64),
    profileHash: input.profile.profileHash,
    manifestId,
    assetSetSha256,
    buildInputLockResourceSha256: "6".repeat(64),
    workerBundleSha256,
  });
  const producer = Object.freeze({
    authority: "independently-admitted-browser-build-producer",
    producerEvidenceId,
    policyId: "browsergrad.test.policy",
    policySha256: "5".repeat(64),
    policyVersion: "1.0",
    buildSubjectId: signatureBinding.buildSubjectId,
    buildSubjectSha256: signatureBinding.buildSubjectSha256,
    statementSha256: signatureBinding.statementSha256,
    signatureEvidenceSha256: signatureBinding.evidenceSha256,
    predicateType: "https://browsergrad.dev/provenance/cpp-cute-browser-build/v1",
    builderId: signatureBinding.builderId,
    keyId: signatureBinding.keyId,
    trustStoreSha256: signatureBinding.trustStoreSha256,
    profileHash: input.profile.profileHash,
    manifestId,
    assetSetSha256,
    buildInputLockResourceSha256: signatureBinding.buildInputLockResourceSha256,
    workerBundleSha256,
    signatureVerified: true,
    manifestSignaturePolicyMatched: true,
    independentTrustPolicyMatched: true,
    producerTrusted: true,
    buildSubjectBound: true,
    exactAssetBytesVerified: false,
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
    backendExecutionObserved: false,
    releaseReady: false,
  });
  return Object.freeze({
    execution,
    executionRecord: Object.freeze({
      validatedResultFrame: frame,
      validatedPackageResult: Object.freeze({ validationId }),
      packageInvocationLineage: lineage,
      productionAuthority: true,
    }),
    frame,
    frameRecord: Object.freeze({
      artifact: input.artifact,
      profile: input.profile,
      requestBinding: input.requestBinding,
      assetManifest: manifest,
    }),
    conformance,
    conformanceInspection: Object.freeze({
      evidenceId: verifierEvidenceId,
      productionConformanceAuthorityMinted: true,
      verifierWorkerExecutionObserved: true,
      rawWasmVerified: true,
      exactInterfaceConformanceObserved: true,
      compilerWorkerExecutionObserved: false,
      loweringAuthorityMinted: false,
      releaseReady: false,
    }),
    conformanceRecord: Object.freeze({ assetManifest: manifest }),
    invocation,
    workerBundle,
    workerInspection,
    signatureBinding,
    signatureRecord: Object.freeze({
      profile: input.profile,
      assetManifest: manifest,
      workerBundle,
    }),
    producer,
    producerRecord: Object.freeze({
      signatureBinding,
      trustPolicy: Object.freeze({ policy: true }),
    }),
  });
}

export function attachCppCuteBrowserSemanticCandidate(
  authority: CppCuteBrowserSemanticAuthorityFixture,
  input: CppCuteBrowserSemanticAuthorityFixtureInput,
  candidateSpecific: Readonly<Record<string, unknown>>,
  recordSpecific: Readonly<Record<string, unknown>>,
): CppCuteBrowserSemanticAuthorityFixture & {
  readonly candidate: Readonly<Record<string, unknown>>;
  readonly candidateRecord: Readonly<Record<string, unknown>>;
} {
  const candidate = Object.freeze({
    candidateId: candidateSpecific.candidateId,
    executionEvidenceId: authority.execution.evidenceId,
    invocationId: authority.execution.invocationId,
    profileHash: input.profile.profileHash,
    requestId: input.requestBinding.requestId,
    requestBindingId: input.requestBinding.bindingId,
    artifactId: input.artifact.artifactId,
    artifactHash: input.artifact.artifactHash,
    artifactBytesSha256: input.artifact.artifactBytesSha256,
    artifactByteLength: input.artifact.artifactByteLength,
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
    ...candidateSpecific,
  });
  return Object.freeze({
    ...authority,
    candidate,
    candidateRecord: Object.freeze({
      execution: authority.execution,
      validatedResultFrame: authority.frame,
      artifact: input.artifact,
      profile: input.profile,
      requestBinding: input.requestBinding,
      observedWasmConformance: authority.conformance,
      commonLoweringAuthorized: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
      ...recordSpecific,
    }),
  });
}
