import {
  unwrapObservedCppCuteBrowserSemanticCandidate,
  type ObservedCppCuteBrowserSemanticCandidate,
  type ObservedCppCuteBrowserSemanticCandidateRecord,
} from "./cpp_cute_browser_semantic_candidate.js";
import {
  unwrapVerifiedCppCuteBrowserBuildProducer,
  type VerifiedCppCuteBrowserBuildProducer,
  type VerifiedCppCuteBrowserBuildProducerRecord,
} from "./cpp_cute_browser_producer_trust.js";
import { unwrapVerifiedCppCuteBrowserBuildSignatureBinding } from
  "./cpp_cute_browser_build_provenance.js";
import { unwrapObservedCppCuteBrowserWorkerExecution } from
  "./cpp_cute_browser_worker_controller.js";
import { unwrapValidatedCppCuteBrowserWorkerResultFrame } from
  "./cpp_cute_browser_worker_protocol.js";
import { unwrapObservedCppCuteBrowserPackageWasmConformance } from
  "./cpp_cute_browser_wasm_verifier_controller.js";
import { inspectVerifiedCppCuteBrowserWorkerBundle } from
  "./cpp_cute_browser_worker_bundle.js";

export interface VerifiedCppCuteBrowserSemanticCandidateProducerBinding {
  readonly candidateRecord: ObservedCppCuteBrowserSemanticCandidateRecord;
  readonly producerRecord: VerifiedCppCuteBrowserBuildProducerRecord;
}

export type CppCuteBrowserSemanticAuthorizationErrorKind =
  | "subject-mismatch"
  | "unverified";

export class CppCuteBrowserSemanticAuthorizationError extends Error {
  constructor(
    readonly kind: CppCuteBrowserSemanticAuthorizationErrorKind,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "CppCuteBrowserSemanticAuthorizationError";
  }
}

/** Exact producer/build-subject cross-binding shared by every browser semantic entry kind. */
export function verifyCppCuteBrowserSemanticCandidateProducerBinding(
  candidate: ObservedCppCuteBrowserSemanticCandidate,
  producer: VerifiedCppCuteBrowserBuildProducer,
): VerifiedCppCuteBrowserSemanticCandidateProducerBinding {
  let candidateRecord: ObservedCppCuteBrowserSemanticCandidateRecord;
  let producerRecord: VerifiedCppCuteBrowserBuildProducerRecord;
  try {
    candidateRecord = unwrapObservedCppCuteBrowserSemanticCandidate(candidate);
  } catch (cause) {
    unverified("$.candidate", "candidate is not an exact observed browser semantic authority", cause);
  }
  try {
    producerRecord = unwrapVerifiedCppCuteBrowserBuildProducer(producer);
  } catch (cause) {
    unverified("$.producer", "producer is not an exact independently admitted authority", cause);
  }
  let executionRecord: ReturnType<typeof unwrapObservedCppCuteBrowserWorkerExecution>;
  let frameRecord: ReturnType<typeof unwrapValidatedCppCuteBrowserWorkerResultFrame>;
  let conformanceRecord: ReturnType<typeof unwrapObservedCppCuteBrowserPackageWasmConformance>;
  let signatureRecord: ReturnType<typeof unwrapVerifiedCppCuteBrowserBuildSignatureBinding>;
  let workerInspection: ReturnType<typeof inspectVerifiedCppCuteBrowserWorkerBundle>;
  try {
    executionRecord = unwrapObservedCppCuteBrowserWorkerExecution(candidateRecord.execution);
    frameRecord = unwrapValidatedCppCuteBrowserWorkerResultFrame(
      candidateRecord.validatedResultFrame,
    );
    conformanceRecord = unwrapObservedCppCuteBrowserPackageWasmConformance(
      candidateRecord.observedWasmConformance,
    );
    signatureRecord = unwrapVerifiedCppCuteBrowserBuildSignatureBinding(
      producerRecord.signatureBinding,
    );
    workerInspection = inspectVerifiedCppCuteBrowserWorkerBundle(signatureRecord.workerBundle);
  } catch (cause) {
    unverified("$", "retained candidate or producer authority chain could not be unwrapped", cause);
  }
  const lineage = executionRecord.packageInvocationLineage;
  const invocation = lineage.invocation;
  const artifact = candidateRecord.artifact;
  const profile = candidateRecord.profile;
  const requestBinding = candidateRecord.requestBinding;

  if (candidate.workerExecutionObserved !== true ||
      candidate.artifactOutcome !== "accepted" ||
      candidate.producerTrusted !== false ||
      candidate.loweringAuthorityMinted !== false ||
      candidate.backendExecutionAuthorized !== false ||
      candidate.releaseReady !== false ||
      candidateRecord.commonLoweringAuthorized !== false ||
      candidateRecord.backendExecutionAuthorized !== false ||
      candidateRecord.releaseReady !== false ||
      candidate.executionEvidenceId !== candidateRecord.execution.evidenceId ||
      candidate.invocationId !== invocation.invocationId ||
      candidate.profileHash !== profile.profileHash ||
      candidate.requestId !== requestBinding.requestId ||
      candidate.requestBindingId !== requestBinding.bindingId ||
      candidate.artifactId !== artifact.artifactId ||
      candidate.artifactHash !== artifact.artifactHash ||
      candidate.artifactBytesSha256 !== artifact.artifactBytesSha256 ||
      candidate.artifactByteLength !== artifact.artifactByteLength ||
      artifact.outcome !== "accepted") {
    mismatch("$.candidate", "candidate projection differs from its retained semantic authority chain");
  }

  if (executionRecord.validatedResultFrame !== candidateRecord.validatedResultFrame ||
      frameRecord.artifact !== artifact ||
      frameRecord.profile !== profile ||
      frameRecord.requestBinding !== requestBinding ||
      lineage.observedWasmConformance !== candidateRecord.observedWasmConformance ||
      conformanceRecord.assetManifest !== frameRecord.assetManifest) {
    mismatch(
      "$.candidate.execution",
      "Worker execution, result frame, verifier observation, and semantic candidate are not one exact lineage",
    );
  }

  if (producer.authority !== "independently-admitted-browser-build-producer" ||
      producer.signatureVerified !== true ||
      producer.manifestSignaturePolicyMatched !== true ||
      producer.independentTrustPolicyMatched !== true ||
      producer.producerTrusted !== true ||
      producer.buildSubjectBound !== true ||
      producer.exactAssetBytesVerified !== false ||
      producer.fullDistributedOutputSetReproducible !== false ||
      producer.licenseReviewComplete !== false ||
      producer.distributionAuthorized !== false ||
      producer.workerExecutionObserved !== false ||
      producer.loweringAuthorityMinted !== false ||
      producer.backendExecutionObserved !== false ||
      producer.releaseReady !== false ||
      producer.buildSubjectId !== producerRecord.signatureBinding.buildSubjectId ||
      producer.buildSubjectSha256 !== producerRecord.signatureBinding.buildSubjectSha256 ||
      producer.statementSha256 !== producerRecord.signatureBinding.statementSha256 ||
      producer.signatureEvidenceSha256 !== producerRecord.signatureBinding.evidenceSha256 ||
      producer.builderId !== producerRecord.signatureBinding.builderId ||
      producer.keyId !== producerRecord.signatureBinding.keyId ||
      producer.trustStoreSha256 !== producerRecord.signatureBinding.trustStoreSha256 ||
      producer.profileHash !== producerRecord.signatureBinding.profileHash ||
      producer.manifestId !== producerRecord.signatureBinding.manifestId ||
      producer.assetSetSha256 !== producerRecord.signatureBinding.assetSetSha256 ||
      producer.buildInputLockResourceSha256 !==
        producerRecord.signatureBinding.buildInputLockResourceSha256 ||
      producer.workerBundleSha256 !== producerRecord.signatureBinding.workerBundleSha256) {
    mismatch("$.producer", "producer projection differs from its retained signature authority chain");
  }

  if (signatureRecord.profile !== profile ||
      signatureRecord.assetManifest !== frameRecord.assetManifest ||
      lineage.workerBundle !== workerInspection ||
      producer.profileHash !== candidate.profileHash ||
      producer.manifestId !== invocation.assetManifestId ||
      producer.assetSetSha256 !== invocation.assetSetSha256 ||
      producer.workerBundleSha256 !== invocation.worker.moduleSha256 ||
      frameRecord.assetManifest.manifestId !== invocation.assetManifestId ||
      frameRecord.assetManifest.assetSetSha256 !== invocation.assetSetSha256) {
    mismatch(
      "$.producer",
      "producer build subject does not cross-bind the exact observed profile, manifest, and Worker bundle",
    );
  }
  return Object.freeze({ candidateRecord, producerRecord });
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserSemanticAuthorizationError("subject-mismatch", path, message);
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserSemanticAuthorizationError(
    "unverified",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
