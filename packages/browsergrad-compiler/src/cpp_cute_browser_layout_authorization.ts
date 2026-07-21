import { hashCanonicalJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapObservedCppCuteBrowserLayoutCandidate,
  type ObservedCppCuteBrowserLayoutCandidate,
} from "./cpp_cute_browser_layout_candidate.js";
import {
  unwrapVerifiedCppCuteBrowserBuildProducer,
  type VerifiedCppCuteBrowserBuildProducer,
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
import {
  issueBrowserCppCuteFrontendArtifactAuthorization,
  type AuthorizedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_authorization.js";

const CAPTURED_OBJECT = Object;
const CAPTURED_REFLECT = Reflect;
const CAPTURED_WEAK_MAP = WeakMap;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const AUTHORIZATIONS = new CAPTURED_WEAK_MAP<
  object,
  AuthorizedCppCuteBrowserLayoutArtifactRecord
>();

declare const authorizedBrowserLayoutArtifactBrand: unique symbol;

/**
 * Local semantic-lowering authority for one exact observed Worker artifact and
 * one independently admitted build producer. It grants no backend,
 * distribution, or release authority.
 */
export interface AuthorizedCppCuteBrowserLayoutArtifact {
  readonly [authorizedBrowserLayoutArtifactBrand]: true;
  readonly authority: "browser-worker-layout-local-semantic-authorization";
  readonly authorizationId: string;
  readonly candidateId: string;
  readonly producerEvidenceId: string;
  readonly executionEvidenceId: string;
  readonly artifactId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly entryId: string;
  readonly layoutSemanticHash: string;
  readonly workerExecutionObserved: true;
  readonly producerTrusted: true;
  readonly localSemanticLoweringAuthorized: true;
  readonly backendExecutionAuthorized: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface AuthorizedCppCuteBrowserLayoutArtifactRecord {
  readonly candidate: ObservedCppCuteBrowserLayoutCandidate;
  readonly producer: VerifiedCppCuteBrowserBuildProducer;
  readonly authorization: AuthorizedCppCuteFrontendArtifact;
  readonly backendExecutionAuthorized: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface AuthorizeCppCuteBrowserLayoutArtifactOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserLayoutAuthorizationErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-SUBJECT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-UNVERIFIED";

export class CppCuteBrowserLayoutAuthorizationError extends Error {
  constructor(
    readonly code: CppCuteBrowserLayoutAuthorizationErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserLayoutAuthorizationError";
  }
}

export async function authorizeCppCuteBrowserLayoutArtifact(
  candidate: ObservedCppCuteBrowserLayoutCandidate,
  producer: VerifiedCppCuteBrowserBuildProducer,
  options: AuthorizeCppCuteBrowserLayoutArtifactOptions = {},
): Promise<AuthorizedCppCuteBrowserLayoutArtifact> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  let candidateRecord: ReturnType<typeof unwrapObservedCppCuteBrowserLayoutCandidate>;
  let producerRecord: ReturnType<typeof unwrapVerifiedCppCuteBrowserBuildProducer>;
  try {
    candidateRecord = unwrapObservedCppCuteBrowserLayoutCandidate(candidate);
  } catch (cause) {
    unverified("$.candidate", "candidate is not an exact observed browser layout authority", cause);
  }
  try {
    producerRecord = unwrapVerifiedCppCuteBrowserBuildProducer(producer);
  } catch (cause) {
    unverified("$.producer", "producer is not an exact independently admitted authority", cause);
  }
  const executionRecord = unwrapObservedCppCuteBrowserWorkerExecution(candidateRecord.execution);
  const frameRecord = unwrapValidatedCppCuteBrowserWorkerResultFrame(
    candidateRecord.validatedResultFrame,
  );
  const conformanceRecord = unwrapObservedCppCuteBrowserPackageWasmConformance(
    candidateRecord.observedWasmConformance,
  );
  const signatureRecord = unwrapVerifiedCppCuteBrowserBuildSignatureBinding(
    producerRecord.signatureBinding,
  );
  const workerInspection = inspectVerifiedCppCuteBrowserWorkerBundle(signatureRecord.workerBundle);
  const lineage = executionRecord.packageInvocationLineage;
  const invocation = lineage.invocation;
  const semantics = candidateRecord.semantics;
  const artifact = candidateRecord.artifact;
  const profile = candidateRecord.profile;
  const requestBinding = candidateRecord.requestBinding;

  if (candidate.authority !== "observed-browser-worker-layout-semantic-candidate" ||
      candidate.workerExecutionObserved !== true ||
      candidate.artifactOutcome !== "accepted" ||
      candidate.sharedLayoutSemanticsPrepared !== true ||
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
      candidate.entryId !== semantics.entry.entryId ||
      candidate.layoutSemanticHash !== semantics.preparedLayout.layoutSemanticHash ||
      candidate.indexMapId !== semantics.preparedLayout.indexMapId ||
      candidate.coordinateRank !== semantics.preparedLayout.coordinateRank ||
      semantics.artifact !== artifact ||
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
      "Worker execution, result frame, verifier observation, and layout candidate are not one exact lineage",
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

  throwIfAborted(signal);
  const evidenceHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-layout-authorization.v1",
    candidateId: candidate.candidateId,
    producerEvidenceId: producer.producerEvidenceId,
    executionEvidenceId: candidate.executionEvidenceId,
    artifactId: candidate.artifactId,
    artifactHash: candidate.artifactHash,
    profileHash: candidate.profileHash,
    requestId: candidate.requestId,
    requestBindingId: candidate.requestBindingId,
    entryId: candidate.entryId,
    layoutSemanticHash: candidate.layoutSemanticHash,
  });
  throwIfAborted(signal);
  const authorization = issueBrowserCppCuteFrontendArtifactAuthorization({
    candidate,
    producer,
    evidenceHash,
  });
  const authorized = NATIVE_OBJECT_FREEZE({
    authority: "browser-worker-layout-local-semantic-authorization",
    authorizationId: `bg.cpp.browser-layout-authorization.sha256.${evidenceHash}`,
    candidateId: candidate.candidateId,
    producerEvidenceId: producer.producerEvidenceId,
    executionEvidenceId: candidate.executionEvidenceId,
    artifactId: candidate.artifactId,
    profileHash: candidate.profileHash,
    requestId: candidate.requestId,
    requestBindingId: candidate.requestBindingId,
    entryId: candidate.entryId,
    layoutSemanticHash: candidate.layoutSemanticHash,
    workerExecutionObserved: true,
    producerTrusted: true,
    localSemanticLoweringAuthorized: true,
    backendExecutionAuthorized: false,
    distributionAuthorized: false,
    releaseReady: false,
  }) as AuthorizedCppCuteBrowserLayoutArtifact;
  weakMapSet(AUTHORIZATIONS, authorized, NATIVE_OBJECT_FREEZE({
    candidate,
    producer,
    authorization,
    backendExecutionAuthorized: false,
    distributionAuthorized: false,
    releaseReady: false,
  }));
  return authorized;
}

export function unwrapAuthorizedCppCuteBrowserLayoutArtifact(
  authorized: AuthorizedCppCuteBrowserLayoutArtifact,
): AuthorizedCppCuteBrowserLayoutArtifactRecord {
  if (typeof authorized !== "object" || authorized === null) unverified("$", "expected opaque browser layout authorization");
  const record = weakMapGet(AUTHORIZATIONS, authorized as object);
  if (record === undefined) unverified("$", "browser layout authorization was not issued by this module instance");
  return record;
}

function normalizeOptions(options: AuthorizeCppCuteBrowserLayoutArtifactOptions): AbortSignal | undefined {
  if (typeof options !== "object" || options === null) {
    mismatch("$.options", "options must be a plain data record");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = NATIVE_REFLECT_APPLY(
      NATIVE_OBJECT_GET_PROTOTYPE_OF,
      CAPTURED_OBJECT,
      [options],
    ) as object | null;
    descriptors = NATIVE_REFLECT_APPLY(
      NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS,
      CAPTURED_OBJECT,
      [options],
    ) as PropertyDescriptorMap;
    keys = NATIVE_REFLECT_APPLY(NATIVE_REFLECT_OWN_KEYS, CAPTURED_REFLECT, [descriptors]) as
      readonly PropertyKey[];
  } catch (cause) {
    unverified("$.options", "options could not be inspected without invoking accessors", cause);
  }
  if (prototype !== CAPTURED_OBJECT.prototype) {
    mismatch("$.options", "options must be a plain data record");
  }
  if (keys.some((key) => key !== "signal")) mismatch("$.options", "options contain unknown fields");
  const descriptor = descriptors.signal;
  if (descriptor === undefined) return undefined;
  if (!("value" in descriptor) || descriptor.enumerable !== true) {
    mismatch("$.options.signal", "signal must be an enumerable plain data property");
  }
  const signal = descriptor.value as unknown;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined || typeof signal !== "object" || signal === null) {
    mismatch("$.options.signal", "signal must be a native AbortSignal");
  }
  try {
    NATIVE_REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []);
  } catch {
    mismatch("$.options.signal", "signal must be a native AbortSignal");
  }
  return signal as AbortSignal;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal !== undefined &&
      NATIVE_REFLECT_APPLY(
        ABORT_SIGNAL_ABORTED_GETTER as (this: AbortSignal) => boolean,
        signal,
        [],
      ) === true) {
    throw new CppCuteBrowserLayoutAuthorizationError(
      "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-CANCELLED",
      "$.signal",
      "browser layout authorization was cancelled",
    );
  }
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserLayoutAuthorizationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-SUBJECT-MISMATCH",
    path,
    message,
  );
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserLayoutAuthorizationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-UNVERIFIED",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
