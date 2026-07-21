import type { WireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapObservedCppCuteBrowserWorkerExecution,
  type ObservedCppCuteBrowserWorkerExecution,
} from "./cpp_cute_browser_worker_controller.js";
import {
  unwrapValidatedCppCuteBrowserWorkerResultFrame,
  type ValidatedCppCuteBrowserWorkerResultFrame,
} from "./cpp_cute_browser_worker_protocol.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  unwrapPreparedCppCuteFrontendRequestBinding,
  type PreparedCppCuteFrontendRequestBinding,
} from "./cpp_cute_frontend_request_binding.js";
import type { VerifiedCppCuteFrontendArtifact } from "./cpp_cute_frontend_artifact.js";
import {
  inspectObservedCppCuteBrowserPackageWasmConformance,
  unwrapObservedCppCuteBrowserPackageWasmConformance,
  type ObservedCppCuteBrowserPackageWasmConformance,
} from "./cpp_cute_browser_wasm_verifier_controller.js";

const CAPTURED_WEAK_MAP = WeakMap;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;

declare const observedBrowserSemanticSubjectBrand: unique symbol;

/** Common public projection shared by kind-specific browser semantic candidates. */
export interface ObservedCppCuteBrowserSemanticCandidate {
  readonly candidateId: string;
  readonly executionEvidenceId: string;
  readonly invocationId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly workerExecutionObserved: true;
  readonly artifactOutcome: "accepted";
  readonly producerTrusted: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionAuthorized: false;
  readonly releaseReady: false;
}

interface ObservedCppCuteBrowserSemanticSubject {
  readonly [observedBrowserSemanticSubjectBrand]: true;
}

export interface ObservedCppCuteBrowserSemanticCandidateRecord {
  readonly execution: ObservedCppCuteBrowserWorkerExecution;
  readonly validatedResultFrame: ValidatedCppCuteBrowserWorkerResultFrame;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
  readonly observedWasmConformance: ObservedCppCuteBrowserPackageWasmConformance;
  readonly commonLoweringAuthorized: false;
  readonly backendExecutionAuthorized: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserSemanticCandidateErrorKind =
  | "subject-mismatch"
  | "unverified";

export class CppCuteBrowserSemanticCandidateError extends Error {
  constructor(
    readonly kind: CppCuteBrowserSemanticCandidateErrorKind,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "CppCuteBrowserSemanticCandidateError";
  }
}

const SUBJECTS = new CAPTURED_WEAK_MAP<
  object,
  ObservedCppCuteBrowserSemanticCandidateRecord
>();
const CANDIDATES = new CAPTURED_WEAK_MAP<
  object,
  ObservedCppCuteBrowserSemanticCandidateRecord
>();

/**
 * Re-authenticates the producer-neutral host-observed Worker lineage once.
 * Kind-specific semantic candidates retain this opaque subject and register
 * only after their own selected-entry semantics have been prepared.
 */
export function authenticateObservedCppCuteBrowserSemanticSubject(
  execution: ObservedCppCuteBrowserWorkerExecution,
): ObservedCppCuteBrowserSemanticSubject {
  let executionRecord: ReturnType<typeof unwrapObservedCppCuteBrowserWorkerExecution>;
  let frameRecord: ReturnType<typeof unwrapValidatedCppCuteBrowserWorkerResultFrame>;
  let verifierInspection: ReturnType<
    typeof inspectObservedCppCuteBrowserPackageWasmConformance
  >;
  try {
    executionRecord = unwrapObservedCppCuteBrowserWorkerExecution(execution);
    frameRecord = unwrapValidatedCppCuteBrowserWorkerResultFrame(
      executionRecord.validatedResultFrame,
    );
    verifierInspection = inspectObservedCppCuteBrowserPackageWasmConformance(
      executionRecord.packageInvocationLineage.observedWasmConformance,
    );
    unwrapObservedCppCuteBrowserPackageWasmConformance(
      executionRecord.packageInvocationLineage.observedWasmConformance,
    );
  } catch (cause) {
    unverified("$.execution", "expected the exact host-observed Worker authority chain", cause);
  }
  const lineage = executionRecord.packageInvocationLineage;
  const invocation = lineage.invocation;
  const artifact = frameRecord.artifact;
  const profile = frameRecord.profile;
  const requestBinding = frameRecord.requestBinding;
  const binding = unwrapPreparedCppCuteFrontendRequestBinding(requestBinding);
  unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  if (executionRecord.productionAuthority !== true ||
      execution.authority !== "host-owned-browser-worker-execution" ||
      execution.workerExecutionObserved !== true ||
      execution.acceptedTerminalMessages !== "1" ||
      execution.loweringAuthorityMinted !== false ||
      execution.releaseReady !== false ||
      execution.invocationId !== invocation.invocationId ||
      execution.profileHash !== invocation.profileHash ||
      execution.requestId !== invocation.requestId ||
      execution.workerModuleSha256 !== invocation.worker.moduleSha256 ||
      execution.invocationNonceSha256 !== invocation.invocationNonceSha256 ||
      execution.verifierEvidenceRegionSha256 !== lineage.verifierEvidenceRegionSha256 ||
      executionRecord.validatedPackageResult.validationId !==
        executionRecord.validatedResultFrame.validationId ||
      executionRecord.validatedResultFrame.invocationId !== invocation.invocationId ||
      executionRecord.validatedResultFrame.requestId !== invocation.requestId ||
      frameRecord.profile.profileHash !== invocation.profileHash ||
      frameRecord.requestBinding.requestId !== invocation.requestId ||
      frameRecord.requestBinding.bindingId !== executionRecord.validatedResultFrame.requestBindingId ||
      frameRecord.artifact.artifactId !== executionRecord.validatedResultFrame.artifactId ||
      frameRecord.artifact.artifactBytesSha256 !==
        executionRecord.validatedResultFrame.artifactBytesSha256 ||
      frameRecord.artifact.outcome !== executionRecord.validatedResultFrame.outcome ||
      binding.artifact !== artifact ||
      binding.request.profileHash !== profile.profileHash ||
      verifierInspection.evidenceId !== lineage.verifierEvidenceId ||
      verifierInspection.productionConformanceAuthorityMinted !== true ||
      verifierInspection.verifierWorkerExecutionObserved !== true ||
      verifierInspection.rawWasmVerified !== true ||
      verifierInspection.exactInterfaceConformanceObserved !== true ||
      verifierInspection.compilerWorkerExecutionObserved !== false ||
      verifierInspection.loweringAuthorityMinted !== false ||
      verifierInspection.releaseReady !== false) {
    mismatch(
      "$.execution",
      "Worker execution, verifier, frame, request binding, and artifact are not one exact authority chain",
    );
  }
  if (artifact.outcome !== "accepted") {
    mismatch("$.artifact.outcome", "rejected Worker artifacts cannot become semantic candidates");
  }
  const subject = NATIVE_OBJECT_FREEZE({}) as ObservedCppCuteBrowserSemanticSubject;
  weakMapSet(SUBJECTS, subject, NATIVE_OBJECT_FREEZE({
    execution,
    validatedResultFrame: executionRecord.validatedResultFrame,
    artifact,
    profile,
    requestBinding,
    observedWasmConformance: lineage.observedWasmConformance,
    commonLoweringAuthorized: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  }));
  return subject;
}

/** Package-internal registration after kind-specific semantic preparation. */
export function registerObservedCppCuteBrowserSemanticCandidate(
  candidate: ObservedCppCuteBrowserSemanticCandidate,
  subject: ObservedCppCuteBrowserSemanticSubject,
): void {
  if (typeof candidate !== "object" || candidate === null) {
    unverified("$.candidate", "expected an opaque browser semantic candidate");
  }
  const record = weakMapGet(SUBJECTS, subject as object);
  if (record === undefined) {
    unverified("$.subject", "semantic subject was not authenticated by this module instance");
  }
  weakMapSet(CANDIDATES, candidate as object, record);
}

export function unwrapObservedCppCuteBrowserSemanticCandidate(
  candidate: ObservedCppCuteBrowserSemanticCandidate,
): ObservedCppCuteBrowserSemanticCandidateRecord {
  if (typeof candidate !== "object" || candidate === null) {
    unverified("$.candidate", "expected an opaque browser semantic candidate");
  }
  const record = weakMapGet(CANDIDATES, candidate as object);
  if (record === undefined) {
    unverified("$.candidate", "semantic candidate was not created by this module instance");
  }
  return record;
}

export function unwrapObservedCppCuteBrowserSemanticSubject(
  subject: ObservedCppCuteBrowserSemanticSubject,
): ObservedCppCuteBrowserSemanticCandidateRecord {
  if (typeof subject !== "object" || subject === null) {
    unverified("$.subject", "expected an opaque browser semantic subject");
  }
  const record = weakMapGet(SUBJECTS, subject as object);
  if (record === undefined) {
    unverified("$.subject", "semantic subject was not authenticated by this module instance");
  }
  return record;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserSemanticCandidateError("subject-mismatch", path, message);
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserSemanticCandidateError(
    "unverified",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
