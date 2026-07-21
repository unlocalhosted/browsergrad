import {
  hashCanonicalJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { ObservedCppCuteBrowserWorkerExecution } from
  "./cpp_cute_browser_worker_controller.js";
import {
  CppCuteBrowserSemanticCandidateError,
  authenticateObservedCppCuteBrowserSemanticSubject,
  registerObservedCppCuteBrowserSemanticCandidate,
  unwrapObservedCppCuteBrowserSemanticSubject,
  type ObservedCppCuteBrowserSemanticCandidate,
  type ObservedCppCuteBrowserSemanticCandidateRecord,
} from "./cpp_cute_browser_semantic_candidate.js";
import {
  CppCuteViewCopyLoweringError,
  prepareVerifiedCppCuteViewCopySemantics,
  type PrepareVerifiedCppCuteViewCopySemanticsOptions,
  type PrepareVerifiedCppCuteViewCopySemanticsRequest,
  type PreparedVerifiedCppCuteViewCopySemantics,
} from "./cpp_cute_view_copy_semantics.js";

const CAPTURED_OBJECT = Object;
const CAPTURED_OBJECT_PROTOTYPE = Object.prototype;
const CAPTURED_WEAK_MAP = WeakMap;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_ARRAY_IS_ARRAY = Array.isArray;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

declare const observedViewCopyCandidateBrand: unique symbol;

/**
 * One exact observed Worker result whose selected view-copy source semantics
 * are supported. Host allocation capacities and offsets are deliberately not
 * part of this producer-evidence candidate.
 */
export interface ObservedCppCuteBrowserViewCopyCandidate
  extends ObservedCppCuteBrowserSemanticCandidate {
  readonly [observedViewCopyCandidateBrand]: true;
  readonly authority: "observed-browser-worker-view-copy-semantic-candidate";
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
  readonly entryId: string;
  readonly entrySubjectHash: string;
  readonly workerExecutionObserved: true;
  readonly artifactOutcome: "accepted";
  readonly sharedViewCopySemanticsPrepared: true;
  readonly producerTrusted: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionAuthorized: false;
  readonly releaseReady: false;
}

export interface ObservedCppCuteBrowserViewCopyCandidateRecord
  extends ObservedCppCuteBrowserSemanticCandidateRecord {
  readonly semantics: PreparedVerifiedCppCuteViewCopySemantics;
}

export type PrepareObservedCppCuteBrowserViewCopyCandidateRequest =
  PrepareVerifiedCppCuteViewCopySemanticsRequest;
export type PrepareObservedCppCuteBrowserViewCopyCandidateOptions =
  PrepareVerifiedCppCuteViewCopySemanticsOptions;

export type CppCuteBrowserViewCopyCandidateErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-SUBJECT-MISMATCH";

export class CppCuteBrowserViewCopyCandidateError extends Error {
  constructor(
    readonly code: CppCuteBrowserViewCopyCandidateErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserViewCopyCandidateError";
  }
}

const CANDIDATES = new CAPTURED_WEAK_MAP<
  object,
  ObservedCppCuteBrowserViewCopyCandidateRecord
>();

export async function prepareObservedCppCuteBrowserViewCopyCandidate(
  execution: ObservedCppCuteBrowserWorkerExecution,
  request: PrepareObservedCppCuteBrowserViewCopyCandidateRequest,
  options: PrepareObservedCppCuteBrowserViewCopyCandidateOptions = {},
): Promise<ObservedCppCuteBrowserViewCopyCandidate> {
  const signal = inspectCandidateSignal(options);
  throwIfAborted(signal);
  let subject: ReturnType<typeof authenticateObservedCppCuteBrowserSemanticSubject>;
  let subjectRecord: ReturnType<typeof unwrapObservedCppCuteBrowserSemanticSubject>;
  try {
    subject = authenticateObservedCppCuteBrowserSemanticSubject(execution);
    subjectRecord = unwrapObservedCppCuteBrowserSemanticSubject(subject);
  } catch (cause) {
    translateSemanticCandidateError(cause);
  }
  let semantics: PreparedVerifiedCppCuteViewCopySemantics;
  try {
    semantics = await prepareVerifiedCppCuteViewCopySemantics(
      subjectRecord.artifact,
      request,
      options,
    );
  } catch (cause) {
    if (cause instanceof CppCuteViewCopyLoweringError &&
        cause.code === "BG-COMPILER-CPP-CUTE-VIEW-COPY-CANCELLED") {
      throwIfAborted(signal);
    }
    throw cause;
  }
  throwIfAborted(signal);
  const candidateHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-worker-view-copy-candidate.v1",
    executionEvidenceId: execution.evidenceId,
    invocationId: execution.invocationId,
    profileHash: execution.profileHash,
    requestId: execution.requestId,
    requestBindingId: subjectRecord.requestBinding.bindingId,
    artifactId: subjectRecord.artifact.artifactId,
    artifactHash: subjectRecord.artifact.artifactHash,
    artifactBytesSha256: subjectRecord.artifact.artifactBytesSha256,
    artifactByteLength: subjectRecord.artifact.artifactByteLength,
    entryId: semantics.entry.entryId,
    entrySubjectHash: semantics.entrySubjectHash,
  });
  throwIfAborted(signal);
  const candidate = NATIVE_OBJECT_FREEZE({
    authority: "observed-browser-worker-view-copy-semantic-candidate",
    candidateId: `bg.cpp.browser-worker-view-copy-candidate.sha256.${candidateHash}`,
    executionEvidenceId: execution.evidenceId,
    invocationId: execution.invocationId,
    profileHash: execution.profileHash,
    requestId: execution.requestId,
    requestBindingId: subjectRecord.requestBinding.bindingId,
    artifactId: subjectRecord.artifact.artifactId,
    artifactHash: subjectRecord.artifact.artifactHash,
    artifactBytesSha256: subjectRecord.artifact.artifactBytesSha256,
    artifactByteLength: subjectRecord.artifact.artifactByteLength,
    entryId: semantics.entry.entryId,
    entrySubjectHash: semantics.entrySubjectHash,
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    sharedViewCopySemanticsPrepared: true,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  }) as ObservedCppCuteBrowserViewCopyCandidate;
  weakMapSet(CANDIDATES, candidate, NATIVE_OBJECT_FREEZE({
    ...subjectRecord,
    semantics,
  }));
  registerObservedCppCuteBrowserSemanticCandidate(candidate, subject);
  return candidate;
}

export function unwrapObservedCppCuteBrowserViewCopyCandidate(
  candidate: ObservedCppCuteBrowserViewCopyCandidate,
): ObservedCppCuteBrowserViewCopyCandidateRecord {
  if (typeof candidate !== "object" || candidate === null) {
    unverified("$.candidate", "expected opaque observed view-copy candidate");
  }
  const record = weakMapGet(CANDIDATES, candidate as object);
  if (record === undefined) {
    unverified("$.candidate", "view-copy candidate was not created by this module instance");
  }
  return record;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function inspectCandidateSignal(
  options: PrepareVerifiedCppCuteViewCopySemanticsOptions,
): AbortSignal | undefined {
  if (typeof options !== "object" || options === null || NATIVE_ARRAY_IS_ARRAY(options)) {
    return undefined;
  }
  let prototype: object | null;
  let descriptor: PropertyDescriptor | undefined;
  try {
    prototype = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_PROTOTYPE_OF, CAPTURED_OBJECT, [options]) as
      object | null;
    descriptor = NATIVE_REFLECT_APPLY(
      NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTOR,
      CAPTURED_OBJECT,
      [options, "signal"],
    ) as PropertyDescriptor | undefined;
  } catch {
    return undefined;
  }
  if (prototype !== CAPTURED_OBJECT_PROTOTYPE && prototype !== null) return undefined;
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    return undefined;
  }
  return descriptor.value as AbortSignal | undefined;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined || ABORT_SIGNAL_ABORTED_GETTER === undefined) return;
  let aborted: unknown;
  try {
    aborted = NATIVE_REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []);
  } catch {
    return;
  }
  if (aborted === true) {
    throw new CppCuteBrowserViewCopyCandidateError(
      "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-CANCELLED",
      "$.signal",
      "observed Worker view-copy candidate preparation was cancelled",
    );
  }
}

function translateSemanticCandidateError(cause: unknown): never {
  if (cause instanceof CppCuteBrowserSemanticCandidateError) {
    if (cause.kind === "subject-mismatch") mismatch(cause.path, cause.message);
    unverified(cause.path, cause.message, cause);
  }
  unverified("$.execution", "expected the exact host-observed Worker authority chain", cause);
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserViewCopyCandidateError(
    "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-SUBJECT-MISMATCH",
    path,
    message,
  );
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserViewCopyCandidateError(
    "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-UNVERIFIED",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
