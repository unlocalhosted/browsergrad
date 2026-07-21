import { hashCanonicalJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapObservedCppCuteBrowserViewCopyCandidate,
  type ObservedCppCuteBrowserViewCopyCandidate,
} from "./cpp_cute_browser_view_copy_candidate.js";
import type { VerifiedCppCuteBrowserBuildProducer } from
  "./cpp_cute_browser_producer_trust.js";
import {
  CppCuteBrowserSemanticAuthorizationError,
  verifyCppCuteBrowserSemanticCandidateProducerBinding,
} from "./cpp_cute_browser_semantic_authorization.js";
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

declare const authorizedBrowserViewCopyArtifactBrand: unique symbol;

/** Local lowering authority for one source-derived view-copy; no storage or backend authority. */
export interface AuthorizedCppCuteBrowserViewCopyArtifact {
  readonly [authorizedBrowserViewCopyArtifactBrand]: true;
  readonly authority: "browser-worker-view-copy-local-semantic-authorization";
  readonly authorizationId: string;
  readonly candidateId: string;
  readonly producerEvidenceId: string;
  readonly executionEvidenceId: string;
  readonly artifactId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly entryId: string;
  readonly entrySubjectHash: string;
  readonly workerExecutionObserved: true;
  readonly producerTrusted: true;
  readonly localSemanticLoweringAuthorized: true;
  readonly backendExecutionAuthorized: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface AuthorizedCppCuteBrowserViewCopyArtifactRecord {
  readonly candidate: ObservedCppCuteBrowserViewCopyCandidate;
  readonly producer: VerifiedCppCuteBrowserBuildProducer;
  readonly authorization: AuthorizedCppCuteFrontendArtifact;
  readonly backendExecutionAuthorized: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface AuthorizeCppCuteBrowserViewCopyArtifactOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserViewCopyAuthorizationErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-SUBJECT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-UNVERIFIED";

export class CppCuteBrowserViewCopyAuthorizationError extends Error {
  constructor(
    readonly code: CppCuteBrowserViewCopyAuthorizationErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserViewCopyAuthorizationError";
  }
}

const AUTHORIZATIONS = new CAPTURED_WEAK_MAP<
  object,
  AuthorizedCppCuteBrowserViewCopyArtifactRecord
>();

export async function authorizeCppCuteBrowserViewCopyArtifact(
  candidate: ObservedCppCuteBrowserViewCopyCandidate,
  producer: VerifiedCppCuteBrowserBuildProducer,
  options: AuthorizeCppCuteBrowserViewCopyArtifactOptions = {},
): Promise<AuthorizedCppCuteBrowserViewCopyArtifact> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  let candidateRecord: ReturnType<typeof unwrapObservedCppCuteBrowserViewCopyCandidate>;
  try {
    candidateRecord = unwrapObservedCppCuteBrowserViewCopyCandidate(candidate);
  } catch (cause) {
    unverified("$.candidate", "candidate is not an exact observed browser view-copy authority", cause);
  }
  let commonBinding: ReturnType<typeof verifyCppCuteBrowserSemanticCandidateProducerBinding>;
  try {
    commonBinding = verifyCppCuteBrowserSemanticCandidateProducerBinding(candidate, producer);
  } catch (cause) {
    translateSemanticAuthorizationError(cause);
  }
  const semantics = candidateRecord.semantics;
  const artifact = candidateRecord.artifact;
  if (candidate.authority !== "observed-browser-worker-view-copy-semantic-candidate" ||
      candidate.workerExecutionObserved !== true ||
      candidate.artifactOutcome !== "accepted" ||
      candidate.sharedViewCopySemanticsPrepared !== true ||
      candidate.producerTrusted !== false ||
      candidate.loweringAuthorityMinted !== false ||
      candidate.backendExecutionAuthorized !== false ||
      candidate.releaseReady !== false ||
      candidate.entryId !== semantics.entry.entryId ||
      candidate.entrySubjectHash !== semantics.entrySubjectHash ||
      semantics.artifact !== artifact ||
      semantics.loweringAuthorityMinted !== false ||
      artifact.outcome !== "accepted") {
    mismatch("$.candidate", "candidate projection differs from its retained view-copy semantic authority chain");
  }
  if (commonBinding.candidateRecord.execution !== candidateRecord.execution ||
      commonBinding.candidateRecord.validatedResultFrame !== candidateRecord.validatedResultFrame ||
      commonBinding.candidateRecord.artifact !== candidateRecord.artifact ||
      commonBinding.candidateRecord.profile !== candidateRecord.profile ||
      commonBinding.candidateRecord.requestBinding !== candidateRecord.requestBinding ||
      commonBinding.candidateRecord.observedWasmConformance !==
        candidateRecord.observedWasmConformance) {
    mismatch(
      "$.candidate",
      "view-copy candidate and common browser semantic authority are not one exact lineage",
    );
  }
  throwIfAborted(signal);
  const evidenceHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-view-copy-authorization.v1",
    candidateId: candidate.candidateId,
    producerEvidenceId: producer.producerEvidenceId,
    executionEvidenceId: candidate.executionEvidenceId,
    artifactId: candidate.artifactId,
    artifactHash: candidate.artifactHash,
    profileHash: candidate.profileHash,
    requestId: candidate.requestId,
    requestBindingId: candidate.requestBindingId,
    entryId: candidate.entryId,
    entrySubjectHash: candidate.entrySubjectHash,
  });
  throwIfAborted(signal);
  const authorization = issueBrowserCppCuteFrontendArtifactAuthorization({
    candidate,
    producer,
    evidenceHash,
  });
  const authorized = NATIVE_OBJECT_FREEZE({
    authority: "browser-worker-view-copy-local-semantic-authorization",
    authorizationId: `bg.cpp.browser-view-copy-authorization.sha256.${evidenceHash}`,
    candidateId: candidate.candidateId,
    producerEvidenceId: producer.producerEvidenceId,
    executionEvidenceId: candidate.executionEvidenceId,
    artifactId: candidate.artifactId,
    profileHash: candidate.profileHash,
    requestId: candidate.requestId,
    requestBindingId: candidate.requestBindingId,
    entryId: candidate.entryId,
    entrySubjectHash: candidate.entrySubjectHash,
    workerExecutionObserved: true,
    producerTrusted: true,
    localSemanticLoweringAuthorized: true,
    backendExecutionAuthorized: false,
    distributionAuthorized: false,
    releaseReady: false,
  }) as AuthorizedCppCuteBrowserViewCopyArtifact;
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

export function unwrapAuthorizedCppCuteBrowserViewCopyArtifact(
  authorized: AuthorizedCppCuteBrowserViewCopyArtifact,
): AuthorizedCppCuteBrowserViewCopyArtifactRecord {
  if (typeof authorized !== "object" || authorized === null) {
    unverified("$", "expected opaque browser view-copy authorization");
  }
  const record = weakMapGet(AUTHORIZATIONS, authorized as object);
  if (record === undefined) {
    unverified("$", "browser view-copy authorization was not issued by this module instance");
  }
  return record;
}

function normalizeOptions(options: AuthorizeCppCuteBrowserViewCopyArtifactOptions): AbortSignal | undefined {
  if (typeof options !== "object" || options === null) mismatch("$.options", "options must be a plain data record");
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_PROTOTYPE_OF, CAPTURED_OBJECT, [options]) as object | null;
    descriptors = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS, CAPTURED_OBJECT, [options]) as PropertyDescriptorMap;
    keys = NATIVE_REFLECT_APPLY(NATIVE_REFLECT_OWN_KEYS, CAPTURED_REFLECT, [descriptors]) as readonly PropertyKey[];
  } catch (cause) {
    unverified("$.options", "options could not be inspected without invoking accessors", cause);
  }
  if (prototype !== CAPTURED_OBJECT.prototype) mismatch("$.options", "options must be a plain data record");
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
    throw new CppCuteBrowserViewCopyAuthorizationError(
      "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-CANCELLED",
      "$.signal",
      "browser view-copy authorization was cancelled",
    );
  }
}

function translateSemanticAuthorizationError(cause: unknown): never {
  if (cause instanceof CppCuteBrowserSemanticAuthorizationError) {
    if (cause.kind === "subject-mismatch") mismatch(cause.path, cause.message);
    unverified(cause.path, cause.message, cause);
  }
  unverified("$", "browser semantic candidate and producer binding could not be verified", cause);
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserViewCopyAuthorizationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-SUBJECT-MISMATCH",
    path,
    message,
  );
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserViewCopyAuthorizationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-UNVERIFIED",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
