import {
  hashCanonicalJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
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
import {
  prepareVerifiedCppCuteLayoutSemantics,
  type LowerAuthorizedCppCuteLayoutEntryOptions,
  type LowerAuthorizedCppCuteLayoutEntryRequest,
  type PreparedVerifiedCppCuteLayoutSemantics,
} from "./cpp_cute_layout_lowering.js";

const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_ARRAY_IS_ARRAY = Array.isArray;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const CAPTURED_OBJECT = Object;
const CAPTURED_OBJECT_PROTOTYPE = Object.prototype;
const CAPTURED_WEAK_MAP = WeakMap;

declare const observedLayoutCandidateBrand: unique symbol;

/**
 * One authenticated Worker result whose selected layout has been prepared by
 * the shared semantic-core layout seam. This remains an inspection candidate:
 * build-producer trust, common lowering, backend execution, and release are
 * deliberately false.
 */
export interface ObservedCppCuteBrowserLayoutCandidate {
  readonly [observedLayoutCandidateBrand]: true;
  readonly authority: "observed-browser-worker-layout-semantic-candidate";
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
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly workerExecutionObserved: true;
  readonly artifactOutcome: "accepted";
  readonly sharedLayoutSemanticsPrepared: true;
  readonly producerTrusted: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionAuthorized: false;
  readonly releaseReady: false;
}

export interface ObservedCppCuteBrowserLayoutCandidateRecord {
  readonly execution: ObservedCppCuteBrowserWorkerExecution;
  readonly validatedResultFrame: ValidatedCppCuteBrowserWorkerResultFrame;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
  readonly observedWasmConformance: ObservedCppCuteBrowserPackageWasmConformance;
  readonly semantics: PreparedVerifiedCppCuteLayoutSemantics;
  readonly commonLoweringAuthorized: false;
  readonly backendExecutionAuthorized: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserLayoutCandidateErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-SUBJECT-MISMATCH";

export class CppCuteBrowserLayoutCandidateError extends Error {
  constructor(
    readonly code: CppCuteBrowserLayoutCandidateErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserLayoutCandidateError";
  }
}

const CANDIDATES = new CAPTURED_WEAK_MAP<
  object,
  ObservedCppCuteBrowserLayoutCandidateRecord
>();

/**
 * Re-authenticates the complete host-observed Worker lineage and prepares its
 * one accepted static layout through the same shared seam used by authorized
 * lowering. It cannot mint an AuthorizedCppCuteFrontendArtifact.
 */
export async function prepareObservedCppCuteBrowserLayoutCandidate(
  execution: ObservedCppCuteBrowserWorkerExecution,
  request: LowerAuthorizedCppCuteLayoutEntryRequest,
  options: LowerAuthorizedCppCuteLayoutEntryOptions = {},
): Promise<ObservedCppCuteBrowserLayoutCandidate> {
  const initialSignal = inspectCandidateSignal(options);
  throwIfAborted(initialSignal);
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
    mismatch("$.artifact.outcome", "rejected Worker artifacts cannot become layout candidates");
  }
  throwIfAborted(options.signal);
  const semantics = await prepareVerifiedCppCuteLayoutSemantics(artifact, request, options);
  throwIfAborted(options.signal);
  const candidateHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-worker-layout-candidate.v1",
    executionEvidenceId: execution.evidenceId,
    invocationId: execution.invocationId,
    profileHash: execution.profileHash,
    requestId: execution.requestId,
    requestBindingId: requestBinding.bindingId,
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    entryId: semantics.entry.entryId,
    layoutSemanticHash: semantics.preparedLayout.layoutSemanticHash,
    indexMapId: semantics.preparedLayout.indexMapId,
  });
  throwIfAborted(options.signal);
  const candidate = NATIVE_OBJECT_FREEZE({
    authority: "observed-browser-worker-layout-semantic-candidate",
    candidateId: `bg.cpp.browser-worker-layout-candidate.sha256.${candidateHash}`,
    executionEvidenceId: execution.evidenceId,
    invocationId: execution.invocationId,
    profileHash: execution.profileHash,
    requestId: execution.requestId,
    requestBindingId: requestBinding.bindingId,
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    entryId: semantics.entry.entryId,
    layoutSemanticHash: semantics.preparedLayout.layoutSemanticHash,
    indexMapId: semantics.preparedLayout.indexMapId,
    coordinateRank: semantics.preparedLayout.coordinateRank,
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    sharedLayoutSemanticsPrepared: true,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  }) as ObservedCppCuteBrowserLayoutCandidate;
  weakMapSet(CANDIDATES, candidate, NATIVE_OBJECT_FREEZE({
    execution,
    validatedResultFrame: executionRecord.validatedResultFrame,
    artifact,
    profile,
    requestBinding,
    observedWasmConformance: lineage.observedWasmConformance,
    semantics,
    commonLoweringAuthorized: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  }));
  return candidate;
}

export function unwrapObservedCppCuteBrowserLayoutCandidate(
  candidate: ObservedCppCuteBrowserLayoutCandidate,
): ObservedCppCuteBrowserLayoutCandidateRecord {
  if (typeof candidate !== "object" || candidate === null) {
    unverified("$.candidate", "expected opaque observed layout candidate");
  }
  const record = weakMapGet(CANDIDATES, candidate as object);
  if (record === undefined) {
    unverified("$.candidate", "layout candidate was not created by this module instance");
  }
  return record;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

/** Reads only a plain enumerable data-property signal; full option validation stays in the lowerer. */
function inspectCandidateSignal(options: LowerAuthorizedCppCuteLayoutEntryOptions): AbortSignal | undefined {
  if (typeof options !== "object" || options === null || NATIVE_ARRAY_IS_ARRAY(options)) return undefined;
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
  if (signal?.aborted === true) {
    throw new CppCuteBrowserLayoutCandidateError(
      "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-CANCELLED",
      "$.signal",
      "observed Worker layout candidate preparation was cancelled",
    );
  }
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserLayoutCandidateError(
    "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-SUBJECT-MISMATCH",
    path,
    message,
  );
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserLayoutCandidateError(
    "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-UNVERIFIED",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
