import {
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
  type CppCuteBrowserWorkerBundleInspection,
} from "./cpp_cute_browser_worker_bundle.js";
import {
  copyCppCuteBrowserWorkerModuleBytes,
  discardCppCuteBrowserWorkerInvocation,
  prepareCppCuteBrowserWorkerInvocation,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  unwrapValidatedCppCuteBrowserWorkerResultFrame,
  validateCppCuteBrowserWorkerResultFrame,
  type CppCuteBrowserWorkerInvocationV1,
  type CppCuteBrowserWorkerInvocationDiscardReason,
  type PrepareCppCuteBrowserWorkerInvocationInput,
  type PreparedCppCuteBrowserWorkerInvocation,
  type ValidatedCppCuteBrowserWorkerResultFrame,
} from "./cpp_cute_browser_worker_protocol.js";
import {
  discardCppCuteBrowserWorkerTransfer,
  prepareCppCuteBrowserWorkerTransfer,
  takeCppCuteBrowserWorkerTransfer,
  type PreparedCppCuteBrowserWorkerTransfer,
  type TakenCppCuteBrowserWorkerTransfer,
} from "./cpp_cute_browser_worker_transfer.js";
import { unwrapPreparedCppCuteFrontendRequest } from "./cpp_cute_frontend_request.js";
import { unwrapPreparedCppCuteFrontendRequestBinding } from "./cpp_cute_frontend_request_binding.js";
import { unwrapPreparedCppCuteBrowserFrontendProfile } from "./cpp_cute_frontend_profile.js";
import {
  unwrapVerifiedCppCuteBrowserVfsInstallation,
} from "./cpp_cute_browser_asset_installation.js";
import {
  prepareCppCuteBrowserWasmVerifierEvidence,
} from "./cpp_cute_browser_wasm_verifier_evidence.js";
import {
  inspectObservedCppCuteBrowserPackageWasmConformance,
  unwrapObservedCppCuteBrowserPackageWasmConformance,
  type ObservedCppCuteBrowserPackageWasmConformance,
} from "./cpp_cute_browser_wasm_verifier_controller.js";

const INPUT_KEYS = Object.freeze([
  "profile",
  "assetManifest",
  "vfsInstallation",
  "request",
  "runtimeAbiAsset",
  "observedWasmConformance",
] as const);
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_CREATE = Object.create;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_SET_HAS = Set.prototype.has;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const INPUT_KEY_SET = new Set<PropertyKey>(INPUT_KEYS);
const PACKAGE_INVOCATIONS = new WeakMap<object, StoredPackageInvocation>();
const VALIDATED_PACKAGE_RESULTS =
  new WeakMap<object, ValidatedCppCuteBrowserPackageInvocationResultRecord>();

export type PrepareCppCuteBrowserPackageInvocationInput = Omit<
  PrepareCppCuteBrowserWorkerInvocationInput,
  "workerModuleBytes" | "verifierEvidence"
> & {
  readonly observedWasmConformance: ObservedCppCuteBrowserPackageWasmConformance;
};

declare const preparedPackageInvocationBrand: unique symbol;

/**
 * Single-use host invocation whose executable JavaScript came only from the
 * exact package Worker bundle. It does not itself observe Worker execution.
 */
export interface PreparedCppCuteBrowserPackageInvocation {
  readonly [preparedPackageInvocationBrand]: true;
  readonly authority: "package-owned-worker-invocation";
  readonly invocationId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly verifierEvidenceRegionSha256: string;
  readonly workerModuleSha256: string;
  readonly workerModuleByteLength: number;
  readonly maxWallTimeMs: number;
  readonly maxArtifactByteLength: number;
  readonly packageWorkerVerified: true;
  readonly callerExecutableBytesAccepted: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface TakenCppCuteBrowserPackageInvocation {
  readonly workerModuleBytes: Uint8Array;
  readonly transfer: TakenCppCuteBrowserWorkerTransfer;
}

/** Compact identity-only lineage retained after heavy protocol state is released. */
export interface CppCuteBrowserPackageInvocationLineage {
  readonly invocationHash: string;
  readonly invocation: CppCuteBrowserWorkerInvocationV1;
  readonly workerBundle: CppCuteBrowserWorkerBundleInspection;
  readonly observedWasmConformance: ObservedCppCuteBrowserPackageWasmConformance;
  readonly verifierEvidenceId: string;
  readonly verifierEvidenceRegionSha256: string;
}

declare const validatedPackageResultBrand: unique symbol;

/**
 * Package-layer composition of one exact protocol validation and its compact
 * host invocation lineage. It proves neither Worker execution nor lowering.
 */
export interface ValidatedCppCuteBrowserPackageInvocationResult {
  readonly [validatedPackageResultBrand]: true;
  readonly authority: "package-owned-worker-result-validation";
  readonly validationId: string;
  readonly invocationId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly workerModuleSha256: string;
  readonly packageWorkerVerified: true;
  readonly protocolResultValidated: true;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface ValidatedCppCuteBrowserPackageInvocationResultRecord {
  readonly validatedResultFrame: ValidatedCppCuteBrowserWorkerResultFrame;
  readonly lineage: CppCuteBrowserPackageInvocationLineage;
}

type PackageInvocationState = "prepared" | "taken" | "consumed";

interface StoredPackageInvocation {
  state: PackageInvocationState;
  readonly lineage: CppCuteBrowserPackageInvocationLineage;
  invocation: PreparedCppCuteBrowserWorkerInvocation | null;
  transfer: PreparedCppCuteBrowserWorkerTransfer | null;
}

export type CppCuteBrowserPackageInvocationErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-UNVERIFIED";

export class CppCuteBrowserPackageInvocationError extends Error {
  constructor(
    readonly code: CppCuteBrowserPackageInvocationErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserPackageInvocationError";
  }
}

/**
 * Verifies the checked package bundle, injects a fresh copy into the protocol
 * preparation seam, and reserves exactly one canonical transfer.
 */
export async function prepareCppCuteBrowserPackageInvocation(
  input: PrepareCppCuteBrowserPackageInvocationInput,
): Promise<PreparedCppCuteBrowserPackageInvocation> {
  const values = exactInput(input);
  const profile = values.profile;
  const request = values.request;
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const effectiveLimits = {
    ...profileRecord.profile.extractionLimits,
    ...requestRecord.request.limits,
  };
  const installation = unwrapVerifiedCppCuteBrowserVfsInstallation(values.vfsInstallation);
  const verifierEvidence = await prepareCppCuteBrowserWasmVerifierEvidence(
    values.observedWasmConformance,
    {
      assetSet: installation.assetSet,
      assetManifest: values.assetManifest,
      runtimeAbiAsset: values.runtimeAbiAsset,
    },
  );
  const bundle = await verifyCppCuteBrowserWorkerBundle();
  const inspection = inspectVerifiedCppCuteBrowserWorkerBundle(bundle);
  const invocation = await prepareCppCuteBrowserWorkerInvocation({
    profile: values.profile,
    assetManifest: values.assetManifest,
    vfsInstallation: values.vfsInstallation,
    request: values.request,
    runtimeAbiAsset: values.runtimeAbiAsset,
    verifierEvidence,
    workerModuleBytes: copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle),
  });
  const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  const lineage = NATIVE_OBJECT_FREEZE({
    invocationHash: invocation.invocationHash,
    invocation: invocationRecord.invocation,
    workerBundle: inspection,
    observedWasmConformance: values.observedWasmConformance,
    verifierEvidenceId: verifierEvidence.sourceEvidenceId,
    verifierEvidenceRegionSha256: verifierEvidence.regionSha256,
  }) as CppCuteBrowserPackageInvocationLineage;
  let transfer: PreparedCppCuteBrowserWorkerTransfer;
  try {
    transfer = prepareCppCuteBrowserWorkerTransfer(invocation);
  } catch (cause) {
    discardCppCuteBrowserWorkerInvocation(invocation, "abandoned");
    throw cause;
  }
  const prepared = NATIVE_OBJECT_FREEZE({
    authority: "package-owned-worker-invocation",
    invocationId: invocation.invocationId,
    profileHash: invocation.profileHash,
    requestId: invocation.requestId,
    invocationNonceSha256: invocationRecord.invocation.invocationNonceSha256,
    verifierEvidenceRegionSha256: invocationRecord.invocation.verifierEvidenceRegionSha256,
    workerModuleSha256: inspection.sha256,
    workerModuleByteLength: inspection.byteLength,
    maxWallTimeMs: effectiveLimits.maxWallTimeMs,
    maxArtifactByteLength: effectiveLimits.maxOutputBytes,
    packageWorkerVerified: true,
    callerExecutableBytesAccepted: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
  }) as PreparedCppCuteBrowserPackageInvocation;
  weakMapSet(PACKAGE_INVOCATIONS, prepared, {
    state: "prepared",
    lineage,
    invocation,
    transfer,
  });
  return prepared;
}

/** Materializes the package Worker and canonical transfer exactly once. */
export function takeCppCuteBrowserPackageInvocation(
  prepared: PreparedCppCuteBrowserPackageInvocation,
): TakenCppCuteBrowserPackageInvocation {
  const stored = storedPackageInvocation(prepared);
  if (stored.state !== "prepared") state("$.prepared", "package invocation was already taken or consumed");
  const invocation = requiredInvocation(stored);
  const reservedTransfer = requiredTransfer(stored);
  let workerModuleBytes: Uint8Array;
  let transfer: TakenCppCuteBrowserWorkerTransfer;
  try {
    workerModuleBytes = copyCppCuteBrowserWorkerModuleBytes(invocation);
    transfer = takeCppCuteBrowserWorkerTransfer(reservedTransfer);
  } catch (cause) {
    stored.state = "consumed";
    stored.invocation = null;
    stored.transfer = null;
    try {
      discardCppCuteBrowserWorkerInvocation(invocation, "worker-unavailable");
    } catch (cleanupCause) {
      throw new AggregateError(
        [cause, cleanupCause],
        "package Worker launch materialization and invocation cleanup both failed",
      );
    }
    throw cause;
  }
  stored.state = "taken";
  stored.transfer = null;
  return NATIVE_OBJECT_FREEZE({ workerModuleBytes, transfer });
}

/** Terminal validation is permitted only after the controller took the launch. */
export async function validateCppCuteBrowserPackageInvocationResult(
  prepared: PreparedCppCuteBrowserPackageInvocation,
  controlBytes: Uint8Array,
  artifactBytes: Uint8Array,
): Promise<ValidatedCppCuteBrowserPackageInvocationResult> {
  const stored = storedPackageInvocation(prepared);
  if (stored.state !== "taken") state("$.prepared", "package invocation has no active taken launch");
  const invocation = requiredInvocation(stored);
  const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  stored.state = "consumed";
  stored.invocation = null;
  const validatedResultFrame = await validateCppCuteBrowserWorkerResultFrame(
    invocation,
    controlBytes,
    artifactBytes,
  );
  validateProtocolResultLineage(
    prepared,
    stored.lineage,
    invocationRecord,
    validatedResultFrame,
  );
  const validated = NATIVE_OBJECT_FREEZE({
    authority: "package-owned-worker-result-validation",
    validationId: validatedResultFrame.validationId,
    invocationId: prepared.invocationId,
    profileHash: prepared.profileHash,
    requestId: prepared.requestId,
    workerModuleSha256: prepared.workerModuleSha256,
    packageWorkerVerified: true,
    protocolResultValidated: true,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
  }) as ValidatedCppCuteBrowserPackageInvocationResult;
  weakMapSet(VALIDATED_PACKAGE_RESULTS, validated, NATIVE_OBJECT_FREEZE({
    validatedResultFrame,
    lineage: stored.lineage,
  }));
  return validated;
}

export function unwrapValidatedCppCuteBrowserPackageInvocationResult(
  validated: ValidatedCppCuteBrowserPackageInvocationResult,
): ValidatedCppCuteBrowserPackageInvocationResultRecord {
  if (typeof validated !== "object" || validated === null) unverifiedResult();
  const record = weakMapGet(VALIDATED_PACKAGE_RESULTS, validated as object);
  if (record === undefined) unverifiedResult();
  return record;
}

/** Releases either a reserved or already-taken invocation without minting authority. */
export function discardCppCuteBrowserPackageInvocation(
  prepared: PreparedCppCuteBrowserPackageInvocation,
  reason: CppCuteBrowserWorkerInvocationDiscardReason,
): void {
  const stored = storedPackageInvocation(prepared);
  if (stored.state === "consumed") state("$.prepared", "package invocation was already consumed");
  const priorState = stored.state;
  const invocation = requiredInvocation(stored);
  const transfer = stored.transfer;
  stored.state = "consumed";
  stored.invocation = null;
  stored.transfer = null;
  if (priorState === "prepared") {
    if (transfer === null) state("$.prepared", "prepared package invocation lost its transfer reservation");
    discardCppCuteBrowserWorkerTransfer(transfer, reason);
  } else {
    discardCppCuteBrowserWorkerInvocation(invocation, reason);
  }
}

function exactInput(
  input: PrepareCppCuteBrowserPackageInvocationInput,
): PrepareCppCuteBrowserPackageInvocationInput {
  if (typeof input !== "object" || input === null ||
      nativeGetPrototypeOf(input) !== Object.prototype) {
    invalid("$.input", "expected one plain input data record");
  }
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    descriptors = nativeGetOwnPropertyDescriptors(input);
    keys = nativeReflectOwnKeys(descriptors);
  } catch (cause) {
    invalid("$.input", "input fields are not safely inspectable", { cause });
  }
  let validKeys = keys.length === INPUT_KEYS.length;
  for (let index = 0; validKeys && index < keys.length; index += 1) {
    validKeys = nativeSetHas(INPUT_KEY_SET, keys[index]);
  }
  if (!validKeys) {
    invalid(
      "$.input",
      "expected exactly fields profile, assetManifest, vfsInstallation, request, runtimeAbiAsset, observedWasmConformance",
    );
  }
  const result = nativeObjectCreate(null) as Record<string, unknown>;
  for (let index = 0; index < INPUT_KEYS.length; index += 1) {
    const key = INPUT_KEYS[index]!;
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`$.input.${key}`, "field must be one enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return result as unknown as PrepareCppCuteBrowserPackageInvocationInput;
}

function storedPackageInvocation(
  prepared: PreparedCppCuteBrowserPackageInvocation,
): StoredPackageInvocation {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = weakMapGet(PACKAGE_INVOCATIONS, prepared as object);
  if (stored === undefined) unverified();
  return stored;
}

function requiredInvocation(stored: StoredPackageInvocation): PreparedCppCuteBrowserWorkerInvocation {
  if (stored.invocation === null) state("$.prepared", "package invocation lost its protocol authority");
  return stored.invocation;
}

function requiredTransfer(stored: StoredPackageInvocation): PreparedCppCuteBrowserWorkerTransfer {
  if (stored.transfer === null) state("$.prepared", "package invocation lost its transfer reservation");
  return stored.transfer;
}

function validateProtocolResultLineage(
  prepared: PreparedCppCuteBrowserPackageInvocation,
  lineage: CppCuteBrowserPackageInvocationLineage,
  invocationRecord: ReturnType<typeof unwrapPreparedCppCuteBrowserWorkerInvocation>,
  frame: ValidatedCppCuteBrowserWorkerResultFrame,
): void {
  let frameRecord: ReturnType<typeof unwrapValidatedCppCuteBrowserWorkerResultFrame>;
  let requestBindingRecord: ReturnType<typeof unwrapPreparedCppCuteFrontendRequestBinding>;
  try {
    frameRecord = unwrapValidatedCppCuteBrowserWorkerResultFrame(frame);
    requestBindingRecord = unwrapPreparedCppCuteFrontendRequestBinding(
      frameRecord.requestBinding,
    );
  } catch (cause) {
    invalid(
      "$.validatedResultFrame",
      "protocol validation did not retain its exact opaque result authority chain",
      { cause },
    );
  }
  const invocation = lineage.invocation;
  const worker = lineage.workerBundle;
  const manifest = frameRecord.assetManifest;
  const request = requestBindingRecord.request;
  let observedInspection: ReturnType<
    typeof inspectObservedCppCuteBrowserPackageWasmConformance
  >;
  try {
    observedInspection = inspectObservedCppCuteBrowserPackageWasmConformance(
      lineage.observedWasmConformance,
    );
    unwrapObservedCppCuteBrowserPackageWasmConformance(
      lineage.observedWasmConformance,
    );
  } catch (cause) {
    invalid(
      "$.lineage.observedWasmConformance",
      "package lineage lost its exact host verifier authority",
      { cause },
    );
  }
  requireLineage(
    invocationRecord.invocation === invocation,
    "$.lineage.invocation",
    "compact lineage does not retain the exact canonical protocol invocation",
  );
  requireLineage(
    frameRecord.profile === invocationRecord.profile,
    "$.validatedResultFrame.profile",
    "protocol result profile is not the exact prepared invocation authority",
  );
  requireLineage(
    manifest === invocationRecord.assetManifest,
    "$.validatedResultFrame.assetManifest",
    "protocol result manifest is not the exact prepared invocation authority",
  );
  requireLineage(
    request === invocationRecord.request,
    "$.validatedResultFrame.requestBinding.request",
    "protocol result request is not the exact prepared invocation authority",
  );
  requireLineage(
    invocation.invocationId === prepared.invocationId &&
      invocation.invocationId ===
        `bg.cpp.browser-worker-invocation.sha256.${lineage.invocationHash}`,
    "$.lineage.invocation.invocationId",
    "compact invocation identity differs from the package invocation",
  );
  requireLineage(
    invocation.profileHash === prepared.profileHash &&
      frameRecord.profile.profileHash === invocation.profileHash,
    "$.lineage.invocation.profileHash",
    "compact invocation profile differs from the protocol result",
  );
  requireLineage(
    invocation.requestId === prepared.requestId &&
      frame.requestId === invocation.requestId &&
      frameRecord.requestBinding.requestId === invocation.requestId &&
      request.requestId === invocation.requestId,
    "$.lineage.invocation.requestId",
    "compact invocation request differs from the protocol result",
  );
  requireLineage(
    request.profileHash === invocation.profileHash,
    "$.validatedResultFrame.requestBinding.request.profileHash",
    "exact request profile differs from the compact invocation",
  );
  requireLineage(
    invocation.invocationNonceSha256 === prepared.invocationNonceSha256,
    "$.lineage.invocation.invocationNonceSha256",
    "compact invocation nonce differs from the package invocation",
  );
  requireLineage(
    invocation.verifierEvidenceId === lineage.verifierEvidenceId &&
      invocation.verifierEvidenceRegionSha256 === lineage.verifierEvidenceRegionSha256 &&
      observedInspection.evidenceId === lineage.verifierEvidenceId,
    "$.lineage.invocation.verifierEvidenceId",
    "compact invocation verifier evidence differs from the exact host authority",
  );
  requireLineage(
    manifest.profileHash === invocation.profileHash &&
      manifest.manifestId === invocation.assetManifestId &&
      manifest.manifestSha256 === invocation.assetManifestSha256 &&
      manifest.assetSetSha256 === invocation.assetSetSha256,
    "$.lineage.invocation.assetManifestId",
    "compact invocation manifest differs from the exact protocol manifest",
  );
  requireLineage(
    invocation.worker.moduleSha256 === prepared.workerModuleSha256 &&
      invocation.worker.moduleByteLength === String(prepared.workerModuleByteLength) &&
      worker.sha256 === prepared.workerModuleSha256 &&
      worker.byteLength === prepared.workerModuleByteLength,
    "$.lineage.invocation.worker",
    "compact invocation Worker differs from the exact package Worker",
  );
  requireLineage(
    worker.staticImportCount === 0 && worker.dynamicImportCount === 0 &&
      worker.packageOwned === true && worker.exactBytesVerified === true &&
      worker.selfContainedModuleGraph === true &&
      worker.workerExecutionObserved === false && worker.releaseReady === false,
    "$.lineage.workerBundle",
    "compact Worker inspection exceeds package bundle authority",
  );
  requireLineage(
    frame.invocationId === invocation.invocationId,
    "$.validatedResultFrame.invocationId",
    "protocol result belongs to a different compact invocation",
  );
  requireLineage(
    frame.requestBindingId === frameRecord.requestBinding.bindingId,
    "$.validatedResultFrame.requestBindingId",
    "protocol result request binding differs from its opaque authority",
  );
  requireLineage(
    frame.artifactId === frameRecord.artifact.artifactId &&
      frame.artifactBytesSha256 === frameRecord.artifact.artifactBytesSha256 &&
      frame.outcome === frameRecord.artifact.outcome,
    "$.validatedResultFrame.artifactId",
    "protocol result artifact differs from its opaque authority",
  );
}

function requireLineage(condition: boolean, path: string, message: string): void {
  if (!condition) invalid(path, message);
}

function nativeGetPrototypeOf(value: object): object | null {
  return Reflect.apply(NATIVE_GET_PROTOTYPE_OF, Object, [value]) as object | null;
}

function nativeGetOwnPropertyDescriptors(value: object): PropertyDescriptorMap {
  return Reflect.apply(NATIVE_GET_OWN_PROPERTY_DESCRIPTORS, Object, [value]) as PropertyDescriptorMap;
}

function nativeReflectOwnKeys(value: object): readonly PropertyKey[] {
  return Reflect.apply(NATIVE_REFLECT_OWN_KEYS, Reflect, [value]) as readonly PropertyKey[];
}

function nativeObjectCreate(prototype: object | null): object {
  return Reflect.apply(NATIVE_OBJECT_CREATE, Object, [prototype]) as object;
}

function nativeSetHas(values: Set<PropertyKey>, value: PropertyKey | undefined): boolean {
  return Reflect.apply(NATIVE_SET_HAS, values, [value]) as boolean;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return Reflect.apply(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  Reflect.apply(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserPackageInvocationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
    path,
    message,
    options,
  );
}

function state(path: string, message: string): never {
  throw new CppCuteBrowserPackageInvocationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-STATE",
    path,
    message,
  );
}

function unverified(): never {
  throw new CppCuteBrowserPackageInvocationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-UNVERIFIED",
    "$.prepared",
    "expected opaque package invocation authority",
  );
}

function unverifiedResult(): never {
  throw new CppCuteBrowserPackageInvocationError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-UNVERIFIED",
    "$.validatedPackageResult",
    "expected opaque package-issued result validation authority",
  );
}
