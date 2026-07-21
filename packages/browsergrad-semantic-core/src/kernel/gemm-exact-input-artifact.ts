import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { sha256Hex } from "../schema/hash.js";
import { encodeWireU64, parseWireU64, type WireI64, type WireU64 } from "../schema/integers.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import type { VerifiedLogicalGemmTileArtifact } from "./gemm-tile-artifact.js";
import {
  prepareLogicalGemmTileSpecialization,
  type PreparedLogicalGemmTileSpecialization,
  type PrepareLogicalGemmTileSpecializationRequest,
} from "./gemm-tile-prepare.js";

export const LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA = "browsergrad.kernel.gemm-exact-f32-input";
export const LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MAJOR = 1;
export const LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MINOR = 0;
export const LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT = 16_777_216n;
export const LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE = "browsergrad.f32-exact-nonnegative-integer-gemm@1";

const AUTHORITY = Object.freeze({
  schema: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
  major: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MAJOR,
});

const DEFAULT_MAX_INPUT_BYTES = 16 * 1024 * 1024;
const MAX_CONFIGURABLE_INPUT_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_PROOF_STEPS = 30_000_000;
const MAX_CONFIGURABLE_PROOF_STEPS = 100_000_000;
const DEFAULT_MAX_CERTIFICATION_MS = 5_000;
const MAX_CONFIGURABLE_CERTIFICATION_MS = 60_000;
const YIELD_INTERVAL_MS = 16;

type InputCommitment = JsonObject & {
  readonly algorithm: "sha256";
  readonly allocationByteLength: WireU64;
  readonly digest: string;
};

type ExactnessProof = JsonObject & {
  readonly profile: typeof LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE;
  readonly byteOrder: "little-endian";
  readonly exactIntegerLimit: WireU64;
  readonly multiplyAdds: WireU64;
  readonly maximumOutputSum: WireU64;
  readonly guarantees: JsonObject & {
    readonly inputs: "finite-nonnegative-integer-f32-with-positive-zero";
    readonly products: "every-product-exact-f32";
    readonly partialSums: "every-subset-sum-exact-f32";
    readonly strictLogicalPolicy: "increasing-k-rne-separate-multiply-add-preserved";
    readonly contraction: "value-preserving-on-certified-inputs";
    readonly reassociation: "value-preserving-on-certified-inputs";
    readonly wgslF32Output: "bit-exact-on-certified-inputs";
  };
};

export type LogicalGemmExactF32InputCertificatePayloadV1 = JsonObject & {
  readonly logicalGemmSemanticHash: string;
  readonly specializationHash: string;
  readonly inputs: JsonObject & {
    readonly lhs: InputCommitment;
    readonly rhs: InputCommitment;
  };
  readonly proof: ExactnessProof;
};

export type VerifiedLogicalGemmExactF32InputCertificate = VerifiedArtifact<LogicalGemmExactF32InputCertificatePayloadV1>;

export interface LogicalGemmExactF32Inputs {
  readonly lhs: Uint8Array;
  readonly rhs: Uint8Array;
}

export interface CertifyLogicalGemmExactF32InputsRequest extends PrepareLogicalGemmTileSpecializationRequest {
  readonly inputs: LogicalGemmExactF32Inputs;
  readonly maxInputBytes?: number;
  readonly maxProofSteps?: number;
  readonly maxCertificationMs?: number;
}

interface CertifiedInputSnapshot {
  readonly lhs: Uint8Array;
  readonly rhs: Uint8Array;
}

interface NativeInputSnapshot extends CertifiedInputSnapshot {
  readonly lhsOriginal: NativeSlots;
  readonly rhsOriginal: NativeSlots;
}

interface NativeSlots {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
}

/** @internal Shared with the canonical constructor to avoid proving the same inputs twice. */
export interface LogicalGemmExactF32InputEvaluation {
  readonly payload: LogicalGemmExactF32InputCertificatePayloadV1;
  readonly prepared: PreparedLogicalGemmTileSpecialization;
  readonly inputs: CertifiedInputSnapshot;
  readonly limits: DecodeLimits;
}

const CERTIFIED_INPUTS = new WeakMap<object, CertifiedInputSnapshot>();

/**
 * Re-verifies a wire certificate against the exact logical GEMM specialization
 * and a synchronous snapshot of the supplied allocation bytes. A certificate
 * cannot authorize later-mutated caller storage: consumers obtain fresh copies
 * of the privately retained certified snapshot with
 * `copyCertifiedLogicalGemmExactF32Inputs`.
 */
export async function verifyLogicalGemmExactF32InputCertificate(
  value: unknown,
  layout: VerifiedLayoutArtifact,
  logicalGemm: VerifiedLogicalGemmTileArtifact,
  request: CertifyLogicalGemmExactF32InputsRequest,
): Promise<VerifiedLogicalGemmExactF32InputCertificate> {
  const evaluation = await evaluateLogicalGemmExactF32Inputs(layout, logicalGemm, request);
  return verifyEvaluatedLogicalGemmExactF32InputCertificate(value, evaluation);
}

export async function decodeLogicalGemmExactF32InputCertificate(
  bytes: Uint8Array,
  layout: VerifiedLayoutArtifact,
  logicalGemm: VerifiedLogicalGemmTileArtifact,
  request: CertifyLogicalGemmExactF32InputsRequest,
): Promise<VerifiedLogicalGemmExactF32InputCertificate> {
  return verifyLogicalGemmExactF32InputCertificate(
    // Decode with the bounded public default before any property on the
    // caller-owned request is observed. Verification then captures the request
    // through own data descriptors and applies its resolved evaluation limits.
    decodeWireJson(bytes),
    layout,
    logicalGemm,
    request,
  );
}

export function logicalGemmExactF32InputCertificatePayload(
  certificate: VerifiedLogicalGemmExactF32InputCertificate,
): LogicalGemmExactF32InputCertificatePayloadV1 {
  const envelope = unwrapVerifiedArtifact(certificate, AUTHORITY);
  if (envelope.schema !== LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA
    || envelope.version.major !== LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MAJOR) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$", "verified artifact is not a browsergrad.kernel.gemm-exact-f32-input@1 certificate");
  }
  return envelope.payload;
}

/**
 * Returns fresh unshared allocation snapshots whose bytes are exactly those
 * committed by the verified certificate. Upload these copies directly; a
 * pre-existing GPU buffer is not authorized by a host-byte certificate.
 */
export function copyCertifiedLogicalGemmExactF32Inputs(
  certificate: VerifiedLogicalGemmExactF32InputCertificate,
): LogicalGemmExactF32Inputs {
  logicalGemmExactF32InputCertificatePayload(certificate);
  const retained = CERTIFIED_INPUTS.get(certificate as object);
  if (retained === undefined) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$", "verified exact-input certificate has no authority-retained byte snapshot");
  }
  return Object.freeze({
    lhs: new Uint8Array(retained.lhs),
    rhs: new Uint8Array(retained.rhs),
  });
}

/** @internal */
export async function evaluateLogicalGemmExactF32Inputs(
  layout: VerifiedLayoutArtifact,
  logicalGemm: VerifiedLogicalGemmTileArtifact,
  requestValue: CertifyLogicalGemmExactF32InputsRequest,
): Promise<LogicalGemmExactF32InputEvaluation> {
  const startedAt = monotonicNow();
  const request = captureCertificationRequest(requestValue);
  const maxInputBytes = resolveBudget(
    request.maxInputBytes,
    DEFAULT_MAX_INPUT_BYTES,
    MAX_CONFIGURABLE_INPUT_BYTES,
    "maxInputBytes",
  );
  const snapshots = captureInputSnapshots(request.inputs, maxInputBytes);
  const maxCertificationMs = resolveBudget(
    request.maxCertificationMs,
    DEFAULT_MAX_CERTIFICATION_MS,
    MAX_CONFIGURABLE_CERTIFICATION_MS,
    "maxCertificationMs",
  );
  ensureCertificationActive(startedAt, maxCertificationMs, request.signal);

  const prepared = await prepareLogicalGemmTileSpecialization(layout, logicalGemm, request);
  ensureCertificationActive(startedAt, maxCertificationMs, request.signal);
  validateInputBindings(snapshots, prepared);

  const inputElements = (prepared.m * prepared.k) + (prepared.k * prepared.n);
  const proofSteps = inputElements + prepared.outputElements + prepared.multiplyAdds;
  const maxProofSteps = resolveBudget(
    request.maxProofSteps,
    DEFAULT_MAX_PROOF_STEPS,
    MAX_CONFIGURABLE_PROOF_STEPS,
    "maxProofSteps",
  );
  if (proofSteps > BigInt(maxProofSteps)) {
    resource("$.maxProofSteps", `exact f32 GEMM certification requires ${proofSteps} proof steps; limit is ${maxProofSteps}`);
  }

  // Hash sequentially: sha256Hex owns a defensive copy, so at most one extra
  // input allocation is live in addition to the bounded retained snapshots.
  const lhsDigest = await sha256Hex(snapshots.lhs);
  ensureCertificationActive(startedAt, maxCertificationMs, request.signal);
  const rhsDigest = await sha256Hex(snapshots.rhs);
  ensureCertificationActive(startedAt, maxCertificationMs, request.signal);
  const maximumOutputSum = await proveExactNonnegativeIntegerGemm(
    snapshots,
    prepared,
    startedAt,
    maxCertificationMs,
    request.signal,
  );
  ensureCertificationActive(startedAt, maxCertificationMs, request.signal);
  const payload: LogicalGemmExactF32InputCertificatePayloadV1 = {
    logicalGemmSemanticHash: prepared.kernelSemanticHash,
    specializationHash: prepared.specializationHash,
    inputs: {
      lhs: {
        algorithm: "sha256",
        allocationByteLength: encodeWireU64(BigInt(snapshots.lhs.byteLength)),
        digest: lhsDigest,
      },
      rhs: {
        algorithm: "sha256",
        allocationByteLength: encodeWireU64(BigInt(snapshots.rhs.byteLength)),
        digest: rhsDigest,
      },
    },
    proof: {
      profile: LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE,
      byteOrder: "little-endian",
      exactIntegerLimit: encodeWireU64(LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT),
      multiplyAdds: encodeWireU64(prepared.multiplyAdds),
      maximumOutputSum: encodeWireU64(maximumOutputSum),
      guarantees: {
        inputs: "finite-nonnegative-integer-f32-with-positive-zero",
        products: "every-product-exact-f32",
        partialSums: "every-subset-sum-exact-f32",
        strictLogicalPolicy: "increasing-k-rne-separate-multiply-add-preserved",
        contraction: "value-preserving-on-certified-inputs",
        reassociation: "value-preserving-on-certified-inputs",
        wgslF32Output: "bit-exact-on-certified-inputs",
      },
    },
  };
  canonicalizeJson(payload, { limits: request.limits });
  return Object.freeze({
    payload,
    prepared,
    inputs: Object.freeze({ lhs: snapshots.lhs, rhs: snapshots.rhs }),
    limits: request.limits,
  });
}

/** @internal */
export function verifyEvaluatedLogicalGemmExactF32InputCertificate(
  value: unknown,
  evaluation: LogicalGemmExactF32InputEvaluation,
): VerifiedLogicalGemmExactF32InputCertificate {
  const envelope = validateWireEnvelope(value, {
    schema: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
    supportedMajor: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MAJOR,
    supportedMinor: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MINOR,
    knownRequiredExtensions: new Set(),
    limits: evaluation.limits,
  });
  const payload = parsePayload(envelope.payload);
  if (canonicalizeJson(payload, { limits: evaluation.limits })
    !== canonicalizeJson(evaluation.payload, { limits: evaluation.limits })) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidArtifact,
      "$.payload",
      "exact-input certificate does not match the supplied logical GEMM specialization and concrete input bytes",
    );
  }
  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: evaluation.payload,
  };
  const certificate = verifyWireArtifact(normalizedEnvelope, {
    schema: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
    supportedMajor: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MAJOR,
    supportedMinor: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_MINOR,
    knownRequiredExtensions: new Set(),
    limits: evaluation.limits,
    validatePayload: (candidate) => parsePayload(candidate),
  }, AUTHORITY) as VerifiedLogicalGemmExactF32InputCertificate;
  // Evaluation snapshots are package-internal move-only data: neither the
  // public constructor nor verifier exposes them. Retain those exact arrays so
  // certification does not transiently double its bounded owned input memory.
  CERTIFIED_INPUTS.set(certificate as object, evaluation.inputs);
  return certificate;
}

function parsePayload(value: JsonValue): LogicalGemmExactF32InputCertificatePayloadV1 {
  const object = closedObject(value, ["logicalGemmSemanticHash", "specializationHash", "inputs", "proof"], "$.payload");
  const inputs = closedObject(field(object, "inputs", "$.payload"), ["lhs", "rhs"], "$.payload.inputs");
  const proof = closedObject(field(object, "proof", "$.payload"), [
    "profile", "byteOrder", "exactIntegerLimit", "multiplyAdds", "maximumOutputSum", "guarantees",
  ], "$.payload.proof");
  if (proof.profile !== LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, "$.payload.proof.profile", "unsupported exact f32 input proof profile");
  }
  if (proof.byteOrder !== "little-endian") {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, "$.payload.proof.byteOrder", "exact f32 input proof requires little-endian allocation bytes");
  }
  const exactIntegerLimit = parseWireU64(field(proof, "exactIntegerLimit", "$.payload.proof"), "$.payload.proof.exactIntegerLimit");
  if (BigInt(exactIntegerLimit) !== LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, "$.payload.proof.exactIntegerLimit", "exact f32 input proof limit must be 2^24");
  }
  const guarantees = closedObject(field(proof, "guarantees", "$.payload.proof"), [
    "inputs", "products", "partialSums", "strictLogicalPolicy", "contraction", "reassociation", "wgslF32Output",
  ], "$.payload.proof.guarantees");
  requireConstant(guarantees, "inputs", "finite-nonnegative-integer-f32-with-positive-zero", "$.payload.proof.guarantees");
  requireConstant(guarantees, "products", "every-product-exact-f32", "$.payload.proof.guarantees");
  requireConstant(guarantees, "partialSums", "every-subset-sum-exact-f32", "$.payload.proof.guarantees");
  requireConstant(guarantees, "strictLogicalPolicy", "increasing-k-rne-separate-multiply-add-preserved", "$.payload.proof.guarantees");
  requireConstant(guarantees, "contraction", "value-preserving-on-certified-inputs", "$.payload.proof.guarantees");
  requireConstant(guarantees, "reassociation", "value-preserving-on-certified-inputs", "$.payload.proof.guarantees");
  requireConstant(guarantees, "wgslF32Output", "bit-exact-on-certified-inputs", "$.payload.proof.guarantees");
  return {
    logicalGemmSemanticHash: digest(field(object, "logicalGemmSemanticHash", "$.payload"), "$.payload.logicalGemmSemanticHash"),
    specializationHash: digest(field(object, "specializationHash", "$.payload"), "$.payload.specializationHash"),
    inputs: {
      lhs: parseInputCommitment(field(inputs, "lhs", "$.payload.inputs"), "$.payload.inputs.lhs"),
      rhs: parseInputCommitment(field(inputs, "rhs", "$.payload.inputs"), "$.payload.inputs.rhs"),
    },
    proof: {
      profile: LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE,
      byteOrder: "little-endian",
      exactIntegerLimit,
      multiplyAdds: parseWireU64(field(proof, "multiplyAdds", "$.payload.proof"), "$.payload.proof.multiplyAdds"),
      maximumOutputSum: parseWireU64(field(proof, "maximumOutputSum", "$.payload.proof"), "$.payload.proof.maximumOutputSum"),
      guarantees: {
        inputs: "finite-nonnegative-integer-f32-with-positive-zero",
        products: "every-product-exact-f32",
        partialSums: "every-subset-sum-exact-f32",
        strictLogicalPolicy: "increasing-k-rne-separate-multiply-add-preserved",
        contraction: "value-preserving-on-certified-inputs",
        reassociation: "value-preserving-on-certified-inputs",
        wgslF32Output: "bit-exact-on-certified-inputs",
      },
    },
  };
}

function parseInputCommitment(value: JsonValue, path: string): InputCommitment {
  const object = closedObject(value, ["algorithm", "allocationByteLength", "digest"], path);
  if (object.algorithm !== "sha256") {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.algorithm`, "exact-input commitment algorithm must be sha256");
  }
  return {
    algorithm: "sha256",
    allocationByteLength: parseWireU64(field(object, "allocationByteLength", path), `${path}.allocationByteLength`),
    digest: digest(field(object, "digest", path), `${path}.digest`),
  };
}

interface CapturedCertificationRequest extends PrepareLogicalGemmTileSpecializationRequest {
  readonly inputs: LogicalGemmExactF32Inputs;
  readonly maxInputBytes?: number;
  readonly maxProofSteps?: number;
  readonly maxCertificationMs?: number;
  readonly limits: DecodeLimits;
}

const REQUEST_FIELDS = [
  "operationId", "bindings", "evaluationLimits", "maxElements", "maxMultiplyAdds", "maxEvaluationSteps",
  "maxPreparationMs", "signal", "inputs", "maxInputBytes", "maxProofSteps", "maxCertificationMs",
] as const;

function captureCertificationRequest(value: CertifyLogicalGemmExactF32InputsRequest): CapturedCertificationRequest {
  const descriptors = plainDataDescriptors(value, REQUEST_FIELDS, ["operationId", "inputs"], "$", "certification request");
  const operationId = descriptors.operationId?.value;
  if (typeof operationId !== "string" || operationId.length === 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.operationId", "operationId must be a non-empty string");
  }
  const evaluationLimitsValue = descriptors.evaluationLimits?.value;
  const limits = resolveDecodeLimits(evaluationLimitsValue === undefined
    ? {}
    : copyPlainDataRecord(evaluationLimitsValue, "$.evaluationLimits") as Partial<DecodeLimits>);
  const bindingsValue = descriptors.bindings?.value;
  const bindings = bindingsValue === undefined
    ? undefined
    : copyPlainDataRecord(bindingsValue, "$.bindings") as Readonly<Record<string, WireI64>>;
  const signalValue = descriptors.signal?.value;
  if (signalValue !== undefined && !(signalValue instanceof AbortSignal)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.signal", "signal must be an AbortSignal");
  }
  const captured: CapturedCertificationRequest = {
    operationId,
    inputs: captureInputBindingObject(descriptors.inputs?.value),
    limits,
    ...(bindings === undefined ? {} : { bindings }),
    evaluationLimits: limits,
    ...copyOptionalNumber(descriptors, "maxElements"),
    ...copyOptionalNumber(descriptors, "maxMultiplyAdds"),
    ...copyOptionalNumber(descriptors, "maxEvaluationSteps"),
    ...copyOptionalNumber(descriptors, "maxPreparationMs"),
    ...copyOptionalNumber(descriptors, "maxInputBytes"),
    ...copyOptionalNumber(descriptors, "maxProofSteps"),
    ...copyOptionalNumber(descriptors, "maxCertificationMs"),
    ...(signalValue === undefined ? {} : { signal: signalValue as AbortSignal }),
  };
  return Object.freeze(captured);
}

function copyOptionalNumber(
  descriptors: PropertyDescriptorMap,
  name: "maxElements" | "maxMultiplyAdds" | "maxEvaluationSteps" | "maxPreparationMs"
    | "maxInputBytes" | "maxProofSteps" | "maxCertificationMs",
): Readonly<Record<string, number>> {
  const value = descriptors[name]?.value;
  return value === undefined ? {} : { [name]: value as number };
}

function captureInputBindingObject(value: unknown): LogicalGemmExactF32Inputs {
  const descriptors = plainDataDescriptors(value, ["lhs", "rhs"], ["lhs", "rhs"], "$.inputs", "input bindings");
  return Object.freeze({
    lhs: descriptors.lhs?.value as Uint8Array,
    rhs: descriptors.rhs?.value as Uint8Array,
  });
}

function copyPlainDataRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "expected a plain data object");
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "expected a plain data object");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const result = Object.create(null) as Record<string, unknown>;
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== "string") invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "symbol keys are forbidden");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `${path}.${key}`, "properties must be enumerable own data properties");
    }
    result[key] = descriptor.value;
  }
  return Object.freeze(result);
}

function plainDataDescriptors(
  value: unknown,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
  label: string,
): PropertyDescriptorMap {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `${label} must be a plain data object`);
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `${label} must expose ordinary own data properties`);
  }
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `${label} must be a plain data object`);
  }
  const allowed = new Set(allowedFields);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== "string" || !allowed.has(key)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `${label} contains an unknown field`);
    }
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `${path}.${key}`, `${label} properties must be enumerable own data properties`);
    }
  }
  for (const fieldName of requiredFields) {
    if (descriptors[fieldName] === undefined) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `${path}.${fieldName}`, "required field is missing");
    }
  }
  return descriptors;
}

function captureInputSnapshots(inputs: LogicalGemmExactF32Inputs, maxInputBytes: number): NativeInputSnapshot {
  const lhsOriginal = typedArraySlots(inputs.lhs, "$.inputs.lhs");
  const rhsOriginal = typedArraySlots(inputs.rhs, "$.inputs.rhs");
  const aggregateBytes = BigInt(lhsOriginal.byteLength) + BigInt(rhsOriginal.byteLength);
  if (aggregateBytes > BigInt(maxInputBytes)) {
    resource("$.maxInputBytes", `exact f32 GEMM certification received ${aggregateBytes} input bytes; limit is ${maxInputBytes}`);
  }
  if (rangesOverlap(lhsOriginal, rhsOriginal)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.inputs", "logical GEMM exact-input allocation bindings must not overlap");
  }
  return Object.freeze({
    lhs: copyNativeBytes(inputs.lhs, lhsOriginal.byteLength),
    rhs: copyNativeBytes(inputs.rhs, rhsOriginal.byteLength),
    lhsOriginal,
    rhsOriginal,
  });
}

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype) as object;
const BUFFER_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "buffer");
const BYTE_OFFSET_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteOffset");
const BYTE_LENGTH_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteLength");
const ARRAY_BUFFER_RESIZABLE_GETTER = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, "resizable")?.get;
const ARRAY_BUFFER_SLICE = ArrayBuffer.prototype.slice;
const UINT8_ARRAY_SET = Uint8Array.prototype.set;

function typedArraySlots(value: Uint8Array, path: string): NativeSlots {
  if (!(value instanceof Uint8Array) || Object.getPrototypeOf(value) !== Uint8Array.prototype) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "exact-input bindings must be direct Uint8Array values");
  }
  try {
    const buffer = BUFFER_GETTER.call(value) as ArrayBufferLike;
    if (!(buffer instanceof ArrayBuffer)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "exact-input bindings must use unshared ArrayBuffer storage");
    }
    if (ARRAY_BUFFER_RESIZABLE_GETTER?.call(buffer) === true) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "exact-input bindings must use fixed-length ArrayBuffer storage");
    }
    try {
      ARRAY_BUFFER_SLICE.call(buffer, 0, 0);
    } catch {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "exact-input binding storage must not be detached");
    }
    return {
      buffer,
      byteOffset: BYTE_OFFSET_GETTER.call(value) as number,
      byteLength: BYTE_LENGTH_GETTER.call(value) as number,
    };
  } catch (error) {
    if (error instanceof SemanticSchemaError) throw error;
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "exact-input binding does not expose native typed-array slots");
  }
}

function copyNativeBytes(value: Uint8Array, byteLength: number): Uint8Array {
  const snapshot = new Uint8Array(byteLength);
  UINT8_ARRAY_SET.call(snapshot, value);
  return snapshot;
}

function requiredGetter(target: object, name: string): (this: unknown) => unknown {
  const getter = Object.getOwnPropertyDescriptor(target, name)?.get;
  if (getter === undefined) throw new Error(`internal: missing typed-array ${name} getter`);
  return getter;
}

function validateInputBindings(snapshots: NativeInputSnapshot, prepared: PreparedLogicalGemmTileSpecialization): void {
  requireExactLength(snapshots.lhsOriginal, prepared.lhs.allocationByteLength, "$.inputs.lhs");
  requireExactLength(snapshots.rhsOriginal, prepared.rhs.allocationByteLength, "$.inputs.rhs");
  requireAlignment(snapshots.lhsOriginal, prepared.lhs.allocationAlignmentBytes, "$.inputs.lhs");
  requireAlignment(snapshots.rhsOriginal, prepared.rhs.allocationAlignmentBytes, "$.inputs.rhs");
}

async function proveExactNonnegativeIntegerGemm(
  inputs: CertifiedInputSnapshot,
  prepared: PreparedLogicalGemmTileSpecialization,
  startedAt: number,
  maxCertificationMs: number,
  signal: AbortSignal | undefined,
): Promise<bigint> {
  const lhsView = new DataView(inputs.lhs.buffer, inputs.lhs.byteOffset, inputs.lhs.byteLength);
  const rhsView = new DataView(inputs.rhs.buffer, inputs.rhs.byteOffset, inputs.rhs.byteLength);
  const lhsBase = safeIndex(prepared.lhs.viewByteOffset, "$.inputs.lhs");
  const rhsBase = safeIndex(prepared.rhs.viewByteOffset, "$.inputs.rhs");
  let yieldAt = startedAt + YIELD_INTERVAL_MS;
  let visited = 0n;

  // Validate each logical input once without retaining a second matrix-sized
  // value representation. The private byte snapshots are the sole owned input
  // storage; subsequent product reads use their already-validated dense bytes.
  for (let row = 0n; row < prepared.m; row += 1n) {
    for (let inner = 0n; inner < prepared.k; inner += 1n) {
      yieldAt = await maybeYield(visited, yieldAt, startedAt, maxCertificationMs, signal);
      const linear = (row * prepared.k) + inner;
      const offset = denseByteOffset(lhsBase, linear, "$.inputs.lhs");
      decodeExactInput(lhsView.getUint32(offset, true), `$.inputs.lhs[${row},${inner}]`);
      visited += 1n;
    }
  }
  for (let inner = 0n; inner < prepared.k; inner += 1n) {
    for (let column = 0n; column < prepared.n; column += 1n) {
      yieldAt = await maybeYield(visited, yieldAt, startedAt, maxCertificationMs, signal);
      const linear = (inner * prepared.n) + column;
      const offset = denseByteOffset(rhsBase, linear, "$.inputs.rhs");
      decodeExactInput(rhsView.getUint32(offset, true), `$.inputs.rhs[${inner},${column}]`);
      visited += 1n;
    }
  }

  if (prepared.outputElements === 0n) {
    ensureCertificationActive(startedAt, maxCertificationMs, signal);
    return 0n;
  }

  let maximumOutputSum = 0;
  for (let row = 0n; row < prepared.m; row += 1n) {
    for (let column = 0n; column < prepared.n; column += 1n) {
      yieldAt = await maybeYield(visited, yieldAt, startedAt, maxCertificationMs, signal);
      visited += 1n;
      let outputSum = 0;
      for (let inner = 0n; inner < prepared.k; inner += 1n) {
        yieldAt = await maybeYield(visited, yieldAt, startedAt, maxCertificationMs, signal);
        const lhsOffset = denseByteOffset(lhsBase, (row * prepared.k) + inner, "$.inputs.lhs");
        const rhsOffset = denseByteOffset(rhsBase, (inner * prepared.n) + column, "$.inputs.rhs");
        const lhs = decodeExactInput(lhsView.getUint32(lhsOffset, true), `$.inputs.lhs[${row},${inner}]`);
        const rhs = decodeExactInput(rhsView.getUint32(rhsOffset, true), `$.inputs.rhs[${inner},${column}]`);
        const product = lhs * rhs;
        if (product > Number(LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT)) {
          unsupported(
            `$.proof.output[${row},${column}].product[${inner}]`,
            `exact integer product ${product} exceeds the inclusive f32 integer limit ${LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT}`,
          );
        }
        outputSum += product;
        if (outputSum > Number(LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT)) {
          unsupported(
            `$.proof.output[${row},${column}]`,
            `nonnegative product total ${outputSum} exceeds the inclusive f32 integer limit ${LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT}`,
          );
        }
        visited += 1n;
      }
      if (outputSum > maximumOutputSum) maximumOutputSum = outputSum;
    }
  }
  ensureCertificationActive(startedAt, maxCertificationMs, signal);
  return BigInt(maximumOutputSum);
}

function decodeExactInput(bits: number, path: string): number {
  const sign = bits >>> 31;
  const exponentBits = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  if (sign !== 0) unsupported(path, "exact f32 GEMM profile requires nonnegative inputs and canonical positive zero");
  if (exponentBits === 0xff) unsupported(path, "exact f32 GEMM profile rejects NaN and infinity");
  if (exponentBits === 0) {
    if (fraction === 0) return 0;
    unsupported(path, "exact f32 GEMM profile rejects nonzero subnormal and nonintegral inputs");
  }
  const significand = BigInt((1 << 23) + fraction);
  const shift = exponentBits - 127 - 23;
  let integer: bigint;
  if (shift >= 0) {
    integer = significand << BigInt(shift);
  } else {
    const divisor = 1n << BigInt(-shift);
    if (significand % divisor !== 0n) unsupported(path, "exact f32 GEMM profile requires integer-valued inputs");
    integer = significand / divisor;
  }
  if (integer > LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT) {
    unsupported(path, `input integer ${integer} exceeds the inclusive f32 integer limit ${LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT}`);
  }
  return Number(integer);
}

async function maybeYield(
  visited: bigint,
  yieldAt: number,
  startedAt: number,
  maxCertificationMs: number,
  signal: AbortSignal | undefined,
): Promise<number> {
  if ((visited & 1023n) !== 0n) return yieldAt;
  ensureCertificationActive(startedAt, maxCertificationMs, signal);
  const now = monotonicNow();
  if (now < yieldAt) return yieldAt;
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  ensureCertificationActive(startedAt, maxCertificationMs, signal);
  return monotonicNow() + YIELD_INTERVAL_MS;
}

function requireExactLength(slots: NativeSlots, expected: bigint, path: string): void {
  if (BigInt(slots.byteLength) !== expected) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `binding length ${slots.byteLength} does not equal declared allocation length ${expected}`);
  }
}

function requireAlignment(slots: NativeSlots, alignment: number, path: string): void {
  if (slots.byteOffset % alignment !== 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `binding byte offset does not satisfy ${alignment}-byte alignment`);
  }
}

function rangesOverlap(left: NativeSlots, right: NativeSlots): boolean {
  if (left.buffer !== right.buffer) return false;
  return left.byteOffset < right.byteOffset + right.byteLength && right.byteOffset < left.byteOffset + left.byteLength;
}

function safeIndex(value: bigint, path: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, path, "byte address cannot be represented as a JavaScript buffer index");
  }
  return Number(value);
}

function denseByteOffset(base: number, linearElement: bigint, path: string): number {
  const offset = BigInt(base) + (linearElement * 4n);
  return safeIndex(offset, path);
}

function resolveBudget(value: number | undefined, fallback: number, maximum: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    resource(`$.${name}`, `${name} must be a positive safe integer no greater than ${maximum}`);
  }
  return resolved;
}

function monotonicNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function ensureCertificationActive(startedAt: number, maxCertificationMs: number, signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) resource("$.signal", "exact f32 GEMM input certification was aborted");
  if (monotonicNow() - startedAt > maxCertificationMs) {
    resource("$.maxCertificationMs", `exact f32 GEMM input certification exceeded ${maxCertificationMs} ms`);
  }
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  const allowed = new Set(fields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unknownField, path, `unknown fields: ${unknown.sort().join(", ")}`);
  }
  for (const name of fields) {
    if (value[name] === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function requireConstant(object: JsonObject, name: string, expected: string, path: string): void {
  if (object[name] !== expected) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.${name}`, `exact f32 input proof requires ${name}=${expected}`);
  }
}

function digest(value: JsonValue, path: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/u.test(value)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "digest must be 64 lowercase hexadecimal digits");
  }
  return value;
}

function resource(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, path, message);
}

function unsupported(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, path, message);
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
