import {
  createVerifiedViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import type { LayoutExpr } from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  SemanticSchemaError,
  encodeWireI64,
  parseWireU64,
  resolveDecodeLimits,
  wireIntegerToBigInt,
  type DecodeLimits,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { unwrapVerifiedCppCuteFrontendArtifact } from "./cpp_cute_frontend_artifact.js";
import {
  unwrapAuthorizedCppCuteFrontendArtifact,
  type AuthorizedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_authorization.js";
import type {
  CppCuteAffineLayoutFactV1,
  CppCuteFrontendPayloadV3,
  CppCuteResolvedFactV1,
} from "./cpp_cute_frontend_types.js";
import {
  CppCuteLayoutLoweringError,
  lowerStaticCppCuteLayoutFact,
} from "./cpp_cute_layout_semantics.js";

const CAPTURED_OBJECT = Object;
const CAPTURED_REFLECT = Reflect;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export interface LowerAuthorizedCppCuteViewCopyEntryRequest {
  readonly entryId: string;
  readonly sourceAllocationByteLength: WireU64;
  readonly destinationAllocationByteLength: WireU64;
  readonly sourceByteOffset: WireU64;
  readonly destinationByteOffset: WireU64;
}

export interface LowerAuthorizedCppCuteViewCopyEntryOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

export type CppCuteViewCopyLoweringErrorCode =
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST"
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-INCONSISTENT-ARTIFACT"
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-ENTRY"
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-PROFILE"
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT"
  | "BG-COMPILER-CPP-CUTE-VIEW-COPY-RESOURCE-LIMIT";

export class CppCuteViewCopyLoweringError extends Error {
  constructor(
    readonly code: CppCuteViewCopyLoweringErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteViewCopyLoweringError";
  }
}

interface NormalizedRequest {
  readonly entryId: string;
  readonly sourceAllocationByteLength: bigint;
  readonly destinationAllocationByteLength: bigint;
  readonly sourceByteOffset: bigint;
  readonly destinationByteOffset: bigint;
}

interface NormalizedOptions {
  readonly limits: DecodeLimits;
  readonly signal?: AbortSignal;
}

type TensorFact = Extract<CppCuteResolvedFactV1, { readonly kind: "tensor" }>;
type IntrinsicFact = Extract<CppCuteResolvedFactV1, { readonly kind: "target-intrinsic" }>;

/**
 * Producer-neutral semantic transition from one exact authorized frontend
 * result. Runtime allocation capacities remain explicit host facts; CuTe
 * `cosize` is never treated as pointer storage authority.
 */
export async function lowerAuthorizedCppCuteViewCopyEntry(
  authorization: AuthorizedCppCuteFrontendArtifact,
  request: LowerAuthorizedCppCuteViewCopyEntryRequest,
  options: LowerAuthorizedCppCuteViewCopyEntryOptions = {},
): Promise<VerifiedViewCopyArtifacts> {
  const normalizedOptions = normalizeOptions(options);
  throwIfAborted(normalizedOptions.signal);
  const normalizedRequest = normalizeRequest(request);
  const authorized = unwrapAuthorizedCppCuteFrontendArtifact(authorization);
  const payload = unwrapVerifiedCppCuteFrontendArtifact(authorized.artifact).envelope.payload;
  const entry = selectedViewCopyEntry(payload, normalizedRequest.entryId);
  const source = tensorFact(payload, entry.sourceTensorFactId, "$.artifact.entry.sourceTensorFactId");
  const destination = tensorFact(payload, entry.destinationTensorFactId, "$.artifact.entry.destinationTensorFactId");
  const sourceLayout = layoutFact(payload, source.layoutFactId, "$.artifact.source.layoutFactId");
  const destinationLayout = layoutFact(payload, destination.layoutFactId, "$.artifact.destination.layoutFactId");
  const intrinsic = viewCopyIntrinsic(payload, entry.operationExpressionId);
  validateProfile(payload, source, destination, sourceLayout, destinationLayout, intrinsic);
  throwIfAborted(normalizedOptions.signal);

  let sourceLayoutExpression: LayoutExpr;
  let destinationLayoutExpression: LayoutExpr;
  try {
    sourceLayoutExpression = lowerStaticCppCuteLayoutFact(
      sourceLayout,
      normalizedOptions.limits,
      "$.artifact.source.layout",
    );
    destinationLayoutExpression = lowerStaticCppCuteLayoutFact(
      destinationLayout,
      normalizedOptions.limits,
      "$.artifact.destination.layout",
    );
  } catch (cause) {
    if (!(cause instanceof CppCuteLayoutLoweringError)) throw cause;
    const code = cause.code.endsWith("RESOURCE-LIMIT")
      ? "BG-COMPILER-CPP-CUTE-VIEW-COPY-RESOURCE-LIMIT"
      : cause.code.endsWith("INCONSISTENT-ARTIFACT")
        ? "BG-COMPILER-CPP-CUTE-VIEW-COPY-INCONSISTENT-ARTIFACT"
        : "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT";
    failure(code, cause.path, cause.message, { cause });
  }

  validateStorage(
    normalizedRequest.sourceAllocationByteLength,
    normalizedRequest.sourceByteOffset,
    positiveRank2SpanElements(sourceLayoutExpression, "$.artifact.source.layout"),
    "$.request.sourceAllocationByteLength",
    "$.request.sourceByteOffset",
  );
  validateStorage(
    normalizedRequest.destinationAllocationByteLength,
    normalizedRequest.destinationByteOffset,
    positiveRank2SpanElements(destinationLayoutExpression, "$.artifact.destination.layout"),
    "$.request.destinationAllocationByteLength",
    "$.request.destinationByteOffset",
  );
  throwIfAborted(normalizedOptions.signal);
  try {
    const artifacts = await createVerifiedViewCopyArtifacts({
      dtype: "f32",
      symbols: [],
      constraints: [],
      source: {
        layout: sourceLayoutExpression,
        allocation: {
          byteLength: constant(normalizedRequest.sourceAllocationByteLength),
          memorySpace: { kind: "global" },
          alignmentBytes: 4,
        },
        byteOffset: constant(normalizedRequest.sourceByteOffset),
        requiredAlignmentBytes: 4,
      },
      destination: {
        layout: destinationLayoutExpression,
        allocation: {
          byteLength: constant(normalizedRequest.destinationAllocationByteLength),
          memorySpace: { kind: "global" },
          alignmentBytes: 4,
        },
        byteOffset: constant(normalizedRequest.destinationByteOffset),
        requiredAlignmentBytes: 4,
      },
      invalidSource: { kind: "reject" },
    }, {
      producer: { id: "browsergrad.compiler.cpp-cute-view-copy-lowering", version: "1" },
      layoutArtifactId: "authorized-cpp-cute-view-copy-layout",
      kernelArtifactId: "authorized-cpp-cute-view-copy-kernel",
      limits: normalizedOptions.limits,
    });
    throwIfAborted(normalizedOptions.signal);
    return artifacts;
  } catch (cause) {
    if (cause instanceof CppCuteViewCopyLoweringError) throw cause;
    if (!(cause instanceof SemanticSchemaError)) throw cause;
    failure(
      cause.diagnostic.code.endsWith("RESOURCE-LIMIT")
        ? "BG-COMPILER-CPP-CUTE-VIEW-COPY-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT",
      cause.diagnostic.path ?? "$.artifacts",
      `shared semantic-core rejected the authorized view-copy: ${cause.message}`,
      { cause },
    );
  }
}

function selectedViewCopyEntry(payload: CppCuteFrontendPayloadV3, entryId: string) {
  if (payload.outcome.kind !== "accepted" ||
      payload.outcome.selectedEntryIds.length !== 1 ||
      payload.outcome.selectedEntryIds[0] !== entryId) {
    failure(
      "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-ENTRY",
      "$.artifact.outcome.selectedEntryIds",
      "initial view-copy lowering requires exactly the explicitly requested selected entry",
    );
  }
  const entry = payload.entries.find((candidate) => candidate.entryId === entryId);
  if (entry === undefined) inconsistent("$.request.entryId", "selected entry disappeared from the authorized artifact");
  if (entry.kind !== "view-copy") {
    failure("BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-ENTRY", "$.request.entryId", "selected entry is not a view-copy");
  }
  return entry;
}

function tensorFact(payload: CppCuteFrontendPayloadV3, factId: string, path: string): TensorFact {
  const fact = payload.facts.find((candidate) => candidate.factId === factId);
  if (fact?.kind !== "tensor") inconsistent(path, "selected tensor fact disappeared or changed kind");
  return fact;
}

function layoutFact(payload: CppCuteFrontendPayloadV3, factId: string, path: string): CppCuteAffineLayoutFactV1 {
  const fact = payload.facts.find((candidate) => candidate.factId === factId);
  if (fact?.kind !== "affine-layout") inconsistent(path, "selected affine-layout fact disappeared or changed kind");
  return fact;
}

function viewCopyIntrinsic(payload: CppCuteFrontendPayloadV3, expressionId: string): IntrinsicFact {
  const expressions = payload.functionBodies.flatMap((body) => body.expressions);
  const expression = expressions.find((candidate) => candidate.expressionId === expressionId);
  if (expression?.kind !== "target-intrinsic") inconsistent("$.artifact.entry.operationExpressionId", "copy expression disappeared or changed kind");
  const fact = payload.facts.find((candidate) => candidate.factId === expression.intrinsicFactId);
  if (fact?.kind !== "target-intrinsic") inconsistent("$.artifact.entry.operationExpressionId", "copy intrinsic fact disappeared or changed kind");
  return fact;
}

function validateProfile(
  payload: CppCuteFrontendPayloadV3,
  source: TensorFact,
  destination: TensorFact,
  sourceLayout: CppCuteAffineLayoutFactV1,
  destinationLayout: CppCuteAffineLayoutFactV1,
  intrinsic: IntrinsicFact,
): void {
  const elementType = payload.types.find((candidate) => candidate.typeId === source.elementTypeId);
  const abi = payload.sourceAbi.types.filter((candidate) => (
    candidate.domain === "device" && candidate.deviceTypeId === source.elementTypeId
  ));
  if (source.elementTypeId !== destination.elementTypeId ||
      elementType?.kind !== "builtin" || elementType.builtin !== "float" ||
      abi.length !== 1 || abi[0]?.sizeBits !== "32" || abi[0].alignmentBits !== "32") {
    unsupportedProfile("$.artifact.source.elementTypeId", "initial view-copy lowering requires one exact f32 device ABI with 32-bit size and alignment");
  }
  if (source.memorySpace !== "global" || destination.memorySpace !== "global" ||
      source.engine.kind !== "global-pointer" || destination.engine.kind !== "global-pointer" ||
      source.engine.nullable || destination.engine.nullable ||
      source.engine.pointerDeclarationId === destination.engine.pointerDeclarationId) {
    unsupportedProfile("$.artifact.entry", "initial view-copy lowering requires distinct non-null global pointer engines");
  }
  validateF32Pointer(payload, source.engine.pointerDeclarationId, true, "$.artifact.source.engine");
  validateF32Pointer(payload, destination.engine.pointerDeclarationId, false, "$.artifact.destination.engine");
  if (sourceLayout.rank !== 2 || destinationLayout.rank !== 2) {
    unsupportedLayout("$.artifact.entry", "initial view-copy lowering requires rank-2 source and destination layouts");
  }
  if (intrinsic.operation.kind !== "copy" || intrinsic.operation.sourceSpace !== "global" ||
      intrinsic.operation.destinationSpace !== "global" || intrinsic.operation.transferBits !== 32 ||
      intrinsic.operation.asynchronous || intrinsic.availability.kind !== "portable-candidate" ||
      !intrinsic.effects.readsMemory || !intrinsic.effects.writesMemory ||
      intrinsic.effects.synchronizes || intrinsic.effects.convergent) {
    unsupportedProfile("$.artifact.entry.operationExpressionId", "initial view-copy lowering requires one synchronous portable 32-bit global copy with exact read/write effects");
  }
}

function validateF32Pointer(
  payload: CppCuteFrontendPayloadV3,
  declarationId: string,
  readOnly: boolean,
  path: string,
): void {
  const declaration = payload.declarations.find((candidate) => candidate.declarationId === declarationId);
  const pointer = declaration?.typeId === null
    ? undefined
    : payload.types.find((candidate) => candidate.typeId === declaration?.typeId);
  const pointee = pointer?.kind === "pointer"
    ? payload.types.find((candidate) => candidate.typeId === pointer.pointeeTypeId)
    : undefined;
  if (declaration?.kind !== "parameter" || declaration.memorySpace !== "global" ||
      pointer?.kind !== "pointer" || pointer.addressSpace !== "global" ||
      pointee?.kind !== "builtin" || pointee.builtin !== "float" ||
      pointee.qualifiers.const !== readOnly) {
    unsupportedProfile(path, "initial view-copy engine must be an exact global f32 parameter pointer with source-only const qualification");
  }
}

function positiveRank2SpanElements(
  layout: LayoutExpr,
  path: string,
): bigint {
  if (layout.kind !== "strided" || layout.shape.length !== 2 || layout.strides.length !== 2) {
    unsupportedLayout(path, "initial view-copy lowering requires two flat static affine modes");
  }
  const values = [...layout.shape, ...layout.strides].map((expression, index) => {
    if (expression.kind !== "const") {
      unsupportedLayout(`${path}.${index < 2 ? "shape" : "strides"}[${index % 2}]`, "initial view-copy layout shape and strides must be static");
    }
    const value = wireIntegerToBigInt(expression.value);
    if (value <= 0n) {
      unsupportedLayout(`${path}.${index < 2 ? "shape" : "strides"}[${index % 2}]`, "initial view-copy layout shape and strides must be positive");
    }
    return value;
  });
  const [shape0, shape1, stride0, stride1] = values;
  if (shape0 === undefined || shape1 === undefined || stride0 === undefined || stride1 === undefined) {
    inconsistent(path, "positive rank-2 layout projection disappeared");
  }
  return (shape0 - 1n) * stride0 + (shape1 - 1n) * stride1 + 1n;
}

function validateStorage(
  byteLength: bigint,
  byteOffset: bigint,
  spanElements: bigint,
  byteLengthPath: string,
  byteOffsetPath: string,
): void {
  if (byteLength % 4n !== 0n) invalidRequest(byteLengthPath, "f32 allocation byte length must be 4-byte aligned");
  if (byteOffset % 4n !== 0n) invalidRequest(byteOffsetPath, "f32 view byte offset must be 4-byte aligned");
  const required = byteOffset + spanElements * 4n;
  if (required > byteLength) {
    unsupportedLayout(byteLengthPath, `explicit allocation has ${byteLength} bytes but the verified affine address span requires ${required}`);
  }
}

function normalizeRequest(request: LowerAuthorizedCppCuteViewCopyEntryRequest): NormalizedRequest {
  const descriptors = closedDataRecord(request, [
    "entryId",
    "sourceAllocationByteLength",
    "destinationAllocationByteLength",
    "sourceByteOffset",
    "destinationByteOffset",
  ], "$.request");
  const entryId = descriptors.entryId?.value;
  if (typeof entryId !== "string" || entryId.length === 0) invalidRequest("$.request.entryId", "entryId must be a non-empty string");
  const sourceAllocationByteLength = unsigned(descriptors.sourceAllocationByteLength?.value, "$.request.sourceAllocationByteLength", true);
  const destinationAllocationByteLength = unsigned(descriptors.destinationAllocationByteLength?.value, "$.request.destinationAllocationByteLength", true);
  const sourceByteOffset = unsigned(descriptors.sourceByteOffset?.value, "$.request.sourceByteOffset", false);
  const destinationByteOffset = unsigned(descriptors.destinationByteOffset?.value, "$.request.destinationByteOffset", false);
  return CAPTURED_OBJECT.freeze({
    entryId,
    sourceAllocationByteLength,
    destinationAllocationByteLength,
    sourceByteOffset,
    destinationByteOffset,
  });
}

function normalizeOptions(options: LowerAuthorizedCppCuteViewCopyEntryOptions): NormalizedOptions {
  const descriptors = closedDataRecord(options, ["limits", "signal"], "$.options", true);
  let limits: DecodeLimits;
  try {
    limits = resolveDecodeLimits(descriptors.limits?.value as Partial<DecodeLimits> | undefined);
  } catch (cause) {
    invalidRequest("$.options.limits", "invalid semantic decode limits", { cause });
  }
  const signal = descriptors.signal?.value as AbortSignal | undefined;
  return CAPTURED_OBJECT.freeze({ limits, ...(signal === undefined ? {} : { signal }) });
}

function closedDataRecord(
  value: unknown,
  expectedKeys: readonly string[],
  path: string,
  allowMissing = false,
): Readonly<Record<string, PropertyDescriptor>> {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_PROTOTYPE_OF, CAPTURED_OBJECT, [value]) as object | null;
    descriptors = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS, CAPTURED_OBJECT, [value]) as PropertyDescriptorMap;
    keys = NATIVE_REFLECT_APPLY(NATIVE_REFLECT_OWN_KEYS, CAPTURED_REFLECT, [value]) as readonly PropertyKey[];
  } catch (cause) {
    invalidRequest(path, "request inspection failed", { cause });
  }
  if (prototype !== CAPTURED_OBJECT.prototype && prototype !== null) invalidRequest(path, "expected a plain object");
  if (keys.some((key) => typeof key !== "string" || !expectedKeys.includes(key)) ||
      (!allowMissing && (keys.length !== expectedKeys.length || expectedKeys.some((key) => !keys.includes(key))))) {
    invalidRequest(path, `expected only ${expectedKeys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = descriptors[String(key)];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalidRequest(`${path}.${String(key)}`, "expected an enumerable data property without accessors");
    }
  }
  return descriptors;
}

function unsigned(value: unknown, path: string, positive: boolean): bigint {
  try {
    const parsed = wireIntegerToBigInt(parseWireU64(value, path));
    if (positive && parsed === 0n) invalidRequest(path, "allocation byte length must be positive");
    if (parsed > 0x7fff_ffff_ffff_ffffn) invalidRequest(path, "value exceeds signed semantic expression range");
    return parsed;
  } catch (cause) {
    if (cause instanceof CppCuteViewCopyLoweringError) throw cause;
    invalidRequest(path, "expected a canonical unsigned wire integer", { cause });
  }
}

function constant(value: bigint) {
  return { kind: "const" as const, value: encodeWireI64(value) };
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  let aborted: unknown;
  try {
    aborted = ABORT_SIGNAL_ABORTED_GETTER === undefined
      ? undefined
      : NATIVE_REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []);
  } catch (cause) {
    invalidRequest("$.options.signal", "signal is not a native AbortSignal", { cause });
  }
  if (aborted === true) failure("BG-COMPILER-CPP-CUTE-VIEW-COPY-CANCELLED", "$.options.signal", "view-copy lowering was aborted");
}

function invalidRequest(path: string, message: string, options?: ErrorOptions): never {
  failure("BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST", path, message, options);
}

function inconsistent(path: string, message: string): never {
  failure("BG-COMPILER-CPP-CUTE-VIEW-COPY-INCONSISTENT-ARTIFACT", path, message);
}

function unsupportedProfile(path: string, message: string): never {
  failure("BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-PROFILE", path, message);
}

function unsupportedLayout(path: string, message: string): never {
  failure("BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT", path, message);
}

function failure(
  code: CppCuteViewCopyLoweringErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteViewCopyLoweringError(code, path, message, options);
}
