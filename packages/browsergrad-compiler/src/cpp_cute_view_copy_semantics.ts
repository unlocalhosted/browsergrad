import type { LayoutExpr } from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  hashCanonicalJson,
  resolveDecodeLimits,
  wireIntegerToBigInt,
  type DecodeLimits,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import type {
  CppCuteAffineLayoutFactV1,
  CppCuteFrontendEntryV1,
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
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export interface PrepareVerifiedCppCuteViewCopySemanticsRequest {
  readonly entryId: string;
}

export interface PrepareVerifiedCppCuteViewCopySemanticsOptions {
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

export interface NormalizedCppCuteViewCopyOptions {
  readonly limits: DecodeLimits;
  readonly signal?: AbortSignal;
}

type TensorFact = Extract<CppCuteResolvedFactV1, { readonly kind: "tensor" }>;
type IntrinsicFact = Extract<CppCuteResolvedFactV1, { readonly kind: "target-intrinsic" }>;

/** Producer-derived view-copy semantics before any host storage geometry is supplied. */
export interface PreparedVerifiedCppCuteViewCopySemantics {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly entry: Extract<CppCuteFrontendEntryV1, { readonly kind: "view-copy" }>;
  readonly sourceTensor: TensorFact;
  readonly destinationTensor: TensorFact;
  readonly sourceLayoutFact: CppCuteAffineLayoutFactV1;
  readonly destinationLayoutFact: CppCuteAffineLayoutFactV1;
  readonly intrinsic: IntrinsicFact;
  readonly sourceLayout: LayoutExpr;
  readonly destinationLayout: LayoutExpr;
  readonly sourceSpanElements: bigint;
  readonly destinationSpanElements: bigint;
  readonly entrySubjectHash: string;
  readonly loweringAuthorityMinted: false;
}

export async function prepareVerifiedCppCuteViewCopySemantics(
  artifact: VerifiedCppCuteFrontendArtifact,
  request: PrepareVerifiedCppCuteViewCopySemanticsRequest,
  options: PrepareVerifiedCppCuteViewCopySemanticsOptions = {},
): Promise<PreparedVerifiedCppCuteViewCopySemantics> {
  const normalizedOptions = normalizeCppCuteViewCopyOptions(options);
  throwIfCppCuteViewCopyAborted(normalizedOptions.signal);
  const entryId = validateCppCuteViewCopyEntryRequest(request);
  const verified = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const payload = verified.envelope.payload;
  const entry = selectedViewCopyEntry(payload, entryId);
  const sourceTensor = tensorFact(payload, entry.sourceTensorFactId, "$.artifact.entry.sourceTensorFactId");
  const destinationTensor = tensorFact(payload, entry.destinationTensorFactId, "$.artifact.entry.destinationTensorFactId");
  const sourceLayoutFact = layoutFact(payload, sourceTensor.layoutFactId, "$.artifact.source.layoutFactId");
  const destinationLayoutFact = layoutFact(payload, destinationTensor.layoutFactId, "$.artifact.destination.layoutFactId");
  const intrinsic = viewCopyIntrinsic(payload, entry.operationExpressionId);
  validateProfile(
    payload,
    sourceTensor,
    destinationTensor,
    sourceLayoutFact,
    destinationLayoutFact,
    intrinsic,
  );
  throwIfCppCuteViewCopyAborted(normalizedOptions.signal);

  let sourceLayout: LayoutExpr;
  let destinationLayout: LayoutExpr;
  try {
    sourceLayout = lowerStaticCppCuteLayoutFact(
      sourceLayoutFact,
      normalizedOptions.limits,
      "$.artifact.source.layout",
    );
    destinationLayout = lowerStaticCppCuteLayoutFact(
      destinationLayoutFact,
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
    cppCuteViewCopyFailure(code, cause.path, cause.message, { cause });
  }
  const sourceSpanElements = positiveRank2SpanElements(sourceLayout, "$.artifact.source.layout");
  const destinationSpanElements = positiveRank2SpanElements(
    destinationLayout,
    "$.artifact.destination.layout",
  );
  throwIfCppCuteViewCopyAborted(normalizedOptions.signal);
  let entrySubjectHash: string;
  try {
    entrySubjectHash = await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.view-copy-entry-subject.v1",
      artifactId: artifact.artifactId,
      artifactHash: artifact.artifactHash,
      artifactBytesSha256: artifact.artifactBytesSha256,
      artifactByteLength: artifact.artifactByteLength,
      entryId: entry.entryId,
      sourceTensorFactId: sourceTensor.factId,
      destinationTensorFactId: destinationTensor.factId,
      sourceLayoutFactId: sourceLayoutFact.factId,
      destinationLayoutFactId: destinationLayoutFact.factId,
      operationExpressionId: entry.operationExpressionId,
      intrinsicFactId: intrinsic.factId,
      sourceLayout,
      destinationLayout,
    }, { limits: normalizedOptions.limits });
  } catch (cause) {
    cppCuteViewCopyFailure(
      "BG-COMPILER-CPP-CUTE-VIEW-COPY-RESOURCE-LIMIT",
      "$.artifact.entry",
      "view-copy semantic subject exceeded canonical hashing limits",
      { cause },
    );
  }
  throwIfCppCuteViewCopyAborted(normalizedOptions.signal);
  return NATIVE_OBJECT_FREEZE({
    artifact,
    entry,
    sourceTensor,
    destinationTensor,
    sourceLayoutFact,
    destinationLayoutFact,
    intrinsic,
    sourceLayout,
    destinationLayout,
    sourceSpanElements,
    destinationSpanElements,
    entrySubjectHash,
    loweringAuthorityMinted: false,
  });
}

export function normalizeCppCuteViewCopyOptions(
  options: PrepareVerifiedCppCuteViewCopySemanticsOptions,
): NormalizedCppCuteViewCopyOptions {
  const descriptors = closedDataRecord(options, ["limits", "signal"], "$.options", true);
  let limits: DecodeLimits;
  try {
    limits = resolveDecodeLimits(descriptors.limits?.value as Partial<DecodeLimits> | undefined);
  } catch (cause) {
    invalidRequest("$.options.limits", "invalid semantic decode limits", { cause });
  }
  const signal = descriptors.signal?.value as AbortSignal | undefined;
  return NATIVE_OBJECT_FREEZE({ limits, ...(signal === undefined ? {} : { signal }) });
}

export function validateCppCuteViewCopyEntryRequest(
  request: PrepareVerifiedCppCuteViewCopySemanticsRequest,
): string {
  const descriptors = closedDataRecord(request, ["entryId"], "$.request");
  const entryId = descriptors.entryId?.value;
  if (typeof entryId !== "string" || entryId.length === 0) {
    invalidRequest("$.request.entryId", "entryId must be a non-empty string");
  }
  return entryId;
}

export function throwIfCppCuteViewCopyAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  let aborted: unknown;
  try {
    aborted = ABORT_SIGNAL_ABORTED_GETTER === undefined
      ? undefined
      : NATIVE_REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []);
  } catch (cause) {
    invalidRequest("$.options.signal", "signal is not a native AbortSignal", { cause });
  }
  if (aborted === true) {
    cppCuteViewCopyFailure(
      "BG-COMPILER-CPP-CUTE-VIEW-COPY-CANCELLED",
      "$.options.signal",
      "view-copy lowering was aborted",
    );
  }
}

export function cppCuteViewCopyFailure(
  code: CppCuteViewCopyLoweringErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteViewCopyLoweringError(code, path, message, options);
}

function selectedViewCopyEntry(payload: CppCuteFrontendPayloadV3, entryId: string) {
  if (payload.outcome.kind !== "accepted" ||
      payload.outcome.selectedEntryIds.length !== 1 ||
      payload.outcome.selectedEntryIds[0] !== entryId) {
    cppCuteViewCopyFailure(
      "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-ENTRY",
      "$.artifact.outcome.selectedEntryIds",
      "initial view-copy lowering requires exactly the explicitly requested selected entry",
    );
  }
  const entry = payload.entries.find((candidate) => candidate.entryId === entryId);
  if (entry === undefined) inconsistent("$.request.entryId", "selected entry disappeared from the authorized artifact");
  if (entry.kind !== "view-copy") {
    cppCuteViewCopyFailure(
      "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-ENTRY",
      "$.request.entryId",
      "selected entry is not a view-copy",
    );
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
  if (expression?.kind !== "target-intrinsic") {
    inconsistent("$.artifact.entry.operationExpressionId", "copy expression disappeared or changed kind");
  }
  const fact = payload.facts.find((candidate) => candidate.factId === expression.intrinsicFactId);
  if (fact?.kind !== "target-intrinsic") {
    inconsistent("$.artifact.entry.operationExpressionId", "copy intrinsic fact disappeared or changed kind");
  }
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

function positiveRank2SpanElements(layout: LayoutExpr, path: string): bigint {
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

function invalidRequest(path: string, message: string, options?: ErrorOptions): never {
  cppCuteViewCopyFailure("BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST", path, message, options);
}

function inconsistent(path: string, message: string): never {
  cppCuteViewCopyFailure("BG-COMPILER-CPP-CUTE-VIEW-COPY-INCONSISTENT-ARTIFACT", path, message);
}

function unsupportedProfile(path: string, message: string): never {
  cppCuteViewCopyFailure("BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-PROFILE", path, message);
}

function unsupportedLayout(path: string, message: string): never {
  cppCuteViewCopyFailure("BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT", path, message);
}
