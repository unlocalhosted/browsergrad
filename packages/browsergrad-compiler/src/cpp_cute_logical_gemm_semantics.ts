import {
  encodeWireU64,
  resolveDecodeLimits,
  type DecodeLimits,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import type {
  CppCuteAffineLayoutFactV1,
  CppCuteFrontendEntryV1,
  CppCuteFrontendPayloadV3,
  CppCuteLogicalGemmTileFactV1,
  CppCuteTensorFactV1,
} from "./cpp_cute_frontend_types.js";
import {
  CppCuteIntegerSemanticsError,
  evaluateStaticCppCuteIntegerExpr,
} from "./cpp_cute_integer_semantics.js";

const CAPTURED_OBJECT = Object;
const CAPTURED_REFLECT = Reflect;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const ABORT_SIGNAL_ABORTED_GETTER = Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export interface PrepareVerifiedCppCuteLogicalGemmTileSemanticsRequest {
  readonly entryId: string;
}

export interface PrepareVerifiedCppCuteLogicalGemmTileSemanticsOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

export interface PreparedVerifiedCppCuteLogicalGemmTileSemantics {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly entry: Extract<CppCuteFrontendEntryV1, { readonly kind: "logical-gemm-tile" }>;
  readonly fact: CppCuteLogicalGemmTileFactV1;
  readonly lhsTensor: CppCuteTensorFactV1;
  readonly rhsTensor: CppCuteTensorFactV1;
  readonly destinationTensor: CppCuteTensorFactV1;
  readonly m: WireU64;
  readonly n: WireU64;
  readonly k: WireU64;
  readonly loweringAuthorityMinted: false;
  readonly nativeExtractorEvidenceClaimed: false;
  readonly nativeFpControlEvidenceClaimed: false;
  readonly sourceBodyParityClaimed: false;
  readonly backendExecutionAuthorized: false;
}

export type CppCuteLogicalGemmTileLoweringErrorCode =
  | "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INVALID-REQUEST"
  | "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INCONSISTENT-ARTIFACT"
  | "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-UNSUPPORTED-ENTRY"
  | "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-UNSUPPORTED-PROFILE"
  | "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-RESOURCE-LIMIT";

export class CppCuteLogicalGemmTileLoweringError extends Error {
  constructor(
    readonly code: CppCuteLogicalGemmTileLoweringErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteLogicalGemmTileLoweringError";
  }
}

export interface NormalizedCppCuteLogicalGemmTileOptions {
  readonly limits: DecodeLimits;
  readonly signal?: AbortSignal;
}

export async function prepareVerifiedCppCuteLogicalGemmTileSemantics(
  artifact: VerifiedCppCuteFrontendArtifact,
  request: PrepareVerifiedCppCuteLogicalGemmTileSemanticsRequest,
  options: PrepareVerifiedCppCuteLogicalGemmTileSemanticsOptions = {},
): Promise<PreparedVerifiedCppCuteLogicalGemmTileSemantics> {
  const verified = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const normalizedOptions = normalizeCppCuteLogicalGemmTileOptions(options);
  throwIfCppCuteLogicalGemmTileAborted(normalizedOptions.signal);
  const entryId = validateCppCuteLogicalGemmTileRequest(request);
  const semantic = prepareCppCuteLogicalGemmTileSemantics(
    verified.envelope.payload,
    entryId,
    normalizedOptions,
  );
  return NATIVE_OBJECT_FREEZE({
    artifact,
    ...semantic,
    loweringAuthorityMinted: false,
    nativeExtractorEvidenceClaimed: false,
    nativeFpControlEvidenceClaimed: false,
    sourceBodyParityClaimed: false,
    backendExecutionAuthorized: false,
  });
}

export function prepareCppCuteLogicalGemmTileSemantics(
  payload: CppCuteFrontendPayloadV3,
  entryId: string,
  options: NormalizedCppCuteLogicalGemmTileOptions,
): Omit<
  PreparedVerifiedCppCuteLogicalGemmTileSemantics,
  "artifact" | "loweringAuthorityMinted" | "nativeExtractorEvidenceClaimed" | "backendExecutionAuthorized"
  | "nativeFpControlEvidenceClaimed" | "sourceBodyParityClaimed"
> {
  if (payload.outcome.kind !== "accepted" ||
      payload.outcome.selectedEntryIds.length !== 1 ||
      payload.outcome.selectedEntryIds[0] !== entryId) {
    failure(
      "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-UNSUPPORTED-ENTRY",
      "$.artifact.outcome.selectedEntryIds",
      "initial logical GEMM lowering requires exactly the explicitly requested selected entry",
    );
  }
  const entry = payload.entries.find((candidate) => candidate.entryId === entryId);
  if (entry === undefined) inconsistent("$.request.entryId", "selected entry disappeared from verified artifact");
  if (entry.kind !== "logical-gemm-tile") {
    failure(
      "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-UNSUPPORTED-ENTRY",
      "$.request.entryId",
      "selected entry is not a logical-gemm-tile typed artifact entry",
    );
  }
  const fact = payload.facts.find((candidate) => candidate.factId === entry.logicalGemmTileFactId);
  if (fact?.kind !== "logical-gemm-tile") {
    inconsistent("$.artifact.entry.logicalGemmTileFactId", "selected logical GEMM fact disappeared or changed kind");
  }
  const lhsTensor = tensor(payload, fact.lhsTensorFactId, "$.artifact.fact.lhsTensorFactId");
  const rhsTensor = tensor(payload, fact.rhsTensorFactId, "$.artifact.fact.rhsTensorFactId");
  const destinationTensor = tensor(
    payload,
    fact.destinationTensorFactId,
    "$.artifact.fact.destinationTensorFactId",
  );
  const lhsShape = denseShape(payload, lhsTensor, "$.artifact.lhs", options.limits);
  const rhsShape = denseShape(payload, rhsTensor, "$.artifact.rhs", options.limits);
  const destinationShape = denseShape(payload, destinationTensor, "$.artifact.destination", options.limits);
  if (lhsShape[0] !== destinationShape[0] || lhsShape[1] !== rhsShape[0] ||
      rhsShape[1] !== destinationShape[1]) {
    inconsistent("$.artifact.fact", "verified logical GEMM tensor shape relation drifted");
  }
  throwIfCppCuteLogicalGemmTileAborted(options.signal);
  return NATIVE_OBJECT_FREEZE({
    entry,
    fact,
    lhsTensor,
    rhsTensor,
    destinationTensor,
    m: encodeWireU64(lhsShape[0]),
    n: encodeWireU64(rhsShape[1]),
    k: encodeWireU64(lhsShape[1]),
  });
}

export function normalizeCppCuteLogicalGemmTileOptions(
  options: PrepareVerifiedCppCuteLogicalGemmTileSemanticsOptions,
): NormalizedCppCuteLogicalGemmTileOptions {
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

export function validateCppCuteLogicalGemmTileRequest(
  request: PrepareVerifiedCppCuteLogicalGemmTileSemanticsRequest,
): string {
  const descriptors = closedDataRecord(request, ["entryId"], "$.request");
  const entryId = descriptors.entryId?.value;
  if (typeof entryId !== "string" || entryId.length === 0) {
    invalidRequest("$.request.entryId", "entryId must be a non-empty string");
  }
  return entryId;
}

export function throwIfCppCuteLogicalGemmTileAborted(signal: AbortSignal | undefined): void {
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
    failure(
      "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-CANCELLED",
      "$.options.signal",
      "logical GEMM lowering was aborted",
    );
  }
}

export function cppCuteLogicalGemmTileFailure(
  code: CppCuteLogicalGemmTileLoweringErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  failure(code, path, message, options);
}

function tensor(payload: CppCuteFrontendPayloadV3, factId: string, path: string): CppCuteTensorFactV1 {
  const fact = payload.facts.find((candidate) => candidate.factId === factId);
  if (fact?.kind !== "tensor") inconsistent(path, "logical GEMM tensor fact disappeared or changed kind");
  return fact;
}

function denseShape(
  payload: CppCuteFrontendPayloadV3,
  tensorFact: CppCuteTensorFactV1,
  path: string,
  limits: DecodeLimits,
): readonly [bigint, bigint] {
  const layout = payload.facts.find((candidate) => candidate.factId === tensorFact.layoutFactId);
  if (layout?.kind !== "affine-layout") inconsistent(`${path}.layoutFactId`, "logical GEMM layout disappeared");
  return staticRank2Shape(layout, path, limits);
}

function staticRank2Shape(
  layout: CppCuteAffineLayoutFactV1,
  path: string,
  limits: DecodeLimits,
): readonly [bigint, bigint] {
  if (layout.shape.kind !== "tuple" || layout.shape.elements.length !== 2 ||
      layout.shape.elements[0]?.kind !== "scalar" || layout.shape.elements[1]?.kind !== "scalar") {
    inconsistent(`${path}.layout`, "verified dense rank-2 logical GEMM layout drifted");
  }
  try {
    const first = evaluateStaticCppCuteIntegerExpr(layout.shape.elements[0].value, {
      path: `${path}.layout.shape[0]`,
      limits,
    });
    const second = evaluateStaticCppCuteIntegerExpr(layout.shape.elements[1].value, {
      path: `${path}.layout.shape[1]`,
      limits,
    });
    if (first === undefined || second === undefined || first <= 0n || second <= 0n) {
      inconsistent(`${path}.layout.shape`, "verified positive static logical GEMM shape drifted");
    }
    return [first, second];
  } catch (cause) {
    if (!(cause instanceof CppCuteIntegerSemanticsError)) throw cause;
    failure(
      cause.kind === "resource-limit"
        ? "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INCONSISTENT-ARTIFACT",
      cause.path,
      cause.message,
      { cause },
    );
  }
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
    descriptors = NATIVE_REFLECT_APPLY(
      NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS,
      CAPTURED_OBJECT,
      [value],
    ) as PropertyDescriptorMap;
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
  failure("BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INVALID-REQUEST", path, message, options);
}

function inconsistent(path: string, message: string): never {
  failure("BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INCONSISTENT-ARTIFACT", path, message);
}

function failure(
  code: CppCuteLogicalGemmTileLoweringErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteLogicalGemmTileLoweringError(code, path, message, options);
}
