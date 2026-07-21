import { unwrapLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { DimExpr } from "../layout/dim-expr.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { parseFloatBits, type FloatBits } from "../schema/float-bits.js";
import { hashCanonicalJson, hashSemanticArtifact } from "../schema/hash.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import type {
  AttentionForwardMask,
  AttentionForwardOperation,
  AttentionForwardReadEffect,
} from "./attention-forward-model.js";
import {
  INITIAL_ATTENTION_FORWARD_MAX_DEPTH,
  INITIAL_ATTENTION_FORWARD_MAX_DIMENSION,
  INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY,
  attentionForwardDefaultScaleBits,
} from "./attention-forward-model.js";

export const ATTENTION_FORWARD_ARTIFACT_SCHEMA = "browsergrad.kernel.attention-forward";
export const ATTENTION_FORWARD_ARTIFACT_MAJOR = 1;
export const ATTENTION_FORWARD_ARTIFACT_MINOR = 0;

const AUTHORITY = Object.freeze({
  schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
  major: ATTENTION_FORWARD_ARTIFACT_MAJOR,
});

export type AttentionForwardArtifactPayloadV1 = JsonObject & {
  readonly layoutSemanticHash: string;
  readonly operation: AttentionForwardOperation;
};

export type VerifiedAttentionForwardArtifact = VerifiedArtifact<AttentionForwardArtifactPayloadV1>;

export interface AttentionForwardArtifactVerificationOptions {
  readonly layout: VerifiedLayoutArtifact;
  readonly limits?: Partial<DecodeLimits>;
}

interface RawPayload {
  readonly layoutSemanticHash: string;
  readonly operation: AttentionForwardOperation;
}

export async function verifyAttentionForwardArtifact(
  value: unknown,
  options: AttentionForwardArtifactVerificationOptions,
): Promise<VerifiedAttentionForwardArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  const envelope = validateWireEnvelope(value, {
    schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
    supportedMajor: ATTENTION_FORWARD_ARTIFACT_MAJOR,
    supportedMinor: ATTENTION_FORWARD_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const raw = parsePayload(envelope.payload);
  const layoutSemanticHash = await hashSemanticArtifact(options.layout, { limits });
  if (raw.layoutSemanticHash !== layoutSemanticHash) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch,
      "$.payload.layoutSemanticHash",
      "attention-forward artifact does not reference the supplied verified layout semantics",
    );
  }

  const provisional = remapOperationId(raw, "provisional");
  const scopeDigest = await hashCanonicalJson({
    domain: "browsergrad.kernel.attention-forward-id-scope.v1",
    payload: provisional,
  }, { limits });
  const normalized = remapOperationId(raw, scopeDigest);
  verifySemantics(normalized, options.layout, limits);

  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: normalized as unknown as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  return verifyWireArtifact(normalizedEnvelope, {
    schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
    supportedMajor: ATTENTION_FORWARD_ARTIFACT_MAJOR,
    supportedMinor: ATTENTION_FORWARD_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }, AUTHORITY) as VerifiedAttentionForwardArtifact;
}

export async function decodeAttentionForwardArtifact(
  bytes: Uint8Array,
  options: AttentionForwardArtifactVerificationOptions,
): Promise<VerifiedAttentionForwardArtifact> {
  return verifyAttentionForwardArtifact(
    decodeWireJson(bytes, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
}

export function attentionForwardArtifactPayload(
  artifact: VerifiedAttentionForwardArtifact,
): AttentionForwardArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact, AUTHORITY);
  if (envelope.schema !== ATTENTION_FORWARD_ARTIFACT_SCHEMA
    || envelope.version.major !== ATTENTION_FORWARD_ARTIFACT_MAJOR) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidArtifact,
      "$",
      "verified artifact is not a browsergrad.kernel.attention-forward@1 artifact",
    );
  }
  return envelope.payload;
}

function parsePayload(value: JsonValue): RawPayload {
  const object = closedObject(value, ["layoutSemanticHash", "operation"], "$.payload");
  const layoutSemanticHash = stringValue(
    field(object, "layoutSemanticHash", "$.payload"),
    "$.payload.layoutSemanticHash",
  );
  if (!/^[0-9a-f]{64}$/u.test(layoutSemanticHash)) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidArtifact,
      "$.payload.layoutSemanticHash",
      "layout semantic hash must be 64 lowercase hexadecimal digits",
    );
  }
  return {
    layoutSemanticHash,
    operation: parseOperation(field(object, "operation", "$.payload"), "$.payload.operation"),
  };
}

function parseOperation(value: JsonValue, path: string): AttentionForwardOperation {
  const object = closedObject(value, [
    "operationId", "kind", "version", "query", "key", "value", "destination",
    "mask", "scale", "inputDomain", "score", "softmax", "weightedValue",
    "numerical", "autodiff", "phases", "overlap",
  ], path);
  if (object.kind !== "scaled-dot-product-attention-forward") {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidArtifact,
      `${path}.kind`,
      "expected scaled-dot-product-attention-forward operation",
    );
  }
  const version = closedObject(field(object, "version", path), ["major", "minor"], `${path}.version`);
  if (version.major !== 1 || version.minor !== 0) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.version`,
      "attention-forward reader supports operation version 1.0 only",
    );
  }
  const query = parseReadEffect(field(object, "query", path), `${path}.query`);
  const key = parseReadEffect(field(object, "key", path), `${path}.key`);
  const inputValue = parseReadEffect(field(object, "value", path), `${path}.value`);
  const destination = closedObject(field(object, "destination", path), ["viewId", "access"], `${path}.destination`);
  requireLiteral(destination, "access", "write", `${path}.destination`);
  const mask = parseMask(field(object, "mask", path), `${path}.mask`);
  const scale = closedObject(field(object, "scale", path), ["source", "value"], `${path}.scale`);
  requireLiteral(
    scale,
    "source",
    "inverse-square-root-query-depth-rounded-to-f32",
    `${path}.scale`,
  );
  const scaleValue = parsePositiveFiniteF32(field(scale, "value", `${path}.scale`), `${path}.scale.value`);
  requireExactObject(
    field(object, "inputDomain", path),
    {
      query: "finite-f32",
      key: "finite-f32",
      value: "finite-f32",
      scaledScores: "finite-f32-required",
      onlineState: "finite-f32-required",
    },
    `${path}.inputDomain`,
  );
  requireExactObject(field(object, "score", path), {
    product: "multiply",
    reduction: "sum",
    reductionAxis: "query-key-depth",
    reductionOrder: "increasing-depth",
    scaleApplication: "after-reduction",
  }, `${path}.score`);
  requireExactObject(field(object, "softmax", path), {
    kind: "stable-max-subtracted",
    scope: "complete-logical-key-range",
    maximumOrder: "increasing-key",
    exponential: "natural-exp",
    sumOrder: "increasing-key",
    normalization: "divide-by-sum",
    fullyMaskedRows: "forbidden",
  }, `${path}.softmax`);
  requireExactObject(field(object, "weightedValue", path), {
    product: "multiply",
    reduction: "sum",
    reductionAxis: "key",
    reductionOrder: "increasing-key",
  }, `${path}.weightedValue`);
  const numerical = parseNumericalPolicy(field(object, "numerical", path), `${path}.numerical`);
  requireExactObject(field(object, "autodiff", path), {
    vjp: "not-defined",
    diagnosticId: "browsergrad.attention-forward-vjp-unavailable",
  }, `${path}.autodiff`);
  const phases = closedObject(field(object, "phases", path), ["order"], `${path}.phases`);
  const phaseOrder = arrayValue(field(phases, "order", `${path}.phases`), `${path}.phases.order`);
  const expectedPhases = ["load", "score", "softmax", "weighted-value", "store"];
  if (phaseOrder.length !== expectedPhases.length
    || phaseOrder.some((phase, index) => phase !== expectedPhases[index])) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.phases.order`,
      "attention-forward phase order must be load, score, softmax, weighted-value, store",
    );
  }
  const overlap = closedObject(field(object, "overlap", path), ["kind"], `${path}.overlap`);
  if (overlap.kind !== "forbid-all") {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.aliasConflict,
      `${path}.overlap.kind`,
      "attention-forward v1 requires pairwise-disjoint allocations",
    );
  }

  return {
    operationId: localId(field(object, "operationId", path), `${path}.operationId`),
    kind: "scaled-dot-product-attention-forward",
    version: { major: 1, minor: 0 },
    query,
    key,
    value: inputValue,
    destination: {
      viewId: localId(field(destination, "viewId", `${path}.destination`), `${path}.destination.viewId`),
      access: "write",
    },
    mask,
    scale: {
      source: "inverse-square-root-query-depth-rounded-to-f32",
      value: scaleValue,
    },
    inputDomain: {
      query: "finite-f32",
      key: "finite-f32",
      value: "finite-f32",
      scaledScores: "finite-f32-required",
      onlineState: "finite-f32-required",
    },
    score: {
      product: "multiply",
      reduction: "sum",
      reductionAxis: "query-key-depth",
      reductionOrder: "increasing-depth",
      scaleApplication: "after-reduction",
    },
    softmax: {
      kind: "stable-max-subtracted",
      scope: "complete-logical-key-range",
      maximumOrder: "increasing-key",
      exponential: "natural-exp",
      sumOrder: "increasing-key",
      normalization: "divide-by-sum",
      fullyMaskedRows: "forbidden",
    },
    weightedValue: {
      product: "multiply",
      reduction: "sum",
      reductionAxis: "key",
      reductionOrder: "increasing-key",
    },
    numerical,
    autodiff: {
      vjp: "not-defined",
      diagnosticId: "browsergrad.attention-forward-vjp-unavailable",
    },
    phases: { order: ["load", "score", "softmax", "weighted-value", "store"] },
    overlap: { kind: "forbid-all" },
  };
}

function parseMask(value: JsonValue, path: string): AttentionForwardMask {
  if (!isJsonObject(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  if (value.kind === "none") {
    closedObject(value, ["kind"], path);
    return { kind: "none" };
  }
  const object = closedObject(value, ["kind", "orientation", "predicate"], path);
  requireLiteral(object, "kind", "causal", path);
  requireLiteral(object, "orientation", "upper-left", path);
  requireLiteral(object, "predicate", "key-index-less-equal-query-index", path);
  return {
    kind: "causal",
    orientation: "upper-left",
    predicate: "key-index-less-equal-query-index",
  };
}

function parseReadEffect(value: JsonValue, path: string): AttentionForwardReadEffect {
  const object = closedObject(value, ["viewId", "access"], path);
  requireLiteral(object, "access", "read", path);
  return { viewId: localId(field(object, "viewId", path), `${path}.viewId`), access: "read" };
}

function parsePositiveFiniteF32(value: JsonValue, path: string): FloatBits {
  const parsed = parseFloatBits(value, path);
  if (parsed.dtype !== "f32") {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.dtype`, "attention scale must be f32 bits");
  }
  const bits = Number.parseInt(parsed.bits, 16) >>> 0;
  const exponent = (bits >>> 23) & 0xff;
  if ((bits & 0x8000_0000) !== 0 || (bits & 0x7fff_ffff) === 0 || exponent === 0xff) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, path, "attention scale must be positive, finite, and nonzero");
  }
  return parsed;
}

function parseNumericalPolicy(value: JsonValue, path: string): AttentionForwardOperation["numerical"] {
  const object = closedObject(value, Object.keys(INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY), path);
  for (const [name, expected] of Object.entries(INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY)) {
    const actual = object[name];
    if (Array.isArray(expected)) {
      const array = arrayValue(actual as JsonValue, `${path}.${name}`);
      if (array.length !== expected.length || array.some((entry, index) => entry !== expected[index])) {
        invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.${name}`, `attention-forward v1 requires the exact ${name} policy`);
      }
    } else if (actual !== expected) {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.${name}`, `attention-forward v1 requires ${name}=${expected}`);
    }
  }
  return {
    ...INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY,
    inputDTypes: [...INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY.inputDTypes],
  };
}

function requireExactObject(
  value: JsonValue,
  expected: Readonly<Record<string, string>>,
  path: string,
): void {
  const object = closedObject(value, Object.keys(expected), path);
  for (const [name, literal] of Object.entries(expected)) requireLiteral(object, name, literal, path);
}

function remapOperationId(raw: RawPayload, scope: string): AttentionForwardArtifactPayloadV1 {
  return {
    layoutSemanticHash: raw.layoutSemanticHash,
    operation: {
      ...raw.operation,
      operationId: scope === "provisional"
        ? "@kernel-operation/0"
        : `bg.entity.kernel-operation.scope-sha256.${scope}.ordinal.0`,
    },
  } as unknown as AttentionForwardArtifactPayloadV1;
}

function verifySemantics(
  payload: AttentionForwardArtifactPayloadV1,
  layoutArtifact: VerifiedLayoutArtifact,
  limits: DecodeLimits,
): void {
  const layout = unwrapLayoutArtifact(layoutArtifact);
  const views = new Map(layout.views.map((view) => [view.viewId, view]));
  const allocations = new Map(layout.allocations.map((allocation) => [allocation.allocationId, allocation]));
  const operation = payload.operation;
  const roles = [
    ["query", operation.query.viewId],
    ["key", operation.key.viewId],
    ["value", operation.value.viewId],
    ["destination", operation.destination.viewId],
  ] as const;
  const resolved = roles.map(([role, viewId]) => {
    const view = views.get(viewId);
    if (view === undefined) {
      invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, `$.payload.operation.${role}.viewId`, `unknown ${role} view`);
    }
    if (view.dtype !== "f32") {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.payload.operation.${role}.viewId`, "attention-forward v1 requires f32 views");
    }
    if (view.shape.length !== 4) {
      invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, `$.payload.operation.${role}.viewId`, "attention-forward v1 requires rank-4 views");
    }
    const allocation = allocations.get(view.allocationId);
    if (allocation === undefined) throw new Error("internal: verified attention allocation disappeared");
    if (allocation.memorySpace.kind !== "global") {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.payload.operation.${role}.viewId`, "attention-forward v1 requires global-memory views");
    }
    for (let axis = 0; axis < view.shape.length; axis += 1) {
      requireInitialDimension(view.shape[axis] as DimExpr, `$.payload.operation.${role}.viewId`, axis);
    }
    return view;
  });
  const [query, key, inputValue, destination] = resolved;
  requireSameDim(query!.shape[0]!, key!.shape[0]!, "batch dimensions");
  requireSameDim(query!.shape[0]!, inputValue!.shape[0]!, "batch dimensions");
  requireSameDim(query!.shape[0]!, destination!.shape[0]!, "batch dimensions");
  requireSameDim(query!.shape[1]!, key!.shape[1]!, "head dimensions");
  requireSameDim(query!.shape[1]!, inputValue!.shape[1]!, "head dimensions");
  requireSameDim(query!.shape[1]!, destination!.shape[1]!, "head dimensions");
  requireSameDim(query!.shape[3]!, key!.shape[3]!, "query/key depth dimensions");
  requireSameDim(key!.shape[2]!, inputValue!.shape[2]!, "key/value sequence dimensions");
  requireSameDim(query!.shape[2]!, destination!.shape[2]!, "query/output sequence dimensions");
  requireSameDim(inputValue!.shape[3]!, destination!.shape[3]!, "value/output depth dimensions");
  const queryDepth = constantDimensionValue(query!.shape[3]!);
  const valueDepth = constantDimensionValue(inputValue!.shape[3]!);
  if (queryDepth > INITIAL_ATTENTION_FORWARD_MAX_DEPTH
    || valueDepth > INITIAL_ATTENTION_FORWARD_MAX_DEPTH) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.operation",
      `attention-forward v1 limits query and value depth to ${INITIAL_ATTENTION_FORWARD_MAX_DEPTH}`,
    );
  }
  const expectedScale = attentionForwardDefaultScaleBits(queryDepth);
  if (operation.scale.value.bits !== expectedScale.bits) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.operation.scale.value.bits",
      "attention-forward scale does not match the exact f32 inverse square root of query depth",
    );
  }
  const allocationIds = resolved.map((view) => view!.allocationId);
  const aliasIds = allocationIds.map((id) => allocations.get(id)?.aliasSetId);
  if (new Set(allocationIds).size !== 4 || new Set(aliasIds).size !== 4) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.aliasConflict,
      "$.payload.operation.overlap",
      "attention-forward requires pairwise-disjoint allocation and alias-set identities",
    );
  }
  canonicalizeJson(payload, { limits });
}

function requireInitialDimension(value: DimExpr, path: string, axis: number): void {
  if (value.kind !== "const") {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.shape[${axis}]`,
      "attention-forward v1 requires positive static dimensions",
    );
  }
  const extent = BigInt(value.value);
  if (extent <= 0n || extent > INITIAL_ATTENTION_FORWARD_MAX_DIMENSION) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.shape[${axis}]`,
      "attention-forward v1 dimensions must be positive and fit the portable u32 profile",
    );
  }
}

function constantDimensionValue(value: DimExpr): bigint {
  if (value.kind !== "const") throw new Error("internal: admitted attention dimension became dynamic");
  return BigInt(value.value);
}

function requireSameDim(left: DimExpr, right: DimExpr, description: string): void {
  if (canonicalizeJson(left) !== canonicalizeJson(right)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, "$.payload.operation", `${description} must be canonically identical`);
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
    if (value[name] === undefined) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
    }
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  }
  return value;
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  return value;
}

function localId(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!/^[A-Za-z_][A-Za-z0-9_.:/-]{0,255}$/u.test(result)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "invalid local or canonical identifier");
  }
  return result;
}

function requireLiteral(object: JsonObject, name: string, expected: string, path: string): void {
  if (object[name] !== expected) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.${name}`,
      `attention-forward v1 requires ${name}=${expected}`,
    );
  }
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
