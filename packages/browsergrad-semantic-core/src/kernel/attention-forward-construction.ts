import { verifyLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { DimExpr } from "../layout/dim-expr.js";
import type { IndexExpr } from "../layout/model.js";
import { layoutArtifactPayload } from "../layout/trace.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import {
  I64_MAX,
  encodeWireI64,
  parseWireU64,
  wireIntegerToBigInt,
  type WireU64,
} from "../schema/integers.js";
import { isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS, resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  ATTENTION_FORWARD_ARTIFACT_SCHEMA,
  attentionForwardArtifactPayload,
  verifyAttentionForwardArtifact,
  type VerifiedAttentionForwardArtifact,
} from "./attention-forward-artifact.js";
import {
  INITIAL_ATTENTION_FORWARD_MAX_DEPTH,
  INITIAL_ATTENTION_FORWARD_MAX_DIMENSION,
  INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY,
  attentionForwardDefaultScaleBits,
} from "./attention-forward-model.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.attention-forward-construction",
  version: "1",
});

const ROLE_IDS = Object.freeze({
  query: { allocationId: "queryAllocation", aliasSetId: "queryAlias", indexMapId: "queryIndexMap", viewId: "queryView" },
  key: { allocationId: "keyAllocation", aliasSetId: "keyAlias", indexMapId: "keyIndexMap", viewId: "keyView" },
  value: { allocationId: "valueAllocation", aliasSetId: "valueAlias", indexMapId: "valueIndexMap", viewId: "valueView" },
  destination: { allocationId: "destinationAllocation", aliasSetId: "destinationAlias", indexMapId: "destinationIndexMap", viewId: "destinationView" },
});
const OPERATION_ID = "attentionForward";

export interface CreateVerifiedDenseAttentionForwardArtifactsRequest {
  readonly batch: WireU64;
  readonly heads: WireU64;
  readonly queryLength: WireU64;
  readonly keyLength: WireU64;
  readonly queryDepth: WireU64;
  readonly valueDepth: WireU64;
  readonly causal: boolean;
}

export interface AttentionForwardArtifactConstructionOptions {
  readonly producer?: WireProducer;
  readonly layoutArtifactId?: string;
  readonly kernelArtifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

export interface AttentionForwardArtifactRole {
  readonly allocationId: string;
  readonly indexMapId: string;
  readonly viewId: string;
}

export interface VerifiedAttentionForwardArtifacts {
  readonly layout: VerifiedLayoutArtifact;
  readonly kernel: VerifiedAttentionForwardArtifact;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly operationId: string;
  readonly query: AttentionForwardArtifactRole;
  readonly key: AttentionForwardArtifactRole;
  readonly value: AttentionForwardArtifactRole;
  readonly destination: AttentionForwardArtifactRole;
}

/**
 * Constructs the initial positive static dense rank-4 f32 attention meaning.
 * Layouts, exact default scale bits, numerical policy, effects, mask meaning,
 * phase ordering, and alias rules are derived rather than caller-authored.
 */
export async function createVerifiedDenseAttentionForwardArtifacts(
  request: CreateVerifiedDenseAttentionForwardArtifactsRequest,
  options: AttentionForwardArtifactConstructionOptions = {},
): Promise<VerifiedAttentionForwardArtifacts> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits);
  const object = closedRecord(snapshot, [
    "batch", "heads", "queryLength", "keyLength", "queryDepth", "valueDepth", "causal",
  ], [
    "batch", "heads", "queryLength", "keyLength", "queryDepth", "valueDepth", "causal",
  ], "$" );
  const batch = initialDimension(parseWireU64(object.batch, "$.batch"), "$.batch");
  const heads = initialDimension(parseWireU64(object.heads, "$.heads"), "$.heads");
  const queryLength = initialDimension(
    parseWireU64(object.queryLength, "$.queryLength"),
    "$.queryLength",
  );
  const keyLength = initialDimension(parseWireU64(object.keyLength, "$.keyLength"), "$.keyLength");
  const queryDepth = initialDepth(parseWireU64(object.queryDepth, "$.queryDepth"), "$.queryDepth");
  const valueDepth = initialDepth(parseWireU64(object.valueDepth, "$.valueDepth"), "$.valueDepth");
  if (typeof object.causal !== "boolean") constructionError("$.causal", "must be a boolean");
  const causal = object.causal;

  const queryShape = [batch, heads, queryLength, queryDepth] as const;
  const keyShape = [batch, heads, keyLength, queryDepth] as const;
  const valueShape = [batch, heads, keyLength, valueDepth] as const;
  const destinationShape = [batch, heads, queryLength, valueDepth] as const;
  const layout = await verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.layoutArtifactId,
    requiredExtensions: [],
    payload: {
      symbols: [],
      constraints: [],
      allocations: [
        allocation(ROLE_IDS.query, denseByteLength(queryShape, "$.query")),
        allocation(ROLE_IDS.key, denseByteLength(keyShape, "$.key")),
        allocation(ROLE_IDS.value, denseByteLength(valueShape, "$.value")),
        allocation(ROLE_IDS.destination, denseByteLength(destinationShape, "$.destination")),
      ],
      indexMaps: [
        denseIndexMap(ROLE_IDS.query.indexMapId, queryShape),
        denseIndexMap(ROLE_IDS.key.indexMapId, keyShape),
        denseIndexMap(ROLE_IDS.value.indexMapId, valueShape),
        denseIndexMap(ROLE_IDS.destination.indexMapId, destinationShape),
      ],
      views: [
        denseView(ROLE_IDS.query, queryShape),
        denseView(ROLE_IDS.key, keyShape),
        denseView(ROLE_IDS.value, valueShape),
        denseView(ROLE_IDS.destination, destinationShape),
      ],
    },
  }, { limits: normalizedOptions.limits });

  const layoutPayload = layoutArtifactPayload(layout);
  const query = role(layoutPayload, 0, "query");
  const key = role(layoutPayload, 1, "key");
  const inputValue = role(layoutPayload, 2, "value");
  const destination = role(layoutPayload, 3, "destination");
  const layoutSemanticHash = await hashSemanticArtifact(layout, { limits: normalizedOptions.limits });
  const kernel = await verifyAttentionForwardArtifact({
    schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.kernelArtifactId,
    requiredExtensions: [],
    payload: {
      layoutSemanticHash,
      operation: {
        operationId: OPERATION_ID,
        kind: "scaled-dot-product-attention-forward",
        version: { major: 1, minor: 0 },
        query: { viewId: query.viewId, access: "read" },
        key: { viewId: key.viewId, access: "read" },
        value: { viewId: inputValue.viewId, access: "read" },
        destination: { viewId: destination.viewId, access: "write" },
        mask: causal
          ? {
              kind: "causal",
              orientation: "upper-left",
              predicate: "key-index-less-equal-query-index",
            }
          : { kind: "none" },
        scale: {
          source: "inverse-square-root-query-depth-rounded-to-f32",
          value: attentionForwardDefaultScaleBits(wireIntegerToBigInt(queryDepth)),
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
        numerical: {
          ...INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY,
          inputDTypes: [...INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY.inputDTypes],
        },
        autodiff: {
          vjp: "not-defined",
          diagnosticId: "browsergrad.attention-forward-vjp-unavailable",
        },
        phases: { order: ["load", "score", "softmax", "weighted-value", "store"] },
        overlap: { kind: "forbid-all" },
      },
    },
  }, { layout, limits: normalizedOptions.limits });
  const kernelPayload = attentionForwardArtifactPayload(kernel);
  return Object.freeze({
    layout,
    kernel,
    layoutSemanticHash,
    kernelSemanticHash: await hashSemanticArtifact(kernel, { limits: normalizedOptions.limits }),
    operationId: kernelPayload.operation.operationId,
    query,
    key,
    value: inputValue,
    destination,
  });
}

interface NormalizedOptions {
  readonly producer: WireProducer;
  readonly layoutArtifactId: string;
  readonly kernelArtifactId: string;
  readonly limits: DecodeLimits;
}

function normalizeOptions(options: AttentionForwardArtifactConstructionOptions): NormalizedOptions {
  const snapshot = snapshotJson(options, DEFAULT_DECODE_LIMITS);
  const object = closedRecord(
    snapshot,
    ["producer", "layoutArtifactId", "kernelArtifactId", "limits"],
    [],
    "$options",
  );
  const rawLimits = object.limits;
  if (rawLimits !== undefined) {
    closedRecord(rawLimits, Object.keys(DEFAULT_DECODE_LIMITS), [], "$options.limits");
  }
  const limits = resolveDecodeLimits(rawLimits === undefined ? {} : rawLimits as Partial<DecodeLimits>);
  const producer = object.producer === undefined ? DEFAULT_PRODUCER : parseProducer(object.producer);
  return Object.freeze({
    producer,
    layoutArtifactId: object.layoutArtifactId === undefined
      ? "attention-forward-layout"
      : nonemptyString(object.layoutArtifactId, "$options.layoutArtifactId"),
    kernelArtifactId: object.kernelArtifactId === undefined
      ? "attention-forward-kernel"
      : nonemptyString(object.kernelArtifactId, "$options.kernelArtifactId"),
    limits,
  });
}

function parseProducer(value: JsonValue): WireProducer {
  const object = closedRecord(value, ["id", "version"], ["id", "version"], "$options.producer");
  return {
    id: nonemptyString(object.id, "$options.producer.id"),
    version: nonemptyString(object.version, "$options.producer.version"),
  };
}

function allocation(ids: typeof ROLE_IDS[keyof typeof ROLE_IDS], byteLength: WireU64) {
  return {
    allocationId: ids.allocationId,
    byteLength: constant(byteLength),
    memorySpace: { kind: "global" as const },
    alignmentBytes: 4,
    aliasSetId: ids.aliasSetId,
  };
}

function denseIndexMap(indexMapId: string, shape: readonly WireU64[]) {
  const strides = shape.map((_, axis) => shape
    .slice(axis + 1)
    .reduce((product, extent) => product * wireIntegerToBigInt(extent), 1n));
  const terms: IndexExpr[] = strides.map((stride, axis) => stride === 1n
    ? { kind: "coordinate", axis }
    : {
        kind: "mul",
        lhs: { kind: "coordinate", axis },
        rhs: { kind: "const", value: encodeWireI64(stride) },
      });
  return {
    indexMapId,
    coordinateRank: shape.length,
    locationUnit: "element" as const,
    location: { kind: "add" as const, terms },
    inBounds: { kind: "bool" as const, value: true },
  };
}

function denseView(ids: typeof ROLE_IDS[keyof typeof ROLE_IDS], shape: readonly WireU64[]) {
  return {
    viewId: ids.viewId,
    allocationId: ids.allocationId,
    dtype: "f32" as const,
    byteOffset: constant(parseWireU64("0")),
    shape: shape.map(constant),
    indexMapId: ids.indexMapId,
    requiredAlignmentBytes: 4,
  };
}

function constant(value: WireU64): DimExpr {
  const bigint = wireIntegerToBigInt(value);
  if (bigint > I64_MAX) throw new Error("internal: admitted attention dimension exceeded signed i64");
  return { kind: "const", value: encodeWireI64(bigint) };
}

function initialDimension(value: WireU64, path: string): WireU64 {
  const bigint = wireIntegerToBigInt(value);
  if (bigint === 0n || bigint > INITIAL_ATTENTION_FORWARD_MAX_DIMENSION) {
    constructionError(path, "must be positive and fit the portable u32 indexing profile");
  }
  return value;
}

function initialDepth(value: WireU64, path: string): WireU64 {
  initialDimension(value, path);
  if (wireIntegerToBigInt(value) > INITIAL_ATTENTION_FORWARD_MAX_DEPTH) {
    constructionError(path, `exceeds the initial depth limit ${INITIAL_ATTENTION_FORWARD_MAX_DEPTH}`);
  }
  return value;
}

function denseByteLength(shape: readonly WireU64[], path: string): WireU64 {
  const elements = shape.reduce((product, extent) => product * wireIntegerToBigInt(extent), 1n);
  const bytes = elements * 4n;
  if (bytes > I64_MAX) {
    constructionError(path, "dense f32 allocation byte length exceeds the signed-i64 layout profile");
  }
  return parseWireU64(bytes.toString(), `${path}.byteLength`);
}

function role(
  payload: ReturnType<typeof layoutArtifactPayload>,
  index: number,
  name: string,
): AttentionForwardArtifactRole {
  const view = payload.views[index];
  const allocation = payload.allocations[index];
  const indexMap = payload.indexMaps[index];
  if (view === undefined || allocation === undefined || indexMap === undefined) {
    throw new Error(`internal: ${name} attention role disappeared after verification`);
  }
  return Object.freeze({
    allocationId: allocation.allocationId,
    indexMapId: indexMap.indexMapId,
    viewId: view.viewId,
  });
}

function snapshotJson(value: unknown, limits: Partial<DecodeLimits>): JsonValue {
  return parseWireJson(canonicalizeJson(value, { limits }), { limits });
}

function closedRecord(
  value: JsonValue | undefined,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
): JsonObject {
  if (value === undefined || !isJsonObject(value)) constructionError(path, "must be a plain JSON object");
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) constructionError(path, `has unknown fields: ${unknown.sort().join(", ")}`);
  for (const field of requiredFields) {
    if (value[field] === undefined) constructionError(`${path}.${field}`, "is required");
  }
  return value;
}

function nonemptyString(value: JsonValue | undefined, path: string): string {
  if (typeof value !== "string" || value.length === 0) constructionError(path, "must be a non-empty string");
  return value;
}

function constructionError(path: string, message: string): never {
  throw schemaError(
    SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
    `attention-forward construction request ${message}`,
    { path },
  );
}
