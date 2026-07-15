import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { hashCanonicalJson } from "../schema/hash.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { parseWireI64 } from "../schema/integers.js";
import { type DecodeLimits, resolveDecodeLimits } from "../schema/limits.js";
import { evaluateConstraintSet, type ShapeConstraint } from "./constraints.js";
import { evaluateDimExpr, type DimExpr, type DimSymbol } from "./dim-expr.js";
import { getBuiltinDType, type BuiltinDTypeId } from "./dtype.js";
import { evaluateIndexExpr, evaluatePredicateExpr } from "./index-eval.js";
import type { AllocationSpec, IndexExpr, IndexMap, MemorySpace, PredicateExpr, TensorView } from "./model.js";

export const LAYOUT_ARTIFACT_SCHEMA = "browsergrad.layout";
export const LAYOUT_ARTIFACT_MAJOR = 1;
export const LAYOUT_ARTIFACT_MINOR = 0;

export type LayoutArtifactPayloadV1 = JsonObject & {
  readonly symbols: readonly DimSymbol[];
  readonly constraints: readonly ShapeConstraint[];
  readonly allocations: readonly AllocationSpec[];
  readonly indexMaps: readonly IndexMap[];
  readonly views: readonly TensorView[];
};

export type VerifiedLayoutArtifact = VerifiedArtifact<LayoutArtifactPayloadV1>;

export interface LayoutArtifactVerificationOptions {
  readonly limits?: Partial<DecodeLimits>;
}

interface RawLayoutArtifact {
  readonly symbols: readonly DimSymbol[];
  readonly constraints: readonly ShapeConstraint[];
  readonly allocations: readonly AllocationSpec[];
  readonly indexMaps: readonly IndexMap[];
  readonly views: readonly TensorView[];
}

interface ParseState {
  readonly limits: DecodeLimits;
  nodes: number;
}

export async function verifyLayoutArtifact(
  value: unknown,
  options: LayoutArtifactVerificationOptions = {},
): Promise<VerifiedLayoutArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  const envelope = validateWireEnvelope(value, {
    schema: LAYOUT_ARTIFACT_SCHEMA,
    supportedMajor: LAYOUT_ARTIFACT_MAJOR,
    supportedMinor: LAYOUT_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const raw = parsePayload(envelope.payload, limits);
  validateReferences(raw);

  const provisional = remapEntityIds(raw, "provisional");
  const scopeDigest = await hashCanonicalJson({
    domain: "browsergrad.layout-id-scope.v1",
    payload: provisional,
  }, { limits });
  const normalized = remapEntityIds(raw, scopeDigest);
  verifySemantics(normalized, limits);

  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: normalized as unknown as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  return verifyWireArtifact(normalizedEnvelope, {
    schema: LAYOUT_ARTIFACT_SCHEMA,
    supportedMajor: LAYOUT_ARTIFACT_MAJOR,
    supportedMinor: LAYOUT_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }) as VerifiedLayoutArtifact;
}

export async function decodeLayoutArtifact(
  bytes: Uint8Array,
  options: LayoutArtifactVerificationOptions = {},
): Promise<VerifiedLayoutArtifact> {
  return verifyLayoutArtifact(decodeWireJson(bytes, options), options);
}

/** @internal Trace implementation uses the already-verified frozen payload. */
export function unwrapLayoutArtifact(artifact: VerifiedLayoutArtifact): LayoutArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact);
  if (envelope.schema !== LAYOUT_ARTIFACT_SCHEMA || envelope.version.major !== LAYOUT_ARTIFACT_MAJOR) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, "$", "verified artifact is not a browsergrad.layout@1 artifact");
  }
  return envelope.payload;
}

function parsePayload(value: JsonValue, limits: DecodeLimits): RawLayoutArtifact {
  const state: ParseState = { limits, nodes: 0 };
  const object = closedObject(value, ["symbols", "constraints", "allocations", "indexMaps", "views"], "$.payload");
  const symbols = arrayField(object, "symbols", "$.payload").map((entry, index) => parseSymbol(entry, state, `$.payload.symbols[${index}]`, 1));
  const constraints = arrayField(object, "constraints", "$.payload").map((entry, index) => parseConstraint(entry, state, `$.payload.constraints[${index}]`, 1));
  const allocations = arrayField(object, "allocations", "$.payload").map((entry, index) => parseAllocation(entry, state, `$.payload.allocations[${index}]`, 1));
  const indexMaps = arrayField(object, "indexMaps", "$.payload").map((entry, index) => parseIndexMap(entry, state, `$.payload.indexMaps[${index}]`, 1));
  const views = arrayField(object, "views", "$.payload").map((entry, index) => parseView(entry, state, `$.payload.views[${index}]`, 1));
  uniqueIds(symbols.map((symbol) => symbol.id), "$.payload.symbols");
  uniqueIds(allocations.map((allocation) => allocation.allocationId), "$.payload.allocations");
  uniqueIds(indexMaps.map((indexMap) => indexMap.indexMapId), "$.payload.indexMaps");
  uniqueIds(views.map((view) => view.viewId), "$.payload.views");
  return { symbols, constraints, allocations, indexMaps, views };
}

function parseSymbol(value: JsonValue, state: ParseState, path: string, depth: number): DimSymbol {
  consume(state, path, depth);
  const object = closedObject(value, ["id", "domain"], path);
  const id = localId(field(object, "id", path), `${path}.id`);
  if (id.startsWith("__bg_")) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.id`, "symbol IDs must not use the reserved __bg_ prefix");
  const domain = closedObject(field(object, "domain", path), ["min", "max"], `${path}.domain`, ["max"]);
  return {
    id,
    domain: {
      min: parseWireI64(field(domain, "min", `${path}.domain`), `${path}.domain.min`),
      ...(domain.max === undefined ? {} : { max: parseWireI64(domain.max, `${path}.domain.max`) }),
    },
  };
}

function parseConstraint(value: JsonValue, state: ParseState, path: string, depth: number): ShapeConstraint {
  consume(state, path, depth);
  const object = objectWithKind(value, path);
  switch (object.kind) {
    case "equal":
    case "lessEqual":
      exactFields(object, ["kind", "lhs", "rhs"], path);
      return {
        kind: object.kind,
        lhs: parseDimExpr(field(object, "lhs", path), state, `${path}.lhs`, depth + 1),
        rhs: parseDimExpr(field(object, "rhs", path), state, `${path}.rhs`, depth + 1),
      };
    case "nonNegative":
    case "positive":
      exactFields(object, ["kind", "value"], path);
      return { kind: object.kind, value: parseDimExpr(field(object, "value", path), state, `${path}.value`, depth + 1) };
    case "divisible":
      exactFields(object, ["kind", "value", "divisor"], path);
      return {
        kind: "divisible",
        value: parseDimExpr(field(object, "value", path), state, `${path}.value`, depth + 1),
        divisor: parseDimExpr(field(object, "divisor", path), state, `${path}.divisor`, depth + 1),
      };
    default: invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.kind`, `unknown constraint kind ${JSON.stringify(object.kind)}`);
  }
}

function parseAllocation(value: JsonValue, state: ParseState, path: string, depth: number): AllocationSpec {
  consume(state, path, depth);
  const object = closedObject(value, ["allocationId", "byteLength", "memorySpace", "alignmentBytes", "aliasSetId"], path);
  return {
    allocationId: localId(field(object, "allocationId", path), `${path}.allocationId`),
    byteLength: parseDimExpr(field(object, "byteLength", path), state, `${path}.byteLength`, depth + 1),
    memorySpace: parseMemorySpace(field(object, "memorySpace", path), `${path}.memorySpace`),
    alignmentBytes: alignment(field(object, "alignmentBytes", path), `${path}.alignmentBytes`),
    aliasSetId: localId(field(object, "aliasSetId", path), `${path}.aliasSetId`),
  };
}

function parseMemorySpace(value: JsonValue, path: string): MemorySpace {
  const object = objectWithKind(value, path);
  switch (object.kind) {
    case "host":
    case "global":
    case "constant":
      exactFields(object, ["kind"], path);
      return { kind: object.kind };
    case "shared": {
      exactFields(object, ["kind", "scope"], path);
      if (object.scope !== "subgroup" && object.scope !== "workgroup" && object.scope !== "cluster") {
        invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.scope`, "invalid shared-memory scope");
      }
      return { kind: "shared", scope: object.scope };
    }
    case "local":
      exactFields(object, ["kind", "scope"], path);
      if (object.scope !== "invocation") invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.scope`, "local scope must be invocation");
      return { kind: "local", scope: "invocation" };
    case "target":
      exactFields(object, ["kind", "targetId", "spaceId"], path);
      return {
        kind: "target",
        targetId: localId(field(object, "targetId", path), `${path}.targetId`),
        spaceId: localId(field(object, "spaceId", path), `${path}.spaceId`),
      };
    default: invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.kind`, `unknown memory-space kind ${JSON.stringify(object.kind)}`);
  }
}

function parseIndexMap(value: JsonValue, state: ParseState, path: string, depth: number): IndexMap {
  consume(state, path, depth);
  const object = closedObject(value, ["indexMapId", "coordinateRank", "locationUnit", "location", "inBounds"], path);
  const coordinateRank = nonnegativeSafeInteger(field(object, "coordinateRank", path), `${path}.coordinateRank`);
  if (coordinateRank > state.limits.maxRank) resource(`${path}.coordinateRank`, `rank ${coordinateRank} exceeds ${state.limits.maxRank}`);
  if (object.locationUnit !== "element" && object.locationUnit !== "byte") {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.locationUnit`, "location unit must be element or byte");
  }
  return {
    indexMapId: localId(field(object, "indexMapId", path), `${path}.indexMapId`),
    coordinateRank,
    locationUnit: object.locationUnit,
    location: parseIndexExpr(field(object, "location", path), coordinateRank, state, `${path}.location`, depth + 1),
    inBounds: parsePredicate(field(object, "inBounds", path), coordinateRank, state, `${path}.inBounds`, depth + 1),
  };
}

function parseView(value: JsonValue, state: ParseState, path: string, depth: number): TensorView {
  consume(state, path, depth);
  const object = closedObject(value, ["viewId", "allocationId", "dtype", "byteOffset", "shape", "indexMapId", "requiredAlignmentBytes"], path);
  const shape = arrayValue(field(object, "shape", path), `${path}.shape`)
    .map((entry, index) => parseDimExpr(entry, state, `${path}.shape[${index}]`, depth + 1));
  if (shape.length > state.limits.maxRank) resource(`${path}.shape`, `rank ${shape.length} exceeds ${state.limits.maxRank}`);
  const dtype = stringValue(field(object, "dtype", path), `${path}.dtype`);
  getBuiltinDType(dtype, `${path}.dtype`);
  return {
    viewId: localId(field(object, "viewId", path), `${path}.viewId`),
    allocationId: localId(field(object, "allocationId", path), `${path}.allocationId`),
    dtype: dtype as BuiltinDTypeId,
    byteOffset: parseDimExpr(field(object, "byteOffset", path), state, `${path}.byteOffset`, depth + 1),
    shape,
    indexMapId: localId(field(object, "indexMapId", path), `${path}.indexMapId`),
    requiredAlignmentBytes: alignment(field(object, "requiredAlignmentBytes", path), `${path}.requiredAlignmentBytes`),
  };
}

function parseDimExpr(value: JsonValue, state: ParseState, path: string, depth: number): DimExpr {
  consume(state, path, depth);
  const object = objectWithKind(value, path);
  switch (object.kind) {
    case "const":
      exactFields(object, ["kind", "value"], path);
      return { kind: "const", value: parseWireI64(field(object, "value", path), `${path}.value`) };
    case "symbol":
      exactFields(object, ["kind", "id"], path);
      return { kind: "symbol", id: localId(field(object, "id", path), `${path}.id`) };
    case "add": {
      exactFields(object, ["kind", "terms"], path);
      const terms = arrayField(object, "terms", path);
      if (terms.length === 0) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidDimExpr, `${path}.terms`, "add requires at least one term");
      return { kind: "add", terms: terms.map((entry, index) => parseDimExpr(entry, state, `${path}.terms[${index}]`, depth + 1)) };
    }
    case "mul":
      exactFields(object, ["kind", "lhs", "rhs"], path);
      return {
        kind: "mul",
        lhs: parseDimExpr(field(object, "lhs", path), state, `${path}.lhs`, depth + 1),
        rhs: parseDimExpr(field(object, "rhs", path), state, `${path}.rhs`, depth + 1),
      };
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      exactFields(object, ["kind", "value", "divisor"], path);
      return {
        kind: object.kind,
        value: parseDimExpr(field(object, "value", path), state, `${path}.value`, depth + 1),
        divisor: parseDimExpr(field(object, "divisor", path), state, `${path}.divisor`, depth + 1),
      };
    case "min":
    case "max": {
      exactFields(object, ["kind", "values"], path);
      const values = arrayField(object, "values", path);
      if (values.length === 0) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidDimExpr, `${path}.values`, `${object.kind} requires at least one value`);
      return { kind: object.kind, values: values.map((entry, index) => parseDimExpr(entry, state, `${path}.values[${index}]`, depth + 1)) };
    }
    default: invalid(LAYOUT_DIAGNOSTIC_CODES.invalidDimExpr, `${path}.kind`, `unknown dimension expression kind ${JSON.stringify(object.kind)}`);
  }
}

function parseIndexExpr(value: JsonValue, rank: number, state: ParseState, path: string, depth: number): IndexExpr {
  consume(state, path, depth);
  const object = objectWithKind(value, path);
  switch (object.kind) {
    case "const":
      exactFields(object, ["kind", "value"], path);
      return { kind: "const", value: parseWireI64(field(object, "value", path), `${path}.value`) };
    case "coordinate": {
      exactFields(object, ["kind", "axis"], path);
      const axis = nonnegativeSafeInteger(field(object, "axis", path), `${path}.axis`);
      if (axis >= rank) invalid(LAYOUT_DIAGNOSTIC_CODES.rankMismatch, `${path}.axis`, `coordinate axis ${axis} is outside rank ${rank}`);
      return { kind: "coordinate", axis };
    }
    case "dimension":
      exactFields(object, ["kind", "symbolId"], path);
      return { kind: "dimension", symbolId: localId(field(object, "symbolId", path), `${path}.symbolId`) };
    case "add": {
      exactFields(object, ["kind", "terms"], path);
      const terms = arrayField(object, "terms", path);
      if (terms.length === 0) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, `${path}.terms`, "index add requires at least one term");
      return { kind: "add", terms: terms.map((entry, index) => parseIndexExpr(entry, rank, state, `${path}.terms[${index}]`, depth + 1)) };
    }
    case "mul":
      exactFields(object, ["kind", "lhs", "rhs"], path);
      return {
        kind: "mul",
        lhs: parseIndexExpr(field(object, "lhs", path), rank, state, `${path}.lhs`, depth + 1),
        rhs: parseIndexExpr(field(object, "rhs", path), rank, state, `${path}.rhs`, depth + 1),
      };
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      exactFields(object, ["kind", "value", "divisor"], path);
      return {
        kind: object.kind,
        value: parseIndexExpr(field(object, "value", path), rank, state, `${path}.value`, depth + 1),
        divisor: parseIndexExpr(field(object, "divisor", path), rank, state, `${path}.divisor`, depth + 1),
      };
    case "min":
    case "max": {
      exactFields(object, ["kind", "values"], path);
      const values = arrayField(object, "values", path);
      if (values.length === 0) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, `${path}.values`, `index ${object.kind} requires at least one value`);
      return { kind: object.kind, values: values.map((entry, index) => parseIndexExpr(entry, rank, state, `${path}.values[${index}]`, depth + 1)) };
    }
    default: invalid(LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, `${path}.kind`, `unknown index expression kind ${JSON.stringify(object.kind)}`);
  }
}

function parsePredicate(value: JsonValue, rank: number, state: ParseState, path: string, depth: number): PredicateExpr {
  consume(state, path, depth);
  const object = objectWithKind(value, path);
  switch (object.kind) {
    case "bool":
      exactFields(object, ["kind", "value"], path);
      if (typeof object.value !== "boolean") invalid(LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, `${path}.value`, "predicate value must be boolean");
      return { kind: "bool", value: object.value };
    case "equal":
    case "lessEqual":
      exactFields(object, ["kind", "lhs", "rhs"], path);
      return {
        kind: object.kind,
        lhs: parseIndexExpr(field(object, "lhs", path), rank, state, `${path}.lhs`, depth + 1),
        rhs: parseIndexExpr(field(object, "rhs", path), rank, state, `${path}.rhs`, depth + 1),
      };
    case "and":
    case "or": {
      exactFields(object, ["kind", "values"], path);
      const values = arrayField(object, "values", path);
      if (values.length === 0) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, `${path}.values`, `${object.kind} requires at least one predicate`);
      return { kind: object.kind, values: values.map((entry, index) => parsePredicate(entry, rank, state, `${path}.values[${index}]`, depth + 1)) };
    }
    case "not":
      exactFields(object, ["kind", "value"], path);
      return { kind: "not", value: parsePredicate(field(object, "value", path), rank, state, `${path}.value`, depth + 1) };
    default: invalid(LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, `${path}.kind`, `unknown predicate kind ${JSON.stringify(object.kind)}`);
  }
}

function validateReferences(raw: RawLayoutArtifact): void {
  const allocations = new Set(raw.allocations.map((entry) => entry.allocationId));
  const indexMaps = new Map(raw.indexMaps.map((entry) => [entry.indexMapId, entry]));
  for (const [index, view] of raw.views.entries()) {
    if (!allocations.has(view.allocationId)) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.danglingReference, `$.payload.views[${index}].allocationId`, `unknown allocation ${view.allocationId}`);
    }
    const indexMap = indexMaps.get(view.indexMapId);
    if (indexMap === undefined) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.danglingReference, `$.payload.views[${index}].indexMapId`, `unknown index map ${view.indexMapId}`);
    }
    if (indexMap.coordinateRank !== view.shape.length) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.rankMismatch, `$.payload.views[${index}].shape`, `view rank ${view.shape.length} does not match index-map rank ${indexMap.coordinateRank}`);
    }
  }
}

function remapEntityIds(raw: RawLayoutArtifact, scope: string): LayoutArtifactPayloadV1 {
  const id = (kind: string, ordinal: number): string => scope === "provisional"
    ? `@${kind}/${ordinal}`
    : `bg.entity.${kind}.scope-sha256.${scope}.ordinal.${ordinal}`;
  const allocationIds = new Map(raw.allocations.map((entry, index) => [entry.allocationId, id("allocation", index)]));
  const indexMapIds = new Map(raw.indexMaps.map((entry, index) => [entry.indexMapId, id("index-map", index)]));
  const aliasIds = new Map<string, string>();
  for (const allocation of raw.allocations) {
    if (!aliasIds.has(allocation.aliasSetId)) aliasIds.set(allocation.aliasSetId, id("alias-set", aliasIds.size));
  }
  return {
    symbols: raw.symbols,
    constraints: raw.constraints,
    allocations: raw.allocations.map((entry) => ({
      ...entry,
      allocationId: requiredMap(allocationIds, entry.allocationId),
      aliasSetId: requiredMap(aliasIds, entry.aliasSetId),
    })),
    indexMaps: raw.indexMaps.map((entry) => ({
      ...entry,
      indexMapId: requiredMap(indexMapIds, entry.indexMapId),
    })),
    views: raw.views.map((entry, index) => ({
      ...entry,
      viewId: id("view", index),
      allocationId: requiredMap(allocationIds, entry.allocationId),
      indexMapId: requiredMap(indexMapIds, entry.indexMapId),
    })),
  } as unknown as LayoutArtifactPayloadV1;
}

function verifySemantics(payload: LayoutArtifactPayloadV1, limits: DecodeLimits): void {
  const environment = { symbols: payload.symbols };
  const constraintResult = evaluateConstraintSet(payload.constraints, environment, { limits });
  if (constraintResult.kind === "violated") {
    invalid(LAYOUT_DIAGNOSTIC_CODES.constraintViolation, `$.payload.constraints[${constraintResult.constraintIndex}]`, "statically violated shape constraint");
  }
  const symbolMinimum = new Map(payload.symbols.map((symbol) => [symbol.id, BigInt(symbol.domain.min)]));
  for (const [index, constraint] of payload.constraints.entries()) {
    if (constraint.kind === "divisible") requireProvablyPositive(constraint.divisor, symbolMinimum, `$.payload.constraints[${index}].divisor`);
    validateDimDivisorsInConstraint(constraint, symbolMinimum, `$.payload.constraints[${index}]`);
  }
  for (const [index, allocation] of payload.allocations.entries()) {
    validateDimDivisors(allocation.byteLength, symbolMinimum, `$.payload.allocations[${index}].byteLength`);
    const result = evaluateDimExpr(allocation.byteLength, environment, { limits });
    if (result.kind === "resolved" && (result.value < 0n || result.value > ((1n << 64n) - 1n))) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, `$.payload.allocations[${index}].byteLength`, "allocation byte length must resolve to u64");
    }
  }
  const allocations = new Map(payload.allocations.map((entry) => [entry.allocationId, entry]));
  for (const [index, indexMap] of payload.indexMaps.entries()) {
    validateIndexDivisors(indexMap.location, symbolMinimum, `$.payload.indexMaps[${index}].location`);
    validatePredicateDivisors(indexMap.inBounds, symbolMinimum, `$.payload.indexMaps[${index}].inBounds`);
    const coordinates = Array.from({ length: indexMap.coordinateRank }, () => 0n);
    evaluateIndexExpr(indexMap.location, { coordinateRank: indexMap.coordinateRank, coordinates, dimensions: environment }, { limits });
    evaluatePredicateExpr(indexMap.inBounds, { coordinateRank: indexMap.coordinateRank, coordinates, dimensions: environment }, { limits });
  }
  const indexMaps = new Map(payload.indexMaps.map((entry) => [entry.indexMapId, entry]));
  for (const [index, view] of payload.views.entries()) {
    const allocation = allocations.get(view.allocationId);
    const indexMap = indexMaps.get(view.indexMapId);
    if (allocation === undefined || indexMap === undefined) throw new Error("internal: verified references disappeared");
    const dtype = getBuiltinDType(view.dtype, `$.payload.views[${index}].dtype`);
    if (view.requiredAlignmentBytes < dtype.alignmentBytes || view.requiredAlignmentBytes % dtype.alignmentBytes !== 0) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, `$.payload.views[${index}].requiredAlignmentBytes`, "view alignment must satisfy dtype alignment");
    }
    if (allocation.alignmentBytes < view.requiredAlignmentBytes || allocation.alignmentBytes % view.requiredAlignmentBytes !== 0) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, `$.payload.views[${index}].requiredAlignmentBytes`, "allocation alignment does not satisfy view alignment");
    }
    validateDimDivisors(view.byteOffset, symbolMinimum, `$.payload.views[${index}].byteOffset`);
    const offset = evaluateDimExpr(view.byteOffset, environment, { limits });
    if (offset.kind === "resolved" && (offset.value < 0n || offset.value > ((1n << 64n) - 1n))) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, `$.payload.views[${index}].byteOffset`, "view byte offset must resolve to u64");
    }
    if (offset.kind === "resolved" && offset.value % BigInt(view.requiredAlignmentBytes) !== 0n) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, `$.payload.views[${index}].byteOffset`, "resolved view offset violates required alignment");
    }
    const allocationLength = evaluateDimExpr(allocation.byteLength, environment, { limits });
    if (offset.kind === "resolved" && allocationLength.kind === "resolved" && offset.value > allocationLength.value) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, `$.payload.views[${index}].byteOffset`, "view offset exceeds allocation byte length");
    }
    for (const [axis, dimension] of view.shape.entries()) {
      validateDimDivisors(dimension, symbolMinimum, `$.payload.views[${index}].shape[${axis}]`);
      const result = evaluateDimExpr(dimension, environment, { limits });
      if (result.kind === "resolved" && result.value < 0n) {
        invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, `$.payload.views[${index}].shape[${axis}]`, "view extent must be non-negative");
      }
    }
  }
}

function validateDimDivisorsInConstraint(constraint: ShapeConstraint, minima: ReadonlyMap<string, bigint>, path: string): void {
  switch (constraint.kind) {
    case "equal":
    case "lessEqual":
      validateDimDivisors(constraint.lhs, minima, `${path}.lhs`);
      validateDimDivisors(constraint.rhs, minima, `${path}.rhs`);
      break;
    case "nonNegative":
    case "positive": validateDimDivisors(constraint.value, minima, `${path}.value`); break;
    case "divisible":
      validateDimDivisors(constraint.value, minima, `${path}.value`);
      validateDimDivisors(constraint.divisor, minima, `${path}.divisor`);
      break;
  }
}

function validateDimDivisors(expression: DimExpr, minima: ReadonlyMap<string, bigint>, path: string): void {
  switch (expression.kind) {
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      requireProvablyPositive(expression.divisor, minima, `${path}.divisor`);
      validateDimDivisors(expression.value, minima, `${path}.value`);
      validateDimDivisors(expression.divisor, minima, `${path}.divisor`);
      break;
    case "add": expression.terms.forEach((term, index) => validateDimDivisors(term, minima, `${path}.terms[${index}]`)); break;
    case "mul":
      validateDimDivisors(expression.lhs, minima, `${path}.lhs`);
      validateDimDivisors(expression.rhs, minima, `${path}.rhs`);
      break;
    case "min":
    case "max": expression.values.forEach((value, index) => validateDimDivisors(value, minima, `${path}.values[${index}]`)); break;
    case "const":
    case "symbol": break;
  }
}

function validateIndexDivisors(expression: IndexExpr, minima: ReadonlyMap<string, bigint>, path: string): void {
  switch (expression.kind) {
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      requireProvablyPositiveIndex(expression.divisor, minima, `${path}.divisor`);
      validateIndexDivisors(expression.value, minima, `${path}.value`);
      validateIndexDivisors(expression.divisor, minima, `${path}.divisor`);
      break;
    case "add": expression.terms.forEach((term, index) => validateIndexDivisors(term, minima, `${path}.terms[${index}]`)); break;
    case "mul":
      validateIndexDivisors(expression.lhs, minima, `${path}.lhs`);
      validateIndexDivisors(expression.rhs, minima, `${path}.rhs`);
      break;
    case "min":
    case "max": expression.values.forEach((value, index) => validateIndexDivisors(value, minima, `${path}.values[${index}]`)); break;
    case "const":
    case "coordinate":
    case "dimension": break;
  }
}

function validatePredicateDivisors(expression: PredicateExpr, minima: ReadonlyMap<string, bigint>, path: string): void {
  switch (expression.kind) {
    case "equal":
    case "lessEqual":
      validateIndexDivisors(expression.lhs, minima, `${path}.lhs`);
      validateIndexDivisors(expression.rhs, minima, `${path}.rhs`);
      break;
    case "and":
    case "or": expression.values.forEach((value, index) => validatePredicateDivisors(value, minima, `${path}.values[${index}]`)); break;
    case "not": validatePredicateDivisors(expression.value, minima, `${path}.value`); break;
    case "bool": break;
  }
}

function requireProvablyPositive(expression: DimExpr, minima: ReadonlyMap<string, bigint>, path: string): void {
  const minimum = dimLowerBound(expression, minima);
  if (minimum === undefined || minimum <= 0n) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.nonpositiveDivisor, path, "divisor positivity must be statically proved from constants and symbol domains");
  }
}

function requireProvablyPositiveIndex(expression: IndexExpr, minima: ReadonlyMap<string, bigint>, path: string): void {
  const minimum = indexLowerBound(expression, minima);
  if (minimum === undefined || minimum <= 0n) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.nonpositiveDivisor, path, "index divisor positivity must be statically proved without coordinate dependence");
  }
}

function dimLowerBound(expression: DimExpr, minima: ReadonlyMap<string, bigint>): bigint | undefined {
  switch (expression.kind) {
    case "const": return BigInt(expression.value);
    case "symbol": return minima.get(expression.id);
    case "add": {
      const values = expression.terms.map((term) => dimLowerBound(term, minima));
      return values.every((value): value is bigint => value !== undefined)
        ? values.reduce((total, value) => total + value, 0n)
        : undefined;
    }
    case "mul": {
      const lhs = dimLowerBound(expression.lhs, minima);
      const rhs = dimLowerBound(expression.rhs, minima);
      return lhs !== undefined && rhs !== undefined && lhs >= 0n && rhs >= 0n ? lhs * rhs : undefined;
    }
    case "min":
    case "max": {
      const values = expression.values.map((value) => dimLowerBound(value, minima));
      if (!values.every((value): value is bigint => value !== undefined)) return undefined;
      return expression.kind === "min"
        ? values.reduce((result, value) => value < result ? value : result)
        : values.reduce((result, value) => value > result ? value : result);
    }
    case "mod": return dimLowerBound(expression.divisor, minima) !== undefined ? 0n : undefined;
    case "floorDiv":
    case "ceilDiv": return undefined;
  }
}

function indexLowerBound(expression: IndexExpr, minima: ReadonlyMap<string, bigint>): bigint | undefined {
  switch (expression.kind) {
    case "const": return BigInt(expression.value);
    case "dimension": return minima.get(expression.symbolId);
    case "coordinate": return undefined;
    case "add": {
      const values = expression.terms.map((term) => indexLowerBound(term, minima));
      return values.every((value): value is bigint => value !== undefined)
        ? values.reduce((total, value) => total + value, 0n)
        : undefined;
    }
    case "mul": {
      const lhs = indexLowerBound(expression.lhs, minima);
      const rhs = indexLowerBound(expression.rhs, minima);
      return lhs !== undefined && rhs !== undefined && lhs >= 0n && rhs >= 0n ? lhs * rhs : undefined;
    }
    case "min":
    case "max": {
      const values = expression.values.map((value) => indexLowerBound(value, minima));
      if (!values.every((value): value is bigint => value !== undefined)) return undefined;
      return expression.kind === "min"
        ? values.reduce((result, value) => value < result ? value : result)
        : values.reduce((result, value) => value > result ? value : result);
    }
    case "mod": return indexLowerBound(expression.divisor, minima) !== undefined ? 0n : undefined;
    case "floorDiv":
    case "ceilDiv": return undefined;
  }
}

function closedObject(
  value: JsonValue,
  fields: readonly string[],
  path: string,
  optional: readonly string[] = [],
): JsonObject {
  if (!isJsonObject(value)) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  exactFields(value, fields, path, optional);
  return value;
}

function objectWithKind(value: JsonValue, path: string): JsonObject & { readonly kind: string } {
  if (!isJsonObject(value) || typeof value.kind !== "string") invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, path, "expected tagged object with string kind");
  return value as JsonObject & { readonly kind: string };
}

function exactFields(object: JsonObject, fields: readonly string[], path: string, optional: readonly string[] = []): void {
  const allowed = new Set(fields);
  const unknown = Object.keys(object).filter((key) => !allowed.has(key));
  if (unknown.length > 0) invalid(LAYOUT_DIAGNOSTIC_CODES.unknownField, path, `unknown fields: ${unknown.sort().join(", ")}`);
  const optionalSet = new Set(optional);
  for (const fieldName of fields) {
    if (!optionalSet.has(fieldName) && object[fieldName] === undefined) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${fieldName}`, "required field is missing");
    }
  }
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function arrayField(object: JsonObject, name: string, path: string): readonly JsonValue[] {
  return arrayValue(field(object, name, path), `${path}.${name}`);
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  return value;
}

function localId(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!/^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$/u.test(result)) invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, path, "invalid local identifier");
  return result;
}

function nonnegativeSafeInteger(value: JsonValue, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, path, "expected non-negative safe integer");
  return value;
}

function alignment(value: JsonValue, path: string): number {
  const result = nonnegativeSafeInteger(value, path);
  if (result === 0 || result > 2 ** 31 || (BigInt(result) & BigInt(result - 1)) !== 0n) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, path, "alignment must be a positive power of two no greater than 2^31");
  }
  return result;
}

function uniqueIds(ids: readonly string[], path: string): void {
  const seen = new Set<string>();
  for (const [index, id] of ids.entries()) {
    if (seen.has(id)) invalid(LAYOUT_DIAGNOSTIC_CODES.duplicateId, `${path}[${index}]`, `duplicate ID ${id}`);
    seen.add(id);
  }
}

function requiredMap(map: ReadonlyMap<string, string>, key: string): string {
  const value = map.get(key);
  if (value === undefined) throw new Error(`internal: missing remap for ${key}`);
  return value;
}

function consume(state: ParseState, path: string, depth: number): void {
  state.nodes += 1;
  if (state.nodes > state.limits.maxNodes) resource(path, `semantic nodes exceed ${state.limits.maxNodes}`);
  if (depth > state.limits.maxDepth) resource(path, `semantic depth exceeds ${state.limits.maxDepth}`);
}

function resource(path: string, message: string): never {
  invalid(LAYOUT_DIAGNOSTIC_CODES.resourceLimit, path, message);
}

function invalid(code: `BG-LAYOUT-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
