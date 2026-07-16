import {
  prepareLayoutExpression,
  traceLayoutExpressionCoordinate,
  type IndexExpr,
  type LayoutExpressionCoordinateRequest,
  type LayoutExpressionCoordinateTrace,
  type LayoutExpr,
  type PreparedLayoutExpression,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  SemanticSchemaError,
  encodeWireI64,
  hashCanonicalJson,
  resolveDecodeLimits,
  type DecodeLimits,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { unwrapVerifiedCppCuteFrontendArtifact } from "./cpp_cute_frontend_artifact.js";
import {
  unwrapAuthorizedCppCuteFrontendArtifact,
  type AuthorizedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_authorization.js";
import type {
  CppCuteAffineLayoutFactV1,
  CppCuteFrontendEntryV1,
  CppCuteFrontendPayloadV3,
  CppCuteHierarchyV1,
  CppCuteMacroExpansionV1,
  CppCuteSourceOriginV1,
  CppCuteSourceSpanV1,
} from "./cpp_cute_frontend_types.js";
import {
  CppCuteIntegerSemanticsError,
  evaluateStaticCppCuteIntegerExpr,
  evaluateStaticCppCuteLayoutSummary,
} from "./cpp_cute_integer_semantics.js";

declare const loweredCppCuteLayoutBrand: unique symbol;

export interface LowerAuthorizedCppCuteLayoutEntryRequest {
  readonly entryId: string;
}

export interface LowerAuthorizedCppCuteLayoutEntryOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

/**
 * Opaque origin-bound compiler result. Public fields describe pure layout
 * meaning only; source/provenance routing remains in the compiler side table.
 */
export interface LoweredCppCuteLayoutEntry {
  readonly [loweredCppCuteLayoutBrand]: never;
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly originHash: string;
}

export interface LoweredCppCuteLayoutEntryRecord {
  readonly authorization: AuthorizedCppCuteFrontendArtifact;
  readonly preparedLayout: PreparedLayoutExpression;
  readonly entry: Extract<CppCuteFrontendEntryV1, { readonly kind: "layout" }>;
  readonly fact: CppCuteAffineLayoutFactV1;
  readonly originSpanRecords: readonly CppCuteSourceSpanV1[];
  readonly macroExpansionRecords: readonly CppCuteMacroExpansionV1[];
}

export type CppCuteLayoutLoweringErrorCode =
  | "BG-COMPILER-CPP-CUTE-LAYOUT-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-LAYOUT-INVALID-REQUEST"
  | "BG-COMPILER-CPP-CUTE-LAYOUT-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-LAYOUT-INCONSISTENT-ARTIFACT"
  | "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-ENTRY"
  | "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-LAYOUT"
  | "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT";

export class CppCuteLayoutLoweringError extends Error {
  constructor(
    readonly code: CppCuteLayoutLoweringErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteLayoutLoweringError";
  }
}

interface LoweredRecord extends LoweredCppCuteLayoutEntryRecord {
  readonly originHash: string;
}

const LOWERED_LAYOUTS = new WeakMap<object, LoweredRecord>();

class LoweredCppCuteLayoutEntryValue {
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly originHash: string;

  constructor(record: LoweredRecord) {
    this.layoutSemanticHash = record.preparedLayout.layoutSemanticHash;
    this.indexMapId = record.preparedLayout.indexMapId;
    this.coordinateRank = record.preparedLayout.coordinateRank;
    this.originHash = record.originHash;
    LOWERED_LAYOUTS.set(this, record);
    Object.freeze(this);
  }
}

/**
 * Lowers one explicitly selected, producer-authorized CuTe layout entry
 * through semantic-core. No merely verified/raw artifact overload exists.
 */
export async function lowerAuthorizedCppCuteLayoutEntry(
  authorization: AuthorizedCppCuteFrontendArtifact,
  request: LowerAuthorizedCppCuteLayoutEntryRequest,
  options: LowerAuthorizedCppCuteLayoutEntryOptions = {},
): Promise<LoweredCppCuteLayoutEntry> {
  const authorized = unwrapAuthorizedCppCuteFrontendArtifact(authorization);
  const normalizedOptions = normalizeOptions(options);
  throwIfAborted(normalizedOptions.signal);
  const entryId = validateRequest(request);
  const artifact = unwrapVerifiedCppCuteFrontendArtifact(authorized.artifact);
  const payload = artifact.envelope.payload;
  const selectedIds = payload.outcome.kind === "accepted" ? payload.outcome.selectedEntryIds : [];
  if (payload.outcome.kind !== "accepted") {
    failure("BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-ENTRY", "$.artifact.outcome", "rejected frontend artifacts cannot enter layout lowering");
  }
  if (selectedIds.length !== 1 || selectedIds[0] !== entryId) {
    failure(
      "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-ENTRY",
      "$.artifact.outcome.selectedEntryIds",
      "initial layout lowering requires exactly the explicitly requested selected entry",
    );
  }
  const entry = payload.entries.find((candidate) => candidate.entryId === entryId);
  if (entry === undefined) inconsistent("$.request.entryId", `selected entry ${entryId} disappeared from the verified artifact`);
  if (entry.kind !== "layout") {
    failure(
      "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-ENTRY",
      "$.request.entryId",
      "layout lowering does not accept tensor/view-copy entries",
    );
  }
  const fact = payload.facts.find((candidate) => candidate.factId === entry.layoutFactId);
  if (fact?.kind !== "affine-layout") inconsistent("$.artifact.entries.layoutFactId", "selected layout fact disappeared or changed kind");
  if (entry.selectedRootDeclarationIds.length !== 1 || entry.selectedRootDeclarationIds[0] !== fact.resultDeclarationId) {
    inconsistent(
      "$.artifact.entries.selectedRootDeclarationIds",
      "selected entry does not own exactly its affine-layout result declaration",
    );
  }
  const selectedRootFacts = payload.facts.filter((candidate) => (
    "resultDeclarationId" in candidate && candidate.resultDeclarationId === fact.resultDeclarationId
  ));
  if (selectedRootFacts.length !== 1 || selectedRootFacts[0] !== fact) {
    inconsistent(
      "$.artifact.facts",
      "selected layout result declaration has ambiguous layout/tensor ownership",
    );
  }

  const layout = lowerLayoutFact(fact, normalizedOptions.limits);
  throwIfAborted(normalizedOptions.signal);
  let preparedLayout: PreparedLayoutExpression;
  try {
    preparedLayout = await prepareLayoutExpression({
      symbols: [],
      constraints: [],
      layout,
    }, {
      producer: { id: "browsergrad.compiler.cpp-cute-layout-lowering", version: "1" },
      artifactId: "authorized-cpp-cute-layout",
      limits: normalizedOptions.limits,
    });
  } catch (error) {
    if (!(error instanceof SemanticSchemaError)) throw error;
    failure(
      error.diagnostic.code.endsWith("RESOURCE-LIMIT")
        ? "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-LAYOUT",
      "$.artifact.layoutFact",
      `shared layout verification rejected the selected CuTe layout: ${error.message}`,
      { cause: error },
    );
  }
  throwIfAborted(normalizedOptions.signal);
  const origin = collectOriginClosure(payload, fact.origin);
  let originHash: string;
  try {
    originHash = await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.layout-origin.v2",
      layoutSemanticHash: preparedLayout.layoutSemanticHash,
      frontendArtifactId: authorization.artifactId,
      frontendArtifactHash: authorization.artifactHash,
      frontendArtifactBytesSha256: authorization.artifactBytesSha256,
      frontendArtifactByteLength: authorization.artifactByteLength,
      profileHash: authorization.profileHash,
      compilationContractHash: authorization.compilationContractHash,
      sourceSetSha256: authorization.sourceSetSha256,
      headerSetSha256: authorization.headerSetSha256,
      inputClosureSha256: authorization.inputClosureSha256,
      evidenceKind: authorization.evidenceKind,
      evidenceHash: authorization.evidenceHash,
      entryId: entry.entryId,
      factId: fact.factId,
      resultDeclarationId: fact.resultDeclarationId,
      origin: fact.origin,
      spans: origin.spans,
      macroExpansions: origin.macroExpansions,
    }, { limits: normalizedOptions.limits });
  } catch (error) {
    failure(
      "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT",
      "$.artifact.layoutFact.origin",
      "layout origin projection exceeded canonical hashing limits",
      { cause: error },
    );
  }
  throwIfAborted(normalizedOptions.signal);
  const record: LoweredRecord = Object.freeze({
    authorization,
    preparedLayout,
    entry,
    fact,
    originSpanRecords: origin.spans,
    macroExpansionRecords: origin.macroExpansions,
    originHash,
  });
  return new LoweredCppCuteLayoutEntryValue(record) as unknown as LoweredCppCuteLayoutEntry;
}

export function traceLoweredCppCuteLayoutCoordinate(
  lowered: LoweredCppCuteLayoutEntry,
  request: LayoutExpressionCoordinateRequest,
): LayoutExpressionCoordinateTrace {
  return traceLayoutExpressionCoordinate(unwrapLoweredCppCuteLayoutEntry(lowered).preparedLayout, request);
}

export function unwrapLoweredCppCuteLayoutEntry(
  lowered: LoweredCppCuteLayoutEntry,
): LoweredCppCuteLayoutEntryRecord {
  if ((typeof lowered !== "object" && typeof lowered !== "function") || lowered === null) {
    failure("BG-COMPILER-CPP-CUTE-LAYOUT-UNVERIFIED", "$", "expected a compiler-authorized lowered CuTe layout");
  }
  const record = LOWERED_LAYOUTS.get(lowered as object);
  if (record === undefined) {
    failure("BG-COMPILER-CPP-CUTE-LAYOUT-UNVERIFIED", "$", "lowered CuTe layout was not created by this module instance");
  }
  return record;
}

function lowerLayoutFact(fact: CppCuteAffineLayoutFactV1, limits: DecodeLimits): LayoutExpr {
  if (!sameTopology(fact.shape, fact.stride)) inconsistent("$.artifact.layoutFact.stride", "shape/stride hierarchy topology drifted after verification");
  const shapeModes = topModes(fact.shape);
  const strideModes = topModes(fact.stride);
  if (fact.rank !== shapeModes.length || strideModes.length !== shapeModes.length) {
    inconsistent("$.artifact.layoutFact.rank", "layout rank drifted after verification");
  }
  const shapeByMode = shapeModes.map((mode, index) => staticLeaves(mode, `$.artifact.layoutFact.shape.mode[${index}]`, limits));
  const strideByMode = strideModes.map((mode, index) => staticLeaves(mode, `$.artifact.layoutFact.stride.mode[${index}]`, limits));
  const shapeLeaves = shapeByMode.flat();
  const strideLeaves = strideByMode.flat();
  if (shapeLeaves.length !== fact.leafRank || strideLeaves.length !== fact.leafRank) {
    inconsistent("$.artifact.layoutFact.leafRank", "layout leaf rank drifted after verification");
  }
  if (shapeLeaves.some((extent) => extent <= 0n)) {
    unsupported("$.artifact.layoutFact.shape", "layout lowering requires positive static shape extents");
  }
  const size = staticValue(fact.size, "$.artifact.layoutFact.size", limits);
  const cosize = staticValue(fact.cosize, "$.artifact.layoutFact.cosize", limits);
  let summary;
  try {
    summary = evaluateStaticCppCuteLayoutSummary(shapeLeaves, strideLeaves, {
      path: "$.artifact.layoutFact",
      limits,
    });
  } catch (error) {
    if (!(error instanceof CppCuteIntegerSemanticsError)) throw error;
    failure(
      error.kind === "resource-limit"
        ? "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-LAYOUT-INCONSISTENT-ARTIFACT",
      error.path,
      error.message,
      { cause: error },
    );
  }
  if (size !== summary.size) inconsistent("$.artifact.layoutFact.size", "layout size drifted after verification");
  if (cosize !== summary.cosize) inconsistent("$.artifact.layoutFact.cosize", "layout cosize drifted after verification");

  const leafLayout: LayoutExpr = {
    kind: "strided",
    shape: shapeLeaves.map(constant),
    strides: strideLeaves.map(constant),
  };
  if (shapeModes.every((mode) => mode.kind === "scalar")) return leafLayout;

  const topShape = shapeByMode.map((mode) => mode.reduce((product, extent) => product * extent, 1n));
  const sourceCoordinates: IndexExpr[] = [];
  shapeByMode.forEach((mode, axis) => {
    let divisor = 1n;
    for (const extent of mode) {
      const coordinate: IndexExpr = divisor === 1n
        ? { kind: "coordinate", axis }
        : {
            kind: "floorDiv",
            value: { kind: "coordinate", axis },
            divisor: indexConstant(divisor),
          };
      sourceCoordinates.push({
        kind: "mod",
        value: coordinate,
        divisor: indexConstant(extent),
      });
      divisor *= extent;
    }
  });
  return {
    kind: "compose",
    source: leafLayout,
    shape: topShape.map(constant),
    sourceCoordinates,
  };
}

function staticLeaves(hierarchy: CppCuteHierarchyV1, path: string, limits: DecodeLimits): bigint[] {
  if (hierarchy.kind === "scalar") return [staticValue(hierarchy.value, `${path}.value`, limits)];
  return hierarchy.elements.flatMap((element, index) => staticLeaves(element, `${path}.elements[${index}]`, limits));
}

function staticValue(
  expression: Parameters<typeof evaluateStaticCppCuteIntegerExpr>[0],
  path: string,
  limits: DecodeLimits,
): bigint {
  try {
    const value = evaluateStaticCppCuteIntegerExpr(expression, { path, limits });
    if (value === undefined) unsupported(path, "initial authorized layout lowering requires static CuTe integer expressions");
    return value;
  } catch (error) {
    if (!(error instanceof CppCuteIntegerSemanticsError)) throw error;
    failure(
      error.kind === "resource-limit"
        ? "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-LAYOUT",
      error.path,
      error.message,
      { cause: error },
    );
  }
}

function topModes(hierarchy: CppCuteHierarchyV1): readonly CppCuteHierarchyV1[] {
  return hierarchy.kind === "scalar" ? [hierarchy] : hierarchy.elements;
}

function sameTopology(left: CppCuteHierarchyV1, right: CppCuteHierarchyV1): boolean {
  if (left.kind !== right.kind) return false;
  if (left.kind === "scalar" || right.kind === "scalar") return true;
  return left.elements.length === right.elements.length && left.elements.every((entry, index) => {
    const other = right.elements[index];
    return other !== undefined && sameTopology(entry, other);
  });
}

function constant(value: bigint) {
  return { kind: "const" as const, value: encodeWireI64(value) };
}

function indexConstant(value: bigint): IndexExpr {
  return { kind: "const", value: encodeWireI64(value) };
}

function collectOriginClosure(
  payload: CppCuteFrontendPayloadV3,
  origin: CppCuteSourceOriginV1,
): { readonly spans: readonly CppCuteSourceSpanV1[]; readonly macroExpansions: readonly CppCuteMacroExpansionV1[] } {
  const spansById = new Map(payload.spans.map((span) => [span.spanId, span]));
  const macrosById = new Map(payload.macroExpansions.map((macro) => [macro.macroExpansionId, macro]));
  const spanIds = new Set<string>();
  const macros: CppCuteMacroExpansionV1[] = [];
  const anchorId = origin.kind === "source" ? origin.spanId : origin.anchorSpanId;
  const anchor = spansById.get(anchorId);
  if (anchor === undefined) inconsistent("$.artifact.layoutFact.origin", "layout origin span disappeared after verification");
  spanIds.add(anchor.spanId);
  let macroId = anchor.macroExpansionId;
  while (macroId !== null) {
    const macro = macrosById.get(macroId);
    if (macro === undefined) inconsistent("$.artifact.layoutFact.origin", "layout macro provenance disappeared after verification");
    macros.push(macro);
    spanIds.add(macro.definitionSpanId);
    spanIds.add(macro.invocationSpanId);
    macroId = macro.parentMacroExpansionId;
  }
  const spans = [...spanIds].sort().map((spanId) => {
    const span = spansById.get(spanId);
    if (span === undefined) inconsistent("$.artifact.layoutFact.origin", "referenced origin span disappeared after verification");
    return span;
  });
  return Object.freeze({
    spans: Object.freeze(spans),
    macroExpansions: Object.freeze(macros),
  });
}

interface NormalizedOptions {
  readonly limits: DecodeLimits;
  readonly signal?: AbortSignal;
}

function normalizeOptions(options: LowerAuthorizedCppCuteLayoutEntryOptions): NormalizedOptions {
  if (typeof options !== "object" || options === null || Array.isArray(options)) {
    invalidRequest("$options", "options must be a plain object");
  }
  const prototype = Object.getPrototypeOf(options);
  if (prototype !== Object.prototype && prototype !== null) invalidRequest("$options", "options must be a plain object");
  const descriptors = Object.getOwnPropertyDescriptors(options);
  for (const key of Reflect.ownKeys(options)) {
    if (typeof key !== "string" || (key !== "limits" && key !== "signal")) invalidRequest("$options", "options contain unknown fields");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalidRequest(`$options.${key}`, "options require enumerable data properties without accessors");
    }
  }
  let limits: DecodeLimits;
  try {
    limits = resolveDecodeLimits(options.limits);
  } catch (error) {
    invalidRequest("$options.limits", "invalid semantic decode limits", { cause: error });
  }
  return Object.freeze({
    limits,
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
}

function validateRequest(request: LowerAuthorizedCppCuteLayoutEntryRequest): string {
  if (typeof request !== "object" || request === null || Array.isArray(request)) invalidRequest("$.request", "request must be a plain object");
  const prototype = Object.getPrototypeOf(request);
  if (prototype !== Object.prototype && prototype !== null) invalidRequest("$.request", "request must be a plain object");
  const descriptors = Object.getOwnPropertyDescriptors(request);
  const keys = Reflect.ownKeys(request);
  if (keys.length !== 1 || keys[0] !== "entryId") invalidRequest("$.request", "request must contain only entryId");
  const descriptor = descriptors.entryId;
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    invalidRequest("$.request.entryId", "entryId must be an enumerable data property");
  }
  if (typeof descriptor.value !== "string" || descriptor.value.length === 0) {
    invalidRequest("$.request.entryId", "entryId must be a non-empty string");
  }
  return descriptor.value;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    failure("BG-COMPILER-CPP-CUTE-LAYOUT-CANCELLED", "$.signal", "authorized layout lowering was aborted");
  }
}

function invalidRequest(path: string, message: string, options?: ErrorOptions): never {
  failure("BG-COMPILER-CPP-CUTE-LAYOUT-INVALID-REQUEST", path, message, options);
}

function inconsistent(path: string, message: string): never {
  failure("BG-COMPILER-CPP-CUTE-LAYOUT-INCONSISTENT-ARTIFACT", path, message);
}

function unsupported(path: string, message: string): never {
  failure("BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-LAYOUT", path, message);
}

function failure(
  code: CppCuteLayoutLoweringErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteLayoutLoweringError(code, path, message, options);
}
