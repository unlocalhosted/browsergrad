import {
  compareCanonicalStrings,
  deepFreezeJson,
  isJsonObject,
  parseWireI64,
  parseWireU64,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type {
  CppCuteAbiBaseV1,
  CppCuteAbiFieldV1,
  CppCuteAddressSpaceV1,
  CppCuteAffineLayoutFactV1,
  CppCuteCallingConventionV1,
  CppCuteConstantV1,
  CppCuteCudaAttributesV1,
  CppCuteDeclarationKindV1,
  CppCuteDeclarationV3,
  CppCuteDiagnosticLocationV3,
  CppCuteDiagnosticPhaseV3,
  CppCuteDiagnosticRelatedLocationV1,
  CppCuteDiagnosticSubjectV3,
  CppCuteExpressionV1,
  CppCuteExtractionRecordV1,
  CppCuteFileRangeV1,
  CppCuteFrontendDiagnosticV3,
  CppCuteFrontendEntryV1,
  CppCuteFrontendOutcomeV1,
  CppCuteFrontendPayloadV3,
  CppCuteFunctionAbiV1,
  CppCuteFunctionBodyV1,
  CppCuteHierarchyV1,
  CppCuteIncludeEdgeV3,
  CppCuteIncludeResolutionV3,
  CppCuteIncludeRootV3,
  CppCuteInputClosureV3,
  CppCuteInputOwnerV3,
  CppCuteIntegerExprV1,
  CppCuteIntrinsicEffectsV1,
  CppCuteMacroExpansionV1,
  CppCuteOverloadResolutionV1,
  CppCuteParameterAbiV1,
  CppCuteResolvedFactV1,
  CppCuteResolvedTypeV1,
  CppCuteSemanticDomainV1,
  CppCuteSemanticPassRecordV1,
  CppCuteSourceAbiV1,
  CppCuteSourceEntityV1,
  CppCuteSourceFileV3,
  CppCuteSourceOriginV1,
  CppCuteSourceSpanV1,
  CppCuteStatementV1,
  CppCuteTargetIntrinsicAvailabilityV1,
  CppCuteTargetIntrinsicFactV1,
  CppCuteTargetIntrinsicOperationV1,
  CppCuteTemplateArgumentV1,
  CppCuteTemplateInstantiationV1,
  CppCuteTensorEngineV1,
  CppCuteTensorFactV1,
  CppCuteTypeAbiV1,
  CppCuteTypeQualifiersV1,
} from "./cpp_cute_frontend_types.js";

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const STABLE_ID = /^bg\.cpp\.([a-z][a-z0-9-]*)\.sha256\.[0-9a-f]{64}$/u;
const CAPABILITY_ID = /^[a-z][a-z0-9.-]*:[a-z][a-z0-9._-]*(?:@[1-9][0-9]*)?$/u;
const DIAGNOSTIC_CODE = /^[a-z][a-z0-9.-]*:[A-Za-z0-9][A-Za-z0-9._-]*$/u;
const CANONICAL_USR = /^c:@/u;
const MACRO_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;
const DEPENDENCY_ID = /^[a-z][a-z0-9._-]*$/u;

export interface CppCuteFrontendArtifactLimits {
  readonly maxIncludeRoots: number;
  readonly maxFiles: number;
  readonly maxIncludeEdges: number;
  readonly maxSpans: number;
  readonly maxMacroExpansions: number;
  readonly maxTypes: number;
  readonly maxConstants: number;
  readonly maxDeclarations: number;
  readonly maxInitializerExpressions: number;
  readonly maxTemplateInstantiations: number;
  readonly maxOverloadResolutions: number;
  readonly maxAbiEntries: number;
  readonly maxFunctionBodies: number;
  readonly maxBodyNodes: number;
  readonly maxFacts: number;
  readonly maxEntries: number;
  readonly maxDiagnostics: number;
  readonly maxRelatedDiagnosticLocations: number;
  readonly maxStringBytes: number;
}

export const DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS: CppCuteFrontendArtifactLimits = Object.freeze({
  maxIncludeRoots: 64,
  maxFiles: 4_096,
  maxIncludeEdges: 16_384,
  maxSpans: 32_768,
  maxMacroExpansions: 8_192,
  maxTypes: 16_384,
  maxConstants: 16_384,
  maxDeclarations: 16_384,
  maxInitializerExpressions: 50_000,
  maxTemplateInstantiations: 8_192,
  maxOverloadResolutions: 8_192,
  maxAbiEntries: 16_384,
  maxFunctionBodies: 4_096,
  maxBodyNodes: 50_000,
  maxFacts: 8_192,
  maxEntries: 256,
  maxDiagnostics: 4_096,
  maxRelatedDiagnosticLocations: 32,
  maxStringBytes: 16 * 1_024,
});

export type CppCuteFrontendArtifactErrorCode =
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-DUPLICATE-ID"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED";

export class CppCuteFrontendArtifactError extends Error {
  constructor(
    readonly code: CppCuteFrontendArtifactErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteFrontendArtifactError";
  }
}

export function parseCppCuteFrontendPayload(
  value: JsonValue,
  limits: CppCuteFrontendArtifactLimits = DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS,
): CppCuteFrontendPayloadV3 {
  const object = closedObject(value, [
    "compilationContractHash",
    "inputs",
    "semanticPasses",
    "semanticGraphOwnerPassId",
    "spans",
    "macroExpansions",
    "types",
    "constants",
    "declarations",
    "initializerExpressions",
    "templateInstantiations",
    "overloadResolutions",
    "sourceEntities",
    "sourceAbi",
    "functionBodies",
    "facts",
    "entries",
    "diagnostics",
    "outcome",
    "extraction",
  ], "$.payload");
  const payload = {
    compilationContractHash: sha256(
      field(object, "compilationContractHash", "$.payload"),
      "$.payload.compilationContractHash",
    ),
    inputs: parseInputs(field(object, "inputs", "$.payload"), limits, "$.payload.inputs"),
    semanticPasses: parseSemanticPasses(
      field(object, "semanticPasses", "$.payload"),
      limits,
      "$.payload.semanticPasses",
    ),
    semanticGraphOwnerPassId: exactString(
      field(object, "semanticGraphOwnerPassId", "$.payload"),
      "cuda-device-sema",
      "$.payload.semanticGraphOwnerPassId",
    ),
    spans: parseSetArray(object, "spans", limits.maxSpans, parseSpan, (entry) => entry.spanId),
    macroExpansions: parseSetArray(
      object,
      "macroExpansions",
      limits.maxMacroExpansions,
      parseMacroExpansion,
      (entry) => entry.macroExpansionId,
    ),
    types: parseSetArray(object, "types", limits.maxTypes, parseType, (entry) => entry.typeId),
    constants: parseSetArray(object, "constants", limits.maxConstants, parseConstant, (entry) => entry.constantId),
    declarations: parseSetArray(
      object,
      "declarations",
      limits.maxDeclarations,
      parseDeclaration,
      (entry) => entry.declarationId,
    ),
    initializerExpressions: parseSetArray(
      object,
      "initializerExpressions",
      limits.maxInitializerExpressions,
      parseExpression,
      (entry) => entry.expressionId,
    ),
    templateInstantiations: parseSetArray(
      object,
      "templateInstantiations",
      limits.maxTemplateInstantiations,
      parseTemplateInstantiation,
      (entry) => entry.instantiationId,
    ),
    overloadResolutions: parseSetArray(
      object,
      "overloadResolutions",
      limits.maxOverloadResolutions,
      parseOverloadResolution,
      (entry) => entry.resolutionId,
    ),
    sourceEntities: parseSetArray(
      object,
      "sourceEntities",
      limits.maxAbiEntries,
      parseSourceEntity,
      (entry) => entry.sourceEntityId,
    ),
    sourceAbi: parseSourceAbi(field(object, "sourceAbi", "$.payload"), limits, "$.payload.sourceAbi"),
    functionBodies: parseSetArray(
      object,
      "functionBodies",
      limits.maxFunctionBodies,
      (entry, path) => parseFunctionBody(entry, limits, path),
      (entry) => entry.bodyId,
    ),
    facts: parseSetArray(object, "facts", limits.maxFacts, parseFact, (entry) => entry.factId),
    entries: parseSetArray(object, "entries", limits.maxEntries, parseEntry, (entry) => entry.entryId),
    diagnostics: parseSetArray(
      object,
      "diagnostics",
      limits.maxDiagnostics,
      (entry, path) => parseDiagnostic(entry, limits, path),
      (entry) => entry.diagnosticId,
    ),
    outcome: parseOutcome(field(object, "outcome", "$.payload"), limits, "$.payload.outcome"),
    extraction: parseExtraction(field(object, "extraction", "$.payload"), "$.payload.extraction"),
  } as CppCuteFrontendPayloadV3;
  return deepFreezeJson(payload);
}

function parseSemanticPasses(
  value: JsonValue,
  limits: CppCuteFrontendArtifactLimits,
  path: string,
): readonly CppCuteSemanticPassRecordV1[] {
  const values = arrayValue(value, path);
  if (values.length !== 2) invalid(path, "artifact requires exactly device extraction then host validation records");
  const expected = [
    {
      ordinal: 0,
      passId: "cuda-device-sema",
      domain: "device",
      role: "semantic-extraction",
      invocationMode: "cuda-device-only",
    },
    {
      ordinal: 1,
      passId: "cuda-host-sema",
      domain: "host",
      role: "validation",
      invocationMode: "cuda-host-only",
    },
  ] as const;
  return values.map((entry, index) => {
    const passPath = `${path}[${index}]`;
    const object = closedObject(entry, [
      "ordinal", "passId", "domain", "role", "invocationMode", "targetTriple", "auxiliaryTargetTriple",
      "deviceArchitecture", "status", "openedFileIds", "includeEdgeIds", "observedInputClosureSha256",
      "sharedSurfaceSha256", "selectedSourceRootEntityIds", "factIds", "diagnosticIds",
    ], passPath);
    const wanted = expected[index];
    if (wanted === undefined) invalid(passPath, "unexpected semantic pass");
    if (object.ordinal !== wanted.ordinal) invalid(`${passPath}.ordinal`, `semantic pass ordinal must equal ${wanted.ordinal}`);
    if (object.passId !== wanted.passId) invalid(`${passPath}.passId`, `semantic pass must be ${wanted.passId}`);
    if (object.domain !== wanted.domain) invalid(`${passPath}.domain`, `semantic pass domain must be ${wanted.domain}`);
    if (object.role !== wanted.role) invalid(`${passPath}.role`, `semantic pass role must be ${wanted.role}`);
    if (object.invocationMode !== wanted.invocationMode) {
      invalid(`${passPath}.invocationMode`, `semantic pass invocationMode must be ${wanted.invocationMode}`);
    }
    const sharedSurface = field(object, "sharedSurfaceSha256", passPath);
    return {
      ordinal: wanted.ordinal,
      passId: wanted.passId,
      domain: wanted.domain,
      role: wanted.role,
      invocationMode: wanted.invocationMode,
      targetTriple: boundedString(field(object, "targetTriple", passPath), `${passPath}.targetTriple`, 256),
      auxiliaryTargetTriple: boundedString(
        field(object, "auxiliaryTargetTriple", passPath),
        `${passPath}.auxiliaryTargetTriple`,
        256,
      ),
      deviceArchitecture: boundedString(
        field(object, "deviceArchitecture", passPath),
        `${passPath}.deviceArchitecture`,
        64,
      ),
      status: enumValue(
        field(object, "status", passPath),
        ["succeeded", "failed", "not-run"] as const,
        `${passPath}.status`,
      ),
      openedFileIds: sortedStableIdSet(
        field(object, "openedFileIds", passPath),
        `${passPath}.openedFileIds`,
        "file",
        limits.maxFiles,
      ),
      includeEdgeIds: sortedStableIdSet(
        field(object, "includeEdgeIds", passPath),
        `${passPath}.includeEdgeIds`,
        "include-edge",
        limits.maxIncludeEdges,
      ),
      observedInputClosureSha256: nullableSha256(
        field(object, "observedInputClosureSha256", passPath),
        `${passPath}.observedInputClosureSha256`,
      ),
      sharedSurfaceSha256: sharedSurface === null
        ? null
        : sha256(sharedSurface, `${passPath}.sharedSurfaceSha256`),
      selectedSourceRootEntityIds: sortedStableIdSet(
        field(object, "selectedSourceRootEntityIds", passPath),
        `${passPath}.selectedSourceRootEntityIds`,
        "source-entity",
        limits.maxDeclarations,
      ),
      factIds: sortedStableIdSet(
        field(object, "factIds", passPath),
        `${passPath}.factIds`,
        "fact",
        limits.maxFacts,
      ),
      diagnosticIds: sortedStableIdSet(
        field(object, "diagnosticIds", passPath),
        `${passPath}.diagnosticIds`,
        "diagnostic",
        limits.maxDiagnostics,
      ),
    };
  });
}

function parseInputs(value: JsonValue, limits: CppCuteFrontendArtifactLimits, path: string): CppCuteInputClosureV3 {
  const object = closedObject(value, [
    "mainFileId",
    "includeRoots",
    "files",
    "includeEdges",
    "sourceSetSha256",
    "headerSetSha256",
    "closureSha256",
  ], path);
  const includeRoots = orderedArrayField(object, "includeRoots", path, limits.maxIncludeRoots).map((entry, index) =>
    parseIncludeRoot(entry, `${path}.includeRoots[${index}]`));
  const files = setArrayField(object, "files", path, limits.maxFiles, parseSourceFile, (entry) => entry.fileId);
  const includeEdges = setArrayField(
    object,
    "includeEdges",
    path,
    limits.maxIncludeEdges,
    parseIncludeEdge,
    (entry) => entry.includeEdgeId,
  );
  return {
    mainFileId: stableId(field(object, "mainFileId", path), `${path}.mainFileId`, "file"),
    includeRoots,
    files,
    includeEdges,
    sourceSetSha256: sha256(field(object, "sourceSetSha256", path), `${path}.sourceSetSha256`),
    headerSetSha256: sha256(field(object, "headerSetSha256", path), `${path}.headerSetSha256`),
    closureSha256: sha256(field(object, "closureSha256", path), `${path}.closureSha256`),
  };
}

function parseIncludeRoot(value: JsonValue, path: string): CppCuteIncludeRootV3 {
  const object = closedObject(value, [
    "includeRootId", "ordinal", "mode", "virtualPath", "manifestSha256", "owner",
  ], path);
  const mode = enumValue(field(object, "mode", path), ["quote", "system"] as const, `${path}.mode`);
  const virtualPath = boundedString(field(object, "virtualPath", path), `${path}.virtualPath`, 4_096);
  validateVirtualPath(virtualPath, `${path}.virtualPath`);
  return {
    includeRootId: dependencyId(field(object, "includeRootId", path), `${path}.includeRootId`),
    ordinal: nonnegativeInteger(field(object, "ordinal", path), `${path}.ordinal`),
    mode,
    virtualPath,
    manifestSha256: sha256(field(object, "manifestSha256", path), `${path}.manifestSha256`),
    owner: parseInputOwner(field(object, "owner", path), `${path}.owner`),
  };
}

function parseSourceFile(value: JsonValue, path: string): CppCuteSourceFileV3 {
  const object = closedObject(value, [
    "fileId", "role", "virtualPath", "contentSha256", "byteLength", "owner", "includeRootId",
  ], path);
  const virtualPath = boundedString(field(object, "virtualPath", path), `${path}.virtualPath`, 4_096);
  validateVirtualPath(virtualPath, `${path}.virtualPath`);
  return {
    fileId: stableId(field(object, "fileId", path), `${path}.fileId`, "file"),
    role: enumValue(field(object, "role", path), [
      "main-source",
      "project-header",
      "system-header",
      "dependency-header",
      "compiler-header",
      "generated-header",
    ] as const, `${path}.role`),
    virtualPath,
    contentSha256: sha256(field(object, "contentSha256", path), `${path}.contentSha256`),
    byteLength: parseWireU64(field(object, "byteLength", path), `${path}.byteLength`),
    owner: parseInputOwner(field(object, "owner", path), `${path}.owner`),
    includeRootId: nullableDependencyId(field(object, "includeRootId", path), `${path}.includeRootId`),
  };
}

function parseInputOwner(value: JsonValue, path: string): CppCuteInputOwnerV3 {
  if (!isJsonObject(value)) invalid(path, "input owner must be an object");
  if (value.kind === "source" || value.kind === "compiler-resource-directory") {
    closedObject(value, ["kind"], path);
    return { kind: value.kind };
  }
  if (value.kind === "dependency") {
    const object = closedObject(value, ["kind", "dependencyId"], path);
    return {
      kind: "dependency",
      dependencyId: dependencyId(field(object, "dependencyId", path), `${path}.dependencyId`),
    };
  }
  invalid(`${path}.kind`, "unknown input owner kind");
}

function parseIncludeEdge(value: JsonValue, path: string): CppCuteIncludeEdgeV3 {
  if (!isJsonObject(value)) invalid(path, "include edge must be an object");
  if (value.kind === "source-directive") {
    const object = closedObject(value, [
      "kind", "includeEdgeId", "includingFileId", "directiveSpanId", "spelling", "mode", "resolution",
    ], path);
    return {
      kind: "source-directive",
      includeEdgeId: stableId(field(object, "includeEdgeId", path), `${path}.includeEdgeId`, "include-edge"),
      includingFileId: stableId(field(object, "includingFileId", path), `${path}.includingFileId`, "file"),
      directiveSpanId: stableId(field(object, "directiveSpanId", path), `${path}.directiveSpanId`, "span"),
      spelling: boundedString(field(object, "spelling", path), `${path}.spelling`, 4_096),
      mode: enumValue(field(object, "mode", path), ["quote", "angle"] as const, `${path}.mode`),
      resolution: parseIncludeResolution(field(object, "resolution", path), `${path}.resolution`),
    };
  }
  if (value.kind === "compiler-forced") {
    const object = closedObject(value, [
      "kind", "includeEdgeId", "fileId", "includeRootId", "compilerOptionOrdinal",
    ], path);
    return {
      kind: "compiler-forced",
      includeEdgeId: stableId(field(object, "includeEdgeId", path), `${path}.includeEdgeId`, "include-edge"),
      fileId: stableId(field(object, "fileId", path), `${path}.fileId`, "file"),
      includeRootId: dependencyId(field(object, "includeRootId", path), `${path}.includeRootId`),
      compilerOptionOrdinal: nonnegativeInteger(
        field(object, "compilerOptionOrdinal", path),
        `${path}.compilerOptionOrdinal`,
      ),
    };
  }
  invalid(`${path}.kind`, "unknown include edge kind");
}

function parseIncludeResolution(value: JsonValue, path: string): CppCuteIncludeResolutionV3 {
  if (!isJsonObject(value)) invalid(path, "include resolution must be an object");
  if (value.kind === "resolved") {
    const object = closedObject(value, ["kind", "fileId", "includeRootId"], path);
    return {
      kind: "resolved",
      fileId: stableId(field(object, "fileId", path), `${path}.fileId`, "file"),
      includeRootId: dependencyId(field(object, "includeRootId", path), `${path}.includeRootId`),
    };
  }
  if (value.kind === "unresolved") {
    const object = closedObject(value, ["kind", "diagnosticId"], path);
    return {
      kind: "unresolved",
      diagnosticId: stableId(field(object, "diagnosticId", path), `${path}.diagnosticId`, "diagnostic"),
    };
  }
  invalid(`${path}.kind`, "unknown include resolution kind");
}

function parseSpan(value: JsonValue, path: string): CppCuteSourceSpanV1 {
  const object = closedObject(value, ["spanId", "spelling", "expansion", "macroExpansionId"], path);
  return {
    spanId: stableId(field(object, "spanId", path), `${path}.spanId`, "span"),
    spelling: parseFileRange(field(object, "spelling", path), `${path}.spelling`),
    expansion: parseFileRange(field(object, "expansion", path), `${path}.expansion`),
    macroExpansionId: nullableStableId(field(object, "macroExpansionId", path), `${path}.macroExpansionId`, "macro"),
  };
}

function parseFileRange(value: JsonValue, path: string): CppCuteFileRangeV1 {
  const object = closedObject(value, ["fileId", "startByte", "endByte"], path);
  return {
    fileId: stableId(field(object, "fileId", path), `${path}.fileId`, "file"),
    startByte: parseWireU64(field(object, "startByte", path), `${path}.startByte`),
    endByte: parseWireU64(field(object, "endByte", path), `${path}.endByte`),
  };
}

function parseMacroExpansion(value: JsonValue, path: string): CppCuteMacroExpansionV1 {
  const object = closedObject(value, [
    "macroExpansionId",
    "macroName",
    "definitionSpanId",
    "invocationSpanId",
    "parentMacroExpansionId",
  ], path);
  const macroName = stringValue(field(object, "macroName", path), `${path}.macroName`);
  if (!MACRO_NAME.test(macroName)) invalid(`${path}.macroName`, "macro name must be a C/C++ identifier");
  return {
    macroExpansionId: stableId(field(object, "macroExpansionId", path), `${path}.macroExpansionId`, "macro"),
    macroName,
    definitionSpanId: stableId(field(object, "definitionSpanId", path), `${path}.definitionSpanId`, "span"),
    invocationSpanId: stableId(field(object, "invocationSpanId", path), `${path}.invocationSpanId`, "span"),
    parentMacroExpansionId: nullableStableId(
      field(object, "parentMacroExpansionId", path),
      `${path}.parentMacroExpansionId`,
      "macro",
    ),
  };
}

function parseOrigin(value: JsonValue, path: string): CppCuteSourceOriginV1 {
  if (!isJsonObject(value)) invalid(path, "source origin must be an object");
  if (value.kind === "source") {
    const object = closedObject(value, ["kind", "spanId"], path);
    return { kind: "source", spanId: stableId(field(object, "spanId", path), `${path}.spanId`, "span") };
  }
  if (value.kind === "implicit") {
    const object = closedObject(value, ["kind", "anchorSpanId", "reason"], path);
    return {
      kind: "implicit",
      anchorSpanId: stableId(field(object, "anchorSpanId", path), `${path}.anchorSpanId`, "span"),
      reason: enumValue(field(object, "reason", path), [
        "implicit-cast",
        "implicit-construction",
        "default-argument",
        "compiler-builtin",
        "template-substitution",
      ] as const, `${path}.reason`),
    };
  }
  invalid(`${path}.kind`, "unknown source origin kind");
}

function parseType(value: JsonValue, path: string): CppCuteResolvedTypeV1 {
  if (!isJsonObject(value)) invalid(path, "resolved type must be an object");
  const common = parseTypeCommon(value, path);
  if (value.kind === "builtin") {
    const object = closedObject(value, ["typeId", "kind", "canonicalName", "qualifiers", "origin", "builtin"], path);
    return {
      ...common,
      kind: "builtin",
      builtin: enumValue(field(object, "builtin", path), [
        "void", "bool", "char", "signed-char", "unsigned-char", "short", "unsigned-short", "int",
        "unsigned-int", "long", "unsigned-long", "long-long", "unsigned-long-long", "half", "bfloat16",
        "float", "double",
      ] as const, `${path}.builtin`),
    };
  }
  if (value.kind === "pointer" || value.kind === "lvalue-reference" || value.kind === "rvalue-reference") {
    const object = closedObject(value, [
      "typeId", "kind", "canonicalName", "qualifiers", "origin", "pointeeTypeId", "addressSpace",
    ], path);
    return {
      ...common,
      kind: value.kind,
      pointeeTypeId: stableId(field(object, "pointeeTypeId", path), `${path}.pointeeTypeId`, "type"),
      addressSpace: parseAddressSpace(field(object, "addressSpace", path), `${path}.addressSpace`),
    };
  }
  if (value.kind === "array") {
    const object = closedObject(value, [
      "typeId", "kind", "canonicalName", "qualifiers", "origin", "elementTypeId", "elementCount",
    ], path);
    return {
      ...common,
      kind: "array",
      elementTypeId: stableId(field(object, "elementTypeId", path), `${path}.elementTypeId`, "type"),
      elementCount: parseWireU64(field(object, "elementCount", path), `${path}.elementCount`),
    };
  }
  if (value.kind === "vector") {
    const object = closedObject(value, [
      "typeId", "kind", "canonicalName", "qualifiers", "origin", "elementTypeId", "elementCount",
    ], path);
    return {
      ...common,
      kind: "vector",
      elementTypeId: stableId(field(object, "elementTypeId", path), `${path}.elementTypeId`, "type"),
      elementCount: positiveInteger(field(object, "elementCount", path), `${path}.elementCount`, 65_536),
    };
  }
  if (value.kind === "function") {
    const object = closedObject(value, [
      "typeId", "kind", "canonicalName", "qualifiers", "origin", "returnTypeId", "parameterTypeIds",
      "variadic", "callingConvention",
    ], path);
    return {
      ...common,
      kind: "function",
      returnTypeId: stableId(field(object, "returnTypeId", path), `${path}.returnTypeId`, "type"),
      parameterTypeIds: stableIdArray(field(object, "parameterTypeIds", path), `${path}.parameterTypeIds`, "type"),
      variadic: booleanValue(field(object, "variadic", path), `${path}.variadic`),
      callingConvention: parseCallingConvention(field(object, "callingConvention", path), `${path}.callingConvention`),
    };
  }
  if (value.kind === "record" || value.kind === "enum") {
    const object = closedObject(value, [
      "typeId", "kind", "canonicalName", "qualifiers", "origin", "declarationId", "complete",
    ], path);
    return {
      ...common,
      kind: value.kind,
      declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration"),
      complete: booleanValue(field(object, "complete", path), `${path}.complete`),
    };
  }
  if (value.kind === "template-specialization") {
    const object = closedObject(value, [
      "typeId", "kind", "canonicalName", "qualifiers", "origin", "templateDeclarationId", "arguments",
    ], path);
    return {
      ...common,
      kind: "template-specialization",
      templateDeclarationId: stableId(
        field(object, "templateDeclarationId", path),
        `${path}.templateDeclarationId`,
        "declaration",
      ),
      arguments: orderedArrayField(object, "arguments", path, 1_024).map((entry, index) =>
        parseTemplateArgument(entry, `${path}.arguments[${index}]`)),
    };
  }
  invalid(`${path}.kind`, "unknown resolved type kind");
}

function parseTypeCommon(value: JsonObject, path: string): {
  readonly typeId: string;
  readonly canonicalName: string;
  readonly qualifiers: CppCuteTypeQualifiersV1;
  readonly origin: CppCuteSourceOriginV1;
} {
  return {
    typeId: stableId(field(value, "typeId", path), `${path}.typeId`, "type"),
    canonicalName: boundedString(field(value, "canonicalName", path), `${path}.canonicalName`, 16_384),
    qualifiers: parseQualifiers(field(value, "qualifiers", path), `${path}.qualifiers`),
    origin: parseOrigin(field(value, "origin", path), `${path}.origin`),
  };
}

function parseQualifiers(value: JsonValue, path: string): CppCuteTypeQualifiersV1 {
  const object = closedObject(value, ["const", "volatile", "restrict"], path);
  return {
    const: booleanValue(field(object, "const", path), `${path}.const`),
    volatile: booleanValue(field(object, "volatile", path), `${path}.volatile`),
    restrict: booleanValue(field(object, "restrict", path), `${path}.restrict`),
  };
}

function parseTemplateArgument(value: JsonValue, path: string): CppCuteTemplateArgumentV1 {
  if (!isJsonObject(value)) invalid(path, "template argument must be an object");
  if (value.kind === "type") {
    const object = closedObject(value, ["kind", "typeId"], path);
    return { kind: "type", typeId: stableId(field(object, "typeId", path), `${path}.typeId`, "type") };
  }
  if (value.kind === "value") {
    const object = closedObject(value, ["kind", "constantId"], path);
    return { kind: "value", constantId: stableId(field(object, "constantId", path), `${path}.constantId`, "constant") };
  }
  if (value.kind === "template") {
    const object = closedObject(value, ["kind", "declarationId"], path);
    return {
      kind: "template",
      declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration"),
    };
  }
  invalid(`${path}.kind`, "unknown template argument kind");
}

function parseConstant(value: JsonValue, path: string): CppCuteConstantV1 {
  if (!isJsonObject(value)) invalid(path, "constant must be an object");
  const common = {
    constantId: stableId(field(value, "constantId", path), `${path}.constantId`, "constant"),
    typeId: stableId(field(value, "typeId", path), `${path}.typeId`, "type"),
    origin: parseOrigin(field(value, "origin", path), `${path}.origin`),
  };
  if (value.kind === "boolean") {
    const object = closedObject(value, ["constantId", "typeId", "origin", "kind", "value"], path);
    return { ...common, kind: "boolean", value: booleanValue(field(object, "value", path), `${path}.value`) };
  }
  if (value.kind === "signed-integer" || value.kind === "unsigned-integer") {
    const object = closedObject(value, ["constantId", "typeId", "origin", "kind", "bitWidth", "value"], path);
    const bitWidth = positiveInteger(field(object, "bitWidth", path), `${path}.bitWidth`, 64);
    return value.kind === "signed-integer"
      ? { ...common, kind: "signed-integer", bitWidth, value: parseWireI64(field(object, "value", path), `${path}.value`) }
      : { ...common, kind: "unsigned-integer", bitWidth, value: parseWireU64(field(object, "value", path), `${path}.value`) };
  }
  if (value.kind === "floating") {
    const object = closedObject(value, ["constantId", "typeId", "origin", "kind", "format", "bits"], path);
    const format = enumValue(field(object, "format", path), ["f16", "bf16", "f32", "f64"] as const, `${path}.format`);
    const bits = stringValue(field(object, "bits", path), `${path}.bits`);
    const digits = format === "f64" ? 16 : format === "f32" ? 8 : 4;
    if (!new RegExp(`^[0-9a-f]{${digits}}$`, "u").test(bits)) invalid(`${path}.bits`, `expected ${digits} exact lowercase hexadecimal digits`);
    return { ...common, kind: "floating", format, bits };
  }
  if (value.kind === "null-pointer") {
    closedObject(value, ["constantId", "typeId", "origin", "kind"], path);
    return { ...common, kind: "null-pointer" };
  }
  if (value.kind === "enum") {
    const object = closedObject(value, [
      "constantId", "typeId", "origin", "kind", "enumDeclarationId", "valueConstantId",
    ], path);
    return {
      ...common,
      kind: "enum",
      enumDeclarationId: stableId(field(object, "enumDeclarationId", path), `${path}.enumDeclarationId`, "declaration"),
      valueConstantId: stableId(field(object, "valueConstantId", path), `${path}.valueConstantId`, "constant"),
    };
  }
  if (value.kind === "aggregate") {
    const object = closedObject(value, ["constantId", "typeId", "origin", "kind", "elementConstantIds"], path);
    return {
      ...common,
      kind: "aggregate",
      elementConstantIds: stableIdArray(field(object, "elementConstantIds", path), `${path}.elementConstantIds`, "constant"),
    };
  }
  invalid(`${path}.kind`, "unknown constant kind");
}

function parseDeclaration(value: JsonValue, path: string): CppCuteDeclarationV3 {
  const object = closedObject(value, [
    "declarationId", "kind", "canonicalUsr", "canonicalName", "lexicalParentId", "semanticParentId", "typeId",
    "targetTypeId", "initializerExpressionId", "origin", "definitionKind", "linkage", "storageDuration", "memorySpace",
    "identitySpanId", "mangledName", "cudaAttributes",
  ], path);
  const canonicalUsr = boundedString(field(object, "canonicalUsr", path), `${path}.canonicalUsr`, 16_384);
  if (!CANONICAL_USR.test(canonicalUsr)) invalid(`${path}.canonicalUsr`, "canonical USR must use the Clang c:@ namespace");
  const kind = enumValue(field(object, "kind", path), [
    "namespace", "type-alias", "record", "enum", "field", "function", "parameter", "variable", "template",
    "template-parameter",
  ] as const, `${path}.kind`) as CppCuteDeclarationKindV1;
  const initializerExpressionId = nullableStableId(
    field(object, "initializerExpressionId", path),
    `${path}.initializerExpressionId`,
    "expression",
  );
  if (kind !== "variable" && initializerExpressionId !== null) {
    invalid(`${path}.initializerExpressionId`, "only variable declarations may carry an initializer expression");
  }
  const definitionKind = enumValue(field(object, "definitionKind", path), [
    "definition", "declaration-only", "builtin", "external",
  ] as const, `${path}.definitionKind`);
  if (initializerExpressionId !== null && definitionKind !== "definition") {
    invalid(`${path}.initializerExpressionId`, "an initializer requires a variable definition");
  }
  return {
    declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration"),
    kind,
    canonicalUsr,
    canonicalName: boundedString(field(object, "canonicalName", path), `${path}.canonicalName`, 16_384),
    lexicalParentId: nullableStableId(field(object, "lexicalParentId", path), `${path}.lexicalParentId`, "declaration"),
    semanticParentId: nullableStableId(field(object, "semanticParentId", path), `${path}.semanticParentId`, "declaration"),
    typeId: nullableStableId(field(object, "typeId", path), `${path}.typeId`, "type"),
    targetTypeId: nullableStableId(field(object, "targetTypeId", path), `${path}.targetTypeId`, "type"),
    initializerExpressionId,
    origin: parseOrigin(field(object, "origin", path), `${path}.origin`),
    identitySpanId: nullableStableId(field(object, "identitySpanId", path), `${path}.identitySpanId`, "span"),
    definitionKind,
    linkage: enumValue(field(object, "linkage", path), [
      "none", "internal", "external", "weak", "linkonce-odr",
    ] as const, `${path}.linkage`),
    storageDuration: enumValue(field(object, "storageDuration", path), [
      "none", "automatic", "static", "thread",
    ] as const, `${path}.storageDuration`),
    memorySpace: parseAddressSpace(field(object, "memorySpace", path), `${path}.memorySpace`),
    mangledName: nullableBoundedString(field(object, "mangledName", path), `${path}.mangledName`, 16_384),
    cudaAttributes: parseCudaAttributes(field(object, "cudaAttributes", path), `${path}.cudaAttributes`),
  };
}

function parseCudaAttributes(value: JsonValue, path: string): CppCuteCudaAttributesV1 {
  const object = closedObject(value, ["host", "device", "global", "forceInline"], path);
  return {
    host: booleanValue(field(object, "host", path), `${path}.host`),
    device: booleanValue(field(object, "device", path), `${path}.device`),
    global: booleanValue(field(object, "global", path), `${path}.global`),
    forceInline: booleanValue(field(object, "forceInline", path), `${path}.forceInline`),
  };
}

function parseTemplateInstantiation(value: JsonValue, path: string): CppCuteTemplateInstantiationV1 {
  const object = closedObject(value, [
    "instantiationId", "templateDeclarationId", "specializationDeclarationId", "arguments",
    "pointOfInstantiationSpanId", "parentInstantiationId", "depth",
  ], path);
  return {
    instantiationId: stableId(field(object, "instantiationId", path), `${path}.instantiationId`, "template-instantiation"),
    templateDeclarationId: stableId(
      field(object, "templateDeclarationId", path),
      `${path}.templateDeclarationId`,
      "declaration",
    ),
    specializationDeclarationId: stableId(
      field(object, "specializationDeclarationId", path),
      `${path}.specializationDeclarationId`,
      "declaration",
    ),
    arguments: orderedArrayField(object, "arguments", path, 1_024).map((entry, index) =>
      parseTemplateArgument(entry, `${path}.arguments[${index}]`)),
    pointOfInstantiationSpanId: stableId(
      field(object, "pointOfInstantiationSpanId", path),
      `${path}.pointOfInstantiationSpanId`,
      "span",
    ),
    parentInstantiationId: nullableStableId(
      field(object, "parentInstantiationId", path),
      `${path}.parentInstantiationId`,
      "template-instantiation",
    ),
    depth: nonnegativeInteger(field(object, "depth", path), `${path}.depth`),
  };
}

function parseOverloadResolution(value: JsonValue, path: string): CppCuteOverloadResolutionV1 {
  const object = closedObject(value, [
    "resolutionId", "origin", "selectedDeclarationId", "candidateDeclarationIds", "argumentTypeIds", "resultTypeId",
  ], path);
  return {
    resolutionId: stableId(field(object, "resolutionId", path), `${path}.resolutionId`, "overload-resolution"),
    origin: parseOrigin(field(object, "origin", path), `${path}.origin`),
    selectedDeclarationId: stableId(
      field(object, "selectedDeclarationId", path),
      `${path}.selectedDeclarationId`,
      "declaration",
    ),
    candidateDeclarationIds: sortedStableIdSet(
      field(object, "candidateDeclarationIds", path),
      `${path}.candidateDeclarationIds`,
      "declaration",
    ),
    argumentTypeIds: stableIdArray(field(object, "argumentTypeIds", path), `${path}.argumentTypeIds`, "type"),
    resultTypeId: stableId(field(object, "resultTypeId", path), `${path}.resultTypeId`, "type"),
  };
}

function parseSourceAbi(value: JsonValue, limits: CppCuteFrontendArtifactLimits, path: string): CppCuteSourceAbiV1 {
  const object = closedObject(value, ["types", "functions"], path);
  return {
    types: setArrayField(
      object,
      "types",
      path,
      limits.maxAbiEntries,
      parseTypeAbi,
      (entry) => `${entry.domain}:${entry.sourceTypeEntityId}`,
    ),
    functions: setArrayField(
      object,
      "functions",
      path,
      limits.maxAbiEntries,
      parseFunctionAbi,
      (entry) => `${entry.domain}:${entry.sourceEntityId}`,
    ),
  };
}

function parseSourceEntity(value: JsonValue, path: string): CppCuteSourceEntityV1 {
  const object = closedObject(value, [
    "sourceEntityId", "entityKind", "canonicalIdentity", "origin", "domains",
  ], path);
  const domains = arrayValue(field(object, "domains", path), `${path}.domains`).map((entry, index) =>
    parseSemanticDomain(entry, `${path}.domains[${index}]`));
  if (domains.length === 0 || domains.length > 2) {
    invalid(`${path}.domains`, "source entity requires one or two semantic domains");
  }
  requireStrictlySorted(domains, (entry) => entry, `${path}.domains`);
  return {
    sourceEntityId: stableId(field(object, "sourceEntityId", path), `${path}.sourceEntityId`, "source-entity"),
    entityKind: enumValue(
      field(object, "entityKind", path),
      ["type", "function", "field", "parameter", "variable"] as const,
      `${path}.entityKind`,
    ),
    canonicalIdentity: boundedString(
      field(object, "canonicalIdentity", path),
      `${path}.canonicalIdentity`,
      4_096,
    ),
    origin: parseOrigin(field(object, "origin", path), `${path}.origin`),
    domains,
  };
}

function parseTypeAbi(value: JsonValue, path: string): CppCuteTypeAbiV1 {
  const object = closedObject(value, [
    "domain", "shared", "sourceTypeEntityId", "deviceTypeId", "sizeBits", "alignmentBits", "fields", "bases",
  ], path);
  return {
    domain: parseSemanticDomain(field(object, "domain", path), `${path}.domain`),
    shared: booleanValue(field(object, "shared", path), `${path}.shared`),
    sourceTypeEntityId: stableId(
      field(object, "sourceTypeEntityId", path),
      `${path}.sourceTypeEntityId`,
      "source-entity",
    ),
    deviceTypeId: nullableStableId(field(object, "deviceTypeId", path), `${path}.deviceTypeId`, "type"),
    sizeBits: parseWireU64(field(object, "sizeBits", path), `${path}.sizeBits`),
    alignmentBits: parseWireU64(field(object, "alignmentBits", path), `${path}.alignmentBits`),
    fields: orderedArrayField(object, "fields", path, 16_384).map((entry, index) =>
      parseAbiField(entry, `${path}.fields[${index}]`)),
    bases: orderedArrayField(object, "bases", path, 16_384).map((entry, index) =>
      parseAbiBase(entry, `${path}.bases[${index}]`)),
  };
}

function parseAbiField(value: JsonValue, path: string): CppCuteAbiFieldV1 {
  const object = closedObject(value, ["sourceEntityId", "sourceTypeEntityId", "bitOffset"], path);
  return {
    sourceEntityId: stableId(field(object, "sourceEntityId", path), `${path}.sourceEntityId`, "source-entity"),
    sourceTypeEntityId: stableId(
      field(object, "sourceTypeEntityId", path),
      `${path}.sourceTypeEntityId`,
      "source-entity",
    ),
    bitOffset: parseWireU64(field(object, "bitOffset", path), `${path}.bitOffset`),
  };
}

function parseAbiBase(value: JsonValue, path: string): CppCuteAbiBaseV1 {
  const object = closedObject(value, ["sourceTypeEntityId", "bitOffset", "virtual"], path);
  return {
    sourceTypeEntityId: stableId(
      field(object, "sourceTypeEntityId", path),
      `${path}.sourceTypeEntityId`,
      "source-entity",
    ),
    bitOffset: parseWireU64(field(object, "bitOffset", path), `${path}.bitOffset`),
    virtual: booleanValue(field(object, "virtual", path), `${path}.virtual`),
  };
}

function parseFunctionAbi(value: JsonValue, path: string): CppCuteFunctionAbiV1 {
  const object = closedObject(value, [
    "domain", "shared", "sourceEntityId", "deviceDeclarationId", "loweredCallingConvention",
    "returnSourceTypeEntityId", "returnPassing", "parameters",
  ], path);
  return {
    domain: parseSemanticDomain(field(object, "domain", path), `${path}.domain`),
    shared: booleanValue(field(object, "shared", path), `${path}.shared`),
    sourceEntityId: stableId(field(object, "sourceEntityId", path), `${path}.sourceEntityId`, "source-entity"),
    deviceDeclarationId: nullableStableId(
      field(object, "deviceDeclarationId", path),
      `${path}.deviceDeclarationId`,
      "declaration",
    ),
    loweredCallingConvention: enumValue(
      field(object, "loweredCallingConvention", path),
      ["c", "cxx-member", "cuda-launch-stub", "nvptx-kernel", "nvptx-device"] as const,
      `${path}.loweredCallingConvention`,
    ),
    returnSourceTypeEntityId: stableId(
      field(object, "returnSourceTypeEntityId", path),
      `${path}.returnSourceTypeEntityId`,
      "source-entity",
    ),
    returnPassing: enumValue(field(object, "returnPassing", path), ["direct", "indirect", "ignore"] as const, `${path}.returnPassing`),
    parameters: orderedArrayField(object, "parameters", path, 16_384).map((entry, index) =>
      parseParameterAbi(entry, `${path}.parameters[${index}]`)),
  };
}

function parseParameterAbi(value: JsonValue, path: string): CppCuteParameterAbiV1 {
  const object = closedObject(value, ["ordinal", "sourceEntityId", "sourceTypeEntityId", "passing"], path);
  return {
    ordinal: boundedNonnegativeInteger(field(object, "ordinal", path), `${path}.ordinal`, 65_535),
    sourceEntityId: stableId(field(object, "sourceEntityId", path), `${path}.sourceEntityId`, "source-entity"),
    sourceTypeEntityId: stableId(
      field(object, "sourceTypeEntityId", path),
      `${path}.sourceTypeEntityId`,
      "source-entity",
    ),
    passing: enumValue(field(object, "passing", path), ["direct", "indirect", "ignore"] as const, `${path}.passing`),
  };
}

function parseFunctionBody(value: JsonValue, limits: CppCuteFrontendArtifactLimits, path: string): CppCuteFunctionBodyV1 {
  const object = closedObject(value, ["bodyId", "declarationId", "rootStatementId", "statements", "expressions"], path);
  const statements = setArrayField(
    object,
    "statements",
    path,
    limits.maxBodyNodes,
    parseStatement,
    (entry) => entry.statementId,
  );
  const expressions = setArrayField(
    object,
    "expressions",
    path,
    limits.maxBodyNodes,
    parseExpression,
    (entry) => entry.expressionId,
  );
  if (statements.length + expressions.length > limits.maxBodyNodes) {
    resource(path, `function body node count exceeds ${limits.maxBodyNodes}`);
  }
  return {
    bodyId: stableId(field(object, "bodyId", path), `${path}.bodyId`, "body"),
    declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration"),
    rootStatementId: stableId(field(object, "rootStatementId", path), `${path}.rootStatementId`, "statement"),
    statements,
    expressions,
  };
}

function parseStatement(value: JsonValue, path: string): CppCuteStatementV1 {
  if (!isJsonObject(value)) invalid(path, "statement must be an object");
  const common = {
    statementId: stableId(field(value, "statementId", path), `${path}.statementId`, "statement"),
    origin: parseOrigin(field(value, "origin", path), `${path}.origin`),
  };
  if (value.kind === "block") {
    const object = closedObject(value, ["statementId", "origin", "kind", "statementIds"], path);
    return { ...common, kind: "block", statementIds: stableIdArray(field(object, "statementIds", path), `${path}.statementIds`, "statement") };
  }
  if (value.kind === "declaration") {
    const object = closedObject(value, ["statementId", "origin", "kind", "declarationId"], path);
    return { ...common, kind: "declaration", declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration") };
  }
  if (value.kind === "expression") {
    const object = closedObject(value, ["statementId", "origin", "kind", "expressionId"], path);
    return { ...common, kind: "expression", expressionId: stableId(field(object, "expressionId", path), `${path}.expressionId`, "expression") };
  }
  if (value.kind === "if") {
    const object = closedObject(value, [
      "statementId", "origin", "kind", "conditionExpressionId", "thenStatementId", "elseStatementId",
    ], path);
    return {
      ...common,
      kind: "if",
      conditionExpressionId: stableId(field(object, "conditionExpressionId", path), `${path}.conditionExpressionId`, "expression"),
      thenStatementId: stableId(field(object, "thenStatementId", path), `${path}.thenStatementId`, "statement"),
      elseStatementId: nullableStableId(field(object, "elseStatementId", path), `${path}.elseStatementId`, "statement"),
    };
  }
  if (value.kind === "for") {
    const object = closedObject(value, [
      "statementId", "origin", "kind", "initializerStatementId", "conditionExpressionId", "incrementExpressionId",
      "bodyStatementId",
    ], path);
    return {
      ...common,
      kind: "for",
      initializerStatementId: nullableStableId(field(object, "initializerStatementId", path), `${path}.initializerStatementId`, "statement"),
      conditionExpressionId: nullableStableId(field(object, "conditionExpressionId", path), `${path}.conditionExpressionId`, "expression"),
      incrementExpressionId: nullableStableId(field(object, "incrementExpressionId", path), `${path}.incrementExpressionId`, "expression"),
      bodyStatementId: stableId(field(object, "bodyStatementId", path), `${path}.bodyStatementId`, "statement"),
    };
  }
  if (value.kind === "while") {
    const object = closedObject(value, [
      "statementId", "origin", "kind", "conditionExpressionId", "bodyStatementId",
    ], path);
    return {
      ...common,
      kind: "while",
      conditionExpressionId: stableId(field(object, "conditionExpressionId", path), `${path}.conditionExpressionId`, "expression"),
      bodyStatementId: stableId(field(object, "bodyStatementId", path), `${path}.bodyStatementId`, "statement"),
    };
  }
  if (value.kind === "return") {
    const object = closedObject(value, ["statementId", "origin", "kind", "expressionId"], path);
    return { ...common, kind: "return", expressionId: nullableStableId(field(object, "expressionId", path), `${path}.expressionId`, "expression") };
  }
  if (value.kind === "break" || value.kind === "continue") {
    closedObject(value, ["statementId", "origin", "kind"], path);
    return { ...common, kind: value.kind };
  }
  invalid(`${path}.kind`, "unknown statement kind");
}

function parseExpression(value: JsonValue, path: string): CppCuteExpressionV1 {
  if (!isJsonObject(value)) invalid(path, "expression must be an object");
  const common = {
    expressionId: stableId(field(value, "expressionId", path), `${path}.expressionId`, "expression"),
    typeId: stableId(field(value, "typeId", path), `${path}.typeId`, "type"),
    valueCategory: enumValue(field(value, "valueCategory", path), ["lvalue", "xvalue", "prvalue"] as const, `${path}.valueCategory`),
    origin: parseOrigin(field(value, "origin", path), `${path}.origin`),
  };
  const base = ["expressionId", "typeId", "valueCategory", "origin", "kind"];
  if (value.kind === "constant") {
    const object = closedObject(value, [...base, "constantId"], path);
    return { ...common, kind: "constant", constantId: stableId(field(object, "constantId", path), `${path}.constantId`, "constant") };
  }
  if (value.kind === "declaration-reference") {
    const object = closedObject(value, [...base, "declarationId"], path);
    return { ...common, kind: "declaration-reference", declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration") };
  }
  if (value.kind === "resolved-call") {
    const object = closedObject(value, [...base, "overloadResolutionId", "argumentExpressionIds"], path);
    return {
      ...common,
      kind: "resolved-call",
      overloadResolutionId: stableId(field(object, "overloadResolutionId", path), `${path}.overloadResolutionId`, "overload-resolution"),
      argumentExpressionIds: stableIdArray(field(object, "argumentExpressionIds", path), `${path}.argumentExpressionIds`, "expression"),
    };
  }
  if (value.kind === "construction") {
    const object = closedObject(value, [...base, "constructorDeclarationId", "argumentExpressionIds"], path);
    return {
      ...common,
      kind: "construction",
      constructorDeclarationId: stableId(field(object, "constructorDeclarationId", path), `${path}.constructorDeclarationId`, "declaration"),
      argumentExpressionIds: stableIdArray(field(object, "argumentExpressionIds", path), `${path}.argumentExpressionIds`, "expression"),
    };
  }
  if (value.kind === "member-access") {
    const object = closedObject(value, [...base, "baseExpressionId", "memberDeclarationId"], path);
    return {
      ...common,
      kind: "member-access",
      baseExpressionId: stableId(field(object, "baseExpressionId", path), `${path}.baseExpressionId`, "expression"),
      memberDeclarationId: stableId(field(object, "memberDeclarationId", path), `${path}.memberDeclarationId`, "declaration"),
    };
  }
  if (value.kind === "subscript") {
    const object = closedObject(value, [...base, "baseExpressionId", "indexExpressionId"], path);
    return {
      ...common,
      kind: "subscript",
      baseExpressionId: stableId(field(object, "baseExpressionId", path), `${path}.baseExpressionId`, "expression"),
      indexExpressionId: stableId(field(object, "indexExpressionId", path), `${path}.indexExpressionId`, "expression"),
    };
  }
  if (value.kind === "cast") {
    const object = closedObject(value, [...base, "castKind", "operandExpressionId"], path);
    return {
      ...common,
      kind: "cast",
      castKind: enumValue(field(object, "castKind", path), [
        "integral", "floating", "pointer", "qualification", "lvalue-to-rvalue",
      ] as const, `${path}.castKind`),
      operandExpressionId: stableId(field(object, "operandExpressionId", path), `${path}.operandExpressionId`, "expression"),
    };
  }
  if (value.kind === "unary") {
    const object = closedObject(value, [...base, "operator", "operandExpressionId"], path);
    return {
      ...common,
      kind: "unary",
      operator: enumValue(field(object, "operator", path), [
        "plus", "minus", "logical-not", "bitwise-not", "dereference", "address-of",
      ] as const, `${path}.operator`),
      operandExpressionId: stableId(field(object, "operandExpressionId", path), `${path}.operandExpressionId`, "expression"),
    };
  }
  if (value.kind === "binary") {
    const object = closedObject(value, [...base, "operator", "leftExpressionId", "rightExpressionId"], path);
    return {
      ...common,
      kind: "binary",
      operator: enumValue(field(object, "operator", path), [
        "add", "subtract", "multiply", "divide", "remainder", "equal", "not-equal", "less", "less-equal",
        "greater", "greater-equal", "logical-and", "logical-or", "bitwise-and", "bitwise-or", "bitwise-xor",
        "shift-left", "shift-right", "assign",
      ] as const, `${path}.operator`),
      leftExpressionId: stableId(field(object, "leftExpressionId", path), `${path}.leftExpressionId`, "expression"),
      rightExpressionId: stableId(field(object, "rightExpressionId", path), `${path}.rightExpressionId`, "expression"),
    };
  }
  if (value.kind === "conditional") {
    const object = closedObject(value, [
      ...base, "conditionExpressionId", "thenExpressionId", "elseExpressionId",
    ], path);
    return {
      ...common,
      kind: "conditional",
      conditionExpressionId: stableId(field(object, "conditionExpressionId", path), `${path}.conditionExpressionId`, "expression"),
      thenExpressionId: stableId(field(object, "thenExpressionId", path), `${path}.thenExpressionId`, "expression"),
      elseExpressionId: stableId(field(object, "elseExpressionId", path), `${path}.elseExpressionId`, "expression"),
    };
  }
  if (value.kind === "target-intrinsic") {
    const object = closedObject(value, [...base, "intrinsicFactId"], path);
    return { ...common, kind: "target-intrinsic", intrinsicFactId: stableId(field(object, "intrinsicFactId", path), `${path}.intrinsicFactId`, "fact") };
  }
  invalid(`${path}.kind`, "unknown expression kind");
}

function parseFact(value: JsonValue, path: string): CppCuteResolvedFactV1 {
  if (!isJsonObject(value)) invalid(path, "resolved fact must be an object");
  if (value.kind === "affine-layout") return parseAffineLayoutFact(value, path);
  if (value.kind === "tensor") return parseTensorFact(value, path);
  if (value.kind === "target-intrinsic") return parseTargetIntrinsicFact(value, path);
  invalid(`${path}.kind`, "unknown resolved fact kind");
}

function parseAffineLayoutFact(value: JsonValue, path: string): CppCuteAffineLayoutFactV1 {
  const object = closedObject(value, [
    "factId", "kind", "origin", "resultDeclarationId", "shape", "stride", "rank", "leafRank", "size", "cosize",
  ], path);
  return {
    factId: stableId(field(object, "factId", path), `${path}.factId`, "fact"),
    kind: "affine-layout",
    origin: parseOrigin(field(object, "origin", path), `${path}.origin`),
    resultDeclarationId: stableId(field(object, "resultDeclarationId", path), `${path}.resultDeclarationId`, "declaration"),
    shape: parseHierarchy(field(object, "shape", path), `${path}.shape`, 1),
    stride: parseHierarchy(field(object, "stride", path), `${path}.stride`, 1),
    rank: positiveInteger(field(object, "rank", path), `${path}.rank`, 1_024),
    leafRank: positiveInteger(field(object, "leafRank", path), `${path}.leafRank`, 1_024),
    size: parseIntegerExpr(field(object, "size", path), `${path}.size`, 1),
    cosize: parseIntegerExpr(field(object, "cosize", path), `${path}.cosize`, 1),
  };
}

function parseHierarchy(value: JsonValue, path: string, depth: number): CppCuteHierarchyV1 {
  if (depth > 64) resource(path, "CuTe hierarchy depth exceeds 64");
  if (!isJsonObject(value)) invalid(path, "CuTe hierarchy must be an object");
  if (value.kind === "scalar") {
    const object = closedObject(value, ["kind", "value"], path);
    return { kind: "scalar", value: parseIntegerExpr(field(object, "value", path), `${path}.value`, depth + 1) };
  }
  if (value.kind === "tuple") {
    const object = closedObject(value, ["kind", "elements"], path);
    const elements = orderedArrayField(object, "elements", path, 1_024).map((entry, index) =>
      parseHierarchy(entry, `${path}.elements[${index}]`, depth + 1));
    if (elements.length === 0) invalid(`${path}.elements`, "CuTe tuple must not be empty");
    return { kind: "tuple", elements };
  }
  invalid(`${path}.kind`, "unknown CuTe hierarchy kind");
}

function parseIntegerExpr(value: JsonValue, path: string, depth: number): CppCuteIntegerExprV1 {
  if (depth > 64) resource(path, "CuTe integer expression depth exceeds 64");
  if (!isJsonObject(value)) invalid(path, "CuTe integer expression must be an object");
  if (value.kind === "integer") {
    const object = closedObject(value, ["kind", "value"], path);
    return { kind: "integer", value: parseWireI64(field(object, "value", path), `${path}.value`) };
  }
  if (value.kind === "runtime") {
    const object = closedObject(value, ["kind", "declarationId"], path);
    return {
      kind: "runtime",
      declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration"),
    };
  }
  if (value.kind === "add" || value.kind === "multiply" || value.kind === "minimum" || value.kind === "maximum") {
    const object = closedObject(value, ["kind", "values"], path);
    const values = orderedArrayField(object, "values", path, 1_024).map((entry, index) =>
      parseIntegerExpr(entry, `${path}.values[${index}]`, depth + 1));
    if (values.length < 2) invalid(`${path}.values`, `${value.kind} requires at least two values`);
    return { kind: value.kind, values };
  }
  if (value.kind === "floor-divide" || value.kind === "ceil-divide" || value.kind === "modulo") {
    const object = closedObject(value, ["kind", "value", "divisor"], path);
    return {
      kind: value.kind,
      value: parseIntegerExpr(field(object, "value", path), `${path}.value`, depth + 1),
      divisor: parseIntegerExpr(field(object, "divisor", path), `${path}.divisor`, depth + 1),
    };
  }
  invalid(`${path}.kind`, "unknown CuTe integer expression kind");
}

function parseTensorFact(value: JsonValue, path: string): CppCuteTensorFactV1 {
  const object = closedObject(value, [
    "factId", "kind", "origin", "resultDeclarationId", "elementTypeId", "layoutFactId", "engine", "memorySpace",
  ], path);
  return {
    factId: stableId(field(object, "factId", path), `${path}.factId`, "fact"),
    kind: "tensor",
    origin: parseOrigin(field(object, "origin", path), `${path}.origin`),
    resultDeclarationId: stableId(field(object, "resultDeclarationId", path), `${path}.resultDeclarationId`, "declaration"),
    elementTypeId: stableId(field(object, "elementTypeId", path), `${path}.elementTypeId`, "type"),
    layoutFactId: stableId(field(object, "layoutFactId", path), `${path}.layoutFactId`, "fact"),
    engine: parseTensorEngine(field(object, "engine", path), `${path}.engine`),
    memorySpace: parseAddressSpace(field(object, "memorySpace", path), `${path}.memorySpace`),
  };
}

function parseTensorEngine(value: JsonValue, path: string): CppCuteTensorEngineV1 {
  if (!isJsonObject(value)) invalid(path, "tensor engine must be an object");
  if (value.kind === "global-pointer") {
    const object = closedObject(value, ["kind", "pointerDeclarationId", "nullable"], path);
    return {
      kind: "global-pointer",
      pointerDeclarationId: stableId(field(object, "pointerDeclarationId", path), `${path}.pointerDeclarationId`, "declaration"),
      nullable: booleanValue(field(object, "nullable", path), `${path}.nullable`),
    };
  }
  if (value.kind === "shared-pointer") {
    const object = closedObject(value, ["kind", "pointerDeclarationId"], path);
    return { kind: "shared-pointer", pointerDeclarationId: stableId(field(object, "pointerDeclarationId", path), `${path}.pointerDeclarationId`, "declaration") };
  }
  if (value.kind === "register-array") {
    const object = closedObject(value, ["kind", "arrayDeclarationId"], path);
    return { kind: "register-array", arrayDeclarationId: stableId(field(object, "arrayDeclarationId", path), `${path}.arrayDeclarationId`, "declaration") };
  }
  if (value.kind === "pointer-array") {
    const object = closedObject(value, ["kind", "pointerDeclarationIds"], path);
    const pointerDeclarationIds = stableIdArray(field(object, "pointerDeclarationIds", path), `${path}.pointerDeclarationIds`, "declaration");
    if (pointerDeclarationIds.length === 0) invalid(`${path}.pointerDeclarationIds`, "pointer array must contain at least one pointer");
    return { kind: "pointer-array", pointerDeclarationIds };
  }
  if (value.kind === "indirect") {
    const object = closedObject(value, ["kind", "engineDeclarationId"], path);
    return { kind: "indirect", engineDeclarationId: stableId(field(object, "engineDeclarationId", path), `${path}.engineDeclarationId`, "declaration") };
  }
  invalid(`${path}.kind`, "unknown tensor engine kind");
}

function parseTargetIntrinsicFact(value: JsonValue, path: string): CppCuteTargetIntrinsicFactV1 {
  const object = closedObject(value, [
    "factId", "kind", "origin", "familyId", "operation", "operandExpressionIds", "resultTypeId", "effects",
    "availability",
  ], path);
  const familyId = capabilityId(field(object, "familyId", path), `${path}.familyId`);
  return {
    factId: stableId(field(object, "factId", path), `${path}.factId`, "fact"),
    kind: "target-intrinsic",
    origin: parseOrigin(field(object, "origin", path), `${path}.origin`),
    familyId,
    operation: parseIntrinsicOperation(field(object, "operation", path), `${path}.operation`),
    operandExpressionIds: stableIdArray(field(object, "operandExpressionIds", path), `${path}.operandExpressionIds`, "expression"),
    resultTypeId: nullableStableId(field(object, "resultTypeId", path), `${path}.resultTypeId`, "type"),
    effects: parseIntrinsicEffects(field(object, "effects", path), `${path}.effects`),
    availability: parseIntrinsicAvailability(field(object, "availability", path), `${path}.availability`),
  };
}

function parseIntrinsicOperation(value: JsonValue, path: string): CppCuteTargetIntrinsicOperationV1 {
  if (!isJsonObject(value)) invalid(path, "target intrinsic operation must be an object");
  if (value.kind === "builtin-index") {
    const object = closedObject(value, ["kind", "scope", "axis"], path);
    return {
      kind: "builtin-index",
      scope: enumValue(field(object, "scope", path), ["grid", "block"] as const, `${path}.scope`),
      axis: enumValue(field(object, "axis", path), ["x", "y", "z"] as const, `${path}.axis`),
    };
  }
  if (value.kind === "barrier") {
    const object = closedObject(value, ["kind", "scope", "memorySemantics"], path);
    return {
      kind: "barrier",
      scope: enumValue(field(object, "scope", path), ["subgroup", "workgroup", "cluster"] as const, `${path}.scope`),
      memorySemantics: enumValue(
        field(object, "memorySemantics", path),
        ["acquire-release", "sequentially-consistent"] as const,
        `${path}.memorySemantics`,
      ),
    };
  }
  if (value.kind === "copy") {
    const object = closedObject(value, [
      "kind", "sourceSpace", "destinationSpace", "transferBits", "asynchronous",
    ], path);
    return {
      kind: "copy",
      sourceSpace: parseAddressSpace(field(object, "sourceSpace", path), `${path}.sourceSpace`),
      destinationSpace: parseAddressSpace(field(object, "destinationSpace", path), `${path}.destinationSpace`),
      transferBits: positiveInteger(field(object, "transferBits", path), `${path}.transferBits`, 65_536),
      asynchronous: booleanValue(field(object, "asynchronous", path), `${path}.asynchronous`),
    };
  }
  if (value.kind === "mma") {
    const object = closedObject(value, ["kind", "m", "n", "k", "aTypeId", "bTypeId", "accumulatorTypeId"], path);
    return {
      kind: "mma",
      m: positiveInteger(field(object, "m", path), `${path}.m`, 65_536),
      n: positiveInteger(field(object, "n", path), `${path}.n`, 65_536),
      k: positiveInteger(field(object, "k", path), `${path}.k`, 65_536),
      aTypeId: stableId(field(object, "aTypeId", path), `${path}.aTypeId`, "type"),
      bTypeId: stableId(field(object, "bTypeId", path), `${path}.bTypeId`, "type"),
      accumulatorTypeId: stableId(field(object, "accumulatorTypeId", path), `${path}.accumulatorTypeId`, "type"),
    };
  }
  if (value.kind === "pipeline") {
    const object = closedObject(value, ["kind", "action", "scope"], path);
    return {
      kind: "pipeline",
      action: enumValue(field(object, "action", path), ["arrive", "commit", "wait", "release"] as const, `${path}.action`),
      scope: enumValue(field(object, "scope", path), ["workgroup", "cluster"] as const, `${path}.scope`),
    };
  }
  if (value.kind === "capability") {
    const object = closedObject(value, ["kind", "capabilityId"], path);
    return { kind: "capability", capabilityId: capabilityId(field(object, "capabilityId", path), `${path}.capabilityId`) };
  }
  invalid(`${path}.kind`, "unknown target intrinsic operation kind");
}

function parseIntrinsicEffects(value: JsonValue, path: string): CppCuteIntrinsicEffectsV1 {
  const object = closedObject(value, ["readsMemory", "writesMemory", "synchronizes", "convergent"], path);
  return {
    readsMemory: booleanValue(field(object, "readsMemory", path), `${path}.readsMemory`),
    writesMemory: booleanValue(field(object, "writesMemory", path), `${path}.writesMemory`),
    synchronizes: booleanValue(field(object, "synchronizes", path), `${path}.synchronizes`),
    convergent: booleanValue(field(object, "convergent", path), `${path}.convergent`),
  };
}

function parseIntrinsicAvailability(value: JsonValue, path: string): CppCuteTargetIntrinsicAvailabilityV1 {
  if (!isJsonObject(value)) invalid(path, "target intrinsic availability must be an object");
  if (value.kind === "portable-candidate") {
    closedObject(value, ["kind"], path);
    return { kind: "portable-candidate" };
  }
  if (value.kind === "requires-capability") {
    const object = closedObject(value, ["kind", "capabilityIds"], path);
    return {
      kind: "requires-capability",
      capabilityIds: sortedCapabilitySet(field(object, "capabilityIds", path), `${path}.capabilityIds`),
    };
  }
  if (value.kind === "recognized-unsupported") {
    const object = closedObject(value, ["kind", "diagnosticId"], path);
    return {
      kind: "recognized-unsupported",
      diagnosticId: stableId(field(object, "diagnosticId", path), `${path}.diagnosticId`, "diagnostic"),
    };
  }
  invalid(`${path}.kind`, "unknown target intrinsic availability kind");
}

function parseEntry(value: JsonValue, path: string): CppCuteFrontendEntryV1 {
  if (!isJsonObject(value)) invalid(path, "frontend entry must be an object");
  if (value.kind === "layout") {
    const object = closedObject(value, ["entryId", "kind", "layoutFactId", "selectedRootDeclarationIds"], path);
    return {
      entryId: stableId(field(object, "entryId", path), `${path}.entryId`, "entry"),
      kind: "layout",
      layoutFactId: stableId(field(object, "layoutFactId", path), `${path}.layoutFactId`, "fact"),
      selectedRootDeclarationIds: sortedStableIdSet(
        field(object, "selectedRootDeclarationIds", path),
        `${path}.selectedRootDeclarationIds`,
        "declaration",
      ),
    };
  }
  if (value.kind === "view-copy") {
    const object = closedObject(value, [
      "entryId", "kind", "sourceTensorFactId", "destinationTensorFactId", "operationExpressionId",
      "selectedRootDeclarationIds",
    ], path);
    return {
      entryId: stableId(field(object, "entryId", path), `${path}.entryId`, "entry"),
      kind: "view-copy",
      sourceTensorFactId: stableId(field(object, "sourceTensorFactId", path), `${path}.sourceTensorFactId`, "fact"),
      destinationTensorFactId: stableId(
        field(object, "destinationTensorFactId", path),
        `${path}.destinationTensorFactId`,
        "fact",
      ),
      operationExpressionId: stableId(
        field(object, "operationExpressionId", path),
        `${path}.operationExpressionId`,
        "expression",
      ),
      selectedRootDeclarationIds: sortedStableIdSet(
        field(object, "selectedRootDeclarationIds", path),
        `${path}.selectedRootDeclarationIds`,
        "declaration",
      ),
    };
  }
  invalid(`${path}.kind`, "unknown frontend entry kind");
}

function parseDiagnostic(
  value: JsonValue,
  limits: CppCuteFrontendArtifactLimits,
  path: string,
): CppCuteFrontendDiagnosticV3 {
  const object = closedObject(value, [
    "diagnosticId", "phase", "severity", "code", "renderedMessage", "location", "subject", "parentDiagnosticId",
  ], path);
  const code = stringValue(field(object, "code", path), `${path}.code`);
  if (!DIAGNOSTIC_CODE.test(code)) invalid(`${path}.code`, "diagnostic code must be namespaced");
  const subject = parseDiagnosticSubject(field(object, "subject", path), `${path}.subject`);
  const location = parseDiagnosticLocation(field(object, "location", path), limits, `${path}.location`);
  if (location.kind === "none" && subject.kind !== "compiler") {
    invalid(`${path}.location`, "locationless frontend diagnostics require a compiler subject");
  }
  return {
    diagnosticId: stableId(field(object, "diagnosticId", path), `${path}.diagnosticId`, "diagnostic"),
    phase: enumValue(field(object, "phase", path), [
      "preprocessing", "parsing", "name-lookup", "overload-resolution",
      "template-instantiation", "cuda-sema", "artifact-extraction",
    ] as const, `${path}.phase`) as CppCuteDiagnosticPhaseV3,
    severity: enumValue(field(object, "severity", path), [
      "remark", "note", "warning", "error", "fatal",
    ] as const, `${path}.severity`),
    code,
    renderedMessage: boundedString(
      field(object, "renderedMessage", path),
      `${path}.renderedMessage`,
      limits.maxStringBytes,
    ),
    location,
    subject,
    parentDiagnosticId: nullableStableId(
      field(object, "parentDiagnosticId", path),
      `${path}.parentDiagnosticId`,
      "diagnostic",
    ),
  };
}

function parseDiagnosticLocation(
  value: JsonValue,
  limits: CppCuteFrontendArtifactLimits,
  path: string,
): CppCuteDiagnosticLocationV3 {
  if (!isJsonObject(value)) invalid(path, "diagnostic location must be an object");
  if (value.kind === "none") {
    closedObject(value, ["kind"], path);
    return { kind: "none" };
  }
  if (value.kind === "source") {
    const object = closedObject(value, ["kind", "primarySpanId", "related"], path);
    const related = orderedArrayField(object, "related", path, limits.maxRelatedDiagnosticLocations).map((entry, index) =>
      parseRelatedDiagnostic(entry, limits, `${path}.related[${index}]`));
    return {
      kind: "source",
      primarySpanId: stableId(field(object, "primarySpanId", path), `${path}.primarySpanId`, "span"),
      related,
    };
  }
  invalid(`${path}.kind`, "unknown diagnostic location kind");
}

function parseRelatedDiagnostic(
  value: JsonValue,
  limits: CppCuteFrontendArtifactLimits,
  path: string,
): CppCuteDiagnosticRelatedLocationV1 {
  const object = closedObject(value, ["spanId", "message"], path);
  return {
    spanId: stableId(field(object, "spanId", path), `${path}.spanId`, "span"),
    message: boundedString(field(object, "message", path), `${path}.message`, limits.maxStringBytes),
  };
}

function parseDiagnosticSubject(value: JsonValue, path: string): CppCuteDiagnosticSubjectV3 {
  if (!isJsonObject(value)) invalid(path, "diagnostic subject must be an object");
  if (value.kind === "compiler") {
    closedObject(value, ["kind"], path);
    return { kind: value.kind };
  }
  if (value.kind === "file") {
    const object = closedObject(value, ["kind", "fileId"], path);
    return { kind: "file", fileId: stableId(field(object, "fileId", path), `${path}.fileId`, "file") };
  }
  if (value.kind === "declaration") {
    const object = closedObject(value, ["kind", "declarationId"], path);
    return { kind: "declaration", declarationId: stableId(field(object, "declarationId", path), `${path}.declarationId`, "declaration") };
  }
  if (value.kind === "type") {
    const object = closedObject(value, ["kind", "typeId"], path);
    return { kind: "type", typeId: stableId(field(object, "typeId", path), `${path}.typeId`, "type") };
  }
  if (value.kind === "expression") {
    const object = closedObject(value, ["kind", "expressionId"], path);
    return { kind: "expression", expressionId: stableId(field(object, "expressionId", path), `${path}.expressionId`, "expression") };
  }
  if (value.kind === "fact") {
    const object = closedObject(value, ["kind", "factId"], path);
    return { kind: "fact", factId: stableId(field(object, "factId", path), `${path}.factId`, "fact") };
  }
  invalid(`${path}.kind`, "unknown diagnostic subject kind");
}

function parseOutcome(
  value: JsonValue,
  limits: CppCuteFrontendArtifactLimits,
  path: string,
): CppCuteFrontendOutcomeV1 {
  if (!isJsonObject(value)) invalid(path, "frontend outcome must be an object");
  if (value.kind === "accepted") {
    const object = closedObject(value, ["kind", "selectedEntryIds"], path);
    return {
      kind: "accepted",
      selectedEntryIds: sortedStableIdSet(
        field(object, "selectedEntryIds", path),
        `${path}.selectedEntryIds`,
        "entry",
        limits.maxEntries,
      ),
    };
  }
  if (value.kind === "rejected") {
    const object = closedObject(value, ["kind", "blockingDiagnosticIds"], path);
    return {
      kind: "rejected",
      blockingDiagnosticIds: sortedStableIdSet(
        field(object, "blockingDiagnosticIds", path),
        `${path}.blockingDiagnosticIds`,
        "diagnostic",
        limits.maxDiagnostics,
      ),
    };
  }
  invalid(`${path}.kind`, "unknown frontend outcome kind");
}

function parseExtraction(value: JsonValue, path: string): CppCuteExtractionRecordV1 {
  const object = closedObject(value, ["compilationContractHash", "inputClosureSha256", "appliedTransforms"], path);
  const transforms = arrayValue(field(object, "appliedTransforms", path), `${path}.appliedTransforms`);
  if (transforms.length !== 0) invalid(`${path}.appliedTransforms`, "CUTE-002 profile forbids source transformations");
  return {
    compilationContractHash: sha256(
      field(object, "compilationContractHash", path),
      `${path}.compilationContractHash`,
    ),
    inputClosureSha256: sha256(field(object, "inputClosureSha256", path), `${path}.inputClosureSha256`),
    appliedTransforms: [],
  };
}

function parseAddressSpace(value: JsonValue, path: string): CppCuteAddressSpaceV1 {
  return enumValue(value, ["generic", "host", "global", "shared", "local", "constant"] as const, path);
}

function parseCallingConvention(value: JsonValue, path: string): CppCuteCallingConventionV1 {
  return enumValue(value, ["c", "cuda-kernel", "cuda-device", "cxx-member"] as const, path);
}

function parseSemanticDomain(value: JsonValue, path: string): CppCuteSemanticDomainV1 {
  return enumValue(value, ["host", "device"] as const, path);
}

function sortedCapabilitySet(value: JsonValue, path: string): readonly string[] {
  const values = arrayValue(value, path).map((entry, index) => capabilityId(entry, `${path}[${index}]`));
  requireStrictlySorted(values, (entry) => entry, path);
  return values;
}

function capabilityId(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!CAPABILITY_ID.test(text)) invalid(path, "invalid namespaced capability ID");
  return text;
}

function stableIdArray(value: JsonValue, path: string, kind: string): readonly string[] {
  return arrayValue(value, path).map((entry, index) => stableId(entry, `${path}[${index}]`, kind));
}

function sortedStableIdSet(
  value: JsonValue,
  path: string,
  kind: string,
  maximum?: number,
): readonly string[] {
  const values = stableIdArray(value, path, kind);
  if (maximum !== undefined && values.length > maximum) {
    resource(path, `${kind} ID count ${values.length} exceeds ${maximum}`);
  }
  requireStrictlySorted(values, (entry) => entry, path);
  return values;
}

function stableId(value: JsonValue, path: string, kind: string): string {
  const text = stringValue(value, path);
  const match = STABLE_ID.exec(text);
  if (match?.[1] !== kind) invalid(path, `expected stable ${kind} ID`);
  return text;
}

function nullableStableId(value: JsonValue, path: string, kind: string): string | null {
  return value === null ? null : stableId(value, path, kind);
}

function dependencyId(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!DEPENDENCY_ID.test(text)) invalid(path, "invalid dependency ID");
  return text;
}

function nullableDependencyId(value: JsonValue, path: string): string | null {
  return value === null ? null : dependencyId(value, path);
}

function sha256(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!SHA256_HEX.test(text)) invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  return text;
}

function nullableSha256(value: JsonValue, path: string): string | null {
  return value === null ? null : sha256(value, path);
}

function parseSetArray<T>(
  object: JsonObject,
  name: string,
  maximum: number,
  parser: (value: JsonValue, path: string) => T,
  id: (value: T) => string,
): readonly T[] {
  return setArrayField(object, name, "$.payload", maximum, parser, id);
}

function setArrayField<T>(
  object: JsonObject,
  name: string,
  parentPath: string,
  maximum: number,
  parser: (value: JsonValue, path: string) => T,
  id: (value: T) => string,
): readonly T[] {
  const path = `${parentPath}.${name}`;
  const values = arrayValue(field(object, name, parentPath), path);
  if (values.length > maximum) resource(path, `${name} count exceeds ${maximum}`);
  const parsed = values.map((entry, index) => parser(entry, `${path}[${index}]`));
  const seen = new Set<string>();
  for (const entry of parsed) {
    const key = id(entry);
    if (seen.has(key)) duplicate(path, `duplicate ID ${key}`);
    seen.add(key);
  }
  return parsed.sort((left, right) => compareCanonicalStrings(id(left), id(right)));
}

function orderedArrayField(object: JsonObject, name: string, parentPath: string, maximum: number): readonly JsonValue[] {
  const path = `${parentPath}.${name}`;
  const values = arrayValue(field(object, name, parentPath), path);
  if (values.length > maximum) resource(path, `${name} count exceeds ${maximum}`);
  return values;
}

function requireStrictlySorted<T>(values: readonly T[], key: (value: T) => string, path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined ||
        compareCanonicalStrings(key(previous), key(current)) >= 0) {
      invalid(path, "set-like entries must be strictly sorted and unique");
    }
  }
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const unknown = Object.keys(value).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length > 0) invalid(path, `unknown closed-record fields: ${unknown.sort().join(", ")}`);
  if (missing.length > 0) invalid(path, `missing required fields: ${missing.sort().join(", ")}`);
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
}

function exactString<const T extends string>(value: JsonValue, expected: T, path: string): T {
  const text = stringValue(value, path);
  if (text !== expected) invalid(path, `expected ${JSON.stringify(expected)}`);
  return expected;
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  const text = stringValue(value, path);
  const bytes = new TextEncoder().encode(text).byteLength;
  if (text.length === 0 || text.includes("\0") || bytes > maximumBytes) {
    invalid(path, `string must be non-empty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return text;
}

function validateVirtualPath(value: string, path: string): void {
  if (!value.startsWith("/") || value.length > 4_096 || value.includes("\\") || value.includes("\0")) {
    invalid(path, "virtual path must be bounded absolute POSIX syntax");
  }
  const segments = value.split("/");
  if (value !== "/" && segments.some((segment, index) => index > 0 && (segment.length === 0 || segment === "." || segment === ".."))) {
    invalid(path, "virtual path must be normalized and must not contain empty, . or .. segments");
  }
}

function nullableBoundedString(value: JsonValue, path: string, maximumBytes: number): string | null {
  return value === null ? null : boundedString(value, path, maximumBytes);
}

function booleanValue(value: JsonValue, path: string): boolean {
  if (typeof value !== "boolean") invalid(path, "expected boolean");
  return value;
}

function enumValue<const T extends readonly string[]>(value: JsonValue, values: T, path: string): T[number] {
  const text = stringValue(value, path);
  if (!values.includes(text)) invalid(path, `expected one of ${values.join(", ")}`);
  return text as T[number];
}

function nonnegativeInteger(value: JsonValue, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) invalid(path, "expected non-negative safe integer");
  return value;
}

function boundedNonnegativeInteger(value: JsonValue, path: string, maximum: number): number {
  const integer = nonnegativeInteger(value, path);
  if (integer > maximum) invalid(path, `expected non-negative safe integer no greater than ${maximum}`);
  return integer;
}

function positiveInteger(value: JsonValue, path: string, maximum: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    invalid(path, `expected positive safe integer no greater than ${maximum}`);
  }
  return value;
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT", path, message);
}

function duplicate(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-ARTIFACT-DUPLICATE-ID", path, message);
}

function invalid(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID", path, message);
}

export function cppCuteFrontendArtifactFailure(
  code: CppCuteFrontendArtifactErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteFrontendArtifactError(code, path, message, options);
}

function fail(code: CppCuteFrontendArtifactErrorCode, path: string, message: string): never {
  cppCuteFrontendArtifactFailure(code, path, message);
}
