import {
  hashCanonicalJson,
  wireIntegerToBigInt,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  cppCuteFrontendArtifactFailure,
  type CppCuteFrontendArtifactErrorCode,
} from "./cpp_cute_frontend_parse.js";
import {
  CppCuteIntegerSemanticsError,
  evaluateStaticCppCuteIntegerExpr,
  evaluateStaticCppCuteLayoutSummary,
} from "./cpp_cute_integer_semantics.js";
import type { DecodeLimits } from "@unlocalhosted/browsergrad-semantic-core/schema";
import type {
  CppCuteConstantV1,
  CppCuteDeclarationV2,
  CppCuteExpressionV1,
  CppCuteFrontendPayloadV2,
  CppCuteHierarchyV1,
  CppCuteIntegerExprV1,
  CppCuteResolvedFactV1,
  CppCuteResolvedTypeV1,
  CppCuteSourceOriginV1,
  CppCuteStatementV1,
  CppCuteTemplateArgumentV1,
} from "./cpp_cute_frontend_types.js";

const MAX_MACRO_DEPTH = 128;
const MAX_TEMPLATE_DEPTH = 128;

export interface VerifiedCppCuteInputHashes {
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly closureSha256: string;
}

export async function verifyCppCuteFrontendPayload(
  payload: CppCuteFrontendPayloadV2,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<VerifiedCppCuteInputHashes> {
  const indexes = buildIndexes(payload);
  verifyInputClosure(payload, indexes);
  verifySpans(payload, indexes);
  verifyMacroExpansions(payload, indexes);
  verifyTypes(payload, indexes);
  verifyConstants(payload, indexes);
  verifyDeclarations(payload, indexes);
  verifyTemplateInstantiations(payload, indexes);
  verifyOverloadResolutions(payload, indexes);
  verifySourceAbi(payload, indexes);
  verifyDeclarationInitializers(payload, indexes);
  verifyFunctionBodies(payload, indexes);
  verifyFacts(payload, indexes, options.limits);
  verifyEntries(payload, indexes);
  verifyDiagnosticsAndOutcome(payload, indexes);

  const hashes = await computeCppCuteInputHashes(payload);
  if (payload.inputs.sourceSetSha256 !== hashes.sourceSetSha256) {
    mismatch("$.payload.inputs.sourceSetSha256", "source-set hash does not match normalized main-source records");
  }
  if (payload.inputs.headerSetSha256 !== hashes.headerSetSha256) {
    mismatch("$.payload.inputs.headerSetSha256", "header-set hash does not match normalized header records");
  }
  if (payload.inputs.closureSha256 !== hashes.closureSha256) {
    mismatch("$.payload.inputs.closureSha256", "input-closure hash does not match normalized files, roots, and include edges");
  }
  if (payload.extraction.profileHash !== payload.profileHash) {
    mismatch("$.payload.extraction.profileHash", "extraction profile hash does not match artifact profileHash");
  }
  if (payload.extraction.inputClosureSha256 !== hashes.closureSha256) {
    mismatch("$.payload.extraction.inputClosureSha256", "extraction record does not bind the verified input closure");
  }
  return Object.freeze(hashes);
}

interface ArtifactIndexes {
  readonly files: ReadonlyMap<string, CppCuteFrontendPayloadV2["inputs"]["files"][number]>;
  readonly includeRoots: ReadonlyMap<string, CppCuteFrontendPayloadV2["inputs"]["includeRoots"][number]>;
  readonly spans: ReadonlyMap<string, CppCuteFrontendPayloadV2["spans"][number]>;
  readonly macros: ReadonlyMap<string, CppCuteFrontendPayloadV2["macroExpansions"][number]>;
  readonly types: ReadonlyMap<string, CppCuteResolvedTypeV1>;
  readonly constants: ReadonlyMap<string, CppCuteConstantV1>;
  readonly declarations: ReadonlyMap<string, CppCuteDeclarationV2>;
  readonly instantiations: ReadonlyMap<string, CppCuteFrontendPayloadV2["templateInstantiations"][number]>;
  readonly resolutions: ReadonlyMap<string, CppCuteFrontendPayloadV2["overloadResolutions"][number]>;
  readonly bodies: ReadonlyMap<string, CppCuteFrontendPayloadV2["functionBodies"][number]>;
  readonly statements: ReadonlyMap<string, CppCuteStatementV1>;
  readonly expressions: ReadonlyMap<string, CppCuteExpressionV1>;
  readonly facts: ReadonlyMap<string, CppCuteResolvedFactV1>;
  readonly entries: ReadonlyMap<string, CppCuteFrontendPayloadV2["entries"][number]>;
  readonly diagnostics: ReadonlyMap<string, CppCuteFrontendPayloadV2["diagnostics"][number]>;
}

function buildIndexes(payload: CppCuteFrontendPayloadV2): ArtifactIndexes {
  const statements = new Map<string, CppCuteStatementV1>();
  const expressions = new Map<string, CppCuteExpressionV1>();
  for (const [expressionIndex, expression] of payload.initializerExpressions.entries()) {
    addUnique(
      expressions,
      expression.expressionId,
      expression,
      `$.payload.initializerExpressions[${expressionIndex}]`,
    );
  }
  for (const [bodyIndex, body] of payload.functionBodies.entries()) {
    for (const statement of body.statements) addUnique(statements, statement.statementId, statement, `$.payload.functionBodies[${bodyIndex}].statements`);
    for (const expression of body.expressions) addUnique(expressions, expression.expressionId, expression, `$.payload.functionBodies[${bodyIndex}].expressions`);
  }
  return {
    files: mapBy(payload.inputs.files, (entry) => entry.fileId),
    includeRoots: mapByChecked(payload.inputs.includeRoots, (entry) => entry.includeRootId, "$.payload.inputs.includeRoots"),
    spans: mapBy(payload.spans, (entry) => entry.spanId),
    macros: mapBy(payload.macroExpansions, (entry) => entry.macroExpansionId),
    types: mapBy(payload.types, (entry) => entry.typeId),
    constants: mapBy(payload.constants, (entry) => entry.constantId),
    declarations: mapBy(payload.declarations, (entry) => entry.declarationId),
    instantiations: mapBy(payload.templateInstantiations, (entry) => entry.instantiationId),
    resolutions: mapBy(payload.overloadResolutions, (entry) => entry.resolutionId),
    bodies: mapBy(payload.functionBodies, (entry) => entry.bodyId),
    statements,
    expressions,
    facts: mapBy(payload.facts, (entry) => entry.factId),
    entries: mapBy(payload.entries, (entry) => entry.entryId),
    diagnostics: mapBy(payload.diagnostics, (entry) => entry.diagnosticId),
  };
}

function verifyInputClosure(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  const roots = payload.inputs.includeRoots;
  for (const [index, root] of roots.entries()) {
    if (root.ordinal !== index) invalid(`$.payload.inputs.includeRoots[${index}].ordinal`, "include-root ordinals must be contiguous and match search precedence");
  }
  uniqueValues(roots.map((root) => root.virtualPath), "$.payload.inputs.includeRoots", "virtual include-root path");
  uniqueValues(payload.inputs.files.map((file) => file.virtualPath), "$.payload.inputs.files", "virtual file path");

  const mainFiles = payload.inputs.files.filter((file) => file.role === "main-source");
  if (mainFiles.length !== 1 || mainFiles[0]?.fileId !== payload.inputs.mainFileId) {
    invalid("$.payload.inputs.mainFileId", "one-TU artifact requires exactly one main-source record matching mainFileId");
  }
  for (const [index, file] of payload.inputs.files.entries()) {
    const path = `$.payload.inputs.files[${index}]`;
    const expectedOwner = file.role === "system-header" || file.role === "dependency-header"
      ? "dependency"
      : file.role === "compiler-header"
        ? "compiler-resource-directory"
        : "source";
    if (file.owner.kind !== expectedOwner) {
      invalid(`${path}.owner`, `${file.role} requires ${expectedOwner} ownership`);
    }
    if (file.role === "main-source") {
      if (file.includeRootId !== null) invalid(`${path}.includeRootId`, "main source cannot belong to an include root");
      continue;
    }
    if (file.includeRootId === null) invalid(`${path}.includeRootId`, `${file.role} requires an include root`);
    const root = ref(indexes.includeRoots, file.includeRootId, `${path}.includeRootId`, "include root");
    if (!sameInputOwner(file.owner, root.owner)) {
      invalid(`${path}.owner`, "file owner must exactly match its bound include-root owner");
    }
    if (!virtualPathContains(root.virtualPath, file.virtualPath)) {
      invalid(`${path}.virtualPath`, "file path must be contained by its bound include root");
    }
  }

  const reachable = new Set<string>([payload.inputs.mainFileId]);
  for (const edge of payload.inputs.includeEdges) {
    if (edge.kind === "compiler-forced") reachable.add(edge.fileId);
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of payload.inputs.includeEdges) {
      if (edge.kind !== "source-directive" || !reachable.has(edge.includingFileId) || edge.resolution.kind !== "resolved") {
        continue;
      }
      if (!reachable.has(edge.resolution.fileId)) {
        reachable.add(edge.resolution.fileId);
        changed = true;
      }
    }
  }

  for (const [index, edge] of payload.inputs.includeEdges.entries()) {
    const path = `$.payload.inputs.includeEdges[${index}]`;
    if (edge.kind === "compiler-forced") {
      const file = ref(indexes.files, edge.fileId, `${path}.fileId`, "file");
      ref(indexes.includeRoots, edge.includeRootId, `${path}.includeRootId`, "include root");
      if (file.includeRootId !== edge.includeRootId) {
        invalid(`${path}.includeRootId`, "forced include root must match the included file binding");
      }
      continue;
    }
    ref(indexes.files, edge.includingFileId, `${path}.includingFileId`, "file");
    const directive = ref(indexes.spans, edge.directiveSpanId, `${path}.directiveSpanId`, "span");
    if (directive.spelling.fileId !== edge.includingFileId) {
      invalid(`${path}.directiveSpanId`, "include directive spelling range must belong to includingFileId");
    }
    if (edge.resolution.kind === "resolved") {
      const file = ref(indexes.files, edge.resolution.fileId, `${path}.resolution.fileId`, "file");
      ref(indexes.includeRoots, edge.resolution.includeRootId, `${path}.resolution.includeRootId`, "include root");
      if (file.includeRootId !== edge.resolution.includeRootId) {
        invalid(`${path}.resolution.includeRootId`, "resolution root must match the included file binding");
      }
    }
  }
  for (const [index, file] of payload.inputs.files.entries()) {
    if (!reachable.has(file.fileId)) invalid(`$.payload.inputs.files[${index}]`, "input closure contains an unreachable file");
  }
}

function sameInputOwner(
  left: CppCuteFrontendPayloadV2["inputs"]["files"][number]["owner"],
  right: CppCuteFrontendPayloadV2["inputs"]["includeRoots"][number]["owner"],
): boolean {
  return left.kind === right.kind && (
    left.kind !== "dependency" || (right.kind === "dependency" && left.dependencyId === right.dependencyId)
  );
}

function virtualPathContains(root: string, candidate: string): boolean {
  return root === "/" ? candidate.startsWith("/") && candidate !== "/" : candidate.startsWith(`${root}/`);
}

function verifySpans(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, span] of payload.spans.entries()) {
    const path = `$.payload.spans[${index}]`;
    verifyRange(span.spelling, `${path}.spelling`, indexes);
    verifyRange(span.expansion, `${path}.expansion`, indexes);
    if (span.macroExpansionId === null) {
      if (!sameRange(span.spelling, span.expansion)) {
        invalid(path, "nonmacro span must have identical spelling and expansion ranges");
      }
    } else {
      ref(indexes.macros, span.macroExpansionId, `${path}.macroExpansionId`, "macro expansion");
    }
  }
}

function verifyRange(
  range: CppCuteFrontendPayloadV2["spans"][number]["spelling"],
  path: string,
  indexes: ArtifactIndexes,
): void {
  const file = ref(indexes.files, range.fileId, `${path}.fileId`, "file");
  const start = wireIntegerToBigInt(range.startByte);
  const end = wireIntegerToBigInt(range.endByte);
  const length = wireIntegerToBigInt(file.byteLength);
  if (start > end || end > length) invalid(path, "source range must be a bounded half-open byte range within its file");
}

function verifyMacroExpansions(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, macro] of payload.macroExpansions.entries()) {
    const path = `$.payload.macroExpansions[${index}]`;
    ref(indexes.spans, macro.definitionSpanId, `${path}.definitionSpanId`, "span");
    ref(indexes.spans, macro.invocationSpanId, `${path}.invocationSpanId`, "span");
    if (macro.parentMacroExpansionId !== null) {
      ref(indexes.macros, macro.parentMacroExpansionId, `${path}.parentMacroExpansionId`, "macro expansion");
    }
  }
  verifyParentForest(
    payload.macroExpansions,
    (entry) => entry.macroExpansionId,
    (entry) => entry.parentMacroExpansionId,
    MAX_MACRO_DEPTH,
    "$.payload.macroExpansions",
    "macro expansion",
  );
}

function verifyTypes(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, type] of payload.types.entries()) {
    const path = `$.payload.types[${index}]`;
    verifyOrigin(type.origin, `${path}.origin`, indexes);
    if (type.kind === "pointer" || type.kind === "lvalue-reference" || type.kind === "rvalue-reference") {
      ref(indexes.types, type.pointeeTypeId, `${path}.pointeeTypeId`, "type");
    } else if (type.kind === "array") {
      ref(indexes.types, type.elementTypeId, `${path}.elementTypeId`, "type");
      if (wireIntegerToBigInt(type.elementCount) === 0n) invalid(`${path}.elementCount`, "resolved array extent must be positive");
    } else if (type.kind === "vector") {
      ref(indexes.types, type.elementTypeId, `${path}.elementTypeId`, "type");
    } else if (type.kind === "function") {
      ref(indexes.types, type.returnTypeId, `${path}.returnTypeId`, "type");
      type.parameterTypeIds.forEach((typeId, parameterIndex) =>
        ref(indexes.types, typeId, `${path}.parameterTypeIds[${parameterIndex}]`, "type"));
    } else if (type.kind === "record" || type.kind === "enum") {
      ref(indexes.declarations, type.declarationId, `${path}.declarationId`, "declaration");
    } else if (type.kind === "template-specialization") {
      ref(indexes.declarations, type.templateDeclarationId, `${path}.templateDeclarationId`, "declaration");
      type.arguments.forEach((argument, argumentIndex) => verifyTemplateArgument(argument, `${path}.arguments[${argumentIndex}]`, indexes));
    }
  }
}

function verifyConstants(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, constant] of payload.constants.entries()) {
    const path = `$.payload.constants[${index}]`;
    ref(indexes.types, constant.typeId, `${path}.typeId`, "type");
    verifyOrigin(constant.origin, `${path}.origin`, indexes);
    if (constant.kind === "signed-integer") verifySignedBitWidth(constant.value, constant.bitWidth, `${path}.value`);
    if (constant.kind === "unsigned-integer") verifyUnsignedBitWidth(constant.value, constant.bitWidth, `${path}.value`);
    if (constant.kind === "enum") {
      ref(indexes.declarations, constant.enumDeclarationId, `${path}.enumDeclarationId`, "declaration");
      ref(indexes.constants, constant.valueConstantId, `${path}.valueConstantId`, "constant");
    }
    if (constant.kind === "aggregate") {
      constant.elementConstantIds.forEach((id, elementIndex) =>
        ref(indexes.constants, id, `${path}.elementConstantIds[${elementIndex}]`, "constant"));
    }
  }
}

function verifyDeclarations(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  uniqueValues(payload.declarations.map((declaration) => declaration.canonicalUsr), "$.payload.declarations", "canonical USR");
  for (const [index, declaration] of payload.declarations.entries()) {
    const path = `$.payload.declarations[${index}]`;
    verifyOrigin(declaration.origin, `${path}.origin`, indexes);
    if (declaration.lexicalParentId !== null) ref(indexes.declarations, declaration.lexicalParentId, `${path}.lexicalParentId`, "declaration");
    if (declaration.semanticParentId !== null) ref(indexes.declarations, declaration.semanticParentId, `${path}.semanticParentId`, "declaration");
    if (declaration.typeId !== null) ref(indexes.types, declaration.typeId, `${path}.typeId`, "type");
    if (declaration.targetTypeId !== null) ref(indexes.types, declaration.targetTypeId, `${path}.targetTypeId`, "type");
    if (declaration.initializerExpressionId !== null) {
      const initializer = ref(
        indexes.expressions,
        declaration.initializerExpressionId,
        `${path}.initializerExpressionId`,
        "expression",
      );
      if (declaration.kind !== "variable" || declaration.definitionKind !== "definition") {
        invalid(`${path}.initializerExpressionId`, "only variable definitions may carry an initializer expression");
      }
      if (initializer.typeId !== declaration.typeId) {
        invalid(`${path}.initializerExpressionId`, "initializer result type must match the variable type");
      }
    }

    if (declaration.kind === "namespace") {
      if (declaration.typeId !== null || declaration.targetTypeId !== null) invalid(path, "namespace declaration cannot carry a type");
    } else if (declaration.kind === "type-alias") {
      if (declaration.targetTypeId === null) invalid(`${path}.targetTypeId`, "type alias requires targetTypeId");
    } else if (declaration.typeId === null) {
      invalid(`${path}.typeId`, `${declaration.kind} declaration requires a resolved type`);
    }
    if (declaration.kind !== "type-alias" && declaration.targetTypeId !== null) {
      invalid(`${path}.targetTypeId`, "only type-alias declarations may carry targetTypeId");
    }
    if (declaration.cudaAttributes.global) {
      if (declaration.kind !== "function" || !declaration.cudaAttributes.device) {
        invalid(`${path}.cudaAttributes.global`, "CUDA global attribute requires a device function declaration");
      }
      const type = declaration.typeId === null ? undefined : indexes.types.get(declaration.typeId);
      if (type?.kind !== "function" || type.callingConvention !== "cuda-kernel") {
        invalid(`${path}.typeId`, "CUDA global function must use cuda-kernel calling convention");
      }
    }
    if (declaration.kind === "field" && declaration.semanticParentId !== null) {
      const parent = indexes.declarations.get(declaration.semanticParentId);
      if (parent?.kind !== "record") invalid(`${path}.semanticParentId`, "field semantic parent must be a record");
    }
    if (declaration.kind === "parameter" && declaration.semanticParentId !== null) {
      const parent = indexes.declarations.get(declaration.semanticParentId);
      if (parent?.kind !== "function" && parent?.kind !== "template") {
        invalid(`${path}.semanticParentId`, "parameter semantic parent must be a function or template");
      }
    }
  }
  verifyParentForest(
    payload.declarations,
    (entry) => entry.declarationId,
    (entry) => entry.lexicalParentId,
    payload.declarations.length + 1,
    "$.payload.declarations",
    "lexical declaration",
  );
  verifyParentForest(
    payload.declarations,
    (entry) => entry.declarationId,
    (entry) => entry.semanticParentId,
    payload.declarations.length + 1,
    "$.payload.declarations",
    "semantic declaration",
  );
}

function verifyDeclarationInitializers(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  const globalExpressions = new Map(payload.initializerExpressions.map((entry) => [entry.expressionId, entry]));
  const bodyByFunction = new Map(payload.functionBodies.map((body) => [body.declarationId, body]));
  const globalParents = new Map<string, number>();
  for (const [index, declaration] of payload.declarations.entries()) {
    if (declaration.initializerExpressionId === null) continue;
    const path = `$.payload.declarations[${index}].initializerExpressionId`;
    const enclosingFunctionId = findEnclosingFunctionId(declaration, indexes);
    if (enclosingFunctionId === null) {
      ref(globalExpressions, declaration.initializerExpressionId, path, "file-scope initializer expression");
      globalParents.set(
        declaration.initializerExpressionId,
        (globalParents.get(declaration.initializerExpressionId) ?? 0) + 1,
      );
      verifyExpressionTree(
        declaration.initializerExpressionId,
        globalExpressions,
        indexes,
        globalParents,
        path,
        new Set(),
      );
    } else {
      const body = bodyByFunction.get(enclosingFunctionId);
      if (body === undefined || !body.expressions.some((entry) => entry.expressionId === declaration.initializerExpressionId)) {
        invalid(path, "local variable initializer must belong to its enclosing function body");
      }
    }
  }
  if (globalParents.size !== globalExpressions.size) {
    invalid("$.payload.initializerExpressions", "file-scope initializer pool contains unreachable expressions");
  }
  for (const [id, count] of globalParents) {
    if (count !== 1) {
      invalid("$.payload.initializerExpressions", `expression ${id} has ${count} owners; exactly one is required`);
    }
  }
}

function findEnclosingFunctionId(
  declaration: CppCuteDeclarationV2,
  indexes: ArtifactIndexes,
): string | null {
  let parentId = declaration.semanticParentId;
  const seen = new Set<string>();
  while (parentId !== null) {
    if (seen.has(parentId)) return null;
    seen.add(parentId);
    const parent = indexes.declarations.get(parentId);
    if (parent === undefined) return null;
    if (parent.kind === "function") return parent.declarationId;
    parentId = parent.semanticParentId;
  }
  return null;
}

function verifyTemplateInstantiations(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, instantiation] of payload.templateInstantiations.entries()) {
    const path = `$.payload.templateInstantiations[${index}]`;
    const template = ref(indexes.declarations, instantiation.templateDeclarationId, `${path}.templateDeclarationId`, "declaration");
    if (template.kind !== "template") invalid(`${path}.templateDeclarationId`, "template instantiation must reference a template declaration");
    ref(indexes.declarations, instantiation.specializationDeclarationId, `${path}.specializationDeclarationId`, "declaration");
    ref(indexes.spans, instantiation.pointOfInstantiationSpanId, `${path}.pointOfInstantiationSpanId`, "span");
    instantiation.arguments.forEach((argument, argumentIndex) =>
      verifyTemplateArgument(argument, `${path}.arguments[${argumentIndex}]`, indexes));
    if (instantiation.depth > MAX_TEMPLATE_DEPTH) resource(`${path}.depth`, `template instantiation depth exceeds ${MAX_TEMPLATE_DEPTH}`);
    if (instantiation.parentInstantiationId === null && instantiation.depth !== 0) {
      invalid(`${path}.depth`, "root template instantiation depth must be zero");
    }
    if (instantiation.parentInstantiationId !== null) {
      const parent = ref(indexes.instantiations, instantiation.parentInstantiationId, `${path}.parentInstantiationId`, "template instantiation");
      if (instantiation.depth !== parent.depth + 1) invalid(`${path}.depth`, "template instantiation depth must equal parent depth plus one");
    }
  }
  verifyParentForest(
    payload.templateInstantiations,
    (entry) => entry.instantiationId,
    (entry) => entry.parentInstantiationId,
    MAX_TEMPLATE_DEPTH + 1,
    "$.payload.templateInstantiations",
    "template instantiation",
  );
}

function verifyOverloadResolutions(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, resolution] of payload.overloadResolutions.entries()) {
    const path = `$.payload.overloadResolutions[${index}]`;
    verifyOrigin(resolution.origin, `${path}.origin`, indexes);
    const selected = ref(indexes.declarations, resolution.selectedDeclarationId, `${path}.selectedDeclarationId`, "declaration");
    if (selected.kind !== "function") invalid(`${path}.selectedDeclarationId`, "selected overload must be a function declaration");
    if (!resolution.candidateDeclarationIds.includes(resolution.selectedDeclarationId)) {
      invalid(`${path}.candidateDeclarationIds`, "selected declaration must be present in overload candidates");
    }
    resolution.candidateDeclarationIds.forEach((id, candidateIndex) => {
      const candidate = ref(indexes.declarations, id, `${path}.candidateDeclarationIds[${candidateIndex}]`, "declaration");
      if (candidate.kind !== "function") invalid(`${path}.candidateDeclarationIds[${candidateIndex}]`, "overload candidate must be a function");
    });
    resolution.argumentTypeIds.forEach((id, argumentIndex) => ref(indexes.types, id, `${path}.argumentTypeIds[${argumentIndex}]`, "type"));
    ref(indexes.types, resolution.resultTypeId, `${path}.resultTypeId`, "type");
  }
}

function verifySourceAbi(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  const typeAbi = new Map(payload.sourceAbi.types.map((entry) => [entry.typeId, entry]));
  for (const [index, abi] of payload.sourceAbi.types.entries()) {
    const path = `$.payload.sourceAbi.types[${index}]`;
    const type = ref(indexes.types, abi.typeId, `${path}.typeId`, "type");
    const size = wireIntegerToBigInt(abi.sizeBits);
    const alignment = wireIntegerToBigInt(abi.alignmentBits);
    if (size === 0n || alignment === 0n || (alignment & (alignment - 1n)) !== 0n) {
      invalid(path, "complete ABI type requires positive size and power-of-two alignment in bits");
    }
    if (type.kind === "record" && !type.complete) invalid(`${path}.typeId`, "incomplete record cannot have a type-layout ABI record");
    for (const [fieldIndex, field] of abi.fields.entries()) {
      const fieldPath = `${path}.fields[${fieldIndex}]`;
      const declaration = ref(indexes.declarations, field.declarationId, `${fieldPath}.declarationId`, "declaration");
      if (declaration.kind !== "field" || declaration.typeId !== field.typeId) invalid(fieldPath, "ABI field must match its resolved field declaration and type");
      ref(indexes.types, field.typeId, `${fieldPath}.typeId`, "type");
      const fieldAbi = typeAbi.get(field.typeId);
      const end = wireIntegerToBigInt(field.bitOffset) + (fieldAbi === undefined ? 1n : wireIntegerToBigInt(fieldAbi.sizeBits));
      if (end > size) invalid(`${fieldPath}.bitOffset`, "ABI field extends beyond record size");
    }
    for (const [baseIndex, base] of abi.bases.entries()) {
      const basePath = `${path}.bases[${baseIndex}]`;
      ref(indexes.types, base.typeId, `${basePath}.typeId`, "type");
      const baseAbi = typeAbi.get(base.typeId);
      const end = wireIntegerToBigInt(base.bitOffset) + (baseAbi === undefined ? 1n : wireIntegerToBigInt(baseAbi.sizeBits));
      if (end > size) invalid(`${basePath}.bitOffset`, "ABI base extends beyond record size");
    }
  }
  verifyValueContainmentCycles(payload, typeAbi);

  for (const [index, abi] of payload.sourceAbi.functions.entries()) {
    const path = `$.payload.sourceAbi.functions[${index}]`;
    const declaration = ref(indexes.declarations, abi.declarationId, `${path}.declarationId`, "declaration");
    if (declaration.kind !== "function") invalid(`${path}.declarationId`, "function ABI must reference a function declaration");
    const functionType = declaration.typeId === null ? undefined : indexes.types.get(declaration.typeId);
    if (functionType?.kind !== "function") invalid(`${path}.declarationId`, "function declaration must reference a resolved function type");
    if (functionType.callingConvention !== abi.callingConvention || functionType.returnTypeId !== abi.returnTypeId) {
      invalid(path, "function ABI calling convention or return type differs from resolved function type");
    }
    if (functionType.parameterTypeIds.length !== abi.parameters.length) invalid(`${path}.parameters`, "function ABI parameter count differs from function type");
    abi.parameters.forEach((parameter, parameterIndex) => {
      const parameterPath = `${path}.parameters[${parameterIndex}]`;
      const parameterDeclaration = ref(indexes.declarations, parameter.declarationId, `${parameterPath}.declarationId`, "declaration");
      if (parameterDeclaration.kind !== "parameter" || parameterDeclaration.semanticParentId !== abi.declarationId) {
        invalid(`${parameterPath}.declarationId`, "ABI parameter must be an ordered parameter of the function declaration");
      }
      if (parameterDeclaration.typeId !== parameter.typeId || functionType.parameterTypeIds[parameterIndex] !== parameter.typeId) {
        invalid(`${parameterPath}.typeId`, "ABI parameter type differs from declaration or function type");
      }
    });
  }
}

function verifyFunctionBodies(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  const declarationsWithBodies = new Set<string>();
  for (const [bodyIndex, body] of payload.functionBodies.entries()) {
    const path = `$.payload.functionBodies[${bodyIndex}]`;
    const declaration = ref(indexes.declarations, body.declarationId, `${path}.declarationId`, "declaration");
    if (declaration.kind !== "function" || declaration.definitionKind !== "definition") {
      invalid(`${path}.declarationId`, "function body requires a defined function declaration");
    }
    if (declarationsWithBodies.has(body.declarationId)) invalid(`${path}.declarationId`, "function declaration has multiple bodies");
    declarationsWithBodies.add(body.declarationId);
    const statementMap = new Map(body.statements.map((entry) => [entry.statementId, entry]));
    const expressionMap = new Map(body.expressions.map((entry) => [entry.expressionId, entry]));
    ref(statementMap, body.rootStatementId, `${path}.rootStatementId`, "statement");
    verifyStatementTree(body.rootStatementId, statementMap, expressionMap, indexes, path);
  }
  for (const [index, declaration] of payload.declarations.entries()) {
    if (declaration.kind === "function" && declaration.definitionKind === "definition" && !declarationsWithBodies.has(declaration.declarationId)) {
      invalid(`$.payload.declarations[${index}].definitionKind`, "defined selected declaration requires one serialized function body");
    }
  }
}

function verifyStatementTree(
  rootId: string,
  statements: ReadonlyMap<string, CppCuteStatementV1>,
  expressions: ReadonlyMap<string, CppCuteExpressionV1>,
  indexes: ArtifactIndexes,
  bodyPath: string,
): void {
  const seenStatements = new Set<string>();
  const activeStatements = new Set<string>();
  const expressionParents = new Map<string, number>();
  const visitStatement = (id: string, loopDepth: number, path: string): void => {
    const statement = ref(statements, id, path, "statement");
    if (activeStatements.has(id)) invalid(path, "structured statement graph contains a cycle");
    if (seenStatements.has(id)) invalid(path, "statement belongs to more than one structured parent");
    seenStatements.add(id);
    activeStatements.add(id);
    verifyOrigin(statement.origin, `${path}.origin`, indexes);
    const expressionRoot = (expressionId: string | null, expressionPath: string): void => {
      if (expressionId === null) return;
      expressionParents.set(expressionId, (expressionParents.get(expressionId) ?? 0) + 1);
      verifyExpressionTree(expressionId, expressions, indexes, expressionParents, expressionPath, new Set());
    };
    if (statement.kind === "block") {
      statement.statementIds.forEach((child, index) => visitStatement(child, loopDepth, `${path}.statementIds[${index}]`));
    } else if (statement.kind === "declaration") {
      const declaration = ref(indexes.declarations, statement.declarationId, `${path}.declarationId`, "declaration");
      expressionRoot(declaration.initializerExpressionId, `${path}.declarationId.initializerExpressionId`);
    } else if (statement.kind === "expression") {
      expressionRoot(statement.expressionId, `${path}.expressionId`);
    } else if (statement.kind === "if") {
      expressionRoot(statement.conditionExpressionId, `${path}.conditionExpressionId`);
      visitStatement(statement.thenStatementId, loopDepth, `${path}.thenStatementId`);
      if (statement.elseStatementId !== null) visitStatement(statement.elseStatementId, loopDepth, `${path}.elseStatementId`);
    } else if (statement.kind === "for") {
      if (statement.initializerStatementId !== null) visitStatement(statement.initializerStatementId, loopDepth, `${path}.initializerStatementId`);
      expressionRoot(statement.conditionExpressionId, `${path}.conditionExpressionId`);
      expressionRoot(statement.incrementExpressionId, `${path}.incrementExpressionId`);
      visitStatement(statement.bodyStatementId, loopDepth + 1, `${path}.bodyStatementId`);
    } else if (statement.kind === "while") {
      expressionRoot(statement.conditionExpressionId, `${path}.conditionExpressionId`);
      visitStatement(statement.bodyStatementId, loopDepth + 1, `${path}.bodyStatementId`);
    } else if (statement.kind === "return") {
      expressionRoot(statement.expressionId, `${path}.expressionId`);
    } else if ((statement.kind === "break" || statement.kind === "continue") && loopDepth === 0) {
      invalid(path, `${statement.kind} statement must be nested in a loop`);
    }
    activeStatements.delete(id);
  };
  visitStatement(rootId, 0, `${bodyPath}.rootStatementId`);
  if (seenStatements.size !== statements.size) invalid(`${bodyPath}.statements`, "function body contains unreachable statements");
  if (expressionParents.size !== expressions.size) invalid(`${bodyPath}.expressions`, "function body contains unreachable expressions");
  for (const [id, count] of expressionParents) {
    if (count !== 1) invalid(`${bodyPath}.expressions`, `expression ${id} has ${count} owners; exactly one is required`);
  }
}

function verifyExpressionTree(
  id: string,
  expressions: ReadonlyMap<string, CppCuteExpressionV1>,
  indexes: ArtifactIndexes,
  parents: Map<string, number>,
  path: string,
  active: Set<string>,
): void {
  const expression = ref(expressions, id, path, "expression");
  if (active.has(id)) invalid(path, "expression graph contains a cycle");
  active.add(id);
  ref(indexes.types, expression.typeId, `${path}.typeId`, "type");
  verifyOrigin(expression.origin, `${path}.origin`, indexes);
  const child = (childId: string, childPath: string): CppCuteExpressionV1 => {
    parents.set(childId, (parents.get(childId) ?? 0) + 1);
    verifyExpressionTree(childId, expressions, indexes, parents, childPath, active);
    return ref(expressions, childId, childPath, "expression");
  };
  if (expression.kind === "constant") {
    const constant = ref(indexes.constants, expression.constantId, `${path}.constantId`, "constant");
    if (constant.typeId !== expression.typeId) invalid(`${path}.typeId`, "constant expression type differs from referenced constant");
  } else if (expression.kind === "declaration-reference") {
    const declaration = ref(indexes.declarations, expression.declarationId, `${path}.declarationId`, "declaration");
    if (declaration.typeId !== expression.typeId) invalid(`${path}.typeId`, "declaration-reference type differs from declaration");
  } else if (expression.kind === "resolved-call") {
    const resolution = ref(indexes.resolutions, expression.overloadResolutionId, `${path}.overloadResolutionId`, "overload resolution");
    if (resolution.resultTypeId !== expression.typeId || resolution.argumentTypeIds.length !== expression.argumentExpressionIds.length) {
      invalid(path, "resolved-call result or argument count differs from overload resolution");
    }
    expression.argumentExpressionIds.forEach((argumentId, index) => {
      const argument = child(argumentId, `${path}.argumentExpressionIds[${index}]`);
      if (argument.typeId !== resolution.argumentTypeIds[index]) invalid(`${path}.argumentExpressionIds[${index}]`, "call argument type differs from overload resolution");
    });
  } else if (expression.kind === "construction") {
    ref(indexes.declarations, expression.constructorDeclarationId, `${path}.constructorDeclarationId`, "declaration");
    expression.argumentExpressionIds.forEach((argumentId, index) => child(argumentId, `${path}.argumentExpressionIds[${index}]`));
  } else if (expression.kind === "member-access") {
    child(expression.baseExpressionId, `${path}.baseExpressionId`);
    const member = ref(indexes.declarations, expression.memberDeclarationId, `${path}.memberDeclarationId`, "declaration");
    if (member.kind !== "field" || member.typeId !== expression.typeId) invalid(`${path}.memberDeclarationId`, "member access must resolve to a field with matching type");
  } else if (expression.kind === "subscript") {
    child(expression.baseExpressionId, `${path}.baseExpressionId`);
    child(expression.indexExpressionId, `${path}.indexExpressionId`);
  } else if (expression.kind === "cast" || expression.kind === "unary") {
    child(expression.operandExpressionId, `${path}.operandExpressionId`);
  } else if (expression.kind === "binary") {
    child(expression.leftExpressionId, `${path}.leftExpressionId`);
    child(expression.rightExpressionId, `${path}.rightExpressionId`);
  } else if (expression.kind === "conditional") {
    child(expression.conditionExpressionId, `${path}.conditionExpressionId`);
    const thenExpression = child(expression.thenExpressionId, `${path}.thenExpressionId`);
    const elseExpression = child(expression.elseExpressionId, `${path}.elseExpressionId`);
    if (thenExpression.typeId !== expression.typeId || elseExpression.typeId !== expression.typeId) {
      invalid(path, "conditional branches must match result type");
    }
  } else if (expression.kind === "target-intrinsic") {
    const fact = ref(indexes.facts, expression.intrinsicFactId, `${path}.intrinsicFactId`, "fact");
    if (fact.kind !== "target-intrinsic" || fact.resultTypeId !== expression.typeId) {
      invalid(`${path}.intrinsicFactId`, "target-intrinsic expression must reference a matching intrinsic fact");
    }
  }
  active.delete(id);
}

function verifyFacts(
  payload: CppCuteFrontendPayloadV2,
  indexes: ArtifactIndexes,
  limits: Partial<DecodeLimits> | undefined,
): void {
  for (const [index, fact] of payload.facts.entries()) {
    const path = `$.payload.facts[${index}]`;
    verifyOrigin(fact.origin, `${path}.origin`, indexes);
    if (fact.kind === "affine-layout") verifyAffineLayoutFact(fact, path, indexes, limits);
    else if (fact.kind === "tensor") verifyTensorFact(fact, path, indexes);
    else verifyTargetIntrinsicFact(fact, path, indexes);
  }
}

function verifyAffineLayoutFact(
  fact: Extract<CppCuteResolvedFactV1, { readonly kind: "affine-layout" }>,
  path: string,
  indexes: ArtifactIndexes,
  limits: Partial<DecodeLimits> | undefined,
): void {
  ref(indexes.declarations, fact.resultDeclarationId, `${path}.resultDeclarationId`, "declaration");
  verifyHierarchyRefs(fact.shape, `${path}.shape`, indexes, limits);
  verifyHierarchyRefs(fact.stride, `${path}.stride`, indexes, limits);
  verifyIntegerExprRefs(fact.size, `${path}.size`, indexes, limits);
  verifyIntegerExprRefs(fact.cosize, `${path}.cosize`, indexes, limits);
  if (!sameHierarchyTopology(fact.shape, fact.stride)) invalid(`${path}.stride`, "shape and stride hierarchy topology must match exactly");
  if (topRank(fact.shape) !== fact.rank) invalid(`${path}.rank`, "rank does not match top-level shape hierarchy");
  if (leafRank(fact.shape) !== fact.leafRank) invalid(`${path}.leafRank`, "leafRank does not match flattened shape hierarchy");

  const shapeLeaves = hierarchyLeaves(fact.shape).map((expression) => evaluateStaticIntegerExpr(expression, `${path}.shape`, limits));
  const strideLeaves = hierarchyLeaves(fact.stride).map((expression) => evaluateStaticIntegerExpr(expression, `${path}.stride`, limits));
  const size = evaluateStaticIntegerExpr(fact.size, `${path}.size`, limits);
  const cosize = evaluateStaticIntegerExpr(fact.cosize, `${path}.cosize`, limits);
  if ([...shapeLeaves, ...strideLeaves, size, cosize].every((value) => value !== undefined)) {
    const shapes = shapeLeaves as bigint[];
    const strides = strideLeaves as bigint[];
    if (shapes.some((extent) => extent <= 0n)) invalid(`${path}.shape`, "static CuTe shape extents must be positive");
    let summary;
    try {
      summary = evaluateStaticCppCuteLayoutSummary(shapes, strides, {
        path,
        ...(limits === undefined ? {} : { limits }),
      });
    } catch (error) {
      if (!(error instanceof CppCuteIntegerSemanticsError)) throw error;
      if (error.kind === "resource-limit") resource(error.path, error.message);
      invalid(error.path, error.message);
    }
    if (size !== summary.size) invalid(`${path}.size`, "static CuTe size does not equal product of shape leaves");
    if (cosize !== summary.cosize) invalid(`${path}.cosize`, "static CuTe cosize does not equal layout(size(layout) - 1) + 1");
  }
}

function verifyTensorFact(
  fact: Extract<CppCuteResolvedFactV1, { readonly kind: "tensor" }>,
  path: string,
  indexes: ArtifactIndexes,
): void {
  ref(indexes.declarations, fact.resultDeclarationId, `${path}.resultDeclarationId`, "declaration");
  ref(indexes.types, fact.elementTypeId, `${path}.elementTypeId`, "type");
  const layout = ref(indexes.facts, fact.layoutFactId, `${path}.layoutFactId`, "fact");
  if (layout.kind !== "affine-layout") invalid(`${path}.layoutFactId`, "tensor layoutFactId must reference an affine-layout fact");
  if (fact.engine.kind === "global-pointer" || fact.engine.kind === "shared-pointer") {
    ref(indexes.declarations, fact.engine.pointerDeclarationId, `${path}.engine.pointerDeclarationId`, "declaration");
  } else if (fact.engine.kind === "register-array") {
    ref(indexes.declarations, fact.engine.arrayDeclarationId, `${path}.engine.arrayDeclarationId`, "declaration");
  } else if (fact.engine.kind === "pointer-array") {
    fact.engine.pointerDeclarationIds.forEach((id, index) => ref(indexes.declarations, id, `${path}.engine.pointerDeclarationIds[${index}]`, "declaration"));
  } else {
    ref(indexes.declarations, fact.engine.engineDeclarationId, `${path}.engine.engineDeclarationId`, "declaration");
  }
  const expectedSpace = fact.engine.kind === "global-pointer" || fact.engine.kind === "pointer-array"
    ? "global"
    : fact.engine.kind === "shared-pointer"
      ? "shared"
      : fact.engine.kind === "register-array"
        ? "local"
        : "generic";
  if (fact.memorySpace !== expectedSpace) invalid(`${path}.memorySpace`, `tensor engine ${fact.engine.kind} requires ${expectedSpace} memory space`);
}

function verifyTargetIntrinsicFact(
  fact: Extract<CppCuteResolvedFactV1, { readonly kind: "target-intrinsic" }>,
  path: string,
  indexes: ArtifactIndexes,
): void {
  fact.operandExpressionIds.forEach((id, index) => ref(indexes.expressions, id, `${path}.operandExpressionIds[${index}]`, "expression"));
  if (fact.resultTypeId !== null) ref(indexes.types, fact.resultTypeId, `${path}.resultTypeId`, "type");
  if (fact.operation.kind === "mma") {
    ref(indexes.types, fact.operation.aTypeId, `${path}.operation.aTypeId`, "type");
    ref(indexes.types, fact.operation.bTypeId, `${path}.operation.bTypeId`, "type");
    ref(indexes.types, fact.operation.accumulatorTypeId, `${path}.operation.accumulatorTypeId`, "type");
  }
  if (fact.availability.kind === "recognized-unsupported") {
    const diagnostic = ref(indexes.diagnostics, fact.availability.diagnosticId, `${path}.availability.diagnosticId`, "diagnostic");
    if (diagnostic.severity === "error" || diagnostic.severity === "fatal") {
      invalid(`${path}.availability.diagnosticId`, "represented unsupported intrinsic must not become a frontend rejection");
    }
  }
}

function verifyEntries(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, entry] of payload.entries.entries()) {
    const path = `$.payload.entries[${index}]`;
    entry.selectedRootDeclarationIds.forEach((id, rootIndex) => ref(indexes.declarations, id, `${path}.selectedRootDeclarationIds[${rootIndex}]`, "declaration"));
    if (entry.selectedRootDeclarationIds.length === 0) invalid(`${path}.selectedRootDeclarationIds`, "frontend entry requires at least one selected root declaration");
    if (entry.kind === "layout") {
      const fact = ref(indexes.facts, entry.layoutFactId, `${path}.layoutFactId`, "fact");
      if (fact.kind !== "affine-layout") invalid(`${path}.layoutFactId`, "layout entry must reference an affine-layout fact");
      if (
        entry.selectedRootDeclarationIds.length !== 1
        || entry.selectedRootDeclarationIds[0] !== fact.resultDeclarationId
      ) {
        invalid(
          `${path}.selectedRootDeclarationIds`,
          "layout entry must select exactly its affine-layout result declaration",
        );
      }
    } else {
      const source = ref(indexes.facts, entry.sourceTensorFactId, `${path}.sourceTensorFactId`, "fact");
      const destination = ref(indexes.facts, entry.destinationTensorFactId, `${path}.destinationTensorFactId`, "fact");
      if (source.kind !== "tensor" || destination.kind !== "tensor") invalid(path, "view-copy entry must reference source and destination tensor facts");
      ref(indexes.expressions, entry.operationExpressionId, `${path}.operationExpressionId`, "expression");
    }
  }
}

function verifyDiagnosticsAndOutcome(payload: CppCuteFrontendPayloadV2, indexes: ArtifactIndexes): void {
  for (const [index, diagnostic] of payload.diagnostics.entries()) {
    const path = `$.payload.diagnostics[${index}]`;
    if (diagnostic.location.kind === "source") {
      ref(indexes.spans, diagnostic.location.primarySpanId, `${path}.location.primarySpanId`, "span");
      diagnostic.location.related.forEach((related, relatedIndex) => ref(
        indexes.spans,
        related.spanId,
        `${path}.location.related[${relatedIndex}].spanId`,
        "span",
      ));
    } else if (
      diagnostic.subject.kind !== "invocation" &&
      diagnostic.subject.kind !== "profile" &&
      diagnostic.subject.kind !== "compiler"
    ) {
      invalid(`${path}.location`, "locationless diagnostics require an invocation, profile, or compiler subject");
    }
    if (diagnostic.subject.kind === "invocation" && diagnostic.phase !== "invocation") {
      invalid(`${path}.phase`, "invocation subject requires invocation phase");
    }
    if (diagnostic.subject.kind === "profile" && diagnostic.phase !== "profile-validation") {
      invalid(`${path}.phase`, "profile subject requires profile-validation phase");
    }
    if (diagnostic.severity === "note") {
      if (diagnostic.parentDiagnosticId === null) invalid(`${path}.parentDiagnosticId`, "note diagnostic requires a parent");
    } else if (diagnostic.parentDiagnosticId !== null) {
      invalid(`${path}.parentDiagnosticId`, "only note diagnostics may have a parent");
    }
    if (diagnostic.parentDiagnosticId !== null) {
      const parent = ref(indexes.diagnostics, diagnostic.parentDiagnosticId, `${path}.parentDiagnosticId`, "diagnostic");
      if (parent.severity === "note") invalid(`${path}.parentDiagnosticId`, "note parent must be a root diagnostic");
    }
    verifyDiagnosticSubject(diagnostic.subject, `${path}.subject`, indexes);
  }

  const blocking = payload.diagnostics
    .filter((diagnostic) => (diagnostic.severity === "error" || diagnostic.severity === "fatal") && diagnostic.parentDiagnosticId === null)
    .map((diagnostic) => diagnostic.diagnosticId)
    .sort();
  if (payload.outcome.kind === "accepted") {
    if (blocking.length > 0) invalid("$.payload.outcome", "accepted frontend outcome cannot contain blocking diagnostics");
    if (payload.outcome.selectedEntryIds.length === 0) invalid("$.payload.outcome.selectedEntryIds", "accepted outcome requires at least one selected entry");
    payload.outcome.selectedEntryIds.forEach((id, index) => ref(indexes.entries, id, `$.payload.outcome.selectedEntryIds[${index}]`, "entry"));
  } else {
    if (blocking.length === 0 || !sameStrings(blocking, payload.outcome.blockingDiagnosticIds)) {
      invalid("$.payload.outcome.blockingDiagnosticIds", "rejected outcome must list exactly all root error/fatal diagnostics");
    }
  }
  for (const [index, edge] of payload.inputs.includeEdges.entries()) {
    if (edge.kind !== "source-directive" || edge.resolution.kind !== "unresolved") continue;
    const diagnostic = ref(indexes.diagnostics, edge.resolution.diagnosticId, `$.payload.inputs.includeEdges[${index}].resolution.diagnosticId`, "diagnostic");
    if (diagnostic.severity !== "error" && diagnostic.severity !== "fatal") {
      invalid(`$.payload.inputs.includeEdges[${index}].resolution.diagnosticId`, "unresolved include requires a blocking diagnostic");
    }
    if (payload.outcome.kind !== "rejected") invalid("$.payload.outcome", "unresolved include requires rejected frontend outcome");
  }
}

function verifyDiagnosticSubject(subject: CppCuteFrontendPayloadV2["diagnostics"][number]["subject"], path: string, indexes: ArtifactIndexes): void {
  if (subject.kind === "file") ref(indexes.files, subject.fileId, `${path}.fileId`, "file");
  else if (subject.kind === "declaration") ref(indexes.declarations, subject.declarationId, `${path}.declarationId`, "declaration");
  else if (subject.kind === "type") ref(indexes.types, subject.typeId, `${path}.typeId`, "type");
  else if (subject.kind === "expression") ref(indexes.expressions, subject.expressionId, `${path}.expressionId`, "expression");
  else if (subject.kind === "fact") ref(indexes.facts, subject.factId, `${path}.factId`, "fact");
}

function verifyTemplateArgument(argument: CppCuteTemplateArgumentV1, path: string, indexes: ArtifactIndexes): void {
  if (argument.kind === "type") ref(indexes.types, argument.typeId, `${path}.typeId`, "type");
  else if (argument.kind === "value") ref(indexes.constants, argument.constantId, `${path}.constantId`, "constant");
  else ref(indexes.declarations, argument.declarationId, `${path}.declarationId`, "declaration");
}

function verifyOrigin(origin: CppCuteSourceOriginV1, path: string, indexes: ArtifactIndexes): void {
  if (origin.kind === "source") ref(indexes.spans, origin.spanId, `${path}.spanId`, "span");
  else ref(indexes.spans, origin.anchorSpanId, `${path}.anchorSpanId`, "span");
}

function verifyHierarchyRefs(
  hierarchy: CppCuteHierarchyV1,
  path: string,
  indexes: ArtifactIndexes,
  limits: Partial<DecodeLimits> | undefined,
): void {
  if (hierarchy.kind === "scalar") verifyIntegerExprRefs(hierarchy.value, `${path}.value`, indexes, limits);
  else hierarchy.elements.forEach((element, index) => (
    verifyHierarchyRefs(element, `${path}.elements[${index}]`, indexes, limits)
  ));
}

function verifyIntegerExprRefs(
  expression: CppCuteIntegerExprV1,
  path: string,
  indexes: ArtifactIndexes,
  limits: Partial<DecodeLimits> | undefined,
): void {
  if (expression.kind === "runtime") {
    const declaration = ref(indexes.declarations, expression.declarationId, `${path}.declarationId`, "declaration");
    if (declaration.typeId === null || !isIntegerType(indexes.types.get(declaration.typeId))) {
      invalid(`${path}.declarationId`, "runtime layout integer must reference an integer-typed declaration");
    }
  } else if (expression.kind === "add" || expression.kind === "multiply" || expression.kind === "minimum" || expression.kind === "maximum") {
    expression.values.forEach((value, index) => verifyIntegerExprRefs(value, `${path}.values[${index}]`, indexes, limits));
  } else if (expression.kind === "floor-divide" || expression.kind === "ceil-divide" || expression.kind === "modulo") {
    verifyIntegerExprRefs(expression.value, `${path}.value`, indexes, limits);
    verifyIntegerExprRefs(expression.divisor, `${path}.divisor`, indexes, limits);
    const divisor = evaluateStaticIntegerExpr(expression.divisor, `${path}.divisor`, limits);
    if (divisor !== undefined && divisor <= 0n) invalid(`${path}.divisor`, "layout integer divisor must be positive");
  }
}

function evaluateStaticIntegerExpr(
  expression: CppCuteIntegerExprV1,
  path: string,
  limits: Partial<DecodeLimits> | undefined,
): bigint | undefined {
  try {
    return evaluateStaticCppCuteIntegerExpr(expression, {
      path,
      ...(limits === undefined ? {} : { limits }),
    });
  } catch (error) {
    if (!(error instanceof CppCuteIntegerSemanticsError)) throw error;
    if (error.kind === "resource-limit") resource(error.path, error.message);
    invalid(error.path, error.message);
  }
}

function hierarchyLeaves(hierarchy: CppCuteHierarchyV1): readonly CppCuteIntegerExprV1[] {
  return hierarchy.kind === "scalar" ? [hierarchy.value] : hierarchy.elements.flatMap(hierarchyLeaves);
}

function topRank(hierarchy: CppCuteHierarchyV1): number {
  return hierarchy.kind === "scalar" ? 1 : hierarchy.elements.length;
}

function leafRank(hierarchy: CppCuteHierarchyV1): number {
  return hierarchyLeaves(hierarchy).length;
}

function sameHierarchyTopology(left: CppCuteHierarchyV1, right: CppCuteHierarchyV1): boolean {
  if (left.kind !== right.kind) return false;
  if (left.kind === "scalar" || right.kind === "scalar") return true;
  return left.elements.length === right.elements.length && left.elements.every((entry, index) => {
    const other = right.elements[index];
    return other !== undefined && sameHierarchyTopology(entry, other);
  });
}

function isIntegerType(type: CppCuteResolvedTypeV1 | undefined): boolean {
  return type?.kind === "builtin" && [
    "bool", "char", "signed-char", "unsigned-char", "short", "unsigned-short", "int", "unsigned-int", "long",
    "unsigned-long", "long-long", "unsigned-long-long",
  ].includes(type.builtin);
}

function verifySignedBitWidth(value: string, bits: number, path: string): void {
  const bigint = BigInt(value);
  const minimum = -(1n << BigInt(bits - 1));
  const maximum = (1n << BigInt(bits - 1)) - 1n;
  if (bigint < minimum || bigint > maximum) invalid(path, `signed constant does not fit declared ${bits}-bit width`);
}

function verifyUnsignedBitWidth(value: string, bits: number, path: string): void {
  const bigint = BigInt(value);
  if (bigint < 0n || bigint >= (1n << BigInt(bits))) invalid(path, `unsigned constant does not fit declared ${bits}-bit width`);
}

function verifyValueContainmentCycles(
  payload: CppCuteFrontendPayloadV2,
  typeAbi: ReadonlyMap<string, CppCuteFrontendPayloadV2["sourceAbi"]["types"][number]>,
): void {
  const edges = new Map<string, string[]>();
  for (const type of payload.types) {
    const contained: string[] = [];
    if (type.kind === "array" || type.kind === "vector") contained.push(type.elementTypeId);
    if (type.kind === "record") contained.push(...(typeAbi.get(type.typeId)?.fields.map((field) => field.typeId) ?? []));
    edges.set(type.typeId, contained);
  }
  const active = new Set<string>();
  const done = new Set<string>();
  const visit = (id: string): void => {
    if (done.has(id)) return;
    if (active.has(id)) invalid("$.payload.sourceAbi.types", "value-recursive type containment requires pointer/reference indirection");
    active.add(id);
    for (const child of edges.get(id) ?? []) visit(child);
    active.delete(id);
    done.add(id);
  };
  for (const id of edges.keys()) visit(id);
}

export async function computeCppCuteInputFileSetHash(
  kind: "source" | "header",
  files: CppCuteFrontendPayloadV2["inputs"]["files"],
): Promise<string> {
  const selected = files
    .filter((file) => kind === "source" ? file.role === "main-source" : file.role !== "main-source")
    .map((file) => ({
      role: file.role,
      virtualPath: file.virtualPath,
      contentSha256: file.contentSha256,
      byteLength: file.byteLength,
      owner: file.owner,
      includeRootId: file.includeRootId,
    }));
  return hashCanonicalJson({
    domain: `browsergrad.compiler.cpp-cute.${kind}-set.v2`,
    files: selected,
  });
}

export async function computeCppCuteInputHashes(payload: CppCuteFrontendPayloadV2): Promise<VerifiedCppCuteInputHashes> {
  const sourceSetSha256 = await computeCppCuteInputFileSetHash("source", payload.inputs.files);
  const headerSetSha256 = await computeCppCuteInputFileSetHash("header", payload.inputs.files);
  const closureSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.input-closure.v2",
    mainFileId: payload.inputs.mainFileId,
    includeRoots: payload.inputs.includeRoots,
    files: payload.inputs.files,
    includeEdges: payload.inputs.includeEdges,
    sourceSetSha256,
    headerSetSha256,
  });
  return { sourceSetSha256, headerSetSha256, closureSha256 };
}

function verifyParentForest<T>(
  values: readonly T[],
  id: (entry: T) => string,
  parent: (entry: T) => string | null,
  maxDepth: number,
  path: string,
  name: string,
): void {
  const map = new Map(values.map((entry) => [id(entry), entry]));
  for (const entry of values) {
    const seen = new Set<string>();
    let current: T | undefined = entry;
    let depth = 0;
    while (current !== undefined) {
      const currentId = id(current);
      if (seen.has(currentId)) invalid(path, `${name} parent graph contains a cycle`);
      seen.add(currentId);
      depth += 1;
      if (depth > maxDepth) resource(path, `${name} parent depth exceeds ${maxDepth}`);
      const parentId = parent(current);
      current = parentId === null ? undefined : map.get(parentId);
    }
  }
}

function sameRange(
  left: CppCuteFrontendPayloadV2["spans"][number]["spelling"],
  right: CppCuteFrontendPayloadV2["spans"][number]["spelling"],
): boolean {
  return left.fileId === right.fileId && left.startByte === right.startByte && left.endByte === right.endByte;
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((entry, index) => entry === right[index]);
}

function uniqueValues(values: readonly string[], path: string, name: string): void {
  if (new Set(values).size !== values.length) invalid(path, `duplicate ${name}`);
}

function mapBy<T>(values: readonly T[], id: (entry: T) => string): ReadonlyMap<string, T> {
  return new Map(values.map((entry) => [id(entry), entry]));
}

function mapByChecked<T>(values: readonly T[], id: (entry: T) => string, path: string): ReadonlyMap<string, T> {
  const result = new Map<string, T>();
  for (const entry of values) addUnique(result, id(entry), entry, path);
  return result;
}

function addUnique<T>(map: Map<string, T>, id: string, value: T, path: string): void {
  if (map.has(id)) fail("BG-COMPILER-CPP-CUTE-ARTIFACT-DUPLICATE-ID", path, `duplicate ID ${id}`);
  map.set(id, value);
}

function ref<T>(map: ReadonlyMap<string, T>, id: string, path: string, name: string): T {
  const value = map.get(id);
  if (value === undefined) fail("BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE", path, `unknown ${name} ${id}`);
  return value;
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT", path, message);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH", path, message);
}

function invalid(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID", path, message);
}

function fail(code: CppCuteFrontendArtifactErrorCode, path: string, message: string): never {
  cppCuteFrontendArtifactFailure(code, path, message);
}
