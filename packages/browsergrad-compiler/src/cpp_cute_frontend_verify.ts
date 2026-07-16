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
  CppCuteDeclarationV3,
  CppCuteExpressionV1,
  CppCuteFrontendPayloadV3,
  CppCuteHierarchyV1,
  CppCuteIntegerExprV1,
  CppCuteResolvedFactV1,
  CppCuteResolvedTypeV1,
  CppCuteSourceEntityV1,
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
  payload: CppCuteFrontendPayloadV3,
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
  await verifySourceAbi(payload, indexes);
  verifyDeclarationInitializers(payload, indexes);
  verifyFunctionBodies(payload, indexes);
  verifyFacts(payload, indexes, options.limits);
  verifyEntries(payload, indexes);
  verifyDiagnosticsAndOutcome(payload, indexes);
  await verifySemanticPasses(payload, indexes);

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
  if (payload.extraction.compilationContractHash !== payload.compilationContractHash) {
    mismatch(
      "$.payload.extraction.compilationContractHash",
      "extraction compilation-contract hash does not match artifact compilationContractHash",
    );
  }
  if (payload.extraction.inputClosureSha256 !== hashes.closureSha256) {
    mismatch("$.payload.extraction.inputClosureSha256", "extraction record does not bind the verified input closure");
  }
  return Object.freeze(hashes);
}

interface ArtifactIndexes {
  readonly files: ReadonlyMap<string, CppCuteFrontendPayloadV3["inputs"]["files"][number]>;
  readonly includeEdges: ReadonlyMap<string, CppCuteFrontendPayloadV3["inputs"]["includeEdges"][number]>;
  readonly includeRoots: ReadonlyMap<string, CppCuteFrontendPayloadV3["inputs"]["includeRoots"][number]>;
  readonly spans: ReadonlyMap<string, CppCuteFrontendPayloadV3["spans"][number]>;
  readonly macros: ReadonlyMap<string, CppCuteFrontendPayloadV3["macroExpansions"][number]>;
  readonly types: ReadonlyMap<string, CppCuteResolvedTypeV1>;
  readonly constants: ReadonlyMap<string, CppCuteConstantV1>;
  readonly declarations: ReadonlyMap<string, CppCuteDeclarationV3>;
  readonly instantiations: ReadonlyMap<string, CppCuteFrontendPayloadV3["templateInstantiations"][number]>;
  readonly resolutions: ReadonlyMap<string, CppCuteFrontendPayloadV3["overloadResolutions"][number]>;
  readonly bodies: ReadonlyMap<string, CppCuteFrontendPayloadV3["functionBodies"][number]>;
  readonly statements: ReadonlyMap<string, CppCuteStatementV1>;
  readonly expressions: ReadonlyMap<string, CppCuteExpressionV1>;
  readonly facts: ReadonlyMap<string, CppCuteResolvedFactV1>;
  readonly entries: ReadonlyMap<string, CppCuteFrontendPayloadV3["entries"][number]>;
  readonly diagnostics: ReadonlyMap<string, CppCuteFrontendPayloadV3["diagnostics"][number]>;
}

function buildIndexes(payload: CppCuteFrontendPayloadV3): ArtifactIndexes {
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
    includeEdges: mapBy(payload.inputs.includeEdges, (entry) => entry.includeEdgeId),
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

function verifyInputClosure(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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
  left: CppCuteFrontendPayloadV3["inputs"]["files"][number]["owner"],
  right: CppCuteFrontendPayloadV3["inputs"]["includeRoots"][number]["owner"],
): boolean {
  return left.kind === right.kind && (
    left.kind !== "dependency" || (right.kind === "dependency" && left.dependencyId === right.dependencyId)
  );
}

function virtualPathContains(root: string, candidate: string): boolean {
  return root === "/" ? candidate.startsWith("/") && candidate !== "/" : candidate.startsWith(`${root}/`);
}

function verifySpans(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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
  range: CppCuteFrontendPayloadV3["spans"][number]["spelling"],
  path: string,
  indexes: ArtifactIndexes,
): void {
  const file = ref(indexes.files, range.fileId, `${path}.fileId`, "file");
  const start = wireIntegerToBigInt(range.startByte);
  const end = wireIntegerToBigInt(range.endByte);
  const length = wireIntegerToBigInt(file.byteLength);
  if (start > end || end > length) invalid(path, "source range must be a bounded half-open byte range within its file");
}

function verifyMacroExpansions(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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

function verifyTypes(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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

function verifyConstants(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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

function verifyDeclarations(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
  uniqueValues(payload.declarations.map((declaration) => declaration.canonicalUsr), "$.payload.declarations", "canonical USR");
  for (const [index, declaration] of payload.declarations.entries()) {
    const path = `$.payload.declarations[${index}]`;
    verifyOrigin(declaration.origin, `${path}.origin`, indexes);
    if (declaration.origin.kind === "source") {
      if (declaration.identitySpanId === null) {
        invalid(`${path}.identitySpanId`, "source declaration requires an identifier/declarator token span");
      }
      const identitySpan = ref(indexes.spans, declaration.identitySpanId, `${path}.identitySpanId`, "span");
      const declarationSpan = ref(indexes.spans, declaration.origin.spanId, `${path}.origin.spanId`, "span");
      if (wireIntegerToBigInt(identitySpan.spelling.startByte) === wireIntegerToBigInt(identitySpan.spelling.endByte) ||
          wireIntegerToBigInt(identitySpan.expansion.startByte) === wireIntegerToBigInt(identitySpan.expansion.endByte)) {
        invalid(`${path}.identitySpanId`, "declaration identity span must be nonempty");
      }
      if (!sourceRangeContains(declarationSpan.spelling, identitySpan.spelling) ||
          !sourceRangeContains(declarationSpan.expansion, identitySpan.expansion)) {
        invalid(`${path}.identitySpanId`, "declaration identity span must stay within declaration origin");
      }
    } else if (declaration.identitySpanId !== null) {
      invalid(`${path}.identitySpanId`, "implicit declaration cannot claim a source identity span");
    }
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

function sourceRangeContains(
  outer: CppCuteFrontendPayloadV3["spans"][number]["spelling"],
  inner: CppCuteFrontendPayloadV3["spans"][number]["spelling"],
): boolean {
  return outer.fileId === inner.fileId &&
    wireIntegerToBigInt(outer.startByte) <= wireIntegerToBigInt(inner.startByte) &&
    wireIntegerToBigInt(outer.endByte) >= wireIntegerToBigInt(inner.endByte);
}

function verifyDeclarationInitializers(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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
  declaration: CppCuteDeclarationV3,
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

function verifyTemplateInstantiations(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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

function verifyOverloadResolutions(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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

async function verifySourceAbi(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): Promise<void> {
  const sourceEntities = await verifySourceEntities(payload, indexes);
  const observedDomains = new Map<string, Set<"host" | "device">>();
  const abiOwners = new Map<string, Set<"host" | "device">>();
  for (const [passIndex, pass] of payload.semanticPasses.entries()) {
    for (const [entityIndex, sourceEntityId] of pass.selectedSourceRootEntityIds.entries()) {
      ref(
        sourceEntities,
        sourceEntityId,
        `$.payload.semanticPasses[${passIndex}].selectedSourceRootEntityIds[${entityIndex}]`,
        "source entity",
      );
      addSourceEntityDomain(observedDomains, sourceEntityId, pass.domain);
    }
  }
  const ownSourceEntity = (
    sourceEntityId: string,
    expectedKind: CppCuteSourceEntityV1["entityKind"],
    domain: "host" | "device",
    path: string,
  ): CppCuteSourceEntityV1 => {
    const entity = ref(sourceEntities, sourceEntityId, path, "source entity");
    if (entity.entityKind !== expectedKind) invalid(path, `source entity must have kind ${expectedKind}`);
    if (!entity.domains.includes(domain)) invalid(path, `source entity is not declared in ${domain} semantic domain`);
    let domains = abiOwners.get(sourceEntityId);
    if (domains === undefined) {
      domains = new Set();
      abiOwners.set(sourceEntityId, domains);
    }
    if (domains.has(domain)) invalid(path, "source entity has multiple ABI owners in one semantic domain");
    domains.add(domain);
    addSourceEntityDomain(observedDomains, sourceEntityId, domain);
    return entity;
  };
  const referenceSourceType = (
    sourceEntityId: string,
    domain: "host" | "device",
    path: string,
  ): CppCuteSourceEntityV1 => {
    const entity = ref(sourceEntities, sourceEntityId, path, "source type entity");
    if (entity.entityKind !== "type") invalid(path, "ABI type reference requires a type source entity");
    if (!entity.domains.includes(domain)) invalid(path, `source type is not declared in ${domain} semantic domain`);
    return entity;
  };
  const typeAbi = new Map(
    payload.sourceAbi.types.map((entry) => [abiDomainKey(entry.domain, entry.sourceTypeEntityId), entry]),
  );
  for (const [index, abi] of payload.sourceAbi.types.entries()) {
    const path = `$.payload.sourceAbi.types[${index}]`;
    const sourceEntity = ownSourceEntity(abi.sourceTypeEntityId, "type", abi.domain, `${path}.sourceTypeEntityId`);
    if (abi.shared !== (sourceEntity.domains.length === 2)) {
      invalid(`${path}.shared`, "ABI shared flag must derive from the verified source entity domain set");
    }
    let deviceType: CppCuteResolvedTypeV1 | undefined;
    if (abi.domain === "device") {
      if (abi.deviceTypeId === null) invalid(`${path}.deviceTypeId`, "device ABI type requires device graph identity");
      deviceType = ref(indexes.types, abi.deviceTypeId, `${path}.deviceTypeId`, "type");
      if (sourceEntity.canonicalIdentity !== deviceType.canonicalName ||
          !sameSourceOrigin(sourceEntity.origin, deviceType.origin)) {
        invalid(`${path}.sourceTypeEntityId`, "device ABI source identity differs from the canonical device type");
      }
    } else if (abi.deviceTypeId !== null) {
      invalid(`${path}.deviceTypeId`, "host ABI cannot reference the device-resolved type graph");
    }
    const size = wireIntegerToBigInt(abi.sizeBits);
    const alignment = wireIntegerToBigInt(abi.alignmentBits);
    if (size === 0n || alignment === 0n || (alignment & (alignment - 1n)) !== 0n || alignment > size) {
      invalid(path, "ABI size/alignment must be positive, power-of-two aligned, and alignment cannot exceed size");
    }
    if (deviceType?.kind === "record" && !deviceType.complete) {
      invalid(`${path}.deviceTypeId`, "incomplete device record cannot have a type-layout ABI record");
    }
    if (deviceType !== undefined && deviceType.kind !== "record" && (abi.fields.length !== 0 || abi.bases.length !== 0)) {
      invalid(path, "non-record device ABI cannot contain fields or bases");
    }
    uniqueValues(abi.fields.map((field) => field.sourceEntityId), `${path}.fields`, "source field identity");
    for (const [fieldIndex, field] of abi.fields.entries()) {
      const fieldPath = `${path}.fields[${fieldIndex}]`;
      ownSourceEntity(field.sourceEntityId, "field", abi.domain, `${fieldPath}.sourceEntityId`);
      referenceSourceType(field.sourceTypeEntityId, abi.domain, `${fieldPath}.sourceTypeEntityId`);
      const fieldAbi = typeAbi.get(abiDomainKey(abi.domain, field.sourceTypeEntityId));
      if (fieldAbi === undefined) invalid(`${fieldPath}.sourceTypeEntityId`, "ABI field type is absent from its target domain");
      const end = wireIntegerToBigInt(field.bitOffset) + wireIntegerToBigInt(fieldAbi.sizeBits);
      if (end > size) invalid(`${fieldPath}.bitOffset`, "ABI field extends beyond record size");
    }
    for (const [baseIndex, base] of abi.bases.entries()) {
      const basePath = `${path}.bases[${baseIndex}]`;
      referenceSourceType(base.sourceTypeEntityId, abi.domain, `${basePath}.sourceTypeEntityId`);
      const baseAbi = typeAbi.get(abiDomainKey(abi.domain, base.sourceTypeEntityId));
      if (baseAbi === undefined) invalid(`${basePath}.sourceTypeEntityId`, "ABI base type is absent from its target domain");
      const end = wireIntegerToBigInt(base.bitOffset) + wireIntegerToBigInt(baseAbi.sizeBits);
      if (end > size) invalid(`${basePath}.bitOffset`, "ABI base extends beyond record size");
    }
  }
  verifyAbiValueContainmentCycles(payload);

  for (const [index, abi] of payload.sourceAbi.functions.entries()) {
    const path = `$.payload.sourceAbi.functions[${index}]`;
    const sourceEntity = ownSourceEntity(abi.sourceEntityId, "function", abi.domain, `${path}.sourceEntityId`);
    if (abi.shared !== (sourceEntity.domains.length === 2)) {
      invalid(`${path}.shared`, "ABI shared flag must derive from the verified source entity domain set");
    }
    let functionType: Extract<CppCuteResolvedTypeV1, { readonly kind: "function" }> | undefined;
    if (abi.domain === "device") {
      if (abi.deviceDeclarationId === null) {
        invalid(`${path}.deviceDeclarationId`, "device function ABI requires device graph declaration identity");
      }
      const declaration = ref(
        indexes.declarations,
        abi.deviceDeclarationId,
        `${path}.deviceDeclarationId`,
        "declaration",
      );
      if (declaration.kind !== "function") {
        invalid(`${path}.deviceDeclarationId`, "device function ABI must reference a function declaration");
      }
      if (sourceEntity.canonicalIdentity !== declaration.canonicalUsr ||
          !sameSourceOrigin(sourceEntity.origin, declaration.origin)) {
        invalid(`${path}.sourceEntityId`, "device function ABI source identity differs from canonical declaration");
      }
      const resolved = declaration.typeId === null ? undefined : indexes.types.get(declaration.typeId);
      if (resolved?.kind !== "function") {
        invalid(`${path}.deviceDeclarationId`, "device function declaration must reference a resolved function type");
      }
      functionType = resolved;
      const expectedConvention = declaration.cudaAttributes.global
        ? "nvptx-kernel"
        : declaration.cudaAttributes.device
          ? "nvptx-device"
          : null;
      if (expectedConvention === null || abi.loweredCallingConvention !== expectedConvention) {
        invalid(`${path}.loweredCallingConvention`, "device ABI convention must match device/global source attributes");
      }
    } else {
      if (abi.deviceDeclarationId !== null) {
        invalid(`${path}.deviceDeclarationId`, "host ABI cannot reference a device-resolved declaration");
      }
      if (abi.loweredCallingConvention === "nvptx-kernel" || abi.loweredCallingConvention === "nvptx-device") {
        invalid(`${path}.loweredCallingConvention`, "NVPTX conventions belong to the device ABI domain");
      }
    }
    const returnAbi = typeAbi.get(abiDomainKey(abi.domain, abi.returnSourceTypeEntityId));
    referenceSourceType(abi.returnSourceTypeEntityId, abi.domain, `${path}.returnSourceTypeEntityId`);
    if (returnAbi === undefined) {
      invalid(`${path}.returnSourceTypeEntityId`, "function return type is absent from its target ABI domain");
    }
    if (functionType !== undefined && returnAbi.deviceTypeId !== functionType.returnTypeId) {
      invalid(`${path}.returnSourceTypeEntityId`, "device function ABI return type differs from resolved function type");
    }
    if (functionType !== undefined && functionType.parameterTypeIds.length !== abi.parameters.length) {
      invalid(`${path}.parameters`, "device function ABI parameter count differs from resolved function type");
    }
    abi.parameters.forEach((parameter, parameterIndex) => {
      const parameterPath = `${path}.parameters[${parameterIndex}]`;
      if (parameter.ordinal !== parameterIndex) {
        invalid(`${parameterPath}.ordinal`, "ABI parameter ordinals must be contiguous and match array position");
      }
      ownSourceEntity(parameter.sourceEntityId, "parameter", abi.domain, `${parameterPath}.sourceEntityId`);
      referenceSourceType(parameter.sourceTypeEntityId, abi.domain, `${parameterPath}.sourceTypeEntityId`);
      const parameterType = typeAbi.get(abiDomainKey(abi.domain, parameter.sourceTypeEntityId));
      if (parameterType === undefined) {
        invalid(`${parameterPath}.sourceTypeEntityId`, "ABI parameter type is absent from its target domain");
      }
      if (functionType !== undefined && parameterType.deviceTypeId !== functionType.parameterTypeIds[parameterIndex]) {
        invalid(`${parameterPath}.sourceTypeEntityId`, "device ABI parameter type differs from resolved function type");
      }
    });
    uniqueValues(abi.parameters.map((parameter) => parameter.sourceEntityId), `${path}.parameters`, "source parameter identity");
  }
  for (const [index, entity] of payload.sourceEntities.entries()) {
    const observed = observedDomains.get(entity.sourceEntityId);
    if (observed === undefined || observed.size !== entity.domains.length ||
        entity.domains.some((domain) => !observed.has(domain))) {
      invalid(
        `$.payload.sourceEntities[${index}].domains`,
        "source entity domains must equal exact semantic-pass and ABI observations",
      );
    }
  }
}

function addSourceEntityDomain(
  domainsByEntity: Map<string, Set<"host" | "device">>,
  sourceEntityId: string,
  domain: "host" | "device",
): void {
  let domains = domainsByEntity.get(sourceEntityId);
  if (domains === undefined) {
    domains = new Set();
    domainsByEntity.set(sourceEntityId, domains);
  }
  domains.add(domain);
}

async function verifySourceEntities(
  payload: CppCuteFrontendPayloadV3,
  indexes: ArtifactIndexes,
): Promise<ReadonlyMap<string, CppCuteSourceEntityV1>> {
  const entities = mapBy(payload.sourceEntities, (entry) => entry.sourceEntityId);
  for (const [index, entity] of payload.sourceEntities.entries()) {
    const path = `$.payload.sourceEntities[${index}]`;
    verifyOrigin(entity.origin, `${path}.origin`, indexes);
    const expected = await deriveSourceEntityId(entity, indexes);
    if (entity.sourceEntityId !== expected) {
      mismatch(`${path}.sourceEntityId`, "source entity ID is not derived from canonical source identity and resolved origin");
    }
  }
  return entities;
}

type CppCuteSourceEntityIdentity = {
  readonly entityKind: CppCuteSourceEntityV1["entityKind"];
  readonly canonicalIdentity: string;
  readonly origin: CppCuteSourceOriginV1;
  readonly domains: readonly ("host" | "device")[];
};

export async function deriveCppCuteSourceEntityId(
  payload: CppCuteFrontendPayloadV3,
  entity: CppCuteSourceEntityIdentity,
): Promise<string> {
  return deriveSourceEntityId(entity, buildIndexes(payload));
}

async function deriveSourceEntityId(
  entity: CppCuteSourceEntityIdentity,
  indexes: ArtifactIndexes,
): Promise<string> {
  const digest = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.source-entity-id.v1",
    entityKind: entity.entityKind,
    canonicalIdentity: entity.canonicalIdentity,
    origin: canonicalSourceOrigin(entity.origin, indexes),
  });
  return `bg.cpp.source-entity.sha256.${digest}`;
}

function canonicalSourceOrigin(origin: CppCuteSourceOriginV1, indexes: ArtifactIndexes): object {
  if (origin.kind === "source") {
    return { kind: "source", span: canonicalSourceSpan(origin.spanId, indexes) };
  }
  return {
    kind: "implicit",
    reason: origin.reason,
    anchor: canonicalSourceSpan(origin.anchorSpanId, indexes),
  };
}

function canonicalSourceSpan(spanId: string, indexes: ArtifactIndexes): object {
  const span = ref(indexes.spans, spanId, "$.payload.sourceEntities.origin", "span");
  return {
    spelling: canonicalSourceRange(span.spelling, indexes),
    expansion: canonicalSourceRange(span.expansion, indexes),
  };
}

function canonicalSourceRange(
  range: CppCuteFrontendPayloadV3["spans"][number]["spelling"],
  indexes: ArtifactIndexes,
): object {
  const file = ref(indexes.files, range.fileId, "$.payload.sourceEntities.origin", "file");
  return {
    virtualPath: file.virtualPath,
    contentSha256: file.contentSha256,
    startByte: range.startByte,
    endByte: range.endByte,
  };
}

function sameSourceOrigin(left: CppCuteSourceOriginV1, right: CppCuteSourceOriginV1): boolean {
  return left.kind === right.kind && (left.kind === "source"
    ? right.kind === "source" && left.spanId === right.spanId
    : right.kind === "implicit" && left.anchorSpanId === right.anchorSpanId && left.reason === right.reason);
}

async function verifySemanticPasses(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): Promise<void> {
  const factOwner = new Map<string, number>();
  const diagnosticOwner = new Map<string, number>();
  const openedFileUnion = new Set<string>();
  const includeEdgeUnion = new Set<string>();
  const sourceEntities = mapBy(payload.sourceEntities, (entry) => entry.sourceEntityId);
  const selectedSourceRootEntityIds = await deriveSelectedSourceRootEntityIds(payload, indexes, sourceEntities);
  for (const [passIndex, pass] of payload.semanticPasses.entries()) {
    const path = `$.payload.semanticPasses[${passIndex}]`;
    for (const [entityIndex, sourceEntityId] of pass.selectedSourceRootEntityIds.entries()) {
      ref(
        sourceEntities,
        sourceEntityId,
        `${path}.selectedSourceRootEntityIds[${entityIndex}]`,
        "source entity",
      );
    }
    for (const [fileIndex, fileId] of pass.openedFileIds.entries()) {
      ref(indexes.files, fileId, `${path}.openedFileIds[${fileIndex}]`, "file");
      openedFileUnion.add(fileId);
    }
    for (const [edgeIndex, edgeId] of pass.includeEdgeIds.entries()) {
      ref(indexes.includeEdges, edgeId, `${path}.includeEdgeIds[${edgeIndex}]`, "include edge");
      includeEdgeUnion.add(edgeId);
    }
    for (const [factIndex, factId] of pass.factIds.entries()) {
      ref(indexes.facts, factId, `${path}.factIds[${factIndex}]`, "fact");
      if (factOwner.has(factId)) invalid(`${path}.factIds[${factIndex}]`, "fact belongs to more than one semantic pass");
      factOwner.set(factId, passIndex);
    }
    for (const [diagnosticIndex, diagnosticId] of pass.diagnosticIds.entries()) {
      ref(
        indexes.diagnostics,
        diagnosticId,
        `${path}.diagnosticIds[${diagnosticIndex}]`,
        "diagnostic",
      );
      if (diagnosticOwner.has(diagnosticId)) {
        invalid(`${path}.diagnosticIds[${diagnosticIndex}]`, "diagnostic belongs to more than one semantic pass");
      }
      diagnosticOwner.set(diagnosticId, passIndex);
    }
    if (pass.status === "not-run") {
      if (pass.openedFileIds.length !== 0 || pass.includeEdgeIds.length !== 0 ||
          pass.observedInputClosureSha256 !== null || pass.sharedSurfaceSha256 !== null ||
          pass.selectedSourceRootEntityIds.length !== 0 || pass.factIds.length !== 0 || pass.diagnosticIds.length !== 0) {
        invalid(`${path}.status`, "not-run semantic pass cannot claim input, root, surface, fact, or diagnostic evidence");
      }
    } else {
      if (!pass.openedFileIds.includes(payload.inputs.mainFileId)) {
        invalid(`${path}.openedFileIds`, "executed semantic pass must open the main source");
      }
      verifySemanticPassInputClosure(payload, passIndex, indexes);
      const observedHash = await computeCppCuteSemanticPassInputClosureHash(payload, passIndex);
      if (pass.observedInputClosureSha256 !== observedHash) {
        mismatch(`${path}.observedInputClosureSha256`, "semantic pass observed-input hash is not canonical");
      }
    }
    const blocking = pass.diagnosticIds.filter((id) => {
      const diagnostic = indexes.diagnostics.get(id);
      return diagnostic?.parentDiagnosticId === null &&
        (diagnostic.severity === "error" || diagnostic.severity === "fatal");
    });
    if (pass.status === "succeeded" && blocking.length !== 0) {
      invalid(`${path}.status`, "succeeded semantic pass cannot own blocking diagnostics");
    }
    if (pass.status === "failed" && blocking.length === 0) {
      invalid(`${path}.status`, "failed semantic pass requires an owned root error or fatal diagnostic");
    }
    if (pass.status === "failed" && pass.selectedSourceRootEntityIds.some(
      (sourceEntityId) => !selectedSourceRootEntityIds.includes(sourceEntityId),
    )) {
      mismatch(`${path}.selectedSourceRootEntityIds`, "failed semantic pass claimed a noncanonical selected root");
    }
    if (pass.status === "succeeded" && !sameStrings(pass.selectedSourceRootEntityIds, selectedSourceRootEntityIds)) {
      mismatch(
        `${path}.selectedSourceRootEntityIds`,
        "successful semantic pass must observe exact content-derived roots of every serialized device entry",
      );
    }
    if (pass.status === "succeeded" && pass.sharedSurfaceSha256 === null) {
      invalid(`${path}.sharedSurfaceSha256`, "successful semantic pass requires shared source-surface evidence");
    }
    if (pass.sharedSurfaceSha256 !== null) {
      const surfaceHash = await computeCppCuteSharedSurfaceHash(payload, pass.domain);
      if (pass.sharedSurfaceSha256 !== surfaceHash) {
        mismatch(`${path}.sharedSurfaceSha256`, "shared source-surface hash is not canonical for pass ABI projection");
      }
    }
  }

  const devicePass = payload.semanticPasses[0];
  const hostPass = payload.semanticPasses[1];
  if (devicePass === undefined || hostPass === undefined) invalid("$.payload.semanticPasses", "two semantic passes required");
  if (devicePass.status === "not-run") {
    invalid("$.payload.semanticPasses[0].status", "device extraction pass must run");
  }
  if (devicePass.status === "failed" && hostPass.status !== "not-run") {
    invalid("$.payload.semanticPasses[1].status", "host validation must not run after device extraction failure");
  }
  if (devicePass.status === "succeeded" && hostPass.status === "not-run") {
    invalid("$.payload.semanticPasses[1].status", "host validation must run after successful device extraction");
  }
  const bothSucceeded = devicePass.status === "succeeded" && hostPass.status === "succeeded";
  if ((payload.outcome.kind === "accepted") !== bothSucceeded) {
    invalid("$.payload.outcome", "accepted outcome is equivalent to successful device extraction and host validation");
  }
  if (bothSucceeded && devicePass.sharedSurfaceSha256 !== hostPass.sharedSurfaceSha256) {
    mismatch("$.payload.semanticPasses", "host and device selected source-surface identities do not converge");
  }
  if (bothSucceeded && (
    devicePass.selectedSourceRootEntityIds.length === 0 ||
    !sameStrings(devicePass.selectedSourceRootEntityIds, hostPass.selectedSourceRootEntityIds)
  )) {
    mismatch(
      "$.payload.semanticPasses[1].selectedSourceRootEntityIds",
      "accepted host/device passes require one nonempty identical selected-root source projection",
    );
  }
  if (devicePass.factIds.length !== payload.facts.length ||
      !sameStrings(devicePass.factIds, payload.facts.map((fact) => fact.factId))) {
    invalid("$.payload.semanticPasses[0].factIds", "device extraction pass must own the complete canonical fact graph");
  }
  if (hostPass.factIds.length !== 0) {
    invalid("$.payload.semanticPasses[1].factIds", "host validation pass cannot contribute to the canonical semantic graph");
  }
  if (openedFileUnion.size !== payload.inputs.files.length ||
      payload.inputs.files.some((file) => !openedFileUnion.has(file.fileId))) {
    invalid("$.payload.semanticPasses", "per-pass opened-file observations must cover the exact union input file set");
  }
  if (includeEdgeUnion.size !== payload.inputs.includeEdges.length ||
      payload.inputs.includeEdges.some((edge) => !includeEdgeUnion.has(edge.includeEdgeId))) {
    invalid("$.payload.semanticPasses", "per-pass include-edge observations must cover the exact union input edge set");
  }

  for (const [factIndex, fact] of payload.facts.entries()) {
    const owner = factOwner.get(fact.factId);
    if (owner === undefined) invalid(`$.payload.facts[${factIndex}]`, "fact is missing semantic-pass ownership");
    if (owner !== 0) {
      invalid(`$.payload.facts[${factIndex}]`, "canonical semantic facts must belong to device extraction pass");
    }
    if (fact.kind === "tensor") {
      const layoutOwner = factOwner.get(fact.layoutFactId);
      if (layoutOwner !== owner) {
        invalid(`$.payload.facts[${factIndex}].layoutFactId`, "tensor and layout facts must share one semantic domain");
      }
    }
    if (fact.kind === "target-intrinsic" && fact.availability.kind === "recognized-unsupported") {
      if (diagnosticOwner.get(fact.availability.diagnosticId) !== owner) {
        invalid(
          `$.payload.facts[${factIndex}].availability.diagnosticId`,
          "target-intrinsic diagnostic must share its semantic domain",
        );
      }
    }
  }
  for (const [diagnosticIndex, diagnostic] of payload.diagnostics.entries()) {
    const owner = diagnosticOwner.get(diagnostic.diagnosticId);
    if (owner === undefined) {
      invalid(`$.payload.diagnostics[${diagnosticIndex}]`, "compiler diagnostic is missing exact semantic-pass ownership");
    }
    if (diagnostic.parentDiagnosticId !== null &&
        diagnosticOwner.get(diagnostic.parentDiagnosticId) !== owner) {
      invalid(`$.payload.diagnostics[${diagnosticIndex}].parentDiagnosticId`, "diagnostic note crosses pass ownership");
    }
    if (owner === 1 && (
      diagnostic.subject.kind === "declaration" ||
      diagnostic.subject.kind === "type" ||
      diagnostic.subject.kind === "expression" ||
      diagnostic.subject.kind === "fact"
    )) {
      invalid(
        `$.payload.diagnostics[${diagnosticIndex}].subject`,
        "host validation diagnostic cannot reference the device canonical graph",
      );
    }
    if (diagnostic.subject.kind === "fact") {
      const expectedOwner = factOwner.get(diagnostic.subject.factId);
      if (owner !== expectedOwner) {
        invalid(`$.payload.diagnostics[${diagnosticIndex}].subject`, "fact diagnostic crosses semantic domains");
      }
    }
  }
  for (const [edgeIndex, edge] of payload.inputs.includeEdges.entries()) {
    if (edge.kind !== "source-directive" || edge.resolution.kind !== "unresolved") continue;
    const owner = diagnosticOwner.get(edge.resolution.diagnosticId);
    const owningPass = owner === undefined ? undefined : payload.semanticPasses[owner];
    if (owningPass === undefined || !owningPass.includeEdgeIds.includes(edge.includeEdgeId)) {
      invalid(
        `$.payload.inputs.includeEdges[${edgeIndex}].resolution.diagnosticId`,
        "unresolved-include diagnostic owner must have observed the unresolved edge",
      );
    }
  }
  for (const [entryIndex, entry] of payload.entries.entries()) {
    const owners = entry.kind === "layout"
      ? [factOwner.get(entry.layoutFactId)]
      : [factOwner.get(entry.sourceTensorFactId), factOwner.get(entry.destinationTensorFactId)];
    if (owners.some((owner) => owner !== 0)) {
      invalid(`$.payload.entries[${entryIndex}]`, "frontend entries must select only the device canonical semantic graph");
    }
  }
  for (const domain of ["device", "host"] as const) {
    const pass = domain === "device" ? devicePass : hostPass;
    if (pass.status === "not-run" && (
      payload.sourceAbi.types.some((entry) => entry.domain === domain) ||
      payload.sourceAbi.functions.some((entry) => entry.domain === domain)
    )) {
      invalid("$.payload.sourceAbi", `not-run ${domain} pass cannot claim ${domain} ABI evidence`);
    }
  }
}

async function deriveSelectedSourceRootEntityIds(
  payload: CppCuteFrontendPayloadV3,
  indexes: ArtifactIndexes,
  sourceEntities: ReadonlyMap<string, CppCuteSourceEntityV1>,
): Promise<readonly string[]> {
  const declarationIds = [...new Set(payload.entries.flatMap((entry) => entry.selectedRootDeclarationIds))].sort();
  const sourceEntityIds = await Promise.all(declarationIds.map(async (declarationId, index) => {
    const path = `$.payload.entries.selectedRootDeclarationIds[${index}]`;
    const declaration = ref(indexes.declarations, declarationId, path, "declaration");
    if (declaration.kind !== "function" && declaration.kind !== "variable") {
      invalid(path, "selected root must be a function or variable source declaration");
    }
    if (!declaration.cudaAttributes.device && !declaration.cudaAttributes.global) {
      invalid(path, "selected root must belong to the canonical device semantic pass");
    }
    const entityKind = declaration.kind;
    const sourceEntityId = await deriveSourceEntityId({
      entityKind,
      canonicalIdentity: declaration.canonicalUsr,
      origin: declaration.origin,
      domains: [],
    }, indexes);
    const entity = ref(sourceEntities, sourceEntityId, path, "selected-root source entity");
    if (entity.entityKind !== entityKind || entity.canonicalIdentity !== declaration.canonicalUsr ||
        !sameSourceOrigin(entity.origin, declaration.origin)) {
      invalid(path, "selected root source identity differs from canonical device declaration");
    }
    return sourceEntityId;
  }));
  return sourceEntityIds.sort();
}

function verifySemanticPassInputClosure(
  payload: CppCuteFrontendPayloadV3,
  passIndex: number,
  indexes: ArtifactIndexes,
): void {
  const pass = payload.semanticPasses[passIndex];
  if (pass === undefined) invalid(`$.payload.semanticPasses[${passIndex}]`, "semantic pass required");
  const path = `$.payload.semanticPasses[${passIndex}]`;
  const opened = new Set(pass.openedFileIds);
  const edgeIds = new Set(pass.includeEdgeIds);
  const reachable = new Set<string>([payload.inputs.mainFileId]);
  for (const [edgeIndex, edgeId] of pass.includeEdgeIds.entries()) {
    const edge = ref(indexes.includeEdges, edgeId, `${path}.includeEdgeIds[${edgeIndex}]`, "include edge");
    if (edge.kind === "compiler-forced") {
      if (!opened.has(edge.fileId)) invalid(`${path}.includeEdgeIds[${edgeIndex}]`, "forced include is absent from pass opened files");
      reachable.add(edge.fileId);
      continue;
    }
    if (!opened.has(edge.includingFileId)) {
      invalid(`${path}.includeEdgeIds[${edgeIndex}]`, "include edge source is absent from pass opened files");
    }
    if (edge.resolution.kind === "resolved" && !opened.has(edge.resolution.fileId)) {
      invalid(`${path}.includeEdgeIds[${edgeIndex}]`, "resolved include is absent from pass opened files");
    }
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of payload.inputs.includeEdges) {
      if (!edgeIds.has(edge.includeEdgeId) || edge.kind !== "source-directive" ||
          !reachable.has(edge.includingFileId) || edge.resolution.kind !== "resolved") continue;
      if (!reachable.has(edge.resolution.fileId)) {
        reachable.add(edge.resolution.fileId);
        changed = true;
      }
    }
  }
  for (const fileId of opened) {
    if (!reachable.has(fileId)) invalid(`${path}.openedFileIds`, `pass opened unreachable file ${fileId}`);
  }
  for (const edge of payload.inputs.includeEdges) {
    if (edge.kind !== "compiler-forced") continue;
    if (!pass.includeEdgeIds.includes(edge.includeEdgeId)) {
      invalid(`${path}.includeEdgeIds`, "executed semantic pass omitted a compiler-forced include edge");
    }
    if (!opened.has(edge.fileId)) invalid(`${path}.openedFileIds`, "pass omitted compiler-forced file");
  }
}

export async function computeCppCuteSemanticPassInputClosureHash(
  payload: CppCuteFrontendPayloadV3,
  passIndex: number,
): Promise<string> {
  const pass = payload.semanticPasses[passIndex];
  if (pass === undefined) invalid(`$.payload.semanticPasses[${passIndex}]`, "semantic pass required");
  const opened = new Set(pass.openedFileIds);
  const edges = new Set(pass.includeEdgeIds);
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.semantic-pass-input-closure.v1",
    passId: pass.passId,
    includeRoots: payload.inputs.includeRoots,
    files: payload.inputs.files.filter((file) => opened.has(file.fileId)),
    includeEdges: payload.inputs.includeEdges.filter((edge) => edges.has(edge.includeEdgeId)),
  });
}

export async function computeCppCuteSharedSurfaceHash(
  payload: CppCuteFrontendPayloadV3,
  domain: "host" | "device",
): Promise<string> {
  const pass = payload.semanticPasses.find((entry) => entry.domain === domain);
  if (pass === undefined) invalid("$.payload.semanticPasses", `missing ${domain} semantic pass`);
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.shared-source-surface.v2",
    selectedSourceRootEntityIds: pass.selectedSourceRootEntityIds,
    types: payload.sourceAbi.types
      .filter((entry) => entry.domain === domain && entry.shared)
      .map((entry) => ({
        sourceTypeEntityId: entry.sourceTypeEntityId,
        fields: entry.fields.map((field) => ({
          sourceEntityId: field.sourceEntityId,
          sourceTypeEntityId: field.sourceTypeEntityId,
        })),
        bases: entry.bases.map((base) => ({
          sourceTypeEntityId: base.sourceTypeEntityId,
          virtual: base.virtual,
        })),
      })),
    functions: payload.sourceAbi.functions
      .filter((entry) => entry.domain === domain && entry.shared)
      .map((entry) => ({
        sourceEntityId: entry.sourceEntityId,
        returnSourceTypeEntityId: entry.returnSourceTypeEntityId,
        parameters: entry.parameters.map((parameter) => ({
          ordinal: parameter.ordinal,
          sourceEntityId: parameter.sourceEntityId,
          sourceTypeEntityId: parameter.sourceTypeEntityId,
        })),
      })),
  });
}

function verifyFunctionBodies(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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
  payload: CppCuteFrontendPayloadV3,
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

function verifyEntries(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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

function verifyDiagnosticsAndOutcome(payload: CppCuteFrontendPayloadV3, indexes: ArtifactIndexes): void {
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
    } else if (diagnostic.subject.kind !== "compiler") {
      invalid(`${path}.location`, "locationless compiler-pass diagnostics require a compiler subject");
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

function verifyDiagnosticSubject(subject: CppCuteFrontendPayloadV3["diagnostics"][number]["subject"], path: string, indexes: ArtifactIndexes): void {
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

function verifyAbiValueContainmentCycles(payload: CppCuteFrontendPayloadV3): void {
  for (const domain of ["host", "device"] as const) {
    const edges = new Map<string, string[]>();
    for (const type of payload.sourceAbi.types.filter((entry) => entry.domain === domain)) {
      edges.set(
        type.sourceTypeEntityId,
        [
          ...type.fields.map((field) => field.sourceTypeEntityId),
          ...type.bases.map((base) => base.sourceTypeEntityId),
        ],
      );
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
}

function abiDomainKey(domain: "host" | "device", id: string): string {
  return `${domain}:${id}`;
}

export async function computeCppCuteInputFileSetHash(
  kind: "source" | "header",
  files: CppCuteFrontendPayloadV3["inputs"]["files"],
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

export async function computeCppCuteInputHashes(payload: CppCuteFrontendPayloadV3): Promise<VerifiedCppCuteInputHashes> {
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
  left: CppCuteFrontendPayloadV3["spans"][number]["spelling"],
  right: CppCuteFrontendPayloadV3["spans"][number]["spelling"],
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
