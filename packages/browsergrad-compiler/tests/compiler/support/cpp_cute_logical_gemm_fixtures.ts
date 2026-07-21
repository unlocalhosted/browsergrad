import { deriveCppCuteFrontendArtifactId } from "../../../src/cpp_cute_frontend_artifact.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR,
  type CppCuteFrontendArtifactV3,
  type CppCuteFrontendPayloadV3,
} from "../../../src/cpp_cute_frontend_types.js";
import {
  computeCppCuteSharedSurfaceHash,
  deriveCppCuteSourceEntityId,
} from "../../../src/cpp_cute_frontend_verify.js";
import {
  CPP_CUTE_FIXTURE_ENTRY_ID,
  CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
  CPP_CUTE_FIXTURE_SPAN_ID,
  CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
  createCppCuteArtifactInput,
} from "./cpp_cute_frontend_fixtures.js";
import {
  CPP_CUTE_VIEW_COPY_DESTINATION_ENGINE_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID,
  CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID,
  CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
  CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
  CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_FUNCTION_TYPE_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID,
  mutateCppCutePayloadToViewCopy,
} from "./cpp_cute_frontend_view_copy_fixtures.js";

function stableId(kind: string, digit: string): string {
  return `bg.cpp.${kind}.sha256.${digit.repeat(64)}`;
}

const origin = { kind: "source" as const, spanId: CPP_CUTE_FIXTURE_SPAN_ID };
const qualifiers = { const: false, volatile: false, restrict: false } as const;

export const CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_TYPE_ID = stableId("type", "d");
export const CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_TYPE_ID = stableId("type", "e");
export const CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_DECLARATION_ID = stableId("declaration", "a");
export const CPP_CUTE_LOGICAL_GEMM_RHS_ENGINE_DECLARATION_ID = stableId("declaration", "b");
export const CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_DECLARATION_ID = stableId("declaration", "c");
export const CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_FACT_ID = stableId("fact", "b");
export const CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID = stableId("fact", "c");
export const CPP_CUTE_LOGICAL_GEMM_FACT_ID = stableId("fact", "d");

export async function createCppCuteLogicalGemmArtifactInput(): Promise<Record<string, unknown>> {
  const artifact = structuredClone(await createCppCuteArtifactInput()) as CppCuteFrontendArtifactV3;
  await mutateCppCutePayloadToLogicalGemm(artifact.payload);
  const minor = CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR;
  return {
    ...artifact,
    version: { major: artifact.version.major, minor },
    artifactId: await deriveCppCuteFrontendArtifactId(artifact.payload, { minor }),
  } as unknown as Record<string, unknown>;
}

export async function mutateCppCutePayloadToLogicalGemm(payload: CppCuteFrontendPayloadV3): Promise<void> {
  await mutateCppCutePayloadToViewCopy(payload);
  const functionType = requiredType(payload, CPP_CUTE_VIEW_COPY_FUNCTION_TYPE_ID, "function");
  const root = requiredDeclaration(payload, CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID);
  const lhsTensorDeclaration = requiredDeclaration(payload, CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID);
  const destinationTensorDeclaration = requiredDeclaration(
    payload,
    CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
  );
  const lhsLayout = requiredFact(payload, CPP_CUTE_FIXTURE_LAYOUT_FACT_ID, "affine-layout");
  const destinationLayout = requiredFact(
    payload,
    CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID,
    "affine-layout",
  );
  const lhsTensor = requiredFact(payload, CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID, "tensor");
  const destinationTensor = requiredFact(
    payload,
    CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
    "tensor",
  );

  Object.assign(functionType, {
    canonicalName: "void (const float *, const float *, float *)",
    parameterTypeIds: [
      CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
      CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
      CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID,
    ],
    callingConvention: "cuda-kernel",
  });
  Object.assign(root, {
    canonicalUsr: "c:@F@logical_gemm#*1f#*1f#*f#",
    canonicalName: "logical_gemm",
    mangledName: "_Z12logical_gemmPKfS0_Pf",
    cudaAttributes: { host: false, device: true, global: true, forceInline: false },
  });
  Object.assign(lhsTensorDeclaration, {
    canonicalUsr: "c:@F@logical_gemm@lhs_tensor",
    canonicalName: "lhs_tensor",
  });
  Object.assign(destinationTensorDeclaration, {
    canonicalUsr: "c:@F@logical_gemm@destination_tensor",
    canonicalName: "destination_tensor",
  });
  Object.assign(
    requiredDeclaration(payload, CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID),
    { canonicalUsr: "c:@F@logical_gemm@FI@lhs", canonicalName: "lhs" },
  );
  Object.assign(
    requiredDeclaration(payload, CPP_CUTE_VIEW_COPY_DESTINATION_ENGINE_DECLARATION_ID),
    { canonicalUsr: "c:@F@logical_gemm@FI@destination", canonicalName: "destination" },
  );

  const rhsLayoutType = {
    typeId: CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_TYPE_ID,
    kind: "template-specialization" as const,
    canonicalName: "cute::Layout<cute::Shape<cute::Int<23>, cute::Int<19>>, cute::Stride<cute::Int<19>, cute::Int<1>>>",
    qualifiers,
    origin,
    templateDeclarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
    arguments: [],
  };
  const rhsTensorType = {
    typeId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_TYPE_ID,
    kind: "template-specialization" as const,
    canonicalName: "cute::Tensor<const float *, RhsLayout>",
    qualifiers,
    origin,
    templateDeclarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
    arguments: [
      { kind: "type" as const, typeId: CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID },
      { kind: "type" as const, typeId: CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_TYPE_ID },
    ],
  };
  const types = [...payload.types, rhsLayoutType, rhsTensorType]
    .sort((left, right) => left.typeId.localeCompare(right.typeId));

  const declarationBase = {
    targetTypeId: null,
    initializerExpressionId: null,
    origin,
    identitySpanId: CPP_CUTE_FIXTURE_SPAN_ID,
    linkage: "none" as const,
    storageDuration: "none" as const,
    memorySpace: "generic" as const,
    mangledName: null,
    cudaAttributes: { host: false, device: true, global: false, forceInline: false },
  };
  const declarations = ([
    ...payload.declarations,
    {
      ...declarationBase,
      declarationId: CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_DECLARATION_ID,
      kind: "variable" as const,
      canonicalUsr: "c:@F@logical_gemm@rhs_layout",
      canonicalName: "rhs_layout",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_TYPE_ID,
      definitionKind: "definition" as const,
      storageDuration: "automatic" as const,
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_LOGICAL_GEMM_RHS_ENGINE_DECLARATION_ID,
      kind: "parameter" as const,
      canonicalUsr: "c:@F@logical_gemm@FI@rhs",
      canonicalName: "rhs",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
      definitionKind: "definition" as const,
      storageDuration: "automatic" as const,
      memorySpace: "global" as const,
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_DECLARATION_ID,
      kind: "variable" as const,
      canonicalUsr: "c:@F@logical_gemm@rhs_tensor",
      canonicalName: "rhs_tensor",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_TYPE_ID,
      definitionKind: "definition" as const,
      storageDuration: "automatic" as const,
    },
  ] satisfies CppCuteFrontendPayloadV3["declarations"])
    .sort((left, right) => left.declarationId.localeCompare(right.declarationId));

  setDenseLayout(lhsLayout, [17, 23]);
  setDenseLayout(destinationLayout, [17, 19]);
  const rhsLayout = denseLayoutFact(
    CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_FACT_ID,
    CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_DECLARATION_ID,
    [23, 19],
  );
  const rhsTensor = {
    factId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID,
    kind: "tensor" as const,
    origin,
    resultDeclarationId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_DECLARATION_ID,
    elementTypeId: CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
    layoutFactId: CPP_CUTE_LOGICAL_GEMM_RHS_LAYOUT_FACT_ID,
    engine: {
      kind: "global-pointer" as const,
      pointerDeclarationId: CPP_CUTE_LOGICAL_GEMM_RHS_ENGINE_DECLARATION_ID,
      nullable: false,
    },
    memorySpace: "global" as const,
  };
  const logicalGemm = {
    factId: CPP_CUTE_LOGICAL_GEMM_FACT_ID,
    kind: "logical-gemm-tile" as const,
    origin,
    functionDeclarationId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
    lhsTensorFactId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID,
    rhsTensorFactId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID,
    destinationTensorFactId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
    logicalTile: { m: "16" as never, n: "16" as never, k: "16" as never },
    boundary: {
      lhs: "zero-fill" as const,
      rhs: "zero-fill" as const,
      destination: "mask-outside-logical-shape" as const,
    },
    accumulation: {
      inputDType: "f32" as const,
      accumulatorDType: "f32" as const,
      outputDType: "f32" as const,
      initialAccumulator: "positive-zero" as const,
      product: "multiply" as const,
      reduction: "sum" as const,
      reductionOrder: "increasing-k" as const,
      rounding: "toward-nearest-ties-even" as const,
      contraction: "forbid" as const,
      reassociation: "forbid" as const,
    },
    phases: {
      order: ["load", "accumulate", "store"] as const,
      participation: "masked-full-logical-tile" as const,
    },
    overlap: { kind: "forbid-all" as const },
  };
  const facts = ([
    lhsLayout,
    destinationLayout,
    rhsLayout,
    lhsTensor,
    rhsTensor,
    destinationTensor,
    logicalGemm,
  ] satisfies CppCuteFrontendPayloadV3["facts"])
    .sort((left, right) => left.factId.localeCompare(right.factId));

  const body = payload.functionBodies.find((candidate) => (
    candidate.declarationId === CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID
  ));
  if (body === undefined) throw new Error("fixture lost function body");
  const rootStatement = body.statements.find((statement) => statement.statementId === body.rootStatementId);
  if (rootStatement?.kind !== "block") throw new Error("fixture lost root block");
  const statementIds = ["a", "b", "c"].map((digit) => stableId("statement", digit));
  const statements: CppCuteFrontendPayloadV3["functionBodies"][number]["statements"] = [
    { ...rootStatement, statementIds },
    {
      statementId: statementIds[0]!,
      kind: "declaration",
      origin,
      declarationId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
    },
    {
      statementId: statementIds[1]!,
      kind: "declaration",
      origin,
      declarationId: CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_DECLARATION_ID,
    },
    {
      statementId: statementIds[2]!,
      kind: "declaration",
      origin,
      declarationId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
    },
  ];
  const functionBodies: CppCuteFrontendPayloadV3["functionBodies"] = [{
    ...body,
    statements,
    expressions: [],
  }];

  const previousFunctionEntity = payload.sourceEntities.find((entity) => entity.entityKind === "function");
  if (previousFunctionEntity === undefined) throw new Error("fixture lost function source identity");
  const functionEntityBody = {
    entityKind: "function" as const,
    canonicalIdentity: root.canonicalUsr,
    origin,
    domains: ["device", "host"] as const,
  };
  const functionSourceEntityId = await deriveCppCuteSourceEntityId(payload, functionEntityBody);
  const sourceEntities = payload.sourceEntities.map((entity) => (
    entity === previousFunctionEntity
      ? { sourceEntityId: functionSourceEntityId, ...functionEntityBody }
      : entity
  )).sort((left, right) => left.sourceEntityId.localeCompare(right.sourceEntityId));

  const nextPayload: CppCuteFrontendPayloadV3 = {
    ...payload,
    types,
    declarations,
    functionBodies,
    facts,
    entries: [{
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
      kind: "logical-gemm-tile",
      logicalGemmTileFactId: CPP_CUTE_LOGICAL_GEMM_FACT_ID,
      selectedRootDeclarationIds: [CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID],
    }],
    diagnostics: [],
    outcome: { kind: "accepted", selectedEntryIds: [CPP_CUTE_FIXTURE_ENTRY_ID] },
    sourceEntities,
    semanticPasses: payload.semanticPasses.map((pass) => ({
      ...pass,
      selectedSourceRootEntityIds: [functionSourceEntityId],
      factIds: pass.domain === "device" ? facts.map((fact) => fact.factId).sort() : [],
      diagnosticIds: [],
    })),
  };
  const semanticPasses = await Promise.all(nextPayload.semanticPasses.map(async (pass) => ({
    ...pass,
    sharedSurfaceSha256: await computeCppCuteSharedSurfaceHash(nextPayload, pass.domain),
  })));
  Object.assign(payload, unshareJsonTree({ ...nextPayload, semanticPasses }));
}

function unshareJsonTree<T>(value: T): T {
  if (Array.isArray(value)) return value.map((entry) => unshareJsonTree(entry)) as T;
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, unshareJsonTree(entry)]),
    ) as T;
  }
  return value;
}

function denseLayoutFact(factId: string, declarationId: string, shape: readonly [number, number]) {
  const fact = {
    factId,
    kind: "affine-layout" as const,
    origin,
    resultDeclarationId: declarationId,
    shape: hierarchy(shape),
    stride: hierarchy([shape[1], 1]),
    rank: 2,
    leafRank: 2,
    size: integer(shape[0] * shape[1]),
    cosize: integer(shape[0] * shape[1]),
  } satisfies CppCuteFrontendPayloadV3["facts"][number];
  return fact;
}

function setDenseLayout(
  fact: Extract<CppCuteFrontendPayloadV3["facts"][number], { readonly kind: "affine-layout" }>,
  shape: readonly [number, number],
): void {
  Object.assign(fact, {
    shape: hierarchy(shape),
    stride: hierarchy([shape[1], 1]),
    rank: 2,
    leafRank: 2,
    size: integer(shape[0] * shape[1]),
    cosize: integer(shape[0] * shape[1]),
  });
}

function hierarchy(values: readonly [number, number]) {
  return {
    kind: "tuple" as const,
    elements: values.map((value) => ({ kind: "scalar" as const, value: integer(value) })),
  };
}

function integer(value: number) {
  return { kind: "integer" as const, value: String(value) as never };
}

function requiredType<K extends CppCuteFrontendPayloadV3["types"][number]["kind"]>(
  payload: CppCuteFrontendPayloadV3,
  typeId: string,
  kind: K,
): Extract<CppCuteFrontendPayloadV3["types"][number], { readonly kind: K }> {
  const type = payload.types.find((candidate) => candidate.typeId === typeId);
  if (type?.kind !== kind) throw new Error(`fixture lost ${kind} type ${typeId}`);
  return type as Extract<CppCuteFrontendPayloadV3["types"][number], { readonly kind: K }>;
}

function requiredDeclaration(payload: CppCuteFrontendPayloadV3, declarationId: string) {
  const declaration = payload.declarations.find((candidate) => candidate.declarationId === declarationId);
  if (declaration === undefined) throw new Error(`fixture lost declaration ${declarationId}`);
  return declaration;
}

function requiredFact<K extends CppCuteFrontendPayloadV3["facts"][number]["kind"]>(
  payload: CppCuteFrontendPayloadV3,
  factId: string,
  kind: K,
): Extract<CppCuteFrontendPayloadV3["facts"][number], { readonly kind: K }> {
  const fact = payload.facts.find((candidate) => candidate.factId === factId);
  if (fact?.kind !== kind) throw new Error(`fixture lost ${kind} fact ${factId}`);
  return fact as Extract<CppCuteFrontendPayloadV3["facts"][number], { readonly kind: K }>;
}
