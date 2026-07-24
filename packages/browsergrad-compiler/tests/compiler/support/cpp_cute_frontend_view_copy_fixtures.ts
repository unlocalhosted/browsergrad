import { deriveCppCuteFrontendArtifactId } from "../../../src/cpp_cute_frontend_artifact.js";
import type { CppCuteFrontendArtifactV3, CppCuteFrontendPayloadV3 } from "../../../src/cpp_cute_frontend_types.js";
import {
  computeCppCuteSharedSurfaceHash,
  deriveCppCuteSourceEntityId,
} from "../../../src/cpp_cute_frontend_verify.js";
import {
  CPP_CUTE_FIXTURE_ENTRY_ID,
  CPP_CUTE_FIXTURE_INT_TYPE_ID,
  CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
  CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID,
  CPP_CUTE_FIXTURE_SPAN_ID,
  CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
  createCppCuteArtifactInput,
} from "./cpp_cute_frontend_fixtures.js";

function stableId(kind: string, digit: string): string {
  return `bg.cpp.${kind}.sha256.${digit.repeat(64)}`;
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

const origin = { kind: "source" as const, spanId: CPP_CUTE_FIXTURE_SPAN_ID };
const qualifiers = { const: false, volatile: false, restrict: false } as const;

export const CPP_CUTE_VIEW_COPY_VOID_TYPE_ID = stableId("type", "0");
export const CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID = stableId("type", "1");
export const CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID = stableId("type", "2");
export const CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID = stableId("type", "a");
export const CPP_CUTE_VIEW_COPY_CONST_FLOAT_TYPE_ID = stableId("type", "b");
export const CPP_CUTE_VIEW_COPY_FUNCTION_TYPE_ID = stableId("type", "5");
export const CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_TYPE_ID = stableId("type", "6");
export const CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_TYPE_ID = stableId("type", "7");
export const CPP_CUTE_VIEW_COPY_CONST_INT_TYPE_ID = stableId("type", "8");
export const CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_TYPE_ID = stableId("type", "9");

export const CPP_CUTE_VIEW_COPY_TENSOR_TEMPLATE_DECLARATION_ID = stableId("declaration", "0");
export const CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_DECLARATION_ID = stableId("declaration", "1");
export const CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID = stableId("declaration", "2");
export const CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID = stableId("declaration", "3");
export const CPP_CUTE_VIEW_COPY_DESTINATION_ENGINE_DECLARATION_ID = stableId("declaration", "4");
export const CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID = stableId("declaration", "8");
export const CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID = stableId("declaration", "9");

export const CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID = stableId("fact", "0");
export const CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID = stableId("fact", "1");
export const CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID = stableId("fact", "2");
export const CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID = stableId("fact", "3");

export const CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID = stableId("expression", "0");
export const CPP_CUTE_VIEW_COPY_SOURCE_EXPRESSION_ID = stableId("expression", "1");
export const CPP_CUTE_VIEW_COPY_DESTINATION_EXPRESSION_ID = stableId("expression", "2");

const CPP_CUTE_VIEW_COPY_BODY_ID = stableId("body", "0");
const CPP_CUTE_VIEW_COPY_ROOT_STATEMENT_ID = stableId("statement", "0");
const CPP_CUTE_VIEW_COPY_SOURCE_STATEMENT_ID = stableId("statement", "1");
const CPP_CUTE_VIEW_COPY_DESTINATION_STATEMENT_ID = stableId("statement", "2");
const CPP_CUTE_VIEW_COPY_OPERATION_STATEMENT_ID = stableId("statement", "3");

export async function createCppCuteViewCopyArtifactInput(): Promise<Record<string, unknown>> {
  const artifact = structuredClone(await createCppCuteArtifactInput()) as CppCuteFrontendArtifactV3;
  const payload = artifact.payload;
  await mutateCppCutePayloadToViewCopy(payload);
  const boundArtifact: CppCuteFrontendArtifactV3 = {
    ...artifact,
    artifactId: await deriveCppCuteFrontendArtifactId(payload),
    payload,
  };
  return boundArtifact as unknown as Record<string, unknown>;
}

/** Exact positive-affine rank-3 transpose fixture shared by lowering and browser authority tests. */
export async function mutateCppCutePayloadToRank3ViewCopy(
  payload: CppCuteFrontendPayloadV3,
): Promise<void> {
  await mutateCppCutePayloadToViewCopy(payload);
  mutateCppCuteViewCopyFlatLayouts(payload, {
    shape: [2, 3, 4],
    sourceStrides: [1, 2, 6],
    destinationStrides: [12, 4, 1],
  });
}

/** Static source broadcast over the leading mode with one dense destination. */
export async function mutateCppCutePayloadToBroadcastViewCopy(
  payload: CppCuteFrontendPayloadV3,
): Promise<void> {
  await mutateCppCutePayloadToViewCopy(payload);
  mutateCppCuteViewCopyFlatLayouts(payload, {
    shape: [3, 2],
    sourceStrides: [0, 1],
    destinationStrides: [2, 1],
  });
}

/** Positive-affine source slice with gaps lowered through the same view ABI. */
export async function mutateCppCutePayloadToStridedSliceViewCopy(
  payload: CppCuteFrontendPayloadV3,
): Promise<void> {
  await mutateCppCutePayloadToViewCopy(payload);
  mutateCppCuteViewCopyFlatLayouts(payload, {
    shape: [3, 2],
    sourceStrides: [2, 7],
    destinationStrides: [2, 1],
  });
}

/** Valid producer artifact used to prove that lowering rejects ranks above its explicit profile. */
export async function mutateCppCutePayloadToRank4ViewCopy(
  payload: CppCuteFrontendPayloadV3,
): Promise<void> {
  await mutateCppCutePayloadToViewCopy(payload);
  mutateCppCuteViewCopyFlatLayouts(payload, {
    shape: [2, 2, 2, 2],
    sourceStrides: [1, 2, 4, 8],
    destinationStrides: [8, 4, 2, 1],
  });
}

export async function mutateCppCutePayloadToViewCopy(payload: CppCuteFrontendPayloadV3): Promise<void> {
  const baseInt = payload.types.find((type) => type.typeId === CPP_CUTE_FIXTURE_INT_TYPE_ID);
  if (baseInt === undefined || baseInt.kind !== "builtin") throw new Error("fixture lost int type");

  const types = ([
    ...payload.types,
    {
      typeId: CPP_CUTE_VIEW_COPY_VOID_TYPE_ID,
      kind: "builtin",
      canonicalName: "void",
      qualifiers,
      origin,
      builtin: "void",
    },
    {
      ...baseInt,
      typeId: CPP_CUTE_VIEW_COPY_CONST_INT_TYPE_ID,
      canonicalName: "const int",
      qualifiers: { ...qualifiers, const: true },
    },
    {
      ...baseInt,
      typeId: CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
      canonicalName: "float",
      builtin: "float",
    },
    {
      ...baseInt,
      typeId: CPP_CUTE_VIEW_COPY_CONST_FLOAT_TYPE_ID,
      canonicalName: "const float",
      builtin: "float",
      qualifiers: { ...qualifiers, const: true },
    },
    {
      typeId: CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
      kind: "pointer",
      canonicalName: "const float *",
      qualifiers,
      origin,
      pointeeTypeId: CPP_CUTE_VIEW_COPY_CONST_FLOAT_TYPE_ID,
      addressSpace: "global",
    },
    {
      typeId: CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID,
      kind: "pointer",
      canonicalName: "float *",
      qualifiers,
      origin,
      pointeeTypeId: CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
      addressSpace: "global",
    },
    {
      typeId: CPP_CUTE_VIEW_COPY_FUNCTION_TYPE_ID,
      kind: "function",
      canonicalName: "void (const float *, float *)",
      qualifiers,
      origin,
      returnTypeId: CPP_CUTE_VIEW_COPY_VOID_TYPE_ID,
      parameterTypeIds: [
        CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
        CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID,
      ],
      variadic: false,
      callingConvention: "cuda-device",
    },
    {
      typeId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_TYPE_ID,
      kind: "template-specialization",
      canonicalName: "cute::Layout<cute::Shape<cute::Int<3>, cute::Int<2>>, cute::Stride<cute::Int<2>, cute::Int<1>>>",
      qualifiers,
      origin,
      templateDeclarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
      arguments: [],
    },
    {
      typeId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_TYPE_ID,
      kind: "template-specialization",
      canonicalName: "cute::Tensor<const float *, SourceLayout>",
      qualifiers,
      origin,
      templateDeclarationId: CPP_CUTE_VIEW_COPY_TENSOR_TEMPLATE_DECLARATION_ID,
      arguments: [
        { kind: "type", typeId: CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID },
        { kind: "type", typeId: CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID },
      ],
    },
    {
      typeId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_TYPE_ID,
      kind: "template-specialization",
      canonicalName: "cute::Tensor<float *, DestinationLayout>",
      qualifiers,
      origin,
      templateDeclarationId: CPP_CUTE_VIEW_COPY_TENSOR_TEMPLATE_DECLARATION_ID,
      arguments: [
        { kind: "type", typeId: CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID },
        { kind: "type", typeId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_TYPE_ID },
      ],
    },
  ] satisfies CppCuteFrontendPayloadV3["types"])
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
    cudaAttributes: { host: true, device: true, global: false, forceInline: false },
  };
  const declarations = ([
    ...payload.declarations,
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_TENSOR_TEMPLATE_DECLARATION_ID,
      kind: "template",
      canonicalUsr: "c:@N@cute@ST>2#T#T@Tensor",
      canonicalName: "cute::Tensor",
      lexicalParentId: null,
      semanticParentId: null,
      typeId: CPP_CUTE_FIXTURE_INT_TYPE_ID,
      definitionKind: "external",
      linkage: "external",
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_DECLARATION_ID,
      kind: "variable",
      canonicalUsr: "c:@destination_layout",
      canonicalName: "destination_layout",
      lexicalParentId: null,
      semanticParentId: null,
      typeId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_TYPE_ID,
      definitionKind: "definition",
      linkage: "internal",
      storageDuration: "static",
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      kind: "function",
      canonicalUsr: "c:@F@copy_views#*1f#*f#",
      canonicalName: "copy_views",
      lexicalParentId: null,
      semanticParentId: null,
      typeId: CPP_CUTE_VIEW_COPY_FUNCTION_TYPE_ID,
      definitionKind: "definition",
      linkage: "external",
      mangledName: "_Z10copy_viewsPKfPf",
      cudaAttributes: { host: false, device: true, global: false, forceInline: false },
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID,
      kind: "parameter",
      canonicalUsr: "c:@F@copy_views@FI@source",
      canonicalName: "source",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_VIEW_COPY_SOURCE_POINTER_TYPE_ID,
      definitionKind: "definition",
      storageDuration: "automatic",
      memorySpace: "global",
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_DESTINATION_ENGINE_DECLARATION_ID,
      kind: "parameter",
      canonicalUsr: "c:@F@copy_views@FI@destination",
      canonicalName: "destination",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_VIEW_COPY_DESTINATION_POINTER_TYPE_ID,
      definitionKind: "definition",
      storageDuration: "automatic",
      memorySpace: "global",
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
      kind: "variable",
      canonicalUsr: "c:@F@copy_views@source_tensor",
      canonicalName: "source_tensor",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_TYPE_ID,
      definitionKind: "definition",
      storageDuration: "automatic",
    },
    {
      ...declarationBase,
      declarationId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
      kind: "variable",
      canonicalUsr: "c:@F@copy_views@destination_tensor",
      canonicalName: "destination_tensor",
      lexicalParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      semanticParentId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
      typeId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_TYPE_ID,
      definitionKind: "definition",
      storageDuration: "automatic",
    },
  ] satisfies CppCuteFrontendPayloadV3["declarations"])
    .sort((left, right) => left.declarationId.localeCompare(right.declarationId));

  const destinationLayout = {
    factId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID,
    kind: "affine-layout" as const,
    origin,
    resultDeclarationId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_DECLARATION_ID,
    shape: {
      kind: "tuple" as const,
      elements: [
        { kind: "scalar" as const, value: { kind: "integer" as const, value: "3" as never } },
        { kind: "scalar" as const, value: { kind: "integer" as const, value: "2" as never } },
      ],
    },
    stride: {
      kind: "tuple" as const,
      elements: [
        { kind: "scalar" as const, value: { kind: "integer" as const, value: "2" as never } },
        { kind: "scalar" as const, value: { kind: "integer" as const, value: "1" as never } },
      ],
    },
    rank: 2,
    leafRank: 2,
    size: { kind: "integer" as const, value: "6" as never },
    cosize: { kind: "integer" as const, value: "6" as never },
  };
  const facts = ([
    ...payload.facts,
    destinationLayout,
    {
      factId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID,
      kind: "tensor",
      origin,
      resultDeclarationId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
      elementTypeId: CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
      layoutFactId: CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
      engine: {
        kind: "global-pointer",
        pointerDeclarationId: CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID,
        nullable: false,
      },
      memorySpace: "global",
    },
    {
      factId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
      kind: "tensor",
      origin,
      resultDeclarationId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
      elementTypeId: CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
      layoutFactId: CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID,
      engine: {
        kind: "global-pointer",
        pointerDeclarationId: CPP_CUTE_VIEW_COPY_DESTINATION_ENGINE_DECLARATION_ID,
        nullable: false,
      },
      memorySpace: "global",
    },
    {
      factId: CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID,
      kind: "target-intrinsic",
      origin,
      familyId: "cute:copy@1",
      operation: {
        kind: "copy",
        sourceSpace: "global",
        destinationSpace: "global",
        transferBits: 32,
        asynchronous: false,
      },
      operandExpressionIds: [
        CPP_CUTE_VIEW_COPY_SOURCE_EXPRESSION_ID,
        CPP_CUTE_VIEW_COPY_DESTINATION_EXPRESSION_ID,
      ],
      resultTypeId: CPP_CUTE_VIEW_COPY_VOID_TYPE_ID,
      effects: {
        readsMemory: true,
        writesMemory: true,
        synchronizes: false,
        convergent: false,
      },
      availability: { kind: "portable-candidate" },
    },
  ] satisfies CppCuteFrontendPayloadV3["facts"])
    .sort((left, right) => left.factId.localeCompare(right.factId));

  const functionBodies: CppCuteFrontendPayloadV3["functionBodies"] = [{
    bodyId: CPP_CUTE_VIEW_COPY_BODY_ID,
    declarationId: CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
    rootStatementId: CPP_CUTE_VIEW_COPY_ROOT_STATEMENT_ID,
    statements: [
      {
        statementId: CPP_CUTE_VIEW_COPY_ROOT_STATEMENT_ID,
        kind: "block",
        origin,
        statementIds: [
          CPP_CUTE_VIEW_COPY_SOURCE_STATEMENT_ID,
          CPP_CUTE_VIEW_COPY_DESTINATION_STATEMENT_ID,
          CPP_CUTE_VIEW_COPY_OPERATION_STATEMENT_ID,
        ],
      },
      {
        statementId: CPP_CUTE_VIEW_COPY_SOURCE_STATEMENT_ID,
        kind: "declaration",
        origin,
        declarationId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
      },
      {
        statementId: CPP_CUTE_VIEW_COPY_DESTINATION_STATEMENT_ID,
        kind: "declaration",
        origin,
        declarationId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
      },
      {
        statementId: CPP_CUTE_VIEW_COPY_OPERATION_STATEMENT_ID,
        kind: "expression",
        origin,
        expressionId: CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID,
      },
    ],
    expressions: [
      {
        expressionId: CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID,
        kind: "target-intrinsic",
        typeId: CPP_CUTE_VIEW_COPY_VOID_TYPE_ID,
        valueCategory: "prvalue",
        origin,
        intrinsicFactId: CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID,
      },
      {
        expressionId: CPP_CUTE_VIEW_COPY_SOURCE_EXPRESSION_ID,
        kind: "declaration-reference",
        typeId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_TYPE_ID,
        valueCategory: "lvalue",
        origin,
        declarationId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
      },
      {
        expressionId: CPP_CUTE_VIEW_COPY_DESTINATION_EXPRESSION_ID,
        kind: "declaration-reference",
        typeId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_TYPE_ID,
        valueCategory: "lvalue",
        origin,
        declarationId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_DECLARATION_ID,
      },
    ],
  }];

  const functionEntityBody = {
    entityKind: "function" as const,
    canonicalIdentity: "c:@F@copy_views#*1f#*f#",
    origin,
    domains: ["device", "host"] as const,
  };
  const functionSourceEntityId = await deriveCppCuteSourceEntityId(payload, functionEntityBody);
  const floatEntityBody = {
    entityKind: "type" as const,
    canonicalIdentity: "float",
    origin,
    domains: ["device", "host"] as const,
  };
  const floatSourceEntityId = await deriveCppCuteSourceEntityId(payload, floatEntityBody);
  const sourceEntities = [
    ...payload.sourceEntities.filter((entity) => entity.entityKind === "type"),
    { sourceEntityId: floatSourceEntityId, ...floatEntityBody },
    { sourceEntityId: functionSourceEntityId, ...functionEntityBody },
  ].sort((left, right) => left.sourceEntityId.localeCompare(right.sourceEntityId));

  const sourceAbi: CppCuteFrontendPayloadV3["sourceAbi"] = {
    ...payload.sourceAbi,
    types: ([
      ...payload.sourceAbi.types,
      {
        domain: "device" as const,
        shared: true,
        sourceTypeEntityId: floatSourceEntityId,
        deviceTypeId: CPP_CUTE_VIEW_COPY_FLOAT_TYPE_ID,
        sizeBits: "32" as never,
        alignmentBits: "32" as never,
        fields: [],
        bases: [],
      },
      {
        domain: "host" as const,
        shared: true,
        sourceTypeEntityId: floatSourceEntityId,
        deviceTypeId: null,
        sizeBits: "32" as never,
        alignmentBits: "32" as never,
        fields: [],
        bases: [],
      },
    ] satisfies CppCuteFrontendPayloadV3["sourceAbi"]["types"]).sort((left, right) => (
      `${left.domain}:${left.sourceTypeEntityId}`.localeCompare(`${right.domain}:${right.sourceTypeEntityId}`)
    )),
  };

  const nextPayload: CppCuteFrontendPayloadV3 = {
    ...payload,
    types,
    declarations,
    functionBodies,
    facts,
    entries: [{
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
      kind: "view-copy",
      sourceTensorFactId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID,
      destinationTensorFactId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
      operationExpressionId: CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID,
      selectedRootDeclarationIds: [CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID],
    }],
    outcome: { kind: "accepted", selectedEntryIds: [CPP_CUTE_FIXTURE_ENTRY_ID] },
    sourceEntities,
    sourceAbi,
    semanticPasses: payload.semanticPasses.map((pass) => ({
      ...pass,
      selectedSourceRootEntityIds: [functionSourceEntityId],
      factIds: pass.domain === "device" ? facts.map((fact) => fact.factId).sort() : [],
    })),
  };
  const semanticPasses = await Promise.all(nextPayload.semanticPasses.map(async (pass) => ({
    ...pass,
    sharedSurfaceSha256: await computeCppCuteSharedSurfaceHash(nextPayload, pass.domain),
  })));
  const boundPayload = unshareJsonTree<CppCuteFrontendPayloadV3>({ ...nextPayload, semanticPasses });
  Object.assign(payload, boundPayload);
}

function mutateCppCuteViewCopyFlatLayouts(
  payload: CppCuteFrontendPayloadV3,
  profile: {
    readonly shape: readonly number[];
    readonly sourceStrides: readonly number[];
    readonly destinationStrides: readonly number[];
  },
): void {
  const sourceTensor = payload.facts.find((fact) => (
    fact.kind === "tensor" && fact.factId === CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID
  ));
  const destinationTensor = payload.facts.find((fact) => (
    fact.kind === "tensor" && fact.factId === CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID
  ));
  if (sourceTensor?.kind !== "tensor" || destinationTensor?.kind !== "tensor") {
    throw new Error("fixture lost view-copy tensor facts");
  }
  const sourceLayout = payload.facts.find((fact) => fact.factId === sourceTensor.layoutFactId);
  const destinationLayout = payload.facts.find((fact) => fact.factId === destinationTensor.layoutFactId);
  if (sourceLayout?.kind !== "affine-layout" || destinationLayout?.kind !== "affine-layout") {
    throw new Error("fixture lost view-copy affine-layout facts");
  }
  if (profile.shape.length !== profile.sourceStrides.length ||
      profile.shape.length !== profile.destinationStrides.length) {
    throw new Error("fixture flat layout shape/stride ranks must match");
  }

  const hierarchy = (values: readonly number[]) => ({
    kind: "tuple" as const,
    elements: values.map((value) => ({
      kind: "scalar" as const,
      value: { kind: "integer" as const, value: String(value) as never },
    })),
  });
  const size = profile.shape.reduce((product, extent) => product * extent, 1);
  const cosize = (strides: readonly number[]) => profile.shape.reduce(
    (span, extent, index) => span + (extent - 1) * (strides[index] ?? 0),
    1,
  );
  const mutateLayout = (
    layout: typeof sourceLayout,
    strides: readonly number[],
  ) => Object.assign(layout, {
    shape: hierarchy(profile.shape),
    stride: hierarchy(strides),
    rank: profile.shape.length,
    leafRank: profile.shape.length,
    size: { kind: "integer" as const, value: String(size) as never },
    cosize: { kind: "integer" as const, value: String(cosize(strides)) as never },
  });
  mutateLayout(sourceLayout, profile.sourceStrides);
  mutateLayout(destinationLayout, profile.destinationStrides);

  const canonicalLayoutName = (strides: readonly number[]) => {
    const shape = profile.shape.map((value) => `cute::Int<${value}>`).join(", ");
    const stride = strides.map((value) => `cute::Int<${value}>`).join(", ");
    return `cute::Layout<cute::Shape<${shape}>, cute::Stride<${stride}>>`;
  };
  const sourceLayoutType = payload.types.find((type) => type.typeId === CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID);
  const destinationLayoutType = payload.types.find((type) => (
    type.typeId === CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_TYPE_ID
  ));
  if (sourceLayoutType?.kind !== "template-specialization" ||
      destinationLayoutType?.kind !== "template-specialization") {
    throw new Error("fixture lost view-copy layout types");
  }
  (sourceLayoutType as { canonicalName: string }).canonicalName = canonicalLayoutName(profile.sourceStrides);
  (destinationLayoutType as { canonicalName: string }).canonicalName = canonicalLayoutName(
    profile.destinationStrides,
  );
}
