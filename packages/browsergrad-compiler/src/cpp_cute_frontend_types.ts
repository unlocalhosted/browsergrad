import type {
  JsonObject,
  WireEnvelope,
  WireI64,
  WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

export const CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA = "browsergrad.compiler.cpp-cute.frontend-artifact";
export const CPP_CUTE_FRONTEND_ARTIFACT_MAJOR = 1;
export const CPP_CUTE_FRONTEND_ARTIFACT_MINOR = 0;

export interface CppCuteFrontendPayloadV1 extends JsonObject {
  readonly profileHash: string;
  readonly inputs: CppCuteInputClosureV1;
  readonly spans: readonly CppCuteSourceSpanV1[];
  readonly macroExpansions: readonly CppCuteMacroExpansionV1[];
  readonly types: readonly CppCuteResolvedTypeV1[];
  readonly constants: readonly CppCuteConstantV1[];
  readonly declarations: readonly CppCuteDeclarationV1[];
  readonly templateInstantiations: readonly CppCuteTemplateInstantiationV1[];
  readonly overloadResolutions: readonly CppCuteOverloadResolutionV1[];
  readonly sourceAbi: CppCuteSourceAbiV1;
  readonly functionBodies: readonly CppCuteFunctionBodyV1[];
  readonly facts: readonly CppCuteResolvedFactV1[];
  readonly entries: readonly CppCuteFrontendEntryV1[];
  readonly diagnostics: readonly CppCuteFrontendDiagnosticV1[];
  readonly outcome: CppCuteFrontendOutcomeV1;
  readonly extraction: CppCuteExtractionRecordV1;
}

export type CppCuteFrontendArtifactV1 = WireEnvelope<CppCuteFrontendPayloadV1>;

export interface CppCuteIncludeRootV1 extends JsonObject {
  readonly includeRootId: string;
  readonly ordinal: number;
  readonly mode: "quote" | "system";
  readonly virtualPath: string;
  readonly manifestSha256: string;
}

export interface CppCuteSourceFileV1 extends JsonObject {
  readonly fileId: string;
  readonly role:
    | "main-source"
    | "project-header"
    | "cuda-header"
    | "cute-header"
    | "compiler-builtin-header"
    | "generated-header";
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
  readonly profileDependency: "none" | "cuda" | "cute" | "frontend";
}

export type CppCuteIncludeResolutionV1 =
  | (JsonObject & {
      readonly kind: "resolved";
      readonly fileId: string;
      readonly includeRootId: string;
    })
  | (JsonObject & {
      readonly kind: "unresolved";
      readonly diagnosticId: string;
    });

export interface CppCuteIncludeEdgeV1 extends JsonObject {
  readonly includeEdgeId: string;
  readonly includingFileId: string;
  readonly directiveSpanId: string;
  readonly spelling: string;
  readonly mode: "quote" | "angle";
  readonly resolution: CppCuteIncludeResolutionV1;
}

export interface CppCuteInputClosureV1 extends JsonObject {
  readonly mainFileId: string;
  /** Include resolution precedence; ordinals must be contiguous. */
  readonly includeRoots: readonly CppCuteIncludeRootV1[];
  readonly files: readonly CppCuteSourceFileV1[];
  readonly includeEdges: readonly CppCuteIncludeEdgeV1[];
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly closureSha256: string;
}

export interface CppCuteFileRangeV1 extends JsonObject {
  readonly fileId: string;
  readonly startByte: WireU64;
  readonly endByte: WireU64;
}

export interface CppCuteSourceSpanV1 extends JsonObject {
  readonly spanId: string;
  readonly spelling: CppCuteFileRangeV1;
  readonly expansion: CppCuteFileRangeV1;
  readonly macroExpansionId: string | null;
}

export type CppCuteImplicitOriginReasonV1 =
  | "implicit-cast"
  | "implicit-construction"
  | "default-argument"
  | "compiler-builtin"
  | "template-substitution";

export type CppCuteSourceOriginV1 =
  | (JsonObject & {
      readonly kind: "source";
      readonly spanId: string;
    })
  | (JsonObject & {
      readonly kind: "implicit";
      readonly anchorSpanId: string;
      readonly reason: CppCuteImplicitOriginReasonV1;
    });

export interface CppCuteMacroExpansionV1 extends JsonObject {
  readonly macroExpansionId: string;
  readonly macroName: string;
  readonly definitionSpanId: string;
  readonly invocationSpanId: string;
  readonly parentMacroExpansionId: string | null;
}

export interface CppCuteTypeQualifiersV1 extends JsonObject {
  readonly const: boolean;
  readonly volatile: boolean;
  readonly restrict: boolean;
}

export type CppCuteAddressSpaceV1 =
  | "generic"
  | "host"
  | "global"
  | "shared"
  | "local"
  | "constant";

interface CppCuteResolvedTypeBaseV1 extends JsonObject {
  readonly typeId: string;
  readonly canonicalName: string;
  readonly qualifiers: CppCuteTypeQualifiersV1;
  readonly origin: CppCuteSourceOriginV1;
}

export type CppCuteBuiltinTypeNameV1 =
  | "void"
  | "bool"
  | "char"
  | "signed-char"
  | "unsigned-char"
  | "short"
  | "unsigned-short"
  | "int"
  | "unsigned-int"
  | "long"
  | "unsigned-long"
  | "long-long"
  | "unsigned-long-long"
  | "half"
  | "bfloat16"
  | "float"
  | "double";

export type CppCuteTemplateArgumentV1 =
  | (JsonObject & { readonly kind: "type"; readonly typeId: string })
  | (JsonObject & { readonly kind: "value"; readonly constantId: string })
  | (JsonObject & { readonly kind: "template"; readonly declarationId: string });

export type CppCuteResolvedTypeV1 =
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "builtin";
      readonly builtin: CppCuteBuiltinTypeNameV1;
    })
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "pointer" | "lvalue-reference" | "rvalue-reference";
      readonly pointeeTypeId: string;
      readonly addressSpace: CppCuteAddressSpaceV1;
    })
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "array";
      readonly elementTypeId: string;
      readonly elementCount: WireU64;
    })
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "vector";
      readonly elementTypeId: string;
      readonly elementCount: number;
    })
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "function";
      readonly returnTypeId: string;
      readonly parameterTypeIds: readonly string[];
      readonly variadic: boolean;
      readonly callingConvention: CppCuteCallingConventionV1;
    })
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "record" | "enum";
      readonly declarationId: string;
      readonly complete: boolean;
    })
  | (CppCuteResolvedTypeBaseV1 & {
      readonly kind: "template-specialization";
      readonly templateDeclarationId: string;
      readonly arguments: readonly CppCuteTemplateArgumentV1[];
    });

interface CppCuteConstantBaseV1 extends JsonObject {
  readonly constantId: string;
  readonly typeId: string;
  readonly origin: CppCuteSourceOriginV1;
}

export type CppCuteConstantV1 =
  | (CppCuteConstantBaseV1 & { readonly kind: "boolean"; readonly value: boolean })
  | (CppCuteConstantBaseV1 & {
      readonly kind: "signed-integer";
      readonly bitWidth: number;
      readonly value: WireI64;
    })
  | (CppCuteConstantBaseV1 & {
      readonly kind: "unsigned-integer";
      readonly bitWidth: number;
      readonly value: WireU64;
    })
  | (CppCuteConstantBaseV1 & {
      readonly kind: "floating";
      readonly format: "f16" | "bf16" | "f32" | "f64";
      readonly bits: string;
    })
  | (CppCuteConstantBaseV1 & { readonly kind: "null-pointer" })
  | (CppCuteConstantBaseV1 & {
      readonly kind: "enum";
      readonly enumDeclarationId: string;
      readonly valueConstantId: string;
    })
  | (CppCuteConstantBaseV1 & {
      readonly kind: "aggregate";
      readonly elementConstantIds: readonly string[];
    });

export type CppCuteDeclarationKindV1 =
  | "namespace"
  | "type-alias"
  | "record"
  | "enum"
  | "field"
  | "function"
  | "parameter"
  | "variable"
  | "template"
  | "template-parameter";

export interface CppCuteCudaAttributesV1 extends JsonObject {
  readonly host: boolean;
  readonly device: boolean;
  readonly global: boolean;
  readonly forceInline: boolean;
}

export interface CppCuteDeclarationV1 extends JsonObject {
  readonly declarationId: string;
  readonly kind: CppCuteDeclarationKindV1;
  readonly canonicalUsr: string;
  readonly canonicalName: string;
  readonly lexicalParentId: string | null;
  readonly semanticParentId: string | null;
  readonly typeId: string | null;
  readonly targetTypeId: string | null;
  readonly origin: CppCuteSourceOriginV1;
  readonly definitionKind: "definition" | "declaration-only" | "builtin" | "external";
  readonly linkage: "none" | "internal" | "external" | "weak" | "linkonce-odr";
  readonly storageDuration: "none" | "automatic" | "static" | "thread";
  readonly memorySpace: CppCuteAddressSpaceV1;
  readonly mangledName: string | null;
  readonly cudaAttributes: CppCuteCudaAttributesV1;
}

export interface CppCuteTemplateInstantiationV1 extends JsonObject {
  readonly instantiationId: string;
  readonly templateDeclarationId: string;
  readonly specializationDeclarationId: string;
  readonly arguments: readonly CppCuteTemplateArgumentV1[];
  readonly pointOfInstantiationSpanId: string;
  readonly parentInstantiationId: string | null;
  readonly depth: number;
}

export interface CppCuteOverloadResolutionV1 extends JsonObject {
  readonly resolutionId: string;
  readonly origin: CppCuteSourceOriginV1;
  readonly selectedDeclarationId: string;
  readonly candidateDeclarationIds: readonly string[];
  readonly argumentTypeIds: readonly string[];
  readonly resultTypeId: string;
}

export type CppCuteCallingConventionV1 =
  | "c"
  | "cuda-kernel"
  | "cuda-device"
  | "cxx-member";

export interface CppCuteAbiFieldV1 extends JsonObject {
  readonly declarationId: string;
  readonly typeId: string;
  readonly bitOffset: WireU64;
}

export interface CppCuteAbiBaseV1 extends JsonObject {
  readonly typeId: string;
  readonly bitOffset: WireU64;
  readonly virtual: boolean;
}

export interface CppCuteTypeAbiV1 extends JsonObject {
  readonly typeId: string;
  readonly sizeBits: WireU64;
  readonly alignmentBits: WireU64;
  readonly fields: readonly CppCuteAbiFieldV1[];
  readonly bases: readonly CppCuteAbiBaseV1[];
}

export interface CppCuteParameterAbiV1 extends JsonObject {
  readonly declarationId: string;
  readonly typeId: string;
  readonly passing: "direct" | "indirect" | "ignore";
}

export interface CppCuteFunctionAbiV1 extends JsonObject {
  readonly declarationId: string;
  readonly callingConvention: CppCuteCallingConventionV1;
  readonly returnTypeId: string;
  readonly returnPassing: "direct" | "indirect" | "ignore";
  readonly parameters: readonly CppCuteParameterAbiV1[];
}

export interface CppCuteSourceAbiV1 extends JsonObject {
  readonly types: readonly CppCuteTypeAbiV1[];
  readonly functions: readonly CppCuteFunctionAbiV1[];
}

interface CppCuteStatementBaseV1 extends JsonObject {
  readonly statementId: string;
  readonly origin: CppCuteSourceOriginV1;
}

export type CppCuteStatementV1 =
  | (CppCuteStatementBaseV1 & { readonly kind: "block"; readonly statementIds: readonly string[] })
  | (CppCuteStatementBaseV1 & { readonly kind: "declaration"; readonly declarationId: string })
  | (CppCuteStatementBaseV1 & { readonly kind: "expression"; readonly expressionId: string })
  | (CppCuteStatementBaseV1 & {
      readonly kind: "if";
      readonly conditionExpressionId: string;
      readonly thenStatementId: string;
      readonly elseStatementId: string | null;
    })
  | (CppCuteStatementBaseV1 & {
      readonly kind: "for";
      readonly initializerStatementId: string | null;
      readonly conditionExpressionId: string | null;
      readonly incrementExpressionId: string | null;
      readonly bodyStatementId: string;
    })
  | (CppCuteStatementBaseV1 & {
      readonly kind: "while";
      readonly conditionExpressionId: string;
      readonly bodyStatementId: string;
    })
  | (CppCuteStatementBaseV1 & { readonly kind: "return"; readonly expressionId: string | null })
  | (CppCuteStatementBaseV1 & { readonly kind: "break" | "continue" });

interface CppCuteExpressionBaseV1 extends JsonObject {
  readonly expressionId: string;
  readonly typeId: string;
  readonly valueCategory: "lvalue" | "xvalue" | "prvalue";
  readonly origin: CppCuteSourceOriginV1;
}

export type CppCuteExpressionV1 =
  | (CppCuteExpressionBaseV1 & { readonly kind: "constant"; readonly constantId: string })
  | (CppCuteExpressionBaseV1 & { readonly kind: "declaration-reference"; readonly declarationId: string })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "resolved-call";
      readonly overloadResolutionId: string;
      readonly argumentExpressionIds: readonly string[];
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "construction";
      readonly constructorDeclarationId: string;
      readonly argumentExpressionIds: readonly string[];
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "member-access";
      readonly baseExpressionId: string;
      readonly memberDeclarationId: string;
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "subscript";
      readonly baseExpressionId: string;
      readonly indexExpressionId: string;
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "cast";
      readonly castKind: "integral" | "floating" | "pointer" | "qualification" | "lvalue-to-rvalue";
      readonly operandExpressionId: string;
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "unary";
      readonly operator: "plus" | "minus" | "logical-not" | "bitwise-not" | "dereference" | "address-of";
      readonly operandExpressionId: string;
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "binary";
      readonly operator:
        | "add"
        | "subtract"
        | "multiply"
        | "divide"
        | "remainder"
        | "equal"
        | "not-equal"
        | "less"
        | "less-equal"
        | "greater"
        | "greater-equal"
        | "logical-and"
        | "logical-or"
        | "bitwise-and"
        | "bitwise-or"
        | "bitwise-xor"
        | "shift-left"
        | "shift-right"
        | "assign";
      readonly leftExpressionId: string;
      readonly rightExpressionId: string;
    })
  | (CppCuteExpressionBaseV1 & {
      readonly kind: "conditional";
      readonly conditionExpressionId: string;
      readonly thenExpressionId: string;
      readonly elseExpressionId: string;
    })
  | (CppCuteExpressionBaseV1 & { readonly kind: "target-intrinsic"; readonly intrinsicFactId: string });

export interface CppCuteFunctionBodyV1 extends JsonObject {
  readonly bodyId: string;
  readonly declarationId: string;
  readonly rootStatementId: string;
  readonly statements: readonly CppCuteStatementV1[];
  readonly expressions: readonly CppCuteExpressionV1[];
}

export type CppCuteIntegerExprV1 =
  | (JsonObject & { readonly kind: "integer"; readonly value: WireI64 })
  | (JsonObject & { readonly kind: "runtime"; readonly declarationId: string })
  | (JsonObject & { readonly kind: "add" | "multiply" | "minimum" | "maximum"; readonly values: readonly CppCuteIntegerExprV1[] })
  | (JsonObject & {
      readonly kind: "floor-divide" | "ceil-divide" | "modulo";
      readonly value: CppCuteIntegerExprV1;
      readonly divisor: CppCuteIntegerExprV1;
    });

export type CppCuteHierarchyV1 =
  | (JsonObject & { readonly kind: "scalar"; readonly value: CppCuteIntegerExprV1 })
  | (JsonObject & { readonly kind: "tuple"; readonly elements: readonly CppCuteHierarchyV1[] });

export interface CppCuteAffineLayoutFactV1 extends JsonObject {
  readonly factId: string;
  readonly kind: "affine-layout";
  readonly origin: CppCuteSourceOriginV1;
  readonly resultDeclarationId: string;
  readonly shape: CppCuteHierarchyV1;
  readonly stride: CppCuteHierarchyV1;
  readonly rank: number;
  readonly leafRank: number;
  readonly size: CppCuteIntegerExprV1;
  readonly cosize: CppCuteIntegerExprV1;
}

export type CppCuteTensorEngineV1 =
  | (JsonObject & { readonly kind: "global-pointer"; readonly pointerDeclarationId: string; readonly nullable: boolean })
  | (JsonObject & { readonly kind: "shared-pointer"; readonly pointerDeclarationId: string })
  | (JsonObject & { readonly kind: "register-array"; readonly arrayDeclarationId: string })
  | (JsonObject & { readonly kind: "pointer-array"; readonly pointerDeclarationIds: readonly string[] })
  | (JsonObject & { readonly kind: "indirect"; readonly engineDeclarationId: string });

export interface CppCuteTensorFactV1 extends JsonObject {
  readonly factId: string;
  readonly kind: "tensor";
  readonly origin: CppCuteSourceOriginV1;
  readonly resultDeclarationId: string;
  readonly elementTypeId: string;
  readonly layoutFactId: string;
  readonly engine: CppCuteTensorEngineV1;
  readonly memorySpace: CppCuteAddressSpaceV1;
}

export interface CppCuteIntrinsicEffectsV1 extends JsonObject {
  readonly readsMemory: boolean;
  readonly writesMemory: boolean;
  readonly synchronizes: boolean;
  readonly convergent: boolean;
}

export type CppCuteTargetIntrinsicOperationV1 =
  | (JsonObject & {
      readonly kind: "builtin-index";
      readonly scope: "grid" | "block";
      readonly axis: "x" | "y" | "z";
    })
  | (JsonObject & {
      readonly kind: "barrier";
      readonly scope: "subgroup" | "workgroup" | "cluster";
      readonly memorySemantics: "acquire-release" | "sequentially-consistent";
    })
  | (JsonObject & {
      readonly kind: "copy";
      readonly sourceSpace: CppCuteAddressSpaceV1;
      readonly destinationSpace: CppCuteAddressSpaceV1;
      readonly transferBits: number;
      readonly asynchronous: boolean;
    })
  | (JsonObject & {
      readonly kind: "mma";
      readonly m: number;
      readonly n: number;
      readonly k: number;
      readonly aTypeId: string;
      readonly bTypeId: string;
      readonly accumulatorTypeId: string;
    })
  | (JsonObject & {
      readonly kind: "pipeline";
      readonly action: "arrive" | "commit" | "wait" | "release";
      readonly scope: "workgroup" | "cluster";
    })
  | (JsonObject & { readonly kind: "capability"; readonly capabilityId: string });

export type CppCuteTargetIntrinsicAvailabilityV1 =
  | (JsonObject & { readonly kind: "portable-candidate" })
  | (JsonObject & { readonly kind: "requires-capability"; readonly capabilityIds: readonly string[] })
  | (JsonObject & { readonly kind: "recognized-unsupported"; readonly diagnosticId: string });

export interface CppCuteTargetIntrinsicFactV1 extends JsonObject {
  readonly factId: string;
  readonly kind: "target-intrinsic";
  readonly origin: CppCuteSourceOriginV1;
  readonly familyId: string;
  readonly operation: CppCuteTargetIntrinsicOperationV1;
  readonly operandExpressionIds: readonly string[];
  readonly resultTypeId: string | null;
  readonly effects: CppCuteIntrinsicEffectsV1;
  readonly availability: CppCuteTargetIntrinsicAvailabilityV1;
}

export type CppCuteResolvedFactV1 =
  | CppCuteAffineLayoutFactV1
  | CppCuteTensorFactV1
  | CppCuteTargetIntrinsicFactV1;

export type CppCuteFrontendEntryV1 =
  | (JsonObject & {
      readonly entryId: string;
      readonly kind: "layout";
      readonly layoutFactId: string;
      readonly selectedRootDeclarationIds: readonly string[];
    })
  | (JsonObject & {
      readonly entryId: string;
      readonly kind: "view-copy";
      readonly sourceTensorFactId: string;
      readonly destinationTensorFactId: string;
      readonly operationExpressionId: string;
      readonly selectedRootDeclarationIds: readonly string[];
    });

export type CppCuteDiagnosticPhaseV1 =
  | "preprocessing"
  | "parsing"
  | "name-lookup"
  | "overload-resolution"
  | "template-instantiation"
  | "cuda-sema"
  | "artifact-extraction";

export type CppCuteDiagnosticSubjectV1 =
  | (JsonObject & { readonly kind: "file"; readonly fileId: string })
  | (JsonObject & { readonly kind: "declaration"; readonly declarationId: string })
  | (JsonObject & { readonly kind: "type"; readonly typeId: string })
  | (JsonObject & { readonly kind: "expression"; readonly expressionId: string })
  | (JsonObject & { readonly kind: "fact"; readonly factId: string });

export interface CppCuteDiagnosticRelatedLocationV1 extends JsonObject {
  readonly spanId: string;
  readonly message: string;
}

export interface CppCuteFrontendDiagnosticV1 extends JsonObject {
  readonly diagnosticId: string;
  readonly phase: CppCuteDiagnosticPhaseV1;
  readonly severity: "remark" | "note" | "warning" | "error" | "fatal";
  readonly code: string;
  readonly renderedMessage: string;
  readonly primarySpanId: string;
  readonly subject: CppCuteDiagnosticSubjectV1;
  readonly parentDiagnosticId: string | null;
  readonly related: readonly CppCuteDiagnosticRelatedLocationV1[];
}

export type CppCuteFrontendOutcomeV1 =
  | (JsonObject & {
      readonly kind: "accepted";
      readonly selectedEntryIds: readonly string[];
    })
  | (JsonObject & {
      readonly kind: "rejected";
      readonly blockingDiagnosticIds: readonly string[];
    });

export interface CppCuteExtractionRecordV1 extends JsonObject {
  readonly profileHash: string;
  readonly inputClosureSha256: string;
  /** CUTE-002 requires unmodified source; v1 therefore requires this to stay empty. */
  readonly appliedTransforms: readonly never[];
}
