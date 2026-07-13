import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslValueType,
} from "@unlocalhosted/browsergrad-kernels";
import type {
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMatrixTileRef,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import {
  semanticInlineAsmLdmatrixAssignments,
  semanticPointerSymbolNeedsRuntimeState,
  walkSemanticOperations,
} from "./semantic_ir.js";
import {
  createBuiltinSemanticSymbolId,
  createGeneratedSemanticSymbolId,
  createSemanticSymbolId,
  createUnresolvedSemanticMemoryId,
  createUnresolvedSemanticSymbolId,
  semanticIdsEqual,
  semanticMemoryIdFromSymbol,
  semanticSymbolIdFromFunction,
} from "./semantic_ids.js";
import {
  isSemanticKernelIrOperation,
  semanticAtomicMemoryRootNames,
  semanticExpressionChildren,
} from "./semantic_ir_walk.js";
import type {
  CudaLiteDiagnostic,
  CudaLiteScalarType,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import { requireSemanticValueType } from "./semantic_value_type.js";
import { semanticPtxIntegerCallInfo } from "./semantic_inline_ptx.js";
import { promotedCudaScalarType } from "./wgsl_value_conversion.js";
import { sizeofCudaType } from "./type_layout.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { createWgslNameMap, safeWgslIdentifier } from "./wgsl_names.js";
import { isCudaBuiltinVectorSymbolName } from "./cuda_builtin_symbols.js";
import { emitBfloatConversionHelpers, emitCuComplexRobustMathHelpers, emitCurandHelpers, emitFp8Helpers, emitHalfConversionHelpers, emitSpecialFloatConstantHelpers } from "./wgsl_support_helpers.js";
import {
  semanticBf162CallArgumentsSupported,
  isSemanticHalf2BooleanComparisonCall,
  isSemanticHalf2ComparisonCall,
  isSemanticHalf2MaskComparisonCall,
  isSemanticFloatVectorType,
  semanticHalf2CallArgumentsSupported,
  semanticVectorAtCallSupported as semanticVectorAtCallContractSupported,
  semanticVectorConstructorCallSupported as semanticVectorConstructorCallContractSupported,
  semanticVectorLerpCallSupported as semanticVectorLerpCallContractSupported,
  semanticExpressionValueType,
  semanticExpressionVectorValueType,
} from "./semantic_vector_intrinsics.js";
import {
  SEMANTIC_CURAND_CALLS,
  semanticCurandArity,
  semanticCurandScalarArgumentIndices,
  semanticCurandStateArgumentIndex,
} from "./semantic_curand_intrinsics.js";
import { isSemanticGeneratedRandomCall } from "./semantic_generated_random_intrinsics.js";
import {
  SEMANTIC_ADDRESS_PREDICATE_CALLS,
  SEMANTIC_LOCAL_ARRAY_FILL_CALLS,
  SEMANTIC_NOOP_CALLS,
  SEMANTIC_SUBGROUP_CALLS,
  semanticAddressPredicateAddressSpace,
  semanticSubgroupScalarArguments,
} from "./semantic_builtin_calls.js";
import {
  isSemanticAtomicCallName,
  semanticAtomicOperation,
  semanticAtomicScalarArgumentIndices,
  semanticAtomicSupportsBfloatAdd,
  semanticAtomicSupportsFloat,
  semanticAtomicUsesF32Storage,
} from "./semantic_atomic_intrinsics.js";
import type { WgslLegalizedSemanticKernelIr } from "./wgsl_legalization.js";
import type { WgslLegalizedIrArtifact } from "./compiler_phases.js";
import {
  createTypedWgslIdentifier,
  createTypedWgslLiteral,
  createTypedWgslZero,
  createTypedWgslCall,
  createTypedWgslMemberAccess,
  createTypedWgslQualifiedAccess,
  createTypedWgslIndexAccess,
  createTypedWgslMemoryRead,
  createTypedWgslScalarMemoryRead,
  createTypedWgslMemoryPathRead,
  createTypedWgslIndexedPlace,
  createTypedWgslLocalPlace,
  createTypedWgslDereferencedPlace,
  createTypedWgslDereferencedIndexedPlace,
  createTypedWgslAddressOf,
  createTypedWgslPlaceRead,
  createTypedWgslPointerIndexRead,
  createTypedWgslBindingAddress,
  createTypedWgslTextureLoad,
  createTypedWgslTextureDescriptorRead,
  createTypedWgslCubemapTextureLoad,
  createTypedWgslAtomicCall,
  createTypedWgslBitcast,
  createTypedWgslConstructor,
  isWgslVectorType,
  isTypedWgslLiteralCode,
  convertTypedWgslExpression,
  legalizeTypedWgslBoolToNumeric,
  emitTypedWgslBinary,
  emitTypedWgslSelect,
  emitTypedWgslUnary,
  type TypedWgslExpression,
  type TypedWgslPlace,
  type WgslPointerType,
} from "./typed_wgsl_expression.js";
import {
  createTypedWgslLocalAssignmentStatement,
  createTypedWgslPlaceAssignmentStatement,
  createTypedWgslCallStatement,
  createTypedWgslReturnStatement,
  createTypedWgslVariableStatement,
} from "./typed_wgsl_statement.js";
import {
  cudaLiteFlatIndicesForDimensions as flatIndicesForDimensions,
  cudaLiteTotalElements as totalElements,
} from "./cuda_lite_values.js";
import { cudaAddressSpacePredicateKind } from "./cuda_pointer_calls.js";
import {
  isCudaComplexCallName,
  isCudaComplexConstructorCallName,
  isCudaComplexScalarCallName,
} from "./cuda_complex_intrinsics.js";
import {
  isMatrixTileByteValueType,
  matrixTileElementCount,
  type MatrixTileLayout,
  type MatrixTileResolvedSpec,
} from "./matrix_tiles.js";
import {
  cudaArithmeticReduceOpForCall,
  cudaVoteOpForCall,
  isCudaWarpReduceCallName,
} from "./cuda_subgroup_calls.js";
import { flattenSemanticInitializerExpressions as flattenInitializerExpressions } from "./semantic_initializers.js";
import {
  emitSemanticFlatArrayType,
  emitSemanticFlatLocalArrayIndexes,
  emitSemanticFlatRankedIndex,
  emitSemanticNestedArrayType,
} from "./semantic_wgsl_memory_layout.js";
import {
  semanticPointerBaseParamName,
  semanticPointerBufferParamName,
  semanticPointerReadHelperName,
  semanticPointerStorageCompatible,
  semanticPointerWriteHelperName,
  semanticStorageOffsetBaseNames,
  semanticStorageOffsetSymbol as storageOffsetSymbol,
  semanticStoragePointerBufferId,
  semanticWgslFunctionStoragePointerParam,
  semanticWgslFunctionSharedPointerParam,
} from "./semantic_wgsl_pointers.js";
import {
  wgslAtomicScalar,
  wgslBindingType,
  wgslScalar,
  wgslValueScalar,
  wgslValueType,
  wgslVectorScalar,
  zeroForType,
} from "./semantic_wgsl_types.js";
import {
  SEMANTIC_MATH_CALLS,
  semanticMathCallArgumentsSupported,
  semanticVectorMinMaxCallValueType,
} from "./semantic_math_intrinsics.js";
import {
  semanticAssignmentBinaryOperator,
  semanticAssignmentOperatorSupported as semanticWgslAssignmentOperatorSupported,
  semanticSurfaceReadValueType,
  semanticVectorAssignmentOperatorSupported,
  semanticVectorBinaryOperatorSupported as semanticWgslVectorBinaryOperatorSupported,
} from "./semantic_expression_contracts.js";
import {
  semanticTextureReadCoordinateShapeSupported,
  semanticTextureSurfaceValueTypeSupported,
} from "./semantic_texture_surface.js";
import {
  semanticBarrierOperationsMatchActiveLaneProof,
  semanticBarrierOperationsMatchUniformityProof,
  semanticBarrierFunctionNames,
  semanticBarrierShapeSupported,
  semanticOperationsContainBarrier,
} from "./semantic_barrier_contracts.js";
import {
  semanticFunctionArgSupported as semanticFunctionArgContractSupported,
  semanticFunctionBodyShapeSupported as semanticFunctionBodyShapeContractSupported,
  semanticFunctionForCall,
  semanticFunctionLocalParamValueTypesSupported,
  semanticFunctionParamContractSupported,
  semanticPointerFunctionBodySupported as semanticPointerFunctionBodyContractSupported,
} from "./semantic_function_calls.js";
import { semanticPointerArgumentMemoryRef as semanticPointerArgMemoryRef } from "./semantic_pointer_arguments.js";
import {
  semanticDirectByteStorageParamSupported,
  semanticDirectByteVectorMemberRef,
} from "./semantic_byte_storage.js";
import { semanticVectorMathCallSupported } from "./semantic_vector_math.js";
import {
  semanticLocalValueTypeSupported,
  semanticScalarValueTypeSupported,
  semanticStorageVectorFieldIndices,
  semanticStorageVectorType,
  semanticValueTypeSupported,
} from "./semantic_value_types.js";
import {
  semanticConstantMemorySymbols as constantMemorySymbols,
  semanticDeviceGlobalMemorySymbols as deviceGlobalMemorySymbols,
  semanticSharedMemorySymbols as sharedMemorySymbols,
  semanticSurfaceSymbols as surfaceSymbols,
  semanticTextureSymbols as textureSymbols,
  semanticUsesBfloatHelper,
  semanticUsesCuComplexRobustMath,
  semanticUsesCurand,
  semanticUsesFp8,
  semanticUsesGenericSurfaceRead,
  semanticUsesGenericSurfaceWrite,
  semanticUsesHalfConversion,
} from "./semantic_wgsl_usage.js";
import {
  emitSemanticBitwiseReduceHelper,
  emitSemanticBallotHelper,
  emitSemanticMatchAnyHelper,
  emitSemanticWarpShuffleHelper,
  legacyShuffleCall,
  legacyVoteCall,
  semanticBitwiseReduceHelper,
  semanticBitwiseReduceHelpers,
  semanticBitwiseReduceOpForCall,
  semanticBitwiseReduceScratchName,
  semanticBallotHelper,
  semanticBallotHelpers,
  semanticMatchAnyHelper,
  semanticMatchAnyHelpers,
  semanticMatchAnyScratchName,
  semanticShuffleOpForCall,
  semanticShuffleTileSize,
  semanticWarpShuffleHelper,
  semanticWarpShuffleHelpers,
  semanticWarpShuffleScratchName,
  semanticWorkgroupSize,
} from "./semantic_wgsl_subgroups.js";
import {
  emitSemanticCooperativeGroupCall,
  emitSemanticCooperativeReduceHelper,
  emitSemanticCooperativeScanHelper,
  emitSemanticCooperativeVectorReduceHelper,
  semanticCooperativeReduceHelperFor,
  semanticCooperativeReduceHelpers,
  semanticCooperativeReduceValue,
  semanticCooperativeScanHelperFor,
  semanticCooperativeScanHelpers,
  semanticCooperativeVectorReduceHelperFor,
  semanticCooperativeVectorReduceHelpers,
  semanticWgslCooperativeGroupCallSupported,
  semanticWgslCooperativeReduceCallSupported,
  semanticWgslCooperativeScanCallSupported,
} from "./semantic_wgsl_cooperative.js";
import {
  semanticCooperativeGroupInfo,
  semanticCooperativeGroupRankParamName,
  semanticCooperativeGroupSizeParamName,
} from "./semantic_cooperative_groups.js";
import { emitSemanticNumericHelpers } from "./semantic_wgsl_numeric_helpers.js";
import {
  emitSemanticSyncthreadsPredicateHelper,
  semanticSyncthreadsPredicateHelperFor,
  semanticSyncthreadsPredicateHelpers,
} from "./semantic_wgsl_sync.js";
import {
  halfConversionModeLiteral,
  wgslRoundBfloat16,
} from "./semantic_wgsl_packed_math.js";
import {
  collectSemanticTextureDescriptorSpecializations,
  emitSemanticTextureDescriptorHelper,
  semanticFunctionCallName,
  semanticOptionsWithTextureDescriptors,
  semanticSpecializedFunctionName,
  semanticTextureDescriptorHelperName,
  semanticTextureDescriptorHelpers,
  type SemanticTextureDescriptorOptions,
  type SemanticTextureDescriptorSpecializations,
} from "./semantic_wgsl_texture_descriptors.js";
import { emitCubeTextureAtlasHelpers } from "./wgsl_texture_surface.js";
import { semanticUniformLayout } from "./semantic_uniform_layout.js";
import { cudaVectorConstructorType, cudaVectorLaneCount, cudaVectorScalarType, cudaVectorSwizzleIndices, isCudaVectorType } from "./vector_types.js";
import {
  rewriteF16BindingsToF32,
  rewriteF16WgslToF32,
} from "./wgsl_feature_usage.js";
import {
  bfloatAtomicAddHelperName,
  emitBfloatAtomicAddHelper,
  emitFloatAtomicAddHelper,
  emitFloatAtomicMaxHelper,
  emitFloatAtomicMinHelper,
  emitFloatAtomicSubHelper,
  emitIntegerAtomicLoopHelpers,
  floatAtomicHelperName,
  integerAtomicLoopHelperName,
  wgslAtomicCalleeForCudaAtomic,
  wgslIntegerLoopAtomicKindForCudaAtomic,
  type WgslAtomicAddressSpace,
  type WgslIntegerLoopAtomicKind,
} from "./wgsl_atomic_helpers.js";

export interface SemanticKernelIrWgslOutput {
  readonly wgsl: string;
  readonly program: ReturnType<typeof defineWgslKernelProgram>;
}

export interface SemanticKernelIrWgslPreflightFailure {
  readonly message: string;
  readonly span: SourceSpan;
}

export interface EmitSemanticKernelIrWgslOptions extends SemanticTextureDescriptorOptions {
  readonly activeCollectivePredicate?: string;
  readonly activeFunction?: string;
  readonly workgroupUniformExpression?: boolean;
}

const UNIFORM_PARAMS_NAME = "bg_uniforms";
const PACKED_SHARED_U8_STORE = "bg_semantic_packed_shared_u8_store";
const PACKED_SHARED_U8_ADD = "bg_semantic_packed_shared_u8_add";
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const LOGICAL_OPERATORS = new Set(["&&", "||"]);
const TYPED_NATIVE_WGSL_MATH_CALLS = new Set([
  "sqrt", "inverseSqrt", "exp", "exp2", "log", "log2", "abs", "floor", "ceil", "trunc",
  "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "pow", "fma",
]);

export function canEmitSemanticKernelIrWgsl(
  legalized: WgslLegalizedIrArtifact<SemanticKernelIrModule>,
  _options: EmitSemanticKernelIrWgslOptions = {},
): boolean {
  return semanticKernelIrWgslPreflightFailureForIr(legalized.ir) === undefined;
}

export function semanticKernelIrWgslPreflightBlocker(
  legalized: WgslLegalizedIrArtifact<SemanticKernelIrModule>,
): string | undefined {
  return semanticKernelIrWgslPreflightFailureForIr(legalized.ir)?.message;
}

export function semanticKernelIrWgslPreflightFailure(
  legalized: WgslLegalizedIrArtifact<SemanticKernelIrModule>,
): SemanticKernelIrWgslPreflightFailure | undefined {
  return semanticKernelIrWgslPreflightFailureForIr(legalized.ir);
}

function semanticKernelIrWgslPreflightFailureForIr(
  ir: SemanticKernelIrModule,
): SemanticKernelIrWgslPreflightFailure | undefined {
  try {
    emitSemanticKernelIrWgslFromIr(ir);
    return undefined;
  } catch (error) {
    if (!(error instanceof CudaLiteCompilerError)) throw error;
    return {
      message: error.message,
      span: error.diagnostics[0]?.span ?? ir.span,
    };
  }
}

function semanticWgslLoweringConstraintFailure(
  ir: SemanticKernelIrModule,
): SemanticKernelIrWgslPreflightFailure | undefined {
  const gridBarrier = semanticGridBarrierInOperations(ir.operations) ??
    ir.functions.map((fn) => semanticGridBarrierInOperations(fn.body)).find((barrier) => barrier !== undefined);
  if (gridBarrier) return { message: "semantic WGSL does not support barrier", span: gridBarrier.span };
  if (!semanticWgslSharedBarrierShapeSupported(ir)) {
    return { message: "semantic WGSL does not support shared-memory barrier shape", span: ir.span };
  }
  const wideByteRef = semanticWidePackedByteRefInOperations(ir.operations) ??
    ir.functions.map((fn) => semanticWidePackedByteRefInOperations(fn.body)).find((ref) => ref !== undefined);
  if (wideByteRef) {
    return { message: "semantic WGSL supports packed byte views up to one 32-bit word", span: wideByteRef.span };
  }
  return undefined;
}

function semanticGridBarrierInOperations(
  operations: readonly SemanticKernelIrOperation[],
): Extract<SemanticKernelIrOperation, { readonly kind: "barrier" }> | undefined {
  for (const operation of operations) {
    if (operation.kind === "barrier" && operation.scope === "grid") return operation;
    const nested = operation.kind === "branch"
      ? [...operation.consequent, ...operation.alternate]
      : operation.kind === "loop"
        ? [...operation.body, ...(operation.continuing ?? [])]
        : operation.kind === "block"
          ? operation.body
          : [];
    const barrier = semanticGridBarrierInOperations(nested);
    if (barrier) return barrier;
  }
  return undefined;
}

function semanticWidePackedByteRefInOperations(
  operations: readonly SemanticKernelIrOperation[],
): SemanticMemoryRef | undefined {
  for (const operation of operations) {
    if (operation.kind === "store" && operation.target.addressSpace === "storage" &&
      operation.target.containerValueType === "uchar" && (operation.target.pointerBaseUnitBytes ?? 0) > 4) {
      return operation.target;
    }
    const nested = operation.kind === "branch"
      ? [...operation.consequent, ...operation.alternate]
      : operation.kind === "loop"
        ? [
            ...(operation.init && isSemanticKernelIrOperation(operation.init) ? [operation.init] : []),
            ...operation.body,
            ...(operation.continuing ?? []),
          ]
        : operation.kind === "block"
          ? operation.body
          : [];
    const nestedWide = semanticWidePackedByteRefInOperations(nested);
    if (nestedWide) return nestedWide;
  }
  return undefined;
}

function semanticWgslSharedBarrierShapeSupported(ir: SemanticKernelIrModule): boolean {
  const shared = sharedMemorySymbols(ir);
  const barrierFunctions = semanticBarrierFunctionNames(ir);
  const containsBarrier = semanticOperationsContainBarrier(ir.operations, barrierFunctions);
  if (shared.length === 0 && !containsBarrier) return true;
  if (!shared.every((symbol) => symbol.dimensions.length === 0 || symbol.dimensions.every((dimension) => dimension > 0))) return false;
  if (ir.functions.some(semanticWgslFunctionHasSharedPointer) && shared.some((symbol) => symbol.dimensions.length > 1)) return false;
  if (!containsBarrier) return operationsHaveNoBarrierOrControlTransfer(ir.operations);
  if (barrierFunctions.size === 0 && semanticDirectBarriersHaveAnalyzerProof(ir)) return true;
  if (!shared.some((symbol) => isSemanticFloatVectorType(symbol.valueType)) && barrierFunctions.size === 0) {
    const activeLaneLowered = semanticOperationsHaveActiveLaneDeclaration(ir.operations);
    return activeLaneLowered && (
      semanticBarrierShapeSupported(ir.operations, barrierFunctions) ||
      semanticBarrierOperationsMatchActiveLaneProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions)
    ) || operationsHaveOnlyTopLevelBarriers(ir.operations);
  }
  const activeLaneLowered = semanticOperationsHaveActiveLaneDeclaration(ir.operations);
  return (
    semanticBarrierShapeSupported(ir.operations, barrierFunctions) ||
    activeLaneLowered && semanticBarrierOperationsMatchActiveLaneProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions) ||
    semanticBarrierOperationsMatchUniformityProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions)
  ) && ir.functions.filter((fn) => barrierFunctions.has(fn.name)).every((fn) =>
    semanticBarrierShapeSupported(fn.body, barrierFunctions) ||
    semanticOperationsHaveActiveLaneDeclaration(fn.body) &&
      semanticBarrierOperationsMatchActiveLaneProof(fn.body, semanticBarrierFunctionProof(ir, fn.name), barrierFunctions) ||
    semanticBarrierOperationsMatchUniformityProof(fn.body, semanticBarrierFunctionProof(ir, fn.name), barrierFunctions)
  );
}

function semanticWgslFunctionHasSharedPointer(fn: SemanticKernelIrModule["functions"][number]): boolean {
  return fn.params.some((param) => param.pointer && param.addressSpace === "shared");
}

function semanticOperationsHaveActiveLaneDeclaration(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) => operation.kind === "declare" && (
    operation.target.name === "bg_active_lane" ||
    operation.target.name.startsWith("bg_barrier_loop_active_") ||
    operation.target.name.startsWith("bg_loop_active_")
  ));
}

function semanticBarrierFunctionProof(
  ir: SemanticKernelIrModule,
  name: string,
): SemanticKernelIrModule["barrierUniformity"]["kernel"] | undefined {
  return ir.barrierUniformity.functions[name] ??
    (name.endsWith("__bg_guarded_barrier")
      ? ir.barrierUniformity.functions[name.slice(0, -"__bg_guarded_barrier".length)]
      : undefined);
}

function semanticDirectBarriersHaveAnalyzerProof(ir: SemanticKernelIrModule): boolean {
  return semanticBarrierOperationsMatchUniformityProof(ir.operations, ir.barrierUniformity.kernel, semanticBarrierFunctionNames(ir));
}

function operationsHaveOnlyTopLevelBarriers(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => operation.kind !== "branch" && operation.kind !== "loop" && operation.kind !== "block");
}

function operationsHaveNoBarrierOrControlTransfer(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "barrier" || operation.kind === "return" || operation.kind === "break" || operation.kind === "continue") return false;
    if (operation.kind === "branch") {
      return operationsHaveNoBarrierOrControlTransfer(operation.consequent) &&
        operationsHaveNoBarrierOrControlTransfer(operation.alternate);
    }
    if (operation.kind === "block" || operation.kind === "loop") return operationsHaveNoBarrierOrControlTransfer(operation.body);
    return true;
  });
}

export function emitSemanticKernelIrWgsl(
  legalized: WgslLegalizedSemanticKernelIr,
  options: EmitSemanticKernelIrWgslOptions = {},
): SemanticKernelIrWgslOutput {
  return emitSemanticKernelIrWgslFromIr(legalized.ir, options);
}

function emitSemanticKernelIrWgslFromIr(
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions = {},
): SemanticKernelIrWgslOutput {
  const constraintFailure = semanticWgslLoweringConstraintFailure(ir);
  if (constraintFailure) throw semanticWgslError(constraintFailure.message, constraintFailure.span);
  const textureSpecializations = collectSemanticTextureDescriptorSpecializations(ir, options);
  const storageOffsetBases = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets);
  const rawNames = new Set(ir.params.map((param) => param.name));
  for (const base of storageOffsetBases) rawNames.add(storageOffsetSymbol(base));
  for (const operation of ir.operations) collectOperationNames(operation, rawNames);
  for (const fn of ir.functions) {
    rawNames.add(fn.name);
    for (const signature of textureSpecializations.get(fn.name)?.values() ?? []) {
      rawNames.add(semanticSpecializedFunctionName(fn.name, signature.key));
    }
    for (const param of fn.params) rawNames.add(param.name);
    for (const param of fn.params.filter((item) => item.pointer && (item.addressSpace === "storage" || item.addressSpace === "shared"))) {
      rawNames.add(semanticPointerBufferParamName(param.name));
      rawNames.add(semanticPointerBaseParamName(param.name));
    }
    for (const operation of fn.body) collectOperationNames(operation, rawNames);
  }
  const surfaces = surfaceSymbols(ir);
  for (const surface of surfaces) {
    rawNames.add(surfaceWidthField(surface.name));
    rawNames.add(surfaceHeightField(surface.name));
  }
  for (const param of ir.params) {
    if (param.pointer && options.pointerBaseOffsets?.[param.name] !== undefined) {
      rawNames.add(pointerBaseOffsetUniformName(param.name));
    }
  }
  const names = createWgslNameMap([...rawNames], [], ir.functions.map((fn) => fn.name));
  const initializedScalarConstants = constantMemorySymbols(ir).filter((symbol) => symbol.initialized && symbol.dimensions.length === 0);
  const initializedConstantArrays = constantMemorySymbols(ir).filter((symbol) => symbol.initialized && symbol.dimensions.length > 0);
  const uniformParams = [
    ...ir.params.filter((param) => param.addressSpace === "uniform").map((param) => ({ ...param, valueType: param.valueType ?? "float" as const })),
    ...constantMemorySymbols(ir)
      .filter((symbol) => !symbol.initialized && symbol.dimensions.length === 0 && !isSemanticFloatVectorType(symbol.valueType))
      .map((symbol) => ({ ...symbol, valueType: symbol.valueType ?? "float" as const })),
    ...surfaces.flatMap((surface) => [
      { name: surfaceWidthField(surface.name), valueType: "uint" as const, span: surface.span },
      { name: surfaceHeightField(surface.name), valueType: "uint" as const, span: surface.span },
    ]),
    ...ir.params.flatMap((param) =>
      param.pointer && options.pointerBaseOffsets?.[param.name] !== undefined
        ? [{ name: pointerBaseOffsetUniformName(param.name), valueType: "uint" as const, span: param.span }]
        : []
    ),
  ];
  const uniformLayout = semanticUniformLayout(uniformParams);
  const constantBuffers = constantMemorySymbols(ir).filter((symbol) => !symbol.initialized && (symbol.dimensions.length > 0 || isSemanticFloatVectorType(symbol.valueType)));
  const deviceGlobalBuffers = deviceGlobalMemorySymbols(ir);
  const textures = textureSymbols(ir);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  const atomicDeviceGlobals = semanticAtomicDeviceGlobalNames(ir.operations, ir.functions);
  const atomicShared = semanticAtomicSharedNames(ir.operations, ir.functions);
  const cooperativeReduceHelpers = semanticCooperativeReduceHelpers(ir);
  const cooperativeScanHelpers = semanticCooperativeScanHelpers(ir);
  const cooperativeVectorReduceHelpers = semanticCooperativeVectorReduceHelpers(ir);
  const syncthreadsPredicateHelpers = semanticSyncthreadsPredicateHelpers(ir);
  const warpShuffleHelpers = ir.subgroupMode === "scalar" ? [] : semanticWarpShuffleHelpers(ir);
  const matchAnyHelpers = ir.subgroupMode === "scalar" ? [] : semanticMatchAnyHelpers(ir);
  const bitwiseReduceHelpers = ir.subgroupMode === "scalar" ? [] : semanticBitwiseReduceHelpers(ir);
  const ballotHelpers = ir.subgroupMode === "scalar" ? [] : [...semanticBallotHelpers(ir)];
  if (cooperativeReduceHelpers.some((helper) => helper.partitioned) &&
    !ballotHelpers.some((helper) => helper.name === semanticBallotHelper().name)) {
    ballotHelpers.push(semanticBallotHelper());
  }
  const f16Mode = effectiveSemanticF16Mode(ir, options);
  const bindings: WgslKernelBindingInput[] = ir.params
    .filter((param) => param.addressSpace === "storage")
    .map((param, binding) => ({
      kind: "storage",
      name: param.name,
      valueType: wgslBindingType(param.valueType),
      access: param.constant ? "read" : "read_write",
      binding,
    }));
  for (const constant of constantBuffers) {
    bindings.push({
      kind: "storage",
      name: constant.name,
      valueType: wgslBindingType(constant.valueType),
      access: "read",
      binding: bindings.length,
    });
  }
  for (const global of deviceGlobalBuffers) {
    bindings.push({
      kind: "storage",
      name: global.name,
      valueType: wgslBindingType(global.valueType),
      access: "read_write",
      binding: bindings.length,
    });
  }
  for (const surface of surfaces) {
    bindings.push({
      kind: "storage",
      name: surface.name,
      valueType: "f32",
      access: "read_write",
      binding: bindings.length,
    });
  }
  for (const texture of textures) {
    bindings.push({
      kind: "texture2d",
      name: texture.name,
      valueType: "f32",
      binding: bindings.length,
    });
  }
  if (uniformParams.length > 0) {
    bindings.push({
      kind: "uniform",
      name: UNIFORM_PARAMS_NAME,
      byteLength: uniformLayout.byteLength,
      binding: bindings.length,
    });
  }

  const lines: string[] = ["// browsergrad-semantic-wgsl: direct semantic IR emission"];
  if (f16Mode === "native" && ir.requiredFeatures.includes("shader-f16")) lines.push("enable f16;");
  if (ir.requiredFeatures.includes("subgroups")) lines.push("enable subgroups;");
  for (const param of ir.params.filter((item) => item.addressSpace === "storage")) {
    const access = param.constant ? "read" : "read_write";
    const elementType = atomicStorage.has(param.name)
      ? `atomic<${wgslAtomicScalar(param.valueType)}>`
      : wgslBindingType(param.valueType);
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, param.name)}) var<storage, ${access}> ${nameFor(param.name, names)}: array<${elementType}>;`);
  }
  for (const constant of constantBuffers) {
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, constant.name)}) var<storage, read> ${nameFor(constant.name, names)}: array<${wgslBindingType(constant.valueType)}>;`);
  }
  for (const global of deviceGlobalBuffers) {
    const elementType = atomicDeviceGlobals.has(global.name)
      ? `atomic<${wgslAtomicScalar(global.valueType)}>`
      : wgslBindingType(global.valueType);
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, global.name)}) var<storage, read_write> ${nameFor(global.name, names)}: array<${elementType}>;`);
  }
  for (const surface of surfaces) {
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, surface.name)}) var<storage, read_write> ${nameFor(surface.name, names)}: array<f32>;`);
  }
  for (const texture of textures) {
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, texture.name)}) var ${nameFor(texture.name, names)}: texture_2d<f32>;`);
  }
  for (const helper of semanticTextureDescriptorHelpers(options, textureSpecializations, names)) {
    lines.push("", ...emitSemanticTextureDescriptorHelper(helper.textureName, helper.descriptor, names));
  }
  if ((ir.bindlessTextures?.length ?? 0) > 0) {
    lines.push("", ...emitSemanticBindlessTextureReadHelper(ir, names, options));
  }
  if (semanticUsesCubemapTextureRead(ir)) lines.push("", ...emitCubeTextureAtlasHelpers());
  lines.push("", ...emitSemanticNumericHelpers());
  if (semanticUsesSpecialFloatConstant(ir)) lines.push("", ...emitSpecialFloatConstantHelpers());
  if (semanticUsesGeneratedRandom(ir)) lines.push("", ...emitSemanticGeneratedRandomHelpers());
  if (semanticUsesBfloatHelper(ir)) {
    lines.push("", ...emitBfloatConversionHelpers());
  }
  if (semanticUsesHalfConversion(ir)) {
    lines.push("", ...emitHalfConversionHelpers());
  }
  const allAtomicOperations = [...ir.operations, ...ir.functions.flatMap((fn) => fn.body)];
  if (semanticUsesIntegerLoopAtomic(allAtomicOperations)) {
    lines.push("", ...emitIntegerAtomicLoopHelpers());
  }
  for (const helper of semanticFloatAtomicHelpers(
    allAtomicOperations,
    ir.functions.some((fn) => fn.params.some((param) => param.pointer && param.addressSpace === "storage" && semanticAtomicUsesF32Storage(param.valueType))),
  )) {
    lines.push("", ...helper);
  }
  if (semanticHasAtomicByteStorage(ir)) {
    lines.push("", ...emitSemanticSignedByteAtomicHelpers());
  }
  if (semanticUsesFp8(ir)) {
    lines.push("", ...emitFp8Helpers());
  }
  if (semanticUsesCurand(ir)) {
    lines.push("", ...emitCurandHelpers());
  }
  if (semanticUsesCuComplexRobustMath(ir)) {
    lines.push("", ...emitCuComplexRobustMathHelpers());
  }
  for (const helper of emitSemanticStoragePointerHelpers(ir, names)) {
    lines.push("", ...helper);
  }
  for (const constant of initializedScalarConstants) {
    lines.push(emitInitializedScalarConstant(constant, ir, names, options));
  }
  for (const constant of initializedConstantArrays) {
    lines.push(emitInitializedConstantArray(constant, ir, names));
  }
  if (semanticUsesGenericSurfaceRead(ir)) {
    lines.push("", ...emitSemanticGenericSurfaceReadHelper(surfaces, names));
  }
  if (semanticUsesGenericSurfaceWrite(ir)) {
    lines.push("", ...emitSemanticGenericSurfaceWriteHelper(surfaces, names));
  }
  for (const surface of surfaces) {
    lines.push("", ...emitSemanticSurfaceReadHelper(surface, names));
  }
  for (const helper of warpShuffleHelpers) {
    lines.push(`var<workgroup> ${semanticWarpShuffleScratchName(helper)}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of matchAnyHelpers) {
    lines.push(`var<workgroup> ${semanticMatchAnyScratchName(helper)}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of bitwiseReduceHelpers) {
    lines.push(`var<workgroup> ${semanticBitwiseReduceScratchName(helper)}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of ballotHelpers) {
    lines.push(`var<workgroup> ${helper.scratchName}: array<u32, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of cooperativeReduceHelpers) {
    lines.push(`var<workgroup> ${helper.scratchName}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of cooperativeScanHelpers) {
    lines.push(`var<workgroup> ${helper.scratchName}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of cooperativeVectorReduceHelpers) {
    lines.push(`var<workgroup> ${helper.scratchName}: array<${wgslValueType(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of syncthreadsPredicateHelpers) {
    lines.push(`var<workgroup> ${helper.scratchName}: array<u32, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const shared of sharedMemorySymbols(ir)) {
    lines.push(`var<workgroup> ${nameFor(shared.name, names)}: ${emitSharedType(shared, atomicShared.has(shared.name))};`);
  }
  if (sharedMemorySymbols(ir).some((shared) => shared.valueType === "uchar")) {
    lines.push("", ...emitSemanticPackedSharedByteHelpers());
  }
  if (uniformParams.length > 0) {
    lines.push("struct Params {");
    for (const param of uniformLayout.fields) {
      const valueType = wgslValueType(param.valueType);
      const size = valueType === "f16" ? "@size(4) " : "";
      lines.push(`  @align(${param.align}) ${size}${nameFor(param.name, names)}: ${valueType},`);
    }
    lines.push("};");
    lines.push(`@group(0) @binding(${bindings.length - 1}) var<uniform> ${UNIFORM_PARAMS_NAME}: Params;`);
  }
  for (const fn of ir.functions) {
    lines.push("", ...emitSemanticFunction(fn, ir, names, options, fn.name, textureSpecializations));
    for (const signature of textureSpecializations.get(fn.name)?.values() ?? []) {
      lines.push("", ...emitSemanticFunction(
        fn,
        ir,
        names,
        semanticOptionsWithTextureDescriptors(options, signature.descriptors),
        semanticSpecializedFunctionName(fn.name, signature.key),
        textureSpecializations,
      ));
    }
  }
  for (const helper of warpShuffleHelpers) {
    lines.push("", ...emitSemanticWarpShuffleHelper(helper, ir));
  }
  for (const helper of matchAnyHelpers) {
    lines.push("", ...emitSemanticMatchAnyHelper(helper, ir));
  }
  for (const helper of bitwiseReduceHelpers) {
    lines.push("", ...emitSemanticBitwiseReduceHelper(helper, ir));
  }
  for (const helper of ballotHelpers) {
    lines.push("", ...emitSemanticBallotHelper(helper, ir));
  }
  for (const helper of cooperativeReduceHelpers) {
    lines.push("", ...emitSemanticCooperativeReduceHelper(helper, ir));
  }
  for (const helper of cooperativeScanHelpers) {
    lines.push("", ...emitSemanticCooperativeScanHelper(helper, ir));
  }
  for (const helper of cooperativeVectorReduceHelpers) {
    lines.push("", ...emitSemanticCooperativeVectorReduceHelper(helper, ir));
  }
  for (const helper of syncthreadsPredicateHelpers) {
    lines.push("", ...emitSemanticSyncthreadsPredicateHelper(helper, ir));
  }
  lines.push(
    "",
    `@compute @workgroup_size(${ir.workgroupSize.join(", ")})`,
    "fn main(",
    "  @builtin(global_invocation_id) global_id: vec3<u32>,",
    "  @builtin(local_invocation_id) local_id: vec3<u32>,",
    "  @builtin(workgroup_id) workgroup_id: vec3<u32>,",
    "  @builtin(num_workgroups) num_workgroups: vec3<u32>",
    ") {",
    ...emitSemanticStorageOffsetDeclarations(ir, names, 1, options),
    ...emitSemanticOperations(ir.operations, ir, names, 1, false, options, textureSpecializations),
    "}",
  );
  const rawWgsl = lines.join("\n");
  const wgsl = f16Mode === "f32" ? rewriteF16WgslToF32(rawWgsl) : rawWgsl;
  const programBindings = f16Mode === "f32" ? rewriteF16BindingsToF32(bindings) : bindings;
  return {
    wgsl,
    program: defineWgslKernelProgram({
      name: ir.name,
      wgsl,
      bindings: programBindings,
      workgroupSize: ir.workgroupSize,
    }),
  };
}

function semanticUsesGeneratedRandom(ir: SemanticKernelIrModule): boolean {
  let used = false;
  const visit = (expression: SemanticExpression) => {
    if (expression.kind === "call" && expression.callee.kind === "symbol" && isSemanticGeneratedRandomCall(expression.callee.name)) used = true;
  };
  walkSemanticOperations(ir.operations, visit);
  for (const fn of ir.functions) walkSemanticOperations(fn.body, visit);
  return used;
}

function semanticUsesSpecialFloatConstant(ir: SemanticKernelIrModule): boolean {
  let used = false;
  const visit = (expression: SemanticExpression) => {
    if (expression.kind === "literal" && typeof expression.value === "number" && !Number.isFinite(expression.value)) used = true;
  };
  walkSemanticOperations(ir.operations, visit);
  for (const fn of ir.functions) walkSemanticOperations(fn.body, visit);
  return used;
}

function emitSemanticGeneratedRandomHelpers(): readonly string[] {
  return [
    "fn bg_random_uniform(state: ptr<function, u32>) -> f32 {",
    "  *state = (*state * 1664525u) + 1013904223u;",
    "  return f32(*state & 0x00ffffffu) / 16777216.0;",
    "}",
    "fn bg_random_normal(state: ptr<function, u32>) -> f32 {",
    "  return bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) - 3.0;",
    "}",
    "fn bg_random_poisson4(state: ptr<function, u32>) -> i32 {",
    "  return i32(bg_random_uniform(state) * 8.0);",
    "}",
  ];
}

function semanticWgslFunctionParamSupported(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): boolean {
  if (param.pointer && param.addressSpace === "shared" && param.valueType === "uchar" && param.pointerCarrierValueType === "uchar") return true;
  if (param.pointer && param.addressSpace === "storage" && param.valueType === "uchar") return true;
  if (!param.pointer && param.addressSpace === "local" && param.valueType === "uchar") return true;
  return semanticFunctionParamContractSupported(param, semanticWgslValueTypeSupported);
}

function effectiveSemanticF16Mode(
  ir: SemanticKernelIrModule,
  options: { readonly f16Mode?: "native" | "f32" },
): "native" | "f32" {
  if (options.f16Mode !== undefined) return options.f16Mode;
  return !ir.requiredFeatures.includes("shader-f16") && semanticIrUsesHalf(ir) ? "f32" : "native";
}

function semanticIrUsesHalf(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value === "half" || value === "half2";
  if (typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some(semanticIrUsesHalf);
  return Object.values(value as Record<string, unknown>).some(semanticIrUsesHalf);
}

function semanticWgslScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticScalarValueTypeSupported(valueType);
}

function semanticWgslValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticValueTypeSupported(valueType);
}

function semanticWgslLocalValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticLocalValueTypeSupported(valueType);
}

function semanticWgslAssignmentMemoryRefSupported(
  expression: SemanticExpression,
  ir?: SemanticKernelIrModule,
): boolean {
  const ref = semanticWgslAssignmentMemoryRef(expression, ir);
  return ref !== undefined &&
    (ir === undefined ? semanticWgslMemoryRefSupported(ref) : semanticWgslTypedMemoryRefSupported(ref, ir)) &&
    !isSemanticFloatVectorType(ref.valueType);
}

function semanticWgslAssignmentMemoryRef(
  expression: SemanticExpression,
  _ir?: SemanticKernelIrModule,
): SemanticMemoryRef | undefined {
  return memoryRefFromIndexExpression(expression);
}

function semanticWgslMemoryRefSupported(ref: SemanticMemoryRef, ir?: SemanticKernelIrModule): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "shared" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local") return false;
  if (ref.fields.length > 0) return semanticWgslVectorFieldMemoryRefSupported(ref);
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return semanticWgslScalarTypeSupported(ref.valueType);
  return ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar", ir));
}

function emitSemanticStoragePointerHelpers(
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly (readonly string[])[] {
  const types = semanticStoragePointerValueTypes(ir);
  return [...types].flatMap((type) => [
    emitSemanticStoragePointerReadHelper(type, ir, names),
    emitSemanticStoragePointerWriteHelper(type, ir, names),
    ...SEMANTIC_POINTER_ATOMIC_CALLS.flatMap((callee) => {
      const helper = emitSemanticStoragePointerAtomicHelper(callee, type, ir, names);
      return helper.length === 0 ? [] : [helper];
    }),
  ]);
}

function semanticStoragePointerValueTypes(ir: SemanticKernelIrModule): ReadonlySet<CudaLiteScalarType> {
  const types = new Set<CudaLiteScalarType>();
  for (const declaration of semanticLocalPointerDeclarations(ir)) {
    if (declaration.target.valueType !== undefined && semanticLocalPointerStorageRef(declaration) !== undefined) {
      types.add(declaration.target.valueType);
    }
  }
  for (const fn of ir.functions) {
    const pointerNames = new Set(fn.params
      .filter((param) => param.pointer && param.addressSpace === "storage")
      .map((param) => param.name));
    if (pointerNames.size === 0) continue;
    const add = (ref: SemanticMemoryRef): void => {
      if (pointerNames.has(ref.base) && ref.valueType !== undefined) types.add(ref.valueType);
    };
    for (const param of fn.params) {
      if (param.pointer && param.addressSpace === "storage" && param.valueType !== undefined) types.add(param.valueType);
    }
    walkSemanticOperations(fn.body, (expression) => {
      const ref = memoryRefFromIndexExpression(expression);
      if (ref) add(ref);
    });
    collectSemanticStoragePointerOperationRefs(fn.body, add);
  }
  return types;
}

function semanticLocalPointerDeclarations(
  ir: SemanticKernelIrModule,
): readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] {
  return [
    ...collectSemanticLocalPointerDeclarations(ir.operations),
    ...ir.functions.flatMap((fn) => collectSemanticLocalPointerDeclarations(fn.body)),
  ];
}

function collectSemanticLocalPointerDeclarations(
  operations: readonly SemanticKernelIrOperation[],
): readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] {
  const declarations: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] = [];
  for (const operation of operations) {
    if (operation.kind === "declare" && semanticPointerSymbolNeedsRuntimeState(operation.target)) declarations.push(operation);
    if (operation.kind === "block") declarations.push(...collectSemanticLocalPointerDeclarations(operation.body));
    if (operation.kind === "branch") {
      declarations.push(...collectSemanticLocalPointerDeclarations(operation.consequent));
      declarations.push(...collectSemanticLocalPointerDeclarations(operation.alternate));
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        declarations.push(...collectSemanticLocalPointerDeclarations([operation.init]));
      }
      declarations.push(...collectSemanticLocalPointerDeclarations(operation.body));
      if (operation.continuing) declarations.push(...collectSemanticLocalPointerDeclarations(operation.continuing));
    }
  }
  return declarations;
}

function semanticLocalPointerStorageRef(
  declaration: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
): SemanticMemoryRef | undefined {
  const ref = declaration.init ? semanticPointerArgMemoryRef(declaration.init) : undefined;
  return ref?.addressSpace === "storage" || ref?.addressSpace === "device-global" ? ref : undefined;
}

function semanticPointerDeclarationNeedsRuntimeState(
  declaration: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
): boolean {
  return semanticPointerSymbolNeedsRuntimeState(declaration.target);
}

function semanticLocalStoragePointerDeclaration(
  ir: SemanticKernelIrModule,
  expression: SemanticExpression,
): Extract<SemanticKernelIrOperation, { readonly kind: "declare" }> | undefined {
  if (expression.kind !== "symbol" || expression.addressSpace !== "local") return undefined;
  return semanticLocalPointerDeclarations(ir).find((operation) =>
    semanticIdsEqual(operation.target.id, expression.id) &&
    semanticPointerDeclarationNeedsRuntimeState(operation) &&
    semanticLocalPointerStorageRef(operation) !== undefined
  );
}

function collectSemanticStoragePointerOperationRefs(
  operations: readonly SemanticKernelIrOperation[],
  add: (ref: SemanticMemoryRef) => void,
): void {
  for (const operation of operations) {
    switch (operation.kind) {
      case "load":
        add(operation.source);
        break;
      case "store":
        add(operation.target);
        operation.reads.forEach(add);
        break;
      case "copy":
        add(operation.source);
        add(operation.target);
        break;
      case "atomic":
        if (operation.target) add(operation.target);
        break;
      case "call":
        operation.reads.forEach(add);
        break;
      case "pointer-rebind":
        add(operation.source);
        break;
      case "branch":
        collectSemanticStoragePointerOperationRefs(operation.consequent, add);
        collectSemanticStoragePointerOperationRefs(operation.alternate, add);
        break;
      case "loop":
        if (operation.init && isSemanticKernelIrOperation(operation.init)) {
          collectSemanticStoragePointerOperationRefs([operation.init], add);
        }
        collectSemanticStoragePointerOperationRefs(operation.body, add);
        if (operation.continuing) collectSemanticStoragePointerOperationRefs(operation.continuing, add);
        break;
      case "block":
        collectSemanticStoragePointerOperationRefs(operation.body, add);
        break;
      case "declare":
      case "dim3-declare":
      case "surface-write":
      case "surface-read-store":
      case "cooperative-group-declare":
      case "barrier":
      case "fence":
      case "inline-asm":
      case "expression":
      case "device-launch":
      case "return":
      case "continue":
      case "break":
        break;
    }
  }
}

const SEMANTIC_POINTER_ATOMIC_CALLS = [
  "atomicAdd", "atomicSub", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExch", "atomicCAS",
] as const;

function emitSemanticStoragePointerReadHelper(
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  return [
    `fn ${semanticPointerReadHelperName(valueType)}(buffer: u32, index: u32) -> ${wgslType} {`,
    "  switch buffer {",
    ...semanticStoragePointerBindings(ir).flatMap((binding) => {
      if (semanticAtomicBytePointerBindingCompatible(valueType, binding, atomicStorage)) {
        return [`    case ${binding.id}u: { return ${emitSemanticAtomicByteStorageReadValue(valueType, nameFor(binding.name, names), "index")}; }`];
      }
      return semanticPointerStorageCompatible(valueType, binding.valueType)
        ? [`    case ${binding.id}u: { return ${emitSemanticStoragePointerReadValue(valueType, nameFor(binding.name, names), "index", atomicStorage.has(binding.name))}; }`]
        : [];
    }),
    "    default: { return " + zeroForType(wgslType) + "; }",
    "  }",
    "}",
  ];
}

function emitSemanticStoragePointerWriteHelper(
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  return [
    `fn ${semanticPointerWriteHelperName(valueType)}(buffer: u32, index: u32, value: ${wgslType}) {`,
    "  switch buffer {",
    ...semanticStoragePointerBindings(ir).flatMap((binding) => {
      if (!binding.constant && semanticAtomicBytePointerBindingCompatible(valueType, binding, atomicStorage)) {
        return [`    case ${binding.id}u: { ${emitSemanticAtomicByteStorageWriteValue(valueType, nameFor(binding.name, names), "index", "value")} return; }`];
      }
      return !binding.constant && semanticPointerStorageCompatible(valueType, binding.valueType)
        ? [`    case ${binding.id}u: { ${emitSemanticStoragePointerWriteValue(valueType, nameFor(binding.name, names), "index", "value", atomicStorage.has(binding.name))} return; }`]
        : [];
    }),
    "    default: { return; }",
    "  }",
    "}",
  ];
}

function emitSemanticStoragePointerAtomicHelper(
  callee: typeof SEMANTIC_POINTER_ATOMIC_CALLS[number],
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  if (!semanticWgslPointerAtomicCallSupported(callee, valueType)) return [];
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  const op = semanticAtomicOperation(callee);
  const cas = op === "cas";
  return [
    `fn ${semanticPointerAtomicHelperName(callee, valueType)}(buffer: u32, index: u32, ${cas ? `compare: ${wgslType}, ` : ""}value: ${wgslType}) -> ${wgslType} {`,
    "  switch buffer {",
    ...semanticStoragePointerBindings(ir).flatMap((binding) => {
      if (!binding.constant && semanticAtomicBytePointerBindingCompatible(valueType, binding, atomicStorage)) {
        return [`    case ${binding.id}u: { return ${emitSemanticAtomicByteStorageAtomicValue(callee, valueType, nameFor(binding.name, names), "index", "compare", "value")}; }`];
      }
      return !binding.constant && atomicStorage.has(binding.name) && semanticPointerStorageCompatible(valueType, binding.valueType)
        ? [`    case ${binding.id}u: { return ${emitSemanticStoragePointerAtomicValue(callee, valueType, nameFor(binding.name, names), "index", "compare", "value")}; }`]
        : [];
    }),
    "    default: { return " + zeroForType(wgslType) + "; }",
    "  }",
    "}",
  ];
}

function semanticStoragePointerBindings(ir: SemanticKernelIrModule): readonly {
  readonly id: number;
  readonly name: string;
  readonly valueType?: CudaLiteScalarType;
  readonly constant: boolean;
}[] {
  return [
    ...ir.params.flatMap((param, index) => param.addressSpace === "storage"
      ? [{ id: index, name: param.name, ...(param.valueType === undefined ? {} : { valueType: param.valueType }), constant: param.constant ?? false }]
      : []),
    ...ir.memory.filter((symbol) => symbol.kind === "device-global").map((symbol, index) => ({
      id: ir.params.length + index,
      name: symbol.name,
      ...(symbol.valueType === undefined ? {} : { valueType: symbol.valueType }),
      constant: false,
    })),
  ];
}

function semanticHasAtomicByteStorage(ir: SemanticKernelIrModule): boolean {
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  return semanticStoragePointerBindings(ir).some((binding) =>
    binding.valueType === "uchar" && atomicStorage.has(binding.name)
  );
}

function semanticAtomicBytePointerBindingCompatible(
  valueType: CudaLiteScalarType,
  binding: { readonly name: string; readonly valueType?: CudaLiteScalarType },
  atomicStorage: ReadonlySet<string>,
): boolean {
  return binding.valueType === "uchar" && atomicStorage.has(binding.name) &&
    !isCudaVectorType(valueType) && sizeofCudaType(valueType) === 4;
}

function emitSemanticAtomicByteStorageReadValue(
  valueType: CudaLiteScalarType,
  storage: string,
  byteIndex: string,
): string {
  const loaded = `atomicLoad(&${storage}[(${byteIndex} >> 2u)])`;
  if (valueType === "int") return `bitcast<i32>(${loaded})`;
  if (valueType === "float" || valueType === "double") return `bitcast<f32>(${loaded})`;
  return loaded;
}

function emitSemanticAtomicByteStorageWriteValue(
  valueType: CudaLiteScalarType,
  storage: string,
  byteIndex: string,
  value: string,
): string {
  const bits = valueType === "uint" ? value : `bitcast<u32>(${value})`;
  return `atomicStore(&${storage}[(${byteIndex} >> 2u)], ${bits});`;
}

function emitSemanticAtomicByteStorageAtomicValue(
  callee: string,
  valueType: CudaLiteScalarType,
  storage: string,
  byteIndex: string,
  compare: string,
  value: string,
): string {
  const access = `${storage}[(${byteIndex} >> 2u)]`;
  if (valueType === "uint") return emitSemanticStoragePointerAtomicValue(callee, valueType, storage, `(${byteIndex} >> 2u)`, compare, value);
  if (valueType === "float" || valueType === "double") {
    return emitSemanticStoragePointerAtomicValue(callee, valueType, storage, `(${byteIndex} >> 2u)`, compare, value);
  }
  const op = semanticAtomicOperation(callee);
  if (op === "min" || op === "max") return `bg_atomic${op === "min" ? "Min" : "Max"}_storage_u32_as_i32(&${access}, ${value})`;
  if (op === "cas") return `bitcast<i32>(atomicCompareExchangeWeak(&${access}, bitcast<u32>(${compare}), bitcast<u32>(${value})).old_value)`;
  const wgslCallee = wgslAtomicCalleeForCudaAtomic(callee);
  return wgslCallee === undefined ? "0" : `bitcast<i32>(${wgslCallee}(&${access}, bitcast<u32>(${value})))`;
}

function emitSemanticSignedByteAtomicHelpers(): readonly string[] {
  const helper = (op: "Min" | "Max", compare: "min" | "max"): readonly string[] => [
    `fn bg_atomic${op}_storage_u32_as_i32(word: ptr<storage, atomic<u32>, read_write>, value: i32) -> i32 {`,
    "  var old_bits = atomicLoad(word);",
    "  loop {",
    "    let old_value = bitcast<i32>(old_bits);",
    `    let next_value = ${compare}(old_value, value);`,
    "    let result = atomicCompareExchangeWeak(word, old_bits, bitcast<u32>(next_value));",
    "    if (result.exchanged) { return old_value; }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
  return [...helper("Min", "min"), "", ...helper("Max", "max")];
}

function semanticPointerAtomicHelperName(callee: string, valueType: CudaLiteScalarType): string {
  if (semanticAtomicOperation(callee) === "cas") return `bg_ptr_atomicCompareExchange_${wgslValueScalar(valueType)}`;
  return `bg_ptr_${callee}_${wgslValueScalar(valueType)}`;
}

function semanticWgslPointerAtomicCallSupported(callee: string, valueType: CudaLiteScalarType): boolean {
  const op = semanticAtomicOperation(callee);
  if (semanticAtomicUsesF32Storage(valueType)) return op === "add" || op === "sub" || op === "min" || op === "max" || op === "exchange" || op === "cas";
  if (valueType === "int" || valueType === "uint") return op === "add" || op === "sub" || op === "min" || op === "max" || op === "and" || op === "or" || op === "xor" || op === "exchange" || op === "cas";
  return false;
}

function emitSemanticStoragePointerReadValue(valueType: CudaLiteScalarType, storage: string, index: string, atomic: boolean): string {
  if (!isCudaVectorType(valueType)) return emitSemanticStoragePointerReadScalarValue(valueType, `${storage}[${index}]`, atomic);
  const laneCount = cudaVectorLaneCount(valueType);
  const scalar = wgslVectorScalar(valueType);
  return `${wgslValueType(valueType)}(${Array.from({ length: laneCount }, (_, lane) =>
    `${scalar}(${emitSemanticStoragePointerReadScalarValue(cudaVectorScalarType(valueType) ?? valueType, `${storage}[(${index} + ${lane}u)]`, atomic)})`
  ).join(", ")})`;
}

function emitSemanticStoragePointerReadScalarValue(valueType: CudaLiteScalarType, access: string, atomic: boolean): string {
  if (!atomic) return access;
  const loaded = `atomicLoad(&${access})`;
  return semanticAtomicUsesF32Storage(valueType) ? `bitcast<f32>(${loaded})` : loaded;
}

function emitSemanticStoragePointerWriteValue(valueType: CudaLiteScalarType, storage: string, index: string, value: string, atomic: boolean): string {
  if (!isCudaVectorType(valueType)) return emitSemanticStoragePointerWriteScalarValue(valueType, `${storage}[${index}]`, value, atomic);
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    emitSemanticStoragePointerWriteScalarValue(cudaVectorScalarType(valueType) ?? valueType, `${storage}[(${index} + ${lane}u)]`, `(${value}).${["x", "y", "z", "w"][lane]}`, atomic)
  ).join(" ");
}

function emitSemanticStoragePointerWriteScalarValue(valueType: CudaLiteScalarType, access: string, value: string, atomic: boolean): string {
  if (!atomic) return `${access} = ${value};`;
  const stored = semanticAtomicUsesF32Storage(valueType) ? `bitcast<u32>(${value})` : value;
  return `atomicStore(&${access}, ${stored});`;
}

function emitSemanticStoragePointerAtomicValue(
  callee: string,
  valueType: CudaLiteScalarType,
  storage: string,
  index: string,
  compare: string,
  value: string,
): string {
  const op = semanticAtomicOperation(callee);
  if (valueType === "float" || valueType === "double") {
    if (op === "exchange") return `bitcast<f32>(atomicExchange(&${storage}[${index}], bitcast<u32>(${value})))`;
    if (op === "cas") return `bitcast<f32>(atomicCompareExchangeWeak(&${storage}[${index}], bitcast<u32>(${compare}), bitcast<u32>(${value})).old_value)`;
    const kind = semanticWgslFloatAtomicCallKind(callee);
    return kind === "Add" || kind === "Sub" || kind === "Min" || kind === "Max"
      ? `${floatAtomicHelperName(kind, "storage")}(&${storage}[${index}], ${value})`
      : "0.0";
  }
  const wgslCallee = wgslAtomicCalleeForCudaAtomic(callee);
  if (wgslCallee === "atomicCompareExchangeWeak") return `atomicCompareExchangeWeak(&${storage}[${index}], ${compare}, ${value}).old_value`;
  return wgslCallee === undefined ? "0" : `${wgslCallee}(&${storage}[${index}], ${value})`;
}

function semanticWgslTypedMemoryRefSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (!semanticWgslMemoryRefSupported(ref, ir)) return false;
  if (semanticWgslLocalPackedHalfView(ref, ir)) return true;
  if (semanticWgslLocalPackedHalf2View(ref, ir)) return true;
  if (semanticWgslLocalScalarBitViewRootType(ref, ir) !== undefined) return true;
  if (semanticWgslLocalVectorBitViewRootType(ref, ir) !== undefined) return true;
  if (semanticWgslSharedScalarBitViewRootType(ref, ir) !== undefined) return true;
  if (semanticWgslSharedVectorBitViewRootType(ref, ir) !== undefined) return true;
  if (semanticWgslLocalPackedByteRawView(ref, ir)) return true;
  if (semanticWgslPackedSharedByteRoot(ref, ir)) return semanticPackedSharedByteViewSupported(ref.valueType);
  if (semanticWgslSharedHalfBitView(ref, ir)) return true;
  if (ref.addressSpace === "shared" && semanticWgslFunctionSharedPointerParam(ir, ref.base)) return true;
  if (ref.addressSpace === "local" && semanticWgslFunctionLocalPointerParam(ir, ref.base)) return true;
  if (ref.addressSpace === "local" && semanticLocalPointerDeclarations(ir).some((operation) =>
    operation.target.name === ref.base && operation.target.pointerRuntimeState === true)) return true;
  if (semanticWgslVectorFieldMemoryRefSupported(ref)) return true;
  if (semanticWgslLocalVectorLaneRefSupported(ref, ir)) return true;
  if (semanticWgslLocalScalarVectorView(ref, ir)) return true;
  if (semanticWgslSharedScalarVectorView(ref, ir)) return true;
  if (semanticWgslSharedVectorScalarView(ref, ir)) return true;
  if (ref.addressSpace !== "local" && ref.addressSpace !== "shared") return true;
  const symbol = ir.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace) ??
    (ref.addressSpace === "local" ? semanticFunctionLocalArraySymbol(ir, ref.base) : undefined);
  return symbol !== undefined && symbol.valueType === ref.valueType;
}

function semanticWgslSharedHalfBitView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  return ref.addressSpace === "shared" && ref.fields.length === 0 && ref.indices.length === 1 &&
    (ref.valueType === "float" || ref.valueType === "uint" || ref.valueType === "int") &&
    sharedMemorySymbols(ir).some((symbol) => symbol.name === ref.base && symbol.valueType === "half");
}

function semanticWgslLocalScalarBitViewRootType(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): CudaLiteScalarType | undefined {
  if (ref.addressSpace !== "local" || ref.fields.length > 0 || ref.indices.length !== 1) return undefined;
  if (ref.valueType !== "float" && ref.valueType !== "uint" && ref.valueType !== "int") return undefined;
  const root = localArraySymbol(ir, ref.base);
  if (root?.valueType !== "float" && root?.valueType !== "uint" && root?.valueType !== "int") return undefined;
  return root.valueType === ref.valueType ? undefined : root.valueType;
}

function semanticWgslSharedScalarBitViewRootType(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): CudaLiteScalarType | undefined {
  if (ref.addressSpace !== "shared" || ref.fields.length > 0 || ref.indices.length !== 1) return undefined;
  if (ref.valueType !== "float" && ref.valueType !== "uint" && ref.valueType !== "int") return undefined;
  const root = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base && symbol.dimensions.length > 0);
  if (root?.valueType !== "float" && root?.valueType !== "uint" && root?.valueType !== "int") return undefined;
  return root.valueType === ref.valueType ? undefined : root.valueType;
}

function semanticWgslSharedVectorBitViewRootType(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): CudaLiteScalarType | undefined {
  if (ref.addressSpace !== "shared" || ref.fields.length > 0 || ref.indices.length !== 1 || !isCudaVectorType(ref.valueType)) return undefined;
  const valueScalar = cudaVectorScalarType(ref.valueType);
  const root = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base && symbol.dimensions.length > 0);
  if (!valueScalar || !root?.valueType || isCudaVectorType(root.valueType)) return undefined;
  return root.valueType !== valueScalar && sizeofCudaType(root.valueType) === sizeofCudaType(valueScalar)
    ? root.valueType
    : undefined;
}

function semanticWgslLocalVectorBitViewRootType(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): CudaLiteScalarType | undefined {
  if (ref.addressSpace !== "local" || ref.fields.length > 0 || ref.indices.length !== 1 || ref.valueType === undefined) return undefined;
  const rootType = ref.containerValueType ?? semanticDeclaredLocalVectorType(ir, ref.base);
  const rootScalar = rootType === undefined ? undefined : cudaVectorScalarType(rootType);
  if (!rootScalar || rootScalar === ref.valueType || sizeofCudaType(rootScalar) !== sizeofCudaType(ref.valueType)) return undefined;
  return rootScalar;
}

function semanticWgslLocalPackedHalfView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  const root = localArraySymbol(ir, ref.base);
  return ref.addressSpace === "local" && ref.pointerBaseIsScalarLane === true && ref.valueType === "half" && root?.valueType === "uint";
}

function semanticWgslLocalPackedHalf2View(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  const root = localArraySymbol(ir, ref.base);
  return ref.addressSpace === "local" && ref.fields.length === 0 && ref.indices.length === 1 && ref.valueType === "half2" && root?.valueType === "uint";
}

function semanticWgslPackedSharedByteRoot(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (ref.addressSpace !== "shared") return false;
  return sharedMemorySymbols(ir).some((symbol) => symbol.name === ref.base && symbol.valueType === "uchar") ||
    ir.functions.some((fn) => fn.params.some((param) =>
      param.name === ref.base && param.pointer && param.addressSpace === "shared" && param.pointerCarrierValueType === "uchar"
    ));
}

function semanticPackedSharedByteViewSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "uchar" || valueType === "uint" || valueType === "int" || valueType === "float" ||
    valueType === "half" || valueType === "bf16" || valueType === "half2" || valueType === "bf162";
}

function semanticWgslVectorFieldMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "device-global" && ref.addressSpace !== "shared" && ref.addressSpace !== "local") return false;
  if (ref.fields.length !== 1) return false;
  const lanes = semanticStorageVectorFieldIndices(ref.containerValueType, ref.fields[0]!);
  if (lanes === undefined || new Set(lanes).size !== lanes.length) return false;
  return ref.indices.length > 0 && ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar"));
}

function semanticWgslLocalVectorLaneRefSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  return ref.addressSpace === "local" &&
    ref.fields.length === 0 &&
    ref.indices.length === 1 &&
    !isSemanticFloatVectorType(ref.valueType) &&
    isSemanticFloatVectorType(semanticDeclaredLocalVectorType(ir, ref.base)) &&
    ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar", ir));
}

function semanticWgslLocalScalarVectorView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  const valueType = ref.valueType;
  if (ref.addressSpace !== "local" || ref.fields.length > 0 || ref.indices.length !== 1 || !valueType || !isSemanticFloatVectorType(valueType)) return false;
  const scalar = cudaVectorScalarType(valueType);
  const local = localArraySymbol(ir, ref.base);
  return scalar !== undefined && local?.valueType === scalar;
}

function semanticDeclaredLocalVectorType(ir: SemanticKernelIrModule, name: string): CudaLiteScalarType | undefined {
  for (const operation of [...ir.operations, ...ir.functions.flatMap((fn) => fn.body)]) {
    const valueType = semanticDeclaredLocalVectorTypeInOperation(operation, name);
    if (valueType !== undefined) return valueType;
  }
  return undefined;
}

function semanticDeclaredLocalVectorTypeInOperation(
  operation: SemanticKernelIrOperation,
  name: string,
): CudaLiteScalarType | undefined {
  if (
    operation.kind === "declare" &&
    operation.target.addressSpace === "local" &&
    operation.target.name === name &&
    operation.target.dimensions.length === 0 &&
    isSemanticFloatVectorType(operation.target.valueType)
  ) return operation.target.valueType;
  if (operation.kind === "branch") {
    for (const child of [...operation.consequent, ...operation.alternate]) {
      const valueType = semanticDeclaredLocalVectorTypeInOperation(child, name);
      if (valueType !== undefined) return valueType;
    }
  }
  if (operation.kind === "loop") {
    if (operation.init && isSemanticKernelIrOperation(operation.init)) {
      const valueType = semanticDeclaredLocalVectorTypeInOperation(operation.init, name);
      if (valueType !== undefined) return valueType;
    }
    for (const child of operation.body) {
      const valueType = semanticDeclaredLocalVectorTypeInOperation(child, name);
      if (valueType !== undefined) return valueType;
    }
  }
  if (operation.kind === "block") {
    for (const child of operation.body) {
      const valueType = semanticDeclaredLocalVectorTypeInOperation(child, name);
      if (valueType !== undefined) return valueType;
    }
  }
  return undefined;
}

function semanticWgslStorageOffsetStoreSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
): boolean {
  return operation.target.addressSpace === "storage" &&
    operation.target.indices.length === 0 &&
    operation.target.fields.length === 0 &&
    (operation.operator === "+=" || operation.operator === "-=") &&
    ir.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage") &&
    semanticWgslExpressionSupported(operation.value, "scalar");
}

function semanticWgslVectorMemberSupported(
  expression: Extract<SemanticExpression, { kind: "member" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (ir && semanticDirectByteVectorMemberRef(expression, ir)) return true;
  const valueType = semanticExpressionValueType(expression.object);
  return semanticWgslExpressionSupported(expression.object, "any", ir) &&
    semanticStorageVectorFieldIndices(valueType, expression.property) !== undefined;
}

function semanticWgslVectorIndexSupported(
  expression: Extract<SemanticExpression, { kind: "index" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  const ref = memoryRefFromIndexExpression(expression);
  if (ref && ir && ref.addressSpace === "local" && semanticWgslFunctionLocalPointerParam(ir, ref.base)) return false;
  if (ref && ir && (semanticWgslLocalScalarVectorView(ref, ir) || semanticWgslLocalVectorBitViewRootType(ref, ir) !== undefined)) return false;
  if (ref && !(ref.addressSpace === "local" && isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.target, ir?.functions)))) return false;
  return isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.target, ir?.functions)) &&
    semanticWgslExpressionSupported(expression.target, "any", ir) &&
    semanticWgslExpressionSupported(expression.index, "scalar", ir);
}

function semanticWgslMathCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any" = "scalar",
  ir?: SemanticKernelIrModule,
): boolean {
  const name = expression.callee.kind === "symbol" ? expression.callee.name : undefined;
  if (semanticVectorMinMaxCallValueType(name, expression.args) !== undefined) {
    return expected === "any" && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "any", ir));
  }
  return semanticMathCallArgumentsSupported(
    name,
    expression.args,
    (arg) => semanticWgslExpressionSupported(arg, "scalar", ir),
  );
}

function semanticWgslCurandCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_CURAND_CALLS.has(expression.callee.name)) return false;
  const stateIndex = semanticCurandStateArgumentIndex(expression.callee.name);
  return stateIndex !== undefined &&
    expression.args.length === semanticCurandArity(expression.callee.name) &&
    semanticCurandStateAddressSpace(expression.args[stateIndex]!) !== undefined &&
    semanticCurandScalarArgumentIndices(expression.callee.name)
      .every((index) => semanticWgslExpressionSupported(expression.args[index]!, "scalar", ir));
}

function semanticWgslGeneratedRandomCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): boolean {
  return expression.callee.kind === "symbol" &&
    isSemanticGeneratedRandomCall(expression.callee.name) &&
    expression.args.length === 1 &&
    expression.args[0]?.kind === "unary" &&
    expression.args[0].operator === "&" &&
    expression.args[0].argument.kind === "symbol" &&
    expression.args[0].argument.addressSpace === "local" &&
    expression.args[0].argument.valueType === "uint";
}

function semanticWgslSubgroupCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace === "function" || !SEMANTIC_SUBGROUP_CALLS.has(expression.callee.name) ||
    ir?.requiredFeatures.includes("subgroups") !== true && ir?.subgroupMode !== "scalar") return false;
  if (expression.callee.name === "bg_subgroup_add" && ir?.subgroupMode !== "scalar") return false;
  const scalarArgs = semanticSubgroupScalarArguments(expression.callee.name, expression.args);
  if (scalarArgs === undefined || !scalarArgs.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir))) return false;
  if (semanticBitwiseReduceOpForCall(expression.callee.name)) {
    const value = expression.args[1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    return valueType === "int" || valueType === "uint";
  }
  return true;
}

function semanticWgslAddressPredicateCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  return expression.callee.kind === "symbol" &&
    SEMANTIC_ADDRESS_PREDICATE_CALLS.has(expression.callee.name) &&
    expression.args.length === 1 &&
    semanticAddressPredicateAddressSpace(expression.args[0]) !== undefined;
}

function semanticWgslTextureReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const texture = expression.texture;
  return semanticTextureSurfaceValueTypeSupported(expression.valueType) &&
    (texture.kind === "symbol" && texture.addressSpace === "texture" ||
      (ir.bindlessTextures?.length ?? 0) > 0 && expression.callee !== "texCubemap" && expression.z === undefined &&
        semanticWgslExpressionSupported(texture, "scalar", ir)) &&
    semanticWgslExpressionSupported(expression.x, "scalar", ir) &&
    semanticWgslExpressionSupported(expression.y, "scalar", ir) &&
    semanticTextureReadCoordinateShapeSupported(expression.callee, expression.z !== undefined) &&
    (expression.z === undefined || semanticWgslExpressionSupported(expression.z, "scalar", ir));
}

function semanticWgslSurfaceReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const target = expression.surface;
  return semanticTextureSurfaceValueTypeSupported(expression.valueType) &&
    target.kind === "symbol" &&
    target.addressSpace === "surface" &&
    semanticWgslExpressionSupported(expression.xBytes, "scalar", ir) &&
    semanticWgslExpressionSupported(expression.y, "scalar", ir) &&
    (expression.z === undefined || semanticWgslExpressionSupported(expression.z, "scalar", ir));
}

function semanticWgslFunctionCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const callee = expression.callee.name;
  const fn = ir.functions.find((item) => item.name === callee);
  if (!fn || !semanticWgslLocalValueTypeSupported(fn.returnType)) return false;
  if (fn.params.some((param) => !semanticWgslFunctionParamSupported(param))) return false;
  if (fn.params.some((param) => param.pointer && param.addressSpace !== "constant") && !semanticWgslPointerFunctionBodySupported(fn)) return false;
  if (!semanticFunctionLocalParamValueTypesSupported(fn, semanticWgslLocalValueTypeSupported)) return false;
  if (!semanticWgslFunctionBodyShapeSupported(fn.body, semanticWgslFunctionHasAtomicPointer(fn))) return false;
  return expression.args.length === fn.params.length &&
    expression.args.every((arg, index) => semanticWgslFunctionArgSupported(arg, fn.params[index], ir));
}

function semanticWgslFunctionArgSupported(
  arg: SemanticExpression,
  param: SemanticKernelIrModule["functions"][number]["params"][number] | undefined,
  ir: SemanticKernelIrModule,
): boolean {
  if (param?.cooperativeGroupKind !== undefined) return arg.kind === "symbol";
  if (param?.pointer && param.addressSpace === "storage" && param.valueType === "uchar") {
    const ref = semanticPointerArgMemoryRef(arg);
    return ref?.addressSpace === "storage" && semanticDirectByteStorageParamSupported(ir, ref.base);
  }
  return semanticFunctionArgContractSupported(arg, param, semanticPointerArgMemoryRef, (item, mode) => semanticWgslExpressionSupported(item, mode, ir));
}

function semanticWgslVectorConstructorSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any",
  ir?: SemanticKernelIrModule,
): boolean {
  return semanticVectorConstructorCallContractSupported(
    expression.callee.kind === "symbol" ? expression.callee.name : undefined,
    expression.args,
    expected,
    (arg, mode) => semanticWgslExpressionSupported(arg, mode, ir),
  );
}

function semanticWgslVectorAtCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  return semanticVectorAtCallContractSupported(
    expression.callee.kind === "symbol" ? expression.callee.name : undefined,
    expression.args,
    (arg) => semanticExpressionVectorValueType(arg, ir?.functions),
    (arg, mode) => semanticWgslExpressionSupported(arg, mode, ir),
  );
}

function semanticWgslVectorLerpCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  return semanticVectorLerpCallContractSupported(
    expression.callee.kind === "symbol" ? expression.callee.name : undefined,
    expression.args,
    (arg) => semanticExpressionVectorValueType(arg, ir?.functions),
    (arg, mode) => semanticWgslExpressionSupported(arg, mode, ir),
  );
}

function semanticWgslHalf2CallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  return semanticHalf2CallArgumentsSupported(
    expression.callee.name,
    expression.args,
    (arg) => semanticExpressionVectorValueType(arg, ir?.functions),
    (arg, expected) => semanticWgslExpressionSupported(arg, expected, ir),
  );
}

function semanticWgslBf162CallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  return semanticBf162CallArgumentsSupported(
    expression.callee.name,
    expression.args,
    (arg) => semanticExpressionVectorValueType(arg, ir?.functions),
    (arg, expected) => semanticWgslExpressionSupported(arg, expected, ir),
  );
}

function semanticWgslFunctionBodyShapeSupported(
  operations: readonly SemanticKernelIrOperation[],
  allowAtomic = false,
): boolean {
  return semanticFunctionBodyShapeContractSupported(operations, { allowBlock: true, allowBarrierFence: true, allowAtomic, allowSharedMemory: true, allowLocalArrays: true });
}

function semanticWgslFunctionHasAtomicPointer(fn: SemanticKernelIrModule["functions"][number]): boolean {
  return fn.params.some((param) => param.pointer && (param.addressSpace === "shared" || param.addressSpace === "storage"));
}

function semanticWgslPointerFunctionBodySupported(fn: SemanticKernelIrModule["functions"][number]): boolean {
  return semanticPointerFunctionBodyContractSupported(fn, memoryRefFromIndexExpression, semanticAtomicCallTarget, {
    allowCooperativeOps: true,
    allowSharedMemory: true,
    allowDeviceGlobals: true,
    allowLocalArrays: true,
    allowConstantMemory: true,
    allowStoragePointerIdentity: true,
  });
}

function semanticWgslAtomicCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const atomicOp = semanticAtomicOperation(expression.callee.name);
  if (!atomicOp) return false;
  const target = semanticAtomicCallTarget(expression);
  if (!target || (target.addressSpace !== "storage" && target.addressSpace !== "device-global" && target.addressSpace !== "shared" && target.addressSpace !== "local")) return false;
  if (!semanticWgslAtomicMemoryRefSupported(target, ir)) return false;
  if (!semanticWgslPointerAtomicSupported(expression.callee.name, target, ir)) return false;
  if (target.addressSpace === "storage" && target.indices.length !== 1 && !semanticWgslFunctionStoragePointerParam(ir, target.base)) return false;
  if (target.fields.length > 0) return false;
  if (!semanticWgslAtomicValueTypeSupported(expression.callee.name, target.valueType)) return false;
  if (!semanticWgslAtomicTargetRootSupported(target, ir)) return false;
  const scalarArgIndices = semanticAtomicScalarArgumentIndices(atomicOp);
  return expression.args.length >= scalarArgIndices.length + 1 &&
    scalarArgIndices.every((index) => semanticWgslExpressionSupported(expression.args[index]!, "scalar", ir));
}

function semanticWgslPointerAtomicSupported(
  callee: string,
  target: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticWgslFunctionStoragePointerParam(ir, target.base) === undefined ||
    semanticWgslPointerAtomicCallSupported(callee, target.valueType ?? "float");
}

function semanticWgslAtomicMemoryRefSupported(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticWgslMemoryRefSupported(ref, ir) ||
    ref.addressSpace === "storage" &&
      ref.indices.length === 0 &&
      semanticWgslFunctionStoragePointerParam(ir, ref.base) !== undefined;
}

function semanticWgslVoidFunctionCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticWgslVoidFunctionCallFailure(operation, ir) === undefined;
}

function semanticWgslVoidFunctionCallFailure(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): string | undefined {
  if (operation.result !== undefined) return "call result requires value-call lowering";
  const fn = semanticFunctionForCall(operation, ir.functions);
  if (!fn) return `unknown function '${operation.callee}'`;
  if (fn.returnType !== "void") return `function '${operation.callee}' does not return void`;
  const unsupportedParam = fn.params.find((param) => !semanticWgslFunctionParamSupported(param));
  if (unsupportedParam) return `unsupported parameter '${unsupportedParam.name}'`;
  if (fn.params.some((param) => param.pointer) && !semanticWgslPointerFunctionBodySupported(fn)) return `unsupported pointer body in '${fn.name}'`;
  if (!semanticFunctionLocalParamValueTypesSupported(fn, semanticWgslLocalValueTypeSupported)) return `unsupported local parameter type in '${fn.name}'`;
  if (operation.args.length !== fn.params.length) return `arity mismatch for '${fn.name}'`;
  const unsupportedArg = operation.args.findIndex((arg, index) => !semanticWgslFunctionArgSupported(arg, fn.params[index], ir));
  if (unsupportedArg >= 0) return `unsupported argument ${unsupportedArg + 1} for '${fn.name}'`;
  if (!semanticWgslFunctionBodyShapeSupported(fn.body, semanticWgslFunctionHasAtomicPointer(fn))) return `unsupported body shape in '${fn.name}'`;
  return undefined;
}

function semanticWgslSurfaceWriteSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-write" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const target = operation.surface;
  return target.kind === "symbol" &&
    target.addressSpace === "surface" &&
    semanticWgslSurfaceValueSupported(operation.value) &&
    semanticWgslExpressionSupported(operation.value, "any", ir) &&
    semanticWgslExpressionSupported(operation.xBytes, "scalar", ir) &&
    semanticWgslExpressionSupported(operation.y, "scalar", ir) &&
    (operation.z === undefined || semanticWgslExpressionSupported(operation.z, "scalar", ir));
}

function semanticWgslSurfaceValueSupported(expression: SemanticExpression): boolean {
  const valueType = semanticExpressionValueType(expression);
  return !isSemanticFloatVectorType(valueType) || isCudaVectorType(valueType);
}

function semanticWgslAtomicTargetRootSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (ref.addressSpace === "local") {
    return semanticLocalPointerDeclarations(ir).some((operation) =>
      operation.target.name === ref.base && operation.target.pointerRuntimeState === true);
  }
  if (ref.addressSpace === "storage") {
    return ir.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && !param.constant) ||
      semanticWgslFunctionStoragePointerParam(ir, ref.base) !== undefined;
  }
  if (ref.addressSpace === "device-global") {
    return ir.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "device-global");
  }
  if (ref.addressSpace === "shared") {
    return ir.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "shared") ||
      semanticWgslFunctionSharedPointerParam(ir, ref.base) !== undefined;
  }
  return false;
}

function semanticWgslExpressionSupported(
  expression: SemanticExpression,
  expected: "scalar" | "any",
  ir?: SemanticKernelIrModule,
): boolean {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number";
    case "symbol":
      if (expected === "scalar" && isCudaVectorType(expression.valueType)) return false;
      return expression.addressSpace === "uniform" ||
        expression.addressSpace === "local" ||
        expression.addressSpace === "constant" ||
        expression.addressSpace === "device-global" ||
        expression.addressSpace === "shared" ||
        isCudaBuiltinVectorSymbolName(expression.name);
    case "pointer-valid":
      return ir === undefined ? false : semanticWgslPointerValidityOwner(expression, ir) !== undefined;
    case "member":
      if (expected === "scalar" && isCudaVectorType(expression.valueType)) return false;
      return expression.object.kind === "symbol" &&
        isCudaBuiltinVectorSymbolName(expression.object.name) &&
        (expression.property === "x" || expression.property === "y" || expression.property === "z") ||
        semanticWgslVectorMemberSupported(expression, ir);
    case "index":
      if (semanticWgslVectorIndexSupported(expression, ir)) return true;
      {
        const ref = memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span);
        const supported = ir === undefined ? semanticWgslMemoryRefSupported(ref) : semanticWgslTypedMemoryRefSupported(ref, ir);
        return supported && (expected === "any" || !isSemanticFloatVectorType(expression.valueType));
      }
    case "cast":
      return !expression.pointer && semanticWgslExpressionSupported(expression.expression, "scalar", ir);
    case "unary":
      if (expected === "scalar" && semanticWgslBf162LocalBitsCastSupported(expression, ir)) return true;
      if (expression.operator === "*") return ir !== undefined && semanticLocalStoragePointerDeclaration(ir, expression.argument) !== undefined;
      return expression.operator !== "&" && semanticWgslExpressionSupported(expression.argument, "scalar", ir);
    case "binary":
      if (isSemanticStoragePointerNullComparison(expression)) return true;
      if (isSemanticStoragePointerIdentityComparison(expression, ir)) return true;
      if (expected === "any" && isSemanticFloatVectorType(expression.valueType) && semanticWgslVectorBinaryOperatorSupported(expression.operator)) {
        return semanticWgslExpressionSupported(expression.left, "any", ir) &&
          semanticWgslExpressionSupported(expression.right, "any", ir);
      }
      return semanticWgslExpressionSupported(expression.left, "scalar", ir) &&
        semanticWgslExpressionSupported(expression.right, "scalar", ir);
    case "conditional":
      return !semanticWgslExpressionContainsSideEffectingCall(expression.consequent, ir) &&
        !semanticWgslExpressionContainsSideEffectingCall(expression.alternate, ir) &&
        semanticWgslExpressionSupported(expression.condition, "scalar", ir) &&
        semanticWgslExpressionSupported(expression.consequent, expected, ir) &&
        semanticWgslExpressionSupported(expression.alternate, expected, ir);
    case "assignment":
      {
        const vectorTarget = isSemanticFloatVectorType(semanticExpressionValueType(expression.target));
        return semanticWgslAssignmentOperatorSupported(expression.operator) &&
        (!vectorTarget || semanticVectorAssignmentOperatorSupported(expression.operator)) &&
        (expression.target.kind === "symbol" && expression.target.addressSpace === "local" ||
          expression.target.kind === "member" && semanticWgslVectorMemberSupported(expression.target, ir) ||
          semanticWgslAssignmentMemoryRefSupported(expression.target, ir)) &&
        semanticWgslExpressionSupported(expression.value, vectorTarget ? "any" : "scalar", ir);
      }
    case "update":
      return (expression.argument.kind === "symbol" && expression.argument.addressSpace === "local" ||
          (ir === undefined
            ? Boolean(memoryRefFromIndexExpression(expression.argument))
            : semanticWgslAssignmentMemoryRefSupported(expression.argument, ir))) &&
        (expression.operator === "++" || expression.operator === "--");
    case "sequence":
      return expression.expressions.every((item) => semanticWgslExpressionSupported(item, "scalar", ir));
    case "call":
      return semanticWgslSharedAddressCallRef(expression) !== undefined ||
        expression.callee.kind === "symbol" && semanticPtxIntegerCallInfo(expression.callee.name) !== undefined &&
          expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir)) ||
        ir !== undefined && semanticWgslCooperativeGroupCallSupported(expression, ir) ||
        ir !== undefined && semanticWgslCoalescedGroupCallSupported(expression, ir) ||
        ir !== undefined && semanticWgslCooperativeReduceCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir)) ||
        ir !== undefined && semanticWgslCooperativeVectorReduceCallSupported(expression, ir) ||
        ir !== undefined && semanticWgslCooperativeScanCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir)) ||
        ir !== undefined && semanticWgslSyncthreadsPredicateCallSupported(expression, ir) ||
        ir !== undefined && semanticWgslFunctionCallSupported(expression, ir) ||
        ir !== undefined && semanticWgslAtomicCallSupported(expression, ir) ||
        semanticWgslCurandCallSupported(expression, ir) &&
          (expected === "any" || !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression, ir?.functions))) ||
        semanticWgslGeneratedRandomCallSupported(expression) ||
        semanticWgslSubgroupCallSupported(expression, ir) ||
        semanticWgslAddressPredicateCallSupported(expression) ||
        semanticWgslMathCallSupported(expression, expected, ir) ||
        semanticWgslHalf2CallSupported(expression, ir) ||
        semanticWgslBf162CallSupported(expression, ir) ||
        semanticWgslVectorConstructorSupported(expression, expected, ir) ||
        expected === "scalar" && semanticWgslVectorAtCallSupported(expression, ir) ||
        expected === "any" && (semanticWgslVectorLerpCallSupported(expression, ir) ||
          expression.callee.kind === "symbol" && semanticVectorMathCallSupported(expression.callee.name, expression.args)) ||
        expected === "scalar" && expression.callee.kind === "symbol" &&
          (expression.callee.name === "dot" || expression.callee.name === "length") &&
          semanticVectorMathCallSupported(expression.callee.name, expression.args);
    case "texture-read":
      return ir !== undefined &&
        (expected === "any" || semanticTextureSurfaceValueTypeSupported(expression.valueType)) &&
        semanticWgslTextureReadSupported(expression, ir);
    case "surface-read":
      return ir !== undefined && (expected === "scalar" || expected === "any") && semanticWgslSurfaceReadSupported(expression, ir);
    case "initializer":
      return false;
  }
}

function semanticWgslExpressionContainsSideEffectingCall(
  expression: SemanticExpression,
  ir?: SemanticKernelIrModule,
  visited: ReadonlySet<string> = new Set(),
): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const callee = expression.callee.name;
    if (isSemanticAtomicCallName(callee)) return true;
    const fn = ir?.functions.find((item) => item.name === callee);
    if (fn && ir && !visited.has(fn.name)) {
      const nextVisited = new Set(visited).add(fn.name);
      if (semanticWgslOperationsHaveObservableSideEffects(fn.body, ir, nextVisited)) return true;
    }
  }
  return semanticExpressionChildren(expression).some((child) =>
    semanticWgslExpressionContainsSideEffectingCall(child, ir, visited)
  );
}

function semanticWgslPointerValidityOwner(
  expression: Extract<SemanticExpression, { readonly kind: "pointer-valid" }>,
  ir: SemanticKernelIrModule,
): CudaLiteSemanticSymbol | undefined {
  const owner = [...ir.params, ...ir.functions.flatMap((fn) => fn.params)]
    .find((symbol) => semanticIdsEqual(symbol.id, expression.pointerId));
  return owner?.name === expression.pointer && owner.pointer && owner.addressSpace === "storage"
    ? owner
    : undefined;
}

function semanticWgslOperationsHaveObservableSideEffects(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  visited: ReadonlySet<string>,
): boolean {
  return operations.some((operation) => {
    if (operation.kind === "store" || operation.kind === "atomic" || operation.kind === "copy" ||
      operation.kind === "surface-write" || operation.kind === "surface-read-store" ||
      operation.kind === "matrix-store" || operation.kind === "device-launch") return true;
    if (operation.kind === "call") {
      const fn = semanticFunctionForCall(operation, ir.functions);
      return fn !== undefined && !visited.has(fn.name) &&
        semanticWgslOperationsHaveObservableSideEffects(fn.body, ir, new Set(visited).add(fn.name));
    }
    if (operation.kind === "branch") {
      return semanticWgslOperationsHaveObservableSideEffects(operation.consequent, ir, visited) ||
        semanticWgslOperationsHaveObservableSideEffects(operation.alternate, ir, visited);
    }
    if (operation.kind === "block" || operation.kind === "loop") {
      return semanticWgslOperationsHaveObservableSideEffects(operation.body, ir, visited);
    }
    return false;
  });
}

function emitSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  allowReturnValue = false,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  return operations.flatMap((operation) => emitSemanticOperation(operation, ir, names, indentLevel, allowReturnValue, options, textureSpecializations));
}

function emitSemanticLocalPointerDeclaration(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions,
): readonly string[] {
  if (!semanticPointerDeclarationNeedsRuntimeState(operation)) return [];
  const prefix = "  ".repeat(indentLevel);
  const ref = semanticLocalPointerStorageRef(operation);
  const buffer = nameFor(semanticPointerBufferParamName(operation.target.name), names);
  const base = nameFor(semanticPointerBaseParamName(operation.target.name), names);
  if (!ref) return [];
  const bufferId = semanticStoragePointerBufferId(ref.base, ir);
  if (bufferId === undefined) throw semanticWgslError(`unknown local pointer storage root '${ref.base}'`, operation.span);
  return [
    `${prefix}var ${buffer}: u32 = ${bufferId}u;`,
    `${prefix}var ${base}: u32 = ${emitSemanticPointerArgBaseIndex(ref, ir, names, options)};`,
  ];
}

function emitSemanticOperation(
  operation: SemanticKernelIrOperation,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  allowReturnValue = false,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  switch (operation.kind) {
    case "dim3-declare":
    case "cooperative-group-declare":
      return [];
    case "declare": {
      if (operation.target.addressSpace === "shared") return [];
      if (operation.target.pointer) return emitSemanticLocalPointerDeclaration(operation, ir, names, indentLevel, options);
      if (operation.target.dimensions.length > 0) {
        return [
          `${prefix}var ${nameFor(operation.target.name, names)}: ${emitLocalArrayType(operation.target)};`,
          ...emitLocalArrayInit(operation, ir, names, indentLevel, options, textureSpecializations),
        ];
      }
      const storageVectorType = semanticStorageVectorType(operation.target.valueType);
      const type = wgslValueType(storageVectorType ?? operation.target.valueType);
      if (operation.init?.kind === "sequence") {
        const sequence = emitSemanticSequenceParts(operation.init, ir, names, indentLevel, options, textureSpecializations);
        const target = nameFor(operation.target.name, names);
        return [
          `${prefix}var ${target}: ${type};`,
          ...sequence.prefix,
          `${prefix}${target} = ${emitSemanticLocalScalarExpressionAs(sequence.value, operation.target.valueType, ir, names, options, textureSpecializations)};`,
        ];
      }
      const init = operation.init
        ? emitSemanticInitExpression(operation.init, operation.target.valueType, ir, names, options, textureSpecializations)
        : storageVectorType !== undefined
        ? createTypedWgslZero(type, operation.span)
        : undefined;
      const statement = createTypedWgslVariableStatement("var", nameFor(operation.target.name, names), type, init, operation.span);
      return [`${prefix}${statement.code}`];
    }
    case "store":
      return emitSemanticStoreOperation(operation, ir, names, indentLevel, options, textureSpecializations);
    case "copy":
      return emitSemanticCopyOperation(operation, ir, names, indentLevel, options);
    case "copy-fence":
      return [`${prefix}// cp.async fence omitted: ${operation.callee}`];
    case "matrix-fill":
      return emitSemanticMatrixFill(operation, ir, names, indentLevel, options, textureSpecializations);
    case "matrix-load":
      return emitSemanticMatrixLoad(operation, ir, names, indentLevel, options, textureSpecializations);
    case "matrix-mma":
      return emitSemanticMatrixMma(operation, ir, names, indentLevel, options, textureSpecializations);
    case "matrix-store":
      return emitSemanticMatrixStore(operation, ir, names, indentLevel, options, textureSpecializations);
    case "surface-write":
      return emitSemanticSurfaceWrite(operation, ir, names, indentLevel, options, textureSpecializations);
    case "surface-read-store":
      return [`${prefix}${emitSemanticSurfaceReadStore(operation, ir, names, options)};`];
    case "atomic":
      return [`${prefix}${emitSemanticAtomic(operation, ir, names, options, textureSpecializations)};`];
    case "call":
      return emitSemanticCall(operation, ir, names, indentLevel, options, textureSpecializations);
    case "pointer-rebind":
      return emitSemanticPointerRebind(operation, ir, names, indentLevel, options);
    case "expression":
      if (isSemanticNoopExpression(operation.expression)) return [];
      if (operation.expression.kind === "assignment") return [`${prefix}${emitSemanticAssignmentStatement(operation.expression, ir, names, options, textureSpecializations)};`];
      if (operation.expression.kind === "sequence") return emitSemanticSequenceStatement(operation.expression, ir, names, indentLevel, options, textureSpecializations);
      if (operation.expression.kind === "update") return [`${prefix}${emitSemanticLocalUpdateStatement(operation.expression, ir, names, options).code}`];
      return [`${prefix}${emitSemanticExpression(operation.expression, ir, names, options, textureSpecializations).code};`];
    case "branch": {
      const conditionOptions = operation.conditionUniformity === "workgroup"
        ? { ...options, workgroupUniformExpression: true }
        : options;
      if (semanticOperationsContainWorkgroupCollective(operation.consequent) || semanticOperationsContainWorkgroupCollective(operation.alternate)) {
        const condition = emitTruthiness(operation.condition, ir, names, conditionOptions);
        return [
          `${prefix}{`,
          ...emitSemanticPredicatedOperations(operation.consequent, condition, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
          `${prefix}}`,
          `${prefix}{`,
          ...emitSemanticPredicatedOperations(operation.alternate, `!(${condition})`, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
          `${prefix}}`,
        ];
      }
      const lines = [`${prefix}if (${emitTruthiness(operation.condition, ir, names, conditionOptions)}) {`];
      lines.push(...emitSemanticOperations(operation.consequent, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      if (operation.alternate.length > 0) {
        lines.push(`${prefix}} else {`);
        lines.push(...emitSemanticOperations(operation.alternate, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      }
      lines.push(`${prefix}}`);
      return lines;
    }
    case "block":
      return [
        `${prefix}{`,
        ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
        `${prefix}}`,
      ];
    case "loop":
      return emitSemanticLoop(operation, ir, names, indentLevel, allowReturnValue, options, textureSpecializations);
    case "barrier":
      return [`${prefix}workgroupBarrier();`];
    case "fence":
      return [`${prefix}storageBarrier();`];
    case "inline-asm":
      {
        const asm = operation.op;
        const ldmatrix = semanticInlineAsmLdmatrixAssignments(operation);
        if (ldmatrix) {
          return ldmatrix.map((assignment) =>
            `${prefix}${emitSemanticAssignmentStatement(assignment, ir, names, options, textureSpecializations)};`
          );
        }
        if (asm?.kind === "mma-m16n8k16") {
          return emitSemanticInlineMma(operation, asm.accumulator, ir, names, prefix, options, textureSpecializations);
        }
        if (asm?.kind === "cp-async-fence" && operation.inputs.length <= (asm.fence === "wait_group" ? 1 : 0) && operation.outputs.length === 0) return [`${prefix}// cp.async inline asm fence omitted`];
        if (asm?.kind === "membar" && operation.inputs.length === 0 && operation.outputs.length === 0) return [`${prefix}storageBarrier();`];
        if (asm?.kind === "bar-sync" && operation.inputs.length === (asm.operand === "input0" ? 1 : 0) && operation.outputs.length === 0) return [`${prefix}workgroupBarrier();`];
      }
      throw semanticWgslError(`semantic WGSL does not support ${operation.kind}`, operation.span);
    case "return":
      if (operation.value) {
        if (!allowReturnValue) throw semanticWgslError("semantic WGSL supports kernel return without value only", operation.span);
        return emitSemanticReturnValue(operation.value, ir, names, indentLevel, options, textureSpecializations);
      }
      return [`${prefix}return;`];
    case "break":
      return [`${prefix}break;`];
    case "continue":
      return [`${prefix}continue;`];
    default:
      throw semanticWgslError(`semantic WGSL does not support ${operation.kind}`, operation.span);
  }
}

function emitSemanticPointerRebind(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "pointer-rebind" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions,
): readonly string[] {
  const bufferId = semanticStoragePointerBufferId(operation.source.base, ir);
  if (bufferId === undefined) throw semanticWgslError(`unknown pointer rebind storage root '${operation.source.base}'`, operation.source.span);
  const prefix = "  ".repeat(indentLevel);
  const buffer = nameFor(semanticPointerBufferParamName(operation.target.name), names);
  const base = nameFor(semanticPointerBaseParamName(operation.target.name), names);
  return [
    `${prefix}${buffer} = ${bufferId}u;`,
    `${prefix}${base} = ${emitSemanticPointerArgBaseIndex(operation.source, ir, names, options)};`,
  ];
}

function emitSemanticMatrixFill(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-fill" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const index = `bg_wmma_i_${operation.span.start}`;
  const value = emitSemanticMatrixCoerce(emitSemanticExpression(operation.value, ir, names, options, textureSpecializations).code, operation.fragment.spec);
  return [
    `${prefix}for (var ${index}: u32 = 0u; ${index} < ${matrixTileElementCount(operation.fragment.spec)}u; ${index} = ${index} + 1u) {`,
    `${prefix}  ${emitSemanticMatrixAccess(operation.fragment, index, ir, names, options)} = ${value};`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixLoad(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-load" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions,
  _textureSpecializations: SemanticTextureDescriptorSpecializations,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const row = `bg_wmma_row_${operation.span.start}`;
  const col = `bg_wmma_col_${operation.span.start}`;
  const [rows, cols] = semanticWgslMatrixRowsCols(operation.fragment.spec);
  const offset = semanticWgslMatrixOffset(row, col, operation.stride, operation.layout, operation.span);
  const read = emitSemanticMemoryRead(semanticWgslMemoryRefOffset(operation.source, offset), ir, names, options);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${prefix}  for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${prefix}    ${emitSemanticMatrixAccess(operation.fragment, tileIndex, ir, names, options)} = ${emitSemanticMatrixCoerce(read, operation.fragment.spec)};`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixMma(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-mma" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions,
  _textureSpecializations: SemanticTextureDescriptorSpecializations,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const row = `bg_wmma_row_${operation.span.start}`;
  const col = `bg_wmma_col_${operation.span.start}`;
  const kk = `bg_wmma_k_${operation.span.start}`;
  const sum = `bg_wmma_sum_${operation.span.start}`;
  const { destination: dst, a, b, accumulator: c } = operation;
  const dstIndex = `(${row} * ${dst.spec.n}u + ${col})`;
  const aIndex = `(${row} * ${dst.spec.k}u + ${kk})`;
  const bIndex = `(${kk} * ${dst.spec.n}u + ${col})`;
  const integer = dst.spec.tileValueType === "s32" && isMatrixTileByteValueType(a.spec.tileValueType) && isMatrixTileByteValueType(b.spec.tileValueType);
  const cValue = emitSemanticMatrixAccess(c, dstIndex, ir, names, options);
  const aValue = emitSemanticMatrixAccess(a, aIndex, ir, names, options);
  const bValue = emitSemanticMatrixAccess(b, bIndex, ir, names, options);
  const init = integer ? emitSemanticMatrixInteger(cValue, c.spec) : `f32(${cValue})`;
  const product = integer
    ? `(${emitSemanticMatrixInteger(aValue, a.spec)} * ${emitSemanticMatrixInteger(bValue, b.spec)})`
    : `(f32(${aValue}) * f32(${bValue}))`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${dst.spec.m}u; ${row} = ${row} + 1u) {`,
    `${prefix}  for (var ${col}: u32 = 0u; ${col} < ${dst.spec.n}u; ${col} = ${col} + 1u) {`,
    `${prefix}    var ${sum}: ${integer ? "i32" : "f32"} = ${init};`,
    `${prefix}    for (var ${kk}: u32 = 0u; ${kk} < ${dst.spec.k}u; ${kk} = ${kk} + 1u) {`,
    `${prefix}      ${sum} = ${sum} + ${product};`,
    `${prefix}    }`,
    `${prefix}    ${emitSemanticMatrixAccess(dst, dstIndex, ir, names, options)} = ${emitSemanticMatrixCoerce(sum, dst.spec)};`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions,
  _textureSpecializations: SemanticTextureDescriptorSpecializations,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const row = `bg_wmma_row_${operation.span.start}`;
  const col = `bg_wmma_col_${operation.span.start}`;
  const [rows, cols] = semanticWgslMatrixRowsCols(operation.fragment.spec);
  const offset = semanticWgslMatrixOffset(row, col, operation.stride, operation.layout, operation.span);
  const target = semanticWgslMemoryRefOffset(operation.target, offset);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  const value = emitSemanticMatrixAccess(operation.fragment, tileIndex, ir, names, options);
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${prefix}  for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${prefix}    ${emitSemanticMemoryWrite(target, value, ir, names, options)};`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixAccess(ref: SemanticMatrixTileRef, index: string, ir: SemanticKernelIrModule, names: ReadonlyMap<string, string>, options: EmitSemanticKernelIrWgslOptions): string {
  const terms = ref.indices.map((item, axis) => {
    const stride = ref.arrayDimensions.slice(axis + 1).reduce((product, value) => product * value, 1) * matrixTileElementCount(ref.spec);
    const emitted = `u32(${emitSemanticExpression(item, ir, names, options).code})`;
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  const base = terms.length === 0 ? undefined : terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
  return `${nameFor(ref.base, names)}[${base ? `(${base} + ${index})` : index}]`;
}

function semanticWgslMatrixOffset(row: string, col: string, stride: SemanticExpression, layout: MatrixTileLayout, span: SourceSpan): SemanticExpression {
  const rowExpression = semanticWgslGeneratedSymbol(row, span);
  const colExpression = semanticWgslGeneratedSymbol(col, span);
  const major = layout === "col_major" || layout === "mem_col_major" ? colExpression : rowExpression;
  const minor = layout === "col_major" || layout === "mem_col_major" ? rowExpression : colExpression;
  return { kind: "binary", operator: "+", left: { kind: "binary", operator: "*", left: major, right: stride, valueType: "uint", span }, right: minor, valueType: "uint", span };
}

function semanticWgslGeneratedSymbol(name: string, span: SourceSpan): SemanticExpression {
  return { kind: "symbol", id: createGeneratedSemanticSymbolId(name, span), name, valueType: "uint", addressSpace: "local", span };
}

function semanticWgslMemoryRefOffset(ref: SemanticMemoryRef, offset: SemanticExpression): SemanticMemoryRef {
  const scaled = ref.pointerBaseUnitBytes === undefined || ref.pointerBaseUnitBytes === 1
    ? offset
    : { kind: "binary", operator: "*", left: offset, right: { kind: "literal", literalKind: "number", value: ref.pointerBaseUnitBytes, valueType: "uint", span: ref.span }, valueType: "uint", span: ref.span } satisfies SemanticExpression;
  if (ref.indices.length === 0) return { ...ref, indices: [scaled] };
  const last = ref.indices[ref.indices.length - 1]!;
  return { ...ref, indices: [...ref.indices.slice(0, -1), { kind: "binary", operator: "+", left: last, right: scaled, valueType: "uint", span: ref.span }] };
}

function semanticWgslMatrixRowsCols(spec: MatrixTileResolvedSpec): readonly [number, number] {
  return spec.role === "matrix_a" ? [spec.m, spec.k] : spec.role === "matrix_b" ? [spec.k, spec.n] : [spec.m, spec.n];
}

function emitSemanticMatrixCoerce(value: string, spec: MatrixTileResolvedSpec): string {
  if (spec.tileValueType === "u8") return `(u32(${value}) & 255u)`;
  if (spec.tileValueType === "s8") return `(i32((u32(${value}) & 255u) << 24u) >> 24)`;
  if (spec.tileValueType === "s32") return `i32(${value})`;
  return `f32(${value})`;
}

function emitSemanticMatrixInteger(value: string, spec: MatrixTileResolvedSpec): string {
  return spec.tileValueType === "u8" ? `i32(u32(${value}) & 255u)` : spec.tileValueType === "s8" ? `(i32((u32(${value}) & 255u) << 24u) >> 24)` : `i32(${value})`;
}

function emitSemanticInlineMma(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "inline-asm" }>,
  accumulator: "f16" | "f32",
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  prefix: string,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): readonly string[] {
  return operation.outputs.map((output, index) => {
    const target = emitSemanticExpression(output, ir, names, options, textureSpecializations).code;
    const a = emitSemanticExpressionAs(operation.inputs[index % 4]!, ir, names, "u32", options, textureSpecializations).code;
    const b = emitSemanticExpressionAs(operation.inputs[4 + (index % 2)]!, ir, names, "u32", options, textureSpecializations).code;
    if (accumulator === "f16") {
      const c = emitSemanticExpressionAs(operation.inputs[6 + index]!, ir, names, "u32", options, textureSpecializations).code;
      const value = `pack2x16float(unpack2x16float(${c}) + (unpack2x16float(${a}) * unpack2x16float(${b})))`;
      return `${prefix}${target} = ${semanticInlineMmaOutputValue(output, value, "u32")};`;
    }
    const cExpression = operation.inputs[6 + index]!;
    const cRaw = emitSemanticExpression(cExpression, ir, names, options, textureSpecializations).code;
    const cType = semanticExpressionValueType(cExpression);
    const c = cType === "uint" || cType === "int" ? `bitcast<f32>(u32(${cRaw}))` : `f32(${cRaw})`;
    const value = `(${c} + dot(unpack2x16float(${a}), unpack2x16float(${b})))`;
    return `${prefix}${target} = ${semanticInlineMmaOutputValue(output, value, "f32")};`;
  });
}

function semanticInlineMmaOutputValue(output: SemanticExpression, value: string, sourceType: "u32" | "f32"): string {
  const outputType = semanticExpressionValueType(output);
  if (outputType === "uint") return sourceType === "u32" ? value : `bitcast<u32>(${value})`;
  if (outputType === "int") return `bitcast<i32>(${value})`;
  if (outputType === "half") return `f16(${value})`;
  return sourceType === "u32" ? `u32(${value})` : value;
}

function emitSemanticPredicatedOperations(
  operations: readonly SemanticKernelIrOperation[],
  predicate: string,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  allowReturnValue: boolean,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): readonly string[] {
  const lines: string[] = [];
  const prefix = "  ".repeat(indentLevel);
  for (const operation of operations) {
    if (operation.kind === "block" && semanticOperationsContainWorkgroupCollective(operation.body)) {
      lines.push(`${prefix}{`);
      lines.push(...emitSemanticPredicatedOperations(operation.body, predicate, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      lines.push(`${prefix}}`);
      continue;
    }
    if (operation.kind === "branch") {
      const condition = emitTruthiness(operation.condition, ir, names, options);
      lines.push(`${prefix}{`);
      lines.push(...emitSemanticPredicatedOperations(operation.consequent, `(${predicate}) && (${condition})`, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      lines.push(`${prefix}}`);
      lines.push(`${prefix}{`);
      lines.push(...emitSemanticPredicatedOperations(operation.alternate, `(${predicate}) && !(${condition})`, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      lines.push(`${prefix}}`);
      continue;
    }
    if (operation.kind === "loop" && semanticOperationsContainWorkgroupCollective(operation.body)) {
      if (operation.loopKind !== "for" || operation.update?.kind === "sequence") {
        throw semanticWgslError("semantic WGSL supports predicated cooperative shuffle only in canonical for loops", operation.span);
      }
      const init = operation.init ? emitSemanticLoopInit(operation.init, ir, names, options, textureSpecializations) : "";
      const condition = operation.condition ? emitTruthiness(operation.condition, ir, names, options) : "true";
      const update = operation.update ? emitSemanticLoopUpdate(operation.update, ir, names, options, textureSpecializations) : "";
      lines.push(`${prefix}for (${init}; ${condition}; ${update}) {`);
      lines.push(...emitSemanticPredicatedOperations(operation.body, predicate, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      lines.push(`${prefix}}`);
      continue;
    }
    if (operation.kind === "declare") {
      const valueType = operation.target.valueType;
      if (operation.target.addressSpace === "local" && !operation.target.pointer && operation.init === undefined) {
        lines.push(...emitSemanticOperation(operation, ir, names, indentLevel, allowReturnValue, options, textureSpecializations));
        continue;
      }
      if (operation.target.addressSpace !== "local" || operation.target.dimensions.length > 0 ||
        valueType === undefined || valueType === "void") {
        throw semanticWgslError("predicated cooperative shuffle requires local scalar declaration", operation.span);
      }
      if (operation.init && semanticExpressionContainsWorkgroupCollective(operation.init)) {
        lines.push(...emitSemanticOperation(
          operation,
          ir,
          names,
          indentLevel,
          allowReturnValue,
          { ...options, activeCollectivePredicate: predicate },
          textureSpecializations,
        ));
      } else {
        const { init: _init, ...declaration } = operation;
        lines.push(...emitSemanticOperation(
          declaration,
          ir,
          names,
          indentLevel,
          allowReturnValue,
          options,
          textureSpecializations,
        ));
        if (operation.init) {
          const target: SemanticExpression = {
            kind: "symbol",
            id: operation.target.id,
            name: operation.target.name,
            valueType,
            addressSpace: "local",
            span: operation.target.span,
          };
          lines.push(`${prefix}if (${predicate}) {`);
          lines.push(`${"  ".repeat(indentLevel + 1)}${emitSemanticAssignmentStatement({
            kind: "assignment",
            operator: "=",
            target,
            value: operation.init,
            valueType,
            span: operation.span,
          }, ir, names, options, textureSpecializations)};`);
          lines.push(`${prefix}}`);
        }
      }
      continue;
    }
    if (operation.kind === "expression" && operation.expression.kind === "assignment" && semanticExpressionContainsWorkgroupCollective(operation.expression.value)) {
      if (operation.expression.target.kind !== "symbol" || operation.expression.target.addressSpace !== "local") {
        throw semanticWgslError("predicated cooperative shuffle requires local scalar assignment", operation.span);
      }
      const valueType = operation.expression.target.valueType;
      if (!valueType || isSemanticFloatVectorType(valueType)) {
        throw semanticWgslError("predicated cooperative shuffle requires typed scalar assignment", operation.span);
      }
      const temporary = nameFor(`bg_collective_${operation.span.start}`, names);
      const collectiveOptions = { ...options, activeCollectivePredicate: predicate };
      lines.push(`${prefix}let ${temporary}: ${wgslValueScalar(valueType)} = ${emitSemanticLocalScalarExpressionAs(operation.expression.value, valueType, ir, names, collectiveOptions, textureSpecializations)};`);
      lines.push(`${prefix}if (${predicate}) {`);
      lines.push(`${"  ".repeat(indentLevel + 1)}${emitSemanticAssignmentStatement({ ...operation.expression, value: { kind: "symbol", id: createGeneratedSemanticSymbolId(temporary, operation.span), name: temporary, valueType, addressSpace: "local", span: operation.span } }, ir, names, options, textureSpecializations)};`);
      lines.push(`${prefix}}`);
      continue;
    }
    if (operation.kind === "store" && semanticExpressionContainsWorkgroupCollective(operation.value) &&
      !operation.target.indices.some(semanticExpressionContainsWorkgroupCollective)) {
      const valueType = operation.target.valueType;
      if (isSemanticFloatVectorType(valueType)) {
        throw semanticWgslError("predicated cooperative store requires typed scalar value", operation.span);
      }
      const temporary = nameFor(`bg_collective_${operation.span.start}`, names);
      const collectiveOptions = { ...options, activeCollectivePredicate: predicate };
      lines.push(`${prefix}let ${temporary}: ${wgslValueScalar(valueType)} = ${emitSemanticLocalScalarExpressionAs(operation.value, valueType, ir, names, collectiveOptions, textureSpecializations)};`);
      lines.push(`${prefix}if (${predicate}) {`);
      lines.push(...emitSemanticOperation({
        ...operation,
        value: { kind: "symbol", id: createGeneratedSemanticSymbolId(temporary, operation.span), name: temporary, valueType, addressSpace: "local", span: operation.span },
      }, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
      lines.push(`${prefix}}`);
      continue;
    }
    if (semanticOperationContainsWorkgroupCollective(operation)) {
      throw semanticWgslError("semantic WGSL does not support this predicated cooperative shuffle shape", operation.span);
    }
    lines.push(`${prefix}if (${predicate}) {`);
    lines.push(...emitSemanticOperation(operation, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
    lines.push(`${prefix}}`);
  }
  return lines;
}

function semanticOperationsContainWorkgroupCollective(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some(semanticOperationContainsWorkgroupCollective);
}

function semanticOperationContainsWorkgroupCollective(operation: SemanticKernelIrOperation): boolean {
  if (operation.kind === "declare") return operation.init !== undefined && semanticExpressionContainsWorkgroupCollective(operation.init);
  if (operation.kind === "store") return semanticExpressionContainsWorkgroupCollective(operation.value) || operation.target.indices.some(semanticExpressionContainsWorkgroupCollective);
  if (operation.kind === "atomic" || operation.kind === "call") return operation.args.some(semanticExpressionContainsWorkgroupCollective);
  if (operation.kind === "expression") return semanticExpressionContainsWorkgroupCollective(operation.expression);
  if (operation.kind === "branch") return semanticExpressionContainsWorkgroupCollective(operation.condition) || semanticOperationsContainWorkgroupCollective(operation.consequent) || semanticOperationsContainWorkgroupCollective(operation.alternate);
  if (operation.kind === "loop") return (operation.init !== undefined && !isSemanticKernelIrOperation(operation.init) && semanticExpressionContainsWorkgroupCollective(operation.init)) ||
    (operation.condition !== undefined && semanticExpressionContainsWorkgroupCollective(operation.condition)) ||
    (operation.update !== undefined && semanticExpressionContainsWorkgroupCollective(operation.update)) ||
    semanticOperationsContainWorkgroupCollective(operation.body);
  if (operation.kind === "block") return semanticOperationsContainWorkgroupCollective(operation.body);
  if (operation.kind === "return") return operation.value !== undefined && semanticExpressionContainsWorkgroupCollective(operation.value);
  return false;
}

function semanticExpressionContainsWorkgroupCollective(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && semanticSyncthreadsPredicateHelperFor(expression) !== undefined) return true;
  if (expression.kind === "call" && expression.callee.kind === "member" && expression.callee.object.kind === "symbol" &&
    (expression.callee.property === "ballot" || expression.callee.property === "any" ||
      expression.callee.property === "all" || expression.callee.property === "shfl")) return true;
  if (expression.kind === "call" && expression.callee.kind === "symbol" && expression.callee.addressSpace !== "function" &&
    (semanticShuffleOpForCall(expression.callee.name) !== undefined ||
      isCudaWarpReduceCallName(expression.callee.name) ||
      cudaVoteOpForCall(expression.callee.name) === "ballot" ||
      cudaVoteOpForCall(expression.callee.name) === "any" ||
      cudaVoteOpForCall(expression.callee.name) === "all" ||
      cudaArithmeticReduceOpForCall(expression.callee.name) !== undefined ||
      expression.callee.name === "__activemask" ||
      expression.callee.name === "cg::reduce" ||
      expression.callee.name === "cooperative_groups::reduce" ||
      expression.callee.name === "cg::inclusive_scan" ||
      expression.callee.name === "cooperative_groups::inclusive_scan" ||
      expression.callee.name === "cg::exclusive_scan" ||
      expression.callee.name === "cooperative_groups::exclusive_scan")) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionContainsWorkgroupCollective);
}

function emitSemanticSurfaceReadStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-read-store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const target = semanticWgslSurfaceReadTarget(operation.target);
  if (!target) throw semanticWgslError("semantic WGSL supports only local scalar/vector surf2Dread targets", operation.span);
  const value = emitSemanticSurfaceRead(
    {
      kind: "surface-read",
      callee: operation.z === undefined ? "surf2Dread" : "surf2DLayeredread",
      surface: operation.surface,
      xBytes: operation.xBytes,
      y: operation.y,
      ...(operation.z === undefined ? {} : { z: operation.z }),
      valueType: semanticSurfaceReadValueType(operation.valueType ?? target.valueType),
      span: operation.span,
    },
    ir,
    names,
    options,
  );
  return `${nameFor(target.name, names)} = ${value}`;
}

function emitSemanticStoreOperation(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const sequence = operation.value.kind === "sequence"
    ? emitSemanticSequenceParts(operation.value, ir, names, indentLevel, options, textureSpecializations)
    : undefined;
  const lowered = sequence ? { ...operation, value: sequence.value } : operation;
  if (semanticWgslLocalPackedByteRawView(lowered.target, ir)) {
    const word = emitSemanticExpressionAs(lowered.value, ir, names, "u32", options, textureSpecializations).code;
    const wordName = nameFor(`bg_packed_byte_word_${lowered.span.start}`, names);
    const target = nameFor(lowered.target.base, names);
    return [
      ...(sequence?.prefix ?? []),
      `${prefix}let ${wordName}: u32 = ${word};`,
      ...["x", "y", "z", "w"].map((field, byte) => `${prefix}${target}.${field} = ((${wordName} >> ${byte * 8}u) & 255u);`),
    ];
  }
  if (semanticDirectVectorStorageStore(lowered, ir)) {
    const vectorType = semanticStorageVectorType(lowered.target.valueType)!;
    const storeIdentity = `${lowered.target.base}_${lowered.span.start}`;
    const valueName = nameFor(`bg_vector_store_value_${storeIdentity}`, names);
    const baseName = nameFor(`bg_vector_store_base_${storeIdentity}`, names);
    const value = emitSemanticVectorOperand(lowered.value, vectorType, ir, names, options, textureSpecializations);
    const base = emitFlatStorageVectorBaseIndex(lowered.target, ir, names, options);
    return [
      ...(sequence?.prefix ?? []),
      `${prefix}let ${valueName}: ${wgslValueType(vectorType)} = ${value};`,
      `${prefix}let ${baseName}: u32 = ${base};`,
      ...emitSemanticVectorMemoryWrite(lowered, ir, names, options, textureSpecializations, valueName, baseName)
        .map((line) => `${prefix}${line};`),
    ];
  }
  return [
    ...(sequence?.prefix ?? []),
    `${prefix}${emitSemanticStore(lowered, ir, names, options, textureSpecializations)};`,
  ];
}

function semanticDirectVectorStorageStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticStorageVectorType(operation.target.valueType) !== undefined &&
    (operation.target.addressSpace === "storage" || operation.target.addressSpace === "device-global") &&
    semanticWgslFunctionStoragePointerParam(ir, operation.target.base) === undefined &&
    !semanticWgslVectorFieldMemoryRefSupported(operation.target) &&
    !semanticAtomicStorageNames(ir.operations, ir.functions).has(operation.target.base) &&
    !semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(operation.target.base);
}

function emitSemanticCopyOperation(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "copy" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const sourceType = operation.source.valueType!;
  const targetType = operation.target.valueType!;
  const sourceBytes = sizeofCudaType(sourceType)!;
  const targetBytes = sizeofCudaType(targetType)!;
  if (sourceType === targetType) {
    return Array.from({ length: operation.bytes / sourceBytes }, (_, offset) => {
      const source = semanticCopyMemoryRefAt(operation.source, offset);
      const target = semanticCopyMemoryRefAt(operation.target, offset);
      return `${prefix}${emitSemanticMemoryWrite(target, emitSemanticMemoryRead(source, ir, names, options), ir, names, options)};`;
    });
  }
  return Array.from({ length: operation.bytes / 4 }, (_, wordOffset) => {
    const word = emitSemanticCopyWordRead(operation.source, sourceType, sourceBytes, wordOffset, ir, names, options);
    return emitSemanticCopyWordWrite(operation.target, targetType, targetBytes, wordOffset, word, ir, names, options)
      .map((line) => `${prefix}${line};`)
      .join("\n");
  });
}

function emitSemanticCopyWordRead(ref: SemanticMemoryRef, valueType: CudaLiteScalarType, valueBytes: number, wordOffset: number, ir: SemanticKernelIrModule, names: ReadonlyMap<string, string>, options: EmitSemanticKernelIrWgslOptions): string {
  if (valueBytes === 1) {
    return Array.from({ length: 4 }, (_, lane) => {
      const value = emitSemanticMemoryRead(semanticCopyMemoryRefAt(ref, wordOffset * 4 + lane), ir, names, options);
      return `(u32(${value}) << ${lane * 8}u)`;
    }).join(" | ");
  }
  if (valueBytes === 2) {
    const low = emitSemanticMemoryRead(semanticCopyMemoryRefAt(ref, wordOffset * 2), ir, names, options);
    const high = emitSemanticMemoryRead(semanticCopyMemoryRefAt(ref, wordOffset * 2 + 1), ir, names, options);
    return `bitcast<u32>(vec2<f16>(${low}, ${high}))`;
  }
  const value = emitSemanticMemoryRead(semanticCopyMemoryRefAt(ref, wordOffset), ir, names, options);
  return valueType === "uint" ? value : `bitcast<u32>(${value})`;
}

function emitSemanticCopyWordWrite(ref: SemanticMemoryRef, valueType: CudaLiteScalarType, valueBytes: number, wordOffset: number, word: string, ir: SemanticKernelIrModule, names: ReadonlyMap<string, string>, options: EmitSemanticKernelIrWgslOptions): readonly string[] {
  if (valueBytes === 1) {
    return Array.from({ length: 4 }, (_, lane) =>
      emitSemanticMemoryWrite(semanticCopyMemoryRefAt(ref, wordOffset * 4 + lane), `((${word}) >> ${lane * 8}u) & 0xffu`, ir, names, options));
  }
  if (valueBytes === 2) {
    const pair = `bitcast<vec2<f16>>(${word})`;
    return [
      emitSemanticMemoryWrite(semanticCopyMemoryRefAt(ref, wordOffset * 2), `${pair}.x`, ir, names, options),
      emitSemanticMemoryWrite(semanticCopyMemoryRefAt(ref, wordOffset * 2 + 1), `${pair}.y`, ir, names, options),
    ];
  }
  const value = valueType === "uint" ? word : `bitcast<${wgslValueType(valueType)}>(${word})`;
  return [emitSemanticMemoryWrite(semanticCopyMemoryRefAt(ref, wordOffset), value, ir, names, options)];
}

function semanticCopyMemoryRefAt(ref: SemanticMemoryRef, offset: number): SemanticMemoryRef {
  if (offset === 0 && ref.indices.length > 0) return ref;
  const offsetExpression: SemanticExpression = {
    kind: "literal",
    literalKind: "number",
    value: offset,
    valueType: "int",
    span: ref.span,
  };
  if (ref.indices.length === 0) return { ...ref, indices: [offsetExpression] };
  const index = ref.indices.at(-1)!;
  return {
    ...ref,
    indices: [
      ...ref.indices.slice(0, -1),
      {
        kind: "binary",
        operator: "+",
        left: index,
        right: offsetExpression,
        valueType: semanticExpressionValueType(index) ?? "int",
        span: ref.span,
      },
    ],
  };
}

function emitSemanticSequenceStatement(
  expression: Extract<SemanticExpression, { readonly kind: "sequence" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const sequence = emitSemanticSequenceParts(expression, ir, names, indentLevel, options, textureSpecializations);
  return [
    ...sequence.prefix,
    ...emitSemanticExpressionStatement(sequence.value, ir, names, indentLevel, options, textureSpecializations),
  ];
}

function emitSemanticSequenceParts(
  expression: Extract<SemanticExpression, { readonly kind: "sequence" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): { readonly prefix: readonly string[]; readonly value: SemanticExpression } {
  const expressions = expression.expressions.length > 0 ? expression.expressions : [zeroExpression(expression.span)];
  const value = expressions.at(-1)!;
  const prefix = expressions.slice(0, -1).flatMap((item) =>
    emitSemanticExpressionStatement(item, ir, names, indentLevel, options, textureSpecializations)
  );
  return { prefix, value };
}

function emitSemanticExpressionStatement(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (isSemanticNoopExpression(expression)) return [];
  const prefix = "  ".repeat(indentLevel);
  if (expression.kind === "assignment") return [`${prefix}${emitSemanticAssignmentStatement(expression, ir, names, options, textureSpecializations)};`];
  if (expression.kind === "sequence") return emitSemanticSequenceStatement(expression, ir, names, indentLevel, options, textureSpecializations);
  if (expression.kind === "update") return [`${prefix}${emitSemanticLocalUpdateStatement(expression, ir, names, options).code}`];
  return [`${prefix}${emitSemanticExpression(expression, ir, names, options, textureSpecializations).code};`];
}

function emitSemanticLocalUpdateStatement(
  expression: Extract<SemanticExpression, { readonly kind: "update" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): ReturnType<typeof createTypedWgslLocalAssignmentStatement> {
  const ref = memoryRefFromIndexExpression(expression.argument);
  const storagePointer = ref === undefined ? undefined : (
    (options.activeFunction === undefined ? undefined : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
      param.name === ref.base && param.pointer && param.addressSpace === "storage"
    )) ?? semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null)
  );
  const localStoragePointer = ref === undefined ? undefined : semanticLocalPointerDeclarations(ir).find((operation) =>
    operation.target.name === ref.base && semanticLocalPointerStorageRef(operation) !== undefined
  );
  const sharedPointer = ref === undefined || options.activeFunction === undefined ? undefined : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
    param.name === ref.base && param.pointer && param.addressSpace === "shared"
  );
  if (ref && sharedPointer && ref.valueType === "uchar" && ref.indices.length === 1 && ref.fields.length === 0) {
    const byteIndex = emitTypedWgslBinary(
      "+",
      createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
      emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options),
      ref.span,
    );
    const wordIndex = emitTypedWgslBinary(">>", byteIndex, createTypedWgslLiteral("2u", "u32", ref.span), ref.span);
    const shift = emitTypedWgslBinary(
      "*",
      emitTypedWgslBinary("&", byteIndex, createTypedWgslLiteral("3u", "u32", ref.span), ref.span),
      createTypedWgslLiteral("8u", "u32", ref.span),
      ref.span,
    );
    const pointerName = nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names);
    const word = createTypedWgslDereferencedIndexedPlace(pointerName, wordIndex, "u32", true, "workgroup", ref.span);
    return createTypedWgslCallStatement(
      PACKED_SHARED_U8_ADD,
      [
        createTypedWgslAddressOf(word),
        shift,
        createTypedWgslLiteral(expression.operator === "++" ? "1" : "-1", "i32", expression.span),
      ],
      expression.span,
    );
  }
  if (ref && (storagePointer || localStoragePointer) && ref.indices.length === 1 && ref.fields.length === 0 && (semanticWgslScalarTypeSupported(ref.valueType) || ref.valueType === "uchar")) {
    const valueType = ref.valueType ?? "float";
    const type = wgslValueType(valueType);
    if (type === "f16" || type === "f32" || type === "i32" || type === "u32") {
      const buffer = createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(ref.base), names), "u32", ref.span);
      const index = localStoragePointer
        ? emitTypedWgslBinary(
            "+",
            createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
            emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options),
            ref.span,
          )
        : emitTypedFlatStorageIndex(ref, ir, names, options);
      const current = createTypedWgslCall(semanticPointerReadHelperName(valueType), [buffer, index], type, ref.span);
      let next = emitTypedWgslBinary(
        expression.operator === "++" ? "+" : "-",
        current,
        createTypedWgslLiteral(type === "u32" ? "1u" : type === "i32" ? "1" : type === "f16" ? "f16(1.0)" : "1.0", type, expression.span),
        expression.span,
      );
      if (valueType === "uchar") next = emitTypedWgslBinary("&", next, createTypedWgslLiteral("0xffu", "u32", expression.span), expression.span);
      return createTypedWgslCallStatement(semanticPointerWriteHelperName(valueType), [buffer, index, next], expression.span);
    }
  }
  if (ref && ref.indices.length === 1 && ref.fields.length === 0 && (semanticWgslScalarTypeSupported(ref.valueType) || ref.valueType === "uchar") &&
    ref.packedByteLanes === undefined && !semanticWgslPackedSharedByteRoot(ref, ir) && !semanticWgslDirectByteRawView(ref, ir) &&
    !semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null) && !semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null)) {
    const type = wgslValueType(ref.valueType);
    if (type === "f16" || type === "f32" || type === "i32" || type === "u32") {
      const index = ref.addressSpace === "storage" || ref.addressSpace === "device-global" || ref.addressSpace === "constant"
        ? emitTypedFlatStorageIndex(ref, ir, names, options)
        : emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
      const atomic = ref.addressSpace === "storage"
        ? semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base)
        : ref.addressSpace === "device-global"
          ? semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(ref.base)
          : ref.addressSpace === "shared" && semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base);
      if (!atomic) {
        const place = createTypedWgslIndexedPlace(
          nameFor(ref.base, names),
          index,
          type,
          false,
          expression.span,
          ref.addressSpace === "shared" ? "workgroup" : ref.addressSpace === "local" ? "function" : "storage",
        );
        return createTypedWgslPlaceAssignmentStatement(
          place,
          expression.operator === "++" ? "+=" : "-=",
          createTypedWgslLiteral(type === "u32" ? "1u" : type === "i32" ? "1" : type === "f16" ? "f16(1.0)" : "1.0", type, expression.span),
          expression.span,
        );
      }
    }
  }
  if (expression.argument.kind !== "symbol") {
    throw semanticWgslError(
      `typed WGSL memory update requires place lowering (${ref?.addressSpace ?? "none"}:${ref?.valueType ?? "none"}:rank${ref?.indices.length ?? 0}:fields${ref?.fields.length ?? 0}:scalar${semanticWgslScalarTypeSupported(ref?.valueType)}:pointer${Boolean(storagePointer)}:base${ref?.base ?? "none"})`,
      expression.span,
    );
  }
  const pointer = semanticLocalStoragePointerDeclaration(ir, expression.argument);
  const target = pointer
    ? nameFor(semanticPointerBaseParamName(expression.argument.name), names)
    : nameFor(expression.argument.name, names);
  const type = pointer ? "u32" : semanticExpressionWgslType(expression.argument, ir);
  if (type === "bool" || isWgslVectorType(type)) {
    throw semanticWgslError(`WGSL update does not support '${type}'`, expression.span);
  }
  if (type !== "u32" && type !== "i32" && type !== "f16" && type !== "f32") {
    throw semanticWgslError(`WGSL update does not support pointer type '${type}'`, expression.span);
  }
  return createTypedWgslLocalAssignmentStatement(
    target,
    type,
    expression.operator === "++" ? "+=" : "-=",
    createTypedWgslLiteral(type === "u32" ? "1u" : type === "i32" ? "1" : type === "f16" ? "f16(1.0)" : "1.0", type, expression.span),
    expression.span,
  );
}

function emitSemanticReturnValue(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  if (expression.kind === "sequence") {
    const sequence = emitSemanticSequenceParts(expression, ir, names, indentLevel, options, textureSpecializations);
    return [
      ...sequence.prefix,
      ...emitSemanticReturnValue(sequence.value, ir, names, indentLevel, options, textureSpecializations),
    ];
  }
  if (expression.kind === "assignment") {
    const lines = emitSemanticExpressionStatement(expression, ir, names, indentLevel, options, textureSpecializations);
    const returnType = semanticActiveFunctionReturnType(ir, options, expression.span);
    const value = emitSemanticAssignmentResultExpression(expression, ir, names, options);
    const statement = createTypedWgslReturnStatement(wgslValueType(returnType), value, expression.span);
    return [...lines, `${prefix}${statement.code}`];
  }
  const returnType = semanticActiveFunctionReturnType(ir, options, expression.span);
  const expectedType = wgslValueType(returnType);
  const source = emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  const value = isSemanticFloatVectorType(returnType)
    ? source
    : source.type === "bool" && returnType !== "bool"
      ? legalizeTypedWgslBoolToNumeric(source, expectedType as "f16" | "f32" | "i32" | "u32")
      : emitSemanticExpressionAs(expression, ir, names, expectedType as WgslValueType, options, textureSpecializations);
  const statement = createTypedWgslReturnStatement(expectedType, value, expression.span);
  return [`${prefix}${statement.code}`];
}

function semanticActiveFunctionReturnType(
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions,
  span: SourceSpan,
): Exclude<CudaLiteScalarType, "void"> {
  const fn = options.activeFunction === undefined
    ? undefined
    : ir.functions.find((candidate) => candidate.name === options.activeFunction);
  if (!fn || fn.returnType === "void") {
    throw semanticWgslError("semantic WGSL value return requires active non-void function", span);
  }
  return fn.returnType;
}

function emitSemanticAssignmentResultExpression(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): TypedWgslExpression {
  if (expression.target.kind === "symbol") {
    return createTypedWgslIdentifier(
      nameFor(expression.target.name, names),
      semanticExpressionWgslType(expression, ir),
      expression.span,
    );
  }
  if (
    expression.target.kind === "member" && semanticWgslVectorMemberSupported(expression.target, ir) ||
    semanticWgslAssignmentMemoryRef(expression.target, ir)
  ) {
    return emitSemanticExpression(expression.target, ir, names, options);
  }
  throw semanticWgslError("semantic WGSL cannot return assignment result", expression.span);
}

function semanticTypedValueFunctionForCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): SemanticKernelIrModule["functions"][number] | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace !== "function") return undefined;
  const callee = expression.callee;
  const fn = ir.functions.find((candidate) =>
    semanticIdsEqual(callee.id, semanticSymbolIdFromFunction(candidate.id))
  );
  if (!fn || fn.returnType === "void") return undefined;
  if (fn.params.some((param) =>
    param.pointer && !(param.addressSpace === "constant" && param.pointerMemoryAlias !== undefined))) {
    return undefined;
  }
  return fn.params.length === expression.args.length ? fn : undefined;
}

function emitSemanticTypedValueFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  fn: SemanticKernelIrModule["functions"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression {
  const args = expression.args.flatMap((arg, index): readonly TypedWgslExpression[] => {
    const param = fn.params[index]!;
    if (param.cooperativeGroupKind !== undefined) return emitSemanticTypedCooperativeGroupArguments(arg, ir, names, options);
    if (param.pointer && param.addressSpace === "constant" && param.pointerMemoryAlias !== undefined) return [];
    if (param.addressSpace === "texture") {
      if (arg.kind !== "symbol" || arg.addressSpace !== "texture") throw semanticWgslError(`texture argument '${param.name}' is not a texture symbol`, arg.span);
      return [createTypedWgslIdentifier(nameFor(arg.name, names), "texture_2d<f32>", arg.span)];
    }
    if (param.addressSpace === "surface") {
      if (arg.kind !== "symbol" || arg.addressSpace !== "surface") throw semanticWgslError(`surface argument '${param.name}' is not a surface symbol`, arg.span);
      const handle = surfaceHandleForName(arg.name, ir);
      if (handle === undefined) throw semanticWgslError(`unknown surface '${arg.name}'`, arg.span);
      return [createTypedWgslLiteral(`${handle}u`, "u32", arg.span)];
    }
    if (param.valueType === "bool") return [emitSemanticBoolExpressionValue(arg, ir, names, options, textureSpecializations)];
    if (param.valueType === "uchar") return [emitSemanticUcharExpressionValue(arg, ir, names, options, textureSpecializations)];
    if (isSemanticFloatVectorType(param.valueType)) return [emitSemanticExpression(arg, ir, names, options, textureSpecializations)];
    return [emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param.valueType), options, textureSpecializations)];
  });
  args.push(
    createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("workgroup_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("num_workgroups", "vec3<u32>", expression.span),
  );
  const calleeName = semanticFunctionCallName(fn.name, fn, expression.args, options, textureSpecializations);
  return createTypedWgslCall(nameFor(calleeName, names), args, wgslValueType(fn.returnType), expression.span);
}

function emitSemanticTruthinessExpression(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): TypedWgslExpression {
  return emitSemanticBoolExpressionValue(expression, ir, names, options);
}

function semanticWgslSurfaceReadTarget(expression: SemanticExpression): { readonly name: string; readonly valueType?: CudaLiteScalarType } | undefined {
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "symbol" && expression.argument.addressSpace === "local") {
    return {
      name: expression.argument.name,
      ...(expression.argument.valueType === undefined ? {} : { valueType: expression.argument.valueType }),
    };
  }
  if (expression.kind === "symbol" && expression.addressSpace === "local") {
    return {
      name: expression.name,
      ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
    };
  }
  return undefined;
}

function emitSemanticSurfaceWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-write" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (!semanticWgslSurfaceWriteSupported(operation, ir) || operation.surface.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL supports only direct scalar surf2Dwrite", operation.span);
  }
  const prefix = "  ".repeat(indentLevel);
  const surfaceName = operation.surface.name;
  const xBytes = emitSemanticExpressionAs(operation.xBytes, ir, names, "i32", options, textureSpecializations).code;
  const y = emitSemanticExpressionAs(operation.y, ir, names, "i32", options, textureSpecializations).code;
  const z = operation.z ? emitSemanticExpressionAs(operation.z, ir, names, "i32", options, textureSpecializations).code : "0";
  const valueType = semanticExpressionVectorValueType(operation.value, ir?.functions);
  const value = isSemanticFloatVectorType(valueType)
    ? emitSemanticExpression(operation.value, ir, names, options, textureSpecializations).code
    : emitSemanticExpressionAs(operation.value, ir, names, "f32", options, textureSpecializations).code;
  const directSurface = surfaceSymbols(ir).find((surface) => surface.name === surfaceName);
  if (isSemanticFloatVectorType(valueType)) {
    return emitSemanticSurfaceVectorWrite(valueType, surfaceName, directSurface, value, xBytes, y, z, names, indentLevel);
  }
  if (!directSurface) return [`${prefix}${GENERIC_SURFACE_WRITE_HELPER_NAME}(${nameFor(surfaceName, names)}, ${value}, ${xBytes}, ${y}, ${z});`];
  return emitSemanticSurfaceWriteBody(directSurface, value, xBytes, y, z, names, indentLevel);
}

function emitSemanticSurfaceVectorWrite(
  valueType: CudaLiteScalarType | undefined,
  surfaceName: string,
  directSurface: SemanticKernelIrModule["params"][number] | undefined,
  value: string,
  xBytes: string,
  y: string,
  z: string,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const fields = ["x", "y", "z", "w"];
  const laneWrites = Array.from({ length: cudaVectorLaneCount(valueType) }).flatMap((_, lane) => {
    const laneValue = semanticSurfaceWriteLaneValue(value, valueType, fields[lane]!);
    const laneXBytes = `(${xBytes} + ${lane * 4})`;
    if (!directSurface) {
      return [`${prefix}${GENERIC_SURFACE_WRITE_HELPER_NAME}(${nameFor(surfaceName, names)}, ${laneValue}, ${laneXBytes}, ${y}, ${z});`];
    }
    return emitSemanticSurfaceWriteBody(directSurface, laneValue, laneXBytes, y, z, names, indentLevel);
  });
  return [
    `${prefix}if (${xBytes} >= 0 && (${xBytes} % 4) == 0) {`,
    ...laneWrites.map((line) => `${prefix}  ${line.slice(prefix.length)}`),
    `${prefix}}`,
  ];
}

function semanticSurfaceWriteLaneValue(value: string, valueType: CudaLiteScalarType | undefined, field: string): string {
  const laneValue = `(${value}).${field}`;
  return wgslVectorScalar(valueType) === "f32" ? laneValue : `f32(${laneValue})`;
}

function emitSemanticStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (semanticWgslStorageOffsetStoreSupported(operation, ir)) {
    const offset = nameFor(storageOffsetSymbol(operation.target.base), names);
    const value = emitSemanticExpressionAs(operation.value, ir, names, "i32", options, textureSpecializations).code;
    return operation.operator === "-=" ? `${offset} = (${offset} - ${value})` : `${offset} = (${offset} + ${value})`;
  }
  if (semanticWgslVectorFieldMemoryRefSupported(operation.target)) {
    return emitSemanticVectorFieldMemoryWrite(operation, ir, names, options, textureSpecializations).join("; ");
  }
  if (
    semanticWgslPackedSharedByteRoot(operation.target, ir) ||
    semanticWgslDirectByteRawView(operation.target, ir) ||
    semanticWgslLocalPackedHalfView(operation.target, ir) ||
    semanticWgslLocalScalarBitViewRootType(operation.target, ir) !== undefined ||
    semanticWgslLocalVectorBitViewRootType(operation.target, ir) !== undefined ||
    semanticWgslSharedScalarBitViewRootType(operation.target, ir) !== undefined
  ) {
    const value = isCudaVectorType(operation.target.valueType)
      ? emitSemanticExpression(operation.value, ir, names, options, textureSpecializations).code
      : emitSemanticScalarStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
    if (operation.operator === "=") return emitSemanticMemoryWrite(operation.target, value, ir, names, options);
    const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
    if (binaryOperator === undefined) throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
    const current = emitSemanticMemoryRead(operation.target, ir, names, options);
    return emitSemanticMemoryWrite(operation.target, `(${current} ${binaryOperator} ${value})`, ir, names, options);
  }
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base, options.activeFunction ?? null)) {
    return emitSemanticPointerMemoryStore(operation, ir, names, options, textureSpecializations);
  }
  if (
    semanticWgslFunctionSharedPointerParam(ir, operation.target.base, options.activeFunction ?? null) &&
    !semanticWgslFunctionSharedPointerAtomicParam(ir, operation.target.base)
  ) {
    if (semanticStorageVectorType(operation.target.valueType) !== undefined) {
      const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
      if (operation.operator !== "=" && binaryOperator === undefined) {
        throw semanticWgslError(`semantic WGSL does not support vector assignment '${operation.operator}'`, operation.span);
      }
      const target = emitSemanticMemoryRef(operation.target, ir, names, options);
      const value = emitSemanticVectorOperand(operation.value, operation.target.valueType as CudaLiteScalarType, ir, names, options, textureSpecializations);
      return `${target} = ${operation.operator === "=" ? value : `(${target} ${binaryOperator} ${value})`}`;
    }
    const value = emitSemanticScalarStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
    if (operation.operator === "=") return emitSemanticMemoryWrite(operation.target, value, ir, names, options);
    const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
    if (binaryOperator === undefined) throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
    const current = emitSemanticMemoryRead(operation.target, ir, names, options);
    return emitSemanticMemoryWrite(operation.target, `(${current} ${binaryOperator} ${value})`, ir, names, options);
  }
  const target = emitSemanticLocalVectorLaneRef(operation.target, ir, names, options, textureSpecializations) ??
    emitSemanticMemoryRef(operation.target, ir, names, options);
  const atomicRoot =
    semanticAtomicStorageNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticAtomicSharedNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticWgslFunctionSharedPointerAtomicParam(ir, operation.target.base);
  if (semanticStorageVectorType(operation.target.valueType) !== undefined) {
    const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
    if (operation.operator !== "=" && binaryOperator === undefined) {
      throw semanticWgslError(`semantic WGSL does not support vector assignment '${operation.operator}'`, operation.span);
    }
    const value = emitSemanticVectorOperand(operation.value, operation.target.valueType as CudaLiteScalarType, ir, names, options, textureSpecializations);
    if (operation.target.addressSpace === "local") {
      const access = emitSemanticMemoryRef(operation.target, ir, names, options);
      return `${access} = ${operation.operator === "=" ? value : `(${access} ${binaryOperator} ${value})`}`;
    }
    if (!atomicRoot && semanticWgslSharedVectorMemoryRef(operation.target, ir)) {
      const access = emitSemanticMemoryRef(operation.target, ir, names, options);
      return `${access} = ${operation.operator === "=" ? value : `(${access} ${binaryOperator} ${value})`}`;
    }
    return emitSemanticVectorMemoryWrite(operation, ir, names, options, textureSpecializations).join("; ");
  }
  if (atomicRoot) {
    if (operation.operator !== "=") {
      throw semanticWgslError(`semantic WGSL does not support atomic storage assignment '${operation.operator}'`, operation.span);
    }
    const atomicValue = emitSemanticAtomicStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
    return `atomicStore(&${target}, ${atomicValue})`;
  }
  const value = emitSemanticScalarStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
  if (operation.operator === "=") return `${target} = ${value}`;
  const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
  if (binaryOperator !== undefined) return `${target} = (${target} ${binaryOperator} ${value})`;
  throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
}

function emitSemanticScalarStoreValue(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  if (valueType === "bool") {
    return `select(0u, 1u, ${emitSemanticBoolExpression(expression, ir, names, options, textureSpecializations)})`;
  }
  if (valueType === "uchar") return emitSemanticUcharExpression(expression, ir, names, options, textureSpecializations);
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations).code;
}

function emitSemanticAtomicStoreValue(
  value: SemanticExpression,
  valueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  if (semanticAtomicUsesF32Storage(valueType) || valueType === "bf16") {
    return `bitcast<u32>(f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations).code}))`;
  }
  return emitSemanticExpressionAs(value, ir, names, wgslAtomicScalar(valueType), options, textureSpecializations).code;
}

function emitSemanticPointerMemoryStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const valueType = operation.target.valueType ?? "float";
  const index = isCudaVectorType(valueType)
    ? emitFlatStorageVectorBaseIndex(operation.target, ir, names, options)
    : emitFlatStorageIndex(operation.target, ir, names, options);
  const buffer = nameFor(semanticPointerBufferParamName(operation.target.base), names);
  const read = `${semanticPointerReadHelperName(valueType)}(${buffer}, ${index})`;
  const value = isSemanticFloatVectorType(valueType)
    ? emitSemanticVectorOperand(operation.value, valueType as CudaLiteScalarType, ir, names, options, textureSpecializations)
    : isCudaVectorType(valueType)
    ? emitSemanticExpression(operation.value, ir, names, options, textureSpecializations).code
    : emitSemanticExpressionAs(operation.value, ir, names, wgslValueScalar(valueType), options, textureSpecializations).code;
  const assigned = operation.operator === "=" ? value : `(${read} ${operation.operator.slice(0, -1)} ${value})`;
  return `${semanticPointerWriteHelperName(valueType)}(${buffer}, ${index}, ${assigned})`;
}

function emitSemanticLocalVectorLaneRef(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string | undefined {
  if (!semanticWgslLocalVectorLaneRefSupported(ref, ir)) return undefined;
  const [index] = ref.indices;
  if (!index) return undefined;
  return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(index, ir, names, "u32", options, textureSpecializations).code}]`;
}

function emitSemanticVectorMemoryWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
  emittedValue?: string,
  emittedBase?: string,
): readonly string[] {
  const valueType = semanticStorageVectorType(operation.target.valueType);
  if (!valueType) throw semanticWgslError("semantic WGSL vector write requires vector target", operation.span);
  const value = emittedValue ?? emitSemanticVectorOperand(operation.value, valueType, ir, names, options, textureSpecializations);
  const base = emittedBase ?? emitFlatStorageVectorBaseIndex(operation.target, ir, names, options);
  const target = nameFor(operation.target.base, names);
  const fields = ["x", "y", "z", "w"];
  const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
  if (operation.operator !== "=" && binaryOperator === undefined) {
    throw semanticWgslError(`semantic WGSL does not support vector assignment '${operation.operator}'`, operation.span);
  }
  const atomicRoot =
    semanticAtomicStorageNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticAtomicSharedNames(ir.operations, ir.functions).has(operation.target.base);
  const sharedBitRootType = semanticWgslSharedVectorBitViewRootType(operation.target, ir);
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) => {
    const access = `${target}[(${base} + ${lane}u)]`;
    const rawLaneValue = `(${value}).${fields[lane]}`;
    const laneValue = sharedBitRootType === undefined
      ? rawLaneValue
      : `bitcast<${wgslValueType(sharedBitRootType)}>(${rawLaneValue})`;
    if (!atomicRoot) {
      return `${access} = ${semanticPackedByteVectorLaneValue(operation.target, operation.operator === "=" ? laneValue : `(${access} ${binaryOperator} ${laneValue})`)}`;
    }
    const scalarType = cudaVectorScalarType(valueType);
    if (scalarType === undefined) throw semanticWgslError("semantic WGSL atomic vector write requires scalar lane type", operation.span);
    const current = semanticAtomicVectorLaneRead(access, scalarType);
    const assigned = operation.operator === "=" ? laneValue : `(${current} ${binaryOperator} ${laneValue})`;
    return `atomicStore(&${access}, ${semanticAtomicVectorLaneStore(assigned, scalarType)})`;
  });
}

function semanticAtomicVectorLaneRead(access: string, valueType: CudaLiteScalarType): string {
  if (valueType === "float" || valueType === "double" || valueType === "bf16") return `bitcast<f32>(atomicLoad(&${access}))`;
  if (valueType === "int") return `bitcast<i32>(atomicLoad(&${access}))`;
  return `atomicLoad(&${access})`;
}

function semanticAtomicVectorLaneStore(value: string, valueType: CudaLiteScalarType): string {
  if (valueType === "float" || valueType === "double" || valueType === "bf16") return `bitcast<u32>(f32(${value}))`;
  if (valueType === "int") return `bitcast<u32>(i32(${value}))`;
  return `u32(${value})`;
}

function semanticPackedByteVectorLaneValue(ref: SemanticMemoryRef, value: string): string {
  return ref.packedByteLanes === undefined ? value : `(u32(${value}) & 255u)`;
}

function emitSemanticVectorFieldMemoryWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const containerType = semanticStorageVectorType(operation.target.containerValueType);
  if (!containerType) throw semanticWgslError("semantic WGSL vector field write requires vector container", operation.span);
  const memoryContainerType = requireSemanticValueType(containerType, "vector field container", operation.span);
  const lanes = semanticStorageVectorFieldIndices(operation.target.containerValueType, operation.target.fields[0] ?? "");
  if (lanes === undefined) throw semanticWgslError("semantic WGSL vector field write requires modeled lanes", operation.span);
  if (semanticWgslSharedVectorMemoryRef({ ...operation.target, valueType: memoryContainerType, fields: [] }, ir)) {
    const target = emitSemanticMemoryRef({ ...operation.target, valueType: memoryContainerType, fields: [] }, ir, names, options);
    const field = lanes.map((lane) => ["x", "y", "z", "w"][lane]).join("");
    const access = `${target}.${field}`;
    const value = lanes.length === 1
      ? emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations).code
      : isCudaVectorType(operation.target.valueType)
      ? emitSemanticVectorOperand(operation.value, operation.target.valueType, ir, names, options, textureSpecializations)
      : undefined;
    if (value === undefined) throw semanticWgslError("semantic WGSL shared vector swizzle write requires vector value", operation.span);
    const assigned = operation.operator === "=" ? value : `(${access} ${operation.operator.slice(0, -1)} ${value})`;
    return [`${access} = ${assigned}`];
  }
  if (operation.target.addressSpace === "local") {
    const target = emitSemanticMemoryRef({ ...operation.target, valueType: memoryContainerType, fields: [] }, ir, names, options);
    const field = lanes.map((lane) => ["x", "y", "z", "w"][lane]).join("");
    if (lanes.length === 1) {
      const access = `${target}.${field}`;
      const value = emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations).code;
      if (operation.operator === "=") return [`${access} = ${value}`];
      return [`${access} = (${access} ${operation.operator.slice(0, -1)} ${value})`];
    }
    const valueType = operation.target.valueType;
    if (!isCudaVectorType(valueType)) throw semanticWgslError("semantic WGSL swizzle write requires vector value", operation.span);
    const value = emitSemanticVectorOperand(operation.value, valueType, ir, names, options, textureSpecializations);
    const access = `${target}.${field}`;
    const assigned = operation.operator === "=" ? value : `(${access} ${operation.operator.slice(0, -1)} ${value})`;
    return [`${access} = ${assigned}`];
  }
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base, options.activeFunction ?? null)) {
    return emitSemanticPointerVectorFieldMemoryWrite(operation, ir, names, options, textureSpecializations);
  }
  const base = emitFlatStorageVectorBaseIndex(operation.target, ir, names, options);
  const target = nameFor(operation.target.base, names);
  const fields = ["x", "y", "z", "w"];
  if (lanes.length === 1) {
    const access = `${target}[(${base} + ${lanes[0]}u)]`;
    const value = emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations).code;
    if (operation.operator === "=") return [`${access} = ${value}`];
    return [`${access} = (${access} ${operation.operator.slice(0, -1)} ${value})`];
  }
  const valueType = operation.target.valueType;
  if (!isCudaVectorType(valueType)) throw semanticWgslError("semantic WGSL swizzle write requires vector value", operation.span);
  const value = emitSemanticVectorOperand(operation.value, valueType, ir, names, options, textureSpecializations);
  const assigned = operation.operator === "="
    ? value
    : `(${wgslValueType(valueType)}(${lanes.map((lane) => `${target}[(${base} + ${lane}u)]`).join(", ")}) ${operation.operator.slice(0, -1)} ${value})`;
  return lanes.map((lane, index) => `${target}[(${base} + ${lane}u)] = (${assigned}).${fields[index]}`);
}

function emitSemanticPointerVectorFieldMemoryWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const containerType = operation.target.containerValueType;
  if (!isCudaVectorType(containerType)) throw semanticWgslError("semantic WGSL pointer swizzle write requires vector container", operation.span);
  const lanes = cudaVectorSwizzleIndices(containerType, operation.target.fields[0] ?? "");
  if (lanes === undefined) throw semanticWgslError("semantic WGSL pointer swizzle write requires modeled lanes", operation.span);
  const valueType = operation.target.valueType;
  const fields = ["x", "y", "z", "w"];
  const buffer = nameFor(semanticPointerBufferParamName(operation.target.base), names);
  const base = emitFlatStorageVectorBaseIndex(operation.target, ir, names, options);
  const read = `${semanticPointerReadHelperName(containerType)}(${buffer}, ${base})`;
  const value = lanes.length === 1
    ? emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations).code
    : isCudaVectorType(valueType)
    ? emitSemanticVectorOperand(operation.value, valueType, ir, names, options, textureSpecializations)
    : undefined;
  if (value === undefined) throw semanticWgslError("semantic WGSL pointer swizzle write requires vector value", operation.span);
  const assigned = operation.operator === "="
    ? value
    : lanes.length === 1
    ? `(${read}.${fields[lanes[0]!]} ${operation.operator.slice(0, -1)} ${value})`
    : `(${wgslValueType(valueType)}(${lanes.map((lane) => `${read}.${fields[lane]}`).join(", ")}) ${operation.operator.slice(0, -1)} ${value})`;
  const laneValues = Array.from({ length: cudaVectorLaneCount(containerType) }, (_, lane) => {
    const assignedIndex = lanes.indexOf(lane);
    return assignedIndex < 0 ? `${read}.${fields[lane]}` : lanes.length === 1 ? assigned : `(${assigned}).${fields[assignedIndex]}`;
  });
  return [`${semanticPointerWriteHelperName(containerType)}(${buffer}, ${base}, ${wgslValueType(containerType)}(${laneValues.join(", ")}))`];
}

function emitSemanticFunction(
  fn: SemanticKernelIrModule["functions"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  rawName = fn.name,
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const mutableParams = semanticFunctionMutableValueParams(fn);
  const params = [
    ...fn.params.flatMap((param) => emitSemanticFunctionParams(param, names, semanticFunctionSharedPointerAtomicParams(fn).has(param.name), mutableParams.has(param.name))),
    "local_id: vec3<u32>",
    "workgroup_id: vec3<u32>",
    "num_workgroups: vec3<u32>",
  ].join(", ");
  const returnType = fn.returnType === "void" ? "" : ` -> ${wgslValueType(fn.returnType)}`;
  return [
    `fn ${nameFor(rawName, names)}(${params})${returnType} {`,
    ...fn.params.filter((param) => mutableParams.has(param.name)).map((param) =>
      `  var ${nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param)} = ${semanticFunctionParamIncomingName(param, names)};`),
    ...emitSemanticOperations(fn.body, ir, names, 1, true, { ...options, activeFunction: fn.name }, textureSpecializations),
    ...(fn.returnType === "void" ? [] : [`  return ${zeroForType(wgslValueType(fn.returnType))};`]),
    "}",
  ];
}

function semanticFunctionMutableValueParams(
  fn: SemanticKernelIrModule["functions"][number],
): ReadonlySet<string> {
  const mutable = new Set<string>();
  const paramNames = new Set(fn.params
    .filter((param) => !param.pointer && !param.constant && param.cooperativeGroupKind === undefined && param.addressSpace === "local")
    .map((param) => param.name));
  walkSemanticOperations(fn.body, (expression) => {
    const target = expression.kind === "assignment"
      ? expression.target
      : expression.kind === "update"
      ? expression.argument
      : undefined;
    const name = target?.kind === "symbol"
      ? target.name
      : target?.kind === "member" && target.object.kind === "symbol"
      ? target.object.name
      : undefined;
    if (name && paramNames.has(name)) mutable.add(name);
  });
  return mutable;
}

function semanticFunctionParamIncomingName(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
  names: ReadonlyMap<string, string>,
): string {
  return `bg_arg_${nameFor(param.name, names)}`;
}

function emitSemanticFunctionParamType(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
  atomicSharedPointer = false,
): string {
  if (param.pointer && param.addressSpace === "local") {
    const valueType = wgslValueType(param.valueType);
    const count = param.dimensions.reduce((product, dimension) => product * dimension, 1);
    return param.dimensions.length === 0
      ? `ptr<function, ${valueType}>`
      : `ptr<function, array<${valueType}, ${count}>>`;
  }
  if (param.pointer && param.addressSpace === "shared") {
    if (param.pointerCarrierValueType === "uchar") {
      const words = Math.ceil((param.dimensions[0] ?? 1) / 4);
      return param.dimensions.length === 0
        ? "ptr<workgroup, atomic<u32>>"
        : `ptr<workgroup, array<atomic<u32>, ${words}>>`;
    }
    const carrierType = param.pointerCarrierValueType ?? param.valueType;
    const element = atomicSharedPointer ? `atomic<${wgslAtomicScalar(carrierType)}>` : wgslValueType(carrierType);
    return param.dimensions.length === 0 ? `ptr<workgroup, ${element}>` : `ptr<workgroup, array<${element}, ${param.dimensions[0] ?? 1}>>`;
  }
  if (param.addressSpace === "texture") return "texture_2d<f32>";
  if (param.addressSpace === "surface") return "u32";
  return wgslValueType(param.valueType);
}

function emitSemanticFunctionParams(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
  names: ReadonlyMap<string, string>,
  atomicSharedPointer = false,
  mutableValueParam = false,
): readonly string[] {
  if (param.pointer && param.addressSpace === "constant" && param.pointerMemoryAlias !== undefined) return [];
  if (param.cooperativeGroupKind !== undefined) {
    return [
      `${semanticCooperativeGroupRankParamName(param.name)}: i32`,
      `${semanticCooperativeGroupSizeParamName(param.name)}: i32`,
    ];
  }
  if (param.pointer && param.addressSpace === "storage") {
    return [
      `${nameFor(semanticPointerBufferParamName(param.name), names)}: u32`,
      `${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`,
    ];
  }
  if (param.pointer && param.addressSpace === "shared") {
    if (param.pointerParamAlias !== undefined) {
      return [`${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`];
    }
    return [
      `${nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param, atomicSharedPointer)}`,
      `${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`,
    ];
  }
  if (param.pointer && param.addressSpace === "local" && param.dimensions.length > 0) {
    return [
      `${nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param)}`,
      `${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`,
    ];
  }
  return [`${mutableValueParam ? semanticFunctionParamIncomingName(param, names) : nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param, atomicSharedPointer)}`];
}

function semanticFunctionParamAliasName(
  fn: SemanticKernelIrModule["functions"][number],
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): string | undefined {
  const aliasId = param.pointerParamAlias;
  return aliasId === undefined ? undefined : fn.params.find((candidate) => semanticIdsEqual(candidate.id, aliasId))?.name;
}

function semanticParamAliasName(
  ir: SemanticKernelIrModule,
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): string | undefined {
  for (const fn of ir.functions) {
    const name = semanticFunctionParamAliasName(fn, param);
    if (name !== undefined) return name;
  }
  return undefined;
}

function emitSemanticAssignmentStatement(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.target.kind === "member" && semanticWgslVectorMemberSupported(expression.target, ir)) {
    const target = emitSemanticMember(expression.target, ir, names, options);
    const targetValueType = semanticExpressionValueType(expression.target);
    const value = isCudaVectorType(targetValueType)
      ? emitSemanticVectorOperand(expression.value, targetValueType, ir, names, options, textureSpecializations)
      : emitSemanticExpressionAs(expression.value, ir, names, wgslVectorScalar(semanticExpressionVectorValueType(expression.target.object, ir?.functions)), options, textureSpecializations).code;
    if (isCudaVectorType(targetValueType) && expression.operator !== "=") {
      return `${target} = ${target} ${expression.operator.slice(0, -1)} ${value}`;
    }
    if (semanticAssignmentBinaryOperator(expression.operator)) return `${target} ${expression.operator} ${value}`;
    return `${target} = ${value}`;
  }
  {
    const ref = semanticWgslAssignmentMemoryRef(expression.target, ir);
    if (ref) {
      const target = emitSemanticMemoryRef(ref, ir, names, options);
      const value = emitSemanticExpressionAs(expression.value, ir, names, wgslValueScalar(ref.valueType), options, textureSpecializations).code;
      if (semanticAssignmentBinaryOperator(expression.operator)) return `${target} ${expression.operator} ${value}`;
      return `${target} = ${value}`;
    }
  }
  if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local assignment targets only", expression.target.span);
  const target = nameFor(expression.target.name, names);
  const targetType = expression.target.valueType;
  const value = targetType !== undefined && isSemanticFloatVectorType(targetType)
    ? emitSemanticVectorOperandExpression(expression.value, targetType, ir, names, options, textureSpecializations)
    : emitSemanticInitExpression(expression.value, targetType, ir, names, options, textureSpecializations);
  if (targetType === "uchar" && expression.operator !== "=") {
    const binaryOperator = expression.operator.slice(0, -1);
    const right = emitSemanticExpressionAs(expression.value, ir, names, "u32", options, textureSpecializations).code;
    return `${target} = ${emitSemanticUcharValue(`(${target} ${binaryOperator} ${right})`)}`;
  }
  const binaryOperator = semanticAssignmentBinaryOperator(expression.operator);
  const promotedType = binaryOperator === undefined
    ? undefined
    : promotedCudaScalarType(targetType, semanticExpressionValueType(expression.value));
  if (binaryOperator !== undefined && promotedType !== undefined && wgslValueScalar(promotedType) !== wgslValueScalar(targetType)) {
    const operationScalar = wgslValueScalar(promotedType);
    const left = `${operationScalar}(${target})`;
    const right = emitSemanticExpressionAs(expression.value, ir, names, operationScalar, options, textureSpecializations).code;
    return `${target} = ${wgslValueScalar(targetType)}((${left} ${binaryOperator} ${right}))`;
  }
  const operator = (binaryOperator ? expression.operator : "=") as Parameters<typeof createTypedWgslLocalAssignmentStatement>[2];
  return createTypedWgslLocalAssignmentStatement(
    target,
    wgslValueType(targetType),
    operator,
    value,
    expression.span,
  ).code.slice(0, -1);
}

function emitLocalArrayInit(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (!operation.init) return [];
  const prefix = "  ".repeat(indentLevel);
  if (operation.init.kind !== "initializer") {
    const value = isSemanticFloatVectorType(operation.target.valueType)
      ? emitSemanticExpression(operation.init, ir, names, options, textureSpecializations).code
      : emitSemanticExpressionAs(operation.init, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations).code;
    return emitLocalArrayFill(
      nameFor(operation.target.name, names),
      operation.target.dimensions,
      value,
      indentLevel,
    );
  }
  return flattenInitializerExpressions(operation.init)
    .slice(0, totalElements(operation.target.dimensions))
    .map((value, index) => {
      const indices = flatIndicesForDimensions(operation.target.dimensions, index)
        .map((item) => `[${item}u]`)
        .join("");
      const emittedValue = isSemanticFloatVectorType(operation.target.valueType)
        ? emitSemanticExpression(value, ir, names, options, textureSpecializations).code
        : emitSemanticExpressionAs(value, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations).code;
      return `${prefix}${nameFor(operation.target.name, names)}${indices} = ${emittedValue};`;
    });
}

function emitSemanticStorageOffsetDeclarations(
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  return [...semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets)]
    .sort()
    .map((base) => {
      const pointerBase = options.pointerBaseOffsets?.[base] === undefined
        ? "0"
        : `i32(${UNIFORM_PARAMS_NAME}.${nameFor(pointerBaseOffsetUniformName(base), names)})`;
      return `${prefix}var ${nameFor(storageOffsetSymbol(base), names)}: i32 = ${pointerBase};`;
    });
}

function emitSemanticAtomic(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const wgslCallee = wgslAtomicCalleeForCudaAtomic(operation.callee);
  const loopAtomicKind = wgslIntegerLoopAtomicKindForCudaAtomic(operation.callee);
  if (!operation.target || (!wgslCallee && !loopAtomicKind && !semanticWgslAtomicValueTypeSupported(operation.callee, operation.target.valueType))) {
    throw semanticWgslError(`semantic WGSL does not support atomic '${operation.callee}'`, operation.span);
  }
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base, options.activeFunction ?? null)) {
    if (operation.target.valueType === undefined) {
      throw semanticWgslError(`atomic '${operation.callee}' target has no value type`, operation.span);
    }
    const pointerCall = emitSemanticPointerAtomicCall({
      kind: "call",
      callee: { kind: "symbol", id: createBuiltinSemanticSymbolId(operation.callee), name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: operation.target.valueType,
      span: operation.span,
    }, operation.target, ir, names, options, textureSpecializations);
    if (!pointerCall) throw semanticWgslError(`semantic WGSL pointer atomic '${operation.callee}' is unsupported`, operation.span);
    return `_ = ${pointerCall}`;
  }
  if (semanticWgslDirectByteRawView(operation.target, ir) && semanticAtomicMemoryRootNames(ir).has(operation.target.base)) {
    const byteIndex = emitFlatStorageIndex({ ...operation.target, valueType: "uchar" }, ir, names, options);
    const operands = operation.args.slice(1, semanticAtomicOperation(operation.callee) === "cas" ? 3 : 2);
    if (operands.length === 0) throw semanticWgslError(`semantic WGSL atomic '${operation.callee}' missing operand`, operation.span);
    const scalar = wgslValueScalar(operation.target.valueType);
    const emitted = operands.map((operand) => emitSemanticExpressionAs(operand!, ir, names, scalar, options, textureSpecializations).code);
    return `_ = ${emitSemanticAtomicByteStorageAtomicValue(
      operation.callee,
      operation.target.valueType ?? "uint",
      nameFor(operation.target.base, names),
      byteIndex,
      emitted[0] ?? "0",
      emitted.at(-1) ?? "0",
    )}`;
  }
  const target = emitSemanticMemoryRef(operation.target, ir, names, options);
  const operands = operation.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  if (operands.length === 0 || operands.some((operand) => operand === undefined)) {
    throw semanticWgslError(`semantic WGSL atomic '${operation.callee}' missing operand`, operation.span);
  }
  if (loopAtomicKind) {
    const value = emitSemanticExpressionAs(operands[0]!, ir, names, "u32", options, textureSpecializations).code;
    return `_ = ${semanticIntegerLoopAtomicHelperName(loopAtomicKind, operation.target, ir)}(&${target}, ${value})`;
  }
  if (semanticAtomicSupportsBfloatAdd(operation.callee, operation.target.valueType)) {
    const value = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations).code;
    return `_ = ${bfloatAtomicAddHelperName(semanticWgslAtomicAddressSpace(operation.target))}(&${target}, ${value})`;
  }
  const floatAtomicKind = semanticAtomicUsesF32Storage(operation.target.valueType) ? semanticWgslFloatAtomicCallKind(operation.callee) : undefined;
  if (floatAtomicKind) {
    const addressSpace = semanticWgslAtomicAddressSpace(operation.target);
    if (floatAtomicKind === "Exchange") {
      const value = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations).code;
      return `_ = atomicExchange(&${target}, bitcast<u32>(${value}))`;
    }
    if (floatAtomicKind === "CompareExchange") {
      const compare = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations).code;
      const value = emitSemanticExpressionAs(operands[1]!, ir, names, "f32", options, textureSpecializations).code;
      return `_ = atomicCompareExchangeWeak(&${target}, bitcast<u32>(${compare}), bitcast<u32>(${value}))`;
    }
    const value = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations).code;
    return `_ = ${floatAtomicHelperName(floatAtomicKind, addressSpace)}(&${target}, ${value})`;
  }
  const emitted = operands.map((operand) =>
    emitSemanticExpressionAs(operand!, ir, names, wgslAtomicScalar(operation.target!.valueType), options, textureSpecializations).code
  );
  return `_ = ${wgslCallee}(&${target}, ${emitted.join(", ")})`;
}

type SemanticFloatAtomicKind = "Add" | "Sub" | "Min" | "Max" | "Exchange" | "CompareExchange";

function semanticWgslFloatAtomicCallKind(callee: string): SemanticFloatAtomicKind | undefined {
  switch (semanticAtomicOperation(callee)) {
    case "add": return "Add";
    case "sub": return "Sub";
    case "min": return "Min";
    case "max": return "Max";
    case "exchange": return "Exchange";
    case "cas": return "CompareExchange";
    default: return undefined;
  }
}

function semanticWgslAtomicValueTypeSupported(callee: string, valueType: CudaLiteScalarType | undefined): boolean {
  const atomicOp = semanticAtomicOperation(callee);
  if (!atomicOp) return false;
  if (valueType === "uint" || valueType === "int") {
    return wgslAtomicCalleeForCudaAtomic(callee) !== undefined || wgslIntegerLoopAtomicKindForCudaAtomic(callee) !== undefined;
  }
  if (semanticAtomicUsesF32Storage(valueType)) return semanticAtomicSupportsFloat(atomicOp);
  return semanticAtomicSupportsBfloatAdd(callee, valueType);
}

function semanticWgslAtomicAddressSpace(ref: SemanticMemoryRef): WgslAtomicAddressSpace {
  return ref.addressSpace === "shared" ? "workgroup" : "storage";
}

function semanticIntegerLoopAtomicHelperName(kind: WgslIntegerLoopAtomicKind, ref: SemanticMemoryRef, ir: SemanticKernelIrModule): string {
  const storageValueType = semanticMemoryRefStorageValueType(ref, ir) ?? ref.valueType ?? "uint";
  return integerAtomicLoopHelperName(kind, {
    valueType: ref.valueType ?? "uint",
    storageValueType,
    storageScalar: wgslAtomicScalar(storageValueType),
    addressSpace: semanticWgslAtomicAddressSpace(ref),
  });
}

function semanticMemoryRefStorageValueType(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): CudaLiteScalarType | undefined {
  if (ref.addressSpace === "storage") {
    return ir.params.find((param) => param.name === ref.base && param.addressSpace === "storage")?.valueType;
  }
  if (ref.addressSpace === "shared" || ref.addressSpace === "device-global") {
    return ir.memory.find((symbol) => symbol.name === ref.base && symbol.kind === ref.addressSpace)?.valueType;
  }
  return ref.valueType;
}

function semanticUsesIntegerLoopAtomic(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) => {
    if (operation.kind === "atomic" && wgslIntegerLoopAtomicKindForCudaAtomic(operation.callee) !== undefined) return true;
    if (operation.kind === "store" && semanticExpressionUsesIntegerLoopAtomic(operation.value)) return true;
    if (operation.kind === "declare" && operation.init && semanticExpressionUsesIntegerLoopAtomic(operation.init)) return true;
    if (operation.kind === "expression" && semanticExpressionUsesIntegerLoopAtomic(operation.expression)) return true;
    if (operation.kind === "branch") return semanticUsesIntegerLoopAtomic(operation.consequent) || semanticUsesIntegerLoopAtomic(operation.alternate);
    if (operation.kind === "loop") return semanticUsesIntegerLoopAtomic(operation.body);
    if (operation.kind === "block") return semanticUsesIntegerLoopAtomic(operation.body);
    return false;
  });
}

function semanticExpressionUsesIntegerLoopAtomic(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && wgslIntegerLoopAtomicKindForCudaAtomic(expression.callee.name) !== undefined) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesIntegerLoopAtomic);
}

function semanticFloatAtomicHelpers(
  operations: readonly SemanticKernelIrOperation[],
  requirePointerFloatRmw = false,
): readonly string[][] {
  const helperKeys = new Set<string>();
  collectSemanticFloatAtomicHelpers(operations, helperKeys);
  if (requirePointerFloatRmw) {
    helperKeys.add("Add:storage");
    helperKeys.add("Sub:storage");
    helperKeys.add("Min:storage");
    helperKeys.add("Max:storage");
  }
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind !== "call" || expression.callee.kind !== "symbol") return;
    const target = semanticAtomicCallTarget(expression);
    if (target && semanticAtomicSupportsBfloatAdd(expression.callee.name, target.valueType)) {
      helperKeys.add(`BfloatAdd:${semanticWgslAtomicAddressSpace(target)}`);
      return;
    }
    if (!target || !semanticAtomicUsesF32Storage(target.valueType)) return;
    const kind = semanticWgslFloatAtomicCallKind(expression.callee.name);
    if (kind && kind !== "Exchange" && kind !== "CompareExchange") {
      helperKeys.add(`${kind}:${semanticWgslAtomicAddressSpace(target)}`);
    }
  });
  return [...helperKeys].flatMap((key) => {
    const [kind, addressSpace] = key.split(":") as [Exclude<SemanticFloatAtomicKind, "Exchange" | "CompareExchange"> | "BfloatAdd", WgslAtomicAddressSpace];
    if (kind === "BfloatAdd") return [emitBfloatAtomicAddHelper(addressSpace)];
    if (kind === "Add") return [emitFloatAtomicAddHelper(addressSpace)];
    if (kind === "Sub") return [emitFloatAtomicSubHelper(addressSpace)];
    if (kind === "Min") return [emitFloatAtomicMinHelper(addressSpace)];
    if (kind === "Max") return [emitFloatAtomicMaxHelper(addressSpace)];
    return [];
  });
}

function collectSemanticFloatAtomicHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helperKeys: Set<string>,
): void {
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target && semanticAtomicSupportsBfloatAdd(operation.callee, operation.target.valueType)) {
      helperKeys.add(`BfloatAdd:${semanticWgslAtomicAddressSpace(operation.target)}`);
    }
    if (operation.kind === "atomic" && operation.target && semanticAtomicUsesF32Storage(operation.target.valueType)) {
      const kind = semanticWgslFloatAtomicCallKind(operation.callee);
      if (kind && kind !== "Exchange" && kind !== "CompareExchange") helperKeys.add(`${kind}:${semanticWgslAtomicAddressSpace(operation.target)}`);
    }
    if (operation.kind === "branch") {
      collectSemanticFloatAtomicHelpers(operation.consequent, helperKeys);
      collectSemanticFloatAtomicHelpers(operation.alternate, helperKeys);
    }
    if (operation.kind === "loop") collectSemanticFloatAtomicHelpers(operation.body, helperKeys);
    if (operation.kind === "block") collectSemanticFloatAtomicHelpers(operation.body, helperKeys);
  }
}

function emitSemanticCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (operation.result !== undefined) {
    const fn = semanticFunctionForCall(operation, ir.functions);
    if (!fn || fn.returnType === "void") throw semanticWgslError(`semantic WGSL call '${operation.callee}' cannot produce a result`, operation.span);
    const call = emitSemanticFunctionCall(semanticCallOperationExpression(operation, fn), ir, names, options, textureSpecializations);
    return [`${"  ".repeat(indentLevel)}${nameFor(operation.result.name, names)} = ${call};`];
  }
  if (operation.callee === "assert") return [];
  if (operation.callee === "printf") return [];
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) {
    const prefix = "  ".repeat(indentLevel);
    return operation.args.map((arg, index) =>
      `${prefix}let ${nameFor(`bg_noop_arg_${operation.span.start}_${index}`, names)} = ${emitSemanticExpression(arg, ir, names, options, textureSpecializations).code};`
    );
  }
  if (operation.callee === "curand_init") return [`${"  ".repeat(indentLevel)}${emitSemanticCurandInit(operation, ir, names, options, textureSpecializations)};`];
  if (operation.callee === "skipahead") {
    return [`${"  ".repeat(indentLevel)}${emitSemanticCurandCall({
      kind: "call",
      callee: { kind: "symbol", id: createBuiltinSemanticSymbolId(operation.callee), name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: "uint",
      span: operation.span,
    }, ir, names, options, textureSpecializations)};`];
  }
  if (semanticWgslVoidFunctionCallSupported(operation, ir)) return [`${"  ".repeat(indentLevel)}${emitSemanticVoidFunctionCall(operation, ir, names, options, textureSpecializations)};`];
  if (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) return emitSemanticLocalArrayFill(operation, ir, names, indentLevel, options, textureSpecializations);
  throw semanticWgslError(`semantic WGSL does not support call '${operation.callee}'`, operation.span);
}

function semanticCallOperationExpression(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  fn: SemanticKernelIrModule["functions"][number],
): Extract<SemanticExpression, { readonly kind: "call" }> {
  return {
    kind: "call",
    callee: { kind: "symbol", id: createSemanticSymbolId("function", fn.name, fn.span), name: operation.callee, addressSpace: "function", valueType: fn.returnType, span: operation.span },
    args: operation.args,
    valueType: fn.returnType,
    span: operation.span,
  };
}

function emitSemanticVoidFunctionCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const fn = semanticFunctionForCall(operation, ir.functions);
  if (!fn) throw semanticWgslError(`semantic WGSL unknown function '${operation.callee}'`, operation.span);
  const callee = semanticFunctionCallName(operation.callee, fn, operation.args, options, textureSpecializations);
  const args = operation.args.flatMap((arg, index) => emitSemanticFunctionArgs(arg, fn.params[index], ir, names, options, textureSpecializations));
  return `${nameFor(callee, names)}(${[...args, "local_id", "workgroup_id", "num_workgroups"].join(", ")})`;
}

function emitSemanticLocalArrayFill(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const [target, valueExpression] = operation.args;
  if (target?.kind !== "symbol" || target.addressSpace !== "local" || valueExpression === undefined) {
    throw semanticWgslError(`${operation.callee} expects local array and fill value`, operation.span);
  }
  const symbol = localArraySymbol(ir, target.name);
  if (!symbol) throw semanticWgslError(`${operation.callee} expects fixed local array '${target.name}'`, target.span);
  const value = isSemanticFloatVectorType(symbol.valueType)
    ? emitSemanticExpression(valueExpression, ir, names, options, textureSpecializations).code
    : emitSemanticExpressionAs(valueExpression, ir, names, wgslValueScalar(symbol.valueType), options, textureSpecializations).code;
  return emitLocalArrayFill(
    nameFor(target.name, names),
    symbol.dimensions,
    value,
    indentLevel,
  );
}

function emitSemanticCurandInit(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const state = operation.args[3];
  const pointer = semanticCurandStatePointer(state, ir, names, options);
  if (!pointer || operation.args.length !== 4) throw semanticWgslError("curand_init expects a modeled state address", operation.span);
  const suffix = pointer.addressSpace === "storage" ? "_storage" : pointer.addressSpace === "workgroup" ? "_workgroup" : "";
  const seed = emitSemanticExpressionAs(operation.args[0]!, ir, names, "u32", options, textureSpecializations).code;
  const sequence = emitSemanticExpressionAs(operation.args[1]!, ir, names, "u32", options, textureSpecializations).code;
  const offset = emitSemanticExpressionAs(operation.args[2]!, ir, names, "u32", options, textureSpecializations).code;
  return `bg_curand_init${suffix}(${seed}, ${sequence}, ${offset}, ${pointer.expression})`;
}

function emitSemanticCurandCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL cuRAND call requires symbol callee", expression.span);
  if (expression.callee.name === "curand_init") {
    return emitSemanticCurandInit({
      kind: "call",
      calleeId: expression.callee.id,
      callee: expression.callee.name,
      args: expression.args,
      reads: [],
      span: expression.span,
    }, ir, names, options, textureSpecializations);
  }
  if (expression.callee.name === "skipahead") {
    const pointer = semanticCurandStatePointer(expression.args[1], ir, names, options);
    if (!pointer) throw semanticWgslError("skipahead expects a modeled state address", expression.span);
    const suffix = pointer.addressSpace === "storage" ? "_storage" : pointer.addressSpace === "workgroup" ? "_workgroup" : "";
    const count = emitSemanticExpressionAs(expression.args[0]!, ir, names, "u32", options, textureSpecializations).code;
    return `bg_curand_skipahead${suffix}(${count}, ${pointer.expression})`;
  }
  const pointer = semanticCurandStatePointer(expression.args[0], ir, names, options);
  if (!pointer) throw semanticWgslError(`${expression.callee.name} expects a modeled state address`, expression.span);
  const suffix = pointer.addressSpace === "storage" ? "_storage" : pointer.addressSpace === "workgroup" ? "_workgroup" : "";
  if (expression.callee.name === "curand") {
    return `bg_curand${suffix}(${pointer.expression})`;
  }
  if (expression.callee.name === "curand_uniform" || expression.callee.name === "curand_uniform_double") {
    return `bg_curand_uniform${suffix}(${pointer.expression})`;
  }
  if (expression.callee.name === "curand_uniform4") {
    return `bg_curand_uniform4${suffix}(${pointer.expression})`;
  }
  if (expression.callee.name === "curand_normal" || expression.callee.name === "curand_normal_double") {
    return `bg_curand_normal${suffix}(${pointer.expression})`;
  }
  if (expression.callee.name === "curand_normal2") {
    return `bg_curand_normal2${suffix}(${pointer.expression})`;
  }
  if (expression.callee.name === "curand_normal4") {
    return `bg_curand_normal4${suffix}(${pointer.expression})`;
  }
  if (expression.callee.name === "curand_log_normal" || expression.callee.name === "curand_log_normal_double") {
    const mean = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations).code;
    const stddev = emitSemanticExpressionAs(expression.args[2]!, ir, names, "f32", options, textureSpecializations).code;
    return `bg_curand_log_normal${suffix}(${pointer.expression}, ${mean}, ${stddev})`;
  }
  if (expression.callee.name === "curand_log_normal2") {
    const mean = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations).code;
    const stddev = emitSemanticExpressionAs(expression.args[2]!, ir, names, "f32", options, textureSpecializations).code;
    return `bg_curand_log_normal2${suffix}(${pointer.expression}, ${mean}, ${stddev})`;
  }
  if (expression.callee.name === "curand_log_normal4") {
    const mean = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations).code;
    const stddev = emitSemanticExpressionAs(expression.args[2]!, ir, names, "f32", options, textureSpecializations).code;
    return `bg_curand_log_normal4${suffix}(${pointer.expression}, ${mean}, ${stddev})`;
  }
  if (expression.callee.name === "curand_poisson") {
    const lambda = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations).code;
    return `bg_curand_poisson${suffix}(${pointer.expression}, ${lambda})`;
  }
  if (expression.callee.name === "curand_poisson4") {
    const lambda = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations).code;
    return `bg_curand_poisson4${suffix}(${pointer.expression}, ${lambda})`;
  }
  throw semanticWgslError(`semantic WGSL does not support cuRAND call '${expression.callee.name}'`, expression.span);
}

function emitSemanticLoop(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  allowReturnValue = false,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  if (operation.loopKind === "for") {
    if (operation.continuing !== undefined || operation.update?.kind === "sequence") {
      const init = operation.init === undefined
        ? []
        : isSemanticKernelIrOperation(operation.init)
          ? emitSemanticOperations([operation.init], ir, names, indentLevel, allowReturnValue, options, textureSpecializations)
          : emitSemanticExpressionStatement(operation.init, ir, names, indentLevel, options, textureSpecializations);
      const condition = operation.condition ? emitTruthiness(operation.condition, ir, names, options) : undefined;
      return [
        ...init,
        `${prefix}loop {`,
        ...(condition === undefined ? [] : [`${"  ".repeat(indentLevel + 1)}if (!(${condition})) { break; }`]),
        ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
        `${"  ".repeat(indentLevel + 1)}continuing {`,
        ...(operation.continuing === undefined
          ? emitSemanticSequenceStatement(operation.update as Extract<SemanticExpression, { readonly kind: "sequence" }>, ir, names, indentLevel + 2, options, textureSpecializations)
          : emitSemanticOperations(operation.continuing, ir, names, indentLevel + 2, allowReturnValue, options, textureSpecializations)),
        `${"  ".repeat(indentLevel + 1)}}`,
        `${prefix}}`,
      ];
    }
    const init = operation.init ? emitSemanticLoopInit(operation.init, ir, names, options, textureSpecializations) : "";
    const condition = operation.condition ? emitTruthiness(operation.condition, ir, names, options) : "true";
    const update = operation.update ? emitSemanticLoopUpdate(operation.update, ir, names, options, textureSpecializations) : "";
    return [
      `${prefix}for (${init}; ${condition}; ${update}) {`,
      ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
      `${prefix}}`,
    ];
  }
  if (operation.loopKind === "while") {
    return [
      `${prefix}while (${operation.condition ? emitTruthiness(operation.condition, ir, names, options) : "true"}) {`,
      ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
      `${prefix}}`,
    ];
  }
  const continuingBreak = operation.continuing?.at(-1);
  const hasTerminalBreak = continuingBreak?.kind === "branch" && continuingBreak.consequent.length === 0 &&
    continuingBreak.alternate.length === 1 && continuingBreak.alternate[0]?.kind === "break";
  const continuingPrefix = operation.continuing === undefined
    ? undefined
    : hasTerminalBreak ? operation.continuing.slice(0, -1) : operation.continuing;
  return [
    `${prefix}loop {`,
    ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
    `${"  ".repeat(indentLevel + 1)}continuing {`,
    ...(continuingPrefix === undefined
      ? [`${"  ".repeat(indentLevel + 2)}break if !(${operation.condition ? emitTruthiness(operation.condition, ir, names, options) : "false"});`]
      : [
          ...emitSemanticOperations(continuingPrefix, ir, names, indentLevel + 2, allowReturnValue, options, textureSpecializations),
          ...(hasTerminalBreak && continuingBreak?.kind === "branch"
            ? [`${"  ".repeat(indentLevel + 2)}break if !(${emitTruthiness(continuingBreak.condition, ir, names, options)});`]
            : []),
        ]),
    `${"  ".repeat(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitSemanticLoopUpdate(
  update: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (isSemanticNoopExpression(update)) return "";
  if (update.kind === "sequence") throw semanticWgslError("semantic WGSL sequence loop updates require loop lowering", update.span);
  if (update.kind === "update") return emitSemanticLocalUpdateStatement(update, ir, names, options).code.slice(0, -1);
  return update.kind === "assignment"
    ? emitSemanticAssignmentStatement(update, ir, names, options, textureSpecializations)
    : emitSemanticExpression(update, ir, names, options, textureSpecializations).code;
}

function emitSemanticLoopInit(
  init: SemanticKernelIrOperation | SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (!isSemanticKernelIrOperation(init)) return emitSemanticExpression(init, ir, names, options, textureSpecializations).code;
  if (init.kind === "declare") {
    const type = wgslScalar(init.target.valueType);
    const value = init.init ? emitSemanticLocalScalarExpressionAs(init.init, init.target.valueType, ir, names, options, textureSpecializations) : zeroForType(type);
    return `var ${nameFor(init.target.name, names)}: ${type} = ${value}`;
  }
  if (init.kind === "expression") return isSemanticNoopExpression(init.expression) ? "" : emitSemanticExpression(init.expression, ir, names, options, textureSpecializations).code;
  throw semanticWgslError(`semantic WGSL does not support ${init.kind} loop initializer`, init.span);
}

function isSemanticNoopExpression(expression: SemanticExpression): boolean {
  return expression.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
}

function emitSemanticExpression(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (expression.kind === "literal" && expression.literalKind === "number") {
    const type = semanticExpressionWgslType(expression, ir);
    if ((type === "f32" || type === "f16") && Math.abs(expression.value) === 3.4028234663852886e38) {
      const value = createTypedWgslBitcast(
        "f32",
        createTypedWgslLiteral(expression.value < 0 ? "0xff7fffffu" : "0x7f7fffffu", "u32", expression.span),
        expression.span,
      );
      return type === "f16" ? convertTypedWgslExpression(value, "f16", true) : value;
    }
    if ((type === "f32" || type === "f16") && !Number.isFinite(expression.value)) {
      const value = Number.isNaN(expression.value)
        ? createTypedWgslCall("bg_f32_nan", [], "f32", expression.span)
        : expression.value < 0
          ? emitTypedWgslUnary("-", createTypedWgslCall("bg_f32_inf", [], "f32", expression.span), expression.span)
          : createTypedWgslCall("bg_f32_inf", [], "f32", expression.span);
      return type === "f16" ? convertTypedWgslExpression(value, "f16", true) : value;
    }
    if (type === "bool" || type === "f16" || type === "f32" || type === "i32" || type === "u32") {
      const code = emitNumberLiteral(expression.value, expression.valueType);
      if (isTypedWgslLiteralCode(code, type)) return createTypedWgslLiteral(code, type, expression.span);
    }
  }
  if (expression.kind === "symbol" && semanticWgslSymbolHasTypedEmission(expression, ir)) {
    return emitSemanticSymbolExpression(expression, ir, names);
  }
  if (expression.kind === "symbol") {
    const constantVector = emitSemanticConstantVectorSymbolExpression(expression, ir, names);
    if (constantVector) return constantVector;
    const scalarMemory = emitSemanticScalarMemorySymbolExpression(expression, ir, names, options);
    if (scalarMemory) return scalarMemory;
  }
  if (expression.kind === "assignment") {
    throw semanticWgslError(`typed WGSL emission missing for 'assignment:${expression.target.kind}:${expression.operator}' expression`, expression.span);
  }
  if (expression.kind === "pointer-valid") {
    const pointer = semanticWgslPointerValidityOwner(expression, ir);
    if (!pointer) throw semanticWgslError(`unresolved pointer validity identity for '${expression.pointer}'`, expression.span);
    return emitTypedWgslBinary(
      "!=",
      createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(pointer.name), names), "u32", expression.span),
      createTypedWgslLiteral("4294967295u", "u32", expression.span),
      expression.span,
    );
  }
  if (expression.kind === "member") {
    const byteVectorMember = semanticDirectByteVectorMemberRef(expression, ir);
    if (byteVectorMember) return emitSemanticTypedDirectByteRawRefRead(byteVectorMember, ir, names, options);
    return emitSemanticMemberExpression(expression, ir, names, options);
  }
  if (expression.kind === "cast" && !expression.pointer) {
    return emitSemanticCastExpression(expression, ir, names, options, textureSpecializations);
  }
  if (expression.kind === "index" && semanticWgslVectorIndexSupported(expression, ir)) {
    return createTypedWgslIndexAccess(
      emitSemanticExpression(expression.target, ir, names, options, textureSpecializations),
      emitSemanticExpressionAs(expression.index, ir, names, "u32", options, textureSpecializations),
      semanticExpressionWgslType(expression, ir),
      expression.span,
    );
  }
  if (expression.kind === "index") {
    const localSpecialRead = emitSemanticTypedLocalSpecialRead(expression, ir, names, options);
    if (localSpecialRead) return localSpecialRead;
    const sharedBitRead = emitSemanticTypedSharedBitRead(expression, ir, names, options);
    if (sharedBitRead) return sharedBitRead;
    const directByteRead = emitSemanticTypedDirectByteRawRead(expression, ir, names, options);
    if (directByteRead) return directByteRead;
    const packedSharedRead = emitSemanticTypedPackedSharedByteRead(expression, ir, names, options);
    if (packedSharedRead) return packedSharedRead;
    const sharedVectorScalarRead = emitSemanticSharedVectorScalarReadExpression(expression, ir, names, options);
    if (sharedVectorScalarRead) return sharedVectorScalarRead;
    const directSharedVectorRead = emitSemanticDirectSharedVectorReadExpression(expression, ir, names, options);
    if (directSharedVectorRead) return directSharedVectorRead;
    const directLocalPointerVectorRead = emitSemanticDirectLocalPointerVectorReadExpression(expression, ir, names, options);
    if (directLocalPointerVectorRead) return directLocalPointerVectorRead;
    const storageVectorScalarRead = emitSemanticStorageVectorScalarReadExpression(expression, ir, names, options);
    if (storageVectorScalarRead) return storageVectorScalarRead;
    const directVectorRead = emitSemanticDirectStorageVectorReadExpression(expression, ir, names, options);
    if (directVectorRead) return directVectorRead;
    const directMemoryRead = emitSemanticDirectMemoryReadExpression(expression, ir, names, options);
    if (directMemoryRead) return directMemoryRead;
  }
  if (expression.kind === "texture-read") {
    return emitSemanticTypedTextureRead(expression, ir, names, options);
  }
  if (expression.kind === "surface-read") {
    return emitSemanticTypedSurfaceRead(expression, ir, names, options);
  }
  if (expression.kind === "sequence") {
    return emitSemanticExpression(
      expression.expressions.at(-1) ?? zeroExpression(expression.span),
      ir,
      names,
      options,
      textureSpecializations,
    );
  }
  if (expression.kind === "call") {
    const coalescedGroup = emitSemanticTypedCoalescedGroupCall(expression, ir, names, options, textureSpecializations);
    if (coalescedGroup) return coalescedGroup;
    const cooperativeGroup = emitSemanticTypedCooperativeGroupCall(expression, ir, options);
    if (cooperativeGroup) return cooperativeGroup;
    const cooperativeReduce = emitSemanticTypedCooperativeReduceCall(expression, ir, names, options, textureSpecializations);
    if (cooperativeReduce) return cooperativeReduce;
    const cooperativeVectorReduce = emitSemanticTypedCooperativeVectorReduceCall(expression, ir, names, options, textureSpecializations);
    if (cooperativeVectorReduce) return cooperativeVectorReduce;
    const cooperativeScan = emitSemanticTypedCooperativeScanCall(expression, ir, names, options, textureSpecializations);
    if (cooperativeScan) return cooperativeScan;
    const syncthreadsPredicate = emitSemanticTypedSyncthreadsPredicateCall(expression, ir, names, options);
    if (syncthreadsPredicate) return syncthreadsPredicate;
    const subgroup = emitSemanticTypedSubgroupCall(expression, ir, names, options, textureSpecializations);
    if (subgroup) return subgroup;
    const generatedRandom = emitSemanticTypedGeneratedRandomCall(expression, ir, names);
    if (generatedRandom) return generatedRandom;
    const curand = emitSemanticTypedLocalCurandCall(expression, ir, names, options, textureSpecializations);
    if (curand) return curand;
    const addressPredicate = emitSemanticTypedAddressPredicateCall(expression);
    if (addressPredicate) return addressPredicate;
    const sharedAddress = emitSemanticTypedSharedAddressCall(expression, ir, names, options);
    if (sharedAddress) return sharedAddress;
    const ptxInteger = emitSemanticTypedPtxIntegerCall(expression, ir, names, options, textureSpecializations);
    if (ptxInteger) return ptxInteger;
    const ptxCompare = emitSemanticTypedPtxCompareCall(expression, ir, names, options, textureSpecializations);
    if (ptxCompare) return ptxCompare;
    const integerAtomic = emitSemanticTypedIntegerAtomicCall(expression, ir, names, options, textureSpecializations);
    if (integerAtomic) return integerAtomic;
    const conversion = emitSemanticTypedConversionIntrinsic(expression, ir, names, options, textureSpecializations);
    if (conversion) return conversion;
    const bf162Conversion = emitSemanticTypedBf162Conversion(expression, ir, names, options, textureSpecializations);
    if (bf162Conversion) return bf162Conversion;
    const bf16 = emitSemanticTypedBf16Call(expression, ir, names, options, textureSpecializations);
    if (bf16) return bf16;
    const half = emitSemanticTypedHalfCall(expression, ir, names, options, textureSpecializations);
    if (half) return half;
    const customMath = emitSemanticTypedCustomMathCall(expression, ir, names, options, textureSpecializations);
    if (customMath) return customMath;
    const minMax = emitSemanticTypedMinMaxCall(expression, ir, names, options, textureSpecializations);
    if (minMax) return minMax;
    const complex = emitSemanticTypedComplexCall(expression, ir, names, options, textureSpecializations);
    if (complex) return complex;
    const vectorMath = emitSemanticTypedVectorMathCall(expression, ir, names, options, textureSpecializations);
    if (vectorMath) return vectorMath;
    const nativeMath = semanticTypedNativeMathCallee(expression, ir);
    if (nativeMath) {
      return createTypedWgslCall(
        nativeMath,
        expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations)),
        "f32",
        expression.span,
      );
    }
    if (semanticWgslVectorConstructorSupported(expression, "any", ir)) {
      return emitSemanticVectorConstructorExpression(expression, ir, names, options, textureSpecializations);
    }
    if (semanticWgslVectorAtCallSupported(expression, ir)) {
      return emitSemanticVectorAtCallExpression(expression, ir, names, options, textureSpecializations);
    }
    const storagePointerFn = semanticTypedStoragePointerFunctionForCall(expression, ir);
    if (storagePointerFn) return emitSemanticTypedStoragePointerFunctionCall(expression, storagePointerFn, ir, names, options, textureSpecializations);
    const sharedPointerFn = semanticTypedSharedPointerFunctionForCall(expression, ir);
    if (sharedPointerFn) return emitSemanticTypedSharedPointerFunctionCall(expression, sharedPointerFn, ir, names, options, textureSpecializations);
    const localPointerFn = semanticTypedLocalPointerFunctionForCall(expression, ir);
    if (localPointerFn) return emitSemanticTypedLocalPointerFunctionCall(expression, localPointerFn, ir, names, options, textureSpecializations);
    const fn = semanticTypedValueFunctionForCall(expression, ir);
    if (fn) return emitSemanticTypedValueFunctionCall(expression, fn, ir, names, options, textureSpecializations);
  }
  if (expression.kind === "binary") {
    return emitSemanticBinary(expression, ir, names, options, textureSpecializations);
  }
  if (expression.kind === "unary") {
    if (semanticWgslBf162LocalBitsCastSupported(expression, ir)) {
      return emitSemanticBf162LocalBitsCast(expression, ir, names, options, textureSpecializations);
    }
    return emitSemanticUnary(expression, ir, names, options, textureSpecializations);
  }
  if (expression.kind === "conditional") {
    return emitSemanticConditional(expression, ir, names, options, textureSpecializations);
  }
  if (expression.kind === "index") {
    const ref = memoryRefFromIndexExpression(expression);
    throw semanticWgslError(`typed WGSL emission missing for 'index:${ref?.addressSpace ?? "none"}:${ref?.valueType ?? "none"}:rank${ref?.indices.length ?? 0}:fields${ref?.fields.length ?? 0}' expression`, expression.span);
  }
  if (expression.kind === "call") {
    const callee = expression.callee.kind === "symbol"
      ? expression.callee.name
      : expression.callee.kind === "member"
        ? `member.${expression.callee.property}`
        : expression.callee.kind;
    throw semanticWgslError(`typed WGSL emission missing for 'call:${callee}' expression`, expression.span);
  }
  if (expression.kind === "symbol") {
    throw semanticWgslError(`typed WGSL emission missing for 'symbol:${expression.addressSpace}:${expression.name}' expression`, expression.span);
  }
  throw semanticWgslError(`typed WGSL emission missing for '${expression.kind}' expression`, expression.span);
}

function emitSemanticTypedPackedSharedByteRead(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || ref.addressSpace !== "shared" || ref.indices.length !== 1 || ref.fields.length !== 0) return undefined;
  const sharedPointer = options.activeFunction === undefined ? undefined : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
    param.name === ref.base && param.pointer && param.addressSpace === "shared" && param.valueType === "uchar"
  );
  const packedRoot = semanticWgslPackedSharedByteRoot(ref, ir);
  if (!sharedPointer && !packedRoot) return undefined;
  const elementBytes = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
  const sourceIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  const offset = elementBytes === 1 || ref.pointerBaseUnitBytes !== undefined
    ? sourceIndex
    : emitTypedWgslBinary("*", sourceIndex, createTypedWgslLiteral(`${elementBytes}u`, "u32", ref.span), ref.span);
  const byteIndex = sharedPointer
    ? emitTypedWgslBinary("+", createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span), offset, ref.span)
    : offset;
  const wordIndex = emitTypedWgslBinary(">>", byteIndex, createTypedWgslLiteral("2u", "u32", ref.span), ref.span);
  const word = sharedPointer
    ? createTypedWgslDereferencedIndexedPlace(nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names), wordIndex, "u32", true, "workgroup", ref.span)
    : createTypedWgslIndexedPlace(nameFor(ref.base, names), wordIndex, "u32", true, ref.span, "workgroup");
  const loaded = createTypedWgslPlaceRead(word);
  if (ref.valueType === "uint") return loaded;
  if (ref.valueType === "int" || ref.valueType === "float") return createTypedWgslBitcast(ref.valueType === "int" ? "i32" : "f32", loaded, ref.span);
  const loadByte = (address: TypedWgslExpression): TypedWgslExpression => {
    const sourceWordIndex = emitTypedWgslBinary(">>", address, createTypedWgslLiteral("2u", "u32", ref.span), ref.span);
    const sourceWord = sharedPointer
      ? createTypedWgslDereferencedIndexedPlace(nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names), sourceWordIndex, "u32", true, "workgroup", ref.span)
      : createTypedWgslIndexedPlace(nameFor(ref.base, names), sourceWordIndex, "u32", true, ref.span, "workgroup");
    const shift = emitTypedWgslBinary(
      "*",
      emitTypedWgslBinary("&", address, createTypedWgslLiteral("3u", "u32", ref.span), ref.span),
      createTypedWgslLiteral("8u", "u32", ref.span),
      ref.span,
    );
    return emitTypedWgslBinary(
      "&",
      emitTypedWgslBinary(">>", createTypedWgslPlaceRead(sourceWord), shift, ref.span),
      createTypedWgslLiteral("255u", "u32", ref.span),
      ref.span,
    );
  };
  if (ref.valueType === "uchar") return loadByte(byteIndex);
  if (ref.valueType !== "half" && ref.valueType !== "bf16" && ref.valueType !== "half2" && ref.valueType !== "bf162") return undefined;
  const byteCount = ref.valueType === "half" || ref.valueType === "bf16" ? 2 : 4;
  const bits = Array.from({ length: byteCount }, (_, offset) => {
    const address = offset === 0
      ? byteIndex
      : emitTypedWgslBinary("+", byteIndex, createTypedWgslLiteral(`${offset}u`, "u32", ref.span), ref.span);
    const byte = loadByte(address);
    return offset === 0
      ? byte
      : emitTypedWgslBinary("<<", byte, createTypedWgslLiteral(`${offset * 8}u`, "u32", ref.span), ref.span);
  }).reduce((left, right) => emitTypedWgslBinary("|", left, right, ref.span));
  if (ref.valueType === "half" || ref.valueType === "half2") {
    const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", ref.span);
    if (ref.valueType === "half") {
      const scalar = createTypedWgslMemberAccess(unpacked, "x", "f32", ref.span);
      return convertTypedWgslExpression(scalar, "f16", true);
    }
    return createTypedWgslConstructor("vec2<f16>", [
      convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", ref.span), "f16", true),
      convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "y", "f32", ref.span), "f16", true),
    ], ref.span);
  }
  const low = createTypedWgslBitcast(
    "f32",
    emitTypedWgslBinary("<<", emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffffu", "u32", ref.span), ref.span), createTypedWgslLiteral("16u", "u32", ref.span), ref.span),
    ref.span,
  );
  if (ref.valueType === "bf16") return low;
  const high = createTypedWgslBitcast("f32", emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffff0000u", "u32", ref.span), ref.span), ref.span);
  return createTypedWgslConstructor("vec2<f32>", [low, high], ref.span);
}

function emitSemanticTypedDirectByteRawRead(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || !semanticWgslDirectByteRawView(ref, ir) || ref.indices.length !== 1 || ref.fields.length !== 0) return undefined;
  return emitSemanticTypedDirectByteRawRefRead(ref, ir, names, options);
}

function emitSemanticTypedDirectByteRawRefRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression {
  const byteIndex = emitTypedFlatStorageIndex({ ...ref, valueType: "uchar" }, ir, names, options);
  const atomic = semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base);
  const loadByte = (address: TypedWgslExpression): TypedWgslExpression => {
    if (!atomic) return createTypedWgslMemoryRead(nameFor(ref.base, names), address, "u32", false, ref.span);
    const wordIndex = emitTypedWgslBinary(">>", address, createTypedWgslLiteral("2u", "u32", ref.span), ref.span);
    const word = createTypedWgslMemoryRead(nameFor(ref.base, names), wordIndex, "u32", true, ref.span);
    const shift = emitTypedWgslBinary(
      "*",
      emitTypedWgslBinary("&", address, createTypedWgslLiteral("3u", "u32", ref.span), ref.span),
      createTypedWgslLiteral("8u", "u32", ref.span),
      ref.span,
    );
    return emitTypedWgslBinary("&", emitTypedWgslBinary(">>", word, shift, ref.span), createTypedWgslLiteral("255u", "u32", ref.span), ref.span);
  };
  const bits = Array.from({ length: 4 }, (_, offset) => {
    const address = offset === 0
      ? byteIndex
      : emitTypedWgslBinary("+", byteIndex, createTypedWgslLiteral(`${offset}u`, "u32", ref.span), ref.span);
    const byte = loadByte(address);
    return offset === 0 ? byte : emitTypedWgslBinary("<<", byte, createTypedWgslLiteral(`${offset * 8}u`, "u32", ref.span), ref.span);
  }).reduce((left, right) => emitTypedWgslBinary("|", left, right, ref.span));
  if (ref.valueType === "uint") return bits;
  if (ref.valueType === "int") return createTypedWgslBitcast("i32", bits, ref.span);
  if (ref.valueType === "float") return createTypedWgslBitcast("f32", bits, ref.span);
  throw semanticWgslError(`direct byte storage cannot produce '${ref.valueType ?? "void"}'`, ref.span);
}

function emitSemanticTypedLocalSpecialRead(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || ref.addressSpace !== "local" || ref.fields.length !== 0 || ref.indices.length !== 1) return undefined;
  const flat = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  const root = localArraySymbol(ir, ref.base);
  const readArrayWord = (index: TypedWgslExpression): TypedWgslExpression | undefined => {
    if (!root) return undefined;
    const path = semanticTypedLocalArrayPath(index, root.dimensions, ref.span);
    return createTypedWgslMemoryPathRead(nameFor(ref.base, names), path, wgslValueType(root.valueType), ref.span);
  };
  if (ref.valueType === "half2" && root?.valueType === "half") {
    const low = readArrayWord(flat);
    const high = readArrayWord(emitTypedWgslBinary("+", flat, createTypedWgslLiteral("1u", "u32", ref.span), ref.span));
    if (!low || !high || low.type !== "f16" || high.type !== "f16") return undefined;
    return createTypedWgslConstructor("vec2<f16>", [low, high], ref.span);
  }
  if (semanticWgslLocalPackedHalf2View(ref, ir)) {
    const word = readArrayWord(flat);
    if (!word || word.type !== "u32") return undefined;
    const unpacked = createTypedWgslCall("unpack2x16float", [word], "vec2<f32>", ref.span);
    return createTypedWgslConstructor("vec2<f16>", [
      convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", ref.span), "f16", true),
      convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "y", "f32", ref.span), "f16", true),
    ], ref.span);
  }
  if (semanticWgslLocalPackedHalfView(ref, ir)) {
    const wordIndex = emitTypedWgslBinary("/", flat, createTypedWgslLiteral("2u", "u32", ref.span), ref.span);
    const word = readArrayWord(wordIndex);
    if (!word || word.type !== "u32") return undefined;
    const unpacked = createTypedWgslCall("unpack2x16float", [word], "vec2<f32>", ref.span);
    const lane = createTypedWgslIndexAccess(unpacked, emitTypedWgslBinary("%", flat, createTypedWgslLiteral("2u", "u32", ref.span), ref.span), "f32", ref.span);
    return convertTypedWgslExpression(lane, "f16", true);
  }
  const scalarBitRoot = semanticWgslLocalScalarBitViewRootType(ref, ir);
  if (scalarBitRoot !== undefined) {
    const source = readArrayWord(flat);
    if (!source) return undefined;
    return createTypedWgslBitcast(wgslValueType(ref.valueType), source, ref.span);
  }
  const vectorBitRoot = semanticWgslLocalVectorBitViewRootType(ref, ir);
  if (vectorBitRoot !== undefined) {
    const vectorType = ref.containerValueType ?? semanticDeclaredLocalVectorType(ir, ref.base);
    if (!vectorType) return undefined;
    const vector = createTypedWgslIdentifier(nameFor(ref.base, names), wgslValueType(vectorType), ref.span);
    const lane = createTypedWgslIndexAccess(vector, flat, wgslValueType(vectorBitRoot), ref.span);
    return createTypedWgslBitcast(wgslValueType(ref.valueType), lane, ref.span);
  }
  return undefined;
}

function emitSemanticTypedSharedBitRead(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || ref.indices.length !== 1 || ref.fields.length !== 0) return undefined;
  const rootType = semanticWgslSharedScalarBitViewRootType(ref, ir);
  if (!rootType) return undefined;
  const root = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
  if (!root) return undefined;
  const atomic = semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base);
  const sourceType = atomic ? wgslAtomicScalar(rootType) : wgslValueType(rootType);
  const index = root.dimensions.length > 1
    ? emitTypedFlatRankedIndex(root.dimensions, ref.indices, ir, names, options, ref.span)
    : emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  let read = createTypedWgslMemoryRead(nameFor(ref.base, names), index, sourceType, atomic, ref.span);
  const semanticRootType = wgslValueType(rootType);
  if (read.type !== semanticRootType) read = createTypedWgslBitcast(semanticRootType, read, ref.span);
  return createTypedWgslBitcast(wgslValueType(ref.valueType), read, ref.span);
}

function emitSemanticTypedSurfaceRead(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression {
  if (!semanticWgslSurfaceReadSupported(expression, ir) || expression.surface.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL supports only modeled surf2Dread", expression.span);
  }
  const surfaceName = expression.surface.name;
  const xBytes = emitSemanticExpressionAs(expression.xBytes, ir, names, "i32", options);
  const y = emitSemanticExpressionAs(expression.y, ir, names, "i32", options);
  const z = expression.z ? emitSemanticExpressionAs(expression.z, ir, names, "i32", options) : createTypedWgslZero("i32", expression.span);
  const directSurface = surfaceSymbols(ir).some((surface) => surface.name === surfaceName);
  const readAt = (x: TypedWgslExpression): TypedWgslExpression => directSurface
    ? createTypedWgslCall(surfaceReadHelperName(surfaceName, names), [x, y, z], "f32", expression.span)
    : createTypedWgslCall(
        GENERIC_SURFACE_READ_HELPER_NAME,
        [emitSemanticExpressionAs(expression.surface, ir, names, "u32", options), x, y, z],
        "f32",
        expression.span,
      );
  if (isSemanticFloatVectorType(expression.valueType)) {
    const targetType = wgslValueType(expression.valueType);
    if (!isWgslVectorType(targetType)) throw semanticWgslError(`invalid surface vector type '${targetType}'`, expression.span);
    const scalar = wgslVectorScalar(expression.valueType);
    const lanes = Array.from({ length: cudaVectorLaneCount(expression.valueType) }, (_, lane): TypedWgslExpression => {
      const x = lane === 0 ? xBytes : emitTypedWgslBinary("+", xBytes, createTypedWgslLiteral(String(lane * 4), "i32", expression.span), expression.span);
      const value = readAt(x);
      return expression.valueType === "bf162" ? roundTypedBf16(value, expression.span) : convertTypedWgslExpression(value, scalar, true);
    });
    const vector = createTypedWgslConstructor(targetType, lanes, expression.span);
    const aligned = emitTypedWgslBinary(
      "&&",
      emitTypedWgslBinary(">=", xBytes, createTypedWgslZero("i32", expression.span), expression.span),
      emitTypedWgslBinary("==", emitTypedWgslBinary("%", xBytes, createTypedWgslLiteral("4", "i32", expression.span), expression.span), createTypedWgslZero("i32", expression.span), expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(createTypedWgslZero(targetType, expression.span), vector, aligned, expression.span);
  }
  const value = readAt(xBytes);
  if (expression.valueType === "bf16") return roundTypedBf16(value, expression.span);
  const target = wgslValueScalar(expression.valueType);
  return target === "f32" ? value : convertTypedWgslExpression(value, target, true);
}

function emitSemanticTypedTextureRead(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression {
  if (!semanticWgslTextureReadSupported(expression, ir)) throw semanticWgslError("semantic WGSL does not support texture read", expression.span);
  const x = emitSemanticExpressionAs(expression.x, ir, names, "f32", options);
  const y = emitSemanticExpressionAs(expression.y, ir, names, "f32", options);
  const directTexture = expression.texture.kind === "symbol" && expression.texture.addressSpace === "texture" ? expression.texture : undefined;
  let read: TypedWgslExpression;
  if (!directTexture) {
    read = createTypedWgslCall(
      SEMANTIC_BINDLESS_TEXTURE_READ_HELPER,
      [emitSemanticExpressionAs(expression.texture, ir, names, "u32", options), x, y],
      "vec4<f32>",
      expression.span,
    );
  } else {
    const texture = nameFor(directTexture.name, names);
    const z = expression.z === undefined ? undefined : emitSemanticExpressionAs(expression.z, ir, names, "f32", options);
    if (expression.callee === "texCubemap") {
      if (!z) throw semanticWgslError("cubemap texture read requires z coordinate", expression.span);
      read = createTypedWgslCubemapTextureLoad(texture, x, y, z, expression.span);
    } else {
      const atlasY = z ? emitTypedWgslBinary("+", y, z, expression.span) : y;
      const descriptor = options.textureDescriptors?.[directTexture.name];
      read = descriptor
        ? createTypedWgslTextureDescriptorRead(semanticTextureDescriptorHelperName(directTexture.name, names, descriptor), texture, x, atlasY, expression.span)
        : createTypedWgslTextureLoad(texture, x, atlasY, expression.span);
    }
  }
  const valueType = expression.valueType;
  if (isSemanticFloatVectorType(valueType)) {
    const targetType = wgslValueType(valueType);
    if (!isWgslVectorType(targetType)) throw semanticWgslError(`invalid texture vector type '${targetType}'`, expression.span);
    const laneCount = cudaVectorLaneCount(valueType);
    const targetScalar = wgslVectorScalar(valueType);
    const fields = ["x", "y", "z", "w"];
    const lanes = Array.from({ length: laneCount }, (_, lane): TypedWgslExpression => {
      const value = createTypedWgslMemberAccess(read, fields[lane]!, "f32", expression.span);
      return valueType === "bf162" ? roundTypedBf16(value, expression.span) : convertTypedWgslExpression(value, targetScalar, true);
    });
    return createTypedWgslConstructor(targetType, lanes, expression.span);
  }
  const red = createTypedWgslMemberAccess(read, "r", "f32", expression.span);
  if (valueType === "bf16") return roundTypedBf16(red, expression.span);
  const target = wgslValueScalar(valueType);
  return target === "f32" ? red : convertTypedWgslExpression(red, target, true);
}

function emitSemanticTypedLocalCurandCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (!semanticWgslCurandCallSupported(expression, ir) || expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  if (name === "curand_init" || name === "skipahead") return undefined;
  const stateIndex = semanticCurandStateArgumentIndex(name);
  const state = stateIndex === undefined ? undefined : expression.args[stateIndex];
  if (!state || state.kind !== "unary" || state.operator !== "&") return undefined;
  const addressSpace = semanticCurandStateAddressSpace(state);
  if (!addressSpace) return undefined;
  let pointer: TypedWgslExpression;
  if (state.argument.kind === "symbol" && state.argument.addressSpace === "local") {
    pointer = createTypedWgslAddressOf(createTypedWgslLocalPlace(nameFor(state.argument.name, names), "u32", state.span));
  } else if (state.argument.kind === "index") {
    const ref = memoryRefFromIndexExpression(state.argument);
    if (!ref || ref.fields.length !== 0 || ref.indices.length !== 1) return undefined;
    const index = ref.addressSpace === "storage" || ref.addressSpace === "device-global"
      ? emitTypedFlatStorageIndex({ ...ref, valueType: "uint" }, ir, names, options)
      : emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
    pointer = createTypedWgslAddressOf(createTypedWgslIndexedPlace(
      nameFor(ref.base, names),
      index,
      "u32",
      false,
      ref.span,
      addressSpace,
    ));
  } else {
    return undefined;
  }
  const helperBase = name === "curand" ? "bg_curand"
    : name === "curand_uniform" || name === "curand_uniform_double" ? "bg_curand_uniform"
    : name === "curand_uniform4" ? "bg_curand_uniform4"
    : name === "curand_normal" || name === "curand_normal_double" ? "bg_curand_normal"
    : name === "curand_normal2" ? "bg_curand_normal2"
    : name === "curand_normal4" ? "bg_curand_normal4"
    : name === "curand_log_normal" || name === "curand_log_normal_double" ? "bg_curand_log_normal"
    : name === "curand_log_normal2" ? "bg_curand_log_normal2"
    : name === "curand_log_normal4" ? "bg_curand_log_normal4"
    : name === "curand_poisson" ? "bg_curand_poisson"
    : name === "curand_poisson4" ? "bg_curand_poisson4"
    : undefined;
  if (!helperBase) return undefined;
  const helper = addressSpace === "function" ? helperBase : `${helperBase}_${addressSpace}`;
  const args: TypedWgslExpression[] = [pointer];
  for (let index = 0; index < expression.args.length; index += 1) {
    if (index === stateIndex) continue;
    args.push(emitSemanticExpressionAs(expression.args[index]!, ir, names, "f32", options, textureSpecializations));
  }
  return createTypedWgslCall(helper, args, semanticExpressionWgslType(expression, ir), expression.span);
}

function emitSemanticTypedGeneratedRandomCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): TypedWgslExpression | undefined {
  if (!semanticWgslGeneratedRandomCallSupported(expression) || expression.callee.kind !== "symbol") return undefined;
  const state = expression.args[0];
  if (!state || state.kind !== "unary" || state.operator !== "&" || state.argument.kind !== "symbol" || state.argument.addressSpace !== "local") return undefined;
  const pointer = createTypedWgslAddressOf(createTypedWgslLocalPlace(nameFor(state.argument.name, names), "u32", state.span));
  return createTypedWgslCall(expression.callee.name, [pointer], semanticExpressionWgslType(expression, ir), expression.span);
}

function emitSemanticTypedPtxCompareCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const info = semanticPtxIntegerCallInfo(expression.callee.name);
  if (info?.family !== "compare") return undefined;
  const [left, right] = expression.args;
  if (!left || !right) return undefined;
  const type = info.signed ? "i32" : "u32";
  const operator = ({ eq: "==", ne: "!=", lt: "<", le: "<=", gt: ">", ge: ">=" } as const)[info.op];
  const condition = emitTypedWgslBinary(
    operator,
    emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations),
    emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations),
    expression.span,
  );
  const resultType = semanticExpressionWgslType(expression, ir) === "i32" ? "i32" : "u32";
  return emitTypedWgslSelect(
    createTypedWgslZero(resultType, expression.span),
    createTypedWgslLiteral(resultType === "i32" ? "1" : "1u", resultType, expression.span),
    condition,
    expression.span,
  );
}

function emitSemanticTypedPtxIntegerCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const info = semanticPtxIntegerCallInfo(expression.callee.name);
  if (!info || info.family === "compare") return undefined;
  const args = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations));
  const u32 = (value: number): TypedWgslExpression => createTypedWgslLiteral(`${value >>> 0}u`, "u32", expression.span);
  const arg = (index: number): TypedWgslExpression => args[index] ?? createTypedWgslZero("u32", expression.span);
  let result: TypedWgslExpression;
  if (info.family === "arithmetic") {
    const product = emitTypedWgslBinary("*", arg(0), arg(1), expression.span);
    result = info.op === "add" ? emitTypedWgslBinary("+", arg(0), arg(1), expression.span)
      : info.op === "sub" ? emitTypedWgslBinary("-", arg(0), arg(1), expression.span)
      : info.op === "mad-lo" ? emitTypedWgslBinary("+", product, arg(2), expression.span)
      : product;
  } else if (info.family === "shift") {
    const amount = arg(1);
    const clamped = createTypedWgslCall("min", [amount, u32(31)], "u32", expression.span);
    const shifted = info.op === "shl"
      ? emitTypedWgslBinary("<<", arg(0), clamped, expression.span)
      : info.signed
        ? createTypedWgslBitcast("u32", emitTypedWgslBinary(">>", createTypedWgslBitcast("i32", arg(0), expression.span), clamped, expression.span), expression.span)
        : emitTypedWgslBinary(">>", arg(0), clamped, expression.span);
    result = info.op === "shr" && info.signed
      ? shifted
      : emitTypedWgslSelect(shifted, createTypedWgslZero("u32", expression.span), emitTypedWgslBinary(">=", amount, u32(32), expression.span), expression.span);
  } else if (info.family === "minmax") {
    if (info.signed) {
      result = createTypedWgslBitcast(
        "u32",
        createTypedWgslCall(info.op, [createTypedWgslBitcast("i32", arg(0), expression.span), createTypedWgslBitcast("i32", arg(1), expression.span)], "i32", expression.span),
        expression.span,
      );
    } else {
      result = createTypedWgslCall(info.op, [arg(0), arg(1)], "u32", expression.span);
    }
  } else if (info.family === "unary") {
    if (info.op === "neg") {
      result = emitTypedWgslBinary("-", createTypedWgslZero("u32", expression.span), arg(0), expression.span);
    } else {
      const negative = emitTypedWgslBinary("!=", emitTypedWgslBinary("&", arg(0), createTypedWgslLiteral("0x80000000u", "u32", expression.span), expression.span), createTypedWgslZero("u32", expression.span), expression.span);
      const mask = emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffffffffu", "u32", expression.span), negative, expression.span);
      result = emitTypedWgslBinary("-", emitTypedWgslBinary("^", arg(0), mask, expression.span), mask, expression.span);
    }
  } else if (info.family === "select") {
    result = emitTypedWgslSelect(arg(1), arg(0), emitTypedWgslBinary("!=", arg(2), createTypedWgslZero("u32", expression.span), expression.span), expression.span);
  } else if (info.family === "prmt") {
    const lanes = Array.from({ length: 4 }, (_, lane): TypedWgslExpression => {
      const control = emitTypedWgslBinary("&", emitTypedWgslBinary(">>", arg(2), u32(lane * 4), expression.span), createTypedWgslLiteral("0xfu", "u32", expression.span), expression.span);
      const source = emitTypedWgslBinary("&", control, u32(7), expression.span);
      const shift = emitTypedWgslBinary("*", emitTypedWgslBinary("&", source, u32(3), expression.span), u32(8), expression.span);
      const byte = emitTypedWgslBinary("&", emitTypedWgslSelect(emitTypedWgslBinary(">>", arg(0), shift, expression.span), emitTypedWgslBinary(">>", arg(1), shift, expression.span), emitTypedWgslBinary(">=", source, u32(4), expression.span), expression.span), createTypedWgslLiteral("0xffu", "u32", expression.span), expression.span);
      const sign = emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffu", "u32", expression.span), emitTypedWgslBinary("!=", emitTypedWgslBinary("&", byte, createTypedWgslLiteral("0x80u", "u32", expression.span), expression.span), createTypedWgslZero("u32", expression.span), expression.span), expression.span);
      const extended = emitTypedWgslSelect(byte, sign, emitTypedWgslBinary("!=", emitTypedWgslBinary("&", control, u32(8), expression.span), createTypedWgslZero("u32", expression.span), expression.span), expression.span);
      return lane === 0 ? extended : emitTypedWgslBinary("<<", extended, u32(lane * 8), expression.span);
    });
    result = lanes.reduce((left, right) => emitTypedWgslBinary("|", left, right, expression.span));
  } else {
    const rows = Array.from({ length: 8 }, (_, row): TypedWgslExpression => {
      const mask = [4, 2, 1].map((bit, index) => (row & bit) === 0 ? emitTypedWgslUnary("~", arg(index), expression.span) : arg(index))
        .reduce((left, right) => emitTypedWgslBinary("&", left, right, expression.span));
      const enabled = emitTypedWgslBinary("!=", emitTypedWgslBinary("&", arg(3), u32(1 << row), expression.span), createTypedWgslZero("u32", expression.span), expression.span);
      return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), mask, enabled, expression.span);
    });
    result = rows.reduce((left, right) => emitTypedWgslBinary("|", left, right, expression.span));
  }
  return expression.valueType === "int" ? createTypedWgslBitcast("i32", result, expression.span) : result;
}

function roundTypedBf16(value: TypedWgslExpression, span: SourceSpan): TypedWgslExpression {
  const bits = createTypedWgslCall("bg_f32_to_bf16_bits_mode", [value, createTypedWgslZero("u32", span)], "u32", span);
  return createTypedWgslBitcast(
    "f32",
    emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", span), span),
    span,
  );
}

function emitSemanticTypedBf16Call(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  const firstArg = expression.args[0];
  if (firstArg === undefined) return undefined;
  const isPair = semanticExpressionVectorValueType(firstArg, ir.functions) === "bf162";
  const scalar = (arg: SemanticExpression): TypedWgslExpression => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations);
  const vector = (arg: SemanticExpression): TypedWgslExpression => emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  const roundPair = (value: TypedWgslExpression): TypedWgslExpression => createTypedWgslConstructor(
    "vec2<f32>",
    [
      roundTypedBf16(createTypedWgslMemberAccess(value, "x", "f32", expression.span), expression.span),
      roundTypedBf16(createTypedWgslMemberAccess(value, "y", "f32", expression.span), expression.span),
    ],
    expression.span,
  );
  if (isPair && ["__hadd2", "__hadd2_rn", "__hsub2", "__hsub2_rn", "__hmul2", "__hmul2_rn", "__h2div"].includes(name)) {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : name === "__h2div" ? "/" : "*";
    return roundPair(emitTypedWgslBinary(operator, vector(left), vector(right), expression.span));
  }
  if (isPair && isSemanticHalf2ComparisonCall(name)) {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const lhs = vector(left);
    const rhs = vector(right);
    const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
    const operator = normalized === "__heq2" || normalized === "__hequ2" ? "=="
      : normalized === "__hne2" || normalized === "__hneu2" ? "!="
      : normalized === "__hgt2" || normalized === "__hgtu2" ? ">"
      : normalized === "__hge2" || normalized === "__hgeu2" ? ">="
      : normalized === "__hlt2" || normalized === "__hltu2" ? "<" : "<=";
    const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
    const unordered = emitTypedWgslBinary("|", emitTypedWgslBinary("!=", lhs, lhs, expression.span), emitTypedWgslBinary("!=", rhs, rhs, expression.span), expression.span);
    const predicate = normalized.includes("u2")
      ? emitTypedWgslBinary("|", unordered, base, expression.span)
      : emitTypedWgslBinary("&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
    if (isSemanticHalf2BooleanComparisonCall(name)) return createTypedWgslCall("all", [predicate], "bool", expression.span);
    if (isSemanticHalf2MaskComparisonCall(name)) {
      const x = createTypedWgslMemberAccess(predicate, "x", "bool", expression.span);
      const y = createTypedWgslMemberAccess(predicate, "y", "bool", expression.span);
      return emitTypedWgslBinary(
        "|",
        emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffffu", "u32", expression.span), x, expression.span),
        emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffff0000u", "u32", expression.span), y, expression.span),
        expression.span,
      );
    }
    return emitTypedWgslSelect(
      createTypedWgslZero("vec2<f32>", expression.span),
      createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span),
      predicate,
      expression.span,
    );
  }
  if (isPair && ["__hceil2", "__hfloor2", "__htrunc2", "__hsqrt2", "__hrsqrt2", "__hrcp2", "h2ceil", "h2floor", "h2trunc", "h2sqrt", "h2rsqrt", "h2rcp"].includes(name)) {
    const value = expression.args[0];
    if (!value) return undefined;
    const operand = vector(value);
    const result = name === "__hrcp2" || name === "h2rcp"
      ? emitTypedWgslBinary("/", createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span), operand, expression.span)
      : createTypedWgslCall(name === "__hceil2" || name === "h2ceil" ? "ceil" : name === "__hfloor2" || name === "h2floor" ? "floor" : name === "__htrunc2" || name === "h2trunc" ? "trunc" : name === "__hsqrt2" || name === "h2sqrt" ? "sqrt" : "inverseSqrt", [operand], "vec2<f32>", expression.span);
    return roundPair(result);
  }
  if (isPair && ["h2exp", "h2exp2", "h2exp10", "h2log", "h2log2", "h2log10", "h2sin", "h2cos", "h2tanh", "h2tanh_approx", "h2rint"].includes(name)) {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = vector(value);
    const emitLane = (field: "x" | "y"): TypedWgslExpression => {
      const lane = createTypedWgslMemberAccess(pair, field, "f32", expression.span);
      if (name === "h2exp10") return createTypedWgslCall("pow", [createTypedWgslLiteral("10.0", "f32", expression.span), lane], "f32", expression.span);
      if (name === "h2log10") return emitTypedWgslBinary("/", createTypedWgslCall("log", [lane], "f32", expression.span), createTypedWgslLiteral("2.302585092994046", "f32", expression.span), expression.span);
      const callee = name === "h2exp" ? "exp" : name === "h2exp2" ? "exp2" : name === "h2log" ? "log" : name === "h2log2" ? "log2" : name === "h2sin" ? "sin" : name === "h2cos" ? "cos" : name === "h2rint" ? "bg_semantic_round_even_f32" : "tanh";
      return createTypedWgslCall(callee, [lane], "f32", expression.span);
    };
    return createTypedWgslConstructor("vec2<f32>", [roundTypedBf16(emitLane("x"), expression.span), roundTypedBf16(emitLane("y"), expression.span)], expression.span);
  }
  if (isPair && name === "__hneg2") {
    const value = expression.args[0];
    return value ? roundPair(emitTypedWgslUnary("-", vector(value), expression.span)) : undefined;
  }
  if (isPair && name === "__habs2") {
    const value = expression.args[0];
    return value ? roundPair(createTypedWgslCall("abs", [vector(value)], "vec2<f32>", expression.span)) : undefined;
  }
  if (isPair && (name === "__hmin2" || name === "__hmax2")) {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    return roundPair(createTypedWgslCall(name === "__hmin2" ? "min" : "max", [vector(left), vector(right)], "vec2<f32>", expression.span));
  }
  if (isPair && (name === "__hmin2_nan" || name === "__hmax2_nan")) {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const lhs = vector(left);
    const rhs = vector(right);
    const result = createTypedWgslCall(name === "__hmin2_nan" ? "min" : "max", [lhs, rhs], "vec2<f32>", expression.span);
    const nan = emitTypedWgslBinary("|", emitTypedWgslBinary("!=", lhs, lhs, expression.span), emitTypedWgslBinary("!=", rhs, rhs, expression.span), expression.span);
    return roundPair(emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span));
  }
  if (isPair && (name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat" || name === "__hfma2_relu")) {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) return undefined;
    let result = createTypedWgslCall("fma", [vector(left), vector(right), vector(addend)], "vec2<f32>", expression.span);
    if (name.endsWith("_sat")) result = createTypedWgslCall("clamp", [result, createTypedWgslZero("vec2<f32>", expression.span), createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span)], "vec2<f32>", expression.span);
    if (name.endsWith("_relu")) result = createTypedWgslCall("max", [result, createTypedWgslZero("vec2<f32>", expression.span)], "vec2<f32>", expression.span);
    return roundPair(result);
  }
  if (isPair && name === "__hcmadd") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) return undefined;
    const lhs = vector(left);
    const rhs = vector(right);
    const acc = vector(addend);
    const lane = (value: TypedWgslExpression, field: "x" | "y"): TypedWgslExpression => createTypedWgslMemberAccess(value, field, "f32", expression.span);
    const real = emitTypedWgslBinary(
      "+",
      emitTypedWgslBinary("-", emitTypedWgslBinary("*", lane(lhs, "x"), lane(rhs, "x"), expression.span), emitTypedWgslBinary("*", lane(lhs, "y"), lane(rhs, "y"), expression.span), expression.span),
      lane(acc, "x"),
      expression.span,
    );
    const imaginary = emitTypedWgslBinary(
      "+",
      emitTypedWgslBinary("+", emitTypedWgslBinary("*", lane(lhs, "x"), lane(rhs, "y"), expression.span), emitTypedWgslBinary("*", lane(lhs, "y"), lane(rhs, "x"), expression.span), expression.span),
      lane(acc, "y"),
      expression.span,
    );
    return createTypedWgslConstructor("vec2<f32>", [roundTypedBf16(real, expression.span), roundTypedBf16(imaginary, expression.span)], expression.span);
  }
  if (isPair && name === "__hisnan2") {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = vector(value);
    return emitTypedWgslSelect(
      createTypedWgslZero("vec2<f32>", expression.span),
      createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span),
      emitTypedWgslBinary("!=", pair, pair, expression.span),
      expression.span,
    );
  }
  if (expression.valueType === "bf16" && (name === "__hdiv" || name === "__hdiv_rn")) {
    const [left, right] = expression.args;
    return left && right ? roundTypedBf16(emitTypedWgslBinary("/", scalar(left), scalar(right), expression.span), expression.span) : undefined;
  }
  if (expression.valueType === "bf16" && (name === "__hfma" || name === "__hfma_rn" || name === "__hfma_sat")) {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) return undefined;
    let result = createTypedWgslCall("fma", [scalar(left), scalar(right), scalar(addend)], "f32", expression.span);
    if (name.endsWith("_sat")) result = createTypedWgslCall("clamp", [result, createTypedWgslZero("f32", expression.span), createTypedWgslLiteral("1.0", "f32", expression.span)], "f32", expression.span);
    return roundTypedBf16(result, expression.span);
  }
  if (expression.valueType === "bf16" && name === "__hfma_relu") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) return undefined;
    const result = createTypedWgslCall("fma", [scalar(left), scalar(right), scalar(addend)], "f32", expression.span);
    return roundTypedBf16(createTypedWgslCall("max", [result, createTypedWgslZero("f32", expression.span)], "f32", expression.span), expression.span);
  }
  if (expression.valueType === "bf16" && name === "__hneg") {
    const value = expression.args[0];
    return value ? roundTypedBf16(emitTypedWgslUnary("-", scalar(value), expression.span), expression.span) : undefined;
  }
  return undefined;
}

function emitSemanticTypedHalfCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  const scalar = (arg: SemanticExpression): TypedWgslExpression => emitSemanticExpressionAs(arg, ir, names, "f16", options, textureSpecializations);
  const vector = (arg: SemanticExpression): TypedWgslExpression => emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  const scalarComparison = /^(?:__h)(eq|ne|gt|ge|lt|le)(u)?$/u.exec(name);
  if (scalarComparison) {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const lhs = scalar(left);
    const rhs = scalar(right);
    const operator = ({ eq: "==", ne: "!=", gt: ">", ge: ">=", lt: "<", le: "<=" } as const)[scalarComparison[1] as "eq" | "ne" | "gt" | "ge" | "lt" | "le"];
    const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
    const unordered = emitTypedWgslBinary("||", emitTypedWgslBinary("!=", lhs, lhs, expression.span), emitTypedWgslBinary("!=", rhs, rhs, expression.span), expression.span);
    const predicate = scalarComparison[2] ? emitTypedWgslBinary("||", unordered, base, expression.span) : emitTypedWgslBinary("&&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), predicate, expression.span);
  }
  if (isSemanticHalf2ComparisonCall(name) && semanticExpressionVectorValueType(expression.args[0]!, ir.functions) === "half2") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const lhs = vector(left);
    const rhs = vector(right);
    const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
    const operator = normalized === "__heq2" || normalized === "__hequ2" ? "=="
      : normalized === "__hne2" || normalized === "__hneu2" ? "!="
      : normalized === "__hgt2" || normalized === "__hgtu2" ? ">"
      : normalized === "__hge2" || normalized === "__hgeu2" ? ">="
      : normalized === "__hlt2" || normalized === "__hltu2" ? "<" : "<=";
    const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
    const unordered = emitTypedWgslBinary("|", emitTypedWgslBinary("!=", lhs, lhs, expression.span), emitTypedWgslBinary("!=", rhs, rhs, expression.span), expression.span);
    const predicate = normalized.includes("u2")
      ? emitTypedWgslBinary("|", unordered, base, expression.span)
      : emitTypedWgslBinary("&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
    if (isSemanticHalf2BooleanComparisonCall(name)) return createTypedWgslCall("all", [predicate], "bool", expression.span);
    if (isSemanticHalf2MaskComparisonCall(name)) {
      const x = createTypedWgslMemberAccess(predicate, "x", "bool", expression.span);
      const y = createTypedWgslMemberAccess(predicate, "y", "bool", expression.span);
      return emitTypedWgslBinary(
        "|",
        emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffffu", "u32", expression.span), x, expression.span),
        emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffff0000u", "u32", expression.span), y, expression.span),
        expression.span,
      );
    }
    return emitTypedWgslSelect(
      createTypedWgslZero("vec2<f16>", expression.span),
      createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span),
      predicate,
      expression.span,
    );
  }
  if (name === "__habs") {
    const value = expression.args[0];
    return value ? createTypedWgslCall("abs", [scalar(value)], "f16", expression.span) : undefined;
  }
  if (name === "__hneg" && semanticExpressionWgslType(expression, ir) === "f16") {
    const value = expression.args[0];
    return value ? emitTypedWgslUnary("-", scalar(value), expression.span) : undefined;
  }
  if (["__hceil", "__hfloor", "__htrunc", "__hsqrt", "__hrsqrt", "hrsqrt", "__hrcp", "hexp"].includes(name)) {
    const value = expression.args[0];
    if (!value) return undefined;
    const operand = scalar(value);
    if (name === "__hrcp") return emitTypedWgslBinary("/", createTypedWgslLiteral("f16(1.0)", "f16", expression.span), operand, expression.span);
    return createTypedWgslCall(
      name === "__hceil" ? "ceil" : name === "__hfloor" ? "floor" : name === "__htrunc" ? "trunc" : name === "__hsqrt" ? "sqrt" : name === "__hrsqrt" || name === "hrsqrt" ? "inverseSqrt" : "exp",
      [operand],
      "f16",
      expression.span,
    );
  }
  if (name === "__hisnan" || name === "__hisinf") {
    const value = expression.args[0];
    if (!value) return undefined;
    const operand = scalar(value);
    const condition = name === "__hisnan"
      ? emitTypedWgslBinary("!=", operand, operand, expression.span)
      : emitTypedWgslBinary(">", createTypedWgslCall("abs", [operand], "f16", expression.span), createTypedWgslLiteral("f16(65504.0)", "f16", expression.span), expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), condition, expression.span);
  }
  if (["__hadd", "__hadd_rn", "__hadd_sat", "__hsub", "__hsub_rn", "__hsub_sat", "__hmul", "__hmul_rn", "__hmul_sat"].includes(name)) {
    if (name === "__hadd" && expression.valueType !== "half") return undefined;
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : "*";
    const value = emitTypedWgslBinary(operator, scalar(left), scalar(right), expression.span);
    if (!name.endsWith("_sat")) return value;
    const clamped = createTypedWgslCall("clamp", [value, createTypedWgslZero("f16", expression.span), createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], "f16", expression.span);
    return emitTypedWgslSelect(clamped, createTypedWgslZero("f16", expression.span), emitTypedWgslBinary("!=", value, value, expression.span), expression.span);
  }
  if (name === "__hmin" || name === "__hmax" || name === "__hmin_nan" || name === "__hmax_nan") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const lhs = scalar(left);
    const rhs = scalar(right);
    const result = createTypedWgslCall(name.includes("min") ? "min" : "max", [lhs, rhs], "f16", expression.span);
    if (!name.endsWith("_nan")) return result;
    const nan = emitTypedWgslBinary("||", emitTypedWgslBinary("!=", lhs, lhs, expression.span), emitTypedWgslBinary("!=", rhs, rhs, expression.span), expression.span);
    return emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span);
  }
  if (["__hadd2", "__hadd2_rn", "__hadd2_sat", "__hsub2", "__hsub2_rn", "__hsub2_sat", "__hmul2", "__hmul2_rn", "__hmul2_sat"].includes(name) && semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : "*";
    const value = emitTypedWgslBinary(operator, vector(left), vector(right), expression.span);
    if (!name.endsWith("_sat")) return value;
    return createTypedWgslCall(
      "clamp",
      [value, createTypedWgslZero("vec2<f16>", expression.span), createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span)],
      "vec2<f16>",
      expression.span,
    );
  }
  if ((name === "__hdiv" || name === "__hdiv_rn") && semanticExpressionWgslType(expression, ir) === "f16") {
    const [left, right] = expression.args;
    return left && right ? emitTypedWgslBinary("/", scalar(left), scalar(right), expression.span) : undefined;
  }
  if ((name === "__hfma" || name === "__hfma_rn" || name === "__hfma_sat") && semanticExpressionWgslType(expression, ir) === "f16") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) return undefined;
    const result = createTypedWgslCall("fma", [scalar(left), scalar(right), scalar(addend)], "f16", expression.span);
    if (!name.endsWith("_sat")) return result;
    return createTypedWgslCall("clamp", [result, createTypedWgslZero("f16", expression.span), createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], "f16", expression.span);
  }
  if (name === "__habs2" && semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
    const value = expression.args[0];
    return value ? createTypedWgslCall("abs", [vector(value)], "vec2<f16>", expression.span) : undefined;
  }
  if (name === "__hneg2" && semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
    const value = expression.args[0];
    return value ? emitTypedWgslUnary("-", vector(value), expression.span) : undefined;
  }
  if (["__hceil2", "__hfloor2", "__htrunc2", "__hsqrt2", "__hrsqrt2", "__hrcp2"].includes(name) && semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
    const value = expression.args[0];
    if (!value) return undefined;
    const operand = vector(value);
    if (name === "__hrcp2") {
      return emitTypedWgslBinary("/", createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span), operand, expression.span);
    }
    return createTypedWgslCall(
      name === "__hceil2" ? "ceil" : name === "__hfloor2" ? "floor" : name === "__htrunc2" ? "trunc" : name === "__hsqrt2" ? "sqrt" : "inverseSqrt",
      [operand],
      "vec2<f16>",
      expression.span,
    );
  }
  if ((name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat") && semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) return undefined;
    const result = createTypedWgslCall("fma", [vector(left), vector(right), vector(addend)], "vec2<f16>", expression.span);
    return name.endsWith("_sat")
      ? createTypedWgslCall("clamp", [result, createTypedWgslZero("vec2<f16>", expression.span), createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span)], "vec2<f16>", expression.span)
      : result;
  }
  if ((name === "__hmin2" || name === "__hmax2" || name === "__hmin2_nan" || name === "__hmax2_nan") && semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const lhs = vector(left);
    const rhs = vector(right);
    const result = createTypedWgslCall(name.includes("min") ? "min" : "max", [lhs, rhs], "vec2<f16>", expression.span);
    if (!name.endsWith("_nan")) return result;
    const nan = emitTypedWgslBinary("|", emitTypedWgslBinary("!=", lhs, lhs, expression.span), emitTypedWgslBinary("!=", rhs, rhs, expression.span), expression.span);
    return emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span);
  }
  if (name === "__hisnan2" && semanticExpressionVectorValueType(expression.args[0]!, ir.functions) === "half2") {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = vector(value);
    return emitTypedWgslSelect(
      createTypedWgslZero("vec2<f16>", expression.span),
      createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span),
      emitTypedWgslBinary("!=", pair, pair, expression.span),
      expression.span,
    );
  }
  if (name === "__floats2half2_rn" || name === "__halves2half2") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    return createTypedWgslConstructor("vec2<f16>", [scalar(left), scalar(right)], expression.span);
  }
  if (name === "__float22half2_rn") {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = vector(value);
    return createTypedWgslConstructor("vec2<f16>", [
      convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "x", "f32", expression.span), "f16", true),
      convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "y", "f32", expression.span), "f16", true),
    ], expression.span);
  }
  if (name === "__float2half2_rn" || name === "__half2half2") {
    const value = expression.args[0];
    return value ? createTypedWgslConstructor("vec2<f16>", [scalar(value)], expression.span) : undefined;
  }
  if (name === "__half22float2") {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = vector(value);
    return createTypedWgslConstructor(
      "vec2<f32>",
      [
        convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "x", "f16", expression.span), "f32", true),
        convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "y", "f16", expression.span), "f32", true),
      ],
      expression.span,
    );
  }
  if (name === "__low2half" || name === "__high2half") {
    const value = expression.args[0];
    return value ? createTypedWgslMemberAccess(vector(value), name === "__low2half" ? "x" : "y", "f16", expression.span) : undefined;
  }
  if (name === "__low2half2" || name === "__high2half2") {
    const value = expression.args[0];
    if (!value) return undefined;
    const lane = createTypedWgslMemberAccess(vector(value), name === "__low2half2" ? "x" : "y", "f16", expression.span);
    return createTypedWgslConstructor("vec2<f16>", [lane], expression.span);
  }
  if (name === "__lows2half2" || name === "__highs2half2") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const field = name === "__lows2half2" ? "x" : "y";
    return createTypedWgslConstructor(
      "vec2<f16>",
      [
        createTypedWgslMemberAccess(vector(left), field, "f16", expression.span),
        createTypedWgslMemberAccess(vector(right), field, "f16", expression.span),
      ],
      expression.span,
    );
  }
  if (name === "__half2_as_uint") {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = vector(value);
    const f32Pair = createTypedWgslConstructor(
      "vec2<f32>",
      [
        convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "x", "f16", expression.span), "f32", true),
        convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "y", "f16", expression.span), "f32", true),
      ],
      expression.span,
    );
    return createTypedWgslCall("pack2x16float", [f32Pair], "u32", expression.span);
  }
  if (name === "__uint_as_half2") {
    const value = expression.args[0];
    if (!value) return undefined;
    const unpacked = createTypedWgslCall(
      "unpack2x16float",
      [emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)],
      "vec2<f32>",
      expression.span,
    );
    return createTypedWgslConstructor("vec2<f16>", [
      convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true),
      convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "y", "f32", expression.span), "f16", true),
    ], expression.span);
  }
  if (name === "__half_as_ushort" || name === "__half_as_short") {
    const value = expression.args[0];
    if (!value) return undefined;
    const pair = createTypedWgslConstructor(
      "vec2<f32>",
      [convertTypedWgslExpression(scalar(value), "f32", true), createTypedWgslZero("f32", expression.span)],
      expression.span,
    );
    const bits = emitTypedWgslBinary("&", createTypedWgslCall("pack2x16float", [pair], "u32", expression.span), createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span);
    if (name === "__half_as_ushort") return bits;
    const shifted = emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
    return emitTypedWgslBinary(">>", createTypedWgslBitcast("i32", shifted, expression.span), createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
  }
  if (name === "__ushort_as_half" || name === "__short_as_half") {
    const value = expression.args[0];
    if (!value) return undefined;
    const source = emitSemanticExpressionAs(value, ir, names, name === "__short_as_half" ? "i32" : "u32", options, textureSpecializations);
    const bits = name === "__short_as_half" ? convertTypedWgslExpression(source, "u32", true) : source;
    const masked = emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span);
    const unpacked = createTypedWgslCall("unpack2x16float", [masked], "vec2<f32>", expression.span);
    return convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true);
  }
  return undefined;
}

function semanticTypedStoragePointerFunctionForCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): SemanticKernelIrModule["functions"][number] | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace !== "function") return undefined;
  const callee = expression.callee;
  const fn = ir.functions.find((candidate) => semanticIdsEqual(callee.id, semanticSymbolIdFromFunction(candidate.id)));
  if (!fn || fn.returnType === "void" || fn.params.length !== expression.args.length) return undefined;
  if (!fn.params.some((param) => param.pointer && param.addressSpace === "storage")) return undefined;
  if (fn.params.some((param) =>
    param.cooperativeGroupKind !== undefined ||
    param.pointer && param.addressSpace !== "storage")) return undefined;
  return fn;
}

function semanticTypedSharedPointerFunctionForCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): SemanticKernelIrModule["functions"][number] | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace !== "function") return undefined;
  const callee = expression.callee;
  const fn = ir.functions.find((candidate) => semanticIdsEqual(callee.id, semanticSymbolIdFromFunction(candidate.id)));
  if (!fn || fn.returnType === "void" || fn.params.length !== expression.args.length) return undefined;
  if (!fn.params.some((param) => param.pointer && param.addressSpace === "shared")) return undefined;
  if (fn.params.some((param) =>
    param.addressSpace === "texture" || param.addressSpace === "surface" ||
    param.pointer && param.addressSpace !== "shared")) return undefined;
  return fn;
}

function emitSemanticTypedSharedPointerFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  fn: SemanticKernelIrModule["functions"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression {
  const args: TypedWgslExpression[] = [];
  expression.args.forEach((arg, index) => {
    const param = fn.params[index]!;
    if (param.cooperativeGroupKind !== undefined) {
      args.push(...emitSemanticTypedCooperativeGroupArguments(arg, ir, names, options));
      return;
    }
    if (param.addressSpace === "texture") {
      if (arg.kind !== "symbol" || arg.addressSpace !== "texture") throw semanticWgslError(`texture argument '${param.name}' is not a texture symbol`, arg.span);
      args.push(createTypedWgslIdentifier(nameFor(arg.name, names), "texture_2d<f32>", arg.span));
      return;
    }
    if (param.addressSpace === "surface") {
      if (arg.kind !== "symbol" || arg.addressSpace !== "surface") throw semanticWgslError(`surface argument '${param.name}' is not a surface symbol`, arg.span);
      const handle = surfaceHandleForName(arg.name, ir);
      if (handle === undefined) throw semanticWgslError(`unknown surface '${arg.name}'`, arg.span);
      args.push(createTypedWgslLiteral(`${handle}u`, "u32", arg.span));
      return;
    }
    if (param.pointer) {
      const ref = semanticPointerArgMemoryRef(arg);
      if (!ref || ref.addressSpace !== "shared") throw semanticWgslError(`shared pointer argument '${param.name}' is not modeled shared memory`, arg.span);
      const owner = options.activeFunction === undefined ? undefined : ir.functions.find((candidate) => candidate.name === options.activeFunction);
      const forwarded = owner?.params.find((candidate) => candidate.name === ref.base && candidate.pointer && candidate.addressSpace === "shared");
      const root = forwarded === undefined ? sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base) : undefined;
      if (!forwarded && !root) throw semanticWgslError(`unknown shared pointer base '${ref.base}'`, ref.span);
      const source = forwarded ?? root!;
      const sourceAtomic = forwarded !== undefined
        ? semanticFunctionSharedPointerAtomicParams(owner!).has(forwarded.name)
        : semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base);
      if (param.pointerParamAlias === undefined) {
        const pointerType = semanticTypedSharedPointerType(source, sourceAtomic, forwarded !== undefined);
        args.push(forwarded
          ? createTypedWgslIdentifier(nameFor(semanticParamAliasName(ir, forwarded) ?? forwarded.name, names), pointerType, ref.span)
          : createTypedWgslBindingAddress(nameFor(ref.base, names), pointerType, ref.span));
      }
      args.push(emitTypedSharedPointerArgBaseIndex(ref, forwarded, root, ir, names, options));
      return;
    }
    if (param.valueType === "bool") {
      args.push(emitSemanticBoolExpressionValue(arg, ir, names, options, textureSpecializations));
    } else if (param.valueType === "uchar") {
      args.push(emitSemanticUcharExpressionValue(arg, ir, names, options, textureSpecializations));
    } else if (isSemanticFloatVectorType(param.valueType)) {
      args.push(emitSemanticExpression(arg, ir, names, options, textureSpecializations));
    } else {
      args.push(emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param.valueType), options, textureSpecializations));
    }
  });
  args.push(
    createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("workgroup_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("num_workgroups", "vec3<u32>", expression.span),
  );
  return createTypedWgslCall(
    nameFor(semanticFunctionCallName(fn.name, fn, expression.args, options, textureSpecializations), names),
    args,
    wgslValueType(fn.returnType),
    expression.span,
  );
}

function semanticTypedLocalPointerFunctionForCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): SemanticKernelIrModule["functions"][number] | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace !== "function") return undefined;
  const callee = expression.callee;
  const fn = ir.functions.find((candidate) => semanticIdsEqual(callee.id, semanticSymbolIdFromFunction(candidate.id)));
  if (!fn || fn.returnType === "void" || fn.params.length !== expression.args.length) return undefined;
  if (!fn.params.some((param) => param.pointer && param.addressSpace === "local")) return undefined;
  if (fn.params.some((param) =>
    param.cooperativeGroupKind !== undefined || param.addressSpace === "texture" || param.addressSpace === "surface" ||
    param.pointer && param.addressSpace !== "local")) return undefined;
  return fn;
}

function emitSemanticTypedLocalPointerFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  fn: SemanticKernelIrModule["functions"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression {
  const args = expression.args.flatMap((arg, index): readonly TypedWgslExpression[] => {
    const param = fn.params[index]!;
    if (param.pointer) {
      const ref = semanticPointerArgMemoryRef(arg);
      if (!ref || ref.addressSpace !== "local") throw semanticWgslError(`local pointer argument '${param.name}' is not modeled local memory`, arg.span);
      const owner = options.activeFunction === undefined ? undefined : ir.functions.find((candidate) => candidate.name === options.activeFunction);
      const forwarded = owner?.params.find((candidate) => candidate.name === ref.base && candidate.pointer && candidate.addressSpace === "local");
      const pointerType = semanticTypedLocalPointerType(forwarded ?? param);
      const pointer = forwarded
        ? createTypedWgslIdentifier(nameFor(forwarded.name, names), pointerType, ref.span)
        : createTypedWgslBindingAddress(nameFor(ref.base, names), pointerType, ref.span);
      if (param.dimensions.length === 0) return [pointer];
      const offset = ref.indices[0] === undefined
        ? createTypedWgslLiteral("0u", "u32", ref.span)
        : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options);
      const base = forwarded?.dimensions.length
        ? emitTypedWgslBinary(
            "+",
            createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(forwarded.name), names), "u32", ref.span),
            offset,
            ref.span,
          )
        : offset;
      return [pointer, base];
    }
    if (param.valueType === "bool") return [emitSemanticBoolExpressionValue(arg, ir, names, options, textureSpecializations)];
    if (param.valueType === "uchar") return [emitSemanticUcharExpressionValue(arg, ir, names, options, textureSpecializations)];
    if (isSemanticFloatVectorType(param.valueType)) return [emitSemanticExpression(arg, ir, names, options, textureSpecializations)];
    return [emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param.valueType), options, textureSpecializations)];
  });
  args.push(
    createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("workgroup_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("num_workgroups", "vec3<u32>", expression.span),
  );
  return createTypedWgslCall(
    nameFor(semanticFunctionCallName(fn.name, fn, expression.args, options, textureSpecializations), names),
    args,
    wgslValueType(fn.returnType),
    expression.span,
  );
}

function semanticTypedLocalPointerType(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): WgslPointerType {
  const element = wgslValueType(param.valueType);
  return param.dimensions.length === 0
    ? `ptr<function,${element}>`
    : `ptr<function,array<${element},${totalElements(param.dimensions)}>>`;
}

function semanticTypedSharedPointerType(
  symbol: SemanticKernelIrModule["functions"][number]["params"][number],
  atomic: boolean,
  functionParam: boolean,
): WgslPointerType {
  if (symbol.pointerCarrierValueType === "uchar" || symbol.valueType === "uchar") {
    const bytes = symbol.dimensions.length === 0 ? 1 : functionParam ? symbol.dimensions[0] ?? 1 : totalElements(symbol.dimensions);
    return symbol.dimensions.length === 0
      ? "ptr<workgroup,atomic<u32>>"
      : `ptr<workgroup,array<atomic<u32>,${Math.ceil(bytes / 4)}>>`;
  }
  const carrier = symbol.pointerCarrierValueType ?? symbol.valueType;
  const element = atomic ? `atomic<${wgslAtomicScalar(carrier)}>` as const : wgslValueType(carrier);
  if (symbol.dimensions.length === 0) return `ptr<workgroup,${element}>`;
  const logicalElements = functionParam ? symbol.dimensions[0] ?? 1 : totalElements(symbol.dimensions);
  const elements = atomic && isCudaVectorType(carrier) ? logicalElements * cudaVectorLaneCount(carrier) : logicalElements;
  return `ptr<workgroup,array<${element},${elements}>>`;
}

function emitTypedSharedPointerArgBaseIndex(
  ref: SemanticMemoryRef,
  pointer: SemanticKernelIrModule["functions"][number]["params"][number] | undefined,
  root: SemanticKernelIrModule["memory"][number] | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression {
  if (pointer) {
    const base = createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span);
    if (ref.indices.length === 0) return base;
    if (pointer.dimensions.length === 0 || ref.indices.length !== 1) throw semanticWgslError(`shared pointer '${ref.base}' index rank mismatch`, ref.span);
    return emitTypedWgslBinary("+", base, emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options), ref.span);
  }
  if (!root) throw semanticWgslError(`unknown shared pointer base '${ref.base}'`, ref.span);
  if (ref.indices.length === 0) return createTypedWgslZero("u32", ref.span);
  return root.dimensions.length > 1
    ? emitTypedFlatRankedIndex(root.dimensions, ref.indices, ir, names, options, ref.span)
    : emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
}

function emitSemanticTypedStoragePointerFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  fn: SemanticKernelIrModule["functions"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression {
  const args: TypedWgslExpression[] = [];
  expression.args.forEach((arg, index) => {
    const param = fn.params[index]!;
    if (param.pointer) {
      const ref = semanticPointerArgMemoryRef(arg);
      if (!ref || ref.addressSpace !== "storage" && ref.addressSpace !== "device-global") {
        throw semanticWgslError(`storage pointer argument '${param.name}' is not modeled storage`, arg.span);
      }
      const forwarded = options.activeFunction === undefined
        ? undefined
        : ir.functions.find((candidate) => candidate.name === options.activeFunction)?.params.find((candidate) =>
          candidate.name === ref.base && candidate.pointer && candidate.addressSpace === "storage");
      if (forwarded) {
        args.push(createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(ref.base), names), "u32", ref.span));
      } else {
        const bufferId = semanticStoragePointerBufferId(ref.base, ir);
        if (bufferId === undefined) throw semanticWgslError(`unknown storage pointer base '${ref.base}'`, ref.span);
        args.push(createTypedWgslLiteral(`${bufferId}u`, "u32", ref.span));
      }
      args.push(emitTypedFlatStorageIndex(ref, ir, names, options));
      return;
    }
    if (param.valueType === "bool") {
      args.push(emitSemanticBoolExpressionValue(arg, ir, names, options, textureSpecializations));
    } else if (isSemanticFloatVectorType(param.valueType)) {
      args.push(emitSemanticExpression(arg, ir, names, options, textureSpecializations));
    } else {
      args.push(emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param.valueType), options, textureSpecializations));
    }
  });
  args.push(
    createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("workgroup_id", "vec3<u32>", expression.span),
    createTypedWgslIdentifier("num_workgroups", "vec3<u32>", expression.span),
  );
  return createTypedWgslCall(
    nameFor(semanticFunctionCallName(fn.name, fn, expression.args, options, textureSpecializations), names),
    args,
    wgslValueType(fn.returnType),
    expression.span,
  );
}

function emitSemanticTypedAddressPredicateCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): TypedWgslExpression | undefined {
  if (!semanticWgslAddressPredicateCallSupported(expression) || expression.callee.kind !== "symbol") return undefined;
  const addressSpace = semanticAddressPredicateAddressSpace(expression.args[0]);
  const kind = cudaAddressSpacePredicateKind(expression.callee.name);
  const matches = kind === "global"
    ? addressSpace === "storage" || addressSpace === "device-global"
    : kind !== undefined && addressSpace === kind;
  return createTypedWgslLiteral(matches ? "1" : "0", "i32", expression.span);
}

function emitSemanticTypedSharedAddressCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = semanticWgslSharedAddressCallRef(expression);
  if (!ref) return undefined;
  const pointer = semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null);
  let index: TypedWgslExpression;
  if (pointer) {
    index = createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span);
    if (ref.indices.length > 0) {
      if (pointer.dimensions.length === 0 || ref.indices.length !== 1) throw semanticWgslError(`shared pointer '${ref.base}' index rank mismatch`, ref.span);
      index = emitTypedWgslBinary("+", index, emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options), ref.span);
    }
  } else {
    const root = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
    if (!root) throw semanticWgslError(`unknown shared address root '${ref.base}'`, ref.span);
    index = ref.indices.length === 0
      ? createTypedWgslZero("u32", ref.span)
      : root.dimensions.length > 1
        ? emitTypedFlatRankedIndex(root.dimensions, ref.indices, ir, names, options, ref.span)
        : emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  }
  const elementBytes = sizeofCudaType(ref.valueType) ?? 1;
  return elementBytes === 1
    ? index
    : emitTypedWgslBinary("*", index, createTypedWgslLiteral(`${elementBytes}u`, "u32", ref.span), ref.span);
}

function emitSemanticTypedBf162Conversion(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  const round = (value: TypedWgslExpression): TypedWgslExpression => {
    const bits = createTypedWgslCall(
      "bg_f32_to_bf16_bits_mode",
      [value, createTypedWgslZero("u32", expression.span)],
      "u32",
      expression.span,
    );
    return createTypedWgslBitcast(
      "f32",
      emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span),
      expression.span,
    );
  };
  if (name === "__halves2bfloat162" || name === "__floats2bfloat162_rn") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    return createTypedWgslConstructor(
      "vec2<f32>",
      [
        round(emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations)),
        round(emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations)),
      ],
      expression.span,
    );
  }
  if (name === "__float22bfloat162_rn") {
    const value = expression.args[0];
    if (!value) return undefined;
    const vector = emitSemanticExpression(value, ir, names, options, textureSpecializations);
    return createTypedWgslConstructor(
      "vec2<f32>",
      [
        round(createTypedWgslMemberAccess(vector, "x", "f32", expression.span)),
        round(createTypedWgslMemberAccess(vector, "y", "f32", expression.span)),
      ],
      expression.span,
    );
  }
  if (name === "__float2bfloat162_rn" || name === "__bfloat162bfloat162") {
    const value = expression.args[0];
    if (!value) return undefined;
    const lane = round(emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations));
    return createTypedWgslConstructor("vec2<f32>", [lane], expression.span);
  }
  if (name === "__bfloat1622float2") {
    const value = expression.args[0];
    return value ? emitSemanticExpression(value, ir, names, options, textureSpecializations) : undefined;
  }
  if (name === "__low2bfloat162" || name === "__high2bfloat162" || name === "__lowhigh2highlow") {
    const value = expression.args[0];
    if (!value) return undefined;
    const vector = emitSemanticExpression(value, ir, names, options, textureSpecializations);
    const low = createTypedWgslMemberAccess(vector, "x", "f32", expression.span);
    const high = createTypedWgslMemberAccess(vector, "y", "f32", expression.span);
    return createTypedWgslConstructor(
      "vec2<f32>",
      name === "__low2bfloat162" ? [low] : name === "__high2bfloat162" ? [high] : [high, low],
      expression.span,
    );
  }
  if (name === "__lows2bfloat162" || name === "__highs2bfloat162") {
    const [left, right] = expression.args;
    if (!left || !right) return undefined;
    const field = name === "__lows2bfloat162" ? "x" : "y";
    return createTypedWgslConstructor(
      "vec2<f32>",
      [
        createTypedWgslMemberAccess(emitSemanticExpression(left, ir, names, options, textureSpecializations), field, "f32", expression.span),
        createTypedWgslMemberAccess(emitSemanticExpression(right, ir, names, options, textureSpecializations), field, "f32", expression.span),
      ],
      expression.span,
    );
  }
  if (name === "__low2bfloat16" || name === "__high2bfloat16" || name === "__low2float" || name === "__high2float") {
    const value = expression.args[0];
    if (!value) return undefined;
    return createTypedWgslMemberAccess(
      emitSemanticExpression(value, ir, names, options, textureSpecializations),
      name.includes("low") ? "x" : "y",
      "f32",
      expression.span,
    );
  }
  if (name === "__bfloat162_as_uint" || name === "__nv_bfloat162_as_uint") {
    const value = expression.args[0];
    if (!value) return undefined;
    const vector = emitSemanticExpression(value, ir, names, options, textureSpecializations);
    const low = emitTypedWgslBinary(
      ">>",
      createTypedWgslBitcast("u32", createTypedWgslMemberAccess(vector, "x", "f32", expression.span), expression.span),
      createTypedWgslLiteral("16u", "u32", expression.span),
      expression.span,
    );
    const high = emitTypedWgslBinary(
      "&",
      createTypedWgslBitcast("u32", createTypedWgslMemberAccess(vector, "y", "f32", expression.span), expression.span),
      createTypedWgslLiteral("0xffff0000u", "u32", expression.span),
      expression.span,
    );
    return emitTypedWgslBinary("|", low, high, expression.span);
  }
  if (name === "__uint_as_bfloat162" || name === "__uint_as_nv_bfloat162") {
    const value = expression.args[0];
    if (!value) return undefined;
    const bits = emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations);
    const low = createTypedWgslBitcast(
      "f32",
      emitTypedWgslBinary("<<", emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span), createTypedWgslLiteral("16u", "u32", expression.span), expression.span),
      expression.span,
    );
    const high = createTypedWgslBitcast("f32", emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffff0000u", "u32", expression.span), expression.span), expression.span);
    return createTypedWgslConstructor("vec2<f32>", [low, high], expression.span);
  }
  return undefined;
}

function emitSemanticTypedSubgroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (!semanticWgslSubgroupCallSupported(expression, ir) || expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  const span = expression.span;
  const localId = createTypedWgslIdentifier("local_id", "vec3<u32>", span);
  const u32 = (value: number): TypedWgslExpression => createTypedWgslLiteral(`${value}u`, "u32", span);
  if (name === "__activemask") {
    if (ir.subgroupMode === "scalar") return u32(1);
    if (options.activeCollectivePredicate !== undefined) return undefined;
    return createTypedWgslCall(
      semanticBallotHelper().name,
      [createTypedWgslLiteral("true", "bool", span), createTypedWgslLiteral("0xffffffffu", "u32", span), localId],
      "u32",
      span,
    );
  }
  const value = expression.args[
    isCudaWarpReduceCallName(name)
      ? expression.args.length - 1
      : name === "bg_subgroup_add" || legacyVoteCall(name) || legacyShuffleCall(name)
        ? 0
        : 1
  ];
  if (!value) return undefined;
  const voteOp = cudaVoteOpForCall(name);
  if (ir.subgroupMode === "scalar") {
    if (voteOp === "any" || voteOp === "all" || voteOp === "ballot") {
      return emitTypedWgslSelect(
        u32(0),
        u32(1),
        emitSemanticTruthinessExpression(value, ir, names, options),
        span,
      );
    }
    if (voteOp === "match-any") return u32(1);
    const valueType = semanticExpressionValueType(value);
    if (!valueType || valueType === "void") return undefined;
    return emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
  }
  if (voteOp === "any" || voteOp === "all" || voteOp === "ballot") {
    const predicate = emitSemanticTruthinessExpression(value, ir, names, options);
    const activeMask = legacyVoteCall(name)
      ? createTypedWgslLiteral("0xffffffffu", "u32", span)
      : emitSemanticExpressionAs(expression.args[0]!, ir, names, "u32", options, textureSpecializations);
    const ballot = (condition: TypedWgslExpression): TypedWgslExpression => createTypedWgslCall(
      semanticBallotHelper().name,
      [condition, activeMask, localId],
      "u32",
      span,
    );
    if (voteOp === "ballot") return ballot(predicate);
    const bits = voteOp === "all" ? ballot(emitTypedWgslUnary("!", predicate, span)) : ballot(predicate);
    const condition = emitTypedWgslBinary(voteOp === "all" ? "==" : "!=", bits, u32(0), span);
    return emitTypedWgslSelect(u32(0), u32(1), condition, span);
  }
  if (voteOp === "match-any") {
    const valueType = semanticExpressionValueType(value);
    if (!valueType || valueType === "void") return undefined;
    return createTypedWgslCall(
      semanticMatchAnyHelper(valueType, 32).name,
      [emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations), u32(32), localId],
      "u32",
      span,
    );
  }
  const bitwiseReduceOp = semanticBitwiseReduceOpForCall(name);
  if (bitwiseReduceOp) {
    const valueType = semanticExpressionValueType(value);
    if (valueType !== "int" && valueType !== "uint") return undefined;
    const scalar = wgslValueScalar(valueType);
    return createTypedWgslCall(
      semanticBitwiseReduceHelper(bitwiseReduceOp, valueType, 32).name,
      [emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations), u32(32), localId],
      scalar,
      span,
    );
  }
  const shuffleOp = semanticShuffleOpForCall(name);
  if (shuffleOp) {
    const valueType = semanticExpressionValueType(value);
    if (!valueType || valueType === "void") return undefined;
    const scalar = wgslValueScalar(valueType);
    const indexArg = legacyShuffleCall(name) ? expression.args[1] : expression.args[2];
    const widthArg = legacyShuffleCall(name) ? expression.args[2] : expression.args[3];
    return createTypedWgslCall(
      semanticWarpShuffleHelper(shuffleOp, valueType, semanticShuffleTileSize(expression)).name,
      [
        emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations),
        indexArg ? emitSemanticExpressionAs(indexArg, ir, names, "u32", options, textureSpecializations) : u32(0),
        widthArg ? emitSemanticExpressionAs(widthArg, ir, names, "u32", options, textureSpecializations) : u32(32),
        localId,
      ],
      scalar,
      span,
    );
  }
  const arithmeticReduceOp = cudaArithmeticReduceOpForCall(name);
  if (arithmeticReduceOp !== undefined) {
    const scalar = semanticExpressionWgslScalar(value);
    const callee = arithmeticReduceOp === "add" ? "subgroupAdd" : arithmeticReduceOp === "min" ? "subgroupMin" : "subgroupMax";
    return createTypedWgslCall(
      callee,
      [emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations)],
      scalar,
      span,
    );
  }
  return undefined;
}

function emitSemanticTypedCooperativeReduceCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (!semanticWgslCooperativeReduceCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir))) {
    return undefined;
  }
  const helper = semanticCooperativeReduceHelperFor(ir, expression);
  const value = semanticCooperativeReduceValue(expression);
  if (!helper || !value) return undefined;
  const scalar = wgslValueScalar(helper.valueType);
  const args: TypedWgslExpression[] = [
    emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations),
  ];
  if (helper.masked) {
    if (helper.partitioned) {
      const groupArg = expression.args[0];
      const group = groupArg?.kind === "symbol" ? semanticCooperativeGroupInfo(ir, groupArg.name) : undefined;
      if (!group?.partitionPredicate) return undefined;
      args.push(createTypedWgslCall(
        semanticBallotHelper().name,
        [
          emitSemanticTruthinessExpression(group.partitionPredicate, ir, names, options),
          createTypedWgslLiteral("0xffffffffu", "u32", expression.span),
          createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
        ],
        "u32",
        expression.span,
      ));
    } else {
      const mask = expression.args[0];
      if (!mask) return undefined;
      args.push(emitSemanticExpressionAs(mask, ir, names, "u32", options, textureSpecializations));
    }
  }
  args.push(createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span));
  return createTypedWgslCall(helper.name, args, scalar, expression.span);
}

function semanticWgslCooperativeVectorReduceCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const helper = semanticCooperativeVectorReduceHelperFor(ir, expression);
  return helper !== undefined && expression.args[1] !== undefined && semanticWgslExpressionSupported(expression.args[1], "any", ir);
}

function emitSemanticTypedCooperativeVectorReduceCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (!semanticWgslCooperativeVectorReduceCallSupported(expression, ir)) return undefined;
  const helper = semanticCooperativeVectorReduceHelperFor(ir, expression);
  const value = expression.args[1];
  if (!helper || !value) return undefined;
  const type = semanticExpressionWgslType(value, ir);
  if (!isWgslVectorType(type)) return undefined;
  return createTypedWgslCall(
    helper.name,
    [
      emitSemanticExpression(value, ir, names, options, textureSpecializations),
      createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
      createTypedWgslIdentifier("workgroup_id", "vec3<u32>", expression.span),
      createTypedWgslIdentifier("num_workgroups", "vec3<u32>", expression.span),
    ],
    type,
    expression.span,
  );
}

function emitSemanticTypedCooperativeScanCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (!semanticWgslCooperativeScanCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir))) {
    return undefined;
  }
  const helper = semanticCooperativeScanHelperFor(ir, expression);
  const value = expression.args[1];
  if (!helper || !value) return undefined;
  const scalar = wgslValueScalar(helper.valueType);
  return createTypedWgslCall(
    helper.name,
    [
      emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations),
      createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
    ],
    scalar,
    expression.span,
  );
}

function semanticWgslSyncthreadsPredicateCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticSyncthreadsPredicateHelperFor(expression) !== undefined &&
    expression.args[0] !== undefined && semanticWgslExpressionSupported(expression.args[0], "scalar", ir);
}

function emitSemanticTypedSyncthreadsPredicateCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  if (!semanticWgslSyncthreadsPredicateCallSupported(expression, ir)) return undefined;
  const helper = semanticSyncthreadsPredicateHelperFor(expression);
  const predicate = expression.args[0];
  if (!helper || !predicate) return undefined;
  return createTypedWgslCall(
    helper.name,
    [
      emitSemanticTruthinessExpression(predicate, ir, names, options),
      createTypedWgslIdentifier("local_id", "vec3<u32>", expression.span),
    ],
    "i32",
    expression.span,
  );
}

function semanticWgslCoalescedGroupCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (ir.subgroupMode === "scalar" || expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") return false;
  const group = semanticCooperativeGroupInfo(ir, expression.callee.object.name);
  if (group?.kind !== "coalesced") return false;
  if (expression.callee.property === "ballot" || expression.callee.property === "any" || expression.callee.property === "all") {
    return expression.args.length === 1 && semanticWgslExpressionSupported(expression.args[0]!, "scalar", ir);
  }
  return expression.callee.property === "shfl" && expression.args.length === 2 &&
    expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
}

function emitSemanticTypedCoalescedGroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (!semanticWgslCoalescedGroupCallSupported(expression, ir) || expression.callee.kind !== "member") return undefined;
  const method = expression.callee.property;
  const value = expression.args[0]!;
  if (method === "ballot") {
    const ballot = createTypedWgslCall(
      "subgroupBallot",
      [emitSemanticTruthinessExpression(value, ir, names, options)],
      "vec4<u32>",
      expression.span,
    );
    return createTypedWgslMemberAccess(ballot, "x", "u32", expression.span);
  }
  if (method === "any" || method === "all") {
    const vote = createTypedWgslCall(
      method === "any" ? "subgroupAny" : "subgroupAll",
      [emitSemanticTruthinessExpression(value, ir, names, options)],
      "bool",
      expression.span,
    );
    return legalizeTypedWgslBoolToNumeric(vote, expression.valueType === "uint" ? "u32" : "i32");
  }
  const valueType = semanticExpressionValueType(value);
  const index = expression.args[1];
  if (!valueType || valueType === "void" || !index) return undefined;
  const scalar = wgslValueScalar(valueType);
  return createTypedWgslCall(
    "subgroupShuffle",
    [
      emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations),
      emitSemanticExpressionAs(index, ir, names, "u32", options, textureSpecializations),
    ],
    scalar,
    expression.span,
  );
}

function emitSemanticTypedCooperativeGroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  if (!semanticWgslCooperativeGroupCallSupported(expression, ir) || expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") {
    return undefined;
  }
  const span = expression.span;
  const groupName = expression.callee.object.name;
  const group = semanticCooperativeGroupInfo(ir, groupName)!;
  const resultType = expression.valueType === "uint" ? "u32" : "i32";
  const groupParam = options.activeFunction === undefined
    ? undefined
    : ir.functions.find((fn) => fn.name === options.activeFunction)?.params
      .find((param) => param.name === groupName && param.cooperativeGroupKind !== undefined);
  if (groupParam && (expression.callee.property === "thread_rank" || expression.callee.property === "size")) {
    const suffix = expression.callee.property === "thread_rank"
      ? semanticCooperativeGroupRankParamName(groupParam.name)
      : semanticCooperativeGroupSizeParamName(groupParam.name);
    return convertTypedWgslExpression(createTypedWgslIdentifier(suffix, "i32", span), resultType);
  }

  const u32 = (value: number): TypedWgslExpression => createTypedWgslLiteral(`${value}u`, "u32", span);
  const i32 = (value: number): TypedWgslExpression => createTypedWgslLiteral(String(value), "i32", span);
  const binary = (operator: "+" | "-" | "*" | "/" | "%", left: TypedWgslExpression, right: TypedWgslExpression): TypedWgslExpression =>
    emitTypedWgslBinary(operator, left, right, span);
  const localRank = semanticTypedCooperativeLocalLinearRank(ir, span);
  const asResult = (value: TypedWgslExpression): TypedWgslExpression => convertTypedWgslExpression(value, resultType);

  if (expression.callee.property === "thread_rank") {
    if (group.kind === "grid") return asResult(semanticTypedCooperativeGlobalLinearRank(ir, span));
    if (group.kind === "tile") return asResult(binary("%", localRank, u32(group.tileSize ?? 32)));
    return asResult(localRank);
  }
  if (expression.callee.property === "size") {
    if (group.kind === "grid") return asResult(semanticTypedCooperativeGridThreadCount(ir, span));
    return resultType === "u32"
      ? u32(group.kind === "tile" ? group.tileSize ?? 32 : semanticCooperativeWorkgroupSize(ir))
      : i32(group.kind === "tile" ? group.tileSize ?? 32 : semanticCooperativeWorkgroupSize(ir));
  }
  if (expression.callee.property === "meta_group_rank") {
    const rank = group.kind === "tile" ? binary("/", localRank, u32(group.tileSize ?? 32)) : u32(0);
    return asResult(rank);
  }
  if (expression.callee.property === "meta_group_size") {
    const size = group.kind === "tile"
      ? Math.ceil(semanticCooperativeWorkgroupSize(ir) / (group.tileSize ?? 32))
      : 1;
    return resultType === "u32" ? u32(size) : i32(size);
  }
  return undefined;
}

function emitSemanticTypedCooperativeGroupArguments(
  arg: SemanticExpression,
  ir: SemanticKernelIrModule,
  _names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): readonly [TypedWgslExpression, TypedWgslExpression] {
  if (arg.kind !== "symbol") throw semanticWgslError("cooperative-group argument must be a semantic symbol", arg.span);
  const call = (property: "thread_rank" | "size"): Extract<SemanticExpression, { readonly kind: "call" }> => ({
    kind: "call",
    callee: { kind: "member", object: arg, property, valueType: "int", span: arg.span },
    args: [],
    valueType: "int",
    span: arg.span,
  });
  const rank = emitSemanticTypedCooperativeGroupCall(call("thread_rank"), ir, options);
  const size = emitSemanticTypedCooperativeGroupCall(call("size"), ir, options);
  if (!rank || !size) throw semanticWgslError(`unknown cooperative group '${arg.name}'`, arg.span);
  return [rank, size];
}

function semanticTypedCooperativeLocalLinearRank(
  ir: SemanticKernelIrModule,
  span: SourceSpan,
): TypedWgslExpression {
  const [x, y, z] = ir.workgroupSize;
  const localId = createTypedWgslIdentifier("local_id", "vec3<u32>", span);
  const lane = (field: "x" | "y" | "z"): TypedWgslExpression => createTypedWgslMemberAccess(localId, field, "u32", span);
  const u32 = (value: number): TypedWgslExpression => createTypedWgslLiteral(`${value}u`, "u32", span);
  let rank = lane("x");
  if (y !== 1 || z !== 1) rank = emitTypedWgslBinary("+", rank, emitTypedWgslBinary("*", lane("y"), u32(x), span), span);
  if (z !== 1) rank = emitTypedWgslBinary("+", rank, emitTypedWgslBinary("*", lane("z"), u32(x * y), span), span);
  return rank;
}

function semanticTypedCooperativeGlobalLinearRank(
  ir: SemanticKernelIrModule,
  span: SourceSpan,
): TypedWgslExpression {
  const workgroupId = createTypedWgslIdentifier("workgroup_id", "vec3<u32>", span);
  const numWorkgroups = createTypedWgslIdentifier("num_workgroups", "vec3<u32>", span);
  const member = (object: TypedWgslExpression, field: "x" | "y" | "z"): TypedWgslExpression =>
    createTypedWgslMemberAccess(object, field, "u32", span);
  const product = (left: TypedWgslExpression, right: TypedWgslExpression): TypedWgslExpression => emitTypedWgslBinary("*", left, right, span);
  const sum = (left: TypedWgslExpression, right: TypedWgslExpression): TypedWgslExpression => emitTypedWgslBinary("+", left, right, span);
  const flatWorkgroup = sum(
    member(workgroupId, "x"),
    sum(
      product(member(workgroupId, "y"), member(numWorkgroups, "x")),
      product(product(member(workgroupId, "z"), member(numWorkgroups, "x")), member(numWorkgroups, "y")),
    ),
  );
  return sum(
    semanticTypedCooperativeLocalLinearRank(ir, span),
    product(createTypedWgslLiteral(`${semanticCooperativeWorkgroupSize(ir)}u`, "u32", span), flatWorkgroup),
  );
}

function semanticTypedCooperativeGridThreadCount(
  ir: SemanticKernelIrModule,
  span: SourceSpan,
): TypedWgslExpression {
  const numWorkgroups = createTypedWgslIdentifier("num_workgroups", "vec3<u32>", span);
  const member = (field: "x" | "y" | "z"): TypedWgslExpression => createTypedWgslMemberAccess(numWorkgroups, field, "u32", span);
  return [member("x"), member("y"), member("z")].reduce(
    (value, factor) => emitTypedWgslBinary("*", value, factor, span),
    createTypedWgslLiteral(`${semanticCooperativeWorkgroupSize(ir)}u`, "u32", span),
  );
}

function semanticCooperativeWorkgroupSize(ir: SemanticKernelIrModule): number {
  return ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
}

function emitSemanticConstantVectorSymbolExpression(
  expression: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): TypedWgslExpression | undefined {
  if (expression.addressSpace !== "constant") return undefined;
  const symbol = ir.memory.find((item) => semanticIdsEqual(item.id, expression.id));
  const valueType = semanticStorageVectorType(symbol?.valueType);
  if (!symbol || !valueType || symbol.initialized) return undefined;
  const targetType = wgslValueType(valueType);
  if (!isWgslVectorType(targetType)) return undefined;
  const laneType = wgslVectorScalar(valueType);
  const lanes = Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    createTypedWgslMemoryRead(
      nameFor(expression.name, names),
      createTypedWgslLiteral(`${lane}u`, "u32", expression.span),
      laneType,
      false,
      expression.span,
    )
  );
  return createTypedWgslConstructor(targetType, lanes, expression.span);
}

function emitSemanticTypedIntegerAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = wgslAtomicCalleeForCudaAtomic(expression.callee.name);
  const target = semanticAtomicCallTarget(expression);
  if (!target) return undefined;
  if (expression.callee.name === "atomicExch" && target.valueType === "int" && semanticWgslDirectByteRawView(target, ir)) {
    const value = expression.args[1];
    if (!value) return undefined;
    const byteIndex = emitTypedFlatStorageIndex({ ...target, valueType: "uchar" }, ir, names, options);
    const wordIndex = emitTypedWgslBinary(">>", byteIndex, createTypedWgslLiteral("2u", "u32", target.span), target.span);
    const place = createTypedWgslIndexedPlace(nameFor(target.base, names), wordIndex, "u32", true, target.span, "storage");
    const old = createTypedWgslAtomicCall(
      "atomicExchange",
      place,
      [createTypedWgslBitcast("u32", emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations), expression.span)],
      expression.span,
    );
    return createTypedWgslBitcast("i32", old, expression.span);
  }
  const pointerParam = options.activeFunction === undefined ? undefined : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
    param.name === target.base && param.pointer && param.addressSpace === "storage"
  );
  const localPointer = expression.args[0] === undefined ? undefined : semanticLocalStoragePointerDeclaration(ir, expression.args[0]);
  if (pointerParam || localPointer) {
    const valueType = target.valueType ?? "float";
    if (!semanticWgslPointerAtomicCallSupported(expression.callee.name, valueType)) return undefined;
    const scalar = wgslValueScalar(valueType);
    const cas = semanticAtomicOperation(expression.callee.name) === "cas";
    const operands = expression.args.slice(1, cas ? 3 : 2);
    if (operands.length !== (cas ? 2 : 1)) return undefined;
    const index = isCudaVectorType(valueType)
      ? emitTypedFlatStorageVectorBaseIndex(target, ir, names, options)
      : emitTypedFlatStorageIndex(target, ir, names, options);
    return createTypedWgslCall(
      semanticPointerAtomicHelperName(expression.callee.name, valueType),
      [
        createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(target.base), names), "u32", target.span),
        index,
        ...operands.map((operand) => emitSemanticExpressionAs(operand, ir, names, scalar, options, textureSpecializations)),
      ],
      scalar,
      expression.span,
    );
  }
  const place = emitSemanticTypedAtomicPlace(target, ir, names, options);
  if (!place) return undefined;
  const loopAtomicKind = wgslIntegerLoopAtomicKindForCudaAtomic(expression.callee.name);
  if (loopAtomicKind) {
    const limit = expression.args[1];
    if (!limit) return undefined;
    return createTypedWgslCall(
      semanticIntegerLoopAtomicHelperName(loopAtomicKind, target, ir),
      [createTypedWgslAddressOf(place), emitSemanticExpressionAs(limit, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (semanticAtomicSupportsBfloatAdd(expression.callee.name, target.valueType)) {
    const value = expression.args[1];
    if (!value) return undefined;
    return createTypedWgslCall(
      bfloatAtomicAddHelperName(semanticWgslAtomicAddressSpace(target)),
      [createTypedWgslAddressOf(place), emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  const floatAtomicKind = semanticAtomicUsesF32Storage(target.valueType) ? semanticWgslFloatAtomicCallKind(expression.callee.name) : undefined;
  if (floatAtomicKind) {
    const [first, second] = expression.args.slice(1);
    if (!first) return undefined;
    if (floatAtomicKind === "Exchange") {
      const old = createTypedWgslAtomicCall(
        "atomicExchange",
        place,
        [createTypedWgslBitcast("u32", emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations), expression.span)],
        expression.span,
      );
      return createTypedWgslBitcast("f32", old, expression.span);
    }
    if (floatAtomicKind === "CompareExchange") {
      if (!second) return undefined;
      const old = createTypedWgslAtomicCall(
        "atomicCompareExchangeWeak",
        place,
        [
          createTypedWgslBitcast("u32", emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations), expression.span),
          createTypedWgslBitcast("u32", emitSemanticExpressionAs(second, ir, names, "f32", options, textureSpecializations), expression.span),
        ],
        expression.span,
      );
      return createTypedWgslBitcast("f32", old, expression.span);
    }
    return createTypedWgslCall(
      floatAtomicHelperName(floatAtomicKind, semanticWgslAtomicAddressSpace(target)),
      [createTypedWgslAddressOf(place), emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (!callee || target.valueType !== "int" && target.valueType !== "uint") return undefined;
  const type = wgslAtomicScalar(target.valueType);
  if (place.type !== type) return undefined;
  const operandCount = callee === "atomicCompareExchangeWeak" ? 2 : 1;
  const operands = expression.args.slice(1, 1 + operandCount).map((operand) =>
    emitSemanticExpressionAs(operand, ir, names, type, options, textureSpecializations)
  );
  if (operands.length !== operandCount) throw semanticWgslError(`atomic '${expression.callee.name}' missing operand`, expression.span);
  return createTypedWgslAtomicCall(callee, place, operands, expression.span);
}

function emitSemanticTypedAtomicPlace(
  target: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslPlace | undefined {
  if (target.fields.length !== 0 || target.indices.length > 1 || target.packedByteLanes !== undefined) return undefined;
  const type = wgslAtomicScalar(target.valueType);
  const sharedPointer = options.activeFunction === undefined ? undefined : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
    param.name === target.base && param.pointer && param.addressSpace === "shared"
  );
  if (sharedPointer) {
    const offset = target.indices[0] === undefined
      ? createTypedWgslZero("u32", target.span)
      : emitSemanticExpressionAs(target.indices[0], ir, names, "u32", options);
    const index = emitTypedWgslBinary(
      "+",
      createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(target.base), names), "u32", target.span),
      offset,
      target.span,
    );
    return createTypedWgslDereferencedIndexedPlace(
      nameFor(semanticParamAliasName(ir, sharedPointer) ?? target.base, names),
      index,
      type,
      true,
      "workgroup",
      target.span,
    );
  }
  if (semanticWgslFunctionStoragePointerParam(ir, target.base, options.activeFunction ?? null)) return undefined;
  const index = target.addressSpace === "storage" || target.addressSpace === "device-global"
    ? emitTypedFlatStorageIndex(target, ir, names, options)
    : target.indices[0] === undefined
      ? createTypedWgslZero("u32", target.span)
      : emitSemanticExpressionAs(target.indices[0], ir, names, "u32", options);
  return createTypedWgslIndexedPlace(
    nameFor(target.base, names),
    index,
    type,
    true,
    target.span,
    target.addressSpace === "shared" ? "workgroup" : "storage",
  );
}

type TypedPackedLaneWidth = 8 | 16;

function typedPackedLaneBits(value: TypedWgslExpression, lane: number, width: TypedPackedLaneWidth, span: SourceSpan): TypedWgslExpression {
  const shifted = lane === 0 ? value : emitTypedWgslBinary(">>", value, createTypedWgslLiteral(`${lane * width}u`, "u32", span), span);
  return emitTypedWgslBinary("&", shifted, createTypedWgslLiteral(width === 8 ? "0xffu" : "0xffffu", "u32", span), span);
}

function typedPackedSignedLane(bits: TypedWgslExpression, width: TypedPackedLaneWidth, span: SourceSpan): TypedWgslExpression {
  const signBit = createTypedWgslLiteral(width === 8 ? "0x80u" : "0x8000u", "u32", span);
  const correction = createTypedWgslLiteral(width === 8 ? "256" : "65536", "i32", span);
  return emitTypedWgslBinary(
    "-",
    convertTypedWgslExpression(bits, "i32", true),
    emitTypedWgslSelect(createTypedWgslZero("i32", span), correction, emitTypedWgslBinary(">=", bits, signBit, span), span),
    span,
  );
}

function typedPackLanes(lanes: readonly TypedWgslExpression[], width: TypedPackedLaneWidth, span: SourceSpan): TypedWgslExpression {
  const mask = createTypedWgslLiteral(width === 8 ? "0xffu" : "0xffffu", "u32", span);
  return lanes.map((lane, index) => {
    const bits = emitTypedWgslBinary("&", lane.type === "u32" ? lane : convertTypedWgslExpression(lane, "u32", true), mask, span);
    return index === 0 ? bits : emitTypedWgslBinary("<<", bits, createTypedWgslLiteral(`${index * width}u`, "u32", span), span);
  }).reduce((left, right) => emitTypedWgslBinary("|", left, right, span));
}

function emitTypedPackedComparison(
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  signed: boolean,
  operator: "==" | "!=" | ">=" | ">" | "<=" | "<",
  reduceAll: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const comparisons = Array.from({ length: 32 / width }, (_, lane) => {
    const lhsBits = typedPackedLaneBits(left, lane, width, span);
    const rhsBits = typedPackedLaneBits(right, lane, width, span);
    return emitTypedWgslBinary(operator, signed ? typedPackedSignedLane(lhsBits, width, span) : lhsBits, signed ? typedPackedSignedLane(rhsBits, width, span) : rhsBits, span);
  });
  if (reduceAll) {
    const predicate = comparisons.slice(1).reduce((result, value) => emitTypedWgslBinary("&&", result, value, span), comparisons[0]!);
    return emitTypedWgslSelect(createTypedWgslZero("u32", span), createTypedWgslLiteral("1u", "u32", span), predicate, span);
  }
  const mask = createTypedWgslLiteral(width === 8 ? "0xffu" : "0xffffu", "u32", span);
  return typedPackLanes(comparisons.map((predicate) => emitTypedWgslSelect(createTypedWgslZero("u32", span), mask, predicate, span)), width, span);
}

function emitTypedPackedUnary(
  value: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  op: "abs" | "sat_abs" | "neg" | "sat_neg",
  span: SourceSpan,
): TypedWgslExpression {
  const minimum = createTypedWgslLiteral(width === 8 ? "-128" : "-32768", "i32", span);
  const maximum = createTypedWgslLiteral(width === 8 ? "127" : "32767", "i32", span);
  const lanes = Array.from({ length: 32 / width }, (_, lane): TypedWgslExpression => {
    const signed = typedPackedSignedLane(typedPackedLaneBits(value, lane, width, span), width, span);
    if (op === "abs") return createTypedWgslCall("abs", [signed], "i32", span);
    if (op === "sat_abs") return createTypedWgslCall("min", [maximum, createTypedWgslCall("abs", [signed], "i32", span)], "i32", span);
    const negated = emitTypedWgslUnary("-", signed, span);
    return op === "neg" ? negated : createTypedWgslCall("clamp", [negated, minimum, maximum], "i32", span);
  });
  return typedPackLanes(lanes, width, span);
}

function emitTypedPackedAverage(
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  signedRounded: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = Array.from({ length: 32 / width }, (_, lane): TypedWgslExpression => {
    const lhsBits = typedPackedLaneBits(left, lane, width, span);
    const rhsBits = typedPackedLaneBits(right, lane, width, span);
    if (!signedRounded) return emitTypedWgslBinary(">>", emitTypedWgslBinary("+", lhsBits, rhsBits, span), createTypedWgslLiteral("1u", "u32", span), span);
    const sum = emitTypedWgslBinary("+", emitTypedWgslBinary("+", typedPackedSignedLane(lhsBits, width, span), typedPackedSignedLane(rhsBits, width, span), span), createTypedWgslLiteral("1", "i32", span), span);
    return emitTypedWgslBinary(">>", sum, createTypedWgslLiteral("1u", "u32", span), span);
  });
  return typedPackLanes(lanes, width, span);
}

function emitTypedPackedDifference(
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  signed: boolean,
  pack: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = Array.from({ length: 32 / width }, (_, lane): TypedWgslExpression => {
    const lhsBits = typedPackedLaneBits(left, lane, width, span);
    const rhsBits = typedPackedLaneBits(right, lane, width, span);
    const lhs = signed ? typedPackedSignedLane(lhsBits, width, span) : convertTypedWgslExpression(lhsBits, "i32", true);
    const rhs = signed ? typedPackedSignedLane(rhsBits, width, span) : convertTypedWgslExpression(rhsBits, "i32", true);
    return createTypedWgslCall("abs", [emitTypedWgslBinary("-", lhs, rhs, span)], "i32", span);
  });
  if (pack) return typedPackLanes(lanes, width, span);
  return lanes.map((lane) => convertTypedWgslExpression(lane, "u32", true)).reduce((sum, lane) => emitTypedWgslBinary("+", sum, lane, span));
}

function emitTypedPackedViadd(
  args: readonly TypedWgslExpression[],
  signed: boolean,
  choose: "min" | "max",
  relu: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = [0, 1].map((lane): TypedWgslExpression => {
    const values = args.map((arg) => {
      const bits = typedPackedLaneBits(arg, lane, 16, span);
      return signed ? typedPackedSignedLane(bits, 16, span) : convertTypedWgslExpression(bits, "i32", true);
    });
    const sum = emitTypedWgslBinary("+", values[0]!, values[1]!, span);
    const selected = createTypedWgslCall(choose, [sum, values[2]!], "i32", span);
    return relu ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", span)], "i32", span) : selected;
  });
  return typedPackLanes(lanes, 16, span);
}

function emitTypedPackedMinMax(
  args: readonly TypedWgslExpression[],
  signed: boolean,
  choose: "min" | "max",
  relu: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = [0, 1].map((lane): TypedWgslExpression => {
    const values = args.map((arg) => {
      const bits = typedPackedLaneBits(arg, lane, 16, span);
      return signed ? typedPackedSignedLane(bits, 16, span) : convertTypedWgslExpression(bits, "i32", true);
    });
    const selected = values.slice(1).reduce((result, value) => createTypedWgslCall(choose, [result, value], "i32", span), values[0]!);
    return relu ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", span)], "i32", span) : selected;
  });
  return typedPackLanes(lanes, 16, span);
}

function emitSemanticTypedCustomMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  if (!callee) return undefined;
  const first = expression.args[0];
  if (callee === "clock") return emitSemanticTypedClock(expression.span);
  if (callee === "tf32") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
  }
  if (callee === "fp8_to_half") {
    const [bits, format] = expression.args;
    if (!bits || !format) throw semanticWgslError(`${expression.callee.name} expects bits and format`, expression.span);
    const value = createTypedWgslCall(
      "bg_fp8_to_f32",
      [
        emitSemanticExpressionAs(bits, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(format, ir, names, "u32", options, textureSpecializations),
      ],
      "f32",
      expression.span,
    );
    return convertTypedWgslExpression(value, "f16", true);
  }
  if (callee === "float_to_fp8") {
    const [value, saturate, format] = expression.args;
    if (!value || !saturate || !format) throw semanticWgslError(`${expression.callee.name} expects value, saturation mode, and format`, expression.span);
    return createTypedWgslCall(
      "bg_f32_to_fp8",
      [
        emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
        emitSemanticExpressionAs(saturate, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(format, ir, names, "u32", options, textureSpecializations),
      ],
      "u32",
      expression.span,
    );
  }
  if (callee === "dp4a" || callee === "dp2a_lo" || callee === "dp2a_hi") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const resultType = expression.valueType === "uint" ? "u32" : "i32";
    const args: TypedWgslExpression[] = [
        emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(addend, ir, names, resultType, options, textureSpecializations),
      ];
    if (callee !== "dp4a") args.push(createTypedWgslLiteral(callee === "dp2a_hi" ? "16u" : "0u", "u32", expression.span));
    return createTypedWgslCall(
      callee === "dp4a"
        ? resultType === "u32" ? "bg_semantic_dp4a_u32" : "bg_semantic_dp4a_i32"
        : resultType === "u32" ? "bg_semantic_dp2a_u32" : "bg_semantic_dp2a_i32",
      args,
      resultType,
      expression.span,
    );
  }
  if (callee === "i16_lane" || callee === "u16_lane") {
    const [value, shift] = expression.args;
    if (!value || !shift) throw semanticWgslError(`${expression.callee.name} expects value and shift`, expression.span);
    const bits = emitTypedWgslBinary(
      "&",
      emitTypedWgslBinary(">>", emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(shift, ir, names, "u32", options, textureSpecializations), expression.span),
      createTypedWgslLiteral("0xffffu", "u32", expression.span),
      expression.span,
    );
    return callee === "i16_lane" ? typedPackedSignedLane(bits, 16, expression.span) : bits;
  }
  if (callee.startsWith("vset") || callee.startsWith("vcmp")) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const laneWidth = callee.endsWith("2") ? 16 : 8;
    const opName = callee.slice(4, -1);
    const signed = opName.endsWith("s");
    const operator = opName === "eq" ? "==" : opName === "ne" ? "!=" : opName.startsWith("ge") ? ">=" : opName.startsWith("gt") ? ">" : opName.startsWith("le") ? "<=" : "<";
    return emitTypedPackedComparison(
      emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
      laneWidth,
      signed,
      operator,
      callee.startsWith("vset"),
      expression.span,
    );
  }
  if (["vabs2", "vabsss2", "vneg2", "vnegss2", "vabs4", "vabsss4", "vneg4", "vnegss4"].includes(callee)) {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const op = callee.startsWith("vabsss") ? "sat_abs" : callee.startsWith("vabs") ? "abs" : callee.startsWith("vnegss") ? "sat_neg" : "neg";
    return emitTypedPackedUnary(
      emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations),
      callee.endsWith("2") ? 16 : 8,
      op,
      expression.span,
    );
  }
  if (/^(?:vabsdiffs|vsads|vsadu)[24]$/u.test(callee)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedPackedDifference(
      emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
      callee.endsWith("2") ? 16 : 8,
      !callee.startsWith("vsadu"),
      callee.startsWith("vabsdiffs"),
      expression.span,
    );
  }
  if (callee === "vhaddu2" || callee === "vhaddu4" || callee === "vavgs2" || callee === "vavgs4") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedPackedAverage(
      emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
      callee.endsWith("2") ? 16 : 8,
      callee.startsWith("vavgs"),
      expression.span,
    );
  }
  if (callee.startsWith("viadd")) {
    const [left, right, compare] = expression.args;
    if (!left || !right || !compare) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const choose = callee.startsWith("viaddmax") ? "max" : "min";
    const relu = callee.endsWith("_relu");
    if (callee.includes("16x2")) {
      return emitTypedPackedViadd(
        expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)),
        callee.includes("_s16x2"), choose, relu, expression.span,
      );
    }
    const type = callee.includes("_s32") ? "i32" : "u32";
    const sum = emitTypedWgslBinary("+", emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations), expression.span);
    const selected = createTypedWgslCall(choose, [sum, emitSemanticExpressionAs(compare, ir, names, type, options, textureSpecializations)], type, expression.span);
    return relu && type === "i32" ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", expression.span)], "i32", expression.span) : selected;
  }
  if (/^(?:vimax|vimin|vibmax|vibmin)/u.test(callee)) {
    const choose = callee.includes("max") ? "max" : "min";
    const relu = callee.endsWith("_relu");
    if (callee.includes("16x2")) {
      return emitTypedPackedMinMax(
        expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)),
        callee.includes("_s16x2"), choose, relu, expression.span,
      );
    }
    const type = callee.includes("_s32") ? "i32" : "u32";
    const values = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, type, options, textureSpecializations));
    if (values.length < 2) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    const selected = values.slice(1).reduce((result, value) => createTypedWgslCall(choose, [result, value], type, expression.span), values[0]!);
    return relu && type === "i32" ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", expression.span)], "i32", expression.span) : selected;
  }
  if (callee === "umin") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      "min",
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (callee === "copysign") {
    const [magnitudeArg, signArg] = expression.args;
    if (!magnitudeArg || !signArg) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const magnitude = createTypedWgslCall("abs", [emitSemanticExpressionAs(magnitudeArg, ir, names, "f32", options, textureSpecializations)], "f32", expression.span);
    const sign = emitSemanticExpressionAs(signArg, ir, names, "f32", options, textureSpecializations);
    const negative = emitTypedWgslBinary(
      "!=",
      emitTypedWgslBinary("&", createTypedWgslBitcast("u32", sign, expression.span), createTypedWgslLiteral("0x80000000u", "u32", expression.span), expression.span),
      createTypedWgslZero("u32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(magnitude, emitTypedWgslUnary("-", magnitude, expression.span), negative, expression.span);
  }
  if (callee === "vadd2") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_vadd2_u32",
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (callee === "vsub2") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_vsub2_u32",
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if ([
    "vaddss2", "vsubss2", "vaddus2", "vsubus2", "vabsdiffu2", "vavgu2", "vminu2", "vmaxu2", "vmins2", "vmaxs2",
    "vadd4", "vsub4", "vaddss4", "vsubss4", "vaddus4", "vsubus4", "vabsdiffu4", "vavgu4", "vminu4", "vmaxu4", "vmins4", "vmaxs4",
  ].includes(callee)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      `bg_semantic_${callee}_u32`,
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (callee === "umul") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedWgslBinary("*", emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations), expression.span);
  }
  if (callee === "umad" || callee === "imad") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const type = callee === "umad" ? "u32" : "i32";
    return emitTypedWgslBinary(
      "+",
      emitTypedWgslBinary("*", emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations), expression.span),
      emitSemanticExpressionAs(addend, ir, names, type, options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "add" || callee === "sub" || callee === "mul") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedWgslBinary(
      callee === "add" ? "+" : callee === "sub" ? "-" : "*",
      emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "sad" || callee === "usad") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const type = callee === "sad" ? "i32" : "u32";
    const lhs = emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations);
    const difference = callee === "sad"
      ? createTypedWgslCall("abs", [emitTypedWgslBinary("-", lhs, rhs, expression.span)], "i32", expression.span)
      : emitTypedWgslBinary("-", createTypedWgslCall("max", [lhs, rhs], "u32", expression.span), createTypedWgslCall("min", [lhs, rhs], "u32", expression.span), expression.span);
    return emitTypedWgslBinary("+", difference, emitSemanticExpressionAs(addend, ir, names, type, options, textureSpecializations), expression.span);
  }
  if (callee === "modf_intpart") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const nonFinite = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("!=", value, value, expression.span),
      emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(createTypedWgslCall("trunc", [value], "f32", expression.span), value, nonFinite, expression.span);
  }
  if (callee === "modf_fraction") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const nan = emitTypedWgslBinary("!=", value, value, expression.span);
    const infinite = emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span);
    const signedZero = emitTypedWgslSelect(createTypedWgslZero("f32", expression.span), createTypedWgslLiteral("-0.0", "f32", expression.span), emitTypedWgslBinary("<", value, createTypedWgslZero("f32", expression.span), expression.span), expression.span);
    const finite = emitTypedWgslBinary("-", value, createTypedWgslCall("trunc", [value], "f32", expression.span), expression.span);
    return emitTypedWgslSelect(emitTypedWgslSelect(finite, signedZero, infinite, expression.span), value, nan, expression.span);
  }
  if (callee === "remquo_quotient") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const x = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const y = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const ratio = emitTypedWgslBinary("/", x, y, expression.span);
    const baseFloat = createTypedWgslCall("floor", [ratio], "f32", expression.span);
    const base = convertTypedWgslExpression(baseFloat, "i32", true);
    const next = emitTypedWgslBinary("+", base, createTypedWgslLiteral("1", "i32", expression.span), expression.span);
    const diff = emitTypedWgslBinary("-", ratio, baseFloat, expression.span);
    const aboveHalf = emitTypedWgslSelect(base, next, emitTypedWgslBinary(">", diff, createTypedWgslLiteral("0.5", "f32", expression.span), expression.span), expression.span);
    const odd = emitTypedWgslBinary("!=", emitTypedWgslBinary("%", base, createTypedWgslLiteral("2", "i32", expression.span), expression.span), createTypedWgslZero("i32", expression.span), expression.span);
    const tie = emitTypedWgslSelect(base, next, odd, expression.span);
    return emitTypedWgslSelect(aboveHalf, tie, emitTypedWgslBinary("==", diff, createTypedWgslLiteral("0.5", "f32", expression.span), expression.span), expression.span);
  }
  if (callee === "remquo_remainder") {
    const [left, right] = expression.args;
    if (!left || !right || expression.callee.kind !== "symbol") throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const quotient = emitSemanticTypedCustomMathCall(
      { ...expression, callee: { ...expression.callee, name: "__bg_remquo_quotient" }, valueType: "int" },
      ir,
      names,
      options,
      textureSpecializations,
    );
    if (!quotient) return undefined;
    const x = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const y = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    return emitTypedWgslBinary("-", x, emitTypedWgslBinary("*", convertTypedWgslExpression(quotient, "f32", true), y, expression.span), expression.span);
  }
  if (callee === "signbit") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const sign = emitTypedWgslBinary("!=", emitTypedWgslBinary("&", createTypedWgslBitcast("u32", value, expression.span), createTypedWgslLiteral("0x80000000u", "u32", expression.span), expression.span), createTypedWgslZero("u32", expression.span), expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), sign, expression.span);
  }
  if (callee === "abs") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const type = semanticExpressionWgslType(expression, ir);
    if (type !== "i32" && type !== "f32") return undefined;
    return createTypedWgslCall("abs", [emitSemanticExpressionAs(first, ir, names, type, options, textureSpecializations)], type, expression.span);
  }
  if (callee === "norm" || callee === "rnorm") {
    if (expression.args.length < 2) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    const values = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations));
    const sum = values
      .map((value) => emitTypedWgslBinary("*", value, value, expression.span))
      .reduce((left, right) => emitTypedWgslBinary("+", left, right, expression.span));
    const norm = createTypedWgslCall("sqrt", [sum], "f32", expression.span);
    return callee === "rnorm" ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), norm, expression.span) : norm;
  }
  if (callee === "mul24" || callee === "umul24") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const type = callee === "mul24" ? "i32" : "u32";
    return emitTypedWgslBinary(
      "*",
      emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "mulhi" || callee === "umulhi") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const type = callee === "mulhi" ? "i32" : "u32";
    return createTypedWgslCall(
      callee === "mulhi" ? "bg_semantic_mulhi_i32" : "bg_semantic_umulhi_u32",
      [emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations)],
      type,
      expression.span,
    );
  }
  if (callee === "byte_perm") {
    const [left, right, selector] = expression.args;
    if (!left || !right || !selector) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_byte_perm_u32",
      [
        emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(selector, ir, names, "u32", options, textureSpecializations),
      ],
      "u32",
      expression.span,
    );
  }
  if (callee === "funnelshift_l" || callee === "funnelshift_lc" || callee === "funnelshift_r" || callee === "funnelshift_rc") {
    const [low, high, shift] = expression.args;
    if (!low || !high || !shift) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    return createTypedWgslCall(
      `bg_semantic_${callee}_u32`,
      [low, high, shift].map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)),
      "u32",
      expression.span,
    );
  }
  if (callee === "rhadd" || callee === "hadd" || callee === "uhadd" || callee === "urhadd") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const type = callee === "uhadd" || callee === "urhadd" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations);
    const xor = emitTypedWgslBinary("^", lhs, rhs, expression.span);
    const half = emitTypedWgslBinary(">>", xor, createTypedWgslLiteral("1u", "u32", expression.span), expression.span);
    const base = callee === "rhadd"
      ? emitTypedWgslBinary("-", emitTypedWgslBinary("|", lhs, rhs, expression.span), half, expression.span)
      : emitTypedWgslBinary("+", emitTypedWgslBinary("&", lhs, rhs, expression.span), half, expression.span);
    return callee === "urhadd"
      ? emitTypedWgslBinary("+", base, emitTypedWgslBinary("&", xor, createTypedWgslLiteral("1u", "u32", expression.span), expression.span), expression.span)
      : base;
  }
  if (callee === "reciprocal" || callee === "cbrt" || callee === "rcbrt") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    if (callee === "reciprocal") return emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span);
    const magnitude = createTypedWgslCall(
      "pow",
      [createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("0.3333333333333333", "f32", expression.span)],
      "f32",
      expression.span,
    );
    const signBits = emitTypedWgslBinary(
      "&",
      createTypedWgslBitcast("u32", value, expression.span),
      createTypedWgslLiteral("0x80000000u", "u32", expression.span),
      expression.span,
    );
    const signed = emitTypedWgslSelect(
      magnitude,
      emitTypedWgslUnary("-", magnitude, expression.span),
      emitTypedWgslBinary("!=", signBits, createTypedWgslZero("u32", expression.span), expression.span),
      expression.span,
    );
    return callee === "rcbrt"
      ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), signed, expression.span)
      : signed;
  }
  if (callee === "builtin_inf") return createTypedWgslCall("bg_f32_inf", [], "f32", expression.span);
  if (callee === "erf" || callee === "erfinv") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      callee === "erf" ? "bg_semantic_erf_f32" : "bg_semantic_erfinv_f32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (callee === "round_even") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_round_even_f32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (callee === "round_away") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const magnitude = createTypedWgslCall("abs", [value], "f32", expression.span);
    const rounded = createTypedWgslCall(
      "floor",
      [emitTypedWgslBinary("+", magnitude, createTypedWgslLiteral("0.5", "f32", expression.span), expression.span)],
      "f32",
      expression.span,
    );
    return emitTypedWgslSelect(
      rounded,
      emitTypedWgslUnary("-", rounded, expression.span),
      emitTypedWgslBinary("<", value, createTypedWgslZero("f32", expression.span), expression.span),
      expression.span,
    );
  }
  if (callee === "saturate") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "clamp",
      [
        emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations),
        createTypedWgslZero("f32", expression.span),
        createTypedWgslLiteral("1.0", "f32", expression.span),
      ],
      "f32",
      expression.span,
    );
  }
  if (callee === "div_ceil") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const scalar = semanticExpressionWgslScalar(left) === "u32" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, scalar, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, scalar, options, textureSpecializations);
    const one = createTypedWgslLiteral(scalar === "u32" ? "1u" : "1", scalar, expression.span);
    const numerator = emitTypedWgslBinary("-", emitTypedWgslBinary("+", lhs, rhs, expression.span), one, expression.span);
    return emitTypedWgslBinary("/", numerator, rhs, expression.span);
  }
  if (callee === "clz" || callee === "clzll") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const count = createTypedWgslCall(
      "countLeadingZeros",
      [emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
    const converted = convertTypedWgslExpression(count, "i32", true);
    return callee === "clzll"
      ? emitTypedWgslBinary("+", converted, createTypedWgslLiteral("32", "i32", expression.span), expression.span)
      : converted;
  }
  if (callee === "ffs" || callee === "popc" || callee === "brev") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
    if (callee === "brev") return createTypedWgslCall("reverseBits", [value], "u32", expression.span);
    const count = createTypedWgslCall(
      callee === "ffs" ? "countTrailingZeros" : "countOneBits",
      [value],
      "u32",
      expression.span,
    );
    if (callee === "popc") return convertTypedWgslExpression(count, "i32", true);
    const oneBased = emitTypedWgslBinary(
      "+",
      convertTypedWgslExpression(count, "i32", true),
      createTypedWgslLiteral("1", "i32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(
      createTypedWgslZero("i32", expression.span),
      oneBased,
      emitTypedWgslBinary("!=", value, createTypedWgslZero("u32", expression.span), expression.span),
      expression.span,
    );
  }
  if (callee === "usad4") {
    const [left, right, addend] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_usad4_u32",
      [
        emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
        addend
          ? emitSemanticExpressionAs(addend, ir, names, "u32", options, textureSpecializations)
          : createTypedWgslZero("u32", expression.span),
      ],
      "u32",
      expression.span,
    );
  }
  if (callee === "divide" || callee === "remainder" || callee === "nextafter" || callee === "hypot" || callee === "rhypot" || callee === "fmod" || callee === "fdim") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    if (callee === "divide") return emitTypedWgslBinary("/", lhs, rhs, expression.span);
    if (callee === "fmod") {
      const quotient = createTypedWgslCall("trunc", [emitTypedWgslBinary("/", lhs, rhs, expression.span)], "f32", expression.span);
      return emitTypedWgslBinary("-", lhs, emitTypedWgslBinary("*", quotient, rhs, expression.span), expression.span);
    }
    if (callee === "fdim") return createTypedWgslCall("max", [emitTypedWgslBinary("-", lhs, rhs, expression.span), createTypedWgslZero("f32", expression.span)], "f32", expression.span);
    if (callee === "remainder" || callee === "nextafter") {
      return createTypedWgslCall(
        callee === "remainder" ? "bg_semantic_remainder_f32" : "bg_semantic_nextafter_f32",
        [lhs, rhs],
        "f32",
        expression.span,
      );
    }
    const norm = createTypedWgslCall(
      "sqrt",
      [emitTypedWgslBinary(
        "+",
        emitTypedWgslBinary("*", lhs, lhs, expression.span),
        emitTypedWgslBinary("*", rhs, rhs, expression.span),
        expression.span,
      )],
      "f32",
      expression.span,
    );
    return callee === "rhypot"
      ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), norm, expression.span)
      : norm;
  }
  if (callee === "ldexp") {
    const [value, exponent] = expression.args;
    if (!value || !exponent) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const scale = createTypedWgslCall(
      "exp2",
      [convertTypedWgslExpression(emitSemanticExpressionAs(exponent, ir, names, "i32", options, textureSpecializations), "f32", true)],
      "f32",
      expression.span,
    );
    return emitTypedWgslBinary("*", emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), scale, expression.span);
  }
  if (callee === "exp10" || callee === "expm1" || callee === "erfc" || callee === "erfcx" || callee === "erfcinv" || callee === "sinpi" || callee === "cospi" || callee === "sinh" || callee === "cosh" || callee === "tanh" || callee === "tgamma" || callee === "lgamma" || callee === "normcdf" || callee === "normcdfinv" || callee === "log10" || callee === "log1p" || callee === "asinh" || callee === "acosh" || callee === "atanh") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    if (callee === "exp10") return createTypedWgslCall("pow", [createTypedWgslLiteral("10.0", "f32", expression.span), value], "f32", expression.span);
    if (callee === "expm1") return emitTypedWgslBinary("-", createTypedWgslCall("exp", [value], "f32", expression.span), createTypedWgslLiteral("1.0", "f32", expression.span), expression.span);
    if (callee === "tgamma" || callee === "lgamma" || callee === "normcdfinv") {
      const helper = callee === "tgamma" ? "bg_semantic_tgamma_f32" : callee === "lgamma" ? "bg_semantic_lgamma_f32" : "bg_semantic_normcdfinv_f32";
      return createTypedWgslCall(helper, [value], "f32", expression.span);
    }
    if (callee === "normcdf") {
      const scaled = emitTypedWgslBinary("*", value, createTypedWgslLiteral("0.7071067811865476", "f32", expression.span), expression.span);
      const shifted = emitTypedWgslBinary("+", createTypedWgslLiteral("1.0", "f32", expression.span), createTypedWgslCall("bg_semantic_erf_f32", [scaled], "f32", expression.span), expression.span);
      return emitTypedWgslBinary("*", createTypedWgslLiteral("0.5", "f32", expression.span), shifted, expression.span);
    }
    if (callee === "log10") {
      return emitTypedWgslBinary("/", createTypedWgslCall("log", [value], "f32", expression.span), createTypedWgslLiteral("2.302585092994046", "f32", expression.span), expression.span);
    }
    if (callee === "log1p") return createTypedWgslCall("log", [emitTypedWgslBinary("+", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span)], "f32", expression.span);
    if (callee === "asinh") {
      const square = emitTypedWgslBinary("*", value, value, expression.span);
      const root = createTypedWgslCall("sqrt", [emitTypedWgslBinary("+", square, createTypedWgslLiteral("1.0", "f32", expression.span), expression.span)], "f32", expression.span);
      return createTypedWgslCall("log", [emitTypedWgslBinary("+", value, root, expression.span)], "f32", expression.span);
    }
    if (callee === "acosh") {
      const square = emitTypedWgslBinary("*", value, value, expression.span);
      const root = createTypedWgslCall("sqrt", [emitTypedWgslBinary("-", square, createTypedWgslLiteral("1.0", "f32", expression.span), expression.span)], "f32", expression.span);
      return createTypedWgslCall("log", [emitTypedWgslBinary("+", value, root, expression.span)], "f32", expression.span);
    }
    if (callee === "atanh") {
      const ratio = emitTypedWgslBinary(
        "/",
        emitTypedWgslBinary("+", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span),
        emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span),
        expression.span,
      );
      return emitTypedWgslBinary("*", createTypedWgslLiteral("0.5", "f32", expression.span), createTypedWgslCall("log", [ratio], "f32", expression.span), expression.span);
    }
    if (callee === "erfc") {
      return emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), createTypedWgslCall("bg_semantic_erf_f32", [value], "f32", expression.span), expression.span);
    }
    if (callee === "erfcx") {
      const complement = emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), createTypedWgslCall("bg_semantic_erf_f32", [value], "f32", expression.span), expression.span);
      return emitTypedWgslBinary("*", createTypedWgslCall("exp", [emitTypedWgslBinary("*", value, value, expression.span)], "f32", expression.span), complement, expression.span);
    }
    if (callee === "erfcinv") {
      return createTypedWgslCall(
        "bg_semantic_erfinv_f32",
        [emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span)],
        "f32",
        expression.span,
      );
    }
    if (callee === "sinpi" || callee === "cospi") {
      return createTypedWgslCall(
        callee === "sinpi" ? "sin" : "cos",
        [emitTypedWgslBinary("*", createTypedWgslLiteral("3.141592653589793", "f32", expression.span), value, expression.span)],
        "f32",
        expression.span,
      );
    }
    if (callee === "tanh") return createTypedWgslCall("tanh", [value], "f32", expression.span);
    const positive = createTypedWgslCall("exp", [value], "f32", expression.span);
    const negative = createTypedWgslCall("exp", [emitTypedWgslUnary("-", value, expression.span)], "f32", expression.span);
    return emitTypedWgslBinary(
      "*",
      createTypedWgslLiteral("0.5", "f32", expression.span),
      emitTypedWgslBinary(callee === "cosh" ? "+" : "-", positive, negative, expression.span),
      expression.span,
    );
  }
  if (callee === "isnan" || callee === "isinf" || callee === "isfinite" || callee === "isnormal") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const condition = callee === "isnan"
      ? emitTypedWgslBinary("!=", value, value, expression.span)
      : callee === "isinf" ? emitTypedWgslBinary(
          ">",
          createTypedWgslCall("abs", [value], "f32", expression.span),
          createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span),
          expression.span,
        ) : callee === "isfinite" ? emitTypedWgslBinary(
          "&&",
          emitTypedWgslBinary("<=", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
          emitTypedWgslBinary("==", value, value, expression.span),
          expression.span,
        ) : emitTypedWgslBinary(
          "&&",
          emitTypedWgslBinary(">=", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("1.1754943508222875e-38", "f32", expression.span), expression.span),
          emitTypedWgslBinary("<=", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
          expression.span,
        );
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), condition, expression.span);
  }
  if (callee === "isunordered") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const unordered = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("!=", lhs, lhs, expression.span),
      emitTypedWgslBinary("!=", rhs, rhs, expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), unordered, expression.span);
  }
  if (callee === "islessgreater") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const ordered = emitTypedWgslBinary("&&", emitTypedWgslBinary("==", lhs, lhs, expression.span), emitTypedWgslBinary("==", rhs, rhs, expression.span), expression.span);
    const different = emitTypedWgslBinary("||", emitTypedWgslBinary("<", lhs, rhs, expression.span), emitTypedWgslBinary(">", lhs, rhs, expression.span), expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), emitTypedWgslBinary("&&", ordered, different, expression.span), expression.span);
  }
  if (callee === "isgreater" || callee === "isgreaterequal" || callee === "isless" || callee === "islessequal") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const operator = callee === "isgreater" ? ">" : callee === "isgreaterequal" ? ">=" : callee === "isless" ? "<" : "<=";
    const ordered = emitTypedWgslBinary("&&", emitTypedWgslBinary("==", lhs, lhs, expression.span), emitTypedWgslBinary("==", rhs, rhs, expression.span), expression.span);
    const comparison = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), emitTypedWgslBinary("&&", ordered, comparison, expression.span), expression.span);
  }
  if (callee === "lerp") {
    if (semanticWgslVectorLerpCallSupported(expression, ir)) return undefined;
    const [start, end, amount] = expression.args;
    if (!start || !end || !amount) throw semanticWgslError("lerp expects three operands", expression.span);
    const from = emitSemanticExpressionAs(start, ir, names, "f32", options, textureSpecializations);
    const to = emitSemanticExpressionAs(end, ir, names, "f32", options, textureSpecializations);
    return createTypedWgslCall(
      "fma",
      [
        emitSemanticExpressionAs(amount, ir, names, "f32", options, textureSpecializations),
        emitTypedWgslBinary("-", to, from, expression.span),
        from,
      ],
      "f32",
      expression.span,
    );
  }
  if (callee === "frexp_exponent") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const zero = createTypedWgslZero("f32", expression.span);
    const nonFiniteOrZero = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("||", emitTypedWgslBinary("==", value, zero, expression.span), emitTypedWgslBinary("!=", value, value, expression.span), expression.span),
      emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
      expression.span,
    );
    const exponent = emitTypedWgslBinary(
      "+",
      convertTypedWgslExpression(createTypedWgslCall("floor", [createTypedWgslCall("log2", [createTypedWgslCall("abs", [value], "f32", expression.span)], "f32", expression.span)], "f32", expression.span), "i32", true),
      createTypedWgslLiteral("1", "i32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(exponent, createTypedWgslZero("i32", expression.span), nonFiniteOrZero, expression.span);
  }
  if (callee === "frexp_mantissa") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const zero = createTypedWgslZero("f32", expression.span);
    const nonFiniteOrZero = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("||", emitTypedWgslBinary("==", value, zero, expression.span), emitTypedWgslBinary("!=", value, value, expression.span), expression.span),
      emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
      expression.span,
    );
    const exponent = emitTypedWgslBinary(
      "+",
      convertTypedWgslExpression(createTypedWgslCall("floor", [createTypedWgslCall("log2", [createTypedWgslCall("abs", [value], "f32", expression.span)], "f32", expression.span)], "f32", expression.span), "i32", true),
      createTypedWgslLiteral("1", "i32", expression.span),
      expression.span,
    );
    const mantissa = emitTypedWgslBinary(
      "/",
      value,
      createTypedWgslCall("exp2", [convertTypedWgslExpression(exponent, "f32", true)], "f32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(mantissa, value, nonFiniteOrZero, expression.span);
  }
  if (callee === "logb") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_logb_f32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (callee === "ilogb") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_ilogb_i32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "i32",
      expression.span,
    );
  }
  if (callee === "float_to_int_rn" || callee === "float_to_int_round") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const rounded = callee === "float_to_int_rn"
      ? createTypedWgslCall("bg_semantic_round_even_f32", [value], "f32", expression.span)
      : emitSemanticTypedCustomMathCall({ ...expression, callee: { ...expression.callee, name: "roundf" }, valueType: "float" }, ir, names, options, textureSpecializations);
    if (!rounded) throw semanticWgslError(`cannot lower '${expression.callee.name}' rounding`, expression.span);
    return convertTypedWgslExpression(rounded, "i32", true);
  }
  return undefined;
}

function emitSemanticTypedClock(span: SourceSpan): TypedWgslExpression {
  const workgroupId = createTypedWgslIdentifier("workgroup_id", "vec3<u32>", span);
  const localId = createTypedWgslIdentifier("local_id", "vec3<u32>", span);
  const member = (object: TypedWgslExpression, field: "x" | "y" | "z"): TypedWgslExpression =>
    createTypedWgslMemberAccess(object, field, "u32", span);
  const term = (object: TypedWgslExpression, field: "x" | "y" | "z", factor: number): TypedWgslExpression =>
    factor === 1
      ? member(object, field)
      : emitTypedWgslBinary("*", member(object, field), createTypedWgslLiteral(`${factor}u`, "u32", span), span);
  return [
    term(workgroupId, "x", 104729),
    term(workgroupId, "y", 1009),
    term(workgroupId, "z", 97),
    term(localId, "x", 1),
    term(localId, "y", 31),
    term(localId, "z", 7),
  ].reduce((left, right) => emitTypedWgslBinary("+", left, right, span));
}

function emitSemanticTypedConversionIntrinsic(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  const value = expression.args[0];
  if (!callee || !value) return undefined;
  if (callee === "float_as_int" || callee === "float_as_uint") {
    return createTypedWgslBitcast(
      callee === "float_as_int" ? "i32" : "u32",
      emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "uint_as_float" || callee === "int_as_float") {
    return createTypedWgslBitcast(
      "f32",
      emitSemanticExpressionAs(value, ir, names, callee === "uint_as_float" ? "u32" : "i32", options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "half_to_float") {
    return convertTypedWgslExpression(
      emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations),
      "f32",
      true,
    );
  }
  if (callee === "bf16_to_float") {
    return emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
  }
  if (callee === "short_as_bf16" || callee === "ushort_as_bf16") {
    const source = emitSemanticExpressionAs(value, ir, names, callee === "short_as_bf16" ? "i32" : "u32", options, textureSpecializations);
    const bits = callee === "short_as_bf16" ? convertTypedWgslExpression(source, "u32", true) : source;
    return createTypedWgslBitcast(
      "f32",
      emitTypedWgslBinary(
        "<<",
        emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span),
        createTypedWgslLiteral("16u", "u32", expression.span),
        expression.span,
      ),
      expression.span,
    );
  }
  if (callee === "bf16_as_ushort" || callee === "bf16_as_short") {
    const bits = emitTypedWgslBinary(
      "&",
      emitTypedWgslBinary(
        ">>",
        createTypedWgslBitcast("u32", emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), expression.span),
        createTypedWgslLiteral("16u", "u32", expression.span),
        expression.span,
      ),
      createTypedWgslLiteral("0xffffu", "u32", expression.span),
      expression.span,
    );
    if (callee === "bf16_as_ushort") return bits;
    const signedBits = emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
    return emitTypedWgslBinary(
      ">>",
      createTypedWgslBitcast("i32", signedBits, expression.span),
      createTypedWgslLiteral("16u", "u32", expression.span),
      expression.span,
    );
  }
  if (callee === "int_to_float" || callee === "uint_to_float") {
    return convertTypedWgslExpression(
      emitSemanticExpressionAs(value, ir, names, callee === "int_to_float" ? "i32" : "u32", options, textureSpecializations),
      "f32",
      true,
    );
  }
  if (callee.startsWith("float_to_int_")) {
    const source = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    const mode = callee.slice("float_to_int_".length);
    const rounded = mode === "rn"
      ? createTypedWgslCall("bg_semantic_round_even_f32", [source], "f32", expression.span)
      : createTypedWgslCall(mode === "ru" ? "ceil" : mode === "rd" ? "floor" : "trunc", [source], "f32", expression.span);
    return convertTypedWgslExpression(rounded, "i32", true);
  }
  {
    const numeric = /^(float|half|bf16)_to_(int|uint|short|ushort|char|uchar)_(rn|rz|ru|rd)$/u.exec(callee);
    if (numeric) {
      const [, sourceKind, targetKind, mode] = numeric;
      const source = sourceKind === "half"
        ? convertTypedWgslExpression(
            emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations),
            "f32",
            true,
          )
        : convertTypedWgslExpression(
            emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
            "f32",
            sourceKind === "bf16",
          );
      const rounded = mode === "rn"
        ? createTypedWgslCall("bg_semantic_round_even_f32", [source], "f32", expression.span)
        : createTypedWgslCall(mode === "ru" ? "ceil" : mode === "rd" ? "floor" : "trunc", [source], "f32", expression.span);
      const unsigned = targetKind === "uint" || targetKind === "ushort" || targetKind === "uchar";
      const legalized = unsigned
        ? createTypedWgslCall("max", [rounded, createTypedWgslZero("f32", expression.span)], "f32", expression.span)
        : rounded;
      const converted = convertTypedWgslExpression(legalized, unsigned ? "u32" : "i32", true);
      if (targetKind === "ushort" || targetKind === "uchar") {
        return emitTypedWgslBinary(
          "&",
          converted,
          createTypedWgslLiteral(targetKind === "ushort" ? "0xffffu" : "0xffu", "u32", expression.span),
          expression.span,
        );
      }
      if (targetKind === "short" || targetKind === "char") {
        const shift = targetKind === "short" ? 16 : 24;
        const bits = emitTypedWgslBinary(
          "<<",
          convertTypedWgslExpression(converted, "u32", true),
          createTypedWgslLiteral(`${shift}u`, "u32", expression.span),
          expression.span,
        );
        return emitTypedWgslBinary(
          ">>",
          createTypedWgslBitcast("i32", bits, expression.span),
          createTypedWgslLiteral(`${shift}u`, "u32", expression.span),
          expression.span,
        );
      }
      return converted;
    }
  }
  if (callee.startsWith("float_to_half_") || callee.startsWith("float_to_bf16_")) {
    const mode = createTypedWgslLiteral(halfConversionModeLiteral(callee), "u32", expression.span);
    const bits = createTypedWgslCall(
      callee.startsWith("float_to_half_") ? "bg_f32_to_f16_bits_mode" : "bg_f32_to_bf16_bits_mode",
      [emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), mode],
      "u32",
      expression.span,
    );
    if (callee.startsWith("float_to_bf16_")) {
      return createTypedWgslBitcast(
        "f32",
        emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span),
        expression.span,
      );
    }
    const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", expression.span);
    return convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true);
  }
  {
    const packed = /^(int|uint|short|ushort)_to_(half|bf16)(?:_(rn|rz|ru|rd))?$/u.exec(callee);
    if (packed) {
      const [, sourceKind, targetKind, mode = "rn"] = packed;
      const sourceScalar = sourceKind === "int" || sourceKind === "short" ? "i32" : "u32";
      let source = emitSemanticExpressionAs(value, ir, names, sourceScalar, options, textureSpecializations);
      if (sourceKind === "short") {
        source = createTypedWgslCall("bg_i16_to_f32", [source], "f32", expression.span);
      } else {
        if (sourceKind === "ushort") {
          source = emitTypedWgslBinary("&", source, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span);
        }
        source = convertTypedWgslExpression(source, "f32", true);
      }
      const modeLiteral = createTypedWgslLiteral(
        mode === "rn" ? "0u" : mode === "rz" ? "1u" : mode === "ru" ? "2u" : "3u",
        "u32",
        expression.span,
      );
      const bits = createTypedWgslCall(
        targetKind === "half" ? "bg_f32_to_f16_bits_mode" : "bg_f32_to_bf16_bits_mode",
        [source, modeLiteral],
        "u32",
        expression.span,
      );
      if (targetKind === "bf16") {
        return createTypedWgslBitcast(
          "f32",
          emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span),
          expression.span,
        );
      }
      const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", expression.span);
      return convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true);
    }
  }
  if (callee === "to_half") {
    const bits = createTypedWgslCall(
      "bg_f32_to_f16_bits_mode",
      [
        emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
        createTypedWgslZero("u32", expression.span),
      ],
      "u32",
      expression.span,
    );
    const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", expression.span);
    return convertTypedWgslExpression(
      createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span),
      "f16",
      true,
    );
  }
  if (callee === "to_bf16") {
    const bits = createTypedWgslCall(
      "bg_f32_to_bf16_bits_mode",
      [
        emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
        createTypedWgslZero("u32", expression.span),
      ],
      "u32",
      expression.span,
    );
    const shifted = emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
    return createTypedWgslBitcast("f32", shifted, expression.span);
  }
  if (callee === "double_to_bf16") {
    return roundTypedBf16(emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), expression.span);
  }
  return undefined;
}

function emitSemanticTypedMinMaxCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  if (callee !== "min" && callee !== "max") return undefined;
  const vectorType = semanticVectorMinMaxCallValueType(expression.callee.name, expression.args);
  if (vectorType !== undefined) {
    return createTypedWgslCall(
      callee,
      expression.args.map((arg) => emitSemanticVectorOperandExpression(arg, vectorType, ir, names, options, textureSpecializations)),
      wgslValueType(vectorType),
      expression.span,
    );
  }
  const scalar = semanticMathCallOperandType(expression.args);
  return createTypedWgslCall(
    callee,
    expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, scalar, options, textureSpecializations)),
    scalar,
    expression.span,
  );
}

function emitSemanticTypedComplexCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol" || !isCudaComplexCallName(expression.callee.name)) return undefined;
  const name = expression.callee.name;
  const operand = (index: number): TypedWgslExpression => {
    const arg = expression.args[index];
    if (!arg) throw semanticWgslError(`${name} expects complex operand ${index + 1}`, expression.span);
    const value = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
    if (value.type !== "vec2<f32>") throw semanticWgslError(`${name} expects complex64 operand`, arg.span);
    return value;
  };
  const lane = (value: TypedWgslExpression, field: "x" | "y"): TypedWgslExpression =>
    createTypedWgslMemberAccess(value, field, "f32", expression.span);
  if (isCudaComplexConstructorCallName(name)) {
    if (expression.args.length !== 2) throw semanticWgslError(`${name} expects two scalar operands`, expression.span);
    return createTypedWgslConstructor("vec2<f32>", expression.args.map((arg) =>
      emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations)
    ), expression.span);
  }
  const left = operand(0);
  if (isCudaComplexScalarCallName(name)) {
    if (name === "cuCrealf" || name === "cuCreal") return lane(left, "x");
    if (name === "cuCimagf" || name === "cuCimag") return lane(left, "y");
    return createTypedWgslCall("bg_cuCabsf", [left], "f32", expression.span);
  }
  if (name === "cuConjf" || name === "cuConj") {
    return createTypedWgslConstructor("vec2<f32>", [lane(left, "x"), emitTypedWgslUnary("-", lane(left, "y"), expression.span)], expression.span);
  }
  const right = operand(1);
  if (name === "cuCaddf" || name === "cuCadd" || name === "cuCsubf" || name === "cuCsub") {
    return emitTypedWgslBinary(name.includes("add") ? "+" : "-", left, right, expression.span);
  }
  if (name === "cuCdivf" || name === "cuCdiv") {
    return createTypedWgslCall("bg_cuCdivf", [left, right], "vec2<f32>", expression.span);
  }
  const product = createTypedWgslConstructor("vec2<f32>", [
    emitTypedWgslBinary("-", emitTypedWgslBinary("*", lane(left, "x"), lane(right, "x"), expression.span), emitTypedWgslBinary("*", lane(left, "y"), lane(right, "y"), expression.span), expression.span),
    emitTypedWgslBinary("+", emitTypedWgslBinary("*", lane(left, "x"), lane(right, "y"), expression.span), emitTypedWgslBinary("*", lane(left, "y"), lane(right, "x"), expression.span), expression.span),
  ], expression.span);
  if (name === "cuCfmaf" || name === "cuCfma") return emitTypedWgslBinary("+", product, operand(2), expression.span);
  return product;
}

function emitSemanticTypedVectorMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (semanticWgslVectorLerpCallSupported(expression, ir)) {
    const [left, right, amount] = expression.args;
    if (!left || !right || !amount) return undefined;
    const valueType = semanticExpressionVectorValueType(left, ir.functions);
    if (!isSemanticFloatVectorType(valueType)) return undefined;
    const typedValueType = requireSemanticValueType(valueType, "vector lerp", expression.span);
    const start = emitSemanticExpression(left, ir, names, options, textureSpecializations);
    const end = emitSemanticExpression(right, ir, names, options, textureSpecializations);
    const factor = emitSemanticVectorOperandExpression(amount, typedValueType, ir, names, options, textureSpecializations);
    return createTypedWgslCall(
      "fma",
      [factor, emitTypedWgslBinary("-", end, start, expression.span), start],
      wgslValueType(typedValueType),
      expression.span,
    );
  }
  if (expression.callee.kind !== "symbol" || !semanticVectorMathCallSupported(expression.callee.name, expression.args)) return undefined;
  if (expression.callee.name !== "normalize" && expression.callee.name !== "length" && expression.callee.name !== "dot" && expression.callee.name !== "cross") return undefined;
  return createTypedWgslCall(
    expression.callee.name,
    expression.args.map((arg) => emitSemanticExpression(arg, ir, names, options, textureSpecializations)),
    semanticExpressionWgslType(expression, ir),
    expression.span,
  );
}

function emitSemanticDirectStorageVectorReadExpression(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  const valueType = ref === undefined ? undefined : semanticStorageVectorType(ref.valueType);
  if (!ref || !valueType || ref.addressSpace !== "storage" && ref.addressSpace !== "device-global" && ref.addressSpace !== "constant" || ref.fields.length !== 0 || ref.indices.length !== 1 || ref.packedByteLanes !== undefined) return undefined;
  const targetType = wgslValueType(valueType);
  if (!isWgslVectorType(targetType)) return undefined;
  const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
  if (pointerParam) {
    return createTypedWgslCall(
      semanticPointerReadHelperName(valueType),
      [
        createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(ref.base), names), "u32", ref.span),
        emitTypedFlatStorageVectorBaseIndex(ref, ir, names, options),
      ],
      targetType,
      expression.span,
    );
  }
  const root = ir.params.find((item) => item.name === ref.base) ?? ir.memory.find((item) => item.name === ref.base);
  if (!root) return undefined;
  const laneType = wgslVectorScalar(valueType);
  const laneCount = cudaVectorLaneCount(valueType);
  const base = emitTypedFlatStorageVectorBaseIndex(ref, ir, names, options);
  const atomic = semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base);
  const lanes = Array.from({ length: laneCount }, (_, lane): TypedWgslExpression => {
    const index = emitTypedWgslBinary("+", base, createTypedWgslLiteral(`${lane}u`, "u32", ref.span), ref.span);
    const storageType = atomic ? "u32" : laneType;
    const read = createTypedWgslMemoryRead(nameFor(ref.base, names), index, storageType, atomic, ref.span);
    return legalizeSemanticMemoryReadValue(read, laneType, ref.span);
  });
  return createTypedWgslConstructor(targetType, lanes, expression.span);
}

function emitSemanticStorageVectorScalarReadExpression(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || ref.fields.length !== 0 || ref.indices.length !== 1 || ref.packedByteLanes !== undefined) return undefined;
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "device-global" && ref.addressSpace !== "constant") return undefined;
  const pointer = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
  const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
  const vectorType = semanticStorageVectorType(pointer?.pointerCarrierValueType ?? root?.valueType);
  if (!vectorType || cudaVectorScalarType(vectorType) !== ref.valueType) return undefined;
  const targetType = wgslValueScalar(ref.valueType);
  if (pointer) {
    return createTypedWgslCall(
      semanticPointerReadHelperName(ref.valueType),
      [
        createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(ref.base), names), "u32", ref.span),
        emitTypedFlatStorageIndex(ref, ir, names, options),
      ],
      targetType,
      ref.span,
    );
  }
  const atomic = ref.addressSpace === "storage"
    ? semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base)
    : ref.addressSpace === "device-global" && semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(ref.base);
  const sourceType = atomic ? wgslAtomicScalar(ref.valueType) : targetType;
  const read = createTypedWgslMemoryRead(nameFor(ref.base, names), emitTypedFlatStorageIndex(ref, ir, names, options), sourceType, atomic, ref.span);
  return read.type === targetType ? read : createTypedWgslBitcast(targetType, read, ref.span);
}

function emitSemanticDirectSharedVectorReadExpression(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  const valueType = ref === undefined ? undefined : semanticStorageVectorType(ref.valueType);
  if (!ref || !valueType || ref.addressSpace !== "shared" || ref.fields.length !== 0 || ref.indices.length !== 1 || ref.packedByteLanes !== undefined) return undefined;
  const targetType = wgslValueType(valueType);
  if (!isWgslVectorType(targetType)) return undefined;
  const sharedPointer = semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null);
  const root = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
  const atomic = semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base) || semanticWgslFunctionSharedPointerAtomicParam(ir, ref.base);
  const directVector = semanticStorageVectorType(sharedPointer?.pointerCarrierValueType ?? sharedPointer?.valueType ?? root?.valueType);
  const sourceIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  if (directVector === valueType && !atomic) {
    if (sharedPointer) {
      const index = emitTypedWgslBinary(
        "+",
        createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
        sourceIndex,
        ref.span,
      );
      return createTypedWgslPointerIndexRead(nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names), index, targetType, ref.span);
    }
    return createTypedWgslMemoryRead(nameFor(ref.base, names), sourceIndex, targetType, false, ref.span);
  }
  const laneType = wgslVectorScalar(valueType);
  const laneCount = cudaVectorLaneCount(valueType);
  const scalarLaneBase = ref.pointerBaseIsScalarLane === true ||
    sharedPointer === undefined && root !== undefined && !isCudaVectorType(root.valueType) && cudaVectorScalarType(valueType) === root.valueType;
  const scalarBase = scalarLaneBase
    ? sourceIndex
    : emitTypedWgslBinary("*", sourceIndex, createTypedWgslLiteral(`${laneCount}u`, "u32", ref.span), ref.span);
  const sourceScalar = sharedPointer?.pointerCarrierValueType ?? sharedPointer?.valueType ?? root?.valueType;
  const sourceLaneType = atomic ? wgslAtomicScalar(sourceScalar) : wgslValueScalar(sourceScalar);
  const lanes = Array.from({ length: laneCount }, (_, lane): TypedWgslExpression => {
    const offset = lane === 0 ? scalarBase : emitTypedWgslBinary("+", scalarBase, createTypedWgslLiteral(`${lane}u`, "u32", ref.span), ref.span);
    const index = sharedPointer
      ? emitTypedWgslBinary("+", createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span), offset, ref.span)
      : offset;
    const read = sharedPointer
      ? createTypedWgslPlaceRead(createTypedWgslDereferencedIndexedPlace(nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names), index, sourceLaneType, atomic, "workgroup", ref.span))
      : createTypedWgslMemoryRead(nameFor(ref.base, names), index, sourceLaneType, atomic, ref.span);
    return read.type === laneType ? read : createTypedWgslBitcast(laneType, read, ref.span);
  });
  return createTypedWgslConstructor(targetType, lanes, expression.span);
}

function emitSemanticDirectLocalPointerVectorReadExpression(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  const valueType = ref === undefined ? undefined : semanticStorageVectorType(ref.valueType);
  const pointer = ref === undefined
    ? undefined
    : semanticWgslFunctionLocalPointerParam(ir, ref.base, options.activeFunction ?? null);
  if (!ref || !valueType || !pointer || pointer.dimensions.length === 0 || ref.indices.length !== 1) return undefined;
  const targetType = wgslValueType(valueType);
  if (!isWgslVectorType(targetType)) return undefined;
  const index = emitTypedWgslBinary(
    "+",
    createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
    emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options),
    ref.span,
  );
  return createTypedWgslPointerIndexRead(nameFor(ref.base, names), index, targetType, ref.span);
}

function emitSemanticSharedVectorScalarReadExpression(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || ref.addressSpace !== "shared" || ref.fields.length !== 0 || ref.indices.length !== 1) return undefined;
  const sharedPointer = semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null);
  const root = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
  const vectorType = semanticStorageVectorType(sharedPointer?.pointerCarrierValueType ?? root?.valueType);
  if (!vectorType || cudaVectorScalarType(vectorType) !== ref.valueType || ref.pointerBaseIsScalarLane !== true && !sharedPointer) return undefined;
  const targetType = wgslValueScalar(ref.valueType);
  const laneCount = cudaVectorLaneCount(vectorType);
  const sourceIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  const scalarIndex = sharedPointer
    ? emitTypedWgslBinary("+", createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span), sourceIndex, ref.span)
    : sourceIndex;
  const atomic = semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base) || semanticWgslFunctionSharedPointerAtomicParam(ir, ref.base);
  if (atomic) {
    const read = sharedPointer
      ? createTypedWgslPlaceRead(createTypedWgslDereferencedIndexedPlace(nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names), scalarIndex, wgslAtomicScalar(ref.valueType), true, "workgroup", ref.span))
      : createTypedWgslMemoryRead(nameFor(ref.base, names), scalarIndex, wgslAtomicScalar(ref.valueType), true, ref.span);
    return read.type === targetType ? read : createTypedWgslBitcast(targetType, read, ref.span);
  }
  const vectorIndex = emitTypedWgslBinary("/", scalarIndex, createTypedWgslLiteral(`${laneCount}u`, "u32", ref.span), ref.span);
  const laneIndex = emitTypedWgslBinary("%", scalarIndex, createTypedWgslLiteral(`${laneCount}u`, "u32", ref.span), ref.span);
  const vector = sharedPointer
    ? createTypedWgslPointerIndexRead(nameFor(semanticParamAliasName(ir, sharedPointer) ?? ref.base, names), vectorIndex, wgslValueType(vectorType), ref.span)
    : createTypedWgslMemoryRead(nameFor(ref.base, names), vectorIndex, wgslValueType(vectorType), false, ref.span);
  return createTypedWgslIndexAccess(vector, laneIndex, targetType, ref.span);
}

function emitTypedFlatStorageVectorBaseIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression {
  const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
  const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
  const rootVector = semanticStorageVectorType(ref.containerValueType) ?? semanticStorageVectorType(pointerParam?.valueType) ?? semanticStorageVectorType(root?.valueType);
  const stride = rootVector === undefined ? 1 : cudaVectorLaneCount(rootVector);
  if (pointerParam) {
    const base = createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span);
    const indices = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "u32", options));
    const offset = indices.length === 0
      ? createTypedWgslZero("u32", ref.span)
      : indices.reduce((left, right) => emitTypedWgslBinary("+", left, right, ref.span));
    const scaled = stride === 1 ? offset : emitTypedWgslBinary("*", offset, createTypedWgslLiteral(`${stride}u`, "u32", ref.span), ref.span);
    return emitTypedWgslBinary("+", base, scaled, ref.span);
  }
  const base = emitTypedFlatStorageIndex(ref, ir, names, options);
  return stride === 1 ? base : emitTypedWgslBinary("*", base, createTypedWgslLiteral(`${stride}u`, "u32", ref.span), ref.span);
}

function emitSemanticScalarMemorySymbolExpression(
  expression: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  if (!semanticWgslScalarTypeSupported(expression.valueType)) return undefined;
  const symbol = ir.memory.find((item) => semanticIdsEqual(item.id, expression.id));
  if (!symbol || symbol.dimensions.length !== 0 || isCudaVectorType(symbol.valueType)) return undefined;
  if (expression.addressSpace === "device-global") {
    const atomic = semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(expression.name);
    const storageType = atomic ? wgslAtomicScalar(symbol.valueType) : wgslBindingType(symbol.valueType);
    const read = createTypedWgslMemoryRead(
      nameFor(expression.name, names),
      createTypedWgslZero("u32", expression.span),
      storageType,
      atomic,
      expression.span,
    );
    return legalizeSemanticMemoryReadValue(read, wgslValueType(symbol.valueType), expression.span);
  }
  if (expression.addressSpace !== "shared") return undefined;
  const atomic = semanticAtomicSharedNames(ir.operations, ir.functions).has(expression.name);
  const storageType = atomic ? wgslAtomicScalar(symbol.valueType) : wgslValueType(symbol.valueType);
  const mode = atomic ? "atomic" : options.workgroupUniformExpression ? "workgroup-uniform" : "plain";
  const read = createTypedWgslScalarMemoryRead(nameFor(expression.name, names), storageType, mode, expression.span);
  return legalizeSemanticMemoryReadValue(read, wgslValueType(symbol.valueType), expression.span);
}

function legalizeSemanticMemoryReadValue(
  read: TypedWgslExpression,
  semanticType: TypedWgslExpression["type"],
  span: SourceSpan,
): TypedWgslExpression {
  if (read.type === semanticType) return read;
  if (semanticType === "bool") return emitTypedWgslBinary("!=", read, createTypedWgslZero(read.type, span), span);
  if (semanticType === "f32" && read.type === "u32") return createTypedWgslBitcast("f32", read, span);
  throw semanticWgslError(`memory read cannot convert '${read.type}' to '${semanticType}'`, span);
}

function emitSemanticDirectMemoryReadExpression(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression | undefined {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref || ref.fields.length !== 0 || ref.indices.length === 0 || !semanticWgslScalarTypeSupported(ref.valueType) && ref.valueType !== "uchar") return undefined;
  if (ref.packedByteLanes !== undefined) return undefined;
  if (
    semanticWgslLocalPackedHalf2View(ref, ir) || semanticWgslLocalPackedHalfView(ref, ir) ||
    semanticWgslLocalScalarBitViewRootType(ref, ir) !== undefined || semanticWgslSharedScalarBitViewRootType(ref, ir) !== undefined ||
    semanticWgslLocalVectorBitViewRootType(ref, ir) !== undefined || semanticWgslDirectByteRawView(ref, ir) ||
    semanticWgslPackedSharedByteRoot(ref, ir) || semanticWgslSharedHalfBitView(ref, ir) || semanticWgslSharedVectorScalarView(ref, ir)
  ) return undefined;
  const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
  const sharedPointerParam = semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null);
  const localPointerParam = semanticWgslFunctionLocalPointerParam(ir, ref.base, options.activeFunction ?? null);
  const semanticType = wgslValueType(ref.valueType);
  if (pointerParam) {
    if (ref.indices.length !== 1) return undefined;
    return createTypedWgslCall(
      semanticPointerReadHelperName(ref.valueType ?? "float"),
      [
        createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(ref.base), names), "u32", ref.span),
        emitTypedFlatStorageIndex(ref, ir, names, options),
      ],
      semanticType,
      expression.span,
    );
  }
  if (sharedPointerParam) {
    if (ref.indices.length !== 1) return undefined;
    const atomic = semanticWgslFunctionSharedPointerAtomicParam(ir, ref.base);
    const carrierValueType = sharedPointerParam.pointerCarrierValueType ?? sharedPointerParam.valueType;
    const carrierType = atomic ? wgslAtomicScalar(carrierValueType) : wgslValueType(carrierValueType);
    if (carrierType !== "f16" && carrierType !== "f32" && carrierType !== "i32" && carrierType !== "u32") return undefined;
    const pointerName = nameFor(semanticParamAliasName(ir, sharedPointerParam) ?? ref.base, names);
    const place = sharedPointerParam.dimensions.length === 0
      ? semanticExpressionIsZero(ref.indices[0]!)
        ? createTypedWgslDereferencedPlace(pointerName, carrierType, atomic, "workgroup", ref.span)
        : undefined
      : createTypedWgslDereferencedIndexedPlace(
          pointerName,
          emitTypedWgslBinary(
            "+",
            createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
            emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options),
            ref.span,
          ),
          carrierType,
          atomic,
          "workgroup",
          ref.span,
        );
    if (!place) return undefined;
    const read = createTypedWgslPlaceRead(place);
    if (read.type === semanticType) return read;
    if (semanticType === "bool") return legalizeSemanticMemoryReadValue(read, semanticType, ref.span);
    return createTypedWgslBitcast(semanticType, read, ref.span);
  }
  if (localPointerParam) {
    if (ref.indices.length !== 1) return undefined;
    const carrierType = wgslValueType(localPointerParam.pointerCarrierValueType ?? localPointerParam.valueType);
    if (carrierType !== "f16" && carrierType !== "f32" && carrierType !== "i32" && carrierType !== "u32") return undefined;
    const place = localPointerParam.dimensions.length === 0
      ? semanticExpressionIsZero(ref.indices[0]!)
        ? createTypedWgslDereferencedPlace(nameFor(ref.base, names), carrierType, false, "function", ref.span)
        : undefined
      : createTypedWgslDereferencedIndexedPlace(
          nameFor(ref.base, names),
          emitTypedWgslBinary(
            "+",
            createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
            emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options),
            ref.span,
          ),
          carrierType,
          false,
          "function",
          ref.span,
        );
    if (!place) return undefined;
    const read = createTypedWgslPlaceRead(place);
    if (read.type === semanticType) return read;
    if (semanticType === "bool") return legalizeSemanticMemoryReadValue(read, semanticType, ref.span);
    return createTypedWgslBitcast(semanticType, read, ref.span);
  }
  const root = ir.params.find((item) => item.name === ref.base) ?? ir.memory.find((item) => item.name === ref.base) ?? localArraySymbol(ir, ref.base);
  if (!root || isCudaVectorType(root.valueType)) return undefined;
  if (ref.addressSpace === "local" && root.dimensions.length > 0) {
    const path = ref.indices.length === root.dimensions.length
      ? ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "u32", options))
      : ref.indices.length === 1
        ? semanticTypedLocalArrayPath(emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options), root.dimensions, ref.span)
        : undefined;
    if (!path) return undefined;
    return createTypedWgslMemoryPathRead(
      nameFor(ref.base, names),
      path,
      semanticType,
      expression.span,
    );
  }
  const atomic = ref.addressSpace === "storage"
    ? semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base)
    : ref.addressSpace === "device-global"
      ? semanticAtomicDeviceGlobalNames(ir.operations, ir.functions).has(ref.base)
      : ref.addressSpace === "shared" && semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base);
  const storageType = atomic
    ? wgslAtomicScalar(ref.valueType)
    : ref.addressSpace === "storage" || ref.addressSpace === "constant" || ref.addressSpace === "device-global"
      ? wgslBindingType(ref.valueType)
      : semanticType;
  const read = createTypedWgslMemoryRead(
    nameFor(ref.base, names),
    root.dimensions.length > 1
      ? emitTypedFlatRankedIndex(root.dimensions, ref.indices, ir, names, options, ref.span)
      : ref.addressSpace === "shared"
        ? emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options)
        : emitTypedFlatStorageIndex(ref, ir, names, options),
    storageType,
    atomic,
    expression.span,
  );
  return legalizeSemanticMemoryReadValue(read, semanticType, expression.span);
}

function emitTypedFlatRankedIndex(
  dimensions: readonly number[],
  indices: readonly SemanticExpression[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  span: SourceSpan,
): TypedWgslExpression {
  if (indices.length === 1) return emitSemanticExpressionAs(indices[0]!, ir, names, "u32", options);
  if (indices.length !== dimensions.length) throw semanticWgslError("typed WGSL array index rank mismatch", span);
  return indices.map((index, offset) => {
    const value = emitSemanticExpressionAs(index, ir, names, "u32", options);
    const stride = dimensions.slice(offset + 1).reduce((size, dimension) => size * dimension, 1);
    return stride === 1 ? value : emitTypedWgslBinary("*", value, createTypedWgslLiteral(`${stride}u`, "u32", span), span);
  }).reduce((left, right) => emitTypedWgslBinary("+", left, right, span));
}

function semanticTypedLocalArrayPath(
  flat: TypedWgslExpression,
  dimensions: readonly number[],
  span: SourceSpan,
): readonly TypedWgslExpression[] {
  return dimensions.map((dimension, offset) => {
    const stride = dimensions.slice(offset + 1).reduce((size, value) => size * value, 1);
    const quotient = stride === 1
      ? flat
      : emitTypedWgslBinary("/", flat, createTypedWgslLiteral(`${stride}u`, "u32", span), span);
    return dimension > 1
      ? emitTypedWgslBinary("%", quotient, createTypedWgslLiteral(`${Math.max(1, dimension)}u`, "u32", span), span)
      : createTypedWgslZero("u32", span);
  });
}

function emitTypedFlatStorageIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): TypedWgslExpression {
  const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
  const localRuntimePointer = ref.addressSpace === "local"
    ? semanticLocalPointerDeclarations(ir).find((operation) =>
        operation.target.name === ref.base && operation.target.pointerRuntimeState === true)
    : undefined;
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
  if (!pointerParam && !localRuntimePointer && !hasOffset && ref.indices.length === 0) return createTypedWgslZero("u32", ref.span);
  if (!pointerParam && !localRuntimePointer && !hasOffset && ref.indices.length === 1) {
    return emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  }
  const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options));
  if (pointerParam || localRuntimePointer) {
    terms.unshift(convertTypedWgslExpression(
      createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
      "i32",
      true,
    ));
  } else if (hasOffset) {
    terms.unshift(createTypedWgslIdentifier(nameFor(storageOffsetSymbol(ref.base), names), "i32", ref.span));
  }
  const index = terms.reduce((left, right) => emitTypedWgslBinary("+", left, right, ref.span));
  return convertTypedWgslExpression(index, "u32", true);
}

function semanticTypedNativeMathCallee(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): string | undefined {
  if (expression.callee.kind !== "symbol" || semanticExpressionWgslType(expression, ir) !== "f32") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  return callee !== undefined && TYPED_NATIVE_WGSL_MATH_CALLS.has(callee) ? callee : undefined;
}

function semanticWgslSymbolHasTypedEmission(
  expression: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.addressSpace === "local" || expression.addressSpace === "uniform" || expression.addressSpace === "surface" || expression.addressSpace === "texture") return true;
  if (expression.addressSpace !== "constant") return false;
  const constant = constantMemorySymbols(ir).find((symbol) => symbol.name === expression.name);
  return constant?.initialized === true || !isSemanticFloatVectorType(expression.valueType);
}

function emitSemanticSymbolExpression(
  expression: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): TypedWgslExpression {
  const type = semanticExpressionWgslType(expression, ir);
  if (expression.addressSpace === "surface") return createTypedWgslIdentifier(nameFor(expression.name, names), "u32", expression.span);
  if (expression.addressSpace === "texture") return createTypedWgslIdentifier(nameFor(expression.name, names), "texture_2d<f32>", expression.span);
  if (expression.addressSpace === "uniform") {
    return createTypedWgslQualifiedAccess(UNIFORM_PARAMS_NAME, nameFor(expression.name, names), type, expression.span);
  }
  if (expression.addressSpace === "constant") {
    const constant = constantMemorySymbols(ir).find((symbol) => symbol.name === expression.name);
    if (!constant?.initialized) {
      return createTypedWgslQualifiedAccess(UNIFORM_PARAMS_NAME, nameFor(expression.name, names), type, expression.span);
    }
  }
  return createTypedWgslIdentifier(nameFor(expression.name, names), type, expression.span);
}

function semanticExpressionWgslType(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
): TypedWgslExpression["type"] {
  if (semanticNativeBoolExpression(expression)) return "bool";
  const vectorType = semanticStorageVectorType(semanticExpressionVectorValueType(expression, ir.functions));
  return vectorType === undefined ? semanticExpressionWgslScalar(expression) : wgslValueType(vectorType);
}

function emitSemanticConditional(
  expression: Extract<SemanticExpression, { readonly kind: "conditional" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression {
  const resultType = semanticExpressionWgslType(expression, ir);
  const emitArm = (arm: SemanticExpression): TypedWgslExpression => {
    if (resultType === "f16" || resultType === "f32" || resultType === "i32" || resultType === "u32") {
      return emitSemanticExpressionAs(arm, ir, names, resultType, options, textureSpecializations);
    }
    return emitSemanticExpression(arm, ir, names, options, textureSpecializations);
  };
  const condition = emitSemanticTruthinessExpression(expression.condition, ir, names, options);
  const emitted = emitTypedWgslSelect(
    emitArm(expression.alternate),
    emitArm(expression.consequent),
    condition,
    expression.span,
  );
  if (emitted.type !== resultType) {
    throw semanticWgslError(
      `WGSL conditional produces '${emitted.type}', semantic IR declares '${resultType}'`,
      expression.span,
    );
  }
  return emitted;
}

function semanticWgslSharedAddressCallRef(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticMemoryRef | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.name !== "__cvta_generic_to_shared") return undefined;
  const arg = expression.args[0];
  if (!arg) return undefined;
  const target = arg.kind === "unary" && arg.operator === "&" ? arg.argument : arg;
  const ref = memoryRefFromIndexExpression(target) ?? (target.kind === "symbol" && target.addressSpace === "shared" && target.valueType !== undefined && target.valueType !== "void" ? {
    baseId: semanticMemoryIdFromSymbol(target.id),
    base: target.name,
    addressSpace: target.addressSpace,
    valueType: target.valueType,
    indices: [],
    fields: [],
    span: target.span,
  } : undefined);
  return ref?.addressSpace === "shared" ? ref : undefined;
}

function emitSemanticSurfaceRead(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (!semanticWgslSurfaceReadSupported(expression, ir) || expression.surface.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL supports only direct scalar surf2Dread", expression.span);
  }
  const surfaceName = expression.surface.name;
  const xBytes = emitSemanticExpressionAs(expression.xBytes, ir, names, "i32", options).code;
  const y = emitSemanticExpressionAs(expression.y, ir, names, "i32", options).code;
  const z = expression.z ? emitSemanticExpressionAs(expression.z, ir, names, "i32", options).code : "0";
  const directSurface = surfaceSymbols(ir).some((surface) => surface.name === surfaceName);
  const readAt = (xBytesExpr: string): string => directSurface
    ? `${surfaceReadHelperName(surfaceName, names)}(${xBytesExpr}, ${y}, ${z})`
    : `${GENERIC_SURFACE_READ_HELPER_NAME}(${nameFor(surfaceName, names)}, ${xBytesExpr}, ${y}, ${z})`;
  if (expression.valueType === "bf162") {
    const vector = `vec2<f32>(${wgslRoundBfloat16(readAt(`(${xBytes} + 0)`))}, ${wgslRoundBfloat16(readAt(`(${xBytes} + 4)`))})`;
    return `select(vec2<f32>(), ${vector}, (${xBytes} >= 0 && (${xBytes} % 4) == 0))`;
  }
  if (isSemanticFloatVectorType(expression.valueType)) {
    const laneType = wgslVectorScalar(expression.valueType);
    const lanes = Array.from({ length: cudaVectorLaneCount(expression.valueType) }, (_, lane) => `${laneType}(${readAt(`(${xBytes} + ${lane * 4})`)})`);
    const vectorType = wgslValueType(expression.valueType);
    const vector = `${vectorType}(${lanes.join(", ")})`;
    return `select(${vectorType}(), ${vector}, (${xBytes} >= 0 && (${xBytes} % 4) == 0))`;
  }
  const read = readAt(xBytes);
  if (expression.valueType === "half") return `f16(${read})`;
  if (expression.valueType === "bf16") return wgslRoundBfloat16(read);
  if (expression.valueType === "uint" || expression.valueType === "uchar") return `u32(${read})`;
  if (expression.valueType === "int") return `i32(${read})`;
  return read;
}

const GENERIC_SURFACE_READ_HELPER_NAME = "bg_sem_surf2dread";
const GENERIC_SURFACE_WRITE_HELPER_NAME = "bg_sem_surf2dwrite";

function emitSemanticGenericSurfaceReadHelper(
  surfaces: readonly SemanticKernelIrModule["params"][number][],
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const lines = [
    `fn ${GENERIC_SURFACE_READ_HELPER_NAME}(surface: u32, x_bytes: i32, y: i32, z: i32) -> f32 {`,
  ];
  for (const [index, surface] of surfaces.entries()) {
    lines.push(`  if (surface == ${index}u) {`);
    lines.push(`    return ${surfaceReadHelperName(surface.name, names)}(x_bytes, y, z);`);
    lines.push("  }");
  }
  lines.push("  return 0.0;");
  lines.push("}");
  return lines;
}

function emitSemanticGenericSurfaceWriteHelper(
  surfaces: readonly SemanticKernelIrModule["params"][number][],
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const lines = [
    `fn ${GENERIC_SURFACE_WRITE_HELPER_NAME}(surface: u32, value: f32, x_bytes: i32, y: i32, z: i32) {`,
  ];
  for (const [index, surface] of surfaces.entries()) {
    lines.push(`  if (surface == ${index}u) {`);
    lines.push(...emitSemanticSurfaceWriteBody(surface, "value", "x_bytes", "y", "z", names, 2));
    lines.push("  }");
  }
  lines.push("}");
  return lines;
}

function emitSemanticSurfaceWriteBody(
  surfaceSymbol: SemanticKernelIrModule["params"][number],
  value: string,
  xBytes: string,
  y: string,
  z: string,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const surfaceName = surfaceSymbol.name;
  const surface = nameFor(surfaceName, names);
  const width = `${UNIFORM_PARAMS_NAME}.${nameFor(surfaceWidthField(surfaceName), names)}`;
  const height = `${UNIFORM_PARAMS_NAME}.${nameFor(surfaceHeightField(surfaceName), names)}`;
  return [
    `${prefix}{`,
    `${prefix}  let bg_x_bytes = ${xBytes};`,
    `${prefix}  if (bg_x_bytes >= 0 && (bg_x_bytes % 4) == 0) {`,
    `${prefix}    let bg_x = bg_x_bytes / 4;`,
    `${prefix}    let bg_y = ${y};`,
    `${prefix}    let bg_z = ${z};`,
    `${prefix}    let bg_index = ((bg_z * i32(${height})) + bg_y) * i32(${width}) + bg_x;`,
    `${prefix}    if (bg_x >= 0 && bg_x < i32(${width}) && bg_y >= 0 && bg_y < i32(${height}) && bg_z >= 0 && bg_index >= 0 && bg_index < i32(arrayLength(&${surface}))) {`,
    `${prefix}      ${surface}[bg_index] = ${value};`,
    `${prefix}    }`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticSurfaceReadHelper(
  surface: SemanticKernelIrModule["params"][number],
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const surfaceName = surface.name;
  const storage = nameFor(surfaceName, names);
  const width = `${UNIFORM_PARAMS_NAME}.${nameFor(surfaceWidthField(surfaceName), names)}`;
  const height = `${UNIFORM_PARAMS_NAME}.${nameFor(surfaceHeightField(surfaceName), names)}`;
  const fn = surfaceReadHelperName(surfaceName, names);
  return [
    `fn ${fn}(x_bytes: i32, y: i32, z: i32) -> f32 {`,
    "  if (x_bytes < 0 || (x_bytes % 4) != 0) {",
    "    return 0.0;",
    "  }",
    "  let x = x_bytes / 4;",
    `  let width = i32(${width});`,
    `  let height = i32(${height});`,
    "  if (x < 0 || x >= width || y < 0 || y >= height || z < 0) {",
    "    return 0.0;",
    "  }",
    "  let index = ((z * height) + y) * width + x;",
    `  if (index >= 0 && index < i32(arrayLength(&${storage}))) {`,
    `    return ${storage}[index];`,
    "  }",
    "  return 0.0;",
    "}",
  ];
}

const SEMANTIC_BINDLESS_TEXTURE_READ_HELPER = "bg_semantic_bindless_texture_read";

function emitSemanticBindlessTextureReadHelper(
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): readonly string[] {
  const lines = [`fn ${SEMANTIC_BINDLESS_TEXTURE_READ_HELPER}(handle: u32, x: f32, y: f32) -> vec4<f32> {`, "  switch handle {"];
  for (const [index, textureName] of (ir.bindlessTextures ?? []).entries()) {
    const texture = nameFor(textureName, names);
    const descriptor = options.textureDescriptors?.[textureName];
    const read = descriptor
      ? `${semanticTextureDescriptorHelperName(textureName, names, descriptor)}(${texture}, x, y)`
      : `textureLoad(${texture}, clamp(vec2<i32>(i32(floor(x)), i32(floor(y))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${texture})) - vec2<i32>(1, 1)), 0)`;
    lines.push(`    case ${index}u: { return ${read}; }`);
  }
  lines.push("    default: { return vec4<f32>(0.0); }", "  }", "}");
  return lines;
}

function semanticUsesCubemapTextureRead(ir: SemanticKernelIrModule): boolean {
  let found = false;
  const visit = (operations: readonly SemanticKernelIrOperation[]): void => {
    walkSemanticOperations(operations, (expression) => {
      if (expression.kind === "texture-read" && expression.callee === "texCubemap") found = true;
    });
  };
  visit(ir.operations);
  for (const fn of ir.functions) visit(fn.body);
  return found;
}

function emitSemanticFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL function call requires symbol callee", expression.span);
  const callee = expression.callee.name;
  const fn = ir.functions.find((item) => item.name === callee);
  if (!fn) throw semanticWgslError(`semantic WGSL unknown function '${callee}'`, expression.span);
  const args = expression.args.flatMap((arg, index) => emitSemanticFunctionArgs(arg, fn.params[index], ir, names, options, textureSpecializations));
  const calleeName = semanticFunctionCallName(callee, fn, expression.args, options, textureSpecializations);
  return `${nameFor(calleeName, names)}(${[...args, "local_id", "workgroup_id", "num_workgroups"].join(", ")})`;
}

function emitSemanticVectorConstructorExpression(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  const valueType = expression.callee.kind === "symbol" ? cudaVectorConstructorType(expression.callee.name) : undefined;
  if (!isSemanticFloatVectorType(valueType)) throw semanticWgslError("semantic WGSL vector constructor requires vector target", expression.span);
  const fields = ["x", "y", "z", "w"];
  const targetLanes = cudaVectorLaneCount(valueType);
  const targetScalar = wgslVectorScalar(valueType);
  const targetType = wgslValueType(valueType);
  if (!isWgslVectorType(targetType)) throw semanticWgslError(`invalid WGSL vector type '${targetType}'`, expression.span);
  if (expression.args.length === 1 && !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.args[0]!, ir?.functions))) {
    const scalar = emitSemanticExpressionAs(expression.args[0]!, ir, names, targetScalar, options, textureSpecializations);
    const lanes = Array.from({ length: targetLanes }, () =>
      convertTypedWgslExpression(scalar, targetScalar, true)
    );
    return createTypedWgslConstructor(targetType, lanes, expression.span);
  }
  const lanes = expression.args.flatMap((arg) => {
    const argType = semanticExpressionVectorValueType(arg, ir?.functions);
    if (isSemanticFloatVectorType(argType)) {
      const value = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
      return Array.from({ length: cudaVectorLaneCount(argType) }, (_, lane) => {
        const member = createTypedWgslMemberAccess(value, fields[lane]!, wgslVectorScalar(argType), arg.span);
        return convertTypedWgslExpression(member, targetScalar, true);
      });
    }
    const scalar = emitSemanticExpressionAs(arg, ir, names, targetScalar, options, textureSpecializations);
    return [convertTypedWgslExpression(scalar, targetScalar, true)];
  });
  while (lanes.length < targetLanes) lanes.push(createTypedWgslZero(targetScalar, expression.span));
  return createTypedWgslConstructor(targetType, lanes.slice(0, targetLanes), expression.span);
}

function emitSemanticVectorAtCallExpression(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  const [target, index] = expression.args;
  if (!target || !index) throw semanticWgslError("semantic WGSL vec_at requires vector and index", expression.span);
  return createTypedWgslIndexAccess(
    emitSemanticExpression(target, ir, names, options, textureSpecializations),
    emitSemanticExpressionAs(index, ir, names, "u32", options, textureSpecializations),
    semanticExpressionWgslType(expression, ir),
    expression.span,
  );
}

function semanticWgslBf162LocalBitsCastSupported(
  expression: Extract<SemanticExpression, { readonly kind: "unary" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.operator !== "*" || expression.valueType !== "uint") return false;
  const arg = expression.argument;
  if (arg.kind !== "cast" || !arg.pointer || arg.valueType !== "uint") return false;
  const address = arg.expression;
  if (address.kind !== "unary" || address.operator !== "&" || address.argument.kind !== "symbol") return false;
  const target = address.argument;
  return target.addressSpace === "local" &&
    semanticExpressionVectorValueType(target, ir?.functions) === "bf162" &&
    (ir === undefined || ir.operations.some((operation) =>
      operation.kind === "declare" &&
      operation.target.name === target.name &&
      operation.target.addressSpace === "local" &&
      operation.target.valueType === "bf162"
    ));
}

function emitSemanticBf162LocalBitsCast(
  expression: Extract<SemanticExpression, { readonly kind: "unary" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  const cast = expression.argument;
  if (cast.kind !== "cast" || cast.expression.kind !== "unary" || cast.expression.argument.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL bf162 bitcast requires local bf162 symbol", expression.span);
  }
  const value = emitSemanticExpression(cast.expression.argument, ir, names, options, textureSpecializations);
  const lane = (property: "x" | "y"): TypedWgslExpression => {
    const member = createTypedWgslMemberAccess(value, property, "f32", expression.span);
    const scalar = convertTypedWgslExpression(member, "f32", true);
    return createTypedWgslBitcast("u32", scalar, expression.span);
  };
  const low = emitTypedWgslBinary(">>", lane("x"), createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
  const high = emitTypedWgslBinary("&", lane("y"), createTypedWgslLiteral("0xffff0000u", "u32", expression.span), expression.span);
  return emitTypedWgslBinary("|", low, high, expression.span);
}

function emitSemanticCastExpression(
  expression: Extract<SemanticExpression, { readonly kind: "cast" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (expression.pointer) throw semanticWgslError("typed WGSL scalar cast cannot lower pointer cast", expression.span);
  if (expression.valueType === "uchar") {
    return emitSemanticUcharExpressionValue(expression.expression, ir, names, options, textureSpecializations);
  }
  if (expression.valueType === "bool") {
    return emitSemanticBoolExpressionValue(expression.expression, ir, names, options, textureSpecializations);
  }
  const value = emitSemanticExpression(expression.expression, ir, names, options, textureSpecializations);
  const sourceType = "valueType" in expression.expression ? expression.expression.valueType : undefined;
  if (expression.valueType === "int" && sourceType === "uint") return createTypedWgslBitcast("i32", value, expression.span);
  if (expression.valueType === "uint" && sourceType === "int") return createTypedWgslBitcast("u32", value, expression.span);
  const targetType = wgslScalar(expression.valueType);
  if (value.type === "bool" && targetType !== "bool") return legalizeTypedWgslBoolToNumeric(value, targetType);
  return convertTypedWgslExpression(value, targetType, true);
}

function emitSemanticFunctionArg(
  arg: SemanticExpression,
  param: SemanticKernelIrModule["functions"][number]["params"][number] | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (param?.addressSpace === "texture") {
    if (arg.kind !== "symbol" || arg.addressSpace !== "texture") throw semanticWgslError("semantic WGSL texture helper argument must be a texture symbol", arg.span);
    return nameFor(arg.name, names);
  }
  if (param?.addressSpace === "surface") {
    if (arg.kind !== "symbol" || arg.addressSpace !== "surface") throw semanticWgslError("semantic WGSL surface helper argument must be a surface symbol", arg.span);
    const handle = surfaceHandleForName(arg.name, ir);
    if (handle === undefined) throw semanticWgslError(`unknown surface '${arg.name}'`, arg.span);
    return `${handle}u`;
  }
  if (param?.valueType === "bool") return emitTruthiness(arg, ir, names, options);
  if (param?.valueType === "uchar") return emitSemanticUcharExpression(arg, ir, names, options, textureSpecializations);
  if (isSemanticFloatVectorType(param?.valueType)) return emitSemanticExpression(arg, ir, names, options, textureSpecializations).code;
  return emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param?.valueType), options, textureSpecializations).code;
}

function emitSemanticFunctionArgs(
  arg: SemanticExpression,
  param: SemanticKernelIrModule["functions"][number]["params"][number] | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (param?.pointer && param.addressSpace === "constant" && param.pointerMemoryAlias !== undefined) return [];
  if (param?.cooperativeGroupKind !== undefined) {
    if (arg.kind !== "symbol") throw semanticWgslError("semantic WGSL cooperative-group argument must be a symbol", arg.span);
    const groupCall = (property: "thread_rank" | "size"): Extract<SemanticExpression, { readonly kind: "call" }> => ({
      kind: "call",
      callee: { kind: "member", object: arg, property, valueType: "int", span: arg.span },
      args: [],
      valueType: "int",
      span: arg.span,
    });
    const rank = emitSemanticCooperativeGroupCall(groupCall("thread_rank"), ir, options.activeFunction);
    const size = emitSemanticCooperativeGroupCall(groupCall("size"), ir, options.activeFunction);
    if (rank === undefined || size === undefined) throw semanticWgslError(`unknown cooperative group '${arg.name}'`, arg.span);
    return [rank, size];
  }
  if (param?.pointer && param.addressSpace === "shared") {
    const ref = semanticPointerArgMemoryRef(arg);
    if (!ref || ref.addressSpace !== "shared") throw semanticWgslError("semantic WGSL shared pointer helper argument must be modeled shared memory", arg.span);
    const base = emitSemanticSharedPointerArgBaseIndex(ref, ir, names);
    const sourceParam = semanticWgslFunctionSharedPointerParam(ir, ref.base);
    const pointer = sourceParam === undefined
      ? `&${nameFor(ref.base, names)}`
      : nameFor(semanticParamAliasName(ir, sourceParam) ?? sourceParam.name, names);
    return param.pointerParamAlias === undefined ? [pointer, base] : [base];
  }
  if (param?.pointer && param.addressSpace === "local") {
    const ref = semanticPointerArgMemoryRef(arg);
    const owner = options.activeFunction === undefined
      ? undefined
      : ir.functions.find((candidate) => candidate.name === options.activeFunction);
    const forwarded = owner?.params.find((candidate) =>
      candidate.name === ref?.base && candidate.pointer && candidate.addressSpace === "local"
    );
    if (ref && forwarded) {
      const pointer = nameFor(ref.base, names);
      if (param.dimensions.length === 0) return [pointer];
      const offset = ref.indices[0] === undefined ? "0u" : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options).code;
      const base = forwarded.dimensions.length > 0
        ? `(${nameFor(semanticPointerBaseParamName(forwarded.name), names)} + ${offset})`
        : offset;
      return [pointer, base];
    }
    const localArray = ref?.addressSpace === "local" ? localArraySymbol(ir, ref.base) : undefined;
    if (ref && localArray && ref.indices.length <= 1 && param.dimensions.length > 0) {
      const base = ref.indices[0] === undefined ? "0u" : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options).code;
      return [`&${nameFor(ref.base, names)}`, base];
    }
    if (!ref || ref.addressSpace !== "local" || ref.indices.length !== 0) {
      throw semanticWgslError("semantic WGSL local pointer helper argument must be a local scalar", arg.span);
    }
    return [`&${nameFor(ref.base, names)}`];
  }
  if (param?.pointer && param.addressSpace === "storage") {
    const ref = semanticPointerArgMemoryRef(arg);
    if (!ref || ref.addressSpace !== "storage" && ref.addressSpace !== "device-global") throw semanticWgslError("semantic WGSL storage pointer helper argument must be modeled storage", arg.span);
    const forwarded = options.activeFunction === undefined
      ? undefined
      : ir.functions.find((candidate) => candidate.name === options.activeFunction)?.params.find((candidate) =>
        candidate.name === ref.base && candidate.pointer && candidate.addressSpace === "storage");
    if (forwarded) {
      return [
        nameFor(semanticPointerBufferParamName(ref.base), names),
        emitSemanticPointerArgBaseIndex(ref, ir, names, options),
      ];
    }
    const bufferId = semanticStoragePointerBufferId(ref.base, ir);
    if (bufferId === undefined) throw semanticWgslError(`semantic WGSL unknown storage pointer base '${ref.base}'`, arg.span);
    return [`${bufferId}u`, emitSemanticPointerArgBaseIndex(ref, ir, names, options)];
  }
  return [emitSemanticFunctionArg(arg, param, ir, names, options, textureSpecializations)];
}

function emitSemanticSharedPointerArgBaseIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  const pointer = semanticWgslFunctionSharedPointerParam(ir, ref.base);
  if (pointer) {
    const base = nameFor(semanticPointerBaseParamName(ref.base), names);
    if (ref.indices.length === 0) return base;
    if (pointer.dimensions.length === 0 || ref.indices.length !== 1) {
      throw semanticWgslError(`shared pointer '${ref.base}' index rank mismatch`, ref.span);
    }
    const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32").code;
    return `(${base} + ${index})`;
  }
  if (ref.indices.length === 0) return "0u";
  const shared = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
  if (!shared) throw semanticWgslError(`unknown shared pointer base '${ref.base}'`, ref.span);
  return emitFlatSharedIndex(shared, ref.indices, ir, names);
}

function emitSemanticPointerArgBaseIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const paramRoot = ir.params.find((param) => param.name === ref.base && param.addressSpace === "storage");
  if (paramRoot) return emitSemanticRootStoragePointerArgBaseIndex(ref, paramRoot, ir, names, options);
  const root = ir.memory.find((symbol) => symbol.name === ref.base);
  const valueType = root?.valueType;
  if (isCudaVectorType(valueType)) {
    const vectorType = valueType as CudaLiteScalarType;
    if (ref.pointerBaseIsScalarLane === true) return emitFlatStorageIndex(ref, ir, names, options);
    return emitFlatStorageVectorBaseIndex({ ...ref, containerValueType: vectorType }, ir, names, options);
  }
  return emitFlatStorageIndex(ref, ir, names, options);
}

function emitSemanticRootStoragePointerArgBaseIndex(
  ref: SemanticMemoryRef,
  root: SemanticKernelIrModule["params"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (!isCudaVectorType(root.valueType)) return emitSemanticRootStorageIndex(ref, ir, names, options);
  const base = emitSemanticRootStorageIndex({ ...ref, valueType: "float" }, ir, names, options);
  if (ref.pointerBaseIsScalarLane === true || !isCudaVectorType(ref.valueType)) return base;
  const stride = cudaVectorLaneCount(root.valueType);
  return stride === 1 ? base : `(${base} * ${stride}u)`;
}

function emitSemanticMemoryRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (semanticWgslLocalPackedHalf2View(ref, ir)) {
    const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
    const root = localArraySymbol(ir, ref.base)!;
    const word = `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(index, root.dimensions)}`;
    const value = `unpack2x16float(${word})`;
    return effectiveSemanticF16Mode(ir, options) === "native" ? `vec2<f16>(${value})` : value;
  }
  if (semanticWgslLocalPackedHalfView(ref, ir)) {
    const halfIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
    const packed = emitSemanticLocalPackedHalfWord(ref, halfIndex, ir, names);
    const lane = `unpack2x16float(${packed})[(${halfIndex} & 1u)]`;
    return effectiveSemanticF16Mode(ir, options) === "native" ? `f16(${lane})` : lane;
  }
  const bitRootType = semanticWgslLocalScalarBitViewRootType(ref, ir) ?? semanticWgslSharedScalarBitViewRootType(ref, ir);
  const vectorBitRootType = semanticWgslLocalVectorBitViewRootType(ref, ir);
  if (bitRootType !== undefined || vectorBitRootType !== undefined) {
    const rootType = requireSemanticValueType(bitRootType ?? vectorBitRootType, `bit view '${ref.base}'`, ref.span);
    const access = vectorBitRootType === undefined
      ? emitSemanticMemoryRef({ ...ref, valueType: rootType }, ir, names, options)
      : emitSemanticLocalVectorBitViewAccess(ref, ir, names, options);
    return `bitcast<${wgslValueType(ref.valueType)}>(${access})`;
  }
  if (semanticWgslDirectByteRawView(ref, ir)) {
    const base = emitFlatStorageIndex({ ...ref, valueType: "uchar" }, ir, names, options);
    const storage = nameFor(ref.base, names);
    if (semanticAtomicMemoryRootNames(ir).has(ref.base)) {
      return emitSemanticAtomicByteStorageReadValue(ref.valueType, storage, base);
    }
    const word = `(${storage}[${base}] | (${storage}[(${base} + 1u)] << 8u) | (${storage}[(${base} + 2u)] << 16u) | (${storage}[(${base} + 3u)] << 24u))`;
    if (ref.valueType === "float") return `bitcast<f32>(${word})`;
    return ref.valueType === "int" ? `bitcast<i32>(${word})` : word;
  }
  if (semanticWgslPackedSharedByteRoot(ref, ir)) {
    return emitSemanticPackedSharedByteRead(ref, ir, names, options);
  }
  if (semanticWgslSharedHalfBitView(ref, ir)) {
    const low = emitSemanticMemoryRef({ ...semanticCopyMemoryRefAt(ref, 0), valueType: "half", containerValueType: "half" }, ir, names, options);
    const high = emitSemanticMemoryRef({ ...semanticCopyMemoryRefAt(ref, 1), valueType: "half", containerValueType: "half" }, ir, names, options);
    const word = `bitcast<u32>(vec2<f16>(${low}, ${high}))`;
    return ref.valueType === "float" ? `bitcast<f32>(${word})` : ref.valueType === "int" ? `bitcast<i32>(${word})` : word;
  }
  if (semanticWgslSharedVectorScalarView(ref, ir)) {
    const target = emitSemanticMemoryRef(ref, ir, names, options);
    if (!semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base)) return target;
    return emitSemanticAtomicLoad(ref, target);
  }
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null)) {
    const valueType = ref.valueType ?? "float";
    const index = isCudaVectorType(valueType) ? emitFlatStorageVectorBaseIndex(ref, ir, names, options) : emitFlatStorageIndex(ref, ir, names, options);
    return `${semanticPointerReadHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(ref.base), names)}, ${index})`;
  }
  if (semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null)) {
    const target = emitSemanticSharedPointerMemoryRef(ref, ir, names, options);
    const param = semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null)!;
    return semanticSharedPointerNeedsBitcast(param)
      ? `bitcast<${wgslValueType(param.valueType)}>(${target})`
      : target;
  }
  return emitSemanticMemoryRef(ref, ir, names, options);
}

function semanticWgslDirectByteRawView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  return ref.addressSpace === "storage" && ref.packedByteLanes === 4 &&
    ir.params.some((param) => param.name === ref.base && param.valueType === "uchar");
}

function semanticWgslLocalPackedByteRawView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  return ref.addressSpace === "local" && ref.packedByteLanes === 4 &&
    semanticWgslDeclaredPackedByteLocal(ir.operations, ref.base);
}

function semanticWgslDeclaredPackedByteLocal(operations: readonly SemanticKernelIrOperation[], name: string): boolean {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.target.name === name) return operation.target.packedByteLanes === 4;
    if (operation.kind === "block" && semanticWgslDeclaredPackedByteLocal(operation.body, name)) return true;
    if (operation.kind === "branch" && (
      semanticWgslDeclaredPackedByteLocal(operation.consequent, name) ||
      semanticWgslDeclaredPackedByteLocal(operation.alternate, name)
    )) return true;
    if (operation.kind === "loop" && semanticWgslDeclaredPackedByteLocal(operation.body, name)) return true;
  }
  return false;
}

function emitSemanticMemoryWrite(
  ref: SemanticMemoryRef,
  value: string,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (semanticWgslLocalPackedHalf2View(ref, ir)) {
    const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
    const root = localArraySymbol(ir, ref.base)!;
    const target = `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(index, root.dimensions)}`;
    return `${target} = pack2x16float(vec2<f32>(${value}))`;
  }
  if (semanticWgslLocalPackedHalfView(ref, ir)) {
    const halfIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
    const word = emitSemanticLocalPackedHalfWord(ref, halfIndex, ir, names);
    const bits = `(pack2x16float(vec2<f32>(f32(${value}), 0.0)) & 0xffffu)`;
    return `${word} = select((${word} & 0xffff0000u) | ${bits}, (${word} & 0x0000ffffu) | (${bits} << 16u), (${halfIndex} & 1u) != 0u)`;
  }
  const bitRootType = semanticWgslLocalScalarBitViewRootType(ref, ir) ?? semanticWgslSharedScalarBitViewRootType(ref, ir);
  const vectorBitRootType = semanticWgslLocalVectorBitViewRootType(ref, ir);
  if (bitRootType !== undefined || vectorBitRootType !== undefined) {
    const rootType = requireSemanticValueType(bitRootType ?? vectorBitRootType, `bit view '${ref.base}'`, ref.span);
    const target = vectorBitRootType === undefined
      ? emitSemanticMemoryRef({ ...ref, valueType: rootType }, ir, names, options)
      : emitSemanticLocalVectorBitViewAccess(ref, ir, names, options);
    return `${target} = bitcast<${wgslValueType(rootType)}>(${value})`;
  }
  if (semanticWgslDirectByteRawView(ref, ir)) {
    const base = emitFlatStorageIndex({ ...ref, valueType: "uchar" }, ir, names, options);
    const storage = nameFor(ref.base, names);
    if (semanticAtomicMemoryRootNames(ir).has(ref.base)) {
      return emitSemanticAtomicByteStorageWriteValue(ref.valueType, storage, base, value);
    }
    const word = ref.valueType === "float" ? `bitcast<u32>(${value})` : `u32(${value})`;
    const temp = `bg_raw_word_${ref.span.start}`;
    const writes = [0, 1, 2, 3].map((byte) =>
      `${storage}[(${base} + ${byte}u)] = ((${temp} >> ${byte * 8}u) & 255u)`
    ).join("; ");
    return `{ let ${temp} = ${word}; ${writes}; }`;
  }
  if (semanticWgslPackedSharedByteRoot(ref, ir)) {
    return emitSemanticPackedSharedByteWrite(ref, value, ir, names, options);
  }
  if (semanticWgslSharedVectorScalarView(ref, ir)) {
    const target = emitSemanticMemoryRef(ref, ir, names, options);
    if (!semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base)) return `${target} = ${value}`;
    const stored = ref.valueType === "float" || ref.valueType === "bf16" ? `bitcast<u32>(${value})` : value;
    return `atomicStore(&${target}, ${stored})`;
  }
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null)) {
    const valueType = ref.valueType ?? "float";
    const index = isCudaVectorType(valueType) ? emitFlatStorageVectorBaseIndex(ref, ir, names, options) : emitFlatStorageIndex(ref, ir, names, options);
    return `${semanticPointerWriteHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(ref.base), names)}, ${index}, ${value})`;
  }
  if (semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null)) {
    const target = emitSemanticSharedPointerMemoryRef(ref, ir, names, options);
    const param = semanticWgslFunctionSharedPointerParam(ir, ref.base, options.activeFunction ?? null)!;
    const stored = semanticSharedPointerNeedsBitcast(param)
      ? `bitcast<${wgslValueType(param.pointerCarrierValueType)}>(${value})`
      : value;
    return `${target} = ${stored}`;
  }
  const target = emitSemanticMemoryRef(ref, ir, names, options);
  return `${target} = ${value}`;
}

function emitSemanticLocalVectorBitViewAccess(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): string {
  const index = ref.indices[0];
  if (!index) throw semanticWgslError(`local vector bit view '${ref.base}' requires one lane index`, ref.span);
  return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(index, ir, names, "u32", options).code}]`;
}

function emitSemanticLocalPackedHalfWord(
  ref: SemanticMemoryRef,
  halfIndex: string,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  const local = localArraySymbol(ir, ref.base);
  if (!local) throw semanticWgslError(`unknown packed local memory '${ref.base}'`, ref.span);
  return `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(`(${halfIndex} / 2u)`, local.dimensions)}`;
}

function emitSemanticPackedSharedByteIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): string {
  const elementBytes = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
  const indexAlreadyBytes = ref.pointerBaseUnitBytes !== undefined;
  const pointer = semanticWgslFunctionSharedPointerParam(ir, ref.base);
  if (pointer) {
    if (ref.indices.length > 1) throw semanticWgslError(`shared pointer '${ref.base}' index rank mismatch`, ref.span);
    const base = nameFor(semanticPointerBaseParamName(ref.base), names);
    const index = ref.indices[0] === undefined ? "0u" : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options).code;
    const offset = elementBytes === 1 || indexAlreadyBytes ? index : `(${index} * ${elementBytes}u)`;
    return `(${base} + ${offset})`;
  }
  const shared = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
  if (!shared) throw semanticWgslError(`unknown packed shared memory '${ref.base}'`, ref.span);
  const index = emitFlatSharedIndex(shared, ref.indices, ir, names);
  return elementBytes === 1 || indexAlreadyBytes ? index : `(${index} * ${elementBytes}u)`;
}

function emitSemanticPackedSharedByteRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): string {
  const byteIndex = emitSemanticPackedSharedByteIndex(ref, ir, names, options);
  const word = `${nameFor(ref.base, names)}[(${byteIndex} >> 2u)]`;
  const loaded = `atomicLoad(&${word})`;
  if (ref.valueType === "uint") return loaded;
  if (ref.valueType === "int") return `bitcast<i32>(${loaded})`;
  if (ref.valueType === "float") return `bitcast<f32>(${loaded})`;
  if (ref.valueType === "half" || ref.valueType === "bf16" || ref.valueType === "half2" || ref.valueType === "bf162") {
    const byteCount = ref.valueType === "half" || ref.valueType === "bf16" ? 2 : 4;
    const bytes = Array.from({ length: byteCount }, (_, offset) => {
      const address = `(${byteIndex} + ${offset}u)`;
      const source = `atomicLoad(&${nameFor(ref.base, names)}[(${address} >> 2u)])`;
      return `((${source} >> ((${address} & 3u) * 8u)) & 255u)`;
    });
    const bits = `(${bytes.map((byte, offset) => offset === 0 ? byte : `(${byte} << ${offset * 8}u)`).join(" | ")})`;
    if (ref.valueType === "half") {
      const value = `unpack2x16float(${bits}).x`;
      return effectiveSemanticF16Mode(ir, options) === "native" ? `f16(${value})` : value;
    }
    if (ref.valueType === "bf16") return `bitcast<f32>((${bits} & 0xffffu) << 16u)`;
    if (ref.valueType === "half2") {
      const value = `unpack2x16float(${bits})`;
      return effectiveSemanticF16Mode(ir, options) === "native" ? `vec2<f16>(${value})` : value;
    }
    return `vec2<f32>(bitcast<f32>((${bits} & 0xffffu) << 16u), bitcast<f32>(${bits} & 0xffff0000u))`;
  }
  const shift = `((${byteIndex} & 3u) * 8u)`;
  return `((${loaded} >> ${shift}) & 255u)`;
}

function emitSemanticPackedSharedByteWrite(
  ref: SemanticMemoryRef,
  value: string,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): string {
  const byteIndex = emitSemanticPackedSharedByteIndex(ref, ir, names, options);
  const word = `${nameFor(ref.base, names)}[(${byteIndex} >> 2u)]`;
  if (ref.valueType === "uint") return `atomicStore(&${word}, ${value})`;
  if (ref.valueType === "int" || ref.valueType === "float") return `atomicStore(&${word}, bitcast<u32>(${value}))`;
  if (ref.valueType === "half" || ref.valueType === "bf16" || ref.valueType === "half2" || ref.valueType === "bf162") {
    const bits = ref.valueType === "half"
      ? `(pack2x16float(vec2<f32>(f32(${value}), 0.0)) & 0xffffu)`
      : ref.valueType === "bf16"
      ? `(bitcast<u32>(f32(${value})) >> 16u)`
      : ref.valueType === "half2"
      ? `pack2x16float(vec2<f32>(${value}))`
      : `((bitcast<u32>(f32((${value}).x)) >> 16u) | (bitcast<u32>(f32((${value}).y)) & 0xffff0000u))`;
    const byteCount = ref.valueType === "half" || ref.valueType === "bf16" ? 2 : 4;
    const temporary = nameFor(`bg_packed_shared_bits_${ref.span.start}`, names);
    const writes = Array.from({ length: byteCount }, (_, offset) => {
      const address = `(${byteIndex} + ${offset}u)`;
      return `${PACKED_SHARED_U8_STORE}(&${nameFor(ref.base, names)}[(${address} >> 2u)], ((${address} & 3u) * 8u), (${temporary} >> ${offset * 8}u))`;
    });
    return `{ let ${temporary}: u32 = ${bits}; ${writes.join("; ")}; }`;
  }
  const shift = `((${byteIndex} & 3u) * 8u)`;
  return `${PACKED_SHARED_U8_STORE}(&${word}, ${shift}, u32(${value}))`;
}

function emitSemanticSharedPointerMemoryRef(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const param = semanticWgslFunctionSharedPointerParam(ir, ref.base);
  if (!param) throw semanticWgslError(`unknown shared pointer '${ref.base}'`, ref.span);
  const pointerName = semanticParamAliasName(ir, param) ?? ref.base;
  if (param.dimensions.length === 0) {
    if (ref.indices.length > 1 || ref.indices[0] && !semanticExpressionIsZero(ref.indices[0])) {
      throw semanticWgslError(`shared scalar pointer '${ref.base}' cannot be indexed`, ref.span);
    }
    return `*${nameFor(pointerName, names)}`;
  }
  const index = ref.indices[0] === undefined ? "0u" : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options).code;
  return `(*${nameFor(pointerName, names)})[(${nameFor(semanticPointerBaseParamName(ref.base), names)} + ${index})]`;
}

function semanticSharedPointerNeedsBitcast(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): boolean {
  return param.pointerCarrierValueType !== undefined && param.pointerCarrierValueType !== param.valueType;
}

function semanticExpressionIsZero(expression: SemanticExpression): boolean {
  return expression.kind === "literal" && typeof expression.value === "number" && expression.value === 0;
}

function emitSemanticExpressionAs(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  targetType: WgslValueType,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  const source = emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  if (source.type === "bool") return legalizeTypedWgslBoolToNumeric(source, targetType);
  if (expression.kind === "literal" && expression.literalKind === "number") {
    if (targetType === "i32" && expression.value > 2147483647) {
      return createTypedWgslBitcast(
        "i32",
        createTypedWgslLiteral(`${Math.trunc(expression.value) >>> 0}u`, "u32", expression.span),
        expression.span,
      );
    }
    const literal = emitNumberLiteral(expression.value, expression.valueType, targetType);
    if (isTypedWgslLiteralCode(literal, targetType)) return createTypedWgslLiteral(literal, targetType, expression.span);
  }
  return convertTypedWgslExpression(source, targetType);
}

function emitSemanticInitExpression(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (valueType === "bool") return emitSemanticBoolExpressionValue(expression, ir, names, options, textureSpecializations);
  if (valueType === "uchar") return emitSemanticUcharExpressionValue(expression, ir, names, options, textureSpecializations);
  if (semanticStorageVectorType(valueType) !== undefined) return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
}

function emitSemanticLocalScalarExpressionAs(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (valueType === "bool") return emitSemanticBoolExpression(expression, ir, names, options, textureSpecializations);
  if (valueType === "uchar") return emitSemanticUcharExpression(expression, ir, names, options, textureSpecializations);
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations).code;
}

function emitSemanticUcharExpression(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  return emitSemanticUcharExpressionValue(expression, ir, names, options, textureSpecializations).code;
}

function emitSemanticUcharExpressionValue(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (expression.kind === "cast" && expression.valueType === "uchar") {
    return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  }
  const value = emitSemanticExpressionAs(expression, ir, names, "i32", options, textureSpecializations);
  const normalized = convertTypedWgslExpression(value, "i32", true);
  const unsigned = convertTypedWgslExpression(normalized, "u32", true);
  return emitTypedWgslBinary("&", unsigned, createTypedWgslLiteral("0xffu", "u32", expression.span), expression.span);
}

function emitSemanticUcharValue(value: string): string {
  return `(u32(i32(${value})) & 0xffu)`;
}

function emitSemanticBoolExpression(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  return emitSemanticBoolExpressionValue(expression, ir, names, options, textureSpecializations).code;
}

function emitSemanticBoolExpressionValue(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (expression.kind === "literal" && typeof expression.value === "number") {
    return createTypedWgslLiteral(expression.value === 0 ? "false" : "true", "bool", expression.span);
  }
  const emitted = emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  if (emitted.type === "bool") return emitted;
  if (semanticNativeBoolExpression(expression)) {
    throw semanticWgslError(`native bool expression produced '${emitted.type}'`, expression.span);
  }
  const sourceType = semanticExpressionWgslScalar(expression);
  const source = emitted.type === sourceType
    ? emitted
    : emitSemanticExpressionAs(expression, ir, names, sourceType, options, textureSpecializations);
  return emitTypedWgslBinary("!=", source, createTypedWgslZero(sourceType, expression.span), expression.span);
}

function semanticNativeBoolExpression(expression: SemanticExpression): boolean {
  if (expression.kind === "pointer-valid") return true;
  if (expression.kind === "call" && expression.callee.kind === "symbol" && isSemanticHalf2BooleanComparisonCall(expression.callee.name)) return true;
  if (semanticExpressionValueType(expression) !== "bool") return false;
  if (expression.kind === "binary" && (COMPARISON_OPERATORS.has(expression.operator) || LOGICAL_OPERATORS.has(expression.operator))) return true;
  if (expression.kind === "unary" && expression.operator === "!") return true;
  return expression.kind === "symbol" && expression.addressSpace === "local";
}

function emitInitializedScalarConstant(
  symbol: SemanticKernelIrModule["memory"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (isSemanticFloatVectorType(symbol.valueType) && symbol.init) {
    const laneCount = cudaVectorLaneCount(symbol.valueType);
    const valueType = wgslValueType(symbol.valueType);
    const values = semanticVectorConstantInitExpressions(symbol.init)
      .slice(0, laneCount)
      .map((value) => emitSemanticExpressionAs(value, ir, names, "f32", options).code);
    while (values.length < laneCount) values.push("0.0");
    return `const ${nameFor(symbol.name, names)}: ${valueType} = ${valueType}(${values.join(", ")});`;
  }
  const valueType = wgslValueType(symbol.valueType);
  return createTypedWgslVariableStatement(
    "const",
    nameFor(symbol.name, names),
    valueType,
    emitSemanticInitExpression(symbol.init ?? zeroExpression(symbol.span), symbol.valueType, ir, names, options),
    symbol.span,
  ).code;
}

function emitSemanticPointerAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  target: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string | undefined {
  if (!semanticWgslFunctionStoragePointerParam(ir, target.base, options.activeFunction ?? null)) return undefined;
  if (expression.callee.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL pointer atomic requires a symbol callee", expression.span);
  }
  const valueType = target.valueType ?? "float";
  if (!semanticWgslPointerAtomicCallSupported(expression.callee.name, valueType)) {
    throw semanticWgslError(`semantic WGSL pointer atomic '${expression.callee.name}' is unsupported`, expression.span);
  }
  const [first, second] = expression.args.slice(1);
  const cas = semanticAtomicOperation(expression.callee.name) === "cas";
  const compare = cas ? first : undefined;
  const value = cas ? second : first;
  if (!value || cas && !compare) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing operand`, expression.span);
  const index = isCudaVectorType(valueType)
    ? emitFlatStorageVectorBaseIndex(target, ir, names, options)
    : emitFlatStorageIndex(target, ir, names, options);
  const args = [
    nameFor(semanticPointerBufferParamName(target.base), names),
    index,
    ...(compare ? [emitSemanticExpressionAs(compare, ir, names, wgslValueScalar(valueType), options, textureSpecializations).code] : []),
    emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations).code,
  ];
  return `${semanticPointerAtomicHelperName(expression.callee.name, valueType)}(${args.join(", ")})`;
}

function emitSemanticMember(
  expression: Extract<SemanticExpression, { readonly kind: "member" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const byteVectorMember = semanticDirectByteVectorMemberRef(expression, ir);
  if (byteVectorMember) return emitSemanticMemoryRead(byteVectorMember, ir, names, options);
  return emitSemanticMemberExpression(expression, ir, names, options).code;
}

function emitSemanticMemberExpression(
  expression: Extract<SemanticExpression, { readonly kind: "member" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): TypedWgslExpression {
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  if (expression.object.kind === "symbol") {
    switch (expression.object.name) {
      case "threadIdx":
        return ir.workgroupSize[axisIndex] === 1
          ? createTypedWgslZero("u32", expression.span)
          : createTypedWgslQualifiedAccess("local_id", expression.property, "u32", expression.span);
      case "blockIdx":
        return createTypedWgslQualifiedAccess("workgroup_id", expression.property, "u32", expression.span);
      case "blockDim":
        return createTypedWgslLiteral(`${ir.workgroupSize[axisIndex]}u`, "u32", expression.span);
      case "gridDim":
        return createTypedWgslQualifiedAccess("num_workgroups", expression.property, "u32", expression.span);
    }
  }
  if (semanticStorageVectorType(semanticExpressionVectorValueType(expression.object, ir?.functions)) === undefined) {
    throw semanticWgslError("semantic WGSL supports builtin vector members only", expression.span);
  }
  return createTypedWgslMemberAccess(
    emitSemanticExpression(expression.object, ir, names, options),
    semanticVectorFieldName(expression),
    semanticExpressionWgslType(expression, ir),
    expression.span,
  );
}

function semanticVectorFieldName(expression: Extract<SemanticExpression, { readonly kind: "member" }>): string {
  const valueType = semanticExpressionVectorValueType(expression.object);
  const fields = semanticStorageVectorFieldIndices(valueType, expression.property);
  return fields?.map((field) => ["x", "y", "z", "w"][field]).join("") ?? expression.property;
}

function emitSemanticUnary(
  expression: Extract<SemanticExpression, { readonly kind: "unary" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  const resultType = semanticExpressionWgslType(expression, ir);
  if (expression.operator === "*") {
    const declaration = semanticLocalStoragePointerDeclaration(ir, expression.argument);
    if (!declaration || expression.argument.kind !== "symbol") {
      throw semanticWgslError("semantic WGSL pointer dereference requires modeled local storage pointer", expression.span);
    }
    const valueType = declaration.target.valueType ?? "float";
    return createTypedWgslCall(
      semanticPointerReadHelperName(valueType),
      [
        createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(expression.argument.name), names), "u32", expression.argument.span),
        createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(expression.argument.name), names), "u32", expression.argument.span),
      ],
      resultType,
      expression.span,
    );
  }
  if (expression.operator === "!") {
    return emitTypedWgslUnary(
      "!",
      emitSemanticTruthinessExpression(expression.argument, ir, names, options),
      expression.span,
    );
  }
  if (expression.operator === "~") {
    const operandType = semanticExpressionWgslScalar(expression) === "u32" ? "u32" : "i32";
    return emitTypedWgslUnary(
      "~",
      emitSemanticExpressionAs(expression.argument, ir, names, operandType, options, textureSpecializations),
      expression.span,
    );
  }
  if (expression.operator === "+" || expression.operator === "-") {
    const operand = emitSemanticExpression(expression.argument, ir, names, options, textureSpecializations);
    if (operand.type !== resultType) {
      throw semanticWgslError(
        `WGSL unary '${expression.operator}' produces '${operand.type}', semantic IR declares '${resultType}'`,
        expression.span,
      );
    }
    return emitTypedWgslUnary(expression.operator, operand, expression.span);
  }
  throw semanticWgslError(`semantic WGSL does not support unary '${expression.operator}'`, expression.span);
}

function emitSemanticBinary(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (isSemanticStoragePointerNullComparison(expression)) {
    return createTypedWgslLiteral(expression.operator === "!=" ? "true" : "false", "bool", expression.span);
  }
  if (isSemanticStoragePointerIdentityComparison(expression, ir)) {
    const left = emitSemanticStoragePointerIdentity(expression.left, ir, names, options);
    const right = emitSemanticStoragePointerIdentity(expression.right, ir, names, options);
    const buffersEqual = emitTypedWgslBinary("==", left.buffer, right.buffer, expression.span);
    const basesEqual = emitTypedWgslBinary("==", left.base, right.base, expression.span);
    const equal = emitTypedWgslBinary("&&", buffersEqual, basesEqual, expression.span);
    return expression.operator === "==" ? equal : emitTypedWgslUnary("!", equal, expression.span);
  }
  if (LOGICAL_OPERATORS.has(expression.operator)) {
    return emitTypedWgslBinary(
      expression.operator as "&&" | "||",
      emitSemanticTruthinessExpression(expression.left, ir, names, options),
      emitSemanticTruthinessExpression(expression.right, ir, names, options),
      expression.span,
    );
  }
  if (isSemanticFloatVectorType(expression.valueType) && semanticWgslVectorBinaryOperatorSupported(expression.operator)) {
    const valueType = expression.valueType as CudaLiteScalarType;
    return emitTypedWgslBinary(
      expression.operator as Parameters<typeof emitTypedWgslBinary>[0],
      emitSemanticVectorOperandExpression(expression.left, valueType, ir, names, options, textureSpecializations),
      emitSemanticVectorOperandExpression(expression.right, valueType, ir, names, options, textureSpecializations),
      expression.span,
    );
  }
  if (expression.operator === "<<" || expression.operator === ">>") {
    const leftType = semanticExpressionWgslScalar(expression.left) === "u32" ? "u32" : "i32";
    const left = emitSemanticExpressionAs(expression.left, ir, names, leftType, options, textureSpecializations);
    const right = emitSemanticExpressionAs(expression.right, ir, names, "u32", options, textureSpecializations);
    return emitTypedWgslBinary(
      expression.operator,
      left,
      right,
      expression.span,
    );
  }
  const operandType = semanticBinaryOperandType(expression);
  const left = emitSemanticExpressionAs(expression.left, ir, names, operandType, options, textureSpecializations);
  const right = emitSemanticExpressionAs(expression.right, ir, names, operandType, options, textureSpecializations);
  const emitted = emitTypedWgslBinary(
    expression.operator as Parameters<typeof emitTypedWgslBinary>[0],
    left,
    right,
    expression.span,
  );
  const expected = semanticExpressionWgslType(expression, ir);
  if (emitted.type !== expected) {
    throw semanticWgslError(
      `WGSL binary '${expression.operator}' produces '${emitted.type}', semantic IR declares '${expected}'`,
      expression.span,
    );
  }
  return emitted;
}

function isSemanticStoragePointerNullComparison(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
): boolean {
  if (expression.operator !== "==" && expression.operator !== "!=") return false;
  const storageParam = (value: SemanticExpression): boolean => value.kind === "symbol" && value.addressSpace === "storage";
  const nullValue = (value: SemanticExpression): boolean =>
    value.kind === "literal" && value.literalKind === "number" && value.value === 0 ||
    value.kind === "symbol" && (value.name === "NULL" || value.name === "nullptr");
  return storageParam(expression.left) && nullValue(expression.right) ||
    storageParam(expression.right) && nullValue(expression.left);
}

function isSemanticStoragePointerIdentityComparison(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (ir === undefined || expression.operator !== "==" && expression.operator !== "!=") return false;
  return semanticStoragePointerSymbol(expression.left, ir) && semanticStoragePointerSymbol(expression.right, ir);
}

function semanticStoragePointerSymbol(expression: SemanticExpression, ir: SemanticKernelIrModule): boolean {
  if (expression.kind !== "symbol" || expression.addressSpace !== "storage") return false;
  return ir.params.some((param) => param.name === expression.name && param.pointer && param.addressSpace === "storage") ||
    ir.functions.some((fn) => fn.params.some((param) =>
      param.name === expression.name && param.pointer && param.addressSpace === "storage"));
}

function emitSemanticStoragePointerIdentity(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
): { readonly buffer: TypedWgslExpression; readonly base: TypedWgslExpression } {
  if (expression.kind !== "symbol") throw semanticWgslError("semantic storage pointer identity requires symbols", expression.span);
  const ownerParam = options.activeFunction === undefined
    ? undefined
    : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
      param.name === expression.name && param.pointer && param.addressSpace === "storage");
  if (ownerParam) {
    return {
      buffer: createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(expression.name), names), "u32", expression.span),
      base: createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(expression.name), names), "u32", expression.span),
    };
  }
  const bufferId = semanticStoragePointerBufferId(expression.name, ir);
  const root = ir.params.find((param) => param.name === expression.name && param.pointer && param.addressSpace === "storage");
  if (bufferId === undefined || root?.valueType === undefined || root.valueType === "void") throw semanticWgslError(`unknown storage pointer '${expression.name}'`, expression.span);
  return {
    buffer: createTypedWgslLiteral(`${bufferId}u`, "u32", expression.span),
    base: emitSemanticRootStoragePointerIdentityBase(root, ir, names, options, expression.span),
  };
}

function emitSemanticRootStoragePointerIdentityBase(
  root: SemanticKernelIrModule["params"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  span: SourceSpan,
): TypedWgslExpression {
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(root.name);
  const base = hasOffset
    ? convertTypedWgslExpression(
        createTypedWgslIdentifier(nameFor(storageOffsetSymbol(root.name), names), "i32", span),
        "u32",
      )
    : createTypedWgslZero("u32", span);
  if (!isCudaVectorType(root.valueType)) return base;
  const stride = cudaVectorLaneCount(root.valueType);
  return stride === 1
    ? base
    : emitTypedWgslBinary("*", base, createTypedWgslLiteral(`${stride}u`, "u32", span), span);
}

function emitSemanticVectorOperand(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  return emitSemanticVectorOperandExpression(expression, valueType, ir, names, options, textureSpecializations).code;
}

function emitSemanticVectorOperandExpression(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): TypedWgslExpression {
  if (semanticStorageVectorType(semanticExpressionVectorValueType(expression, ir?.functions)) !== undefined) {
    return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  }
  const laneCount = cudaVectorLaneCount(valueType);
  const vectorScalar = wgslVectorScalar(valueType);
  const scalar = emitSemanticExpressionAs(expression, ir, names, vectorScalar, options, textureSpecializations);
  const lanes = Array.from({ length: laneCount }, () =>
    convertTypedWgslExpression(scalar, vectorScalar, true)
  );
  const vectorType = wgslValueType(valueType);
  if (!isWgslVectorType(vectorType)) {
    throw new TypeError(`expected vector WGSL type for '${valueType}', received '${vectorType}'`);
  }
  return createTypedWgslConstructor(vectorType, lanes, expression.span);
}

function semanticBinaryOperandType(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): WgslValueType {
  const left = semanticExpressionWgslScalar(expression.left);
  const right = semanticExpressionWgslScalar(expression.right);
  if (
    COMPARISON_OPERATORS.has(expression.operator) &&
    expression.left.kind === "cast" &&
    expression.right.kind === "cast" &&
    expression.left.valueType === expression.right.valueType &&
    (expression.left.valueType === "int" || expression.left.valueType === "uint")
  ) {
    return wgslValueScalar(expression.left.valueType);
  }
  const result = wgslValueScalar(expression.valueType);
  if (left === "f32" || right === "f32" || result === "f32") return "f32";
  if (left === "u32" || right === "u32" || result === "u32") return "u32";
  return "i32";
}

function semanticMathCallOperandType(args: readonly SemanticExpression[]): WgslValueType {
  const types = args.map(semanticExpressionWgslScalar);
  if (types.includes("f32")) return "f32";
  if (types.includes("f16")) return "f16";
  if (types.includes("u32")) return "u32";
  return "i32";
}

function emitTruthiness(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (expression.kind === "symbol" && expression.addressSpace === "storage") return "true";
  if (semanticExpressionValueType(expression) === "bool" && semanticNativeBoolExpression(expression)) {
    return emitSemanticExpression(expression, ir, names, options).code;
  }
  if (expression.kind === "binary" && (COMPARISON_OPERATORS.has(expression.operator) || LOGICAL_OPERATORS.has(expression.operator))) {
    return emitSemanticBinary(expression, ir, names, options).code;
  }
  const scalar = semanticExpressionWgslScalar(expression);
  const zero = scalar === "u32" ? "0u" : scalar === "f32" ? "0.0" : scalar === "f16" ? "f16(0.0)" : "0";
  return emitTypedWgslBinary(
    "!=",
    emitSemanticExpressionAs(expression, ir, names, scalar, options),
    createTypedWgslLiteral(zero, scalar, expression.span),
    expression.span,
  ).code;
}

function emitSemanticMemoryRef(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (ref.fields.length > 0) throw semanticWgslError("semantic WGSL supports scalar memory refs only", ref.span);
  if (ref.addressSpace === "storage") {
    if (ref.indices.length === 0) throw semanticWgslError("semantic WGSL supports indexed storage refs only", ref.span);
    return `${nameFor(ref.base, names)}[${emitFlatStorageIndex(ref, ir, names, options)}]`;
  }
  if (ref.addressSpace === "constant") {
    const symbol = constantMemorySymbols(ir).find((item) => item.name === ref.base);
    if (!symbol) throw semanticWgslError(`unknown constant memory '${ref.base}'`, ref.span);
    return `${nameFor(ref.base, names)}[${emitFlatConstantIndex(symbol, ref.indices, ir, names, ref.span, options)}]`;
  }
  if (ref.addressSpace === "device-global") {
    const symbol = deviceGlobalMemorySymbols(ir).find((item) => item.name === ref.base);
    if (!symbol) throw semanticWgslError(`unknown device-global memory '${ref.base}'`, ref.span);
    return `${nameFor(ref.base, names)}[${emitFlatDeviceGlobalIndex(symbol, ref.indices, ir, names, ref.span)}]`;
  }
  if (ref.addressSpace === "local") {
    const pointerParam = semanticWgslFunctionLocalPointerParam(ir, ref.base, options.activeFunction ?? null);
    if (pointerParam) {
      if (pointerParam.dimensions.length > 0) {
        if (ref.indices.length !== 1) throw semanticWgslError(`local array pointer '${ref.base}' requires one flat index`, ref.span);
        const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32").code;
        return `(*${nameFor(ref.base, names)})[${nameFor(semanticPointerBaseParamName(ref.base), names)} + ${index}]`;
      }
      if (ref.indices.length > 1 || ref.indices[0] && !semanticExpressionIsZero(ref.indices[0])) {
        throw semanticWgslError(`local scalar pointer '${ref.base}' cannot be indexed`, ref.span);
      }
      return `*${nameFor(ref.base, names)}`;
    }
    const local = localArraySymbol(ir, ref.base);
    if (!local && ref.indices.length === 0) return nameFor(ref.base, names);
    if (!local) throw semanticWgslError(`unknown local memory '${ref.base}'`, ref.span);
    if (ref.indices.length === 1 && local.dimensions.length > 1) {
      const flat = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32").code;
      return `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(flat, local.dimensions)}`;
    }
    if (ref.indices.length !== local.dimensions.length) throw semanticWgslError(`local memory '${ref.base}' index rank mismatch`, ref.span);
    return `${nameFor(ref.base, names)}${ref.indices.map((index) => `[${emitSemanticExpressionAs(index, ir, names, "u32").code}]`).join("")}`;
  }
  if (ref.addressSpace === "shared") {
    if (semanticWgslFunctionSharedPointerParam(ir, ref.base)) {
      return emitSemanticSharedPointerMemoryRef(ref, ir, names, options);
    }
    const shared = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
    if (!shared) throw semanticWgslError(`unknown shared memory '${ref.base}'`, ref.span);
    if (shared.dimensions.length === 0) {
      if (ref.indices.length !== 0) throw semanticWgslError(`shared memory '${ref.base}' index rank mismatch`, ref.span);
      return nameFor(ref.base, names);
    }
    return `${nameFor(ref.base, names)}[${emitFlatSharedIndex(shared, ref.indices, ir, names)}]`;
  }
  throw semanticWgslError(`semantic WGSL does not support ${ref.addressSpace} memory refs`, ref.span);
}

function semanticWgslFunctionLocalPointerParam(
  ir: SemanticKernelIrModule,
  name: string,
  owner?: string | null,
): SemanticKernelIrModule["functions"][number]["params"][number] | undefined {
  if (owner !== undefined) {
    if (owner === null) return undefined;
    return ir.functions.find((fn) => fn.name === owner)?.params.find((param) =>
      param.name === name && param.pointer && param.addressSpace === "local"
    );
  }
  return ir.functions.flatMap((fn) => fn.params).find((param) =>
    param.name === name && param.pointer && param.addressSpace === "local"
  );
}

function emitSemanticAtomicLoad(ref: SemanticMemoryRef, memoryRef: string): string {
  const loaded = `atomicLoad(&${memoryRef})`;
  return ref.valueType === "float" || ref.valueType === "bf16" ? `bitcast<f32>(${loaded})` : loaded;
}

function semanticWgslSharedVectorMemoryRef(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (ref.addressSpace !== "shared" || !isSemanticFloatVectorType(ref.valueType)) return false;
  const pointer = semanticWgslFunctionSharedPointerParam(ir, ref.base);
  if (pointer) return pointer.valueType === ref.valueType;
  return sharedMemorySymbols(ir).some((symbol) => symbol.name === ref.base && symbol.valueType === ref.valueType);
}

function semanticWgslSharedScalarVectorView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  const valueType = ref.valueType;
  if (ref.addressSpace !== "shared" || !valueType || !isSemanticFloatVectorType(valueType) || ref.indices.length === 0) return false;
  const scalar = cudaVectorScalarType(valueType);
  return scalar !== undefined && sharedMemorySymbols(ir).some((symbol) =>
    symbol.name === ref.base && symbol.valueType === scalar,
  );
}

function semanticWgslSharedVectorScalarView(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (ref.addressSpace !== "shared" || ref.pointerBaseIsScalarLane !== true || ref.fields.length > 0 || ref.indices.length !== 1) return false;
  const scalar = ref.valueType;
  return scalar !== undefined && sharedMemorySymbols(ir).some((symbol) =>
    symbol.name === ref.base && isCudaVectorType(symbol.valueType) && cudaVectorScalarType(symbol.valueType) === scalar,
  );
}

function semanticCurandStateAddressSpace(expression: SemanticExpression | undefined): "function" | "storage" | "workgroup" | undefined {
  if (!expression || expression.kind !== "unary" || expression.operator !== "&") return undefined;
  const target = expression.argument;
  if (target.kind === "symbol" && target.addressSpace === "local") return "function";
  if (target.kind !== "index") return undefined;
  const ref = memoryRefFromIndexExpression(target);
  if (!ref) return undefined;
  if (ref.addressSpace === "local") return "function";
  if (ref.addressSpace === "shared") return "workgroup";
  if (ref.addressSpace === "storage" || ref.addressSpace === "device-global") return "storage";
  return undefined;
}

function semanticCurandStatePointer(
  expression: SemanticExpression | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): { readonly addressSpace: "function" | "storage" | "workgroup"; readonly expression: string } | undefined {
  const addressSpace = semanticCurandStateAddressSpace(expression);
  if (!addressSpace || !expression || expression.kind !== "unary" || expression.operator !== "&") return undefined;
  if (expression.argument.kind === "symbol") {
    return { addressSpace, expression: `&${nameFor(expression.argument.name, names)}` };
  }
  if (expression.argument.kind === "index") {
    const ref = memoryRefFromIndexExpression(expression.argument);
    if (!ref) return undefined;
    return { addressSpace, expression: `&${emitSemanticMemoryRef({ ...ref, valueType: "uint" }, ir, names, options)}` };
  }
  return undefined;
}

function memoryRefFromIndexExpression(expression: SemanticExpression): SemanticMemoryRef | undefined {
  if (expression.kind === "symbol" && expression.addressSpace === "device-global") {
    const valueType = requireSemanticValueType(expression.valueType, `device global '${expression.name}'`, expression.span);
    return {
      baseId: semanticMemoryIdFromSymbol(expression.id),
      base: expression.name,
      addressSpace: expression.addressSpace,
      valueType,
      indices: [],
      fields: [],
      span: expression.span,
    };
  }
  if (expression.kind !== "index") return undefined;
  const flattened = flattenMemoryRef(expression);
  if (!flattened || (flattened.base.addressSpace !== "storage" && flattened.base.addressSpace !== "shared" && flattened.base.addressSpace !== "constant" && flattened.base.addressSpace !== "device-global" && flattened.base.addressSpace !== "local")) return undefined;
  return {
    baseId: semanticMemoryIdFromSymbol(flattened.base.id),
    base: flattened.base.name,
    addressSpace: flattened.base.addressSpace,
    valueType: expression.valueType,
    ...(expression.target.kind === "symbol" && expression.target.valueType !== undefined ? { containerValueType: expression.target.valueType } : {}),
    ...(expression.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(expression.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: expression.pointerBaseUnitBytes }),
    ...(expression.packedByteLanes === undefined ? {} : { packedByteLanes: expression.packedByteLanes }),
    indices: flattened.indices,
    fields: [],
    span: expression.span,
  };
}

function flattenMemoryRef(expression: SemanticExpression): {
  readonly base: Extract<SemanticExpression, { readonly kind: "symbol" }>;
  readonly indices: readonly SemanticExpression[];
} | undefined {
  if (expression.kind === "symbol") return { base: expression, indices: [] };
  if (expression.kind === "cast" && expression.pointer) return flattenMemoryRef(expression.expression);
  if (expression.kind !== "index") return undefined;
  const target = flattenMemoryRef(expression.target);
  if (!target) return undefined;
  return { base: target.base, indices: [...target.indices, expression.index] };
}

function unsupportedMemoryRef(span: SourceSpan): SemanticMemoryRef {
  return { baseId: createUnresolvedSemanticMemoryId("", span), base: "", addressSpace: "unknown", valueType: "int", indices: [], fields: [], span };
}

function surfaceHandleForName(name: string, ir: SemanticKernelIrModule): number | undefined {
  const index = surfaceSymbols(ir).findIndex((surface) => surface.name === name);
  return index < 0 ? undefined : index;
}

function surfaceWidthField(name: string): string {
  return `${name}_width`;
}

function surfaceHeightField(name: string): string {
  return `${name}_height`;
}

function surfaceReadHelperName(name: string, names: ReadonlyMap<string, string>): string {
  return `bg_sem_surf2dread_${nameFor(name, names)}`;
}

function localArraySymbol(ir: SemanticKernelIrModule, name: string): SemanticKernelIrModule["params"][number] | undefined {
  return ir.memory.find((symbol) => symbol.kind === "local" && symbol.name === name && symbol.dimensions.length > 0) ??
    semanticFunctionLocalArraySymbol(ir, name);
}

function semanticFunctionLocalArraySymbol(
  ir: SemanticKernelIrModule,
  name: string,
): SemanticKernelIrModule["params"][number] | undefined {
  const matches: SemanticKernelIrModule["params"][number][] = [];
  const collect = (operations: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of operations) {
      if (operation.kind === "declare" && operation.target.addressSpace === "local" && operation.target.name === name && operation.target.dimensions.length > 0) {
        matches.push(operation.target);
      }
      if (operation.kind === "branch") {
        collect(operation.consequent);
        collect(operation.alternate);
      }
      if (operation.kind === "loop" || operation.kind === "block") collect(operation.body);
    }
  };
  for (const fn of ir.functions) collect(fn.body);
  const first = matches[0];
  if (!first) return undefined;
  return matches.every((item) =>
    item.valueType === first.valueType &&
    item.dimensions.length === first.dimensions.length &&
    item.dimensions.every((dimension, index) => dimension === first.dimensions[index])
  ) ? first : undefined;
}

function emitLocalArrayType(symbol: SemanticKernelIrModule["params"][number]): string {
  return emitSemanticNestedArrayType(symbol.dimensions, wgslValueType(symbol.valueType));
}

function emitInitializedConstantArray(
  symbol: SemanticKernelIrModule["memory"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  const elementType = wgslScalar(symbol.valueType);
  const length = totalElements(symbol.dimensions);
  const arrayType = `array<${elementType}, ${length}>`;
  const values = flattenInitializerExpressions(symbol.init ?? zeroExpression(symbol.span))
    .slice(0, length)
    .map((value) => emitSemanticExpressionAs(value, ir, names, wgslValueScalar(symbol.valueType)).code);
  while (values.length < length) values.push(zeroForType(elementType));
  return `const ${nameFor(symbol.name, names)}: ${arrayType} = ${arrayType}(${values.join(", ")});`;
}

function semanticVectorConstantInitCallSupported(expression: SemanticExpression): boolean {
  return expression.kind === "call" && semanticWgslVectorConstructorSupported(expression, "any");
}

function semanticVectorConstantInitExpressions(expression: SemanticExpression): readonly SemanticExpression[] {
  if (expression.kind === "initializer") return flattenInitializerExpressions(expression);
  if (expression.kind === "call" && semanticVectorConstantInitCallSupported(expression)) return expression.args;
  return [expression];
}

function emitLocalArrayFill(
  name: string,
  dimensions: readonly number[],
  value: string,
  indentLevel: number,
  indexes: readonly string[] = [],
): readonly string[] {
  if (indexes.length === dimensions.length) {
    return [`${"  ".repeat(indentLevel)}${name}${indexes.map((index) => `[${index}]`).join("")} = ${value};`];
  }
  const loopName = `fill_${name}_${indexes.length}`;
  const lines = [
    `${"  ".repeat(indentLevel)}for (var ${loopName}: i32 = 0; ${loopName} < ${dimensions[indexes.length] ?? 0}; ${loopName} = ${loopName} + 1) {`,
  ];
  lines.push(...emitLocalArrayFill(name, dimensions, value, indentLevel + 1, [...indexes, loopName]));
  lines.push(`${"  ".repeat(indentLevel)}}`);
  return lines;
}

function emitSharedType(symbol: SemanticKernelIrModule["memory"][number], atomic: boolean): string {
  if (symbol.valueType === "uchar") {
    const bytes = symbol.dimensions.length === 0 ? 1 : totalElements(symbol.dimensions);
    return symbol.dimensions.length === 0 ? "atomic<u32>" : `array<atomic<u32>, ${Math.ceil(bytes / 4)}>`;
  }
  if (atomic && isCudaVectorType(symbol.valueType)) {
    const lanes = cudaVectorLaneCount(symbol.valueType);
    const elements = symbol.dimensions.length === 0 ? lanes : totalElements(symbol.dimensions) * lanes;
    return `array<atomic<${wgslAtomicScalar(symbol.valueType)}>, ${elements}>`;
  }
  const element = atomic ? `atomic<${wgslAtomicScalar(symbol.valueType)}>` : wgslValueType(symbol.valueType);
  if (symbol.dimensions.length === 0) return element;
  return emitSemanticFlatArrayType(symbol.dimensions, element);
}

function emitSemanticPackedSharedByteHelpers(): readonly string[] {
  return [
    `fn ${PACKED_SHARED_U8_STORE}(word: ptr<workgroup, atomic<u32>>, shift: u32, value: u32) {`,
    "  let mask = 255u << shift;",
    "  var old_bits = atomicLoad(word);",
    "  loop {",
    "    let new_bits = (old_bits & ~mask) | ((value & 255u) << shift);",
    "    let result = atomicCompareExchangeWeak(word, old_bits, new_bits);",
    "    if (result.exchanged) { return; }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
    `fn ${PACKED_SHARED_U8_ADD}(word: ptr<workgroup, atomic<u32>>, shift: u32, delta: i32) -> u32 {`,
    "  let mask = 255u << shift;",
    "  var old_bits = atomicLoad(word);",
    "  loop {",
    "    let old_value = (old_bits >> shift) & 255u;",
    "    let new_value = u32(i32(old_value) + delta) & 255u;",
    "    let new_bits = (old_bits & ~mask) | (new_value << shift);",
    "    let result = atomicCompareExchangeWeak(word, old_bits, new_bits);",
    "    if (result.exchanged) { return old_value; }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

function emitFlatStorageIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null)) {
    const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options).code);
    terms.unshift(`i32(${nameFor(semanticPointerBaseParamName(ref.base), names)})`);
    return `u32(${terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`})`;
  }
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
  if (!hasOffset && ref.indices.length === 0) return "0u";
  if (!hasOffset && ref.indices.length === 1) {
    return emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
  }
  const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options).code);
  if (hasOffset) {
    terms.unshift(nameFor(storageOffsetSymbol(ref.base), names));
  }
  const expression = terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
  return `u32(${expression})`;
}

function emitSemanticRootStorageIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
  if (!hasOffset && ref.indices.length === 0) return "0u";
  if (!hasOffset && ref.indices.length === 1) {
    return emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
  }
  const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options).code);
  if (hasOffset) terms.unshift(nameFor(storageOffsetSymbol(ref.base), names));
  const expression = terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
  return `u32(${expression})`;
}

function emitFlatStorageVectorBaseIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
  if (pointerParam) {
    const indexTerms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "u32", options).code);
    const valueType = semanticStorageVectorType(ref.containerValueType) ?? semanticStorageVectorType(pointerParam.valueType);
    const stride = valueType === undefined ? 1 : cudaVectorLaneCount(valueType);
    const index = indexTerms.length === 0 ? "0u" : indexTerms.length === 1 ? indexTerms[0]! : `(${indexTerms.join(" + ")})`;
    const offset = stride === 1 ? index : `(${index} * ${stride}u)`;
    return `(${nameFor(semanticPointerBaseParamName(ref.base), names)} + ${offset})`;
  }
  const base = emitFlatStorageIndex({ ...ref, valueType: "float" }, ir, names, options);
  const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
  const valueType = semanticStorageVectorType(ref.containerValueType) ?? semanticStorageVectorType(root?.valueType);
  const stride = valueType === undefined ? 1 : cudaVectorLaneCount(valueType);
  return stride === 1 ? base : `(${base} * ${stride}u)`;
}

function emitFlatSharedIndex(
  symbol: SemanticKernelIrModule["memory"][number],
  indices: readonly SemanticExpression[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (indices.length === 0) return "0u";
  if (indices.length === 1) return emitSemanticExpressionAs(indices[0]!, ir, names, "u32").code;
  return emitSemanticFlatRankedIndex(
    "shared memory",
    symbol.name,
    symbol.dimensions,
    indices,
    symbol.span,
    (index) => emitSemanticExpressionAs(index, ir, names, "u32").code,
  );
}

function emitFlatDeviceGlobalIndex(
  symbol: SemanticKernelIrModule["memory"][number],
  indices: readonly SemanticExpression[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  span: SourceSpan,
): string {
  if (symbol.dimensions.length === 0) {
    if (indices.length > 1) throw semanticWgslError(`device-global memory '${symbol.name}' index rank mismatch`, span);
    return indices[0] ? emitSemanticExpressionAs(indices[0], ir, names, "u32").code : "0u";
  }
  if (indices.length === 1 && symbol.dimensions.length > 1) {
    return emitSemanticExpressionAs(indices[0]!, ir, names, "u32").code;
  }
  if (indices.length !== symbol.dimensions.length) {
    throw semanticWgslError(`device-global memory '${symbol.name}' index rank mismatch`, span);
  }
  return emitSemanticFlatRankedIndex(
    "device-global memory",
    symbol.name,
    symbol.dimensions,
    indices,
    span,
    (index) => emitSemanticExpressionAs(index, ir, names, "u32").code,
  );
}

function emitFlatConstantIndex(
  symbol: SemanticKernelIrModule["memory"][number],
  indices: readonly SemanticExpression[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  span: SourceSpan,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (symbol.dimensions.length === 0) {
    if (indices.length !== 1) throw semanticWgslError(`constant memory '${symbol.name}' index rank mismatch`, span);
    return emitSemanticExpressionAs(indices[0]!, ir, names, "u32", options).code;
  }
  if (indices.length === 1 && symbol.dimensions.length > 1) {
    return emitSemanticExpressionAs(indices[0]!, ir, names, "u32", options).code;
  }
  if (indices.length !== symbol.dimensions.length) {
    throw semanticWgslError(`constant memory '${symbol.name}' index rank mismatch`, span);
  }
  return emitSemanticFlatRankedIndex(
    "constant memory",
    symbol.name,
    symbol.dimensions,
    indices,
    span,
    (index) => emitSemanticExpressionAs(index, ir, names, "u32", options).code,
  );
}

function emitFlatLocalArrayIndexes(flat: string, dimensions: readonly number[]): string {
  return emitSemanticFlatLocalArrayIndexes(flat, dimensions);
}

function collectOperationNames(
  operation: SemanticKernelIrOperation,
  names: Set<string>,
): void {
  if (operation.kind === "declare") {
    names.add(operation.target.name);
    if (operation.target.pointer) {
      names.add(semanticPointerBufferParamName(operation.target.name));
      names.add(semanticPointerBaseParamName(operation.target.name));
    }
  }
  if (operation.kind === "pointer-rebind") {
    names.add(semanticPointerBufferParamName(operation.target.name));
    names.add(semanticPointerBaseParamName(operation.target.name));
  }
  if (operation.kind === "branch") {
    for (const child of [...operation.consequent, ...operation.alternate]) collectOperationNames(child, names);
  }
  if (operation.kind === "loop") {
    if (operation.init && isSemanticKernelIrOperation(operation.init)) collectOperationNames(operation.init, names);
    for (const child of operation.body) collectOperationNames(child, names);
  }
}

function semanticExpressionWgslScalar(expression: SemanticExpression): WgslValueType {
  switch (expression.kind) {
    case "call": {
      if (expression.callee.kind === "symbol") {
        if (expression.callee.name === "__half2_as_uint") return "u32";
        if (expression.callee.name === "__low2half" || expression.callee.name === "__high2half") return "f16";
        if (expression.callee.name === "__low2float" || expression.callee.name === "__high2float") return "f32";
        if (expression.callee.name === "__low2bfloat16" || expression.callee.name === "__high2bfloat16") return "f32";
        if (expression.callee.name === "__bfloat162_as_uint" || expression.callee.name === "__nv_bfloat162_as_uint") return "u32";
        const mathCallee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
        if (mathCallee && semanticMathCallReturnsFloat(expression.callee.name)) return "f32";
        if (mathCallee === "hadd" && expression.valueType === "half") return "f16";
        if (mathCallee && semanticMathCallReturnsHalf(mathCallee)) return "f16";
        if (mathCallee && (mathCallee.startsWith("half_to_int_") || mathCallee.startsWith("half_to_short_") || mathCallee === "half_as_short")) return "i32";
        if (mathCallee === "half_isinf") return "i32";
        if (mathCallee && (mathCallee.startsWith("half_to_uint_") || mathCallee.startsWith("half_to_ushort_") || mathCallee === "half_as_ushort" || mathCallee === "float_to_fp8" || mathCallee.startsWith("half_") && !semanticMathCallReturnsHalf(mathCallee))) return "u32";
        if (mathCallee && mathCallee.startsWith("bf16_to_int_")) return "i32";
        if (mathCallee && mathCallee === "bf16_as_short") return "i32";
        if (mathCallee && (mathCallee.startsWith("bf16_to_uint_") || mathCallee === "bf16_as_ushort")) return "u32";
        if (mathCallee === "mul24" || mathCallee === "mulhi") return "i32";
        if (mathCallee === "umul24" || mathCallee === "umulhi" || mathCallee === "umul" || mathCallee === "umin") return "u32";
      }
      if (semanticWgslMathCallSupported(expression) && (expression.valueType === undefined || expression.valueType === "float")) return "f32";
      const atomicType = semanticAtomicCallValueType(expression);
      return atomicType ? wgslAtomicScalar(atomicType) : wgslValueScalar(expression.valueType);
    }
    case "texture-read":
      return wgslValueScalar(expression.valueType);
    case "surface-read":
      return wgslValueScalar(expression.valueType);
    case "binary": {
      const left = semanticExpressionWgslScalar(expression.left);
      const right = semanticExpressionWgslScalar(expression.right);
      const result = wgslValueScalar(expression.valueType);
      if (left === "f32" || right === "f32" || result === "f32") return "f32";
      if (left === "u32" || right === "u32" || result === "u32") return "u32";
      return "i32";
    }
    case "conditional": {
      const consequent = semanticExpressionWgslScalar(expression.consequent);
      const alternate = semanticExpressionWgslScalar(expression.alternate);
      const result = wgslValueScalar(expression.valueType);
      if (consequent === "f32" || alternate === "f32" || result === "f32") return "f32";
      if (consequent === "u32" || alternate === "u32" || result === "u32") return "u32";
      return "i32";
    }
    case "sequence":
      return expression.expressions.length > 0
        ? semanticExpressionWgslScalar(expression.expressions.at(-1)!)
        : wgslValueScalar(expression.valueType);
    default:
      return wgslValueScalar(semanticExpressionValueType(expression));
  }
}

function semanticMathCallReturnsFloat(name: string): boolean {
  const callee = SEMANTIC_MATH_CALLS.get(name);
  return callee === "builtin_inf" || callee === "uint_as_float" || callee === "int_as_float" || callee === "half_to_float";
}

function semanticMathCallReturnsHalf(callee: string): boolean {
  return callee === "to_half" ||
    callee === "int_to_half" ||
    callee === "uint_to_half" ||
    callee.startsWith("float_to_half_") ||
    callee.startsWith("int_to_half_") ||
    callee.startsWith("uint_to_half_") ||
    callee.startsWith("short_to_half_") ||
    callee.startsWith("ushort_to_half_") ||
    callee === "short_as_half" ||
    callee === "ushort_as_half" ||
    callee === "fp8_to_half" ||
    callee === "half_abs" ||
    callee === "half_ceil" ||
    callee === "half_floor" ||
    callee === "half_rcp" ||
    callee === "half_rsqrt" ||
    callee === "half_sqrt" ||
    callee === "half_trunc" ||
    callee === "half_neg" ||
    callee === "half_add" ||
    callee === "half_add_sat" ||
    callee === "half_sub" ||
    callee === "half_sub_sat" ||
    callee === "half_mul" ||
    callee === "half_mul_sat" ||
    callee === "half_div" ||
    callee === "half_fma" ||
    callee === "half_fma_sat" ||
    callee === "half_exp" ||
    callee === "half_min" ||
    callee === "half_max" ||
    callee === "half_min_nan" ||
    callee === "half_max_nan";
}

function emitNumberLiteral(value: number, valueType: CudaLiteScalarType | undefined, expectedType?: WgslValueType): string {
  const type = expectedType ?? wgslScalar(valueType);
  if (type === "f32" && Math.abs(value) === 3.4028234663852886e38) {
    return value < 0 ? "bitcast<f32>(0xff7fffffu)" : "bitcast<f32>(0x7f7fffffu)";
  }
  if (!Number.isFinite(value)) {
    const f32 = Number.isNaN(value) ? "bg_f32_nan()" : value < 0 ? "-bg_f32_inf()" : "bg_f32_inf()";
    return type === "f16" ? `f16(${f32})` : f32;
  }
  if (type === "bool") return value === 0 ? "false" : "true";
  if (type === "u32") return `${Math.trunc(value) >>> 0}u`;
  if (type === "i32" && value > 2147483647) return `bitcast<i32>(${Math.trunc(value) >>> 0}u)`;
  if (type === "i32") return String(Math.trunc(value));
  const literal = String(value);
  const floatLiteral = /[.eE]/u.test(literal) ? literal : `${literal}.0`;
  if (type === "f16") return `f16(${floatLiteral})`;
  return floatLiteral;
}

function zeroExpression(span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: 0, valueType: "int", span };
}

function bindingIndexFor(bindings: readonly WgslKernelBindingInput[], name: string): number {
  const binding = bindings.find((item) => item.name === name)?.binding;
  return binding ?? 0;
}

function semanticAtomicStorageNames(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "storage") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicStorageNamesFromOperation(operation, functions)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicStorageNames(operation.consequent, functions)) names.add(name);
      for (const name of semanticAtomicStorageNames(operation.alternate, functions)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicStorageNames([operation.init], functions)) names.add(name);
      }
      for (const name of semanticAtomicStorageNames(operation.body, functions)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicStorageNames(operation.continuing, functions)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicStorageNames(operation.body, functions)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicDeviceGlobalNames(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "device-global") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicDeviceGlobalNamesFromOperation(operation, functions)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.consequent, functions)) names.add(name);
      for (const name of semanticAtomicDeviceGlobalNames(operation.alternate, functions)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicDeviceGlobalNames([operation.init], functions)) names.add(name);
      }
      for (const name of semanticAtomicDeviceGlobalNames(operation.body, functions)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicDeviceGlobalNames(operation.continuing, functions)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.body, functions)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicSharedNames(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "shared") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicSharedNamesFromOperation(operation, functions)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicSharedNames(operation.consequent, functions)) names.add(name);
      for (const name of semanticAtomicSharedNames(operation.alternate, functions)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicSharedNames([operation.init], functions)) names.add(name);
      }
      for (const name of semanticAtomicSharedNames(operation.body, functions)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicSharedNames(operation.continuing, functions)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicSharedNames(operation.body, functions)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicSharedNamesFromOperation(
  operation: SemanticKernelIrOperation,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  if (operation.kind === "call") {
    const fn = semanticFunctionForCall(operation, functions);
    const expression: Extract<SemanticExpression, { readonly kind: "call" }> = {
      kind: "call",
      callee: { kind: "symbol", id: fn ? createSemanticSymbolId("function", fn.name, fn.span) : createUnresolvedSemanticSymbolId(operation.callee, operation.span), name: operation.callee, addressSpace: "function", span: operation.span },
      args: operation.args,
      valueType: "void",
      span: operation.span,
    };
    for (const name of semanticAtomicSharedNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const expression of expressions) {
    for (const name of semanticAtomicSharedNamesFromExpression(expression, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicSharedNamesFromExpression(
  expression: SemanticExpression,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === "shared") names.add(target.base);
  if (expression.kind === "call") {
    for (const name of semanticAtomicSharedNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicSharedNamesFromExpression(child, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicSharedNamesFromFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const callee = expression.callee;
  if (callee.kind !== "symbol") return new Set();
  const fn = functions.find((item) => item.name === callee.name);
  if (!fn) return new Set();
  const pointerAtomicParams = semanticFunctionSharedPointerAtomicParams(fn);
  const names = new Set<string>();
  for (const [index, param] of fn.params.entries()) {
    if (!pointerAtomicParams.has(param.name)) continue;
    const ref = semanticPointerArgMemoryRef(expression.args[index] ?? zeroExpression(expression.span));
    if (ref?.addressSpace === "shared") names.add(ref.base);
  }
  return names;
}

function semanticFunctionSharedPointerAtomicParams(
  fn: SemanticKernelIrModule["functions"][number],
): ReadonlySet<string> {
  const pointerParams = new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "shared").map((param) => param.name));
  const names = new Set<string>();
  for (const name of semanticAtomicSharedNames(fn.body)) {
    if (!pointerParams.has(name)) continue;
    names.add(name);
    const param = fn.params.find((candidate) => candidate.name === name);
    const alias = param === undefined ? undefined : semanticFunctionParamAliasName(fn, param);
    if (alias !== undefined) names.add(alias);
  }
  return names;
}

function semanticWgslFunctionSharedPointerAtomicParam(ir: SemanticKernelIrModule, name: string): boolean {
  return ir.functions.some((fn) => semanticFunctionSharedPointerAtomicParams(fn).has(name));
}

function semanticAtomicStorageNamesFromOperation(
  operation: SemanticKernelIrOperation,
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  if (operation.kind === "call") {
    const fn = semanticFunctionForCall(operation, functions);
    const expression: Extract<SemanticExpression, { readonly kind: "call" }> = {
      kind: "call",
      callee: { kind: "symbol", id: fn ? createSemanticSymbolId("function", fn.name, fn.span) : createUnresolvedSemanticSymbolId(operation.callee, operation.span), name: operation.callee, addressSpace: "function", span: operation.span },
      args: operation.args,
      valueType: "void",
      span: operation.span,
    };
    for (const name of semanticAtomicStorageNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const expression of expressions) {
    for (const name of semanticAtomicStorageNamesFromExpression(expression, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicStorageNamesFromExpression(
  expression: SemanticExpression,
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === "storage") names.add(target.base);
  if (expression.kind === "call") {
    for (const name of semanticAtomicStorageNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicStorageNamesFromExpression(child, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicStorageNamesFromFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const callee = expression.callee;
  if (callee.kind !== "symbol") return new Set();
  const fn = functions.find((item) => item.name === callee.name);
  if (!fn) return new Set();
  const pointerAtomicParams = semanticFunctionStoragePointerAtomicParams(fn);
  const names = new Set<string>();
  for (const [index, param] of fn.params.entries()) {
    if (!pointerAtomicParams.has(param.name)) continue;
    const ref = semanticPointerArgMemoryRef(expression.args[index] ?? zeroExpression(expression.span));
    if (ref?.addressSpace === "storage") names.add(ref.base);
  }
  return names;
}

function semanticFunctionStoragePointerAtomicParams(
  fn: SemanticKernelIrModule["functions"][number],
): ReadonlySet<string> {
  const pointerParams = new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "storage").map((param) => param.name));
  const names = new Set<string>();
  for (const name of semanticAtomicStorageNames(fn.body)) {
    if (pointerParams.has(name)) names.add(name);
  }
  return names;
}

function semanticAtomicDeviceGlobalNamesFromOperation(
  operation: SemanticKernelIrOperation,
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const name of semanticAtomicNamesFromOperation(operation, "device-global")) names.add(name);
  if (operation.kind === "call") {
    const fn = semanticFunctionForCall(operation, functions);
    if (fn) {
      const atomicParams = semanticFunctionStoragePointerAtomicParams(fn);
      for (const [index, param] of fn.params.entries()) {
        if (!atomicParams.has(param.name)) continue;
        const ref = semanticPointerArgMemoryRef(operation.args[index] ?? zeroExpression(operation.span));
        if (ref?.addressSpace === "device-global") names.add(ref.base);
      }
    }
  }
  return names;
}

function semanticAtomicNamesFromOperation(
  operation: SemanticKernelIrOperation,
  addressSpace: "storage" | "device-global" | "shared",
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  for (const expression of expressions) {
    for (const name of semanticAtomicNamesFromExpression(expression, addressSpace)) names.add(name);
  }
  return names;
}

function semanticAtomicNamesFromExpression(
  expression: SemanticExpression,
  addressSpace: "storage" | "device-global" | "shared",
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === addressSpace) names.add(target.base);
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicNamesFromExpression(child, addressSpace)) names.add(name);
  }
  return names;
}

function semanticAtomicCallTarget(expression: Extract<SemanticExpression, { readonly kind: "call" }>): SemanticMemoryRef | undefined {
  if (
    expression.callee.kind !== "symbol" ||
    !isSemanticAtomicCallName(expression.callee.name)
  ) return undefined;
  const firstArg = expression.args[0];
  if (!firstArg) return undefined;
  if (
    firstArg.kind === "cast" &&
    firstArg.pointer &&
    (firstArg.valueType === "uint" || firstArg.valueType === "int") &&
    firstArg.expression.kind === "unary" &&
    firstArg.expression.operator === "&" &&
    firstArg.expression.argument.kind === "index"
  ) {
    const ref = memoryRefFromIndexExpression(firstArg.expression.argument);
    return ref ? { ...ref, valueType: firstArg.valueType } : undefined;
  }
  if (firstArg.kind === "unary" && firstArg.operator === "&" && firstArg.argument.kind === "index") {
    return memoryRefFromIndexExpression(firstArg.argument);
  }
  if (
    firstArg.kind === "unary" &&
    firstArg.operator === "&" &&
    firstArg.argument.kind === "symbol" &&
    (firstArg.argument.addressSpace === "device-global" || firstArg.argument.addressSpace === "shared")
  ) {
    const valueType = requireSemanticValueType(firstArg.argument.valueType, `atomic target '${firstArg.argument.name}'`, firstArg.argument.span);
    return {
      baseId: semanticMemoryIdFromSymbol(firstArg.argument.id),
      base: firstArg.argument.name,
      addressSpace: firstArg.argument.addressSpace,
      valueType,
      indices: [],
      fields: [],
      span: firstArg.argument.span,
    };
  }
  if (firstArg.kind === "index") return memoryRefFromIndexExpression(firstArg);
  if (firstArg.kind === "symbol" && (firstArg.addressSpace === "storage" || firstArg.addressSpace === "shared" || firstArg.addressSpace === "local")) {
    const valueType = requireSemanticValueType(firstArg.valueType, `atomic target '${firstArg.name}'`, firstArg.span);
    return {
      baseId: semanticMemoryIdFromSymbol(firstArg.id),
      base: firstArg.name,
      addressSpace: firstArg.addressSpace,
      valueType,
      indices: [],
      fields: [],
      span: firstArg.span,
    };
  }
  return undefined;
}

function semanticAtomicCallValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  if (expression.kind !== "call") return undefined;
  return semanticAtomicCallTarget(expression)?.valueType;
}

function nameFor(name: string, names: ReadonlyMap<string, string>): string {
  if (isCudaBuiltinVectorSymbolName(name)) return name;
  return names.get(name) ?? safeWgslIdentifier(name);
}

function semanticWgslError(message: string, span: SourceSpan): CudaLiteCompilerError {
  const diagnostic: CudaLiteDiagnostic = {
    code: "semantic-wgsl-unsupported",
    severity: "error",
    message,
    span,
  };
  return new CudaLiteCompilerError(message, [diagnostic]);
}
