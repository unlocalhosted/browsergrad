import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslValueType,
} from "@unlocalhosted/browsergrad-kernels";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMatrixTileRef,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { semanticInlineAsmLdmatrixAssignments, walkSemanticOperations } from "./semantic_ir.js";
import {
  isSemanticKernelIrOperation,
  semanticExpressionChildren,
  semanticOperationsReferenceRoot,
} from "./semantic_ir_walk.js";
import type {
  CudaLiteDiagnostic,
  CudaLiteScalarType,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import {
  emitInlineArithmeticWgsl,
  emitInlineBytePermWgsl,
  emitInlineCompareWgsl,
  emitInlineLop3Wgsl,
  emitInlineMinMaxWgsl,
  emitInlineSelectWgsl,
  emitInlineShiftWgsl,
  emitInlineUnaryIntWgsl,
} from "./features/inline_ptx/wgsl.js";
import { semanticPtxIntegerCallInfo } from "./semantic_inline_ptx.js";
import { promotedCudaScalarType } from "./wgsl_value_conversion.js";
import { sizeofCudaType } from "./type_layout.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { createWgslNameMap, safeWgslIdentifier } from "./wgsl_names.js";
import { isCudaBuiltinVectorSymbolName } from "./cuda_builtin_symbols.js";
import { emitBfloatConversionHelpers, emitCurandHelpers, emitFp8Helpers, emitHalfConversionHelpers, emitSpecialFloatConstantHelpers } from "./wgsl_support_helpers.js";
import {
  SEMANTIC_BF162_BINARY_VECTOR_CALLS,
  SEMANTIC_BF162_BOOL_COMPARISON_CALLS,
  SEMANTIC_BF162_MASK_COMPARISON_CALLS,
  SEMANTIC_BF162_MINMAX_VECTOR_CALLS,
  SEMANTIC_BF162_SCALAR_CALLS,
  SEMANTIC_BF162_TERNARY_VECTOR_CALLS,
  SEMANTIC_BF162_UNARY_VECTOR_CALLS,
  SEMANTIC_BF162_VECTOR_CALLS,
  SEMANTIC_BF162_VECTOR_COMPARISON_CALLS,
  SEMANTIC_HALF2_SCALAR_CALLS,
  SEMANTIC_HALF2_VECTOR_CALLS,
  semanticBf162CallArgumentsSupported,
  isSemanticHalf2BooleanComparisonCall,
  isSemanticHalf2ComparisonCall,
  isSemanticHalf2MaskComparisonCall,
  isSemanticHalf2UnaryCall,
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
  SEMANTIC_CURAND_VECTOR_CALLS,
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
  semanticAssertCallSupported,
  semanticNoopCallSupported,
  semanticPrintfCallSupported,
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
import {
  cudaLiteFlatIndicesForDimensions as flatIndicesForDimensions,
  cudaLiteTotalElements as totalElements,
} from "./cuda_lite_values.js";
import { cudaAddressSpacePredicateKind } from "./cuda_pointer_calls.js";
import {
  isCudaBarrierCallName,
  isCudaCooperativeBarrierCallName,
  isCudaFenceCallName,
} from "./cuda_sync_calls.js";
import { isCudaCpAsyncFenceCall } from "./cuda_cp_async.js";
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
  semanticLocalArrayFillCallSupported,
  semanticLocalArrayInitSupported as semanticLocalArrayInitContractSupported,
} from "./semantic_local_arrays.js";
import {
  semanticFunctionArgSupported as semanticFunctionArgContractSupported,
  semanticFunctionBodyShapeSupported as semanticFunctionBodyShapeContractSupported,
  semanticFunctionLocalParamValueTypesSupported,
  semanticFunctionParamContractSupported,
  semanticPointerFunctionBodySupported as semanticPointerFunctionBodyContractSupported,
} from "./semantic_function_calls.js";
import { semanticPointerArgumentMemoryRef as semanticPointerArgMemoryRef } from "./semantic_pointer_arguments.js";
import { semanticDirectByteStorageParamSupported } from "./semantic_byte_storage.js";
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
  emitSemanticCooperativeReduceCall,
  emitSemanticCooperativeReduceHelper,
  semanticCooperativeReduceHelperFor,
  semanticCooperativeReduceHelpers,
  semanticCooperativeReduceValue,
  semanticWgslCooperativeGroupCallSupported,
  semanticWgslCooperativeReduceCallSupported,
} from "./semantic_wgsl_cooperative.js";
import {
  semanticCooperativeGroupInfo,
  semanticCooperativeGroupRankParamName,
  semanticCooperativeGroupSizeParamName,
} from "./semantic_cooperative_groups.js";
import { emitSemanticNumericHelpers } from "./semantic_wgsl_numeric_helpers.js";
import {
  emitRoundEvenWgsl,
  emitSemanticVCompareExpression,
  emitSemanticVPackedAbsDiffExpression,
  emitSemanticVPackedAverageExpression,
  emitSemanticVPackedSadExpression,
  emitSemanticVPackedUnaryExpression,
  emitSemanticVSetExpression,
  emitSemanticViMinMax16x2Expression,
  emitSemanticViadd16x2Expression,
  halfConversionModeLiteral,
  wgslReluBf162,
  wgslReluBfloat16,
  wgslRoundBfloat16,
  wgslSaturateBf162,
  wgslSaturateBfloat16,
  wgslSaturateHalf,
  wgslSaturateHalf2,
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
}

const UNIFORM_PARAMS_NAME = "bg_uniforms";
const PACKED_SHARED_U8_STORE = "bg_semantic_packed_shared_u8_store";
const PACKED_SHARED_U8_ADD = "bg_semantic_packed_shared_u8_add";
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const LOGICAL_OPERATORS = new Set(["&&", "||"]);

export function canEmitSemanticKernelIrWgsl(
  ir: SemanticKernelIrModule,
  _options: EmitSemanticKernelIrWgslOptions = {},
): boolean {
  return semanticKernelIrWgslPreflightFailure(ir) === undefined;
}

export function semanticKernelIrWgslPreflightBlocker(
  ir: SemanticKernelIrModule,
): string | undefined {
  return semanticKernelIrWgslPreflightFailure(ir)?.message;
}

export function semanticKernelIrWgslPreflightFailure(
  ir: SemanticKernelIrModule,
): SemanticKernelIrWgslPreflightFailure | undefined {
  const unsupported = unsupportedSemanticWgslOperation(ir.operations, ir);
  if (unsupported) return { message: `semantic WGSL does not support ${unsupported.kind}`, span: unsupported.span };
  if (!semanticWgslRequiredFeaturesSupported(ir.requiredFeatures)) {
    return { message: "semantic WGSL does not support required WebGPU features yet", span: ir.span };
  }
  const unsupportedParam = ir.params.find((param) => !semanticWgslParamSupported(param, ir));
  if (unsupportedParam) return { message: `semantic WGSL does not support parameter '${unsupportedParam.name}'`, span: unsupportedParam.span };
  if (!semanticWgslSharedBarrierShapeSupported(ir)) {
    return { message: "semantic WGSL does not support shared-memory barrier shape", span: ir.span };
  }
  const unsupportedMemory = ir.memory.find((symbol) => !semanticWgslMemorySymbolSupported(symbol));
  if (unsupportedMemory) return { message: `semantic WGSL does not support memory '${unsupportedMemory.name}'`, span: unsupportedMemory.span };
  return undefined;
}

export function emitSemanticKernelIrWgsl(
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions = {},
): SemanticKernelIrWgslOutput {
  const failure = semanticKernelIrWgslPreflightFailure(ir);
  if (failure) throw semanticWgslError(failure.message, failure.span);

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
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions);
  const atomicDeviceGlobals = semanticAtomicDeviceGlobalNames(ir.operations);
  const atomicShared = semanticAtomicSharedNames(ir.operations, ir.functions);
  const cooperativeReduceHelpers = semanticCooperativeReduceHelpers(ir);
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
  if (semanticUsesFp8(ir)) {
    lines.push("", ...emitFp8Helpers());
  }
  if (semanticUsesCurand(ir)) {
    lines.push("", ...emitCurandHelpers());
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

function unsupportedSemanticWgslOperation(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  allowReturnValue = false,
  predicated = false,
): SemanticKernelIrOperation | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "dim3-declare":
      case "cooperative-group-declare":
        break;
      case "declare":
        if (operation.init && semanticExpressionContainsPartitionedReduce(operation.init, ir) && !predicated) return operation;
        if (operation.target.addressSpace === "shared") {
          if (operation.target.pointer || operation.target.valueType !== "uchar" && !semanticWgslValueTypeSupported(operation.target.valueType)) return operation;
          break;
        }
        if (operation.target.addressSpace !== "local" || operation.target.pointer) return operation;
        if (operation.target.valueType === "uchar" && operation.target.dimensions.length > 0 && operation.target.matrixTile === undefined) return operation;
        if (!semanticWgslLocalValueTypeSupported(operation.target.valueType)) return operation;
        if (operation.target.dimensions.length > 0 && operation.init && !semanticWgslLocalArrayInitSupported(operation.init, operation.target.valueType, ir)) return operation;
        if (operation.target.dimensions.length === 0) {
          const vectorTarget = isSemanticFloatVectorType(operation.target.valueType);
          if (operation.init && !semanticWgslExpressionSupported(operation.init, vectorTarget ? "any" : "scalar", ir)) {
            return unsupportedSemanticWgslNestedExpressionOperation(operation.init, ir) ?? operation;
          }
        }
        break;
      case "store":
        if (!semanticWgslAssignmentOperatorSupported(operation.operator)) return operation;
        if (isSemanticFloatVectorType(operation.target.valueType) && !semanticVectorAssignmentOperatorSupported(operation.operator)) return operation;
        if (semanticWgslVectorFieldMemoryRefSupported(operation.target) && !semanticVectorAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticWgslTypedMemoryRefSupported(operation.target, ir) && !semanticWgslStorageOffsetStoreSupported(operation, ir)) return operation;
        if (
          operation.target.addressSpace === "storage" &&
          !semanticWgslStorageBaseSupported(operation.target.base, ir)
        ) return operation;
        if (!semanticWgslStoreValueSupported(operation, ir)) return operation;
        break;
      case "copy":
        if (!semanticWgslCopySupported(operation, ir)) return operation;
        break;
      case "copy-fence":
        if (!isCudaCpAsyncFenceCall(operation.callee)) return operation;
        break;
      case "matrix-fill":
        if (!semanticWgslMatrixRefSupported(operation.fragment, ir) || !semanticWgslExpressionSupported(operation.value, "scalar", ir)) return operation;
        break;
      case "matrix-load":
        if (!semanticWgslMatrixRefSupported(operation.fragment, ir) || !semanticWgslTypedMemoryRefSupported(operation.source, ir) || !semanticWgslExpressionSupported(operation.stride, "scalar", ir)) return operation;
        break;
      case "matrix-mma":
        if (![operation.destination, operation.a, operation.b, operation.accumulator].every((ref) => semanticWgslMatrixRefSupported(ref, ir))) return operation;
        break;
      case "matrix-store":
        if (!semanticWgslTypedMemoryRefSupported(operation.target, ir) || !semanticWgslMatrixRefSupported(operation.fragment, ir) || !semanticWgslExpressionSupported(operation.stride, "scalar", ir)) return operation;
        break;
      case "surface-write":
        if (!semanticWgslSurfaceWriteSupported(operation, ir)) return operation;
        break;
      case "surface-read-store":
        if (!semanticWgslSurfaceReadStoreSupported(operation, ir)) return operation;
        break;
      case "atomic":
        if (!semanticWgslAtomicSupported(operation, ir)) return operation;
        break;
      case "call":
        if (!semanticWgslCallSupported(operation, ir)) {
          const fn = ir.functions.find((item) => item.name === operation.callee);
          const nested = fn ? unsupportedSemanticWgslOperation(fn.body, ir, true) : undefined;
          return nested ?? operation;
        }
        break;
      case "expression":
        if (!semanticWgslExpressionSupported(operation.expression, "scalar", ir)) return operation;
        break;
      case "branch":
        if (!semanticWgslConditionSupported(operation.condition, ir)) return operation;
        if (
          (semanticOperationsContainWorkgroupCollective(operation.consequent) || semanticOperationsContainWorkgroupCollective(operation.alternate)) &&
          (!semanticPredicatedOperationsSupported(operation.consequent) || !semanticPredicatedOperationsSupported(operation.alternate))
        ) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.consequent, ir, allowReturnValue, true) ??
          unsupportedSemanticWgslOperation(operation.alternate, ir, allowReturnValue, true);
          if (unsupported) return unsupported;
        }
        break;
      case "block":
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.body, ir, allowReturnValue, predicated);
          if (unsupported) return unsupported;
        }
        break;
      case "loop":
        if (operation.init && !semanticWgslLoopInitSupported(operation.init, ir)) return operation;
        if (operation.condition && !semanticWgslConditionSupported(operation.condition, ir)) return operation;
        if (operation.update && !semanticWgslExpressionSupported(operation.update, "scalar", ir)) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.body, ir, allowReturnValue, predicated) ??
            (operation.continuing === undefined ? undefined : unsupportedSemanticWgslOperation(operation.continuing, ir, allowReturnValue, predicated));
          if (unsupported) return unsupported;
        }
        break;
      case "barrier":
        if (!semanticWgslBarrierSupported(operation, ir)) return operation;
        break;
      case "fence":
        if (!isCudaFenceCallName(operation.callee)) return operation;
        break;
      case "inline-asm":
        {
          const asm = operation.op;
          const ldmatrix = semanticInlineAsmLdmatrixAssignments(operation);
          if (ldmatrix?.every((expression) => semanticWgslExpressionSupported(expression, "scalar", ir))) break;
          if (semanticWgslInlineMmaSupported(operation, ir)) break;
          if (asm?.kind === "cp-async-fence") {
            if (operation.inputs.length > (asm.fence === "wait_group" ? 1 : 0) || operation.outputs.length !== 0) return operation;
            break;
          }
          if (asm?.kind === "membar") {
            if (operation.inputs.length !== 0 || operation.outputs.length !== 0) return operation;
            break;
          }
          if (asm?.kind === "bar-sync") {
            if (operation.inputs.length !== (asm.operand === "input0" ? 1 : 0) || operation.outputs.length !== 0) return operation;
            break;
          }
          return operation;
        }
      case "return":
        if (operation.value && (!allowReturnValue || !semanticWgslExpressionSupported(operation.value, "any", ir))) return operation;
        break;
      case "break":
      case "continue":
        break;
      default:
        return operation;
    }
  }
  return undefined;
}

function semanticWgslMatrixRefSupported(ref: SemanticMatrixTileRef, ir: SemanticKernelIrModule): boolean {
  const symbol = ir.memory.find((item) => item.name === ref.base && item.kind === "local");
  return symbol?.matrixTile !== undefined && ref.indices.length === ref.arrayDimensions.length &&
    ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar", ir));
}

function unsupportedSemanticWgslNestedExpressionOperation(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  seen = new Set<string>(),
): SemanticKernelIrOperation | undefined {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && !seen.has(expression.callee.name)) {
    const calleeName = expression.callee.name;
    const fn = ir.functions.find((candidate) => candidate.name === calleeName);
    if (fn) {
      const nextSeen = new Set(seen).add(fn.name);
      const unsupported = unsupportedSemanticWgslOperation(fn.body, ir, true);
      if (unsupported) return unsupported;
      for (const operation of fn.body) {
        const nested = unsupportedSemanticWgslOperationExpression(operation);
        if (!nested) continue;
        const result = unsupportedSemanticWgslNestedExpressionOperation(nested, ir, nextSeen);
        if (result) return result;
      }
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    const unsupported = unsupportedSemanticWgslNestedExpressionOperation(child, ir, seen);
    if (unsupported) return unsupported;
  }
  return undefined;
}

function unsupportedSemanticWgslOperationExpression(
  operation: SemanticKernelIrOperation,
): SemanticExpression | undefined {
  if (operation.kind === "declare") return operation.init;
  if (operation.kind === "expression") return operation.expression;
  if (operation.kind === "return") return operation.value;
  if (operation.kind === "store") return operation.value;
  return undefined;
}

function semanticWgslParamSupported(
  param: SemanticKernelIrModule["params"][number],
  ir: SemanticKernelIrModule,
): boolean {
  if (param.addressSpace === "storage") {
    return Boolean(param.pointer) && (param.valueType === "uchar"
      ? semanticDirectByteStorageParamSupported(ir, param.name)
      : param.valueType === "complex64" || semanticWgslValueTypeSupported(param.valueType));
  }
  if (param.valueType === "uchar") return false;
  if (param.addressSpace === "uniform") return semanticWgslScalarTypeSupported(param.valueType) || isCudaVectorType(param.valueType);
  if (param.addressSpace === "texture") return param.valueType === "texture2d";
  if (param.addressSpace === "surface") return param.valueType === "surface2d";
  if (param.addressSpace === "pool") return !semanticOperationsReferenceRoot(ir.operations, param.name);
  return false;
}

function semanticWgslFunctionParamSupported(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): boolean {
  if (param.pointer && param.addressSpace === "shared" && param.valueType === "uchar" && param.pointerCarrierValueType === "uchar") return true;
  if (param.pointer && param.addressSpace === "storage" && param.valueType === "uchar") return true;
  if (!param.pointer && param.addressSpace === "local" && param.valueType === "uchar") return true;
  return semanticFunctionParamContractSupported(param, semanticWgslValueTypeSupported);
}

function semanticWgslMemorySymbolSupported(symbol: SemanticKernelIrModule["memory"][number]): boolean {
  if (symbol.kind === "local" || symbol.kind === "shared") return true;
  if (symbol.kind === "constant") {
    if (!semanticWgslValueTypeSupported(symbol.valueType)) return false;
    return !symbol.initialized ||
      symbol.init !== undefined && (
        symbol.dimensions.length === 0
          ? isSemanticFloatVectorType(symbol.valueType)
            ? initializedVectorConstantSupported(symbol)
            : semanticWgslExpressionSupported(symbol.init, "scalar")
          : initializedConstantArraySupported(symbol)
      );
  }
  if (symbol.kind === "device-global") return symbol.valueType !== "uchar" && semanticWgslScalarTypeSupported(symbol.valueType);
  if (symbol.kind === "texture") return symbol.valueType === "texture2d";
  return false;
}

function semanticWgslSharedBarrierShapeSupported(ir: SemanticKernelIrModule): boolean {
  const shared = sharedMemorySymbols(ir);
  const barrierFunctions = semanticBarrierFunctionNames(ir);
  const containsBarrier = semanticOperationsContainBarrier(ir.operations, barrierFunctions);
  if (shared.length === 0 && !containsBarrier) return true;
  const hasSharedPointer = ir.functions.some(semanticWgslFunctionHasSharedPointer);
  if (!shared.every((symbol) =>
    symbol.dimensions.length === 0 ||
    symbol.dimensions.every((dimension) => dimension > 0)
  )) return false;
  // Direct shared references use row-major flattened indices. Pointer-helper ABI
  // remains one-dimensional until its parameter type and base arithmetic are flattened.
  if (hasSharedPointer && shared.some((symbol) => symbol.dimensions.length > 1)) return false;
  if (!containsBarrier) return operationsHaveNoBarrierOrControlTransfer(ir.operations);
  if (barrierFunctions.size === 0 && semanticDirectBarriersHaveAnalyzerProof(ir)) return true;
  if (!shared.some((symbol) => isSemanticFloatVectorType(symbol.valueType)) && barrierFunctions.size === 0) {
    const activeLaneLowered = ir.operations.some((operation) => operation.kind === "declare" && operation.target.name === "bg_active_lane");
    return (activeLaneLowered && (
      semanticBarrierShapeSupported(ir.operations, barrierFunctions) ||
      semanticBarrierOperationsMatchActiveLaneProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions)
    )) || operationsHaveOnlyTopLevelBarriers(ir.operations);
  }
  const activeLaneLowered = ir.operations.some((operation) => operation.kind === "declare" && operation.target.name === "bg_active_lane");
  return (semanticBarrierShapeSupported(ir.operations, barrierFunctions) ||
      activeLaneLowered && semanticBarrierOperationsMatchActiveLaneProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions) ||
      semanticBarrierOperationsMatchUniformityProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions)) &&
    ir.functions.filter((fn) => barrierFunctions.has(fn.name)).every((fn) =>
      semanticBarrierShapeSupported(fn.body, barrierFunctions) ||
      semanticBarrierOperationsMatchUniformityProof(fn.body, ir.barrierUniformity.functions[fn.name], barrierFunctions)
    );
}

function semanticDirectBarriersHaveAnalyzerProof(ir: SemanticKernelIrModule): boolean {
  return semanticBarrierOperationsMatchUniformityProof(ir.operations, ir.barrierUniformity.kernel, semanticBarrierFunctionNames(ir));
}

function semanticWgslBarrierSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "barrier" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (operation.scope === "grid") return false;
  if (isCudaCooperativeBarrierCallName(operation.callee)) {
    const kind = operation.groupName === undefined ? undefined : semanticCooperativeGroupKind(ir, operation.groupName);
    return kind !== undefined && kind !== "grid";
  }
  return isCudaBarrierCallName(operation.callee);
}

function semanticCooperativeGroupKind(ir: SemanticKernelIrModule, name: string): string | undefined {
  for (const operation of ir.operations) {
    const kind = semanticCooperativeGroupKindInOperations([operation], name);
    if (kind !== undefined) return kind;
  }
  for (const fn of ir.functions) {
    const kind = semanticCooperativeGroupKindInOperations(fn.body, name);
    if (kind !== undefined) return kind;
    const paramKind = fn.params.find((param) => param.name === name)?.cooperativeGroupKind;
    if (paramKind !== undefined) return paramKind;
  }
  return undefined;
}

function semanticCooperativeGroupKindInOperations(
  operations: readonly SemanticKernelIrOperation[],
  name: string,
): string | undefined {
  for (const operation of operations) {
    if (operation.kind === "cooperative-group-declare" && operation.declaration.name === name) return operation.declaration.groupKind;
    const nested = operation.kind === "branch"
      ? [...operation.consequent, ...operation.alternate]
      : operation.kind === "loop" || operation.kind === "block"
      ? operation.body
      : undefined;
    if (nested) {
      const kind = semanticCooperativeGroupKindInOperations(nested, name);
      if (kind !== undefined) return kind;
    }
  }
  return undefined;
}

function operationsHaveOnlyTopLevelBarriers(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) =>
    operation.kind !== "branch" &&
    operation.kind !== "loop" &&
    operation.kind !== "block"
  );
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

function semanticWgslRequiredFeaturesSupported(requiredFeatures: readonly string[]): boolean {
  return requiredFeatures.every((feature) => feature === "shader-f16" || feature === "subgroups");
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

function semanticWgslLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  ir: SemanticKernelIrModule,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? unsupportedSemanticWgslOperation([init], ir) === undefined
    : semanticWgslExpressionSupported(init, "scalar", ir);
}

function semanticWgslMemoryRefSupported(ref: SemanticMemoryRef, ir?: SemanticKernelIrModule): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "shared" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local") return false;
  if (ref.fields.length > 0) return semanticWgslVectorFieldMemoryRefSupported(ref);
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return semanticWgslScalarTypeSupported(ref.valueType);
  return ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar", ir));
}

function semanticWgslStorageBaseSupported(base: string, ir: SemanticKernelIrModule): boolean {
  return ir.params.some((param) => param.name === base && param.addressSpace === "storage") ||
    ir.functions.some((fn) => fn.params.some((param) => param.name === base && param.pointer && param.addressSpace === "storage"));
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
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions);
  return [
    `fn ${semanticPointerReadHelperName(valueType)}(buffer: u32, index: u32) -> ${wgslType} {`,
    "  switch buffer {",
    ...ir.params.flatMap((param, index) =>
      param.addressSpace === "storage" && semanticPointerStorageCompatible(valueType, param.valueType)
        ? [`    case ${index}u: { return ${emitSemanticStoragePointerReadValue(valueType, nameFor(param.name, names), "index", atomicStorage.has(param.name))}; }`]
        : []
    ),
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
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions);
  return [
    `fn ${semanticPointerWriteHelperName(valueType)}(buffer: u32, index: u32, value: ${wgslType}) {`,
    "  switch buffer {",
    ...ir.params.flatMap((param, index) =>
      param.addressSpace === "storage" && !param.constant && semanticPointerStorageCompatible(valueType, param.valueType)
        ? [`    case ${index}u: { ${emitSemanticStoragePointerWriteValue(valueType, nameFor(param.name, names), "index", "value", atomicStorage.has(param.name))} return; }`]
        : []
    ),
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
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions);
  const op = semanticAtomicOperation(callee);
  const cas = op === "cas";
  return [
    `fn ${semanticPointerAtomicHelperName(callee, valueType)}(buffer: u32, index: u32, ${cas ? `compare: ${wgslType}, ` : ""}value: ${wgslType}) -> ${wgslType} {`,
    "  switch buffer {",
    ...ir.params.flatMap((param, index) =>
      param.addressSpace === "storage" && !param.constant && atomicStorage.has(param.name) && semanticPointerStorageCompatible(valueType, param.valueType)
        ? [`    case ${index}u: { return ${emitSemanticStoragePointerAtomicValue(callee, valueType, nameFor(param.name, names), "index", "compare", "value")}; }`]
        : []
    ),
    "    default: { return " + zeroForType(wgslType) + "; }",
    "  }",
    "}",
  ];
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
  if (semanticWgslLocalPackedByteRawView(ref, ir)) return true;
  if (semanticWgslPackedSharedByteRoot(ref, ir)) return semanticPackedSharedByteViewSupported(ref.valueType);
  if (ref.addressSpace === "shared" && semanticWgslFunctionSharedPointerParam(ir, ref.base)) return true;
  if (ref.addressSpace === "local" && semanticWgslFunctionLocalPointerParam(ir, ref.base)) return true;
  if (semanticWgslVectorFieldMemoryRefSupported(ref)) return true;
  if (semanticWgslLocalVectorLaneRefSupported(ref, ir)) return true;
  if (semanticWgslLocalScalarVectorView(ref, ir)) return true;
  if (semanticWgslSharedScalarVectorView(ref, ir)) return true;
  if (ref.addressSpace !== "local" && ref.addressSpace !== "shared") return true;
  const symbol = ir.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace) ??
    (ref.addressSpace === "local" ? semanticFunctionLocalArraySymbol(ir, ref.base) : undefined);
  return symbol !== undefined && symbol.valueType === ref.valueType;
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
  return valueType === "uchar" || valueType === "uint" || valueType === "int" || valueType === "float";
}

function semanticWgslCopySupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "copy" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const sourceBytes = operation.source.valueType === undefined ? undefined : sizeofCudaType(operation.source.valueType);
  const targetBytes = operation.target.valueType === undefined ? undefined : sizeofCudaType(operation.target.valueType);
  return operation.bytes >= 1 &&
    operation.bytes <= 64 &&
    operation.bytes % 4 === 0 &&
    sourceBytes !== undefined &&
    targetBytes !== undefined &&
    (sourceBytes === 2 || sourceBytes === 4 || sourceBytes === 1 && semanticWgslByteCopyRoot(operation.source, ir)) &&
    (targetBytes === 2 || targetBytes === 4 || targetBytes === 1 && semanticWgslByteCopyRoot(operation.target, ir)) &&
    operation.bytes % sourceBytes === 0 &&
    operation.bytes % targetBytes === 0 &&
    operation.source.fields.length === 0 &&
    operation.target.fields.length === 0 &&
    operation.target.addressSpace !== "constant" &&
    semanticWgslTypedMemoryRefSupported(operation.source, ir) &&
    semanticWgslTypedMemoryRefSupported(operation.target, ir);
}

function semanticWgslByteCopyRoot(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  return semanticWgslPackedSharedByteRoot(ref, ir) ||
    ref.addressSpace === "storage" && ir.params.some((param) => param.name === ref.base && param.valueType === "uchar");
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

function semanticWgslAtomicSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const atomicOp = semanticAtomicOperation(operation.callee);
  if (!atomicOp) return false;
  if (!operation.target || (operation.target.addressSpace !== "storage" && operation.target.addressSpace !== "device-global" && operation.target.addressSpace !== "shared")) return false;
  if (!semanticWgslAtomicMemoryRefSupported(operation.target, ir)) return false;
  if (!semanticWgslPointerAtomicSupported(operation.callee, operation.target, ir)) return false;
  if (operation.target.addressSpace === "storage" && operation.target.indices.length !== 1 && !semanticWgslFunctionStoragePointerParam(ir, operation.target.base)) return false;
  if (operation.target.fields.length > 0) return false;
  if (!semanticWgslAtomicValueTypeSupported(operation.callee, operation.target.valueType)) return false;
  if (!semanticWgslAtomicTargetRootSupported(operation.target, ir)) {
    return false;
  }
  const scalarArgIndices = semanticAtomicScalarArgumentIndices(atomicOp);
  return operation.args.length >= scalarArgIndices.length + 1 &&
    scalarArgIndices.every((index) => semanticWgslExpressionSupported(operation.args[index]!, "scalar", ir));
}

function semanticWgslStoreValueSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const targetVectorType = semanticStorageVectorType(operation.target.valueType);
  const valueVectorType = semanticExpressionVectorValueType(operation.value, ir?.functions);
  if (semanticWgslVectorFieldMemoryRefSupported(operation.target)) {
    return isSemanticFloatVectorType(targetVectorType)
      ? valueVectorType === targetVectorType && semanticWgslExpressionSupported(operation.value, "any", ir)
      : semanticWgslScalarStoreValueSupported(operation.value, ir);
  }
  if (isSemanticFloatVectorType(targetVectorType)) {
    return semanticVectorAssignmentOperatorSupported(operation.operator) &&
      (valueVectorType === targetVectorType
        ? semanticWgslExpressionSupported(operation.value, "any", ir)
        : semanticWgslExpressionSupported(operation.value, "scalar", ir));
  }
  return semanticWgslScalarStoreValueSupported(operation.value, ir);
}

function semanticWgslScalarStoreValueSupported(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
): boolean {
  switch (expression.kind) {
    case "symbol":
    case "index":
      return !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression, ir?.functions)) &&
        semanticWgslExpressionSupported(expression, "scalar", ir);
    case "call":
      return semanticWgslScalarCallSupported(expression, ir);
    case "binary":
      if (expression.valueType === "bool") return semanticWgslExpressionSupported(expression, "scalar", ir);
      return semanticWgslScalarStoreValueSupported(expression.left, ir) &&
        semanticWgslScalarStoreValueSupported(expression.right, ir);
    case "conditional":
      return semanticWgslConditionSupported(expression.condition, ir) &&
        semanticWgslScalarStoreValueSupported(expression.consequent, ir) &&
        semanticWgslScalarStoreValueSupported(expression.alternate, ir);
    case "sequence": {
      const last = expression.expressions.at(-1);
      return last !== undefined &&
        expression.expressions.slice(0, -1).every((item) => semanticWgslExpressionSupported(item, "scalar", ir)) &&
        semanticWgslScalarStoreValueSupported(last, ir);
    }
    case "texture-read":
      return !isSemanticFloatVectorType(expression.valueType) && semanticWgslTextureReadSupported(expression, ir);
    case "surface-read":
      return !isSemanticFloatVectorType(expression.valueType) && semanticWgslSurfaceReadSupported(expression, ir);
    default:
      return !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression, ir?.functions)) &&
        semanticWgslExpressionSupported(expression, "scalar", ir);
  }
}

function semanticWgslScalarCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (semanticWgslCooperativeGroupCallSupported(expression, ir)) return true;
  if (expression.callee.kind !== "symbol") return false;
  const callee = expression.callee.name;
  if (semanticPtxIntegerCallInfo(callee) !== undefined) {
    return expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  if (callee === "__cvta_generic_to_shared") return semanticWgslSharedAddressCallRef(expression) !== undefined;
  if (SEMANTIC_CURAND_VECTOR_CALLS.has(callee) || SEMANTIC_HALF2_VECTOR_CALLS.has(callee) || SEMANTIC_BF162_VECTOR_CALLS.has(callee) || cudaVectorConstructorType(callee)) return false;
  const fn = ir.functions.find((item) => item.name === callee);
  if (fn && isSemanticFloatVectorType(fn.returnType)) return false;
  return semanticWgslCooperativeReduceCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir)) ||
    (callee === "dot" || callee === "length") && semanticVectorMathCallSupported(callee, expression.args) ||
    semanticWgslFunctionCallSupported(expression, ir) ||
    semanticWgslAtomicCallSupported(expression, ir) ||
    semanticWgslCurandCallSupported(expression, ir) ||
    semanticWgslGeneratedRandomCallSupported(expression) ||
    semanticWgslSubgroupCallSupported(expression, ir) ||
    semanticWgslAddressPredicateCallSupported(expression) ||
    semanticWgslMathCallSupported(expression, "scalar", ir) ||
    SEMANTIC_HALF2_SCALAR_CALLS.has(callee) && semanticWgslHalf2CallSupported(expression, ir) ||
    SEMANTIC_BF162_SCALAR_CALLS.has(callee) && semanticWgslBf162CallSupported(expression, ir) ||
    semanticWgslVectorAtCallSupported(expression, ir);
}

function semanticWgslVectorMemberSupported(
  expression: Extract<SemanticExpression, { kind: "member" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  const valueType = semanticExpressionValueType(expression.object);
  return semanticWgslExpressionSupported(expression.object, "any", ir) &&
    semanticStorageVectorFieldIndices(valueType, expression.property) !== undefined;
}

function semanticWgslVectorIndexSupported(
  expression: Extract<SemanticExpression, { kind: "index" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  const ref = memoryRefFromIndexExpression(expression);
  if (ref && ir && (semanticWgslLocalScalarVectorView(ref, ir) || semanticWgslLocalVectorBitViewRootType(ref, ir) !== undefined)) return false;
  if (ref && !(ref.addressSpace === "local" && isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.target, ir?.functions)))) return false;
  return isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.target, ir?.functions)) &&
    semanticWgslExpressionSupported(expression.target, "any", ir) &&
    semanticWgslExpressionSupported(expression.index, "scalar", ir);
}

function semanticWgslLocalArrayInitSupported(
  expression: SemanticExpression,
  targetValueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticLocalArrayInitContractSupported(expression, targetValueType, (item, expected) => semanticWgslExpressionSupported(item, expected, ir));
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
    texture.kind === "symbol" &&
    texture.addressSpace === "texture" &&
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
    expression.args.every((arg, index) => semanticWgslFunctionArgSupported(arg, fn.params[index], ir)) &&
    unsupportedSemanticWgslOperation(fn.body, ir, true) === undefined;
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

function semanticWgslFunctionHasSharedPointer(fn: SemanticKernelIrModule["functions"][number]): boolean {
  return fn.params.some((param) => param.pointer && param.addressSpace === "shared");
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
  if (!target || (target.addressSpace !== "storage" && target.addressSpace !== "device-global" && target.addressSpace !== "shared")) return false;
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

function semanticWgslCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (operation.result !== undefined) {
    const fn = ir.functions.find((item) => item.name === operation.callee);
    return fn !== undefined &&
      fn.returnType !== "void" &&
      fn.returnType === operation.result.valueType &&
      semanticWgslFunctionCallSupported(semanticCallOperationExpression(operation, fn.returnType), ir);
  }
  if (operation.callee === "assert") return semanticAssertCallSupported(operation.args, (arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  if (operation.callee === "printf") return semanticPrintfCallSupported(operation.args, (arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) {
    return semanticNoopCallSupported(operation.callee, operation.args, (arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  if (operation.callee === "curand_init") {
    return semanticWgslCurandCallSupported({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: "uint",
      span: operation.span,
    }, ir);
  }
  if (operation.callee === "skipahead") {
    return semanticWgslCurandCallSupported({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: "uint",
      span: operation.span,
    }, ir);
  }
  if (semanticWgslVoidFunctionCallSupported(operation, ir)) return true;
  return semanticLocalArrayFillCallSupported(
    operation,
    (name) => localArraySymbol(ir, name),
    (item, expected) => semanticWgslExpressionSupported(item, expected, ir),
  );
}

function semanticWgslVoidFunctionCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (operation.result !== undefined) return false;
  const fn = ir.functions.find((item) => item.name === operation.callee);
  if (!fn || fn.returnType !== "void") return false;
  if (fn.params.some((param) => !semanticWgslFunctionParamSupported(param))) return false;
  if (fn.params.some((param) => param.pointer) && !semanticWgslPointerFunctionBodySupported(fn)) return false;
  if (!semanticFunctionLocalParamValueTypesSupported(fn, semanticWgslLocalValueTypeSupported)) return false;
  return operation.args.length === fn.params.length &&
    operation.args.every((arg, index) => semanticWgslFunctionArgSupported(arg, fn.params[index], ir)) &&
    semanticWgslFunctionBodyShapeSupported(fn.body, semanticWgslFunctionHasAtomicPointer(fn)) &&
    unsupportedSemanticWgslOperation(fn.body, ir, true) === undefined;
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

function semanticWgslSurfaceReadStoreSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-read-store" }>,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticWgslSurfaceReadTarget(operation.target) !== undefined &&
    semanticWgslSurfaceReadSupported(
      {
        kind: "surface-read",
        callee: operation.z === undefined ? "surf2Dread" : "surf2DLayeredread",
        surface: operation.surface,
        xBytes: operation.xBytes,
        y: operation.y,
        ...(operation.z === undefined ? {} : { z: operation.z }),
        valueType: semanticSurfaceReadValueType(operation.valueType ?? semanticWgslSurfaceReadTarget(operation.target)?.valueType),
        span: operation.span,
      },
      ir,
    );
}

function semanticWgslAtomicTargetRootSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
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
      return expression.operator !== "*" && expression.operator !== "&" && semanticWgslExpressionSupported(expression.argument, "scalar", ir);
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
        ir !== undefined && semanticWgslCooperativeReduceCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir)) ||
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
      const fn = ir.functions.find((item) => item.name === operation.callee);
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
      if (operation.target.dimensions.length > 0) {
        return [
          `${prefix}var ${nameFor(operation.target.name, names)}: ${emitLocalArrayType(operation.target)};`,
          ...emitLocalArrayInit(operation, ir, names, indentLevel, options, textureSpecializations),
        ];
      }
      const type = wgslValueType(operation.target.valueType);
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
        ? ` = ${emitSemanticInitExpression(operation.init, operation.target.valueType, ir, names, options, textureSpecializations)}`
        : isSemanticFloatVectorType(operation.target.valueType)
        ? ` = ${zeroForType(wgslValueType(operation.target.valueType))}`
        : "";
      return [`${prefix}var ${nameFor(operation.target.name, names)}: ${type}${init};`];
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
    case "expression":
      if (isSemanticNoopExpression(operation.expression)) return [];
      if (operation.expression.kind === "assignment") return [`${prefix}${emitSemanticAssignmentStatement(operation.expression, ir, names, options, textureSpecializations)};`];
      if (operation.expression.kind === "sequence") return emitSemanticSequenceStatement(operation.expression, ir, names, indentLevel, options, textureSpecializations);
      return [`${prefix}${emitSemanticExpression(operation.expression, ir, names, options, textureSpecializations)};`];
    case "branch": {
      if (semanticOperationsContainWorkgroupCollective(operation.consequent) || semanticOperationsContainWorkgroupCollective(operation.alternate)) {
        const condition = emitTruthiness(operation.condition, ir, names, options);
        return [
          `${prefix}{`,
          ...emitSemanticPredicatedOperations(operation.consequent, condition, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
          `${prefix}}`,
          `${prefix}{`,
          ...emitSemanticPredicatedOperations(operation.alternate, `!(${condition})`, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
          `${prefix}}`,
        ];
      }
      const lines = [`${prefix}if (${emitTruthiness(operation.condition, ir, names, options)}) {`];
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
            `${prefix}${emitSemanticExpression(assignment, ir, names, options, textureSpecializations)};`
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
  const value = emitSemanticMatrixCoerce(emitSemanticExpression(operation.value, ir, names, options, textureSpecializations), operation.fragment.spec);
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
    const emitted = `u32(${emitSemanticExpression(item, ir, names, options)})`;
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
  return { kind: "symbol", name, valueType: "uint", addressSpace: "local", span };
}

function semanticWgslMemoryRefOffset(ref: SemanticMemoryRef, offset: SemanticExpression): SemanticMemoryRef {
  if (ref.indices.length === 0) return { ...ref, indices: [offset] };
  const last = ref.indices[ref.indices.length - 1]!;
  return { ...ref, indices: [...ref.indices.slice(0, -1), { kind: "binary", operator: "+", left: last, right: offset, valueType: "uint", span: ref.span }] };
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

function semanticWgslInlineMmaSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "inline-asm" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const op = operation.op;
  const countsMatch = op?.kind === "mma-m16n8k16" &&
    (op.accumulator === "f16"
      ? operation.outputs.length === 2 && operation.inputs.length === 8
      : operation.outputs.length === 4 && operation.inputs.length === 10);
  if (!countsMatch) return false;
  return operation.inputs.every((input) => semanticWgslExpressionSupported(input, "scalar", ir)) &&
    operation.outputs.every((output) => semanticWgslInlineOutputSupported(output, ir));
}

function semanticWgslInlineOutputSupported(output: SemanticExpression, ir: SemanticKernelIrModule): boolean {
  const assignment: SemanticExpression = {
    kind: "assignment",
    operator: "=",
    target: output,
    value: { kind: "literal", literalKind: "number", value: 0, valueType: "uint", span: output.span },
    valueType: "uint",
    span: output.span,
  };
  return semanticWgslExpressionSupported(assignment, "scalar", ir);
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
    const target = emitSemanticExpression(output, ir, names, options, textureSpecializations);
    const a = emitSemanticExpressionAs(operation.inputs[index % 4]!, ir, names, "u32", options, textureSpecializations);
    const b = emitSemanticExpressionAs(operation.inputs[4 + (index % 2)]!, ir, names, "u32", options, textureSpecializations);
    if (accumulator === "f16") {
      const c = emitSemanticExpressionAs(operation.inputs[6 + index]!, ir, names, "u32", options, textureSpecializations);
      const value = `pack2x16float(unpack2x16float(${c}) + (unpack2x16float(${a}) * unpack2x16float(${b})))`;
      return `${prefix}${target} = ${semanticInlineMmaOutputValue(output, value, "u32")};`;
    }
    const cExpression = operation.inputs[6 + index]!;
    const cRaw = emitSemanticExpression(cExpression, ir, names, options, textureSpecializations);
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
      if (!valueType || valueType === "void" || isSemanticFloatVectorType(valueType)) {
        throw semanticWgslError("predicated cooperative shuffle requires typed scalar assignment", operation.span);
      }
      const temporary = nameFor(`bg_collective_${operation.span.start}`, names);
      const collectiveOptions = { ...options, activeCollectivePredicate: predicate };
      lines.push(`${prefix}let ${temporary}: ${wgslValueScalar(valueType)} = ${emitSemanticLocalScalarExpressionAs(operation.expression.value, valueType, ir, names, collectiveOptions, textureSpecializations)};`);
      lines.push(`${prefix}if (${predicate}) {`);
      lines.push(`${"  ".repeat(indentLevel + 1)}${emitSemanticAssignmentStatement({ ...operation.expression, value: { kind: "symbol", name: temporary, valueType, addressSpace: "local", span: operation.span } }, ir, names, options, textureSpecializations)};`);
      lines.push(`${prefix}}`);
      continue;
    }
    if (operation.kind === "store" && semanticExpressionContainsWorkgroupCollective(operation.value) &&
      !operation.target.indices.some(semanticExpressionContainsWorkgroupCollective)) {
      const valueType = operation.target.valueType;
      if (!valueType || valueType === "void" || isSemanticFloatVectorType(valueType)) {
        throw semanticWgslError("predicated cooperative store requires typed scalar value", operation.span);
      }
      const temporary = nameFor(`bg_collective_${operation.span.start}`, names);
      const collectiveOptions = { ...options, activeCollectivePredicate: predicate };
      lines.push(`${prefix}let ${temporary}: ${wgslValueScalar(valueType)} = ${emitSemanticLocalScalarExpressionAs(operation.value, valueType, ir, names, collectiveOptions, textureSpecializations)};`);
      lines.push(`${prefix}if (${predicate}) {`);
      lines.push(...emitSemanticOperation({
        ...operation,
        value: { kind: "symbol", name: temporary, valueType, addressSpace: "local", span: operation.span },
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

function semanticPredicatedOperationsSupported(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "block" && semanticOperationsContainWorkgroupCollective(operation.body)) {
      return semanticPredicatedOperationsSupported(operation.body);
    }
    if (operation.kind === "branch") {
      return semanticPredicatedOperationsSupported(operation.consequent) && semanticPredicatedOperationsSupported(operation.alternate);
    }
    if (operation.kind === "loop" && semanticOperationsContainWorkgroupCollective(operation.body)) {
      return operation.loopKind === "for" && operation.update?.kind !== "sequence" && semanticPredicatedOperationsSupported(operation.body);
    }
    if (operation.kind === "declare") {
      return operation.target.addressSpace === "local" && !operation.target.pointer && (
        operation.init === undefined ||
        operation.target.dimensions.length === 0 &&
          operation.target.valueType !== undefined && operation.target.valueType !== "void"
      );
    }
    if (operation.kind === "expression" && operation.expression.kind === "assignment" && semanticExpressionContainsWorkgroupCollective(operation.expression.value)) {
      const target = operation.expression.target;
      return target.kind === "symbol" && target.addressSpace === "local" && target.valueType !== undefined && target.valueType !== "void" && !isSemanticFloatVectorType(target.valueType);
    }
    if (operation.kind === "store" && semanticExpressionContainsWorkgroupCollective(operation.value) &&
      !operation.target.indices.some(semanticExpressionContainsWorkgroupCollective)) {
      return operation.target.valueType !== undefined && operation.target.valueType !== "void" && !isSemanticFloatVectorType(operation.target.valueType);
    }
    return !semanticOperationContainsWorkgroupCollective(operation);
  });
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
  if (expression.kind === "call" && expression.callee.kind === "symbol" && expression.callee.addressSpace !== "function" &&
    (semanticShuffleOpForCall(expression.callee.name) !== undefined ||
      isCudaWarpReduceCallName(expression.callee.name) ||
      cudaVoteOpForCall(expression.callee.name) === "ballot" ||
      cudaVoteOpForCall(expression.callee.name) === "any" ||
      cudaVoteOpForCall(expression.callee.name) === "all" ||
      cudaArithmeticReduceOpForCall(expression.callee.name) !== undefined ||
      expression.callee.name === "__activemask" ||
      expression.callee.name === "cg::reduce" ||
      expression.callee.name === "cooperative_groups::reduce")) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionContainsWorkgroupCollective);
}

function semanticExpressionContainsPartitionedReduce(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" &&
    (expression.callee.name === "cg::reduce" || expression.callee.name === "cooperative_groups::reduce") &&
    expression.args[0]?.kind === "symbol" &&
    semanticCooperativeGroupInfo(ir, expression.args[0].name)?.partitioned === true) return true;
  return semanticExpressionChildren(expression).some((child) => semanticExpressionContainsPartitionedReduce(child, ir));
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
    const word = emitSemanticExpressionAs(lowered.value, ir, names, "u32", options, textureSpecializations);
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
    !semanticAtomicDeviceGlobalNames(ir.operations).has(operation.target.base);
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
  return [`${prefix}${emitSemanticExpression(expression, ir, names, options, textureSpecializations)};`];
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
    return [...lines, `${prefix}return ${emitSemanticAssignmentResult(expression, ir, names, options)};`];
  }
  const returnType = options.activeFunction === undefined
    ? undefined
    : ir.functions.find((fn) => fn.name === options.activeFunction)?.returnType;
  const value = returnType === undefined
    ? emitSemanticExpression(expression, ir, names, options, textureSpecializations)
    : semanticExpressionValueType(expression) === "bool" && returnType !== "bool"
      ? `select(${emitNumberLiteral(0, returnType)}, ${emitNumberLiteral(1, returnType)}, ${emitTruthiness(expression, ir, names, options)})`
      : emitSemanticLocalScalarExpressionAs(expression, returnType, ir, names, options, textureSpecializations);
  return [`${prefix}return ${value};`];
}

function emitSemanticAssignmentResult(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (expression.target.kind === "symbol") return nameFor(expression.target.name, names);
  if (expression.target.kind === "member" && semanticWgslVectorMemberSupported(expression.target, ir)) {
    return emitSemanticMember(expression.target, ir, names, options);
  }
  const ref = semanticWgslAssignmentMemoryRef(expression.target, ir);
  if (ref) return emitSemanticMemoryRef(ref, ir, names, options);
  throw semanticWgslError("semantic WGSL cannot return assignment result", expression.span);
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
  const xBytes = emitSemanticExpressionAs(operation.xBytes, ir, names, "i32", options, textureSpecializations);
  const y = emitSemanticExpressionAs(operation.y, ir, names, "i32", options, textureSpecializations);
  const z = operation.z ? emitSemanticExpressionAs(operation.z, ir, names, "i32", options, textureSpecializations) : "0";
  const valueType = semanticExpressionVectorValueType(operation.value, ir?.functions);
  const value = isSemanticFloatVectorType(valueType)
    ? emitSemanticExpression(operation.value, ir, names, options, textureSpecializations)
    : emitSemanticExpressionAs(operation.value, ir, names, "f32", options, textureSpecializations);
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
    const value = emitSemanticExpressionAs(operation.value, ir, names, "i32", options, textureSpecializations);
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
    semanticWgslLocalVectorBitViewRootType(operation.target, ir) !== undefined
  ) {
    const value = emitSemanticScalarStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
    if (operation.operator === "=") return emitSemanticMemoryWrite(operation.target, value, ir, names, options);
    const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
    if (binaryOperator === undefined) throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
    const current = emitSemanticMemoryRead(operation.target, ir, names, options);
    return emitSemanticMemoryWrite(operation.target, `(${current} ${binaryOperator} ${value})`, ir, names, options);
  }
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base)) {
    return emitSemanticPointerMemoryStore(operation, ir, names, options, textureSpecializations);
  }
  if (
    semanticWgslFunctionSharedPointerParam(ir, operation.target.base) &&
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
  if (
    semanticAtomicStorageNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations).has(operation.target.base) ||
    semanticAtomicSharedNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticWgslFunctionSharedPointerAtomicParam(ir, operation.target.base)
  ) {
    if (operation.operator !== "=") {
      throw semanticWgslError(`semantic WGSL does not support atomic storage assignment '${operation.operator}'`, operation.span);
    }
    const atomicValue = emitSemanticAtomicStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
    return `atomicStore(&${target}, ${atomicValue})`;
  }
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
    if (semanticWgslSharedVectorMemoryRef(operation.target, ir)) {
      const access = emitSemanticMemoryRef(operation.target, ir, names, options);
      return `${access} = ${operation.operator === "=" ? value : `(${access} ${binaryOperator} ${value})`}`;
    }
    return emitSemanticVectorMemoryWrite(operation, ir, names, options, textureSpecializations).join("; ");
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
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
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
    return `bitcast<u32>(f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)}))`;
  }
  return emitSemanticExpressionAs(value, ir, names, wgslAtomicScalar(valueType), options, textureSpecializations);
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
    ? emitSemanticExpression(operation.value, ir, names, options, textureSpecializations)
    : emitSemanticExpressionAs(operation.value, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
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
  return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(index, ir, names, "u32", options, textureSpecializations)}]`;
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
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    `${target}[(${base} + ${lane}u)] = ${semanticPackedByteVectorLaneValue(operation.target, operation.operator === "=" ? `(${value}).${fields[lane]}` : `(${target}[(${base} + ${lane}u)] ${binaryOperator} (${value}).${fields[lane]})`)}`
  );
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
  const lanes = semanticStorageVectorFieldIndices(operation.target.containerValueType, operation.target.fields[0] ?? "");
  if (lanes === undefined) throw semanticWgslError("semantic WGSL vector field write requires modeled lanes", operation.span);
  if (semanticWgslSharedVectorMemoryRef({ ...operation.target, valueType: containerType, fields: [] }, ir)) {
    const target = emitSemanticMemoryRef({ ...operation.target, valueType: containerType, fields: [] }, ir, names, options);
    const field = lanes.map((lane) => ["x", "y", "z", "w"][lane]).join("");
    const access = `${target}.${field}`;
    const value = lanes.length === 1
      ? emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations)
      : isCudaVectorType(operation.target.valueType)
      ? emitSemanticVectorOperand(operation.value, operation.target.valueType, ir, names, options, textureSpecializations)
      : undefined;
    if (value === undefined) throw semanticWgslError("semantic WGSL shared vector swizzle write requires vector value", operation.span);
    const assigned = operation.operator === "=" ? value : `(${access} ${operation.operator.slice(0, -1)} ${value})`;
    return [`${access} = ${assigned}`];
  }
  if (operation.target.addressSpace === "local") {
    const target = emitSemanticMemoryRef({ ...operation.target, valueType: containerType, fields: [] }, ir, names, options);
    const field = lanes.map((lane) => ["x", "y", "z", "w"][lane]).join("");
    if (lanes.length === 1) {
      const access = `${target}.${field}`;
      const value = emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations);
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
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base)) {
    return emitSemanticPointerVectorFieldMemoryWrite(operation, ir, names, options, textureSpecializations);
  }
  const base = emitFlatStorageVectorBaseIndex(operation.target, ir, names, options);
  const target = nameFor(operation.target.base, names);
  const fields = ["x", "y", "z", "w"];
  if (lanes.length === 1) {
    const access = `${target}[(${base} + ${lanes[0]}u)]`;
    const value = emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations);
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
    ? emitSemanticExpressionAs(operation.value, ir, names, wgslVectorScalar(containerType), options, textureSpecializations)
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
  if (param.pointer && param.addressSpace === "local") return `ptr<function, ${wgslValueType(param.valueType)}>`;
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
  if (param.pointer && param.addressSpace === "constant" && param.pointerAliasOf !== undefined) return [];
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
    if (param.pointerAliasOf !== undefined) {
      return [`${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`];
    }
    return [
      `${nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param, atomicSharedPointer)}`,
      `${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`,
    ];
  }
  return [`${mutableValueParam ? semanticFunctionParamIncomingName(param, names) : nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param, atomicSharedPointer)}`];
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
      : emitSemanticExpressionAs(expression.value, ir, names, wgslVectorScalar(semanticExpressionVectorValueType(expression.target.object, ir?.functions)), options, textureSpecializations);
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
      const value = emitSemanticExpressionAs(expression.value, ir, names, wgslValueScalar(ref.valueType), options, textureSpecializations);
      if (semanticAssignmentBinaryOperator(expression.operator)) return `${target} ${expression.operator} ${value}`;
      return `${target} = ${value}`;
    }
  }
  if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local assignment targets only", expression.target.span);
  const target = nameFor(expression.target.name, names);
  const targetType = expression.target.valueType;
  const value = targetType !== undefined && isSemanticFloatVectorType(targetType)
    ? emitSemanticVectorOperand(expression.value, targetType, ir, names, options, textureSpecializations)
    : emitSemanticLocalScalarExpressionAs(expression.value, targetType, ir, names, options, textureSpecializations);
  if (targetType === "uchar" && expression.operator !== "=") {
    const binaryOperator = expression.operator.slice(0, -1);
    const right = emitSemanticExpressionAs(expression.value, ir, names, "u32", options, textureSpecializations);
    return `${target} = ${emitSemanticUcharValue(`(${target} ${binaryOperator} ${right})`)}`;
  }
  const binaryOperator = semanticAssignmentBinaryOperator(expression.operator);
  const promotedType = binaryOperator === undefined
    ? undefined
    : promotedCudaScalarType(targetType, semanticExpressionValueType(expression.value));
  if (binaryOperator !== undefined && promotedType !== undefined && wgslValueScalar(promotedType) !== wgslValueScalar(targetType)) {
    const operationScalar = wgslValueScalar(promotedType);
    const left = `${operationScalar}(${target})`;
    const right = emitSemanticExpressionAs(expression.value, ir, names, operationScalar, options, textureSpecializations);
    return `${target} = ${wgslValueScalar(targetType)}((${left} ${binaryOperator} ${right}))`;
  }
  if (binaryOperator) return `${target} ${expression.operator} ${value}`;
  return `${target} = ${value}`;
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
      ? emitSemanticExpression(operation.init, ir, names, options, textureSpecializations)
      : emitSemanticExpressionAs(operation.init, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations);
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
        ? emitSemanticExpression(value, ir, names, options, textureSpecializations)
        : emitSemanticExpressionAs(value, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations);
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
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base)) {
    const pointerCall = emitSemanticPointerAtomicCall({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      ...(operation.target.valueType === undefined ? {} : { valueType: operation.target.valueType }),
      span: operation.span,
    }, operation.target, ir, names, options, textureSpecializations);
    if (!pointerCall) throw semanticWgslError(`semantic WGSL pointer atomic '${operation.callee}' is unsupported`, operation.span);
    return `_ = ${pointerCall}`;
  }
  const target = emitSemanticMemoryRef(operation.target, ir, names, options);
  const operands = operation.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  if (operands.length === 0 || operands.some((operand) => operand === undefined)) {
    throw semanticWgslError(`semantic WGSL atomic '${operation.callee}' missing operand`, operation.span);
  }
  if (loopAtomicKind) {
    const value = emitSemanticExpressionAs(operands[0]!, ir, names, "u32", options, textureSpecializations);
    return `_ = ${semanticIntegerLoopAtomicHelperName(loopAtomicKind, operation.target, ir)}(&${target}, ${value})`;
  }
  if (semanticAtomicSupportsBfloatAdd(operation.callee, operation.target.valueType)) {
    const value = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations);
    return `_ = ${bfloatAtomicAddHelperName(semanticWgslAtomicAddressSpace(operation.target))}(&${target}, ${value})`;
  }
  const floatAtomicKind = semanticAtomicUsesF32Storage(operation.target.valueType) ? semanticWgslFloatAtomicCallKind(operation.callee) : undefined;
  if (floatAtomicKind) {
    const addressSpace = semanticWgslAtomicAddressSpace(operation.target);
    if (floatAtomicKind === "Exchange") {
      const value = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations);
      return `_ = atomicExchange(&${target}, bitcast<u32>(${value}))`;
    }
    if (floatAtomicKind === "CompareExchange") {
      const compare = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations);
      const value = emitSemanticExpressionAs(operands[1]!, ir, names, "f32", options, textureSpecializations);
      return `_ = atomicCompareExchangeWeak(&${target}, bitcast<u32>(${compare}), bitcast<u32>(${value}))`;
    }
    const value = emitSemanticExpressionAs(operands[0]!, ir, names, "f32", options, textureSpecializations);
    return `_ = ${floatAtomicHelperName(floatAtomicKind, addressSpace)}(&${target}, ${value})`;
  }
  const emitted = operands.map((operand) =>
    emitSemanticExpressionAs(operand!, ir, names, wgslAtomicScalar(operation.target!.valueType), options, textureSpecializations)
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
    const fn = ir.functions.find((item) => item.name === operation.callee);
    if (!fn || fn.returnType === "void") throw semanticWgslError(`semantic WGSL call '${operation.callee}' cannot produce a result`, operation.span);
    const call = emitSemanticFunctionCall(semanticCallOperationExpression(operation, fn.returnType), ir, names, options, textureSpecializations);
    return [`${"  ".repeat(indentLevel)}${nameFor(operation.result.name, names)} = ${call};`];
  }
  if (operation.callee === "assert") return [];
  if (operation.callee === "printf") return [];
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) {
    const prefix = "  ".repeat(indentLevel);
    return operation.args.map((arg, index) =>
      `${prefix}let ${nameFor(`bg_noop_arg_${operation.span.start}_${index}`, names)} = ${emitSemanticExpression(arg, ir, names, options, textureSpecializations)};`
    );
  }
  if (operation.callee === "curand_init") return [`${"  ".repeat(indentLevel)}${emitSemanticCurandInit(operation, ir, names, options, textureSpecializations)};`];
  if (operation.callee === "skipahead") {
    return [`${"  ".repeat(indentLevel)}${emitSemanticCurandCall({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
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
  valueType: CudaLiteScalarType,
): Extract<SemanticExpression, { readonly kind: "call" }> {
  return {
    kind: "call",
    callee: { kind: "symbol", name: operation.callee, addressSpace: "function", valueType, span: operation.span },
    args: operation.args,
    valueType,
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
  const fn = ir.functions.find((item) => item.name === operation.callee);
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
    ? emitSemanticExpression(valueExpression, ir, names, options, textureSpecializations)
    : emitSemanticExpressionAs(valueExpression, ir, names, wgslValueScalar(symbol.valueType), options, textureSpecializations);
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
  const seed = emitSemanticExpressionAs(operation.args[0]!, ir, names, "u32", options, textureSpecializations);
  const sequence = emitSemanticExpressionAs(operation.args[1]!, ir, names, "u32", options, textureSpecializations);
  const offset = emitSemanticExpressionAs(operation.args[2]!, ir, names, "u32", options, textureSpecializations);
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
    const count = emitSemanticExpressionAs(expression.args[0]!, ir, names, "u32", options, textureSpecializations);
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
    const mean = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations);
    const stddev = emitSemanticExpressionAs(expression.args[2]!, ir, names, "f32", options, textureSpecializations);
    return `bg_curand_log_normal${suffix}(${pointer.expression}, ${mean}, ${stddev})`;
  }
  if (expression.callee.name === "curand_log_normal2") {
    const mean = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations);
    const stddev = emitSemanticExpressionAs(expression.args[2]!, ir, names, "f32", options, textureSpecializations);
    return `bg_curand_log_normal2${suffix}(${pointer.expression}, ${mean}, ${stddev})`;
  }
  if (expression.callee.name === "curand_log_normal4") {
    const mean = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations);
    const stddev = emitSemanticExpressionAs(expression.args[2]!, ir, names, "f32", options, textureSpecializations);
    return `bg_curand_log_normal4${suffix}(${pointer.expression}, ${mean}, ${stddev})`;
  }
  if (expression.callee.name === "curand_poisson") {
    const lambda = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations);
    return `bg_curand_poisson${suffix}(${pointer.expression}, ${lambda})`;
  }
  if (expression.callee.name === "curand_poisson4") {
    const lambda = emitSemanticExpressionAs(expression.args[1]!, ir, names, "f32", options, textureSpecializations);
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
    if (operation.update?.kind === "sequence") {
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
        ...emitSemanticSequenceStatement(operation.update, ir, names, indentLevel + 2, options, textureSpecializations),
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
  return update.kind === "assignment"
    ? emitSemanticAssignmentStatement(update, ir, names, options, textureSpecializations)
    : emitSemanticExpression(update, ir, names, options, textureSpecializations);
}

function emitSemanticLoopInit(
  init: SemanticKernelIrOperation | SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (!isSemanticKernelIrOperation(init)) return emitSemanticExpression(init, ir, names, options, textureSpecializations);
  if (init.kind === "declare") {
    const type = wgslScalar(init.target.valueType);
    const value = init.init ? emitSemanticLocalScalarExpressionAs(init.init, init.target.valueType, ir, names, options, textureSpecializations) : zeroForType(type);
    return `var ${nameFor(init.target.name, names)}: ${type} = ${value}`;
  }
  if (init.kind === "expression") return isSemanticNoopExpression(init.expression) ? "" : emitSemanticExpression(init.expression, ir, names, options, textureSpecializations);
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
): string {
  switch (expression.kind) {
    case "literal":
      if (typeof expression.value !== "number") throw semanticWgslError("semantic WGSL supports numeric literals only", expression.span);
      return emitNumberLiteral(expression.value, expression.valueType);
    case "symbol":
      if (expression.addressSpace === "uniform") return `${UNIFORM_PARAMS_NAME}.${nameFor(expression.name, names)}`;
      if (expression.addressSpace === "constant") {
        const constant = constantMemorySymbols(ir).find((symbol) => symbol.name === expression.name);
        if (constant?.initialized) return nameFor(expression.name, names);
        if (isSemanticFloatVectorType(expression.valueType)) {
          return emitSemanticVectorMemoryRead({
            base: expression.name,
            addressSpace: "constant",
            valueType: expression.valueType as CudaLiteScalarType,
            indices: [zeroExpression(expression.span)],
            fields: [],
            span: expression.span,
          }, ir, names, options);
        }
        return `${UNIFORM_PARAMS_NAME}.${nameFor(expression.name, names)}`;
      }
      if (expression.addressSpace === "device-global") {
        const ref = `${nameFor(expression.name, names)}[0u]`;
        return semanticAtomicDeviceGlobalNames(ir.operations).has(expression.name) ? `atomicLoad(&${ref})` : ref;
      }
      if (expression.addressSpace === "shared" && (semanticAtomicSharedNames(ir.operations, ir.functions).has(expression.name) || semanticWgslFunctionSharedPointerAtomicParam(ir, expression.name))) {
        return `atomicLoad(&${nameFor(expression.name, names)})`;
      }
      return nameFor(expression.name, names);
    case "member":
      return emitSemanticMember(expression, ir, names, options);
    case "index": {
      if (semanticWgslVectorIndexSupported(expression, ir)) {
        const target = emitSemanticExpression(expression.target, ir, names, options, textureSpecializations);
        const index = emitSemanticExpressionAs(expression.index, ir, names, "u32", options, textureSpecializations);
        return `${target}[${index}]`;
      }
      const ref = memoryRefFromIndexExpression(expression);
      if (ref) {
        if (semanticWgslDirectByteRawView(ref, ir)) return emitSemanticMemoryRead(ref, ir, names, options);
        if (semanticWgslPackedSharedByteRoot(ref, ir)) return emitSemanticMemoryRead(ref, ir, names, options);
        if (semanticWgslLocalPackedHalfView(ref, ir)) return emitSemanticMemoryRead(ref, ir, names, options);
        if (semanticWgslLocalPackedHalf2View(ref, ir)) return emitSemanticMemoryRead(ref, ir, names, options);
        if (semanticWgslLocalScalarBitViewRootType(ref, ir) !== undefined) return emitSemanticMemoryRead(ref, ir, names, options);
        if (semanticWgslLocalVectorBitViewRootType(ref, ir) !== undefined) return emitSemanticMemoryRead(ref, ir, names, options);
        if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
          return emitSemanticMemoryRead(ref, ir, names, options);
        }
        if (semanticWgslLocalScalarVectorView(ref, ir)) {
          return emitSemanticVectorMemoryRead(ref, ir, names, options);
        }
        if (isSemanticFloatVectorType(ref.valueType) && ref.addressSpace === "local") {
          return emitSemanticMemoryRef(ref, ir, names, options);
        }
        if (semanticStorageVectorType(ref.valueType) !== undefined) return emitSemanticVectorMemoryRead(ref, ir, names, options);
        const memoryRef = emitSemanticMemoryRef(ref, ir, names, options);
        if (
          semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base) ||
          semanticAtomicDeviceGlobalNames(ir.operations).has(ref.base) ||
          semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base) ||
          semanticWgslFunctionSharedPointerAtomicParam(ir, ref.base)
        ) return emitSemanticAtomicLoad(ref, memoryRef);
        return memoryRef;
      }
      throw semanticWgslError("semantic WGSL does not support index target", expression.span);
    }
    case "cast":
      return emitSemanticCast(expression, ir, names, options, textureSpecializations);
    case "unary":
      if (semanticWgslBf162LocalBitsCastSupported(expression, ir)) return emitSemanticBf162LocalBitsCast(expression, ir, names, options, textureSpecializations);
      return emitSemanticUnary(expression, ir, names, options, textureSpecializations);
    case "binary":
      return emitSemanticBinary(expression, ir, names, options, textureSpecializations);
    case "conditional":
      return `select(${emitSemanticExpression(expression.alternate, ir, names, options, textureSpecializations)}, ${emitSemanticExpression(expression.consequent, ir, names, options, textureSpecializations)}, ${emitTruthiness(expression.condition, ir, names, options)})`;
    case "assignment":
      if (
        expression.target.kind === "member" && semanticWgslVectorMemberSupported(expression.target, ir) ||
        semanticWgslAssignmentMemoryRefSupported(expression.target, ir)
      ) return `(${emitSemanticAssignmentStatement(expression, ir, names, options, textureSpecializations)})`;
      if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local assignment targets only", expression.target.span);
      {
        const target = nameFor(expression.target.name, names);
        const targetType = expression.target.valueType;
        const value = targetType !== undefined && isSemanticFloatVectorType(targetType)
          ? emitSemanticVectorOperand(expression.value, targetType, ir, names, options, textureSpecializations)
          : emitSemanticLocalScalarExpressionAs(expression.value, expression.target.valueType, ir, names, options, textureSpecializations);
        if (targetType === "uchar" && expression.operator !== "=") {
          const binaryOperator = expression.operator.slice(0, -1);
          const right = emitSemanticExpressionAs(expression.value, ir, names, "u32", options, textureSpecializations);
          return `(${target} = ${emitSemanticUcharValue(`(${target} ${binaryOperator} ${right})`)})`;
        }
        if (semanticAssignmentBinaryOperator(expression.operator)) return `(${target} ${expression.operator} ${value})`;
        return `(${target} = ${value})`;
      }
    case "update":
      return emitSemanticUpdate(expression, ir, names, options);
    case "sequence":
      return emitSemanticExpression(expression.expressions.at(-1) ?? zeroExpression(expression.span), ir, names, options, textureSpecializations);
    case "call":
      if (expression.callee.kind === "symbol" && expression.callee.name === "__cvta_generic_to_shared") {
        const ref = semanticWgslSharedAddressCallRef(expression);
        if (!ref) throw semanticWgslError("__cvta_generic_to_shared requires modeled shared memory", expression.span);
        const elementBytes = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
        const index = emitSemanticSharedPointerArgBaseIndex(ref, ir, names);
        return elementBytes === 1 ? index : `(${index} * ${elementBytes}u)`;
      }
      if (semanticWgslCooperativeGroupCallSupported(expression, ir)) {
        const emitted = emitSemanticCooperativeGroupCall(
          expression,
          ir,
          options.activeFunction,
          semanticBallotHelper().scratchName,
        );
        if (emitted !== undefined) return emitted;
      }
      if (semanticWgslCooperativeReduceCallSupported(expression, ir, (value) => semanticWgslExpressionSupported(value, "scalar", ir))) {
        const helper = semanticCooperativeReduceHelperFor(ir, expression);
        const value = semanticCooperativeReduceValue(expression);
        if (helper && value) {
          const partitionGroup = helper.partitioned && expression.args[0]?.kind === "symbol"
            ? semanticCooperativeGroupInfo(ir, expression.args[0].name)
            : undefined;
          const partitionPredicate = options.activeCollectivePredicate ??
            (partitionGroup?.partitionPredicate === undefined
              ? undefined
              : emitTruthiness(partitionGroup.partitionPredicate, ir, names, options));
          const mask = helper.partitioned && partitionPredicate !== undefined
            ? `${semanticBallotHelper().name}(${partitionPredicate}, 0xffffffffu, local_id)`
            : helper.masked && expression.args[0]
              ? emitSemanticExpressionAs(expression.args[0], ir, names, "u32", options, textureSpecializations)
              : undefined;
          const emitted = emitSemanticCooperativeReduceCall(
            expression,
            ir,
            emitSemanticExpressionAs(value, ir, names, wgslValueScalar(helper.valueType), options, textureSpecializations),
            mask,
          );
          if (emitted !== undefined) return emitted;
        }
      }
      if (semanticWgslAtomicCallSupported(expression, ir)) return emitSemanticAtomicCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslCurandCallSupported(expression, ir)) return emitSemanticCurandCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslGeneratedRandomCallSupported(expression)) {
        const state = expression.args[0]!;
        if (state.kind !== "unary" || state.argument.kind !== "symbol") throw semanticWgslError("generated random helper expects local uint state", expression.span);
        return `${expression.callee.kind === "symbol" ? expression.callee.name : ""}(&${nameFor(state.argument.name, names)})`;
      }
      if (semanticWgslSubgroupCallSupported(expression, ir)) return emitSemanticSubgroupCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslAddressPredicateCallSupported(expression)) return emitSemanticAddressPredicateCall(expression);
      if (expression.callee.kind === "symbol" && semanticPtxIntegerCallInfo(expression.callee.name) !== undefined) {
        return emitSemanticPtxIntegerCall(expression, ir, names, options, textureSpecializations);
      }
      if (semanticWgslVectorConstructorSupported(expression, "any", ir)) return emitSemanticVectorConstructor(expression, ir, names, options, textureSpecializations);
      if (semanticWgslVectorAtCallSupported(expression, ir)) return emitSemanticVectorAtCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslVectorLerpCallSupported(expression, ir)) return emitSemanticVectorLerpCall(expression, ir, names, options, textureSpecializations);
      if (expression.callee.kind === "symbol" && semanticVectorMathCallSupported(expression.callee.name, expression.args)) {
        return emitSemanticVectorMathCall(expression, ir, names, options, textureSpecializations);
      }
      if (semanticWgslHalf2CallSupported(expression, ir)) return emitSemanticHalf2Call(expression, ir, names, options, textureSpecializations);
      if (semanticWgslBf162CallSupported(expression, ir)) return emitSemanticBf162Call(expression, ir, names, options, textureSpecializations);
      if (semanticWgslFunctionCallSupported(expression, ir)) return emitSemanticFunctionCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslMathCallSupported(expression, "any", ir)) return emitSemanticMathCall(expression, ir, names, options, textureSpecializations);
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
    case "texture-read":
      return emitSemanticTextureRead(expression, ir, names, options);
    case "surface-read":
      return emitSemanticSurfaceRead(expression, ir, names, options);
    case "initializer":
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
  }
}

function semanticWgslSharedAddressCallRef(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticMemoryRef | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.name !== "__cvta_generic_to_shared") return undefined;
  const arg = expression.args[0];
  if (!arg) return undefined;
  const target = arg.kind === "unary" && arg.operator === "&" ? arg.argument : arg;
  const ref = memoryRefFromIndexExpression(target) ?? (target.kind === "symbol" && target.addressSpace === "shared" ? {
    base: target.name,
    addressSpace: target.addressSpace,
    ...(target.valueType === undefined ? {} : { valueType: target.valueType }),
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
  const xBytes = emitSemanticExpressionAs(expression.xBytes, ir, names, "i32", options);
  const y = emitSemanticExpressionAs(expression.y, ir, names, "i32", options);
  const z = expression.z ? emitSemanticExpressionAs(expression.z, ir, names, "i32", options) : "0";
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

function emitSemanticTextureRead(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (!semanticWgslTextureReadSupported(expression, ir) || expression.texture.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL supports only direct texture reads", expression.span);
  }
  const x = emitSemanticExpressionAs(expression.x, ir, names, "f32", options);
  const y = emitSemanticExpressionAs(expression.y, ir, names, "f32", options);
  const atlasY = expression.z === undefined || expression.callee === "texCubemap"
    ? y
    : `(${y} + ${emitSemanticExpressionAs(expression.z, ir, names, "f32", options)})`;
  const texture = nameFor(expression.texture.name, names);
  const descriptor = expression.callee === "texCubemap" ? undefined : options.textureDescriptors?.[expression.texture.name];
  const read = expression.callee === "texCubemap"
    ? emitSemanticCubemapTextureRead(texture, x, y, emitSemanticExpressionAs(expression.z!, ir, names, "f32", options))
    : descriptor
    ? `${semanticTextureDescriptorHelperName(expression.texture.name, names, descriptor)}(${texture}, ${x}, ${atlasY})`
    : `textureLoad(${texture}, clamp(vec2<i32>(i32(floor(${x})), i32(floor(${atlasY}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${texture})) - vec2<i32>(1, 1)), 0)`;
  if (isSemanticFloatVectorType(expression.valueType)) return emitSemanticTextureVectorRead(read, expression.valueType);
  if (expression.valueType === "half") return `f16(${read}.r)`;
  if (expression.valueType === "bf16") return wgslRoundBfloat16(`${read}.r`);
  if (expression.valueType === "uint" || expression.valueType === "uchar") return `u32(${read}.r)`;
  if (expression.valueType === "int") return `i32(${read}.r)`;
  return `${read}.r`;
}

function emitSemanticCubemapTextureRead(texture: string, x: string, y: string, z: string): string {
  const width = `f32(textureDimensions(${texture}).x)`;
  const cubeX = `((bg_cube_u(${x}, ${y}, ${z}) + 1.0) * 0.5 * (${width} - 1.0))`;
  const cubeY = `((bg_cube_v(${x}, ${y}, ${z}) + 1.0) * 0.5 * (${width} - 1.0) + bg_cube_face(${x}, ${y}, ${z}) * ${width})`;
  return `textureLoad(${texture}, clamp(vec2<i32>(i32(floor(${cubeX})), i32(floor(${cubeY}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${texture})) - vec2<i32>(1, 1)), 0)`;
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

function emitSemanticTextureVectorRead(read: string, valueType: CudaLiteScalarType): string {
  if (valueType === "float2") return `${read}.xy`;
  if (valueType === "float3") return `${read}.xyz`;
  if (valueType === "float4") return read;
  if (valueType === "bf162") return `vec2<f32>(${wgslRoundBfloat16(`${read}.x`)}, ${wgslRoundBfloat16(`${read}.y`)})`;
  const laneCount = cudaVectorLaneCount(valueType);
  const vectorType = wgslValueType(valueType);
  const scalarType = wgslVectorScalar(valueType);
  const fields = ["x", "y", "z", "w"];
  return `${vectorType}(${Array.from({ length: laneCount }, (_, lane) => `${scalarType}((${read}).${fields[lane]})`).join(", ")})`;
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

function emitSemanticVectorConstructor(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const valueType = expression.callee.kind === "symbol" ? cudaVectorConstructorType(expression.callee.name) : undefined;
  if (!isSemanticFloatVectorType(valueType)) throw semanticWgslError("semantic WGSL vector constructor requires vector target", expression.span);
  const fields = ["x", "y", "z", "w"];
  const targetLanes = cudaVectorLaneCount(valueType);
  const targetScalar = wgslVectorScalar(valueType);
  const targetType = wgslValueType(valueType);
  if (expression.args.length === 1 && !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.args[0]!, ir?.functions))) {
    const scalar = emitSemanticExpressionAs(expression.args[0]!, ir, names, targetScalar, options, textureSpecializations);
    return `${targetType}(${Array.from({ length: targetLanes }, () => `${targetScalar}(${scalar})`).join(", ")})`;
  }
  const lanes = expression.args.flatMap((arg) => {
    const argType = semanticExpressionVectorValueType(arg, ir?.functions);
    if (isSemanticFloatVectorType(argType)) {
      const value = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
      return Array.from({ length: cudaVectorLaneCount(argType) }, (_, lane) => `${targetScalar}((${value}).${fields[lane]})`);
    }
    return [`${targetScalar}(${emitSemanticExpressionAs(arg, ir, names, targetScalar, options, textureSpecializations)})`];
  });
  while (lanes.length < targetLanes) lanes.push(zeroForType(targetScalar));
  return `${targetType}(${lanes.slice(0, targetLanes).join(", ")})`;
}

function emitSemanticVectorAtCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const [target, index] = expression.args;
  if (!target || !index) throw semanticWgslError("semantic WGSL vec_at requires vector and index", expression.span);
  return `${emitSemanticExpression(target, ir, names, options, textureSpecializations)}[${emitSemanticExpressionAs(index, ir, names, "u32", options, textureSpecializations)}]`;
}

function emitSemanticVectorLerpCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const [left, right, amount] = expression.args;
  if (!left || !right || !amount) throw semanticWgslError("semantic WGSL vector lerp requires three operands", expression.span);
  const valueType = semanticExpressionVectorValueType(left, ir?.functions);
  if (!isSemanticFloatVectorType(valueType)) throw semanticWgslError("semantic WGSL vector lerp requires vector endpoints", expression.span);
  const start = emitSemanticExpression(left, ir, names, options, textureSpecializations);
  const end = emitSemanticExpression(right, ir, names, options, textureSpecializations);
  const factor = emitSemanticVectorOperand(amount, valueType as CudaLiteScalarType, ir, names, options, textureSpecializations);
  return `fma(${factor}, (${end} - ${start}), ${start})`;
}

function emitSemanticVectorMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  if (expression.callee.kind !== "symbol" || !semanticVectorMathCallSupported(expression.callee.name, expression.args)) {
    throw semanticWgslError("semantic WGSL vector math call is unsupported", expression.span);
  }
  return `${expression.callee.name}(${expression.args.map((arg) =>
    emitSemanticExpression(arg, ir, names, options, textureSpecializations)).join(", ")})`;
}

function emitSemanticHalf2Call(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL half2 call requires symbol callee", expression.span);
  const name = expression.callee.name;
  const emitHalf2 = (arg: SemanticExpression): string => emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  if (isSemanticHalf2UnaryCall(name)) {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    return emitSemanticHalf2UnaryCall(name, emitHalf2(arg));
  }
  if (isSemanticHalf2ComparisonCall(name)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two half2 operands`, expression.span);
    return emitSemanticHalf2ComparisonCall(name, emitHalf2(left), emitHalf2(right));
  }
  if (name === "__hadd2" || name === "__hadd2_rn" || name === "__hadd2_sat" || name === "__hsub2" || name === "__hsub2_rn" || name === "__hsub2_sat" || name === "__hmul2" || name === "__hmul2_rn" || name === "__hmul2_sat") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two half2 operands`, expression.span);
    const operator = name === "__hadd2" || name === "__hadd2_rn" || name === "__hadd2_sat" ? "+" : name === "__hsub2" || name === "__hsub2_rn" || name === "__hsub2_sat" ? "-" : "*";
    const value = `(${emitHalf2(left)} ${operator} ${emitHalf2(right)})`;
    return name.endsWith("_sat") ? wgslSaturateHalf2(value) : value;
  }
  if (name === "__hmin2" || name === "__hmax2" || name === "__hmin2_nan" || name === "__hmax2_nan") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two half2 operands`, expression.span);
    const lhs = emitHalf2(left);
    const rhs = emitHalf2(right);
    if (name === "__hmin2_nan" || name === "__hmax2_nan") return emitSemanticHalf2NanMinMax(name === "__hmin2_nan" ? "min" : "max", lhs, rhs);
    return `${name === "__hmin2" ? "min" : "max"}(${lhs}, ${rhs})`;
  }
  if (name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${name} expects three half2 operands`, expression.span);
    const value = `fma(${emitHalf2(left)}, ${emitHalf2(right)}, ${emitHalf2(addend)})`;
    return name === "__hfma2_sat" ? wgslSaturateHalf2(value) : value;
  }
  if (name === "__half22float2") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    return `vec2<f32>(${emitHalf2(arg)})`;
  }
  if (name === "__float22half2_rn") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one float2 operand`, expression.span);
    return `vec2<f16>(${emitSemanticExpression(arg, ir, names, options, textureSpecializations)})`;
  }
  if (name === "__half2_as_uint") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    const emitted = emitHalf2(arg);
    return `pack2x16float(vec2<f32>(f32((${emitted}).x), f32((${emitted}).y)))`;
  }
  if (name === "__uint_as_half2") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one uint operand`, expression.span);
    return `vec2<f16>(unpack2x16float(${emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)}))`;
  }
  if (name === "__low2half" || name === "__high2half") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    return `(${emitHalf2(arg)}).${name === "__low2half" ? "x" : "y"}`;
  }
  if (name === "__low2float" || name === "__high2float") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    return `f32((${emitHalf2(arg)}).${name === "__low2float" ? "x" : "y"})`;
  }
  if (name === "__halves2half2") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two half operands`, expression.span);
    return `vec2<f16>(${emitSemanticExpressionAs(left, ir, names, "f16", options, textureSpecializations)}, ${emitSemanticExpressionAs(right, ir, names, "f16", options, textureSpecializations)})`;
  }
  if (name === "__half2half2") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half operand`, expression.span);
    const emitted = emitSemanticExpressionAs(arg, ir, names, "f16", options, textureSpecializations);
    return `vec2<f16>(${emitted}, ${emitted})`;
  }
  if (name === "__low2half2" || name === "__high2half2") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    const emitted = `(${emitHalf2(arg)}).${name === "__low2half2" ? "x" : "y"}`;
    return `vec2<f16>(${emitted}, ${emitted})`;
  }
  if (name === "__lows2half2" || name === "__highs2half2") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two half2 operands`, expression.span);
    const lane = name === "__lows2half2" ? "x" : "y";
    return `vec2<f16>((${emitHalf2(left)}).${lane}, (${emitHalf2(right)}).${lane})`;
  }
  if (name === "__lowhigh2highlow") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one half2 operand`, expression.span);
    const emitted = emitHalf2(arg);
    return `vec2<f16>((${emitted}).y, (${emitted}).x)`;
  }
  if (name === "__float2half2_rn") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one scalar operand`, expression.span);
    const emitted = emitSemanticExpressionAs(arg, ir, names, "f16", options, textureSpecializations);
    return `vec2<f16>(${emitted}, ${emitted})`;
  }
  if (name === "__floats2half2_rn") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two scalar operands`, expression.span);
    return `vec2<f16>(${emitSemanticExpressionAs(left, ir, names, "f16", options, textureSpecializations)}, ${emitSemanticExpressionAs(right, ir, names, "f16", options, textureSpecializations)})`;
  }
  throw semanticWgslError(`semantic WGSL does not support half2 call '${name}'`, expression.span);
}

function emitSemanticHalf2UnaryCall(name: string, value: string): string {
  switch (name) {
    case "__habs2": return `abs(${value})`;
    case "__hceil2": return `vec2<f16>(ceil(vec2<f32>(${value})))`;
    case "__hfloor2": return `vec2<f16>(floor(vec2<f32>(${value})))`;
    case "__hneg2": return `(-${value})`;
    case "__hrcp2": return `vec2<f16>(vec2<f32>(1.0) / vec2<f32>(${value}))`;
    case "__hrsqrt2": return `vec2<f16>(inverseSqrt(vec2<f32>(${value})))`;
    case "__hsqrt2": return `vec2<f16>(sqrt(vec2<f32>(${value})))`;
    case "__htrunc2": return `vec2<f16>(trunc(vec2<f32>(${value})))`;
    case "__hisnan2": return `select(vec2<f16>(0.0), vec2<f16>(1.0), ${emitSemanticHalf2IsNanPredicate(value)})`;
    default: return value;
  }
}

function emitSemanticHalf2ComparisonCall(name: string, left: string, right: string): string {
  const predicate = emitSemanticHalf2ComparisonPredicate(name, left, right);
  if (isSemanticHalf2MaskComparisonCall(name)) return `((select(0u, 0xffffu, (${predicate}).x)) | (select(0u, 0xffff0000u, (${predicate}).y)))`;
  if (isSemanticHalf2BooleanComparisonCall(name)) return `all(${predicate})`;
  return `select(vec2<f16>(0.0), vec2<f16>(1.0), ${predicate})`;
}

function emitSemanticHalf2ComparisonPredicate(name: string, left: string, right: string): string {
  const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
  const ordered = `!(${emitSemanticHalf2IsNanPredicate(left)} | ${emitSemanticHalf2IsNanPredicate(right)})`;
  const unordered = `(${emitSemanticHalf2IsNanPredicate(left)} | ${emitSemanticHalf2IsNanPredicate(right)})`;
  const op = normalized === "__heq2" || normalized === "__hequ2"
    ? "=="
    : normalized === "__hne2" || normalized === "__hneu2"
      ? "!="
      : normalized === "__hgt2" || normalized === "__hgtu2"
        ? ">"
        : normalized === "__hge2" || normalized === "__hgeu2"
          ? ">="
          : normalized === "__hlt2" || normalized === "__hltu2"
            ? "<"
            : "<=";
  const base = `((${left}) ${op} (${right}))`;
  return normalized.includes("u2") ? `(${unordered} | ${base})` : `(${ordered} & ${base})`;
}

function emitSemanticHalf2IsNanPredicate(value: string): string {
  const bits = `bitcast<vec2<u32>>(vec2<f32>(${value}))`;
  return `((${bits} & vec2<u32>(0x7fffffffu)) > vec2<u32>(0x7f800000u))`;
}

function emitSemanticBf162ComparisonPredicate(name: string, left: string, right: string): string {
  const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
  const ordered = `!(${emitSemanticBf162IsNanPredicate(left)} | ${emitSemanticBf162IsNanPredicate(right)})`;
  const unordered = `(${emitSemanticBf162IsNanPredicate(left)} | ${emitSemanticBf162IsNanPredicate(right)})`;
  const op = normalized === "__heq2" || normalized === "__hequ2"
    ? "=="
    : normalized === "__hne2" || normalized === "__hneu2"
      ? "!="
      : normalized === "__hgt2" || normalized === "__hgtu2"
        ? ">"
        : normalized === "__hge2" || normalized === "__hgeu2"
          ? ">="
          : normalized === "__hlt2" || normalized === "__hltu2"
            ? "<"
            : "<=";
  const base = `((${left}) ${op} (${right}))`;
  return normalized.includes("u2") ? `(${unordered} | ${base})` : `(${ordered} & ${base})`;
}

function emitSemanticBf162IsNanPredicate(value: string): string {
  const bits = `bitcast<vec2<u32>>(vec2<f32>(${value}))`;
  return `((${bits} & vec2<u32>(0x7fffffffu)) > vec2<u32>(0x7f800000u))`;
}

function emitSemanticHalfIsNanPredicate(value: string): string {
  return `((bitcast<u32>(f32(${value})) & 0x7fffffffu) > 0x7f800000u)`;
}

function emitSemanticBf16IsNanPredicate(value: string): string {
  return `((bitcast<u32>(f32(${value})) & 0x7fffffffu) > 0x7f800000u)`;
}

function emitSemanticHalfNanMinMax(op: "min" | "max", left: string, right: string): string {
  return `select(${op}(${left}, ${right}), (${left} + ${right}), ${emitSemanticHalfIsNanPredicate(left)} || ${emitSemanticHalfIsNanPredicate(right)})`;
}

function emitSemanticBf16NanMinMax(op: "min" | "max", left: string, right: string): string {
  return wgslRoundBfloat16(`select(${op}(${left}, ${right}), (${left} + ${right}), ${emitSemanticBf16IsNanPredicate(left)} || ${emitSemanticBf16IsNanPredicate(right)})`);
}

function emitSemanticHalf2NanMinMax(op: "min" | "max", left: string, right: string): string {
  return `select(${op}(${left}, ${right}), (${left} + ${right}), ${emitSemanticHalf2IsNanPredicate(left)} | ${emitSemanticHalf2IsNanPredicate(right)})`;
}

function emitSemanticBf162NanMinMax(op: "min" | "max", left: string, right: string): string {
  const value = `select(${op}(${left}, ${right}), (${left} + ${right}), ${emitSemanticBf162IsNanPredicate(left)} | ${emitSemanticBf162IsNanPredicate(right)})`;
  return `vec2<f32>(${wgslRoundBfloat16(`(${value}).x`)}, ${wgslRoundBfloat16(`(${value}).y`)})`;
}

function emitSemanticBf162UnaryCall(name: string, value: string): string {
  const lane = (body: (lane: "x" | "y") => string): string =>
    `vec2<f32>(${wgslRoundBfloat16(body("x"))}, ${wgslRoundBfloat16(body("y"))})`;
  switch (name) {
    case "__habs2": return lane((l) => `abs((${value}).${l})`);
    case "__hneg2": return lane((l) => `-(${value}).${l}`);
    case "h2ceil": return lane((l) => `ceil((${value}).${l})`);
    case "h2floor": return lane((l) => `floor((${value}).${l})`);
    case "h2rcp": return lane((l) => `(1.0 / (${value}).${l})`);
    case "h2rsqrt": return lane((l) => `inverseSqrt((${value}).${l})`);
    case "h2sqrt": return lane((l) => `sqrt((${value}).${l})`);
    case "h2trunc": return lane((l) => `trunc((${value}).${l})`);
    case "h2exp": return lane((l) => `exp((${value}).${l})`);
    case "h2exp2": return lane((l) => `exp2((${value}).${l})`);
    case "h2exp10": return lane((l) => `pow(10.0, (${value}).${l})`);
    case "h2log": return lane((l) => `log((${value}).${l})`);
    case "h2log2": return lane((l) => `log2((${value}).${l})`);
    case "h2log10": return lane((l) => `(log((${value}).${l}) / 2.302585092994046)`);
    case "h2sin": return lane((l) => `sin((${value}).${l})`);
    case "h2cos": return lane((l) => `cos((${value}).${l})`);
    case "h2tanh":
    case "h2tanh_approx": return lane((l) => `tanh((${value}).${l})`);
    case "h2rint": return lane((l) => `bg_semantic_round_even_f32((${value}).${l})`);
    default: return value;
  }
}

function emitSemanticBf162Call(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL bf162 call requires symbol callee", expression.span);
  const name = expression.callee.name;
  const emitBf162 = (arg: SemanticExpression): string => emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  const emitBf162Lane = (left: string, right: string, operator: string): string =>
    `vec2<f32>(${wgslRoundBfloat16(`(${left}).x ${operator} (${right}).x`)}, ${wgslRoundBfloat16(`(${left}).y ${operator} (${right}).y`)})`;
  if (SEMANTIC_BF162_UNARY_VECTOR_CALLS.has(name)) {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    const value = emitBf162(arg);
    return emitSemanticBf162UnaryCall(name, value);
  }
  if (SEMANTIC_BF162_BINARY_VECTOR_CALLS.has(name)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two bf162 operands`, expression.span);
    const lhs = emitBf162(left);
    const rhs = emitBf162(right);
    const operator = name === "__hadd2" || name === "__hadd2_rn" || name === "__hadd2_sat" ? "+" : name === "__hsub2" || name === "__hsub2_rn" || name === "__hsub2_sat" ? "-" : name === "__h2div" ? "/" : "*";
    const value = emitBf162Lane(lhs, rhs, operator);
    return name.endsWith("_sat") ? wgslSaturateBf162(value) : value;
  }
  if (SEMANTIC_BF162_TERNARY_VECTOR_CALLS.has(name)) {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${name} expects three bf162 operands`, expression.span);
    const lhs = emitBf162(left);
    const rhs = emitBf162(right);
    const acc = emitBf162(addend);
    if (name === "__hcmadd") {
      return `vec2<f32>(${wgslRoundBfloat16(`((${lhs}).x * (${rhs}).x - (${lhs}).y * (${rhs}).y + (${acc}).x)`)}, ${wgslRoundBfloat16(`((${lhs}).x * (${rhs}).y + (${lhs}).y * (${rhs}).x + (${acc}).y)`)})`;
    }
    const value = `vec2<f32>(${wgslRoundBfloat16(`fma((${lhs}).x, (${rhs}).x, (${acc}).x)`)}, ${wgslRoundBfloat16(`fma((${lhs}).y, (${rhs}).y, (${acc}).y)`)})`;
    if (name === "__hfma2_sat") return wgslSaturateBf162(value);
    if (name === "__hfma2_relu") return wgslReluBf162(value);
    return value;
  }
  if (SEMANTIC_BF162_MINMAX_VECTOR_CALLS.has(name)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two bf162 operands`, expression.span);
    const lhs = emitBf162(left);
    const rhs = emitBf162(right);
    if (name === "__hmin2_nan" || name === "__hmax2_nan") return emitSemanticBf162NanMinMax(name === "__hmin2_nan" ? "min" : "max", lhs, rhs);
    const op = name === "__hmin2" ? "min" : "max";
    return `vec2<f32>(${wgslRoundBfloat16(`${op}((${lhs}).x, (${rhs}).x)`)}, ${wgslRoundBfloat16(`${op}((${lhs}).y, (${rhs}).y)`)})`;
  }
  if (SEMANTIC_BF162_VECTOR_COMPARISON_CALLS.has(name)) {
    const [left, right] = expression.args;
    if (!left) throw semanticWgslError(`${name} expects bf162 operand`, expression.span);
    if (name === "__hisnan2") {
      const value = emitBf162(left);
      return `select(vec2<f32>(0.0), vec2<f32>(1.0), ${emitSemanticBf162IsNanPredicate(value)})`;
    }
    if (!right) throw semanticWgslError(`${name} expects two bf162 operands`, expression.span);
    return `select(vec2<f32>(0.0), vec2<f32>(1.0), ${emitSemanticBf162ComparisonPredicate(name, emitBf162(left), emitBf162(right))})`;
  }
  if (SEMANTIC_BF162_MASK_COMPARISON_CALLS.has(name) || SEMANTIC_BF162_BOOL_COMPARISON_CALLS.has(name)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two bf162 operands`, expression.span);
    const predicate = emitSemanticBf162ComparisonPredicate(name, emitBf162(left), emitBf162(right));
    if (SEMANTIC_BF162_BOOL_COMPARISON_CALLS.has(name)) return `all(${predicate})`;
    return `((select(0u, 0xffffu, (${predicate}).x)) | (select(0u, 0xffff0000u, (${predicate}).y)))`;
  }
  if (name === "__bfloat1622float2") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    return emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  }
  if (name === "__float22bfloat162_rn") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one float2 operand`, expression.span);
    const emitted = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
    return `vec2<f32>(${wgslRoundBfloat16(`(${emitted}).x`)}, ${wgslRoundBfloat16(`(${emitted}).y`)})`;
  }
  if (name === "__bfloat162bfloat162" || name === "__float2bfloat162_rn") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one scalar operand`, expression.span);
    const emitted = wgslRoundBfloat16(emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations));
    return `vec2<f32>(${emitted}, ${emitted})`;
  }
  if (name === "__halves2bfloat162") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two bf16 operands`, expression.span);
    return `vec2<f32>(${wgslRoundBfloat16(emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations))}, ${wgslRoundBfloat16(emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations))})`;
  }
  if (name === "__floats2bfloat162_rn") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two scalar operands`, expression.span);
    return `vec2<f32>(${wgslRoundBfloat16(emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations))}, ${wgslRoundBfloat16(emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations))})`;
  }
  if (name === "__bfloat162_as_uint" || name === "__nv_bfloat162_as_uint") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    const emitted = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
    return `((bitcast<u32>(f32((${emitted}).x)) >> 16u) | (bitcast<u32>(f32((${emitted}).y)) & 0xffff0000u))`;
  }
  if (name === "__uint_as_bfloat162" || name === "__uint_as_nv_bfloat162") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one uint operand`, expression.span);
    const bits = emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations);
    return `vec2<f32>(bitcast<f32>((${bits} & 0x0000ffffu) << 16u), bitcast<f32>(${bits} & 0xffff0000u))`;
  }
  if (name === "__low2bfloat16" || name === "__high2bfloat16") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    return wgslRoundBfloat16(`(${emitSemanticExpression(arg, ir, names, options, textureSpecializations)}).${name === "__low2bfloat16" ? "x" : "y"}`);
  }
  if (name === "__low2float" || name === "__high2float") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    return `f32((${emitSemanticExpression(arg, ir, names, options, textureSpecializations)}).${name === "__low2float" ? "x" : "y"})`;
  }
  if (name === "__low2bfloat162" || name === "__high2bfloat162") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    const emitted = wgslRoundBfloat16(`(${emitSemanticExpression(arg, ir, names, options, textureSpecializations)}).${name === "__low2bfloat162" ? "x" : "y"}`);
    return `vec2<f32>(${emitted}, ${emitted})`;
  }
  if (name === "__lows2bfloat162" || name === "__highs2bfloat162") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${name} expects two bf162 operands`, expression.span);
    const lane = name === "__lows2bfloat162" ? "x" : "y";
    return `vec2<f32>(${wgslRoundBfloat16(`(${emitSemanticExpression(left, ir, names, options, textureSpecializations)}).${lane}`)}, ${wgslRoundBfloat16(`(${emitSemanticExpression(right, ir, names, options, textureSpecializations)}).${lane}`)})`;
  }
  if (name === "__lowhigh2highlow") {
    const [arg] = expression.args;
    if (!arg) throw semanticWgslError(`${name} expects one bf162 operand`, expression.span);
    const emitted = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
    return `vec2<f32>(${wgslRoundBfloat16(`(${emitted}).y`)}, ${wgslRoundBfloat16(`(${emitted}).x`)})`;
  }
  throw semanticWgslError(`semantic WGSL does not support bf162 call '${name}'`, expression.span);
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
): string {
  const cast = expression.argument;
  if (cast.kind !== "cast" || cast.expression.kind !== "unary" || cast.expression.argument.kind !== "symbol") {
    throw semanticWgslError("semantic WGSL bf162 bitcast requires local bf162 symbol", expression.span);
  }
  const value = emitSemanticExpression(cast.expression.argument, ir, names, options, textureSpecializations);
  return `((bitcast<u32>(f32((${value}).x)) >> 16u) | (bitcast<u32>(f32((${value}).y)) & 0xffff0000u))`;
}

function emitSemanticCast(
  expression: Extract<SemanticExpression, { readonly kind: "cast" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.valueType === "uchar") {
    return emitSemanticUcharExpression(expression.expression, ir, names, options, textureSpecializations);
  }
  const value = emitSemanticExpression(expression.expression, ir, names, options, textureSpecializations);
  const sourceType = "valueType" in expression.expression ? expression.expression.valueType : undefined;
  if (expression.valueType === "int" && sourceType === "uint") return `bitcast<i32>(${value})`;
  if (expression.valueType === "uint" && sourceType === "int") return `bitcast<u32>(${value})`;
  return `${wgslScalar(expression.valueType)}(${value})`;
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
  if (isSemanticFloatVectorType(param?.valueType)) return emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  return emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param?.valueType), options, textureSpecializations);
}

function emitSemanticFunctionArgs(
  arg: SemanticExpression,
  param: SemanticKernelIrModule["functions"][number]["params"][number] | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (param?.pointer && param.addressSpace === "constant" && param.pointerAliasOf !== undefined) return [];
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
      : nameFor(sourceParam.pointerAliasOf ?? sourceParam.name, names);
    return param.pointerAliasOf === undefined ? [pointer, base] : [base];
  }
  if (param?.pointer && param.addressSpace === "local") {
    const ref = semanticPointerArgMemoryRef(arg);
    const owner = options.activeFunction === undefined
      ? undefined
      : ir.functions.find((candidate) => candidate.name === options.activeFunction);
    const forwarded = ref?.indices.length === 1 && isSemanticZeroLiteral(ref.indices[0]) &&
      owner?.params.some((candidate) => candidate.name === ref.base && candidate.pointer && candidate.addressSpace === "local");
    if (ref && forwarded) return [nameFor(ref.base, names)];
    if (!ref || ref.addressSpace !== "local" || ref.indices.length !== 0) {
      throw semanticWgslError("semantic WGSL local pointer helper argument must be a local scalar", arg.span);
    }
    return [`&${nameFor(ref.base, names)}`];
  }
  if (param?.pointer && param.addressSpace === "storage") {
    const ref = semanticPointerArgMemoryRef(arg);
    if (!ref || ref.addressSpace !== "storage") throw semanticWgslError("semantic WGSL storage pointer helper argument must be modeled storage", arg.span);
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

function isSemanticZeroLiteral(expression: SemanticExpression | undefined): boolean {
  return expression?.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
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
    const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32");
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
  if (isSemanticFloatVectorType(valueType)) {
    const vectorType = valueType as CudaLiteScalarType;
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
  if (!isSemanticFloatVectorType(root.valueType)) return emitSemanticRootStorageIndex(ref, ir, names, options);
  const base = emitSemanticRootStorageIndex({ ...ref, valueType: "float" }, ir, names, options);
  if (ref.pointerBaseIsScalarLane === true || !isSemanticFloatVectorType(ref.valueType)) return base;
  const stride = cudaVectorLaneCount(root.valueType);
  return stride === 1 ? base : `(${base} * ${stride}u)`;
}

function emitSemanticUpdate(
  expression: Extract<SemanticExpression, { readonly kind: "update" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const ref = memoryRefFromIndexExpression(expression.argument);
  if (ref) {
    if (semanticWgslPackedSharedByteRoot(ref, ir) && ref.valueType === "uchar") {
      const byteIndex = emitSemanticPackedSharedByteIndex(ref, ir, names, options);
      const word = `${nameFor(ref.base, names)}[(${byteIndex} >> 2u)]`;
      const shift = `((${byteIndex} & 3u) * 8u)`;
      const delta = expression.operator === "++" ? "1" : "-1";
      return `${PACKED_SHARED_U8_ADD}(&${word}, ${shift}, ${delta})`;
    }
    const target = emitSemanticMemoryRead(ref, ir, names, options);
    const next = `(${target} ${expression.operator === "++" ? "+" : "-"} ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))})`;
    return emitSemanticMemoryWrite(ref, next, ir, names, options);
  }
  if (expression.argument.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar or modeled memory updates only", expression.span);
  const name = nameFor(expression.argument.name, names);
  if (expression.operator === "++") return `${name} += ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))}`;
  if (expression.operator === "--") return `${name} -= ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))}`;
  throw semanticWgslError(`semantic WGSL does not support update '${expression.operator}'`, expression.span);
}

function emitSemanticMemoryRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (semanticWgslLocalPackedHalf2View(ref, ir)) {
    const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
    const root = localArraySymbol(ir, ref.base)!;
    const word = `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(index, root.dimensions)}`;
    const value = `unpack2x16float(${word})`;
    return effectiveSemanticF16Mode(ir, options) === "native" ? `vec2<f16>(${value})` : value;
  }
  if (semanticWgslLocalPackedHalfView(ref, ir)) {
    const halfIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
    const packed = emitSemanticLocalPackedHalfWord(ref, halfIndex, ir, names);
    const lane = `unpack2x16float(${packed})[(${halfIndex} & 1u)]`;
    return effectiveSemanticF16Mode(ir, options) === "native" ? `f16(${lane})` : lane;
  }
  const bitRootType = semanticWgslLocalScalarBitViewRootType(ref, ir);
  const vectorBitRootType = semanticWgslLocalVectorBitViewRootType(ref, ir);
  if (bitRootType !== undefined || vectorBitRootType !== undefined) {
    const rootType = bitRootType ?? vectorBitRootType!;
    const access = vectorBitRootType === undefined
      ? emitSemanticMemoryRef({ ...ref, valueType: rootType }, ir, names, options)
      : emitSemanticLocalVectorBitViewAccess(ref, ir, names, options);
    return `bitcast<${wgslValueType(ref.valueType)}>(${access})`;
  }
  if (semanticWgslDirectByteRawView(ref, ir)) {
    const base = emitFlatStorageIndex({ ...ref, valueType: "uchar" }, ir, names, options);
    const storage = nameFor(ref.base, names);
    const word = `(${storage}[${base}] | (${storage}[(${base} + 1u)] << 8u) | (${storage}[(${base} + 2u)] << 16u) | (${storage}[(${base} + 3u)] << 24u))`;
    if (ref.valueType === "float") return `bitcast<f32>(${word})`;
    return ref.valueType === "int" ? `bitcast<i32>(${word})` : word;
  }
  if (semanticWgslPackedSharedByteRoot(ref, ir)) {
    return emitSemanticPackedSharedByteRead(ref, ir, names, options);
  }
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
    const valueType = ref.valueType ?? "float";
    const index = isCudaVectorType(valueType) ? emitFlatStorageVectorBaseIndex(ref, ir, names, options) : emitFlatStorageIndex(ref, ir, names, options);
    return `${semanticPointerReadHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(ref.base), names)}, ${index})`;
  }
  if (semanticWgslFunctionSharedPointerParam(ir, ref.base)) {
    const target = emitSemanticSharedPointerMemoryRef(ref, ir, names, options);
    const param = semanticWgslFunctionSharedPointerParam(ir, ref.base)!;
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
    const index = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
    const root = localArraySymbol(ir, ref.base)!;
    const target = `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(index, root.dimensions)}`;
    return `${target} = pack2x16float(vec2<f32>(${value}))`;
  }
  if (semanticWgslLocalPackedHalfView(ref, ir)) {
    const halfIndex = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
    const word = emitSemanticLocalPackedHalfWord(ref, halfIndex, ir, names);
    const bits = `(pack2x16float(vec2<f32>(f32(${value}), 0.0)) & 0xffffu)`;
    return `${word} = select((${word} & 0xffff0000u) | ${bits}, (${word} & 0x0000ffffu) | (${bits} << 16u), (${halfIndex} & 1u) != 0u)`;
  }
  const bitRootType = semanticWgslLocalScalarBitViewRootType(ref, ir);
  const vectorBitRootType = semanticWgslLocalVectorBitViewRootType(ref, ir);
  if (bitRootType !== undefined || vectorBitRootType !== undefined) {
    const rootType = bitRootType ?? vectorBitRootType!;
    const target = vectorBitRootType === undefined
      ? emitSemanticMemoryRef({ ...ref, valueType: rootType }, ir, names, options)
      : emitSemanticLocalVectorBitViewAccess(ref, ir, names, options);
    return `${target} = bitcast<${wgslValueType(rootType)}>(${value})`;
  }
  if (semanticWgslDirectByteRawView(ref, ir)) {
    const base = emitFlatStorageIndex({ ...ref, valueType: "uchar" }, ir, names, options);
    const storage = nameFor(ref.base, names);
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
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
    const valueType = ref.valueType ?? "float";
    const index = isCudaVectorType(valueType) ? emitFlatStorageVectorBaseIndex(ref, ir, names, options) : emitFlatStorageIndex(ref, ir, names, options);
    return `${semanticPointerWriteHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(ref.base), names)}, ${index}, ${value})`;
  }
  if (semanticWgslFunctionSharedPointerParam(ir, ref.base)) {
    const target = emitSemanticSharedPointerMemoryRef(ref, ir, names, options);
    const param = semanticWgslFunctionSharedPointerParam(ir, ref.base)!;
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
  return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(index, ir, names, "u32", options)}]`;
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
    const index = ref.indices[0] === undefined ? "0u" : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options);
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
  const pointerName = param.pointerAliasOf ?? ref.base;
  if (param.dimensions.length === 0) {
    if (ref.indices.length > 1 || ref.indices[0] && !semanticExpressionIsZero(ref.indices[0])) {
      throw semanticWgslError(`shared scalar pointer '${ref.base}' cannot be indexed`, ref.span);
    }
    return `*${nameFor(pointerName, names)}`;
  }
  const index = ref.indices[0] === undefined ? "0u" : emitSemanticExpressionAs(ref.indices[0], ir, names, "u32", options);
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
): string {
  if (expression.kind === "literal" && typeof expression.value === "number") {
    return emitNumberLiteral(expression.value, expression.valueType, targetType);
  }
  if (expression.kind === "unary" && expression.operator === "~" && (targetType === "u32" || targetType === "i32")) {
    return `~(${emitSemanticExpressionAs(expression.argument, ir, names, targetType, options, textureSpecializations)})`;
  }
  const emitted = emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  const atomicValueType = semanticAtomicCallValueType(expression);
  if (atomicValueType) {
    const sourceType = wgslAtomicScalar(atomicValueType);
    if (sourceType === targetType) return emitted;
    return `${targetType}(${emitted})`;
  }
  if (expression.kind === "call" && semanticWgslMathCallSupported(expression)) {
    const sourceType = semanticExpressionWgslScalar(expression);
    if (sourceType === targetType) return emitted;
    return `${targetType}(${emitted})`;
  }
  const sourceType = semanticExpressionWgslScalar(expression);
  if (sourceType === targetType) return emitted;
  return `${targetType}(${emitted})`;
}

function emitSemanticInitExpression(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (valueType === "bool") return emitSemanticBoolExpression(expression, ir, names, options, textureSpecializations);
  if (valueType === "uchar") return emitSemanticUcharExpression(expression, ir, names, options, textureSpecializations);
  if (isSemanticFloatVectorType(valueType)) return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
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
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
}

function emitSemanticUcharExpression(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.kind === "cast" && expression.valueType === "uchar") {
    return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  }
  return emitSemanticUcharValue(emitSemanticExpressionAs(expression, ir, names, "i32", options, textureSpecializations));
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
  if (expression.kind === "literal" && typeof expression.value === "number") return expression.value === 0 ? "false" : "true";
  const emitted = emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  if (semanticNativeBoolExpression(expression)) return emitted;
  const sourceType = semanticExpressionWgslScalar(expression);
  if (sourceType === "u32") return `(${emitted} != 0u)`;
  if (sourceType === "i32") return `(${emitted} != 0)`;
  return `(${emitted} != 0.0)`;
}

function semanticNativeBoolExpression(expression: SemanticExpression): boolean {
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
      .map((value) => emitSemanticExpressionAs(value, ir, names, "f32", options));
    while (values.length < laneCount) values.push("0.0");
    return `const ${nameFor(symbol.name, names)}: ${valueType} = ${valueType}(${values.join(", ")});`;
  }
  return `const ${nameFor(symbol.name, names)}: ${wgslValueType(symbol.valueType)} = ${emitSemanticInitExpression(symbol.init ?? zeroExpression(symbol.span), symbol.valueType, ir, names, options)};`;
}

function emitSemanticAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL atomic call requires symbol callee", expression.span);
  const wgslCallee = wgslAtomicCalleeForCudaAtomic(expression.callee.name);
  const loopAtomicKind = wgslIntegerLoopAtomicKindForCudaAtomic(expression.callee.name);
  const target = semanticAtomicCallTarget(expression);
  if (!target || (!wgslCallee && !loopAtomicKind && !semanticWgslAtomicValueTypeSupported(expression.callee.name, target.valueType))) {
    throw semanticWgslError(`semantic WGSL does not support atomic '${expression.callee.name}'`, expression.span);
  }
  const pointerAtomic = emitSemanticPointerAtomicCall(expression, target, ir, names, options, textureSpecializations);
  if (pointerAtomic) return pointerAtomic;
  const memoryRef = emitSemanticMemoryRef(target, ir, names, options);
  const operands = expression.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  if (loopAtomicKind) {
    const [limit] = operands;
    if (!limit) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing limit`, expression.span);
    return `${semanticIntegerLoopAtomicHelperName(loopAtomicKind, target, ir)}(&${memoryRef}, ${emitSemanticExpressionAs(limit, ir, names, "u32", options, textureSpecializations)})`;
  }
  if (semanticAtomicSupportsBfloatAdd(expression.callee.name, target.valueType)) {
    const [value] = operands;
    if (!value) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing value`, expression.span);
    return `${bfloatAtomicAddHelperName(semanticWgslAtomicAddressSpace(target))}(&${memoryRef}, ${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`;
  }
  const floatAtomicKind = semanticAtomicUsesF32Storage(target.valueType) ? semanticWgslFloatAtomicCallKind(expression.callee.name) : undefined;
  if (floatAtomicKind) {
    if (floatAtomicKind === "Exchange") {
      const [value] = operands;
      if (!value) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing value`, expression.span);
      return `bitcast<f32>(atomicExchange(&${memoryRef}, bitcast<u32>(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})))`;
    }
    if (floatAtomicKind === "CompareExchange") {
      const [compare, value] = operands;
      if (!compare || !value) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing operand`, expression.span);
      const emittedCompare = emitSemanticExpressionAs(compare, ir, names, "f32", options, textureSpecializations);
      const emittedValue = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
      return `bitcast<f32>(atomicCompareExchangeWeak(&${memoryRef}, bitcast<u32>(${emittedCompare}), bitcast<u32>(${emittedValue})).old_value)`;
    }
    const [value] = operands;
    if (!value) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing value`, expression.span);
    return `${floatAtomicHelperName(floatAtomicKind, semanticWgslAtomicAddressSpace(target))}(&${memoryRef}, ${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`;
  }
  const emitted = operands.map((operand) => emitSemanticExpressionAs(operand, ir, names, wgslAtomicScalar(target.valueType), options, textureSpecializations));
  const call = `${wgslCallee}(&${memoryRef}, ${emitted.join(", ")})`;
  return wgslCallee === "atomicCompareExchangeWeak" ? `${call}.old_value` : call;
}

function emitSemanticPointerAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  target: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string | undefined {
  if (!semanticWgslFunctionStoragePointerParam(ir, target.base)) return undefined;
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
    ...(compare ? [emitSemanticExpressionAs(compare, ir, names, wgslValueScalar(valueType), options, textureSpecializations)] : []),
    emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations),
  ];
  return `${semanticPointerAtomicHelperName(expression.callee.name, valueType)}(${args.join(", ")})`;
}

function emitSemanticAddressPredicateCall(expression: Extract<SemanticExpression, { readonly kind: "call" }>): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL address predicate requires symbol callee", expression.span);
  const addressSpace = semanticAddressPredicateAddressSpace(expression.args[0]);
  const kind = cudaAddressSpacePredicateKind(expression.callee.name);
  const matches =
    kind === "global" ? addressSpace === "storage" || addressSpace === "device-global" :
      kind !== undefined ? addressSpace === kind :
        false;
  return matches ? "1" : "0";
}

function emitSemanticPtxIntegerCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic PTX integer call requires symbol callee", expression.span);
  const info = semanticPtxIntegerCallInfo(expression.callee.name);
  if (!info) throw semanticWgslError(`unknown semantic PTX integer call '${expression.callee.name}'`, expression.span);
  const args = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations));
  const emitted = info.family === "arithmetic"
    ? emitInlineArithmeticWgsl(info.op, args[0] ?? "0u", args[1] ?? "0u", args[2] ?? "0u")
    : info.family === "shift"
      ? emitInlineShiftWgsl(info.op, args[0] ?? "0u", args[1] ?? "0u", info.signed)
      : info.family === "minmax"
        ? emitInlineMinMaxWgsl(info.op, args[0] ?? "0u", args[1] ?? "0u", info.signed)
        : info.family === "unary"
          ? emitInlineUnaryIntWgsl(info.op, args[0] ?? "0u")
          : info.family === "prmt"
            ? emitInlineBytePermWgsl(args[0] ?? "0u", args[1] ?? "0u", args[2] ?? "0u")
            : info.family === "lop3"
              ? emitInlineLop3Wgsl(args[0] ?? "0u", args[1] ?? "0u", args[2] ?? "0u", args[3] ?? "0u")
              : info.family === "select"
                ? emitInlineSelectWgsl(args[0] ?? "0u", args[1] ?? "0u", args[2] ?? "0u")
                : emitInlineCompareWgsl(
                    info.op,
                    info.signed ? `bitcast<i32>(${args[0] ?? "0u"})` : args[0] ?? "0u",
                    info.signed ? `bitcast<i32>(${args[1] ?? "0u"})` : args[1] ?? "0u",
                  );
  return expression.valueType === "int" ? `bitcast<i32>(${emitted})` : emitted;
}

function emitSemanticSubgroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL subgroup call requires symbol callee", expression.span);
  const name = expression.callee.name;
  if (name === "__activemask") {
    if (ir.subgroupMode === "scalar") return "1u";
    return `${semanticBallotHelper().name}(${options.activeCollectivePredicate ?? "true"}, 0xffffffffu, local_id)`;
  }
  const value = expression.args[isCudaWarpReduceCallName(name) ? expression.args.length - 1 : name === "bg_subgroup_add" || legacyVoteCall(name) || legacyShuffleCall(name) ? 0 : 1];
  if (!value) throw semanticWgslError(`${name} expects value operand`, expression.span);
  const voteOp = cudaVoteOpForCall(name);
  if (ir.subgroupMode === "scalar") {
    if (voteOp === "any" || voteOp === "all" || voteOp === "ballot") {
      return `select(0u, 1u, ${emitTruthiness(value, ir, names, options)})`;
    }
    if (voteOp === "match-any") return "1u";
    const valueType = semanticExpressionValueType(value);
    if (!valueType || valueType === "void") throw semanticWgslError(`${name} expects scalar value operand`, expression.span);
    return emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
  }
  if (voteOp === "any" || voteOp === "all" || voteOp === "ballot") {
    const predicate = emitTruthiness(value, ir, names, options);
    const activeMask = legacyVoteCall(name)
      ? "0xffffffffu"
      : emitSemanticExpressionAs(expression.args[0]!, ir, names, "u32", options, textureSpecializations);
    const ballot = `${semanticBallotHelper().name}(${predicate}, ${activeMask}, local_id)`;
    if (voteOp === "any") return `select(0u, 1u, ${ballot} != 0u)`;
    if (voteOp === "all") {
      const failed = `${semanticBallotHelper().name}(!(${predicate}), ${activeMask}, local_id)`;
      return `select(0u, 1u, ${failed} == 0u)`;
    }
    return ballot;
  }
  if (voteOp === "match-any") {
    const valueType = semanticExpressionValueType(value);
    if (!valueType || valueType === "void") throw semanticWgslError(`${name} expects scalar value operand`, expression.span);
    const helper = semanticMatchAnyHelper(valueType, 32);
    return `${helper.name}(${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations)}, 32u, local_id)`;
  }
  const bitwiseReduceOp = semanticBitwiseReduceOpForCall(name);
  if (bitwiseReduceOp) {
    const valueType = semanticExpressionValueType(value);
    if (valueType !== "int" && valueType !== "uint") throw semanticWgslError(`${name} expects int or uint value operand`, expression.span);
    const helper = semanticBitwiseReduceHelper(bitwiseReduceOp, valueType, 32);
    return `${helper.name}(${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations)}, 32u, local_id)`;
  }
  const shuffleOp = semanticShuffleOpForCall(name);
  if (shuffleOp) {
    const valueType = semanticExpressionValueType(value);
    if (!valueType || valueType === "void") throw semanticWgslError(`${name} expects scalar value operand`, expression.span);
    const helper = semanticWarpShuffleHelper(shuffleOp, valueType, semanticShuffleTileSize(expression));
    const indexArg = legacyShuffleCall(name) ? expression.args[1] : expression.args[2];
    const widthArg = legacyShuffleCall(name) ? expression.args[2] : expression.args[3];
    const index = indexArg ? emitSemanticExpressionAs(indexArg, ir, names, "u32", options, textureSpecializations) : "0u";
    const width = widthArg ? emitSemanticExpressionAs(widthArg, ir, names, "u32", options, textureSpecializations) : "32u";
    return `${helper.name}(${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations)}, ${index}, ${width}, local_id)`;
  }
  const arithmeticReduceOp = cudaArithmeticReduceOpForCall(name);
  if (arithmeticReduceOp !== undefined) {
    const scalar = semanticExpressionWgslScalar(value);
    const wgslCall = arithmeticReduceOp === "add" ? "subgroupAdd" : arithmeticReduceOp === "min" ? "subgroupMin" : "subgroupMax";
    return `${wgslCall}(${emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations)})`;
  }
  throw semanticWgslError(`semantic WGSL does not support subgroup call '${name}'`, expression.span);
}

function emitSemanticMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL math call requires symbol callee", expression.span);
  const wgslCallee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  if (!wgslCallee) throw semanticWgslError(`semantic WGSL does not support math call '${expression.callee.name}'`, expression.span);
  if (wgslCallee === "clock") {
    return "u32(workgroup_id.x * 104729u + workgroup_id.y * 1009u + workgroup_id.z * 97u + local_id.x + local_id.y * 31u + local_id.z * 7u)";
  }
  if (wgslCallee === "min" || wgslCallee === "max") {
    const vectorType = semanticVectorMinMaxCallValueType(expression.callee.name, expression.args);
    if (vectorType !== undefined) {
      return `${wgslCallee}(${expression.args.map((arg) => emitSemanticVectorOperand(arg, vectorType, ir, names, options, textureSpecializations)).join(", ")})`;
    }
    const scalar = semanticMathCallOperandType(expression.args);
    return `${wgslCallee}(${expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, scalar, options, textureSpecializations)).join(", ")})`;
  }
  if (wgslCallee === "div_ceil") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const scalar = semanticExpressionWgslScalar(left) === "u32" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, scalar, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, scalar, options, textureSpecializations);
    return `(((${lhs} + ${rhs}) - ${scalar === "u32" ? "1u" : "1"}) / ${rhs})`;
  }
  if (wgslCallee === "assert") return "0";
  if (wgslCallee === "tf32") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
  }
  if (wgslCallee === "float_as_int" || wgslCallee === "float_as_uint") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    return wgslCallee === "float_as_int" ? `bitcast<i32>(${emitted})` : `bitcast<u32>(${emitted})`;
  }
  if (wgslCallee === "half_to_float" || wgslCallee === "to_half" || wgslCallee === "int_to_half" || wgslCallee === "uint_to_half" || wgslCallee === "half_as_short" || wgslCallee === "half_as_ushort" || wgslCallee === "short_as_half" || wgslCallee === "ushort_as_half") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    if (wgslCallee === "half_to_float") return `f32(${emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations)})`;
    if (wgslCallee === "to_half") return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)}, 0u)).x)`;
    if (wgslCallee === "int_to_half") return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)}), 0u)).x)`;
    if (wgslCallee === "uint_to_half") return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)}), 0u)).x)`;
    if (wgslCallee === "half_as_short") return `((bitcast<i32>((pack2x16float(vec2<f32>(f32(${emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations)}), 0.0)) & 0xffffu) << 16u)) >> 16)`;
    if (wgslCallee === "half_as_ushort") return `(pack2x16float(vec2<f32>(f32(${emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations)}), 0.0)) & 0xffffu)`;
    if (wgslCallee === "short_as_half") return `f16(unpack2x16float(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)} & 0xffffu).x)`;
    return `f16(unpack2x16float(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)}).x)`;
  }
  if (
    wgslCallee === "bf16_to_float" ||
    wgslCallee === "to_bf16" ||
    wgslCallee === "double_to_bf16" ||
    wgslCallee === "int_to_bf16" ||
    wgslCallee === "uint_to_bf16" ||
    wgslCallee === "bf16_as_short" ||
    wgslCallee === "bf16_as_ushort" ||
    wgslCallee === "short_as_bf16" ||
    wgslCallee === "ushort_as_bf16"
  ) {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    if (wgslCallee === "bf16_to_float") return `f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`;
    if (wgslCallee === "to_bf16") return wgslRoundBfloat16(emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations));
    if (wgslCallee === "double_to_bf16") return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`);
    if (wgslCallee === "int_to_bf16") return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)})`);
    if (wgslCallee === "uint_to_bf16") return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)})`);
    if (wgslCallee === "bf16_as_short") return `((bitcast<i32>(((bitcast<u32>(f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})) >> 16u) & 0xffffu) << 16u)) >> 16)`;
    if (wgslCallee === "bf16_as_ushort") return `((bitcast<u32>(f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})) >> 16u) & 0xffffu)`;
    if (wgslCallee === "short_as_bf16") return `bitcast<f32>((u32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)}) & 0xffffu) << 16u)`;
    return `bitcast<f32>(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)} << 16u)`;
  }
  if (
    wgslCallee.startsWith("float_to_half_") ||
    wgslCallee.startsWith("int_to_half_") ||
    wgslCallee.startsWith("uint_to_half_") ||
    wgslCallee.startsWith("short_to_half_") ||
    wgslCallee.startsWith("ushort_to_half_")
  ) {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const mode = halfConversionModeLiteral(wgslCallee);
    if (wgslCallee.startsWith("float_to_half_")) {
      return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)}, ${mode})).x)`;
    }
    if (wgslCallee.startsWith("int_to_half_")) {
      return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)}), ${mode})).x)`;
    }
    if (wgslCallee.startsWith("uint_to_half_")) {
      return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)}), ${mode})).x)`;
    }
    if (wgslCallee.startsWith("short_to_half_")) {
      return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(bg_i16_to_f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)}), ${mode})).x)`;
    }
    return `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)} & 0xffffu), ${mode})).x)`;
  }
  if (
    wgslCallee.startsWith("float_to_bf16_") ||
    wgslCallee.startsWith("int_to_bf16_") ||
    wgslCallee.startsWith("uint_to_bf16_") ||
    wgslCallee.startsWith("short_to_bf16_") ||
    wgslCallee.startsWith("ushort_to_bf16_")
  ) {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const mode = halfConversionModeLiteral(wgslCallee);
    if (wgslCallee.startsWith("float_to_bf16_")) {
      return wgslRoundBfloat16(emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), mode);
    }
    if (wgslCallee.startsWith("int_to_bf16_")) {
      return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)})`, mode);
    }
    if (wgslCallee.startsWith("uint_to_bf16_")) {
      return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)})`, mode);
    }
    if (wgslCallee.startsWith("short_to_bf16_")) {
      return wgslRoundBfloat16(`bg_bf16_i16_to_f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)})`, mode);
    }
    return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)} & 0xffffu)`, mode);
  }
  if (wgslCallee === "fp8_to_half") {
    const [bits, mode] = expression.args;
    if (!bits || !mode) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return `f16(bg_fp8_to_f32(${emitSemanticExpressionAs(bits, ir, names, "u32", options, textureSpecializations)}, ${emitSemanticExpressionAs(mode, ir, names, "u32", options, textureSpecializations)}))`;
  }
  if (wgslCallee === "float_to_fp8") {
    const [value, saturate, mode] = expression.args;
    if (!value || !saturate || !mode) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    return `bg_f32_to_fp8(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)}, ${emitSemanticExpressionAs(saturate, ir, names, "u32", options, textureSpecializations)}, ${emitSemanticExpressionAs(mode, ir, names, "u32", options, textureSpecializations)})`;
  }
  if (wgslCallee.startsWith("half_to_int_") || wgslCallee.startsWith("half_to_short_") || wgslCallee.startsWith("half_to_uint_") || wgslCallee.startsWith("half_to_ushort_")) {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = `f32(${emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations)})`;
    const rounded = wgslCallee.endsWith("_rn")
      ? emitRoundEvenWgsl(emitted)
      : wgslCallee.endsWith("_rz")
      ? `trunc(${emitted})`
      : wgslCallee.endsWith("_ru")
      ? `ceil(${emitted})`
      : `floor(${emitted})`;
    return wgslCallee.startsWith("half_to_uint_") || wgslCallee.startsWith("half_to_ushort_") ? `u32(max(${rounded}, 0.0))` : `i32(${rounded})`;
  }
  if (wgslCallee === "bf16_to_float" || wgslCallee === "to_bf16" || wgslCallee === "double_to_bf16" || wgslCallee === "int_to_bf16" || wgslCallee === "uint_to_bf16" || wgslCallee === "bf16_as_ushort" || wgslCallee === "ushort_as_bf16") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    if (wgslCallee === "bf16_to_float") return `f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`;
    if (wgslCallee === "to_bf16") return wgslRoundBfloat16(emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations));
    if (wgslCallee === "double_to_bf16") return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`);
    if (wgslCallee === "int_to_bf16") return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "i32", options, textureSpecializations)})`);
    if (wgslCallee === "uint_to_bf16") return wgslRoundBfloat16(`f32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)})`);
    if (wgslCallee === "bf16_as_ushort") return `((bitcast<u32>(f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})) >> 16u) & 0xffffu)`;
    return `bitcast<f32>(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)} << 16u)`;
  }
  if (
    wgslCallee.startsWith("bf16_to_int_") ||
    wgslCallee.startsWith("bf16_to_uint_") ||
    wgslCallee.startsWith("bf16_to_short_") ||
    wgslCallee.startsWith("bf16_to_ushort_") ||
    wgslCallee === "bf16_to_char_rz" ||
    wgslCallee === "bf16_to_uchar_rz"
  ) {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = `f32(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)})`;
    const rounded = wgslCallee.endsWith("_rn")
      ? emitRoundEvenWgsl(emitted)
      : wgslCallee.endsWith("_rz")
      ? `trunc(${emitted})`
      : wgslCallee.endsWith("_ru")
      ? `ceil(${emitted})`
      : `floor(${emitted})`;
    if (wgslCallee.startsWith("bf16_to_uint_")) return `u32(max(${rounded}, 0.0))`;
    if (wgslCallee.startsWith("bf16_to_ushort_")) return `(u32(max(${rounded}, 0.0)) & 0xffffu)`;
    if (wgslCallee === "bf16_to_uchar_rz") return `(u32(max(${rounded}, 0.0)) & 0xffu)`;
    if (wgslCallee.startsWith("bf16_to_short_")) return `((bitcast<i32>((u32(i32(${rounded})) & 0xffffu) << 16u)) >> 16)`;
    if (wgslCallee === "bf16_to_char_rz") return `((bitcast<i32>((u32(i32(${rounded})) & 0xffu) << 24u)) >> 24)`;
    return `i32(${rounded})`;
  }
  if (wgslCallee === "half_abs" || wgslCallee === "half_ceil" || wgslCallee === "half_floor" || wgslCallee === "half_rcp" || wgslCallee === "half_rsqrt" || wgslCallee === "half_sqrt" || wgslCallee === "half_trunc" || wgslCallee === "half_neg" || wgslCallee === "half_exp") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    if (expression.valueType === "bf16") {
      const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
      if (wgslCallee === "half_abs") return wgslRoundBfloat16(`abs(${emitted})`);
      if (wgslCallee === "half_ceil") return wgslRoundBfloat16(`ceil(${emitted})`);
      if (wgslCallee === "half_floor") return wgslRoundBfloat16(`floor(${emitted})`);
      if (wgslCallee === "half_rcp") return wgslRoundBfloat16(`(1.0 / ${emitted})`);
      if (wgslCallee === "half_rsqrt") return wgslRoundBfloat16(`inverseSqrt(${emitted})`);
      if (wgslCallee === "half_sqrt") return wgslRoundBfloat16(`sqrt(${emitted})`);
      if (wgslCallee === "half_trunc") return wgslRoundBfloat16(`trunc(${emitted})`);
      if (wgslCallee === "half_exp") return wgslRoundBfloat16(`exp(${emitted})`);
      if (wgslCallee === "half_neg") return wgslRoundBfloat16(`(-${emitted})`);
    }
    const emitted = emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations);
    if (wgslCallee === "half_abs") return `abs(${emitted})`;
    if (wgslCallee === "half_ceil") return `f16(ceil(f32(${emitted})))`;
    if (wgslCallee === "half_floor") return `f16(floor(f32(${emitted})))`;
    if (wgslCallee === "half_rcp") return `f16(1.0 / f32(${emitted}))`;
    if (wgslCallee === "half_rsqrt") return `f16(inverseSqrt(f32(${emitted})))`;
    if (wgslCallee === "half_sqrt") return `f16(sqrt(f32(${emitted})))`;
    if (wgslCallee === "half_trunc") return `f16(trunc(f32(${emitted})))`;
    if (wgslCallee === "half_exp") return `f16(exp(f32(${emitted})))`;
    return `(-${emitted})`;
  }
  if (wgslCallee === "half_fma" || wgslCallee === "half_fma_sat" || wgslCallee === "half_fma_relu") {
    const [first, second, third] = expression.args;
    if (!first || !second || !third) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    if (expression.valueType === "bf16") {
      const value = `fma(${emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)}, ${emitSemanticExpressionAs(second, ir, names, "f32", options, textureSpecializations)}, ${emitSemanticExpressionAs(third, ir, names, "f32", options, textureSpecializations)})`;
      if (wgslCallee === "half_fma_sat") return wgslSaturateBfloat16(value);
      if (wgslCallee === "half_fma_relu") return wgslReluBfloat16(value);
      return wgslRoundBfloat16(value);
    }
    const value = `fma(${emitSemanticExpressionAs(first, ir, names, "f16", options, textureSpecializations)}, ${emitSemanticExpressionAs(second, ir, names, "f16", options, textureSpecializations)}, ${emitSemanticExpressionAs(third, ir, names, "f16", options, textureSpecializations)})`;
    if (wgslCallee === "half_fma_sat") return wgslSaturateHalf(value);
    if (wgslCallee === "half_fma_relu") return `max(${value}, f16(0.0))`;
    return value;
  }
  if (wgslCallee === "half_isnan" || wgslCallee === "half_isinf") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    if (semanticExpressionValueType(value) === "bf16") {
      const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
      if (wgslCallee === "half_isnan") return `select(0u, 1u, ${emitSemanticBf16IsNanPredicate(emitted)})`;
      return `select(0, select(-1, 1, ((bitcast<u32>(f32(${emitted})) & 0x80000000u) == 0u)), ((bitcast<u32>(f32(${emitted})) & 0x7fffffffu) == 0x7f800000u))`;
    }
    const emitted = emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations);
    if (wgslCallee === "half_isnan") return `select(0u, 1u, ${emitSemanticHalfIsNanPredicate(emitted)})`;
    return `select(0, select(-1, 1, ((bitcast<u32>(f32(${emitted})) & 0x80000000u) == 0u)), ((bitcast<u32>(f32(${emitted})) & 0x7fffffffu) == 0x7f800000u))`;
  }
  if (wgslCallee === "half_add" || wgslCallee === "half_add_sat" || wgslCallee === "half_sub" || wgslCallee === "half_sub_sat" || wgslCallee === "half_mul" || wgslCallee === "half_mul_sat" || wgslCallee === "half_div" || wgslCallee === "half_min" || wgslCallee === "half_max" || wgslCallee === "half_min_nan" || wgslCallee === "half_max_nan" || wgslCallee === "half_eq" || wgslCallee === "half_ne" || wgslCallee === "half_gt" || wgslCallee === "half_ge" || wgslCallee === "half_lt" || wgslCallee === "half_le" || wgslCallee === "half_equ" || wgslCallee === "half_neu" || wgslCallee === "half_gtu" || wgslCallee === "half_geu" || wgslCallee === "half_ltu" || wgslCallee === "half_leu") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const hasBf16Operand = expression.args.some((arg) => semanticExpressionValueType(arg) === "bf16");
    if (expression.valueType === "bf16" || hasBf16Operand) {
      const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
      const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
      if (wgslCallee === "half_add") return wgslRoundBfloat16(`(${lhs} + ${rhs})`);
      if (wgslCallee === "half_add_sat") return wgslSaturateBfloat16(`(${lhs} + ${rhs})`);
      if (wgslCallee === "half_sub") return wgslRoundBfloat16(`(${lhs} - ${rhs})`);
      if (wgslCallee === "half_sub_sat") return wgslSaturateBfloat16(`(${lhs} - ${rhs})`);
      if (wgslCallee === "half_mul") return wgslRoundBfloat16(`(${lhs} * ${rhs})`);
      if (wgslCallee === "half_mul_sat") return wgslSaturateBfloat16(`(${lhs} * ${rhs})`);
      if (wgslCallee === "half_div") return wgslRoundBfloat16(`(${lhs} / ${rhs})`);
      if (wgslCallee === "half_min") return wgslRoundBfloat16(`min(${lhs}, ${rhs})`);
      if (wgslCallee === "half_max") return wgslRoundBfloat16(`max(${lhs}, ${rhs})`);
      if (wgslCallee === "half_min_nan") return emitSemanticBf16NanMinMax("min", lhs, rhs);
      if (wgslCallee === "half_max_nan") return emitSemanticBf16NanMinMax("max", lhs, rhs);
      const operator =
        wgslCallee === "half_eq" || wgslCallee === "half_equ" ? "==" :
        wgslCallee === "half_ne" || wgslCallee === "half_neu" ? "!=" :
        wgslCallee === "half_gt" || wgslCallee === "half_gtu" ? ">" :
        wgslCallee === "half_ge" || wgslCallee === "half_geu" ? ">=" :
        wgslCallee === "half_lt" || wgslCallee === "half_ltu" ? "<" :
        "<=";
      const comparison = `(${lhs} ${operator} ${rhs})`;
      const predicate = wgslCallee.endsWith("u")
        ? `(${emitSemanticBf16IsNanPredicate(lhs)} || ${emitSemanticBf16IsNanPredicate(rhs)} || ${comparison})`
        : comparison;
      return `select(0u, 1u, ${predicate})`;
    }
    const lhs = emitSemanticExpressionAs(left, ir, names, "f16", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f16", options, textureSpecializations);
    if (wgslCallee === "half_add") return `(${lhs} + ${rhs})`;
    if (wgslCallee === "half_add_sat") return wgslSaturateHalf(`(${lhs} + ${rhs})`);
    if (wgslCallee === "half_sub") return `(${lhs} - ${rhs})`;
    if (wgslCallee === "half_sub_sat") return wgslSaturateHalf(`(${lhs} - ${rhs})`);
    if (wgslCallee === "half_mul") return `(${lhs} * ${rhs})`;
    if (wgslCallee === "half_mul_sat") return wgslSaturateHalf(`(${lhs} * ${rhs})`);
    if (wgslCallee === "half_div") return `(${lhs} / ${rhs})`;
    if (wgslCallee === "half_min") return `min(${lhs}, ${rhs})`;
    if (wgslCallee === "half_max") return `max(${lhs}, ${rhs})`;
    if (wgslCallee === "half_min_nan") return emitSemanticHalfNanMinMax("min", lhs, rhs);
    if (wgslCallee === "half_max_nan") return emitSemanticHalfNanMinMax("max", lhs, rhs);
    const operator =
      wgslCallee === "half_eq" || wgslCallee === "half_equ" ? "==" :
      wgslCallee === "half_ne" || wgslCallee === "half_neu" ? "!=" :
      wgslCallee === "half_gt" || wgslCallee === "half_gtu" ? ">" :
      wgslCallee === "half_ge" || wgslCallee === "half_geu" ? ">=" :
      wgslCallee === "half_lt" || wgslCallee === "half_ltu" ? "<" :
      "<=";
    const comparison = `(${lhs} ${operator} ${rhs})`;
    const predicate = wgslCallee.endsWith("u")
      ? `(${emitSemanticHalfIsNanPredicate(lhs)} || ${emitSemanticHalfIsNanPredicate(rhs)} || ${comparison})`
      : comparison;
    return `select(0u, 1u, ${predicate})`;
  }
  if (wgslCallee === "clz" || wgslCallee === "clzll" || wgslCallee === "ffs" || wgslCallee === "popc" || wgslCallee === "brev") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations);
    if (wgslCallee === "clz") return `i32(countLeadingZeros(${emitted}))`;
    if (wgslCallee === "clzll") return `(i32(countLeadingZeros(${emitted})) + 32)`;
    if (wgslCallee === "ffs") return `select(0, (i32(countTrailingZeros(${emitted})) + 1), (${emitted} != 0u))`;
    if (wgslCallee === "popc") return `i32(countOneBits(${emitted}))`;
    return `reverseBits(${emitted})`;
  }
  if (
    wgslCallee === "mul24" ||
    wgslCallee === "umul24" ||
    wgslCallee === "mulhi" ||
    wgslCallee === "umulhi" ||
    wgslCallee === "rhadd" ||
    wgslCallee === "uhadd" ||
    wgslCallee === "urhadd" ||
    wgslCallee === "hadd" ||
    wgslCallee === "umul" ||
    wgslCallee === "umin"
  ) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    if (wgslCallee === "mul24") {
      const lhs = emitSemanticExpressionAs(left, ir, names, "i32", options, textureSpecializations);
      const rhs = emitSemanticExpressionAs(right, ir, names, "i32", options, textureSpecializations);
      return `(${lhs} * ${rhs})`;
    }
    if (wgslCallee === "umul24" || wgslCallee === "umul") {
      const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
      const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
      return `(${lhs} * ${rhs})`;
    }
    if (wgslCallee === "mulhi") {
      const lhs = emitSemanticExpressionAs(left, ir, names, "i32", options, textureSpecializations);
      const rhs = emitSemanticExpressionAs(right, ir, names, "i32", options, textureSpecializations);
      return `bg_semantic_mulhi_i32(${lhs}, ${rhs})`;
    }
    if (wgslCallee === "umulhi") {
      const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
      const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
      return `bg_semantic_umulhi_u32(${lhs}, ${rhs})`;
    }
    if (wgslCallee === "umin") {
      const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
      const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
      return `min(${lhs}, ${rhs})`;
    }
    if (wgslCallee === "hadd" && expression.valueType === "half") {
      return `(${emitSemanticExpressionAs(left, ir, names, "f16", options, textureSpecializations)} + ${emitSemanticExpressionAs(right, ir, names, "f16", options, textureSpecializations)})`;
    }
    if (wgslCallee === "hadd" && expression.valueType === "bf16") {
      return wgslRoundBfloat16(`(${emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations)} + ${emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations)})`);
    }
    const scalar = wgslCallee === "uhadd" || wgslCallee === "urhadd" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, scalar, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, scalar, options, textureSpecializations);
    if (wgslCallee === "rhadd") return `((${lhs} | ${rhs}) - ((${lhs} ^ ${rhs}) >> 1u))`;
    if (wgslCallee === "hadd") return `((${lhs} & ${rhs}) + ((${lhs} ^ ${rhs}) >> 1u))`;
    if (wgslCallee === "uhadd") return `((${lhs} & ${rhs}) + ((${lhs} ^ ${rhs}) >> 1u))`;
    return `((${lhs} & ${rhs}) + ((${lhs} ^ ${rhs}) >> 1u) + ((${lhs} ^ ${rhs}) & 1u))`;
  }
  if (wgslCallee.startsWith("viadd")) {
    const [first, second, third] = expression.args;
    if (!first || !second || !third) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const choose: "max" | "min" = wgslCallee.startsWith("viaddmax") ? "max" : "min";
    const relu = wgslCallee.endsWith("_relu");
    if (wgslCallee.includes("16x2")) {
      const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
      const c = emitSemanticExpressionAs(third, ir, names, "u32", options, textureSpecializations);
      return emitSemanticViadd16x2Expression(a, b, c, wgslCallee.includes("_s16x2"), choose, relu);
    }
    const scalar = wgslCallee.includes("_s32") ? "i32" : "u32";
    const a = emitSemanticExpressionAs(first, ir, names, scalar, options, textureSpecializations);
    const b = emitSemanticExpressionAs(second, ir, names, scalar, options, textureSpecializations);
    const c = emitSemanticExpressionAs(third, ir, names, scalar, options, textureSpecializations);
    const selected = `${choose}((${a} + ${b}), ${c})`;
    return relu ? `max(${selected}, 0)` : selected;
  }
  if (wgslCallee.startsWith("vimax") || wgslCallee.startsWith("vimin") || wgslCallee.startsWith("vibmax") || wgslCallee.startsWith("vibmin")) {
    const choose: "max" | "min" = wgslCallee.includes("max") ? "max" : "min";
    const relu = wgslCallee.endsWith("_relu");
    if (wgslCallee.includes("16x2")) {
      const args = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations));
      return emitSemanticViMinMax16x2Expression(args, wgslCallee.includes("_s16x2"), choose, relu);
    }
    const scalar = wgslCallee.includes("_s32") ? "i32" : "u32";
    const args = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, scalar, options, textureSpecializations));
    const selected = args.slice(1).reduce((acc, arg) => `${choose}(${acc}, ${arg})`, args[0] ?? `${scalar}(0)`);
    return relu ? `max(${selected}, 0)` : selected;
  }
  if (wgslCallee === "vabs2" || wgslCallee === "vabsss2" || wgslCallee === "vneg2" || wgslCallee === "vnegss2" || wgslCallee === "vabs4" || wgslCallee === "vabsss4" || wgslCallee === "vneg4" || wgslCallee === "vnegss4") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations);
    const laneWidth = wgslCallee.endsWith("2") ? 16 : 8;
    const op =
      wgslCallee.startsWith("vabsss") ? "sat_abs" :
      wgslCallee.startsWith("vabs") ? "abs" :
      wgslCallee.startsWith("vnegss") ? "sat_neg" :
      "neg";
    return emitSemanticVPackedUnaryExpression(emitted, laneWidth, op);
  }
  if (wgslCallee === "vabsdiffs2" || wgslCallee === "vabsdiffs4" || wgslCallee === "vsads2" || wgslCallee === "vsadu2" || wgslCallee === "vsads4" || wgslCallee === "vsadu4") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
    const laneWidth = wgslCallee.endsWith("2") ? 16 : 8;
    if (wgslCallee.startsWith("vabsdiffs")) return emitSemanticVPackedAbsDiffExpression(lhs, rhs, laneWidth);
    return emitSemanticVPackedSadExpression(lhs, rhs, laneWidth, wgslCallee.startsWith("vsads"));
  }
  if (wgslCallee === "vhaddu2" || wgslCallee === "vhaddu4" || wgslCallee === "vavgs2" || wgslCallee === "vavgs4") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
    return emitSemanticVPackedAverageExpression(lhs, rhs, wgslCallee.endsWith("2") ? 16 : 8, wgslCallee.startsWith("vavgs"));
  }
  if (wgslCallee === "vadd2" || wgslCallee === "vsub2" || wgslCallee === "vaddss2" || wgslCallee === "vsubss2" || wgslCallee === "vaddus2" || wgslCallee === "vsubus2" || wgslCallee === "vabsdiffu2" || wgslCallee === "vavgu2" || wgslCallee === "vminu2" || wgslCallee === "vmaxu2" || wgslCallee === "vmins2" || wgslCallee === "vmaxs2" || wgslCallee === "vadd4" || wgslCallee === "vsub4" || wgslCallee === "vaddss4" || wgslCallee === "vsubss4" || wgslCallee === "vaddus4" || wgslCallee === "vsubus4" || wgslCallee === "vabsdiffu4" || wgslCallee === "vavgu4" || wgslCallee === "vminu4" || wgslCallee === "vmaxu4" || wgslCallee === "vmins4" || wgslCallee === "vmaxs4") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
    return `bg_semantic_${wgslCallee}_u32(${lhs}, ${rhs})`;
  }
  if (wgslCallee.startsWith("vset")) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
    const laneWidth = wgslCallee.endsWith("2") ? 16 : 8;
    const opName = wgslCallee.slice(4, -1);
    const signed = opName.endsWith("s");
    const operator =
      opName === "eq" ? "==" :
      opName === "ne" ? "!=" :
      opName.startsWith("ge") ? ">=" :
      opName.startsWith("gt") ? ">" :
      opName.startsWith("le") ? "<=" :
      "<";
    return emitSemanticVSetExpression(lhs, rhs, laneWidth, signed, operator);
  }
  if (wgslCallee.startsWith("vcmp")) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations);
    const laneWidth = wgslCallee.endsWith("2") ? 16 : 8;
    const opName = wgslCallee.slice(4, -1);
    const signed = opName.endsWith("s");
    const operator =
      opName === "eq" ? "==" :
      opName === "ne" ? "!=" :
      opName.startsWith("ge") ? ">=" :
      opName.startsWith("gt") ? ">" :
      opName.startsWith("le") ? "<=" :
      "<";
    return emitSemanticVCompareExpression(lhs, rhs, laneWidth, signed, operator);
  }
  if (wgslCallee === "imad" || wgslCallee === "umad" || wgslCallee === "sad" || wgslCallee === "usad" || wgslCallee === "usad4" || wgslCallee === "dp4a" || wgslCallee === "dp2a_lo" || wgslCallee === "dp2a_hi" || wgslCallee === "byte_perm" || wgslCallee.startsWith("funnelshift_")) {
    const [first, second, third] = expression.args;
    if (!first || !second || (!third && wgslCallee !== "usad4")) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    if (wgslCallee === "imad") {
      const a = emitSemanticExpressionAs(first, ir, names, "i32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "i32", options, textureSpecializations);
      const c = emitSemanticExpressionAs(third!, ir, names, "i32", options, textureSpecializations);
      return `((${a} * ${b}) + ${c})`;
    }
    if (wgslCallee === "umad") {
      const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
      const c = emitSemanticExpressionAs(third!, ir, names, "u32", options, textureSpecializations);
      return `((${a} * ${b}) + ${c})`;
    }
    if (wgslCallee === "sad") {
      const a = emitSemanticExpressionAs(first, ir, names, "i32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "i32", options, textureSpecializations);
      const c = emitSemanticExpressionAs(third!, ir, names, "u32", options, textureSpecializations);
      return `(select((u32(${b}) - u32(${a})), (u32(${a}) - u32(${b})), (${a} >= ${b})) + ${c})`;
    }
    if (wgslCallee === "usad") {
      const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
      const c = emitSemanticExpressionAs(third!, ir, names, "u32", options, textureSpecializations);
      return `(max(${a}, ${b}) - min(${a}, ${b}) + ${c})`;
    }
    if (wgslCallee === "usad4") {
      const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
      const c = third ? emitSemanticExpressionAs(third, ir, names, "u32", options, textureSpecializations) : "0u";
      return `bg_semantic_usad4_u32(${a}, ${b}, ${c})`;
    }
    if (wgslCallee === "dp4a") {
      const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
      if (expression.valueType === "uint") {
        const c = emitSemanticExpressionAs(third!, ir, names, "u32", options, textureSpecializations);
        return `bg_semantic_dp4a_u32(${a}, ${b}, ${c})`;
      }
      const c = emitSemanticExpressionAs(third!, ir, names, "i32", options, textureSpecializations);
      return `bg_semantic_dp4a_i32(${a}, ${b}, ${c})`;
    }
    if (wgslCallee === "dp2a_lo" || wgslCallee === "dp2a_hi") {
      const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
      const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
      const byteShift = wgslCallee === "dp2a_hi" ? "16u" : "0u";
      if (expression.valueType === "uint") {
        const c = emitSemanticExpressionAs(third!, ir, names, "u32", options, textureSpecializations);
        return `bg_semantic_dp2a_u32(${a}, ${b}, ${c}, ${byteShift})`;
      }
      const c = emitSemanticExpressionAs(third!, ir, names, "i32", options, textureSpecializations);
      return `bg_semantic_dp2a_i32(${a}, ${b}, ${c}, ${byteShift})`;
    }
    const a = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
    const b = emitSemanticExpressionAs(second, ir, names, "u32", options, textureSpecializations);
    const c = emitSemanticExpressionAs(third!, ir, names, "u32", options, textureSpecializations);
    if (wgslCallee === "byte_perm") return `bg_semantic_byte_perm_u32(${a}, ${b}, ${c})`;
    return `bg_semantic_${wgslCallee}_u32(${a}, ${b}, ${c})`;
  }
  if (wgslCallee === "add" || wgslCallee === "sub" || wgslCallee === "mul") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const operator = wgslCallee === "add" ? "+" : wgslCallee === "sub" ? "-" : "*";
    return `(${emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations)} ${operator} ${emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations)})`;
  }
  if (wgslCallee === "divide") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return `(${emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations)} / ${emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations)})`;
  }
  if (wgslCallee === "ldexp") {
    const [value, exponent] = expression.args;
    if (!value || !exponent) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    const scale = emitSemanticExpressionAs(exponent, ir, names, "i32", options, textureSpecializations);
    return `(${emitted} * exp2(f32(${scale})))`;
  }
  if (wgslCallee === "fmod" || wgslCallee === "remainder" || wgslCallee === "fdim" || wgslCallee === "nextafter") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    if (wgslCallee === "fmod") return `(${lhs} - trunc(${lhs} / ${rhs}) * ${rhs})`;
    if (wgslCallee === "remainder") return `bg_semantic_remainder_f32(${lhs}, ${rhs})`;
    if (wgslCallee === "nextafter") return `bg_semantic_nextafter_f32(${lhs}, ${rhs})`;
    return `max((${lhs} - ${rhs}), 0.0)`;
  }
  if (wgslCallee === "hypot" || wgslCallee === "rhypot" || wgslCallee === "norm" || wgslCallee === "rnorm") {
    const emitted = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations));
    if (emitted.length < 2) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    const sum = emitted.map((arg) => `(${arg} * ${arg})`).join(" + ");
    const norm = `sqrt(${sum})`;
    return wgslCallee === "rhypot" || wgslCallee === "rnorm" ? `(1.0 / ${norm})` : norm;
  }
  if (
    wgslCallee === "exp10" ||
    wgslCallee === "expm1" ||
    wgslCallee === "erf" ||
    wgslCallee === "erfc" ||
    wgslCallee === "erfcx" ||
    wgslCallee === "erfinv" ||
    wgslCallee === "erfcinv" ||
    wgslCallee === "normcdf" ||
    wgslCallee === "normcdfinv" ||
    wgslCallee === "tgamma" ||
    wgslCallee === "lgamma" ||
    wgslCallee === "log10" ||
    wgslCallee === "log1p" ||
    wgslCallee === "sinpi" ||
    wgslCallee === "cospi" ||
    wgslCallee === "round_away" ||
    wgslCallee === "round_even" ||
    wgslCallee === "logb" ||
    wgslCallee === "ilogb" ||
    wgslCallee === "sinh" ||
    wgslCallee === "cosh" ||
    wgslCallee === "asinh" ||
    wgslCallee === "acosh" ||
    wgslCallee === "atanh" ||
    wgslCallee === "cbrt" ||
    wgslCallee === "rcbrt" ||
    wgslCallee === "reciprocal" ||
    wgslCallee.startsWith("float_to_")
  ) {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    if (wgslCallee === "exp10") return `pow(10.0, ${emitted})`;
    if (wgslCallee === "expm1") return `(exp(${emitted}) - 1.0)`;
    if (wgslCallee === "erf") return `bg_semantic_erf_f32(${emitted})`;
    if (wgslCallee === "erfc") return `(1.0 - bg_semantic_erf_f32(${emitted}))`;
    if (wgslCallee === "erfcx") return `(exp(${emitted} * ${emitted}) * (1.0 - bg_semantic_erf_f32(${emitted})))`;
    if (wgslCallee === "erfinv") return `bg_semantic_erfinv_f32(${emitted})`;
    if (wgslCallee === "erfcinv") return `bg_semantic_erfinv_f32(1.0 - ${emitted})`;
    if (wgslCallee === "normcdf") return `(0.5 * (1.0 + bg_semantic_erf_f32((${emitted} * 0.7071067811865476))))`;
    if (wgslCallee === "normcdfinv") return `bg_semantic_normcdfinv_f32(${emitted})`;
    if (wgslCallee === "tgamma") return `bg_semantic_tgamma_f32(${emitted})`;
    if (wgslCallee === "lgamma") return `bg_semantic_lgamma_f32(${emitted})`;
    if (wgslCallee === "log10") return `(log(${emitted}) / 2.302585092994046)`;
    if (wgslCallee === "log1p") return `log(1.0 + ${emitted})`;
    if (wgslCallee === "sinpi") return `sin(3.141592653589793 * ${emitted})`;
    if (wgslCallee === "cospi") return `cos(3.141592653589793 * ${emitted})`;
    if (wgslCallee === "round_away") return `select(floor(abs(${emitted}) + 0.5), -floor(abs(${emitted}) + 0.5), (${emitted} < 0.0))`;
    if (wgslCallee === "round_even") return emitRoundEvenWgsl(emitted);
    if (wgslCallee === "logb") return `bg_semantic_logb_f32(${emitted})`;
    if (wgslCallee === "ilogb") return `bg_semantic_ilogb_i32(${emitted})`;
    if (wgslCallee === "sinh") return `(0.5 * (exp(${emitted}) - exp(-${emitted})))`;
    if (wgslCallee === "cosh") return `(0.5 * (exp(${emitted}) + exp(-${emitted})))`;
    if (wgslCallee === "asinh") return `log(${emitted} + sqrt((${emitted} * ${emitted}) + 1.0))`;
    if (wgslCallee === "acosh") return `log(${emitted} + sqrt((${emitted} * ${emitted}) - 1.0))`;
    if (wgslCallee === "atanh") return `(0.5 * log((1.0 + ${emitted}) / (1.0 - ${emitted})))`;
    const signedCbrt = `select(pow(abs(${emitted}), 0.3333333333333333), -pow(abs(${emitted}), 0.3333333333333333), (${emitted} < 0.0))`;
    if (wgslCallee === "cbrt") return signedCbrt;
    if (wgslCallee === "rcbrt") return `(1.0 / ${signedCbrt})`;
    if (wgslCallee === "float_to_int_rn") return `i32(bg_semantic_round_even_f32(${emitted}))`;
    if (wgslCallee === "float_to_int_round") return `i32(select(floor(abs(${emitted}) + 0.5), -floor(abs(${emitted}) + 0.5), (${emitted} < 0.0)))`;
    if (wgslCallee === "float_to_int_rz") return `i32(trunc(${emitted}))`;
    if (wgslCallee === "float_to_int_ru") return `i32(ceil(${emitted}))`;
    if (wgslCallee === "float_to_int_rd") return `i32(floor(${emitted}))`;
    if (wgslCallee === "float_to_uint_rn") return `u32(max(bg_semantic_round_even_f32(${emitted}), 0.0))`;
    if (wgslCallee === "float_to_uint_rz") return `u32(max(trunc(${emitted}), 0.0))`;
    if (wgslCallee === "float_to_uint_ru") return `u32(max(ceil(${emitted}), 0.0))`;
    if (wgslCallee === "float_to_uint_rd") return `u32(max(floor(${emitted}), 0.0))`;
    return `(1.0 / ${emitted})`;
  }
  if (wgslCallee === "int_to_float" || wgslCallee === "uint_to_float") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const scalar = wgslCallee === "int_to_float" ? "i32" : "u32";
    return `f32(${emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations)})`;
  }
  if (wgslCallee === "builtin_inf") return "bitcast<f32>(0x7f800000u)";
  if (wgslCallee === "uint_as_float" || wgslCallee === "int_as_float") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const scalar = wgslCallee === "uint_as_float" ? "u32" : "i32";
    return `bitcast<f32>(${emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations)})`;
  }
  if (wgslCallee === "saturate") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError("__saturatef expects one operand", expression.span);
    return `clamp(${emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations)}, 0.0, 1.0)`;
  }
  if (wgslCallee === "copysign") {
    const [magnitude, sign] = expression.args;
    if (!magnitude || !sign) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(magnitude, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(sign, ir, names, "f32", options, textureSpecializations);
    return `select(abs(${lhs}), -abs(${lhs}), ((bitcast<u32>(${rhs}) & 0x80000000u) != 0u))`;
  }
  if (wgslCallee === "isnan" || wgslCallee === "isinf" || wgslCallee === "isfinite" || wgslCallee === "signbit" || wgslCallee === "isnormal") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    const absValue = `abs(${emitted})`;
    const condition =
      wgslCallee === "isnan" ? `((${emitted}) != (${emitted}))` :
      wgslCallee === "isinf" ? `(${absValue} > 3.4028234663852886e38)` :
      wgslCallee === "isfinite" ? `((${absValue} <= 3.4028234663852886e38) && ((${emitted}) == (${emitted})))` :
      wgslCallee === "signbit" ? `((bitcast<u32>(${emitted}) & 0x80000000u) != 0u)` :
      `((${absValue} >= 1.1754943508222875e-38) && (${absValue} <= 3.4028234663852886e38))`;
    return `select(0u, 1u, ${condition})`;
  }
  if (wgslCallee === "isgreater" || wgslCallee === "isgreaterequal" || wgslCallee === "isless" || wgslCallee === "islessequal" || wgslCallee === "islessgreater" || wgslCallee === "isunordered") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const unordered = `((${lhs}) != (${lhs}) || (${rhs}) != (${rhs}))`;
    const comparison =
      wgslCallee === "isgreater" ? `((${lhs}) > (${rhs}))` :
      wgslCallee === "isgreaterequal" ? `((${lhs}) >= (${rhs}))` :
      wgslCallee === "isless" ? `((${lhs}) < (${rhs}))` :
      wgslCallee === "islessequal" ? `((${lhs}) <= (${rhs}))` :
      wgslCallee === "islessgreater" ? `(((${lhs}) < (${rhs})) || ((${lhs}) > (${rhs})))` :
      "";
    const condition = wgslCallee === "isunordered" ? unordered : `(!${unordered} && ${comparison})`;
    return `select(0u, 1u, ${condition})`;
  }
  if (wgslCallee === "lerp") {
    const [left, right, factor] = expression.args;
    if (!left || !right || !factor) throw semanticWgslError("lerp expects three operands", expression.span);
    const start = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const end = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const amount = emitSemanticExpressionAs(factor, ir, names, "f32", options, textureSpecializations);
    return `fma(${amount}, (${end} - ${start}), ${start})`;
  }
  if (wgslCallee === "modf_intpart" || wgslCallee === "modf_fraction") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    const nonFinite = `((${emitted} != ${emitted}) || (abs(${emitted}) > 3.4028234663852886e38))`;
    if (wgslCallee === "modf_intpart") return `select(trunc(${emitted}), ${emitted}, ${nonFinite})`;
    const infinityFraction = `select(0.0, -0.0, ${emitted} < 0.0)`;
    return `select(select((${emitted} - trunc(${emitted})), ${infinityFraction}, abs(${emitted}) > 3.4028234663852886e38), ${emitted}, ${emitted} != ${emitted})`;
  }
  if (wgslCallee === "frexp_exponent" || wgslCallee === "frexp_mantissa") {
    const [value] = expression.args;
    if (!value) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const emitted = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    const nonFiniteOrZero = `((${emitted} == 0.0) || (${emitted} != ${emitted}) || (abs(${emitted}) > 3.4028234663852886e38))`;
    const exponent = `(i32(floor(log2(abs(${emitted})))) + 1)`;
    if (wgslCallee === "frexp_exponent") return `select(${exponent}, 0, ${nonFiniteOrZero})`;
    return `select((${emitted} / exp2(f32(${exponent}))), ${emitted}, ${nonFiniteOrZero})`;
  }
  if (wgslCallee === "remquo_quotient" || wgslCallee === "remquo_remainder") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const x = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const y = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const ratio = `(${x} / ${y})`;
    const base = `floor(${ratio})`;
    const diff = `(${ratio} - ${base})`;
    const quotient = `select(select(i32(${base}), i32(${base}) + 1, ${diff} > 0.5), select(i32(${base}), i32(${base}) + 1, (i32(${base}) % 2) != 0), ${diff} == 0.5)`;
    if (wgslCallee === "remquo_quotient") return quotient;
    return `(${x} - f32(${quotient}) * ${y})`;
  }
  if (wgslCallee === "i16_lane" || wgslCallee === "u16_lane") {
    const [value, shift] = expression.args;
    if (!value || !shift) throw semanticWgslError(`${expression.callee.name} expects value and shift`, expression.span);
    const bits = `((u32(${emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations)}) >> u32(${emitSemanticExpressionAs(shift, ir, names, "i32", options, textureSpecializations)})) & 0xffffu)`;
    if (wgslCallee === "u16_lane") return bits;
    return `(i32(${bits}) - select(0, 65536, ${bits} >= 0x8000u))`;
  }
  return `${wgslCallee}(${expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations)).join(", ")})`;
}

function emitSemanticMember(
  expression: Extract<SemanticExpression, { readonly kind: "member" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  if (expression.object.kind === "symbol") {
    switch (expression.object.name) {
      case "threadIdx":
        return ir.workgroupSize[axisIndex] === 1 ? "0u" : `local_id.${expression.property}`;
      case "blockIdx":
        return `workgroup_id.${expression.property}`;
      case "blockDim":
        return `${ir.workgroupSize[axisIndex]}u`;
      case "gridDim":
        return `num_workgroups.${expression.property}`;
    }
  }
  if (semanticStorageVectorType(semanticExpressionVectorValueType(expression.object, ir?.functions)) === undefined) {
    throw semanticWgslError("semantic WGSL supports builtin vector members only", expression.span);
  }
  return `${emitSemanticExpression(expression.object, ir, names, options)}.${semanticVectorFieldName(expression)}`;
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
): string {
  if (expression.operator === "!") return `!(${emitTruthiness(expression.argument, ir, names, options)})`;
  if (expression.operator === "~") {
    const operandType = semanticExpressionWgslScalar(expression) === "u32" ? "u32" : "i32";
    return `~(${emitSemanticExpressionAs(expression.argument, ir, names, operandType, options, textureSpecializations)})`;
  }
  if (expression.operator === "+") return emitSemanticExpression(expression.argument, ir, names, options, textureSpecializations);
  if (expression.operator === "-") return `-(${emitSemanticExpression(expression.argument, ir, names, options, textureSpecializations)})`;
  throw semanticWgslError(`semantic WGSL does not support unary '${expression.operator}'`, expression.span);
}

function emitSemanticBinary(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (isSemanticStoragePointerNullComparison(expression)) return expression.operator === "!=" ? "true" : "false";
  if (isSemanticStoragePointerIdentityComparison(expression, ir)) {
    const left = emitSemanticStoragePointerIdentity(expression.left, ir, names, options);
    const right = emitSemanticStoragePointerIdentity(expression.right, ir, names, options);
    const equal = `((${left.buffer}) == (${right.buffer}) && (${left.base}) == (${right.base}))`;
    return expression.operator === "==" ? equal : `!${equal}`;
  }
  if (LOGICAL_OPERATORS.has(expression.operator)) {
    return `(${emitTruthiness(expression.left, ir, names, options)} ${expression.operator} ${emitTruthiness(expression.right, ir, names, options)})`;
  }
  if (isSemanticFloatVectorType(expression.valueType) && semanticWgslVectorBinaryOperatorSupported(expression.operator)) {
    const valueType = expression.valueType as CudaLiteScalarType;
    return `(${emitSemanticVectorOperand(expression.left, valueType, ir, names, options, textureSpecializations)} ${expression.operator} ${emitSemanticVectorOperand(expression.right, valueType, ir, names, options, textureSpecializations)})`;
  }
  if (expression.operator === "<<" || expression.operator === ">>") {
    const leftType = semanticExpressionWgslScalar(expression.left) === "u32" ? "u32" : "i32";
    const left = emitSemanticExpressionAs(expression.left, ir, names, leftType, options, textureSpecializations);
    const right = emitSemanticExpressionAs(expression.right, ir, names, "u32", options, textureSpecializations);
    return `(${left} ${expression.operator} ${right})`;
  }
  const operandType = semanticBinaryOperandType(expression);
  const left = emitSemanticExpressionAs(expression.left, ir, names, operandType, options, textureSpecializations);
  const right = emitSemanticExpressionAs(expression.right, ir, names, operandType, options, textureSpecializations);
  return `(${left} ${expression.operator} ${right})`;
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
): { readonly buffer: string; readonly base: string } {
  if (expression.kind !== "symbol") throw semanticWgslError("semantic storage pointer identity requires symbols", expression.span);
  const ownerParam = options.activeFunction === undefined
    ? undefined
    : ir.functions.find((fn) => fn.name === options.activeFunction)?.params.find((param) =>
      param.name === expression.name && param.pointer && param.addressSpace === "storage");
  if (ownerParam) {
    return {
      buffer: nameFor(semanticPointerBufferParamName(expression.name), names),
      base: nameFor(semanticPointerBaseParamName(expression.name), names),
    };
  }
  const bufferId = semanticStoragePointerBufferId(expression.name, ir);
  const root = ir.params.find((param) => param.name === expression.name && param.pointer && param.addressSpace === "storage");
  if (bufferId === undefined || root?.valueType === undefined) throw semanticWgslError(`unknown storage pointer '${expression.name}'`, expression.span);
  return {
    buffer: `${bufferId}u`,
    base: emitSemanticRootStoragePointerArgBaseIndex({
      base: expression.name,
      addressSpace: "storage",
      valueType: root.valueType,
      indices: [],
      fields: [],
      span: expression.span,
    }, root, ir, names, options),
  };
}

function emitSemanticVectorOperand(
  expression: SemanticExpression,
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (isSemanticFloatVectorType(semanticExpressionVectorValueType(expression, ir?.functions))) {
    return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  }
  const laneCount = cudaVectorLaneCount(valueType);
  const vectorScalar = wgslVectorScalar(valueType);
  const scalar = emitSemanticExpressionAs(expression, ir, names, vectorScalar, options, textureSpecializations);
  return `vec${laneCount}<${vectorScalar}>(${Array.from({ length: laneCount }, () => `${vectorScalar}(${scalar})`).join(", ")})`;
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
    return emitSemanticExpression(expression, ir, names, options);
  }
  if (expression.kind === "binary" && (COMPARISON_OPERATORS.has(expression.operator) || LOGICAL_OPERATORS.has(expression.operator))) {
    return emitSemanticBinary(expression, ir, names, options);
  }
  const scalar = semanticExpressionWgslScalar(expression);
  const zero = scalar === "u32" ? "0u" : scalar === "f32" ? "0.0" : "0";
  return `(${emitSemanticExpression(expression, ir, names, options)} != ${zero})`;
}

function semanticWgslConditionSupported(expression: SemanticExpression, ir?: SemanticKernelIrModule): boolean {
  return expression.kind === "symbol" && expression.addressSpace === "storage" ||
    semanticWgslExpressionSupported(expression, "scalar", ir);
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
    return `${nameFor(ref.base, names)}[${emitFlatConstantIndex(symbol, ref.indices, ir, names, ref.span)}]`;
  }
  if (ref.addressSpace === "device-global") {
    const symbol = deviceGlobalMemorySymbols(ir).find((item) => item.name === ref.base);
    if (!symbol) throw semanticWgslError(`unknown device-global memory '${ref.base}'`, ref.span);
    return `${nameFor(ref.base, names)}[${emitFlatDeviceGlobalIndex(symbol, ref.indices, ir, names, ref.span)}]`;
  }
  if (ref.addressSpace === "local") {
    if (semanticWgslFunctionLocalPointerParam(ir, ref.base)) {
      if (ref.indices.length > 1 || ref.indices[0] && !semanticExpressionIsZero(ref.indices[0])) {
        throw semanticWgslError(`local scalar pointer '${ref.base}' cannot be indexed`, ref.span);
      }
      return `*${nameFor(ref.base, names)}`;
    }
    const local = localArraySymbol(ir, ref.base);
    if (!local && ref.indices.length === 0) return nameFor(ref.base, names);
    if (!local) throw semanticWgslError(`unknown local memory '${ref.base}'`, ref.span);
    if (ref.indices.length === 1 && local.dimensions.length > 1) {
      const flat = emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32");
      return `${nameFor(ref.base, names)}${emitFlatLocalArrayIndexes(flat, local.dimensions)}`;
    }
    if (ref.indices.length !== local.dimensions.length) throw semanticWgslError(`local memory '${ref.base}' index rank mismatch`, ref.span);
    return `${nameFor(ref.base, names)}${ref.indices.map((index) => `[${emitSemanticExpressionAs(index, ir, names, "u32")}]`).join("")}`;
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
): SemanticKernelIrModule["functions"][number]["params"][number] | undefined {
  return ir.functions.flatMap((fn) => fn.params).find((param) =>
    param.name === name && param.pointer && param.addressSpace === "local" && param.dimensions.length === 0
  );
}

function emitSemanticAtomicLoad(ref: SemanticMemoryRef, memoryRef: string): string {
  const loaded = `atomicLoad(&${memoryRef})`;
  return ref.valueType === "float" || ref.valueType === "bf16" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitSemanticVectorMemoryRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const valueType = semanticStorageVectorType(ref.valueType);
  if (!valueType) throw semanticWgslError("semantic WGSL vector read requires vector memory type", ref.span);
  if (semanticWgslLocalScalarVectorView(ref, ir)) {
    const scalar = cudaVectorScalarType(valueType);
    if (scalar === undefined) throw semanticWgslError("semantic WGSL local vector view requires scalar lanes", ref.span);
    return `${wgslValueType(valueType)}(${Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
      emitSemanticMemoryRef(semanticCopyMemoryRefAt({ ...ref, valueType: scalar }, lane), ir, names, options)
    ).join(", ")})`;
  }
  if (semanticWgslSharedVectorMemoryRef(ref, ir)) return emitSemanticMemoryRef(ref, ir, names, options);
  const base = emitFlatStorageVectorBaseIndex(ref, ir, names, options);
  const storage = nameFor(ref.base, names);
  const laneCount = cudaVectorLaneCount(valueType);
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations).has(ref.base) ||
    semanticAtomicSharedNames(ir.operations, ir.functions).has(ref.base) ||
    semanticWgslFunctionSharedPointerAtomicParam(ir, ref.base);
  return `${wgslValueType(valueType)}(${Array.from({ length: laneCount }, (_, lane) => {
    const access = `${storage}[(${base} + ${lane}u)]`;
    return atomicStorage ? `bitcast<f32>(atomicLoad(&${access}))` : access;
  }).join(", ")})`;
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
    return {
      base: expression.name,
      addressSpace: expression.addressSpace,
      ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
      indices: [],
      fields: [],
      span: expression.span,
    };
  }
  if (expression.kind !== "index") return undefined;
  const flattened = flattenMemoryRef(expression);
  if (!flattened || (flattened.base.addressSpace !== "storage" && flattened.base.addressSpace !== "shared" && flattened.base.addressSpace !== "constant" && flattened.base.addressSpace !== "device-global" && flattened.base.addressSpace !== "local")) return undefined;
  return {
    base: flattened.base.name,
    addressSpace: flattened.base.addressSpace,
    ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
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
  return { base: "", addressSpace: "unknown", indices: [], fields: [], span };
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
    .map((value) => emitSemanticExpressionAs(value, ir, names, wgslValueScalar(symbol.valueType)));
  while (values.length < length) values.push(zeroForType(elementType));
  return `const ${nameFor(symbol.name, names)}: ${arrayType} = ${arrayType}(${values.join(", ")});`;
}

function initializedConstantArraySupported(symbol: SemanticKernelIrModule["memory"][number]): boolean {
  if (!symbol.init || symbol.init.kind !== "initializer") return false;
  return flattenInitializerExpressions(symbol.init)
    .slice(0, totalElements(symbol.dimensions))
    .every((value) => semanticWgslExpressionSupported(value, "scalar"));
}

function initializedVectorConstantSupported(symbol: SemanticKernelIrModule["memory"][number]): boolean {
  if (!symbol.init) return false;
  if (symbol.init.kind !== "initializer" && !semanticVectorConstantInitCallSupported(symbol.init)) return false;
  return semanticVectorConstantInitExpressions(symbol.init)
    .slice(0, cudaVectorLaneCount(symbol.valueType))
    .every((value) => semanticWgslExpressionSupported(value, "scalar"));
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
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
    const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options));
    terms.unshift(`i32(${nameFor(semanticPointerBaseParamName(ref.base), names)})`);
    return `u32(${terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`})`;
  }
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
  if (!hasOffset && ref.indices.length === 0) return "0u";
  if (!hasOffset && ref.indices.length === 1) {
    return emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  }
  const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options));
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
    return emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32", options);
  }
  const terms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "i32", options));
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
  const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base);
  if (pointerParam) {
    const indexTerms = ref.indices.map((index) => emitSemanticExpressionAs(index, ir, names, "u32", options));
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
  if (indices.length === 1) return emitSemanticExpressionAs(indices[0]!, ir, names, "u32");
  return emitSemanticFlatRankedIndex(
    "shared memory",
    symbol.name,
    symbol.dimensions,
    indices,
    symbol.span,
    (index) => emitSemanticExpressionAs(index, ir, names, "u32"),
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
    return indices[0] ? emitSemanticExpressionAs(indices[0], ir, names, "u32") : "0u";
  }
  if (indices.length === 1 && symbol.dimensions.length > 1) {
    return emitSemanticExpressionAs(indices[0]!, ir, names, "u32");
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
    (index) => emitSemanticExpressionAs(index, ir, names, "u32"),
  );
}

function emitFlatConstantIndex(
  symbol: SemanticKernelIrModule["memory"][number],
  indices: readonly SemanticExpression[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  span: SourceSpan,
): string {
  if (symbol.dimensions.length === 0) {
    if (indices.length !== 1) throw semanticWgslError(`constant memory '${symbol.name}' index rank mismatch`, span);
    return emitSemanticExpressionAs(indices[0]!, ir, names, "u32");
  }
  if (indices.length === 1 && symbol.dimensions.length > 1) {
    return emitSemanticExpressionAs(indices[0]!, ir, names, "u32");
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
    (index) => emitSemanticExpressionAs(index, ir, names, "u32"),
  );
}

function emitFlatLocalArrayIndexes(flat: string, dimensions: readonly number[]): string {
  return emitSemanticFlatLocalArrayIndexes(flat, dimensions);
}

function collectOperationNames(
  operation: SemanticKernelIrOperation,
  names: Set<string>,
): void {
  if (operation.kind === "declare") names.add(operation.target.name);
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

function semanticAtomicDeviceGlobalNames(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "device-global") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicDeviceGlobalNamesFromOperation(operation)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.consequent)) names.add(name);
      for (const name of semanticAtomicDeviceGlobalNames(operation.alternate)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicDeviceGlobalNames([operation.init])) names.add(name);
      }
      for (const name of semanticAtomicDeviceGlobalNames(operation.body)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicDeviceGlobalNames(operation.continuing)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.body)) names.add(name);
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
    const expression: Extract<SemanticExpression, { readonly kind: "call" }> = {
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "function", span: operation.span },
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
    const alias = fn.params.find((param) => param.name === name)?.pointerAliasOf;
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
    const expression: Extract<SemanticExpression, { readonly kind: "call" }> = {
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "function", span: operation.span },
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

function semanticAtomicDeviceGlobalNamesFromOperation(operation: SemanticKernelIrOperation): ReadonlySet<string> {
  const names = new Set<string>();
  for (const name of semanticAtomicNamesFromOperation(operation, "device-global")) names.add(name);
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
    return {
      base: firstArg.argument.name,
      addressSpace: firstArg.argument.addressSpace,
      ...(firstArg.argument.valueType === undefined ? {} : { valueType: firstArg.argument.valueType }),
      indices: [],
      fields: [],
      span: firstArg.argument.span,
    };
  }
  if (firstArg.kind === "index") return memoryRefFromIndexExpression(firstArg);
  if (firstArg.kind === "symbol" && (firstArg.addressSpace === "storage" || firstArg.addressSpace === "shared")) {
    return {
      base: firstArg.name,
      addressSpace: firstArg.addressSpace,
      ...(firstArg.valueType === undefined ? {} : { valueType: firstArg.valueType }),
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
