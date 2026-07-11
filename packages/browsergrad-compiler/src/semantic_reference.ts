import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
  isWgslFloat16Array,
  type WgslTexture2DInput,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import {
  bfloat16BitsToFloat32,
  roundFloat32ToBfloat16,
  roundFloat32ToBfloat16Bits,
} from "./bfloat_rounding.js";
import { isCudaBuiltinVectorSymbolName } from "./cuda_builtin_symbols.js";
import { roundFloat32ToFloat16 } from "./half_rounding.js";
import { validateCudaKernelLaunch } from "./launch.js";
import {
  cloneReferenceBuffers,
  cloneReferenceSurfaces,
} from "./reference_inputs.js";
import {
  cudaLiteFlatIndexForDimensions as flatIndexForDimensions,
  cudaLiteTotalElements as totalElements,
  cudaLiteTruthy as truthy,
} from "./cuda_lite_values.js";
import { cudaVibMinMaxInfo } from "./cuda_math_calls.js";
import { cudaAddressSpacePredicateKind } from "./cuda_pointer_calls.js";
import {
  isCudaBarrierCallName,
  isCudaCooperativeBarrierCallName,
  isCudaFenceCallName,
} from "./cuda_sync_calls.js";
import { isCudaCpAsyncFenceCall } from "./cuda_cp_async.js";
import {
  cudaArithmeticReduceOpForCall,
  cudaBitwiseReduceOpForCall,
  cudaShuffleOpForCall,
  cudaVoteOpForCall,
  isCudaLegacyShuffleCallName as legacyShuffleCall,
  isCudaLegacyVoteCallName as legacyVoteCall,
  isCudaShuffleCallName,
  isCudaWarpReduceCallName,
} from "./cuda_subgroup_calls.js";
import { referenceTypedArrayForScalar as typedArrayForScalar } from "./reference_scalars.js";
import { flattenSemanticInitializerExpressions as flattenInitializerExpressions } from "./semantic_initializers.js";
import {
  freezeReferenceTrace,
  type MutableReferenceTrace,
} from "./reference_trace.js";
import {
  referenceFloat32ToUintBits as float32ToUintBits,
  referenceUintBitsToFloat32 as uintBitsToFloat32,
} from "./reference_bitcasts.js";
import {
  referenceVectorFromTuple,
  type ReferenceVector3,
} from "./reference_vectors.js";
import { isSemanticGeneratedRandomCall } from "./semantic_generated_random_intrinsics.js";
import { deviceGlobalBufferInputs } from "./webgpu_inputs.js";
import type {
  CompiledCudaLiteKernel,
  CompiledKernelInput,
  CudaLiteDiagnostic,
  CudaLiteScalarType,
  CudaLiteTextureDescriptor,
  KernelLaunch,
  ReferenceKernelResult,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import { sizeofCudaType } from "./type_layout.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { semanticInlineAsmLdmatrixAssignments } from "./semantic_ir.js";
import { semanticOperationsReferenceRoot } from "./semantic_ir_walk.js";
import {
  SEMANTIC_BF162_BINARY_VECTOR_CALLS,
  SEMANTIC_BF162_BOOL_COMPARISON_CALLS,
  SEMANTIC_BF162_MASK_COMPARISON_CALLS,
  SEMANTIC_BF162_MINMAX_VECTOR_CALLS,
  SEMANTIC_BF162_TERNARY_VECTOR_CALLS,
  SEMANTIC_BF162_UNARY_VECTOR_CALLS,
  SEMANTIC_BF162_VECTOR_COMPARISON_CALLS,
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
  isSemanticCurandCallName,
  semanticCurandArity,
  semanticCurandScalarArgumentIndices,
  semanticCurandStateArgumentIndex,
} from "./semantic_curand_intrinsics.js";
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
  SEMANTIC_ATOMIC_OPS,
  semanticAtomicOperation,
  semanticAtomicReferenceValueTypeSupported,
  semanticAtomicScalarArgumentIndices,
  type SemanticAtomicOp,
} from "./semantic_atomic_intrinsics.js";
import {
  referenceCurandAdvance as curandAdvance,
  referenceCurandNext as curandNext,
  referenceCurandNormalPair as curandNormalPair,
  referenceCurandPoissonDraw as curandPoissonDraw,
} from "./reference_curand.js";
import {
  semanticMathCallArgumentsSupported,
  semanticVectorMinMaxCallValueType,
} from "./semantic_math_intrinsics.js";
import {
  semanticAssignmentBinaryOperator,
  semanticAssignmentOperatorSupported as semanticReferenceAssignmentOperatorSupported,
  semanticSurfaceReadValueType,
  semanticVectorAssignmentOperatorSupported,
  semanticVectorBinaryOperatorSupported as semanticReferenceVectorBinaryOperatorSupported,
} from "./semantic_expression_contracts.js";
import {
  semanticTextureReadCoordinateShapeSupported,
  semanticTextureSurfaceValueTypeSupported,
} from "./semantic_texture_surface.js";
import {
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
  semanticLocalScalarValueTypeSupported,
  semanticLocalValueTypeSupported,
  semanticScalarValueTypeSupported,
  semanticStorageVectorFieldIndices,
  semanticStorageVectorType,
  semanticValueTypeSupported,
} from "./semantic_value_types.js";
import { assertCudaTrapLaunchPreconditions } from "./trap_preconditions.js";
import { cudaVectorConstructorType, cudaVectorFieldIndex, cudaVectorLaneCount, cudaVectorScalarType, cudaVectorSwizzleIndices, isCudaVectorType } from "./vector_types.js";
import {
  semanticCooperativeGroupInfo,
  semanticCooperativeGroupRankParamName,
  semanticCooperativeGroupSizeParamName,
} from "./semantic_cooperative_groups.js";

type SemanticValue = number | ReferenceVector3 | number[];
type SemanticControl = "fallthrough" | "return" | "break" | "continue";
type SemanticBarrierScope = "subgroup" | "workgroup" | "grid";
type SemanticBarrierGenerator = Generator<SemanticBarrierScope, SemanticControl, void>;
type SemanticCurandState =
  | { readonly kind: "local"; readonly name: string; readonly span: SourceSpan }
  | { readonly kind: "memory"; readonly ref: SemanticMemoryRef };
interface SemanticReferenceContext {
  readonly compiled: CompiledCudaLiteKernel;
  readonly buffers: Map<string, WgslTypedArray>;
  readonly constants: Map<string, number | WgslTypedArray>;
  readonly deviceGlobals: Map<string, WgslTypedArray>;
  readonly textures: Readonly<Record<string, WgslTexture2DInput>>;
  readonly textureDescriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>;
  readonly surfaces: Readonly<Record<string, WgslTexture2DInput>>;
  readonly sharedMemory: Map<string, WgslTypedArray>;
  readonly sharedOffsets: Map<string, number>;
  readonly storageOffsets: Map<string, number>;
  readonly localPointerTargets: Map<string, { readonly ref: SemanticMemoryRef; readonly context: SemanticReferenceContext }>;
  readonly scalars: Readonly<Record<string, number>>;
  readonly vectors: Readonly<Record<string, WgslTypedArray>>;
  readonly locals: Map<string, SemanticValue>;
  readonly localDimensions: Map<string, readonly number[]>;
  readonly blockIdx: ReferenceVector3;
  readonly threadIdx: ReferenceVector3;
  readonly blockDim: ReferenceVector3;
  readonly gridDim: ReferenceVector3;
  readonly blockContexts: readonly SemanticReferenceContext[];
  readonly trace: MutableTrace;
  activeCollectiveContexts?: readonly SemanticReferenceContext[];
  returnValue?: SemanticValue;
}

type MutableTrace = MutableReferenceTrace;

export function canRunCompiledKernelSemanticReference(compiled: CompiledCudaLiteKernel): boolean {
  return compiled.kernelIr.params.every((param) => semanticReferenceParamSupported(param, compiled)) &&
    compiled.kernelIr.memory.every(semanticReferenceMemorySymbolSupported) &&
    semanticReferenceTextureDescriptorsSupported(compiled) &&
    semanticReferenceSharedShapeSupported(compiled) &&
    unsupportedSemanticReferenceOperation(compiled.kernelIr.operations, compiled) === undefined &&
    !semanticReferenceOperationsContainUnsupportedCallableSignatures(compiled.kernelIr.operations, compiled) &&
    !semanticReferenceOperationsContainUnsupportedCalls(compiled.kernelIr.operations, compiled) &&
    compiled.kernelIr.functions.every((fn) => !semanticReferenceOperationsContainUnsupportedCalls(fn.body, compiled));
}

export function runCompiledKernelSemanticReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): ReferenceKernelResult {
  const unsupported = unsupportedSemanticReferenceOperation(compiled.kernelIr.operations, compiled);
  if (unsupported) throw semanticReferenceError(`semantic reference does not support ${unsupported.kind}`, unsupported.span);
  if (!semanticReferenceSharedShapeSupported(compiled)) {
    throw semanticReferenceError("semantic reference does not support complex shared-memory barrier shape", compiled.kernelIr.span);
  }
  validateCudaKernelLaunch(launch, compiled.kernelIr.workgroupSize);
  validateSemanticReferenceInput(compiled, input);
  assertCudaTrapLaunchPreconditions(compiled, input.scalars ?? {});

  const buffers = cloneReferenceBuffers(input.buffers);
  const constants = semanticReferenceConstants(compiled, input);
  const deviceGlobals = cloneReferenceBuffers(deviceGlobalBufferInputs(compiled, input));
  const surfaces = cloneReferenceSurfaces(input.surfaces ?? {});
  const traces: MutableTrace[] = [];
  const blockDim = referenceVectorFromTuple(launch.blockDim);
  const gridDim = referenceVectorFromTuple(launch.gridDim);
  const scalars = input.scalars ?? {};
  const vectors = input.vectors ?? {};
  const contextBlocks: SemanticReferenceContext[][] = [];
  for (let bz = 0; bz < launch.gridDim[2]; bz++) {
    for (let by = 0; by < launch.gridDim[1]; by++) {
      for (let bx = 0; bx < launch.gridDim[0]; bx++) {
        const sharedMemory = semanticReferenceSharedMemory(compiled);
        const blockContexts: SemanticReferenceContext[] = [];
        for (let tz = 0; tz < launch.blockDim[2]; tz++) {
          for (let ty = 0; ty < launch.blockDim[1]; ty++) {
            for (let tx = 0; tx < launch.blockDim[0]; tx++) {
              const trace: MutableTrace = {
                blockIdx: [bx, by, bz],
                threadIdx: [tx, ty, tz],
                reads: [],
                writes: [],
                sharedReads: [],
                sharedWrites: [],
              };
              traces.push(trace);
              blockContexts.push({
                compiled,
                buffers,
                constants,
                deviceGlobals,
                textures: input.textures ?? {},
                textureDescriptors: compiled.textureDescriptors ?? {},
                surfaces,
                sharedMemory,
                sharedOffsets: new Map(),
                storageOffsets: new Map(),
                localPointerTargets: new Map(),
                scalars,
                vectors,
                locals: new Map(),
                localDimensions: new Map(),
                blockIdx: referenceVectorFromTuple([bx, by, bz]),
                threadIdx: referenceVectorFromTuple([tx, ty, tz]),
                blockDim,
                gridDim,
                blockContexts,
                trace,
              });
            }
          }
        }
        contextBlocks.push(blockContexts);
      }
    }
  }
  const barrierFunctions = semanticBarrierFunctionNames(compiled.kernelIr);
  if (semanticOperationsContainSubgroupCall(compiled.kernelIr.operations)) {
    for (const blockContexts of contextBlocks) runSemanticCollectiveOperations(compiled.kernelIr.operations, blockContexts);
  } else if (semanticOperationsContainBarrier(compiled.kernelIr.operations, barrierFunctions)) {
    runSemanticBarrierScheduler(compiled.kernelIr.operations, contextBlocks.flat(), barrierFunctions);
  } else {
    for (const context of contextBlocks.flat()) execSemanticOperations(compiled.kernelIr.operations, context);
  }

  const readback = input.readback ?? compiled.kernelIr.params
    .filter((param) =>
      param.addressSpace === "storage" && param.pointer && !param.constant ||
      param.addressSpace === "surface"
    )
    .map((param) => param.name)
    .concat(compiled.kernelIr.memory.filter((symbol) => symbol.kind === "device-global").map((symbol) => symbol.name));
  return {
    buffers: Object.fromEntries(readback.map((name) => {
      const buffer = buffers.get(name) ?? deviceGlobals.get(name) ?? surfaces[name]?.data;
      if (!buffer) throw semanticReferenceError(`missing readback buffer '${name}'`, compiled.kernelIr.span);
      return [name, buffer];
    })),
    trace: traces.map(freezeReferenceTrace),
  };
}

function unsupportedSemanticReferenceOperation(
  operations: readonly SemanticKernelIrOperation[],
  compiled: CompiledCudaLiteKernel,
  allowReturnValue = false,
): SemanticKernelIrOperation | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "dim3-declare":
      case "cooperative-group-declare":
        break;
      case "declare":
        if (operation.target.addressSpace === "shared") {
          if (operation.target.pointer || operation.target.valueType !== "uchar" && !semanticReferenceValueTypeSupported(operation.target.valueType)) return operation;
          break;
        }
        if (operation.target.addressSpace !== "local" || operation.target.pointer) return operation;
        if (!semanticReferenceLocalValueTypeSupported(operation.target.valueType)) return operation;
        if (operation.target.dimensions.length > 0 && operation.init && !semanticReferenceLocalArrayInitSupported(operation.init, operation.target.valueType, compiled)) return operation;
        if (operation.target.dimensions.length === 0) {
          const vectorTarget = isSemanticFloatVectorType(operation.target.valueType);
          if (operation.init && !semanticReferenceExpressionSupported(operation.init, vectorTarget ? "any" : "scalar", compiled)) return operation;
        }
        break;
      case "store":
        if (!semanticReferenceAssignmentOperatorSupported(operation.operator)) return operation;
        if (isSemanticFloatVectorType(operation.target.valueType) && !semanticVectorAssignmentOperatorSupported(operation.operator)) return operation;
        if (semanticReferenceVectorFieldMemoryRefSupported(operation.target) && !semanticVectorAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticReferenceTypedMemoryRefSupported(operation.target, compiled) && !semanticReferenceStorageOffsetStoreSupported(operation, compiled)) return operation;
        if (
          operation.target.addressSpace === "storage" &&
          !semanticReferenceStorageBaseSupported(operation.target.base, compiled)
        ) return operation;
        if (!semanticReferenceValueExpressionSupported(operation.value, compiled)) return operation;
        break;
      case "copy":
        if (!semanticReferenceCopySupported(operation, compiled)) return operation;
        break;
      case "copy-fence":
        if (!isCudaCpAsyncFenceCall(operation.callee)) return operation;
        break;
      case "surface-write":
        if (!semanticReferenceSurfaceWriteSupported(operation, compiled)) return operation;
        break;
      case "surface-read-store":
        if (!semanticReferenceSurfaceReadStoreSupported(operation, compiled)) return operation;
        break;
      case "atomic":
        if (!semanticReferenceAtomicSupported(operation, compiled)) return operation;
        break;
      case "call":
        if (!semanticReferenceCallSupported(operation, compiled)) return operation;
        break;
      case "expression":
        if (!semanticReferenceExpressionSupported(operation.expression, "scalar", compiled)) return operation;
        break;
      case "branch":
        if (!semanticReferenceConditionSupported(operation.condition, compiled)) return operation;
        break;
      case "block":
        break;
      case "loop":
        if (operation.init && !semanticReferenceLoopInitSupported(operation.init, compiled)) return operation;
        if (operation.condition && !semanticReferenceConditionSupported(operation.condition, compiled)) return operation;
        if (operation.update && !semanticReferenceExpressionSupported(operation.update, "scalar", compiled)) return operation;
        break;
      case "return":
        if (operation.value && (!allowReturnValue || !semanticReferenceExpressionSupported(operation.value, "any", compiled))) return operation;
        break;
      case "barrier":
        if (!isCudaBarrierCallName(operation.callee) && !isCudaCooperativeBarrierCallName(operation.callee)) return operation;
        break;
      case "fence":
        if (!isCudaFenceCallName(operation.callee)) return operation;
        break;
      case "inline-asm":
        {
          const asm = operation.op;
          const ldmatrix = semanticInlineAsmLdmatrixAssignments(operation);
          if (ldmatrix?.every((expression) => semanticReferenceExpressionSupported(expression, "scalar", compiled))) break;
          if (semanticReferenceInlineMmaSupported(operation, compiled)) break;
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
      case "break":
      case "continue":
        break;
      default:
        return operation;
    }
    if (operation.kind === "branch") {
      return unsupportedSemanticReferenceOperation(operation.consequent, compiled, allowReturnValue) ??
        unsupportedSemanticReferenceOperation(operation.alternate, compiled, allowReturnValue);
    }
    if (operation.kind === "block") return unsupportedSemanticReferenceOperation(operation.body, compiled, allowReturnValue);
    if (operation.kind === "loop") return unsupportedSemanticReferenceOperation(operation.body, compiled, allowReturnValue);
  }
  return undefined;
}

function semanticReferenceParamSupported(
  param: CompiledCudaLiteKernel["kernelIr"]["params"][number],
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (param.addressSpace === "storage") {
    return Boolean(param.pointer) && (param.valueType === "uchar"
      ? semanticDirectByteStorageParamSupported(compiled.kernelIr, param.name)
      : param.valueType === "complex64" || semanticReferenceValueTypeSupported(param.valueType));
  }
  if (param.addressSpace === "uniform") return semanticReferenceScalarTypeSupported(param.valueType) || isCudaVectorType(param.valueType);
  if (param.addressSpace === "texture") return param.valueType === "texture2d";
  if (param.addressSpace === "surface") return param.valueType === "surface2d";
  if (param.addressSpace === "pool") return !semanticOperationsReferenceRoot(compiled.kernelIr.operations, param.name);
  return false;
}

function semanticReferenceOperationsContainUnsupportedCallableSignatures(
  operations: readonly SemanticKernelIrOperation[],
  compiled: CompiledCudaLiteKernel,
): boolean {
  for (const operation of operations) {
    if (operation.kind === "call") {
      const fn = compiled.kernelIr.functions.find((item) => item.name === operation.callee);
      if (fn && fn.params.some((param) => !semanticReferenceFunctionParamSupported(param))) {
        return true;
      }
    } else if (operation.kind === "branch") {
      if (
        semanticReferenceOperationsContainUnsupportedCallableSignatures(operation.consequent, compiled) ||
        semanticReferenceOperationsContainUnsupportedCallableSignatures(operation.alternate, compiled)
      ) return true;
    } else if (operation.kind === "loop") {
      if (semanticReferenceOperationsContainUnsupportedCallableSignatures(operation.body, compiled)) return true;
    } else if (operation.kind === "block") {
      if (semanticReferenceOperationsContainUnsupportedCallableSignatures(operation.body, compiled)) return true;
    }
  }
  return false;
}

function semanticReferenceMemorySymbolSupported(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): boolean {
  if (symbol.kind === "local" || symbol.kind === "shared") return true;
  if (symbol.kind === "constant") {
    if (!semanticReferenceValueTypeSupported(symbol.valueType)) return false;
    return !symbol.initialized ||
      symbol.init !== undefined && (
        symbol.dimensions.length === 0
          ? semanticReferenceExpressionSupported(symbol.init, isSemanticFloatVectorType(symbol.valueType) ? "any" : "scalar")
          : initializedConstantArraySupported(symbol)
      );
  }
  if (symbol.kind === "device-global") return semanticReferenceScalarTypeSupported(symbol.valueType);
  if (symbol.kind === "texture") return symbol.valueType === "texture2d";
  return false;
}

function semanticReferenceFunctionParamSupported(
  param: CompiledCudaLiteKernel["kernelIr"]["functions"][number]["params"][number],
): boolean {
  if (param.pointer && param.addressSpace === "shared" && param.valueType === "uchar" && param.pointerCarrierValueType === "uchar") return true;
  if (param.pointer && param.addressSpace === "storage" && param.valueType === "uchar") return true;
  if (!param.pointer && param.addressSpace === "local" && param.valueType === "uchar") return true;
  return semanticFunctionParamContractSupported(param, semanticReferenceValueTypeSupported);
}

function semanticReferenceSharedShapeSupported(compiled: CompiledCudaLiteKernel): boolean {
  const ir = compiled.kernelIr;
  const shared = ir.memory.filter((symbol) => symbol.kind === "shared");
  const barrierFunctions = semanticBarrierFunctionNames(ir);
  const containsBarrier = semanticOperationsContainBarrier(ir.operations, barrierFunctions);
  if (shared.length === 0 && !containsBarrier) return true;
  if (!shared.every((symbol) => symbol.dimensions.length === 0 || symbol.dimensions.every((dimension) => dimension > 0))) return false;
  if (!containsBarrier) return true;
  if (barrierFunctions.size > 0) {
    return (semanticBarrierShapeSupported(ir.operations, barrierFunctions) ||
        semanticBarrierOperationsMatchUniformityProof(ir.operations, ir.barrierUniformity.kernel, barrierFunctions)) &&
      ir.functions.filter((fn) => barrierFunctions.has(fn.name)).every((fn) =>
        semanticBarrierShapeSupported(fn.body, barrierFunctions) ||
        semanticBarrierOperationsMatchUniformityProof(fn.body, ir.barrierUniformity.functions[fn.name], barrierFunctions)
      );
  }
  if (shared.length === 0) return true;
  const proof = compiled.kernelIr.barrierUniformity.kernel;
  return (ir.operations.some((operation) => operation.kind === "declare" && operation.target.name === "bg_active_lane") &&
      semanticBarrierShapeSupported(ir.operations, barrierFunctions)) ||
    semanticBarrierOperationsMatchUniformityProof(ir.operations, proof, barrierFunctions);
}

function semanticReferenceLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? unsupportedSemanticReferenceOperation([init], compiled) === undefined
    : semanticReferenceExpressionSupported(init, "scalar", compiled);
}

function semanticReferenceScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticScalarValueTypeSupported(valueType);
}

function semanticReferenceValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticValueTypeSupported(valueType);
}

function semanticReferenceLocalValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticLocalValueTypeSupported(valueType);
}

function semanticReferenceMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local" && ref.addressSpace !== "shared") {
    return false;
  }
  if (ref.fields.length > 0) return semanticReferenceVectorFieldMemoryRefSupported(ref);
  if (ref.addressSpace === "local" && ref.indices.length === 0) return semanticLocalScalarValueTypeSupported(ref.valueType);
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length === 0) return false;
  return ref.indices.every((index) => semanticReferenceExpressionSupported(index, "scalar"));
}

function semanticReferenceStorageBaseSupported(base: string, compiled: CompiledCudaLiteKernel): boolean {
  return compiled.kernelIr.params.some((param) => param.name === base && param.addressSpace === "storage") ||
    compiled.kernelIr.functions.some((fn) => fn.params.some((param) => param.name === base && param.pointer && param.addressSpace === "storage"));
}

function semanticReferenceTypedMemoryRefSupported(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  if (!semanticReferenceMemoryRefSupported(ref)) return false;
  if (semanticReferenceLocalPackedHalfView(ref, compiled)) return true;
  if (semanticReferenceLocalPackedByteRawView(ref, compiled)) return true;
  if (semanticReferencePackedSharedByteRoot(ref, compiled)) return semanticPackedSharedByteViewSupported(ref.valueType);
  if (semanticReferenceVectorFieldMemoryRefSupported(ref)) return true;
  if (semanticReferenceLocalScalarVectorView(ref, compiled)) return true;
  if (semanticReferenceSharedScalarVectorView(ref, compiled)) return true;
  if (ref.addressSpace !== "local" && ref.addressSpace !== "shared") return true;
  const symbol = compiled.kernelIr.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace);
  return symbol === undefined || symbol.valueType === ref.valueType;
}

function semanticReferenceLocalPackedHalfView(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  return ref.addressSpace === "local" && ref.pointerBaseIsScalarLane === true && ref.valueType === "half" &&
    compiled.kernelIr.memory.some((symbol) => symbol.kind === "local" && symbol.name === ref.base && symbol.valueType === "uint");
}

function semanticReferencePackedSharedByteRoot(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  if (ref.addressSpace !== "shared") return false;
  return compiled.kernelIr.memory.some((symbol) => symbol.kind === "shared" && symbol.name === ref.base && symbol.valueType === "uchar") ||
    compiled.kernelIr.functions.some((fn) => fn.params.some((param) =>
      param.name === ref.base && param.pointer && param.addressSpace === "shared" && param.pointerCarrierValueType === "uchar"
    ));
}

function semanticPackedSharedByteViewSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "uchar" || valueType === "uint" || valueType === "int" || valueType === "float";
}

function semanticReferenceSharedScalarVectorView(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  const valueType = ref.valueType;
  if (ref.addressSpace !== "shared" || !valueType || !isSemanticFloatVectorType(valueType) || ref.indices.length === 0) return false;
  const scalar = cudaVectorScalarType(valueType);
  return scalar !== undefined && compiled.kernelIr.memory.some((symbol) =>
    symbol.name === ref.base && symbol.kind === "shared" && symbol.valueType === scalar,
  );
}

function semanticReferenceLocalScalarVectorView(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  const valueType = ref.valueType;
  if (ref.addressSpace !== "local" || ref.fields.length > 0 || ref.indices.length !== 1 || !valueType || !isSemanticFloatVectorType(valueType)) return false;
  const scalar = cudaVectorScalarType(valueType);
  return scalar !== undefined && compiled.kernelIr.memory.some((symbol) =>
    symbol.name === ref.base && symbol.kind === "local" && symbol.valueType === scalar,
  );
}

function semanticReferenceCopySupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "copy" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return operation.elements >= 1 &&
    operation.elements <= 16 &&
    operation.source.valueType !== undefined &&
    operation.source.valueType === operation.target.valueType &&
    operation.source.fields.length === 0 &&
    operation.target.fields.length === 0 &&
    operation.target.addressSpace !== "constant" &&
    semanticReferenceTypedMemoryRefSupported(operation.source, compiled) &&
    semanticReferenceTypedMemoryRefSupported(operation.target, compiled);
}

function semanticReferenceVectorFieldMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.fields.length !== 1) return false;
  const lanes = semanticStorageVectorFieldIndices(ref.containerValueType, ref.fields[0]!);
  if (lanes === undefined || new Set(lanes).size !== lanes.length) return false;
  return ref.indices.length > 0 && ref.indices.every((index) => semanticReferenceExpressionSupported(index, "scalar"));
}

function semanticReferenceStorageOffsetStoreSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return operation.target.addressSpace === "storage" &&
    operation.target.indices.length === 0 &&
    operation.target.fields.length === 0 &&
    (operation.operator === "+=" || operation.operator === "-=") &&
    compiled.kernelIr.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage") &&
    semanticReferenceExpressionSupported(operation.value, "scalar", compiled);
}

function semanticReferenceAtomicSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const atomicOp = semanticAtomicOperation(operation.callee);
  if (!atomicOp) return false;
  if (!operation.target || !semanticReferenceAtomicMemoryRefSupported(operation.target, compiled)) return false;
  if (!semanticAtomicReferenceValueTypeSupported(atomicOp, operation.target.valueType)) return false;
  if (!semanticReferenceAtomicTargetRootSupported(operation.target, compiled)) {
    return false;
  }
  const scalarArgIndices = semanticAtomicScalarArgumentIndices(atomicOp);
  return operation.args.length >= scalarArgIndices.length + 1 &&
    scalarArgIndices.every((index) => semanticReferenceExpressionSupported(operation.args[index]!, "scalar", compiled));
}

function semanticReferenceValueExpressionSupported(expression: SemanticExpression, compiled: CompiledCudaLiteKernel): boolean {
  return semanticReferenceExpressionSupported(expression, "scalar", compiled) ||
    semanticReferenceExpressionSupported(expression, "any", compiled) && isSemanticFloatVectorType(semanticExpressionValueType(expression)) ||
    expression.kind === "call" && (semanticReferenceAtomicCallSupported(expression, compiled) || semanticReferenceCurandCallSupported(expression, compiled) || semanticReferenceGeneratedRandomCallSupported(expression) || semanticReferenceSubgroupCallSupported(expression) || semanticReferenceAddressPredicateCallSupported(expression) || semanticReferenceMathCallSupported(expression, "any", compiled) || semanticReferenceHalf2CallSupported(expression, compiled) || semanticReferenceBf162CallSupported(expression, compiled) || semanticReferenceVectorConstructorSupported(expression, "any", compiled) || semanticReferenceVectorAtCallSupported(expression, compiled) || semanticReferenceVectorLerpCallSupported(expression, compiled) || semanticReferenceVectorMathCallSupported(expression)) ||
    expression.kind === "texture-read" && semanticReferenceTextureReadSupported(expression, compiled) ||
    expression.kind === "surface-read" && semanticReferenceSurfaceReadSupported(expression, compiled);
}

function semanticReferenceLocalArrayInitSupported(
  expression: SemanticExpression,
  targetValueType: CudaLiteScalarType | undefined,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return semanticLocalArrayInitContractSupported(expression, targetValueType, (item, expected) => semanticReferenceExpressionSupported(item, expected, compiled));
}

function semanticReferenceMathCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any" = "scalar",
  compiled?: CompiledCudaLiteKernel,
): boolean {
  const name = expression.callee.kind === "symbol" ? expression.callee.name : undefined;
  if (semanticVectorMinMaxCallValueType(name, expression.args) !== undefined) {
    return expected === "any" && expression.args.every((arg) => semanticReferenceExpressionSupported(arg, "any", compiled));
  }
  return semanticMathCallArgumentsSupported(
    name,
    expression.args,
    (arg) => semanticReferenceExpressionSupported(arg, "scalar", compiled),
  );
}

function semanticReferenceVectorMathCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  return expression.callee.kind === "symbol" && semanticVectorMathCallSupported(expression.callee.name, expression.args);
}

function semanticReferenceCurandCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol" || !isSemanticCurandCallName(expression.callee.name)) return false;
  const stateIndex = semanticCurandStateArgumentIndex(expression.callee.name);
  return stateIndex !== undefined &&
    expression.args.length === semanticCurandArity(expression.callee.name) &&
    semanticCurandState(expression.args[stateIndex]!) !== undefined &&
    semanticCurandScalarArgumentIndices(expression.callee.name)
      .every((index) => semanticReferenceExpressionSupported(expression.args[index]!, "scalar", compiled));
}

function semanticReferenceGeneratedRandomCallSupported(
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

function semanticReferenceSubgroupCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace === "function" || !SEMANTIC_SUBGROUP_CALLS.has(expression.callee.name)) return false;
  const scalarArgs = semanticSubgroupScalarArguments(expression.callee.name, expression.args);
  return scalarArgs !== undefined && scalarArgs.every((arg) => semanticReferenceExpressionSupported(arg, "scalar"));
}

function semanticReferenceAddressPredicateCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  return expression.callee.kind === "symbol" &&
    SEMANTIC_ADDRESS_PREDICATE_CALLS.has(expression.callee.name) &&
    expression.args.length === 1 &&
    semanticAddressPredicateAddressSpace(expression.args[0]) !== undefined;
}

function semanticReferenceTextureReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return semanticTextureSurfaceValueTypeSupported(expression.valueType) &&
    expression.texture.kind === "symbol" &&
    expression.texture.addressSpace === "texture" &&
    semanticReferenceExpressionSupported(expression.x, "scalar", compiled) &&
    semanticReferenceExpressionSupported(expression.y, "scalar", compiled) &&
    semanticTextureReadCoordinateShapeSupported(expression.callee, expression.z !== undefined) &&
    (expression.z === undefined || semanticReferenceExpressionSupported(expression.z, "scalar", compiled));
}

function semanticReferenceTextureDescriptorsSupported(_compiled: CompiledCudaLiteKernel): boolean {
  return true;
}

function semanticReferenceSurfaceReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const surface = expression.surface;
  return semanticTextureSurfaceValueTypeSupported(expression.valueType) &&
    surface.kind === "symbol" &&
    surface.addressSpace === "surface" &&
    semanticReferenceExpressionSupported(expression.xBytes, "scalar", compiled) &&
    semanticReferenceExpressionSupported(expression.y, "scalar", compiled) &&
    (expression.z === undefined || semanticReferenceExpressionSupported(expression.z, "scalar", compiled));
}

function semanticReferenceFunctionCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const callee = expression.callee.name;
  const fn = compiled.kernelIr.functions.find((item) => item.name === callee);
  if (!fn || !semanticReferenceLocalValueTypeSupported(fn.returnType)) return false;
  if (fn.params.some((param) => !semanticReferenceFunctionParamSupported(param))) return false;
  if (fn.params.some((param) => param.pointer && param.addressSpace !== "constant") && !semanticReferencePointerFunctionBodySupported(fn)) return false;
  if (!semanticFunctionLocalParamValueTypesSupported(fn, semanticReferenceLocalValueTypeSupported)) return false;
  if (!semanticReferenceFunctionBodyShapeSupported(fn.body, semanticReferenceFunctionHasSharedPointer(fn))) return false;
  return expression.args.length === fn.params.length &&
    expression.args.every((arg, index) => semanticReferenceFunctionArgSupported(arg, fn.params[index], compiled)) &&
    unsupportedSemanticReferenceOperation(fn.body, compiled, true) === undefined;
}

function semanticReferenceCooperativeGroupCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (!(expression.callee.kind === "member" &&
    expression.callee.object.kind === "symbol" &&
    expression.args.length === 0 &&
    (expression.callee.property === "thread_rank" ||
      expression.callee.property === "size" ||
      expression.callee.property === "meta_group_rank" ||
      expression.callee.property === "meta_group_size"))) return false;
  const group = semanticCooperativeGroupInfo(compiled.kernelIr, expression.callee.object.name);
  return group !== undefined && group.kind !== "coalesced";
}

function semanticReferenceCooperativeReduceCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol" ||
    (expression.callee.name !== "cg::reduce" && expression.callee.name !== "cooperative_groups::reduce")) return false;
  const [groupArg, valueArg, operationArg] = expression.args;
  if (groupArg?.kind !== "symbol" || !valueArg || operationArg?.kind !== "call" ||
    operationArg.callee.kind !== "symbol" || !operationArg.callee.name.endsWith("::plus")) return false;
  const group = semanticCooperativeGroupInfo(compiled.kernelIr, groupArg.name);
  return group?.partitioned === true && semanticReferenceExpressionSupported(valueArg, "scalar", compiled);
}

function semanticReferenceFunctionArgSupported(
  arg: SemanticExpression,
  param: CompiledCudaLiteKernel["kernelIr"]["functions"][number]["params"][number] | undefined,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (param?.pointer && param.addressSpace === "storage" && param.valueType === "uchar") {
    const ref = semanticPointerArgMemoryRef(arg);
    return ref?.addressSpace === "storage" && semanticDirectByteStorageParamSupported(compiled.kernelIr, ref.base);
  }
  const supported = semanticFunctionArgContractSupported(
    arg,
    param,
    semanticPointerArgMemoryRef,
    (item, mode) => semanticReferenceExpressionSupported(item, mode, compiled),
  );
  return supported;
}

function semanticReferenceVectorConstructorSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any",
  compiled?: CompiledCudaLiteKernel,
): boolean {
  return semanticVectorConstructorCallContractSupported(
    expression.callee.kind === "symbol" ? expression.callee.name : undefined,
    expression.args,
    expected,
    (arg, mode) => semanticReferenceExpressionSupported(arg, mode, compiled),
  );
}

function semanticReferenceVectorIndexSupported(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  const ref = memoryRefFromIndexExpression(expression);
  if (ref && !(ref.addressSpace === "local" && isSemanticFloatVectorType(semanticExpressionValueType(expression.target)))) return false;
  return isSemanticFloatVectorType(semanticExpressionValueType(expression.target)) &&
    semanticReferenceExpressionSupported(expression.target, "any", compiled) &&
    semanticReferenceExpressionSupported(expression.index, "scalar", compiled);
}

function semanticReferenceVectorAtCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  return semanticVectorAtCallContractSupported(
    expression.callee.kind === "symbol" ? expression.callee.name : undefined,
    expression.args,
    semanticExpressionValueType,
    (arg, mode) => semanticReferenceExpressionSupported(arg, mode, compiled),
  );
}

function semanticReferenceVectorLerpCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  return semanticVectorLerpCallContractSupported(
    expression.callee.kind === "symbol" ? expression.callee.name : undefined,
    expression.args,
    semanticExpressionValueType,
    (arg, mode) => semanticReferenceExpressionSupported(arg, mode, compiled),
  );
}

function semanticReferenceHalf2CallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  return semanticHalf2CallArgumentsSupported(
    expression.callee.name,
    expression.args,
    semanticExpressionVectorValueType,
    (arg, expected) => semanticReferenceExpressionSupported(arg, expected, compiled),
  );
}

function semanticReferenceBf162CallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  return semanticBf162CallArgumentsSupported(
    expression.callee.name,
    expression.args,
    semanticExpressionVectorValueType,
    (arg, expected) => semanticReferenceExpressionSupported(arg, expected, compiled),
  );
}

function semanticReferenceFunctionBodyShapeSupported(
  operations: readonly SemanticKernelIrOperation[],
  allowAtomic = false,
): boolean {
  return semanticFunctionBodyShapeContractSupported(operations, { allowBlock: true, allowBarrierFence: true, allowAtomic, allowSharedMemory: true, allowLocalArrays: true });
}

function semanticReferenceFunctionHasSharedPointer(fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number]): boolean {
  return fn.params.some((param) => param.pointer && param.addressSpace === "shared");
}

function semanticReferencePointerFunctionBodySupported(fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number]): boolean {
  return semanticPointerFunctionBodyContractSupported(fn, memoryRefFromIndexExpression, semanticAtomicCallTarget, {
    allowCooperativeOps: true,
    allowSharedMemory: true,
    allowDeviceGlobals: true,
    allowLocalArrays: true,
    allowConstantMemory: true,
    allowStoragePointerIdentity: true,
  });
}

function semanticReferenceAtomicCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const atomicOp = semanticAtomicOperation(expression.callee.name);
  if (!atomicOp) return false;
  const target = semanticAtomicCallTarget(expression);
  if (!target || !semanticReferenceAtomicMemoryRefSupported(target, compiled)) return false;
  if (!semanticAtomicReferenceValueTypeSupported(atomicOp, target.valueType)) return false;
  if (!semanticReferenceAtomicTargetRootSupported(target, compiled)) {
    return false;
  }
  const scalarArgIndices = semanticAtomicScalarArgumentIndices(atomicOp);
  return expression.args.length >= scalarArgIndices.length + 1 &&
    scalarArgIndices.every((index) => semanticReferenceExpressionSupported(expression.args[index]!, "scalar"));
}

function semanticReferenceAtomicMemoryRefSupported(
  ref: SemanticMemoryRef,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return semanticReferenceMemoryRefSupported(ref) ||
    ref.addressSpace === "storage" &&
      ref.indices.length === 0 &&
      compiled.kernelIr.functions.some((fn) =>
        fn.params.some((param) => param.name === ref.base && param.pointer && param.addressSpace === "storage")
      );
}

function semanticReferenceCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (operation.result !== undefined) {
    const fn = compiled.kernelIr.functions.find((item) => item.name === operation.callee);
    return fn !== undefined &&
      fn.returnType !== "void" &&
      fn.returnType === operation.result.valueType &&
      semanticReferenceFunctionCallSupported(semanticCallOperationExpression(operation, fn.returnType), compiled);
  }
  if (operation.callee === "assert") return semanticAssertCallSupported(operation.args, (arg) => semanticReferenceExpressionSupported(arg, "scalar", compiled));
  if (operation.callee === "printf") return semanticPrintfCallSupported(operation.args, (arg) => semanticReferenceExpressionSupported(arg, "scalar", compiled));
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) {
    return semanticNoopCallSupported(operation.callee, operation.args, (arg) => semanticReferenceExpressionSupported(arg, "scalar", compiled));
  }
  if (operation.callee === "curand_init") {
    return semanticReferenceCurandCallSupported({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: "uint",
      span: operation.span,
    }, compiled);
  }
  if (operation.callee === "skipahead") {
    return semanticReferenceCurandCallSupported({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: "uint",
      span: operation.span,
    }, compiled);
  }
  if (semanticReferenceVoidFunctionCallSupported(operation, compiled)) return true;
  return semanticLocalArrayFillCallSupported(
    operation,
    (name) => compiled.kernelIr.memory.find((item) =>
      item.kind === "local" &&
      item.name === name &&
      item.dimensions.length > 0
    ),
    (item, expected) => semanticReferenceExpressionSupported(item, expected, compiled),
  );
}

function semanticReferenceVoidFunctionCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (operation.result !== undefined) return false;
  const fn = compiled.kernelIr.functions.find((item) => item.name === operation.callee);
  if (!fn || fn.returnType !== "void") return false;
  if (fn.params.some((param) => !semanticReferenceFunctionParamSupported(param))) return false;
  if (fn.params.some((param) => param.pointer) && !semanticReferencePointerFunctionBodySupported(fn)) return false;
  if (!semanticFunctionLocalParamValueTypesSupported(fn, semanticReferenceLocalValueTypeSupported)) return false;
  return operation.args.length === fn.params.length &&
    operation.args.every((arg, index) => semanticReferenceFunctionArgSupported(arg, fn.params[index], compiled)) &&
    semanticReferenceFunctionBodyShapeSupported(fn.body, semanticReferenceFunctionHasSharedPointer(fn)) &&
    unsupportedSemanticReferenceOperation(fn.body, compiled, true) === undefined;
}

function semanticReferenceSurfaceWriteSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-write" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const surface = operation.surface;
  return surface.kind === "symbol" &&
    surface.addressSpace === "surface" &&
    semanticReferenceSurfaceValueSupported(operation.value) &&
    semanticReferenceExpressionSupported(operation.value, "any", compiled) &&
    semanticReferenceExpressionSupported(operation.xBytes, "scalar", compiled) &&
    semanticReferenceExpressionSupported(operation.y, "scalar", compiled) &&
    (operation.z === undefined || semanticReferenceExpressionSupported(operation.z, "scalar", compiled));
}

function semanticReferenceSurfaceValueSupported(expression: SemanticExpression): boolean {
  const valueType = semanticExpressionValueType(expression);
  return !isSemanticFloatVectorType(valueType) || isCudaVectorType(valueType);
}

function semanticReferenceSurfaceReadStoreSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-read-store" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const target = semanticReferenceSurfaceReadTarget(operation.target);
  return target !== undefined &&
    semanticReferenceSurfaceReadSupported(
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
      compiled,
    );
}

function semanticReferenceAtomicTargetRootSupported(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  if (ref.addressSpace === "storage") {
    return compiled.kernelIr.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && !param.constant) ||
      compiled.kernelIr.functions.some((fn) =>
        fn.params.some((param) => param.name === ref.base && param.pointer && param.addressSpace === "storage")
      );
  }
  if (ref.addressSpace === "device-global") {
    return compiled.kernelIr.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "device-global");
  }
  if (ref.addressSpace === "shared") {
    return compiled.kernelIr.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "shared") ||
      compiled.kernelIr.functions.some((fn) =>
        fn.params.some((param) => param.name === ref.base && param.pointer && param.addressSpace === "shared")
      );
  }
  return false;
}

function semanticReferenceExpressionSupported(
  expression: SemanticExpression,
  expected: "scalar" | "any",
  compiled?: CompiledCudaLiteKernel,
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
        isBuiltinVectorSymbol(expression.name);
    case "member":
      if (expected === "scalar" && isCudaVectorType(expression.valueType)) return false;
      return isBuiltinVectorMember(expression) || semanticReferenceVectorMemberSupported(expression, compiled);
    case "index":
      if (semanticReferenceVectorIndexSupported(expression, compiled)) return true;
      {
        const ref = memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span);
        const supported = compiled === undefined ? semanticReferenceMemoryRefSupported(ref) : semanticReferenceTypedMemoryRefSupported(ref, compiled);
        return supported && (expected === "any" || !isSemanticFloatVectorType(expression.valueType));
      }
    case "cast":
      return !expression.pointer && semanticReferenceExpressionSupported(expression.expression, "scalar", compiled);
    case "unary":
      if (expected === "scalar" && semanticReferenceBf162LocalBitsCastSupported(expression, compiled)) return true;
      return expression.operator !== "*" && expression.operator !== "&" && semanticReferenceExpressionSupported(expression.argument, "scalar", compiled);
    case "binary":
      if (isStoragePointerNullComparison(expression)) return true;
      if (compiled !== undefined && isStoragePointerIdentityComparison(expression, compiled)) return true;
      if (expected === "any" && isSemanticFloatVectorType(expression.valueType) && semanticReferenceVectorBinaryOperatorSupported(expression.operator)) {
        return semanticReferenceExpressionSupported(expression.left, "any", compiled) &&
          semanticReferenceExpressionSupported(expression.right, "any", compiled);
      }
      return semanticReferenceExpressionSupported(expression.left, "scalar", compiled) &&
        semanticReferenceExpressionSupported(expression.right, "scalar", compiled);
    case "conditional":
      return semanticReferenceConditionSupported(expression.condition, compiled) &&
        semanticReferenceExpressionSupported(expression.consequent, expected, compiled) &&
        semanticReferenceExpressionSupported(expression.alternate, expected, compiled);
    case "assignment":
      {
        const vectorTarget = isSemanticFloatVectorType(semanticExpressionValueType(expression.target));
        return semanticReferenceAssignmentOperatorSupported(expression.operator) &&
        (!vectorTarget || semanticVectorAssignmentOperatorSupported(expression.operator)) &&
        (expression.target.kind === "symbol" && expression.target.addressSpace === "local" ||
          expression.target.kind === "member" && semanticReferenceVectorMemberSupported(expression.target, compiled) ||
          semanticReferenceAssignmentMemoryRefSupported(expression.target, compiled)) &&
        semanticReferenceExpressionSupported(expression.value, vectorTarget ? "any" : "scalar", compiled);
      }
    case "sequence":
      return expression.expressions.every((item) => semanticReferenceExpressionSupported(item, "scalar", compiled));
    case "update":
      return (expression.argument.kind === "symbol" && expression.argument.addressSpace === "local" ||
          (compiled === undefined
            ? Boolean(memoryRefFromIndexExpression(expression.argument))
            : semanticReferenceAssignmentMemoryRefSupported(expression.argument, compiled))) &&
        (expression.operator === "++" || expression.operator === "--");
    case "call":
      return semanticReferenceSharedAddressCallSupported(expression) ||
        compiled !== undefined && semanticReferenceCooperativeGroupCallSupported(expression, compiled) ||
        compiled !== undefined && semanticReferenceCooperativeReduceCallSupported(expression, compiled) ||
        compiled !== undefined && semanticReferenceFunctionCallSupported(expression, compiled) ||
        compiled !== undefined && semanticReferenceAtomicCallSupported(expression, compiled) ||
        (semanticReferenceCurandCallSupported(expression, compiled) || semanticReferenceGeneratedRandomCallSupported(expression)) &&
          (expected === "any" || !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression))) ||
        semanticReferenceSubgroupCallSupported(expression) ||
        semanticReferenceAddressPredicateCallSupported(expression) ||
        semanticReferenceMathCallSupported(expression, expected, compiled) ||
        semanticReferenceHalf2CallSupported(expression, compiled) ||
        semanticReferenceBf162CallSupported(expression, compiled) ||
        semanticReferenceVectorConstructorSupported(expression, expected, compiled) ||
        expected === "scalar" && semanticReferenceVectorAtCallSupported(expression, compiled) ||
        expected === "any" && (semanticReferenceVectorLerpCallSupported(expression, compiled) || semanticReferenceVectorMathCallSupported(expression)) ||
        expected === "scalar" && semanticReferenceVectorMathCallSupported(expression) && expression.callee.kind === "symbol" &&
          (expression.callee.name === "dot" || expression.callee.name === "length");
    case "texture-read":
      return compiled !== undefined &&
        (expected === "any" || semanticTextureSurfaceValueTypeSupported(expression.valueType)) &&
        semanticReferenceTextureReadSupported(expression, compiled);
    case "surface-read":
      return compiled !== undefined && (expected === "scalar" || expected === "any") && semanticReferenceSurfaceReadSupported(expression, compiled);
    case "initializer":
      return false;
  }
}

function unsupportedMemoryRef(span: SourceSpan): SemanticMemoryRef {
  return { base: "", addressSpace: "unknown", indices: [], fields: [], span };
}

function semanticReferenceOperationsContainUnsupportedCalls(
  operations: readonly SemanticKernelIrOperation[],
  compiled: CompiledCudaLiteKernel,
): boolean {
  return operations.some((operation) => {
    if (operation.kind === "declare" && operation.init) return semanticReferenceExpressionContainsUnsupportedCall(operation.init, compiled);
    if (operation.kind === "store") {
      return operation.target.indices.some((index) => semanticReferenceExpressionContainsUnsupportedCall(index, compiled)) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.value, compiled);
    }
    if (operation.kind === "surface-write") {
      return semanticReferenceExpressionContainsUnsupportedCall(operation.surface, compiled) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.value, compiled) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.xBytes, compiled) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.y, compiled) ||
        Boolean(operation.z && semanticReferenceExpressionContainsUnsupportedCall(operation.z, compiled));
    }
    if (operation.kind === "surface-read-store") {
      return semanticReferenceExpressionContainsUnsupportedCall(operation.target, compiled) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.surface, compiled) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.xBytes, compiled) ||
        semanticReferenceExpressionContainsUnsupportedCall(operation.y, compiled) ||
        Boolean(operation.z && semanticReferenceExpressionContainsUnsupportedCall(operation.z, compiled));
    }
    if (operation.kind === "atomic") return operation.args.some((arg) => semanticReferenceExpressionContainsUnsupportedCall(arg, compiled));
    if (operation.kind === "call") return operation.args.some((arg) => semanticReferenceExpressionContainsUnsupportedCall(arg, compiled));
    if (operation.kind === "expression") return semanticReferenceExpressionContainsUnsupportedCall(operation.expression, compiled);
    if (operation.kind === "branch") {
      return semanticReferenceExpressionContainsUnsupportedCall(operation.condition, compiled) ||
        semanticReferenceOperationsContainUnsupportedCalls(operation.consequent, compiled) ||
        semanticReferenceOperationsContainUnsupportedCalls(operation.alternate, compiled);
    }
    if (operation.kind === "loop") {
      return Boolean(operation.init && (isSemanticKernelIrOperation(operation.init)
        ? semanticReferenceOperationsContainUnsupportedCalls([operation.init], compiled)
        : semanticReferenceExpressionContainsUnsupportedCall(operation.init, compiled))) ||
        Boolean(operation.condition && semanticReferenceExpressionContainsUnsupportedCall(operation.condition, compiled)) ||
        Boolean(operation.update && semanticReferenceExpressionContainsUnsupportedCall(operation.update, compiled)) ||
        semanticReferenceOperationsContainUnsupportedCalls(operation.body, compiled);
    }
    if (operation.kind === "return") return Boolean(operation.value && semanticReferenceExpressionContainsUnsupportedCall(operation.value, compiled));
    if (operation.kind === "block") return semanticReferenceOperationsContainUnsupportedCalls(operation.body, compiled);
    return false;
  });
}

function semanticReferenceExpressionContainsUnsupportedCall(
  expression: SemanticExpression,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.kind === "call") {
    if (semanticReferenceCooperativeReduceCallSupported(expression, compiled)) return false;
    return !(semanticReferenceAtomicCallSupported(expression, compiled) ||
      semanticReferenceSharedAddressCallSupported(expression) ||
      semanticReferenceCooperativeGroupCallSupported(expression, compiled) ||
      semanticReferenceCooperativeReduceCallSupported(expression, compiled) ||
      semanticReferenceCurandCallSupported(expression, compiled) ||
      semanticReferenceGeneratedRandomCallSupported(expression) ||
      semanticReferenceFunctionCallSupported(expression, compiled) ||
      semanticReferenceSubgroupCallSupported(expression) ||
      semanticReferenceAddressPredicateCallSupported(expression) ||
      semanticReferenceMathCallSupported(expression, "any", compiled) ||
      semanticReferenceHalf2CallSupported(expression, compiled) ||
      semanticReferenceBf162CallSupported(expression, compiled) ||
      semanticReferenceVectorConstructorSupported(expression, "any", compiled) ||
      semanticReferenceVectorAtCallSupported(expression, compiled) ||
      semanticReferenceVectorLerpCallSupported(expression, compiled) ||
      semanticReferenceVectorMathCallSupported(expression)) ||
      expression.args.some((arg) => semanticReferenceExpressionContainsUnsupportedCall(arg, compiled));
  }
  if (expression.kind === "texture-read") {
    return !semanticReferenceTextureReadSupported(expression, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.texture, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.x, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.y, compiled) ||
      expression.z !== undefined && semanticReferenceExpressionContainsUnsupportedCall(expression.z, compiled);
  }
  if (expression.kind === "surface-read") {
    return !semanticReferenceSurfaceReadSupported(expression, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.surface, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.xBytes, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.y, compiled) ||
      Boolean(expression.z && semanticReferenceExpressionContainsUnsupportedCall(expression.z, compiled));
  }
  return semanticReferenceExpressionChildren(expression).some((child) => semanticReferenceExpressionContainsUnsupportedCall(child, compiled));
}

function semanticReferenceExpressionChildren(expression: SemanticExpression): readonly SemanticExpression[] {
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return [];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "cast":
      return [expression.expression];
    case "unary":
    case "update":
      return [expression.argument];
    case "binary":
      return [expression.left, expression.right];
    case "conditional":
      return [expression.condition, expression.consequent, expression.alternate];
    case "assignment":
      return [expression.target, expression.value];
    case "initializer":
      return expression.elements;
    case "sequence":
      return expression.expressions;
    case "call":
      return expression.args;
    case "texture-read":
      return [expression.texture, expression.x, expression.y, ...(expression.z ? [expression.z] : [])];
    case "surface-read":
      return [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])];
  }
}

function semanticReferenceAssignmentMemoryRefSupported(
  expression: SemanticExpression,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  const ref = semanticReferenceAssignmentMemoryRef(expression, compiled);
  return ref !== undefined &&
    (compiled === undefined ? semanticReferenceMemoryRefSupported(ref) : semanticReferenceTypedMemoryRefSupported(ref, compiled)) &&
    !isSemanticFloatVectorType(ref.valueType);
}

function semanticReferenceAssignmentMemoryRef(
  expression: SemanticExpression,
  _compiled?: CompiledCudaLiteKernel,
): SemanticMemoryRef | undefined {
  return memoryRefFromIndexExpression(expression);
}

function isStoragePointerNullComparison(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): boolean {
  if (expression.operator !== "==" && expression.operator !== "!=") return false;
  return isStorageSymbol(expression.left) && isNullLiteral(expression.right) ||
    isStorageSymbol(expression.right) && isNullLiteral(expression.left);
}

function isStoragePointerIdentityComparison(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.operator !== "==" && expression.operator !== "!=") return false;
  return semanticReferenceStoragePointerSymbol(expression.left, compiled) &&
    semanticReferenceStoragePointerSymbol(expression.right, compiled);
}

function semanticReferenceStoragePointerSymbol(
  expression: SemanticExpression,
  compiled: CompiledCudaLiteKernel,
): expression is Extract<SemanticExpression, { readonly kind: "symbol" }> {
  if (expression.kind !== "symbol" || expression.addressSpace !== "storage") return false;
  return compiled.kernelIr.params.some((param) =>
    param.name === expression.name && param.pointer && param.addressSpace === "storage") ||
    compiled.kernelIr.functions.some((fn) => fn.params.some((param) =>
      param.name === expression.name && param.pointer && param.addressSpace === "storage"));
}

function semanticReferenceConditionSupported(expression: SemanticExpression, compiled?: CompiledCudaLiteKernel): boolean {
  return expression.kind === "symbol" && expression.addressSpace === "storage" ||
    semanticReferenceExpressionSupported(expression, "scalar", compiled);
}

function isStorageSymbol(expression: SemanticExpression): boolean {
  return expression.kind === "symbol" && expression.addressSpace === "storage";
}

function isZeroLiteral(expression: SemanticExpression): boolean {
  return expression.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
}

function isNullLiteral(expression: SemanticExpression): boolean {
  return isZeroLiteral(expression) || expression.kind === "symbol" && (expression.name === "NULL" || expression.name === "nullptr");
}

function isBuiltinVectorSymbol(name: string): boolean {
  return isCudaBuiltinVectorSymbolName(name);
}

function isBuiltinVectorMember(expression: Extract<SemanticExpression, { kind: "member" }>): boolean {
  return expression.object.kind === "symbol" &&
    isBuiltinVectorSymbol(expression.object.name) &&
    (expression.property === "x" || expression.property === "y" || expression.property === "z");
}

function semanticReferenceVectorMemberSupported(
  expression: Extract<SemanticExpression, { kind: "member" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  const valueType = semanticExpressionValueType(expression.object);
  return semanticReferenceExpressionSupported(expression.object, "any", compiled) &&
    semanticStorageVectorFieldIndices(valueType, expression.property) !== undefined;
}

function execSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: SemanticReferenceContext,
): SemanticControl {
  for (const operation of operations) {
    switch (operation.kind) {
      case "dim3-declare":
        break;
      case "cooperative-group-declare":
        recordSemanticPartitionMembership(operation.declaration, context);
        break;
      case "declare":
        if (operation.target.addressSpace === "shared") break;
        if (operation.target.dimensions.length > 0) context.localDimensions.set(operation.target.name, operation.target.dimensions);
        context.locals.set(operation.target.name, semanticDeclareValue(operation, context));
        break;
      case "store":
        if (semanticReferenceStorageOffsetStoreSupported(operation, context.compiled)) {
          const delta = Math.trunc(evalNumber(operation.value, context));
          const current = context.storageOffsets.get(operation.target.base) ?? 0;
          context.storageOffsets.set(operation.target.base, operation.operator === "-=" ? current - delta : current + delta);
          break;
        }
        writeMemoryValue(operation.target, storeValueExpression(operation, context), context);
        break;
      case "copy":
        execSemanticCopy(operation, context);
        break;
      case "copy-fence":
        break;
      case "surface-write":
        execSemanticSurfaceWrite(operation, context);
        break;
      case "surface-read-store":
        execSemanticSurfaceReadStore(operation, context);
        break;
      case "atomic":
        execSemanticAtomic(operation, context);
        break;
      case "call":
        execSemanticCall(operation, context);
        break;
      case "expression":
        evalSemanticExpression(operation.expression, context);
        break;
      case "branch":
        if (truthy(evalNumber(operation.condition, context))) {
          const control = execSemanticScopedOperations(operation.consequent, context);
          if (control !== "fallthrough") return control;
        } else {
          const control = execSemanticScopedOperations(operation.alternate, context);
          if (control !== "fallthrough") return control;
        }
        break;
      case "block":
        {
          const control = execSemanticScopedOperations(operation.body, context);
          if (control !== "fallthrough") return control;
        }
        break;
      case "loop":
        {
          const control = execSemanticLoop(operation, context);
          if (control !== "fallthrough") return control;
        }
        break;
      case "return":
        if (operation.value) context.returnValue = evalSemanticExpression(operation.value, context);
        return "return";
      case "barrier":
        break;
      case "fence":
        break;
      case "inline-asm":
        {
          const asm = operation.op;
          const ldmatrix = semanticInlineAsmLdmatrixAssignments(operation);
          if (ldmatrix) {
            for (const assignment of ldmatrix) evalSemanticExpression(assignment, context);
            break;
          }
          if (asm?.kind === "mma-m16n8k16") {
            execSemanticInlineMma(operation, asm.accumulator, context);
            break;
          }
          const cpAsyncFenceSupported = asm?.kind === "cp-async-fence" && operation.inputs.length <= (asm.fence === "wait_group" ? 1 : 0) && operation.outputs.length === 0;
          const membarSupported = asm?.kind === "membar" && operation.inputs.length === 0 && operation.outputs.length === 0;
          const barSyncSupported = asm?.kind === "bar-sync" && operation.inputs.length === (asm.operand === "input0" ? 1 : 0) && operation.outputs.length === 0;
          if (!cpAsyncFenceSupported && !membarSupported && !barSyncSupported) {
            throw semanticReferenceError(`semantic reference does not support ${operation.kind}`, operation.span);
          }
        }
        break;
      case "break":
        return "break";
      case "continue":
        return "continue";
      default:
        throw semanticReferenceError(`semantic reference does not support ${operation.kind}`, operation.span);
    }
  }
  return "fallthrough";
}

function execSemanticCopy(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "copy" }>,
  context: SemanticReferenceContext,
): void {
  for (let offset = 0; offset < operation.elements; offset++) {
    const source = semanticCopyMemoryRefAt(operation.source, offset);
    const target = semanticCopyMemoryRefAt(operation.target, offset);
    writeMemoryValue(target, readMemoryValue(source, context), context);
  }
}

function semanticReferenceInlineMmaSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "inline-asm" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const op = operation.op;
  const countsMatch = op?.kind === "mma-m16n8k16" &&
    (op.accumulator === "f16"
      ? operation.outputs.length === 2 && operation.inputs.length === 8
      : operation.outputs.length === 4 && operation.inputs.length === 10);
  if (!countsMatch) return false;
  return operation.inputs.every((input) => semanticReferenceExpressionSupported(input, "scalar", compiled)) &&
    operation.outputs.every((output) => semanticReferenceInlineOutputSupported(output, compiled));
}

function semanticReferenceInlineOutputSupported(
  output: SemanticExpression,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const assignment: SemanticExpression = {
    kind: "assignment",
    operator: "=",
    target: output,
    value: { kind: "literal", literalKind: "number", value: 0, valueType: "uint", span: output.span },
    valueType: "uint",
    span: output.span,
  };
  return semanticReferenceExpressionSupported(assignment, "scalar", compiled);
}

function execSemanticInlineMma(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "inline-asm" }>,
  accumulator: "f16" | "f32",
  context: SemanticReferenceContext,
): void {
  const inputs = operation.inputs.map((input) => Number(evalSemanticExpression(input, context)));
  if (accumulator === "f16") {
    for (let index = 0; index < operation.outputs.length; index++) {
      const a = inputs[index % 4]! >>> 0;
      const b = inputs[4 + (index % 2)]! >>> 0;
      const c = inputs[6 + index]! >>> 0;
      const lane0 = float16BitsToFloat32(c & 0xffff) + float16BitsToFloat32(a & 0xffff) * float16BitsToFloat32(b & 0xffff);
      const lane1 = float16BitsToFloat32(c >>> 16) + float16BitsToFloat32(a >>> 16) * float16BitsToFloat32(b >>> 16);
      assignSemanticInlineOutput(operation.outputs[index]!, (float32ToFloat16Bits(lane0) | (float32ToFloat16Bits(lane1) << 16)) >>> 0, context);
    }
    return;
  }
  for (let index = 0; index < operation.outputs.length; index++) {
    const a = inputs[index % 4]! >>> 0;
    const b = inputs[4 + (index % 2)]! >>> 0;
    const cExpression = operation.inputs[6 + index]!;
    const rawC = inputs[6 + index]!;
    const cType = semanticExpressionValueType(cExpression);
    const c = cType === "uint" || cType === "int" ? uintBitsToFloat32(rawC >>> 0) : rawC;
    const product = float16BitsToFloat32(a & 0xffff) * float16BitsToFloat32(b & 0xffff) +
      float16BitsToFloat32(a >>> 16) * float16BitsToFloat32(b >>> 16);
    const output = operation.outputs[index]!;
    const outputType = semanticExpressionValueType(output);
    const value = outputType === "uint" || outputType === "int" ? float32ToUintBits(c + product) : c + product;
    assignSemanticInlineOutput(output, value, context);
  }
}

function assignSemanticInlineOutput(
  target: SemanticExpression,
  value: number,
  context: SemanticReferenceContext,
): void {
  evalSemanticExpression({
    kind: "assignment",
    operator: "=",
    target,
    value: { kind: "literal", literalKind: "number", value, ...("valueType" in target && target.valueType ? { valueType: target.valueType } : {}), span: target.span },
    ...(semanticExpressionValueType(target) === undefined ? {} : { valueType: semanticExpressionValueType(target)! }),
    span: target.span,
  }, context);
}

function execSemanticScopedOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: SemanticReferenceContext,
): SemanticControl {
  const savedLocals = new Map<string, SemanticValue | undefined>();
  const savedDimensions = new Map<string, readonly number[] | undefined>();
  for (const operation of operations) {
    if (operation.kind !== "declare" || savedLocals.has(operation.target.name)) continue;
    savedLocals.set(operation.target.name, context.locals.get(operation.target.name));
    savedDimensions.set(operation.target.name, context.localDimensions.get(operation.target.name));
  }
  try {
    return execSemanticOperations(operations, context);
  } finally {
    for (const [name, value] of savedLocals) {
      if (value === undefined) context.locals.delete(name);
      else context.locals.set(name, value);
    }
    for (const [name, dimensions] of savedDimensions) {
      if (dimensions === undefined) context.localDimensions.delete(name);
      else context.localDimensions.set(name, dimensions);
    }
  }
}

function* execSemanticBarrierOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: SemanticReferenceContext,
  barrierFunctions: ReadonlySet<string>,
): SemanticBarrierGenerator {
  for (const operation of operations) {
    switch (operation.kind) {
      case "dim3-declare":
      case "cooperative-group-declare":
        break;
      case "declare":
        if (operation.target.addressSpace !== "shared") {
          if (operation.target.dimensions.length > 0) context.localDimensions.set(operation.target.name, operation.target.dimensions);
          context.locals.set(operation.target.name, semanticDeclareValue(operation, context));
        }
        break;
      case "store":
      case "copy":
      case "copy-fence":
      case "surface-write":
      case "surface-read-store":
      case "atomic":
      case "expression":
      case "fence":
        execSemanticOperations([operation], context);
        break;
      case "call":
        if (barrierFunctions.has(operation.callee)) {
          yield* execSemanticBarrierFunctionCall(operation, context, barrierFunctions);
        } else {
          execSemanticOperations([operation], context);
        }
        break;
      case "branch": {
        const control = yield* execSemanticBarrierScopedOperations(
          truthy(evalNumber(operation.condition, context)) ? operation.consequent : operation.alternate,
          context,
          barrierFunctions,
        );
        if (control !== "fallthrough") return control;
        break;
      }
      case "block": {
        const control = yield* execSemanticBarrierScopedOperations(operation.body, context, barrierFunctions);
        if (control !== "fallthrough") return control;
        break;
      }
      case "loop": {
        const control = yield* execSemanticBarrierLoop(operation, context, barrierFunctions);
        if (control !== "fallthrough") return control;
        break;
      }
      case "return":
        if (operation.value) context.returnValue = evalSemanticExpression(operation.value, context);
        return "return";
      case "barrier":
        yield operation.scope;
        break;
      case "inline-asm":
        if (operation.op?.kind !== "bar-sync") {
          execSemanticOperations([operation], context);
          break;
        }
        yield "workgroup";
        break;
      case "break":
        return "break";
      case "continue":
        return "continue";
      default:
        throw semanticReferenceError(`semantic reference does not support ${operation.kind}`, operation.span);
    }
  }
  return "fallthrough";
}

function* execSemanticBarrierScopedOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: SemanticReferenceContext,
  barrierFunctions: ReadonlySet<string>,
): SemanticBarrierGenerator {
  const savedLocals = new Map<string, SemanticValue | undefined>();
  const savedDimensions = new Map<string, readonly number[] | undefined>();
  for (const operation of operations) {
    if (operation.kind !== "declare" || savedLocals.has(operation.target.name)) continue;
    savedLocals.set(operation.target.name, context.locals.get(operation.target.name));
    savedDimensions.set(operation.target.name, context.localDimensions.get(operation.target.name));
  }
  try {
    return yield* execSemanticBarrierOperations(operations, context, barrierFunctions);
  } finally {
    for (const [name, value] of savedLocals) {
      if (value === undefined) context.locals.delete(name);
      else context.locals.set(name, value);
    }
    for (const [name, dimensions] of savedDimensions) {
      if (dimensions === undefined) context.localDimensions.delete(name);
      else context.localDimensions.set(name, dimensions);
    }
  }
}

function* execSemanticBarrierFunctionCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
  barrierFunctions: ReadonlySet<string>,
): SemanticBarrierGenerator {
  const fn = context.compiled.kernelIr.functions.find((item) => item.name === operation.callee);
  if (!fn) throw semanticReferenceError(`semantic reference unknown barrier function '${operation.callee}'`, operation.span);
  if ((operation.result === undefined) !== (fn.returnType === "void")) throw semanticReferenceError(`semantic reference barrier call '${operation.callee}' result contract mismatch`, operation.span);
  const child = createSemanticFunctionContext(fn, operation.args, context, operation.span);
  const control = yield* execSemanticBarrierOperations(fn.body, child, barrierFunctions);
  if (control === "break" || control === "continue") {
    throw semanticReferenceError(`semantic reference function '${fn.name}' leaked ${control} across call boundary`, fn.span);
  }
  assignSemanticCallResult(operation, fn, child, context);
  return "fallthrough";
}

function* execSemanticBarrierLoop(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>,
  context: SemanticReferenceContext,
  barrierFunctions: ReadonlySet<string>,
): SemanticBarrierGenerator {
  if (operation.loopKind === "for" && operation.init) execSemanticLoopInit(operation.init, context);
  for (let guard = 0; ; guard++) {
    if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
    if (operation.loopKind !== "do-while" && operation.condition !== undefined && !truthy(evalNumber(operation.condition, context))) return "fallthrough";
    const control = yield* execSemanticBarrierScopedOperations(operation.body, context, barrierFunctions);
    if (control === "return") return control;
    if (control === "break") return "fallthrough";
    if (operation.loopKind === "for" && operation.update) evalNumber(operation.update, context);
    if (operation.loopKind === "do-while" && (!operation.condition || !truthy(evalNumber(operation.condition, context)))) return "fallthrough";
  }
}

function execSemanticSurfaceReadStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-read-store" }>,
  context: SemanticReferenceContext,
): void {
  const target = semanticReferenceSurfaceReadTarget(operation.target);
  if (!target) throw semanticReferenceError("semantic reference supports only local scalar/vector surf2Dread targets", operation.span);
  const value = evalSemanticSurfaceRead(
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
    context,
  );
  context.locals.set(target.name, value);
}

function semanticReferenceSurfaceReadTarget(expression: SemanticExpression): { readonly name: string; readonly valueType?: CudaLiteScalarType } | undefined {
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

function execSemanticSurfaceWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-write" }>,
  context: SemanticReferenceContext,
): void {
  if (!semanticReferenceSurfaceWriteSupported(operation, context.compiled) || operation.surface.kind !== "symbol") {
    throw semanticReferenceError("semantic reference supports only direct scalar surf2Dwrite", operation.span);
  }
  const surface = context.surfaces[operation.surface.name];
  if (!surface) throw semanticReferenceError(`missing surface input '${operation.surface.name}'`, operation.surface.span);
  const surfaceName = operation.surface.name;
  const xBytes = Math.trunc(evalNumber(operation.xBytes, context));
  const aligned = xBytes % 4 === 0;
  const x = Math.trunc(xBytes / 4);
  const y = Math.trunc(evalNumber(operation.y, context));
  const z = operation.z ? Math.trunc(evalNumber(operation.z, context)) : 0;
  const value = evalSemanticExpression(operation.value, context);
  if (Array.isArray(value)) {
    value.forEach((laneValue, lane) => {
      writeSemanticSurfaceLane(surface, surfaceName, xBytes + lane * 4, y, z, laneValue, context);
    });
    return;
  }
  if (typeof value !== "number") throw semanticReferenceError("semantic surface write value is not scalar/vector", operation.value.span);
  writeSemanticSurfaceLane(surface, surfaceName, aligned ? x * 4 : xBytes, y, z, value, context);
}

function writeSemanticSurfaceLane(
  surface: WgslTexture2DInput,
  surfaceName: string,
  xBytes: number,
  y: number,
  z: number,
  value: number,
  context: SemanticReferenceContext,
): void {
  const aligned = xBytes % 4 === 0;
  const x = Math.trunc(xBytes / 4);
  const index = ((z * surface.height) + y) * surface.width + x;
  const ok = aligned && xBytes >= 0 && x >= 0 && y >= 0 && z >= 0 && x < surface.width && y < surface.height && index >= 0 && index < surface.data.length;
  if (ok) surface.data[index] = value;
  context.trace.writes.push({ name: surfaceName, index, value, ok });
}

function storeValueExpression(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (semanticReferenceVectorFieldMemoryRefSupported(operation.target)) {
    const right = isSemanticFloatVectorType(operation.target.valueType)
      ? evalSemanticExpression(operation.value, context)
      : evalNumber(operation.value, context);
    if (operation.operator === "=") return right;
    const left = readMemoryValue(operation.target, context);
    return evalVectorFieldAssignment(operation.operator, left, right, operation.span);
  }
  if (semanticStorageVectorType(operation.target.valueType) !== undefined) {
    const right = evalSemanticExpression(operation.value, context);
    if (operation.operator === "=") return right;
    const binaryOperator = semanticAssignmentBinaryOperator(operation.operator);
    if (binaryOperator === undefined) throw semanticReferenceError(`semantic reference does not support assignment '${operation.operator}'`, operation.span);
    return evalVectorBinary(binaryOperator, readMemoryValue(operation.target, context), right, operation.span);
  }
  const right = evalNumber(operation.value, context);
  if (operation.operator === "=") return right;
  return applySemanticScalarAssignment(operation.operator, readMemory(operation.target, context), right, operation.span);
}

function evalVectorFieldAssignment(operator: string, left: SemanticValue, right: SemanticValue, span: SourceSpan): SemanticValue {
  const binaryOperator = semanticAssignmentBinaryOperator(operator);
  if (binaryOperator !== undefined) return Array.isArray(left) || Array.isArray(right)
    ? evalVectorBinary(binaryOperator, left, right, span)
    : evalBinary(binaryOperator, Number(left), Number(right));
  throw semanticReferenceError(`semantic reference does not support assignment '${operator}'`, span);
}

function semanticDeclareValue(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (operation.target.dimensions.length > 0) {
    const vectorTarget = isSemanticFloatVectorType(operation.target.valueType);
    const zeroValue = vectorTarget && operation.target.valueType !== undefined ? zeroSemanticVector(operation.target.valueType) : 0;
    const values: SemanticValue[] = Array.from({ length: totalElements(operation.target.dimensions) }, () =>
      Array.isArray(zeroValue) ? [...zeroValue] : zeroValue
    );
    if (operation.init?.kind === "initializer") {
      for (const [index, expression] of flattenInitializerExpressions(operation.init).entries()) {
        if (index >= values.length) break;
        values[index] = vectorTarget ? evalSemanticExpression(expression, context) : evalNumber(expression, context);
      }
    } else if (operation.init) {
      const fillValue = vectorTarget ? evalSemanticExpression(operation.init, context) : evalNumber(operation.init, context);
      for (let index = 0; index < values.length; index++) values[index] = Array.isArray(fillValue) ? [...fillValue] : fillValue;
    }
    return values as number[];
  }
  if (operation.init) {
    const value = evalSemanticExpression(operation.init, context);
    return typeof value === "number" ? coerceSemanticScalarValue(value, operation.target.valueType) : value;
  }
  if (isSemanticFloatVectorType(operation.target.valueType)) {
    return Array.from({ length: cudaVectorLaneCount(operation.target.valueType) }, () => 0);
  }
  return 0;
}

function zeroSemanticVector(valueType: CudaLiteScalarType): number[] {
  return Array.from({ length: cudaVectorLaneCount(valueType) }, () => 0);
}

function execSemanticAtomic(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  context: SemanticReferenceContext,
): void {
  const atomicOp = semanticAtomicOperation(operation.callee);
  if (!operation.target || !atomicOp) {
    throw semanticReferenceError(`semantic reference does not support atomic '${operation.callee}'`, operation.span);
  }
  const value = operation.args[1];
  if (!value) throw semanticReferenceError(`semantic reference atomic '${operation.callee}' missing operand`, operation.span);
  const oldValue = readAtomicMemory(operation.target, context);
  const nextValue = semanticAtomicValue(atomicOp, oldValue, evalNumber(value, context), operation, context);
  writeAtomicMemory(operation.target, nextValue, context);
}

function execSemanticCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): void {
  if (operation.result !== undefined) {
    const fn = context.compiled.kernelIr.functions.find((item) => item.name === operation.callee);
    if (!fn || fn.returnType === "void") throw semanticReferenceError(`semantic reference call '${operation.callee}' cannot produce a result`, operation.span);
    const child = runSemanticFunction(fn, operation.args, context, operation.span);
    assignSemanticCallResult(operation, fn, child, context);
    return;
  }
  if (operation.callee === "assert") {
    if (operation.args[0]) evalNumber(operation.args[0], context);
    return;
  }
  if (operation.callee === "printf") return;
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) {
    for (const arg of operation.args) evalSemanticExpression(arg, context);
    return;
  }
  if (operation.callee === "curand_init") {
    execSemanticCurandInit(operation, context);
    return;
  }
  if (operation.callee === "skipahead") {
    evalSemanticExpression({
      kind: "call",
      callee: { kind: "symbol", name: operation.callee, addressSpace: "builtin", span: operation.span },
      args: operation.args,
      valueType: "uint",
      span: operation.span,
    }, context);
    return;
  }
  if (semanticReferenceVoidFunctionCallSupported(operation, context.compiled)) {
    execSemanticVoidFunctionCall(operation, context);
    return;
  }
  const fn = context.compiled.kernelIr.functions.find((item) => item.name === operation.callee);
  if (fn?.returnType === "void" && operation.args.length === fn.params.length) {
    runSemanticFunction(fn, operation.args, context, operation.span);
    return;
  }
  if (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) {
    execSemanticLocalArrayFill(operation, context);
    return;
  }
  throw semanticReferenceError(`semantic reference does not support call '${operation.callee}'`, operation.span);
}

function assignSemanticCallResult(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number],
  child: SemanticReferenceContext,
  context: SemanticReferenceContext,
): void {
  if (operation.result === undefined) return;
  if (child.returnValue === undefined) throw semanticReferenceError(`semantic reference function '${fn.name}' did not return value`, fn.span);
  if (!Array.isArray(child.returnValue) && typeof child.returnValue !== "number") {
    throw semanticReferenceError(`semantic reference function '${fn.name}' returned a non-scalar value`, fn.span);
  }
  const value = Array.isArray(child.returnValue)
    ? child.returnValue
    : coerceSemanticScalarValue(child.returnValue, operation.result.valueType);
  context.locals.set(operation.result.name, value);
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

function execSemanticVoidFunctionCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): void {
  const fn = context.compiled.kernelIr.functions.find((item) => item.name === operation.callee);
  if (!fn) throw semanticReferenceError(`semantic reference unknown function '${operation.callee}'`, operation.span);
  runSemanticFunction(fn, operation.args, context, operation.span);
}

function execSemanticLocalArrayFill(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): void {
  const [target, valueExpression] = operation.args;
  if (target?.kind !== "symbol" || target.addressSpace !== "local" || valueExpression === undefined) {
    throw semanticReferenceError(`${operation.callee} expects local array and fill value`, operation.span);
  }
  const local = context.locals.get(target.name);
  if (!Array.isArray(local)) throw semanticReferenceError(`${operation.callee} expects fixed local array '${target.name}'`, target.span);
  const symbol = context.compiled.kernelIr.memory.find((item) => item.kind === "local" && item.name === target.name);
  const localValues = local as SemanticValue[];
  const value = isSemanticFloatVectorType(symbol?.valueType)
    ? evalSemanticExpression(valueExpression, context)
    : evalNumber(valueExpression, context);
  for (let index = 0; index < localValues.length; index++) {
    localValues[index] = Array.isArray(value) ? [...value] : value;
  }
}

function execSemanticCurandInit(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): void {
  const state = semanticCurandState(operation.args[3]);
  if (!state) throw semanticReferenceError("curand_init expects state address", operation.span);
  const seed = evalNumber(operation.args[0]!, context) >>> 0;
  const sequence = evalNumber(operation.args[1]!, context) >>> 0;
  const offset = evalNumber(operation.args[2]!, context) >>> 0;
  semanticCurandStateWrite(state, curandNext((seed ^ Math.imul(sequence, 747796405) ^ offset ^ 2891336453) >>> 0), context);
}

function evalSemanticCurandCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic cuRAND call requires symbol callee", expression.span);
  if (expression.callee.name === "skipahead") {
    const state = semanticCurandState(expression.args[1]);
    if (!state) throw semanticReferenceError("skipahead expects state address", expression.span);
    const count = evalNumber(expression.args[0]!, context) >>> 0;
    semanticCurandStateWrite(state, curandAdvance(semanticCurandStateRead(state, context) >>> 0, count), context);
    return 0;
  }
  const state = semanticCurandState(expression.args[0]);
  if (!state) throw semanticReferenceError(`${expression.callee.name} expects state address`, expression.span);
  if (expression.callee.name === "curand_uniform" || expression.callee.name === "curand_uniform_double") {
    const next = curandNext(semanticCurandStateRead(state, context) >>> 0);
    semanticCurandStateWrite(state, next, context);
    return (next + 1) * 2.3283064365386963e-10;
  }
  if (expression.callee.name === "curand_uniform4") {
    let current = semanticCurandStateRead(state, context) >>> 0;
    const lanes: number[] = [];
    for (let lane = 0; lane < 4; lane++) {
      current = curandNext(current);
      lanes.push((current + 1) * 2.3283064365386963e-10);
    }
    semanticCurandStateWrite(state, current, context);
    return lanes;
  }
  if (expression.callee.name === "curand") {
    const next = curandNext(semanticCurandStateRead(state, context) >>> 0);
    semanticCurandStateWrite(state, next, context);
    return next;
  }
  if (expression.callee.name === "curand_normal" || expression.callee.name === "curand_normal_double") {
    const first = curandNext(semanticCurandStateRead(state, context) >>> 0);
    const second = curandNext(first);
    semanticCurandStateWrite(state, second, context);
    const u1 = Math.max((first + 1) * 2.3283064365386963e-10, 1.1754943508222875e-38);
    const u2 = (second + 1) * 2.3283064365386963e-10;
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  if (expression.callee.name === "curand_normal2") {
    const first = curandNext(semanticCurandStateRead(state, context) >>> 0);
    const second = curandNext(first);
    semanticCurandStateWrite(state, second, context);
    const u1 = Math.max((first + 1) * 2.3283064365386963e-10, 1.1754943508222875e-38);
    const u2 = (second + 1) * 2.3283064365386963e-10;
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }
  if (expression.callee.name === "curand_normal4") {
    const first = curandNext(semanticCurandStateRead(state, context) >>> 0);
    const second = curandNext(first);
    const third = curandNext(second);
    const fourth = curandNext(third);
    semanticCurandStateWrite(state, fourth, context);
    const a = curandNormalPair(first, second);
    const b = curandNormalPair(third, fourth);
    return [...a, ...b];
  }
  if (expression.callee.name === "curand_log_normal" || expression.callee.name === "curand_log_normal_double") {
    const first = curandNext(semanticCurandStateRead(state, context) >>> 0);
    const second = curandNext(first);
    semanticCurandStateWrite(state, second, context);
    const u1 = Math.max((first + 1) * 2.3283064365386963e-10, 1.1754943508222875e-38);
    const u2 = (second + 1) * 2.3283064365386963e-10;
    const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.exp(evalNumber(expression.args[1]!, context) + evalNumber(expression.args[2]!, context) * normal);
  }
  if (expression.callee.name === "curand_log_normal2") {
    const first = curandNext(semanticCurandStateRead(state, context) >>> 0);
    const second = curandNext(first);
    semanticCurandStateWrite(state, second, context);
    const u1 = Math.max((first + 1) * 2.3283064365386963e-10, 1.1754943508222875e-38);
    const u2 = (second + 1) * 2.3283064365386963e-10;
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    const mean = evalNumber(expression.args[1]!, context);
    const stddev = evalNumber(expression.args[2]!, context);
    return [
      Math.exp(mean + stddev * radius * Math.cos(angle)),
      Math.exp(mean + stddev * radius * Math.sin(angle)),
    ];
  }
  if (expression.callee.name === "curand_log_normal4") {
    const first = curandNext(semanticCurandStateRead(state, context) >>> 0);
    const second = curandNext(first);
    const third = curandNext(second);
    const fourth = curandNext(third);
    semanticCurandStateWrite(state, fourth, context);
    const mean = evalNumber(expression.args[1]!, context);
    const stddev = evalNumber(expression.args[2]!, context);
    return [...curandNormalPair(first, second), ...curandNormalPair(third, fourth)]
      .map((normal) => Math.exp(mean + stddev * normal));
  }
  if (expression.callee.name === "curand_poisson" || expression.callee.name === "curand_poisson4") {
    const start = semanticCurandStateRead(state, context) >>> 0;
    const lambda = Math.fround(Math.max(0, evalNumber(expression.args[1]!, context)));
    if (expression.callee.name === "curand_poisson") {
      const [value, current] = curandPoissonDraw(start, lambda);
      semanticCurandStateWrite(state, current, context);
      return value;
    }
    let current = start;
    const values: number[] = [];
    for (let lane = 0; lane < 4; lane++) {
      const [value, next] = curandPoissonDraw(current, lambda);
      values.push(value);
      current = next;
    }
    semanticCurandStateWrite(state, current, context);
    return values;
  }
  throw semanticReferenceError(`semantic reference does not support cuRAND call '${expression.callee.name}'`, expression.span);
}

function evalSemanticGeneratedRandomCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (!semanticReferenceGeneratedRandomCallSupported(expression) || expression.callee.kind !== "symbol") {
    throw semanticReferenceError("generated random helper expects local uint state", expression.span);
  }
  const address = expression.args[0]!;
  if (address.kind !== "unary" || address.argument.kind !== "symbol") {
    throw semanticReferenceError("generated random helper expects local uint state", expression.span);
  }
  const stateName = address.argument.name;
  const uniform = (): number => {
    const current = Number(context.locals.get(stateName) ?? 0) >>> 0;
    const next = (Math.imul(current, 1664525) + 1013904223) >>> 0;
    context.locals.set(stateName, next);
    return Math.fround((next & 0x00ffffff) / 16777216);
  };
  if (expression.callee.name === "bg_random_uniform") return uniform();
  if (expression.callee.name === "bg_random_normal") {
    let sum = uniform();
    for (let draw = 1; draw < 6; draw++) sum = Math.fround(sum + uniform());
    return Math.fround(sum - 3);
  }
  return Math.trunc(Math.fround(uniform() * 8));
}

function semanticCurandState(expression: SemanticExpression | undefined): SemanticCurandState | undefined {
  if (!expression) return undefined;
  if (expression.kind === "unary" && expression.operator === "&") {
    if (expression.argument.kind === "symbol" && expression.argument.addressSpace === "local") {
      return { kind: "local", name: expression.argument.name, span: expression.argument.span };
    }
    if (expression.argument.kind === "index") {
      const ref = memoryRefFromIndexExpression(expression.argument);
      return ref && (ref.addressSpace === "storage" || ref.addressSpace === "device-global" || ref.addressSpace === "shared" || ref.addressSpace === "local")
        ? { kind: "memory", ref: { ...ref, valueType: "uint" } }
        : undefined;
    }
  }
  return undefined;
}

function semanticCurandStateRead(state: SemanticCurandState, context: SemanticReferenceContext): number {
  return state.kind === "local"
    ? Number(context.locals.get(state.name) ?? 0)
    : readMemory(state.ref, context);
}

function semanticCurandStateWrite(state: SemanticCurandState, value: number, context: SemanticReferenceContext): void {
  if (state.kind === "local") {
    context.locals.set(state.name, value >>> 0);
    return;
  }
  writeMemory(state.ref, value >>> 0, context);
}

function semanticAtomicValue(
  atomicOp: SemanticAtomicOp,
  oldValue: number,
  value: number,
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }> | Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  switch (atomicOp) {
    case "add": return semanticAtomicTargetType(operation) === "bf16" ? roundFloat32ToBfloat16(oldValue + value, "rn") : oldValue + value;
    case "sub": return oldValue - value;
    case "min": return Math.min(oldValue, value);
    case "max": return Math.max(oldValue, value);
    case "and": return Math.trunc(oldValue) & Math.trunc(value);
    case "or": return Math.trunc(oldValue) | Math.trunc(value);
    case "xor": return Math.trunc(oldValue) ^ Math.trunc(value);
    case "exchange": return value;
    case "inc": return oldValue >= value ? 0 : oldValue + 1;
    case "dec": return oldValue === 0 || oldValue > value ? value : oldValue - 1;
    case "cas": {
      const replacement = operation.args[2];
      const callee = operation.kind === "atomic"
        ? operation.callee
        : operation.callee.kind === "symbol" ? operation.callee.name : "<expr>";
      if (!replacement) throw semanticReferenceError(`semantic reference atomic '${callee}' missing replacement`, operation.span);
      const targetType = operation.kind === "atomic"
        ? operation.target?.valueType
        : semanticAtomicCallTarget(operation)?.valueType;
      const matches = targetType === "float" || targetType === "double"
        ? float32ToUintBits(oldValue) === float32ToUintBits(value)
        : oldValue === value;
      return matches ? evalNumber(replacement, context) : oldValue;
    }
  }
}

function semanticAtomicTargetType(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }> | Extract<SemanticExpression, { readonly kind: "call" }>,
): CudaLiteScalarType | undefined {
  return operation.kind === "atomic"
    ? operation.target?.valueType
    : semanticAtomicCallTarget(operation)?.valueType;
}

function execSemanticLoop(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>,
  context: SemanticReferenceContext,
): SemanticControl {
  if (operation.loopKind === "for") {
    if (operation.init) execSemanticLoopInit(operation.init, context);
    for (let guard = 0; operation.condition === undefined || truthy(evalNumber(operation.condition, context)); guard++) {
      if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
      const control = execSemanticScopedOperations(operation.body, context);
      if (control === "return") return control;
      if (control === "break") return "fallthrough";
      if (operation.update) evalNumber(operation.update, context);
    }
    return "fallthrough";
  }
  if (operation.loopKind === "while") {
    for (let guard = 0; operation.condition === undefined || truthy(evalNumber(operation.condition, context)); guard++) {
      if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
      const control = execSemanticScopedOperations(operation.body, context);
      if (control === "return") return control;
      if (control === "break") return "fallthrough";
    }
    return "fallthrough";
  }
  for (let guard = 0; ; guard++) {
    if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
    const control = execSemanticScopedOperations(operation.body, context);
    if (control === "return") return control;
    if (control === "break") return "fallthrough";
    if (!operation.condition || !truthy(evalNumber(operation.condition, context))) return "fallthrough";
  }
}

function execSemanticLoopInit(
  init: SemanticKernelIrOperation | SemanticExpression,
  context: SemanticReferenceContext,
): void {
  if (isSemanticKernelIrOperation(init)) execSemanticOperations([init], context);
  else evalNumber(init, context);
}

function evalNumber(expression: SemanticExpression, context: SemanticReferenceContext): number {
  const value = evalSemanticExpression(expression, context);
  if (typeof value !== "number") throw semanticReferenceError("semantic expression is not scalar", expression.span);
  return value;
}

function semanticReferenceActiveMask(context: SemanticReferenceContext): number {
  if (context.compiled.subgroupMode === "scalar") return 1;
  const rank = semanticLocalLinearRank(context);
  const warpBase = Math.floor(rank / 32) * 32;
  let mask = 0;
  for (const peer of semanticWarpContexts(context)) {
    const peerRank = semanticLocalLinearRank(peer);
    if (peerRank < warpBase || peerRank >= warpBase + 32) continue;
    mask |= 1 << (peerRank - warpBase);
  }
  return mask >>> 0;
}

function evalSemanticSubgroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic subgroup call requires symbol callee", expression.span);
  const name = expression.callee.name;
  if (name === "__activemask") return semanticReferenceActiveMask(context);
  const value = expression.args[isCudaWarpReduceCallName(name) ? expression.args.length - 1 : legacyVoteCall(name) || legacyShuffleCall(name) ? 0 : 1];
  if (!value) throw semanticReferenceError(`${name} expects value operand`, expression.span);
  const voteOp = cudaVoteOpForCall(name);
  if (context.compiled.subgroupMode === "scalar") {
    const scalar = evalNumber(value, context);
    if (voteOp !== undefined) return truthy(scalar) ? 1 : 0;
    return scalar;
  }
  const mask = expression.args.length === 2 &&
    (isCudaWarpReduceCallName(name) || voteOp !== undefined || cudaArithmeticReduceOpForCall(name) !== undefined)
    ? evalNumber(expression.args[0]!, context) >>> 0
    : undefined;
  const peers = semanticWarpContexts(context).filter((peer) =>
    mask === undefined || (mask & (1 << (semanticLocalLinearRank(peer) % 32))) !== 0
  );
  if (voteOp === "any") return peers.some((peer) => truthy(evalNumber(value, peer))) ? 1 : 0;
  if (voteOp === "all") return peers.every((peer) => truthy(evalNumber(value, peer))) ? 1 : 0;
  if (voteOp === "ballot") {
    const activeMask = legacyVoteCall(name) ? 0xffffffff : evalNumber(expression.args[0]!, context) >>> 0;
    let mask = 0;
    for (const peer of peers) {
      const lane = semanticLocalLinearRank(peer) % 32;
      if ((activeMask & (1 << lane)) === 0) continue;
      if (!truthy(evalNumber(value, peer))) continue;
      mask |= 1 << lane;
    }
    return mask >>> 0;
  }
  if (voteOp === "match-any") {
    const current = evalNumber(value, context);
    let mask = 0;
    for (const peer of peers) {
      if (evalNumber(value, peer) !== current) continue;
      mask |= 1 << (semanticLocalLinearRank(peer) % 32);
    }
    return mask >>> 0;
  }
  const arithmeticReduceOp = cudaArithmeticReduceOpForCall(name);
  if (arithmeticReduceOp === "add") return peers.reduce((sum, peer) => sum + evalNumber(value, peer), 0);
  if (arithmeticReduceOp === "min") return Math.min(...peers.map((peer) => evalNumber(value, peer)));
  if (arithmeticReduceOp === "max") return Math.max(...peers.map((peer) => evalNumber(value, peer)));
  const bitwiseReduceOp = cudaBitwiseReduceOpForCall(name);
  if (bitwiseReduceOp === "and") return peers.reduce((acc, peer) => acc & (evalNumber(value, peer) | 0), -1) >>> 0;
  if (bitwiseReduceOp === "or") return peers.reduce((acc, peer) => acc | (evalNumber(value, peer) | 0), 0) >>> 0;
  if (bitwiseReduceOp === "xor") return peers.reduce((acc, peer) => acc ^ (evalNumber(value, peer) | 0), 0) >>> 0;
  if (isCudaShuffleCallName(name)) {
    const indexArg = legacyShuffleCall(name) ? expression.args[1] : expression.args[2];
    const widthArg = legacyShuffleCall(name) ? expression.args[2] : expression.args[3];
    const index = indexArg ? Math.trunc(evalNumber(indexArg, context)) : 0;
    const width = semanticShuffleWidth(widthArg ? evalNumber(widthArg, context) : 32);
    const rank = semanticLocalLinearRank(context);
    const lane = rank % width;
    const base = rank - lane;
    const targetLane = semanticShuffleTargetLane(name, lane, index, width);
    const targetRank = base + targetLane;
    const target = peers.find((peer) => semanticLocalLinearRank(peer) === targetRank) ?? context;
    return evalNumber(value, target);
  }
  throw semanticReferenceError(`semantic reference does not support subgroup call '${name}'`, expression.span);
}

function semanticWarpContexts(context: SemanticReferenceContext): readonly SemanticReferenceContext[] {
  const rank = semanticLocalLinearRank(context);
  const warpBase = Math.floor(rank / 32) * 32;
  const warpEnd = Math.min(warpBase + 32, semanticBlockSize(context));
  return (context.activeCollectiveContexts ?? context.blockContexts).filter((peer) => {
    const peerRank = semanticLocalLinearRank(peer);
    return peerRank >= warpBase && peerRank < warpEnd;
  });
}

function semanticShuffleWidth(width: number): number {
  if (!Number.isFinite(width)) return 32;
  return Math.max(1, Math.min(32, Math.trunc(width)));
}

function semanticShuffleTargetLane(name: string, lane: number, index: number, width: number): number {
  const operand = Math.max(0, Math.trunc(index));
  const op = cudaShuffleOpForCall(name);
  if (op === "sync") return operand % width;
  if (op === "down") return lane + operand < width ? lane + operand : lane;
  if (op === "up") return lane >= operand ? lane - operand : lane;
  const xorLane = lane ^ operand;
  return xorLane < width ? xorLane : lane;
}

function semanticLocalLinearRank(context: SemanticReferenceContext): number {
  return context.threadIdx.x +
    context.threadIdx.y * context.blockDim.x +
    context.threadIdx.z * context.blockDim.x * context.blockDim.y;
}

function semanticBlockSize(context: SemanticReferenceContext): number {
  return context.blockDim.x * context.blockDim.y * context.blockDim.z;
}

function evalSemanticAddressPredicateCall(expression: Extract<SemanticExpression, { readonly kind: "call" }>): number {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic address predicate requires symbol callee", expression.span);
  const addressSpace = semanticAddressPredicateAddressSpace(expression.args[0]);
  if (!addressSpace) return 0;
  const kind = cudaAddressSpacePredicateKind(expression.callee.name);
  if (kind === "global") return addressSpace === "storage" || addressSpace === "device-global" ? 1 : 0;
  if (kind !== undefined) return addressSpace === kind ? 1 : 0;
  throw semanticReferenceError(`semantic reference does not support address predicate '${expression.callee.name}'`, expression.span);
}

function evalSemanticExpression(expression: SemanticExpression, context: SemanticReferenceContext): SemanticValue {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number" ? expression.value : 0;
    case "symbol":
      return symbolValue(expression.name, context, expression.span);
    case "member":
      return memberValue(evalSemanticExpression(expression.object, context), semanticExpressionValueType(expression.object), expression.property, expression.span);
    case "index":
      return readIndexExpression(expression, context);
    case "cast":
      return castNumber(evalNumber(expression.expression, context), expression.valueType);
    case "unary":
      if (semanticReferenceBf162LocalBitsCastSupported(expression, context.compiled)) return evalSemanticBf162LocalBitsCast(expression, context);
      return evalUnary(expression.operator, evalNumber(expression.argument, context));
    case "binary": {
      if (isStoragePointerIdentityComparison(expression, context.compiled)) {
        const left = semanticReferenceStoragePointerIdentity(expression.left, context);
        const right = semanticReferenceStoragePointerIdentity(expression.right, context);
        const equal = left.buffer === right.buffer && left.base === right.base;
        return expression.operator === "==" ? Number(equal) : Number(!equal);
      }
      const left = evalSemanticExpression(expression.left, context);
      const right = evalSemanticExpression(expression.right, context);
      if (Array.isArray(left) || Array.isArray(right)) {
        return evalVectorBinary(expression.operator, left, right, expression.span);
      }
      if (typeof left !== "number" || typeof right !== "number") {
        throw semanticReferenceError("semantic reference scalar binary requires scalar operands", expression.span);
      }
      return evalBinary(expression.operator, left, right);
    }
    case "conditional":
      return truthy(evalNumber(expression.condition, context))
        ? evalSemanticExpression(expression.consequent, context)
        : evalSemanticExpression(expression.alternate, context);
    case "assignment":
      {
        const vectorTarget = isSemanticFloatVectorType(semanticExpressionValueType(expression.target));
        if (!semanticReferenceAssignmentOperatorSupported(expression.operator) ||
        (vectorTarget && !semanticVectorAssignmentOperatorSupported(expression.operator)) ||
        (expression.target.kind !== "symbol" &&
          (expression.target.kind !== "member" || !semanticReferenceVectorMemberSupported(expression.target, context.compiled)) &&
          !semanticReferenceAssignmentMemoryRefSupported(expression.target, context.compiled))) {
          throw semanticReferenceError("semantic reference supports only modeled local assignment expressions", expression.span);
        }
      if (expression.target.kind === "member" && semanticReferenceVectorMemberSupported(expression.target, context.compiled)) return assignLocalVectorMember(expression, context);
      {
        const ref = semanticReferenceAssignmentMemoryRef(expression.target, context.compiled);
        if (ref) return assignMemoryRef(expression, ref, context);
      }
      if (expression.target.kind !== "symbol") throw semanticReferenceError("semantic reference assignment requires local symbol target", expression.target.span);
      {
        if (vectorTarget) {
          const right = evalSemanticExpression(expression.value, context);
          const left = evalSemanticExpression(expression.target, context);
          const value = expression.operator === "=" ? right : evalVectorBinary(expression.operator.slice(0, -1), left, right, expression.span);
          context.locals.set(expression.target.name, value);
          return value;
        }
        const right = evalNumber(expression.value, context);
        const value = applySemanticScalarAssignment(expression.operator, evalNumber(expression.target, context), right, expression.span);
        const assigned = coerceSemanticScalarValue(value, expression.target.valueType);
        context.locals.set(expression.target.name, assigned);
        return assigned;
      }
      }
    case "sequence": {
      let value = 0;
      for (const item of expression.expressions) value = evalNumber(item, context);
      return value;
    }
    case "call":
      if (semanticReferenceSharedAddressCallSupported(expression)) return evalSemanticSharedAddressCall(expression, context);
      if (semanticReferenceCooperativeGroupCallSupported(expression, context.compiled)) return evalSemanticCooperativeGroupCall(expression, context);
      if (semanticReferenceCooperativeReduceCallSupported(expression, context.compiled)) return evalSemanticCooperativeReduceCall(expression, context);
      if (semanticReferenceAtomicCallSupported(expression, context.compiled)) return evalSemanticAtomicCall(expression, context);
      if (semanticReferenceCurandCallSupported(expression, context.compiled)) return evalSemanticCurandCall(expression, context);
      if (semanticReferenceGeneratedRandomCallSupported(expression)) return evalSemanticGeneratedRandomCall(expression, context);
      if (semanticReferenceSubgroupCallSupported(expression)) return evalSemanticSubgroupCall(expression, context);
      if (semanticReferenceAddressPredicateCallSupported(expression)) return evalSemanticAddressPredicateCall(expression);
      if (semanticReferenceVectorConstructorSupported(expression, "any", context.compiled)) return evalSemanticVectorConstructor(expression, context);
      if (semanticReferenceVectorAtCallSupported(expression, context.compiled)) return evalSemanticVectorAtCall(expression, context);
      if (semanticReferenceVectorLerpCallSupported(expression, context.compiled)) return evalSemanticVectorLerpCall(expression, context);
      if (semanticReferenceVectorMathCallSupported(expression)) return evalSemanticVectorMathCall(expression, context);
      if (semanticReferenceHalf2CallSupported(expression, context.compiled)) return evalSemanticHalf2Call(expression, context);
      if (semanticReferenceBf162CallSupported(expression, context.compiled)) return evalSemanticBf162Call(expression, context);
      if (semanticReferenceFunctionCallSupported(expression, context.compiled)) return evalSemanticFunctionCall(expression, context);
      if (semanticReferenceMathCallSupported(expression, "any", context.compiled)) return evalSemanticMathCall(expression, context);
      throw semanticReferenceError(`semantic reference does not support ${expression.kind} expression`, expression.span);
    case "texture-read":
      return evalSemanticTextureRead(expression, context);
    case "surface-read":
      return evalSemanticSurfaceRead(expression, context);
    case "initializer":
      throw semanticReferenceError(`semantic reference does not support ${expression.kind} expression`, expression.span);
    case "update":
      return evalUpdate(expression, context);
  }
}

function semanticReferenceSharedAddressCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): boolean {
  if (expression.callee.kind !== "symbol" || expression.callee.name !== "__cvta_generic_to_shared") return false;
  const arg = expression.args[0];
  if (!arg) return false;
  const ref = semanticReferenceSharedAddressMemoryRef(arg);
  return ref?.addressSpace === "shared";
}

function evalSemanticSharedAddressCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  const arg = expression.args[0];
  const ref = arg ? semanticReferenceSharedAddressMemoryRef(arg) : undefined;
  if (!ref || ref.addressSpace !== "shared") throw semanticReferenceError("__cvta_generic_to_shared requires modeled shared memory", expression.span);
  const elementBytes = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
  return (flatIndex(ref, context) + (context.sharedOffsets.get(ref.base) ?? 0)) * elementBytes;
}

function semanticReferenceSharedAddressMemoryRef(arg: SemanticExpression): SemanticMemoryRef | undefined {
  return memoryRefFromIndexExpression(arg) ?? (arg.kind === "symbol" && arg.addressSpace === "shared" ? {
    base: arg.name,
    addressSpace: arg.addressSpace,
    ...(arg.valueType === undefined ? {} : { valueType: arg.valueType }),
    indices: [],
    fields: [],
    span: arg.span,
  } : undefined);
}

function semanticReferenceStoragePointerIdentity(
  expression: SemanticExpression,
  context: SemanticReferenceContext,
): { readonly buffer: WgslTypedArray; readonly base: number } {
  if (expression.kind !== "symbol") throw semanticReferenceError("semantic storage pointer identity requires symbols", expression.span);
  const buffer = context.buffers.get(expression.name);
  if (!buffer) throw semanticReferenceError(`missing storage pointer '${expression.name}'`, expression.span);
  return { buffer, base: context.storageOffsets.get(expression.name) ?? 0 };
}

function evalSemanticVectorMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.callee.kind !== "symbol" || !semanticVectorMathCallSupported(expression.callee.name, expression.args)) {
    throw semanticReferenceError("semantic reference vector math call is unsupported", expression.span);
  }
  const left = evalSemanticExpression(expression.args[0]!, context);
  if (!Array.isArray(left)) throw semanticReferenceError(`${expression.callee.name} expects a vector`, expression.span);
  if (expression.callee.name === "length" || expression.callee.name === "normalize") {
    const length = Math.sqrt(left.reduce((sum, lane) => sum + lane * lane, 0));
    return expression.callee.name === "length" ? length : left.map((lane) => lane / length);
  }
  const right = evalSemanticExpression(expression.args[1]!, context);
  if (!Array.isArray(right)) throw semanticReferenceError(`${expression.callee.name} expects vectors`, expression.span);
  if (expression.callee.name === "dot") return left.reduce((sum, lane, index) => sum + lane * (right[index] ?? 0), 0);
  return [
    left[1]! * right[2]! - left[2]! * right[1]!,
    left[2]! * right[0]! - left[0]! * right[2]!,
    left[0]! * right[1]! - left[1]! * right[0]!,
  ];
}

function evalSemanticSurfaceRead(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (!semanticReferenceSurfaceReadSupported(expression, context.compiled) || expression.surface.kind !== "symbol") {
    throw semanticReferenceError("semantic reference supports only direct scalar/vector surf2Dread", expression.span);
  }
  const surfaceName = expression.surface.name;
  const surface = context.surfaces[surfaceName];
  if (!surface) throw semanticReferenceError(`missing surface input '${surfaceName}'`, expression.surface.span);
  const xBytes = Math.trunc(evalNumber(expression.xBytes, context));
  const y = Math.trunc(evalNumber(expression.y, context));
  const z = expression.z ? Math.trunc(evalNumber(expression.z, context)) : 0;
  if (isSemanticFloatVectorType(expression.valueType)) {
    return Array.from({ length: cudaVectorLaneCount(expression.valueType) }, (_, lane) => {
      const value = evalSemanticSurfaceLane(surface, surfaceName, xBytes + lane * 4, y, z, context);
      if (expression.valueType === "half2") return roundSemanticHalf(value);
      return expression.valueType === "bf162" ? roundSemanticBfloat16(value) : value;
    });
  }
  const value = evalSemanticSurfaceLane(surface, surfaceName, xBytes, y, z, context);
  if (expression.valueType === "half") return roundSemanticHalf(value);
  if (expression.valueType === "bf16") return roundSemanticBfloat16(value);
  return value;
}

function evalSemanticSurfaceLane(
  surface: WgslTexture2DInput,
  surfaceName: string,
  xBytes: number,
  y: number,
  z: number,
  context: SemanticReferenceContext,
): number {
  const aligned = xBytes % 4 === 0;
  const x = Math.trunc(xBytes / 4);
  const index = ((z * surface.height) + y) * surface.width + x;
  const ok = aligned && xBytes >= 0 && x >= 0 && y >= 0 && z >= 0 && x < surface.width && y < surface.height && index >= 0 && index < surface.data.length;
  const value = ok ? surface.data[index] ?? 0 : 0;
  context.trace.reads.push({ name: surfaceName, index, value, ok });
  return value;
}

function evalSemanticTextureRead(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (!semanticReferenceTextureReadSupported(expression, context.compiled) || expression.texture.kind !== "symbol") {
    throw semanticReferenceError("semantic reference supports only direct scalar/vector texture reads", expression.span);
  }
  const texture = context.textures[expression.texture.name];
  if (!texture) throw semanticReferenceError(`missing texture input '${expression.texture.name}'`, expression.texture.span);
  const channels = texture.channels ?? 1;
  const descriptor = context.textureDescriptors[expression.texture.name] ?? {};
  const xValue = evalNumber(expression.x, context);
  const yValue = evalNumber(expression.y, context);
  if (expression.callee === "texCubemap") {
    const z = evalNumber(expression.z!, context);
    const cube = semanticCubemapTextureCoord(xValue, yValue, z, texture.width, texture.height);
    return evalSemanticTextureValue(expression.valueType, (lane) => texture.data[(cube.y * texture.width + cube.x) * channels + lane] ?? 0);
  }
  const atlasY = expression.z === undefined ? yValue : yValue + evalNumber(expression.z, context);
  if (descriptor.filterMode === "linear") {
    return evalSemanticTextureValue(expression.valueType, (lane) => evalSemanticLinearTextureRead(texture, descriptor, xValue, atlasY, channels, lane));
  }
  const x = semanticTextureCoord(xValue, texture.width, descriptor, "x");
  const y = semanticTextureCoord(atlasY, texture.height, descriptor, "y");
  return evalSemanticTextureValue(expression.valueType, (lane) => texture.data[(y * texture.width + x) * channels + lane] ?? 0);
}

function semanticCubemapTextureCoord(x: number, y: number, z: number, width: number, height: number): { readonly x: number; readonly y: number } {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  const face = ax >= ay && ax >= az ? (x >= 0 ? 0 : 1) : ay >= az ? (y >= 0 ? 2 : 3) : (z >= 0 ? 4 : 5);
  const u = ax >= ay && ax >= az ? z / Math.max(ax, 1e-6) : x / Math.max(ay >= az ? ay : az, 1e-6);
  const v = ax >= ay && ax >= az ? y / Math.max(ax, 1e-6) : ay >= az ? z / Math.max(ay, 1e-6) : y / Math.max(az, 1e-6);
  return {
    x: semanticTextureCoord((u + 1) * 0.5 * (width - 1), width, {}, "x"),
    y: semanticTextureCoord((v + 1) * 0.5 * (width - 1) + face * width, height, {}, "y"),
  };
}

function evalSemanticLinearTextureRead(
  texture: WgslTexture2DInput,
  descriptor: CudaLiteTextureDescriptor,
  xValue: number,
  yValue: number,
  channels: number,
  lane = 0,
): number {
  const x = semanticLinearTextureAxis(xValue, texture.width, descriptor, "x");
  const y = semanticLinearTextureAxis(yValue, texture.height, descriptor, "y");
  const v00 = texture.data[(y.i0 * texture.width + x.i0) * channels + lane] ?? 0;
  const v10 = texture.data[(y.i0 * texture.width + x.i1) * channels + lane] ?? 0;
  const v01 = texture.data[(y.i1 * texture.width + x.i0) * channels + lane] ?? 0;
  const v11 = texture.data[(y.i1 * texture.width + x.i1) * channels + lane] ?? 0;
  const top = v00 + (v10 - v00) * x.alpha;
  const bottom = v01 + (v11 - v01) * x.alpha;
  return top + (bottom - top) * y.alpha;
}

function evalSemanticTextureValue(
  valueType: CudaLiteScalarType | undefined,
  laneValue: (lane: number) => number,
): SemanticValue {
  if (valueType === "half") return roundSemanticHalf(laneValue(0));
  if (valueType === "bf16") return roundSemanticBfloat16(laneValue(0));
  if (!isSemanticFloatVectorType(valueType)) return laneValue(0);
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) => {
    const value = laneValue(lane);
    if (valueType === "half2") return roundSemanticHalf(value);
    return valueType === "bf162" ? roundSemanticBfloat16(value) : value;
  });
}

function semanticTextureCoord(
  value: number,
  extent: number,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): number {
  const scaled = descriptor.normalizedCoords ? value * extent : value;
  return semanticTextureIndex(Math.floor(scaled), extent, descriptor, axis);
}

function semanticLinearTextureAxis(
  value: number,
  extent: number,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): { readonly i0: number; readonly i1: number; readonly alpha: number } {
  const scaled = descriptor.normalizedCoords ? value * extent : value;
  const base = scaled - 0.5;
  const i0 = Math.floor(base);
  return {
    i0: semanticTextureIndex(i0, extent, descriptor, axis),
    i1: semanticTextureIndex(i0 + 1, extent, descriptor, axis),
    alpha: base - i0,
  };
}

function semanticTextureIndex(
  value: number,
  extent: number,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): number {
  const mode = descriptor.addressMode?.[axis === "x" ? 0 : 1] ?? "clamp";
  if (mode === "wrap") return ((value % extent) + extent) % extent;
  return Math.max(0, Math.min(extent - 1, value));
}

function evalUpdate(
  expression: Extract<SemanticExpression, { readonly kind: "update" }>,
  context: SemanticReferenceContext,
): number {
  const delta = expression.operator === "++" ? 1 : expression.operator === "--" ? -1 : 0;
  const ref = memoryRefFromIndexExpression(expression.argument);
  if (ref) {
    const oldValue = readMemory(ref, context);
    const next = oldValue + delta;
    writeMemory(ref, next, context);
    return expression.prefix ? next : oldValue;
  }
  if (expression.argument.kind !== "symbol") {
    throw semanticReferenceError("semantic reference supports only local scalar or modeled memory updates", expression.span);
  }
  const oldValue = evalNumber(expression.argument, context);
  const next = oldValue + delta;
  context.locals.set(expression.argument.name, next);
  return expression.prefix ? next : oldValue;
}

function readIndexExpression(expression: Extract<SemanticExpression, { kind: "index" }>, context: SemanticReferenceContext): SemanticValue {
  if (semanticReferenceVectorIndexSupported(expression, context.compiled)) {
    const value = evalSemanticExpression(expression.target, context);
    if (!Array.isArray(value)) throw semanticReferenceError("semantic reference vector index target is not a vector", expression.target.span);
    const index = Math.trunc(evalNumber(expression.index, context));
    return value[index] ?? 0;
  }
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref) throw semanticReferenceError("semantic reference supports only direct storage indexing", expression.span);
  if (semanticStorageVectorType(ref.valueType) !== undefined) return readVectorMemory(ref, context);
  return readMemory(ref, context);
}

function evalSemanticAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference atomic call requires symbol callee", expression.span);
  const atomicOp = semanticAtomicOperation(expression.callee.name);
  const target = semanticAtomicCallTarget(expression);
  const value = expression.args[1];
  if (!atomicOp || !target || !value) {
    throw semanticReferenceError(`semantic reference does not support atomic '${expression.callee.name}'`, expression.span);
  }
  const oldValue = readAtomicMemory(target, context);
  writeAtomicMemory(target, semanticAtomicValue(atomicOp, oldValue, evalNumber(value, context), expression, context), context);
  return oldValue;
}

function evalSemanticMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference math call requires symbol callee", expression.span);
  const vectorType = semanticVectorMinMaxCallValueType(expression.callee.name, expression.args);
  if (vectorType !== undefined) {
    const [left, right] = expression.args.map((arg) => evalSemanticExpression(arg, context));
    if (left === undefined || right === undefined) throw semanticReferenceError(`${expression.callee.name} expects two operands`, expression.span);
    return evalSemanticVectorMinMax(expression.callee.name, left, right, expression.span);
  }
  const args = expression.args.map((arg) => evalNumber(arg, context));
  const vib = cudaVibMinMaxInfo(expression.callee.name);
  if (vib) {
    return vib.packed
      ? viMinMax16x2(args.slice(0, 2), vib.signed, vib.choose, false)
      : viMinMaxScalar(args.slice(0, 2), vib.signed, vib.choose, false);
  }
  switch (expression.callee.name) {
    case "clock":
    case "clock64":
      return context.blockIdx.x * 104729 +
        context.blockIdx.y * 1009 +
        context.blockIdx.z * 97 +
        context.threadIdx.x +
        context.threadIdx.y * 31 +
        context.threadIdx.z * 7;
    case "sqrt":
    case "sqrtf":
    case "__fsqrt_rn": return Math.sqrt(args[0] ?? 0);
    case "rsqrt":
    case "rsqrtf":
    case "__frsqrt_rn": return 1 / Math.sqrt(args[0] ?? 0);
    case "exp":
    case "expf":
    case "__expf": return Math.exp(args[0] ?? 0);
    case "exp2":
    case "exp2f":
    case "__exp2f": return 2 ** (args[0] ?? 0);
    case "exp10":
    case "exp10f":
    case "__exp10f": return 10 ** (args[0] ?? 0);
    case "expm1":
    case "expm1f": return Math.expm1(args[0] ?? 0);
    case "erf":
    case "erff": return evalSemanticErf(args[0] ?? 0);
    case "erfc":
    case "erfcf": return 1 - evalSemanticErf(args[0] ?? 0);
    case "erfcx":
    case "erfcxf": return Math.exp((args[0] ?? 0) * (args[0] ?? 0)) * (1 - evalSemanticErf(args[0] ?? 0));
    case "erfinv":
    case "erfinvf": return evalSemanticErfinv(args[0] ?? 0);
    case "erfcinv":
    case "erfcinvf": return evalSemanticErfinv(1 - (args[0] ?? 0));
    case "normcdf":
    case "normcdff": return 0.5 * (1 + evalSemanticErf((args[0] ?? 0) * Math.SQRT1_2));
    case "normcdfinv":
    case "normcdfinvf": return evalSemanticNormcdfinv(args[0] ?? 0);
    case "tgamma":
    case "tgammaf": return evalSemanticGamma(args[0] ?? 0);
    case "lgamma":
    case "lgammaf": return evalSemanticLgamma(args[0] ?? 0);
    case "log":
    case "logf":
    case "__logf": return Math.log(args[0] ?? 0);
    case "log2":
    case "log2f":
    case "__log2f": return Math.log2(args[0] ?? 0);
    case "log10":
    case "log10f":
    case "__log10f": return Math.log10(args[0] ?? 0);
    case "log1p":
    case "log1pf": return Math.log1p(args[0] ?? 0);
    case "fabs":
    case "fabsf":
    case "abs": return Math.abs(args[0] ?? 0);
    case "floor":
    case "floorf": return Math.floor(args[0] ?? 0);
    case "ceil":
    case "ceilf": return Math.ceil(args[0] ?? 0);
    case "trunc":
    case "truncf": return Math.trunc(args[0] ?? 0);
    case "round":
    case "roundf": return roundAwayFromZero(args[0] ?? 0);
    case "rint":
    case "rintf":
    case "nearbyint":
    case "nearbyintf": return roundTiesToEvenNumber(args[0] ?? 0);
    case "sin":
    case "sinf":
    case "__sinf": return Math.sin(args[0] ?? 0);
    case "sinpi":
    case "sinpif": return Math.sin(Math.PI * (args[0] ?? 0));
    case "cos":
    case "cosf":
    case "__cosf": return Math.cos(args[0] ?? 0);
    case "cospi":
    case "cospif": return Math.cos(Math.PI * (args[0] ?? 0));
    case "tan":
    case "tanf":
    case "__tanf": return Math.tan(args[0] ?? 0);
    case "asin":
    case "asinf": return Math.asin(args[0] ?? 0);
    case "acos":
    case "acosf": return Math.acos(args[0] ?? 0);
    case "atan":
    case "atanf": return Math.atan(args[0] ?? 0);
    case "atan2":
    case "atan2f": return Math.atan2(args[0] ?? 0, args[1] ?? 0);
    case "sinh":
    case "sinhf": return Math.sinh(args[0] ?? 0);
    case "cosh":
    case "coshf": return Math.cosh(args[0] ?? 0);
    case "tanh":
    case "tanhf":
    case "__tanhf": return Math.tanh(args[0] ?? 0);
    case "asinh":
    case "asinhf": return Math.asinh(args[0] ?? 0);
    case "acosh":
    case "acoshf": return Math.acosh(args[0] ?? 0);
    case "atanh":
    case "atanhf": return Math.atanh(args[0] ?? 0);
    case "cbrt":
    case "cbrtf": return Math.cbrt(args[0] ?? 0);
    case "rcbrt":
    case "rcbrtf": return 1 / Math.cbrt(args[0] ?? 0);
    case "ldexp":
    case "ldexpf":
    case "scalbn":
    case "scalbnf":
    case "scalbln":
    case "scalblnf": return (args[0] ?? 0) * (2 ** Math.trunc(args[1] ?? 0));
    case "fmod":
    case "fmodf": return (args[0] ?? 0) - Math.trunc((args[0] ?? 0) / (args[1] ?? 0)) * (args[1] ?? 0);
    case "remainder":
    case "remainderf": return (args[0] ?? 0) - roundTiesToEvenNumber((args[0] ?? 0) / (args[1] ?? 0)) * (args[1] ?? 0);
    case "logb":
    case "logbf": return evalLogb(args[0] ?? 0);
    case "ilogb":
    case "ilogbf": return evalIlogb(args[0] ?? 0);
    case "fdim":
    case "fdimf": return Math.max((args[0] ?? 0) - (args[1] ?? 0), 0);
    case "nextafter":
    case "nextafterf":
    case "nexttoward":
    case "nexttowardf": return evalNextafter(args[0] ?? 0, args[1] ?? 0);
    case "hypot":
    case "hypotf": return Math.hypot(args[0] ?? 0, args[1] ?? 0);
    case "rhypot":
    case "rhypotf": return 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0);
    case "norm3df": return Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0);
    case "norm4df": return Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, args[3] ?? 0);
    case "rnorm3df": return 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0);
    case "rnorm4df": return 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, args[3] ?? 0);
    case "lrint":
    case "lrintf":
    case "llrint":
    case "llrintf":
    case "__float2int_rn": return roundTiesToEvenNumber(args[0] ?? 0) | 0;
    case "lround":
    case "lroundf":
    case "llround":
    case "llroundf": return roundAwayFromZero(args[0] ?? 0) | 0;
    case "__float2int_rz": return Math.trunc(args[0] ?? 0) | 0;
    case "__float2int_ru": return Math.ceil(args[0] ?? 0) | 0;
    case "__float2int_rd": return Math.floor(args[0] ?? 0) | 0;
    case "__float2uint_rn": return roundTiesToEvenNumber(args[0] ?? 0) >>> 0;
    case "__float2uint_rz": return Math.trunc(args[0] ?? 0) >>> 0;
    case "__float2uint_ru": return Math.ceil(args[0] ?? 0) >>> 0;
    case "__float2uint_rd": return Math.floor(args[0] ?? 0) >>> 0;
    case "__int2float_rn":
    case "__int2float_rz":
    case "__int2float_ru":
    case "__int2float_rd": return Math.trunc(args[0] ?? 0) | 0;
    case "__uint2float_rn":
    case "__uint2float_rz":
    case "__uint2float_ru":
    case "__uint2float_rd": return Math.trunc(args[0] ?? 0) >>> 0;
    case "__half2float": return args[0] ?? 0;
    case "__float2half":
    case "__float2half_rn": return roundSemanticHalf(args[0] ?? 0);
    case "__float2half_rz": return roundFloat32ToFloat16(args[0] ?? 0, "rz");
    case "__float2half_ru": return roundFloat32ToFloat16(args[0] ?? 0, "ru");
    case "__float2half_rd": return roundFloat32ToFloat16(args[0] ?? 0, "rd");
    case "__int2half_rn": return roundSemanticHalf(Math.trunc(args[0] ?? 0) | 0);
    case "__int2half_rz": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) | 0, "rz");
    case "__int2half_ru": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) | 0, "ru");
    case "__int2half_rd": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) | 0, "rd");
    case "__uint2half_rn": return roundSemanticHalf(Math.trunc(args[0] ?? 0) >>> 0);
    case "__uint2half_rz": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) >>> 0, "rz");
    case "__uint2half_ru": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) >>> 0, "ru");
    case "__uint2half_rd": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) >>> 0, "rd");
    case "__short2half_rn": return roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "rn");
    case "__short2half_rz": return roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "rz");
    case "__short2half_ru": return roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "ru");
    case "__short2half_rd": return roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "rd");
    case "__ushort2half_rn": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rn");
    case "__ushort2half_rz": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rz");
    case "__ushort2half_ru": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "ru");
    case "__ushort2half_rd": return roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rd");
    case "__half_as_short": return signExtend16(float32ToFloat16Bits(args[0] ?? 0));
    case "__half_as_ushort": return float32ToFloat16Bits(args[0] ?? 0) >>> 0;
    case "__short_as_half": return float16BitsToFloat32(Math.trunc(args[0] ?? 0) & 0xffff);
    case "__ushort_as_half": return float16BitsToFloat32(Math.trunc(args[0] ?? 0) & 0xffff);
    case "__half2int_rn": return roundTiesToEvenNumber(args[0] ?? 0) | 0;
    case "__half2int_rz": return Math.trunc(args[0] ?? 0) | 0;
    case "__half2int_ru": return Math.ceil(args[0] ?? 0) | 0;
    case "__half2int_rd": return Math.floor(args[0] ?? 0) | 0;
    case "__half2short_rn": return signExtend16(roundTiesToEvenNumber(args[0] ?? 0));
    case "__half2short_rz": return signExtend16(Math.trunc(args[0] ?? 0));
    case "__half2short_ru": return signExtend16(Math.ceil(args[0] ?? 0));
    case "__half2short_rd": return signExtend16(Math.floor(args[0] ?? 0));
    case "__half2uint_rn": return roundTiesToEvenNumber(args[0] ?? 0) >>> 0;
    case "__half2uint_rz": return Math.trunc(args[0] ?? 0) >>> 0;
    case "__half2uint_ru": return Math.ceil(args[0] ?? 0) >>> 0;
    case "__half2uint_rd": return Math.floor(args[0] ?? 0) >>> 0;
    case "__half2ushort_rn": return (roundTiesToEvenNumber(args[0] ?? 0) & 0xffff) >>> 0;
    case "__half2ushort_rz": return (Math.trunc(args[0] ?? 0) & 0xffff) >>> 0;
    case "__half2ushort_ru": return (Math.ceil(args[0] ?? 0) & 0xffff) >>> 0;
    case "__half2ushort_rd": return (Math.floor(args[0] ?? 0) & 0xffff) >>> 0;
    case "__nv_cvt_fp8_to_halfraw": return roundSemanticHalf(semanticFp8ToFloat32(args[0] ?? 0, args[1] ?? 0));
    case "__nv_cvt_float_to_fp8": return semanticFloat32ToFp8(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0);
    case "__habs": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.abs(args[0] ?? 0))
      : roundSemanticHalf(Math.abs(args[0] ?? 0));
    case "__hceil": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.ceil(args[0] ?? 0))
      : roundSemanticHalf(Math.ceil(args[0] ?? 0));
    case "__hfloor": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.floor(args[0] ?? 0))
      : roundSemanticHalf(Math.floor(args[0] ?? 0));
    case "__hrcp": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(1 / (args[0] ?? 0))
      : roundSemanticHalf(1 / (args[0] ?? 0));
    case "__hrsqrt":
    case "hrsqrt": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(1 / Math.sqrt(args[0] ?? 0))
      : roundSemanticHalf(1 / Math.sqrt(args[0] ?? 0));
    case "__hsqrt": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.sqrt(args[0] ?? 0))
      : roundSemanticHalf(Math.sqrt(args[0] ?? 0));
    case "__htrunc": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.trunc(args[0] ?? 0))
      : roundSemanticHalf(Math.trunc(args[0] ?? 0));
    case "__hneg": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(-(args[0] ?? 0))
      : roundSemanticHalf(-(args[0] ?? 0));
    case "__hadd_rn": return expression.valueType === "bf16"
      ? roundSemanticBfloat16((args[0] ?? 0) + (args[1] ?? 0))
      : roundSemanticHalf((args[0] ?? 0) + (args[1] ?? 0));
    case "__hadd_sat": return expression.valueType === "bf16"
      ? saturateSemanticBfloat16((args[0] ?? 0) + (args[1] ?? 0))
      : saturateSemanticHalf((args[0] ?? 0) + (args[1] ?? 0));
    case "__hsub":
    case "__hsub_rn": return expression.valueType === "bf16"
      ? roundSemanticBfloat16((args[0] ?? 0) - (args[1] ?? 0))
      : roundSemanticHalf((args[0] ?? 0) - (args[1] ?? 0));
    case "__hsub_sat": return expression.valueType === "bf16"
      ? saturateSemanticBfloat16((args[0] ?? 0) - (args[1] ?? 0))
      : saturateSemanticHalf((args[0] ?? 0) - (args[1] ?? 0));
    case "__hmul":
    case "__hmul_rn": return expression.valueType === "bf16"
      ? roundSemanticBfloat16((args[0] ?? 0) * (args[1] ?? 0))
      : roundSemanticHalf((args[0] ?? 0) * (args[1] ?? 0));
    case "__hmul_sat": return expression.valueType === "bf16"
      ? saturateSemanticBfloat16((args[0] ?? 0) * (args[1] ?? 0))
      : saturateSemanticHalf((args[0] ?? 0) * (args[1] ?? 0));
    case "__hdiv":
    case "__hdiv_rn": return expression.valueType === "bf16"
      ? roundSemanticBfloat16((args[0] ?? 0) / (args[1] ?? 0))
      : roundSemanticHalf((args[0] ?? 0) / (args[1] ?? 0));
    case "__hfma":
    case "__hfma_rn": return expression.valueType === "bf16"
      ? roundSemanticBfloat16((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0))
      : roundSemanticHalf((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0));
    case "__hfma_sat": return expression.valueType === "bf16"
      ? saturateSemanticBfloat16((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0))
      : saturateSemanticHalf((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0));
    case "__hfma_relu": return expression.valueType === "bf16"
      ? reluSemanticBfloat16((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0))
      : roundSemanticHalf(Math.max((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0), 0));
    case "hexp": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.exp(args[0] ?? 0))
      : roundSemanticHalf(Math.exp(args[0] ?? 0));
    case "__hmin": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.min(args[0] ?? 0, args[1] ?? 0))
      : roundSemanticHalf(Math.min(args[0] ?? 0, args[1] ?? 0));
    case "__hmax": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.max(args[0] ?? 0, args[1] ?? 0))
      : roundSemanticHalf(Math.max(args[0] ?? 0, args[1] ?? 0));
    case "__hmin_nan": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.min(args[0] ?? 0, args[1] ?? 0))
      : roundSemanticHalf(Math.min(args[0] ?? 0, args[1] ?? 0));
    case "__hmax_nan": return expression.valueType === "bf16"
      ? roundSemanticBfloat16(Math.max(args[0] ?? 0, args[1] ?? 0))
      : roundSemanticHalf(Math.max(args[0] ?? 0, args[1] ?? 0));
    case "__hisnan": return Number.isNaN(args[0] ?? 0) ? 1 : 0;
    case "__hisinf": {
      const value = args[0] ?? 0;
      return value === Infinity ? 1 : value === -Infinity ? -1 : 0;
    }
    case "__heq": return (args[0] ?? 0) === (args[1] ?? 0) ? 1 : 0;
    case "__hne": return (args[0] ?? 0) !== (args[1] ?? 0) ? 1 : 0;
    case "__hgt": return (args[0] ?? 0) > (args[1] ?? 0) ? 1 : 0;
    case "__hge": return (args[0] ?? 0) >= (args[1] ?? 0) ? 1 : 0;
    case "__hlt": return (args[0] ?? 0) < (args[1] ?? 0) ? 1 : 0;
    case "__hle": return (args[0] ?? 0) <= (args[1] ?? 0) ? 1 : 0;
    case "__hequ": return unorderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left === right);
    case "__hneu": return unorderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left !== right);
    case "__hgtu": return unorderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left > right);
    case "__hgeu": return unorderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left >= right);
    case "__hltu": return unorderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left < right);
    case "__hleu": return unorderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left <= right);
    case "__bfloat162float": return args[0] ?? 0;
    case "__float2bfloat16":
    case "__float2bfloat16_rn": return roundSemanticBfloat16(args[0] ?? 0);
    case "__float2bfloat16_rz": return roundFloat32ToBfloat16(args[0] ?? 0, "rz");
    case "__float2bfloat16_ru": return roundFloat32ToBfloat16(args[0] ?? 0, "ru");
    case "__float2bfloat16_rd": return roundFloat32ToBfloat16(args[0] ?? 0, "rd");
    case "__double2bfloat16": return roundSemanticBfloat16(args[0] ?? 0);
    case "__int2bfloat16_rn": return roundSemanticBfloat16(Math.trunc(args[0] ?? 0) | 0);
    case "__int2bfloat16_rz": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "rz");
    case "__int2bfloat16_ru": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "ru");
    case "__int2bfloat16_rd": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "rd");
    case "__ll2bfloat16_rn": return roundSemanticBfloat16(Math.trunc(args[0] ?? 0) | 0);
    case "__ll2bfloat16_rz": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "rz");
    case "__ll2bfloat16_ru": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "ru");
    case "__ll2bfloat16_rd": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "rd");
    case "__uint2bfloat16_rn": return roundSemanticBfloat16(Math.trunc(args[0] ?? 0) >>> 0);
    case "__uint2bfloat16_rz": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "rz");
    case "__uint2bfloat16_ru": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "ru");
    case "__uint2bfloat16_rd": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "rd");
    case "__ull2bfloat16_rn": return roundSemanticBfloat16(Math.trunc(args[0] ?? 0) >>> 0);
    case "__ull2bfloat16_rz": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "rz");
    case "__ull2bfloat16_ru": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "ru");
    case "__ull2bfloat16_rd": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "rd");
    case "__short2bfloat16_rn": return roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "rn");
    case "__short2bfloat16_rz": return roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "rz");
    case "__short2bfloat16_ru": return roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "ru");
    case "__short2bfloat16_rd": return roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "rd");
    case "__ushort2bfloat16_rn": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rn");
    case "__ushort2bfloat16_rz": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rz");
    case "__ushort2bfloat16_ru": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "ru");
    case "__ushort2bfloat16_rd": return roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rd");
    case "__bfloat16_as_short": return signExtend16(roundFloat32ToBfloat16Bits(args[0] ?? 0));
    case "__bfloat16_as_ushort":
    case "__nv_bfloat16_as_ushort": return (float32ToUintBits(roundSemanticBfloat16(args[0] ?? 0)) >>> 16) & 0xffff;
    case "__short_as_bfloat16":
    case "__ushort_as_bfloat16": return bfloat16BitsToFloat32(args[0] ?? 0);
    case "__bfloat162int_rn": return roundTiesToEvenNumber(args[0] ?? 0) | 0;
    case "__bfloat162int_rz": return Math.trunc(args[0] ?? 0) | 0;
    case "__bfloat162int_ru": return Math.ceil(args[0] ?? 0) | 0;
    case "__bfloat162int_rd": return Math.floor(args[0] ?? 0) | 0;
    case "__bfloat162ll_rn": return roundTiesToEvenNumber(args[0] ?? 0) | 0;
    case "__bfloat162ll_rz": return Math.trunc(args[0] ?? 0) | 0;
    case "__bfloat162ll_ru": return Math.ceil(args[0] ?? 0) | 0;
    case "__bfloat162ll_rd": return Math.floor(args[0] ?? 0) | 0;
    case "__bfloat162uint_rn": return roundTiesToEvenNumber(args[0] ?? 0) >>> 0;
    case "__bfloat162uint_rz": return Math.trunc(args[0] ?? 0) >>> 0;
    case "__bfloat162uint_ru": return Math.ceil(args[0] ?? 0) >>> 0;
    case "__bfloat162uint_rd": return Math.floor(args[0] ?? 0) >>> 0;
    case "__bfloat162ull_rn": return roundTiesToEvenNumber(args[0] ?? 0) >>> 0;
    case "__bfloat162ull_rz": return Math.trunc(args[0] ?? 0) >>> 0;
    case "__bfloat162ull_ru": return Math.ceil(args[0] ?? 0) >>> 0;
    case "__bfloat162ull_rd": return Math.floor(args[0] ?? 0) >>> 0;
    case "__bfloat162short_rn": return signExtend16(roundTiesToEvenNumber(args[0] ?? 0));
    case "__bfloat162short_rz": return signExtend16(Math.trunc(args[0] ?? 0));
    case "__bfloat162short_ru": return signExtend16(Math.ceil(args[0] ?? 0));
    case "__bfloat162short_rd": return signExtend16(Math.floor(args[0] ?? 0));
    case "__bfloat162ushort_rn": return roundTiesToEvenNumber(args[0] ?? 0) & 0xffff;
    case "__bfloat162ushort_rz": return Math.trunc(args[0] ?? 0) & 0xffff;
    case "__bfloat162ushort_ru": return Math.ceil(args[0] ?? 0) & 0xffff;
    case "__bfloat162ushort_rd": return Math.floor(args[0] ?? 0) & 0xffff;
    case "__bfloat162char_rz": return signExtend8(Math.trunc(args[0] ?? 0));
    case "__bfloat162uchar_rz": return Math.trunc(args[0] ?? 0) & 0xff;
    case "wmma::__float_to_tf32": return args[0] ?? 0;
    case "__clz": return Math.clz32(args[0] ?? 0);
    case "__clzll": return Math.clz32(args[0] ?? 0) + 32;
    case "__ffs": return evalSemanticFfs(args[0] ?? 0);
    case "__ffsll": return evalSemanticFfs(args[0] ?? 0);
    case "__popc": return popCount32(args[0] ?? 0);
    case "__popcll": return popCount32(args[0] ?? 0);
    case "__brev": return reverseBits32(args[0] ?? 0);
    case "__brevll": return reverseBits32(args[0] ?? 0);
    case "__mul24": return Math.imul(args[0] ?? 0, args[1] ?? 0);
    case "__umul24": return Math.imul(args[0] ?? 0, args[1] ?? 0) >>> 0;
    case "__mulhi":
    case "__mul64hi": return signedMulHi32(args[0] ?? 0, args[1] ?? 0);
    case "__umulhi":
    case "__umul64hi": return unsignedMulHi32(args[0] ?? 0, args[1] ?? 0);
    case "__byte_perm": return bytePerm(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0);
    case "__funnelshift_l": return funnelShiftLeft(args[0] ?? 0, args[1] ?? 0, Math.trunc(args[2] ?? 0) & 31);
    case "__funnelshift_lc": return funnelShiftLeft(args[0] ?? 0, args[1] ?? 0, Math.max(0, Math.min(32, Math.trunc(args[2] ?? 0))));
    case "__funnelshift_r": return funnelShiftRight(args[0] ?? 0, args[1] ?? 0, Math.trunc(args[2] ?? 0) & 31);
    case "__funnelshift_rc": return funnelShiftRight(args[0] ?? 0, args[1] ?? 0, Math.max(0, Math.min(32, Math.trunc(args[2] ?? 0))));
    case "__rhadd": return roundedSignedAverage(args[0] ?? 0, args[1] ?? 0);
    case "__uhadd": return unsignedAverage(args[0] ?? 0, args[1] ?? 0);
    case "__urhadd": return roundedUnsignedAverage(args[0] ?? 0, args[1] ?? 0);
    case "__hadd": return expression.valueType === "half"
      ? roundSemanticHalf((args[0] ?? 0) + (args[1] ?? 0))
      : expression.valueType === "bf16"
      ? roundSemanticBfloat16((args[0] ?? 0) + (args[1] ?? 0))
      : signedAverage(args[0] ?? 0, args[1] ?? 0);
    case "__float_as_int": return float32ToUintBits(args[0] ?? 0) | 0;
    case "__float_as_uint": return float32ToUintBits(args[0] ?? 0) >>> 0;
    case "__sad": return (Math.abs((Math.trunc(args[0] ?? 0) | 0) - (Math.trunc(args[1] ?? 0) | 0)) + (Math.trunc(args[2] ?? 0) >>> 0)) >>> 0;
    case "__usad": return (Math.abs((Math.trunc(args[0] ?? 0) >>> 0) - (Math.trunc(args[1] ?? 0) >>> 0)) + (Math.trunc(args[2] ?? 0) >>> 0)) >>> 0;
    case "__usad4": return u8x4SadAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0);
    case "__viaddmax_s32": return viaddScalar(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "max", false);
    case "__viaddmax_s32_relu": return viaddScalar(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "max", true);
    case "__viaddmin_s32": return viaddScalar(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "min", false);
    case "__viaddmin_s32_relu": return viaddScalar(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "min", true);
    case "__viaddmax_u32": return viaddScalar(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, false, "max", false);
    case "__viaddmin_u32": return viaddScalar(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, false, "min", false);
    case "__viaddmax_s16x2": return viadd16x2(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "max", false);
    case "__viaddmax_s16x2_relu": return viadd16x2(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "max", true);
    case "__viaddmin_s16x2": return viadd16x2(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "min", false);
    case "__viaddmin_s16x2_relu": return viadd16x2(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, true, "min", true);
    case "__viaddmax_u16x2": return viadd16x2(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, false, "max", false);
    case "__viaddmin_u16x2": return viadd16x2(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, false, "min", false);
    case "__vimax_s32_relu": return viMinMaxScalar(args, true, "max", true);
    case "__vimin_s32_relu": return viMinMaxScalar(args, true, "min", true);
    case "__vimax_s16x2_relu": return viMinMax16x2(args, true, "max", true);
    case "__vimin_s16x2_relu": return viMinMax16x2(args, true, "min", true);
    case "__vimax3_s32": return viMinMaxScalar(args, true, "max", false);
    case "__vimax3_s32_relu": return viMinMaxScalar(args, true, "max", true);
    case "__vimin3_s32": return viMinMaxScalar(args, true, "min", false);
    case "__vimin3_s32_relu": return viMinMaxScalar(args, true, "min", true);
    case "__vimax3_u32": return viMinMaxScalar(args, false, "max", false);
    case "__vimin3_u32": return viMinMaxScalar(args, false, "min", false);
    case "__vimax3_s16x2": return viMinMax16x2(args, true, "max", false);
    case "__vimax3_s16x2_relu": return viMinMax16x2(args, true, "max", true);
    case "__vimin3_s16x2": return viMinMax16x2(args, true, "min", false);
    case "__vimin3_s16x2_relu": return viMinMax16x2(args, true, "min", true);
    case "__vimax3_u16x2": return viMinMax16x2(args, false, "max", false);
    case "__vimin3_u16x2": return viMinMax16x2(args, false, "min", false);
    case "__vadd2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => a + b);
    case "__vsub2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => a - b);
    case "__vabs2": return packedUnary(args[0] ?? 0, 16, true, (a) => Math.abs(a));
    case "__vabsss2": return packedUnary(args[0] ?? 0, 16, true, (a) => Math.min(32767, Math.abs(a)));
    case "__vneg2": return packedUnary(args[0] ?? 0, 16, true, (a) => -a);
    case "__vnegss2": return packedUnary(args[0] ?? 0, 16, true, (a) => Math.min(32767, Math.max(-32768, -a)));
    case "__vaddss2": return i16x2SaturatingBinary(args[0] ?? 0, args[1] ?? 0, (a, b) => a + b);
    case "__vsubss2": return i16x2SaturatingBinary(args[0] ?? 0, args[1] ?? 0, (a, b) => a - b);
    case "__vaddus2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.min(0xffff, a + b));
    case "__vsubus2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.max(0, a - b));
    case "__vabsdiffu2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.abs(a - b));
    case "__vabsdiffs2": return i16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.abs(a - b));
    case "__vsads2": return packedSad(args[0] ?? 0, args[1] ?? 0, 16, true);
    case "__vsadu2": return packedSad(args[0] ?? 0, args[1] ?? 0, 16, false);
    case "__vhaddu2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => (a + b) >> 1);
    case "__vavgs2": return i16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => (a + b + 1) >> 1);
    case "__vavgu2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => (a + b + 1) >> 1);
    case "__vminu2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, Math.min);
    case "__vmaxu2": return u16x2Binary(args[0] ?? 0, args[1] ?? 0, Math.max);
    case "__vmins2": return i16x2Binary(args[0] ?? 0, args[1] ?? 0, Math.min);
    case "__vmaxs2": return i16x2Binary(args[0] ?? 0, args[1] ?? 0, Math.max);
    case "__vcmpeq2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a === b);
    case "__vcmpne2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a !== b);
    case "__vcmpges2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a >= b);
    case "__vcmpgeu2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a >= b);
    case "__vcmpgts2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a > b);
    case "__vcmpgtu2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a > b);
    case "__vcmples2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a <= b);
    case "__vcmpleu2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a <= b);
    case "__vcmplts2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a < b);
    case "__vcmpltu2": return vcompare(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a < b);
    case "__vseteq2": return vset(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a === b);
    case "__vsetne2": return vset(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a !== b);
    case "__vsetges2": return vset(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a >= b);
    case "__vsetgeu2": return vset(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a >= b);
    case "__vsetgts2": return vset(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a > b);
    case "__vsetgtu2": return vset(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a > b);
    case "__vsetles2": return vset(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a <= b);
    case "__vsetleu2": return vset(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a <= b);
    case "__vsetlts2": return vset(args[0] ?? 0, args[1] ?? 0, 16, true, (a, b) => a < b);
    case "__vsetltu2": return vset(args[0] ?? 0, args[1] ?? 0, 16, false, (a, b) => a < b);
    case "__vadd4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => a + b);
    case "__vsub4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => a - b);
    case "__vabs4": return packedUnary(args[0] ?? 0, 8, true, (a) => Math.abs(a));
    case "__vabsss4": return packedUnary(args[0] ?? 0, 8, true, (a) => Math.min(127, Math.abs(a)));
    case "__vneg4": return packedUnary(args[0] ?? 0, 8, true, (a) => -a);
    case "__vnegss4": return packedUnary(args[0] ?? 0, 8, true, (a) => Math.min(127, Math.max(-128, -a)));
    case "__vaddss4": return i8x4SaturatingBinary(args[0] ?? 0, args[1] ?? 0, (a, b) => a + b);
    case "__vsubss4": return i8x4SaturatingBinary(args[0] ?? 0, args[1] ?? 0, (a, b) => a - b);
    case "__vaddus4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.min(0xff, a + b));
    case "__vsubus4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.max(0, a - b));
    case "__vabsdiffu4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.abs(a - b));
    case "__vabsdiffs4": return i8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => Math.abs(a - b));
    case "__vsads4": return packedSad(args[0] ?? 0, args[1] ?? 0, 8, true);
    case "__vsadu4": return packedSad(args[0] ?? 0, args[1] ?? 0, 8, false);
    case "__vhaddu4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => (a + b) >> 1);
    case "__vavgs4": return i8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => (a + b + 1) >> 1);
    case "__vavgu4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, (a, b) => (a + b + 1) >> 1);
    case "__vminu4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, Math.min);
    case "__vmaxu4": return u8x4Binary(args[0] ?? 0, args[1] ?? 0, Math.max);
    case "__vmins4": return i8x4Binary(args[0] ?? 0, args[1] ?? 0, Math.min);
    case "__vmaxs4": return i8x4Binary(args[0] ?? 0, args[1] ?? 0, Math.max);
    case "__vcmpeq4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a === b);
    case "__vcmpne4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a !== b);
    case "__vcmpges4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a >= b);
    case "__vcmpgeu4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a >= b);
    case "__vcmpgts4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a > b);
    case "__vcmpgtu4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a > b);
    case "__vcmples4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a <= b);
    case "__vcmpleu4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a <= b);
    case "__vcmplts4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a < b);
    case "__vcmpltu4": return vcompare(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a < b);
    case "__vseteq4": return vset(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a === b);
    case "__vsetne4": return vset(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a !== b);
    case "__vsetges4": return vset(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a >= b);
    case "__vsetgeu4": return vset(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a >= b);
    case "__vsetgts4": return vset(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a > b);
    case "__vsetgtu4": return vset(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a > b);
    case "__vsetles4": return vset(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a <= b);
    case "__vsetleu4": return vset(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a <= b);
    case "__vsetlts4": return vset(args[0] ?? 0, args[1] ?? 0, 8, true, (a, b) => a < b);
    case "__vsetltu4": return vset(args[0] ?? 0, args[1] ?? 0, 8, false, (a, b) => a < b);
    case "__dp4a": return expression.valueType === "uint"
      ? u8x4DotAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0)
      : i8x4DotAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0);
    case "__dp2a_lo": return expression.valueType === "uint"
      ? u16x2U8x2DotAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, 0)
      : i16x2I8x2DotAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, 0);
    case "__dp2a_hi": return expression.valueType === "uint"
      ? u16x2U8x2DotAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, 16)
      : i16x2I8x2DotAdd(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, 16);
    case "IMAD": return (Math.imul(args[0] ?? 0, args[1] ?? 0) + (Math.trunc(args[2] ?? 0) | 0)) | 0;
    case "UMUL": return Math.imul(args[0] ?? 0, args[1] ?? 0) >>> 0;
    case "UMAD": return (Math.imul(args[0] ?? 0, args[1] ?? 0) + (Math.trunc(args[2] ?? 0) >>> 0)) >>> 0;
    case "umin": return Math.min(Math.trunc(args[0] ?? 0) >>> 0, Math.trunc(args[1] ?? 0) >>> 0) >>> 0;
    case "assert": return 0;
    case "fmin":
    case "fminf":
    case "min": return Math.min(args[0] ?? 0, args[1] ?? 0);
    case "fmax":
    case "fmaxf":
    case "max": return Math.max(args[0] ?? 0, args[1] ?? 0);
    case "pow":
    case "powf":
    case "__powf": return Math.pow(args[0] ?? 0, args[1] ?? 0);
    case "__fdividef":
    case "fdividef":
    case "__fdiv_rn": return (args[0] ?? 0) / (args[1] ?? 0);
    case "__frcp_rn": return 1 / (args[0] ?? 0);
    case "__fadd_rn": return (args[0] ?? 0) + (args[1] ?? 0);
    case "__fsub_rn": return (args[0] ?? 0) - (args[1] ?? 0);
    case "__fmul_rn": return (args[0] ?? 0) * (args[1] ?? 0);
    case "__builtin_inff":
    case "__builtin_huge_valf": return Infinity;
    case "__uint_as_float":
    case "__int_as_float": return uintBitsToFloat32(args[0] ?? 0);
    case "__saturatef": return Math.min(1, Math.max(0, args[0] ?? 0));
    case "copysign":
    case "copysignf": return Math.sign(args[1] ?? 0) < 0 || Object.is(args[1] ?? 0, -0) ? -Math.abs(args[0] ?? 0) : Math.abs(args[0] ?? 0);
    case "isinf":
    case "isinff":
    case "__isinff": return !Number.isFinite(args[0] ?? 0) && !Number.isNaN(args[0] ?? 0) ? 1 : 0;
    case "isfinite":
    case "isfinitef":
    case "finite":
    case "finitef":
    case "__finitef": return Number.isFinite(args[0] ?? 0) ? 1 : 0;
    case "isnan":
    case "isnanf":
    case "__isnanf":
    case "isNan": return Number.isNaN(args[0] ?? 0) ? 1 : 0;
    case "signbit":
    case "signbitf": return Math.sign(args[0] ?? 0) < 0 || Object.is(args[0] ?? 0, -0) ? 1 : 0;
    case "isnormal": {
      const value = Math.abs(args[0] ?? 0);
      return Number.isFinite(value) && value >= 1.1754943508222875e-38 && value <= 3.4028234663852886e38 ? 1 : 0;
    }
    case "isgreater": return orderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left > right);
    case "isgreaterequal": return orderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left >= right);
    case "isless": return orderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left < right);
    case "islessequal": return orderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left <= right);
    case "islessgreater": return orderedCompare(args[0] ?? 0, args[1] ?? 0, (left, right) => left !== right);
    case "isunordered": return Number.isNaN(args[0] ?? 0) || Number.isNaN(args[1] ?? 0) ? 1 : 0;
    case "fma":
    case "fmaf":
    case "__fmaf_rn": return (args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0);
    case "lerp": return (args[0] ?? 0) + (args[2] ?? 0) * ((args[1] ?? 0) - (args[0] ?? 0));
    case "div_ceil":
    case "ceil_div": return Math.trunc((Math.trunc(args[0] ?? 0) + Math.trunc(args[1] ?? 1) - 1) / Math.trunc(args[1] ?? 1));
    case "__bg_modf_intpart": return modfIntpart(args[0] ?? 0);
    case "__bg_modf_fraction": return modfFraction(args[0] ?? 0);
    case "__bg_frexp_exponent": return frexpExponent(args[0] ?? 0);
    case "__bg_frexp_mantissa": return frexpMantissa(args[0] ?? 0);
    case "__bg_remquo_quotient": return roundTiesToEvenNumber((args[0] ?? 0) / (args[1] ?? 1));
    case "__bg_remquo_remainder": {
      const quotient = roundTiesToEvenNumber((args[0] ?? 0) / (args[1] ?? 1));
      return (args[0] ?? 0) - quotient * (args[1] ?? 1);
    }
    case "__bg_i16_lane": return signExtend16(((Math.trunc(args[0] ?? 0) >>> 0) >>> (Math.trunc(args[1] ?? 0) & 16)) & 0xffff);
    case "__bg_u16_lane": return (((Math.trunc(args[0] ?? 0) >>> 0) >>> (Math.trunc(args[1] ?? 0) & 16)) & 0xffff) >>> 0;
    default:
      throw semanticReferenceError(`semantic reference does not support math call '${expression.callee.name}'`, expression.span);
  }
}

function frexpExponent(value: number): number {
  return value === 0 || !Number.isFinite(value) ? 0 : Math.floor(Math.log2(Math.abs(value))) + 1;
}

function orderedCompare(left: number, right: number, compare: (left: number, right: number) => boolean): number {
  return !Number.isNaN(left) && !Number.isNaN(right) && compare(left, right) ? 1 : 0;
}

function unorderedCompare(left: number, right: number, compare: (left: number, right: number) => boolean): number {
  return Number.isNaN(left) || Number.isNaN(right) || compare(left, right) ? 1 : 0;
}

function roundSemanticHalf(value: number): number {
  return float16BitsToFloat32(float32ToFloat16Bits(value));
}

function saturateSemanticHalf(value: number): number {
  return Number.isNaN(value) ? 0 : roundSemanticHalf(Math.min(1, Math.max(0, value)));
}

function roundSemanticBfloat16(value: number): number {
  return roundFloat32ToBfloat16(value, "rn");
}

function saturateSemanticBfloat16(value: number): number {
  return Number.isNaN(value) ? 0 : roundSemanticBfloat16(Math.min(1, Math.max(0, value)));
}

function reluSemanticBfloat16(value: number): number {
  return Number.isNaN(value) ? roundSemanticBfloat16(Number.NaN) : roundSemanticBfloat16(Math.max(0, value));
}

function semanticFp8ToFloat32(bits: number, mode: number): number {
  const value = Math.trunc(bits) & 0xff;
  const sign = (value & 0x80) === 0 ? 1 : -1;
  if ((Math.trunc(mode) >>> 0) === 1) return semanticFp8E5M2ToFloat32(value, sign);
  return semanticFp8E4M3ToFloat32(value, sign);
}

function semanticFp8E4M3ToFloat32(value: number, sign: number): number {
  const exponent = (value >>> 3) & 0x0f;
  const mantissa = value & 0x07;
  if (exponent === 0 && mantissa === 0) return sign < 0 ? -0 : 0;
  if (exponent === 0) return sign * mantissa * 2 ** -9;
  if (exponent === 0x0f && mantissa === 0x07) return Number.NaN;
  return sign * (1 + mantissa / 8) * 2 ** (exponent - 7);
}

function semanticFp8E5M2ToFloat32(value: number, sign: number): number {
  const exponent = (value >>> 2) & 0x1f;
  const mantissa = value & 0x03;
  if (exponent === 0 && mantissa === 0) return sign < 0 ? -0 : 0;
  if (exponent === 0) return sign * mantissa * 2 ** -16;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 4) * 2 ** (exponent - 15);
}

function semanticFloat32ToFp8(value: number, saturate: number, mode: number): number {
  return (Math.trunc(mode) >>> 0) === 1
    ? semanticFloat32ToFp8Format(value, saturate, { mantissaBits: 2, bias: 15, maxExponent: 30, maxMantissa: 3, nanBits: 0x7f, infBits: 0x7c })
    : semanticFloat32ToFp8Format(value, saturate, { mantissaBits: 3, bias: 7, maxExponent: 15, maxMantissa: 6, nanBits: 0x7f });
}

function semanticFloat32ToFp8Format(
  value: number,
  saturate: number,
  format: {
    readonly mantissaBits: number;
    readonly bias: number;
    readonly maxExponent: number;
    readonly maxMantissa: number;
    readonly nanBits: number;
    readonly infBits?: number;
  },
): number {
  if (Number.isNaN(value)) return format.nanBits;
  const signBit = Object.is(value, -0) || value < 0 ? 0x80 : 0;
  let magnitude = Math.abs(value);
  if (magnitude === 0) return signBit;
  const mantissaScale = 1 << format.mantissaBits;
  const maxFinite = (1 + format.maxMantissa / mantissaScale) * 2 ** (format.maxExponent - format.bias);
  if (magnitude > maxFinite) {
    if ((Math.trunc(saturate) >>> 0) === 1) magnitude = maxFinite;
    else return signBit | (format.infBits ?? format.nanBits);
  }
  const rawExponent = Math.floor(Math.log2(magnitude));
  let exponent = rawExponent + format.bias;
  if (exponent <= 0) {
    const mantissa = Math.max(0, Math.min(format.maxMantissa, roundTiesToEvenNumber(magnitude / 2 ** (1 - format.bias) * mantissaScale)));
    return signBit | mantissa;
  }
  let mantissa = roundTiesToEvenNumber((magnitude / 2 ** rawExponent - 1) * mantissaScale);
  if (mantissa === mantissaScale) {
    exponent++;
    mantissa = 0;
  }
  if (exponent > format.maxExponent || (exponent === format.maxExponent && mantissa > format.maxMantissa)) {
    if ((Math.trunc(saturate) >>> 0) !== 1) return signBit | (format.infBits ?? format.nanBits);
    exponent = format.maxExponent;
    mantissa = format.maxMantissa;
  }
  return signBit | (exponent << format.mantissaBits) | mantissa;
}

function evalNextafter(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (Object.is(x, y) || x === y) return y;
  if (x === 0) return uintBitsToFloat32((y < 0 || Object.is(y, -0)) ? 0x80000001 : 0x00000001);
  let bits = float32ToUintBits(x);
  bits = x > 0
    ? (x < y ? bits + 1 : bits - 1)
    : (x < y ? bits - 1 : bits + 1);
  return uintBitsToFloat32(bits >>> 0);
}

function evalSemanticFfs(value: number): number {
  const bits = Math.trunc(value) >>> 0;
  return bits === 0 ? 0 : 32 - Math.clz32(bits & -bits);
}

function popCount32(value: number): number {
  let bits = Math.trunc(value) >>> 0;
  bits -= (bits >>> 1) & 0x55555555;
  bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
  return (((bits + (bits >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function reverseBits32(value: number): number {
  let bits = Math.trunc(value) >>> 0;
  bits = ((bits >>> 1) & 0x55555555) | ((bits & 0x55555555) << 1);
  bits = ((bits >>> 2) & 0x33333333) | ((bits & 0x33333333) << 2);
  bits = ((bits >>> 4) & 0x0f0f0f0f) | ((bits & 0x0f0f0f0f) << 4);
  bits = ((bits >>> 8) & 0x00ff00ff) | ((bits & 0x00ff00ff) << 8);
  return ((bits >>> 16) | (bits << 16)) >>> 0;
}

function signedMulHi32(xValue: number, yValue: number): number {
  const x = BigInt(Math.trunc(xValue) | 0);
  const y = BigInt(Math.trunc(yValue) | 0);
  return Number((x * y) >> 32n) | 0;
}

function unsignedMulHi32(xValue: number, yValue: number): number {
  const x = BigInt(Math.trunc(xValue) >>> 0);
  const y = BigInt(Math.trunc(yValue) >>> 0);
  return Number((x * y) >> 32n) >>> 0;
}

function bytePerm(xValue: number, yValue: number, selectorValue: number): number {
  const x = Math.trunc(xValue) >>> 0;
  const y = Math.trunc(yValue) >>> 0;
  const selector = Math.trunc(selectorValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const source = (selector >>> (lane * 4)) & 0x7;
    const input = source < 4 ? x : y;
    out |= ((input >>> ((source & 3) * 8)) & 0xff) << (lane * 8);
  }
  return out >>> 0;
}

function funnelShiftLeft(loValue: number, hiValue: number, shiftValue: number): number {
  const lo = Math.trunc(loValue) >>> 0;
  const hi = Math.trunc(hiValue) >>> 0;
  const shift = Math.trunc(shiftValue);
  if (shift <= 0) return lo;
  if (shift >= 32) return hi;
  return ((lo << shift) | (hi >>> (32 - shift))) >>> 0;
}

function funnelShiftRight(loValue: number, hiValue: number, shiftValue: number): number {
  const lo = Math.trunc(loValue) >>> 0;
  const hi = Math.trunc(hiValue) >>> 0;
  const shift = Math.trunc(shiftValue);
  if (shift <= 0) return lo;
  if (shift >= 32) return hi;
  return ((lo >>> shift) | (hi << (32 - shift))) >>> 0;
}

function roundedSignedAverage(xValue: number, yValue: number): number {
  const x = BigInt(Math.trunc(xValue) | 0);
  const y = BigInt(Math.trunc(yValue) | 0);
  return Number((x + y + 1n) >> 1n) | 0;
}

function signedAverage(xValue: number, yValue: number): number {
  const x = BigInt(Math.trunc(xValue) | 0);
  const y = BigInt(Math.trunc(yValue) | 0);
  return Number((x + y) >> 1n) | 0;
}

function unsignedAverage(xValue: number, yValue: number): number {
  const x = Math.trunc(xValue) >>> 0;
  const y = Math.trunc(yValue) >>> 0;
  return ((x & y) + ((x ^ y) >>> 1)) >>> 0;
}

function roundedUnsignedAverage(xValue: number, yValue: number): number {
  const x = Math.trunc(xValue) >>> 0;
  const y = Math.trunc(yValue) >>> 0;
  return ((x & y) + ((x ^ y) >>> 1) + ((x ^ y) & 1)) >>> 0;
}

function u8x4SadAdd(aValue: number, bValue: number, addValue = 0): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = Math.trunc(addValue) >>> 0;
  for (let lane = 0; lane < 4; lane++) {
    out = (out + Math.abs(((a >>> (lane * 8)) & 0xff) - ((b >>> (lane * 8)) & 0xff))) >>> 0;
  }
  return out;
}

function u8x4Binary(aValue: number, bValue: number, op: (a: number, b: number) => number): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const laneValue = op((a >>> shift) & 0xff, (b >>> shift) & 0xff) & 0xff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function u16x2Binary(aValue: number, bValue: number, op: (a: number, b: number) => number): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 2; lane++) {
    const shift = lane * 16;
    const laneValue = op((a >>> shift) & 0xffff, (b >>> shift) & 0xffff) & 0xffff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function i8x4Binary(aValue: number, bValue: number, op: (a: number, b: number) => number): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const laneValue = op(signExtend8((a >>> shift) & 0xff), signExtend8((b >>> shift) & 0xff)) & 0xff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function i16x2Binary(aValue: number, bValue: number, op: (a: number, b: number) => number): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 2; lane++) {
    const shift = lane * 16;
    const laneValue = op(signExtend16((a >>> shift) & 0xffff), signExtend16((b >>> shift) & 0xffff)) & 0xffff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function viaddScalar(aValue: number, bValue: number, cValue: number, signed: boolean, choose: "max" | "min", relu: boolean): number {
  const add = signed
    ? ((Math.trunc(aValue) | 0) + (Math.trunc(bValue) | 0)) | 0
    : ((Math.trunc(aValue) >>> 0) + (Math.trunc(bValue) >>> 0)) >>> 0;
  const c = signed ? Math.trunc(cValue) | 0 : Math.trunc(cValue) >>> 0;
  const selected = choose === "max" ? Math.max(add, c) : Math.min(add, c);
  const value = relu ? Math.max(selected, 0) : selected;
  return signed ? value | 0 : value >>> 0;
}

function viadd16x2(aValue: number, bValue: number, cValue: number, signed: boolean, choose: "max" | "min", relu: boolean): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const c = Math.trunc(cValue) >>> 0;
  let out = 0;
  for (let shift = 0; shift < 32; shift += 16) {
    const leftBits = (a >>> shift) & 0xffff;
    const rightBits = (b >>> shift) & 0xffff;
    const cmpBits = (c >>> shift) & 0xffff;
    const left = signed ? signExtend16(leftBits) : leftBits;
    const right = signed ? signExtend16(rightBits) : rightBits;
    const cmp = signed ? signExtend16(cmpBits) : cmpBits;
    const selected = choose === "max" ? Math.max(left + right, cmp) : Math.min(left + right, cmp);
    const value = relu ? Math.max(selected, 0) : selected;
    out = (out | ((value & 0xffff) << shift)) >>> 0;
  }
  return out >>> 0;
}

function viMinMaxScalar(values: readonly number[], signed: boolean, choose: "max" | "min", relu: boolean): number {
  const operands = values.map((value) => signed ? Math.trunc(value) | 0 : Math.trunc(value) >>> 0);
  const selected = choose === "max" ? Math.max(...operands) : Math.min(...operands);
  const value = relu ? Math.max(selected, 0) : selected;
  return signed ? value | 0 : value >>> 0;
}

function viMinMax16x2(values: readonly number[], signed: boolean, choose: "max" | "min", relu: boolean): number {
  const operands = values.map((value) => Math.trunc(value) >>> 0);
  let out = 0;
  for (let shift = 0; shift < 32; shift += 16) {
    const lanes = operands.map((operand) => {
      const bits = (operand >>> shift) & 0xffff;
      return signed ? signExtend16(bits) : bits;
    });
    const selected = choose === "max" ? Math.max(...lanes) : Math.min(...lanes);
    const value = relu ? Math.max(selected, 0) : selected;
    out = (out | ((value & 0xffff) << shift)) >>> 0;
  }
  return out >>> 0;
}

function packedUnary(value: number, laneWidth: 8 | 16, signed: boolean, op: (a: number) => number): number {
  const input = Math.trunc(value) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  let out = 0;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const bits = (input >>> shift) & mask;
    const lane = signed ? laneWidth === 8 ? signExtend8(bits) : signExtend16(bits) : bits;
    out = (out | ((op(lane) & mask) << shift)) >>> 0;
  }
  return out >>> 0;
}

function packedSad(aValue: number, bValue: number, laneWidth: 8 | 16, signed: boolean): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  let out = 0;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const leftBits = (a >>> shift) & mask;
    const rightBits = (b >>> shift) & mask;
    const left = signed ? laneWidth === 8 ? signExtend8(leftBits) : signExtend16(leftBits) : leftBits;
    const right = signed ? laneWidth === 8 ? signExtend8(rightBits) : signExtend16(rightBits) : rightBits;
    out += Math.abs(left - right);
  }
  return out >>> 0;
}

function vset(aValue: number, bValue: number, laneWidth: 8 | 16, signed: boolean, op: (a: number, b: number) => boolean): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const leftBits = (a >>> shift) & mask;
    const rightBits = (b >>> shift) & mask;
    const left = signed ? laneWidth === 8 ? signExtend8(leftBits) : signExtend16(leftBits) : leftBits;
    const right = signed ? laneWidth === 8 ? signExtend8(rightBits) : signExtend16(rightBits) : rightBits;
    if (!op(left, right)) return 0;
  }
  return 1;
}

function vcompare(aValue: number, bValue: number, laneWidth: 8 | 16, signed: boolean, op: (a: number, b: number) => boolean): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  let out = 0;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const leftBits = (a >>> shift) & mask;
    const rightBits = (b >>> shift) & mask;
    const left = signed ? laneWidth === 8 ? signExtend8(leftBits) : signExtend16(leftBits) : leftBits;
    const right = signed ? laneWidth === 8 ? signExtend8(rightBits) : signExtend16(rightBits) : rightBits;
    if (op(left, right)) out |= mask << shift;
  }
  return out >>> 0;
}

function i8x4SaturatingBinary(aValue: number, bValue: number, op: (a: number, b: number) => number): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const laneValue = Math.min(127, Math.max(-128, op(signExtend8((a >>> shift) & 0xff), signExtend8((b >>> shift) & 0xff)))) & 0xff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function i16x2SaturatingBinary(aValue: number, bValue: number, op: (a: number, b: number) => number): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 2; lane++) {
    const shift = lane * 16;
    const laneValue = Math.min(32767, Math.max(-32768, op(signExtend16((a >>> shift) & 0xffff), signExtend16((b >>> shift) & 0xffff)))) & 0xffff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function i8x4DotAdd(aValue: number, bValue: number, addValue = 0): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = Math.trunc(addValue) | 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    out = (out + Math.imul(signExtend8((a >>> shift) & 0xff), signExtend8((b >>> shift) & 0xff))) | 0;
  }
  return out;
}

function u8x4DotAdd(aValue: number, bValue: number, addValue = 0): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = Math.trunc(addValue) >>> 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    out = (out + (((a >>> shift) & 0xff) * ((b >>> shift) & 0xff))) >>> 0;
  }
  return out;
}

function signExtend8(value: number): number {
  return (value << 24) >> 24;
}

function i16x2I8x2DotAdd(aValue: number, bValue: number, addValue = 0, byteShift: 0 | 16): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const left0 = signExtend16(a & 0xffff);
  const left1 = signExtend16((a >>> 16) & 0xffff);
  const right0 = signExtend8((b >>> byteShift) & 0xff);
  const right1 = signExtend8((b >>> (byteShift + 8)) & 0xff);
  return ((Math.trunc(addValue) | 0) + Math.imul(left0, right0) + Math.imul(left1, right1)) | 0;
}

function u16x2U8x2DotAdd(aValue: number, bValue: number, addValue = 0, byteShift: 0 | 16): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const left0 = a & 0xffff;
  const left1 = (a >>> 16) & 0xffff;
  const right0 = (b >>> byteShift) & 0xff;
  const right1 = (b >>> (byteShift + 8)) & 0xff;
  return ((Math.trunc(addValue) >>> 0) + (left0 * right0) + (left1 * right1)) >>> 0;
}

function signExtend16(value: number): number {
  return (value << 16) >> 16;
}

function evalSemanticErf(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (!Number.isFinite(value)) return Math.sign(value);
  const sign = value < 0 ? -1 : 1;
  const magnitude = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * magnitude);
  const polynomial = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-magnitude * magnitude));
}

function evalSemanticErfinv(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === -1) return Number.NEGATIVE_INFINITY;
  if (value === 1) return Number.POSITIVE_INFINITY;
  if (value < -1 || value > 1) return NaN;
  if (value === 0) return 0;
  const a = 0.147;
  const log = Math.log(1 - value * value);
  const first = (2 / (Math.PI * a)) + (log / 2);
  let estimate = Math.sign(value) * Math.sqrt(Math.sqrt(first * first - log / a) - first);
  for (let i = 0; i < 2; i++) {
    estimate -= (evalSemanticErf(estimate) - value) / (1.1283791670955126 * Math.exp(-estimate * estimate));
  }
  return estimate;
}

function evalSemanticNormcdfinv(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === 0) return Number.NEGATIVE_INFINITY;
  if (value === 1) return Number.POSITIVE_INFINITY;
  if (value < 0 || value > 1) return NaN;
  return Math.SQRT2 * evalSemanticErfinv(2 * value - 1);
}

const SEMANTIC_LANCZOS_COEFFICIENTS = [
  0.9999999999998099,
  676.5203681218851,
  -1259.1392167224028,
  771.3234287776531,
  -176.6150291621406,
  12.507343278686905,
  -0.13857109526572012,
  9.984369578019572e-6,
  1.5056327351493116e-7,
] as const;

function evalSemanticGamma(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === Infinity) return Infinity;
  if (value === -Infinity) return NaN;
  if (value <= 0 && Number.isInteger(value)) return NaN;
  if (value < 0.5) return Math.PI / (Math.sin(Math.PI * value) * evalSemanticGamma(1 - value));
  const z = value - 1;
  let x = SEMANTIC_LANCZOS_COEFFICIENTS[0];
  for (let i = 1; i < SEMANTIC_LANCZOS_COEFFICIENTS.length; i++) x += SEMANTIC_LANCZOS_COEFFICIENTS[i]! / (z + i);
  const t = z + 7.5;
  return Math.sqrt(2 * Math.PI) * (t ** (z + 0.5)) * Math.exp(-t) * x;
}

function evalSemanticLgamma(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (!Number.isFinite(value)) return Infinity;
  if (value <= 0 && Number.isInteger(value)) return Infinity;
  return Math.log(Math.abs(evalSemanticGamma(value)));
}

function frexpMantissa(value: number): number {
  const exponent = frexpExponent(value);
  return exponent === 0 ? value : value / 2 ** exponent;
}

function roundTiesToEvenNumber(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function roundAwayFromZero(value: number): number {
  if (!Number.isFinite(value)) return value;
  const magnitude = Math.floor(Math.abs(value) + 0.5);
  return value < 0 || Object.is(value, -0) ? -magnitude : magnitude;
}

function evalLogb(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === 0) return -Infinity;
  if (!Number.isFinite(value)) return Infinity;
  return Math.floor(Math.log2(Math.abs(value)));
}

function evalIlogb(value: number): number {
  if (Number.isNaN(value) || !Number.isFinite(value)) return 2147483647;
  if (value === 0) return -2147483648;
  return Math.floor(Math.log2(Math.abs(value))) | 0;
}

function modfIntpart(value: number): number {
  return Number.isFinite(value) ? Math.trunc(value) : value;
}

function modfFraction(value: number): number {
  if (Number.isNaN(value)) return value;
  if (!Number.isFinite(value)) return value < 0 ? -0 : 0;
  return value - Math.trunc(value);
}

function evalSemanticFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference function call requires symbol callee", expression.span);
  const callee = expression.callee.name;
  const fn = context.compiled.kernelIr.functions.find((item) => item.name === callee);
  if (!fn) throw semanticReferenceError(`semantic reference unknown function '${callee}'`, expression.span);
  const child = runSemanticFunction(fn, expression.args, context, expression.span);
  if (child.returnValue === undefined) {
    throw semanticReferenceError(`semantic reference function '${fn.name}' did not return value`, fn.span);
  }
  return typeof child.returnValue === "number"
    ? coerceSemanticScalarValue(child.returnValue, fn.returnType)
    : child.returnValue;
}

function evalSemanticCooperativeGroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") {
    throw semanticReferenceError("semantic reference cooperative-group call requires symbol receiver", expression.span);
  }
  if (
    expression.callee.property !== "thread_rank" &&
    expression.callee.property !== "size" &&
    expression.callee.property !== "meta_group_rank" &&
    expression.callee.property !== "meta_group_size"
  ) {
    throw semanticReferenceError("semantic reference cooperative-group call requires rank, size, or meta-group topology", expression.span);
  }
  return semanticReferenceCooperativeGroupValue(expression.callee.object.name, expression.callee.property, context);
}

function evalSemanticCooperativeReduceCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  const [groupArg, valueArg] = expression.args;
  if (groupArg?.kind !== "symbol" || !valueArg) {
    throw semanticReferenceError("semantic cooperative reduce requires group and value", expression.span);
  }
  return semanticReferenceCooperativeContexts(groupArg.name, context)
    .reduce((sum, peer) => sum + evalNumber(valueArg, peer), 0);
}

function semanticReferenceCooperativeGroupValue(
  name: string,
  property: "thread_rank" | "size" | "meta_group_rank" | "meta_group_size",
  context: SemanticReferenceContext,
): number {
  const group = semanticCooperativeGroupInfo(context.compiled.kernelIr, name);
  if (!group) throw semanticReferenceError(`unknown cooperative group '${name}'`, context.compiled.kernelIr.span);
  const workgroupSize = context.blockDim.x * context.blockDim.y * context.blockDim.z;
  const localRank = context.threadIdx.x + context.blockDim.x * (context.threadIdx.y + context.blockDim.y * context.threadIdx.z);
  if (property === "meta_group_rank") return group.kind === "tile" ? Math.floor(localRank / (group.tileSize ?? 32)) : 0;
  if (property === "meta_group_size") return group.kind === "tile" ? Math.ceil(workgroupSize / (group.tileSize ?? 32)) : 1;
  const localName = property === "thread_rank"
    ? semanticCooperativeGroupRankParamName(name)
    : semanticCooperativeGroupSizeParamName(name);
  const local = context.locals.get(localName);
  if (typeof local === "number") return local;
  if (group.partitioned && (property === "thread_rank" || property === "size")) {
    const peers = semanticReferenceCooperativeContexts(name, context);
    if (property === "size") return peers.length;
    return peers.findIndex((peer) => semanticLocalLinearRank(peer) === localRank);
  }
  if (property === "thread_rank") {
    if (group.kind === "grid") {
      const blockRank = context.blockIdx.x + context.gridDim.x * (context.blockIdx.y + context.gridDim.y * context.blockIdx.z);
      return localRank + workgroupSize * blockRank;
    }
    if (group.kind === "tile") return localRank % (group.tileSize ?? 32);
    return localRank;
  }
  if (group.kind === "grid") return workgroupSize * context.gridDim.x * context.gridDim.y * context.gridDim.z;
  if (group.kind === "tile") return group.tileSize ?? 32;
  return workgroupSize;
}

function semanticReferenceCooperativeContexts(
  name: string,
  context: SemanticReferenceContext,
): readonly SemanticReferenceContext[] {
  const group = semanticCooperativeGroupInfo(context.compiled.kernelIr, name);
  if (!group) return [];
  const parent = group.partitionParent
    ? semanticCooperativeGroupInfo(context.compiled.kernelIr, group.partitionParent)
    : undefined;
  const tileSize = group.tileSize ?? parent?.tileSize ?? 32;
  const rank = semanticLocalLinearRank(context);
  const base = Math.floor(rank / tileSize) * tileSize;
  const peers = context.blockContexts.filter((peer) => {
    const peerRank = semanticLocalLinearRank(peer);
    return peerRank >= base && peerRank < base + tileSize;
  });
  if (!group.partitioned || !group.partitionPredicate) return peers;
  const membershipName = semanticPartitionMembershipName(name);
  const selected = context.locals.get(membershipName);
  if (typeof selected !== "number") {
    throw semanticReferenceError(`partition membership for '${name}' was not initialized`, context.compiled.kernelIr.span);
  }
  return peers.filter((peer) => peer.locals.get(membershipName) === selected);
}

function recordSemanticPartitionMembership(
  declaration: Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }>["declaration"],
  context: SemanticReferenceContext,
): void {
  if (!declaration.partitionPredicate) return;
  context.locals.set(
    semanticPartitionMembershipName(declaration.name),
    truthy(evalNumber(declaration.partitionPredicate, context)) ? 1 : 0,
  );
}

function semanticPartitionMembershipName(name: string): string {
  return `${name}__bg_partition_membership`;
}

function evalSemanticVectorConstructor(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  const valueType = expression.callee.kind === "symbol" ? cudaVectorConstructorType(expression.callee.name) : undefined;
  if (!isSemanticFloatVectorType(valueType)) throw semanticReferenceError("semantic reference vector constructor requires vector target", expression.span);
  const targetLanes = cudaVectorLaneCount(valueType);
  const values = expression.args.map((arg) => evalSemanticExpression(arg, context));
  if (values.length === 1 && typeof values[0] === "number") return Array.from({ length: targetLanes }, () => values[0] as number);
  const lanes = values.flatMap((value) => Array.isArray(value) ? value : [value]);
  return Array.from({ length: targetLanes }, (_, lane) => {
    const value = lanes[lane];
    return typeof value === "number" ? value : 0;
  });
}

function evalSemanticVectorAtCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  const [target, indexExpression] = expression.args;
  if (!target || !indexExpression) throw semanticReferenceError("semantic reference vec_at requires vector and index", expression.span);
  const value = evalSemanticExpression(target, context);
  if (!Array.isArray(value)) throw semanticReferenceError("semantic reference vec_at target is not a vector", target.span);
  const index = Math.trunc(evalNumber(indexExpression, context));
  return value[index] ?? 0;
}

function evalSemanticVectorLerpCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number[] {
  const [leftExpression, rightExpression, amountExpression] = expression.args;
  if (!leftExpression || !rightExpression || !amountExpression) throw semanticReferenceError("semantic reference vector lerp requires three operands", expression.span);
  const left = evalSemanticExpression(leftExpression, context);
  const right = evalSemanticExpression(rightExpression, context);
  if (!Array.isArray(left) || !Array.isArray(right)) throw semanticReferenceError("semantic reference vector lerp requires vector endpoints", expression.span);
  const amount = evalNumber(amountExpression, context);
  return Array.from({ length: Math.max(left.length, right.length) }, (_, lane) => {
    const start = left[lane] ?? 0;
    return start + amount * ((right[lane] ?? 0) - start);
  });
}

function evalSemanticHalf2Call(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference half2 call requires symbol callee", expression.span);
  const name = expression.callee.name;
  const half2Arg = (index: number): number[] => {
    const arg = expression.args[index];
    if (!arg) throw semanticReferenceError(`${name} missing half2 operand`, expression.span);
    const value = evalSemanticExpression(arg, context);
    if (!Array.isArray(value)) throw semanticReferenceError(`${name} expects half2 operand`, arg.span);
    return [value[0] ?? 0, value[1] ?? 0];
  };
  const scalarArg = (index: number): number => {
    const arg = expression.args[index];
    if (!arg) throw semanticReferenceError(`${name} missing scalar operand`, expression.span);
    return evalNumber(arg, context);
  };
  if (isSemanticHalf2UnaryCall(name)) {
    const value = half2Arg(0);
    return value.map((lane) => roundSemanticHalf(evalSemanticHalf2UnaryLane(name, lane)));
  }
  if (isSemanticHalf2ComparisonCall(name)) {
    const left = half2Arg(0);
    const right = half2Arg(1);
    const lanes = semanticHalf2ComparisonLanes(name, left, right);
    if (isSemanticHalf2BooleanComparisonCall(name)) return lanes.every(Boolean) ? 1 : 0;
    if (isSemanticHalf2MaskComparisonCall(name)) return ((lanes[0] ? 0xffff : 0) | (lanes[1] ? 0xffff0000 : 0)) >>> 0;
    return lanes.map((lane) => lane ? 1 : 0);
  }
  if (name === "__hadd2" || name === "__hadd2_rn" || name === "__hadd2_sat" || name === "__hsub2" || name === "__hsub2_rn" || name === "__hsub2_sat" || name === "__hmul2" || name === "__hmul2_rn" || name === "__hmul2_sat" || name === "__hmin2" || name === "__hmax2" || name === "__hmin2_nan" || name === "__hmax2_nan") {
    const left = half2Arg(0);
    const right = half2Arg(1);
    return [0, 1].map((lane) => {
      const lhs = left[lane] ?? 0;
      const rhs = right[lane] ?? 0;
      if (name === "__hadd2" || name === "__hadd2_rn") return roundSemanticHalf(lhs + rhs);
      if (name === "__hadd2_sat") return saturateSemanticHalf(lhs + rhs);
      if (name === "__hsub2" || name === "__hsub2_rn") return roundSemanticHalf(lhs - rhs);
      if (name === "__hsub2_sat") return saturateSemanticHalf(lhs - rhs);
      if (name === "__hmul2" || name === "__hmul2_rn") return roundSemanticHalf(lhs * rhs);
      if (name === "__hmul2_sat") return saturateSemanticHalf(lhs * rhs);
      if (name === "__hmin2" || name === "__hmin2_nan") return roundSemanticHalf(Math.min(lhs, rhs));
      return roundSemanticHalf(Math.max(lhs, rhs));
    });
  }
  if (name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat") {
    const left = half2Arg(0);
    const right = half2Arg(1);
    const addend = half2Arg(2);
    return [0, 1].map((lane) => {
      const value = (left[lane] ?? 0) * (right[lane] ?? 0) + (addend[lane] ?? 0);
      return name === "__hfma2_sat" ? saturateSemanticHalf(value) : roundSemanticHalf(value);
    });
  }
  if (name === "__half22float2") return half2Arg(0);
  if (name === "__float22half2_rn") return half2Arg(0).map(roundSemanticHalf);
  if (name === "__half2_as_uint") {
    const value = half2Arg(0);
    const low = float32ToFloat16Bits(value[0] ?? 0) & 0xffff;
    const high = float32ToFloat16Bits(value[1] ?? 0) & 0xffff;
    return ((high << 16) | low) >>> 0;
  }
  if (name === "__uint_as_half2") {
    const bits = Math.trunc(scalarArg(0)) >>> 0;
    return [float16BitsToFloat32(bits & 0xffff), float16BitsToFloat32(bits >>> 16)];
  }
  if (name === "__low2half") return roundSemanticHalf(half2Arg(0)[0] ?? 0);
  if (name === "__high2half") return roundSemanticHalf(half2Arg(0)[1] ?? 0);
  if (name === "__low2float") return half2Arg(0)[0] ?? 0;
  if (name === "__high2float") return half2Arg(0)[1] ?? 0;
  if (name === "__halves2half2") return [roundSemanticHalf(scalarArg(0)), roundSemanticHalf(scalarArg(1))];
  if (name === "__half2half2") {
    const value = roundSemanticHalf(scalarArg(0));
    return [value, value];
  }
  if (name === "__low2half2" || name === "__high2half2") {
    const value = roundSemanticHalf(half2Arg(0)[name === "__low2half2" ? 0 : 1] ?? 0);
    return [value, value];
  }
  if (name === "__lows2half2" || name === "__highs2half2") {
    const lane = name === "__lows2half2" ? 0 : 1;
    return [roundSemanticHalf(half2Arg(0)[lane] ?? 0), roundSemanticHalf(half2Arg(1)[lane] ?? 0)];
  }
  if (name === "__lowhigh2highlow") {
    const value = half2Arg(0);
    return [roundSemanticHalf(value[1] ?? 0), roundSemanticHalf(value[0] ?? 0)];
  }
  if (name === "__float2half2_rn") {
    const value = roundSemanticHalf(scalarArg(0));
    return [value, value];
  }
  if (name === "__floats2half2_rn") return [roundSemanticHalf(scalarArg(0)), roundSemanticHalf(scalarArg(1))];
  throw semanticReferenceError(`semantic reference does not support half2 call '${name}'`, expression.span);
}

function evalSemanticHalf2UnaryLane(name: string, value: number): number {
  switch (name) {
    case "__habs2": return Math.abs(value);
    case "__hceil2": return Math.ceil(value);
    case "__hfloor2": return Math.floor(value);
    case "__hneg2": return -value;
    case "__hrcp2": return 1 / value;
    case "__hrsqrt2": return 1 / Math.sqrt(value);
    case "__hsqrt2": return Math.sqrt(value);
    case "__htrunc2": return Math.trunc(value);
    case "__hisnan2": return Number.isNaN(value) ? 1 : 0;
    default: return value;
  }
}

function semanticHalf2ComparisonLanes(name: string, left: readonly number[], right: readonly number[]): [boolean, boolean] {
  return [0, 1].map((lane) => semanticHalf2ComparisonLane(name, left[lane] ?? 0, right[lane] ?? 0)) as [boolean, boolean];
}

function semanticHalf2ComparisonLane(name: string, left: number, right: number): boolean {
  const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
  const unordered = Number.isNaN(left) || Number.isNaN(right);
  const base = normalized === "__heq2" || normalized === "__hequ2"
    ? left === right
    : normalized === "__hne2" || normalized === "__hneu2"
      ? left !== right
      : normalized === "__hgt2" || normalized === "__hgtu2"
        ? left > right
        : normalized === "__hge2" || normalized === "__hgeu2"
          ? left >= right
          : normalized === "__hlt2" || normalized === "__hltu2"
            ? left < right
            : left <= right;
  return normalized.includes("u2") ? unordered || base : !unordered && base;
}

function evalSemanticBf162Call(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference bf162 call requires symbol callee", expression.span);
  const name = expression.callee.name;
  const vectorArg = (index: number): number[] => {
    const arg = expression.args[index];
    if (!arg) throw semanticReferenceError(`${name} missing bf162 operand`, expression.span);
    const value = evalSemanticExpression(arg, context);
    if (!Array.isArray(value)) throw semanticReferenceError(`${name} expects bf162 operand`, arg.span);
    return [value[0] ?? 0, value[1] ?? 0];
  };
  const scalarArg = (index: number): number => {
    const arg = expression.args[index];
    if (!arg) throw semanticReferenceError(`${name} missing scalar operand`, expression.span);
    return evalNumber(arg, context);
  };
  if (SEMANTIC_BF162_UNARY_VECTOR_CALLS.has(name)) {
    const value = vectorArg(0);
    return value.map((lane) => roundSemanticBfloat16(evalSemanticBf162UnaryLane(name, lane)));
  }
  if (SEMANTIC_BF162_BINARY_VECTOR_CALLS.has(name)) {
    const left = vectorArg(0);
    const right = vectorArg(1);
    return [0, 1].map((lane) => {
      const lhs = left[lane] ?? 0;
      const rhs = right[lane] ?? 0;
      if (name === "__hadd2" || name === "__hadd2_rn") return roundSemanticBfloat16(lhs + rhs);
      if (name === "__hadd2_sat") return saturateSemanticBfloat16(lhs + rhs);
      if (name === "__hsub2" || name === "__hsub2_rn") return roundSemanticBfloat16(lhs - rhs);
      if (name === "__hsub2_sat") return saturateSemanticBfloat16(lhs - rhs);
      if (name === "__hmul2" || name === "__hmul2_rn") return roundSemanticBfloat16(lhs * rhs);
      if (name === "__hmul2_sat") return saturateSemanticBfloat16(lhs * rhs);
      return roundSemanticBfloat16(lhs / rhs);
    });
  }
  if (SEMANTIC_BF162_TERNARY_VECTOR_CALLS.has(name)) {
    const left = vectorArg(0);
    const right = vectorArg(1);
    const addend = vectorArg(2);
    if (name === "__hcmadd") {
      const real = (left[0] ?? 0) * (right[0] ?? 0) - (left[1] ?? 0) * (right[1] ?? 0) + (addend[0] ?? 0);
      const imag = (left[0] ?? 0) * (right[1] ?? 0) + (left[1] ?? 0) * (right[0] ?? 0) + (addend[1] ?? 0);
      return [roundSemanticBfloat16(real), roundSemanticBfloat16(imag)];
    }
    return [0, 1].map((lane) => {
      const value = (left[lane] ?? 0) * (right[lane] ?? 0) + (addend[lane] ?? 0);
      if (name === "__hfma2_sat") return saturateSemanticBfloat16(value);
      if (name === "__hfma2_relu") return reluSemanticBfloat16(value);
      return roundSemanticBfloat16(value);
    });
  }
  if (SEMANTIC_BF162_MINMAX_VECTOR_CALLS.has(name)) {
    const left = vectorArg(0);
    const right = vectorArg(1);
    return [0, 1].map((lane) => {
      const lhs = left[lane] ?? 0;
      const rhs = right[lane] ?? 0;
      if (name === "__hmin2" || name === "__hmin2_nan") return roundSemanticBfloat16(Math.min(lhs, rhs));
      return roundSemanticBfloat16(Math.max(lhs, rhs));
    });
  }
  if (SEMANTIC_BF162_VECTOR_COMPARISON_CALLS.has(name)) {
    if (name === "__hisnan2") return vectorArg(0).map((lane) => roundSemanticBfloat16(Number.isNaN(lane) ? 1 : 0));
    const left = vectorArg(0);
    const right = vectorArg(1);
    return semanticHalf2ComparisonLanes(name, left, right).map((lane) => roundSemanticBfloat16(lane ? 1 : 0));
  }
  if (SEMANTIC_BF162_MASK_COMPARISON_CALLS.has(name) || SEMANTIC_BF162_BOOL_COMPARISON_CALLS.has(name)) {
    const left = vectorArg(0);
    const right = vectorArg(1);
    const lanes = semanticHalf2ComparisonLanes(name, left, right);
    if (SEMANTIC_BF162_BOOL_COMPARISON_CALLS.has(name)) return lanes.every(Boolean) ? 1 : 0;
    return ((lanes[0] ? 0xffff : 0) | (lanes[1] ? 0xffff0000 : 0)) >>> 0;
  }
  if (name === "__bfloat1622float2") return vectorArg(0);
  if (name === "__float22bfloat162_rn") return vectorArg(0).map(roundSemanticBfloat16);
  if (name === "__bfloat162bfloat162" || name === "__float2bfloat162_rn") {
    const value = roundSemanticBfloat16(scalarArg(0));
    return [value, value];
  }
  if (name === "__halves2bfloat162" || name === "__floats2bfloat162_rn") return [roundSemanticBfloat16(scalarArg(0)), roundSemanticBfloat16(scalarArg(1))];
  if (name === "__low2bfloat16") return roundSemanticBfloat16(vectorArg(0)[0] ?? 0);
  if (name === "__high2bfloat16") return roundSemanticBfloat16(vectorArg(0)[1] ?? 0);
  if (name === "__low2float") return vectorArg(0)[0] ?? 0;
  if (name === "__high2float") return vectorArg(0)[1] ?? 0;
  if (name === "__low2bfloat162" || name === "__high2bfloat162") {
    const value = roundSemanticBfloat16(vectorArg(0)[name === "__low2bfloat162" ? 0 : 1] ?? 0);
    return [value, value];
  }
  if (name === "__lows2bfloat162" || name === "__highs2bfloat162") {
    const lane = name === "__lows2bfloat162" ? 0 : 1;
    return [roundSemanticBfloat16(vectorArg(0)[lane] ?? 0), roundSemanticBfloat16(vectorArg(1)[lane] ?? 0)];
  }
  if (name === "__lowhigh2highlow") {
    const value = vectorArg(0);
    return [roundSemanticBfloat16(value[1] ?? 0), roundSemanticBfloat16(value[0] ?? 0)];
  }
  if (name === "__bfloat162_as_uint" || name === "__nv_bfloat162_as_uint") {
    const value = vectorArg(0);
    const low = (float32ToUintBits(roundSemanticBfloat16(value[0] ?? 0)) >>> 16) & 0xffff;
    const high = float32ToUintBits(roundSemanticBfloat16(value[1] ?? 0)) & 0xffff0000;
    return (high | low) >>> 0;
  }
  if (name === "__uint_as_bfloat162" || name === "__uint_as_nv_bfloat162") {
    const bits = Math.trunc(scalarArg(0)) >>> 0;
    return [bfloat16BitsToFloat32(bits & 0xffff), bfloat16BitsToFloat32(bits >>> 16)];
  }
  throw semanticReferenceError(`semantic reference does not support bf162 call '${name}'`, expression.span);
}

function evalSemanticBf162UnaryLane(name: string, value: number): number {
  switch (name) {
    case "__habs2": return Math.abs(value);
    case "__hneg2": return -value;
    case "h2ceil": return Math.ceil(value);
    case "h2floor": return Math.floor(value);
    case "h2rcp": return 1 / value;
    case "h2rsqrt": return 1 / Math.sqrt(value);
    case "h2sqrt": return Math.sqrt(value);
    case "h2trunc": return Math.trunc(value);
    case "h2exp": return Math.exp(value);
    case "h2exp2": return 2 ** value;
    case "h2exp10": return 10 ** value;
    case "h2log": return Math.log(value);
    case "h2log2": return Math.log2(value);
    case "h2log10": return Math.log10(value);
    case "h2sin": return Math.sin(value);
    case "h2cos": return Math.cos(value);
    case "h2tanh":
    case "h2tanh_approx": return Math.tanh(value);
    case "h2rint": return roundTiesToEvenNumber(value);
    default: return value;
  }
}

function semanticReferenceBf162LocalBitsCastSupported(
  expression: Extract<SemanticExpression, { readonly kind: "unary" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  if (expression.operator !== "*" || expression.valueType !== "uint") return false;
  const arg = expression.argument;
  if (arg.kind !== "cast" || !arg.pointer || arg.valueType !== "uint") return false;
  const address = arg.expression;
  if (address.kind !== "unary" || address.operator !== "&" || address.argument.kind !== "symbol") return false;
  const target = address.argument;
  return target.addressSpace === "local" &&
    semanticExpressionVectorValueType(target) === "bf162" &&
    (compiled === undefined || compiled.kernelIr.operations.some((operation) =>
      operation.kind === "declare" &&
      operation.target.name === target.name &&
      operation.target.addressSpace === "local" &&
      operation.target.valueType === "bf162"
    ));
}

function evalSemanticBf162LocalBitsCast(
  expression: Extract<SemanticExpression, { readonly kind: "unary" }>,
  context: SemanticReferenceContext,
): number {
  const cast = expression.argument;
  if (cast.kind !== "cast" || cast.expression.kind !== "unary" || cast.expression.argument.kind !== "symbol") {
    throw semanticReferenceError("semantic reference bf162 bitcast requires local bf162 symbol", expression.span);
  }
  const value = evalSemanticExpression(cast.expression.argument, context);
  if (!Array.isArray(value)) throw semanticReferenceError("semantic reference bf162 bitcast expects vector local", expression.span);
  const low = (float32ToUintBits(roundSemanticBfloat16(value[0] ?? 0)) >>> 16) & 0xffff;
  const high = float32ToUintBits(roundSemanticBfloat16(value[1] ?? 0)) & 0xffff0000;
  return (high | low) >>> 0;
}

function assignLocalVectorMember(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (expression.target.kind !== "member" || expression.target.object.kind !== "symbol") {
    throw semanticReferenceError("semantic reference vector assignment requires local vector member", expression.target.span);
  }
  const current = context.locals.get(expression.target.object.name);
  if (!Array.isArray(current)) throw semanticReferenceError(`missing local vector '${expression.target.object.name}'`, expression.target.span);
  const valueType = semanticExpressionValueType(expression.target.object);
  const lanes = cudaVectorSwizzleIndices(valueType, expression.target.property);
  if (lanes !== undefined && lanes.length > 1) {
    const right = evalSemanticExpression(expression.value, context);
    const assigned = expression.operator === "="
      ? right
      : evalVectorBinary(expression.operator.slice(0, -1), lanes.map((lane) => Number(current[lane] ?? 0)), right, expression.span);
    if (!Array.isArray(assigned)) throw semanticReferenceError("semantic reference vector swizzle assignment requires vector value", expression.value.span);
    const next = [...current];
    lanes.forEach((lane, index) => {
      next[lane] = Number(assigned[index] ?? 0);
    });
    context.locals.set(expression.target.object.name, next);
    return assigned;
  }
  const lane = lanes?.[0] ?? (valueType === undefined ? undefined : cudaVectorFieldIndex(valueType, expression.target.property));
  if (lane === undefined) throw semanticReferenceError("semantic reference vector assignment requires modeled lane", expression.target.span);
  const right = evalNumber(expression.value, context);
  const left = Number(current[lane] ?? 0);
  const binaryOperator = semanticAssignmentBinaryOperator(expression.operator);
  const value = binaryOperator === undefined ? right : evalBinary(binaryOperator, left, right);
  current[lane] = value;
  context.locals.set(expression.target.object.name, current);
  return value;
}

function assignMemoryRef(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  ref: SemanticMemoryRef,
  context: SemanticReferenceContext,
): number {
  const right = evalNumber(expression.value, context);
  const value = applySemanticScalarAssignment(expression.operator, readMemory(ref, context), right, expression.span);
  writeMemory(ref, value, context);
  return value;
}

function createSemanticFunctionContext(
  fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number],
  args: readonly SemanticExpression[],
  context: SemanticReferenceContext,
  span: SourceSpan,
): SemanticReferenceContext {
  const locals = new Map<string, SemanticValue>();
  const textures = { ...context.textures };
  const textureDescriptors = { ...context.textureDescriptors };
  const surfaces = { ...context.surfaces };
  const buffers = new Map(context.buffers);
  const sharedMemory = new Map(context.sharedMemory);
  const sharedOffsets = new Map(context.sharedOffsets);
  const storageOffsets = new Map(context.storageOffsets);
  const localPointerTargets = new Map(context.localPointerTargets);
  for (const [index, param] of fn.params.entries()) {
    const arg = args[index];
    if (!arg) throw semanticReferenceError(`semantic reference function '${fn.name}' missing argument`, span);
    if (param.cooperativeGroupKind !== undefined) {
      if (arg.kind !== "symbol") throw semanticReferenceError(`semantic reference function '${fn.name}' cooperative-group argument must be a symbol`, arg.span);
      locals.set(semanticCooperativeGroupRankParamName(param.name), semanticReferenceCooperativeGroupValue(arg.name, "thread_rank", context));
      locals.set(semanticCooperativeGroupSizeParamName(param.name), semanticReferenceCooperativeGroupValue(arg.name, "size", context));
      continue;
    }
    if (param.addressSpace === "texture") {
      if (arg.kind !== "symbol" || arg.addressSpace !== "texture") throw semanticReferenceError(`semantic reference function '${fn.name}' texture argument must be a texture symbol`, arg.span);
      const texture = context.textures[arg.name];
      if (!texture) throw semanticReferenceError(`missing texture input '${arg.name}'`, arg.span);
      textures[param.name] = texture;
      const descriptor = context.textureDescriptors[arg.name];
      if (descriptor !== undefined) textureDescriptors[param.name] = descriptor;
      continue;
    }
    if (param.addressSpace === "surface") {
      if (arg.kind !== "symbol" || arg.addressSpace !== "surface") throw semanticReferenceError(`semantic reference function '${fn.name}' surface argument must be a surface symbol`, arg.span);
      const surface = context.surfaces[arg.name];
      if (!surface) throw semanticReferenceError(`missing surface input '${arg.name}'`, arg.span);
      surfaces[param.name] = surface;
      continue;
    }
    if (param.pointer && param.addressSpace === "constant" && param.pointerAliasOf !== undefined) continue;
    if (param.pointer && param.addressSpace === "storage") {
      const ref = semanticPointerArgMemoryRef(arg);
      if (!ref || ref.addressSpace !== "storage") throw semanticReferenceError(`semantic reference function '${fn.name}' pointer argument must be modeled storage`, arg.span);
      const buffer = context.buffers.get(ref.base);
      if (!buffer || typeof buffer === "number") throw semanticReferenceError(`missing buffer input '${ref.base}'`, arg.span);
      buffers.set(param.name, buffer);
      storageOffsets.set(param.name, semanticReferencePointerArgBaseIndex(ref, context));
      continue;
    }
    if (param.pointer && param.addressSpace === "shared") {
      const ref = semanticPointerArgMemoryRef(arg);
      if (!ref || ref.addressSpace !== "shared") {
        throw semanticReferenceError(`semantic reference function '${fn.name}' pointer argument must be modeled shared memory`, arg.span);
      }
      const buffer = context.sharedMemory.get(ref.base);
      if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, arg.span);
      sharedMemory.set(param.name, buffer);
      sharedOffsets.set(param.name, semanticReferencePointerArgBaseIndex(ref, context));
      continue;
    }
    if (param.pointer && param.addressSpace === "local") {
      const ref = semanticPointerArgMemoryRef(arg);
      if (ref?.indices.length === 1 && semanticReferenceZeroLiteral(ref.indices[0])) {
        const forwarded = context.localPointerTargets.get(ref.base);
        if (forwarded) {
          localPointerTargets.set(param.name, forwarded);
          continue;
        }
      }
      if (!ref || ref.addressSpace !== "local" || ref.indices.length !== 0) {
        throw semanticReferenceError(`semantic reference function '${fn.name}' pointer argument must be a local scalar`, arg.span);
      }
      localPointerTargets.set(param.name, { ref, context });
      continue;
    }
    locals.set(param.name, isSemanticFloatVectorType(param.valueType)
      ? evalSemanticExpression(arg, context)
      : coerceSemanticScalarValue(evalNumber(arg, context), param.valueType));
  }
  const child: SemanticReferenceContext = {
    compiled: context.compiled,
    buffers,
    constants: context.constants,
    deviceGlobals: context.deviceGlobals,
    textures,
    textureDescriptors,
    surfaces,
    sharedMemory,
    sharedOffsets,
    storageOffsets,
    localPointerTargets,
    scalars: context.scalars,
    vectors: context.vectors,
    locals,
    localDimensions: new Map(),
    blockIdx: context.blockIdx,
    threadIdx: context.threadIdx,
    blockDim: context.blockDim,
    gridDim: context.gridDim,
    blockContexts: context.blockContexts,
    trace: context.trace,
  };
  return child;
}

function semanticReferenceZeroLiteral(expression: SemanticExpression | undefined): boolean {
  return expression?.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
}

function runSemanticFunction(
  fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number],
  args: readonly SemanticExpression[],
  context: SemanticReferenceContext,
  span: SourceSpan,
): SemanticReferenceContext {
  const child = createSemanticFunctionContext(fn, args, context, span);
  const control = execSemanticOperations(fn.body, child);
  if (fn.returnType !== "void" && control !== "return") {
    throw semanticReferenceError(`semantic reference function '${fn.name}' did not return scalar`, fn.span);
  }
  return child;
}

function semanticReferencePointerArgBaseIndex(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  const root = context.compiled.kernelIr.params.find((param) => param.name === ref.base) ??
    context.compiled.kernelIr.memory.find((symbol) => symbol.name === ref.base);
  const index = flatIndex(ref, context);
  const valueType = root?.valueType;
  return ref.pointerBaseIsScalarLane !== true && isSemanticFloatVectorType(valueType) && isSemanticFloatVectorType(ref.valueType)
    ? index * cudaVectorLaneCount(valueType)
    : index;
}

function semanticAtomicCallTarget(expression: Extract<SemanticExpression, { readonly kind: "call" }>): SemanticMemoryRef | undefined {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_ATOMIC_OPS.has(expression.callee.name)) return undefined;
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
  const indices: SemanticExpression[] = [expression.index];
  let target = expression.target;
  while (target.kind === "index") {
    indices.unshift(target.index);
    target = target.target;
  }
  if (target.kind === "cast" && target.pointer) target = target.expression;
  if (target.kind !== "symbol" || (target.addressSpace !== "storage" && target.addressSpace !== "constant" && target.addressSpace !== "device-global" && target.addressSpace !== "local" && target.addressSpace !== "shared")) return undefined;
  return {
    base: target.name,
    addressSpace: target.addressSpace,
    ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
    ...(expression.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(expression.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: expression.pointerBaseUnitBytes }),
    ...(expression.packedByteLanes === undefined ? {} : { packedByteLanes: expression.packedByteLanes }),
    indices,
    fields: [],
    span: expression.span,
  };
}

function readMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  if (semanticReferenceVectorFieldMemoryRefSupported(ref)) {
    const value = readMemoryValue(ref, context);
    if (typeof value !== "number") throw semanticReferenceError("semantic reference scalar read received vector value", ref.span);
    return value;
  }
  if (ref.addressSpace === "local") {
    const target = context.localPointerTargets.get(ref.base);
    if (target) return readMemory(localPointerTargetRef(target.ref, ref), target.context);
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) {
      if (ref.indices.length === 0 && typeof buffer === "number") return buffer;
      throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    }
    if (semanticReferenceLocalPackedHalfView(ref, context.compiled)) {
      const halfIndex = flatIndex(ref, context);
      const word = Number(buffer[Math.trunc(halfIndex / 2)] ?? 0) >>> 0;
      return float16BitsToFloat32(halfIndex % 2 === 0 ? word & 0xffff : word >>> 16);
    }
    const index = flatIndex(ref, context);
    return Number(buffer[index] ?? 0);
  }
  if (ref.addressSpace === "shared") {
    const buffer = context.sharedMemory.get(ref.base);
    if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, ref.span);
    if (semanticReferencePackedSharedByteRoot(ref, context.compiled)) {
      return readPackedSemanticSharedByteView(ref, buffer, context);
    }
    const index = flatIndex(ref, context);
    const ok = index >= 0 && index < buffer.length;
    const value = ok ? Number(buffer[index]) : 0;
    context.trace.sharedReads.push({ name: ref.base, index, value, ok });
    return value;
  }
  if (semanticReferenceDirectByteRawView(ref, context.compiled)) {
    const buffer = context.buffers.get(ref.base);
    if (!buffer || typeof buffer === "number") throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
    const base = flatIndex(ref, context);
    let word = 0;
    for (let byte = 0; byte < 4; byte++) word |= (Number(buffer[base + byte] ?? 0) & 0xff) << (byte * 8);
    const unsigned = word >>> 0;
    if (ref.valueType === "float") return uintBitsToFloat32(unsigned);
    return ref.valueType === "int" ? word | 0 : unsigned;
  }
  const buffer = ref.addressSpace === "constant"
    ? context.constants.get(ref.base)
    : ref.addressSpace === "device-global"
    ? context.deviceGlobals.get(ref.base)
    : context.buffers.get(ref.base);
  if (!buffer || typeof buffer === "number") throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  const index = flatIndex(ref, context);
  const ok = index >= 0 && index < buffer.length;
  const value = ok ? Number(buffer[index]) : 0;
  context.trace.reads.push({ name: ref.base, index, value, ok });
  return value;
}

function readMemoryValue(ref: SemanticMemoryRef, context: SemanticReferenceContext): SemanticValue {
  if (!semanticReferenceVectorFieldMemoryRefSupported(ref)) {
    return isSemanticFloatVectorType(ref.valueType) ? readVectorMemory(ref, context) : readMemory(ref, context);
  }
  const lanes = vectorFieldMemoryLanes(ref);
  if (!lanes) throw semanticReferenceError("semantic reference vector field read requires modeled lanes", ref.span);
  const values = readVectorContainerMemory(ref, context);
  if (lanes.length === 1) return Number(values[lanes[0]!] ?? 0);
  return lanes.map((lane) => Number(values[lane] ?? 0));
}

function writeMemory(ref: SemanticMemoryRef, value: number, context: SemanticReferenceContext): void {
  if (ref.addressSpace === "constant") throw semanticReferenceError(`cannot write constant memory '${ref.base}'`, ref.span);
  if (ref.addressSpace === "local") {
    const target = context.localPointerTargets.get(ref.base);
    if (target) {
      writeMemory(localPointerTargetRef(target.ref, ref), value, target.context);
      return;
    }
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) {
      if (ref.indices.length === 0 && typeof buffer === "number") {
        context.locals.set(ref.base, value);
        return;
      }
      throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    }
    if (semanticReferenceLocalPackedHalfView(ref, context.compiled)) {
      const halfIndex = flatIndex(ref, context);
      const wordIndex = Math.trunc(halfIndex / 2);
      const previous = Number(buffer[wordIndex] ?? 0) >>> 0;
      const bits = float32ToFloat16Bits(value) & 0xffff;
      buffer[wordIndex] = halfIndex % 2 === 0
        ? ((previous & 0xffff0000) | bits) >>> 0
        : ((previous & 0x0000ffff) | (bits << 16)) >>> 0;
      return;
    }
    const index = flatIndex(ref, context);
    if (index >= 0 && index < buffer.length) buffer[index] = value;
    return;
  }
  if (ref.addressSpace === "shared") {
    const buffer = context.sharedMemory.get(ref.base);
    if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, ref.span);
    if (semanticReferencePackedSharedByteRoot(ref, context.compiled)) {
      writePackedSemanticSharedByteView(ref, value, buffer, context);
      return;
    }
    const index = flatIndex(ref, context);
    const ok = index >= 0 && index < buffer.length;
    if (ok) buffer[index] = value;
    context.trace.sharedWrites.push({ name: ref.base, index, value, ok });
    return;
  }
  if (semanticReferenceDirectByteRawView(ref, context.compiled)) {
    const buffer = context.buffers.get(ref.base);
    if (!buffer || typeof buffer === "number") throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
    const base = flatIndex(ref, context);
    const word = ref.valueType === "float" ? float32ToUintBits(value) : Math.trunc(value) >>> 0;
    for (let byte = 0; byte < 4; byte++) {
      const index = base + byte;
      const byteValue = (word >>> (byte * 8)) & 0xff;
      const ok = index >= 0 && index < buffer.length;
      if (ok) buffer[index] = byteValue;
      context.trace.writes.push({ name: ref.base, index, value: byteValue, ok });
    }
    return;
  }
  const buffer = ref.addressSpace === "device-global" ? context.deviceGlobals.get(ref.base) : context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  const index = flatIndex(ref, context);
  const ok = index >= 0 && index < buffer.length;
  if (ok) buffer[index] = value;
  context.trace.writes.push({ name: ref.base, index, value, ok });
}

function localPointerTargetRef(target: SemanticMemoryRef, access: SemanticMemoryRef): SemanticMemoryRef {
  return {
    ...target,
    ...(access.valueType === undefined && target.valueType === undefined ? {} : { valueType: access.valueType ?? target.valueType! }),
    indices: [...target.indices, ...access.indices.filter((index) => !(index.kind === "literal" && index.value === 0))],
    fields: [...target.fields, ...access.fields],
    span: access.span,
  };
}

function packedSemanticSharedByteIndex(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  const elementBytes = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
  const pointer = context.compiled.kernelIr.functions
    .flatMap((fn) => fn.params)
    .find((param) => param.name === ref.base && param.pointer && param.addressSpace === "shared");
  if (!pointer) return flatIndex(ref, context) * (ref.pointerBaseUnitBytes === undefined ? elementBytes : 1);
  if (ref.indices.length > 1) throw semanticReferenceError(`shared pointer '${ref.base}' index rank mismatch`, ref.span);
  const offset = context.sharedOffsets.get(ref.base) ?? 0;
  const index = ref.indices[0] === undefined ? 0 : Math.trunc(evalNumber(ref.indices[0], context));
  return offset + index * (ref.pointerBaseUnitBytes === undefined ? elementBytes : 1);
}

function readPackedSemanticSharedByteView(
  ref: SemanticMemoryRef,
  buffer: WgslTypedArray,
  context: SemanticReferenceContext,
): number {
  const byteIndex = packedSemanticSharedByteIndex(ref, context);
  const byteCount = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
  let bits = 0;
  for (let lane = 0; lane < byteCount; lane++) {
    const index = byteIndex + lane;
    const ok = index >= 0 && index < buffer.length;
    const value = ok ? Number(buffer[index]) & 0xff : 0;
    context.trace.sharedReads.push({ name: ref.base, index, value, ok });
    bits = (bits | value << lane * 8) >>> 0;
  }
  if (ref.valueType === "float") return uintBitsToFloat32(bits);
  if (ref.valueType === "int") return bits | 0;
  return ref.valueType === "uchar" ? bits & 0xff : bits >>> 0;
}

function writePackedSemanticSharedByteView(
  ref: SemanticMemoryRef,
  value: number,
  buffer: WgslTypedArray,
  context: SemanticReferenceContext,
): void {
  const byteIndex = packedSemanticSharedByteIndex(ref, context);
  const byteCount = sizeofCudaType(ref.valueType ?? "uchar") ?? 1;
  const bits = ref.valueType === "float" ? float32ToUintBits(value) : value >>> 0;
  for (let lane = 0; lane < byteCount; lane++) {
    const index = byteIndex + lane;
    const byte = bits >>> lane * 8 & 0xff;
    const ok = index >= 0 && index < buffer.length;
    if (ok) buffer[index] = byte;
    context.trace.sharedWrites.push({ name: ref.base, index, value: byte, ok });
  }
}

function readAtomicMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  const value = readMemory(ref, context);
  return atomicMemoryUsesFloatBits(ref, context) ? float32ToUintBits(value) : value;
}

function writeAtomicMemory(ref: SemanticMemoryRef, value: number, context: SemanticReferenceContext): void {
  writeMemory(ref, atomicMemoryUsesFloatBits(ref, context) ? uintBitsToFloat32(value) : value, context);
}

function atomicMemoryUsesFloatBits(ref: SemanticMemoryRef, context: SemanticReferenceContext): boolean {
  return ref.addressSpace === "storage" &&
    (ref.valueType === "uint" || ref.valueType === "int") &&
    context.compiled.kernelIr.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && param.valueType === "float");
}

function readVectorMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number[] {
  const valueType = semanticStorageVectorType(ref.valueType);
  if (!valueType) throw semanticReferenceError("semantic reference vector read requires vector memory type", ref.span);
  const laneCount = cudaVectorLaneCount(valueType);
  if (ref.addressSpace === "local") {
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    if (semanticReferenceLocalScalarVectorView(ref, context.compiled)) {
      const base = flatIndex(ref, context);
      return Array.from({ length: laneCount }, (_, lane) => Number(buffer[base + lane] ?? 0));
    }
    const value = buffer[flatIndex(ref, context)];
    if (!Array.isArray(value)) throw semanticReferenceError(`local vector array '${ref.base}' contains scalar value`, ref.span);
    return Array.from({ length: laneCount }, (_, lane) => Number(value[lane] ?? 0));
  }
  if (ref.addressSpace === "shared") {
    const buffer = context.sharedMemory.get(ref.base);
    if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, ref.span);
    const base = semanticReferenceSharedVectorBase(ref, context, laneCount);
    return Array.from({ length: laneCount }, (_, lane) => {
      const index = base + lane;
      const ok = index >= 0 && index < buffer.length;
      const value = ok ? Number(buffer[index]) : 0;
      context.trace.sharedReads.push({ name: ref.base, index, value, ok });
      return value;
    });
  }
  const base = flatIndex(ref, context) * semanticReferenceVectorStorageStride(ref, context);
  const buffer = ref.addressSpace === "constant"
    ? context.constants.get(ref.base)
    : ref.addressSpace === "device-global"
    ? context.deviceGlobals.get(ref.base)
    : context.buffers.get(ref.base);
  if (!buffer || typeof buffer === "number") throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  return Array.from({ length: laneCount }, (_, lane) => {
    const index = base + lane;
    const ok = index >= 0 && index < buffer.length;
    const value = ok ? Number(buffer[index]) : 0;
    context.trace.reads.push({ name: ref.base, index, value, ok });
    return value;
  });
}

function writeMemoryValue(ref: SemanticMemoryRef, value: SemanticValue, context: SemanticReferenceContext): void {
  if (semanticReferenceLocalPackedByteRawView(ref, context.compiled)) {
    if (typeof value !== "number") throw semanticReferenceError("semantic reference packed byte-vector write received vector value", ref.span);
    const word = ref.valueType === "float" ? float32ToUintBits(value) : value >>> 0;
    context.locals.set(ref.base, Array.from({ length: 4 }, (_, byte) => (word >>> (byte * 8)) & 0xff));
    return;
  }
  if (semanticReferenceVectorFieldMemoryRefSupported(ref)) {
    const lanes = vectorFieldMemoryLanes(ref);
    if (!lanes) throw semanticReferenceError("semantic reference vector field write requires modeled lanes", ref.span);
    const next = readVectorContainerMemory(ref, context);
    if (lanes.length === 1) {
      if (typeof value !== "number") throw semanticReferenceError("semantic reference scalar vector-field write received vector value", ref.span);
      next[lanes[0]!] = value;
    } else {
      if (!Array.isArray(value)) throw semanticReferenceError("semantic reference swizzle write received scalar value", ref.span);
      lanes.forEach((lane, index) => {
        next[lane] = Number(value[index] ?? 0);
      });
    }
    writeVectorContainerMemory(ref, next, context);
    return;
  }
  const vectorType = semanticStorageVectorType(ref.valueType);
  if (!vectorType) {
    if (typeof value !== "number") throw semanticReferenceError("semantic reference scalar write received vector value", ref.span);
    writeMemory(ref, coerceSemanticScalarValue(value, ref.valueType), context);
    return;
  }
  if (!Array.isArray(value)) throw semanticReferenceError("semantic reference vector write received scalar value", ref.span);
  const laneCount = cudaVectorLaneCount(vectorType);
  if (ref.addressSpace === "local") {
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    const index = flatIndex(ref, context);
    if (index >= 0 && index < buffer.length) {
      (buffer as SemanticValue[])[index] = Array.from({ length: laneCount }, (_, lane) => Number(value[lane] ?? 0));
    }
    return;
  }
  if (ref.addressSpace === "shared") {
    const buffer = context.sharedMemory.get(ref.base);
    if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, ref.span);
    const base = semanticReferenceSharedVectorBase(ref, context, laneCount);
    for (let lane = 0; lane < laneCount; lane++) {
      const index = base + lane;
      const laneValue = value[lane] ?? 0;
      const ok = index >= 0 && index < buffer.length;
      if (ok) buffer[index] = laneValue;
      context.trace.sharedWrites.push({ name: ref.base, index, value: laneValue, ok });
    }
    return;
  }
  const base = flatIndex(ref, context) * semanticReferenceVectorStorageStride(ref, context);
  const buffer = ref.addressSpace === "device-global" ? context.deviceGlobals.get(ref.base) : context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  for (let lane = 0; lane < laneCount; lane++) {
    const index = base + lane;
    const laneValue = ref.packedByteLanes === undefined ? value[lane] ?? 0 : (Number(value[lane] ?? 0) & 0xff);
    const ok = index >= 0 && index < buffer.length;
    if (ok) buffer[index] = laneValue;
    context.trace.writes.push({ name: ref.base, index, value: laneValue, ok });
  }
}

function semanticReferenceDirectByteRawView(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  return ref.addressSpace === "storage" && ref.packedByteLanes === 4 &&
    compiled.kernelIr.params.some((param) => param.name === ref.base && param.valueType === "uchar");
}

function semanticReferenceLocalPackedByteRawView(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  return ref.addressSpace === "local" && ref.packedByteLanes === 4 &&
    semanticReferenceDeclaredPackedByteLocal(compiled.kernelIr.operations, ref.base);
}

function semanticReferenceDeclaredPackedByteLocal(operations: readonly SemanticKernelIrOperation[], name: string): boolean {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.target.name === name) return operation.target.packedByteLanes === 4;
    if (operation.kind === "block" && semanticReferenceDeclaredPackedByteLocal(operation.body, name)) return true;
    if (operation.kind === "branch" && (
      semanticReferenceDeclaredPackedByteLocal(operation.consequent, name) ||
      semanticReferenceDeclaredPackedByteLocal(operation.alternate, name)
    )) return true;
    if (operation.kind === "loop" && semanticReferenceDeclaredPackedByteLocal(operation.body, name)) return true;
  }
  return false;
}

function semanticReferenceSharedVectorBase(
  ref: SemanticMemoryRef,
  context: SemanticReferenceContext,
  laneCount: number,
): number {
  const index = flatIndex(ref, context);
  const root = context.compiled.kernelIr.memory.find((symbol) => symbol.name === ref.base && symbol.kind === "shared");
  if (root !== undefined) return isSemanticFloatVectorType(root.valueType) ? index * laneCount : index;
  const param = context.compiled.kernelIr.functions
    .flatMap((fn) => fn.params)
    .find((symbol) => symbol.name === ref.base && symbol.pointer && symbol.addressSpace === "shared");
  if (!isSemanticFloatVectorType(param?.valueType)) return index;
  const offset = context.sharedOffsets.get(ref.base) ?? 0;
  return offset + (index - offset) * laneCount;
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

function vectorFieldMemoryLanes(ref: SemanticMemoryRef): readonly number[] | undefined {
  return ref.fields.length === 1
    ? semanticStorageVectorFieldIndices(ref.containerValueType, ref.fields[0]!)
    : undefined;
}

function readVectorContainerMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number[] {
  const containerType = semanticStorageVectorType(ref.containerValueType);
  if (!containerType) throw semanticReferenceError("semantic reference vector field requires vector container", ref.span);
  const laneCount = cudaVectorLaneCount(containerType);
  if (ref.addressSpace === "local") {
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    const value = buffer[flatIndex(ref, context)];
    if (Array.isArray(value)) return Array.from({ length: laneCount }, (_, lane) => Number(value[lane] ?? 0));
  }
  const base = flatIndex(ref, context) * laneCount;
  const buffer = ref.addressSpace === "constant"
    ? context.constants.get(ref.base)
    : ref.addressSpace === "device-global"
    ? context.deviceGlobals.get(ref.base)
    : ref.addressSpace === "shared"
    ? context.sharedMemory.get(ref.base)
    : context.buffers.get(ref.base);
  if (!buffer || typeof buffer === "number") throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  return Array.from({ length: laneCount }, (_, lane) => {
    const index = base + lane;
    const ok = index >= 0 && index < buffer.length;
    const value = ok ? Number(buffer[index]) : 0;
    if (ref.addressSpace === "shared") context.trace.sharedReads.push({ name: ref.base, index, value, ok });
    else context.trace.reads.push({ name: ref.base, index, value, ok });
    return value;
  });
}

function writeVectorContainerMemory(ref: SemanticMemoryRef, value: readonly number[], context: SemanticReferenceContext): void {
  const containerType = semanticStorageVectorType(ref.containerValueType);
  if (!containerType) throw semanticReferenceError("semantic reference vector field requires vector container", ref.span);
  const laneCount = cudaVectorLaneCount(containerType);
  if (ref.addressSpace === "local") {
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    const index = flatIndex(ref, context);
    if (index >= 0 && index < buffer.length) (buffer as SemanticValue[])[index] = Array.from({ length: laneCount }, (_, lane) => Number(value[lane] ?? 0));
    return;
  }
  if (ref.addressSpace === "constant") throw semanticReferenceError(`cannot write constant memory '${ref.base}'`, ref.span);
  const base = flatIndex(ref, context) * laneCount;
  const buffer = ref.addressSpace === "device-global"
    ? context.deviceGlobals.get(ref.base)
    : ref.addressSpace === "shared"
    ? context.sharedMemory.get(ref.base)
    : context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  for (let lane = 0; lane < laneCount; lane++) {
    const index = base + lane;
    const laneValue = Number(value[lane] ?? 0);
    const ok = index >= 0 && index < buffer.length;
    if (ok) buffer[index] = laneValue;
    if (ref.addressSpace === "shared") context.trace.sharedWrites.push({ name: ref.base, index, value: laneValue, ok });
    else context.trace.writes.push({ name: ref.base, index, value: laneValue, ok });
  }
}

function semanticReferenceVectorStorageStride(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  const root = context.compiled.kernelIr.params.find((param) => param.name === ref.base) ??
    context.compiled.kernelIr.memory.find((symbol) => symbol.name === ref.base);
  const valueType = semanticStorageVectorType(root?.valueType);
  return valueType === undefined ? 1 : cudaVectorLaneCount(valueType);
}

function flatIndex(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  if (ref.addressSpace === "local" || ref.addressSpace === "shared") {
    const symbol = context.compiled.kernelIr.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace);
    const localDimensions = ref.addressSpace === "local" ? context.localDimensions.get(ref.base) : undefined;
    if (ref.addressSpace === "shared" && !symbol) {
      if (!context.sharedMemory.has(ref.base)) throw semanticReferenceError(`unknown shared array '${ref.base}'`, ref.span);
      const pointerParam = context.compiled.kernelIr.functions
        .flatMap((fn) => fn.params)
        .find((param) => param.name === ref.base && param.pointer && param.addressSpace === "shared");
      if (pointerParam?.dimensions.length === 0) {
        if (ref.indices.length > 1 || ref.indices[0] && Math.trunc(evalNumber(ref.indices[0]!, context)) !== 0) {
          throw semanticReferenceError(`shared scalar pointer '${ref.base}' cannot be indexed`, ref.span);
        }
        return context.sharedOffsets.get(ref.base) ?? 0;
      }
      if (ref.indices.length !== 1) throw semanticReferenceError(`shared pointer '${ref.base}' index rank mismatch`, ref.span);
      return (context.sharedOffsets.get(ref.base) ?? 0) + Math.trunc(evalNumber(ref.indices[0]!, context));
    }
    if (!symbol && !localDimensions) throw semanticReferenceError(`unknown ${ref.addressSpace} array '${ref.base}'`, ref.span);
    if (ref.addressSpace === "shared" && ref.indices.length === 0) return 0;
    const dimensions = ref.addressSpace === "shared"
      ? semanticReferenceSharedDimensions(context.compiled, symbol!)
      : localDimensions ?? symbol!.dimensions;
    if (ref.addressSpace === "local" && ref.indices.length === 1 && dimensions.length > 1) {
      return Math.trunc(evalNumber(ref.indices[0]!, context));
    }
    if (ref.indices.length !== dimensions.length) throw semanticReferenceError(`${ref.addressSpace} array '${ref.base}' index rank mismatch`, ref.span);
    return flatIndexForDimensions(dimensions, ref.indices.map((index) => Math.trunc(evalNumber(index, context))));
  }
  if (ref.addressSpace === "device-global") {
    const symbol = context.compiled.kernelIr.memory.find((item) => item.name === ref.base && item.kind === "device-global");
    if (symbol?.dimensions.length) {
      if (ref.indices.length !== symbol.dimensions.length) throw semanticReferenceError(`device-global array '${ref.base}' index rank mismatch`, ref.span);
      return flatIndexForDimensions(symbol.dimensions, ref.indices.map((index) => Math.trunc(evalNumber(index, context))));
    }
    if (ref.indices.length === 0) return 0;
    if (ref.indices.length === 1) return Math.trunc(evalNumber(ref.indices[0]!, context));
    throw semanticReferenceError("semantic reference supports scalar/1D device-global indexing", ref.span);
  }
  if (ref.addressSpace === "storage") {
    const offset = context.storageOffsets.get(ref.base) ?? 0;
    if (ref.indices.length === 0) return offset;
    return offset + ref.indices.reduce((sum, index) => sum + Math.trunc(evalNumber(index, context)), 0);
  }
  if (ref.addressSpace === "constant") {
    const symbol = context.compiled.kernelIr.memory.find((item) => item.name === ref.base && item.kind === "constant");
    if (symbol?.dimensions.length && ref.indices.length === symbol.dimensions.length) {
      return flatIndexForDimensions(symbol.dimensions, ref.indices.map((index) => Math.trunc(evalNumber(index, context))));
    }
  }
  if (ref.indices.length !== 1) throw semanticReferenceError("semantic reference supports only 1D constant indexing", ref.span);
  return Math.trunc(evalNumber(ref.indices[0]!, context));
}

function symbolValue(name: string, context: SemanticReferenceContext, span: SourceSpan): SemanticValue {
  if (name === "NULL" || name === "nullptr") return 0;
  if (name === "threadIdx") return context.threadIdx;
  if (name === "blockIdx") return context.blockIdx;
  if (name === "blockDim") return context.blockDim;
  if (name === "gridDim") return context.gridDim;
  if (context.locals.has(name)) return context.locals.get(name)!;
  const scalar = context.scalars[name];
  if (scalar !== undefined) return scalar;
  const vector = context.vectors[name];
  if (vector !== undefined) return Array.from(vector, Number);
  const constant = context.constants.get(name);
  if (typeof constant === "number") return constant;
  const constantSymbol = context.compiled.kernelIr.memory.find((symbol) => symbol.name === name && symbol.kind === "constant");
  if (constantSymbol && isSemanticFloatVectorType(constantSymbol.valueType)) {
    return readVectorMemory({
      base: name,
      addressSpace: "constant",
      valueType: constantSymbol.valueType as CudaLiteScalarType,
      indices: [{ kind: "literal", literalKind: "number", value: 0, valueType: "int", span }],
      fields: [],
      span,
    }, context);
  }
  const global = context.compiled.kernelIr.memory.find((symbol) => symbol.name === name && symbol.kind === "device-global");
  if (global && global.dimensions.length === 0) return readMemory({ base: name, addressSpace: "device-global", indices: [], fields: [], span }, context);
  const shared = context.compiled.kernelIr.memory.find((symbol) => symbol.name === name && symbol.kind === "shared");
  if (shared && shared.dimensions.length === 0) return readMemory({ base: name, addressSpace: "shared", indices: [], fields: [], span }, context);
  const storageParam = context.compiled.kernelIr.params.find((param) => param.name === name && param.addressSpace === "storage");
  if (storageParam) return context.buffers.has(name) ? 1 : 0;
  throw semanticReferenceError(`unknown semantic reference symbol '${name}'`, span);
}

function memberValue(
  value: SemanticValue,
  valueType: CudaLiteScalarType | undefined,
  property: string,
  span: SourceSpan,
): SemanticValue {
  if (Array.isArray(value)) {
    const indices = cudaVectorSwizzleIndices(valueType, property);
    if (indices && indices.length > 1) return indices.map((index) => value[index] ?? 0);
    const index = indices?.[0] ?? vectorFieldIndex(property);
    if (index !== undefined) return value[index] ?? 0;
    throw semanticReferenceError(`unsupported semantic member '${property}'`, span);
  }
  if (typeof value === "number") throw semanticReferenceError("semantic member target is not a vector", span);
  if (property === "x") return value.x;
  if (property === "y") return value.y;
  if (property === "z") return value.z;
  throw semanticReferenceError(`unsupported semantic member '${property}'`, span);
}

function vectorFieldIndex(property: string): number | undefined {
  if (property === "x" || property === "r" || property === "s0") return 0;
  if (property === "y" || property === "g" || property === "s1") return 1;
  if (property === "z" || property === "b" || property === "s2") return 2;
  if (property === "w" || property === "a" || property === "s3") return 3;
  return undefined;
}

function evalUnary(operator: string, value: number): number {
  if (operator === "+") return value;
  if (operator === "-") return -value;
  if (operator === "!") return truthy(value) ? 0 : 1;
  if (operator === "~") return ~Math.trunc(value);
  return value;
}

function evalBinary(operator: string, left: number, right: number): number {
  switch (operator) {
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return right === 0 ? 0 : left / right;
    case "%": return right === 0 ? 0 : left % right;
    case "<<": return Math.trunc(left) << Math.trunc(right);
    case ">>": return Math.trunc(left) >> Math.trunc(right);
    case "&": return Math.trunc(left) & Math.trunc(right);
    case "|": return Math.trunc(left) | Math.trunc(right);
    case "^": return Math.trunc(left) ^ Math.trunc(right);
    case "&&": return truthy(left) && truthy(right) ? 1 : 0;
    case "||": return truthy(left) || truthy(right) ? 1 : 0;
    case "==": return left === right ? 1 : 0;
    case "!=": return left !== right ? 1 : 0;
    case "<": return left < right ? 1 : 0;
    case "<=": return left <= right ? 1 : 0;
    case ">": return left > right ? 1 : 0;
    case ">=": return left >= right ? 1 : 0;
    default: return 0;
  }
}

function applySemanticScalarAssignment(
  operator: string,
  left: number,
  right: number,
  span: SourceSpan,
): number {
  if (operator === "=") return right;
  const binaryOperator = semanticAssignmentBinaryOperator(operator);
  if (binaryOperator === undefined) {
    throw semanticReferenceError(`semantic reference does not support assignment '${operator}'`, span);
  }
  return evalBinary(binaryOperator, left, right);
}

function evalVectorBinary(operator: string, left: SemanticValue, right: SemanticValue, span: SourceSpan): number[] {
  if (!semanticReferenceVectorBinaryOperatorSupported(operator)) {
    throw semanticReferenceError(`semantic reference does not support vector binary '${operator}'`, span);
  }
  const leftValues = Array.isArray(left) ? left : typeof left === "number" ? [left] : [];
  const rightValues = Array.isArray(right) ? right : typeof right === "number" ? [right] : [];
  const laneCount = Math.max(leftValues.length, rightValues.length);
  return Array.from({ length: laneCount }, (_, lane) => {
    const leftValue = leftValues.length === 1 ? leftValues[0]! : leftValues[lane] ?? 0;
    const rightValue = rightValues.length === 1 ? rightValues[0]! : rightValues[lane] ?? 0;
    return evalBinary(operator, leftValue, rightValue);
  });
}

function evalSemanticVectorMinMax(name: string, left: SemanticValue, right: SemanticValue, span: SourceSpan): number[] {
  const leftValues = Array.isArray(left) ? left : typeof left === "number" ? [left] : [];
  const rightValues = Array.isArray(right) ? right : typeof right === "number" ? [right] : [];
  const laneCount = Math.max(leftValues.length, rightValues.length);
  if (laneCount === 0) throw semanticReferenceError(`${name} expects numeric operands`, span);
  const chooseMax = name === "fmax" || name === "fmaxf" || name === "max";
  return Array.from({ length: laneCount }, (_, lane) => {
    const leftValue = leftValues.length === 1 ? leftValues[0]! : leftValues[lane] ?? 0;
    const rightValue = rightValues.length === 1 ? rightValues[0]! : rightValues[lane] ?? 0;
    return chooseMax ? Math.max(leftValue, rightValue) : Math.min(leftValue, rightValue);
  });
}

function castNumber(value: number, valueType: CudaLiteScalarType): number {
  if (valueType === "int") return Math.trunc(value);
  if (valueType === "uint") return Math.trunc(value) >>> 0;
  if (valueType === "uchar") return Math.trunc(value) & 0xff;
  if (valueType === "bool") return truthy(value) ? 1 : 0;
  if (valueType === "half") return roundSemanticHalf(value);
  if (valueType === "bf16") return roundSemanticBfloat16(value);
  return value;
}

function coerceSemanticScalarValue(value: number, valueType: CudaLiteScalarType | undefined): number {
  return valueType === "int" || valueType === "uint" || valueType === "uchar" || valueType === "bool"
    ? castNumber(value, valueType)
    : value;
}

function validateSemanticReferenceInput(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): void {
  for (const param of compiled.kernelIr.params) {
    if (param.addressSpace === "storage") {
      const buffer = input.buffers[param.name];
      const valueType = semanticStorageScalarType(param.valueType);
      if (!buffer) throw semanticReferenceError(`missing buffer input '${param.name}'`, param.span);
      if (valueType === "float" && !(buffer instanceof Float32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Float32Array`, param.span);
      }
      if (valueType === "int" && !(buffer instanceof Int32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Int32Array`, param.span);
      }
      if (valueType === "uint" && !(buffer instanceof Uint32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Uint32Array`, param.span);
      }
      if (valueType === "half" && !isWgslFloat16Array(buffer)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Float16Array`, param.span);
      }
      if (valueType === "bf16" && !(buffer instanceof Float32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Float32Array`, param.span);
      }
    } else if (param.addressSpace === "uniform") {
      if (isCudaVectorType(param.valueType)) {
        const vector = input.vectors?.[param.name];
        if (vector === undefined) throw semanticReferenceError(`missing vector input '${param.name}'`, param.span);
        if (vector.length < cudaVectorLaneCount(param.valueType)) {
          throw semanticReferenceError(`vector '${param.name}' has insufficient lanes`, param.span);
        }
      } else if (input.scalars?.[param.name] === undefined) {
        throw semanticReferenceError(`missing scalar input '${param.name}'`, param.span);
      }
    } else if (param.addressSpace === "texture") {
      if (!input.textures?.[param.name]) throw semanticReferenceError(`missing texture input '${param.name}'`, param.span);
    } else if (param.addressSpace === "surface") {
      if (!input.surfaces?.[param.name]) throw semanticReferenceError(`missing surface input '${param.name}'`, param.span);
    } else if (param.addressSpace === "pool" && !semanticOperationsReferenceRoot(compiled.kernelIr.operations, param.name)) {
      continue;
    } else {
      throw semanticReferenceError(`semantic reference does not support ${param.addressSpace} parameter '${param.name}'`, param.span);
    }
  }
  for (const texture of compiled.kernelIr.memory.filter((symbol) => symbol.kind === "texture")) {
    if (!input.textures?.[texture.name]) throw semanticReferenceError(`missing texture input '${texture.name}'`, texture.span);
  }
  for (const constant of compiled.kernelIr.memory.filter((symbol) => symbol.kind === "constant")) {
    if (constant.initialized) continue;
    const value = input.constants?.[constant.name];
    if (value === undefined) throw semanticReferenceError(`missing constant input '${constant.name}'`, constant.span);
    if (constant.dimensions.length === 0 && !isSemanticFloatVectorType(constant.valueType) && typeof value !== "number") {
      throw semanticReferenceError(`constant '${constant.name}' expects scalar number`, constant.span);
    }
    if ((constant.dimensions.length > 0 || isSemanticFloatVectorType(constant.valueType)) && typeof value === "number") {
      throw semanticReferenceError(`constant '${constant.name}' expects typed array`, constant.span);
    }
  }
}

function semanticStorageScalarType(valueType: CudaLiteScalarType | undefined): CudaLiteScalarType | undefined {
  return isCudaVectorType(valueType) ? cudaVectorScalarType(valueType) : valueType;
}

function runSemanticBarrierScheduler(
  operations: readonly SemanticKernelIrOperation[],
  contexts: readonly SemanticReferenceContext[],
  barrierFunctions: ReadonlySet<string>,
): void {
  const generators = contexts.map((context) => execSemanticBarrierOperations(operations, context, barrierFunctions));
  const active = generators.map(() => true);
  const waiting: Array<SemanticBarrierScope | undefined> = generators.map(() => undefined);
  while (active.some(Boolean)) {
    let advanced = false;
    for (const [index, generator] of generators.entries()) {
      if (!active[index] || waiting[index] !== undefined) continue;
      advanced = true;
      const next = generator.next();
      if (next.done) {
        active[index] = false;
        if (next.value === "break" || next.value === "continue") {
          throw semanticReferenceError(`semantic reference unexpected ${next.value} across shared-memory phase`, contexts[index]!.compiled.kernelIr.span);
        }
      } else waiting[index] = next.value;
    }

    const activeIndices = active.flatMap((isActive, index) => isActive ? [index] : []);
    if (activeIndices.length === 0) break;
    let released = false;
    for (const scope of ["grid", "workgroup", "subgroup"] as const) {
      const groups = new Map<string, number[]>();
      for (const index of activeIndices) {
        const key = semanticBarrierGroupKey(contexts[index]!, scope);
        const members = groups.get(key) ?? [];
        members.push(index);
        groups.set(key, members);
      }
      for (const members of groups.values()) {
        if (!members.every((index) => waiting[index] === scope)) continue;
        for (const index of members) waiting[index] = undefined;
        released = true;
      }
    }
    if (advanced || released) continue;
    throw semanticReferenceError("semantic reference barrier mismatch: active threads reached incompatible barrier scopes", contexts[0]?.compiled.kernelIr.span ?? operations[0]?.span ?? { start: 0, end: 0, line: 1, column: 1 });
  }
}

function semanticBarrierGroupKey(context: SemanticReferenceContext, scope: SemanticBarrierScope): string {
  if (scope === "grid") return "grid";
  const block = `${context.blockIdx.x},${context.blockIdx.y},${context.blockIdx.z}`;
  if (scope === "workgroup") return block;
  const linearThread = context.threadIdx.x + context.blockDim.x * (context.threadIdx.y + context.blockDim.y * context.threadIdx.z);
  return `${block}:subgroup:${Math.floor(linearThread / 32)}`;
}

function runSemanticCollectiveOperations(
  operations: readonly SemanticKernelIrOperation[],
  contexts: readonly SemanticReferenceContext[],
): ReadonlyMap<SemanticReferenceContext, SemanticControl> {
  const controls = new Map<SemanticReferenceContext, SemanticControl>();
  const active = (): SemanticReferenceContext[] => contexts.filter((context) => !controls.has(context));
  for (const operation of operations) {
    const current = active();
    if (current.length === 0) break;
    if (operation.kind === "branch") {
      for (const context of current) context.activeCollectiveContexts = current;
      let consequent: SemanticReferenceContext[];
      try {
        consequent = current.filter((context) => truthy(evalNumber(operation.condition, context)));
      } finally {
        for (const context of current) delete context.activeCollectiveContexts;
      }
      const alternate = current.filter((context) => !consequent.includes(context));
      mergeSemanticCollectiveControls(controls, runSemanticCollectiveScopedOperations(operation.consequent, consequent));
      mergeSemanticCollectiveControls(controls, runSemanticCollectiveScopedOperations(operation.alternate, alternate));
      continue;
    }
    if (operation.kind === "block") {
      mergeSemanticCollectiveControls(controls, runSemanticCollectiveScopedOperations(operation.body, current));
      continue;
    }
    if (operation.kind === "loop") {
      mergeSemanticCollectiveControls(controls, runSemanticCollectiveLoop(operation, current));
      continue;
    }
    for (const context of current) context.activeCollectiveContexts = current;
    try {
      for (const context of current) {
        const control = execSemanticOperations([operation], context);
        if (control !== "fallthrough") controls.set(context, control);
      }
    } finally {
      for (const context of current) delete context.activeCollectiveContexts;
    }
  }
  return controls;
}

function runSemanticCollectiveScopedOperations(
  operations: readonly SemanticKernelIrOperation[],
  contexts: readonly SemanticReferenceContext[],
): ReadonlyMap<SemanticReferenceContext, SemanticControl> {
  const saved = contexts.map((context) => {
    const locals = new Map<string, SemanticValue | undefined>();
    for (const operation of operations) {
      if (operation.kind === "declare" && !locals.has(operation.target.name)) locals.set(operation.target.name, context.locals.get(operation.target.name));
    }
    return [context, locals] as const;
  });
  try {
    return runSemanticCollectiveOperations(operations, contexts);
  } finally {
    for (const [context, locals] of saved) {
      for (const [name, value] of locals) {
        if (value === undefined) context.locals.delete(name);
        else context.locals.set(name, value);
      }
    }
  }
}

function runSemanticCollectiveLoop(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>,
  contexts: readonly SemanticReferenceContext[],
): ReadonlyMap<SemanticReferenceContext, SemanticControl> {
  if (operation.loopKind === "for" && operation.init) {
    for (const context of contexts) execSemanticLoopInit(operation.init, context);
  }
  const completed = new Map<SemanticReferenceContext, SemanticControl>();
  let active = [...contexts];
  for (let guard = 0; active.length > 0; guard++) {
    if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
    if (operation.loopKind !== "do-while" && operation.condition) {
      const continuing: SemanticReferenceContext[] = [];
      for (const context of active) {
        if (truthy(evalNumber(operation.condition, context))) continuing.push(context);
        else completed.set(context, "fallthrough");
      }
      active = continuing;
      if (active.length === 0) break;
    }
    const bodyControls = runSemanticCollectiveScopedOperations(operation.body, active);
    const continuing: SemanticReferenceContext[] = [];
    for (const context of active) {
      const control = bodyControls.get(context) ?? "fallthrough";
      if (control === "return") completed.set(context, control);
      else if (control === "break") completed.set(context, "fallthrough");
      else continuing.push(context);
    }
    active = continuing;
    if (operation.loopKind === "for" && operation.update) {
      for (const context of active) evalNumber(operation.update, context);
    }
    if (operation.loopKind === "do-while" && operation.condition) {
      const repeating: SemanticReferenceContext[] = [];
      for (const context of active) {
        if (truthy(evalNumber(operation.condition, context))) repeating.push(context);
        else completed.set(context, "fallthrough");
      }
      active = repeating;
    }
  }
  return completed;
}

function mergeSemanticCollectiveControls(
  target: Map<SemanticReferenceContext, SemanticControl>,
  source: ReadonlyMap<SemanticReferenceContext, SemanticControl>,
): void {
  for (const [context, control] of source) {
    if (control !== "fallthrough") target.set(context, control);
  }
}

function semanticOperationsContainSubgroupCall(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) => {
    if (operation.kind === "declare") return operation.init !== undefined && semanticExpressionContainsSubgroupCall(operation.init);
    if (operation.kind === "store") return semanticExpressionContainsSubgroupCall(operation.value) || operation.target.indices.some(semanticExpressionContainsSubgroupCall);
    if (operation.kind === "atomic") return operation.args.some(semanticExpressionContainsSubgroupCall) || operation.target?.indices.some(semanticExpressionContainsSubgroupCall) === true;
    if (operation.kind === "call") return SEMANTIC_SUBGROUP_CALLS.has(operation.callee) || operation.args.some(semanticExpressionContainsSubgroupCall);
    if (operation.kind === "expression") return semanticExpressionContainsSubgroupCall(operation.expression);
    if (operation.kind === "branch") return semanticExpressionContainsSubgroupCall(operation.condition) ||
      semanticOperationsContainSubgroupCall(operation.consequent) ||
      semanticOperationsContainSubgroupCall(operation.alternate);
    if (operation.kind === "loop") return (operation.init === undefined
      ? false
      : isSemanticKernelIrOperation(operation.init)
        ? semanticOperationsContainSubgroupCall([operation.init])
        : semanticExpressionContainsSubgroupCall(operation.init)) ||
      (operation.condition !== undefined && semanticExpressionContainsSubgroupCall(operation.condition)) ||
      (operation.update !== undefined && semanticExpressionContainsSubgroupCall(operation.update)) ||
      semanticOperationsContainSubgroupCall(operation.body);
    if (operation.kind === "return") return operation.value !== undefined && semanticExpressionContainsSubgroupCall(operation.value);
    if (operation.kind === "block") return semanticOperationsContainSubgroupCall(operation.body);
    return false;
  });
}

function semanticExpressionContainsSubgroupCall(expression: SemanticExpression): boolean {
  if (expression.kind === "call") {
    if (expression.callee.kind === "symbol" && expression.callee.addressSpace !== "function" &&
      (SEMANTIC_SUBGROUP_CALLS.has(expression.callee.name) ||
        expression.callee.name === "cg::reduce" ||
        expression.callee.name === "cooperative_groups::reduce")) return true;
    return semanticExpressionContainsSubgroupCall(expression.callee) || expression.args.some(semanticExpressionContainsSubgroupCall);
  }
  if (expression.kind === "member") return semanticExpressionContainsSubgroupCall(expression.object);
  if (expression.kind === "index") return semanticExpressionContainsSubgroupCall(expression.target) || semanticExpressionContainsSubgroupCall(expression.index);
  if (expression.kind === "texture-read") return semanticExpressionContainsSubgroupCall(expression.texture) || semanticExpressionContainsSubgroupCall(expression.x) || semanticExpressionContainsSubgroupCall(expression.y) || (expression.z !== undefined && semanticExpressionContainsSubgroupCall(expression.z));
  if (expression.kind === "surface-read") return semanticExpressionContainsSubgroupCall(expression.surface) || semanticExpressionContainsSubgroupCall(expression.xBytes) || semanticExpressionContainsSubgroupCall(expression.y) || (expression.z !== undefined && semanticExpressionContainsSubgroupCall(expression.z));
  if (expression.kind === "cast") return semanticExpressionContainsSubgroupCall(expression.expression);
  if (expression.kind === "unary") return semanticExpressionContainsSubgroupCall(expression.argument);
  if (expression.kind === "binary") return semanticExpressionContainsSubgroupCall(expression.left) || semanticExpressionContainsSubgroupCall(expression.right);
  if (expression.kind === "conditional") return semanticExpressionContainsSubgroupCall(expression.condition) || semanticExpressionContainsSubgroupCall(expression.consequent) || semanticExpressionContainsSubgroupCall(expression.alternate);
  if (expression.kind === "assignment") return semanticExpressionContainsSubgroupCall(expression.target) || semanticExpressionContainsSubgroupCall(expression.value);
  if (expression.kind === "update") return semanticExpressionContainsSubgroupCall(expression.argument);
  if (expression.kind === "initializer") return expression.elements.some(semanticExpressionContainsSubgroupCall);
  if (expression.kind === "sequence") return expression.expressions.some(semanticExpressionContainsSubgroupCall);
  return false;
}

function semanticReferenceSharedMemory(compiled: CompiledCudaLiteKernel): Map<string, WgslTypedArray> {
  const out = new Map<string, WgslTypedArray>();
  for (const symbol of compiled.kernelIr.memory.filter((item) => item.kind === "shared")) {
    const vectorType = semanticStorageVectorType(symbol.valueType);
    const lanes = vectorType === undefined ? 1 : cudaVectorLaneCount(vectorType);
    out.set(
      symbol.name,
      typedArrayForScalar(symbol.valueType, totalElements(semanticReferenceSharedDimensions(compiled, symbol)) * lanes),
    );
  }
  return out;
}

function semanticReferenceSharedDimensions(
  _compiled: CompiledCudaLiteKernel,
  symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number],
): readonly number[] {
  return symbol.dimensions;
}

function semanticReferenceConstants(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): Map<string, number | WgslTypedArray> {
  const constants = new Map<string, number | WgslTypedArray>();
  for (const constant of compiled.kernelIr.memory.filter((symbol) => symbol.kind === "constant")) {
    const value = input.constants?.[constant.name];
    if (value !== undefined) constants.set(constant.name, value);
    else if (constant.initialized && constant.init !== undefined && (constant.dimensions.length > 0 || isSemanticFloatVectorType(constant.valueType))) {
      constants.set(constant.name, initializedConstantArrayValue(constant));
    } else if (constant.initialized && constant.dimensions.length === 0 && constant.init !== undefined) {
      constants.set(constant.name, evalConstantInitNumber(constant.init));
    }
  }
  return constants;
}

function initializedConstantArraySupported(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): boolean {
  if (!symbol.init) return false;
  if (symbol.init.kind !== "initializer" && !(isSemanticFloatVectorType(symbol.valueType) && semanticVectorConstantInitCallSupported(symbol.init))) return false;
  const length = isSemanticFloatVectorType(symbol.valueType)
    ? cudaVectorLaneCount(symbol.valueType)
    : totalElements(symbol.dimensions);
  return semanticVectorConstantInitExpressions(symbol.init)
    .slice(0, length)
    .every((value) => semanticReferenceExpressionSupported(value, "scalar"));
}

function initializedConstantArrayValue(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): WgslTypedArray {
  const length = isSemanticFloatVectorType(symbol.valueType)
    ? cudaVectorLaneCount(symbol.valueType)
    : totalElements(symbol.dimensions);
  const array = typedArrayForScalar(symbol.valueType, length);
  if (!symbol.init) return array;
  const values = semanticVectorConstantInitExpressions(symbol.init)
    .slice(0, length)
    .map(evalConstantInitNumber);
  for (let index = 0; index < values.length; index++) array[index] = values[index] ?? 0;
  return array;
}

function semanticVectorConstantInitCallSupported(expression: SemanticExpression): boolean {
  return expression.kind === "call" && semanticReferenceVectorConstructorSupported(expression, "any");
}

function semanticVectorConstantInitExpressions(expression: SemanticExpression): readonly SemanticExpression[] {
  if (expression.kind === "initializer") return flattenInitializerExpressions(expression);
  if (expression.kind === "call" && semanticVectorConstantInitCallSupported(expression)) return expression.args;
  return [expression];
}

function evalConstantInitNumber(expression: SemanticExpression): number {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number" ? expression.value : 0;
    case "cast":
      return castNumber(evalConstantInitNumber(expression.expression), expression.valueType);
    case "unary":
      return evalUnary(expression.operator, evalConstantInitNumber(expression.argument));
    case "binary":
      return evalBinary(expression.operator, evalConstantInitNumber(expression.left), evalConstantInitNumber(expression.right));
    case "conditional":
      return truthy(evalConstantInitNumber(expression.condition))
        ? evalConstantInitNumber(expression.consequent)
        : evalConstantInitNumber(expression.alternate);
    default:
      return 0;
  }
}

function semanticReferenceError(message: string, span: SourceSpan): CudaLiteCompilerError {
  const diagnostic: CudaLiteDiagnostic = {
    code: "semantic-reference-unsupported",
    severity: "error",
    message,
    span,
  };
  return new CudaLiteCompilerError(message, [diagnostic]);
}

function isSemanticKernelIrOperation(
  value: SemanticKernelIrOperation | SemanticExpression,
): value is SemanticKernelIrOperation {
  switch (value.kind) {
    case "declare":
    case "dim3-declare":
    case "cooperative-group-declare":
    case "load":
    case "store":
    case "surface-write":
    case "surface-read-store":
    case "atomic":
    case "expression":
    case "branch":
    case "loop":
    case "barrier":
    case "fence":
    case "device-launch":
    case "inline-asm":
    case "return":
    case "continue":
    case "break":
    case "block":
      return true;
    case "call":
      return typeof value.callee === "string";
    default:
      return false;
  }
}
