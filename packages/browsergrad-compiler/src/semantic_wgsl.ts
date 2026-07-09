import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslValueType,
} from "@unlocalhosted/browsergrad-kernels";
import type {
  SemanticAddressSpace,
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { walkSemanticOperations } from "./semantic_ir.js";
import type {
  CudaLiteDiagnostic,
  CudaLiteTextureDescriptor,
  CudaLiteScalarType,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { createWgslNameMap, safeWgslIdentifier } from "./wgsl_names.js";
import { emitBfloatConversionHelpers, emitCurandHelpers, emitFp8Helpers, emitHalfConversionHelpers } from "./wgsl_support_helpers.js";
import { classifyInlineAsm } from "./features/inline_ptx/model.js";
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
  isSemanticHalf2BooleanComparisonCall,
  isSemanticHalf2ComparisonCall,
  isSemanticHalf2MaskComparisonCall,
  isSemanticHalf2UnaryCall,
  isSemanticFloatVectorType,
  semanticExpressionValueType,
  semanticExpressionVectorValueType,
} from "./semantic_vector_intrinsics.js";
import {
  SEMANTIC_CURAND_CALLS,
  SEMANTIC_CURAND_DISTRIBUTION_CALLS,
  SEMANTIC_CURAND_STATE_ONLY_CALLS,
  SEMANTIC_CURAND_VECTOR_CALLS,
} from "./semantic_curand_intrinsics.js";
import {
  SEMANTIC_ADDRESS_PREDICATE_CALLS,
  SEMANTIC_LOCAL_ARRAY_FILL_CALLS,
  SEMANTIC_NOOP_CALLS,
  SEMANTIC_SUBGROUP_CALLS,
} from "./semantic_builtin_calls.js";
import {
  isSemanticAtomicCallName,
  semanticAtomicOperation,
  semanticAtomicSupportsBfloatAdd,
  semanticAtomicSupportsFloat,
} from "./semantic_atomic_intrinsics.js";
import {
  cudaLiteDimensionStride as dimensionStride,
  cudaLiteFlatIndicesForDimensions as flatIndicesForDimensions,
  cudaLiteTotalElements as totalElements,
} from "./cuda_lite_values.js";
import { flattenSemanticInitializerExpressions as flattenInitializerExpressions } from "./semantic_initializers.js";
import {
  SEMANTIC_BFLOAT_HELPER_CALLS,
  SEMANTIC_FP8_CALLS,
  SEMANTIC_HALF_CONVERSION_CALLS,
  SEMANTIC_MATH_CALLS,
  semanticMathCallArity,
} from "./semantic_math_intrinsics.js";
import { cudaVectorConstructorType, cudaVectorLaneCount, cudaVectorScalarType, cudaVectorSwizzleIndices, cudaVectorSwizzleType, isCudaVectorType } from "./vector_types.js";
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

export interface EmitSemanticKernelIrWgslOptions {
  readonly f16Mode?: "native" | "f32";
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
  readonly textureDescriptors?: Readonly<Record<string, CudaLiteTextureDescriptor>>;
}

interface SemanticTextureDescriptorSignature {
  readonly key: string;
  readonly descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>;
}

type SemanticTextureDescriptorSpecializations = ReadonlyMap<string, ReadonlyMap<string, SemanticTextureDescriptorSignature>>;
type SemanticWgslValueType =
  | WgslValueType
  | "bool"
  | "vec2<f32>"
  | "vec3<f32>"
  | "vec4<f32>"
  | "vec2<f16>"
  | "vec2<i32>"
  | "vec3<i32>"
  | "vec4<i32>"
  | "vec2<u32>"
  | "vec3<u32>"
  | "vec4<u32>";

interface SemanticTextureDescriptorHelper {
  readonly textureName: string;
  readonly descriptor: CudaLiteTextureDescriptor;
}

type SemanticShuffleOp = "sync" | "down" | "up" | "xor";

interface SemanticWarpShuffleHelper {
  readonly key: string;
  readonly name: string;
  readonly op: SemanticShuffleOp;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
}

interface SemanticMatchAnyHelper {
  readonly key: string;
  readonly name: string;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
}

type SemanticBitwiseReduceOp = "and" | "or" | "xor";

interface SemanticBitwiseReduceHelper {
  readonly key: string;
  readonly name: string;
  readonly op: SemanticBitwiseReduceOp;
  readonly valueType: "int" | "uint";
  readonly tileSize: number;
}

const UNIFORM_PARAMS_NAME = "bg_uniforms";
const BUILTIN_VECTOR_NAMES = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const LOGICAL_OPERATORS = new Set(["&&", "||"]);

export function canEmitSemanticKernelIrWgsl(
  ir: SemanticKernelIrModule,
  _options: EmitSemanticKernelIrWgslOptions = {},
): boolean {
  return unsupportedSemanticWgslOperation(ir.operations, ir) === undefined &&
    semanticWgslRequiredFeaturesSupported(ir.requiredFeatures) &&
    ir.params.every(semanticWgslParamSupported) &&
    semanticWgslSharedBarrierShapeSupported(ir) &&
    ir.memory.every(semanticWgslMemorySymbolSupported);
}

export function emitSemanticKernelIrWgsl(
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions = {},
): SemanticKernelIrWgslOutput {
  const unsupported = unsupportedSemanticWgslOperation(ir.operations, ir);
  if (unsupported) throw semanticWgslError(`semantic WGSL does not support ${unsupported.kind}`, unsupported.span);
  if (!semanticWgslRequiredFeaturesSupported(ir.requiredFeatures)) throw semanticWgslError("semantic WGSL does not support required WebGPU features yet", ir.span);
  const unsupportedParam = ir.params.find((param) => !semanticWgslParamSupported(param));
  if (unsupportedParam) throw semanticWgslError(`semantic WGSL does not support parameter '${unsupportedParam.name}'`, unsupportedParam.span);

  const textureSpecializations = collectSemanticTextureDescriptorSpecializations(ir, options);
  const storageOffsetBases = semanticStorageOffsetBaseNames(ir.operations, ir, options);
  const rawNames = new Set(ir.params.map((param) => param.name));
  for (const base of storageOffsetBases) rawNames.add(storageOffsetSymbol(base));
  for (const operation of ir.operations) collectOperationNames(operation, rawNames);
  for (const fn of ir.functions) {
    rawNames.add(fn.name);
    for (const signature of textureSpecializations.get(fn.name)?.values() ?? []) {
      rawNames.add(semanticSpecializedFunctionName(fn.name, signature.key));
    }
    for (const param of fn.params) rawNames.add(param.name);
    for (const param of fn.params.filter((item) => item.pointer && item.addressSpace === "storage")) {
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
  const names = createWgslNameMap([...rawNames]);
  const initializedScalarConstants = constantMemorySymbols(ir).filter((symbol) => symbol.initialized && symbol.dimensions.length === 0);
  const initializedConstantArrays = constantMemorySymbols(ir).filter((symbol) => symbol.initialized && symbol.dimensions.length > 0);
  const uniformParams = [
    ...ir.params.filter((param) => param.addressSpace === "uniform"),
    ...constantMemorySymbols(ir).filter((symbol) => !symbol.initialized && symbol.dimensions.length === 0 && !isSemanticFloatVectorType(symbol.valueType)),
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
  const constantBuffers = constantMemorySymbols(ir).filter((symbol) => !symbol.initialized && (symbol.dimensions.length > 0 || isSemanticFloatVectorType(symbol.valueType)));
  const deviceGlobalBuffers = deviceGlobalMemorySymbols(ir);
  const textures = textureSymbols(ir);
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions);
  const atomicDeviceGlobals = semanticAtomicDeviceGlobalNames(ir.operations);
  const atomicShared = semanticAtomicSharedNames(ir.operations);
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
      byteLength: Math.max(16, uniformParams.length * 4),
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
  lines.push("", ...emitSemanticNumericHelpers());
  if (semanticUsesBfloatHelper(ir)) {
    lines.push("", ...emitBfloatConversionHelpers());
  }
  if (semanticUsesHalfConversion(ir)) {
    lines.push("", ...emitHalfConversionHelpers());
  }
  if (semanticUsesIntegerLoopAtomic(ir.operations)) {
    lines.push("", ...emitIntegerAtomicLoopHelpers());
  }
  for (const helper of semanticFloatAtomicHelpers(ir.operations)) {
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
  for (const helper of semanticWarpShuffleHelpers(ir)) {
    lines.push(`var<workgroup> ${semanticWarpShuffleScratchName(helper)}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of semanticMatchAnyHelpers(ir)) {
    lines.push(`var<workgroup> ${semanticMatchAnyScratchName(helper)}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const helper of semanticBitwiseReduceHelpers(ir)) {
    lines.push(`var<workgroup> ${semanticBitwiseReduceScratchName(helper)}: array<${wgslValueScalar(helper.valueType)}, ${semanticWorkgroupSize(ir)}>;`);
  }
  for (const shared of sharedMemorySymbols(ir)) {
    lines.push(`var<workgroup> ${nameFor(shared.name, names)}: ${emitSharedType(shared, atomicShared.has(shared.name))};`);
  }
  if (uniformParams.length > 0) {
    lines.push("struct Params {");
    for (const param of uniformParams) {
      const scalar = wgslUniformScalar(param.valueType);
      const align = scalar === "f16" ? "@align(4) " : "";
      lines.push(`  ${align}${nameFor(param.name, names)}: ${scalar},`);
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
  for (const helper of semanticWarpShuffleHelpers(ir)) {
    lines.push("", ...emitSemanticWarpShuffleHelper(helper, ir));
  }
  for (const helper of semanticMatchAnyHelpers(ir)) {
    lines.push("", ...emitSemanticMatchAnyHelper(helper, ir));
  }
  for (const helper of semanticBitwiseReduceHelpers(ir)) {
    lines.push("", ...emitSemanticBitwiseReduceHelper(helper, ir));
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

function unsupportedSemanticWgslOperation(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  allowReturnValue = false,
): SemanticKernelIrOperation | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        if (operation.target.addressSpace === "shared") {
          if (operation.target.pointer || !semanticWgslScalarTypeSupported(operation.target.valueType)) return operation;
          break;
        }
        if (operation.target.addressSpace !== "local" || operation.target.pointer) return operation;
        if (!semanticWgslValueTypeSupported(operation.target.valueType)) return operation;
        if (operation.target.dimensions.length > 0 && operation.init && !semanticWgslLocalArrayInitSupported(operation.init, operation.target.valueType, ir)) return operation;
        if (operation.target.dimensions.length === 0) {
          const vectorTarget = isSemanticFloatVectorType(operation.target.valueType);
          if (operation.init && !semanticWgslExpressionSupported(operation.init, vectorTarget ? "any" : "scalar", ir)) return operation;
        }
        break;
      case "store":
        if (!semanticWgslAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticWgslTypedMemoryRefSupported(operation.target, ir) && !semanticWgslStorageOffsetStoreSupported(operation, ir)) return operation;
        if (
          operation.target.addressSpace === "storage" &&
          !semanticWgslStorageBaseSupported(operation.target.base, ir)
        ) return operation;
        if (!semanticWgslStoreValueSupported(operation, ir)) return operation;
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
        if (!semanticWgslCallSupported(operation, ir)) return operation;
        break;
      case "expression":
        if (!semanticWgslExpressionSupported(operation.expression, "scalar", ir)) return operation;
        break;
      case "branch":
        if (!semanticWgslExpressionSupported(operation.condition, "scalar", ir)) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.consequent, ir, allowReturnValue) ??
          unsupportedSemanticWgslOperation(operation.alternate, ir, allowReturnValue);
          if (unsupported) return unsupported;
        }
        break;
      case "block":
        if (operationsContainDeclare(operation.body)) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.body, ir, allowReturnValue);
          if (unsupported) return unsupported;
        }
        break;
      case "loop":
        if (operation.init && !semanticWgslLoopInitSupported(operation.init, ir)) return operation;
        if (operation.condition && !semanticWgslExpressionSupported(operation.condition, "scalar", ir)) return operation;
        if (operation.update && !semanticWgslExpressionSupported(operation.update, "scalar", ir)) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.body, ir, allowReturnValue);
          if (unsupported) return unsupported;
        }
        break;
      case "barrier":
        if (operation.callee !== "__syncthreads" && operation.callee !== "__syncwarp") return operation;
        break;
      case "fence":
        if (
          operation.callee !== "__threadfence" &&
          operation.callee !== "__threadfence_block" &&
          operation.callee !== "__threadfence_system"
        ) return operation;
        break;
      case "inline-asm":
        {
          const asm = classifyInlineAsm(operation.statement.template);
          const outputs = operation.statement.outputs ?? (operation.statement.output === undefined ? [] : [operation.statement.output]);
          if (asm?.kind === "cp-async-fence") {
            if (operation.statement.inputs.length > (asm.fence === "wait_group" ? 1 : 0) || outputs.length !== 0) return operation;
            break;
          }
          if (asm?.kind === "membar") {
            if (operation.statement.inputs.length !== 0 || outputs.length !== 0) return operation;
            break;
          }
          if (asm?.kind === "bar-sync") {
            if (operation.statement.inputs.length !== (asm.operand === "input0" ? 1 : 0) || outputs.length !== 0) return operation;
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

function semanticWgslParamSupported(param: SemanticKernelIrModule["params"][number]): boolean {
  if (param.addressSpace === "storage") return Boolean(param.pointer) && semanticWgslValueTypeSupported(param.valueType);
  if (param.addressSpace === "uniform") return semanticWgslScalarTypeSupported(param.valueType);
  if (param.addressSpace === "texture") return param.valueType === "texture2d";
  if (param.addressSpace === "surface") return param.valueType === "surface2d";
  return false;
}

function semanticWgslFunctionParamSupported(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
): boolean {
  if (param.pointer) return param.addressSpace === "storage" && semanticWgslValueTypeSupported(param.valueType);
  return param.addressSpace === "local" || param.addressSpace === "texture" || param.addressSpace === "surface";
}

function operationsContainDeclare(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) =>
    operation.kind === "declare" ||
    operation.kind === "branch" && (operationsContainDeclare(operation.consequent) || operationsContainDeclare(operation.alternate)) ||
    operation.kind === "loop" && operationsContainDeclare(operation.body) ||
    operation.kind === "block" && operationsContainDeclare(operation.body)
  );
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
  if (symbol.kind === "device-global") return semanticWgslScalarTypeSupported(symbol.valueType);
  if (symbol.kind === "texture") return symbol.valueType === "texture2d";
  return false;
}

function semanticWgslSharedBarrierShapeSupported(ir: SemanticKernelIrModule): boolean {
  const shared = sharedMemorySymbols(ir);
  if (shared.length === 0 && !operationsContainBarrier(ir.operations)) return true;
  if (!shared.every((symbol) => symbol.dimensions.length === 1 && (symbol.dimensions[0] ?? 0) > 0)) return false;
  if (!operationsContainBarrier(ir.operations)) return operationsHaveNoBarrierOrControlTransfer(ir.operations);
  return operationsHaveOnlyTopLevelBarriers(ir.operations);
}

function operationsContainBarrier(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) =>
    operation.kind === "barrier" ||
    operation.kind === "branch" && (operationsContainBarrier(operation.consequent) || operationsContainBarrier(operation.alternate)) ||
    operation.kind === "loop" && operationsContainBarrier(operation.body) ||
    operation.kind === "block" && operationsContainBarrier(operation.body)
  );
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
  return valueType === "float" || valueType === "half" || valueType === "bf16" || valueType === "int" || valueType === "uint" || valueType === "bool";
}

function semanticWgslValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticWgslScalarTypeSupported(valueType) || isSemanticFloatVectorType(valueType);
}

function semanticWgslAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
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
  return expression.kind === "index" ? memoryRefFromIndexExpression(expression) : undefined;
}

function semanticWgslVectorBinaryOperatorSupported(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

function semanticWgslLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  ir: SemanticKernelIrModule,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? unsupportedSemanticWgslOperation([init], ir) === undefined
    : semanticWgslExpressionSupported(init, "scalar");
}

function semanticWgslMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "shared" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local") return false;
  if (ref.fields.length > 0) return semanticWgslVectorFieldMemoryRefSupported(ref);
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return semanticWgslScalarTypeSupported(ref.valueType);
  return ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar"));
}

function semanticWgslStorageBaseSupported(base: string, ir: SemanticKernelIrModule): boolean {
  return ir.params.some((param) => param.name === base && param.addressSpace === "storage") ||
    ir.functions.some((fn) => fn.params.some((param) => param.name === base && param.pointer && param.addressSpace === "storage"));
}

function semanticWgslFunctionStoragePointerParam(
  ir: SemanticKernelIrModule,
  base: string,
): SemanticKernelIrModule["functions"][number]["params"][number] | undefined {
  for (const fn of ir.functions) {
    const param = fn.params.find((item) => item.name === base && item.pointer && item.addressSpace === "storage");
    if (param) return param;
  }
  return undefined;
}

function semanticStoragePointerBufferId(base: string, ir: SemanticKernelIrModule): number | undefined {
  const index = ir.params.findIndex((param) => param.name === base && param.addressSpace === "storage");
  return index < 0 ? undefined : index;
}

function emitSemanticStoragePointerHelpers(
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly (readonly string[])[] {
  const types = new Set<CudaLiteScalarType>();
  for (const fn of ir.functions) {
    for (const param of fn.params) {
      if (param.pointer && param.addressSpace === "storage" && param.valueType !== undefined) types.add(param.valueType);
    }
  }
  return [...types].flatMap((type) => [
    emitSemanticStoragePointerReadHelper(type, ir, names),
    emitSemanticStoragePointerWriteHelper(type, ir, names),
    emitSemanticStoragePointerAtomicCasHelper(type, ir, names),
  ]);
}

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

function emitSemanticStoragePointerAtomicCasHelper(
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions);
  return [
    `fn ${semanticPointerAtomicCasHelperName(valueType)}(buffer: u32, index: u32, compare: ${wgslType}, value: ${wgslType}) -> ${wgslType} {`,
    "  switch buffer {",
    ...ir.params.flatMap((param, index) =>
      param.addressSpace === "storage" && !param.constant && atomicStorage.has(param.name) && semanticPointerStorageCompatible(valueType, param.valueType)
        ? [`    case ${index}u: { return ${emitSemanticStoragePointerAtomicCasValue(valueType, nameFor(param.name, names), "index", "compare", "value")}; }`]
        : []
    ),
    "    default: { return " + zeroForType(wgslType) + "; }",
    "  }",
    "}",
  ];
}

function semanticPointerStorageCompatible(pointerType: CudaLiteScalarType, storageType: CudaLiteScalarType | undefined): boolean {
  if (storageType === undefined) return false;
  return pointerType === storageType ||
    isCudaVectorType(storageType) && cudaVectorScalarType(storageType) === pointerType ||
    isCudaVectorType(pointerType) && cudaVectorScalarType(pointerType) === storageType ||
    isCudaVectorType(pointerType) && cudaVectorScalarType(pointerType) === cudaVectorScalarType(storageType);
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
  return valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitSemanticStoragePointerWriteValue(valueType: CudaLiteScalarType, storage: string, index: string, value: string, atomic: boolean): string {
  if (!isCudaVectorType(valueType)) return emitSemanticStoragePointerWriteScalarValue(valueType, `${storage}[${index}]`, value, atomic);
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    emitSemanticStoragePointerWriteScalarValue(cudaVectorScalarType(valueType) ?? valueType, `${storage}[(${index} + ${lane}u)]`, `(${value}).${["x", "y", "z", "w"][lane]}`, atomic)
  ).join(" ");
}

function emitSemanticStoragePointerWriteScalarValue(valueType: CudaLiteScalarType, access: string, value: string, atomic: boolean): string {
  if (!atomic) return `${access} = ${value};`;
  const stored = valueType === "float" ? `bitcast<u32>(${value})` : value;
  return `atomicStore(&${access}, ${stored});`;
}

function emitSemanticStoragePointerAtomicCasValue(
  valueType: CudaLiteScalarType,
  storage: string,
  index: string,
  compare: string,
  value: string,
): string {
  if (valueType === "float" || valueType === "double") {
    return `bitcast<f32>(atomicCompareExchangeWeak(&${storage}[${index}], bitcast<u32>(${compare}), bitcast<u32>(${value})).old_value)`;
  }
  return `atomicCompareExchangeWeak(&${storage}[${index}], ${compare}, ${value}).old_value`;
}

function semanticPointerReadHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_read_${semanticPointerHelperTypeName(valueType)}`;
}

function semanticPointerWriteHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_write_${semanticPointerHelperTypeName(valueType)}`;
}

function semanticPointerAtomicCasHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_atomicCompareExchange_${semanticPointerHelperTypeName(valueType)}`;
}

function semanticPointerHelperTypeName(valueType: CudaLiteScalarType): string {
  const scalar = wgslValueScalar(valueType);
  return isCudaVectorType(valueType) ? `${scalar}x${cudaVectorLaneCount(valueType)}` : scalar;
}

function semanticWgslTypedMemoryRefSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (!semanticWgslMemoryRefSupported(ref)) return false;
  if (semanticWgslVectorFieldMemoryRefSupported(ref)) return true;
  if (semanticWgslLocalVectorLaneRefSupported(ref, ir)) return true;
  if (ref.addressSpace !== "local" && ref.addressSpace !== "shared") return true;
  const symbol = ir.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace);
  return symbol !== undefined && symbol.valueType === ref.valueType;
}

function semanticWgslVectorFieldMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "device-global" && ref.addressSpace !== "shared" && ref.addressSpace !== "local") return false;
  if (ref.fields.length !== 1 || !isCudaVectorType(ref.containerValueType)) return false;
  const lanes = cudaVectorSwizzleIndices(ref.containerValueType, ref.fields[0]!);
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
  if (operation.target.addressSpace === "storage" && operation.target.indices.length !== 1 && !semanticWgslFunctionStoragePointerParam(ir, operation.target.base)) return false;
  if (operation.target.fields.length > 0) return false;
  if (!semanticWgslAtomicValueTypeSupported(operation.callee, operation.target.valueType)) return false;
  if (!semanticWgslAtomicTargetRootSupported(operation.target, ir)) {
    return false;
  }
  const expectedArgs = atomicOp === "cas" ? 3 : 2;
  return operation.args.length >= expectedArgs &&
    operation.args.slice(1, expectedArgs).every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslStoreValueSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const targetVectorType = operation.target.valueType;
  const valueVectorType = semanticExpressionVectorValueType(operation.value, ir?.functions);
  if (semanticWgslVectorFieldMemoryRefSupported(operation.target)) {
    return isSemanticFloatVectorType(targetVectorType)
      ? valueVectorType === targetVectorType && semanticWgslExpressionSupported(operation.value, "any", ir)
      : semanticWgslScalarStoreValueSupported(operation.value, ir);
  }
  if (isSemanticFloatVectorType(targetVectorType)) {
    return operation.operator === "=" &&
      valueVectorType === targetVectorType &&
      semanticWgslExpressionSupported(operation.value, "any", ir);
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
      return semanticWgslScalarStoreValueSupported(expression.left, ir) &&
        semanticWgslScalarStoreValueSupported(expression.right, ir);
    case "conditional":
      return semanticWgslExpressionSupported(expression.condition, "scalar", ir) &&
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
  if (expression.callee.kind !== "symbol") return false;
  const callee = expression.callee.name;
  if (SEMANTIC_CURAND_VECTOR_CALLS.has(callee) || SEMANTIC_HALF2_VECTOR_CALLS.has(callee) || SEMANTIC_BF162_VECTOR_CALLS.has(callee) || cudaVectorConstructorType(callee)) return false;
  const fn = ir.functions.find((item) => item.name === callee);
  if (fn && isSemanticFloatVectorType(fn.returnType)) return false;
  return semanticWgslFunctionCallSupported(expression, ir) ||
    semanticWgslAtomicCallSupported(expression, ir) ||
    semanticWgslCurandCallSupported(expression, ir) ||
    semanticWgslSubgroupCallSupported(expression, ir) ||
    semanticWgslAddressPredicateCallSupported(expression) ||
    semanticWgslMathCallSupported(expression) ||
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
    isCudaVectorType(valueType) &&
    cudaVectorSwizzleType(valueType, expression.property) !== undefined;
}

function semanticWgslVectorIndexSupported(
  expression: Extract<SemanticExpression, { kind: "index" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  const ref = memoryRefFromIndexExpression(expression);
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
  const expected = isSemanticFloatVectorType(targetValueType) ? "any" : "scalar";
  if (expression.kind === "initializer") {
    return flattenInitializerExpressions(expression).every((item) => semanticWgslExpressionSupported(item, expected, ir));
  }
  return semanticWgslExpressionSupported(expression, expected, ir);
}

function semanticWgslMathCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_MATH_CALLS.has(expression.callee.name)) return false;
  const arity = semanticMathCallArity(expression.callee.name);
  return expression.args.length === arity && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslCurandCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_CURAND_CALLS.has(expression.callee.name)) return false;
  if (expression.callee.name === "curand_init") {
    return expression.args.length === 4 &&
      expression.args.slice(0, 3).every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir)) &&
      semanticCurandStateAddressSpace(expression.args[3]!) !== undefined;
  }
  if (SEMANTIC_CURAND_STATE_ONLY_CALLS.has(expression.callee.name)) {
    return expression.args.length === 1 && semanticCurandStateAddressSpace(expression.args[0]!) !== undefined;
  }
  if (expression.callee.name === "curand_poisson" || expression.callee.name === "curand_poisson4") {
    return expression.args.length === 2 &&
      semanticCurandStateAddressSpace(expression.args[0]!) !== undefined &&
      semanticWgslExpressionSupported(expression.args[1]!, "scalar", ir);
  }
  if (expression.callee.name === "skipahead") {
    return expression.args.length === 2 &&
      semanticWgslExpressionSupported(expression.args[0]!, "scalar", ir) &&
      semanticCurandStateAddressSpace(expression.args[1]!) !== undefined;
  }
  if (SEMANTIC_CURAND_DISTRIBUTION_CALLS.has(expression.callee.name)) {
    return expression.args.length === 3 &&
      semanticCurandStateAddressSpace(expression.args[0]!) !== undefined &&
      expression.args.slice(1).every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  return false;
}

function semanticWgslSubgroupCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_SUBGROUP_CALLS.has(expression.callee.name) || ir?.requiredFeatures.includes("subgroups") !== true) return false;
  if (expression.callee.name === "__activemask") return expression.args.length === 0;
  if (legacyVoteCall(expression.callee.name)) {
    return expression.args.length === 1 && semanticWgslExpressionSupported(expression.args[0]!, "scalar", ir);
  }
  if (legacyShuffleCall(expression.callee.name)) {
    return (expression.args.length === 2 || expression.args.length === 3) &&
      expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  if (semanticBitwiseReduceOpForCall(expression.callee.name)) {
    const value = expression.args[1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    return expression.args.length === 2 &&
      (valueType === "int" || valueType === "uint") &&
      expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  if (semanticShuffleOpForCall(expression.callee.name)) {
    return (expression.args.length === 3 || expression.args.length === 4) &&
      expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  return expression.args.length === 2 && semanticWgslExpressionSupported(expression.args[1]!, "scalar", ir);
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
  return semanticWgslTextureValueTypeSupported(expression.valueType) &&
    texture.kind === "symbol" &&
    texture.addressSpace === "texture" &&
    semanticWgslExpressionSupported(expression.x, "scalar", ir) &&
    semanticWgslExpressionSupported(expression.y, "scalar", ir);
}

function semanticWgslTextureValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" ||
    valueType === "half" ||
    valueType === "bf16" ||
    valueType === "uint" ||
    valueType === "int" ||
    valueType === "uchar" ||
    isSemanticFloatVectorType(valueType);
}

function semanticWgslSurfaceReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const target = expression.surface;
  return (expression.valueType === "float" ||
      expression.valueType === "half" ||
      expression.valueType === "bf16" ||
      expression.valueType === "uint" ||
      expression.valueType === "int" ||
      expression.valueType === "uchar" ||
      isSemanticFloatVectorType(expression.valueType)) &&
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
  if (!fn || !semanticWgslValueTypeSupported(fn.returnType)) return false;
  if (fn.params.some((param) => !semanticWgslFunctionParamSupported(param))) return false;
  if (fn.params.some((param) => param.pointer) && !semanticWgslPointerFunctionBodySupported(fn)) return false;
  if (fn.params.some((param) => param.addressSpace === "local" && !semanticWgslValueTypeSupported(param.valueType))) return false;
  if (!semanticWgslFunctionBodyShapeSupported(fn.body)) return false;
  return expression.args.length === fn.params.length &&
    expression.args.every((arg, index) => semanticWgslFunctionArgSupported(arg, fn.params[index], ir)) &&
    unsupportedSemanticWgslOperation(fn.body, ir, true) === undefined;
}

function semanticWgslFunctionArgSupported(
  arg: SemanticExpression,
  param: SemanticKernelIrModule["functions"][number]["params"][number] | undefined,
  ir: SemanticKernelIrModule,
): boolean {
  if (!param) return false;
  if (param.pointer) {
    const ref = semanticPointerArgMemoryRef(arg);
    return param.addressSpace === "storage" && ref?.addressSpace === "storage";
  }
  if (param.addressSpace === "texture") return arg.kind === "symbol" && arg.addressSpace === "texture";
  if (param.addressSpace === "surface") return arg.kind === "symbol" && arg.addressSpace === "surface";
  return semanticWgslExpressionSupported(arg, isSemanticFloatVectorType(param.valueType) ? "any" : "scalar", ir);
}

function semanticWgslVectorConstructorSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any",
  ir?: SemanticKernelIrModule,
): boolean {
  if (expected === "scalar" || expression.callee.kind !== "symbol") return false;
  const valueType = cudaVectorConstructorType(expression.callee.name);
  return isSemanticFloatVectorType(valueType) &&
    expression.args.length > 0 &&
    expression.args.every((arg) => semanticWgslExpressionSupported(arg, "any", ir));
}

function semanticWgslVectorAtCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  return expression.callee.kind === "symbol" &&
    expression.callee.name === "vec_at" &&
    expression.args.length === 2 &&
    expression.args[0] !== undefined &&
    expression.args[1] !== undefined &&
    isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.args[0], ir?.functions)) &&
    semanticWgslExpressionSupported(expression.args[0], "any", ir) &&
    semanticWgslExpressionSupported(expression.args[1], "scalar", ir);
}

function semanticWgslVectorLerpCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  const [left, right, amount] = expression.args;
  if (expression.callee.kind !== "symbol" || expression.callee.name !== "lerp" || !left || !right || !amount) return false;
  const valueType = semanticExpressionVectorValueType(left, ir?.functions);
  return isSemanticFloatVectorType(valueType) &&
    semanticExpressionVectorValueType(right, ir?.functions) === valueType &&
    semanticWgslExpressionSupported(left, "any", ir) &&
    semanticWgslExpressionSupported(right, "any", ir) &&
    semanticWgslExpressionSupported(amount, "scalar", ir);
}

function semanticWgslHalf2CallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const name = expression.callee.name;
  if (!SEMANTIC_HALF2_VECTOR_CALLS.has(name) && !SEMANTIC_HALF2_SCALAR_CALLS.has(name)) return false;
  if (isSemanticHalf2UnaryCall(name)) {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "half2" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (isSemanticHalf2ComparisonCall(name)) {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "half2" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (name === "__hadd2" || name === "__hadd2_rn" || name === "__hadd2_sat" || name === "__hsub2" || name === "__hsub2_rn" || name === "__hsub2_sat" || name === "__hmul2" || name === "__hmul2_rn" || name === "__hmul2_sat" || name === "__hmin2" || name === "__hmax2" || name === "__hmin2_nan" || name === "__hmax2_nan") {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "half2" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat") {
    return expression.args.length === 3 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "half2" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (name === "__half22float2" || name === "__half2_as_uint" || name === "__low2half" || name === "__high2half" || name === "__low2float" || name === "__high2float" || name === "__low2half2" || name === "__high2half2" || name === "__lowhigh2highlow") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "half2" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (name === "__halves2half2") {
    return expression.args.length === 2 && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  if (name === "__half2half2") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticWgslExpressionSupported(arg, "scalar", ir);
  }
  if (name === "__lows2half2" || name === "__highs2half2") {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "half2" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (name === "__uint_as_half2") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticWgslExpressionSupported(arg, "scalar", ir);
  }
  if (name === "__float22half2_rn") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "float2" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (name === "__float2half2_rn") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticWgslExpressionSupported(arg, "scalar", ir);
  }
  if (name === "__floats2half2_rn") {
    return expression.args.length === 2 && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  return false;
}

function semanticWgslBf162CallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const name = expression.callee.name;
  if (!SEMANTIC_BF162_VECTOR_CALLS.has(name) && !SEMANTIC_BF162_SCALAR_CALLS.has(name)) return false;
  if (SEMANTIC_BF162_UNARY_VECTOR_CALLS.has(name)) {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (SEMANTIC_BF162_BINARY_VECTOR_CALLS.has(name)) {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (SEMANTIC_BF162_TERNARY_VECTOR_CALLS.has(name)) {
    return expression.args.length === 3 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (SEMANTIC_BF162_MINMAX_VECTOR_CALLS.has(name)) {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (SEMANTIC_BF162_VECTOR_COMPARISON_CALLS.has(name)) {
    const arity = name === "__hisnan2" ? 1 : 2;
    return expression.args.length === arity && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (SEMANTIC_BF162_MASK_COMPARISON_CALLS.has(name) || SEMANTIC_BF162_BOOL_COMPARISON_CALLS.has(name)) {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (name === "__bfloat1622float2") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (name === "__float22bfloat162_rn") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "float2" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (name === "__halves2bfloat162" || name === "__floats2bfloat162_rn") {
    return expression.args.length === 2 && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
  }
  if (name === "__bfloat162bfloat162" || name === "__float2bfloat162_rn") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticWgslExpressionSupported(arg, "scalar", ir);
  }
  if (name === "__low2bfloat16" || name === "__high2bfloat16" || name === "__low2float" || name === "__high2float" || name === "__low2bfloat162" || name === "__high2bfloat162" || name === "__lowhigh2highlow") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  if (name === "__lows2bfloat162" || name === "__highs2bfloat162") {
    return expression.args.length === 2 && expression.args.every((arg) => semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir));
  }
  if (name === "__uint_as_bfloat162" || name === "__uint_as_nv_bfloat162") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticWgslExpressionSupported(arg, "scalar", ir);
  }
  if (name === "__bfloat162_as_uint" || name === "__nv_bfloat162_as_uint") {
    const [arg] = expression.args;
    return expression.args.length === 1 && arg !== undefined && semanticExpressionVectorValueType(arg, ir?.functions) === "bf162" && semanticWgslExpressionSupported(arg, "any", ir);
  }
  return false;
}

function semanticWgslFunctionBodyShapeSupported(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "declare") return operation.target.addressSpace === "local" && !operation.target.pointer && operation.target.dimensions.length === 0;
    if (operation.kind === "store") return operation.target.addressSpace === "local" || operation.target.addressSpace === "storage";
    if (operation.kind === "surface-write") return true;
    if (operation.kind === "call") return true;
    if (operation.kind === "barrier" || operation.kind === "fence") return true;
    if (operation.kind === "branch") return semanticWgslFunctionBodyShapeSupported(operation.consequent) && semanticWgslFunctionBodyShapeSupported(operation.alternate);
    if (operation.kind === "loop") return semanticWgslFunctionBodyShapeSupported(operation.body);
    return operation.kind === "expression" || operation.kind === "return" || operation.kind === "break" || operation.kind === "continue";
  });
}

function semanticWgslPointerFunctionBodySupported(fn: SemanticKernelIrModule["functions"][number]): boolean {
  const pointerParams = new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "storage").map((param) => param.name));
  return pointerParams.size > 0 && fn.body.every((operation) => semanticWgslPointerFunctionOperationSupported(operation, pointerParams));
}

function semanticWgslPointerFunctionOperationSupported(
  operation: SemanticKernelIrOperation,
  pointerParams: ReadonlySet<string>,
): boolean {
  if (operation.kind === "atomic") return operation.target !== undefined && pointerParams.has(operation.target.base);
  if (operation.kind === "store") return pointerParams.has(operation.target.base) && operation.target.fields.length > 0;
  if (operation.kind === "return" && operation.value) return semanticWgslPointerFunctionExpressionSupported(operation.value, pointerParams);
  if (operation.kind === "expression" && operation.expression.kind === "update") {
    const ref = memoryRefFromIndexExpression(operation.expression.argument);
    return ref !== undefined && pointerParams.has(ref.base);
  }
  if (operation.kind === "expression") return semanticWgslPointerFunctionExpressionSupported(operation.expression, pointerParams);
  return false;
}

function semanticWgslPointerFunctionExpressionSupported(
  expression: SemanticExpression,
  pointerParams: ReadonlySet<string>,
): boolean {
  if (expression.kind !== "call") return false;
  const target = semanticAtomicCallTarget(expression);
  return target !== undefined && pointerParams.has(target.base);
}

function collectSemanticTextureDescriptorSpecializations(
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions,
): SemanticTextureDescriptorSpecializations {
  if (options.textureDescriptors === undefined) return new Map();
  const out = new Map<string, Map<string, SemanticTextureDescriptorSignature>>();
  let changed = true;
  while (changed) {
    changed = false;
    changed = collectSemanticTextureDescriptorSpecializationsFromOperations(ir.operations, options.textureDescriptors, ir, out) || changed;
    for (const fn of ir.functions) {
      for (const signature of out.get(fn.name)?.values() ?? []) {
        const scope = { ...options.textureDescriptors, ...signature.descriptors };
        changed = collectSemanticTextureDescriptorSpecializationsFromOperations(fn.body, scope, ir, out) || changed;
      }
    }
  }
  return out;
}

function collectSemanticTextureDescriptorSpecializationsFromOperations(
  operations: readonly SemanticKernelIrOperation[],
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
  ir: SemanticKernelIrModule,
  out: Map<string, Map<string, SemanticTextureDescriptorSignature>>,
): boolean {
  let changed = false;
  for (const operation of operations) {
    for (const expression of semanticOperationExpressions(operation)) {
      changed = collectSemanticTextureDescriptorSpecializationsFromExpression(expression, scope, ir, out) || changed;
    }
    if (operation.kind === "call") {
      changed = addSemanticTextureDescriptorSignature(operation.callee, operation.args, scope, ir, out) || changed;
    }
    if (operation.kind === "branch") {
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.consequent, scope, ir, out) || changed;
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.alternate, scope, ir, out) || changed;
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        changed = collectSemanticTextureDescriptorSpecializationsFromOperations([operation.init], scope, ir, out) || changed;
      }
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.body, scope, ir, out) || changed;
    }
    if (operation.kind === "block") {
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.body, scope, ir, out) || changed;
    }
  }
  return changed;
}

function semanticOperationExpressions(operation: SemanticKernelIrOperation): readonly SemanticExpression[] {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(...operation.target.indices, operation.value);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "atomic") expressions.push(...operation.args, ...(operation.target?.indices ?? []));
  if (operation.kind === "call") expressions.push(...operation.args);
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  return expressions;
}

function collectSemanticTextureDescriptorSpecializationsFromExpression(
  expression: SemanticExpression,
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
  ir: SemanticKernelIrModule,
  out: Map<string, Map<string, SemanticTextureDescriptorSignature>>,
): boolean {
  let changed = false;
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    changed = addSemanticTextureDescriptorSignature(expression.callee.name, expression.args, scope, ir, out);
  }
  for (const child of semanticExpressionChildren(expression)) {
    changed = collectSemanticTextureDescriptorSpecializationsFromExpression(child, scope, ir, out) || changed;
  }
  return changed;
}

function addSemanticTextureDescriptorSignature(
  callee: string,
  args: readonly SemanticExpression[],
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
  ir: SemanticKernelIrModule,
  out: Map<string, Map<string, SemanticTextureDescriptorSignature>>,
): boolean {
  const fn = ir.functions.find((item) => item.name === callee);
  if (!fn) return false;
  const signature = semanticTextureDescriptorSignatureForCall(fn, args, scope);
  if (!signature) return false;
  let signatures = out.get(fn.name);
  if (!signatures) {
    signatures = new Map();
    out.set(fn.name, signatures);
  }
  if (signatures.has(signature.key)) return false;
  signatures.set(signature.key, signature);
  return true;
}

function semanticFunctionCallName(
  callee: string,
  fn: SemanticKernelIrModule["functions"][number],
  args: readonly SemanticExpression[],
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  const signature = semanticTextureDescriptorSignatureForCall(fn, args, options.textureDescriptors ?? {});
  if (!signature || !textureSpecializations.get(callee)?.has(signature.key)) return callee;
  return semanticSpecializedFunctionName(callee, signature.key);
}

function semanticTextureDescriptorSignatureForCall(
  fn: SemanticKernelIrModule["functions"][number],
  args: readonly SemanticExpression[],
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
): SemanticTextureDescriptorSignature | undefined {
  const descriptors: Record<string, CudaLiteTextureDescriptor> = {};
  for (const [index, param] of fn.params.entries()) {
    if (param.addressSpace !== "texture") continue;
    const arg = args[index];
    if (arg?.kind !== "symbol" || arg.addressSpace !== "texture") continue;
    const descriptor = scope[arg.name];
    if (descriptor !== undefined) descriptors[param.name] = descriptor;
  }
  if (Object.keys(descriptors).length === 0) return undefined;
  const key = semanticTextureDescriptorSignatureKey(descriptors);
  return { key, descriptors };
}

function semanticTextureDescriptorSignatureKey(
  descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>,
): string {
  return Object.entries(descriptors)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, descriptor]) => `${name}=${semanticTextureDescriptorKey(descriptor)}`)
    .join(",");
}

function semanticTextureDescriptorKey(descriptor: CudaLiteTextureDescriptor): string {
  return JSON.stringify({
    normalizedCoords: descriptor.normalizedCoords ?? false,
    addressMode: descriptor.addressMode ?? ["clamp", "clamp"],
    filterMode: descriptor.filterMode ?? "point",
  });
}

function semanticSpecializedFunctionName(name: string, key: string): string {
  return `${name}__bg_tex_${semanticStableHash(key)}`;
}

function semanticStableHash(value: string): string {
  let hash = 5381;
  for (let i = 0; i < value.length; i++) hash = ((hash * 33) ^ value.charCodeAt(i)) >>> 0;
  return hash.toString(36);
}

function semanticOptionsWithTextureDescriptors(
  options: EmitSemanticKernelIrWgslOptions,
  descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>,
): EmitSemanticKernelIrWgslOptions {
  const next: EmitSemanticKernelIrWgslOptions = {
    textureDescriptors: Object.assign({}, options.textureDescriptors, descriptors),
  };
  if (options.pointerBaseOffsets !== undefined) return { ...next, pointerBaseOffsets: options.pointerBaseOffsets };
  return next;
}

function semanticTextureDescriptorHelpers(
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
  names: ReadonlyMap<string, string>,
): readonly SemanticTextureDescriptorHelper[] {
  const helpers = new Map<string, SemanticTextureDescriptorHelper>();
  const add = (textureName: string, descriptor: CudaLiteTextureDescriptor): void => {
    helpers.set(semanticTextureDescriptorHelperName(textureName, names, descriptor), { textureName, descriptor });
  };
  for (const [textureName, descriptor] of Object.entries(options.textureDescriptors ?? {})) add(textureName, descriptor);
  for (const specializations of textureSpecializations.values()) {
    for (const signature of specializations.values()) {
      for (const [textureName, descriptor] of Object.entries(signature.descriptors)) add(textureName, descriptor);
    }
  }
  return [...helpers.values()];
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
  if (target.addressSpace === "storage" && target.indices.length !== 1 && !semanticWgslFunctionStoragePointerParam(ir, target.base)) return false;
  if (target.fields.length > 0) return false;
  if (!semanticWgslAtomicValueTypeSupported(expression.callee.name, target.valueType)) return false;
  if (!semanticWgslAtomicTargetRootSupported(target, ir)) return false;
  const expectedArgs = atomicOp === "cas" ? 3 : 2;
  return expression.args.length >= expectedArgs &&
    expression.args.slice(1, expectedArgs).every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslAtomicMemoryRefSupported(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): boolean {
  return semanticWgslMemoryRefSupported(ref) ||
    ref.addressSpace === "storage" &&
      ref.indices.length === 0 &&
      semanticWgslFunctionStoragePointerParam(ir, ref.base) !== undefined;
}

function semanticWgslCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (operation.callee === "assert") return operation.args.length === 1 && semanticWgslExpressionSupported(operation.args[0]!, "scalar", ir);
  if (operation.callee === "printf") return semanticWgslPrintfSupported(operation, ir);
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) return operation.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
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
  if (!SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) return false;
  const [target, value] = operation.args;
  const symbol = target?.kind === "symbol" ? localArraySymbol(ir, target.name) : undefined;
  return target?.kind === "symbol" &&
    target.addressSpace === "local" &&
    symbol !== undefined &&
    value !== undefined &&
    semanticWgslExpressionSupported(value, isSemanticFloatVectorType(symbol.valueType) ? "any" : "scalar", ir);
}

function semanticWgslPrintfSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const [format, ...args] = operation.args;
  return format?.kind === "literal" &&
    format.literalKind === "string" &&
    args.every((arg) => semanticWgslExpressionSupported(arg, "scalar", ir));
}

function semanticWgslVoidFunctionCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const fn = ir.functions.find((item) => item.name === operation.callee);
  if (!fn || fn.returnType !== "void") return false;
  if (fn.params.some((param) => !semanticWgslFunctionParamSupported(param))) return false;
  if (fn.params.some((param) => param.pointer) && !semanticWgslPointerFunctionBodySupported(fn)) return false;
  return operation.args.length === fn.params.length &&
    operation.args.every((arg, index) => semanticWgslFunctionArgSupported(arg, fn.params[index], ir)) &&
    semanticWgslFunctionBodyShapeSupported(fn.body) &&
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
    return ir.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "shared");
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
        BUILTIN_VECTOR_NAMES.has(expression.name);
    case "member":
      if (expected === "scalar" && isCudaVectorType(expression.valueType)) return false;
      return expression.object.kind === "symbol" &&
        BUILTIN_VECTOR_NAMES.has(expression.object.name) &&
        (expression.property === "x" || expression.property === "y" || expression.property === "z") ||
        semanticWgslVectorMemberSupported(expression, ir);
    case "index":
      if (semanticWgslVectorIndexSupported(expression, ir)) return true;
      if (expected === "any" && isSemanticFloatVectorType(expression.valueType)) {
        const ref = memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span);
        return ir === undefined ? semanticWgslMemoryRefSupported(ref) : semanticWgslTypedMemoryRefSupported(ref, ir);
      }
      {
        const ref = memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span);
        return expected === "scalar" && (ir === undefined ? semanticWgslMemoryRefSupported(ref) : semanticWgslTypedMemoryRefSupported(ref, ir));
      }
    case "cast":
      return !expression.pointer && semanticWgslExpressionSupported(expression.expression, "scalar", ir);
    case "unary":
      if (expected === "scalar" && semanticWgslBf162LocalBitsCastSupported(expression, ir)) return true;
      return expression.operator !== "*" && expression.operator !== "&" && semanticWgslExpressionSupported(expression.argument, "scalar", ir);
    case "binary":
      if (expected === "any" && isSemanticFloatVectorType(expression.valueType) && semanticWgslVectorBinaryOperatorSupported(expression.operator)) {
        return semanticWgslExpressionSupported(expression.left, "any", ir) &&
          semanticWgslExpressionSupported(expression.right, "any", ir);
      }
      return semanticWgslExpressionSupported(expression.left, "scalar", ir) &&
        semanticWgslExpressionSupported(expression.right, "scalar", ir);
    case "conditional":
      return semanticWgslExpressionSupported(expression.condition, "scalar", ir) &&
        semanticWgslExpressionSupported(expression.consequent, expected, ir) &&
        semanticWgslExpressionSupported(expression.alternate, expected, ir);
    case "assignment":
      {
        const vectorMemberTarget = expression.target.kind === "member" &&
          isSemanticFloatVectorType(semanticExpressionValueType(expression.target));
        return semanticWgslAssignmentOperatorSupported(expression.operator) &&
        (expression.target.kind === "symbol" && expression.target.addressSpace === "local" ||
          expression.target.kind === "member" && semanticWgslVectorMemberSupported(expression.target, ir) ||
          semanticWgslAssignmentMemoryRefSupported(expression.target, ir)) &&
        semanticWgslExpressionSupported(expression.value, vectorMemberTarget ? "any" : "scalar", ir);
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
      return ir !== undefined && semanticWgslFunctionCallSupported(expression, ir) ||
        ir !== undefined && semanticWgslAtomicCallSupported(expression, ir) ||
        semanticWgslCurandCallSupported(expression, ir) &&
          (expected === "any" || !isSemanticFloatVectorType(semanticExpressionVectorValueType(expression, ir?.functions))) ||
        semanticWgslSubgroupCallSupported(expression, ir) ||
        semanticWgslAddressPredicateCallSupported(expression) ||
        semanticWgslMathCallSupported(expression) ||
        semanticWgslHalf2CallSupported(expression, ir) ||
        semanticWgslBf162CallSupported(expression, ir) ||
        semanticWgslVectorConstructorSupported(expression, expected, ir) ||
        expected === "scalar" && semanticWgslVectorAtCallSupported(expression, ir) ||
        expected === "any" && semanticWgslVectorLerpCallSupported(expression, ir);
    case "texture-read":
      return ir !== undefined &&
        (expected === "any" || semanticWgslTextureValueTypeSupported(expression.valueType)) &&
        semanticWgslTextureReadSupported(expression, ir);
    case "surface-read":
      return ir !== undefined && (expected === "scalar" || expected === "any") && semanticWgslSurfaceReadSupported(expression, ir);
    case "initializer":
      return false;
  }
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
        const asm = classifyInlineAsm(operation.statement.template);
        const outputs = operation.statement.outputs ?? (operation.statement.output === undefined ? [] : [operation.statement.output]);
        if (asm?.kind === "cp-async-fence" && operation.statement.inputs.length <= (asm.fence === "wait_group" ? 1 : 0) && outputs.length === 0) return [`${prefix}// cp.async inline asm fence omitted`];
        if (asm?.kind === "membar" && operation.statement.inputs.length === 0 && outputs.length === 0) return [`${prefix}storageBarrier();`];
        if (asm?.kind === "bar-sync" && operation.statement.inputs.length === (asm.operand === "input0" ? 1 : 0) && outputs.length === 0) return [`${prefix}workgroupBarrier();`];
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
  if (operation.value.kind !== "sequence") {
    return [`${prefix}${emitSemanticStore(operation, ir, names, options, textureSpecializations)};`];
  }
  const sequence = emitSemanticSequenceParts(operation.value, ir, names, indentLevel, options, textureSpecializations);
  return [
    ...sequence.prefix,
    `${prefix}${emitSemanticStore({ ...operation, value: sequence.value }, ir, names, options, textureSpecializations)};`,
  ];
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
  return [`${prefix}return ${emitSemanticExpression(expression, ir, names, options, textureSpecializations)};`];
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

function semanticSurfaceReadValueType(valueType: CudaLiteScalarType | undefined): Exclude<CudaLiteScalarType, "void"> {
  return valueType === undefined || valueType === "void" ? "float" : valueType;
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
  if (semanticWgslFunctionStoragePointerParam(ir, operation.target.base)) {
    return emitSemanticPointerMemoryStore(operation, ir, names, options, textureSpecializations);
  }
  const target = emitSemanticLocalVectorLaneRef(operation.target, ir, names, options, textureSpecializations) ??
    emitSemanticMemoryRef(operation.target, ir, names, options);
  if (
    semanticAtomicStorageNames(ir.operations, ir.functions).has(operation.target.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations).has(operation.target.base) ||
    semanticAtomicSharedNames(ir.operations).has(operation.target.base)
  ) {
    if (operation.operator !== "=") {
      throw semanticWgslError(`semantic WGSL does not support atomic storage assignment '${operation.operator}'`, operation.span);
    }
    const atomicValue = emitSemanticAtomicStoreValue(operation.value, operation.target.valueType, ir, names, options, textureSpecializations);
    return `atomicStore(&${target}, ${atomicValue})`;
  }
  if (isSemanticFloatVectorType(operation.target.valueType)) {
    if (operation.operator !== "=") throw semanticWgslError(`semantic WGSL does not support vector assignment '${operation.operator}'`, operation.span);
    if (operation.target.addressSpace === "local") {
      return `${emitSemanticMemoryRef(operation.target, ir, names, options)} = ${emitSemanticExpression(operation.value, ir, names, options, textureSpecializations)}`;
    }
    return emitSemanticVectorMemoryWrite(operation, ir, names, options, textureSpecializations).join("; ");
  }
  const value = emitSemanticExpressionAs(operation.value, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations);
  if (operation.operator === "=") return `${target} = ${value}`;
  if (operation.operator === "+=") return `${target} = (${target} + ${value})`;
  if (operation.operator === "-=") return `${target} = (${target} - ${value})`;
  throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
}

function emitSemanticAtomicStoreValue(
  value: SemanticExpression,
  valueType: CudaLiteScalarType | undefined,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  if (valueType === "float") {
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
  const value = isCudaVectorType(valueType)
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
): readonly string[] {
  const valueType = operation.target.valueType;
  if (!isSemanticFloatVectorType(valueType)) throw semanticWgslError("semantic WGSL vector write requires vector target", operation.span);
  const value = emitSemanticExpression(operation.value, ir, names, options, textureSpecializations);
  const base = emitFlatStorageVectorBaseIndex(operation.target, ir, names, options);
  const target = nameFor(operation.target.base, names);
  const fields = ["x", "y", "z", "w"];
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    `${target}[(${base} + ${lane}u)] = (${value}).${fields[lane]}`
  );
}

function emitSemanticVectorFieldMemoryWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const containerType = operation.target.containerValueType;
  if (!isCudaVectorType(containerType)) throw semanticWgslError("semantic WGSL vector field write requires vector container", operation.span);
  const lanes = cudaVectorSwizzleIndices(containerType, operation.target.fields[0] ?? "");
  if (lanes === undefined) throw semanticWgslError("semantic WGSL vector field write requires modeled lanes", operation.span);
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
  const params = [
    ...fn.params.flatMap((param) => emitSemanticFunctionParams(param, names)),
    "local_id: vec3<u32>",
    "workgroup_id: vec3<u32>",
    "num_workgroups: vec3<u32>",
  ].join(", ");
  const returnType = fn.returnType === "void" ? "" : ` -> ${wgslValueType(fn.returnType)}`;
  return [
    `fn ${nameFor(rawName, names)}(${params})${returnType} {`,
    ...emitSemanticOperations(fn.body, ir, names, 1, true, options, textureSpecializations),
    ...(fn.returnType === "void" ? [] : [`  return ${zeroForType(wgslValueType(fn.returnType))};`]),
    "}",
  ];
}

function emitSemanticFunctionParamType(param: SemanticKernelIrModule["functions"][number]["params"][number]): string {
  if (param.addressSpace === "texture") return "texture_2d<f32>";
  if (param.addressSpace === "surface") return "u32";
  return wgslValueType(param.valueType);
}

function emitSemanticFunctionParams(
  param: SemanticKernelIrModule["functions"][number]["params"][number],
  names: ReadonlyMap<string, string>,
): readonly string[] {
  if (param.pointer && param.addressSpace === "storage") {
    return [
      `${nameFor(semanticPointerBufferParamName(param.name), names)}: u32`,
      `${nameFor(semanticPointerBaseParamName(param.name), names)}: u32`,
    ];
  }
  return [`${nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param)}`];
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
    if (expression.operator === "+=") return `${target} += ${value}`;
    if (expression.operator === "-=") return `${target} -= ${value}`;
    return `${target} = ${value}`;
  }
  {
    const ref = semanticWgslAssignmentMemoryRef(expression.target, ir);
    if (ref) {
      const target = emitSemanticMemoryRef(ref, ir, names, options);
      const value = emitSemanticExpressionAs(expression.value, ir, names, wgslValueScalar(ref.valueType), options, textureSpecializations);
      if (expression.operator === "+=") return `${target} += ${value}`;
      if (expression.operator === "-=") return `${target} -= ${value}`;
      return `${target} = ${value}`;
    }
  }
  if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar assignment targets only", expression.target.span);
  const target = nameFor(expression.target.name, names);
  const value = emitSemanticLocalScalarExpressionAs(expression.value, expression.target.valueType, ir, names, options, textureSpecializations);
  if (expression.operator === "+=") return `${target} += ${value}`;
  if (expression.operator === "-=") return `${target} -= ${value}`;
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
  return [...semanticStorageOffsetBaseNames(ir.operations, ir, options)]
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
  const floatAtomicKind = operation.target.valueType === "float" ? semanticWgslFloatAtomicCallKind(operation.callee) : undefined;
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
  if (valueType === "float") return semanticAtomicSupportsFloat(atomicOp);
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

function semanticFloatAtomicHelpers(operations: readonly SemanticKernelIrOperation[]): readonly string[][] {
  const helperKeys = new Set<string>();
  collectSemanticFloatAtomicHelpers(operations, helperKeys);
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind !== "call" || expression.callee.kind !== "symbol") return;
    const target = semanticAtomicCallTarget(expression);
    if (target && semanticAtomicSupportsBfloatAdd(expression.callee.name, target.valueType)) {
      helperKeys.add(`BfloatAdd:${semanticWgslAtomicAddressSpace(target)}`);
      return;
    }
    if (target?.valueType !== "float") return;
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
    if (operation.kind === "atomic" && operation.target?.valueType === "float") {
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

function semanticWarpShuffleHelpers(ir: SemanticKernelIrModule): readonly SemanticWarpShuffleHelper[] {
  const helpers = new Map<string, SemanticWarpShuffleHelper>();
  collectSemanticWarpShuffleHelpers(ir.operations, helpers);
  for (const fn of ir.functions) collectSemanticWarpShuffleHelpers(fn.body, helpers);
  return [...helpers.values()];
}

function collectSemanticWarpShuffleHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helpers: Map<string, SemanticWarpShuffleHelper>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.init) collectSemanticWarpShuffleExpressionHelpers(operation.init, helpers);
    if (operation.kind === "store") {
      collectSemanticWarpShuffleExpressionHelpers(operation.value, helpers);
      operation.target.indices.forEach((index) => collectSemanticWarpShuffleExpressionHelpers(index, helpers));
    }
    if (operation.kind === "atomic") {
      operation.args.forEach((arg) => collectSemanticWarpShuffleExpressionHelpers(arg, helpers));
      operation.target?.indices.forEach((index) => collectSemanticWarpShuffleExpressionHelpers(index, helpers));
    }
    if (operation.kind === "call") operation.args.forEach((arg) => collectSemanticWarpShuffleExpressionHelpers(arg, helpers));
    if (operation.kind === "expression") collectSemanticWarpShuffleExpressionHelpers(operation.expression, helpers);
    if (operation.kind === "branch") {
      collectSemanticWarpShuffleExpressionHelpers(operation.condition, helpers);
      collectSemanticWarpShuffleHelpers(operation.consequent, helpers);
      collectSemanticWarpShuffleHelpers(operation.alternate, helpers);
    }
    if (operation.kind === "loop") {
      if (operation.init !== undefined) {
        if (isSemanticKernelIrOperation(operation.init)) collectSemanticWarpShuffleHelpers([operation.init], helpers);
        else collectSemanticWarpShuffleExpressionHelpers(operation.init, helpers);
      }
      if (operation.condition) collectSemanticWarpShuffleExpressionHelpers(operation.condition, helpers);
      if (operation.update) collectSemanticWarpShuffleExpressionHelpers(operation.update, helpers);
      collectSemanticWarpShuffleHelpers(operation.body, helpers);
    }
    if (operation.kind === "return" && operation.value) collectSemanticWarpShuffleExpressionHelpers(operation.value, helpers);
    if (operation.kind === "block") collectSemanticWarpShuffleHelpers(operation.body, helpers);
  }
}

function collectSemanticWarpShuffleExpressionHelpers(
  expression: SemanticExpression,
  helpers: Map<string, SemanticWarpShuffleHelper>,
): void {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const op = semanticShuffleOpForCall(expression.callee.name);
    const value = expression.args[legacyShuffleCall(expression.callee.name) ? 0 : 1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    if (op && valueType && valueType !== "void") {
      const helper = semanticWarpShuffleHelper(op, valueType, 32);
      helpers.set(helper.key, helper);
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    collectSemanticWarpShuffleExpressionHelpers(child, helpers);
  }
}

function semanticMatchAnyHelpers(ir: SemanticKernelIrModule): readonly SemanticMatchAnyHelper[] {
  const helpers = new Map<string, SemanticMatchAnyHelper>();
  collectSemanticMatchAnyHelpers(ir.operations, helpers);
  for (const fn of ir.functions) collectSemanticMatchAnyHelpers(fn.body, helpers);
  return [...helpers.values()];
}

function collectSemanticMatchAnyHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helpers: Map<string, SemanticMatchAnyHelper>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.init) collectSemanticMatchAnyExpressionHelpers(operation.init, helpers);
    if (operation.kind === "store") {
      collectSemanticMatchAnyExpressionHelpers(operation.value, helpers);
      operation.target.indices.forEach((index) => collectSemanticMatchAnyExpressionHelpers(index, helpers));
    }
    if (operation.kind === "atomic") {
      operation.args.forEach((arg) => collectSemanticMatchAnyExpressionHelpers(arg, helpers));
      operation.target?.indices.forEach((index) => collectSemanticMatchAnyExpressionHelpers(index, helpers));
    }
    if (operation.kind === "call") operation.args.forEach((arg) => collectSemanticMatchAnyExpressionHelpers(arg, helpers));
    if (operation.kind === "expression") collectSemanticMatchAnyExpressionHelpers(operation.expression, helpers);
    if (operation.kind === "branch") {
      collectSemanticMatchAnyExpressionHelpers(operation.condition, helpers);
      collectSemanticMatchAnyHelpers(operation.consequent, helpers);
      collectSemanticMatchAnyHelpers(operation.alternate, helpers);
    }
    if (operation.kind === "loop") {
      if (operation.init !== undefined) {
        if (isSemanticKernelIrOperation(operation.init)) collectSemanticMatchAnyHelpers([operation.init], helpers);
        else collectSemanticMatchAnyExpressionHelpers(operation.init, helpers);
      }
      if (operation.condition) collectSemanticMatchAnyExpressionHelpers(operation.condition, helpers);
      if (operation.update) collectSemanticMatchAnyExpressionHelpers(operation.update, helpers);
      collectSemanticMatchAnyHelpers(operation.body, helpers);
    }
    if (operation.kind === "return" && operation.value) collectSemanticMatchAnyExpressionHelpers(operation.value, helpers);
    if (operation.kind === "block") collectSemanticMatchAnyHelpers(operation.body, helpers);
  }
}

function collectSemanticMatchAnyExpressionHelpers(
  expression: SemanticExpression,
  helpers: Map<string, SemanticMatchAnyHelper>,
): void {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && expression.callee.name === "__match_any_sync") {
    const value = expression.args[1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    if (valueType && valueType !== "void") {
      const helper = semanticMatchAnyHelper(valueType, 32);
      helpers.set(helper.key, helper);
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    collectSemanticMatchAnyExpressionHelpers(child, helpers);
  }
}

function semanticMatchAnyHelper(
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): SemanticMatchAnyHelper {
  const key = `${valueType}:${tileSize}`;
  return {
    key,
    name: `bg_semantic_match_any_${safeWgslIdentifier(valueType)}_${tileSize}`,
    valueType,
    tileSize,
  };
}

function semanticMatchAnyScratchName(helper: SemanticMatchAnyHelper): string {
  return `${helper.name}_scratch`;
}

function emitSemanticMatchAnyHelper(
  helper: SemanticMatchAnyHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticWorkgroupSize(ir);
  const scratch = semanticMatchAnyScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, width_arg: u32, local_id: vec3<u32>) -> u32 {`,
    `  let bg_linear_rank: u32 = ${semanticLocalLinearRank(ir)};`,
    `  let bg_tile_lane: u32 = bg_linear_rank % ${helper.tileSize}u;`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_logical_lane: u32 = bg_tile_lane % bg_width;",
    "  let bg_group_base: u32 = bg_linear_rank - bg_logical_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    "  var bg_mask: u32 = 0u;",
    "  var bg_lane: u32 = 0u;",
    "  while (bg_lane < bg_width) {",
    "    let bg_source_rank: u32 = bg_group_base + bg_lane;",
    `    if (bg_source_rank < ${workgroupSize}u && ${scratch}[bg_source_rank] == value_arg) {`,
    "      bg_mask = bg_mask | (1u << bg_lane);",
    "    }",
    "    bg_lane = bg_lane + 1u;",
    "  }",
    "  workgroupBarrier();",
    "  return bg_mask;",
    "}",
  ];
}

function semanticBitwiseReduceHelpers(ir: SemanticKernelIrModule): readonly SemanticBitwiseReduceHelper[] {
  const helpers = new Map<string, SemanticBitwiseReduceHelper>();
  collectSemanticBitwiseReduceHelpers(ir.operations, helpers);
  for (const fn of ir.functions) collectSemanticBitwiseReduceHelpers(fn.body, helpers);
  return [...helpers.values()];
}

function collectSemanticBitwiseReduceHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helpers: Map<string, SemanticBitwiseReduceHelper>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.init) collectSemanticBitwiseReduceExpressionHelpers(operation.init, helpers);
    if (operation.kind === "store") {
      collectSemanticBitwiseReduceExpressionHelpers(operation.value, helpers);
      operation.target.indices.forEach((index) => collectSemanticBitwiseReduceExpressionHelpers(index, helpers));
    }
    if (operation.kind === "atomic") {
      operation.args.forEach((arg) => collectSemanticBitwiseReduceExpressionHelpers(arg, helpers));
      operation.target?.indices.forEach((index) => collectSemanticBitwiseReduceExpressionHelpers(index, helpers));
    }
    if (operation.kind === "call") operation.args.forEach((arg) => collectSemanticBitwiseReduceExpressionHelpers(arg, helpers));
    if (operation.kind === "expression") collectSemanticBitwiseReduceExpressionHelpers(operation.expression, helpers);
    if (operation.kind === "branch") {
      collectSemanticBitwiseReduceExpressionHelpers(operation.condition, helpers);
      collectSemanticBitwiseReduceHelpers(operation.consequent, helpers);
      collectSemanticBitwiseReduceHelpers(operation.alternate, helpers);
    }
    if (operation.kind === "loop") {
      if (operation.init !== undefined) {
        if (isSemanticKernelIrOperation(operation.init)) collectSemanticBitwiseReduceHelpers([operation.init], helpers);
        else collectSemanticBitwiseReduceExpressionHelpers(operation.init, helpers);
      }
      if (operation.condition) collectSemanticBitwiseReduceExpressionHelpers(operation.condition, helpers);
      if (operation.update) collectSemanticBitwiseReduceExpressionHelpers(operation.update, helpers);
      collectSemanticBitwiseReduceHelpers(operation.body, helpers);
    }
    if (operation.kind === "return" && operation.value) collectSemanticBitwiseReduceExpressionHelpers(operation.value, helpers);
    if (operation.kind === "block") collectSemanticBitwiseReduceHelpers(operation.body, helpers);
  }
}

function collectSemanticBitwiseReduceExpressionHelpers(
  expression: SemanticExpression,
  helpers: Map<string, SemanticBitwiseReduceHelper>,
): void {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const op = semanticBitwiseReduceOpForCall(expression.callee.name);
    const value = expression.args[1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    if (op && (valueType === "int" || valueType === "uint")) {
      const helper = semanticBitwiseReduceHelper(op, valueType, 32);
      helpers.set(helper.key, helper);
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    collectSemanticBitwiseReduceExpressionHelpers(child, helpers);
  }
}

function semanticBitwiseReduceOpForCall(name: string): SemanticBitwiseReduceOp | undefined {
  if (name === "__reduce_and_sync") return "and";
  if (name === "__reduce_or_sync") return "or";
  if (name === "__reduce_xor_sync") return "xor";
  return undefined;
}

function semanticBitwiseReduceHelper(
  op: SemanticBitwiseReduceOp,
  valueType: "int" | "uint",
  tileSize: number,
): SemanticBitwiseReduceHelper {
  const key = `${op}:${valueType}:${tileSize}`;
  return {
    key,
    name: `bg_semantic_reduce_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
  };
}

function semanticBitwiseReduceScratchName(helper: SemanticBitwiseReduceHelper): string {
  return `${helper.name}_scratch`;
}

function emitSemanticBitwiseReduceHelper(
  helper: SemanticBitwiseReduceHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticWorkgroupSize(ir);
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  const scratch = semanticBitwiseReduceScratchName(helper);
  const operator = helper.op === "and" ? "&" : helper.op === "or" ? "|" : "^";
  return [
    `fn ${helper.name}(value_arg: ${type}, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = ${semanticLocalLinearRank(ir)};`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_tile_lane: u32 = bg_linear_rank % bg_width;",
    "  let bg_tile_base: u32 = bg_linear_rank - bg_tile_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    `  var bg_stride: u32 = ${start}u;`,
    "  while (bg_stride > 0u) {",
    `    if (bg_stride < bg_width && bg_tile_lane < bg_stride && (bg_tile_lane + bg_stride) < bg_width && (bg_linear_rank + bg_stride) < ${workgroupSize}u) {`,
    `      ${scratch}[bg_linear_rank] = ${scratch}[bg_linear_rank] ${operator} ${scratch}[bg_linear_rank + bg_stride];`,
    "    }",
    "    workgroupBarrier();",
    "    bg_stride = bg_stride / 2u;",
    "  }",
    `  let bg_result: ${type} = ${scratch}[bg_tile_base];`,
    "  workgroupBarrier();",
    "  return bg_result;",
    "}",
  ];
}

function semanticShuffleOpForCall(name: string): SemanticShuffleOp | undefined {
  if (name === "__shfl" || name === "__shfl_sync") return "sync";
  if (name === "__shfl_down" || name === "__shfl_down_sync") return "down";
  if (name === "__shfl_up" || name === "__shfl_up_sync") return "up";
  if (name === "__shfl_xor" || name === "__shfl_xor_sync") return "xor";
  return undefined;
}

function semanticWarpShuffleHelper(
  op: SemanticShuffleOp,
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): SemanticWarpShuffleHelper {
  const key = `${op}:${valueType}:${tileSize}`;
  return {
    key,
    name: `bg_semantic_warp_shuffle_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
  };
}

function semanticWarpShuffleScratchName(helper: SemanticWarpShuffleHelper): string {
  return `${helper.name}_scratch`;
}

function emitSemanticWarpShuffleHelper(
  helper: SemanticWarpShuffleHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticWorkgroupSize(ir);
  const scratch = semanticWarpShuffleScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, index_arg: u32, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = ${semanticLocalLinearRank(ir)};`,
    `  let bg_tile_lane: u32 = bg_linear_rank % ${helper.tileSize}u;`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_logical_lane: u32 = bg_tile_lane % bg_width;",
    "  let bg_group_base: u32 = bg_linear_rank - bg_logical_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    ...emitSemanticWarpShuffleSourceLines(helper, workgroupSize),
    `  let bg_result: ${type} = ${scratch}[bg_source_rank];`,
    "  workgroupBarrier();",
    "  return bg_result;",
    "}",
  ];
}

function emitSemanticWarpShuffleSourceLines(helper: SemanticWarpShuffleHelper, workgroupSize: number): readonly string[] {
  switch (helper.op) {
    case "sync":
      return [
        "  let bg_source_lane: u32 = index_arg % bg_width;",
        "  let bg_source_candidate: u32 = bg_group_base + bg_source_lane;",
        `  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_source_candidate < ${workgroupSize}u);`,
      ];
    case "down":
      return [
        "  let bg_source_lane: u32 = bg_logical_lane + index_arg;",
        "  let bg_source_candidate: u32 = bg_linear_rank + index_arg;",
        `  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_source_lane < bg_width && bg_source_candidate < ${workgroupSize}u);`,
      ];
    case "up":
      return [
        "  let bg_source_candidate: u32 = bg_linear_rank - min(index_arg, bg_linear_rank);",
        "  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_logical_lane >= index_arg);",
      ];
    case "xor":
      return [
        "  let bg_source_lane: u32 = bg_logical_lane ^ index_arg;",
        "  let bg_source_candidate: u32 = bg_group_base + bg_source_lane;",
        `  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_source_lane < bg_width && bg_source_candidate < ${workgroupSize}u);`,
      ];
  }
}

function semanticWorkgroupSize(ir: SemanticKernelIrModule): number {
  return ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
}

function semanticLocalLinearRank(ir: SemanticKernelIrModule): string {
  return `(local_id.x + local_id.y * ${ir.workgroupSize[0]}u + local_id.z * ${ir.workgroupSize[0] * ir.workgroupSize[1]}u)`;
}

function emitSemanticCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  if (operation.callee === "assert") return [];
  if (operation.callee === "printf") return [];
  if (SEMANTIC_NOOP_CALLS.has(operation.callee)) return [];
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
  return [
    `${prefix}loop {`,
    ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations),
    `${"  ".repeat(indentLevel + 1)}continuing {`,
    `${"  ".repeat(indentLevel + 2)}break if !(${operation.condition ? emitTruthiness(operation.condition, ir, names, options) : "false"});`,
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
      if (expression.addressSpace === "shared" && semanticAtomicSharedNames(ir.operations).has(expression.name)) {
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
        if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
          return emitSemanticMemoryRead(ref, ir, names, options);
        }
        if (isSemanticFloatVectorType(ref.valueType) && ref.addressSpace === "local") {
          return emitSemanticMemoryRef(ref, ir, names, options);
        }
        if (isSemanticFloatVectorType(ref.valueType)) return emitSemanticVectorMemoryRead(ref, ir, names, options);
        const memoryRef = emitSemanticMemoryRef(ref, ir, names, options);
        if (
          semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base) ||
          semanticAtomicDeviceGlobalNames(ir.operations).has(ref.base) ||
          semanticAtomicSharedNames(ir.operations).has(ref.base)
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
      if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar assignment targets only", expression.target.span);
      {
        const target = nameFor(expression.target.name, names);
        const value = emitSemanticLocalScalarExpressionAs(expression.value, expression.target.valueType, ir, names, options, textureSpecializations);
        if (expression.operator === "+=") return `(${target} += ${value})`;
        if (expression.operator === "-=") return `(${target} -= ${value})`;
        return `(${target} = ${value})`;
      }
    case "update":
      return emitSemanticUpdate(expression, ir, names, options);
    case "sequence":
      return emitSemanticExpression(expression.expressions.at(-1) ?? zeroExpression(expression.span), ir, names, options, textureSpecializations);
    case "call":
      if (semanticWgslAtomicCallSupported(expression, ir)) return emitSemanticAtomicCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslCurandCallSupported(expression, ir)) return emitSemanticCurandCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslSubgroupCallSupported(expression, ir)) return emitSemanticSubgroupCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslAddressPredicateCallSupported(expression)) return emitSemanticAddressPredicateCall(expression);
      if (semanticWgslVectorConstructorSupported(expression, "any", ir)) return emitSemanticVectorConstructor(expression, ir, names, options, textureSpecializations);
      if (semanticWgslVectorAtCallSupported(expression, ir)) return emitSemanticVectorAtCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslVectorLerpCallSupported(expression, ir)) return emitSemanticVectorLerpCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslHalf2CallSupported(expression, ir)) return emitSemanticHalf2Call(expression, ir, names, options, textureSpecializations);
      if (semanticWgslBf162CallSupported(expression, ir)) return emitSemanticBf162Call(expression, ir, names, options, textureSpecializations);
      if (semanticWgslFunctionCallSupported(expression, ir)) return emitSemanticFunctionCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslMathCallSupported(expression)) return emitSemanticMathCall(expression, ir, names, options, textureSpecializations);
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
    case "texture-read":
      return emitSemanticTextureRead(expression, ir, names, options);
    case "surface-read":
      return emitSemanticSurfaceRead(expression, ir, names, options);
    case "initializer":
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
  }
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
    throw semanticWgslError("semantic WGSL supports only direct tex2D<float> reads", expression.span);
  }
  const x = emitSemanticExpressionAs(expression.x, ir, names, "f32", options);
  const y = emitSemanticExpressionAs(expression.y, ir, names, "f32", options);
  const texture = nameFor(expression.texture.name, names);
  const descriptor = options.textureDescriptors?.[expression.texture.name];
  const read = descriptor
    ? `${semanticTextureDescriptorHelperName(expression.texture.name, names, descriptor)}(${texture}, ${x}, ${y})`
    : `textureLoad(${texture}, clamp(vec2<i32>(i32(floor(${x})), i32(floor(${y}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${texture})) - vec2<i32>(1, 1)), 0)`;
  if (isSemanticFloatVectorType(expression.valueType)) return emitSemanticTextureVectorRead(read, expression.valueType);
  if (expression.valueType === "half") return `f16(${read}.r)`;
  if (expression.valueType === "bf16") return wgslRoundBfloat16(`${read}.r`);
  if (expression.valueType === "uint" || expression.valueType === "uchar") return `u32(${read}.r)`;
  if (expression.valueType === "int") return `i32(${read}.r)`;
  return `${read}.r`;
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

function emitSemanticTextureDescriptorHelper(
  textureName: string,
  descriptor: CudaLiteTextureDescriptor,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const texture = "bg_texture";
  const helper = semanticTextureDescriptorHelperName(textureName, names, descriptor);
  if (descriptor.filterMode === "linear") {
    return [
      `fn ${helper}(${texture}: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {`,
      `  let dims = textureDimensions(${texture});`,
      `  let sx = ${semanticTextureScaledCoord("x", "dims.x", descriptor)};`,
      `  let sy = ${semanticTextureScaledCoord("y", "dims.y", descriptor)};`,
      "  let xb = sx - 0.5;",
      "  let yb = sy - 0.5;",
      "  let x0f = floor(xb);",
      "  let y0f = floor(yb);",
      "  let ax = xb - x0f;",
      "  let ay = yb - y0f;",
      `  let x0 = ${semanticTextureIndex("i32(x0f)", "dims.x", descriptor, "x")};`,
      `  let x1 = ${semanticTextureIndex("(i32(x0f) + 1)", "dims.x", descriptor, "x")};`,
      `  let y0 = ${semanticTextureIndex("i32(y0f)", "dims.y", descriptor, "y")};`,
      `  let y1 = ${semanticTextureIndex("(i32(y0f) + 1)", "dims.y", descriptor, "y")};`,
      `  let v00 = textureLoad(${texture}, vec2<i32>(x0, y0), 0);`,
      `  let v10 = textureLoad(${texture}, vec2<i32>(x1, y0), 0);`,
      `  let v01 = textureLoad(${texture}, vec2<i32>(x0, y1), 0);`,
      `  let v11 = textureLoad(${texture}, vec2<i32>(x1, y1), 0);`,
      "  return mix(mix(v00, v10, ax), mix(v01, v11, ax), ay);",
      "}",
    ];
  }
  return [
    `fn ${helper}(${texture}: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {`,
    `  let dims = textureDimensions(${texture});`,
    `  let ix = ${semanticTextureIndex(`i32(floor(${semanticTextureScaledCoord("x", "dims.x", descriptor)}))`, "dims.x", descriptor, "x")};`,
    `  let iy = ${semanticTextureIndex(`i32(floor(${semanticTextureScaledCoord("y", "dims.y", descriptor)}))`, "dims.y", descriptor, "y")};`,
    `  return textureLoad(${texture}, vec2<i32>(ix, iy), 0);`,
    "}",
  ];
}

function emitSemanticNumericHelpers(): readonly string[] {
  return [
    "fn bg_semantic_round_even_f32(value: f32) -> f32 {",
    "  if (value != value || abs(value) > 3.4028234663852886e38) { return value; }",
    "  let lower = floor(value);",
    "  let diff = value - lower;",
    "  if (diff < 0.5) { return lower; }",
    "  if (diff > 0.5) { return lower + 1.0; }",
    "  let half_lower = floor(lower * 0.5);",
    "  return select(lower + 1.0, lower, (half_lower * 2.0) == lower);",
    "}",
    "fn bg_semantic_remainder_f32(x: f32, y: f32) -> f32 {",
    "  return x - bg_semantic_round_even_f32(x / y) * y;",
    "}",
    "fn bg_semantic_logb_f32(value: f32) -> f32 {",
    "  if (value != value) { return value; }",
    "  let runtime_zero = select(0.0, value, false);",
    "  if (value == 0.0) { return -1.0 / runtime_zero; }",
    "  if (abs(value) > 3.4028234663852886e38) { return 1.0 / runtime_zero; }",
    "  return floor(log2(abs(value)));",
    "}",
    "fn bg_semantic_ilogb_i32(value: f32) -> i32 {",
    "  if (value != value || abs(value) > 3.4028234663852886e38) { return 2147483647; }",
    "  if (value == 0.0) { return -2147483648; }",
    "  return i32(floor(log2(abs(value))));",
    "}",
    "fn bg_semantic_nextafter_f32(x: f32, y: f32) -> f32 {",
    "  if (x != x || y != y) { return x + y; }",
    "  if (x == y) { return y; }",
    "  if (x == 0.0) {",
    "    return bitcast<f32>(select(0x00000001u, 0x80000001u, (bitcast<u32>(y) & 0x80000000u) != 0u));",
    "  }",
    "  var bits = bitcast<u32>(x);",
    "  if (x > 0.0) {",
    "    bits = select(bits - 1u, bits + 1u, x < y);",
    "  } else {",
    "    bits = select(bits + 1u, bits - 1u, x < y);",
    "  }",
    "  return bitcast<f32>(bits);",
    "}",
    "fn bg_semantic_runtime_nan_f32(value: f32) -> f32 {",
    "  let runtime_zero = select(0.0, value, false);",
    "  return runtime_zero / runtime_zero;",
    "}",
    "fn bg_semantic_runtime_inf_f32(value: f32) -> f32 {",
    "  let runtime_zero = select(0.0, value, false);",
    "  return 1.0 / runtime_zero;",
    "}",
    "fn bg_semantic_erf_f32(value: f32) -> f32 {",
    "  if (value != value) { return value; }",
    "  if (abs(value) > 3.4028234663852886e38) { return select(-1.0, 1.0, value > 0.0); }",
    "  let magnitude = abs(value);",
    "  let t = 1.0 / (1.0 + 0.3275911 * magnitude);",
    "  let polynomial = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;",
    "  return select(-1.0, 1.0, value >= 0.0) * (1.0 - (polynomial * exp(-(magnitude * magnitude))));",
    "}",
    "fn bg_semantic_erfinv_f32(value: f32) -> f32 {",
    "  if (value != value) { return value; }",
    "  if (value <= -1.0) { return select(bg_semantic_runtime_nan_f32(value), -bg_semantic_runtime_inf_f32(value), value == -1.0); }",
    "  if (value >= 1.0) { return select(bg_semantic_runtime_nan_f32(value), bg_semantic_runtime_inf_f32(value), value == 1.0); }",
    "  if (value == 0.0) { return 0.0; }",
    "  let a = 0.147;",
    "  let l = log(1.0 - value * value);",
    "  let first = (2.0 / (3.141592653589793 * a)) + (l * 0.5);",
    "  var y = select(-1.0, 1.0, value >= 0.0) * sqrt(sqrt(first * first - l / a) - first);",
    "  y -= (bg_semantic_erf_f32(y) - value) / (1.1283791670955126 * exp(-(y * y)));",
    "  y -= (bg_semantic_erf_f32(y) - value) / (1.1283791670955126 * exp(-(y * y)));",
    "  return y;",
    "}",
    "fn bg_semantic_normcdfinv_f32(value: f32) -> f32 {",
    "  if (value != value) { return value; }",
    "  if (value <= 0.0) { return select(bg_semantic_runtime_nan_f32(value), -bg_semantic_runtime_inf_f32(value), value == 0.0); }",
    "  if (value >= 1.0) { return select(bg_semantic_runtime_nan_f32(value), bg_semantic_runtime_inf_f32(value), value == 1.0); }",
    "  return 1.4142135623730951 * bg_semantic_erfinv_f32(2.0 * value - 1.0);",
    "}",
    "fn bg_semantic_tgamma_core_f32(value: f32) -> f32 {",
    "  let z = value - 1.0;",
    "  var x = 0.9999999999998099;",
    "  x += 676.5203681218851 / (z + 1.0);",
    "  x += -1259.1392167224028 / (z + 2.0);",
    "  x += 771.3234287776531 / (z + 3.0);",
    "  x += -176.6150291621406 / (z + 4.0);",
    "  x += 12.507343278686905 / (z + 5.0);",
    "  x += -0.13857109526572012 / (z + 6.0);",
    "  x += 0.000009984369578019572 / (z + 7.0);",
    "  x += 0.00000015056327351493116 / (z + 8.0);",
    "  let t = z + 7.5;",
    "  return 2.5066282746310002 * pow(t, z + 0.5) * exp(-t) * x;",
    "}",
    "fn bg_semantic_lgamma_core_f32(value: f32) -> f32 {",
    "  let z = value - 1.0;",
    "  var x = 0.9999999999998099;",
    "  x += 676.5203681218851 / (z + 1.0);",
    "  x += -1259.1392167224028 / (z + 2.0);",
    "  x += 771.3234287776531 / (z + 3.0);",
    "  x += -176.6150291621406 / (z + 4.0);",
    "  x += 12.507343278686905 / (z + 5.0);",
    "  x += -0.13857109526572012 / (z + 6.0);",
    "  x += 0.000009984369578019572 / (z + 7.0);",
    "  x += 0.00000015056327351493116 / (z + 8.0);",
    "  let t = z + 7.5;",
    "  return 0.9189385332046727 + (z + 0.5) * log(t) - t + log(abs(x));",
    "}",
    "fn bg_semantic_tgamma_f32(value: f32) -> f32 {",
    "  if (value != value) { return value; }",
    "  if (abs(value) > 3.4028234663852886e38) { return select(value, bg_semantic_runtime_nan_f32(value), value < 0.0); }",
    "  if (value <= 0.0 && value == trunc(value)) { return bg_semantic_runtime_nan_f32(value); }",
    "  if (value < 0.5) {",
    "    return 3.141592653589793 / (sin(3.141592653589793 * value) * bg_semantic_tgamma_core_f32(1.0 - value));",
    "  }",
    "  return bg_semantic_tgamma_core_f32(value);",
    "}",
    "fn bg_semantic_lgamma_f32(value: f32) -> f32 {",
    "  if (value != value) { return value; }",
    "  if (abs(value) > 3.4028234663852886e38) { return bg_semantic_runtime_inf_f32(value); }",
    "  if (value <= 0.0 && value == trunc(value)) { return bg_semantic_runtime_inf_f32(value); }",
    "  if (value < 0.5) {",
    "    return log(3.141592653589793) - log(abs(sin(3.141592653589793 * value))) - bg_semantic_lgamma_core_f32(1.0 - value);",
    "  }",
    "  return bg_semantic_lgamma_core_f32(value);",
    "}",
    "fn bg_semantic_umulhi_u32(x: u32, y: u32) -> u32 {",
    "  let x_lo = x & 0xffffu;",
    "  let x_hi = x >> 16u;",
    "  let y_lo = y & 0xffffu;",
    "  let y_hi = y >> 16u;",
    "  let lo_carry = (((x_lo * y_lo) >> 16u) + ((x_lo * y_hi) & 0xffffu) + ((x_hi * y_lo) & 0xffffu)) >> 16u;",
    "  return (x_hi * y_hi) + ((x_lo * y_hi) >> 16u) + ((x_hi * y_lo) >> 16u) + lo_carry;",
    "}",
    "fn bg_semantic_mulhi_i32(x: i32, y: i32) -> i32 {",
    "  return i32(bg_semantic_umulhi_u32(u32(x), u32(y))) - select(0, y, x < 0) - select(0, x, y < 0);",
    "}",
    "fn bg_semantic_byte_perm_u32(x: u32, y: u32, selector: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var lane = 0u; lane < 4u; lane = lane + 1u) {",
    "    let source = (selector >> (lane * 4u)) & 0x7u;",
    "    let input = select(x, y, source >= 4u);",
    "    let byte = (input >> ((source & 0x3u) * 8u)) & 0xffu;",
    "    out = out | (byte << (lane * 8u));",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_funnelshift_l_u32(lo: u32, hi: u32, shift: u32) -> u32 {",
    "  let s = shift & 31u;",
    "  if (s == 0u) { return lo; }",
    "  return (lo << s) | (hi >> (32u - s));",
    "}",
    "fn bg_semantic_funnelshift_lc_u32(lo: u32, hi: u32, shift: u32) -> u32 {",
    "  let s = min(shift, 32u);",
    "  if (s == 0u) { return lo; }",
    "  if (s == 32u) { return hi; }",
    "  return (lo << s) | (hi >> (32u - s));",
    "}",
    "fn bg_semantic_funnelshift_r_u32(lo: u32, hi: u32, shift: u32) -> u32 {",
    "  let s = shift & 31u;",
    "  if (s == 0u) { return lo; }",
    "  return (lo >> s) | (hi << (32u - s));",
    "}",
    "fn bg_semantic_funnelshift_rc_u32(lo: u32, hi: u32, shift: u32) -> u32 {",
    "  let s = min(shift, 32u);",
    "  if (s == 0u) { return lo; }",
    "  if (s == 32u) { return hi; }",
    "  return (lo >> s) | (hi << (32u - s));",
    "}",
    "fn bg_semantic_usad4_u32(a: u32, b: u32, c: u32) -> u32 {",
    "  var out = c;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out + max(left, right) - min(left, right);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vadd2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (((left + right) & 0xffffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vsub2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (((left - right) & 0xffffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vaddss2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left_bits = (a >> shift) & 0xffffu;",
    "    let right_bits = (b >> shift) & 0xffffu;",
    "    let left = i32(left_bits) - select(0, 65536, left_bits >= 0x8000u);",
    "    let right = i32(right_bits) - select(0, 65536, right_bits >= 0x8000u);",
    "    out = out | ((u32(clamp(left + right, -32768, 32767)) & 0xffffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vsubss2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left_bits = (a >> shift) & 0xffffu;",
    "    let right_bits = (b >> shift) & 0xffffu;",
    "    let left = i32(left_bits) - select(0, 65536, left_bits >= 0x8000u);",
    "    let right = i32(right_bits) - select(0, 65536, right_bits >= 0x8000u);",
    "    out = out | ((u32(clamp(left - right, -32768, 32767)) & 0xffffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vaddus2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (min(0xffffu, left + right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vsubus2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (select(0u, left - right, left >= right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vabsdiffu2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | ((max(left, right) - min(left, right)) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vavgu2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (((left + right + 1u) >> 1u) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vminu2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (min(left, right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vmaxu2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left = (a >> shift) & 0xffffu;",
    "    let right = (b >> shift) & 0xffffu;",
    "    out = out | (max(left, right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vmins2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left_bits = (a >> shift) & 0xffffu;",
    "    let right_bits = (b >> shift) & 0xffffu;",
    "    let left = i32(left_bits) - select(0, 65536, left_bits >= 0x8000u);",
    "    let right = i32(right_bits) - select(0, 65536, right_bits >= 0x8000u);",
    "    out = out | ((u32(min(left, right)) & 0xffffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vmaxs2_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 16u) {",
    "    let left_bits = (a >> shift) & 0xffffu;",
    "    let right_bits = (b >> shift) & 0xffffu;",
    "    let left = i32(left_bits) - select(0, 65536, left_bits >= 0x8000u);",
    "    let right = i32(right_bits) - select(0, 65536, right_bits >= 0x8000u);",
    "    out = out | ((u32(max(left, right)) & 0xffffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vadd4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (((left + right) & 0xffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vsub4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (((left - right) & 0xffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vaddss4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left_bits = (a >> shift) & 0xffu;",
    "    let right_bits = (b >> shift) & 0xffu;",
    "    let left = i32(left_bits) - select(0, 256, left_bits >= 0x80u);",
    "    let right = i32(right_bits) - select(0, 256, right_bits >= 0x80u);",
    "    out = out | ((u32(clamp(left + right, -128, 127)) & 0xffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vsubss4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left_bits = (a >> shift) & 0xffu;",
    "    let right_bits = (b >> shift) & 0xffu;",
    "    let left = i32(left_bits) - select(0, 256, left_bits >= 0x80u);",
    "    let right = i32(right_bits) - select(0, 256, right_bits >= 0x80u);",
    "    out = out | ((u32(clamp(left - right, -128, 127)) & 0xffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vaddus4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (min(0xffu, left + right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vsubus4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (select(0u, left - right, left >= right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vabsdiffu4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | ((max(left, right) - min(left, right)) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vavgu4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (((left + right + 1u) >> 1u) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vminu4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (min(left, right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vmaxu4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = (a >> shift) & 0xffu;",
    "    let right = (b >> shift) & 0xffu;",
    "    out = out | (max(left, right) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vmins4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left_bits = (a >> shift) & 0xffu;",
    "    let right_bits = (b >> shift) & 0xffu;",
    "    let left = i32(left_bits) - select(0, 256, left_bits >= 0x80u);",
    "    let right = i32(right_bits) - select(0, 256, right_bits >= 0x80u);",
    "    out = out | ((u32(min(left, right)) & 0xffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_vmaxs4_u32(a: u32, b: u32) -> u32 {",
    "  var out = 0u;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left_bits = (a >> shift) & 0xffu;",
    "    let right_bits = (b >> shift) & 0xffu;",
    "    let left = i32(left_bits) - select(0, 256, left_bits >= 0x80u);",
    "    let right = i32(right_bits) - select(0, 256, right_bits >= 0x80u);",
    "    out = out | ((u32(max(left, right)) & 0xffu) << shift);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_dp4a_i32(a: u32, b: u32, c: i32) -> i32 {",
    "  var out = c;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    let left = i32(((a >> shift) & 0xffu) << 24u) >> 24u;",
    "    let right = i32(((b >> shift) & 0xffu) << 24u) >> 24u;",
    "    out = out + (left * right);",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_dp4a_u32(a: u32, b: u32, c: u32) -> u32 {",
    "  var out = c;",
    "  for (var shift = 0u; shift < 32u; shift = shift + 8u) {",
    "    out = out + (((a >> shift) & 0xffu) * ((b >> shift) & 0xffu));",
    "  }",
    "  return out;",
    "}",
    "fn bg_semantic_dp2a_i32(a: u32, b: u32, c: i32, byte_shift: u32) -> i32 {",
    "  let left0_bits = a & 0xffffu;",
    "  let left1_bits = (a >> 16u) & 0xffffu;",
    "  let right0_bits = (b >> byte_shift) & 0xffu;",
    "  let right1_bits = (b >> (byte_shift + 8u)) & 0xffu;",
    "  let left0 = i32(left0_bits) - select(0, 65536, left0_bits >= 0x8000u);",
    "  let left1 = i32(left1_bits) - select(0, 65536, left1_bits >= 0x8000u);",
    "  let right0 = i32(right0_bits) - select(0, 256, right0_bits >= 0x80u);",
    "  let right1 = i32(right1_bits) - select(0, 256, right1_bits >= 0x80u);",
    "  return c + (left0 * right0) + (left1 * right1);",
    "}",
    "fn bg_semantic_dp2a_u32(a: u32, b: u32, c: u32, byte_shift: u32) -> u32 {",
    "  let left0 = a & 0xffffu;",
    "  let left1 = (a >> 16u) & 0xffffu;",
    "  let right0 = (b >> byte_shift) & 0xffu;",
    "  let right1 = (b >> (byte_shift + 8u)) & 0xffu;",
    "  return c + (left0 * right0) + (left1 * right1);",
    "}",
  ];
}

function semanticTextureDescriptorHelperName(
  textureName: string,
  names: ReadonlyMap<string, string>,
  descriptor: CudaLiteTextureDescriptor,
): string {
  return `bg_sem_tex2d_${safeWgslIdentifier(nameFor(textureName, names))}_${semanticStableHash(semanticTextureDescriptorKey(descriptor))}`;
}

function semanticTextureScaledCoord(
  value: string,
  extent: string,
  descriptor: CudaLiteTextureDescriptor,
): string {
  return descriptor.normalizedCoords ? `(${value} * f32(${extent}))` : value;
}

function semanticTextureIndex(
  value: string,
  extent: string,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): string {
  const mode = descriptor.addressMode?.[axis === "x" ? 0 : 1] ?? "clamp";
  const signedExtent = `i32(${extent})`;
  if (mode === "wrap") return `(((${value}) % ${signedExtent}) + ${signedExtent}) % ${signedExtent}`;
  return `clamp(${value}, 0, (${signedExtent} - 1))`;
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
  if (param?.pointer && param.addressSpace === "storage") {
    const ref = semanticPointerArgMemoryRef(arg);
    if (!ref || ref.addressSpace !== "storage") throw semanticWgslError("semantic WGSL storage pointer helper argument must be modeled storage", arg.span);
    const bufferId = semanticStoragePointerBufferId(ref.base, ir);
    if (bufferId === undefined) throw semanticWgslError(`semantic WGSL unknown storage pointer base '${ref.base}'`, arg.span);
    return [`${bufferId}u`, emitSemanticPointerArgBaseIndex(ref, ir, names, options)];
  }
  return [emitSemanticFunctionArg(arg, param, ir, names, options, textureSpecializations)];
}

function semanticPointerArgMemoryRef(expression: SemanticExpression): SemanticMemoryRef | undefined {
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "index") {
    return memoryRefFromIndexExpression(expression.argument);
  }
  if (expression.kind === "index") return memoryRefFromIndexExpression(expression);
  if (expression.kind === "symbol" && expression.addressSpace === "storage") {
    return {
      base: expression.name,
      addressSpace: expression.addressSpace,
      ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
      indices: [],
      fields: [],
      span: expression.span,
    };
  }
  return undefined;
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
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
    const valueType = ref.valueType ?? "float";
    const index = isCudaVectorType(valueType) ? emitFlatStorageVectorBaseIndex(ref, ir, names, options) : emitFlatStorageIndex(ref, ir, names, options);
    return `${semanticPointerReadHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(ref.base), names)}, ${index})`;
  }
  return emitSemanticMemoryRef(ref, ir, names, options);
}

function emitSemanticMemoryWrite(
  ref: SemanticMemoryRef,
  value: string,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (semanticWgslFunctionStoragePointerParam(ir, ref.base)) {
    const valueType = ref.valueType ?? "float";
    const index = isCudaVectorType(valueType) ? emitFlatStorageVectorBaseIndex(ref, ir, names, options) : emitFlatStorageIndex(ref, ir, names, options);
    return `${semanticPointerWriteHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(ref.base), names)}, ${index}, ${value})`;
  }
  const target = emitSemanticMemoryRef(ref, ir, names, options);
  return `${target} = ${value}`;
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
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
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
  const floatAtomicKind = target.valueType === "float" ? semanticWgslFloatAtomicCallKind(expression.callee.name) : undefined;
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
  if (expression.callee.kind !== "symbol" || expression.callee.name !== "atomicCAS" && expression.callee.name !== "atomicCAS_system") {
    throw semanticWgslError(`semantic WGSL pointer atomic '${expression.callee.kind === "symbol" ? expression.callee.name : "<expr>"}' is unsupported`, expression.span);
  }
  const [compare, value] = expression.args.slice(1);
  if (!compare || !value) throw semanticWgslError(`semantic WGSL atomic '${expression.callee.name}' missing operand`, expression.span);
  const valueType = target.valueType ?? "float";
  const index = isCudaVectorType(valueType)
    ? emitFlatStorageVectorBaseIndex(target, ir, names, options)
    : emitFlatStorageIndex(target, ir, names, options);
  return `${semanticPointerAtomicCasHelperName(valueType)}(${nameFor(semanticPointerBufferParamName(target.base), names)}, ${index}, ${emitSemanticExpressionAs(compare, ir, names, wgslValueScalar(valueType), options, textureSpecializations)}, ${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations)})`;
}

function emitSemanticAddressPredicateCall(expression: Extract<SemanticExpression, { readonly kind: "call" }>): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL address predicate requires symbol callee", expression.span);
  const addressSpace = semanticAddressPredicateAddressSpace(expression.args[0]);
  const matches =
    expression.callee.name === "__isGlobal" ? addressSpace === "storage" || addressSpace === "device-global" :
      expression.callee.name === "__isShared" ? addressSpace === "shared" :
        expression.callee.name === "__isConstant" ? addressSpace === "constant" :
          expression.callee.name === "__isLocal" ? addressSpace === "local" :
            false;
  return matches ? "1" : "0";
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
  if (name === "__activemask") return "subgroupBallot(true).x";
  const value = expression.args[legacyVoteCall(name) || legacyShuffleCall(name) ? 0 : 1];
  if (!value) throw semanticWgslError(`${name} expects value operand`, expression.span);
  if (name === "__any" || name === "__all" || name === "__ballot" || name === "__any_sync" || name === "__all_sync" || name === "__ballot_sync") {
    const predicate = emitTruthiness(value, ir, names, options);
    if (name === "__any" || name === "__any_sync") return `select(0u, 1u, subgroupAny(${predicate}))`;
    if (name === "__all" || name === "__all_sync") return `select(0u, 1u, subgroupAll(${predicate}))`;
    return `subgroupBallot(${predicate}).x`;
  }
  if (name === "__match_any_sync") {
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
    const helper = semanticWarpShuffleHelper(shuffleOp, valueType, 32);
    const indexArg = legacyShuffleCall(name) ? expression.args[1] : expression.args[2];
    const widthArg = legacyShuffleCall(name) ? expression.args[2] : expression.args[3];
    const index = indexArg ? emitSemanticExpressionAs(indexArg, ir, names, "u32", options, textureSpecializations) : "0u";
    const width = widthArg ? emitSemanticExpressionAs(widthArg, ir, names, "u32", options, textureSpecializations) : "32u";
    return `${helper.name}(${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(valueType), options, textureSpecializations)}, ${index}, ${width}, local_id)`;
  }
  if (name === "__reduce_add_sync" || name === "__reduce_min_sync" || name === "__reduce_max_sync") {
    const scalar = semanticExpressionWgslScalar(value);
    const wgslCall = name === "__reduce_add_sync" ? "subgroupAdd" : name === "__reduce_min_sync" ? "subgroupMin" : "subgroupMax";
    return `${wgslCall}(${emitSemanticExpressionAs(value, ir, names, scalar, options, textureSpecializations)})`;
  }
  throw semanticWgslError(`semantic WGSL does not support subgroup call '${name}'`, expression.span);
}

function legacyVoteCall(name: string): boolean {
  return name === "__any" || name === "__all" || name === "__ballot";
}

function legacyShuffleCall(name: string): boolean {
  return name === "__shfl" || name === "__shfl_down" || name === "__shfl_up" || name === "__shfl_xor";
}

function semanticAddressPredicateAddressSpace(expression: SemanticExpression | undefined): SemanticAddressSpace | undefined {
  if (!expression) return undefined;
  if (expression.kind === "symbol") return expression.addressSpace;
  if (expression.kind === "index") return expression.addressSpace;
  if (expression.kind === "member") return semanticAddressPredicateAddressSpace(expression.object);
  if (expression.kind === "cast" && expression.pointer) return semanticAddressPredicateAddressSpace(expression.expression);
  if (expression.kind === "unary" && expression.operator === "&") return semanticAddressPredicateAddressSpace(expression.argument);
  if (expression.kind === "conditional") {
    const consequent = semanticAddressPredicateAddressSpace(expression.consequent);
    const alternate = semanticAddressPredicateAddressSpace(expression.alternate);
    return consequent === alternate ? consequent : undefined;
  }
  return undefined;
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

function emitSemanticVSetExpression(lhs: string, rhs: string, laneWidth: 8 | 16, signed: boolean, operator: string): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const comparisons: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : leftBits;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : rightBits;
    comparisons.push(`(${left} ${operator} ${right})`);
  }
  return `select(0u, 1u, ${comparisons.join(" && ")})`;
}

function emitSemanticVCompareExpression(lhs: string, rhs: string, laneWidth: 8 | 16, signed: boolean, operator: string): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : leftBits;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : rightBits;
    lanes.push(`(select(0u, ${mask}, (${left} ${operator} ${right})) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

function emitSemanticVPackedUnaryExpression(value: string, laneWidth: 8 | 16, op: "abs" | "sat_abs" | "neg" | "sat_neg"): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const minValue = laneWidth === 8 ? "-128" : "-32768";
  const maxValue = laneWidth === 8 ? "127" : "32767";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const bits = `((${value} >> ${shift}) & ${mask})`;
    const signed = `(i32(${bits}) - select(0, ${signSub}, ${bits} >= ${signBit}))`;
    const result =
      op === "abs" ? `abs(${signed})` :
      op === "sat_abs" ? `min(${maxValue}, abs(${signed}))` :
      op === "neg" ? `(-${signed})` :
      `clamp(-${signed}, ${minValue}, ${maxValue})`;
    lanes.push(`((u32(${result}) & ${mask}) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

function emitSemanticVPackedAbsDiffExpression(lhs: string, rhs: string, laneWidth: 8 | 16): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))`;
    const right = `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))`;
    lanes.push(`((u32(abs(${left} - ${right})) & ${mask}) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

function emitSemanticVPackedSadExpression(lhs: string, rhs: string, laneWidth: 8 | 16, signed: boolean): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : `i32(${leftBits})`;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : `i32(${rightBits})`;
    lanes.push(`u32(abs(${left} - ${right}))`);
  }
  return `(${lanes.join(" + ")})`;
}

function emitSemanticVPackedAverageExpression(lhs: string, rhs: string, laneWidth: 8 | 16, signedRounded: boolean): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    if (signedRounded) {
      const left = `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))`;
      const right = `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))`;
      lanes.push(`((u32((${left} + ${right} + 1) >> 1u) & ${mask}) << ${shift})`);
    } else {
      lanes.push(`(((${leftBits} + ${rightBits}) >> 1u) << ${shift})`);
    }
  }
  return `(${lanes.join(" | ")})`;
}

function emitSemanticViadd16x2Expression(lhs: string, rhs: string, cmpValue: string, signed: boolean, choose: "max" | "min", relu: boolean): string {
  const lanes: string[] = [];
  for (const shift of ["0u", "16u"]) {
    const leftBits = `((${lhs} >> ${shift}) & 0xffffu)`;
    const rightBits = `((${rhs} >> ${shift}) & 0xffffu)`;
    const cmpBits = `((${cmpValue} >> ${shift}) & 0xffffu)`;
    const left = signed ? `(i32(${leftBits}) - select(0, 65536, ${leftBits} >= 0x8000u))` : `i32(${leftBits})`;
    const right = signed ? `(i32(${rightBits}) - select(0, 65536, ${rightBits} >= 0x8000u))` : `i32(${rightBits})`;
    const cmp = signed ? `(i32(${cmpBits}) - select(0, 65536, ${cmpBits} >= 0x8000u))` : `i32(${cmpBits})`;
    const selected = `${choose}((${left} + ${right}), ${cmp})`;
    const value = relu ? `max(${selected}, 0)` : selected;
    lanes.push(`((u32(${value}) & 0xffffu) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

function emitSemanticViMinMax16x2Expression(inputs: readonly string[], signed: boolean, choose: "max" | "min", relu: boolean): string {
  const lanes: string[] = [];
  for (const shift of ["0u", "16u"]) {
    const values = inputs.map((input) => {
      const bits = `((${input} >> ${shift}) & 0xffffu)`;
      return signed ? `(i32(${bits}) - select(0, 65536, ${bits} >= 0x8000u))` : `i32(${bits})`;
    });
    const selected = values.slice(1).reduce((acc, value) => `${choose}(${acc}, ${value})`, values[0] ?? "0");
    const value = relu ? `max(${selected}, 0)` : selected;
    lanes.push(`((u32(${value}) & 0xffffu) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

function emitRoundEvenWgsl(emitted: string): string {
  return `bg_semantic_round_even_f32(${emitted})`;
}

function halfConversionModeLiteral(callee: string): "0u" | "1u" | "2u" | "3u" {
  if (callee.endsWith("_rn")) return "0u";
  if (callee.endsWith("_rz")) return "1u";
  if (callee.endsWith("_ru")) return "2u";
  return "3u";
}

function wgslRoundBfloat16(value: string, mode: "0u" | "1u" | "2u" | "3u" = "0u"): string {
  return `bitcast<f32>(bg_f32_to_bf16_bits_mode(f32(${value}), ${mode}) << 16u)`;
}

function wgslSaturateHalf(value: string): string {
  return `select(clamp(${value}, f16(0.0), f16(1.0)), f16(0.0), (${value}) != (${value}))`;
}

function wgslSaturateHalf2(value: string): string {
  return `select(clamp(${value}, vec2<f16>(0.0), vec2<f16>(1.0)), vec2<f16>(0.0), (${value}) != (${value}))`;
}

function wgslSaturateBfloat16(value: string): string {
  return wgslRoundBfloat16(`select(clamp(${value}, 0.0, 1.0), 0.0, (${value}) != (${value}))`);
}

function wgslReluBfloat16(value: string): string {
  return wgslRoundBfloat16(`select(max(${value}, 0.0), ${value}, (${value}) != (${value}))`);
}

function wgslSaturateBf162(value: string): string {
  return `select(clamp(${value}, vec2<f32>(0.0), vec2<f32>(1.0)), vec2<f32>(0.0), (${value}) != (${value}))`;
}

function wgslReluBf162(value: string): string {
  return `select(max(${value}, vec2<f32>(0.0)), ${value}, (${value}) != (${value}))`;
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
  if (!isSemanticFloatVectorType(semanticExpressionVectorValueType(expression.object, ir?.functions))) {
    throw semanticWgslError("semantic WGSL supports builtin vector members only", expression.span);
  }
  return `${emitSemanticExpression(expression.object, ir, names, options)}.${semanticVectorFieldName(expression)}`;
}

function semanticVectorFieldName(expression: Extract<SemanticExpression, { readonly kind: "member" }>): string {
  const valueType = semanticExpressionVectorValueType(expression.object);
  const fields = valueType === undefined ? undefined : cudaVectorSwizzleIndices(valueType, expression.property);
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
  if (expression.operator === "~") return `~(${emitSemanticExpression(expression.argument, ir, names, options, textureSpecializations)})`;
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
  if (LOGICAL_OPERATORS.has(expression.operator)) {
    return `(${emitTruthiness(expression.left, ir, names, options)} ${expression.operator} ${emitTruthiness(expression.right, ir, names, options)})`;
  }
  if (isSemanticFloatVectorType(expression.valueType) && semanticWgslVectorBinaryOperatorSupported(expression.operator)) {
    const valueType = expression.valueType as CudaLiteScalarType;
    return `(${emitSemanticVectorOperand(expression.left, valueType, ir, names, options, textureSpecializations)} ${expression.operator} ${emitSemanticVectorOperand(expression.right, valueType, ir, names, options, textureSpecializations)})`;
  }
  const operandType = semanticBinaryOperandType(expression);
  const left = emitSemanticExpressionAs(expression.left, ir, names, operandType, options, textureSpecializations);
  const right = emitSemanticExpressionAs(expression.right, ir, names, operandType, options, textureSpecializations);
  return `(${left} ${expression.operator} ${right})`;
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

function emitTruthiness(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (semanticExpressionValueType(expression) === "bool") {
    return emitSemanticExpression(expression, ir, names, options);
  }
  if (expression.kind === "binary" && (COMPARISON_OPERATORS.has(expression.operator) || LOGICAL_OPERATORS.has(expression.operator))) {
    return emitSemanticBinary(expression, ir, names, options);
  }
  const scalar = semanticExpressionWgslScalar(expression);
  const zero = scalar === "u32" ? "0u" : scalar === "f32" ? "0.0" : "0";
  return `(${emitSemanticExpression(expression, ir, names, options)} != ${zero})`;
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
    const local = localMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
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

function emitSemanticAtomicLoad(ref: SemanticMemoryRef, memoryRef: string): string {
  const loaded = `atomicLoad(&${memoryRef})`;
  return ref.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitSemanticVectorMemoryRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (!isSemanticFloatVectorType(ref.valueType)) throw semanticWgslError("semantic WGSL vector read requires vector memory type", ref.span);
  const base = emitFlatStorageVectorBaseIndex(ref, ir, names, options);
  const storage = nameFor(ref.base, names);
  const laneCount = cudaVectorLaneCount(ref.valueType);
  const atomicStorage = semanticAtomicStorageNames(ir.operations, ir.functions).has(ref.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations).has(ref.base) ||
    semanticAtomicSharedNames(ir.operations).has(ref.base);
  return `${wgslValueType(ref.valueType)}(${Array.from({ length: laneCount }, (_, lane) => {
    const access = `${storage}[(${base} + ${lane}u)]`;
    return atomicStorage ? `bitcast<f32>(atomicLoad(&${access}))` : access;
  }).join(", ")})`;
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
  if (expression.kind !== "index") return undefined;
  const flattened = flattenMemoryRef(expression);
  if (!flattened || (flattened.base.addressSpace !== "storage" && flattened.base.addressSpace !== "shared" && flattened.base.addressSpace !== "constant" && flattened.base.addressSpace !== "device-global" && flattened.base.addressSpace !== "local")) return undefined;
  return {
    base: flattened.base.name,
    addressSpace: flattened.base.addressSpace,
    ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
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

function sharedMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "shared");
}

function constantMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "constant");
}

function deviceGlobalMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "device-global");
}

function textureSymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  const byName = new Map<string, SemanticKernelIrModule["memory"][number]>();
  for (const param of ir.params.filter((symbol) => symbol.addressSpace === "texture")) byName.set(param.name, param);
  for (const symbol of ir.memory.filter((item) => item.kind === "texture")) byName.set(symbol.name, symbol);
  return [...byName.values()];
}

function surfaceSymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["params"][number][] {
  return ir.params.filter((symbol) => symbol.addressSpace === "surface");
}

function surfaceHandleForName(name: string, ir: SemanticKernelIrModule): number | undefined {
  const index = surfaceSymbols(ir).findIndex((surface) => surface.name === name);
  return index < 0 ? undefined : index;
}

function semanticUsesGenericSurfaceRead(ir: SemanticKernelIrModule): boolean {
  return ir.functions.some((fn) => fn.params.some((param) => param.addressSpace === "surface") && semanticOperationsUseSurfaceParamRead(fn.body, new Set(fn.params.filter((param) => param.addressSpace === "surface").map((param) => param.name))));
}

function semanticUsesGenericSurfaceWrite(ir: SemanticKernelIrModule): boolean {
  return ir.functions.some((fn) => fn.params.some((param) => param.addressSpace === "surface") && semanticOperationsUseSurfaceParamWrite(fn.body, new Set(fn.params.filter((param) => param.addressSpace === "surface").map((param) => param.name))));
}

function semanticUsesFp8(ir: SemanticKernelIrModule): boolean {
  return semanticOperationsUseFp8(ir.operations) || ir.functions.some((fn) => semanticOperationsUseFp8(fn.body));
}

function semanticUsesHalfConversion(ir: SemanticKernelIrModule): boolean {
  return semanticOperationsUseHalfConversion(ir.operations) || ir.functions.some((fn) => semanticOperationsUseHalfConversion(fn.body));
}

function semanticUsesBfloatHelper(ir: SemanticKernelIrModule): boolean {
  return ir.params.some((param) => param.valueType === "bf16" || param.valueType === "bf162") ||
    ir.memory.some((memory) => memory.valueType === "bf16" || memory.valueType === "bf162") ||
    semanticOperationsUseBfloatHelper(ir.operations) ||
    ir.functions.some((fn) =>
      fn.params.some((param) => param.valueType === "bf16" || param.valueType === "bf162") ||
      semanticOperationsUseBfloatHelper(fn.body));
}

function semanticUsesCurand(ir: SemanticKernelIrModule): boolean {
  return semanticOperationsUseCurand(ir.operations) || ir.functions.some((fn) => semanticOperationsUseCurand(fn.body));
}

function semanticOperationsUseCurand(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (operation.kind === "call" && SEMANTIC_CURAND_CALLS.has(operation.callee)) return true;
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesCurand)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseCurand(operation.consequent) || semanticOperationsUseCurand(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseCurand(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseCurand(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesCurand(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_CURAND_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesCurand);
}

function semanticOperationsUseFp8(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesFp8)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseFp8(operation.consequent) || semanticOperationsUseFp8(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseFp8(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseFp8(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesFp8(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_FP8_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesFp8);
}

function semanticOperationsUseHalfConversion(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesHalfConversion)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseHalfConversion(operation.consequent) || semanticOperationsUseHalfConversion(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseHalfConversion(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseHalfConversion(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesHalfConversion(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_HALF_CONVERSION_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesHalfConversion);
}

function semanticOperationsUseBfloatHelper(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (operation.kind === "declare" && (operation.target.valueType === "bf16" || operation.target.valueType === "bf162")) return true;
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesBfloatHelper)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseBfloatHelper(operation.consequent) || semanticOperationsUseBfloatHelper(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseBfloatHelper(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseBfloatHelper(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesBfloatHelper(expression: SemanticExpression): boolean {
  const valueType = semanticExpressionValueType(expression);
  if (valueType === "bf16" || valueType === "bf162") return true;
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_BFLOAT_HELPER_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesBfloatHelper);
}

function semanticOperationsUseSurfaceParamWrite(
  operations: readonly SemanticKernelIrOperation[],
  surfaceParams: ReadonlySet<string>,
): boolean {
  for (const operation of operations) {
    if (operation.kind === "surface-write" && operation.surface.kind === "symbol" && surfaceParams.has(operation.surface.name)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseSurfaceParamWrite(operation.consequent, surfaceParams) || semanticOperationsUseSurfaceParamWrite(operation.alternate, surfaceParams))) return true;
    if (operation.kind === "loop" && semanticOperationsUseSurfaceParamWrite(operation.body, surfaceParams)) return true;
  }
  return false;
}

function semanticOperationsUseSurfaceParamRead(
  operations: readonly SemanticKernelIrOperation[],
  surfaceParams: ReadonlySet<string>,
): boolean {
  for (const operation of operations) {
    if (operation.kind === "return" && operation.value && semanticExpressionUsesSurfaceParamRead(operation.value, surfaceParams)) return true;
    if (operation.kind === "expression" && semanticExpressionUsesSurfaceParamRead(operation.expression, surfaceParams)) return true;
    if (operation.kind === "declare" && operation.init && semanticExpressionUsesSurfaceParamRead(operation.init, surfaceParams)) return true;
    if (operation.kind === "store" && semanticExpressionUsesSurfaceParamRead(operation.value, surfaceParams)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseSurfaceParamRead(operation.consequent, surfaceParams) || semanticOperationsUseSurfaceParamRead(operation.alternate, surfaceParams))) return true;
    if (operation.kind === "loop" && semanticOperationsUseSurfaceParamRead(operation.body, surfaceParams)) return true;
  }
  return false;
}

function semanticExpressionUsesSurfaceParamRead(
  expression: SemanticExpression,
  surfaceParams: ReadonlySet<string>,
): boolean {
  if (expression.kind === "surface-read") return expression.surface.kind === "symbol" && surfaceParams.has(expression.surface.name);
  if (expression.kind === "call") return expression.args.some((arg) => semanticExpressionUsesSurfaceParamRead(arg, surfaceParams));
  if (expression.kind === "member") return semanticExpressionUsesSurfaceParamRead(expression.object, surfaceParams);
  if (expression.kind === "index") return semanticExpressionUsesSurfaceParamRead(expression.target, surfaceParams) || semanticExpressionUsesSurfaceParamRead(expression.index, surfaceParams);
  if (expression.kind === "cast") return semanticExpressionUsesSurfaceParamRead(expression.expression, surfaceParams);
  if (expression.kind === "unary" || expression.kind === "update") return semanticExpressionUsesSurfaceParamRead(expression.argument, surfaceParams);
  if (expression.kind === "binary") return semanticExpressionUsesSurfaceParamRead(expression.left, surfaceParams) || semanticExpressionUsesSurfaceParamRead(expression.right, surfaceParams);
  if (expression.kind === "conditional") return semanticExpressionUsesSurfaceParamRead(expression.condition, surfaceParams) || semanticExpressionUsesSurfaceParamRead(expression.consequent, surfaceParams) || semanticExpressionUsesSurfaceParamRead(expression.alternate, surfaceParams);
  if (expression.kind === "assignment") return semanticExpressionUsesSurfaceParamRead(expression.target, surfaceParams) || semanticExpressionUsesSurfaceParamRead(expression.value, surfaceParams);
  if (expression.kind === "initializer") return expression.elements.some((item) => semanticExpressionUsesSurfaceParamRead(item, surfaceParams));
  if (expression.kind === "sequence") return expression.expressions.some((item) => semanticExpressionUsesSurfaceParamRead(item, surfaceParams));
  return false;
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

function localMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "local" && symbol.dimensions.length > 0);
}

function localArraySymbol(ir: SemanticKernelIrModule, name: string): SemanticKernelIrModule["memory"][number] | undefined {
  return ir.memory.find((symbol) => symbol.kind === "local" && symbol.name === name && symbol.dimensions.length > 0);
}

function emitLocalArrayType(symbol: SemanticKernelIrModule["memory"][number]): string {
  return symbol.dimensions.reduceRight<string>(
    (element, dimension) => `array<${element}, ${Math.max(1, dimension)}>`,
    wgslValueType(symbol.valueType),
  );
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
  const element = atomic ? `atomic<${wgslAtomicScalar(symbol.valueType)}>` : wgslScalar(symbol.valueType);
  if (symbol.dimensions.length === 0) return element;
  return `array<${element}, ${Math.max(1, totalElements(symbol.dimensions))}>`;
}

function storageOffsetSymbol(base: string): string {
  return `${base}__bg_ptr_offset`;
}

function semanticPointerBufferParamName(base: string): string {
  return `${base}_buffer`;
}

function semanticPointerBaseParamName(base: string): string {
  return `${base}_base`;
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
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options).has(ref.base);
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
  const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options).has(ref.base);
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
    const valueType = isSemanticFloatVectorType(ref.containerValueType) ? ref.containerValueType : pointerParam.valueType;
    const stride = isSemanticFloatVectorType(valueType) ? cudaVectorLaneCount(valueType) : 1;
    const index = indexTerms.length === 0 ? "0u" : indexTerms.length === 1 ? indexTerms[0]! : `(${indexTerms.join(" + ")})`;
    const offset = stride === 1 ? index : `(${index} * ${stride}u)`;
    return `(${nameFor(semanticPointerBaseParamName(ref.base), names)} + ${offset})`;
  }
  const base = emitFlatStorageIndex({ ...ref, valueType: "float" }, ir, names, options);
  const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
  const valueType = isSemanticFloatVectorType(ref.containerValueType) ? ref.containerValueType : root?.valueType;
  const stride = isSemanticFloatVectorType(valueType) ? cudaVectorLaneCount(valueType) : 1;
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
  if (indices.length !== symbol.dimensions.length) {
    throw semanticWgslError(`shared memory '${symbol.name}' index rank mismatch`, symbol.span);
  }
  const terms = indices.map((index, offset) => {
    const stride = dimensionStride(symbol.dimensions, offset);
    const emitted = emitSemanticExpressionAs(index, ir, names, "u32");
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
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
  if (indices.length !== symbol.dimensions.length) {
    throw semanticWgslError(`device-global memory '${symbol.name}' index rank mismatch`, span);
  }
  const terms = indices.map((index, offset) => {
    const stride = dimensionStride(symbol.dimensions, offset);
    const emitted = emitSemanticExpressionAs(index, ir, names, "u32");
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
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
  if (indices.length !== symbol.dimensions.length) {
    throw semanticWgslError(`constant memory '${symbol.name}' index rank mismatch`, span);
  }
  const terms = indices.map((index, offset) => {
    const stride = dimensionStride(symbol.dimensions, offset);
    const emitted = emitSemanticExpressionAs(index, ir, names, "u32");
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function emitFlatLocalArrayIndexes(flat: string, dimensions: readonly number[]): string {
  return dimensions.map((dimension, offset) => {
    const stride = dimensionStride(dimensions, offset);
    const quotient = stride === 1 ? flat : `(${flat} / ${stride}u)`;
    return `[${dimension > 1 ? `(${quotient} % ${Math.max(1, dimension)}u)` : "0u"}]`;
  }).join("");
}

function semanticStorageOffsetBaseNames(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  options: EmitSemanticKernelIrWgslOptions = {},
): Set<string> {
  const out = new Set(ir.params
    .filter((param) =>
      param.addressSpace === "storage" &&
      param.pointer &&
      options.pointerBaseOffsets?.[param.name] !== undefined
    )
    .map((param) => param.name));
  collectSemanticStorageOffsetBaseNames(operations, out);
  return out;
}

function collectSemanticStorageOffsetBaseNames(
  operations: readonly SemanticKernelIrOperation[],
  out: Set<string>,
): void {
  for (const operation of operations) {
    if (
      operation.kind === "store" &&
      operation.target.addressSpace === "storage" &&
      operation.target.indices.length === 0 &&
      operation.target.fields.length === 0 &&
      (operation.operator === "+=" || operation.operator === "-=")
    ) out.add(operation.target.base);
    if (operation.kind === "branch") collectSemanticStorageOffsetBaseNames([...operation.consequent, ...operation.alternate], out);
    if (operation.kind === "loop") collectSemanticStorageOffsetBaseNames(operation.body, out);
  }
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

function wgslBindingType(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  if (scalar !== "bool") return scalar;
  return "u32";
}

function wgslScalar(valueType: CudaLiteScalarType | undefined): WgslValueType | "bool" {
  if (valueType === "half" || valueType === "half2") return "f16";
  const scalarType = isCudaVectorType(valueType) ? cudaVectorScalarType(valueType) : valueType;
  if (scalarType === "int") return "i32";
  if (scalarType === "uint") return "u32";
  if (scalarType === "half") return "f16";
  if (valueType === "bool") return "bool";
  return "f32";
}

function wgslValueType(valueType: CudaLiteScalarType | undefined): SemanticWgslValueType {
  if (valueType === "float2") return "vec2<f32>";
  if (valueType === "float3") return "vec3<f32>";
  if (valueType === "float4") return "vec4<f32>";
  if (valueType === "half2") return "vec2<f16>";
  if (valueType === "bf162") return "vec2<f32>";
  if (valueType === "int2") return "vec2<i32>";
  if (valueType === "int3") return "vec3<i32>";
  if (valueType === "int4") return "vec4<i32>";
  if (valueType === "uint2") return "vec2<u32>";
  if (valueType === "uint3") return "vec3<u32>";
  if (valueType === "uint4") return "vec4<u32>";
  return wgslScalar(valueType);
}

function wgslVectorScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "half2") return "f16";
  if (valueType === "int2" || valueType === "int3" || valueType === "int4") return "i32";
  if (valueType === "uint2" || valueType === "uint3" || valueType === "uint4") return "u32";
  return "f32";
}

function wgslValueScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  return scalar === "bool" ? "u32" : scalar;
}

function wgslAtomicScalar(valueType: CudaLiteScalarType | undefined): Extract<WgslValueType, "i32" | "u32"> {
  return valueType === "int" ? "i32" : "u32";
}

function wgslUniformScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "half") return "f16";
  if (valueType === "int") return "i32";
  if (valueType === "uint" || valueType === "bool") return "u32";
  return "f32";
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
      return "f32";
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
  if (type === "u32") return `${Math.trunc(value) >>> 0}u`;
  if (type === "i32" && value > 2147483647) return `bitcast<i32>(${Math.trunc(value) >>> 0}u)`;
  if (type === "i32") return String(Math.trunc(value));
  if (type === "f16") return `f16(${Number.isInteger(value) ? `${value}.0` : String(value)})`;
  if (Number.isInteger(value)) return `${value}.0`;
  return String(value);
}

function zeroExpression(span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: 0, valueType: "int", span };
}

function zeroForType(valueType: SemanticWgslValueType): string {
  if (valueType === "u32") return "0u";
  if (valueType === "i32") return "0";
  if (valueType === "bool") return "false";
  if (valueType === "f16") return "f16(0.0)";
  if (valueType === "vec2<f16>") return "vec2<f16>(f16(0.0))";
  if (valueType === "vec2<f32>") return "vec2<f32>(0.0)";
  if (valueType === "vec3<f32>") return "vec3<f32>(0.0)";
  if (valueType === "vec4<f32>") return "vec4<f32>(0.0)";
  if (valueType === "vec2<i32>") return "vec2<i32>(0)";
  if (valueType === "vec3<i32>") return "vec3<i32>(0)";
  if (valueType === "vec4<i32>") return "vec4<i32>(0)";
  if (valueType === "vec2<u32>") return "vec2<u32>(0u)";
  if (valueType === "vec3<u32>") return "vec3<u32>(0u)";
  if (valueType === "vec4<u32>") return "vec4<u32>(0u)";
  return "0.0";
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
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.body)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicSharedNames(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "shared") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicNamesFromOperation(operation, "shared")) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicSharedNames(operation.consequent)) names.add(name);
      for (const name of semanticAtomicSharedNames(operation.alternate)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicSharedNames([operation.init])) names.add(name);
      }
      for (const name of semanticAtomicSharedNames(operation.body)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicSharedNames(operation.body)) names.add(name);
    }
  }
  return names;
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

function semanticExpressionChildren(expression: SemanticExpression): readonly SemanticExpression[] {
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return [];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "call":
      return [expression.callee, ...expression.args];
    case "texture-read":
      return [expression.texture, expression.x, expression.y];
    case "surface-read":
      return [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])];
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
  }
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
  if (firstArg.kind === "symbol" && firstArg.addressSpace === "storage") {
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
  if (BUILTIN_VECTOR_NAMES.has(name)) return name;
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
