import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslValueType,
} from "@unlocalhosted/browsergrad-kernels";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import type {
  CudaLiteDiagnostic,
  CudaLiteTextureDescriptor,
  CudaLiteScalarType,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { createWgslNameMap, safeWgslIdentifier } from "./wgsl_names.js";
import { cudaVectorConstructorType, cudaVectorFieldIndex, cudaVectorLaneCount, isCudaVectorType } from "./vector_types.js";

export interface SemanticKernelIrWgslOutput {
  readonly wgsl: string;
  readonly program: ReturnType<typeof defineWgslKernelProgram>;
}

export interface EmitSemanticKernelIrWgslOptions {
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
  readonly textureDescriptors?: Readonly<Record<string, CudaLiteTextureDescriptor>>;
}

interface SemanticTextureDescriptorSignature {
  readonly key: string;
  readonly descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>;
}

type SemanticTextureDescriptorSpecializations = ReadonlyMap<string, ReadonlyMap<string, SemanticTextureDescriptorSignature>>;
type SemanticWgslValueType = WgslValueType | "bool" | "vec2<f32>" | "vec3<f32>" | "vec4<f32>";

interface SemanticTextureDescriptorHelper {
  readonly textureName: string;
  readonly descriptor: CudaLiteTextureDescriptor;
}

const UNIFORM_PARAMS_NAME = "bg_uniforms";
const BUILTIN_VECTOR_NAMES = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const LOGICAL_OPERATORS = new Set(["&&", "||"]);
const SEMANTIC_MATH_CALLS = new Map([
  ["sqrt", "sqrt"],
  ["sqrtf", "sqrt"],
  ["__fsqrt_rn", "sqrt"],
  ["rsqrt", "inverseSqrt"],
  ["rsqrtf", "inverseSqrt"],
  ["__frsqrt_rn", "inverseSqrt"],
  ["exp", "exp"],
  ["expf", "exp"],
  ["__expf", "exp"],
  ["log", "log"],
  ["logf", "log"],
  ["__logf", "log"],
  ["fabs", "abs"],
  ["fabsf", "abs"],
  ["abs", "abs"],
  ["floor", "floor"],
  ["floorf", "floor"],
  ["ceil", "ceil"],
  ["ceilf", "ceil"],
  ["trunc", "trunc"],
  ["truncf", "trunc"],
  ["sin", "sin"],
  ["sinf", "sin"],
  ["__sinf", "sin"],
  ["cos", "cos"],
  ["cosf", "cos"],
  ["__cosf", "cos"],
  ["tan", "tan"],
  ["tanf", "tan"],
  ["__tanf", "tan"],
  ["atan", "atan"],
  ["atanf", "atan"],
  ["atan2", "atan2"],
  ["atan2f", "atan2"],
  ["tanh", "tanh"],
  ["tanhf", "tanh"],
  ["__tanhf", "tanh"],
  ["fmin", "min"],
  ["fminf", "min"],
  ["min", "min"],
  ["fmax", "max"],
  ["fmaxf", "max"],
  ["max", "max"],
  ["pow", "pow"],
  ["powf", "pow"],
  ["__fdividef", "divide"],
  ["fma", "fma"],
  ["fmaf", "fma"],
  ["__fmaf_rn", "fma"],
  ["lerp", "lerp"],
  ["div_ceil", "div_ceil"],
  ["ceil_div", "div_ceil"],
  ["__bg_modf_intpart", "modf_intpart"],
  ["__bg_modf_fraction", "modf_fraction"],
  ["__bg_frexp_exponent", "frexp_exponent"],
  ["__bg_frexp_mantissa", "frexp_mantissa"],
  ["__bg_remquo_quotient", "remquo_quotient"],
  ["__bg_remquo_remainder", "remquo_remainder"],
]);
const SEMANTIC_LOCAL_ARRAY_FILL_CALLS = new Set(["fill_1D_regs", "fill_2D_regs", "fill_3D_regs"]);
const WGSL_ATOMIC_CALLEES = new Map([
  ["atomicAdd", "atomicAdd"],
  ["atomicAdd_system", "atomicAdd"],
  ["atomicSub", "atomicSub"],
  ["atomicSub_system", "atomicSub"],
  ["atomicMin", "atomicMin"],
  ["atomicMin_system", "atomicMin"],
  ["atomicMax", "atomicMax"],
  ["atomicMax_system", "atomicMax"],
  ["atomicAnd", "atomicAnd"],
  ["atomicAnd_system", "atomicAnd"],
  ["atomicOr", "atomicOr"],
  ["atomicOr_system", "atomicOr"],
  ["atomicXor", "atomicXor"],
  ["atomicXor_system", "atomicXor"],
  ["atomicExch", "atomicExchange"],
  ["atomicExch_system", "atomicExchange"],
  ["atomicCAS", "atomicCompareExchangeWeak"],
  ["atomicCAS_system", "atomicCompareExchangeWeak"],
]);

export function canEmitSemanticKernelIrWgsl(
  ir: SemanticKernelIrModule,
  _options: EmitSemanticKernelIrWgslOptions = {},
): boolean {
  return unsupportedSemanticWgslOperation(ir.operations, ir) === undefined &&
    ir.requiredFeatures.length === 0 &&
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
  if (ir.requiredFeatures.length > 0) throw semanticWgslError("semantic WGSL does not support required WebGPU features yet", ir.span);
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
    ...constantMemorySymbols(ir).filter((symbol) => !symbol.initialized && symbol.dimensions.length === 0),
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
  const constantBuffers = constantMemorySymbols(ir).filter((symbol) => !symbol.initialized && symbol.dimensions.length > 0);
  const deviceGlobalBuffers = deviceGlobalMemorySymbols(ir);
  const textures = textureSymbols(ir);
  const atomicStorage = semanticAtomicStorageNames(ir.operations);
  const atomicDeviceGlobals = semanticAtomicDeviceGlobalNames(ir.operations);
  const atomicShared = semanticAtomicSharedNames(ir.operations);
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
  for (const param of ir.params.filter((item) => item.addressSpace === "storage")) {
    const access = param.constant ? "read" : "read_write";
    const elementType = atomicStorage.has(param.name)
      ? `atomic<${wgslAtomicScalar(param.valueType)}>`
      : wgslScalar(param.valueType);
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, param.name)}) var<storage, ${access}> ${nameFor(param.name, names)}: array<${elementType}>;`);
  }
  for (const constant of constantBuffers) {
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, constant.name)}) var<storage, read> ${nameFor(constant.name, names)}: array<${wgslScalar(constant.valueType)}>;`);
  }
  for (const global of deviceGlobalBuffers) {
    const elementType = atomicDeviceGlobals.has(global.name)
      ? `atomic<${wgslAtomicScalar(global.valueType)}>`
      : wgslScalar(global.valueType);
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
  for (const constant of initializedScalarConstants) {
    lines.push(`const ${nameFor(constant.name, names)}: ${wgslScalar(constant.valueType)} = ${emitSemanticExpressionAs(constant.init ?? zeroExpression(constant.span), ir, names, wgslValueScalar(constant.valueType), options)};`);
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
  for (const shared of sharedMemorySymbols(ir)) {
    lines.push(`var<workgroup> ${nameFor(shared.name, names)}: ${emitSharedType(shared, atomicShared.has(shared.name))};`);
  }
  if (uniformParams.length > 0) {
    lines.push("struct Params {");
    for (const param of uniformParams) lines.push(`  ${nameFor(param.name, names)}: ${wgslUniformScalar(param.valueType)},`);
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
  const wgsl = lines.join("\n");
  return {
    wgsl,
    program: defineWgslKernelProgram({
      name: ir.name,
      wgsl,
      bindings,
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
        if (operation.target.dimensions.length > 0 && operation.init && !semanticWgslLocalArrayInitSupported(operation.init)) return operation;
        if (operation.target.dimensions.length === 0) {
          const vectorTarget = isSemanticWgslFloatVectorType(operation.target.valueType);
          if (operation.init && !semanticWgslExpressionSupported(operation.init, vectorTarget ? "any" : "scalar", ir)) return operation;
        }
        break;
      case "store":
        if (!semanticWgslAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticWgslMemoryRefSupported(operation.target) && !semanticWgslStorageOffsetStoreSupported(operation, ir)) return operation;
        if (
          operation.target.addressSpace === "storage" &&
          !ir.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage")
        ) return operation;
        if (!semanticWgslValueExpressionSupported(operation.value, ir)) return operation;
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
        if (operation.callee !== "__syncthreads") return operation;
        break;
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
    if (!semanticWgslScalarTypeSupported(symbol.valueType)) return false;
    return !symbol.initialized ||
      symbol.init !== undefined && (
        symbol.dimensions.length === 0
          ? semanticWgslExpressionSupported(symbol.init, "scalar")
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

function semanticWgslScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" || valueType === "int" || valueType === "uint";
}

function semanticWgslValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticWgslScalarTypeSupported(valueType) || isSemanticWgslFloatVectorType(valueType);
}

function semanticWgslAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
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
  if (ref.fields.length > 0) return false;
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return false;
  return ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar"));
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
  if (!WGSL_ATOMIC_CALLEES.has(operation.callee)) return false;
  if (!operation.target || (operation.target.addressSpace !== "storage" && operation.target.addressSpace !== "device-global" && operation.target.addressSpace !== "shared")) return false;
  if (!semanticWgslMemoryRefSupported(operation.target)) return false;
  if (operation.target.addressSpace === "storage" && operation.target.indices.length !== 1) return false;
  if (operation.target.fields.length > 0) return false;
  if (operation.target.valueType !== "uint" && operation.target.valueType !== "int") return false;
  if (!semanticWgslAtomicTargetRootSupported(operation.target, ir)) {
    return false;
  }
  const expectedArgs = operation.callee === "atomicCAS" || operation.callee === "atomicCAS_system" ? 3 : 2;
  return operation.args.length >= expectedArgs &&
    operation.args.slice(1, expectedArgs).every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslValueExpressionSupported(expression: SemanticExpression, ir: SemanticKernelIrModule): boolean {
  return semanticWgslExpressionSupported(expression, "scalar", ir) ||
    semanticWgslExpressionSupported(expression, "any", ir) && isSemanticWgslFloatVectorType(semanticExpressionVectorValueType(expression, ir)) ||
    expression.kind === "call" && (semanticWgslAtomicCallSupported(expression, ir) || semanticWgslMathCallSupported(expression) || semanticWgslVectorConstructorSupported(expression, "any", ir)) ||
    expression.kind === "texture-read" && semanticWgslTextureReadSupported(expression, ir) ||
    expression.kind === "surface-read" && semanticWgslSurfaceReadSupported(expression, ir);
}

function semanticWgslVectorMemberSupported(
  expression: Extract<SemanticExpression, { kind: "member" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  const valueType = semanticExpressionValueType(expression.object);
  return semanticWgslExpressionSupported(expression.object, "any", ir) &&
    isCudaVectorType(valueType) &&
    cudaVectorFieldIndex(valueType, expression.property) !== undefined;
}

function semanticWgslVectorIndexSupported(
  expression: Extract<SemanticExpression, { kind: "index" }>,
  ir?: SemanticKernelIrModule,
): boolean {
  return isSemanticWgslFloatVectorType(semanticExpressionVectorValueType(expression.target, ir)) &&
    semanticWgslExpressionSupported(expression.target, "any", ir) &&
    semanticWgslExpressionSupported(expression.index, "scalar", ir);
}

function semanticWgslLocalArrayInitSupported(expression: SemanticExpression): boolean {
  return expression.kind === "initializer" &&
    flattenInitializerExpressions(expression).every((item) => semanticWgslExpressionSupported(item, "scalar"));
}

function semanticWgslMathCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_MATH_CALLS.has(expression.callee.name)) return false;
  const arity = semanticMathCallArity(expression.callee.name);
  return expression.args.length === arity && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslTextureReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const texture = expression.texture;
  return (expression.valueType === "float" || isSemanticWgslFloatVectorType(expression.valueType)) &&
    texture.kind === "symbol" &&
    texture.addressSpace === "texture" &&
    semanticWgslExpressionSupported(expression.x, "scalar", ir) &&
    semanticWgslExpressionSupported(expression.y, "scalar", ir);
}

function semanticWgslSurfaceReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const target = expression.surface;
  return (expression.valueType === "float" ||
      expression.valueType === "uint" ||
      expression.valueType === "int" ||
      isSemanticWgslFloatVectorType(expression.valueType)) &&
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
  if (fn.params.some((param) => param.pointer || (param.addressSpace !== "local" && param.addressSpace !== "texture" && param.addressSpace !== "surface"))) return false;
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
  if (param.addressSpace === "texture") return arg.kind === "symbol" && arg.addressSpace === "texture";
  if (param.addressSpace === "surface") return arg.kind === "symbol" && arg.addressSpace === "surface";
  return semanticWgslExpressionSupported(arg, isSemanticWgslFloatVectorType(param.valueType) ? "any" : "scalar", ir);
}

function semanticWgslVectorConstructorSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any",
  ir?: SemanticKernelIrModule,
): boolean {
  if (expected === "scalar" || expression.callee.kind !== "symbol") return false;
  const valueType = cudaVectorConstructorType(expression.callee.name);
  return isSemanticWgslFloatVectorType(valueType) &&
    expression.args.length > 0 &&
    expression.args.every((arg) => semanticWgslExpressionSupported(arg, "any", ir));
}

function semanticWgslFunctionBodyShapeSupported(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "declare") return operation.target.addressSpace === "local" && !operation.target.pointer && operation.target.dimensions.length === 0;
    if (operation.kind === "store") return operation.target.addressSpace === "local";
    if (operation.kind === "surface-write") return true;
    if (operation.kind === "call") return true;
    if (operation.kind === "branch") return semanticWgslFunctionBodyShapeSupported(operation.consequent) && semanticWgslFunctionBodyShapeSupported(operation.alternate);
    if (operation.kind === "loop") return semanticWgslFunctionBodyShapeSupported(operation.body);
    return operation.kind === "expression" || operation.kind === "return" || operation.kind === "break" || operation.kind === "continue";
  });
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
  if (expression.callee.kind !== "symbol" || !WGSL_ATOMIC_CALLEES.has(expression.callee.name)) return false;
  const target = semanticAtomicCallTarget(expression);
  if (!target || (target.addressSpace !== "storage" && target.addressSpace !== "device-global" && target.addressSpace !== "shared")) return false;
  if (!semanticWgslMemoryRefSupported(target)) return false;
  if (target.addressSpace === "storage" && target.indices.length !== 1) return false;
  if (target.fields.length > 0) return false;
  if (target.valueType !== "uint" && target.valueType !== "int") return false;
  if (!semanticWgslAtomicTargetRootSupported(target, ir)) return false;
  const expectedArgs = expression.callee.name === "atomicCAS" || expression.callee.name === "atomicCAS_system" ? 3 : 2;
  return expression.args.length >= expectedArgs &&
    expression.args.slice(1, expectedArgs).every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (semanticWgslVoidFunctionCallSupported(operation, ir)) return true;
  if (!SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) return false;
  const [target, value] = operation.args;
  return target?.kind === "symbol" &&
    target.addressSpace === "local" &&
    value !== undefined &&
    semanticWgslExpressionSupported(value, "scalar", ir) &&
    localArraySymbol(ir, target.name) !== undefined;
}

function semanticWgslVoidFunctionCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  const fn = ir.functions.find((item) => item.name === operation.callee);
  if (!fn || fn.returnType !== "void") return false;
  if (fn.params.some((param) => param.pointer || (param.addressSpace !== "local" && param.addressSpace !== "texture" && param.addressSpace !== "surface"))) return false;
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
    semanticWgslExpressionSupported(operation.value, "any", ir) &&
    semanticWgslExpressionSupported(operation.xBytes, "scalar", ir) &&
    semanticWgslExpressionSupported(operation.y, "scalar", ir) &&
    (operation.z === undefined || semanticWgslExpressionSupported(operation.z, "scalar", ir));
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
        valueType: operation.valueType === "uint" || operation.valueType === "int" ? operation.valueType : "float",
        span: operation.span,
      },
      ir,
    );
}

function semanticWgslAtomicTargetRootSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (ref.addressSpace === "storage") {
    return ir.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && !param.constant);
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
      return expression.object.kind === "symbol" &&
        BUILTIN_VECTOR_NAMES.has(expression.object.name) &&
        (expression.property === "x" || expression.property === "y" || expression.property === "z") ||
        semanticWgslVectorMemberSupported(expression, ir);
    case "index":
      if (semanticWgslVectorIndexSupported(expression, ir)) return true;
      if (expected === "any" && isSemanticWgslFloatVectorType(expression.valueType)) {
        return semanticWgslMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
      }
      return expected === "scalar" && semanticWgslMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
    case "cast":
      return !expression.pointer && semanticWgslExpressionSupported(expression.expression, "scalar", ir);
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" && semanticWgslExpressionSupported(expression.argument, "scalar", ir);
    case "binary":
      return semanticWgslExpressionSupported(expression.left, "scalar", ir) &&
        semanticWgslExpressionSupported(expression.right, "scalar", ir);
    case "conditional":
      return semanticWgslExpressionSupported(expression.condition, "scalar", ir) &&
        semanticWgslExpressionSupported(expression.consequent, expected, ir) &&
        semanticWgslExpressionSupported(expression.alternate, expected, ir);
    case "assignment":
      return semanticWgslAssignmentOperatorSupported(expression.operator) &&
        expression.target.kind === "symbol" &&
        expression.target.addressSpace === "local" &&
        semanticWgslExpressionSupported(expression.value, "scalar", ir);
    case "update":
      return expression.argument.kind === "symbol" &&
        expression.argument.addressSpace === "local" &&
        (expression.operator === "++" || expression.operator === "--");
    case "sequence":
      return expression.expressions.every((item) => semanticWgslExpressionSupported(item, "scalar", ir));
    case "call":
      return ir !== undefined && semanticWgslFunctionCallSupported(expression, ir) ||
        semanticWgslMathCallSupported(expression) ||
        semanticWgslVectorConstructorSupported(expression, expected, ir);
    case "texture-read":
      return ir !== undefined &&
        (expected === "any" || expression.valueType === "float") &&
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
      const init = operation.init
        ? ` = ${emitSemanticInitExpression(operation.init, operation.target.valueType, ir, names, options, textureSpecializations)}`
        : isSemanticWgslFloatVectorType(operation.target.valueType)
        ? ` = ${zeroForType(wgslValueType(operation.target.valueType))}`
        : "";
      return [`${prefix}var ${nameFor(operation.target.name, names)}: ${type}${init};`];
    }
    case "store":
      return [`${prefix}${emitSemanticStore(operation, ir, names, options, textureSpecializations)};`];
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
    case "return":
      if (operation.value) {
        if (!allowReturnValue) throw semanticWgslError("semantic WGSL supports kernel return without value only", operation.span);
        return [`${prefix}return ${emitSemanticExpression(operation.value, ir, names, options, textureSpecializations)};`];
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
  const valueType = semanticExpressionValueType(operation.value);
  const value = isSemanticWgslFloatVectorType(valueType)
    ? emitSemanticExpression(operation.value, ir, names, options, textureSpecializations)
    : emitSemanticExpressionAs(operation.value, ir, names, "f32", options, textureSpecializations);
  const directSurface = surfaceSymbols(ir).find((surface) => surface.name === surfaceName);
  if (isSemanticWgslFloatVectorType(valueType)) {
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
  return Array.from({ length: cudaVectorLaneCount(valueType) }).flatMap((_, lane) => {
    const laneValue = `(${value}).${fields[lane]}`;
    const laneXBytes = `(${xBytes} + ${lane * 4})`;
    if (!directSurface) {
      return [`${prefix}${GENERIC_SURFACE_WRITE_HELPER_NAME}(${nameFor(surfaceName, names)}, ${laneValue}, ${laneXBytes}, ${y}, ${z});`];
    }
    return emitSemanticSurfaceWriteBody(directSurface, laneValue, laneXBytes, y, z, names, indentLevel);
  });
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
  const target = emitSemanticMemoryRef(operation.target, ir, names, options);
  if (
    semanticAtomicStorageNames(ir.operations).has(operation.target.base) ||
    semanticAtomicDeviceGlobalNames(ir.operations).has(operation.target.base) ||
    semanticAtomicSharedNames(ir.operations).has(operation.target.base)
  ) {
    if (operation.operator !== "=") {
      throw semanticWgslError(`semantic WGSL does not support atomic storage assignment '${operation.operator}'`, operation.span);
    }
    const atomicValue = emitSemanticExpressionAs(operation.value, ir, names, wgslAtomicScalar(operation.target.valueType), options, textureSpecializations);
    return `atomicStore(&${target}, ${atomicValue})`;
  }
  if (isSemanticWgslFloatVectorType(operation.target.valueType)) {
    if (operation.operator !== "=") throw semanticWgslError(`semantic WGSL does not support vector assignment '${operation.operator}'`, operation.span);
    return emitSemanticVectorMemoryWrite(operation, ir, names, options, textureSpecializations).join("; ");
  }
  const value = emitSemanticExpressionAs(operation.value, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations);
  if (operation.operator === "=") return `${target} = ${value}`;
  if (operation.operator === "+=") return `${target} = (${target} + ${value})`;
  if (operation.operator === "-=") return `${target} = (${target} - ${value})`;
  throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
}

function emitSemanticVectorMemoryWrite(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const valueType = operation.target.valueType;
  if (!isSemanticWgslFloatVectorType(valueType)) throw semanticWgslError("semantic WGSL vector write requires vector target", operation.span);
  const value = emitSemanticExpression(operation.value, ir, names, options, textureSpecializations);
  const base = emitFlatStorageVectorBaseIndex(operation.target, ir, names, options);
  const target = nameFor(operation.target.base, names);
  const fields = ["x", "y", "z", "w"];
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    `${target}[(${base} + ${lane}u)] = (${value}).${fields[lane]}`
  );
}

function emitSemanticFunction(
  fn: SemanticKernelIrModule["functions"][number],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  rawName = fn.name,
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
  const params = fn.params.map((param) => `${nameFor(param.name, names)}: ${emitSemanticFunctionParamType(param)}`).join(", ");
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

function emitSemanticAssignmentStatement(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar assignment targets only", expression.target.span);
  const target = nameFor(expression.target.name, names);
  const value = emitSemanticExpressionAs(expression.value, ir, names, wgslValueScalar(expression.target.valueType), options, textureSpecializations);
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
  if (!operation.init || operation.init.kind !== "initializer") return [];
  const prefix = "  ".repeat(indentLevel);
  return flattenInitializerExpressions(operation.init)
    .slice(0, totalElements(operation.target.dimensions))
    .map((value, index) => {
      const indices = flatIndicesForDimensions(operation.target.dimensions, index)
        .map((item) => `[${item}u]`)
        .join("");
      return `${prefix}${nameFor(operation.target.name, names)}${indices} = ${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(operation.target.valueType), options, textureSpecializations)};`;
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
  const wgslCallee = WGSL_ATOMIC_CALLEES.get(operation.callee);
  if (!operation.target || !wgslCallee) {
    throw semanticWgslError(`semantic WGSL does not support atomic '${operation.callee}'`, operation.span);
  }
  const target = emitSemanticMemoryRef(operation.target, ir, names, options);
  const operands = operation.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  if (operands.length === 0 || operands.some((operand) => operand === undefined)) {
    throw semanticWgslError(`semantic WGSL atomic '${operation.callee}' missing operand`, operation.span);
  }
  const emitted = operands.map((operand) =>
    emitSemanticExpressionAs(operand!, ir, names, wgslAtomicScalar(operation.target!.valueType), options, textureSpecializations)
  );
  return `_ = ${wgslCallee}(&${target}, ${emitted.join(", ")})`;
}

function emitSemanticCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): readonly string[] {
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
  return `${nameFor(callee, names)}(${operation.args.map((arg, index) => emitSemanticFunctionArg(arg, fn.params[index], ir, names, options, textureSpecializations)).join(", ")})`;
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
    throw semanticWgslError(`${operation.callee} expects local array and scalar value`, operation.span);
  }
  const symbol = localArraySymbol(ir, target.name);
  if (!symbol) throw semanticWgslError(`${operation.callee} expects fixed local array '${target.name}'`, target.span);
  return emitLocalArrayFill(
    nameFor(target.name, names),
    symbol.dimensions,
    emitSemanticExpressionAs(valueExpression, ir, names, wgslValueScalar(symbol.valueType), options, textureSpecializations),
    indentLevel,
  );
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
    const value = init.init ? emitSemanticExpressionAs(init.init, ir, names, wgslValueScalar(init.target.valueType), options, textureSpecializations) : zeroForType(type);
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
      if (expression.addressSpace === "constant") return `${UNIFORM_PARAMS_NAME}.${nameFor(expression.name, names)}`;
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
        if (isSemanticWgslFloatVectorType(ref.valueType)) return emitSemanticVectorMemoryRead(ref, ir, names, options);
        const memoryRef = emitSemanticMemoryRef(ref, ir, names, options);
        if (
          semanticAtomicStorageNames(ir.operations).has(ref.base) ||
          semanticAtomicDeviceGlobalNames(ir.operations).has(ref.base) ||
          semanticAtomicSharedNames(ir.operations).has(ref.base)
        ) return `atomicLoad(&${memoryRef})`;
        return memoryRef;
      }
      throw semanticWgslError("semantic WGSL does not support index target", expression.span);
    }
    case "cast":
      return `${wgslScalar(expression.valueType)}(${emitSemanticExpression(expression.expression, ir, names, options, textureSpecializations)})`;
    case "unary":
      return emitSemanticUnary(expression, ir, names, options, textureSpecializations);
    case "binary":
      return emitSemanticBinary(expression, ir, names, options, textureSpecializations);
    case "conditional":
      return `select(${emitSemanticExpression(expression.alternate, ir, names, options, textureSpecializations)}, ${emitSemanticExpression(expression.consequent, ir, names, options, textureSpecializations)}, ${emitTruthiness(expression.condition, ir, names, options)})`;
    case "assignment":
      if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar assignment targets only", expression.target.span);
      {
        const target = nameFor(expression.target.name, names);
        const value = emitSemanticExpressionAs(expression.value, ir, names, wgslValueScalar(expression.valueType), options, textureSpecializations);
        if (expression.operator === "+=") return `(${target} += ${value})`;
        if (expression.operator === "-=") return `(${target} -= ${value})`;
        return `(${target} = ${value})`;
      }
    case "update":
      return emitSemanticUpdate(expression, names);
    case "sequence":
      return emitSemanticExpression(expression.expressions.at(-1) ?? zeroExpression(expression.span), ir, names, options, textureSpecializations);
    case "call":
      if (semanticWgslAtomicCallSupported(expression, ir)) return emitSemanticAtomicCall(expression, ir, names, options, textureSpecializations);
      if (semanticWgslVectorConstructorSupported(expression, "any", ir)) return emitSemanticVectorConstructor(expression, ir, names, options, textureSpecializations);
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
  if (isSemanticWgslFloatVectorType(expression.valueType)) {
    const lanes = Array.from({ length: cudaVectorLaneCount(expression.valueType) }, (_, lane) => `f32(${readAt(`(${xBytes} + ${lane * 4})`)})`);
    return `vec${lanes.length}<f32>(${lanes.join(", ")})`;
  }
  const read = readAt(xBytes);
  if (expression.valueType === "uint") return `u32(${read})`;
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
  if (expression.valueType === "float2") return `${read}.xy`;
  if (expression.valueType === "float3") return `${read}.xyz`;
  if (expression.valueType === "float4") return read;
  return `${read}.r`;
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
  const args = expression.args.map((arg, index) => emitSemanticFunctionArg(arg, fn.params[index], ir, names, options, textureSpecializations));
  const calleeName = semanticFunctionCallName(callee, fn, expression.args, options, textureSpecializations);
  return `${nameFor(calleeName, names)}(${args.join(", ")})`;
}

function emitSemanticVectorConstructor(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  const valueType = expression.callee.kind === "symbol" ? cudaVectorConstructorType(expression.callee.name) : undefined;
  if (!isSemanticWgslFloatVectorType(valueType)) throw semanticWgslError("semantic WGSL vector constructor requires float vector target", expression.span);
  const fields = ["x", "y", "z", "w"];
  const targetLanes = cudaVectorLaneCount(valueType);
  if (expression.args.length === 1 && !isSemanticWgslFloatVectorType(semanticExpressionVectorValueType(expression.args[0]!, ir))) {
    const scalar = emitSemanticExpressionAs(expression.args[0]!, ir, names, "f32", options, textureSpecializations);
    return `vec${targetLanes}<f32>(${Array.from({ length: targetLanes }, () => `f32(${scalar})`).join(", ")})`;
  }
  const lanes = expression.args.flatMap((arg) => {
    const argType = semanticExpressionVectorValueType(arg, ir);
    if (isSemanticWgslFloatVectorType(argType)) {
      const value = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
      return Array.from({ length: cudaVectorLaneCount(argType) }, (_, lane) => `f32((${value}).${fields[lane]})`);
    }
    return [`f32(${emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations)})`];
  });
  while (lanes.length < targetLanes) lanes.push("0.0");
  return `vec${targetLanes}<f32>(${lanes.slice(0, targetLanes).join(", ")})`;
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
  if (isSemanticWgslFloatVectorType(param?.valueType)) return emitSemanticExpression(arg, ir, names, options, textureSpecializations);
  return emitSemanticExpressionAs(arg, ir, names, wgslValueScalar(param?.valueType), options, textureSpecializations);
}

function emitSemanticUpdate(
  expression: Extract<SemanticExpression, { readonly kind: "update" }>,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.argument.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar updates only", expression.span);
  const name = nameFor(expression.argument.name, names);
  if (expression.operator === "++") return `${name} += ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))}`;
  if (expression.operator === "--") return `${name} -= ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))}`;
  throw semanticWgslError(`semantic WGSL does not support update '${expression.operator}'`, expression.span);
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
    if (targetType === "f32") return emitted;
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
  if (isSemanticWgslFloatVectorType(valueType)) return emitSemanticExpression(expression, ir, names, options, textureSpecializations);
  return emitSemanticExpressionAs(expression, ir, names, wgslValueScalar(valueType), options, textureSpecializations);
}

function emitSemanticAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
  textureSpecializations: SemanticTextureDescriptorSpecializations = new Map(),
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL atomic call requires symbol callee", expression.span);
  const wgslCallee = WGSL_ATOMIC_CALLEES.get(expression.callee.name);
  const target = semanticAtomicCallTarget(expression);
  if (!wgslCallee || !target) throw semanticWgslError(`semantic WGSL does not support atomic '${expression.callee.name}'`, expression.span);
  const memoryRef = emitSemanticMemoryRef(target, ir, names, options);
  const operands = expression.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  const emitted = operands.map((operand) => emitSemanticExpressionAs(operand, ir, names, wgslAtomicScalar(target.valueType), options, textureSpecializations));
  const call = `${wgslCallee}(&${memoryRef}, ${emitted.join(", ")})`;
  return wgslCallee === "atomicCompareExchangeWeak" ? `${call}.old_value` : call;
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
  if (wgslCallee === "div_ceil") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const scalar = semanticExpressionWgslScalar(left) === "u32" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, scalar, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, scalar, options, textureSpecializations);
    return `(((${lhs} + ${rhs}) - ${scalar === "u32" ? "1u" : "1"}) / ${rhs})`;
  }
  if (wgslCallee === "divide") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return `(${emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations)} / ${emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations)})`;
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
  return `${wgslCallee}(${expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations)).join(", ")})`;
}

function semanticMathCallArity(name: string): number {
  return name === "fmin" ||
    name === "fminf" ||
    name === "min" ||
    name === "fmax" ||
    name === "fmaxf" ||
    name === "max" ||
    name === "pow" ||
    name === "powf" ||
    name === "__fdividef" ||
    name === "div_ceil" ||
    name === "ceil_div" ||
    name === "__bg_remquo_quotient" ||
    name === "__bg_remquo_remainder" ||
    name === "atan2" ||
    name === "atan2f"
    ? 2
    : name === "fma" ||
      name === "fmaf" ||
      name === "__fmaf_rn" ||
      name === "lerp"
    ? 3
    : 1;
}

function emitSemanticMember(
  expression: Extract<SemanticExpression, { readonly kind: "member" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (expression.object.kind !== "symbol") throw semanticWgslError("semantic WGSL supports builtin vector members only", expression.span);
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  switch (expression.object.name) {
    case "threadIdx":
      return ir.workgroupSize[axisIndex] === 1 ? "0u" : `local_id.${expression.property}`;
    case "blockIdx":
      return `workgroup_id.${expression.property}`;
    case "blockDim":
      return `${ir.workgroupSize[axisIndex]}u`;
    case "gridDim":
      return `num_workgroups.${expression.property}`;
    default:
      return `${emitSemanticExpression(expression.object, ir, names, options)}.${expression.property}`;
  }
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
  const operandType = semanticBinaryOperandType(expression);
  const left = emitSemanticExpressionAs(expression.left, ir, names, operandType, options, textureSpecializations);
  const right = emitSemanticExpressionAs(expression.right, ir, names, operandType, options, textureSpecializations);
  return `(${left} ${expression.operator} ${right})`;
}

function semanticBinaryOperandType(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): WgslValueType {
  const left = semanticExpressionWgslScalar(expression.left);
  const right = semanticExpressionWgslScalar(expression.right);
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
  if (expression.kind === "binary" && (COMPARISON_OPERATORS.has(expression.operator) || LOGICAL_OPERATORS.has(expression.operator))) {
    return emitSemanticBinary(expression, ir, names, options);
  }
  return `(${emitSemanticExpression(expression, ir, names, options)} != 0)`;
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

function emitSemanticVectorMemoryRead(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  if (!isSemanticWgslFloatVectorType(ref.valueType)) throw semanticWgslError("semantic WGSL vector read requires vector memory type", ref.span);
  const base = emitFlatStorageVectorBaseIndex(ref, ir, names, options);
  const storage = nameFor(ref.base, names);
  const laneCount = cudaVectorLaneCount(ref.valueType);
  return `vec${laneCount}<f32>(${Array.from({ length: laneCount }, (_, lane) => `${storage}[(${base} + ${lane}u)]`).join(", ")})`;
}

function memoryRefFromIndexExpression(expression: Extract<SemanticExpression, { readonly kind: "index" }>): SemanticMemoryRef | undefined {
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
    wgslScalar(symbol.valueType),
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

function totalElements(dimensions: readonly number[]): number {
  return dimensions.length === 0 ? 1 : dimensions.reduce((product, dimension) => product * dimension, 1);
}

function storageOffsetSymbol(base: string): string {
  return `${base}__bg_ptr_offset`;
}

function flattenInitializerExpressions(expression: SemanticExpression): readonly SemanticExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

function flatIndicesForDimensions(dimensions: readonly number[], flatIndex: number): readonly number[] {
  return dimensions.map((_, offset) => {
    const stride = dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
    return Math.floor(flatIndex / stride) % Math.max(1, dimensions[offset] ?? 1);
  });
}

function emitFlatStorageIndex(
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
  if (hasOffset) {
    terms.unshift(nameFor(storageOffsetSymbol(ref.base), names));
  }
  const expression = terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
  return `u32(${expression})`;
}

function emitFlatStorageVectorBaseIndex(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: EmitSemanticKernelIrWgslOptions = {},
): string {
  const base = emitFlatStorageIndex({ ...ref, valueType: "float" }, ir, names, options);
  const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
  const valueType = root?.valueType;
  const stride = isSemanticWgslFloatVectorType(valueType) ? cudaVectorLaneCount(valueType) : 1;
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
    const stride = symbol.dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
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
    const stride = symbol.dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
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
    const stride = symbol.dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
    const emitted = emitSemanticExpressionAs(index, ir, names, "u32");
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function emitFlatLocalArrayIndexes(flat: string, dimensions: readonly number[]): string {
  return dimensions.map((dimension, offset) => {
    const stride = dimensions.slice(offset + 1).reduce((product, item) => product * item, 1);
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
  if (valueType === "int") return "i32";
  if (valueType === "uint") return "u32";
  if (valueType === "bool") return "bool";
  return "f32";
}

function wgslValueType(valueType: CudaLiteScalarType | undefined): SemanticWgslValueType {
  if (valueType === "float2") return "vec2<f32>";
  if (valueType === "float3") return "vec3<f32>";
  if (valueType === "float4") return "vec4<f32>";
  return wgslScalar(valueType);
}

function wgslValueScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  return scalar === "bool" ? "u32" : scalar;
}

function wgslAtomicScalar(valueType: CudaLiteScalarType | undefined): Extract<WgslValueType, "i32" | "u32"> {
  return valueType === "int" ? "i32" : "u32";
}

function wgslUniformScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "int") return "i32";
  if (valueType === "uint" || valueType === "bool") return "u32";
  return "f32";
}

function semanticExpressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

function semanticExpressionVectorValueType(
  expression: SemanticExpression,
  ir?: SemanticKernelIrModule,
): CudaLiteScalarType | undefined {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const calleeName = expression.callee.name;
    return cudaVectorConstructorType(calleeName) ?? ir?.functions.find((fn) => fn.name === calleeName)?.returnType ?? semanticExpressionValueType(expression);
  }
  return semanticExpressionValueType(expression);
}

function semanticExpressionWgslScalar(expression: SemanticExpression): WgslValueType {
  switch (expression.kind) {
    case "call": {
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

function isSemanticWgslFloatVectorType(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float2" || valueType === "float3" || valueType === "float4";
}

function emitNumberLiteral(value: number, valueType: CudaLiteScalarType | undefined, expectedType?: WgslValueType): string {
  const type = expectedType ?? wgslScalar(valueType);
  if (type === "u32") return `${Math.trunc(value) >>> 0}u`;
  if (type === "i32" && value > 2147483647) return `bitcast<i32>(${Math.trunc(value) >>> 0}u)`;
  if (type === "i32") return String(Math.trunc(value));
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
  if (valueType === "vec2<f32>") return "vec2<f32>(0.0)";
  if (valueType === "vec3<f32>") return "vec3<f32>(0.0)";
  if (valueType === "vec4<f32>") return "vec4<f32>(0.0)";
  return "0.0";
}

function bindingIndexFor(bindings: readonly WgslKernelBindingInput[], name: string): number {
  const binding = bindings.find((item) => item.name === name)?.binding;
  return binding ?? 0;
}

function semanticAtomicStorageNames(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "storage") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicStorageNamesFromOperation(operation)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicStorageNames(operation.consequent)) names.add(name);
      for (const name of semanticAtomicStorageNames(operation.alternate)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicStorageNames([operation.init])) names.add(name);
      }
      for (const name of semanticAtomicStorageNames(operation.body)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicStorageNames(operation.body)) names.add(name);
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

function semanticAtomicStorageNamesFromOperation(operation: SemanticKernelIrOperation): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  for (const expression of expressions) {
    for (const name of semanticAtomicStorageNamesFromExpression(expression)) names.add(name);
  }
  return names;
}

function semanticAtomicStorageNamesFromExpression(expression: SemanticExpression): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === "storage") names.add(target.base);
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicStorageNamesFromExpression(child)) names.add(name);
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
  const firstArg = expression.args[0];
  if (!firstArg) return undefined;
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
