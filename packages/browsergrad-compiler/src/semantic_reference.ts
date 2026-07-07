import type { WgslTexture2DInput, WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { validateCudaKernelLaunch } from "./launch.js";
import { deviceGlobalBufferInputs } from "./webgpu_inputs.js";
import type {
  CompiledCudaLiteKernel,
  CompiledKernelInput,
  CudaLiteDiagnostic,
  CudaLiteScalarType,
  CudaLiteTextureDescriptor,
  KernelLaunch,
  KernelMemoryAccess,
  KernelThreadTrace,
  ReferenceKernelResult,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { cudaVectorConstructorType, cudaVectorFieldIndex, cudaVectorLaneCount, isCudaVectorType } from "./vector_types.js";

type SemanticValue = number | Vector3 | number[];
type SemanticAtomicOp = "add" | "sub" | "min" | "max" | "and" | "or" | "xor" | "exchange" | "cas";
type SemanticControl = "fallthrough" | "return" | "break" | "continue";

const SEMANTIC_ATOMIC_OPS = new Map<string, SemanticAtomicOp>([
  ["atomicAdd", "add"],
  ["atomicAdd_system", "add"],
  ["atomicSub", "sub"],
  ["atomicSub_system", "sub"],
  ["atomicMin", "min"],
  ["atomicMin_system", "min"],
  ["atomicMax", "max"],
  ["atomicMax_system", "max"],
  ["atomicAnd", "and"],
  ["atomicAnd_system", "and"],
  ["atomicOr", "or"],
  ["atomicOr_system", "or"],
  ["atomicXor", "xor"],
  ["atomicXor_system", "xor"],
  ["atomicExch", "exchange"],
  ["atomicExch_system", "exchange"],
  ["atomicCAS", "cas"],
  ["atomicCAS_system", "cas"],
]);
const SEMANTIC_MATH_CALLS = new Set([
  "sqrt", "sqrtf", "__fsqrt_rn", "rsqrt", "rsqrtf", "__frsqrt_rn",
  "exp", "expf", "__expf", "log", "logf", "__logf", "fabs", "fabsf", "abs",
  "floor", "floorf", "ceil", "ceilf", "trunc", "truncf", "sin", "sinf", "__sinf", "cos", "cosf", "__cosf",
  "tan", "tanf", "__tanf", "atan", "atanf", "atan2", "atan2f", "tanh", "tanhf", "__tanhf",
  "fmin", "fminf", "min", "fmax", "fmaxf", "max", "pow", "powf",
  "__powf", "__fdividef", "fdividef", "__fadd_rn", "__fsub_rn", "__fmul_rn", "__fdiv_rn",
  "__saturatef", "copysign", "copysignf", "fma", "fmaf", "__fmaf_rn", "lerp", "div_ceil", "ceil_div",
  "__bg_modf_intpart", "__bg_modf_fraction",
  "__bg_frexp_exponent", "__bg_frexp_mantissa",
  "__bg_remquo_quotient", "__bg_remquo_remainder",
]);
const SEMANTIC_LOCAL_ARRAY_FILL_CALLS = new Set(["fill_1D_regs", "fill_2D_regs", "fill_3D_regs"]);

interface Vector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface SemanticReferenceContext {
  readonly compiled: CompiledCudaLiteKernel;
  readonly buffers: Map<string, WgslTypedArray>;
  readonly constants: Map<string, number | WgslTypedArray>;
  readonly deviceGlobals: Map<string, WgslTypedArray>;
  readonly textures: Readonly<Record<string, WgslTexture2DInput>>;
  readonly textureDescriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>;
  readonly surfaces: Readonly<Record<string, WgslTexture2DInput>>;
  readonly sharedMemory: Map<string, WgslTypedArray>;
  readonly storageOffsets: Map<string, number>;
  readonly scalars: Readonly<Record<string, number>>;
  readonly locals: Map<string, SemanticValue>;
  readonly blockIdx: Vector3;
  readonly threadIdx: Vector3;
  readonly blockDim: Vector3;
  readonly gridDim: Vector3;
  readonly trace: MutableTrace;
  returnValue?: SemanticValue;
}

interface MutableTrace {
  blockIdx: [number, number, number];
  threadIdx: [number, number, number];
  reads: KernelMemoryAccess[];
  writes: KernelMemoryAccess[];
  sharedReads: KernelMemoryAccess[];
  sharedWrites: KernelMemoryAccess[];
}

export function canRunCompiledKernelSemanticReference(compiled: CompiledCudaLiteKernel): boolean {
  return compiled.kernelIr.params.every(semanticReferenceParamSupported) &&
    compiled.kernelIr.memory.every(semanticReferenceMemorySymbolSupported) &&
    semanticReferenceTextureDescriptorsSupported(compiled) &&
    semanticReferenceSharedShapeSupported(compiled) &&
    unsupportedSemanticReferenceOperation(compiled.kernelIr.operations, compiled) === undefined &&
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

  const buffers = cloneBuffers(input.buffers);
  const constants = semanticReferenceConstants(compiled, input);
  const deviceGlobals = cloneBuffers(deviceGlobalBufferInputs(compiled, input));
  const surfaces = cloneSurfaces(input.surfaces ?? {});
  const traces: MutableTrace[] = [];
  const blockDim = vectorFromTuple(launch.blockDim);
  const gridDim = vectorFromTuple(launch.gridDim);
  const scalars = input.scalars ?? {};
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
                storageOffsets: new Map(),
                scalars,
                locals: new Map(),
                blockIdx: { x: bx, y: by, z: bz },
                threadIdx: { x: tx, y: ty, z: tz },
                blockDim,
                gridDim,
                trace,
              });
            }
          }
        }
        if (sharedMemory.size > 0) runSemanticBlockPhases(compiled.kernelIr.operations, blockContexts);
        else for (const context of blockContexts) execSemanticOperations(compiled.kernelIr.operations, context);
      }
    }
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
    trace: traces.map(freezeTrace),
  };
}

function unsupportedSemanticReferenceOperation(
  operations: readonly SemanticKernelIrOperation[],
  compiled: CompiledCudaLiteKernel,
  allowReturnValue = false,
): SemanticKernelIrOperation | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        if (operation.target.addressSpace === "shared") {
          if (operation.target.pointer || !semanticReferenceScalarTypeSupported(operation.target.valueType)) return operation;
          break;
        }
        if (operation.target.addressSpace !== "local" || operation.target.pointer) return operation;
        if (!semanticReferenceValueTypeSupported(operation.target.valueType)) return operation;
        if (operation.target.dimensions.length > 0 && operation.init && !semanticReferenceLocalArrayInitSupported(operation.init)) return operation;
        if (operation.target.dimensions.length === 0) {
          const vectorTarget = isSemanticReferenceFloatVectorType(operation.target.valueType);
          if (operation.init && !semanticReferenceExpressionSupported(operation.init, vectorTarget ? "any" : "scalar", compiled)) return operation;
        }
        break;
      case "store":
        if (!semanticReferenceAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticReferenceMemoryRefSupported(operation.target) && !semanticReferenceStorageOffsetStoreSupported(operation, compiled)) return operation;
        if (
          operation.target.addressSpace === "storage" &&
          !compiled.kernelIr.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage")
        ) return operation;
        if (!semanticReferenceValueExpressionSupported(operation.value, compiled)) return operation;
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
        if (!semanticReferenceExpressionSupported(operation.condition, "scalar", compiled)) return operation;
        break;
      case "block":
        if (operationsContainDeclare(operation.body)) return operation;
        break;
      case "loop":
        if (operation.init && !semanticReferenceLoopInitSupported(operation.init, compiled)) return operation;
        if (operation.condition && !semanticReferenceExpressionSupported(operation.condition, "scalar", compiled)) return operation;
        if (operation.update && !semanticReferenceExpressionSupported(operation.update, "scalar", compiled)) return operation;
        break;
      case "return":
        if (operation.value && (!allowReturnValue || !semanticReferenceExpressionSupported(operation.value, "any", compiled))) return operation;
        break;
      case "barrier":
        if (operation.callee !== "__syncthreads") return operation;
        break;
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

function semanticReferenceParamSupported(param: CompiledCudaLiteKernel["kernelIr"]["params"][number]): boolean {
  if (param.addressSpace === "storage") return Boolean(param.pointer) && semanticReferenceValueTypeSupported(param.valueType);
  if (param.addressSpace === "uniform") return semanticReferenceScalarTypeSupported(param.valueType);
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

function semanticReferenceMemorySymbolSupported(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): boolean {
  if (symbol.kind === "local" || symbol.kind === "shared") return true;
  if (symbol.kind === "constant") {
    if (!semanticReferenceValueTypeSupported(symbol.valueType)) return false;
    return !symbol.initialized ||
      symbol.init !== undefined && (
        symbol.dimensions.length === 0
          ? semanticReferenceExpressionSupported(symbol.init, isSemanticReferenceFloatVectorType(symbol.valueType) ? "any" : "scalar")
          : initializedConstantArraySupported(symbol)
      );
  }
  if (symbol.kind === "device-global") return semanticReferenceScalarTypeSupported(symbol.valueType);
  if (symbol.kind === "texture") return symbol.valueType === "texture2d";
  return false;
}

function semanticReferenceSharedShapeSupported(compiled: CompiledCudaLiteKernel): boolean {
  if (!compiled.kernelIr.memory.some((symbol) => symbol.kind === "shared")) return true;
  return operationsHaveOnlyTopLevelSharedBarriers(compiled.kernelIr.operations);
}

function operationsHaveOnlyTopLevelSharedBarriers(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "branch") {
      return operationsHaveNoBarrierOrControlTransfer(operation.consequent) &&
        operationsHaveNoBarrierOrControlTransfer(operation.alternate);
    }
    if (operation.kind === "block") return operationsHaveNoBarrierOrControlTransfer(operation.body);
    return operation.kind !== "loop";
  });
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

function semanticReferenceLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? unsupportedSemanticReferenceOperation([init], compiled) === undefined
    : semanticReferenceExpressionSupported(init, "scalar", compiled);
}

function semanticReferenceScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" || valueType === "int" || valueType === "uint";
}

function semanticReferenceValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticReferenceScalarTypeSupported(valueType) || isSemanticReferenceFloatVectorType(valueType);
}

function semanticReferenceMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local" && ref.addressSpace !== "shared") {
    return false;
  }
  if (ref.fields.length > 0) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length === 0) return false;
  return ref.indices.every((index) => semanticReferenceExpressionSupported(index, "scalar"));
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
  const atomicOp = SEMANTIC_ATOMIC_OPS.get(operation.callee);
  if (!atomicOp) return false;
  if (!operation.target || !semanticReferenceMemoryRefSupported(operation.target)) return false;
  if (operation.target.valueType !== "uint" && operation.target.valueType !== "int") return false;
  if (!semanticReferenceAtomicTargetRootSupported(operation.target, compiled)) {
    return false;
  }
  const expectedArgs = atomicOp === "cas" ? 3 : 2;
  return operation.args.length >= expectedArgs &&
    operation.args.slice(1, expectedArgs).every((arg) => semanticReferenceExpressionSupported(arg, "scalar"));
}

function semanticReferenceValueExpressionSupported(expression: SemanticExpression, compiled: CompiledCudaLiteKernel): boolean {
  return semanticReferenceExpressionSupported(expression, "scalar", compiled) ||
    semanticReferenceExpressionSupported(expression, "any", compiled) && isSemanticReferenceFloatVectorType(semanticExpressionValueType(expression)) ||
    expression.kind === "call" && (semanticReferenceAtomicCallSupported(expression, compiled) || semanticReferenceMathCallSupported(expression) || semanticReferenceVectorConstructorSupported(expression, "any", compiled) || semanticReferenceVectorAtCallSupported(expression, compiled)) ||
    expression.kind === "texture-read" && semanticReferenceTextureReadSupported(expression, compiled) ||
    expression.kind === "surface-read" && semanticReferenceSurfaceReadSupported(expression, compiled);
}

function semanticReferenceLocalArrayInitSupported(expression: SemanticExpression): boolean {
  return expression.kind === "initializer" &&
    flattenInitializerExpressions(expression).every((item) => semanticReferenceExpressionSupported(item, "scalar"));
}

function semanticReferenceMathCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_MATH_CALLS.has(expression.callee.name)) return false;
  const arity = semanticMathCallArity(expression.callee.name);
  return expression.args.length === arity && expression.args.every((arg) => semanticReferenceExpressionSupported(arg, "scalar"));
}

function semanticReferenceTextureReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return (expression.valueType === "float" || isSemanticReferenceFloatVectorType(expression.valueType)) &&
    expression.texture.kind === "symbol" &&
    expression.texture.addressSpace === "texture" &&
    semanticReferenceExpressionSupported(expression.x, "scalar", compiled) &&
    semanticReferenceExpressionSupported(expression.y, "scalar", compiled);
}

function semanticReferenceTextureDescriptorsSupported(_compiled: CompiledCudaLiteKernel): boolean {
  return true;
}

function semanticReferenceSurfaceReadSupported(
  expression: Extract<SemanticExpression, { readonly kind: "surface-read" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const surface = expression.surface;
  return (expression.valueType === "float" ||
      expression.valueType === "uint" ||
      expression.valueType === "int" ||
      isSemanticReferenceFloatVectorType(expression.valueType)) &&
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
  if (!fn || !semanticReferenceValueTypeSupported(fn.returnType)) return false;
  if (fn.params.some((param) => param.pointer || (param.addressSpace !== "local" && param.addressSpace !== "texture" && param.addressSpace !== "surface"))) return false;
  if (fn.params.some((param) => param.addressSpace === "local" && !semanticReferenceValueTypeSupported(param.valueType))) return false;
  if (!semanticReferenceFunctionBodyShapeSupported(fn.body)) return false;
  return expression.args.length === fn.params.length &&
    expression.args.every((arg, index) => semanticReferenceFunctionArgSupported(arg, fn.params[index], compiled)) &&
    unsupportedSemanticReferenceOperation(fn.body, compiled, true) === undefined;
}

function semanticReferenceFunctionArgSupported(
  arg: SemanticExpression,
  param: CompiledCudaLiteKernel["kernelIr"]["functions"][number]["params"][number] | undefined,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (!param) return false;
  if (param.addressSpace === "texture") return arg.kind === "symbol" && arg.addressSpace === "texture";
  if (param.addressSpace === "surface") return arg.kind === "symbol" && arg.addressSpace === "surface";
  return semanticReferenceExpressionSupported(arg, isSemanticReferenceFloatVectorType(param.valueType) ? "any" : "scalar", compiled);
}

function semanticReferenceVectorConstructorSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  expected: "scalar" | "any",
  compiled?: CompiledCudaLiteKernel,
): boolean {
  if (expected === "scalar" || expression.callee.kind !== "symbol") return false;
  const valueType = cudaVectorConstructorType(expression.callee.name);
  return isSemanticReferenceFloatVectorType(valueType) &&
    expression.args.length > 0 &&
    expression.args.every((arg) => semanticReferenceExpressionSupported(arg, "any", compiled));
}

function semanticReferenceVectorIndexSupported(
  expression: Extract<SemanticExpression, { readonly kind: "index" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  const ref = memoryRefFromIndexExpression(expression);
  if (ref && !(ref.addressSpace === "local" && expression.target.kind === "symbol" && isSemanticReferenceFloatVectorType(expression.target.valueType))) return false;
  return isSemanticReferenceFloatVectorType(semanticExpressionValueType(expression.target)) &&
    semanticReferenceExpressionSupported(expression.target, "any", compiled) &&
    semanticReferenceExpressionSupported(expression.index, "scalar", compiled);
}

function semanticReferenceVectorAtCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled?: CompiledCudaLiteKernel,
): boolean {
  return expression.callee.kind === "symbol" &&
    expression.callee.name === "vec_at" &&
    expression.args.length === 2 &&
    expression.args[0] !== undefined &&
    expression.args[1] !== undefined &&
    isSemanticReferenceFloatVectorType(semanticExpressionValueType(expression.args[0])) &&
    semanticReferenceExpressionSupported(expression.args[0], "any", compiled) &&
    semanticReferenceExpressionSupported(expression.args[1], "scalar", compiled);
}

function semanticReferenceFunctionBodyShapeSupported(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "declare") return operation.target.addressSpace === "local" && !operation.target.pointer && operation.target.dimensions.length === 0;
    if (operation.kind === "store") return operation.target.addressSpace === "local";
    if (operation.kind === "surface-write") return true;
    if (operation.kind === "call") return true;
    if (operation.kind === "branch") return semanticReferenceFunctionBodyShapeSupported(operation.consequent) && semanticReferenceFunctionBodyShapeSupported(operation.alternate);
    if (operation.kind === "block") return semanticReferenceFunctionBodyShapeSupported(operation.body);
    if (operation.kind === "loop") return semanticReferenceFunctionBodyShapeSupported(operation.body);
    return operation.kind === "expression" || operation.kind === "return" || operation.kind === "break" || operation.kind === "continue";
  });
}

function semanticReferenceAtomicCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const atomicOp = SEMANTIC_ATOMIC_OPS.get(expression.callee.name);
  if (!atomicOp) return false;
  const target = semanticAtomicCallTarget(expression);
  if (!target || !semanticReferenceMemoryRefSupported(target)) return false;
  if (target.valueType !== "uint" && target.valueType !== "int") return false;
  if (!semanticReferenceAtomicTargetRootSupported(target, compiled)) {
    return false;
  }
  const expectedArgs = atomicOp === "cas" ? 3 : 2;
  return expression.args.length >= expectedArgs &&
    expression.args.slice(1, expectedArgs).every((arg) => semanticReferenceExpressionSupported(arg, "scalar"));
}

function semanticReferenceCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (semanticReferenceVoidFunctionCallSupported(operation, compiled)) return true;
  if (!SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) return false;
  const [target, value] = operation.args;
  return target?.kind === "symbol" &&
    target.addressSpace === "local" &&
    value !== undefined &&
    semanticReferenceExpressionSupported(value, "scalar", compiled) &&
    compiled.kernelIr.memory.some((symbol) =>
      symbol.kind === "local" &&
      symbol.name === target.name &&
      symbol.dimensions.length > 0
    );
}

function semanticReferenceVoidFunctionCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const fn = compiled.kernelIr.functions.find((item) => item.name === operation.callee);
  if (!fn || fn.returnType !== "void") return false;
  if (fn.params.some((param) => param.pointer || (param.addressSpace !== "local" && param.addressSpace !== "texture" && param.addressSpace !== "surface"))) return false;
  return operation.args.length === fn.params.length &&
    operation.args.every((arg, index) => semanticReferenceFunctionArgSupported(arg, fn.params[index], compiled)) &&
    semanticReferenceFunctionBodyShapeSupported(fn.body) &&
    unsupportedSemanticReferenceOperation(fn.body, compiled, true) === undefined;
}

function semanticReferenceSurfaceWriteSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-write" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const surface = operation.surface;
  return surface.kind === "symbol" &&
    surface.addressSpace === "surface" &&
    semanticReferenceExpressionSupported(operation.value, "any", compiled) &&
    semanticReferenceExpressionSupported(operation.xBytes, "scalar", compiled) &&
    semanticReferenceExpressionSupported(operation.y, "scalar", compiled) &&
    (operation.z === undefined || semanticReferenceExpressionSupported(operation.z, "scalar", compiled));
}

function semanticReferenceSurfaceReadStoreSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "surface-read-store" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return semanticReferenceSurfaceReadTarget(operation.target) !== undefined &&
    semanticReferenceSurfaceReadSupported(
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
      compiled,
    );
}

function semanticReferenceAtomicTargetRootSupported(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  if (ref.addressSpace === "storage") {
    return compiled.kernelIr.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && !param.constant);
  }
  if (ref.addressSpace === "device-global") {
    return compiled.kernelIr.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "device-global");
  }
  if (ref.addressSpace === "shared") {
    return compiled.kernelIr.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "shared");
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
        isBuiltinVectorSymbol(expression.name);
    case "member":
      return isBuiltinVectorMember(expression) || semanticReferenceVectorMemberSupported(expression, compiled);
    case "index":
      if (semanticReferenceVectorIndexSupported(expression, compiled)) return true;
      if (expected === "any" && isSemanticReferenceFloatVectorType(expression.valueType)) {
        return semanticReferenceMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
      }
      return expected === "scalar" && semanticReferenceMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
    case "cast":
      return !expression.pointer && semanticReferenceExpressionSupported(expression.expression, "scalar", compiled);
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" && semanticReferenceExpressionSupported(expression.argument, "scalar", compiled);
    case "binary":
      if (isStoragePointerNullComparison(expression)) return true;
      if (expected === "any" && isSemanticReferenceFloatVectorType(expression.valueType) && semanticReferenceVectorBinaryOperatorSupported(expression.operator)) {
        return semanticReferenceExpressionSupported(expression.left, "any", compiled) &&
          semanticReferenceExpressionSupported(expression.right, "any", compiled);
      }
      return semanticReferenceExpressionSupported(expression.left, "scalar", compiled) &&
        semanticReferenceExpressionSupported(expression.right, "scalar", compiled);
    case "conditional":
      return semanticReferenceExpressionSupported(expression.condition, "scalar", compiled) &&
        semanticReferenceExpressionSupported(expression.consequent, expected, compiled) &&
        semanticReferenceExpressionSupported(expression.alternate, expected, compiled);
    case "assignment":
      return semanticReferenceAssignmentOperatorSupported(expression.operator) &&
        (expression.target.kind === "symbol" && expression.target.addressSpace === "local" ||
          expression.target.kind === "member" && semanticReferenceVectorMemberSupported(expression.target, compiled)) &&
        semanticReferenceExpressionSupported(expression.value, "scalar", compiled);
    case "sequence":
      return expression.expressions.every((item) => semanticReferenceExpressionSupported(item, "scalar", compiled));
    case "update":
      return expression.argument.kind === "symbol" &&
        expression.argument.addressSpace === "local" &&
        (expression.operator === "++" || expression.operator === "--");
    case "call":
      return compiled !== undefined && semanticReferenceFunctionCallSupported(expression, compiled) ||
        semanticReferenceMathCallSupported(expression) ||
        semanticReferenceVectorConstructorSupported(expression, expected, compiled) ||
        expected === "scalar" && semanticReferenceVectorAtCallSupported(expression, compiled);
    case "texture-read":
      return compiled !== undefined &&
        (expected === "any" || expression.valueType === "float") &&
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
    return !(semanticReferenceAtomicCallSupported(expression, compiled) ||
      semanticReferenceFunctionCallSupported(expression, compiled) ||
      semanticReferenceMathCallSupported(expression) ||
      semanticReferenceVectorConstructorSupported(expression, "any", compiled) ||
      semanticReferenceVectorAtCallSupported(expression, compiled)) ||
      expression.args.some((arg) => semanticReferenceExpressionContainsUnsupportedCall(arg, compiled));
  }
  if (expression.kind === "texture-read") {
    return !semanticReferenceTextureReadSupported(expression, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.texture, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.x, compiled) ||
      semanticReferenceExpressionContainsUnsupportedCall(expression.y, compiled);
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
      return [expression.texture, expression.x, expression.y];
    case "surface-read":
      return [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])];
  }
}

function semanticReferenceAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
}

function semanticReferenceVectorBinaryOperatorSupported(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

function isStoragePointerNullComparison(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): boolean {
  if (expression.operator !== "==" && expression.operator !== "!=") return false;
  return isStorageSymbol(expression.left) && isNullLiteral(expression.right) ||
    isStorageSymbol(expression.right) && isNullLiteral(expression.left);
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
  return name === "threadIdx" || name === "blockIdx" || name === "blockDim" || name === "gridDim";
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
    isCudaVectorType(valueType) &&
    cudaVectorFieldIndex(valueType, expression.property) !== undefined;
}

function semanticExpressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

function execSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: SemanticReferenceContext,
): SemanticControl {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        if (operation.target.addressSpace === "shared") break;
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
        evalNumber(operation.expression, context);
        break;
      case "branch":
        if (truthy(evalNumber(operation.condition, context))) {
          const control = execSemanticOperations(operation.consequent, context);
          if (control !== "fallthrough") return control;
        } else {
          const control = execSemanticOperations(operation.alternate, context);
          if (control !== "fallthrough") return control;
        }
        break;
      case "block":
        {
          const control = execSemanticOperations(operation.body, context);
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

function semanticSurfaceReadValueType(valueType: CudaLiteScalarType | undefined): Exclude<CudaLiteScalarType, "void"> {
  return valueType === undefined || valueType === "void" ? "float" : valueType;
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
  if (isSemanticReferenceFloatVectorType(operation.target.valueType)) {
    if (operation.operator !== "=") throw semanticReferenceError("semantic reference supports only direct vector assignment", operation.span);
    return evalSemanticExpression(operation.value, context);
  }
  const right = evalNumber(operation.value, context);
  if (operation.operator === "=") return right;
  const left = readMemory(operation.target, context);
  if (operation.operator === "+=") return left + right;
  if (operation.operator === "-=") return left - right;
  throw semanticReferenceError(`semantic reference does not support assignment '${operation.operator}'`, operation.span);
}

function semanticDeclareValue(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  if (operation.target.dimensions.length > 0) {
    const values = Array.from({ length: totalElements(operation.target.dimensions) }, () => 0);
    if (operation.init?.kind === "initializer") {
      for (const [index, expression] of flattenInitializerExpressions(operation.init).entries()) {
        if (index >= values.length) break;
        values[index] = evalNumber(expression, context);
      }
    }
    return values;
  }
  if (operation.init) return evalSemanticExpression(operation.init, context);
  if (isSemanticReferenceFloatVectorType(operation.target.valueType)) {
    return Array.from({ length: cudaVectorLaneCount(operation.target.valueType) }, () => 0);
  }
  return 0;
}

function execSemanticAtomic(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  context: SemanticReferenceContext,
): void {
  const atomicOp = SEMANTIC_ATOMIC_OPS.get(operation.callee);
  if (!operation.target || !atomicOp) {
    throw semanticReferenceError(`semantic reference does not support atomic '${operation.callee}'`, operation.span);
  }
  const value = operation.args[1];
  if (!value) throw semanticReferenceError(`semantic reference atomic '${operation.callee}' missing operand`, operation.span);
  const oldValue = readMemory(operation.target, context);
  const nextValue = semanticAtomicValue(atomicOp, oldValue, evalNumber(value, context), operation, context);
  writeMemory(operation.target, nextValue, context);
}

function execSemanticCall(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): void {
  if (semanticReferenceVoidFunctionCallSupported(operation, context.compiled)) {
    execSemanticVoidFunctionCall(operation, context);
    return;
  }
  if (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) {
    execSemanticLocalArrayFill(operation, context);
    return;
  }
  throw semanticReferenceError(`semantic reference does not support call '${operation.callee}'`, operation.span);
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
    throw semanticReferenceError(`${operation.callee} expects local array and scalar value`, operation.span);
  }
  const local = context.locals.get(target.name);
  if (!Array.isArray(local)) throw semanticReferenceError(`${operation.callee} expects fixed local array '${target.name}'`, target.span);
  local.fill(evalNumber(valueExpression, context));
}

function semanticAtomicValue(
  atomicOp: SemanticAtomicOp,
  oldValue: number,
  value: number,
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }> | Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  switch (atomicOp) {
    case "add": return oldValue + value;
    case "sub": return oldValue - value;
    case "min": return Math.min(oldValue, value);
    case "max": return Math.max(oldValue, value);
    case "and": return Math.trunc(oldValue) & Math.trunc(value);
    case "or": return Math.trunc(oldValue) | Math.trunc(value);
    case "xor": return Math.trunc(oldValue) ^ Math.trunc(value);
    case "exchange": return value;
    case "cas": {
      const replacement = operation.args[2];
      const callee = operation.kind === "atomic"
        ? operation.callee
        : operation.callee.kind === "symbol" ? operation.callee.name : "<expr>";
      if (!replacement) throw semanticReferenceError(`semantic reference atomic '${callee}' missing replacement`, operation.span);
      return oldValue === value ? evalNumber(replacement, context) : oldValue;
    }
  }
}

function execSemanticLoop(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>,
  context: SemanticReferenceContext,
): SemanticControl {
  if (operation.loopKind === "for") {
    if (operation.init) execSemanticLoopInit(operation.init, context);
    for (let guard = 0; operation.condition === undefined || truthy(evalNumber(operation.condition, context)); guard++) {
      if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
      const control = execSemanticOperations(operation.body, context);
      if (control === "return") return control;
      if (control === "break") return "fallthrough";
      if (operation.update) evalNumber(operation.update, context);
    }
    return "fallthrough";
  }
  if (operation.loopKind === "while") {
    for (let guard = 0; operation.condition === undefined || truthy(evalNumber(operation.condition, context)); guard++) {
      if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
      const control = execSemanticOperations(operation.body, context);
      if (control === "return") return control;
      if (control === "break") return "fallthrough";
    }
    return "fallthrough";
  }
  for (let guard = 0; ; guard++) {
    if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
    const control = execSemanticOperations(operation.body, context);
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

function evalSemanticExpression(expression: SemanticExpression, context: SemanticReferenceContext): SemanticValue {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number" ? expression.value : 0;
    case "symbol":
      return symbolValue(expression.name, context, expression.span);
    case "member":
      return memberValue(evalSemanticExpression(expression.object, context), expression.property, expression.span);
    case "index":
      return readIndexExpression(expression, context);
    case "cast":
      return castNumber(evalNumber(expression.expression, context), expression.valueType);
    case "unary":
      return evalUnary(expression.operator, evalNumber(expression.argument, context));
    case "binary": {
      const left = evalSemanticExpression(expression.left, context);
      const right = evalSemanticExpression(expression.right, context);
      if (Array.isArray(left) || Array.isArray(right) || isSemanticReferenceFloatVectorType(expression.valueType)) {
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
      if (!semanticReferenceAssignmentOperatorSupported(expression.operator) ||
        (expression.target.kind !== "symbol" && (expression.target.kind !== "member" || !semanticReferenceVectorMemberSupported(expression.target, context.compiled)))) {
        throw semanticReferenceError("semantic reference supports only scalar local assignment expressions", expression.span);
      }
      if (expression.target.kind === "member" && semanticReferenceVectorMemberSupported(expression.target, context.compiled)) return assignLocalVectorMember(expression, context);
      if (expression.target.kind !== "symbol") throw semanticReferenceError("semantic reference assignment requires local symbol target", expression.target.span);
      {
        const right = evalNumber(expression.value, context);
        const left = expression.operator === "=" ? 0 : evalNumber(expression.target, context);
        const value = expression.operator === "+=" ? left + right : expression.operator === "-=" ? left - right : right;
        context.locals.set(expression.target.name, value);
        return value;
      }
    case "sequence": {
      let value = 0;
      for (const item of expression.expressions) value = evalNumber(item, context);
      return value;
    }
    case "call":
      if (semanticReferenceAtomicCallSupported(expression, context.compiled)) return evalSemanticAtomicCall(expression, context);
      if (semanticReferenceVectorConstructorSupported(expression, "any", context.compiled)) return evalSemanticVectorConstructor(expression, context);
      if (semanticReferenceVectorAtCallSupported(expression, context.compiled)) return evalSemanticVectorAtCall(expression, context);
      if (semanticReferenceFunctionCallSupported(expression, context.compiled)) return evalSemanticFunctionCall(expression, context);
      if (semanticReferenceMathCallSupported(expression)) return evalSemanticMathCall(expression, context);
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
  if (isSemanticReferenceFloatVectorType(expression.valueType)) {
    return Array.from({ length: cudaVectorLaneCount(expression.valueType) }, (_, lane) => evalSemanticSurfaceLane(surface, surfaceName, xBytes + lane * 4, y, z, context));
  }
  return evalSemanticSurfaceLane(surface, surfaceName, xBytes, y, z, context);
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
    throw semanticReferenceError("semantic reference supports only direct scalar/vector tex2D reads", expression.span);
  }
  const texture = context.textures[expression.texture.name];
  if (!texture) throw semanticReferenceError(`missing texture input '${expression.texture.name}'`, expression.texture.span);
  const channels = texture.channels ?? 1;
  const descriptor = context.textureDescriptors[expression.texture.name] ?? {};
  if (descriptor.filterMode === "linear") {
    return evalSemanticTextureValue(expression.valueType, (lane) => evalSemanticLinearTextureRead(texture, descriptor, expression, context, channels, lane));
  }
  const x = semanticTextureCoord(evalNumber(expression.x, context), texture.width, descriptor, "x");
  const y = semanticTextureCoord(evalNumber(expression.y, context), texture.height, descriptor, "y");
  return evalSemanticTextureValue(expression.valueType, (lane) => texture.data[(y * texture.width + x) * channels + lane] ?? 0);
}

function evalSemanticLinearTextureRead(
  texture: WgslTexture2DInput,
  descriptor: CudaLiteTextureDescriptor,
  expression: Extract<SemanticExpression, { readonly kind: "texture-read" }>,
  context: SemanticReferenceContext,
  channels: number,
  lane = 0,
): number {
  const x = semanticLinearTextureAxis(evalNumber(expression.x, context), texture.width, descriptor, "x");
  const y = semanticLinearTextureAxis(evalNumber(expression.y, context), texture.height, descriptor, "y");
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
  if (!isSemanticReferenceFloatVectorType(valueType)) return laneValue(0);
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) => laneValue(lane));
}

function isSemanticReferenceFloatVectorType(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float2" || valueType === "float3" || valueType === "float4";
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
  if (expression.argument.kind !== "symbol") {
    throw semanticReferenceError("semantic reference supports only local scalar updates", expression.span);
  }
  const oldValue = evalNumber(expression.argument, context);
  const delta = expression.operator === "++" ? 1 : expression.operator === "--" ? -1 : 0;
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
  if (isSemanticReferenceFloatVectorType(ref.valueType)) return readVectorMemory(ref, context);
  return readMemory(ref, context);
}

function evalSemanticAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference atomic call requires symbol callee", expression.span);
  const atomicOp = SEMANTIC_ATOMIC_OPS.get(expression.callee.name);
  const target = semanticAtomicCallTarget(expression);
  const value = expression.args[1];
  if (!atomicOp || !target || !value) {
    throw semanticReferenceError(`semantic reference does not support atomic '${expression.callee.name}'`, expression.span);
  }
  const oldValue = readMemory(target, context);
  writeMemory(target, semanticAtomicValue(atomicOp, oldValue, evalNumber(value, context), expression, context), context);
  return oldValue;
}

function evalSemanticMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference math call requires symbol callee", expression.span);
  const args = expression.args.map((arg) => evalNumber(arg, context));
  switch (expression.callee.name) {
    case "sqrt":
    case "sqrtf":
    case "__fsqrt_rn": return Math.sqrt(args[0] ?? 0);
    case "rsqrt":
    case "rsqrtf":
    case "__frsqrt_rn": return 1 / Math.sqrt(args[0] ?? 0);
    case "exp":
    case "expf":
    case "__expf": return Math.exp(args[0] ?? 0);
    case "log":
    case "logf":
    case "__logf": return Math.log(args[0] ?? 0);
    case "fabs":
    case "fabsf":
    case "abs": return Math.abs(args[0] ?? 0);
    case "floor":
    case "floorf": return Math.floor(args[0] ?? 0);
    case "ceil":
    case "ceilf": return Math.ceil(args[0] ?? 0);
    case "trunc":
    case "truncf": return Math.trunc(args[0] ?? 0);
    case "sin":
    case "sinf":
    case "__sinf": return Math.sin(args[0] ?? 0);
    case "cos":
    case "cosf":
    case "__cosf": return Math.cos(args[0] ?? 0);
    case "tan":
    case "tanf":
    case "__tanf": return Math.tan(args[0] ?? 0);
    case "atan":
    case "atanf": return Math.atan(args[0] ?? 0);
    case "atan2":
    case "atan2f": return Math.atan2(args[0] ?? 0, args[1] ?? 0);
    case "tanh":
    case "tanhf":
    case "__tanhf": return Math.tanh(args[0] ?? 0);
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
    case "__fadd_rn": return (args[0] ?? 0) + (args[1] ?? 0);
    case "__fsub_rn": return (args[0] ?? 0) - (args[1] ?? 0);
    case "__fmul_rn": return (args[0] ?? 0) * (args[1] ?? 0);
    case "__saturatef": return Math.min(1, Math.max(0, args[0] ?? 0));
    case "copysign":
    case "copysignf": return Math.sign(args[1] ?? 0) < 0 || Object.is(args[1] ?? 0, -0) ? -Math.abs(args[0] ?? 0) : Math.abs(args[0] ?? 0);
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
    default:
      throw semanticReferenceError(`semantic reference does not support math call '${expression.callee.name}'`, expression.span);
  }
}

function frexpExponent(value: number): number {
  return value === 0 || !Number.isFinite(value) ? 0 : Math.floor(Math.log2(Math.abs(value))) + 1;
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
  return child.returnValue;
}

function evalSemanticVectorConstructor(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): SemanticValue {
  const valueType = expression.callee.kind === "symbol" ? cudaVectorConstructorType(expression.callee.name) : undefined;
  if (!isSemanticReferenceFloatVectorType(valueType)) throw semanticReferenceError("semantic reference vector constructor requires float vector target", expression.span);
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

function assignLocalVectorMember(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.target.kind !== "member" || expression.target.object.kind !== "symbol") {
    throw semanticReferenceError("semantic reference vector assignment requires local vector member", expression.target.span);
  }
  const current = context.locals.get(expression.target.object.name);
  if (!Array.isArray(current)) throw semanticReferenceError(`missing local vector '${expression.target.object.name}'`, expression.target.span);
  const valueType = semanticExpressionValueType(expression.target.object);
  const lane = valueType === undefined ? undefined : cudaVectorFieldIndex(valueType, expression.target.property);
  if (lane === undefined) throw semanticReferenceError("semantic reference vector assignment requires modeled lane", expression.target.span);
  const right = evalNumber(expression.value, context);
  const left = Number(current[lane] ?? 0);
  const value = expression.operator === "+=" ? left + right : expression.operator === "-=" ? left - right : right;
  current[lane] = value;
  context.locals.set(expression.target.object.name, current);
  return value;
}

function runSemanticFunction(
  fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number],
  args: readonly SemanticExpression[],
  context: SemanticReferenceContext,
  span: SourceSpan,
): SemanticReferenceContext {
  const locals = new Map<string, SemanticValue>();
  const textures = { ...context.textures };
  const textureDescriptors = { ...context.textureDescriptors };
  const surfaces = { ...context.surfaces };
  for (const [index, param] of fn.params.entries()) {
    const arg = args[index];
    if (!arg) throw semanticReferenceError(`semantic reference function '${fn.name}' missing argument`, span);
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
    locals.set(param.name, isSemanticReferenceFloatVectorType(param.valueType) ? evalSemanticExpression(arg, context) : evalNumber(arg, context));
  }
  const child: SemanticReferenceContext = {
    compiled: context.compiled,
    buffers: context.buffers,
    constants: context.constants,
    deviceGlobals: context.deviceGlobals,
    textures,
    textureDescriptors,
    surfaces,
    sharedMemory: context.sharedMemory,
    storageOffsets: new Map(context.storageOffsets),
    scalars: context.scalars,
    locals,
    blockIdx: context.blockIdx,
    threadIdx: context.threadIdx,
    blockDim: context.blockDim,
    gridDim: context.gridDim,
    trace: context.trace,
  };
  const control = execSemanticOperations(fn.body, child);
  if (fn.returnType !== "void" && control !== "return") {
    throw semanticReferenceError(`semantic reference function '${fn.name}' did not return scalar`, fn.span);
  }
  return child;
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
    name === "__powf" ||
    name === "__fdividef" ||
    name === "fdividef" ||
    name === "__fadd_rn" ||
    name === "__fsub_rn" ||
    name === "__fmul_rn" ||
    name === "__fdiv_rn" ||
    name === "copysign" ||
    name === "copysignf" ||
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

function memoryRefFromIndexExpression(expression: SemanticExpression): SemanticMemoryRef | undefined {
  if (expression.kind !== "index") return undefined;
  const indices: SemanticExpression[] = [expression.index];
  let target = expression.target;
  while (target.kind === "index") {
    indices.unshift(target.index);
    target = target.target;
  }
  if (target.kind !== "symbol" || (target.addressSpace !== "storage" && target.addressSpace !== "constant" && target.addressSpace !== "device-global" && target.addressSpace !== "local" && target.addressSpace !== "shared")) return undefined;
  return {
    base: target.name,
    addressSpace: target.addressSpace,
    ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
    indices,
    fields: [],
    span: expression.span,
  };
}

function readMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  if (ref.addressSpace === "local") {
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    const index = flatIndex(ref, context);
    return Number(buffer[index] ?? 0);
  }
  if (ref.addressSpace === "shared") {
    const buffer = context.sharedMemory.get(ref.base);
    if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, ref.span);
    const index = flatIndex(ref, context);
    const ok = index >= 0 && index < buffer.length;
    const value = ok ? Number(buffer[index]) : 0;
    context.trace.sharedReads.push({ name: ref.base, index, value, ok });
    return value;
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

function writeMemory(ref: SemanticMemoryRef, value: number, context: SemanticReferenceContext): void {
  if (ref.addressSpace === "constant") throw semanticReferenceError(`cannot write constant memory '${ref.base}'`, ref.span);
  if (ref.addressSpace === "local") {
    const buffer = context.locals.get(ref.base);
    if (!Array.isArray(buffer)) throw semanticReferenceError(`missing local array '${ref.base}'`, ref.span);
    const index = flatIndex(ref, context);
    if (index >= 0 && index < buffer.length) buffer[index] = value;
    return;
  }
  if (ref.addressSpace === "shared") {
    const buffer = context.sharedMemory.get(ref.base);
    if (!buffer) throw semanticReferenceError(`missing shared memory '${ref.base}'`, ref.span);
    const index = flatIndex(ref, context);
    const ok = index >= 0 && index < buffer.length;
    if (ok) buffer[index] = value;
    context.trace.sharedWrites.push({ name: ref.base, index, value, ok });
    return;
  }
  const buffer = ref.addressSpace === "device-global" ? context.deviceGlobals.get(ref.base) : context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  const index = flatIndex(ref, context);
  const ok = index >= 0 && index < buffer.length;
  if (ok) buffer[index] = value;
  context.trace.writes.push({ name: ref.base, index, value, ok });
}

function readVectorMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number[] {
  if (!isSemanticReferenceFloatVectorType(ref.valueType)) throw semanticReferenceError("semantic reference vector read requires vector memory type", ref.span);
  const laneCount = cudaVectorLaneCount(ref.valueType);
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
  if (!isSemanticReferenceFloatVectorType(ref.valueType)) {
    if (typeof value !== "number") throw semanticReferenceError("semantic reference scalar write received vector value", ref.span);
    writeMemory(ref, value, context);
    return;
  }
  if (!Array.isArray(value)) throw semanticReferenceError("semantic reference vector write received scalar value", ref.span);
  const laneCount = cudaVectorLaneCount(ref.valueType);
  const base = flatIndex(ref, context) * semanticReferenceVectorStorageStride(ref, context);
  const buffer = ref.addressSpace === "device-global" ? context.deviceGlobals.get(ref.base) : context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  for (let lane = 0; lane < laneCount; lane++) {
    const index = base + lane;
    const laneValue = value[lane] ?? 0;
    const ok = index >= 0 && index < buffer.length;
    if (ok) buffer[index] = laneValue;
    context.trace.writes.push({ name: ref.base, index, value: laneValue, ok });
  }
}

function semanticReferenceVectorStorageStride(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  const root = context.compiled.kernelIr.params.find((param) => param.name === ref.base) ??
    context.compiled.kernelIr.memory.find((symbol) => symbol.name === ref.base);
  const valueType = root?.valueType;
  return isSemanticReferenceFloatVectorType(valueType) ? cudaVectorLaneCount(valueType) : 1;
}

function flatIndex(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  if (ref.addressSpace === "local" || ref.addressSpace === "shared") {
    const symbol = context.compiled.kernelIr.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace);
    if (!symbol) throw semanticReferenceError(`unknown ${ref.addressSpace} array '${ref.base}'`, ref.span);
    const dimensions = ref.addressSpace === "shared" ? semanticReferenceSharedDimensions(context.compiled, symbol) : symbol.dimensions;
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
  const constant = context.constants.get(name);
  if (typeof constant === "number") return constant;
  const constantSymbol = context.compiled.kernelIr.memory.find((symbol) => symbol.name === name && symbol.kind === "constant");
  if (constantSymbol && isSemanticReferenceFloatVectorType(constantSymbol.valueType)) {
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

function memberValue(value: SemanticValue, property: string, span: SourceSpan): number {
  if (Array.isArray(value)) {
    const index = vectorFieldIndex(property);
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

function castNumber(value: number, valueType: CudaLiteScalarType): number {
  if (valueType === "int") return Math.trunc(value);
  if (valueType === "uint") return Math.trunc(value) >>> 0;
  if (valueType === "bool") return truthy(value) ? 1 : 0;
  return value;
}

function validateSemanticReferenceInput(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): void {
  for (const param of compiled.kernelIr.params) {
    if (param.addressSpace === "storage") {
      const buffer = input.buffers[param.name];
      if (!buffer) throw semanticReferenceError(`missing buffer input '${param.name}'`, param.span);
      if (param.valueType === "float" && !(buffer instanceof Float32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Float32Array`, param.span);
      }
      if (param.valueType === "int" && !(buffer instanceof Int32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Int32Array`, param.span);
      }
      if (param.valueType === "uint" && !(buffer instanceof Uint32Array)) {
        throw semanticReferenceError(`buffer '${param.name}' expects Uint32Array`, param.span);
      }
    } else if (param.addressSpace === "uniform") {
      if (input.scalars?.[param.name] === undefined) throw semanticReferenceError(`missing scalar input '${param.name}'`, param.span);
    } else if (param.addressSpace === "texture") {
      if (!input.textures?.[param.name]) throw semanticReferenceError(`missing texture input '${param.name}'`, param.span);
    } else if (param.addressSpace === "surface") {
      if (!input.surfaces?.[param.name]) throw semanticReferenceError(`missing surface input '${param.name}'`, param.span);
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
    if (constant.dimensions.length === 0 && !isSemanticReferenceFloatVectorType(constant.valueType) && typeof value !== "number") {
      throw semanticReferenceError(`constant '${constant.name}' expects scalar number`, constant.span);
    }
    if ((constant.dimensions.length > 0 || isSemanticReferenceFloatVectorType(constant.valueType)) && typeof value === "number") {
      throw semanticReferenceError(`constant '${constant.name}' expects typed array`, constant.span);
    }
  }
}

function runSemanticBlockPhases(
  operations: readonly SemanticKernelIrOperation[],
  contexts: readonly SemanticReferenceContext[],
): void {
  for (const segment of semanticBarrierSegments(operations)) {
    for (const context of contexts) {
      const control = execSemanticOperations(segment, context);
      if (control === "break" || control === "continue") {
        throw semanticReferenceError(`semantic reference unexpected ${control} across shared-memory phase`, context.compiled.kernelIr.span);
      }
    }
  }
}

function semanticBarrierSegments(operations: readonly SemanticKernelIrOperation[]): readonly (readonly SemanticKernelIrOperation[])[] {
  const segments: SemanticKernelIrOperation[][] = [[]];
  for (const operation of operations) {
    if (operation.kind === "barrier") {
      segments.push([]);
      continue;
    }
    segments.at(-1)!.push(operation);
  }
  return segments;
}

function semanticReferenceSharedMemory(compiled: CompiledCudaLiteKernel): Map<string, WgslTypedArray> {
  const out = new Map<string, WgslTypedArray>();
  for (const symbol of compiled.kernelIr.memory.filter((item) => item.kind === "shared")) {
    out.set(
      symbol.name,
      typedArrayForScalar(symbol.valueType, totalElements(semanticReferenceSharedDimensions(compiled, symbol))),
    );
  }
  return out;
}

function semanticReferenceSharedDimensions(
  compiled: CompiledCudaLiteKernel,
  symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number],
): readonly number[] {
  const dynamicLeading = compiled.dynamicSharedMemory?.[symbol.name];
  if (dynamicLeading === undefined) return symbol.dimensions;
  return [dynamicLeading, ...symbol.dimensions];
}

function semanticReferenceConstants(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): Map<string, number | WgslTypedArray> {
  const constants = new Map<string, number | WgslTypedArray>();
  for (const constant of compiled.kernelIr.memory.filter((symbol) => symbol.kind === "constant")) {
    const value = input.constants?.[constant.name];
    if (value !== undefined) constants.set(constant.name, value);
    else if (constant.initialized && constant.init !== undefined && (constant.dimensions.length > 0 || isSemanticReferenceFloatVectorType(constant.valueType))) {
      constants.set(constant.name, initializedConstantArrayValue(constant));
    } else if (constant.initialized && constant.dimensions.length === 0 && constant.init !== undefined) {
      constants.set(constant.name, evalConstantInitNumber(constant.init));
    }
  }
  return constants;
}

function initializedConstantArraySupported(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): boolean {
  if (!symbol.init) return false;
  if (symbol.init.kind !== "initializer" && !(isSemanticReferenceFloatVectorType(symbol.valueType) && semanticVectorConstantInitCallSupported(symbol.init))) return false;
  const length = isSemanticReferenceFloatVectorType(symbol.valueType)
    ? cudaVectorLaneCount(symbol.valueType)
    : totalElements(symbol.dimensions);
  return semanticVectorConstantInitExpressions(symbol.init)
    .slice(0, length)
    .every((value) => semanticReferenceExpressionSupported(value, "scalar"));
}

function initializedConstantArrayValue(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): WgslTypedArray {
  const length = isSemanticReferenceFloatVectorType(symbol.valueType)
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

function typedArrayForScalar(valueType: CudaLiteScalarType | undefined, length: number): WgslTypedArray {
  if (valueType === "int") return new Int32Array(length);
  if (valueType === "uint") return new Uint32Array(length);
  return new Float32Array(length);
}

function cloneBuffers(buffers: Readonly<Record<string, WgslTypedArray>>): Map<string, WgslTypedArray> {
  return new Map(Object.entries(buffers).map(([name, buffer]) => [name, cloneTypedArray(buffer)] as const));
}

function cloneTypedArray(buffer: WgslTypedArray): WgslTypedArray {
  if (buffer instanceof Float32Array) return new Float32Array(buffer);
  if (buffer instanceof Int32Array) return new Int32Array(buffer);
  if (buffer instanceof Uint32Array) return new Uint32Array(buffer);
  return new Float32Array(buffer as Float32Array);
}

function cloneSurfaces(surfaces: NonNullable<CompiledKernelInput["surfaces"]>): Record<string, WgslTexture2DInput> {
  return Object.fromEntries(Object.entries(surfaces).map(([name, surface]) => [
    name,
    { ...surface, data: new Float32Array(surface.data) },
  ]));
}

function totalElements(dimensions: readonly number[]): number {
  return dimensions.length === 0 ? 1 : dimensions.reduce((product, dimension) => product * dimension, 1);
}

function flattenInitializerExpressions(expression: SemanticExpression): readonly SemanticExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

function flatIndexForDimensions(dimensions: readonly number[], indices: readonly number[]): number {
  return indices.reduce((sum, index, offset) => {
    const stride = dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
    return sum + index * stride;
  }, 0);
}

function vectorFromTuple(value: readonly [number, number, number]): Vector3 {
  return { x: value[0], y: value[1], z: value[2] };
}

function truthy(value: number): boolean {
  return value !== 0 && !Number.isNaN(value);
}

function freezeTrace(trace: MutableTrace): KernelThreadTrace {
  return {
    blockIdx: trace.blockIdx,
    threadIdx: trace.threadIdx,
    reads: trace.reads.map((item) => ({ ...item })),
    writes: trace.writes.map((item) => ({ ...item })),
    sharedReads: trace.sharedReads.map((item) => ({ ...item })),
    sharedWrites: trace.sharedWrites.map((item) => ({ ...item })),
  };
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
