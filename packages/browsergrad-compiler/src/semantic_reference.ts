import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { validateCudaKernelLaunch } from "./launch.js";
import { deviceGlobalBufferInputs } from "./webgpu_inputs.js";
import type {
  CompiledCudaLiteKernel,
  CompiledKernelInput,
  CudaLiteDiagnostic,
  CudaLiteScalarType,
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
  "floor", "floorf", "ceil", "ceilf", "sin", "sinf", "__sinf", "cos", "cosf", "__cosf",
  "tan", "tanf", "__tanf", "atan", "atanf", "atan2", "atan2f", "tanh", "tanhf", "__tanhf",
  "fmin", "fminf", "min", "fmax", "fmaxf", "max", "pow", "powf",
  "__fdividef", "fma", "fmaf", "__fmaf_rn", "lerp", "div_ceil", "ceil_div",
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
    semanticReferenceSharedShapeSupported(compiled) &&
    unsupportedSemanticReferenceOperation(compiled.kernelIr.operations, compiled) === undefined;
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
    .filter((param) => param.addressSpace === "storage" && param.pointer && !param.constant)
    .map((param) => param.name)
    .concat(compiled.kernelIr.memory.filter((symbol) => symbol.kind === "device-global").map((symbol) => symbol.name));
  return {
    buffers: Object.fromEntries(readback.map((name) => {
      const buffer = buffers.get(name) ?? deviceGlobals.get(name);
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
        if (!semanticReferenceScalarTypeSupported(operation.target.valueType)) return operation;
        if (operation.target.dimensions.length > 0 && operation.init && !semanticReferenceLocalArrayInitSupported(operation.init)) return operation;
        if (operation.target.dimensions.length === 0 && operation.init && !semanticReferenceExpressionSupported(operation.init, "scalar", compiled)) return operation;
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
      case "loop":
        if (operation.init && !semanticReferenceLoopInitSupported(operation.init, compiled)) return operation;
        if (operation.condition && !semanticReferenceExpressionSupported(operation.condition, "scalar", compiled)) return operation;
        if (operation.update && !semanticReferenceExpressionSupported(operation.update, "scalar", compiled)) return operation;
        break;
      case "return":
        if (operation.value && (!allowReturnValue || !semanticReferenceExpressionSupported(operation.value, "scalar", compiled))) return operation;
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
    if (operation.kind === "loop") return unsupportedSemanticReferenceOperation(operation.body, compiled, allowReturnValue);
  }
  return undefined;
}

function semanticReferenceParamSupported(param: CompiledCudaLiteKernel["kernelIr"]["params"][number]): boolean {
  if (param.addressSpace === "storage") return Boolean(param.pointer) && semanticReferenceScalarTypeSupported(param.valueType);
  if (param.addressSpace === "uniform") return semanticReferenceScalarTypeSupported(param.valueType);
  return false;
}

function semanticReferenceMemorySymbolSupported(symbol: CompiledCudaLiteKernel["kernelIr"]["memory"][number]): boolean {
  if (symbol.kind === "local" || symbol.kind === "shared") return true;
  if (symbol.kind === "constant") return !symbol.initialized && semanticReferenceScalarTypeSupported(symbol.valueType);
  if (symbol.kind === "device-global") return semanticReferenceScalarTypeSupported(symbol.valueType);
  return false;
}

function semanticReferenceSharedShapeSupported(compiled: CompiledCudaLiteKernel): boolean {
  if (!compiled.kernelIr.memory.some((symbol) => symbol.kind === "shared")) return true;
  return operationsHaveOnlyTopLevelSharedBarriers(compiled.kernelIr.operations);
}

function operationsHaveOnlyTopLevelSharedBarriers(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) =>
    operation.kind !== "branch" &&
    operation.kind !== "loop" &&
    operation.kind !== "block"
  );
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

function semanticReferenceMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local" && ref.addressSpace !== "shared") {
    return false;
  }
  if (ref.fields.length > 0) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "shared" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "storage" && ref.indices.length === 0) return false;
  if (ref.addressSpace === "constant" && ref.indices.length !== 1) return false;
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
    expression.kind === "call" && (semanticReferenceAtomicCallSupported(expression, compiled) || semanticReferenceMathCallSupported(expression));
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

function semanticReferenceFunctionCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  if (expression.callee.kind !== "symbol") return false;
  const callee = expression.callee.name;
  const fn = compiled.kernelIr.functions.find((item) => item.name === callee);
  if (!fn || !semanticReferenceScalarTypeSupported(fn.returnType)) return false;
  if (fn.params.some((param) => param.pointer || param.addressSpace !== "local")) return false;
  if (fn.params.some((param) => !semanticReferenceScalarTypeSupported(param.valueType))) return false;
  if (!semanticReferenceFunctionBodyShapeSupported(fn.body)) return false;
  return expression.args.length === fn.params.length &&
    expression.args.every((arg) => semanticReferenceExpressionSupported(arg, "scalar", compiled)) &&
    unsupportedSemanticReferenceOperation(fn.body, compiled, true) === undefined;
}

function semanticReferenceFunctionBodyShapeSupported(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) => {
    if (operation.kind === "declare") return operation.target.addressSpace === "local" && !operation.target.pointer && operation.target.dimensions.length === 0;
    if (operation.kind === "store") return operation.target.addressSpace === "local";
    if (operation.kind === "branch") return semanticReferenceFunctionBodyShapeSupported(operation.consequent) && semanticReferenceFunctionBodyShapeSupported(operation.alternate);
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

function semanticReferenceAtomicTargetRootSupported(ref: SemanticMemoryRef, compiled: CompiledCudaLiteKernel): boolean {
  if (ref.addressSpace === "storage") {
    return compiled.kernelIr.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && !param.constant);
  }
  if (ref.addressSpace === "device-global") {
    return compiled.kernelIr.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "device-global");
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
      return expression.addressSpace === "uniform" ||
        expression.addressSpace === "local" ||
        expression.addressSpace === "constant" ||
        expression.addressSpace === "device-global" ||
        isBuiltinVectorSymbol(expression.name);
    case "member":
      return isBuiltinVectorMember(expression);
    case "index":
      return expected === "scalar" && semanticReferenceMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
    case "cast":
      return !expression.pointer && semanticReferenceExpressionSupported(expression.expression, "scalar", compiled);
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" && semanticReferenceExpressionSupported(expression.argument, "scalar", compiled);
    case "binary":
      if (isStoragePointerNullComparison(expression)) return true;
      return semanticReferenceExpressionSupported(expression.left, "scalar", compiled) &&
        semanticReferenceExpressionSupported(expression.right, "scalar", compiled);
    case "conditional":
      return semanticReferenceExpressionSupported(expression.condition, "scalar", compiled) &&
        semanticReferenceExpressionSupported(expression.consequent, expected, compiled) &&
        semanticReferenceExpressionSupported(expression.alternate, expected, compiled);
    case "assignment":
      return semanticReferenceAssignmentOperatorSupported(expression.operator) &&
        expression.target.kind === "symbol" &&
        expression.target.addressSpace === "local" &&
        semanticReferenceExpressionSupported(expression.value, "scalar", compiled);
    case "sequence":
      return expression.expressions.every((item) => semanticReferenceExpressionSupported(item, "scalar", compiled));
    case "update":
      return expression.argument.kind === "symbol" &&
        expression.argument.addressSpace === "local" &&
        (expression.operator === "++" || expression.operator === "--");
    case "call":
      return compiled !== undefined && semanticReferenceFunctionCallSupported(expression, compiled) ||
        semanticReferenceMathCallSupported(expression);
    case "initializer":
      return false;
  }
}

function unsupportedMemoryRef(span: SourceSpan): SemanticMemoryRef {
  return { base: "", addressSpace: "unknown", indices: [], fields: [], span };
}

function semanticReferenceAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
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
        writeMemory(operation.target, storeValue(operation, context), context);
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

function storeValue(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  context: SemanticReferenceContext,
): number {
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
  return operation.init ? evalNumber(operation.init, context) : 0;
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
  if (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) {
    execSemanticLocalArrayFill(operation, context);
    return;
  }
  throw semanticReferenceError(`semantic reference does not support call '${operation.callee}'`, operation.span);
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
    case "binary":
      return evalBinary(expression.operator, evalNumber(expression.left, context), evalNumber(expression.right, context));
    case "conditional":
      return truthy(evalNumber(expression.condition, context))
        ? evalSemanticExpression(expression.consequent, context)
        : evalSemanticExpression(expression.alternate, context);
    case "assignment":
      if (!semanticReferenceAssignmentOperatorSupported(expression.operator) || expression.target.kind !== "symbol") {
        throw semanticReferenceError("semantic reference supports only scalar local assignment expressions", expression.span);
      }
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
      if (semanticReferenceFunctionCallSupported(expression, context.compiled)) return evalSemanticFunctionCall(expression, context);
      if (semanticReferenceMathCallSupported(expression)) return evalSemanticMathCall(expression, context);
      throw semanticReferenceError(`semantic reference does not support ${expression.kind} expression`, expression.span);
    case "initializer":
      throw semanticReferenceError(`semantic reference does not support ${expression.kind} expression`, expression.span);
    case "update":
      return evalUpdate(expression, context);
  }
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

function readIndexExpression(expression: Extract<SemanticExpression, { kind: "index" }>, context: SemanticReferenceContext): number {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref) throw semanticReferenceError("semantic reference supports only direct storage indexing", expression.span);
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
    case "powf": return Math.pow(args[0] ?? 0, args[1] ?? 0);
    case "__fdividef": return (args[0] ?? 0) / (args[1] ?? 0);
    case "fma":
    case "fmaf":
    case "__fmaf_rn": return (args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0);
    case "lerp": return (args[0] ?? 0) + (args[2] ?? 0) * ((args[1] ?? 0) - (args[0] ?? 0));
    case "div_ceil":
    case "ceil_div": return Math.trunc((Math.trunc(args[0] ?? 0) + Math.trunc(args[1] ?? 1) - 1) / Math.trunc(args[1] ?? 1));
    default:
      throw semanticReferenceError(`semantic reference does not support math call '${expression.callee.name}'`, expression.span);
  }
}

function evalSemanticFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  context: SemanticReferenceContext,
): number {
  if (expression.callee.kind !== "symbol") throw semanticReferenceError("semantic reference function call requires symbol callee", expression.span);
  const callee = expression.callee.name;
  const fn = context.compiled.kernelIr.functions.find((item) => item.name === callee);
  if (!fn) throw semanticReferenceError(`semantic reference unknown function '${callee}'`, expression.span);
  const locals = new Map<string, SemanticValue>();
  for (const [index, param] of fn.params.entries()) {
    const arg = expression.args[index];
    if (!arg) throw semanticReferenceError(`semantic reference function '${fn.name}' missing argument`, expression.span);
    locals.set(param.name, evalNumber(arg, context));
  }
  const child: SemanticReferenceContext = {
    compiled: context.compiled,
    buffers: context.buffers,
    constants: context.constants,
    deviceGlobals: context.deviceGlobals,
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
  if (control !== "return" || typeof child.returnValue !== "number") {
    throw semanticReferenceError(`semantic reference function '${fn.name}' did not return scalar`, fn.span);
  }
  return child.returnValue;
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
  if (firstArg.kind === "unary" && firstArg.operator === "&" && firstArg.argument.kind === "symbol" && firstArg.argument.addressSpace === "device-global") {
    return {
      base: firstArg.argument.name,
      addressSpace: "device-global",
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

function flatIndex(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  if (ref.addressSpace === "local" || ref.addressSpace === "shared") {
    const symbol = context.compiled.kernelIr.memory.find((item) => item.name === ref.base && item.kind === ref.addressSpace);
    if (!symbol) throw semanticReferenceError(`unknown ${ref.addressSpace} array '${ref.base}'`, ref.span);
    if (ref.addressSpace === "shared" && symbol.dimensions.length === 0 && ref.indices.length === 1) {
      return Math.trunc(evalNumber(ref.indices[0]!, context));
    }
    if (ref.indices.length !== symbol.dimensions.length) throw semanticReferenceError(`${ref.addressSpace} array '${ref.base}' index rank mismatch`, ref.span);
    return flatIndexForDimensions(symbol.dimensions, ref.indices.map((index) => Math.trunc(evalNumber(index, context))));
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
  const global = context.compiled.kernelIr.memory.find((symbol) => symbol.name === name && symbol.kind === "device-global");
  if (global && global.dimensions.length === 0) return readMemory({ base: name, addressSpace: "device-global", indices: [], fields: [], span }, context);
  const storageParam = context.compiled.kernelIr.params.find((param) => param.name === name && param.addressSpace === "storage");
  if (storageParam) return context.buffers.has(name) ? 1 : 0;
  throw semanticReferenceError(`unknown semantic reference symbol '${name}'`, span);
}

function memberValue(value: SemanticValue, property: string, span: SourceSpan): number {
  if (typeof value === "number" || Array.isArray(value)) throw semanticReferenceError("semantic member target is not a vector", span);
  if (property === "x") return value.x;
  if (property === "y") return value.y;
  if (property === "z") return value.z;
  throw semanticReferenceError(`unsupported semantic member '${property}'`, span);
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
    } else {
      throw semanticReferenceError(`semantic reference does not support ${param.addressSpace} parameter '${param.name}'`, param.span);
    }
  }
  for (const constant of compiled.kernelIr.memory.filter((symbol) => symbol.kind === "constant")) {
    if (constant.initialized) continue;
    const value = input.constants?.[constant.name];
    if (value === undefined) throw semanticReferenceError(`missing constant input '${constant.name}'`, constant.span);
    if (constant.dimensions.length === 0 && typeof value !== "number") {
      throw semanticReferenceError(`constant '${constant.name}' expects scalar number`, constant.span);
    }
    if (constant.dimensions.length > 0 && typeof value === "number") {
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
      typedArrayForScalar(symbol.valueType, compiled.dynamicSharedMemory?.[symbol.name] ?? totalElements(symbol.dimensions)),
    );
  }
  return out;
}

function semanticReferenceConstants(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): Map<string, number | WgslTypedArray> {
  const constants = new Map<string, number | WgslTypedArray>();
  for (const constant of compiled.kernelIr.memory.filter((symbol) => symbol.kind === "constant")) {
    const value = input.constants?.[constant.name];
    if (value !== undefined) constants.set(constant.name, value);
  }
  return constants;
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
