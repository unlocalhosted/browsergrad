import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { validateCudaKernelLaunch } from "./launch.js";
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

type SemanticValue = number | Vector3;

interface Vector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface SemanticReferenceContext {
  readonly compiled: CompiledCudaLiteKernel;
  readonly buffers: Map<string, WgslTypedArray>;
  readonly scalars: Readonly<Record<string, number>>;
  readonly locals: Map<string, SemanticValue>;
  readonly blockIdx: Vector3;
  readonly threadIdx: Vector3;
  readonly blockDim: Vector3;
  readonly gridDim: Vector3;
  readonly trace: MutableTrace;
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
    compiled.kernelIr.memory.every((symbol) => symbol.kind === "local" || symbol.kind === "shared") &&
    unsupportedSemanticReferenceOperation(compiled.kernelIr.operations, compiled) === undefined;
}

export function runCompiledKernelSemanticReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): ReferenceKernelResult {
  const unsupported = unsupportedSemanticReferenceOperation(compiled.kernelIr.operations, compiled);
  if (unsupported) throw semanticReferenceError(`semantic reference does not support ${unsupported.kind}`, unsupported.span);
  validateCudaKernelLaunch(launch, compiled.kernelIr.workgroupSize);
  validateSemanticReferenceInput(compiled, input);

  const buffers = cloneBuffers(input.buffers);
  const traces: MutableTrace[] = [];
  const blockDim = vectorFromTuple(launch.blockDim);
  const gridDim = vectorFromTuple(launch.gridDim);
  const scalars = input.scalars ?? {};
  for (let bz = 0; bz < launch.gridDim[2]; bz++) {
    for (let by = 0; by < launch.gridDim[1]; by++) {
      for (let bx = 0; bx < launch.gridDim[0]; bx++) {
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
              execSemanticOperations(compiled.kernelIr.operations, {
                compiled,
                buffers,
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
      }
    }
  }

  const readback = input.readback ?? compiled.kernelIr.params
    .filter((param) => param.addressSpace === "storage" && param.pointer && !param.constant)
    .map((param) => param.name);
  return {
    buffers: Object.fromEntries(readback.map((name) => {
      const buffer = buffers.get(name);
      if (!buffer) throw semanticReferenceError(`missing readback buffer '${name}'`, compiled.kernelIr.span);
      return [name, buffer];
    })),
    trace: traces.map(freezeTrace),
  };
}

function unsupportedSemanticReferenceOperation(
  operations: readonly SemanticKernelIrOperation[],
  compiled: CompiledCudaLiteKernel,
): SemanticKernelIrOperation | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        if (operation.target.addressSpace !== "local" || operation.target.pointer || operation.target.dimensions.length > 0) return operation;
        if (!semanticReferenceScalarTypeSupported(operation.target.valueType)) return operation;
        if (operation.init && !semanticReferenceExpressionSupported(operation.init, "scalar")) return operation;
        break;
      case "store":
        if (operation.operator !== "=") return operation;
        if (!semanticReferenceMemoryRefSupported(operation.target)) return operation;
        if (!compiled.kernelIr.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage")) return operation;
        if (!semanticReferenceExpressionSupported(operation.value, "scalar")) return operation;
        break;
      case "expression":
        if (!semanticReferenceExpressionSupported(operation.expression, "scalar")) return operation;
        break;
      case "branch":
        if (!semanticReferenceExpressionSupported(operation.condition, "scalar")) return operation;
        break;
      default:
        return operation;
    }
    if (operation.kind === "branch") {
      return unsupportedSemanticReferenceOperation(operation.consequent, compiled) ??
        unsupportedSemanticReferenceOperation(operation.alternate, compiled);
    }
  }
  return undefined;
}

function semanticReferenceParamSupported(param: CompiledCudaLiteKernel["kernelIr"]["params"][number]): boolean {
  if (param.addressSpace === "storage") return Boolean(param.pointer) && param.valueType === "float";
  if (param.addressSpace === "uniform") return semanticReferenceScalarTypeSupported(param.valueType);
  return false;
}

function semanticReferenceScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" || valueType === "int" || valueType === "uint";
}

function semanticReferenceMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  return ref.addressSpace === "storage" &&
    ref.indices.length === 1 &&
    ref.fields.length === 0 &&
    semanticReferenceExpressionSupported(ref.indices[0]!, "scalar");
}

function semanticReferenceExpressionSupported(expression: SemanticExpression, expected: "scalar" | "any"): boolean {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number";
    case "symbol":
      return expression.addressSpace === "uniform" || expression.addressSpace === "local" || isBuiltinVectorSymbol(expression.name);
    case "member":
      return isBuiltinVectorMember(expression);
    case "index":
      return expected === "scalar" && semanticReferenceMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
    case "cast":
      return !expression.pointer && semanticReferenceExpressionSupported(expression.expression, "scalar");
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" && semanticReferenceExpressionSupported(expression.argument, "scalar");
    case "binary":
      return semanticReferenceExpressionSupported(expression.left, "scalar") &&
        semanticReferenceExpressionSupported(expression.right, "scalar");
    case "conditional":
      return semanticReferenceExpressionSupported(expression.condition, "scalar") &&
        semanticReferenceExpressionSupported(expression.consequent, expected) &&
        semanticReferenceExpressionSupported(expression.alternate, expected);
    case "assignment":
      return expression.operator === "=" &&
        expression.target.kind === "symbol" &&
        expression.target.addressSpace === "local" &&
        semanticReferenceExpressionSupported(expression.value, "scalar");
    case "sequence":
      return expression.expressions.every((item) => semanticReferenceExpressionSupported(item, "scalar"));
    case "call":
    case "update":
    case "initializer":
      return false;
  }
}

function unsupportedMemoryRef(span: SourceSpan): SemanticMemoryRef {
  return { base: "", addressSpace: "unknown", indices: [], fields: [], span };
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
): void {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        context.locals.set(operation.target.name, operation.init ? evalNumber(operation.init, context) : 0);
        break;
      case "store":
        writeMemory(operation.target, evalNumber(operation.value, context), context);
        break;
      case "expression":
        evalNumber(operation.expression, context);
        break;
      case "branch":
        if (truthy(evalNumber(operation.condition, context))) execSemanticOperations(operation.consequent, context);
        else execSemanticOperations(operation.alternate, context);
        break;
      default:
        throw semanticReferenceError(`semantic reference does not support ${operation.kind}`, operation.span);
    }
  }
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
      if (expression.operator !== "=" || expression.target.kind !== "symbol") {
        throw semanticReferenceError("semantic reference supports only scalar local assignment expressions", expression.span);
      }
      {
        const value = evalNumber(expression.value, context);
        context.locals.set(expression.target.name, value);
        return value;
      }
    case "sequence": {
      let value = 0;
      for (const item of expression.expressions) value = evalNumber(item, context);
      return value;
    }
    case "call":
    case "update":
    case "initializer":
      throw semanticReferenceError(`semantic reference does not support ${expression.kind} expression`, expression.span);
  }
}

function readIndexExpression(expression: Extract<SemanticExpression, { kind: "index" }>, context: SemanticReferenceContext): number {
  const ref = memoryRefFromIndexExpression(expression);
  if (!ref) throw semanticReferenceError("semantic reference supports only direct storage indexing", expression.span);
  return readMemory(ref, context);
}

function memoryRefFromIndexExpression(expression: SemanticExpression): SemanticMemoryRef | undefined {
  if (expression.kind !== "index") return undefined;
  const indices: SemanticExpression[] = [expression.index];
  let target = expression.target;
  while (target.kind === "index") {
    indices.unshift(target.index);
    target = target.target;
  }
  if (target.kind !== "symbol" || target.addressSpace !== "storage") return undefined;
  return {
    base: target.name,
    addressSpace: "storage",
    ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
    indices,
    fields: [],
    span: expression.span,
  };
}

function readMemory(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  const buffer = context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  const index = flatIndex(ref, context);
  const ok = index >= 0 && index < buffer.length;
  const value = ok ? Number(buffer[index]) : 0;
  context.trace.reads.push({ name: ref.base, index, value, ok });
  return value;
}

function writeMemory(ref: SemanticMemoryRef, value: number, context: SemanticReferenceContext): void {
  const buffer = context.buffers.get(ref.base);
  if (!buffer) throw semanticReferenceError(`missing buffer input '${ref.base}'`, ref.span);
  const index = flatIndex(ref, context);
  const ok = index >= 0 && index < buffer.length;
  if (ok) buffer[index] = value;
  context.trace.writes.push({ name: ref.base, index, value, ok });
}

function flatIndex(ref: SemanticMemoryRef, context: SemanticReferenceContext): number {
  if (ref.indices.length !== 1) throw semanticReferenceError("semantic reference supports only 1D storage indexing", ref.span);
  return Math.trunc(evalNumber(ref.indices[0]!, context));
}

function symbolValue(name: string, context: SemanticReferenceContext, span: SourceSpan): SemanticValue {
  if (name === "threadIdx") return context.threadIdx;
  if (name === "blockIdx") return context.blockIdx;
  if (name === "blockDim") return context.blockDim;
  if (name === "gridDim") return context.gridDim;
  if (context.locals.has(name)) return context.locals.get(name)!;
  const scalar = context.scalars[name];
  if (scalar !== undefined) return scalar;
  throw semanticReferenceError(`unknown semantic reference symbol '${name}'`, span);
}

function memberValue(value: SemanticValue, property: string, span: SourceSpan): number {
  if (typeof value === "number") throw semanticReferenceError("semantic member target is not a vector", span);
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
    } else if (param.addressSpace === "uniform") {
      if (input.scalars?.[param.name] === undefined) throw semanticReferenceError(`missing scalar input '${param.name}'`, param.span);
    } else {
      throw semanticReferenceError(`semantic reference does not support ${param.addressSpace} parameter '${param.name}'`, param.span);
    }
  }
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
    sharedReads: [],
    sharedWrites: [],
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
