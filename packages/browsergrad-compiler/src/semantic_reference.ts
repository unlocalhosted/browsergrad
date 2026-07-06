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
type SemanticAtomicOp = "add" | "sub" | "min" | "max" | "and" | "or" | "xor" | "exchange" | "cas";

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
        if (!semanticReferenceAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticReferenceMemoryRefSupported(operation.target)) return operation;
        if (!compiled.kernelIr.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage")) return operation;
        if (!semanticReferenceValueExpressionSupported(operation.value, compiled)) return operation;
        break;
      case "atomic":
        if (!semanticReferenceAtomicSupported(operation, compiled)) return operation;
        break;
      case "expression":
        if (!semanticReferenceExpressionSupported(operation.expression, "scalar")) return operation;
        break;
      case "branch":
        if (!semanticReferenceExpressionSupported(operation.condition, "scalar")) return operation;
        break;
      case "loop":
        if (operation.init && !semanticReferenceLoopInitSupported(operation.init, compiled)) return operation;
        if (operation.condition && !semanticReferenceExpressionSupported(operation.condition, "scalar")) return operation;
        if (operation.update && !semanticReferenceExpressionSupported(operation.update, "scalar")) return operation;
        break;
      default:
        return operation;
    }
    if (operation.kind === "branch") {
      return unsupportedSemanticReferenceOperation(operation.consequent, compiled) ??
        unsupportedSemanticReferenceOperation(operation.alternate, compiled);
    }
    if (operation.kind === "loop") return unsupportedSemanticReferenceOperation(operation.body, compiled);
  }
  return undefined;
}

function semanticReferenceParamSupported(param: CompiledCudaLiteKernel["kernelIr"]["params"][number]): boolean {
  if (param.addressSpace === "storage") return Boolean(param.pointer) && semanticReferenceScalarTypeSupported(param.valueType);
  if (param.addressSpace === "uniform") return semanticReferenceScalarTypeSupported(param.valueType);
  return false;
}

function semanticReferenceLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  compiled: CompiledCudaLiteKernel,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? unsupportedSemanticReferenceOperation([init], compiled) === undefined
    : semanticReferenceExpressionSupported(init, "scalar");
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

function semanticReferenceAtomicSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  compiled: CompiledCudaLiteKernel,
): boolean {
  const atomicOp = SEMANTIC_ATOMIC_OPS.get(operation.callee);
  if (!atomicOp) return false;
  if (!operation.target || !semanticReferenceMemoryRefSupported(operation.target)) return false;
  if (operation.target.valueType !== "uint" && operation.target.valueType !== "int") return false;
  if (!compiled.kernelIr.params.some((param) => param.name === operation.target?.base && param.addressSpace === "storage" && !param.constant)) {
    return false;
  }
  const expectedArgs = atomicOp === "cas" ? 3 : 2;
  return operation.args.length >= expectedArgs &&
    operation.args.slice(1, expectedArgs).every((arg) => semanticReferenceExpressionSupported(arg, "scalar"));
}

function semanticReferenceValueExpressionSupported(expression: SemanticExpression, compiled: CompiledCudaLiteKernel): boolean {
  return semanticReferenceExpressionSupported(expression, "scalar") ||
    expression.kind === "call" && semanticReferenceAtomicCallSupported(expression, compiled);
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
  if (!compiled.kernelIr.params.some((param) => param.name === target.base && param.addressSpace === "storage" && !param.constant)) {
    return false;
  }
  const expectedArgs = atomicOp === "cas" ? 3 : 2;
  return expression.args.length >= expectedArgs &&
    expression.args.slice(1, expectedArgs).every((arg) => semanticReferenceExpressionSupported(arg, "scalar"));
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
      if (isStoragePointerNullComparison(expression)) return true;
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
    case "update":
      return expression.argument.kind === "symbol" &&
        expression.argument.addressSpace === "local" &&
        (expression.operator === "++" || expression.operator === "--");
    case "call":
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
): void {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        context.locals.set(operation.target.name, operation.init ? evalNumber(operation.init, context) : 0);
        break;
      case "store":
        writeMemory(operation.target, storeValue(operation, context), context);
        break;
      case "atomic":
        execSemanticAtomic(operation, context);
        break;
      case "expression":
        evalNumber(operation.expression, context);
        break;
      case "branch":
        if (truthy(evalNumber(operation.condition, context))) execSemanticOperations(operation.consequent, context);
        else execSemanticOperations(operation.alternate, context);
        break;
      case "loop":
        execSemanticLoop(operation, context);
        break;
      default:
        throw semanticReferenceError(`semantic reference does not support ${operation.kind}`, operation.span);
    }
  }
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
): void {
  if (operation.loopKind === "for") {
    if (operation.init) execSemanticLoopInit(operation.init, context);
    for (let guard = 0; operation.condition === undefined || truthy(evalNumber(operation.condition, context)); guard++) {
      if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
      execSemanticOperations(operation.body, context);
      if (operation.update) evalNumber(operation.update, context);
    }
    return;
  }
  if (operation.loopKind === "while") {
    for (let guard = 0; operation.condition === undefined || truthy(evalNumber(operation.condition, context)); guard++) {
      if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
      execSemanticOperations(operation.body, context);
    }
    return;
  }
  for (let guard = 0; ; guard++) {
    if (guard > 1_000_000) throw semanticReferenceError("semantic reference loop exceeded iteration cap", operation.span);
    execSemanticOperations(operation.body, context);
    if (!operation.condition || !truthy(evalNumber(operation.condition, context))) return;
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
      if (semanticReferenceAtomicCallSupported(expression, context.compiled)) return evalSemanticAtomicCall(expression, context);
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

function semanticAtomicCallTarget(expression: Extract<SemanticExpression, { readonly kind: "call" }>): SemanticMemoryRef | undefined {
  const firstArg = expression.args[0];
  if (!firstArg) return undefined;
  if (firstArg.kind === "unary" && firstArg.operator === "&" && firstArg.argument.kind === "index") {
    return memoryRefFromIndexExpression(firstArg.argument);
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
  if (name === "NULL" || name === "nullptr") return 0;
  if (name === "threadIdx") return context.threadIdx;
  if (name === "blockIdx") return context.blockIdx;
  if (name === "blockDim") return context.blockDim;
  if (name === "gridDim") return context.gridDim;
  if (context.locals.has(name)) return context.locals.get(name)!;
  const scalar = context.scalars[name];
  if (scalar !== undefined) return scalar;
  const storageParam = context.compiled.kernelIr.params.find((param) => param.name === name && param.addressSpace === "storage");
  if (storageParam) return context.buffers.has(name) ? 1 : 0;
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
