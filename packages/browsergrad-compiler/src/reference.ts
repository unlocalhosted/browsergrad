import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CudaLiteExpression,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelLaunch,
  type KernelMemoryAccess,
  type KernelThreadTrace,
  type ReferenceKernelResult,
} from "./types.js";

type LocalValue = number | Vector3;
interface Vector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface LValue {
  readonly name: string;
  readonly space: "local" | "buffer" | "shared";
  readonly index?: number;
}

interface ThreadContext {
  readonly blockIdx: Vector3;
  readonly threadIdx: Vector3;
  readonly blockDim: Vector3;
  readonly gridDim: Vector3;
  readonly buffers: Map<string, WgslTypedArray>;
  readonly scalars: Readonly<Record<string, number>>;
  readonly locals: Map<string, LocalValue>;
  readonly shared: Map<string, SharedArrayValue>;
  readonly trace: MutableTrace;
}

interface SharedArrayValue {
  readonly dimensions: readonly number[];
  readonly data: WgslTypedArray;
}

interface MutableTrace {
  readonly blockIdx: readonly [number, number, number];
  readonly threadIdx: readonly [number, number, number];
  readonly reads: KernelMemoryAccess[];
  readonly writes: KernelMemoryAccess[];
  readonly sharedReads: KernelMemoryAccess[];
  readonly sharedWrites: KernelMemoryAccess[];
}

type BarrierGenerator = Generator<"barrier", void, void>;

export function runCompiledKernelReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): ReferenceKernelResult {
  validateLaunch(launch, compiled.ir.workgroupSize);
  validateInputs(compiled, input);
  const buffers = cloneBuffers(input.buffers);
  const scalars = input.scalars ?? {};
  const traces: MutableTrace[] = [];

  for (let bz = 0; bz < launch.gridDim[2]; bz++) {
    for (let by = 0; by < launch.gridDim[1]; by++) {
      for (let bx = 0; bx < launch.gridDim[0]; bx++) {
        runBlock(compiled, buffers, scalars, {
          x: bx,
          y: by,
          z: bz,
        }, vectorFromTuple(launch.blockDim), vectorFromTuple(launch.gridDim), traces);
      }
    }
  }

  const readback = input.readback ??
    compiled.ir.params.filter((param) => param.pointer && !param.constant).map((param) => param.name);
  const result: Record<string, WgslTypedArray> = {};
  for (const name of readback) {
    const buffer = buffers.get(name);
    if (!buffer) throw compilerFailure(`missing readback buffer '${name}'`);
    result[name] = cloneTypedArray(buffer);
  }
  return { buffers: result, trace: traces.map(freezeTrace) };
}

function runBlock(
  compiled: CompiledCudaLiteKernel,
  buffers: Map<string, WgslTypedArray>,
  scalars: Readonly<Record<string, number>>,
  blockIdx: Vector3,
  blockDim: Vector3,
  gridDim: Vector3,
  traces: MutableTrace[],
): void {
  const shared = allocateShared(compiled.ir.sharedDeclarations);
  const generators: BarrierGenerator[] = [];
  const active: boolean[] = [];

  for (let tz = 0; tz < blockDim.z; tz++) {
    for (let ty = 0; ty < blockDim.y; ty++) {
      for (let tx = 0; tx < blockDim.x; tx++) {
        const trace: MutableTrace = {
          blockIdx: [blockIdx.x, blockIdx.y, blockIdx.z],
          threadIdx: [tx, ty, tz],
          reads: [],
          writes: [],
          sharedReads: [],
          sharedWrites: [],
        };
        traces.push(trace);
        const context: ThreadContext = {
          blockIdx,
          threadIdx: { x: tx, y: ty, z: tz },
          blockDim,
          gridDim,
          buffers,
          scalars,
          locals: new Map(),
          shared,
          trace,
        };
        generators.push(execStatements(compiled.ir.body, context));
        active.push(true);
      }
    }
  }

  while (active.some(Boolean)) {
    let activeBefore = 0;
    let barriers = 0;
    for (let i = 0; i < generators.length; i++) {
      if (!active[i]) continue;
      activeBefore++;
      const next = generators[i]!.next();
      if (next.done) {
        active[i] = false;
      } else {
        barriers++;
      }
    }
    if (barriers > 0 && barriers !== activeBefore) {
      throw compilerFailure("barrier mismatch: not every active thread reached __syncthreads()");
    }
  }
}

function* execStatements(
  statements: readonly CudaLiteStatement[],
  context: ThreadContext,
): BarrierGenerator {
  for (const statement of statements) {
    switch (statement.kind) {
      case "var":
        execVar(statement, context);
        break;
      case "expr":
        if (isBarrier(statement.expression)) {
          yield "barrier";
        } else {
          evalExpression(statement.expression, context);
        }
        break;
      case "if":
        if (truthy(evalExpression(statement.condition, context))) {
          yield* execStatements(statement.consequent, context);
        } else if (statement.alternate) {
          yield* execStatements(statement.alternate, context);
        }
        break;
      case "for":
        if (statement.init) {
          if (statement.init.kind === "var") execVar(statement.init, context);
          else evalExpression(statement.init, context);
        }
        while (statement.condition ? truthy(evalExpression(statement.condition, context)) : true) {
          yield* execStatements(statement.body, context);
          if (statement.update) evalExpression(statement.update, context);
        }
        break;
    }
  }
}

function execVar(statement: CudaLiteVarDecl, context: ThreadContext): void {
  if (statement.storage === "shared") return;
  context.locals.set(statement.name, statement.init ? evalExpression(statement.init, context) : 0);
}

function evalExpression(expression: CudaLiteExpression, context: ThreadContext): number {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier":
      return valueAsNumber(readIdentifier(expression.name, context), expression.name);
    case "member": {
      const object = readExpressionObject(expression.object, context);
      return object[expression.property];
    }
    case "index": {
      const lvalue = resolveLValue(expression, context);
      return readLValue(lvalue, context);
    }
    case "unary": {
      if (expression.operator === "&") return 0;
      const value = evalExpression(expression.argument, context);
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      return truthy(value) ? 0 : 1;
    }
    case "binary":
      return evalBinary(expression.operator, expression.left, expression.right, context);
    case "assignment":
      return evalAssignment(expression.operator, expression.left, expression.right, context);
    case "update": {
      const lvalue = resolveLValue(expression.argument, context);
      const current = readLValue(lvalue, context);
      const next = expression.operator === "++" ? current + 1 : current - 1;
      writeLValue(lvalue, next, context);
      return expression.prefix ? next : current;
    }
    case "call":
      return evalCall(expression, context);
  }
}

function readExpressionObject(expression: CudaLiteExpression, context: ThreadContext): Vector3 {
  if (expression.kind !== "identifier") {
    throw compilerFailure("member access only supports CUDA-lite builtin vectors");
  }
  const value = readIdentifier(expression.name, context);
  if (typeof value === "number") throw compilerFailure(`'${expression.name}' is not a vector`);
  return value;
}

function readIdentifier(name: string, context: ThreadContext): LocalValue {
  if (name === "threadIdx") return context.threadIdx;
  if (name === "blockIdx") return context.blockIdx;
  if (name === "blockDim") return context.blockDim;
  if (name === "gridDim") return context.gridDim;
  if (context.locals.has(name)) return context.locals.get(name)!;
  if (Object.prototype.hasOwnProperty.call(context.scalars, name)) return context.scalars[name]!;
  throw compilerFailure(`unknown identifier '${name}'`);
}

function evalBinary(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): number {
  if (operator === "&&") {
    return truthy(evalExpression(leftExpression, context)) && truthy(evalExpression(rightExpression, context)) ? 1 : 0;
  }
  if (operator === "||") {
    return truthy(evalExpression(leftExpression, context)) || truthy(evalExpression(rightExpression, context)) ? 1 : 0;
  }
  const left = evalExpression(leftExpression, context);
  const right = evalExpression(rightExpression, context);
  switch (operator) {
    case "+":
      return left + right;
    case "-":
      return left - right;
    case "*":
      return left * right;
    case "/":
      return left / right;
    case "%":
      return left % right;
    case "<":
      return left < right ? 1 : 0;
    case "<=":
      return left <= right ? 1 : 0;
    case ">":
      return left > right ? 1 : 0;
    case ">=":
      return left >= right ? 1 : 0;
    case "==":
      return left === right ? 1 : 0;
    case "!=":
      return left !== right ? 1 : 0;
    default:
      throw compilerFailure(`unsupported binary operator '${operator}'`);
  }
}

function evalAssignment(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): number {
  const lvalue = resolveLValue(leftExpression, context);
  const right = evalExpression(rightExpression, context);
  const current = operator === "=" ? 0 : readLValue(lvalue, context);
  const value = operator === "="
    ? right
    : operator === "+="
      ? current + right
      : operator === "-="
        ? current - right
        : operator === "*="
          ? current * right
          : current / right;
  writeLValue(lvalue, value, context);
  return value;
}

function evalCall(expression: Extract<CudaLiteExpression, { kind: "call" }>, context: ThreadContext): number {
  const name = expression.kind === "call" && expression.callee.kind === "identifier"
    ? expression.callee.name
    : undefined;
  const args = expression.args.map((arg) => evalExpression(arg, context));
  switch (name) {
    case "__syncthreads":
      return 0;
    case "min":
      return Math.min(...args);
    case "max":
      return Math.max(...args);
    case "sqrtf":
      return Math.sqrt(args[0] ?? 0);
    case "expf":
      return Math.exp(args[0] ?? 0);
    case "logf":
      return Math.log(args[0] ?? 0);
    case "bg_subgroup_add":
      return args[0] ?? 0;
    case "atomicAdd": {
      const first = expression.args[0];
      if (!first || first.kind !== "unary" || first.operator !== "&") {
        throw compilerFailure("atomicAdd expects address-of target");
      }
      const lvalue = resolveLValue(first.argument, context);
      const current = readLValue(lvalue, context);
      writeLValue(lvalue, current + (args[1] ?? 0), context);
      return current;
    }
    default:
      throw compilerFailure(`unsupported call '${name ?? "<expr>"}'`);
  }
}

function resolveLValue(expression: CudaLiteExpression, context: ThreadContext): LValue {
  if (expression.kind === "identifier") return { name: expression.name, space: "local" };
  const chain: number[] = [];
  let cursor: CudaLiteExpression = expression;
  while (cursor.kind === "index") {
    chain.unshift(Math.trunc(evalExpression(cursor.index, context)));
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") throw compilerFailure("unsupported lvalue");
  const shared = context.shared.get(cursor.name);
  if (shared) {
    return { name: cursor.name, space: "shared", index: flattenIndex(shared.dimensions, chain) };
  }
  if (context.buffers.has(cursor.name)) {
    if (chain.length !== 1) throw compilerFailure(`buffer '${cursor.name}' expects one-dimensional indexing`);
    return { name: cursor.name, space: "buffer", index: chain[0]! };
  }
  throw compilerFailure(`unknown lvalue '${cursor.name}'`);
}

function readLValue(lvalue: LValue, context: ThreadContext): number {
  if (lvalue.space === "local") return valueAsNumber(readIdentifier(lvalue.name, context), lvalue.name);
  if (lvalue.index === undefined) throw compilerFailure(`missing index for '${lvalue.name}'`);
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    const ok = lvalue.index >= 0 && lvalue.index < buffer.length;
    const value = ok ? Number(buffer[lvalue.index]!) : 0;
    context.trace.reads.push({ name: lvalue.name, index: lvalue.index, value, ok });
    return value;
  }
  const shared = context.shared.get(lvalue.name);
  if (!shared) throw compilerFailure(`missing shared array '${lvalue.name}'`);
  const ok = lvalue.index >= 0 && lvalue.index < shared.data.length;
  const value = ok ? Number(shared.data[lvalue.index]!) : 0;
  context.trace.sharedReads.push({ name: lvalue.name, index: lvalue.index, value, ok });
  return value;
}

function writeLValue(lvalue: LValue, value: number, context: ThreadContext): void {
  if (lvalue.space === "local") {
    context.locals.set(lvalue.name, value);
    return;
  }
  if (lvalue.index === undefined) throw compilerFailure(`missing index for '${lvalue.name}'`);
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    const ok = lvalue.index >= 0 && lvalue.index < buffer.length;
    if (ok) buffer[lvalue.index] = value;
    context.trace.writes.push({ name: lvalue.name, index: lvalue.index, value, ok });
    return;
  }
  const shared = context.shared.get(lvalue.name);
  if (!shared) throw compilerFailure(`missing shared array '${lvalue.name}'`);
  const ok = lvalue.index >= 0 && lvalue.index < shared.data.length;
  if (ok) shared.data[lvalue.index] = value;
  context.trace.sharedWrites.push({ name: lvalue.name, index: lvalue.index, value, ok });
}

function allocateShared(declarations: readonly CudaLiteVarDecl[]): Map<string, SharedArrayValue> {
  const shared = new Map<string, SharedArrayValue>();
  for (const declaration of declarations) {
    const length = declaration.dimensions.reduce((product, item) => product * item, 1);
    const data = declaration.valueType === "int"
      ? new Int32Array(length)
      : declaration.valueType === "uint"
        ? new Uint32Array(length)
        : declaration.valueType === "half"
          ? new Float16Array(length)
          : new Float32Array(length);
    shared.set(declaration.name, { dimensions: declaration.dimensions, data });
  }
  return shared;
}

function flattenIndex(dimensions: readonly number[], indices: readonly number[]): number {
  if (dimensions.length !== indices.length) {
    throw compilerFailure(`expected ${dimensions.length} indices, got ${indices.length}`);
  }
  let flat = 0;
  for (let i = 0; i < dimensions.length; i++) {
    flat = flat * dimensions[i]! + indices[i]!;
  }
  return flat;
}

function isBarrier(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" &&
    expression.callee.kind === "identifier" &&
    expression.callee.name === "__syncthreads";
}

function cloneBuffers(
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Map<string, WgslTypedArray> {
  const out = new Map<string, WgslTypedArray>();
  for (const [name, buffer] of Object.entries(buffers)) {
    out.set(name, cloneTypedArray(buffer));
  }
  return out;
}

function cloneTypedArray<T extends WgslTypedArray>(value: T): T {
  return value.slice() as T;
}

function vectorFromTuple(value: readonly [number, number, number]): Vector3 {
  return { x: value[0], y: value[1], z: value[2] };
}

function valueAsNumber(value: LocalValue, name: string): number {
  if (typeof value === "number") return value;
  throw compilerFailure(`'${name}' is a vector, not a scalar`);
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

function validateLaunch(launch: KernelLaunch, workgroupSize: readonly [number, number, number]): void {
  for (let axis = 0; axis < 3; axis++) {
    const block = launch.blockDim[axis];
    const grid = launch.gridDim[axis];
    const expected = workgroupSize[axis];
    if (block !== expected) {
      throw compilerFailure("launch.blockDim must match compiled workgroupSize for reference/WebGPU parity");
    }
    if (grid === undefined || !Number.isInteger(grid) || grid <= 0) {
      throw compilerFailure("launch.gridDim values must be positive integers");
    }
  }
}

function validateInputs(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): void {
  for (const param of compiled.ir.params) {
    if (param.pointer) {
      const buffer = input.buffers[param.name];
      if (!buffer) throw compilerFailure(`missing buffer input '${param.name}'`);
      if (param.valueType === "int" && !(buffer instanceof Int32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Int32Array`);
      }
      if (param.valueType === "uint" && !(buffer instanceof Uint32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Uint32Array`);
      }
      if (param.valueType === "float" && !(buffer instanceof Float32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Float32Array`);
      }
      if (param.valueType === "half" && !(buffer instanceof Float16Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Float16Array`);
      }
    } else if (input.scalars?.[param.name] === undefined) {
      throw compilerFailure(`missing scalar input '${param.name}'`);
    }
  }
}

function compilerFailure(message: string): CudaLiteCompilerError {
  return new CudaLiteCompilerError(message, [{
    code: "reference-runtime-error",
    severity: "error",
    message,
    span: { start: 0, end: 0, line: 1, column: 1 },
  }]);
}
