import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { expressionName } from "./analyzer.js";
import { createCudaRuntimePlan } from "./runtime_plan.js";
import type {
  CompiledCudaLiteKernel,
  CompiledKernelInput,
  CudaLiteCallExpression,
  CudaLiteExpression,
  CudaLiteStatement,
  KernelLaunch,
} from "./types.js";

export interface CudaPeerCopyPlan {
  readonly supported: boolean;
  readonly reason?: string;
  readonly copies: readonly CudaPeerCopyOperation[];
}

export interface CudaPeerCopyOperation {
  readonly expression: CudaLiteCallExpression;
  readonly dstRoot: string;
  readonly srcRoot: string;
  readonly dstOffset: number;
  readonly srcOffset: number;
  readonly elementCount: number;
  readonly valueType: "float" | "int" | "uint";
}

type HostEvalValue = number | readonly [number, number, number];

export function createCudaPeerCopyPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaPeerCopyPlan {
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (!runtimePlan.operations.some((operation) => operation.kind === "peer-copy")) {
    return unsupported("no peer-copy operation found");
  }
  if (!runtimePlan.operations.every((operation) => operation.kind === "peer-copy" || operation.kind === "device-sync")) {
    return unsupported("runtime operations besides peer-copy/device sync require reference runtime");
  }
  const copies = collectHostPeerCopies(compiled.ir.body, input, launch);
  if (copies.length === 0) return unsupported("no host-liftable peer-copy operations");
  return { supported: true, copies };
}

function unsupported(reason: string): CudaPeerCopyPlan {
  return { supported: false, reason, copies: [] };
}

function collectHostPeerCopies(
  statements: readonly CudaLiteStatement[],
  input: CompiledKernelInput,
  launch: KernelLaunch,
): readonly CudaPeerCopyOperation[] {
  const out: CudaPeerCopyOperation[] = [];
  const parentHasSingleInvocation = launch.gridDim.every((axis) => axis === 1) && launch.blockDim.every((axis) => axis === 1);
  let unsafe = false;

  const visit = (
    items: readonly CudaLiteStatement[],
    env: ReadonlyMap<string, HostEvalValue>,
    singleInvocationGuard: boolean,
  ): boolean => {
    let current = new Map(env);
    let containsPeerCopy = false;
    for (let index = 0; index < items.length; index++) {
      const item = items[index]!;
      if (item.kind === "dim3") {
        const value = evaluateVectorExpressions(item.args, current, input);
        if (value) current.set(item.name, value);
        continue;
      }
      if (item.kind === "var" && !item.pointer && item.storage === "local" && item.init) {
        const value = evaluateHostNumber(item.init, current, input);
        if (value !== undefined) current.set(item.name, value);
        continue;
      }
      if (item.kind === "if") {
        const before = out.length;
        if (isSingleInvocationGuard(item.condition)) {
          containsPeerCopy = visit(item.consequent, current, true) || containsPeerCopy;
          if (out.length > before && hasParentSideEffectsAfterPeerCopy(items.slice(index + 1))) unsafe = true;
          continue;
        }
        const condition = evaluateHostNumber(item.condition, current, input);
        if (condition === undefined) return containsPeerCopy;
        containsPeerCopy = visit(condition !== 0 ? item.consequent : item.alternate ?? [], current, singleInvocationGuard) || containsPeerCopy;
        if (out.length > before && hasParentSideEffectsAfterPeerCopy(items.slice(index + 1))) unsafe = true;
        continue;
      }
      if (item.kind === "expr" && isPeerCopyCall(item.expression)) {
        if (!(singleInvocationGuard || parentHasSingleInvocation)) unsafe = true;
        else if (hasParentSideEffectsAfterPeerCopy(items.slice(index + 1))) unsafe = true;
        else {
          const operation = createPeerCopyOperation(item.expression, current, input);
          if (!operation) unsafe = true;
          else {
            out.push(operation);
            containsPeerCopy = true;
          }
        }
      }
    }
    return containsPeerCopy;
  };

  visit(statements, new Map(), parentHasSingleInvocation);
  return unsafe ? [] : out;
}

function createPeerCopyOperation(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): CudaPeerCopyOperation | undefined {
  const dst = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const src = expression.args[2] ? evaluatePointerArgument(expression.args[2], env, input) : undefined;
  const byteCount = expression.args[4] ? evaluateHostNumber(expression.args[4], env, input) : undefined;
  if (!dst || !src || byteCount === undefined || byteCount < 0) return undefined;
  const dstBuffer = input.buffers[dst.root];
  const srcBuffer = input.buffers[src.root];
  if (!dstBuffer || !srcBuffer) return undefined;
  const valueType = compatibleCopyValueType(dstBuffer, srcBuffer);
  if (!valueType) return undefined;
  const elementSize = dstBuffer.BYTES_PER_ELEMENT;
  if (Math.trunc(byteCount) % elementSize !== 0) return undefined;
  return {
    expression,
    dstRoot: dst.root,
    srcRoot: src.root,
    dstOffset: dst.offset,
    srcOffset: src.offset,
    elementCount: Math.trunc(byteCount) / elementSize,
    valueType,
  };
}

function compatibleCopyValueType(
  dst: WgslTypedArray,
  src: WgslTypedArray,
): "float" | "int" | "uint" | undefined {
  if (dst.constructor !== src.constructor) return undefined;
  if (dst instanceof Float32Array) return "float";
  if (dst instanceof Int32Array) return "int";
  if (dst instanceof Uint32Array) return "uint";
  return undefined;
}

function evaluatePointerArgument(
  expression: CudaLiteExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): { readonly root: string; readonly offset: number } | undefined {
  if (expression.kind === "identifier") return { root: expression.name, offset: 0 };
  if (expression.kind !== "binary" || (expression.operator !== "+" && expression.operator !== "-")) return undefined;
  if (expression.left.kind !== "identifier") return undefined;
  const offset = evaluateHostNumber(expression.right, env, input);
  if (offset === undefined) return undefined;
  const value = Math.trunc(offset) * (expression.operator === "-" ? -1 : 1);
  if (value < 0) return undefined;
  return { root: expression.left.name, offset: value };
}

function hasParentSideEffectsAfterPeerCopy(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    switch (statement.kind) {
      case "dim3":
      case "cooperative-group":
        continue;
      case "expr":
        if (isHostNoopExpression(statement.expression) || isPeerCopyCall(statement.expression)) continue;
        return true;
      case "if":
        if (hasParentSideEffectsAfterPeerCopy(statement.consequent) || hasParentSideEffectsAfterPeerCopy(statement.alternate ?? [])) return true;
        continue;
      case "var":
        if (statement.storage === "local" && !statement.pointer) continue;
        return true;
      case "asm":
      case "for":
      case "kernel-launch":
      case "return":
      case "continue":
        return true;
    }
  }
  return false;
}

function isPeerCopyCall(expression: CudaLiteExpression): expression is CudaLiteCallExpression {
  return expression.kind === "call" && expressionName(expression.callee) === "cudaMemcpyPeerAsync";
}

function isHostNoopExpression(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "cudaDeviceSynchronize" || name === "printf";
}

function evaluateVectorExpressions(
  expressions: readonly CudaLiteExpression[],
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): readonly [number, number, number] | undefined {
  const x = expressions[0] ? evaluateHostNumber(expressions[0], env, input) : 1;
  const y = expressions[1] ? evaluateHostNumber(expressions[1], env, input) : 1;
  const z = expressions[2] ? evaluateHostNumber(expressions[2], env, input) : 1;
  if (x === undefined || y === undefined || z === undefined) return undefined;
  return [Math.max(1, Math.trunc(x)), Math.max(1, Math.trunc(y)), Math.max(1, Math.trunc(z))];
}

function evaluateHostNumber(
  expression: CudaLiteExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): number | undefined {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier": {
      const local = env.get(expression.name);
      if (typeof local === "number") return local;
      return input.scalars?.[expression.name];
    }
    case "cast":
      return evaluateHostNumber(expression.expression, env, input);
    case "call":
      if (expressionName(expression.callee) === "sizeof" && expression.args[0]?.kind === "identifier") {
        return sizeofType(expression.args[0].name);
      }
      return undefined;
    case "member": {
      if (expression.object.kind !== "identifier") return undefined;
      const vector = env.get(expression.object.name);
      if (!isHostVector(vector)) return undefined;
      return expression.property === "x" ? vector[0] : expression.property === "y" ? vector[1] : expression.property === "z" ? vector[2] : undefined;
    }
    case "unary": {
      const value = evaluateHostNumber(expression.argument, env, input);
      if (value === undefined) return undefined;
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      if (expression.operator === "!") return value === 0 ? 1 : 0;
      return undefined;
    }
    case "binary": {
      const left = evaluateHostNumber(expression.left, env, input);
      const right = evaluateHostNumber(expression.right, env, input);
      if (left === undefined || right === undefined) return undefined;
      return evaluateHostBinary(expression.operator, left, right);
    }
    case "conditional": {
      const condition = evaluateHostNumber(expression.condition, env, input);
      if (condition === undefined) return undefined;
      return evaluateHostNumber(condition !== 0 ? expression.consequent : expression.alternate, env, input);
    }
    default:
      return undefined;
  }
}

function isHostVector(value: HostEvalValue | undefined): value is readonly [number, number, number] {
  return Array.isArray(value) && value.length === 3;
}

function evaluateHostBinary(operator: string, left: number, right: number): number | undefined {
  switch (operator) {
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return left / right;
    case "%": return left % right;
    case "<<": return Math.trunc(left) << Math.trunc(right);
    case ">>": return Math.trunc(left) >> Math.trunc(right);
    case "&": return Math.trunc(left) & Math.trunc(right);
    case "^": return Math.trunc(left) ^ Math.trunc(right);
    case "|": return Math.trunc(left) | Math.trunc(right);
    case "<": return left < right ? 1 : 0;
    case "<=": return left <= right ? 1 : 0;
    case ">": return left > right ? 1 : 0;
    case ">=": return left >= right ? 1 : 0;
    case "==": return left === right ? 1 : 0;
    case "!=": return left !== right ? 1 : 0;
    case "&&": return left !== 0 && right !== 0 ? 1 : 0;
    case "||": return left !== 0 || right !== 0 ? 1 : 0;
    default: return undefined;
  }
}

function isSingleInvocationGuard(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "binary") return false;
  const left = threadIdxXGuardSide(expression.left);
  const right = literalGuardSide(expression.right);
  if (left && right !== undefined) return guardAllowsOnlyThreadZero(expression.operator, right);
  const flippedLeft = threadIdxXGuardSide(expression.right);
  const flippedRight = literalGuardSide(expression.left);
  if (flippedLeft && flippedRight !== undefined) return guardAllowsOnlyThreadZero(flipComparison(expression.operator), flippedRight);
  if (expression.operator === "&&") return isSingleInvocationGuard(expression.left) || isSingleInvocationGuard(expression.right);
  return false;
}

function threadIdxXGuardSide(expression: CudaLiteExpression): boolean {
  return expression.kind === "member" &&
    expression.property === "x" &&
    expression.object.kind === "identifier" &&
    expression.object.name === "threadIdx";
}

function literalGuardSide(expression: CudaLiteExpression): number | undefined {
  return expression.kind === "number" ? expression.value : undefined;
}

function guardAllowsOnlyThreadZero(operator: string, literal: number): boolean {
  switch (operator) {
    case "==": return literal === 0;
    case "<": return literal <= 1;
    case "<=": return literal < 1;
    default: return false;
  }
}

function flipComparison(operator: string): string {
  switch (operator) {
    case "<": return ">";
    case "<=": return ">=";
    case ">": return "<";
    case ">=": return "<=";
    default: return operator;
  }
}

function sizeofType(typeName: string): number {
  switch (typeName) {
    case "half": return 2;
    case "float":
    case "int":
    case "uint":
    case "unsigned":
    case "size_t":
    default:
      return 4;
  }
}
