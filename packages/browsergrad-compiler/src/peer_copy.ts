import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { expressionName } from "./analyzer.js";
import {
  evaluateHostNumber,
  evaluatePointerArgument,
  evaluateVectorExpressions,
  isSingleInvocationGuard,
  type HostEvalValue,
} from "./host_eval.js";
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
  if (dst.offset < 0 || src.offset < 0) return undefined;
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
