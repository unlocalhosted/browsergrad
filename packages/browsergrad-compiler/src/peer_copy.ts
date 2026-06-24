import type { WgslResidentBuffer, WgslTypedArray, WgslValueType } from "@unlocalhosted/browsergrad-kernels";
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
  readonly blocker?: CudaPeerCopyBlocker;
  readonly copies: readonly CudaPeerCopyOperation[];
}

export interface CudaPeerCopyBlocker {
  readonly code: CudaPeerCopyBlockerCode;
  readonly message: string;
}

export type CudaPeerCopyBlockerCode =
  | "no-peer-copy"
  | "mixed-runtime-operations"
  | "no-host-liftable-peer-copy"
  | "unsafe-parent-side-effects"
  | "branch-not-host-evaluable"
  | "parent-not-single-invocation"
  | "arguments-not-host-evaluable";

export interface CudaPeerCopyOperation {
  readonly expression: CudaLiteCallExpression;
  readonly dstRoot: string;
  readonly srcRoot: string;
  readonly dstOffset: number;
  readonly srcOffset: number;
  readonly elementCount: number;
  readonly valueType: "float" | "int" | "uint";
}

export type CudaRuntimeCopyPlan = CudaPeerCopyPlan;
export type CudaRuntimeCopyBlocker = CudaPeerCopyBlocker;
export type CudaRuntimeCopyBlockerCode = CudaPeerCopyBlockerCode;
export type CudaRuntimeCopyOperation = CudaPeerCopyOperation;

interface HostPeerCopyCollection {
  readonly copies: readonly CudaPeerCopyOperation[];
  readonly reason?: string;
  readonly blocker?: CudaPeerCopyBlocker;
}

interface CopyBufferView {
  readonly valueType: "float" | "int" | "uint";
  readonly elementSize: number;
  readonly elementLength: number;
}

export function createCudaPeerCopyPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaPeerCopyPlan {
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (!runtimePlan.operations.some((operation) => operation.kind === "runtime-copy")) {
    return unsupported("no-peer-copy", "no peer-copy operation found");
  }
  if (!runtimePlan.operations.every((operation) => operation.kind === "runtime-copy" || operation.kind === "device-sync")) {
    return unsupported("mixed-runtime-operations", "runtime operations besides peer-copy/device sync require reference runtime");
  }
  const copyCollection = collectHostPeerCopies(compiled.ir.body, input, launch);
  const copies = copyCollection.copies;
  if (copyCollection.blocker) return unsupportedWithBlocker(copyCollection.blocker);
  if (copies.length === 0) return unsupported("no-host-liftable-peer-copy", copyCollection.reason ?? "no host-liftable peer-copy operations");
  return { supported: true, copies };
}

export function createCudaRuntimeCopyPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaRuntimeCopyPlan {
  return createCudaPeerCopyPlan(compiled, input, launch);
}

function unsupported(code: CudaPeerCopyBlockerCode, message: string): CudaPeerCopyPlan {
  return unsupportedWithBlocker({ code, message });
}

function unsupportedWithBlocker(blocker: CudaPeerCopyBlocker): CudaPeerCopyPlan {
  return { supported: false, reason: blocker.message, blocker, copies: [] };
}

function collectHostPeerCopies(
  statements: readonly CudaLiteStatement[],
  input: CompiledKernelInput,
  launch: KernelLaunch,
): HostPeerCopyCollection {
  const out: CudaPeerCopyOperation[] = [];
  const parentHasSingleInvocation = launch.gridDim.every((axis) => axis === 1) && launch.blockDim.every((axis) => axis === 1);
  let unsafeBlocker: CudaPeerCopyBlocker | undefined;
  const markUnsafe = (code: CudaPeerCopyBlockerCode, message: string): void => {
    unsafeBlocker ??= { code, message };
  };

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
          if (out.length > before && hasParentSideEffectsAfterPeerCopy(items.slice(index + 1))) {
            markUnsafe("unsafe-parent-side-effects", "parent side effects after peer copy cannot be replayed in host-lifted sequence");
          }
          continue;
        }
        const condition = evaluateHostNumber(item.condition, current, input);
        if (condition === undefined) {
          if (containsPeerCopyCall(item.consequent) || containsPeerCopyCall(item.alternate ?? [])) {
            markUnsafe("branch-not-host-evaluable", "peer-copy branch condition must be host-evaluable or a single-invocation guard");
          }
          return containsPeerCopy;
        }
        containsPeerCopy = visit(condition !== 0 ? item.consequent : item.alternate ?? [], current, singleInvocationGuard) || containsPeerCopy;
        if (out.length > before && hasParentSideEffectsAfterPeerCopy(items.slice(index + 1))) {
          markUnsafe("unsafe-parent-side-effects", "parent side effects after peer copy cannot be replayed in host-lifted sequence");
        }
        continue;
      }
      if (item.kind === "expr" && isPeerCopyCall(item.expression)) {
        if (!(singleInvocationGuard || parentHasSingleInvocation)) {
          markUnsafe("parent-not-single-invocation", "peer copy must be single-invocation guarded or parent launch must be one thread");
        }
        else if (hasParentSideEffectsAfterPeerCopy(items.slice(index + 1))) {
          markUnsafe("unsafe-parent-side-effects", "parent side effects after peer copy cannot be replayed in host-lifted sequence");
        }
        else {
          const operation = createPeerCopyOperation(item.expression, current, input);
          if (!operation) {
            markUnsafe("arguments-not-host-evaluable", "peer-copy arguments must resolve to typed buffer aliases, non-negative offsets, and element-aligned byte count");
          }
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
  return unsafeBlocker ? { copies: [], reason: unsafeBlocker.message, blocker: unsafeBlocker } : { copies: out };
}

function createPeerCopyOperation(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): CudaPeerCopyOperation | undefined {
  const copyShape = cudaRuntimeCopyShape(expression);
  if (!copyShape) return undefined;
  const dst = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const srcArg = expression.args[copyShape.srcIndex];
  const countArg = expression.args[copyShape.countIndex];
  const src = srcArg ? evaluatePointerArgument(srcArg, env, input) : undefined;
  const byteCount = countArg ? evaluateHostNumber(countArg, env, input) : undefined;
  if (!dst || !src || byteCount === undefined || byteCount < 0) return undefined;
  if (dst.offset < 0 || src.offset < 0) return undefined;
  const dstBuffer = copyBufferViewFor(input, dst.root);
  const srcBuffer = copyBufferViewFor(input, src.root);
  if (!dstBuffer || !srcBuffer || dstBuffer.valueType !== srcBuffer.valueType) return undefined;
  const elementSize = dstBuffer.elementSize;
  if (Math.trunc(byteCount) % elementSize !== 0) return undefined;
  const elementCount = Math.trunc(byteCount) / elementSize;
  if (dst.offset + elementCount > dstBuffer.elementLength || src.offset + elementCount > srcBuffer.elementLength) return undefined;
  return {
    expression,
    dstRoot: dst.root,
    srcRoot: src.root,
    dstOffset: dst.offset,
    srcOffset: src.offset,
    elementCount,
    valueType: dstBuffer.valueType,
  };
}

function copyBufferViewFor(input: CompiledKernelInput, name: string): CopyBufferView | undefined {
  const typed = input.buffers[name];
  const resident = input.residentBuffers?.[name];
  if (typed && resident) return undefined;
  if (typed) return copyTypedArrayView(typed);
  if (resident) return copyResidentBufferView(resident);
  return undefined;
}

function copyTypedArrayView(buffer: WgslTypedArray): CopyBufferView | undefined {
  if (buffer instanceof Float32Array) return { valueType: "float", elementSize: Float32Array.BYTES_PER_ELEMENT, elementLength: buffer.length };
  if (buffer instanceof Int32Array) return { valueType: "int", elementSize: Int32Array.BYTES_PER_ELEMENT, elementLength: buffer.length };
  if (buffer instanceof Uint32Array) return { valueType: "uint", elementSize: Uint32Array.BYTES_PER_ELEMENT, elementLength: buffer.length };
  return undefined;
}

function copyResidentBufferView(buffer: WgslResidentBuffer): CopyBufferView | undefined {
  const valueType = copyValueTypeForWgsl(buffer.valueType);
  if (!valueType) return undefined;
  const elementSize = elementSizeForWgsl(buffer.valueType);
  return { valueType, elementSize, elementLength: Math.trunc(buffer.byteLength / elementSize) };
}

function copyValueTypeForWgsl(valueType: WgslValueType): "float" | "int" | "uint" | undefined {
  if (valueType === "f32") return "float";
  if (valueType === "i32") return "int";
  if (valueType === "u32") return "uint";
  return undefined;
}

function elementSizeForWgsl(valueType: WgslValueType): number {
  return valueType === "f16" ? 2 : 4;
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
      case "while":
      case "do-while":
      case "kernel-launch":
      case "return":
      case "continue":
      case "break":
        return true;
    }
  }
  return false;
}

function isPeerCopyCall(expression: CudaLiteExpression): expression is CudaLiteCallExpression {
  return expression.kind === "call" && cudaRuntimeCopyShape(expression) !== undefined;
}

function containsPeerCopyCall(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && isPeerCopyCall(statement.expression)) return true;
    if (statement.kind === "if" && (containsPeerCopyCall(statement.consequent) || containsPeerCopyCall(statement.alternate ?? []))) return true;
    if ((statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while" || statement.kind === "block") && containsPeerCopyCall(statement.body)) return true;
  }
  return false;
}

function isHostNoopExpression(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "cudaDeviceSynchronize" ||
    name === "cudaStreamCreate" ||
    name === "cudaStreamCreateWithFlags" ||
    name === "cudaStreamDestroy" ||
    name === "cudaStreamSynchronize" ||
    name === "cudaEventCreate" ||
    name === "cudaEventCreateWithFlags" ||
    name === "cudaEventDestroy" ||
    name === "cudaEventRecord" ||
    name === "cudaEventSynchronize" ||
    name === "printf";
}

function cudaRuntimeCopyShape(
  expression: CudaLiteCallExpression,
): { readonly srcIndex: number; readonly countIndex: number } | undefined {
  const name = expressionName(expression.callee);
  if (name === "cudaMemcpy" || name === "cudaMemcpyAsync") return { srcIndex: 1, countIndex: 2 };
  if (name === "cudaMemcpyPeerAsync") return { srcIndex: 2, countIndex: 4 };
  return undefined;
}
