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
  CudaLiteDeviceGlobal,
  CudaLiteExpression,
  CudaLiteScalarType,
  CudaLiteStatement,
  KernelLaunch,
} from "./types.js";
import { cudaVectorLaneCount, cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";

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

export type CudaPeerCopyOperation =
  | CudaPeerCopyBufferOperation
  | CudaPeerByteCopyBufferOperation
  | CudaPeerFillBufferOperation
  | CudaPeerByteFillBufferOperation;

export interface CudaPeerCopyBufferOperation {
  readonly kind: "copy";
  readonly expression: CudaLiteCallExpression;
  readonly dstRoot: string;
  readonly srcRoot: string;
  readonly dstOffset: number;
  readonly srcOffset: number;
  readonly elementCount: number;
  readonly valueType: "float" | "int" | "uint";
}

export interface CudaPeerByteCopyBufferOperation {
  readonly kind: "copy-bytes";
  readonly expression: CudaLiteCallExpression;
  readonly dstRoot: string;
  readonly srcRoot: string;
  readonly dstByteOffset: number;
  readonly srcByteOffset: number;
  readonly byteCount: number;
}

export interface CudaPeerFillBufferOperation {
  readonly kind: "fill";
  readonly expression: CudaLiteCallExpression;
  readonly dstRoot: string;
  readonly dstOffset: number;
  readonly elementCount: number;
  readonly valueType: "float" | "int" | "uint";
  readonly byteValue: number;
}

export interface CudaPeerByteFillBufferOperation {
  readonly kind: "fill-bytes";
  readonly expression: CudaLiteCallExpression;
  readonly dstRoot: string;
  readonly dstByteOffset: number;
  readonly byteCount: number;
  readonly byteValue: number;
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
  const copyCollection = collectHostPeerCopies(compiled.analysis.kernel.body, input, launch, compiled.analysis.deviceGlobals);
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
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
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
          const operations = createPeerCopyOperations(item.expression, current, input, deviceGlobals);
          if (!operations) {
            const callName = expressionName(item.expression.callee) ?? "runtime copy";
            markUnsafe("arguments-not-host-evaluable", `${callName} arguments must resolve to typed buffer aliases, non-negative integer byte ranges`);
          }
          else {
            out.push(...operations);
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

function createPeerCopyOperations(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): readonly CudaPeerCopyOperation[] | undefined {
  if (isRuntimeSymbolMemsetCall(expression)) {
    const fill = createPeerSymbolFillOperation(expression, env, input, deviceGlobals);
    return fill ? [fill] : undefined;
  }
  if (isRuntimeMemsetCall(expression)) {
    if (isRuntimeMemset2DCall(expression)) return createPeerFill2DOperations(expression, env, input, deviceGlobals);
    const fill = createPeerFillOperation(expression, env, input, deviceGlobals);
    return fill ? [fill] : undefined;
  }
  const copyShape = cudaRuntimeCopyShape(expression);
  if (!copyShape) return undefined;
  if (copyShape.kind === "symbol") return createPeerSymbolCopyOperation(expression, env, input, deviceGlobals, copyShape);
  if (copyShape.kind === "copy2d") return createPeerCopy2DOperations(expression, env, input, deviceGlobals, copyShape);
  const dst = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const srcArg = expression.args[copyShape.srcIndex];
  const countArg = expression.args[copyShape.countIndex];
  const src = srcArg ? evaluatePointerArgument(srcArg, env, input) : undefined;
  const byteCount = countArg ? evaluateHostNumber(countArg, env, input) : undefined;
  if (!dst || !src || byteCount === undefined || byteCount < 0) return undefined;
  if (dst.offset < 0 || src.offset < 0) return undefined;
  const dstBuffer = copyBufferViewFor(input, dst.root, deviceGlobals);
  const srcBuffer = copyBufferViewFor(input, src.root, deviceGlobals);
  if (!dstBuffer || !srcBuffer) return undefined;
  const copyView = copyCompatibleBufferView(dstBuffer, srcBuffer);
  if (!copyView) return undefined;
  const elementSize = copyView.elementSize;
  if (!Number.isInteger(byteCount)) return undefined;
  const dstByteOffset = dst.offset * elementSize;
  const srcByteOffset = src.offset * elementSize;
  const byteLength = Math.trunc(byteCount);
  if (dstByteOffset + byteLength > dstBuffer.elementLength * elementSize || srcByteOffset + byteLength > srcBuffer.elementLength * elementSize) return undefined;
  if (byteLength % elementSize !== 0) {
    return [{
      kind: "copy-bytes",
      expression,
      dstRoot: dst.root,
      srcRoot: src.root,
      dstByteOffset,
      srcByteOffset,
      byteCount: byteLength,
    }];
  }
  const elementCount = byteLength / elementSize;
  return [{
    kind: "copy",
    expression,
    dstRoot: dst.root,
    srcRoot: src.root,
    dstOffset: dst.offset,
    srcOffset: src.offset,
    elementCount,
    valueType: copyView.valueType,
  }];
}

function createPeerSymbolCopyOperation(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
  copyShape: Extract<CudaRuntimeCopyShape, { readonly kind: "symbol" }>,
): readonly CudaPeerCopyOperation[] | undefined {
  const symbol = expression.args[copyShape.symbolIndex] ? evaluatePointerArgument(expression.args[copyShape.symbolIndex]!, env, input) : undefined;
  const pointer = expression.args[copyShape.pointerIndex] ? evaluatePointerArgument(expression.args[copyShape.pointerIndex]!, env, input) : undefined;
  const byteCount = expression.args[copyShape.countIndex] ? evaluateHostNumber(expression.args[copyShape.countIndex]!, env, input) : undefined;
  const offsetBytes = copyShape.offsetIndex === undefined ? 0 : evaluateHostNumber(expression.args[copyShape.offsetIndex]!, env, input);
  if (!symbol || !pointer || byteCount === undefined || offsetBytes === undefined) return undefined;
  if (symbol.offset < 0 || pointer.offset < 0 || byteCount < 0 || offsetBytes < 0) return undefined;
  const symbolBuffer = copyBufferViewFor(input, symbol.root, deviceGlobals);
  const pointerBuffer = copyBufferViewFor(input, pointer.root, deviceGlobals);
  if (!symbolBuffer || !pointerBuffer) return undefined;
  const copyView = copyCompatibleBufferView(symbolBuffer, pointerBuffer);
  if (!copyView) return undefined;
  const elementSize = copyView.elementSize;
  if (!Number.isInteger(byteCount) || !Number.isInteger(offsetBytes)) return undefined;
  const byteLength = Math.trunc(byteCount);
  const symbolByteOffset = (symbol.offset * elementSize) + Math.trunc(offsetBytes);
  const pointerByteOffset = pointer.offset * elementSize;
  const srcRoot = copyShape.direction === "to-symbol" ? pointer.root : symbol.root;
  const dstRoot = copyShape.direction === "to-symbol" ? symbol.root : pointer.root;
  const srcByteOffset = copyShape.direction === "to-symbol" ? pointerByteOffset : symbolByteOffset;
  const dstByteOffset = copyShape.direction === "to-symbol" ? symbolByteOffset : pointerByteOffset;
  const srcBuffer = copyShape.direction === "to-symbol" ? pointerBuffer : symbolBuffer;
  const dstBuffer = copyShape.direction === "to-symbol" ? symbolBuffer : pointerBuffer;
  if (srcByteOffset + byteLength > srcBuffer.elementLength * elementSize || dstByteOffset + byteLength > dstBuffer.elementLength * elementSize) return undefined;
  const aligned = byteLength % elementSize === 0 && srcByteOffset % elementSize === 0 && dstByteOffset % elementSize === 0;
  if (!aligned) {
    return [{
      kind: "copy-bytes",
      expression,
      dstRoot,
      srcRoot,
      dstByteOffset,
      srcByteOffset,
      byteCount: byteLength,
    }];
  }
  const elementCount = byteLength / elementSize;
  const srcOffset = srcByteOffset / elementSize;
  const dstOffset = dstByteOffset / elementSize;
  return [{
    kind: "copy",
    expression,
    dstRoot,
    srcRoot,
    dstOffset,
    srcOffset,
    elementCount,
    valueType: copyView.valueType,
  }];
}

function createPeerCopy2DOperations(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
  copyShape: Extract<CudaRuntimeCopyShape, { readonly kind: "copy2d" }>,
): readonly CudaPeerCopyOperation[] | undefined {
  const dst = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const dstPitchBytes = expression.args[1] ? evaluateHostNumber(expression.args[1], env, input) : undefined;
  const src = expression.args[copyShape.srcIndex] ? evaluatePointerArgument(expression.args[copyShape.srcIndex]!, env, input) : undefined;
  const srcPitchBytes = expression.args[3] ? evaluateHostNumber(expression.args[3], env, input) : undefined;
  const rowBytes = expression.args[4] ? evaluateHostNumber(expression.args[4], env, input) : undefined;
  const rows = expression.args[5] ? evaluateHostNumber(expression.args[5], env, input) : undefined;
  if (!dst || !src || dstPitchBytes === undefined || srcPitchBytes === undefined || rowBytes === undefined || rows === undefined) return undefined;
  if (dst.offset < 0 || src.offset < 0 || dstPitchBytes < 0 || srcPitchBytes < 0 || rowBytes < 0 || rows < 0) return undefined;
  const dstBuffer = copyBufferViewFor(input, dst.root, deviceGlobals);
  const srcBuffer = copyBufferViewFor(input, src.root, deviceGlobals);
  if (!dstBuffer || !srcBuffer) return undefined;
  const copyView = copyCompatibleBufferView(dstBuffer, srcBuffer);
  if (!copyView) return undefined;
  const elementSize = copyView.elementSize;
  const byteValues = [dstPitchBytes, srcPitchBytes, rowBytes];
  if (byteValues.some((value) => !Number.isInteger(value))) return undefined;
  if (!Number.isInteger(rows)) return undefined;
  const rowCount = Math.trunc(rows);
  const alignedRows = byteValues.every((value) => Math.trunc(value) % elementSize === 0);
  const dstByteLength = dstBuffer.elementLength * elementSize;
  const srcByteLength = srcBuffer.elementLength * elementSize;
  const dstPitch = Math.trunc(dstPitchBytes) / elementSize;
  const srcPitch = Math.trunc(srcPitchBytes) / elementSize;
  const elementCount = Math.trunc(rowBytes) / elementSize;
  const out: CudaPeerCopyOperation[] = [];
  if (rowCount === 0) {
    return [{
      kind: "copy",
      expression,
      dstRoot: dst.root,
      srcRoot: src.root,
      dstOffset: dst.offset,
      srcOffset: src.offset,
      elementCount: 0,
      valueType: copyView.valueType,
    }];
  }
  for (let row = 0; row < rowCount; row++) {
    if (alignedRows) {
      const dstOffset = dst.offset + row * dstPitch;
      const srcOffset = src.offset + row * srcPitch;
      if (dstOffset + elementCount > dstBuffer.elementLength || srcOffset + elementCount > srcBuffer.elementLength) return undefined;
      out.push({
        kind: "copy",
        expression,
        dstRoot: dst.root,
        srcRoot: src.root,
        dstOffset,
        srcOffset,
        elementCount,
        valueType: copyView.valueType,
      });
    } else {
      const dstByteOffset = (dst.offset * elementSize) + row * Math.trunc(dstPitchBytes);
      const srcByteOffset = (src.offset * elementSize) + row * Math.trunc(srcPitchBytes);
      const byteCount = Math.trunc(rowBytes);
      if (dstByteOffset + byteCount > dstByteLength || srcByteOffset + byteCount > srcByteLength) return undefined;
      out.push({
        kind: "copy-bytes",
        expression,
        dstRoot: dst.root,
        srcRoot: src.root,
        dstByteOffset,
        srcByteOffset,
        byteCount,
      });
    }
  }
  return out;
}

function createPeerFillOperation(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): CudaPeerFillBufferOperation | CudaPeerByteFillBufferOperation | undefined {
  const dst = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const value = expression.args[1] ? evaluateHostNumber(expression.args[1], env, input) : undefined;
  const count = expression.args[2] ? evaluateHostNumber(expression.args[2], env, input) : undefined;
  if (!dst || value === undefined || count === undefined || count < 0) return undefined;
  if (dst.offset < 0) return undefined;
  const dstBuffer = copyBufferViewFor(input, dst.root, deviceGlobals);
  if (!dstBuffer) return undefined;
  const elementSize = dstBuffer.elementSize;
  const byteCount = Math.trunc(count);
  const byteOffset = dst.offset * elementSize;
  const byteLength = dstBuffer.elementLength * elementSize;
  if (!Number.isInteger(count) || byteOffset + byteCount > byteLength) return undefined;
  if (byteCount % elementSize !== 0) {
    return {
      kind: "fill-bytes",
      expression,
      dstRoot: dst.root,
      dstByteOffset: byteOffset,
      byteCount,
      byteValue: Math.trunc(value) & 0xff,
    };
  }
  const elementCount = byteCount / elementSize;
  return {
    kind: "fill",
    expression,
    dstRoot: dst.root,
    dstOffset: dst.offset,
    elementCount,
    valueType: dstBuffer.valueType,
    byteValue: Math.trunc(value) & 0xff,
  };
}

function createPeerSymbolFillOperation(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): CudaPeerFillBufferOperation | CudaPeerByteFillBufferOperation | undefined {
  const symbol = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const value = expression.args[1] ? evaluateHostNumber(expression.args[1], env, input) : undefined;
  const count = expression.args[2] ? evaluateHostNumber(expression.args[2], env, input) : undefined;
  const offsetBytes = expression.args[3] ? evaluateHostNumber(expression.args[3], env, input) : 0;
  if (!symbol || value === undefined || count === undefined || offsetBytes === undefined) return undefined;
  if (symbol.offset < 0 || count < 0 || offsetBytes < 0) return undefined;
  const symbolBuffer = copyBufferViewFor(input, symbol.root, deviceGlobals);
  if (!symbolBuffer) return undefined;
  const elementSize = symbolBuffer.elementSize;
  if (!Number.isInteger(count) || !Number.isInteger(offsetBytes)) return undefined;
  const byteCount = Math.trunc(count);
  const byteOffset = (symbol.offset * elementSize) + Math.trunc(offsetBytes);
  const byteLength = symbolBuffer.elementLength * elementSize;
  if (byteOffset + byteCount > byteLength) return undefined;
  const byteValue = Math.trunc(value) & 0xff;
  if (byteCount % elementSize !== 0 || byteOffset % elementSize !== 0) {
    return {
      kind: "fill-bytes",
      expression,
      dstRoot: symbol.root,
      dstByteOffset: byteOffset,
      byteCount,
      byteValue,
    };
  }
  return {
    kind: "fill",
    expression,
    dstRoot: symbol.root,
    dstOffset: byteOffset / elementSize,
    elementCount: byteCount / elementSize,
    valueType: symbolBuffer.valueType,
    byteValue,
  };
}

function createPeerFill2DOperations(
  expression: CudaLiteCallExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): readonly CudaPeerCopyOperation[] | undefined {
  const dst = expression.args[0] ? evaluatePointerArgument(expression.args[0], env, input) : undefined;
  const pitchBytes = expression.args[1] ? evaluateHostNumber(expression.args[1], env, input) : undefined;
  const value = expression.args[2] ? evaluateHostNumber(expression.args[2], env, input) : undefined;
  const rowBytes = expression.args[3] ? evaluateHostNumber(expression.args[3], env, input) : undefined;
  const rows = expression.args[4] ? evaluateHostNumber(expression.args[4], env, input) : undefined;
  if (!dst || pitchBytes === undefined || value === undefined || rowBytes === undefined || rows === undefined) return undefined;
  if (dst.offset < 0 || pitchBytes < 0 || rowBytes < 0 || rows < 0) return undefined;
  const dstBuffer = copyBufferViewFor(input, dst.root, deviceGlobals);
  if (!dstBuffer) return undefined;
  const elementSize = dstBuffer.elementSize;
  const byteValues = [pitchBytes, rowBytes];
  if (byteValues.some((byteValue) => !Number.isInteger(byteValue))) return undefined;
  if (!Number.isInteger(rows)) return undefined;
  const byteLength = dstBuffer.elementLength * elementSize;
  const alignedRows = byteValues.every((byteValue) => Math.trunc(byteValue) % elementSize === 0);
  const pitch = Math.trunc(pitchBytes) / elementSize;
  const elementCount = Math.trunc(rowBytes) / elementSize;
  const rowCount = Math.trunc(rows);
  const out: CudaPeerCopyOperation[] = [];
  if (rowCount === 0) {
    return [{
      kind: "fill",
      expression,
      dstRoot: dst.root,
      dstOffset: dst.offset,
      elementCount: 0,
      valueType: dstBuffer.valueType,
      byteValue: Math.trunc(value) & 0xff,
    }];
  }
  for (let row = 0; row < rowCount; row++) {
    if (alignedRows) {
      const dstOffset = dst.offset + row * pitch;
      if (dstOffset + elementCount > dstBuffer.elementLength) return undefined;
      out.push({
        kind: "fill",
        expression,
        dstRoot: dst.root,
        dstOffset,
        elementCount,
        valueType: dstBuffer.valueType,
        byteValue: Math.trunc(value) & 0xff,
      });
    } else {
      const dstByteOffset = (dst.offset * elementSize) + row * Math.trunc(pitchBytes);
      const byteCount = Math.trunc(rowBytes);
      if (dstByteOffset + byteCount > byteLength) return undefined;
      out.push({
        kind: "fill-bytes",
        expression,
        dstRoot: dst.root,
        dstByteOffset,
        byteCount,
        byteValue: Math.trunc(value) & 0xff,
      });
    }
  }
  return out;
}

function copyBufferViewFor(input: CompiledKernelInput, name: string, deviceGlobals: readonly CudaLiteDeviceGlobal[]): CopyBufferView | undefined {
  const typed = input.buffers[name];
  const resident = input.residentBuffers?.[name];
  const constant = input.constants?.[name];
  const deviceGlobal = input.deviceGlobals?.[name];
  if (typed && resident) return undefined;
  if (typed) return copyTypedArrayView(typed);
  if (deviceGlobal) return copyTypedArrayView(deviceGlobal);
  if (constant && typeof constant !== "number") return copyTypedArrayView(constant);
  if (resident) return copyResidentBufferView(resident);
  const global = deviceGlobals.find((item) => item.name === name);
  if (global) return copyDeviceGlobalView(global);
  return undefined;
}

function copyCompatibleBufferView(
  dst: CopyBufferView | undefined,
  src: CopyBufferView | undefined,
): Pick<CopyBufferView, "elementSize" | "valueType"> | undefined {
  if (!dst || !src || dst.elementSize !== src.elementSize) return undefined;
  if (dst.valueType === src.valueType) return { elementSize: dst.elementSize, valueType: dst.valueType };
  if (dst.elementSize === Uint32Array.BYTES_PER_ELEMENT) return { elementSize: dst.elementSize, valueType: "uint" };
  return undefined;
}

function copyTypedArrayView(buffer: WgslTypedArray): CopyBufferView | undefined {
  if (buffer instanceof Float32Array) return { valueType: "float", elementSize: Float32Array.BYTES_PER_ELEMENT, elementLength: buffer.length };
  if (buffer instanceof Int32Array) return { valueType: "int", elementSize: Int32Array.BYTES_PER_ELEMENT, elementLength: buffer.length };
  if (buffer instanceof Uint32Array) return { valueType: "uint", elementSize: Uint32Array.BYTES_PER_ELEMENT, elementLength: buffer.length };
  return undefined;
}

function copyDeviceGlobalView(global: CudaLiteDeviceGlobal): CopyBufferView | undefined {
  const valueType = copyValueTypeForCuda(global.valueType);
  if (!valueType) return undefined;
  const lanes = isCudaVectorType(global.valueType) ? cudaVectorLaneCount(global.valueType) : 1;
  const elements = global.dimensions.length === 0
    ? 1
    : global.dimensions.reduce((product, dimension) => product * dimension, 1);
  return { valueType, elementSize: 4, elementLength: elements * lanes };
}

function copyValueTypeForCuda(valueType: CudaLiteScalarType): "float" | "int" | "uint" | undefined {
  const scalar = cudaVectorScalarType(valueType) ?? valueType;
  if (scalar === "float" || scalar === "double") return "float";
  if (scalar === "int") return "int";
  if (scalar === "uint" || scalar === "uchar" || scalar === "bool" || scalar === "voidptr") return "uint";
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
  return expression.kind === "call" && (cudaRuntimeCopyShape(expression) !== undefined || isRuntimeMemsetCall(expression) || isRuntimeSymbolMemsetCall(expression));
}

function isRuntimeMemsetCall(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "cudaMemset" || name === "cudaMemsetAsync" || isRuntimeMemset2DCall(expression);
}

function isRuntimeMemset2DCall(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "cudaMemset2D" || name === "cudaMemset2DAsync";
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
  if (name !== undefined && isRuntimeQueryWriteCall(name)) return false;
  return name === "cudaDeviceSynchronize" ||
    name === "cudaCtxResetPersistingL2Cache" ||
    name === "cudaDeviceReset" ||
    name === "cudaThreadExit" ||
    name === "cudaThreadSynchronize" ||
    name === "cudaDeviceGetAttribute" ||
    name === "cudaDeviceGetLimit" ||
    name === "cudaThreadGetLimit" ||
    name === "cudaDeviceSetLimit" ||
    name === "cudaThreadSetLimit" ||
    name === "cudaDeviceCanAccessPeer" ||
    name === "cudaDeviceEnablePeerAccess" ||
    name === "cudaDeviceDisablePeerAccess" ||
    name === "cudaGetDeviceFlags" ||
    name === "cudaSetDeviceFlags" ||
    name === "cudaMemGetInfo" ||
    name === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
    name === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
    name === "cudaOccupancyMaxPotentialBlockSize" ||
    name === "cudaOccupancyMaxPotentialBlockSizeWithFlags" ||
    name === "cudaOccupancyAvailableDynamicSMemPerBlock" ||
    name === "cudaDeviceGetCacheConfig" ||
    name === "cudaDeviceSetCacheConfig" ||
    name === "cudaDeviceGetSharedMemConfig" ||
    name === "cudaThreadGetCacheConfig" ||
    name === "cudaDeviceSetSharedMemConfig" ||
    name === "cudaThreadSetCacheConfig" ||
    name === "cudaThreadExchangeStreamCaptureMode" ||
    name === "cudaDeviceGetStreamPriorityRange" ||
    name === "cudaFree" ||
    name === "cudaFreeAsync" ||
    name === "cudaMemAdvise" ||
    name === "cudaMemPrefetchAsync" ||
    name === "cudaStreamAttachMemAsync" ||
    name === "cudaStreamCreate" ||
    name === "cudaStreamCreateWithFlags" ||
    name === "cudaStreamCreateWithPriority" ||
    name === "cudaStreamDestroy" ||
    name === "cudaStreamGetDevice" ||
    name === "cudaStreamGetFlags" ||
    name === "cudaStreamGetId" ||
    name === "cudaStreamGetPriority" ||
    name === "cudaStreamIsCapturing" ||
    name === "cudaStreamGetCaptureInfo" ||
    name === "cudaStreamGetCaptureInfo_v2" ||
    name === "cudaStreamBeginCapture" ||
    name === "cudaStreamEndCapture" ||
    name === "cudaStreamUpdateCaptureDependencies" ||
    name === "cudaGraphCreate" ||
    name === "cudaGraphInstantiate" ||
    name === "cudaGraphInstantiateWithFlags" ||
    name === "cudaGraphUpload" ||
    name === "cudaGraphExecUpdate" ||
    name === "cudaGraphDestroy" ||
    name === "cudaGraphExecDestroy" ||
    name === "cudaStreamQuery" ||
    name === "cudaStreamSynchronize" ||
    name === "cudaStreamWaitEvent" ||
    name === "cudaSetDevice" ||
    name === "cudaGetDevice" ||
    name === "cudaGetDeviceCount" ||
    name === "cudaRuntimeGetVersion" ||
    name === "cudaDriverGetVersion" ||
    name === "cudaFuncSetAttribute" ||
    name === "cudaFuncSetCacheConfig" ||
    name === "cudaFuncSetSharedMemConfig" ||
    name === "cudaGetLastError" ||
    name === "cudaPeekAtLastError" ||
    name === "cudaProfilerStart" ||
    name === "cudaProfilerStop" ||
    name === "cudaEventCreate" ||
    name === "cudaEventCreateWithFlags" ||
    name === "cudaEventDestroy" ||
    name === "cudaEventQuery" ||
    name === "cudaEventRecord" ||
    name === "cudaEventRecordWithFlags" ||
    name === "cudaEventSynchronize" ||
    name === "printf";
}

function isRuntimeQueryWriteCall(name: string): boolean {
  return name === "cudaGetDevice" ||
    name === "cudaGetDeviceCount" ||
    name === "cudaDeviceGetAttribute" ||
    name === "cudaDeviceGetLimit" ||
    name === "cudaThreadGetLimit" ||
    name === "cudaDeviceCanAccessPeer" ||
    name === "cudaGetDeviceFlags" ||
    name === "cudaMemGetInfo" ||
    name === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
    name === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
    name === "cudaOccupancyMaxPotentialBlockSize" ||
    name === "cudaOccupancyMaxPotentialBlockSizeWithFlags" ||
    name === "cudaOccupancyAvailableDynamicSMemPerBlock" ||
    name === "cudaDeviceGetCacheConfig" ||
    name === "cudaDeviceGetSharedMemConfig" ||
    name === "cudaThreadGetCacheConfig" ||
    name === "cudaThreadExchangeStreamCaptureMode" ||
    name === "cudaDeviceGetStreamPriorityRange" ||
    name === "cudaStreamCreate" ||
    name === "cudaStreamCreateWithFlags" ||
    name === "cudaStreamCreateWithPriority" ||
    name === "cudaStreamGetDevice" ||
    name === "cudaStreamGetFlags" ||
    name === "cudaStreamGetId" ||
    name === "cudaStreamGetPriority" ||
    name === "cudaStreamIsCapturing" ||
    name === "cudaStreamGetCaptureInfo" ||
    name === "cudaStreamGetCaptureInfo_v2" ||
    name === "cudaStreamEndCapture" ||
    name === "cudaGraphCreate" ||
    name === "cudaGraphInstantiate" ||
    name === "cudaGraphInstantiateWithFlags" ||
    name === "cudaGraphExecUpdate" ||
    name === "cudaEventCreate" ||
    name === "cudaEventCreateWithFlags" ||
    name === "cudaRuntimeGetVersion" ||
    name === "cudaDriverGetVersion" ||
    name === "cudaEventElapsedTime";
}

type CudaRuntimeCopyShape =
  | { readonly kind: "copy1d"; readonly srcIndex: number; readonly countIndex: number }
  | { readonly kind: "copy2d"; readonly srcIndex: number }
  | { readonly kind: "symbol"; readonly direction: "to-symbol" | "from-symbol"; readonly symbolIndex: number; readonly pointerIndex: number; readonly srcIndex: number; readonly countIndex: number; readonly offsetIndex?: number };

function cudaRuntimeCopyShape(
  expression: CudaLiteCallExpression,
): CudaRuntimeCopyShape | undefined {
  const name = expressionName(expression.callee);
  if (name === "cudaMemcpy" || name === "cudaMemcpyAsync") return { kind: "copy1d", srcIndex: 1, countIndex: 2 };
  if (name === "cudaMemcpy2D" || name === "cudaMemcpy2DAsync") return { kind: "copy2d", srcIndex: 2 };
  if (name === "cudaMemcpyPeer" || name === "cudaMemcpyPeerAsync") return { kind: "copy1d", srcIndex: 2, countIndex: 4 };
  if (name === "cudaMemcpyToSymbol" || name === "cudaMemcpyToSymbolAsync") {
    return { kind: "symbol", direction: "to-symbol", symbolIndex: 0, pointerIndex: 1, srcIndex: 1, countIndex: 2, ...(expression.args[3] ? { offsetIndex: 3 } : {}) };
  }
  if (name === "cudaMemcpyFromSymbol" || name === "cudaMemcpyFromSymbolAsync") {
    return { kind: "symbol", direction: "from-symbol", symbolIndex: 1, pointerIndex: 0, srcIndex: 1, countIndex: 2, ...(expression.args[3] ? { offsetIndex: 3 } : {}) };
  }
  return undefined;
}

function isRuntimeSymbolMemsetCall(expression: CudaLiteCallExpression): boolean {
  const name = expressionName(expression.callee);
  return name === "cudaMemsetToSymbol" || name === "cudaMemsetToSymbolAsync";
}
