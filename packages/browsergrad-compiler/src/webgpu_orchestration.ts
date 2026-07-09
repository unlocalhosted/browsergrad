import {
  defineWgslKernelProgram,
  float32ToFloat16Bits,
  type WgslKernelProgram,
  type WgslKernelRunInput,
  type WgslKernelSequenceStep,
  type WgslResidentBuffer,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { createCudaHostDynamicLaunchPlan } from "./dynamic_launch.js";
import { CUDA_INTRINSICS } from "./intrinsics.js";
import { createCudaLaunchValidationDiagnostics } from "./launch.js";
import { createCudaPeerCopyPlan, type CudaPeerCopyOperation } from "./peer_copy.js";
import { poolOffsetName } from "./pool_bindings.js";
import { deviceLaunchTreeIsExternallySilent } from "./runtime_elision.js";
import { createCudaGridSyncPhasePlan, createCudaRuntimePlan } from "./runtime_plan.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  cudaWebGpuMemoryPoolDataAliases,
  cudaWebGpuUniformParamDescriptors,
  deviceGlobalBufferInputs,
  memoryPoolBufferInputs,
  memoryPoolStorageMetadata,
  surfaceBufferInputs,
} from "./webgpu_inputs.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type CudaLiteDiagnostic,
  type KernelLaunch,
} from "./types.js";

export type CudaWebGpuExecutionPlanKind =
  | "single-dispatch"
  | "runtime-elided-single-dispatch"
  | "grid-sync-phases"
  | "host-dynamic-launch"
  | "host-copy";

export type CudaWebGpuExecutionMode = "direct" | "host-orchestrated" | "unsupported";

export type CudaWebGpuExecutionBlockerKind =
  | "launch"
  | "grid-sync"
  | "device-launch"
  | "runtime-copy"
  | "runtime";

export interface CudaWebGpuExecutionBlocker {
  readonly kind: CudaWebGpuExecutionBlockerKind;
  readonly code: string;
  readonly message: string;
}

export type CudaWebGpuExecutionPlan =
  | {
      readonly supported: true;
      readonly kind: CudaWebGpuExecutionPlanKind;
      readonly steps: readonly WgslKernelSequenceStep[];
      readonly input: WgslKernelRunInput;
    }
  | {
      readonly supported: false;
      readonly reason: string;
      readonly blockers: readonly CudaWebGpuExecutionBlocker[];
      readonly diagnostics: readonly CudaLiteDiagnostic[];
    };

export interface CudaWebGpuExecutionStatus {
  readonly canRunOnWebGpu: boolean;
  readonly mode: CudaWebGpuExecutionMode;
  readonly kind?: CudaWebGpuExecutionPlanKind;
  readonly requiresHostOrchestration: boolean;
  readonly reason?: string;
  readonly blockers: readonly CudaWebGpuExecutionBlocker[];
  readonly diagnostics: readonly CudaLiteDiagnostic[];
}

export interface CudaWebGpuExecutionPlanOptions {
  readonly compileKernel?: (
    source: string,
    options?: CompileCudaLiteOptions,
  ) => CompiledCudaLiteKernel;
  readonly maxHostExpandedParentInvocations?: number;
  readonly maxHostDynamicLaunchDepth?: number;
  readonly hostDynamicLaunchDepth?: number;
}

const peerCopyProgramCache = new Map<"float" | "int" | "uint", WgslKernelProgram>();
const peerFillProgramCache = new Map<"float" | "int" | "uint", WgslKernelProgram>();
let peerByteCopyProgramCache: WgslKernelProgram | undefined;
let peerByteFillProgramCache: WgslKernelProgram | undefined;
const DEFAULT_MAX_HOST_DYNAMIC_LAUNCH_DEPTH = 8;
const HOST_SIDE_EFFECT_FREE_CALLS = new Set([
  "cudaDeviceSynchronize",
  "cudaCtxResetPersistingL2Cache",
  "cudaDeviceReset",
  "cudaThreadExit",
  "cudaThreadSynchronize",
  "cudaDeviceGetAttribute",
  "cudaDeviceGetLimit",
  "cudaThreadGetLimit",
  "cudaDeviceSetLimit",
  "cudaThreadSetLimit",
  "cudaDeviceCanAccessPeer",
  "cudaDeviceEnablePeerAccess",
  "cudaDeviceDisablePeerAccess",
  "cudaGetDeviceFlags",
  "cudaSetDeviceFlags",
  "cudaMemGetInfo",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cudaOccupancyMaxPotentialBlockSize",
  "cudaOccupancyMaxPotentialBlockSizeWithFlags",
  "cudaDeviceGetCacheConfig",
  "cudaDeviceSetCacheConfig",
  "cudaDeviceGetSharedMemConfig",
  "cudaThreadGetCacheConfig",
  "cudaDeviceSetSharedMemConfig",
  "cudaThreadSetCacheConfig",
  "cudaThreadExchangeStreamCaptureMode",
  "cudaDeviceGetStreamPriorityRange",
  "cudaFree",
  "cudaFreeAsync",
  "cudaMemAdvise",
  "cudaMemPrefetchAsync",
  "cudaStreamAttachMemAsync",
  "cudaStreamCreate",
  "cudaStreamCreateWithFlags",
  "cudaStreamCreateWithPriority",
  "cudaStreamDestroy",
  "cudaStreamGetDevice",
  "cudaStreamGetFlags",
  "cudaStreamGetId",
  "cudaStreamGetPriority",
  "cudaStreamIsCapturing",
  "cudaStreamGetCaptureInfo",
  "cudaStreamBeginCapture",
  "cudaStreamEndCapture",
  "cudaStreamUpdateCaptureDependencies",
  "cudaGraphCreate",
  "cudaGraphInstantiate",
  "cudaGraphInstantiateWithFlags",
  "cudaGraphDestroy",
  "cudaGraphExecDestroy",
  "cudaStreamQuery",
  "cudaStreamSynchronize",
  "cudaStreamWaitEvent",
  "cudaSetDevice",
  "cudaGetDevice",
  "cudaGetDeviceCount",
  "cudaRuntimeGetVersion",
  "cudaDriverGetVersion",
  "cudaFuncSetAttribute",
  "cudaFuncSetCacheConfig",
  "cudaFuncSetSharedMemConfig",
  "cudaGetLastError",
  "cudaPeekAtLastError",
  "cudaProfilerStart",
  "cudaProfilerStop",
  "cudaEventCreate",
  "cudaEventCreateWithFlags",
  "cudaEventDestroy",
  "cudaEventQuery",
  "cudaEventRecord",
  "cudaEventRecordWithFlags",
  "cudaEventSynchronize",
  "deviceAllocate",
  "max",
  "min",
  "printf",
  "sizeof",
  "alignof",
  "__syncthreads",
  "__syncwarp",
  "streamOrderedAllocate",
  ...CUDA_INTRINSICS.map((intrinsic) => intrinsic.name),
]);
const HOST_RUNTIME_QUERY_WRITE_CALLS = new Set([
  "cudaGetDevice",
  "cudaGetDeviceCount",
  "cudaDeviceGetAttribute",
  "cudaDeviceGetLimit",
  "cudaThreadGetLimit",
  "cudaDeviceCanAccessPeer",
  "cudaGetDeviceFlags",
  "cudaMemGetInfo",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cudaOccupancyMaxPotentialBlockSize",
  "cudaOccupancyMaxPotentialBlockSizeWithFlags",
  "cudaDeviceGetCacheConfig",
  "cudaDeviceGetSharedMemConfig",
  "cudaThreadGetCacheConfig",
  "cudaThreadExchangeStreamCaptureMode",
  "cudaDeviceGetStreamPriorityRange",
  "cudaStreamGetDevice",
  "cudaStreamGetFlags",
  "cudaStreamGetId",
  "cudaStreamGetPriority",
  "cudaStreamIsCapturing",
  "cudaStreamGetCaptureInfo",
  "cudaStreamEndCapture",
  "cudaGraphCreate",
  "cudaGraphInstantiate",
  "cudaGraphInstantiateWithFlags",
  "cudaRuntimeGetVersion",
  "cudaDriverGetVersion",
  "cudaEventElapsedTime",
]);

type DynamicChildCompileResult =
  | { readonly compiled: CompiledCudaLiteKernel }
  | { readonly blocker: CudaWebGpuExecutionBlocker };

export function createCudaWebGpuExecutionPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: CudaWebGpuExecutionPlanOptions = {},
): CudaWebGpuExecutionPlan {
  const launchDiagnostics = createCudaLaunchValidationDiagnostics(launch, compiled.kernelIr.workgroupSize);
  if (launchDiagnostics.length > 0) {
    const launchBlockers = launchDiagnostics.map((diagnostic) => webGpuBlocker("launch", diagnostic.code, diagnostic.message));
    return {
      supported: false,
      reason: formatWebGpuBlockers(launchBlockers),
      blockers: launchBlockers,
      diagnostics: launchDiagnostics,
    };
  }
  const runtimePlan = createCudaRuntimePlan(compiled);
  const blockers: CudaWebGpuExecutionBlocker[] = [];

  const gridSyncPhasePlan = createCudaGridSyncPhasePlan(compiled);
  const gridSyncPlan = createGridSyncWebGpuPlan(compiled, input, launch, gridSyncPhasePlan);
  if (gridSyncPlan) return gridSyncPlan;
  if (runtimePlan.operations.some((operation) => operation.kind === "grid-sync") && !gridSyncPhasePlan.supported) {
    blockers.push(webGpuBlocker("grid-sync", "grid-sync-phase-unsupported", gridSyncPhasePlan.reason));
  }

  if (runtimePlan.operations.some((operation) => operation.kind === "device-launch")) {
    if (deviceLaunchTreeIsExternallySilent(compiled)) return createRuntimeElidedSingleDispatchWebGpuPlan(compiled, input, launch);
    const depth = options.hostDynamicLaunchDepth ?? 0;
    const maxDepth = normalizeMaxHostDynamicLaunchDepth(options.maxHostDynamicLaunchDepth);
    if (depth >= maxDepth) {
      blockers.push(webGpuBlocker(
        "device-launch",
        "host-dynamic-launch-depth-exceeded",
        `host dynamic launch depth exceeded ${maxDepth}`,
      ));
    } else {
      const hostDynamicPlan = createCudaHostDynamicLaunchPlan(
        compiled,
        input,
        launch,
        options.maxHostExpandedParentInvocations === undefined
          ? {}
          : { maxHostExpandedParentInvocations: options.maxHostExpandedParentInvocations },
      );
      const dynamicLaunchPlan = createHostLiftedDynamicWebGpuPlan(compiled, input, launch, options, hostDynamicPlan);
      if (dynamicLaunchPlan) return dynamicLaunchPlan;
      if (hostDynamicPlan.supported && hostDynamicPlan.launches.length === 0) return createSingleDispatchWebGpuPlan(compiled, input, launch);
      if (!hostDynamicPlan.supported) {
        blockers.push(webGpuBlocker(
          "device-launch",
          hostDynamicPlan.blocker?.code ?? "host-dynamic-launch-unsupported",
          hostDynamicPlan.reason ?? "host-lifted dynamic launch unsupported",
        ));
      }
    }
  }

  const peerCopyRuntimePlan = createCudaPeerCopyPlan(compiled, input, launch);
  const peerCopyPlan = createHostLiftedPeerCopyWebGpuPlan(compiled, input, launch, peerCopyRuntimePlan);
  if (peerCopyPlan) return peerCopyPlan;
  if (runtimePlan.operations.some((operation) => operation.kind === "runtime-copy") && !peerCopyRuntimePlan.supported) {
    blockers.push(webGpuBlocker(
      "runtime-copy",
      peerCopyRuntimePlan.blocker?.code ?? "host-copy-unsupported",
      peerCopyRuntimePlan.reason ?? "host-lifted runtime copy unsupported",
    ));
  }

  const unsupported = createReferenceOnlyRuntimePlan(compiled, blockers);
  if (unsupported) return unsupported;

  return createSingleDispatchWebGpuPlan(compiled, input, launch);
}

export function summarizeCudaWebGpuExecutionPlan(
  plan: CudaWebGpuExecutionPlan,
): CudaWebGpuExecutionStatus {
  if (!plan.supported) {
    return {
      canRunOnWebGpu: false,
      mode: "unsupported",
      requiresHostOrchestration: false,
      reason: plan.reason,
      blockers: plan.blockers,
      diagnostics: plan.diagnostics,
    };
  }

  const requiresHostOrchestration = plan.kind !== "single-dispatch" && plan.kind !== "runtime-elided-single-dispatch";
  return {
    canRunOnWebGpu: true,
    mode: requiresHostOrchestration ? "host-orchestrated" : "direct",
    kind: plan.kind,
    requiresHostOrchestration,
    blockers: [],
    diagnostics: [],
  };
}

export function normalizeCudaWebGpuReadback(
  compiled: CompiledCudaLiteKernel,
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = { ...buffers };
  for (const [poolName, dataName] of cudaWebGpuMemoryPoolDataAliases(compiled)) {
    const data = buffers[dataName];
    if (data) out[poolName] = data;
  }
  return out;
}

export function normalizeCudaWebGpuReadbackNames(
  compiled: CompiledCudaLiteKernel,
  names: readonly string[],
): readonly string[] {
  const aliases = cudaWebGpuMemoryPoolDataAliases(compiled);
  return [...new Set(names.map((name) => aliases.get(name) ?? name))];
}

function createGridSyncWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  gridSyncPhasePlan: ReturnType<typeof createCudaGridSyncPhasePlan>,
): CudaWebGpuExecutionPlan | undefined {
  if (!gridSyncPhasePlan.supported || gridSyncPhasePlan.modules.length <= 1) return undefined;
  const wgslInput = createWgslRunInput(compiled, input);
  const dispatchCount = dispatchCountForLaunch(launch);
  const steps = gridSyncPhasePlan.modules.map((module): WgslKernelSequenceStep => ({
    program: emitKernelIrWgsl(module, {
      features: featureOptionsFor(module.requiredFeatures),
      ...(compiled.f16Mode === undefined ? {} : { f16Mode: compiled.f16Mode }),
      ...(compiled.subgroupMode === undefined ? {} : { subgroupMode: compiled.subgroupMode }),
      ...(compiled.textureDescriptors === undefined ? {} : { textureDescriptors: compiled.textureDescriptors }),
    }).program,
    launch: { dispatchCount },
    ...(wgslInput.uniforms === undefined ? {} : { uniforms: wgslInput.uniforms }),
  }));
  return {
    supported: true,
    kind: "grid-sync-phases",
    steps,
    input: wgslInput,
  };
}

function createHostLiftedPeerCopyWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  plan: ReturnType<typeof createCudaPeerCopyPlan>,
): CudaWebGpuExecutionPlan | undefined {
  if (!plan.supported || plan.copies.length === 0) return undefined;
  const parentInput = createWgslRunInput(compiled, input);
  const steps: WgslKernelSequenceStep[] = [{
    program: compiled.wgslProgram,
    launch: { dispatchCount: dispatchCountForLaunch(launch) },
    ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
  }];

  appendPeerCopySteps(steps, plan.copies);

  return {
    supported: true,
    kind: "host-copy",
    steps,
    input: {
      buffers: { ...parentInput.buffers },
      ...(parentInput.residentBuffers === undefined ? {} : { residentBuffers: parentInput.residentBuffers }),
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      ...(parentInput.readback === undefined ? {} : { readback: parentInput.readback }),
    },
  };
}

function createHostLiftedDynamicWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: CudaWebGpuExecutionPlanOptions,
  plan: ReturnType<typeof createCudaHostDynamicLaunchPlan>,
): CudaWebGpuExecutionPlan | undefined {
  if (!plan.supported || plan.launches.length === 0) return undefined;
  if (!options.compileKernel) {
    return unsupportedWebGpuPlan(compiled, [
      webGpuBlocker("device-launch", "dynamic-child-compiler-unavailable", "dynamic child compiler unavailable for WebGPU host orchestration"),
    ]);
  }

  const parentInput = createWgslRunInput(compiled, input);
  const buffers: Record<string, WgslTypedArray> = { ...parentInput.buffers };
  const residentBuffers = { ...parentInput.residentBuffers };
  const parentDispatchNeeded = hostDynamicParentDispatchNeeded(compiled.kernelIr.operations);
  const poolOffsetUpdates = plan.poolOffsetUpdates ?? {};
  if (parentDispatchNeeded && Object.keys(poolOffsetUpdates).length > 0) {
    return unsupportedWebGpuPlan(compiled, [
      webGpuBlocker(
        "device-launch",
        "parent-side-effects-with-host-pool-allocation",
        "host-lifted DevicePool allocation cannot replay parent side effects without double allocation",
      ),
    ]);
  }
  if (!parentDispatchNeeded) applyHostDynamicPoolOffsetUpdates(buffers, poolOffsetUpdates);
  const steps: WgslKernelSequenceStep[] = [];
  if (parentDispatchNeeded) {
    steps.push({
      program: compiled.wgslProgram,
      launch: { dispatchCount: dispatchCountForLaunch(launch) },
      ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
    });
  }
  const childCompileCache = new Map<string, CompiledCudaLiteKernel>();

  for (const item of plan.launches) {
    const childCompileResult = getOrCompileDynamicChild(
      compiled,
      item,
      childCompileCache,
      options.compileKernel,
    );
    if ("blocker" in childCompileResult) return unsupportedWebGpuPlan(compiled, [childCompileResult.blocker]);
    const childCompiled = childCompileResult.compiled;
    const childLaunch = { gridDim: item.gridDim, blockDim: item.blockDim };
    const childExecutionPlan = createCudaWebGpuExecutionPlan(
      childCompiled,
      item.input,
      childLaunch,
      childExecutionPlanOptions(options),
    );
    if (!childExecutionPlan.supported) {
      const firstBlocker = childExecutionPlan.blockers[0];
      return unsupportedWebGpuPlan(compiled, [
        webGpuBlocker(
          firstBlocker?.kind ?? "device-launch",
          firstBlocker?.code ?? "dynamic-child-runtime-unsupported",
          firstBlocker?.message ?? `dynamic child kernel '${item.kernel.name}' needs unsupported runtime orchestration`,
        ),
      ]);
    }
    appendExecutionPlanWithAliases(steps, buffers, residentBuffers, childExecutionPlan, item.storageAliases);
  }

  return {
    supported: true,
    kind: "host-dynamic-launch",
    steps,
    input: {
      buffers,
      ...(Object.keys(residentBuffers).length === 0 ? {} : { residentBuffers }),
      ...(parentInput.storageMetadata === undefined ? {} : { storageMetadata: parentInput.storageMetadata }),
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      ...(parentInput.readback === undefined ? {} : { readback: parentInput.readback }),
    },
  };
}

function applyHostDynamicPoolOffsetUpdates(
  buffers: Record<string, WgslTypedArray>,
  updates: Readonly<Record<string, number>>,
): void {
  for (const [poolName, offset] of Object.entries(updates)) {
    const name = poolOffsetName(poolName);
    const existing = buffers[name];
    const next = existing instanceof Uint32Array ? new Uint32Array(existing) : new Uint32Array(1);
    next[0] = offset >>> 0;
    buffers[name] = next;
  }
}

function hostDynamicParentDispatchNeeded(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some(operationNeedsParentDispatch);
}

function operationNeedsParentDispatch(operation: SemanticKernelIrOperation): boolean {
  switch (operation.kind) {
    case "block":
      return operation.body.some(operationNeedsParentDispatch);
    case "declare":
      return operation.init === undefined ? false : expressionNeedsParentDispatch(operation.init);
    case "dim3-declare":
      return operation.args.some(expressionNeedsParentDispatch);
    case "cooperative-group-declare":
    case "device-launch":
    case "return":
    case "continue":
    case "break":
      return false;
    case "inline-asm":
    case "load":
      return true;
    case "store":
      return true;
    case "surface-write":
      return true;
    case "surface-read-store":
      return true;
    case "atomic":
      return true;
    case "call":
      return semanticCallNeedsParentDispatch(operation.callee, operation.args);
    case "expression":
      return expressionNeedsParentDispatch(operation.expression);
    case "branch":
      return expressionNeedsParentDispatch(operation.condition) ||
        operation.consequent.some(operationNeedsParentDispatch) ||
        operation.alternate.some(operationNeedsParentDispatch);
    case "loop":
    case "barrier":
    case "fence":
      return true;
  }
}

function expressionNeedsParentDispatch(expression: SemanticExpression): boolean {
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return false;
    case "initializer":
      return expression.elements.some(expressionNeedsParentDispatch);
    case "cast":
      return expressionNeedsParentDispatch(expression.expression);
    case "member":
      return expressionNeedsParentDispatch(expression.object);
    case "index":
      return expressionNeedsParentDispatch(expression.target) || expressionNeedsParentDispatch(expression.index);
    case "unary":
      return expressionNeedsParentDispatch(expression.argument);
    case "binary":
      return expressionNeedsParentDispatch(expression.left) || expressionNeedsParentDispatch(expression.right);
    case "conditional":
      return expressionNeedsParentDispatch(expression.condition) ||
        expressionNeedsParentDispatch(expression.consequent) ||
        expressionNeedsParentDispatch(expression.alternate);
    case "update":
      return expression.argument.kind !== "symbol";
    case "assignment":
      return expression.target.kind !== "symbol" || expressionNeedsParentDispatch(expression.value);
    case "sequence":
      return expression.expressions.some(expressionNeedsParentDispatch);
    case "call": {
      const name = semanticCallName(expression.callee);
      return semanticCallNeedsParentDispatch(name, expression.args);
    }
    case "texture-read":
      return expressionNeedsParentDispatch(expression.texture) ||
        expressionNeedsParentDispatch(expression.x) ||
        expressionNeedsParentDispatch(expression.y);
    case "surface-read":
      return expressionNeedsParentDispatch(expression.surface) ||
        expressionNeedsParentDispatch(expression.xBytes) ||
        expressionNeedsParentDispatch(expression.y) ||
        Boolean(expression.z && expressionNeedsParentDispatch(expression.z));
  }
}

function semanticCallNeedsParentDispatch(name: string | undefined, args: readonly SemanticExpression[]): boolean {
  if (name !== undefined && HOST_RUNTIME_QUERY_WRITE_CALLS.has(name)) return true;
  if (name !== undefined && HOST_SIDE_EFFECT_FREE_CALLS.has(name)) {
    return args.some(expressionNeedsParentDispatch);
  }
  return true;
}

function semanticCallName(expression: SemanticExpression): string | undefined {
  if (expression.kind === "symbol") return expression.name;
  if (expression.kind === "member") {
    const objectName = semanticCallName(expression.object);
    return objectName ? `${objectName}.${expression.property}` : undefined;
  }
  return undefined;
}

function childExecutionPlanOptions(options: CudaWebGpuExecutionPlanOptions): CudaWebGpuExecutionPlanOptions {
  return {
    ...(options.compileKernel === undefined ? {} : { compileKernel: options.compileKernel }),
    ...(options.maxHostExpandedParentInvocations === undefined ? {} : { maxHostExpandedParentInvocations: options.maxHostExpandedParentInvocations }),
    ...(options.maxHostDynamicLaunchDepth === undefined ? {} : { maxHostDynamicLaunchDepth: options.maxHostDynamicLaunchDepth }),
    hostDynamicLaunchDepth: (options.hostDynamicLaunchDepth ?? 0) + 1,
  };
}

function appendExecutionPlanWithAliases(
  steps: WgslKernelSequenceStep[],
  buffers: Record<string, WgslTypedArray>,
  residentBuffers: Record<string, WgslResidentBuffer>,
  plan: Extract<CudaWebGpuExecutionPlan, { readonly supported: true }>,
  aliases: Readonly<Record<string, string>>,
): void {
  for (const [name, value] of Object.entries(plan.input.buffers)) {
    const storageName = aliases[name] ?? name;
    if (!buffersShareStorage(buffers[storageName], value)) buffers[storageName] = value;
  }
  for (const [name, value] of Object.entries(plan.input.residentBuffers ?? {})) {
    residentBuffers[aliases[name] ?? name] = value;
  }
  for (const step of plan.steps) {
    const storageAliases = composeStorageAliases(step.storageAliases, aliases);
    steps.push({
      ...step,
      ...(storageAliases === undefined ? {} : { storageAliases }),
    });
  }
}

function buffersShareStorage(left: WgslTypedArray | undefined, right: WgslTypedArray): boolean {
  return left !== undefined &&
    left.buffer === right.buffer &&
    left.byteOffset === right.byteOffset &&
    left.byteLength === right.byteLength;
}

function composeStorageAliases(
  stepAliases: Readonly<Record<string, string>> | undefined,
  parentAliases: Readonly<Record<string, string>>,
): Readonly<Record<string, string>> | undefined {
  const out: Record<string, string> = { ...parentAliases };
  for (const [from, to] of Object.entries(stepAliases ?? {})) {
    out[from] = parentAliases[to] ?? to;
  }
  return Object.keys(out).length === 0 ? undefined : out;
}

function createSingleDispatchWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaWebGpuExecutionPlan {
  return createSingleDispatchWebGpuPlanWithKind(compiled, input, launch, "single-dispatch");
}

function createRuntimeElidedSingleDispatchWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaWebGpuExecutionPlan {
  return createSingleDispatchWebGpuPlanWithKind(compiled, input, launch, "runtime-elided-single-dispatch");
}

function createSingleDispatchWebGpuPlanWithKind(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  kind: "single-dispatch" | "runtime-elided-single-dispatch",
): CudaWebGpuExecutionPlan {
  const wgslInput = createWgslRunInput(compiled, input);
  return {
    supported: true,
    kind,
    steps: [{
      program: compiled.wgslProgram,
      launch: { dispatchCount: dispatchCountForLaunch(launch) },
      ...(wgslInput.uniforms === undefined ? {} : { uniforms: wgslInput.uniforms }),
    }],
    input: wgslInput,
  };
}

function createReferenceOnlyRuntimePlan(
  compiled: CompiledCudaLiteKernel,
  blockers: readonly CudaWebGpuExecutionBlocker[],
): CudaWebGpuExecutionPlan | undefined {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "cuda-dynamic-launch-host-orchestration" ||
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return undefined;
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (runtimePlan.operations.every((operation) => operation.kind === "device-sync")) return undefined;
  const labels = [...new Set(runtimePlan.operations.map((operation) => operation.kind))].join(", ");
  const reason = labels.length > 0
    ? `CUDA runtime orchestration is reference-only (${labels}); WebGPU host orchestration is not implemented yet`
    : "CUDA runtime orchestration is reference-only; WebGPU host orchestration is not implemented yet";
  const message = blockers.length > 0
    ? `${reason}: ${formatWebGpuBlockers(blockers)}`
    : reason;
  return {
    supported: false,
    reason: message,
    blockers,
    diagnostics: [{
      ...diagnostic,
      severity: "error",
      message,
    }],
  };
}

function unsupportedWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  blockers: readonly CudaWebGpuExecutionBlocker[],
): CudaWebGpuExecutionPlan {
  const reason = formatWebGpuBlockers(blockers);
  return {
    supported: false,
    reason,
    blockers,
    diagnostics: referenceRuntimeDiagnostics(compiled, reason),
  };
}

function referenceRuntimeDiagnostics(
  compiled: CompiledCudaLiteKernel,
  message: string,
): readonly CudaLiteDiagnostic[] {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "cuda-dynamic-launch-host-orchestration" ||
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return [];
  return [{
    ...diagnostic,
    severity: "error",
    message,
  }];
}

function webGpuBlocker(
  kind: CudaWebGpuExecutionBlockerKind,
  code: string,
  message: string,
): CudaWebGpuExecutionBlocker {
  return { kind, code, message };
}

function formatWebGpuBlockers(blockers: readonly CudaWebGpuExecutionBlocker[]): string {
  return blockers.map((blocker) => `${blocker.kind}/${blocker.code}: ${blocker.message}`).join("; ");
}

function normalizeMaxHostDynamicLaunchDepth(value: number | undefined): number {
  if (value === undefined) return DEFAULT_MAX_HOST_DYNAMIC_LAUNCH_DEPTH;
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError("maxHostDynamicLaunchDepth must be a non-negative integer");
  }
  return value;
}

function definePeerCopyProgram(copy: CudaPeerCopyOperation): WgslKernelProgram {
  if (copy.kind !== "copy") throw new Error("peer copy program requires a copy operation");
  const cached = peerCopyProgramCache.get(copy.valueType);
  if (cached) return cached;
  const valueType = copy.valueType === "float" ? "f32" : copy.valueType === "int" ? "i32" : "u32";
  const program = defineWgslKernelProgram({
    name: `bg_peer_copy_${copy.valueType}`,
    workgroupSize: [64, 1, 1],
    bindings: [
      { kind: "storage", name: "bg_peer_dst", valueType, access: "read_write", binding: 0 },
      { kind: "storage", name: "bg_peer_src", valueType, access: "read", binding: 1 },
      { kind: "uniform", name: "params", byteLength: 16, binding: 2 },
    ],
    wgsl: [
      "struct Params {",
      "  dst_base: u32,",
      "  src_base: u32,",
      "  count: u32,",
      "};",
      "@group(0) @binding(0) var<storage, read_write> bg_peer_dst: array<" + valueType + ">;",
      "@group(0) @binding(1) var<storage, read> bg_peer_src: array<" + valueType + ">;",
      "@group(0) @binding(2) var<uniform> params: Params;",
      "@compute @workgroup_size(64, 1, 1)",
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
      "  let index = gid.x;",
      "  if (index < params.count) {",
      "    bg_peer_dst[params.dst_base + index] = bg_peer_src[params.src_base + index];",
      "  }",
      "}",
    ].join("\n"),
  });
  peerCopyProgramCache.set(copy.valueType, program);
  return program;
}

function definePeerByteCopyProgram(copy: CudaPeerCopyOperation): WgslKernelProgram {
  if (copy.kind !== "copy-bytes") throw new Error("peer byte-copy program requires a byte-copy operation");
  if (peerByteCopyProgramCache) return peerByteCopyProgramCache;
  const program = defineWgslKernelProgram({
    name: "bg_peer_copy_bytes",
    workgroupSize: [64, 1, 1],
    bindings: [
      { kind: "storage", name: "bg_peer_dst", valueType: "u32", access: "read_write", binding: 0 },
      { kind: "storage", name: "bg_peer_src", valueType: "u32", access: "read", binding: 1 },
      { kind: "uniform", name: "params", byteLength: 16, binding: 2 },
    ],
    wgsl: [
      "struct Params {",
      "  dst_byte_base: u32,",
      "  src_byte_base: u32,",
      "  byte_count: u32,",
      "};",
      "@group(0) @binding(0) var<storage, read_write> bg_peer_dst: array<u32>;",
      "@group(0) @binding(1) var<storage, read> bg_peer_src: array<u32>;",
      "@group(0) @binding(2) var<uniform> params: Params;",
      "@compute @workgroup_size(64, 1, 1)",
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
      "  let start_byte = params.dst_byte_base;",
      "  let end_byte = start_byte + params.byte_count;",
      "  let word_index = (start_byte >> 2u) + gid.x;",
      "  let word_base = word_index << 2u;",
      "  var mask: u32 = 0u;",
      "  var value_bits: u32 = 0u;",
      "  for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {",
      "    let dst_byte_index = word_base + lane;",
      "    if (dst_byte_index >= start_byte && dst_byte_index < end_byte) {",
      "      let copy_offset = dst_byte_index - start_byte;",
      "      let src_byte_index = params.src_byte_base + copy_offset;",
      "      let src_word = bg_peer_src[src_byte_index >> 2u];",
      "      let src_byte = (src_word >> ((src_byte_index & 3u) * 8u)) & 255u;",
      "      let shift = lane * 8u;",
      "      mask = mask | (255u << shift);",
      "      value_bits = value_bits | (src_byte << shift);",
      "    }",
      "  }",
      "  if (mask != 0u) {",
      "    let current = bg_peer_dst[word_index];",
      "    bg_peer_dst[word_index] = (current & ~mask) | value_bits;",
      "  }",
      "}",
    ].join("\n"),
  });
  peerByteCopyProgramCache = program;
  return program;
}


function definePeerFillProgram(fill: CudaPeerCopyOperation): WgslKernelProgram {
  if (fill.kind !== "fill") throw new Error("peer fill program requires a fill operation");
  const cached = peerFillProgramCache.get(fill.valueType);
  if (cached) return cached;
  const valueType = fill.valueType === "float" ? "f32" : fill.valueType === "int" ? "i32" : "u32";
  const fillValue = fill.valueType === "float"
    ? "bitcast<f32>(params.fill_value)"
    : fill.valueType === "int"
      ? "bitcast<i32>(params.fill_value)"
      : "params.fill_value";
  const program = defineWgslKernelProgram({
    name: `bg_peer_fill_${fill.valueType}`,
    workgroupSize: [64, 1, 1],
    bindings: [
      { kind: "storage", name: "bg_peer_dst", valueType, access: "read_write", binding: 0 },
      { kind: "uniform", name: "params", byteLength: 16, binding: 1 },
    ],
    wgsl: [
      "struct Params {",
      "  dst_base: u32,",
      "  count: u32,",
      "  fill_value: u32,",
      "};",
      "@group(0) @binding(0) var<storage, read_write> bg_peer_dst: array<" + valueType + ">;",
      "@group(0) @binding(1) var<uniform> params: Params;",
      "@compute @workgroup_size(64, 1, 1)",
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
      "  let index = gid.x;",
      "  if (index < params.count) {",
      "    bg_peer_dst[params.dst_base + index] = " + fillValue + ";",
      "  }",
      "}",
    ].join("\n"),
  });
  peerFillProgramCache.set(fill.valueType, program);
  return program;
}

function definePeerByteFillProgram(fill: CudaPeerCopyOperation): WgslKernelProgram {
  if (fill.kind !== "fill-bytes") throw new Error("peer byte-fill program requires a byte-fill operation");
  if (peerByteFillProgramCache) return peerByteFillProgramCache;
  const program = defineWgslKernelProgram({
    name: "bg_peer_fill_bytes",
    workgroupSize: [64, 1, 1],
    bindings: [
      { kind: "storage", name: "bg_peer_dst", valueType: "u32", access: "read_write", binding: 0 },
      { kind: "uniform", name: "params", byteLength: 16, binding: 1 },
    ],
    wgsl: [
      "struct Params {",
      "  dst_byte_base: u32,",
      "  byte_count: u32,",
      "  fill_value: u32,",
      "};",
      "@group(0) @binding(0) var<storage, read_write> bg_peer_dst: array<u32>;",
      "@group(0) @binding(1) var<uniform> params: Params;",
      "@compute @workgroup_size(64, 1, 1)",
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
      "  let start_byte = params.dst_byte_base;",
      "  let end_byte = start_byte + params.byte_count;",
      "  let word_index = (start_byte >> 2u) + gid.x;",
      "  let word_base = word_index << 2u;",
      "  var mask: u32 = 0u;",
      "  var value_bits: u32 = 0u;",
      "  for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {",
      "    let byte_index = word_base + lane;",
      "    if (byte_index >= start_byte && byte_index < end_byte) {",
      "      let shift = lane * 8u;",
      "      mask = mask | (255u << shift);",
      "      value_bits = value_bits | ((params.fill_value & 255u) << shift);",
      "    }",
      "  }",
      "  if (mask != 0u) {",
      "    let current = bg_peer_dst[word_index];",
      "    bg_peer_dst[word_index] = (current & ~mask) | value_bits;",
      "  }",
      "}",
    ].join("\n"),
  });
  peerByteFillProgramCache = program;
  return program;
}

function packPeerCopyParams(copy: CudaPeerCopyOperation): Uint8Array {
  if (copy.kind !== "copy") throw new Error("peer copy params require a copy operation");
  const bytes = new Uint8Array(16);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, copy.dstOffset, true);
  view.setUint32(4, copy.srcOffset, true);
  view.setUint32(8, copy.elementCount, true);
  return bytes;
}

function packPeerByteCopyParams(copy: CudaPeerCopyOperation): Uint8Array {
  if (copy.kind !== "copy-bytes") throw new Error("peer byte-copy params require a byte-copy operation");
  const bytes = new Uint8Array(16);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, copy.dstByteOffset, true);
  view.setUint32(4, copy.srcByteOffset, true);
  view.setUint32(8, copy.byteCount, true);
  return bytes;
}

function packPeerFillParams(fill: CudaPeerCopyOperation): Uint8Array {
  if (fill.kind !== "fill") throw new Error("peer fill params require a fill operation");
  const bytes = new Uint8Array(16);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, fill.dstOffset, true);
  view.setUint32(4, fill.elementCount, true);
  view.setUint32(8, repeatedBytePattern(fill.byteValue), true);
  return bytes;
}

function packPeerByteFillParams(fill: CudaPeerCopyOperation): Uint8Array {
  if (fill.kind !== "fill-bytes") throw new Error("peer byte-fill params require a byte-fill operation");
  const bytes = new Uint8Array(16);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, fill.dstByteOffset, true);
  view.setUint32(4, fill.byteCount, true);
  view.setUint32(8, fill.byteValue & 0xff, true);
  return bytes;
}

function repeatedBytePattern(byteValue: number): number {
  const byte = byteValue & 0xff;
  return (byte | (byte << 8) | (byte << 16) | (byte << 24)) >>> 0;
}

function appendPeerCopySteps(
  steps: WgslKernelSequenceStep[],
  copies: readonly CudaPeerCopyOperation[],
  storageAliases: Readonly<Record<string, string>> = {},
): void {
  for (const copy of copies) {
    if (copy.kind === "copy-bytes") {
      steps.push({
        program: definePeerByteCopyProgram(copy),
        launch: { dispatchCount: [Math.max(byteRangeWordCount(copy.dstByteOffset, copy.byteCount), 1), 1, 1] },
        storageAliases: {
          bg_peer_dst: storageAliases[copy.dstRoot] ?? copy.dstRoot,
          bg_peer_src: storageAliases[copy.srcRoot] ?? copy.srcRoot,
        },
        uniforms: { params: packPeerByteCopyParams(copy) },
      });
      continue;
    }
    if (copy.kind === "fill-bytes") {
      steps.push({
        program: definePeerByteFillProgram(copy),
        launch: { dispatchCount: [Math.max(byteRangeWordCount(copy.dstByteOffset, copy.byteCount), 1), 1, 1] },
        storageAliases: {
          bg_peer_dst: storageAliases[copy.dstRoot] ?? copy.dstRoot,
        },
        uniforms: { params: packPeerByteFillParams(copy) },
      });
      continue;
    }
    if (copy.kind === "fill") {
      steps.push({
        program: definePeerFillProgram(copy),
        launch: { dispatchCount: [Math.max(copy.elementCount, 1), 1, 1] },
        storageAliases: {
          bg_peer_dst: storageAliases[copy.dstRoot] ?? copy.dstRoot,
        },
        uniforms: { params: packPeerFillParams(copy) },
      });
      continue;
    }
    steps.push({
      program: definePeerCopyProgram(copy),
      launch: { dispatchCount: [Math.max(copy.elementCount, 1), 1, 1] },
      storageAliases: {
        bg_peer_dst: storageAliases[copy.dstRoot] ?? copy.dstRoot,
        bg_peer_src: storageAliases[copy.srcRoot] ?? copy.srcRoot,
      },
      uniforms: { params: packPeerCopyParams(copy) },
    });
  }
}

function byteRangeWordCount(byteOffset: number, byteCount: number): number {
  const firstWord = Math.trunc(byteOffset / 4);
  const lastByteExclusive = byteOffset + byteCount;
  const lastWordExclusive = Math.trunc((lastByteExclusive + 3) / 4);
  return Math.max(0, lastWordExclusive - firstWord);
}

function getOrCompileDynamicChild(
  parent: CompiledCudaLiteKernel,
  item: {
    readonly kernel: { readonly name: string };
    readonly blockDim: readonly [number, number, number];
    readonly pointerBaseOffsets: Readonly<Record<string, number>>;
  },
  cache: Map<string, CompiledCudaLiteKernel>,
  compileKernel: NonNullable<CudaWebGpuExecutionPlanOptions["compileKernel"]>,
): DynamicChildCompileResult {
  const key = JSON.stringify({
    kernelName: item.kernel.name,
    blockDim: item.blockDim,
    pointerBaseOffsets: item.pointerBaseOffsets,
  });
  const cached = cache.get(key);
  if (cached) return { compiled: cached };
  try {
    const compiled = compileKernel(parent.ast.source, {
      kernelName: item.kernel.name,
      features: featureOptionsFor(parent.kernelIr.requiredFeatures),
      referenceDynamicParallelism: true,
      referenceGridSync: true,
      referenceCudaRuntime: true,
      ...(parent.f16Mode === undefined ? {} : { f16Mode: parent.f16Mode }),
      ...(parent.subgroupMode === undefined ? {} : { subgroupMode: parent.subgroupMode }),
      ...(parent.textureDescriptors === undefined ? {} : { textureDescriptors: parent.textureDescriptors }),
      workgroupSize: item.blockDim,
      pointerBaseOffsets: item.pointerBaseOffsets,
    });
    cache.set(key, compiled);
    return { compiled };
  } catch (error) {
    return {
      blocker: webGpuBlocker(
        "device-launch",
        "dynamic-child-compile-failed",
        dynamicChildCompileFailureMessage(item.kernel.name, error),
      ),
    };
  }
}

function dynamicChildCompileFailureMessage(kernelName: string, error: unknown): string {
  const prefix = `dynamic child kernel '${kernelName}' could not be compiled for WebGPU`;
  if (error instanceof CudaLiteCompilerError && error.diagnostics.length > 0) {
    const details = error.diagnostics.map((diagnostic) => `${diagnostic.code}: ${diagnostic.message}`).join("; ");
    return `${prefix}: ${details}`;
  }
  if (error instanceof Error && error.message.length > 0) return `${prefix}: ${error.message}`;
  return prefix;
}

function createWgslRunInput(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): WgslKernelRunInput {
  const uniforms = packCudaWebGpuUniformParams(compiled, input);
  const buffers = {
    ...input.buffers,
    ...surfaceBufferInputs(compiled, input),
    ...memoryPoolBufferInputs(compiled, input),
    ...constantBufferInputs(compiled, input),
    ...deviceGlobalBufferInputs(compiled, input),
  };
  const storageMetadata = memoryPoolStorageMetadata(compiled);
  const readback = input.readback === undefined
    ? cudaWebGpuDefaultReadbackNames(compiled)
    : normalizeCudaWebGpuReadbackNames(compiled, input.readback);
  return {
    buffers,
    ...(input.residentBuffers === undefined ? {} : { residentBuffers: input.residentBuffers }),
    ...(Object.keys(storageMetadata).length === 0 ? {} : { storageMetadata }),
    ...(input.textures === undefined ? {} : { textures: input.textures }),
    ...(uniforms.byteLength === 0 ? {} : { uniforms: { bg_uniforms: uniforms } }),
    readback,
  };
}

function dispatchCountForLaunch(launch: KernelLaunch): readonly [number, number, number] {
  return [
    launch.gridDim[0] * launch.blockDim[0],
    launch.gridDim[1] * launch.blockDim[1],
    launch.gridDim[2] * launch.blockDim[2],
  ];
}

function featureOptionsFor(
  requiredFeatures: readonly string[],
): Partial<Record<"shader-f16" | "subgroups" | "compatibility", boolean>> {
  return {
    ...(requiredFeatures.includes("shader-f16") ? { "shader-f16": true } : {}),
    ...(requiredFeatures.includes("subgroups") ? { subgroups: true } : {}),
  };
}

export function packCudaWebGpuUniformParams(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Uint8Array {
  const scalarParams = cudaWebGpuUniformParamDescriptors(compiled);
  if (scalarParams.length === 0) return new Uint8Array(0);
  const bytes = new Uint8Array(Math.max(16, scalarParams.length * 4));
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < scalarParams.length; i++) {
    const param = scalarParams[i]!;
    const value = param.kind === "surface-dimension"
      ? (param.name.endsWith("_width") ? input.surfaces?.[param.surface]?.width : input.surfaces?.[param.surface]?.height)
      : param.kind === "pointer-base"
      ? compiled.pointerBaseOffsets?.[param.pointerBase]
      : param.kind === "scalar"
      ? input.scalars?.[param.name]
      : input.constants?.[param.name];
    if (value === undefined) {
      const kind = param.kind === "surface-dimension" ? "surface input" : param.kind === "scalar" ? "scalar input" : "constant input";
      throw new CudaLiteCompilerError(`missing ${kind} '${param.name}'`, [{
        code: "missing-scalar",
        severity: "error",
        message: `missing ${kind} '${param.name}'`,
        span: param.span,
      }]);
    }
    if (typeof value !== "number") {
      throw new CudaLiteCompilerError(`constant '${param.name}' must be a scalar number`, [{
        code: "invalid-constant-input",
        severity: "error",
        message: `constant '${param.name}' must be a scalar number`,
        span: param.span,
      }]);
    }
    const offset = i * 4;
    if (param.valueType === "int") view.setInt32(offset, Math.trunc(value), true);
    else if (param.valueType === "uint") view.setUint32(offset, Math.trunc(value), true);
    else if (param.valueType === "half" && compiled.f16Mode !== "f32") view.setUint16(offset, float16Bits(value), true);
    else if (param.valueType === "bool") view.setUint32(offset, value ? 1 : 0, true);
    else view.setFloat32(offset, value, true);
  }
  return bytes;
}

function float16Bits(value: number): number {
  return float32ToFloat16Bits(value);
}
