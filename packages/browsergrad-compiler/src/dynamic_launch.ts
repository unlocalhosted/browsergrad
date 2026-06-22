import type { WgslResidentBuffer, WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { expressionName, rootIdentifier } from "./analyzer.js";
import {
  evaluateHostNumber,
  evaluatePointerArgument,
  evaluateVectorExpressions,
  isHostVector,
  isSingleInvocationGuard,
  type HostEvalValue,
} from "./host_eval.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { createCudaRuntimePlan } from "./runtime_plan.js";
import type {
  CompiledCudaLiteKernel,
  CompiledKernelInput,
  CudaLiteExpression,
  CudaLiteKernel,
  CudaLiteKernelLaunchStatement,
  CudaLiteParam,
  CudaLiteStatement,
  KernelLaunch,
} from "./types.js";

export interface CudaHostDynamicLaunchPlan {
  readonly supported: boolean;
  readonly reason?: string;
  readonly blocker?: CudaHostDynamicLaunchBlocker;
  readonly launches: readonly CudaHostDynamicLaunch[];
}

export interface CudaHostDynamicLaunchBlocker {
  readonly code: CudaHostDynamicLaunchBlockerCode;
  readonly message: string;
}

export type CudaHostDynamicLaunchBlockerCode =
  | "no-device-launch"
  | "mixed-runtime-operations"
  | "too-many-parent-invocations"
  | "no-host-liftable-launch"
  | "unknown-child-kernel"
  | "child-runtime-unsupported"
  | "child-launch-dimensions-not-host-evaluable"
  | "child-arguments-not-host-evaluable"
  | "unsafe-parent-side-effects"
  | "branch-not-host-evaluable";

export interface CudaHostDynamicLaunchPlanOptions {
  readonly maxHostExpandedParentInvocations?: number;
}

export interface CudaHostDynamicLaunch {
  readonly statement: CudaLiteKernelLaunchStatement;
  readonly kernel: CudaLiteKernel;
  readonly gridDim: readonly [number, number, number];
  readonly blockDim: readonly [number, number, number];
  readonly input: CompiledKernelInput;
  readonly storageAliases: Readonly<Record<string, string>>;
  readonly pointerBaseOffsets: Readonly<Record<string, number>>;
}

interface HostLiftedLaunch {
  readonly statement: CudaLiteKernelLaunchStatement;
  readonly env: ReadonlyMap<string, HostEvalValue>;
}

interface HostLiftedLaunchCollection {
  readonly launches: readonly HostLiftedLaunch[];
  readonly reason?: string;
  readonly blocker?: CudaHostDynamicLaunchBlocker;
}

type MemoryPoolInput = NonNullable<CompiledKernelInput["memoryPools"]>[string];
const DEFAULT_MAX_HOST_EXPANDED_PARENT_INVOCATIONS = 4096;

export function createCudaHostDynamicLaunchPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: CudaHostDynamicLaunchPlanOptions = {},
): CudaHostDynamicLaunchPlan {
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (!runtimePlan.operations.some((operation) => operation.kind === "device-launch")) {
    return unsupported("no-device-launch", "no device-side launch found");
  }
  if (!runtimePlan.operations.every((operation) => operation.kind === "device-launch" || operation.kind === "device-sync")) {
    return unsupported("mixed-runtime-operations", "runtime operations besides device launch/device sync require reference runtime");
  }
  const parentInvocations = launch.gridDim[0] * launch.gridDim[1] * launch.gridDim[2] *
    launch.blockDim[0] * launch.blockDim[1] * launch.blockDim[2];
  const maxParentInvocations = options.maxHostExpandedParentInvocations ?? DEFAULT_MAX_HOST_EXPANDED_PARENT_INVOCATIONS;
  if (parentInvocations > maxParentInvocations) {
    return unsupported(
      "too-many-parent-invocations",
      `host-expanded dynamic launch needs ${parentInvocations} parent invocations; max is ${maxParentInvocations}`,
    );
  }

  const launchCollection = collectHostLiftedLaunches(compiled.ir.body, input, launch);
  const launches = launchCollection.launches;
  if (launches.length === 0) {
    if (!launchCollection.blocker) return { supported: true, launches: [] };
    return unsupportedWithBlocker(launchCollection.blocker);
  }

  const planned: CudaHostDynamicLaunch[] = [];
  for (const item of launches) {
    const childKernel = compiled.ast.kernels.find((kernel) => kernel.name === item.statement.callee);
    if (!childKernel) return unsupported("unknown-child-kernel", `unknown dynamic kernel '${item.statement.callee}'`);
    const childBlock = evaluateLaunchVector(item.statement.block, item.env, input);
    const childGrid = evaluateLaunchVector(item.statement.grid, item.env, input);
    if (!childBlock || !childGrid) return unsupported("child-launch-dimensions-not-host-evaluable", "child launch dimensions must be host-evaluable");
    const childInput = createChildKernelInput(childKernel.params, item.statement, item.env, input);
    if (!childInput) return unsupported("child-arguments-not-host-evaluable", "child launch arguments must be host-evaluable storage aliases or scalar values");
    planned.push({
      statement: item.statement,
      kernel: childKernel,
      gridDim: childGrid,
      blockDim: childBlock,
      input: childInput.input,
      storageAliases: childInput.storageAliases,
      pointerBaseOffsets: childInput.pointerBaseOffsets,
    });
  }
  return { supported: true, launches: planned };
}

function unsupported(code: CudaHostDynamicLaunchBlockerCode, message: string): CudaHostDynamicLaunchPlan {
  return unsupportedWithBlocker({ code, message });
}

function unsupportedWithBlocker(blocker: CudaHostDynamicLaunchBlocker): CudaHostDynamicLaunchPlan {
  return { supported: false, reason: blocker.message, blocker, launches: [] };
}

function collectHostLiftedLaunches(
  statements: readonly CudaLiteStatement[],
  input: CompiledKernelInput,
  launch: KernelLaunch,
): HostLiftedLaunchCollection {
  const out: HostLiftedLaunch[] = [];
  let unsafeBlocker: CudaHostDynamicLaunchBlocker | undefined;
  const markUnsafe = (code: CudaHostDynamicLaunchBlockerCode, message: string): void => {
    unsafeBlocker ??= { code, message };
  };
  const visit = (
    items: readonly CudaLiteStatement[],
    env: ReadonlyMap<string, HostEvalValue>,
  ): { readonly containsLaunch: boolean; readonly returned: boolean; readonly env: ReadonlyMap<string, HostEvalValue> } => {
    let current = new Map(env);
    let containsLaunch = false;
    for (let index = 0; index < items.length; index++) {
      const item = items[index]!;
      if (item.kind === "dim3") {
        const value = evaluateVectorExpressions(item.args, current, input);
        if (value) current.set(item.name, value);
        continue;
      }
      if (item.kind === "var" && !item.pointer && item.storage === "local" && item.init) {
        const value = evaluateHostNumber(item.init, current, input);
        if (value !== undefined) current.set(item.name, coerceHostScalar(item.valueType, value));
        continue;
      }
      if (item.kind === "if") {
        const before = out.length;
        const condition = evaluateHostNumber(item.condition, current, input);
        if (condition === undefined) {
          if (isSingleInvocationGuard(item.condition)) {
            const result = visit(item.consequent, current);
            containsLaunch = result.containsLaunch || containsLaunch;
            if (out.length > before && hasParentSideEffectsAfterLaunch(items.slice(index + 1))) {
              markUnsafe("unsafe-parent-side-effects", "parent side effects after device-side launch cannot be replayed in host-lifted sequence");
            }
            if (result.returned) return { containsLaunch, returned: true, env: result.env };
            continue;
          }
          if (
            containsKernelLaunch(item.consequent) ||
            containsKernelLaunch(item.alternate ?? []) ||
            containsKernelLaunch(items.slice(index + 1))
          ) {
            markUnsafe("branch-not-host-evaluable", "device-side launch branch condition must be host-evaluable or a single-invocation guard");
          }
          return { containsLaunch, returned: false, env: current };
        }
        const result = visit(condition !== 0 ? item.consequent : item.alternate ?? [], current);
        containsLaunch = result.containsLaunch || containsLaunch;
        current = new Map(result.env);
        if (out.length > before && hasParentSideEffectsAfterLaunch(items.slice(index + 1))) {
          markUnsafe("unsafe-parent-side-effects", "parent side effects after device-side launch cannot be replayed in host-lifted sequence");
        }
        if (result.returned) return { containsLaunch, returned: true, env: current };
        continue;
      }
      if (item.kind === "expr") {
        applyHostExpressionEffect(item.expression, current, input);
        continue;
      }
      if (item.kind === "return") return { containsLaunch, returned: true, env: current };
      if (item.kind === "kernel-launch") {
        if (hasParentSideEffectsAfterLaunch(items.slice(index + 1))) {
          markUnsafe("unsafe-parent-side-effects", "parent side effects after device-side launch cannot be replayed in host-lifted sequence");
        }
        out.push({ statement: item, env: current });
        containsLaunch = true;
      }
    }
    return { containsLaunch, returned: false, env: current };
  };
  forEachParentInvocation(launch, (env) => {
    if (!unsafeBlocker) visit(statements, env);
  });
  return unsafeBlocker ? { launches: [], reason: unsafeBlocker.message, blocker: unsafeBlocker } : { launches: out };
}

function coerceHostScalar(valueType: CudaLiteParam["valueType"], value: number): number {
  if (valueType === "int" || valueType === "uint" || valueType === "bool") return Math.trunc(value);
  return value;
}

function applyHostExpressionEffect(
  expression: CudaLiteExpression,
  env: Map<string, HostEvalValue>,
  input: CompiledKernelInput,
): void {
  if (expression.kind !== "assignment" || expression.left.kind !== "identifier") return;
  if (expression.operator !== "=") return;
  const value = evaluateHostNumber(expression.right, env, input);
  if (value !== undefined) env.set(expression.left.name, value);
}

function forEachParentInvocation(
  launch: KernelLaunch,
  visit: (env: ReadonlyMap<string, HostEvalValue>) => void,
): void {
  for (let bz = 0; bz < launch.gridDim[2]; bz++) {
    for (let by = 0; by < launch.gridDim[1]; by++) {
      for (let bx = 0; bx < launch.gridDim[0]; bx++) {
        for (let tz = 0; tz < launch.blockDim[2]; tz++) {
          for (let ty = 0; ty < launch.blockDim[1]; ty++) {
            for (let tx = 0; tx < launch.blockDim[0]; tx++) {
              visit(new Map<string, HostEvalValue>([
                ["blockIdx", [bx, by, bz]],
                ["threadIdx", [tx, ty, tz]],
                ["blockDim", launch.blockDim],
                ["gridDim", launch.gridDim],
              ]));
            }
          }
        }
      }
    }
  }
}

function hasParentSideEffectsAfterLaunch(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    switch (statement.kind) {
      case "dim3":
      case "cooperative-group":
        continue;
      case "expr":
        if (isHostNoopExpression(statement.expression)) continue;
        return true;
      case "if":
        if (hasParentSideEffectsAfterLaunch(statement.consequent) || hasParentSideEffectsAfterLaunch(statement.alternate ?? [])) return true;
        continue;
      case "var":
        if (statement.storage === "local" && !statement.pointer) continue;
        return true;
      case "kernel-launch":
        continue;
      case "asm":
      case "for":
      case "return":
      case "continue":
        return true;
    }
  }
  return false;
}

function containsKernelLaunch(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "kernel-launch") return true;
    if (statement.kind === "if" && (containsKernelLaunch(statement.consequent) || containsKernelLaunch(statement.alternate ?? []))) return true;
    if (statement.kind === "for" && containsKernelLaunch(statement.body)) return true;
  }
  return false;
}

function isHostNoopExpression(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "cudaDeviceSynchronize" || name === "printf";
}

function createChildKernelInput(
  params: readonly CudaLiteParam[],
  statement: CudaLiteKernelLaunchStatement,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): {
  readonly input: CompiledKernelInput;
  readonly storageAliases: Readonly<Record<string, string>>;
  readonly pointerBaseOffsets: Readonly<Record<string, number>>;
} | undefined {
  const scalars: Record<string, number> = {};
  const buffers: Record<string, WgslTypedArray> = {};
  const residentBuffers: Record<string, WgslResidentBuffer> = {};
  const memoryPools: Record<string, MemoryPoolInput> = {};
  const storageAliases: Record<string, string> = {};
  const pointerBaseOffsets: Record<string, number> = {};
  for (const [index, param] of params.entries()) {
    const arg = statement.args[index];
    if (!arg) return undefined;
    if (param.pointer) {
      if (param.valueType === "devicepool") {
        const root = arg.kind === "identifier" ? rootIdentifier(arg) : undefined;
        if (!root) return undefined;
        const pool = input.memoryPools?.[root];
        if (!pool) return undefined;
        memoryPools[param.name] = pool;
        if (root !== param.name) {
          storageAliases[poolDataName(param.name)] = poolDataName(root);
          storageAliases[poolOffsetName(param.name)] = poolOffsetName(root);
        }
        continue;
      }
      const pointer = evaluatePointerArgument(arg, env, input);
      if (!pointer) return undefined;
      if (pointer.offset < 0) return undefined;
      const root = pointer.root;
      const buffer = input.buffers[root];
      const resident = input.residentBuffers?.[root];
      if (buffer && resident) return undefined;
      if (buffer) buffers[param.name] = buffer;
      else if (resident) residentBuffers[param.name] = resident;
      else return undefined;
      if (root !== param.name) storageAliases[param.name] = root;
      if (pointer.offset !== 0) pointerBaseOffsets[param.name] = pointer.offset;
    } else {
      const value = evaluateHostNumber(arg, env, input);
      if (value === undefined) return undefined;
      scalars[param.name] = value;
    }
  }
  return {
    input: {
      ...input,
      buffers,
      ...(Object.keys(residentBuffers).length === 0 ? {} : { residentBuffers }),
      memoryPools: { ...input.memoryPools, ...memoryPools },
      scalars: { ...input.scalars, ...scalars },
    },
    storageAliases,
    pointerBaseOffsets,
  };
}

function evaluateLaunchVector(
  expressions: readonly CudaLiteExpression[],
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): readonly [number, number, number] | undefined {
  if (expressions.length === 1 && expressions[0]?.kind === "identifier") {
    const value = env.get(expressions[0].name);
    if (isHostVector(value)) return value;
  }
  return evaluateVectorExpressions(expressions, env, input);
}
