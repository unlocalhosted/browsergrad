import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { walkCudaLiteExpressions } from "./ast_queries.js";
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
  readonly launches: readonly CudaHostDynamicLaunch[];
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
}

type MemoryPoolInput = NonNullable<CompiledKernelInput["memoryPools"]>[string];

export function createCudaHostDynamicLaunchPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaHostDynamicLaunchPlan {
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (!runtimePlan.operations.some((operation) => operation.kind === "device-launch")) {
    return unsupported("no device-side launch found");
  }
  if (!runtimePlan.operations.every((operation) => operation.kind === "device-launch" || operation.kind === "device-sync")) {
    return unsupported("runtime operations besides device launch/device sync require reference runtime");
  }
  if (launch.gridDim.some((axis) => axis !== 1)) {
    return unsupported("parent gridDim must be [1, 1, 1] for host-lifted dynamic launch");
  }

  const launchCollection = collectHostLiftedLaunches(compiled.ir.body, input, launch);
  const launches = launchCollection.launches;
  if (launches.length === 0) return unsupported(launchCollection.reason ?? "no host-liftable device-side launches");

  const planned: CudaHostDynamicLaunch[] = [];
  for (const item of launches) {
    const childKernel = compiled.ast.kernels.find((kernel) => kernel.name === item.statement.callee);
    if (!childKernel) return unsupported(`unknown dynamic kernel '${item.statement.callee}'`);
    const childRuntimeGap = unsupportedHostLiftChildRuntime(childKernel);
    if (childRuntimeGap) return unsupported(childRuntimeGap);
    const childBlock = evaluateLaunchVector(item.statement.block, item.env, input);
    const childGrid = evaluateLaunchVector(item.statement.grid, item.env, input);
    if (!childBlock || !childGrid) return unsupported("child launch dimensions must be host-evaluable");
    const childInput = createChildKernelInput(childKernel.params, item.statement, item.env, input);
    if (!childInput) return unsupported("child launch arguments must be host-evaluable storage aliases or scalar values");
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

function unsupported(reason: string): CudaHostDynamicLaunchPlan {
  return { supported: false, reason, launches: [] };
}

function collectHostLiftedLaunches(
  statements: readonly CudaLiteStatement[],
  input: CompiledKernelInput,
  launch: KernelLaunch,
): HostLiftedLaunchCollection {
  const out: HostLiftedLaunch[] = [];
  const initial = new Map<string, HostEvalValue>();
  const parentHasSingleInvocation = launch.gridDim.every((axis) => axis === 1) && launch.blockDim.every((axis) => axis === 1);
  let unsafeReason: string | undefined;
  const markUnsafe = (reason: string): void => {
    unsafeReason ??= reason;
  };
  const visit = (
    items: readonly CudaLiteStatement[],
    env: ReadonlyMap<string, HostEvalValue>,
    singleInvocationGuard: boolean,
  ): boolean => {
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
        if (value !== undefined) current.set(item.name, value);
        continue;
      }
      if (item.kind === "if") {
        const before = out.length;
        if (isSingleInvocationGuard(item.condition)) {
          containsLaunch = visit(item.consequent, current, true) || containsLaunch;
          if (out.length > before && hasParentSideEffectsAfterLaunch(items.slice(index + 1))) {
            markUnsafe("parent side effects after device-side launch cannot be replayed in host-lifted sequence");
          }
          continue;
        }
        const condition = evaluateHostNumber(item.condition, current, input);
        if (condition === undefined) {
          if (containsKernelLaunch(item.consequent) || containsKernelLaunch(item.alternate ?? [])) {
            markUnsafe("device-side launch branch condition must be host-evaluable or a single-invocation guard");
          }
          return containsLaunch;
        }
        containsLaunch = visit(condition !== 0 ? item.consequent : item.alternate ?? [], current, singleInvocationGuard) || containsLaunch;
        if (out.length > before && hasParentSideEffectsAfterLaunch(items.slice(index + 1))) {
          markUnsafe("parent side effects after device-side launch cannot be replayed in host-lifted sequence");
        }
        continue;
      }
      if (item.kind === "kernel-launch") {
        if (!(singleInvocationGuard || parentHasSingleInvocation)) {
          markUnsafe("device-side launch must be single-invocation guarded or parent launch must be one thread");
        }
        else {
          if (hasParentSideEffectsAfterLaunch(items.slice(index + 1))) {
            markUnsafe("parent side effects after device-side launch cannot be replayed in host-lifted sequence");
          }
          out.push({ statement: item, env: current });
          containsLaunch = true;
        }
      }
    }
    return containsLaunch;
  };
  visit(statements, initial, parentHasSingleInvocation);
  return unsafeReason ? { launches: [], reason: unsafeReason } : { launches: out };
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

function unsupportedHostLiftChildRuntime(kernel: CudaLiteKernel): string | undefined {
  if (containsKernelLaunch(kernel.body)) {
    return `child kernel '${kernel.name}' contains nested device-side launches`;
  }
  let reason: string | undefined;
  walkCudaLiteExpressions(kernel.body, (expression) => {
    if (reason || expression.kind !== "call") return;
    if (isGridSyncCall(expression)) reason = `child kernel '${kernel.name}' requires grid-sync orchestration`;
  });
  return reason;
}

function containsKernelLaunch(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "kernel-launch") return true;
    if (statement.kind === "if" && (containsKernelLaunch(statement.consequent) || containsKernelLaunch(statement.alternate ?? []))) return true;
    if (statement.kind === "for" && containsKernelLaunch(statement.body)) return true;
  }
  return false;
}

function isGridSyncCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" &&
    expression.callee.kind === "member" &&
    expression.callee.property === "sync";
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
      if (!buffer) return undefined;
      buffers[param.name] = buffer;
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
