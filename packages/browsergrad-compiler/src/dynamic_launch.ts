import type { WgslResidentBuffer, WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { isCudaHostDynamicNoopCall } from "./cuda_host_silent_calls.js";
import { isHostPoolPointer as isBaseHostPoolPointer, type HostEvalPoolPointer, type HostEvalValue } from "./host_eval.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { createCudaRuntimePlan } from "./runtime_plan.js";
import { semanticPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import {
  isSemanticKernelIrOperation,
  type CudaLiteSemanticLaunchableEntry,
  type CudaLiteSemanticSymbol,
  type SemanticDeviceLaunch,
  type SemanticExpression,
  type SemanticKernelIrOperation,
} from "./semantic_ir.js";
import { semanticIdsEqual } from "./semantic_ids.js";
import type {
  CompiledCudaLiteKernel,
  CompiledKernelInput,
  CudaLiteScalarType,
  KernelLaunch,
} from "./types.js";

export interface CudaHostDynamicLaunchPlan {
  readonly supported: boolean;
  readonly reason?: string;
  readonly blocker?: CudaHostDynamicLaunchBlocker;
  readonly launches: readonly CudaHostDynamicLaunch[];
  readonly poolOffsetUpdates?: Readonly<Record<string, number>>;
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
  | "pool-allocation-not-single-invocation"
  | "pool-allocation-order-sensitive"
  | "branch-not-host-evaluable";

export interface CudaHostDynamicLaunchPlanOptions {
  readonly maxHostExpandedParentInvocations?: number;
}

export interface CudaHostDynamicLaunch {
  readonly launch: SemanticDeviceLaunch;
  readonly kernel: CudaLiteSemanticLaunchableEntry;
  readonly gridDim: readonly [number, number, number];
  readonly blockDim: readonly [number, number, number];
  readonly input: CompiledKernelInput;
  readonly storageAliases: Readonly<Record<string, string>>;
  readonly pointerBaseOffsets: Readonly<Record<string, number>>;
}

interface HostLiftedLaunch {
  readonly launch: SemanticDeviceLaunch;
  readonly env: ReadonlyMap<string, SemanticHostValue>;
}

interface SemanticHostPointer {
  readonly kind: "pointer";
  readonly root: string;
  readonly offset: number;
}

type SemanticHostValue = HostEvalValue | SemanticHostPointer;

interface HostLiftedLaunchCollection {
  readonly launches: readonly HostLiftedLaunch[];
  readonly reason?: string;
  readonly blocker?: CudaHostDynamicLaunchBlocker;
  readonly expandedPoolAllocation: boolean;
  readonly poolOffsetUpdates: Readonly<Record<string, number>>;
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
  if (!runtimePlan.operations.every((operation) => operation.kind === "device-launch" || operation.kind === "device-sync" || operation.kind === "runtime-copy")) {
    return unsupported("mixed-runtime-operations", "runtime operations besides device launch/device sync require reference runtime");
  }
  const parentInvocations = launch.gridDim[0] * launch.gridDim[1] * launch.gridDim[2] *
    launch.blockDim[0] * launch.blockDim[1] * launch.blockDim[2];
  const maxParentInvocations = normalizeMaxHostExpandedParentInvocations(options.maxHostExpandedParentInvocations);
  if (parentInvocations > maxParentInvocations) {
    return unsupported(
      "too-many-parent-invocations",
      `host-expanded dynamic launch needs ${parentInvocations} parent invocations; max is ${maxParentInvocations}`,
    );
  }

  const launchCollection = collectHostLiftedLaunches(compiled.kernelIr.operations, input, launch);
  const launches = launchCollection.launches;
  if (launches.length === 0) {
    if (!launchCollection.blocker) return { supported: true, launches: [], poolOffsetUpdates: launchCollection.poolOffsetUpdates };
    return unsupportedWithBlocker(launchCollection.blocker);
  }

  const planned: CudaHostDynamicLaunch[] = [];
  for (const item of launches) {
    const childKernel = findLaunchableKernel(compiled, item.launch);
    if (!childKernel) return unsupported("unknown-child-kernel", `unknown dynamic kernel '${item.launch.callee}'`);
    const childBlock = evaluateSemanticLaunchVector(item.launch.block, item.env, input);
    const childGrid = evaluateSemanticLaunchVector(item.launch.grid, item.env, input);
    if (!childBlock || !childGrid) return unsupported("child-launch-dimensions-not-host-evaluable", "child launch dimensions must be host-evaluable");
    const childInput = createChildKernelInput(childKernel.params, item.launch, item.env, input);
    if (!childInput) return unsupported("child-arguments-not-host-evaluable", "child launch arguments must be host-evaluable storage aliases or scalar values");
    planned.push({
      launch: item.launch,
      kernel: childKernel,
      gridDim: childGrid,
      blockDim: childBlock,
      input: childInput.input,
      storageAliases: childInput.storageAliases,
      pointerBaseOffsets: childInput.pointerBaseOffsets,
    });
  }
  if (launchCollection.expandedPoolAllocation && !hostDynamicLaunchesAreOrderStable(planned)) {
    return unsupported(
      "pool-allocation-order-sensitive",
      "expanded DevicePool allocation needs child launches to be order-stable except pointer base offsets",
    );
  }
  return { supported: true, launches: planned, poolOffsetUpdates: launchCollection.poolOffsetUpdates };
}

function findLaunchableKernel(
  compiled: CompiledCudaLiteKernel,
  launch: SemanticDeviceLaunch,
): CudaLiteSemanticLaunchableEntry | undefined {
  return compiled.kernelIr.launchableEntries.find((entry) => semanticIdsEqual(entry.id, launch.calleeId));
}

function unsupported(code: CudaHostDynamicLaunchBlockerCode, message: string): CudaHostDynamicLaunchPlan {
  return unsupportedWithBlocker({ code, message });
}

function unsupportedWithBlocker(blocker: CudaHostDynamicLaunchBlocker): CudaHostDynamicLaunchPlan {
  return { supported: false, reason: blocker.message, blocker, launches: [] };
}

function collectHostLiftedLaunches(
  operations: readonly SemanticKernelIrOperation[],
  input: CompiledKernelInput,
  launch: KernelLaunch,
): HostLiftedLaunchCollection {
  const out: HostLiftedLaunch[] = [];
  const parentInvocations = launch.gridDim[0] * launch.gridDim[1] * launch.gridDim[2] *
    launch.blockDim[0] * launch.blockDim[1] * launch.blockDim[2];
  const hasExpandedParent = parentInvocations > 1;
  const poolOffsets = new Map(
    Object.entries(input.memoryPools ?? {}).map(([name, pool]) => [name, pool.offset?.[0] ?? 0] as const),
  );
  const initialPoolOffsets = new Map(poolOffsets);
  let unsafeBlocker: CudaHostDynamicLaunchBlocker | undefined;
  let expandedPoolAllocation = false;
  const markUnsafe = (code: CudaHostDynamicLaunchBlockerCode, message: string): void => {
    unsafeBlocker ??= { code, message };
  };
  const visit = (
    items: readonly SemanticKernelIrOperation[],
    env: ReadonlyMap<string, SemanticHostValue>,
  ): { readonly containsLaunch: boolean; readonly control: "none" | "return" | "break" | "continue"; readonly env: ReadonlyMap<string, SemanticHostValue> } => {
    let current = new Map(env);
    let containsLaunch = false;
    for (let index = 0; index < items.length; index++) {
      const item = items[index]!;
      if (item.kind === "dim3-declare") {
        const value = evaluateSemanticVectorExpressions(item.args, current, input);
        if (value) current.set(item.target.name, value);
        continue;
      }
      if (item.kind === "declare" && !item.target.pointer && item.target.addressSpace === "local" && item.init) {
        const value = evaluateSemanticHostNumber(item.init, current, input);
        if (value !== undefined) current.set(item.target.name, coerceHostScalar(item.target.valueType, value));
        continue;
      }
      if (item.kind === "declare" && item.target.pointer && item.target.addressSpace === "local" && item.init) {
        const pointer = evaluateSemanticHostPointer(item.init, current, input, poolOffsets, () => {
          if (hasExpandedParent) expandedPoolAllocation = true;
        });
        if (pointer) current.set(item.target.name, pointer);
        continue;
      }
      if (item.kind === "pool-allocate") {
        if (item.pool.kind !== "device-pool") {
          markUnsafe("child-arguments-not-host-evaluable", "raw pool allocation cannot be host-lifted for dynamic launch planning");
          continue;
        }
        const pool = input.memoryPools?.[item.pool.name];
        const sizeBytes = evaluateSemanticHostNumber(item.sizeBytes, current, input);
        if (!pool || sizeBytes === undefined) {
          markUnsafe("child-arguments-not-host-evaluable", "DevicePool allocation size and input must be host-evaluable");
          continue;
        }
        const bytes = Math.max(0, Math.trunc(sizeBytes));
        const oldOffset = poolOffsets.get(item.pool.name) ?? pool.offset?.[0] ?? 0;
        poolOffsets.set(item.pool.name, oldOffset + bytes);
        if (hasExpandedParent) expandedPoolAllocation = true;
        current.set(item.target.name, {
          kind: "pool-pointer",
          poolName: item.pool.name,
          byteOffset: oldOffset + bytes > pool.data.byteLength ? -1 : oldOffset,
        });
        continue;
      }
      if (item.kind === "branch") {
        const before = out.length;
        const condition = evaluateSemanticHostNumber(item.condition, current, input);
        if (condition === undefined) {
          if (
            semanticOperationsContainDeviceLaunch(item.consequent) ||
            semanticOperationsContainDeviceLaunch(item.alternate) ||
            semanticOperationsContainDeviceLaunch(items.slice(index + 1))
          ) {
            markUnsafe("branch-not-host-evaluable", "device-side launch branch condition must be host-evaluable or a single-invocation guard");
          }
          return { containsLaunch, control: "none", env: current };
        }
        const result = visit(condition !== 0 ? item.consequent : item.alternate, current);
        containsLaunch = result.containsLaunch || containsLaunch;
        current = new Map(result.env);
        if (out.length > before && semanticParentSideEffectsAfterLaunch(items.slice(index + 1))) {
          markUnsafe("unsafe-parent-side-effects", "parent side effects after device-side launch cannot be replayed in host-lifted sequence");
        }
        if (result.control !== "none") return { containsLaunch, control: result.control, env: current };
        continue;
      }
      if (item.kind === "block") {
        const result = visit(item.body, current);
        containsLaunch = result.containsLaunch || containsLaunch;
        current = new Map(result.env);
        if (result.control !== "none") return { containsLaunch, control: result.control, env: current };
        continue;
      }
      if (item.kind === "loop") {
        if (item.init) applySemanticLoopInit(item.init, current, input, poolOffsets, () => {
          if (hasExpandedParent) expandedPoolAllocation = true;
        });
        for (let iteration = 0; iteration <= 1_000_000; iteration++) {
          if (iteration === 1_000_000) {
            markUnsafe("branch-not-host-evaluable", "device-side launch loop exceeded host evaluation cap");
            break;
          }
          if (item.loopKind !== "do-while" && item.condition && evaluateSemanticHostNumber(item.condition, current, input) === 0) break;
          const result = visit(item.body, current);
          containsLaunch = result.containsLaunch || containsLaunch;
          current = new Map(result.env);
          if (result.control === "return") return { containsLaunch, control: "return", env: current };
          if (result.control === "break") break;
          if (item.continuing) {
            const continuing = visit(item.continuing, current);
            containsLaunch = continuing.containsLaunch || containsLaunch;
            current = new Map(continuing.env);
            if (continuing.control === "return") return { containsLaunch, control: "return", env: current };
            if (continuing.control === "break") break;
          }
          if (item.update) applySemanticExpressionEffect(item.update, current, input);
          if (item.loopKind === "do-while" && item.condition && evaluateSemanticHostNumber(item.condition, current, input) === 0) break;
          if (!item.condition && item.loopKind !== "for") break;
        }
        continue;
      }
      if (item.kind === "expression") {
        applySemanticExpressionEffect(item.expression, current, input);
        continue;
      }
      if (item.kind === "return") return { containsLaunch, control: "return", env: current };
      if (item.kind === "break" || item.kind === "continue") return { containsLaunch, control: item.kind, env: current };
      if (item.kind === "device-launch") {
        if (semanticParentSideEffectsAfterLaunch(items.slice(index + 1))) {
          markUnsafe("unsafe-parent-side-effects", "parent side effects after device-side launch cannot be replayed in host-lifted sequence");
        }
        out.push({ launch: item.launch, env: current });
        containsLaunch = true;
        continue;
      }
      if (item.kind === "call" && isCudaHostDynamicNoopCall(item.callee)) continue;
    }
    return { containsLaunch, control: "none", env: current };
  };
  forEachParentInvocation(launch, (env) => {
    if (!unsafeBlocker) visit(operations, env);
  });
  return unsafeBlocker
    ? { launches: [], reason: unsafeBlocker.message, blocker: unsafeBlocker, expandedPoolAllocation, poolOffsetUpdates: poolOffsetUpdates(initialPoolOffsets, poolOffsets) }
    : { launches: out, expandedPoolAllocation, poolOffsetUpdates: poolOffsetUpdates(initialPoolOffsets, poolOffsets) };
}

function poolOffsetUpdates(
  before: ReadonlyMap<string, number>,
  after: ReadonlyMap<string, number>,
): Readonly<Record<string, number>> {
  const out: Record<string, number> = {};
  for (const [name, value] of after) {
    if ((before.get(name) ?? 0) !== value) out[name] = value >>> 0;
  }
  return out;
}

function coerceHostScalar(valueType: CudaLiteScalarType | undefined, value: number): number {
  if (valueType === "int" || valueType === "uint" || valueType === "bool") return Math.trunc(value);
  return value;
}

function applySemanticExpressionEffect(
  expression: SemanticExpression,
  env: Map<string, SemanticHostValue>,
  input: CompiledKernelInput,
): void {
  if (expression.kind === "sequence") {
    expression.expressions.forEach((item) => applySemanticExpressionEffect(item, env, input));
    return;
  }
  if (expression.kind === "update" && expression.argument.kind === "symbol") {
    const current = evaluateSemanticHostNumber(expression.argument, env, input);
    if (typeof current === "number") env.set(expression.argument.name, current + (expression.operator === "++" ? 1 : -1));
    return;
  }
  if (expression.kind !== "assignment" || expression.target.kind !== "symbol") return;
  const current = evaluateSemanticHostNumber(expression.target, env, input);
  const value = evaluateSemanticHostNumber(expression.value, env, input);
  if (value === undefined) return;
  const next = expression.operator === "=" ? value : current === undefined ? undefined : evaluateSemanticHostBinary(expression.operator.slice(0, -1), current, value);
  if (next !== undefined) env.set(expression.target.name, next);
}

function evaluateSemanticHostPointer(
  expression: SemanticExpression,
  env: ReadonlyMap<string, SemanticHostValue>,
  input: CompiledKernelInput,
  poolOffsets: Map<string, number>,
  onPoolAllocation: () => void,
): SemanticHostPointer | HostEvalPoolPointer | undefined {
  if (expression.kind === "symbol") {
    const pointer = env.get(expression.name);
    if (isSemanticHostPoolPointer(pointer) || isSemanticHostPointer(pointer)) return pointer;
    if (expression.addressSpace === "storage" || expression.addressSpace === "device-global" || expression.addressSpace === "constant") {
      return { kind: "pointer", root: expression.name, offset: 0 };
    }
    return undefined;
  }
  if (expression.kind === "cast" && expression.pointer) {
    return evaluateSemanticHostPointer(expression.expression, env, input, poolOffsets, onPoolAllocation);
  }
  if (expression.kind === "conditional") {
    const condition = evaluateSemanticHostNumber(expression.condition, env, input);
    return condition === undefined
      ? undefined
      : evaluateSemanticHostPointer(condition === 0 ? expression.alternate : expression.consequent, env, input, poolOffsets, onPoolAllocation);
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const base = evaluateSemanticHostPointer(expression.left, env, input, poolOffsets, onPoolAllocation);
    const offset = evaluateSemanticHostNumber(expression.right, env, input);
    if (!base || offset === undefined || isSemanticHostPoolPointer(base)) return undefined;
    return { ...base, offset: base.offset + Math.trunc(offset) * (expression.operator === "-" ? -1 : 1) };
  }
  const ref = semanticPointerArgumentMemoryRef(expression);
  if (ref) {
    const offset = ref.indices.reduce<number | undefined>((sum, index) => {
      const value = evaluateSemanticHostNumber(index, env, input);
      return sum === undefined || value === undefined ? undefined : sum + Math.trunc(value);
    }, 0);
    if (offset !== undefined) return { kind: "pointer", root: ref.base, offset };
  }
  if (expression.kind !== "call" || expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  if (name !== "deviceAllocate" && name !== "streamOrderedAllocate") return undefined;
  if (expression.args.length !== 2) return undefined;
  const poolName = semanticPoolNameFromAllocatorArg(expression.args[0]);
  if (!poolName) return undefined;
  const pool = input.memoryPools?.[poolName];
  if (!pool) return undefined;
  const sizeBytes = expression.args[1] ? Math.max(0, Math.trunc(evaluateSemanticHostNumber(expression.args[1], env, input) ?? NaN)) : NaN;
  if (!Number.isFinite(sizeBytes)) return undefined;
  const oldOffset = poolOffsets.get(poolName) ?? pool.offset?.[0] ?? 0;
  poolOffsets.set(poolName, oldOffset + sizeBytes);
  onPoolAllocation();
  const capacity = pool.data.byteLength;
  return {
    kind: "pool-pointer",
    poolName,
    byteOffset: oldOffset + sizeBytes > capacity ? -1 : oldOffset,
  };
}

function evaluateSemanticHostNumber(
  expression: SemanticExpression,
  env: ReadonlyMap<string, SemanticHostValue>,
  input: CompiledKernelInput,
): number | undefined {
  switch (expression.kind) {
    case "literal": return expression.literalKind === "number" ? expression.value : undefined;
    case "symbol": {
      if (expression.name === "nullptr" || expression.name === "NULL") return 0;
      const local = env.get(expression.name);
      if (typeof local === "number") return local;
      if (isSemanticHostPoolPointer(local)) return local.byteOffset < 0 ? 0 : local.byteOffset + 1;
      if (isSemanticHostPointer(local)) return local.offset + 1;
      return input.scalars?.[expression.name];
    }
    case "pointer-valid": {
      const pointer = env.get(expression.pointer);
      return pointer === undefined || isSemanticHostPoolPointer(pointer) && pointer.byteOffset < 0 ? 0 : 1;
    }
    case "member": {
      if (expression.object.kind !== "symbol") return undefined;
      const vector = env.get(expression.object.name);
      if (!isSemanticHostVector(vector)) return undefined;
      return expression.property === "x" ? vector[0] : expression.property === "y" ? vector[1] : expression.property === "z" ? vector[2] : undefined;
    }
    case "cast": return evaluateSemanticHostNumber(expression.expression, env, input);
    case "unary": {
      const value = evaluateSemanticHostNumber(expression.argument, env, input);
      if (value === undefined) return undefined;
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      if (expression.operator === "!") return value === 0 ? 1 : 0;
      if (expression.operator === "~") return ~Math.trunc(value);
      return undefined;
    }
    case "binary": {
      const left = evaluateSemanticHostNumber(expression.left, env, input);
      if (left === undefined) return undefined;
      if (expression.operator === "&&" && left === 0) return 0;
      if (expression.operator === "||" && left !== 0) return 1;
      const right = evaluateSemanticHostNumber(expression.right, env, input);
      return right === undefined ? undefined : evaluateSemanticHostBinary(expression.operator, left, right);
    }
    case "conditional": {
      const condition = evaluateSemanticHostNumber(expression.condition, env, input);
      return condition === undefined ? undefined : evaluateSemanticHostNumber(condition === 0 ? expression.alternate : expression.consequent, env, input);
    }
    case "assignment":
    case "update": {
      const mutable = new Map(env);
      applySemanticExpressionEffect(expression, mutable, input);
      const target = expression.kind === "assignment" ? expression.target : expression.argument;
      return target.kind === "symbol" ? evaluateSemanticHostNumber(target, mutable, input) : undefined;
    }
    case "sequence": {
      const mutable = new Map(env);
      let value: number | undefined;
      for (const item of expression.expressions) {
        applySemanticExpressionEffect(item, mutable, input);
        value = evaluateSemanticHostNumber(item, mutable, input);
      }
      return value;
    }
    case "call": return evaluateSemanticHostCall(expression, env, input);
    default: return undefined;
  }
}

function evaluateSemanticHostCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  env: ReadonlyMap<string, SemanticHostValue>,
  input: CompiledKernelInput,
): number | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const name = expression.callee.name;
  if ((name === "sizeof" || name === "alignof") && expression.args[0]?.kind === "symbol") {
    return name === "sizeof" ? sizeofCudaType(expression.args[0].name) ?? 4 : alignofCudaType(expression.args[0].name) ?? 4;
  }
  const args = expression.args.map((arg) => evaluateSemanticHostNumber(arg, env, input));
  if (args.some((arg) => arg === undefined)) return undefined;
  const first = args[0] ?? 0;
  const second = args[1] ?? 0;
  switch (name) {
    case "ceil": case "ceilf": return Math.ceil(first);
    case "floor": case "floorf": return Math.floor(first);
    case "round": case "roundf": return first < 0 ? Math.ceil(first - 0.5) : Math.floor(first + 0.5);
    case "trunc": case "truncf": return Math.trunc(first);
    case "min": case "fmin": case "fminf": return Math.min(first, second);
    case "max": case "fmax": case "fmaxf": return Math.max(first, second);
    case "div_ceil": return Math.ceil(first / Math.max(1, second));
    default: return undefined;
  }
}

function evaluateSemanticHostBinary(operator: string, left: number, right: number): number | undefined {
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

function isSemanticHostPointer(value: SemanticHostValue | undefined): value is SemanticHostPointer {
  return typeof value === "object" && value !== null && !Array.isArray(value) && "kind" in value && value.kind === "pointer";
}

function isSemanticHostPoolPointer(value: SemanticHostValue | undefined): value is HostEvalPoolPointer {
  return isBaseHostPoolPointer(value as HostEvalValue | undefined);
}

function isSemanticHostVector(value: SemanticHostValue | undefined): value is readonly [number, number, number] {
  return Array.isArray(value) && value.length === 3;
}

function evaluateSemanticVectorExpressions(
  expressions: readonly SemanticExpression[],
  env: ReadonlyMap<string, SemanticHostValue>,
  input: CompiledKernelInput,
): readonly [number, number, number] | undefined {
  const x = expressions[0] ? evaluateSemanticHostNumber(expressions[0], env, input) : 1;
  const y = expressions[1] ? evaluateSemanticHostNumber(expressions[1], env, input) : 1;
  const z = expressions[2] ? evaluateSemanticHostNumber(expressions[2], env, input) : 1;
  if (x === undefined || y === undefined || z === undefined) return undefined;
  return [Math.max(1, Math.trunc(x)), Math.max(1, Math.trunc(y)), Math.max(1, Math.trunc(z))];
}

function applySemanticLoopInit(
  init: SemanticKernelIrOperation | SemanticExpression,
  env: Map<string, SemanticHostValue>,
  input: CompiledKernelInput,
  poolOffsets: Map<string, number>,
  onPoolAllocation: () => void,
): void {
  if (init.kind === "declare") {
    if (!init.init) return;
    if (init.target.pointer) {
      const pointer = evaluateSemanticHostPointer(init.init, env, input, poolOffsets, onPoolAllocation);
      if (pointer) env.set(init.target.name, pointer);
      return;
    }
    const value = evaluateSemanticHostNumber(init.init, env, input);
    if (value !== undefined) env.set(init.target.name, coerceHostScalar(init.target.valueType, value));
    return;
  }
  if (init.kind === "dim3-declare") {
    const value = evaluateSemanticVectorExpressions(init.args, env, input);
    if (value) env.set(init.target.name, value);
    return;
  }
  if (init.kind === "expression") {
    applySemanticExpressionEffect(init.expression, env, input);
    return;
  }
  if (!isSemanticKernelIrOperation(init)) applySemanticExpressionEffect(init, env, input);
}

function hostDynamicLaunchesAreOrderStable(launches: readonly CudaHostDynamicLaunch[]): boolean {
  if (launches.length <= 1) return true;
  const first = hostDynamicLaunchOrderSignature(launches[0]!);
  return launches.every((launch) => hostDynamicLaunchOrderSignature(launch) === first);
}

function hostDynamicLaunchOrderSignature(launch: CudaHostDynamicLaunch): string {
  return JSON.stringify({
    kernel: launch.kernel.name,
    gridDim: launch.gridDim,
    blockDim: launch.blockDim,
    scalars: sortedScalarRecord(launch.input.scalars ?? {}),
    storageAliases: sortedStringRecord(launch.storageAliases),
    bufferNames: Object.keys(launch.input.buffers).sort(),
    deviceGlobalNames: Object.keys(launch.input.deviceGlobals ?? {}).sort(),
    residentBufferNames: Object.keys(launch.input.residentBuffers ?? {}).sort(),
    memoryPoolNames: Object.keys(launch.input.memoryPools ?? {}).sort(),
    pointerOffsetNames: Object.keys(launch.pointerBaseOffsets).sort(),
  });
}

function sortedScalarRecord(values: Readonly<Record<string, number>>): Record<string, number> {
  return Object.fromEntries(Object.entries(values).sort(([left], [right]) => left.localeCompare(right)));
}

function sortedStringRecord(values: Readonly<Record<string, string>>): Record<string, string> {
  return Object.fromEntries(Object.entries(values).sort(([left], [right]) => left.localeCompare(right)));
}

function semanticPoolNameFromAllocatorArg(expression: SemanticExpression | undefined): string | undefined {
  if (!expression) return undefined;
  if (expression.kind === "symbol") return expression.name;
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "symbol") {
    return expression.argument.name;
  }
  return undefined;
}

function forEachParentInvocation(
  launch: KernelLaunch,
  visit: (env: ReadonlyMap<string, SemanticHostValue>) => void,
): void {
  for (let bz = 0; bz < launch.gridDim[2]; bz++) {
    for (let by = 0; by < launch.gridDim[1]; by++) {
      for (let bx = 0; bx < launch.gridDim[0]; bx++) {
        for (let tz = 0; tz < launch.blockDim[2]; tz++) {
          for (let ty = 0; ty < launch.blockDim[1]; ty++) {
            for (let tx = 0; tx < launch.blockDim[0]; tx++) {
              visit(new Map<string, SemanticHostValue>([
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

function semanticParentSideEffectsAfterLaunch(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) => {
    if (operation.kind === "dim3-declare" || operation.kind === "cooperative-group-declare" || operation.kind === "device-launch" || operation.kind === "return") return false;
    if (operation.kind === "declare") return !operation.target.pointer && operation.target.addressSpace !== "local";
    if (operation.kind === "call") return !isCudaHostDynamicNoopCall(operation.callee);
    if (operation.kind === "branch") return semanticParentSideEffectsAfterLaunch(operation.consequent) || semanticParentSideEffectsAfterLaunch(operation.alternate);
    if (operation.kind === "block" || operation.kind === "loop") return semanticParentSideEffectsAfterLaunch(operation.body);
    if (operation.kind === "expression") return expressionHasExternallyVisibleEffect(operation.expression);
    return operation.kind !== "break" && operation.kind !== "continue" && operation.kind !== "copy-fence";
  });
}

function expressionHasExternallyVisibleEffect(expression: SemanticExpression): boolean {
  if (expression.kind === "assignment") return expression.target.kind !== "symbol" || expression.target.addressSpace !== "local";
  if (expression.kind === "update") return expression.argument.kind !== "symbol" || expression.argument.addressSpace !== "local";
  if (expression.kind === "call") return expression.callee.kind !== "symbol" || !isCudaHostDynamicNoopCall(expression.callee.name);
  if (expression.kind === "sequence") return expression.expressions.some(expressionHasExternallyVisibleEffect);
  return false;
}

function normalizeMaxHostExpandedParentInvocations(value: number | undefined): number {
  if (value === undefined) return DEFAULT_MAX_HOST_EXPANDED_PARENT_INVOCATIONS;
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError("maxHostExpandedParentInvocations must be a non-negative integer");
  }
  return value;
}

function semanticOperationsContainDeviceLaunch(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) => operation.kind === "device-launch" ||
    operation.kind === "branch" && (semanticOperationsContainDeviceLaunch(operation.consequent) || semanticOperationsContainDeviceLaunch(operation.alternate)) ||
    (operation.kind === "loop" || operation.kind === "block") && semanticOperationsContainDeviceLaunch(operation.body));
}

function createChildKernelInput(
  params: readonly CudaLiteSemanticSymbol[],
  launch: SemanticDeviceLaunch,
  env: ReadonlyMap<string, SemanticHostValue>,
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
    const arg = launch.args[index];
    if (!arg) return undefined;
    if (param.pointer) {
      if (param.valueType === "devicepool") {
        const root = semanticPoolNameFromAllocatorArg(arg);
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
      const pointer = evaluateSemanticHostPointer(arg, env, input, new Map(), () => {});
      const poolPointer = arg.kind === "symbol" ? env.get(arg.name) : undefined;
      if (isSemanticHostPoolPointer(poolPointer)) {
        if (poolPointer.byteOffset < 0) return undefined;
        const pool = input.memoryPools?.[poolPointer.poolName];
        if (!pool) return undefined;
        buffers[param.name] = typedPoolView(pool.data, param.valueType);
        storageAliases[param.name] = poolDataName(poolPointer.poolName);
        pointerBaseOffsets[param.name] = Math.trunc(poolPointer.byteOffset / elementByteSize(param.valueType));
        continue;
      }
      if (!pointer || isSemanticHostPoolPointer(pointer)) return undefined;
      if (pointer.offset < 0) return undefined;
      const root = pointer.root;
      const buffer = input.buffers[root];
      const global = input.deviceGlobals?.[root];
      const resident = input.residentBuffers?.[root];
      if ((buffer || global) && resident) return undefined;
      if (buffer) buffers[param.name] = buffer;
      else if (global) buffers[param.name] = global;
      else if (resident) residentBuffers[param.name] = resident;
      else return undefined;
      if (root !== param.name) storageAliases[param.name] = root;
      if (pointer.offset !== 0) pointerBaseOffsets[param.name] = pointer.offset;
    } else {
      const value = evaluateSemanticHostNumber(arg, env, input);
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

function typedPoolView(data: Uint32Array, valueType: CudaLiteScalarType | undefined): WgslTypedArray {
  if (valueType === "int") return new Int32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  if (valueType === "uint" || valueType === "voidptr" || valueType === "bool") return data;
  return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
}

function elementByteSize(valueType: CudaLiteScalarType | undefined): number {
  return valueType === "half" ? 2 : 4;
}

function evaluateSemanticLaunchVector(
  expressions: readonly SemanticExpression[],
  env: ReadonlyMap<string, SemanticHostValue>,
  input: CompiledKernelInput,
): readonly [number, number, number] | undefined {
  if (expressions.length === 1 && expressions[0]?.kind === "symbol") {
    const value = env.get(expressions[0].name);
    if (isSemanticHostVector(value)) return value;
  }
  return evaluateSemanticVectorExpressions(expressions, env, input);
}
