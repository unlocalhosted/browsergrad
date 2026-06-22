import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { expressionName, rootIdentifier } from "./analyzer.js";
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
}

interface HostLiftedLaunch {
  readonly statement: CudaLiteKernelLaunchStatement;
  readonly env: ReadonlyMap<string, HostEvalValue>;
}

type HostEvalValue = number | readonly [number, number, number];

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

  const launches = collectHostLiftedLaunches(compiled.ir.body, input, launch);
  if (launches.length === 0) return unsupported("no host-liftable device-side launches");

  const planned: CudaHostDynamicLaunch[] = [];
  for (const item of launches) {
    const childKernel = compiled.ast.kernels.find((kernel) => kernel.name === item.statement.callee);
    if (!childKernel) return unsupported(`unknown dynamic kernel '${item.statement.callee}'`);
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
): readonly HostLiftedLaunch[] {
  const out: HostLiftedLaunch[] = [];
  const initial = new Map<string, HostEvalValue>();
  const parentHasSingleInvocation = launch.gridDim.every((axis) => axis === 1) && launch.blockDim.every((axis) => axis === 1);
  let unsafe = false;
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
          if (out.length > before && hasHostSideEffects(items.slice(index + 1))) unsafe = true;
          continue;
        }
        const condition = evaluateHostNumber(item.condition, current, input);
        if (condition === undefined) return containsLaunch;
        containsLaunch = visit(condition !== 0 ? item.consequent : item.alternate ?? [], current, singleInvocationGuard) || containsLaunch;
        if (out.length > before && hasHostSideEffects(items.slice(index + 1))) unsafe = true;
        continue;
      }
      if (item.kind === "kernel-launch") {
        if (!(singleInvocationGuard || parentHasSingleInvocation)) unsafe = true;
        else {
          if (hasHostSideEffects(items.slice(index + 1))) unsafe = true;
          out.push({ statement: item, env: current });
          containsLaunch = true;
        }
      }
    }
    return containsLaunch;
  };
  visit(statements, initial, parentHasSingleInvocation);
  return unsafe ? [] : out;
}

function hasHostSideEffects(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    switch (statement.kind) {
      case "dim3":
      case "cooperative-group":
        continue;
      case "expr":
        if (isHostNoopExpression(statement.expression)) continue;
        return true;
      case "if":
        if (hasHostSideEffects(statement.consequent) || hasHostSideEffects(statement.alternate ?? [])) return true;
        continue;
      case "var":
        if (statement.storage === "local" && !statement.pointer) continue;
        return true;
      case "kernel-launch":
      case "asm":
      case "for":
      case "return":
      case "continue":
        return true;
    }
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
): { readonly input: CompiledKernelInput; readonly storageAliases: Readonly<Record<string, string>> } | undefined {
  const scalars: Record<string, number> = {};
  const buffers: Record<string, WgslTypedArray> = {};
  const storageAliases: Record<string, string> = {};
  for (const [index, param] of params.entries()) {
    const arg = statement.args[index];
    if (!arg) return undefined;
    if (param.pointer) {
      const root = arg.kind === "identifier" ? rootIdentifier(arg) : undefined;
      if (!root) return undefined;
      const buffer = input.buffers[root];
      if (!buffer) return undefined;
      buffers[param.name] = buffer;
      if (root !== param.name) storageAliases[param.name] = root;
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
      scalars: { ...input.scalars, ...scalars },
    },
    storageAliases,
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
  return Boolean(flippedLeft && flippedRight !== undefined && guardAllowsOnlyThreadZero(flipComparison(expression.operator), flippedRight));
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

function guardAllowsOnlyThreadZero(operator: string, value: number): boolean {
  if (operator === "==" && value === 0) return true;
  if (operator === "<" && value <= 1) return true;
  if (operator === "<=" && value < 1) return true;
  return false;
}

function flipComparison(operator: string): string {
  if (operator === "<") return ">";
  if (operator === "<=") return ">=";
  if (operator === ">") return "<";
  if (operator === ">=") return "<=";
  return operator;
}
