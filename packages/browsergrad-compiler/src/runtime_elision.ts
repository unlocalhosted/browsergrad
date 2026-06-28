import { expressionName, rootIdentifier } from "./analyzer.js";
import { CUDA_INTRINSICS } from "./intrinsics.js";
import type {
  CompiledCudaLiteKernel,
  CudaLiteDeviceFunction,
  CudaLiteDeviceGlobal,
  CudaLiteExpression,
  CudaLiteKernel,
  CudaLiteParam,
  CudaLiteStatement,
} from "./types.js";

const SIDE_EFFECT_FREE_CALLS = new Set([
  "max",
  "min",
  "printf",
  "sizeof",
  "alignof",
  "__syncthreads",
  "__syncwarp",
  ...CUDA_INTRINSICS.map((intrinsic) => intrinsic.name),
]);

const SIDE_EFFECT_FREE_MEMBER_CALLS = new Set([
  "any",
  "all",
  "ballot",
  "shfl",
  "shfl_down",
  "shfl_up",
  "shfl_xor",
  "sync",
  "thread_rank",
  "size",
  "meta_group_rank",
  "meta_group_size",
]);

export function deviceLaunchTreeIsExternallySilent(compiled: CompiledCudaLiteKernel): boolean {
  const entry = compiled.ast.kernels.find((kernel) => kernel.name === compiled.ir.name);
  if (!entry) return false;
  const externalRoots = initialExternalRoots(entry, compiled.ast.deviceGlobals);
  return !launchableHasExternalWrite(compiled, entry, externalRoots, new Set());
}

function launchableHasExternalWrite(
  compiled: CompiledCudaLiteKernel,
  launchable: CudaLiteKernel | CudaLiteDeviceFunction,
  externalRoots: ReadonlySet<string>,
  visiting: Set<string>,
): boolean {
  const key = `${launchable.name}:${[...externalRoots].sort().join(",")}`;
  if (visiting.has(key)) return false;
  visiting.add(key);

  const aliases = new Set<string>();
  const hasExternal = (expression: CudaLiteExpression | undefined): boolean => {
    if (!expression) return false;
    const root = rootIdentifier(expression);
    if (root !== undefined && (externalRoots.has(root) || aliases.has(root))) return true;
    switch (expression.kind) {
      case "assignment":
        return hasExternal(expression.left) || hasExternal(expression.right);
      case "conditional":
        return hasExternal(expression.consequent) || hasExternal(expression.alternate);
      case "sequence":
        return expression.expressions.some(hasExternal);
      case "initializer":
        return expression.elements.some(hasExternal);
      default:
        return false;
    }
  };

  const expressionHasWrite = (expression: CudaLiteExpression): boolean => {
    switch (expression.kind) {
      case "assignment": {
        const leftHasExternal = hasExternal(expression.left);
        const rightHasExternal = hasExternal(expression.right);
        const rightHasWrite = expressionHasWrite(expression.right);
        updateExternalAlias(expression.left, rightHasExternal);
        return leftHasExternal || rightHasWrite;
      }
      case "update":
        return hasExternal(expression.argument);
      case "call":
        return callHasExternalWrite(compiled, expression, hasExternal, expressionHasWrite, visiting);
      case "initializer":
        return expression.elements.some(expressionHasWrite);
      case "cast":
        return expressionHasWrite(expression.expression);
      case "member":
        return expressionHasWrite(expression.object);
      case "unary":
        return expressionHasWrite(expression.argument);
      case "index":
        return expressionHasWrite(expression.target) || expressionHasWrite(expression.index);
      case "binary":
        return expressionHasWrite(expression.left) || expressionHasWrite(expression.right);
      case "conditional":
        return expressionHasWrite(expression.condition) ||
          expressionHasWrite(expression.consequent) ||
          expressionHasWrite(expression.alternate);
      case "sequence":
        return expression.expressions.some(expressionHasWrite);
      case "identifier":
      case "number":
      case "string":
        return false;
    }
  };

  const updateExternalAlias = (target: CudaLiteExpression, aliasesExternal: boolean): void => {
    if (target.kind !== "identifier") return;
    if (aliasesExternal) aliases.add(target.name);
  };

  const statementsHaveWrite = (statements: readonly CudaLiteStatement[]): boolean => {
    for (const statement of statements) {
      switch (statement.kind) {
        case "var":
          if (statement.init && expressionHasWrite(statement.init)) return true;
          if (statement.pointer && statement.init && hasExternal(statement.init)) aliases.add(statement.name);
          break;
        case "dim3":
          if (statement.args.some(expressionHasWrite)) return true;
          break;
        case "expr":
          if (expressionHasWrite(statement.expression)) return true;
          break;
        case "if":
          if (expressionHasWrite(statement.condition) ||
            statementsHaveWrite(statement.consequent) ||
            statementsHaveWrite(statement.alternate ?? [])) return true;
          break;
        case "for":
          if (statement.init?.kind === "var") {
            if (statement.init.init && expressionHasWrite(statement.init.init)) return true;
            if (statement.init.pointer && statement.init.init && hasExternal(statement.init.init)) aliases.add(statement.init.name);
          } else if (statement.init && expressionHasWrite(statement.init)) {
            return true;
          }
          if ((statement.condition && expressionHasWrite(statement.condition)) ||
            statementsHaveWrite(statement.body) ||
            (statement.update && expressionHasWrite(statement.update))) return true;
          break;
        case "while":
        case "do-while":
          if (expressionHasWrite(statement.condition) || statementsHaveWrite(statement.body)) return true;
          break;
        case "block":
          if (statementsHaveWrite(statement.body)) return true;
          break;
        case "return":
          if (statement.value && expressionHasWrite(statement.value)) return true;
          break;
        case "asm":
          if (hasExternal(statement.output) || (statement.outputs ?? []).some(hasExternal)) return true;
          if (statement.inputs.some(expressionHasWrite)) return true;
          break;
        case "kernel-launch":
          if (kernelLaunchHasExternalWrite(compiled, statement, hasExternal, visiting)) return true;
          break;
        case "cooperative-group":
        case "continue":
        case "break":
          break;
      }
    }
    return false;
  };

  const hasWrite = statementsHaveWrite(launchable.body);
  visiting.delete(key);
  return hasWrite;
}

function callHasExternalWrite(
  compiled: CompiledCudaLiteKernel,
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  hasExternal: (expression: CudaLiteExpression | undefined) => boolean,
  expressionHasWrite: (expression: CudaLiteExpression) => boolean,
  visiting: Set<string>,
): boolean {
  const name = expressionName(expression.callee);
  if (name === undefined && expression.callee.kind === "member" && SIDE_EFFECT_FREE_MEMBER_CALLS.has(expression.callee.property)) {
    return expressionHasWrite(expression.callee.object) || expression.args.some(expressionHasWrite);
  }
  if (name !== undefined && isRuntimeMemoryMutationCall(name)) return true;
  if (name !== undefined && isExternalMutationCall(name)) return hasExternal(expression.args[0]) || expression.args.some(expressionHasWrite);
  if (name === "surf1Dwrite" || name === "surf2Dwrite" || name === "surf2DLayeredwrite" || name === "surf3Dwrite") {
    return expression.args.some(hasExternal) || expression.args.some(expressionHasWrite);
  }
  if (name !== undefined) {
    const fn = compiled.ast.functions.find((candidate) => candidate.name === name);
    if (fn) {
      return launchableHasExternalWrite(compiled, fn, externalRootsForCall(fn.params, expression.args, hasExternal, compiled.ast.deviceGlobals), visiting);
    }
  }
  if (expression.args.some(expressionHasWrite)) return true;
  if (name !== undefined && SIDE_EFFECT_FREE_CALLS.has(name)) return false;
  return name === undefined || expression.args.some(hasExternal);
}

function kernelLaunchHasExternalWrite(
  compiled: CompiledCudaLiteKernel,
  statement: Extract<CudaLiteStatement, { kind: "kernel-launch" }>,
  hasExternal: (expression: CudaLiteExpression | undefined) => boolean,
  visiting: Set<string>,
): boolean {
  const kernel = compiled.ast.kernels.find((candidate) => candidate.name === statement.callee) ??
    compiled.ast.functions.find((candidate) => candidate.name === statement.callee);
  if (!kernel) return true;
  return launchableHasExternalWrite(compiled, kernel, externalRootsForCall(kernel.params, statement.args, hasExternal, compiled.ast.deviceGlobals), visiting);
}

function externalRootsForCall(
  params: readonly CudaLiteParam[],
  args: readonly CudaLiteExpression[],
  hasExternal: (expression: CudaLiteExpression | undefined) => boolean,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): ReadonlySet<string> {
  const roots = new Set(deviceGlobals.map((global) => global.name));
  for (const [index, param] of params.entries()) {
    if (param.pointer && !param.constant && hasExternal(args[index])) roots.add(param.name);
  }
  return roots;
}

function initialExternalRoots(
  kernel: CudaLiteKernel,
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): ReadonlySet<string> {
  return new Set([
    ...deviceGlobals.map((global) => global.name),
    ...kernel.params
      .filter((param) => (param.pointer && !param.constant) || param.valueType === "surface2d")
      .map((param) => param.name),
  ]);
}

function isExternalMutationCall(name: string): boolean {
  return name === "atomicAdd" ||
    name === "atomicAdd_system" ||
    name === "atomicSub" ||
    name === "atomicSub_system" ||
    name === "atomicMin" ||
    name === "atomicMin_system" ||
    name === "atomicMax" ||
    name === "atomicMax_system" ||
    name === "atomicAnd" ||
    name === "atomicAnd_system" ||
    name === "atomicOr" ||
    name === "atomicOr_system" ||
    name === "atomicXor" ||
    name === "atomicXor_system" ||
    name === "atomicInc" ||
    name === "atomicInc_system" ||
    name === "atomicDec" ||
    name === "atomicDec_system" ||
    name === "atomicExch" ||
    name === "atomicExch_system" ||
    name === "atomicCAS" ||
    name === "atomicCAS_system";
}

function isRuntimeMemoryMutationCall(name: string): boolean {
  return name === "deviceAllocate" ||
    name === "streamOrderedAllocate" ||
    name === "cudaMemcpy" ||
    name === "cudaMemcpyAsync" ||
    name === "cudaMemcpyPeerAsync";
}
