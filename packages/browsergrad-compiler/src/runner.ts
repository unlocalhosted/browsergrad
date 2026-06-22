import {
  runWgslKernelProgram,
  runWgslKernelProgramSequence,
  type KernelDevice,
  type WgslKernelSequenceStep,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import { analyzeCudaLite, expressionName, lowerAnalyzedCudaLiteToKernelIr, rootIdentifier } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import { parseCudaLite } from "./parser.js";
import { runCompiledKernelReference } from "./reference.js";
import { createCudaGridSyncPhasePlan, createCudaRuntimePlan } from "./runtime_plan.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type CudaLiteExpression,
  type CudaLiteKernelLaunchStatement,
  type CudaLiteStatement,
  type KernelLaunch,
  type ReferenceKernelResult,
} from "./types.js";
import { formatCudaLiteDiagnostics } from "./diagnostics.js";

export function compileCudaLiteKernel(
  source: string,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteKernel {
  const ast = parseCudaLite(source);
  const analysis = analyzeCudaLite(ast, options);
  const errors = analysis.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  if (errors.length > 0) {
    throw new CudaLiteCompilerError(
      `CUDA-lite compile failed\n${formatCudaLiteDiagnostics(source, errors)}`,
      errors,
    );
  }
  const ir = lowerAnalyzedCudaLiteToKernelIr(analysis, options);
  const emitted = emitKernelIrWgsl(
    ir,
    options.features === undefined ? {} : { features: options.features },
  );
  const loweringPlan = createCudaLoweringPlan(analysis.diagnostics);
  return {
    ast,
    analysis,
    ir,
    wgsl: emitted.wgsl,
    wgslProgram: emitted.program,
    diagnostics: analysis.diagnostics,
    loweringPlan,
  };
}

export { runCompiledKernelReference };

export async function runCompiledKernelWebGpu(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): Promise<ReferenceKernelResult> {
  validateLaunch(launch, compiled.ir.workgroupSize);
  const gridSyncPhasePlan = createCudaGridSyncPhasePlan(compiled.ir);
  if (gridSyncPhasePlan.supported && gridSyncPhasePlan.modules.length > 1) {
    const wgslInput = createWgslRunInput(compiled, input);
    const dispatchCount = dispatchCountForLaunch(launch);
    const programs = gridSyncPhasePlan.modules.map((module) =>
      emitKernelIrWgsl(module, { features: featureOptionsFor(module.requiredFeatures) }).program
    );
    const result = await runWgslKernelProgramSequence(
      device,
      programs.map((program) => ({ program, launch: { dispatchCount } })),
      wgslInput,
    );
    return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
  }
  const dynamicResult = await tryRunHostLiftedDynamicLaunch(device, compiled, input, launch);
  if (dynamicResult) return dynamicResult;
  rejectReferenceOnlyRuntime(compiled);
  const wgslInput = createWgslRunInput(compiled, input);
  const dispatchCount = dispatchCountForLaunch(launch);
  const result = await runWgslKernelProgram(device, compiled.wgslProgram, wgslInput, { dispatchCount });
  return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
}

interface HostLiftedLaunch {
  readonly statement: CudaLiteKernelLaunchStatement;
  readonly env: ReadonlyMap<string, HostEvalValue>;
}

type HostEvalValue = number | readonly [number, number, number];

async function tryRunHostLiftedDynamicLaunch(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): Promise<ReferenceKernelResult | undefined> {
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (!runtimePlan.operations.some((operation) => operation.kind === "device-launch")) return undefined;
  if (!runtimePlan.operations.every((operation) => operation.kind === "device-launch" || operation.kind === "device-sync")) {
    return undefined;
  }
  if (launch.gridDim.some((axis) => axis !== 1)) return undefined;

  const launches = collectHostLiftedLaunches(compiled.ir.body, input, launch);
  if (launches.length === 0) return undefined;

  const parentInput = createWgslRunInput(compiled, input);
  const buffers: Record<string, WgslTypedArray> = { ...parentInput.buffers };
  const steps: WgslKernelSequenceStep[] = [{
    program: compiled.wgslProgram,
    launch: { dispatchCount: dispatchCountForLaunch(launch) },
    ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
  }];

  for (const item of launches) {
    const childKernel = compiled.ast.kernels.find((kernel) => kernel.name === item.statement.callee);
    if (!childKernel) return undefined;
    const childBlock = evaluateLaunchVector(item.statement.block, item.env, input);
    const childGrid = evaluateLaunchVector(item.statement.grid, item.env, input);
    if (!childBlock || !childGrid) return undefined;
    let childCompiled: CompiledCudaLiteKernel;
    try {
      childCompiled = compileCudaLiteKernel(compiled.ast.source, {
        kernelName: childKernel.name,
        features: featureOptionsFor(compiled.ir.requiredFeatures),
        workgroupSize: childBlock,
      });
    } catch {
      return undefined;
    }
    const childRuntime = createCudaRuntimePlan(childCompiled);
    if (!childRuntime.operations.every((operation) => operation.kind === "device-sync")) return undefined;
    const childInput = createChildKernelInput(childKernel.params, item.statement, item.env, input);
    if (!childInput) return undefined;
    const childWgslInput = createWgslRunInput(childCompiled, childInput.input);
    for (const [name, value] of Object.entries(childWgslInput.buffers)) {
      buffers[childInput.storageAliases[name] ?? name] = value;
    }
    steps.push({
      program: childCompiled.wgslProgram,
      launch: { dispatchCount: dispatchCountForLaunch({ gridDim: childGrid, blockDim: childBlock }) },
      storageAliases: childInput.storageAliases,
      ...(childWgslInput.uniforms === undefined ? {} : { uniforms: childWgslInput.uniforms }),
    });
  }

  const result = await runWgslKernelProgramSequence(
    device,
    steps,
    {
      buffers,
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      readback: parentInput.readback,
    },
  );
  return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
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
  params: readonly CompiledCudaLiteKernel["ir"]["params"][number][],
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

function createWgslRunInput(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): {
  readonly buffers: Record<string, WgslTypedArray>;
  readonly textures?: NonNullable<CompiledKernelInput["textures"]>;
  readonly uniforms?: { readonly params: Uint8Array };
  readonly readback: readonly string[];
} {
  const uniforms = packScalarParams(compiled, input);
  const buffers = {
    ...input.buffers,
    ...surfaceBufferInputs(compiled, input),
    ...memoryPoolBufferInputs(compiled, input),
    ...constantBufferInputs(compiled, input),
  };
  const readback = input.readback ??
    [
      ...compiled.ir.params
        .filter((param) => (param.pointer && !param.constant) || param.valueType === "surface2d")
        .map((param) => param.valueType === "devicepool" ? poolDataName(param.name) : param.name),
      ...collectExternalDevicePoolNames(compiled.ir.body).map(poolDataName),
    ];
  return {
    buffers,
    ...(input.textures === undefined ? {} : { textures: input.textures }),
    ...(uniforms.byteLength === 0 ? {} : { uniforms: { params: uniforms } }),
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

function rejectReferenceOnlyRuntime(compiled: CompiledCudaLiteKernel): void {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return;
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (runtimePlan.operations.every((operation) => operation.kind === "device-sync")) return;
  const labels = [...new Set(runtimePlan.operations.map((operation) => operation.kind))].join(", ");
  const message = labels.length > 0
    ? `CUDA runtime orchestration is reference-only (${labels}); WebGPU host orchestration is not implemented yet`
    : "CUDA runtime orchestration is reference-only; WebGPU host orchestration is not implemented yet";
  throw new CudaLiteCompilerError(message, [{
    ...diagnostic,
    severity: "error",
    message,
  }]);
}

function packScalarParams(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Uint8Array {
  const scalarParams = [
    ...compiled.ir.params.filter((param) => !param.pointer && param.valueType !== "surface2d"),
    ...compiled.ir.constants.filter((constant) => constant.dimensions.length === 0),
    ...compiled.ir.params.filter((param) => param.valueType === "surface2d").flatMap((param) => [
      { name: `${param.name}_width`, valueType: "uint" as const, surface: param.name, span: param.span },
      { name: `${param.name}_height`, valueType: "uint" as const, surface: param.name, span: param.span },
    ]),
  ];
  if (scalarParams.length === 0) return new Uint8Array(0);
  const bytes = new Uint8Array(Math.max(16, scalarParams.length * 4));
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < scalarParams.length; i++) {
    const param = scalarParams[i]!;
    const value = "surface" in param
      ? (param.name.endsWith("_width") ? input.surfaces?.[param.surface]?.width : input.surfaces?.[param.surface]?.height)
      : "pointer" in param
      ? input.scalars?.[param.name]
      : input.constants?.[param.name];
    if (value === undefined) {
      const kind = "surface" in param ? "surface input" : "pointer" in param ? "scalar input" : "constant input";
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
    else if (param.valueType === "half") view.setUint16(offset, float16Bits(value), true);
    else if (param.valueType === "bool") view.setUint32(offset, value ? 1 : 0, true);
    else view.setFloat32(offset, value, true);
  }
  return bytes;
}

function surfaceBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const surface of compiled.ir.params.filter((param) => param.valueType === "surface2d")) {
    const value = input.surfaces?.[surface.name];
    if (!value) {
      throw new CudaLiteCompilerError(`missing surface input '${surface.name}'`, [{
        code: "missing-surface",
        severity: "error",
        message: `missing surface input '${surface.name}'`,
        span: surface.span,
      }]);
    }
    out[surface.name] = value.data;
  }
  return out;
}

function memoryPoolBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const pool of [
    ...compiled.ir.params.filter(isDevicePoolParam).map((param) => ({ name: param.name, span: param.span })),
    ...collectExternalDevicePoolNames(compiled.ir.body).map((name) => ({ name, span: compiled.ir.body[0]?.span ?? compiled.ir.params[0]?.span ?? { start: 0, end: 0, line: 1, column: 1 } })),
  ]) {
    const value = input.memoryPools?.[pool.name];
    if (!value) {
      throw new CudaLiteCompilerError(`missing memory pool input '${pool.name}'`, [{
        code: "missing-memory-pool",
        severity: "error",
        message: `missing memory pool input '${pool.name}'`,
        span: pool.span,
      }]);
    }
    if (!(value.data instanceof Uint32Array)) {
      throw new CudaLiteCompilerError(`memory pool '${pool.name}' expects Uint32Array data`, [{
        code: "invalid-memory-pool",
        severity: "error",
        message: `memory pool '${pool.name}' expects Uint32Array data`,
        span: pool.span,
      }]);
    }
    const offset = value.offset ?? new Uint32Array([0]);
    if (!(offset instanceof Uint32Array) || offset.length < 1) {
      throw new CudaLiteCompilerError(`memory pool '${pool.name}' offset expects Uint32Array length >= 1`, [{
        code: "invalid-memory-pool",
        severity: "error",
        message: `memory pool '${pool.name}' offset expects Uint32Array length >= 1`,
        span: pool.span,
      }]);
    }
    out[poolDataName(pool.name)] = value.data;
    out[poolOffsetName(pool.name)] = offset;
  }
  return out;
}

function normalizePoolReadback(
  compiled: CompiledCudaLiteKernel,
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = { ...buffers };
  for (const pool of compiled.ir.params.filter(isDevicePoolParam)) {
    const data = buffers[poolDataName(pool.name)];
    if (data) out[pool.name] = data;
  }
  for (const poolName of collectExternalDevicePoolNames(compiled.ir.body)) {
    const data = buffers[poolDataName(poolName)];
    if (data) out[poolName] = data;
  }
  return out;
}

function constantBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const constant of compiled.ir.constants.filter((item) => item.dimensions.length > 0)) {
    const value = input.constants?.[constant.name];
    if (!value || typeof value === "number") {
      throw new CudaLiteCompilerError(`missing constant buffer '${constant.name}'`, [{
        code: "missing-constant",
        severity: "error",
        message: `missing constant buffer '${constant.name}'`,
        span: constant.span,
      }]);
    }
    out[constant.name] = value;
  }
  return out;
}

function isDevicePoolParam(param: { readonly pointer: boolean; readonly valueType: string }): boolean {
  return param.pointer && param.valueType === "devicepool";
}

function poolDataName(name: string): string {
  return `${name}_pool`;
}

function poolOffsetName(name: string): string {
  return `${name}_offset`;
}

function float16Bits(value: number): number {
  const half = new Float16Array([value]);
  return new Uint16Array(half.buffer)[0] ?? 0;
}

function validateLaunch(launch: KernelLaunch, workgroupSize: readonly [number, number, number]): void {
  for (let axis = 0; axis < 3; axis++) {
    if (launch.blockDim[axis] !== workgroupSize[axis]) {
      throw new CudaLiteCompilerError("launch.blockDim must match compiled workgroupSize", [{
        code: "launch-workgroup-mismatch",
        severity: "error",
        message: "launch.blockDim must match compiled workgroupSize",
        span: { start: 0, end: 0, line: 1, column: 1 },
      }]);
    }
  }
}
