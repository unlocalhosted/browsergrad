import {
  runWgslKernelProgram,
  type KernelDevice,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { analyzeCudaLite, lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import { parseCudaLite } from "./parser.js";
import { runCompiledKernelReference } from "./reference.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type CudaLiteExpression,
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
  rejectReferenceOnlyRuntime(compiled);
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
      ...collectExternalPoolNames(compiled.ir.body).map(poolDataName),
    ];
  const result = await runWgslKernelProgram(
    device,
    compiled.wgslProgram,
    {
      buffers,
      ...(input.textures === undefined ? {} : { textures: input.textures }),
      ...(uniforms.byteLength === 0 ? {} : { uniforms: { params: uniforms } }),
      readback,
    },
    {
      dispatchCount: [
        launch.gridDim[0] * launch.blockDim[0],
        launch.gridDim[1] * launch.blockDim[1],
        launch.gridDim[2] * launch.blockDim[2],
      ],
    },
  );
  return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
}

function rejectReferenceOnlyRuntime(compiled: CompiledCudaLiteKernel): void {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return;
  throw new CudaLiteCompilerError("CUDA runtime orchestration is reference-only; WebGPU host orchestration is not implemented yet", [{
    ...diagnostic,
    severity: "error",
    message: "CUDA runtime orchestration is reference-only; WebGPU host orchestration is not implemented yet",
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
    ...collectExternalPoolNames(compiled.ir.body).map((name) => ({ name, span: compiled.ir.body[0]?.span ?? compiled.ir.params[0]?.span ?? { start: 0, end: 0, line: 1, column: 1 } })),
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
  for (const poolName of collectExternalPoolNames(compiled.ir.body)) {
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

function collectExternalPoolNames(statements: readonly CudaLiteStatement[]): readonly string[] {
  const pools = new Set<string>();
  const visit = (expression: CudaLiteExpression): void => {
    if (expression.kind === "call") {
      const callName = expression.kind === "call" && expression.callee.kind === "identifier"
        ? expression.callee.name
        : undefined;
      if ((callName === "deviceAllocate" || callName === "streamOrderedAllocate") && expression.args.length === 2) {
        const first = expression.args[0];
        if (first?.kind === "unary" && first.operator === "&" && first.argument.kind === "identifier") {
          pools.add(first.argument.name);
        }
      }
      visit(expression.callee);
      for (const arg of expression.args) visit(arg);
      return;
    }
    if (expression.kind === "cast") visit(expression.expression);
    else if (expression.kind === "member") visit(expression.object);
    else if (expression.kind === "index") {
      visit(expression.target);
      visit(expression.index);
    } else if (expression.kind === "unary" || expression.kind === "update") visit(expression.argument);
    else if (expression.kind === "binary") {
      visit(expression.left);
      visit(expression.right);
    } else if (expression.kind === "conditional") {
      visit(expression.condition);
      visit(expression.consequent);
      visit(expression.alternate);
    } else if (expression.kind === "assignment") {
      visit(expression.left);
      visit(expression.right);
    }
  };
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.init) visit(item.init);
      if (item.kind === "expr") visit(item.expression);
      if (item.kind === "if") {
        visit(item.condition);
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.init) visit(item.init.init);
        else if (item.init && item.init.kind !== "var") visit(item.init);
        if (item.condition) visit(item.condition);
        if (item.update) visit(item.update);
        walk(item.body);
      }
      if (item.kind === "return" && item.value) visit(item.value);
    }
  };
  walk(statements);
  return [...pools].sort();
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
