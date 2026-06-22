import {
  runWgslKernelProgram,
  type KernelDevice,
} from "@unlocalhosted/browsergrad-kernels";
import { analyzeCudaLite, lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import { parseCudaLite } from "./parser.js";
import { runCompiledKernelReference } from "./reference.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
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
  return {
    ast,
    analysis,
    ir,
    wgsl: emitted.wgsl,
    wgslProgram: emitted.program,
    diagnostics: analysis.diagnostics,
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
  const uniforms = packScalarParams(compiled, input);
  const result = await runWgslKernelProgram(
    device,
    compiled.wgslProgram,
    {
      buffers: input.buffers,
      ...(uniforms.byteLength === 0 ? {} : { uniforms: { params: uniforms } }),
      readback: input.readback ??
        compiled.ir.params.filter((param) => param.pointer && !param.constant).map((param) => param.name),
    },
    {
      dispatchCount: [
        launch.gridDim[0] * launch.blockDim[0],
        launch.gridDim[1] * launch.blockDim[1],
        launch.gridDim[2] * launch.blockDim[2],
      ],
    },
  );
  return { buffers: result.buffers, trace: [] };
}

function packScalarParams(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Uint8Array {
  const scalarParams = compiled.ir.params.filter((param) => !param.pointer);
  if (scalarParams.length === 0) return new Uint8Array(0);
  const bytes = new Uint8Array(Math.max(16, scalarParams.length * 4));
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < scalarParams.length; i++) {
    const param = scalarParams[i]!;
    const value = input.scalars?.[param.name];
    if (value === undefined) {
      throw new CudaLiteCompilerError(`missing scalar input '${param.name}'`, [{
        code: "missing-scalar",
        severity: "error",
        message: `missing scalar input '${param.name}'`,
        span: param.span,
      }]);
    }
    const offset = i * 4;
    if (param.valueType === "int") view.setInt32(offset, Math.trunc(value), true);
    else if (param.valueType === "uint") view.setUint32(offset, Math.trunc(value), true);
    else view.setFloat32(offset, value, true);
  }
  return bytes;
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
