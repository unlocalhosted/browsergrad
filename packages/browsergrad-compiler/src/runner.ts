import {
  prepareWgslKernelProgramSequence,
  runWgslKernelProgramSequence,
  type KernelDevice,
  type WgslPreparedKernelSequence,
  type WgslPreparedKernelSequenceRunOptions,
} from "@unlocalhosted/browsergrad-kernels";
import { analyzeCudaLite, lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import { parseCudaLite } from "./parser.js";
import { runCompiledKernelReference } from "./reference.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  createCudaWebGpuExecutionPlan,
  normalizeCudaWebGpuReadback,
  normalizeCudaWebGpuReadbackNames,
  type CudaWebGpuExecutionPlanKind,
} from "./webgpu_orchestration.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type KernelLaunch,
  type ReferenceKernelResult,
} from "./types.js";
import { formatCudaLiteDiagnostics } from "./diagnostics.js";

export type PreparedCompiledKernelWebGpuRunOptions = WgslPreparedKernelSequenceRunOptions;

export interface PreparedCompiledKernelWebGpu {
  readonly kind: CudaWebGpuExecutionPlanKind;
  readonly stepCount: number;
  run(options?: PreparedCompiledKernelWebGpuRunOptions): Promise<ReferenceKernelResult>;
  destroy(): void;
}

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
    {
      ...(options.features === undefined ? {} : { features: options.features }),
      ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
    },
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
    ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
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
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
    compileKernel: compileCudaLiteKernel,
  });
  if (!executionPlan.supported) {
    throw new CudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics);
  }
  const result = await runWgslKernelProgramSequence(
    device,
    executionPlan.steps,
    executionPlan.input,
  );
  return { buffers: normalizeCudaWebGpuReadback(compiled, result.buffers), trace: [] };
}

export async function prepareCompiledKernelWebGpu(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): Promise<PreparedCompiledKernelWebGpu> {
  validateLaunch(launch, compiled.ir.workgroupSize);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
    compileKernel: compileCudaLiteKernel,
  });
  if (!executionPlan.supported) {
    throw new CudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics);
  }
  const prepared = await prepareWgslKernelProgramSequence(
    device,
    executionPlan.steps,
    executionPlan.input,
  );
  return new PreparedCompiledKernelWebGpuImpl(compiled, executionPlan.kind, prepared);
}

class PreparedCompiledKernelWebGpuImpl implements PreparedCompiledKernelWebGpu {
  readonly stepCount: number;
  private destroyed = false;

  constructor(
    private readonly compiled: CompiledCudaLiteKernel,
    readonly kind: PreparedCompiledKernelWebGpu["kind"],
    private readonly prepared: WgslPreparedKernelSequence,
  ) {
    this.stepCount = prepared.stepCount;
  }

  async run(options?: PreparedCompiledKernelWebGpuRunOptions): Promise<ReferenceKernelResult> {
    if (this.destroyed) {
      throw new CudaLiteCompilerError("prepared compiled WebGPU kernel has been destroyed", [{
        code: "prepared-webgpu-kernel-destroyed",
        severity: "error",
        message: "prepared compiled WebGPU kernel has been destroyed",
        span: { start: 0, end: 0, line: 1, column: 1 },
      }]);
    }
    const result = await this.prepared.run(normalizePreparedRunOptions(this.compiled, options));
    return { buffers: normalizeCudaWebGpuReadback(this.compiled, result.buffers), trace: [] };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.prepared.destroy();
  }
}

function normalizePreparedRunOptions(
  compiled: CompiledCudaLiteKernel,
  options: PreparedCompiledKernelWebGpuRunOptions | undefined,
): PreparedCompiledKernelWebGpuRunOptions | undefined {
  if (options?.readback === undefined) return options;
  return { readback: normalizeCudaWebGpuReadbackNames(compiled, options.readback) };
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
