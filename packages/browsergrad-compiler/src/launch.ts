import {
  CudaLiteCompilerError,
  type CudaLiteDiagnostic,
  type KernelLaunch,
  type SourceSpan,
} from "./types.js";

const LAUNCH_SPAN: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };

export function createCudaLaunchValidationDiagnostics(
  launch: KernelLaunch,
  workgroupSize: readonly [number, number, number],
): readonly CudaLiteDiagnostic[] {
  const diagnostics: CudaLiteDiagnostic[] = [];
  for (let axis = 0; axis < 3; axis++) {
    const grid = launch.gridDim[axis]!;
    const block = launch.blockDim[axis]!;
    const expected = workgroupSize[axis]!;
    if (!Number.isInteger(grid) || grid <= 0) {
      diagnostics.push(error("launch-grid-dim-invalid", `launch.gridDim[${axis}] must be a positive integer`));
    }
    if (!Number.isInteger(block) || block <= 0) {
      diagnostics.push(error("launch-block-dim-invalid", `launch.blockDim[${axis}] must be a positive integer`));
      continue;
    }
    if (block !== expected) {
      diagnostics.push(error(
        "launch-workgroup-mismatch",
        `launch.blockDim[${axis}] must match compiled workgroupSize[${axis}] (${expected})`,
      ));
    }
  }
  return diagnostics;
}

export function validateCudaKernelLaunch(
  launch: KernelLaunch,
  workgroupSize: readonly [number, number, number],
): void {
  const diagnostics = createCudaLaunchValidationDiagnostics(launch, workgroupSize);
  if (diagnostics.length === 0) return;
  throw new CudaLiteCompilerError(diagnostics[0]?.message ?? "invalid CUDA-lite launch", diagnostics);
}

function error(code: string, message: string): CudaLiteDiagnostic {
  return {
    code,
    severity: "error",
    message,
    span: LAUNCH_SPAN,
  };
}
