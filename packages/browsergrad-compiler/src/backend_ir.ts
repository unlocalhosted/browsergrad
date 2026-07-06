import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CudaLiteDiagnostic,
  type KernelIrModule,
} from "./types.js";

const BACKEND_IR_BY_KERNEL = new WeakMap<CompiledCudaLiteKernel, KernelIrModule>();

export function attachInternalBackendIr(
  compiled: CompiledCudaLiteKernel,
  backendIr: KernelIrModule,
): CompiledCudaLiteKernel {
  BACKEND_IR_BY_KERNEL.set(compiled, backendIr);
  return compiled;
}

export function internalBackendIrFor(compiled: CompiledCudaLiteKernel): KernelIrModule {
  const backendIr = BACKEND_IR_BY_KERNEL.get(compiled);
  if (backendIr) return backendIr;
  throw new CudaLiteCompilerError("compiled CUDA-lite kernel is missing its internal backend IR", [{
    code: "internal-backend-ir-missing",
    severity: "error",
    message: "compiled CUDA-lite kernel is missing its internal backend IR",
    span: fallbackSpan(compiled),
  }]);
}

function fallbackSpan(compiled: CompiledCudaLiteKernel): CudaLiteDiagnostic["span"] {
  return compiled.kernelIr.span ?? compiled.semantic.span ?? compiled.ast.span ?? { start: 0, end: 0, line: 1, column: 1 };
}
