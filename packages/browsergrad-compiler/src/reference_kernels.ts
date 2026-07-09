import { collectKernelLaunchCallees } from "./ast_queries.js";
import { lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import type {
  CompiledCudaLiteKernel,
  CudaLiteDeviceFunction,
  CudaLiteKernel,
  KernelIrModule,
} from "./types.js";

export function collectReferenceKernels(compiled: CompiledCudaLiteKernel): Map<string, CudaLiteKernel> {
  const kernels = new Map(compiled.ast.kernels.map((kernel) => [kernel.name, kernel] as const));
  const launched = new Set<string>();
  for (const kernel of compiled.ast.kernels) {
    for (const name of collectKernelLaunchCallees(kernel.body)) launched.add(name);
  }
  for (const fn of compiled.ast.functions) {
    for (const name of collectKernelLaunchCallees(fn.body)) launched.add(name);
  }
  for (const fn of compiled.ast.functions) {
    if (launched.has(fn.name)) kernels.set(fn.name, referenceDeviceFunctionAsKernel(fn));
  }
  return kernels;
}

export function collectReferenceFunctions(functions: readonly CudaLiteDeviceFunction[]): Map<string, readonly CudaLiteDeviceFunction[]> {
  const byName = new Map<string, CudaLiteDeviceFunction[]>();
  for (const fn of functions) {
    const overloads = byName.get(fn.name);
    if (overloads) overloads.push(fn);
    else byName.set(fn.name, [fn]);
  }
  return byName;
}

export function resolveReferenceDeviceFunction(
  functions: ReadonlyMap<string, readonly CudaLiteDeviceFunction[]>,
  name: string,
  argCount: number,
): CudaLiteDeviceFunction | undefined {
  const overloads = functions.get(name);
  if (!overloads || overloads.length === 0) return undefined;
  return overloads.find((fn) => fn.params.length === argCount) ?? overloads[0];
}

export function referenceKernelIrFor(compiled: CompiledCudaLiteKernel): KernelIrModule {
  return lowerAnalyzedCudaLiteToKernelIr(compiled.analysis, {
    workgroupSize: compiled.kernelIr.workgroupSize,
    ...(compiled.dynamicSharedMemory === undefined ? {} : { dynamicSharedMemory: compiled.dynamicSharedMemory }),
  });
}

function referenceDeviceFunctionAsKernel(fn: CudaLiteDeviceFunction): CudaLiteKernel {
  return {
    kind: "kernel",
    name: fn.name,
    params: fn.params,
    body: fn.body,
    span: fn.span,
  };
}
