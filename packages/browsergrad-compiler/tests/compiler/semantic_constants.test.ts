import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  compileCudaLiteKernel,
  parseCudaLite,
  runCompiledKernelSemanticReference,
} from "../../src/index.js";

describe("CUDA-lite compiler: semantic constants", () => {
  it("preserves constexpr declarations and eliminates resolved branches", () => {
    const source = `
__global__ void constexpr_branch(int *out) {
  constexpr bool enabled = ((4 / 2) == 2) && (3 < 4);
  constexpr int count = enabled ? 2 : 1;
  if constexpr (enabled && count > 1) {
    out[0] = 7;
  } else {
    out[0] = 9;
  }
}`;
    const ast = parseCudaLite(source);
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(ast.kernels[0]?.body.filter((statement) => statement.kind === "var")).toEqual([
      expect.objectContaining({ name: "enabled", constant: true }),
      expect.objectContaining({ name: "count", constant: true }),
    ]);
    expect(JSON.stringify(compiled.kernelIr.operations)).not.toContain('"kind":"branch"');
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.out as Int32Array]).toEqual([7]);
  });

  it("removes host-orchestration diagnostics for dead device launches", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(int *out) { out[0] = 99; }
__global__ void parent(int *out) {
  constexpr bool launch_child = false;
  if constexpr (launch_child) {
    child<<<1, 1>>>(out);
    cudaDeviceSynchronize();
  }
  out[0] = 5;
}`, { kernelName: "parent", workgroupSize: [1, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("cuda-dynamic-launch-host-orchestration");
    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
    expect(compiled.kernelIr.operations.some((operation) => operation.kind === "device-launch")).toBe(false);
  });
});
