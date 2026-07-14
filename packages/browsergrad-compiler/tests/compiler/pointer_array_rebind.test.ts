import { describe, expect, it } from "vitest";
import {
  compileCudaLiteKernelForWebGpu,
  runCompiledKernelSemanticReference,
} from "../../src/index";

describe("CUDA-lite compiler: Pointer-array rebinding", () => {
  it("evaluates a side-effecting assignment index once before its ordered rebind", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ uint pointer_array_assignment_index_helper(uint *ptr, uint add) {
  atomicAdd(ptr, add);
  return 1u;
}

__global__ void pointer_array_assignment_index_once(uint *storage) {
  uint *ptrs[2];
  ptrs[0] = storage;
  ptrs[1] = storage + 1;
  ptrs[pointer_array_assignment_index_helper(storage, 1u)] = storage + 2;
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl!.match(/\bpointer_array_assignment_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/var bg__bg_pointer_array_index_\d+_\d+: u32 = pointer_array_assignment_index_helper\(/u);
    const rebinds = compiled.kernelIr.operations.filter((operation) => operation.kind === "pointer-array-rebind");
    expect(rebinds).toHaveLength(3);
    const dynamicRebindIndex = compiled.kernelIr.operations.findIndex((operation) =>
      operation.kind === "pointer-array-rebind" && operation.slot.kind === "symbol"
    );
    const dynamicIndexDeclaration = compiled.kernelIr.operations.findIndex((operation) =>
      operation.kind === "declare" && operation.target.name.includes("pointer.array.index")
    );
    expect(dynamicRebindIndex).toBeGreaterThan(dynamicIndexDeclaration);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { storage: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...result.buffers.storage as Uint32Array]).toEqual([1, 0, 0, 0]);
  });
});
