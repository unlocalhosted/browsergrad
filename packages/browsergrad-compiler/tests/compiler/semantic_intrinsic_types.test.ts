import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  compileCudaLiteKernel,
  runCompiledKernelSemanticReference,
} from "../../src/index.js";

describe("CUDA-lite compiler: semantic intrinsic types", () => {
  it("preserves float2 return type for half2 conversion over local packed words", () => {
    const compiled = compileCudaLiteKernel(`
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
__global__ void local_half2_to_float2(float *out) {
  uint words[1];
  words[0] = 0x40003c00u;
  float2 value = __half22float2(HALF2(words[0]));
  out[0] = value.x;
  out[1] = value.y;
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const declaration = compiled.kernelIr.operations.find((operation) =>
      operation.kind === "declare" && operation.target.name === "value");
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(declaration).toMatchObject({ kind: "declare", init: { kind: "call", valueType: "float2" } });
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
  });
});
