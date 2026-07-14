import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  CudaLiteCompilerError,
  parseCudaLite,
  runCompiledKernelReference,
} from "../../src/index.js";

describe("CUDA-lite parser", () => {
  it("lowers CuTe _N static-integer aliases without accepting wider CuTe syntax", () => {
    const compiled = compileCudaLiteKernel(`
      __global__ void cuteStaticAliases(int *out) {
        constexpr auto one = _1{};
        auto width = cute::_8{};
        auto empty = _0{};
        if (threadIdx.x == 0) out[0] = one + width + empty;
      }`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.out as Int32Array]).toEqual([9]);
    expect(() => parseCudaLite(`
      __global__ void cuteStaticAliasInitializer(int *out) {
        auto value = _4{1};
        out[0] = value;
      }`)).toThrow(/CuTe _N value must use empty braces/);
    expect(() => parseCudaLite(`
      __global__ void cuteStaticAliasOverflow(int *out) {
        auto value = cute::_2147483648{};
        out[0] = value;
      }`)).toThrow(/CuTe _N value must be a non-negative i32 integer constant/);
    let parserError: unknown;
    try {
      parseCudaLite(`
        __global__ void cuteStaticAliasObject(int *out) {
          Layout<_1> layout;
          out[0] = 0;
        }`);
    } catch (error) {
      parserError = error;
    }
    expect(parserError).toBeInstanceOf(CudaLiteCompilerError);
    expect((parserError as CudaLiteCompilerError).diagnostics.map((diagnostic) => diagnostic.code))
      .toContain("unsupported-cute-object");
  });
});
