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

  it("scalarizes static rank-one CuTe layouts without accepting a runtime object graph", () => {
    const compiled = compileCudaLiteKernel(`
      __global__ void cuteStaticLayouts(int *out) {
        using namespace cute;
        constexpr auto compact = make_layout(make_shape(_4{}));
        constexpr auto strided = cute::make_layout(cute::make_shape(cute::Int<4>{}), cute::make_stride(cute::_2{}));
        constexpr auto direct = cute::Layout<cute::_4, cute::_2>{};
        int tid = threadIdx.x;
        if (tid < size(compact)) {
          out[tid] = compact(tid) + strided(tid) + direct(tid) + cute::rank(strided) + cute::cosize(strided);
        }
      }`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.ast.kernels[0]?.body).toHaveLength(2);
    expect(compiled.wgsl).not.toContain("make_layout");
    expect([...result.buffers.out as Int32Array]).toEqual([8, 13, 18, 23]);

    const invalidLayoutSources: readonly [source: string, code: string][] = [
      [`
        __global__ void dynamicCuteLayout(int *out) {
          auto layout = make_layout(make_shape(threadIdx.x));
          out[0] = 0;
        }`, "invalid-cute-static-layout"],
      [`
        __global__ void rankTwoCuteLayout(int *out) {
          auto layout = make_layout(make_shape(_2{}, _2{}));
          out[0] = 0;
        }`, "unsupported-cute-static-layout"],
      [`
        __global__ void escapedCuteLayout(int *out) {
          auto layout = make_layout(make_shape(_4{}));
          int value = layout;
          out[0] = value;
        }`, "unsupported-cute-static-layout-use"],
    ];
    for (const [source, code] of invalidLayoutSources) {
      let error: unknown;
      try {
        parseCudaLite(source);
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(CudaLiteCompilerError);
      expect((error as CudaLiteCompilerError).diagnostics.map((diagnostic) => diagnostic.code)).toContain(code);
    }
  });
});
