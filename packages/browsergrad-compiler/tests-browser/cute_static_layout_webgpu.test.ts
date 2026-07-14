import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  type KernelDevice,
  createDevice,
} from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernel,
  runCompiledKernelWebGpu,
} from "../src/index";

const STATIC_RANK1_LAYOUT = `
__global__ void staticRank1Layout(int *out) {
  using namespace cute;
  constexpr auto compact = make_layout(make_shape(_4{}));
  constexpr auto strided = cute::make_layout(cute::make_shape(cute::Int<4>{}), cute::make_stride(cute::_2{}));
  int tid = threadIdx.x;
  if (tid < size(compact)) {
    out[tid] = compact(tid) + strided(tid) + cute::rank(strided) + cute::cosize(strided);
  }
}
`;

describe("real WebGPU — static rank-one CuTe layouts", () => {
  let device: KernelDevice | undefined;

  beforeAll(async () => {
    if (!navigator.gpu) return;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    device = await createDevice({ device: await adapter.requestDevice() });
  });

  afterAll(() => {
    device?.clearCache();
    device?.gpu.destroy();
  });

  it("executes scalarized static shape, stride, and layout queries", async () => {
    if (!device) return;
    const compiled = compileCudaLiteKernel(STATIC_RANK1_LAYOUT, { workgroupSize: [4, 1, 1] });
    const result = await runCompiledKernelWebGpu(
      device,
      compiled,
      { buffers: { out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect([...result.buffers.out as Int32Array]).toEqual([8, 11, 14, 17]);
  });
});
