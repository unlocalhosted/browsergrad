import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("GPU tensor plan scaffold", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("plans Conv3d forward/backward as primitive tensor IR with liveness", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      forwardOps: string[];
      gxOps: string[];
      gwOps: string[];
      gbOps: string[];
      materializedBuffers: number;
      peakLiveBytes: number;
      hasCustom: boolean;
      legacyWebgpuHasConv3d: boolean;
    }>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np

x = bg.tensor(np.arange(1, 1 + 1*4*3*4*3, dtype=np.float32).reshape(1,4,3,4,3) / 20)
w = bg.tensor(np.arange(1, 1 + 6*2*2*2*2, dtype=np.float32).reshape(6,2,2,2,2) / 40)
b = bg.tensor(np.linspace(-0.1, 0.1, 6, dtype=np.float32))
out = F.conv3d(x, w, b, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2)
gx = bg.func.grad(lambda inp: F.conv3d(inp, w, b, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2).sum())(x)
gw = bg.func.grad(lambda weight: F.conv3d(x, weight, b, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2).sum())(w)
gb = bg.func.grad(lambda bias: F.conv3d(x, w, bias, stride=(1,1,1), padding=(1,1,0), dilation=(1,2,1), groups=2).sum())(b)

fwd_plan = bg.gpu_plan_summary(out)
gx_plan = bg.gpu_plan_summary(gx)
gw_plan = bg.gpu_plan_summary(gw)
gb_plan = bg.gpu_plan_summary(gb)
{
    "forwardOps": fwd_plan["ops"],
    "gxOps": gx_plan["ops"],
    "gwOps": gw_plan["ops"],
    "gbOps": gb_plan["ops"],
    "materializedBuffers": sum(1 for buf in fwd_plan["buffers"] if buf["materialize"]),
    "peakLiveBytes": fwd_plan["peak_live_bytes"],
    "hasCustom": fwd_plan["has_custom_ops"],
    "legacyWebgpuHasConv3d": "CONV3D" in bg.webgpu_supported_opcodes(),
}
`);
    expect(result.forwardOps).toContain("CONV3D");
    expect(result.gxOps).toContain("CONV3D_BACKWARD_INPUT");
    expect(result.gwOps).toContain("CONV3D_BACKWARD_WEIGHT");
    expect(result.gbOps).toContain("CONV3D_BACKWARD_BIAS");
    expect(result.materializedBuffers).toBe(1);
    expect(result.peakLiveBytes).toBeGreaterThan(0);
    expect(result.hasCustom).toBe(false);
    expect(result.legacyWebgpuHasConv3d).toBe(false);
  });

  it("refuses CUSTOM ops in the core GPU tensor plan", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg

q = bg.randn(1, 1, 2, 4)
k = bg.randn(1, 1, 2, 4)
v = bg.randn(1, 1, 2, 4)
out = bg.kernels.flash_attention(q, k, v)
try:
    bg.gpu_plan_summary(out)
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/GpuPlanUnsupported/);
    expect(err).toMatch(/refuses CUSTOM/);
    expect(err).toMatch(/primitive IR/);
  });
});
