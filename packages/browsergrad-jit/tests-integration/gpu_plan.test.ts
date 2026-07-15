import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { DENSE_PERMUTATION_VIEW_COPY_FIXTURES } from "../../../test-support/dense-permutation-view-copy-fixtures";
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

  it("emits one closed semantic PERMUTE request from the post-fusion plan", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      schema: string;
      version: { major: number; minor: number };
      envelopeKeys: string[];
      requestKeys: string[];
      request: {
        kind: string;
        valueId: number;
        inputShape: string[];
        axes: number[];
        dtype: string;
      };
      permuteValueId: number;
      permuteArgErased: boolean;
      ops: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._gpu_plan import build_gpu_execution_submission

x = bg.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
out = bg.exp(x + x).permute(1, 0)
submission = build_gpu_execution_submission(out._uop)
plan = submission.plan_summary()
semantic = submission.semantic_request_summary()
request = semantic["requests"][0]
permute_step = next(step for step in plan["steps"] if step["op"] == "PERMUTE")
{
    "schema": semantic["schema"],
    "version": semantic["version"],
    "envelopeKeys": sorted(semantic.keys()),
    "requestKeys": sorted(request.keys()),
    "request": request,
    "permuteValueId": int(permute_step["value_id"]),
    "permuteArgErased": permute_step["arg"] is None,
    "ops": plan["ops"],
}
`);

    expect(result).toMatchObject({
      schema: "browsergrad.jit.tensor-plan-semantic-requests",
      version: { major: 1, minor: 0 },
      envelopeKeys: ["requests", "schema", "version"],
      requestKeys: ["axes", "dtype", "inputShape", "kind", "valueId"],
      request: {
        kind: "dense-permutation-view-copy",
        inputShape: ["2", "3"],
        axes: [1, 0],
        dtype: "f32",
      },
    });
    expect(result.ops).toContain("FUSED_ELEMENTWISE");
    expect(result.ops).not.toContain("EXP");
    expect(result.request.valueId).toBe(result.permuteValueId);
    expect(result.permuteArgErased).toBe(true);
  });

  it("emits every shared dense-permutation fixture without plan-owned meaning", async () => {
    const target = await getJitTarget();
    const casesJson = JSON.stringify(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map((fixture) => ({
      id: fixture.id,
      request: fixture.request,
      outputShape: fixture.outputShape.map(Number),
    })));
    const result = await target.run<Array<{
      id: string;
      request: {
        kind: string;
        inputShape: string[];
        axes: number[];
        dtype: string;
        valueId: number;
      };
      permuteValueId: number;
      permuteArgErased: boolean;
      outputShape: number[];
    }>>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._gpu_plan import build_gpu_execution_submission

fixtures = json.loads(${JSON.stringify(casesJson)})
result = []
for fixture in fixtures:
    request = fixture["request"]
    shape = tuple(int(extent) for extent in request["inputShape"])
    tensor = bg.from_numpy(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    output = tensor.permute(*request["axes"])
    submission = build_gpu_execution_submission(output._uop)
    plan = submission.plan_summary()
    semantic = submission.semantic_request_summary()
    permute_step = next(step for step in plan["steps"] if step["op"] == "PERMUTE")
    result.append({
        "id": fixture["id"],
        "request": semantic["requests"][0],
        "permuteValueId": int(permute_step["value_id"]),
        "permuteArgErased": permute_step["arg"] is None,
        "outputShape": list(permute_step["shape"]),
    })
result
`);

    expect(result.map(({ id }) => id)).toEqual(
      DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map(({ id }) => id),
    );
    result.forEach((actual, index) => {
      const fixture = DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases[index]!;
      const { valueId, ...semanticProjection } = actual.request;
      expect(semanticProjection).toEqual(fixture.request);
      expect(valueId).toBe(actual.permuteValueId);
      expect(actual.permuteArgErased).toBe(true);
      expect(actual.outputShape).toEqual(fixture.outputShape.map(Number));
    });
  });

  it("fail-closes PERMUTE requests outside the initial static f32 rank-2/3 profile", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
from browsergrad_jit._gpu_plan import build_gpu_execution_submission
from browsergrad_jit._ir import UOp, buffer, load, OP_PERMUTE

def source(name, shape, dtype="float32"):
    return load(buffer(name, shape, dtype))

cases = {
    "dtype": UOp(OP_PERMUTE, (source("f16", (2, 3), "float16"),), (3, 2), "float16", arg={"axes": (1, 0)}),
    "rank": UOp(OP_PERMUTE, (source("rank4", (1, 2, 3, 4)),), (4, 3, 2, 1), "float32", arg={"axes": (3, 2, 1, 0)}),
    "zero": UOp(OP_PERMUTE, (source("zero", (0, 3)),), (3, 0), "float32", arg={"axes": (1, 0)}),
    "axes": UOp(OP_PERMUTE, (source("axes", (2, 3)),), (2, 2), "float32", arg={"axes": (0, 0)}),
    "shape": UOp(OP_PERMUTE, (source("shape", (2, 3)),), (2, 3), "float32", arg={"axes": (1, 0)}),
    "closed": UOp(OP_PERMUTE, (source("closed", (2, 3)),), (3, 2), "float32", arg={"axes": (1, 0), "offset": 0}),
}
errors = {}
for name, node in cases.items():
    try:
        build_gpu_execution_submission(node)
        errors[name] = "no_error"
    except Exception as exc:
        errors[name] = type(exc).__name__ + ": " + str(exc)
errors
`);

    expect(errors.dtype).toMatch(/GpuPlanUnsupported.*requires float32/);
    expect(errors.rank).toMatch(/GpuPlanUnsupported.*requires rank 2 or 3/);
    expect(errors.zero).toMatch(/GpuPlanUnsupported.*positive static integer/);
    expect(errors.axes).toMatch(/GpuPlanUnsupported.*exact permutation/);
    expect(errors.shape).toMatch(/GpuPlanUnsupported.*does not match derived shape/);
    expect(errors.closed).toMatch(/GpuPlanUnsupported.*unsupported fields/);
  });
});
