import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("typed attention-forward contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed IR and matches the bounded stable CPU reference", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      aliasOp: string;
      shape: number[];
      dtype: string;
      requiresGrad: boolean;
      maxDiff: number;
      aliasDiff: number;
      owning: boolean;
      support: Record<string, unknown>;
    }>(`
import browsergrad_jit as bg
import numpy as np

q_np = np.array([[[[1.0, 0.5], [0.25, -0.75]]]], dtype=np.float32)
k_np = np.array([[[[0.5, -0.25], [1.25, 0.75], [-0.5, 1.0]]]], dtype=np.float32)
v_np = np.array([[[[2.0, 3.0], [5.0, 7.0], [-1.0, 4.0]]]], dtype=np.float32)
q = bg.from_numpy(q_np)
k = bg.from_numpy(k_np)
v = bg.from_numpy(v_np)
out = bg.kernels.attention_forward(q, k, v)
alias = bg.kernels.flash_attention(q, k, v)
actual = out.numpy()
alias_actual = alias.numpy()
scale = np.float32(1.0 / np.sqrt(q_np.shape[-1]))
scores = np.matmul(q_np, np.swapaxes(k_np, -1, -2)) * scale
maximum = scores.max(axis=-1, keepdims=True)
exponentials = np.exp(scores - maximum)
reference = np.matmul(exponentials / exponentials.sum(axis=-1, keepdims=True), v_np)
support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == "ATTENTION_FORWARD"
)
{
    "op": out._uop.op,
    "aliasOp": alias._uop.op,
    "shape": list(out.shape),
    "dtype": out.dtype,
    "requiresGrad": out.requires_grad,
    "maxDiff": float(np.max(np.abs(actual - reference))),
    "aliasDiff": float(np.max(np.abs(alias_actual - reference))),
    "owning": bool(actual.flags.owndata),
    "support": support,
}
`);

    expect(result.op).toBe("ATTENTION_FORWARD");
    expect(result.aliasOp).toBe("ATTENTION_FORWARD");
    expect(result.shape).toEqual([1, 1, 2, 2]);
    expect(result.dtype).toBe("float32");
    expect(result.requiresGrad).toBe(false);
    expect(result.maxDiff).toBeLessThan(1e-6);
    expect(result.aliasDiff).toBeLessThan(1e-6);
    expect(result.owning).toBe(true);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.kernels.attention-forward.v1",
      retiredOpaqueOperationId: "jit.custom.flash-attention.v0",
      decisions: {
        cpu: "supported-numpy-owning-stable-attention-forward",
        webgpu: "supported-legacy-row-wise-online-softmax-f32",
      },
    });
  });

  it("rejects unsupported shape, dtype, autograd, mask, scale, and numerical domains", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

q = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
k = bg.from_numpy(np.ones((1, 1, 3, 4), dtype=np.float32))
v = bg.from_numpy(np.ones((1, 1, 3, 4), dtype=np.float32))
mask = bg.from_numpy(np.zeros((1, 1, 2, 3), dtype=np.float32))
nonfinite = np.ones((1, 1, 2, 4), dtype=np.float32)
nonfinite[0, 0, 0, 0] = np.inf

{
    "rank": error(lambda: bg.kernels.attention_forward(
        bg.from_numpy(np.ones((1, 2, 4), dtype=np.float32)), k, v
    )),
    "dtype": error(lambda: bg.kernels.attention_forward(
        bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float64)), k, v
    )),
    "heads": error(lambda: bg.kernels.attention_forward(
        q, bg.from_numpy(np.ones((1, 2, 3, 4), dtype=np.float32)), v
    )),
    "depth": error(lambda: bg.kernels.attention_forward(
        q, k, bg.from_numpy(np.ones((1, 1, 3, 5), dtype=np.float32))
    )),
    "autograd": error(lambda: bg.kernels.attention_forward(
        bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32), requires_grad=True),
        k,
        v,
    )),
    "mask": error(lambda: bg.kernels.flash_attention(q, k, v, mask=mask)),
    "scale": error(lambda: bg.kernels.flash_attention(q, k, v, scale=0.5)),
    "nonfinite": error(lambda: bg.kernels.attention_forward(
        bg.from_numpy(nonfinite), k, v
    ).numpy()),
}
`);

    expect(errors.rank).toMatch(/^ShapeError: .*rank-4/);
    expect(errors.dtype).toMatch(/^ShapeError: .*float32/);
    expect(errors.heads).toMatch(/^ShapeError: .*batch\/head/);
    expect(errors.depth).toMatch(/^ShapeError: .*head depth/);
    expect(errors.autograd).toMatch(/^NoBackwardError: .*no admitted VJP/);
    expect(errors.mask).toMatch(/^JitNotImplementedError: .*additive masks/);
    expect(errors.scale).toMatch(/^JitNotImplementedError: .*custom scale/);
    expect(errors.nonfinite).toMatch(/^RealizationError: .*finite float32/);
  });

  it("revalidates hostile mutation and refuses unowned plan, vmap, and export boundaries", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

q = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
k = bg.from_numpy(np.ones((1, 1, 3, 4), dtype=np.float32))
v = bg.from_numpy(np.ones((1, 1, 3, 4), dtype=np.float32))
plan_node = bg.kernels.attention_forward(q, k, v)
export_node = bg.kernels.attention_forward(q, k, v)
forged_node = bg.kernels.attention_forward(q, k, v)
forged_node._uop.arg["scale"] = 0.25
batched_q = bg.from_numpy(np.ones((2, 1, 1, 2, 4), dtype=np.float32))

{
    "forgedCpu": error(forged_node.numpy),
    "plan": error(lambda: bg.gpu_plan_summary(plan_node)),
    "onnx": error(lambda: bg.onnx.export_inference(
        export_node, input_buffers=(q, k, v)
    )),
    "vmap": error(lambda: bg.func.vmap(
        lambda item: bg.kernels.attention_forward(item, k, v)
    )(batched_q)),
}
`);

    expect(errors.forgedCpu).toMatch(/^RealizationError: .*canonical float32/);
    expect(errors.plan).toMatch(/^GpuPlanUnsupported: .*Gate 5 attention artifact/);
    expect(errors.onnx).toMatch(/^OnnxUnmappableOp: .*ATTENTION_FORWARD/);
    expect(errors.vmap).toMatch(/^JitNotImplementedError: .*not vmappable/);
  });
});
