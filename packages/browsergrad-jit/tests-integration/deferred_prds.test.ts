/**
 * Tests for the previously-deferred PRDs that landed:
 *   - PRD-011 (WebNN spike, behind bg.experimental.webnn)
 *   - PRD-012b (cost model + producer-consumer pairs)
 *   - PRD-012c (transformer_block megakernel constructor)
 *   - PRD-014b (remaining vmap rules + refusal stubs)
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-011 WebNN spike (bg.experimental.webnn)", () => {
  beforeAll(async () => { await getJitTarget(); }, 120_000);
  beforeEach(async () => { await clearNamespace(await getJitTarget()); });

  it("is_available is False outside browser", async () => {
    const target = await getJitTarget();
    const r = await target.run<boolean>(`
import browsergrad_jit as bg
bg.experimental.webnn.is_available()
`);
    expect(r).toBe(false);
  });

  it("matmul constructs an OP_CUSTOM('webnn_matmul') UOp", async () => {
    const target = await getJitTarget();
    const r = await target.run<{ op: string; arg_op: string; shape: number[] }>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.ones((4, 8), dtype=np.float32))
b = bg.from_numpy(np.ones((8, 6), dtype=np.float32))
y = bg.experimental.webnn.matmul(a, b)
{
    "op": y._uop.op,
    "arg_op": y._uop.arg["op"],
    "shape": list(y._uop.shape),
}
`);
    expect(r.op).toBe("CUSTOM");
    expect(r.arg_op).toBe("webnn_matmul");
    expect(r.shape).toEqual([4, 6]);
  });

  it("non-2D inputs raise", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.ones((2, 4, 8), dtype=np.float32))
b = bg.from_numpy(np.ones((8, 6), dtype=np.float32))
try:
    bg.experimental.webnn.matmul(a, b)
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/2-D only/);
  });
});

describe("PRD-012b cost model + producer-consumer", () => {
  beforeAll(async () => { await getJitTarget(); }, 120_000);
  beforeEach(async () => { await clearNamespace(await getJitTarget()); });

  it("estimate_flops returns 2*M*N*K for matmul", async () => {
    const target = await getJitTarget();
    const f = await target.run<number>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.ones((4, 8), dtype=np.float32))
b = bg.from_numpy(np.ones((8, 6), dtype=np.float32))
y = a @ b
bg.jit.cost_model.estimate_flops(y._uop)
`);
    expect(f).toBe(2 * 4 * 6 * 8);
  });

  it("pick_tier returns 'numpy' for tiny ops and 'gpu' for large ones", async () => {
    const target = await getJitTarget();
    const r = await target.run<{ tiny: string; big: string }>(`
import browsergrad_jit as bg
import numpy as np
tiny_a = bg.from_numpy(np.ones((2, 2), dtype=np.float32))
tiny_b = bg.from_numpy(np.ones((2, 2), dtype=np.float32))
tiny = tiny_a @ tiny_b

big_a = bg.from_numpy(np.ones((256, 256), dtype=np.float32))
big_b = bg.from_numpy(np.ones((256, 256), dtype=np.float32))
big = big_a @ big_b
{
    "tiny": bg.jit.cost_model.pick_tier(tiny._uop),
    "big": bg.jit.cost_model.pick_tier(big._uop),
}
`);
    expect(r.tiny).toBe("numpy");
    expect(r.big).toBe("gpu");
  });

  it("find_producer_consumer_pairs detects bias-add → matmul", async () => {
    const target = await getJitTarget();
    const n = await target.run<number>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.ones((4, 8), dtype=np.float32))
b = bg.from_numpy(np.ones((4, 8), dtype=np.float32))
w = bg.from_numpy(np.ones((8, 6), dtype=np.float32))
# (x + b) @ w  — bias-add producer feeding matmul
y = (x + b) @ w
pairs = bg.jit.cost_model.find_producer_consumer_pairs(y._uop)
len(pairs)
`);
    expect(n).toBeGreaterThanOrEqual(1);
  });

  it("cost_stats returns the rollup dict", async () => {
    const target = await getJitTarget();
    const stats = await target.run<{
      total_flops: number;
      nodes: number;
      gpu_picks: number;
      numpy_picks: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.ones((64, 64), dtype=np.float32))
b = bg.from_numpy(np.ones((64, 64), dtype=np.float32))
y = a @ b
bg.jit.cost_model.cost_stats(y._uop)
`);
    expect(stats.total_flops).toBe(2 * 64 * 64 * 64);
    expect(stats.nodes).toBeGreaterThan(0);
    expect(stats.gpu_picks + stats.numpy_picks).toBe(stats.nodes);
  });
});

describe("PRD-012c transformer_block megakernel constructor", () => {
  beforeAll(async () => { await getJitTarget(); }, 120_000);
  beforeEach(async () => { await clearNamespace(await getJitTarget()); });

  it("builds an OP_CUSTOM('transformer_block') UOp", async () => {
    const target = await getJitTarget();
    const r = await target.run<{ op: string; arg_op: string; shape: number[] }>(`
import browsergrad_jit as bg
import numpy as np
B, S, D = 2, 16, 64
x = bg.from_numpy(np.zeros((B, S, D), dtype=np.float32))
w_qkv = bg.from_numpy(np.zeros((D, 3 * D), dtype=np.float32))
w_o = bg.from_numpy(np.zeros((D, D), dtype=np.float32))
w_ff1 = bg.from_numpy(np.zeros((D, 4 * D), dtype=np.float32))
w_ff2 = bg.from_numpy(np.zeros((4 * D, D), dtype=np.float32))

y = bg.kernels.transformer_block(x, w_qkv, w_o, w_ff1, w_ff2, num_heads=4)
{
    "op": y._uop.op,
    "arg_op": y._uop.arg["op"],
    "shape": list(y._uop.shape),
}
`);
    expect(r.op).toBe("CUSTOM");
    expect(r.arg_op).toBe("transformer_block");
    expect(r.shape).toEqual([2, 16, 64]);
  });

  it("refuses non-3D inputs", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.zeros((16, 64), dtype=np.float32))  # 2-D
w = bg.from_numpy(np.zeros((64, 192), dtype=np.float32))
try:
    bg.kernels.transformer_block(x, w, w, w, w)
    result = "no_error"
except TypeError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/\(B, S, D\)/);
  });
});

describe("PRD-014b remaining vmap rules + refusal stubs", () => {
  beforeAll(async () => { await getJitTarget(); }, 120_000);
  beforeEach(async () => { await clearNamespace(await getJitTarget()); });

  it("ISNAN rule passes through with batch dim", async () => {
    const target = await getJitTarget();
    const r = await target.run<{ shape: number[]; values: boolean[][] }>(`
import browsergrad_jit as bg
import numpy as np
batched = bg.from_numpy(np.array(
    [[1.0, float("nan")], [3.0, 4.0]], dtype=np.float32))
def has_nan(x):
    from browsergrad_jit._ir import UOp, OP_ISNAN
    u = UOp(op=OP_ISNAN, inputs=(x._uop,), shape=x.shape, dtype="bool", arg=None)
    return bg.TensorProxy(u, session=x._get_session())
mapped = bg.func.vmap(has_nan)(batched)
arr = mapped.numpy()
{"shape": list(arr.shape), "values": arr.tolist()}
`);
    expect(r.shape).toEqual([2, 2]);
    expect(r.values).toEqual([[false, true], [false, false]]);
  });

  it("RANDOM raises with a clear pointer", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_RANDOM
def fn(x):
    seed = (1, 2)
    u = UOp(op=OP_RANDOM, inputs=(), shape=(3,), dtype="float32",
            arg={"seed_key": seed, "dist": "uniform"})
    return bg.TensorProxy(u, session=x._get_session())
try:
    bg.func.vmap(fn)(bg.from_numpy(np.zeros((2, 4), dtype=np.float32)))
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/OP_RANDOM/);
    expect(err).toMatch(/key split/);
  });
});
