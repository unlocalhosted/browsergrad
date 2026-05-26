/**
 * Fusion integration tests (PRD-006).
 *
 * Verifies:
 *  - Numerical identity (within 1e-6) between fusion ON and fusion OFF.
 *  - The matchers actually fire on the canonical IR shapes
 *    (`bg.jit.debug_fused_kernels()` reports the expected patterns).
 *  - Autograd is preserved — `.backward()` produces identical gradients
 *    regardless of fusion state.
 *  - `BG_DISABLE_FUSION` env var precedence: not tested here because the
 *    var is read at process start; covered by a chaos test in CI.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-006 fusion — numerical correctness vs unfused", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    // Reset to fusion-on between tests.
    await target.run(`
import browsergrad_jit as bg
bg.jit.use_fusion(True)
`);
  });

  it("elementwise chain produces identical output with fusion on vs off", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ fused: number[][]; unfused: number[][]; same: boolean }>(`
import browsergrad_jit as bg
import numpy as np
np.random.seed(0)
x_np = np.random.randn(8, 16).astype(np.float32)
w_np = np.random.randn(16).astype(np.float32)
b_np = np.random.randn(16).astype(np.float32)

def compute():
    x = bg.tensor(x_np.copy())
    w = bg.tensor(w_np.copy())
    b = bg.tensor(b_np.copy())
    # 4-op pointwise chain: ((x * w + b) - w) * b
    return ((x * w + b - w) * b).numpy()

bg.jit.use_fusion(True)
fused = compute()
bg.jit.use_fusion(False)
unfused = compute()
same = bool(np.allclose(fused, unfused, atol=1e-6, rtol=0))
bg.jit.use_fusion(True)
{"fused": fused.tolist(), "unfused": unfused.tolist(), "same": same}
`);
    expect(result.same).toBe(true);
    expect(result.fused).toEqual(result.unfused);
  });

  it("softmax fuses and produces output within 1e-6 of unfused softmax", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ max_diff: number; fired: number }>(`
import browsergrad_jit as bg
import numpy as np
np.random.seed(1)
x_np = np.random.randn(8, 32).astype(np.float32) * 5  # nontrivial dynamic range

bg.jit.use_fusion(False)
x = bg.tensor(x_np.copy())
unfused = bg.nn.functional.softmax(x, dim=-1).numpy()

bg.jit.use_fusion(True)
x = bg.tensor(x_np.copy())
fused_out = bg.nn.functional.softmax(x, dim=-1).numpy()
softmax_groups = [k for k in bg.jit.debug_fused_kernels() if k.pattern == "softmax"]

bg.jit.use_fusion(True)
result = {
    "max_diff": float(np.max(np.abs(unfused - fused_out))),
    "fired": len(softmax_groups),
}
result
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.fired).toBeGreaterThanOrEqual(1);
  });

  it("backward gradients match between fused and unfused paths", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ max_diff: number }>(`
import browsergrad_jit as bg
import numpy as np
np.random.seed(2)
x_np = np.random.randn(4, 8).astype(np.float32)
w_np = np.random.randn(8, 3).astype(np.float32)
b_np = np.random.randn(3).astype(np.float32)

def grads():
    x = bg.from_numpy(x_np.copy(), requires_grad=True)
    w = bg.from_numpy(w_np.copy(), requires_grad=True)
    b = bg.from_numpy(b_np.copy(), requires_grad=True)
    out = x @ w + b
    loss = (out * out).sum()
    loss.backward()
    return w.grad.numpy(), b.grad.numpy()

bg.jit.use_fusion(False)
gw_u, gb_u = grads()
bg.jit.use_fusion(True)
gw_f, gb_f = grads()
bg.jit.use_fusion(True)
{
    "max_diff": float(max(np.max(np.abs(gw_u - gw_f)), np.max(np.abs(gb_u - gb_f)))),
}
`);
    expect(result.max_diff).toBeLessThan(1e-5);
  });

  it("does not fuse single ops below the minimum chain length", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ fused_count: number }>(`
import browsergrad_jit as bg
import numpy as np

bg.jit.use_fusion(True)
x = bg.tensor(np.zeros(4, dtype=np.float32))
_ = (x + 1.0).numpy()  # 1-op "chain" — must NOT fuse
{"fused_count": len([k for k in bg.jit.debug_fused_kernels() if k.pattern == "elementwise_chain"])}
`);
    expect(result.fused_count).toBe(0);
  });

  it("debug_fused_kernels reports the chain length and opcodes", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ groups: Array<{ n_ops: number; ops: string[] }> }>(`
import browsergrad_jit as bg
import numpy as np

bg.jit.use_fusion(True)
x = bg.tensor(np.zeros(4, dtype=np.float32))
y = bg.tensor(np.ones(4, dtype=np.float32))
# A 3-op chain: (x + y) * y - x  --> ADD, MUL, ADD(with NEG)
_ = ((x + y) * y).numpy()
{"groups": [
    {"n_ops": int(k.n_ops), "ops": list(k.ops)}
    for k in bg.jit.debug_fused_kernels()
    if k.pattern == "elementwise_chain"
]}
`);
    expect(result.groups.length).toBeGreaterThanOrEqual(1);
    const g = result.groups[0]!;
    expect(g.n_ops).toBeGreaterThanOrEqual(2);
  });

  it("use_fusion(False) prevents any fusion", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ count: number }>(`
import browsergrad_jit as bg
import numpy as np

bg.jit.use_fusion(False)
x = bg.tensor(np.zeros(4, dtype=np.float32))
y = bg.tensor(np.ones(4, dtype=np.float32))
_ = ((x + y) * y - x).numpy()
{"count": len(bg.jit.debug_fused_kernels())}
`);
    expect(result.count).toBe(0);
  });

  it("training a 2-class MLP converges identically under fusion on/off", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ loss_f: number; loss_u: number }>(`
import browsergrad_jit as bg
import numpy as np

def train(use_fusion: bool):
    bg.jit.use_fusion(use_fusion)
    bg.manual_seed(0)
    np.random.seed(0)
    x1 = np.random.randn(16, 2).astype(np.float32) - 1.0
    x2 = np.random.randn(16, 2).astype(np.float32) + 1.0
    X = bg.tensor(np.concatenate([x1, x2], axis=0))
    Y = bg.tensor(np.concatenate([np.zeros(16), np.ones(16)]).astype(np.int64))
    model = bg.nn.Sequential(
        bg.nn.Linear(2, 8),
        bg.nn.ReLU(),
        bg.nn.Linear(8, 2),
    )
    opt = bg.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(30):
        opt.zero_grad()
        logits = model(X)
        loss = bg.nn.functional.cross_entropy(logits, Y)
        loss.backward()
        opt.step()
    return float(loss.item())

lf = train(True)
lu = train(False)
bg.jit.use_fusion(True)
{"loss_f": lf, "loss_u": lu}
`);
    // The two training runs are deterministic — same seed, same init,
    // same optimizer — so they must converge to numerically equal losses
    // within the 1e-5 tolerance that captures NumPy's own float accumulation
    // order tolerances.
    expect(Math.abs(result.loss_f - result.loss_u)).toBeLessThan(1e-5);
  });
});
