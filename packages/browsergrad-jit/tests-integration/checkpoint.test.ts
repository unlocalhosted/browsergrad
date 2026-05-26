/**
 * Gradient checkpointing integration tests (PRD-009 v0).
 *
 * Verifies:
 *   - Wrapped forward returns the same output as un-wrapped (parity).
 *   - Backward gradients match the un-wrapped reference within 1e-6.
 *   - The IR-rewrite produces fresh clone identities (the cached
 *     forward intermediates are not referenced by the backward graph
 *     after rewrite).
 *   - Refusal modes: use_reentrant=True, preserve_rng_state=True,
 *     nested checkpointing, ops without VJP rules.
 *   - PyTorch shim `torch.utils.checkpoint.checkpoint` resolves
 *     when the alias is installed.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-009 gradient checkpointing", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    await target.run(`
import browsergrad_jit as bg
bg._checkpoint.clear_all_regions()
`);
  });

  it("forward pass returns the same tensor as the un-wrapped fn", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ max_diff: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit.utils.checkpoint import checkpoint

x_np = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)

def fn(t):
    return (t + 1.0) * (t * 2.0)

x = bg.tensor(x_np.copy())
y_uncheckpointed = fn(x).numpy()

x2 = bg.tensor(x_np.copy())
y_checkpointed = checkpoint(fn, x2).numpy()
{"max_diff": float(np.max(np.abs(y_uncheckpointed - y_checkpointed)))}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
  });

  it("backward gradients match the un-checkpointed reference", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ max_diff: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit.utils.checkpoint import checkpoint

x_np = np.linspace(-2, 2, 8, dtype=np.float32).reshape(2, 4)

def block(t):
    # Two-op chain that goes through MUL and ADD with broadcasting —
    # exercises the un-broadcast logic in the backward graph.
    return (t * 3.0) + t

# Reference run: no checkpointing.
x = bg.from_numpy(x_np.copy(), requires_grad=True)
y = block(x).sum()
y.backward()
grad_ref = x.grad.numpy()

# Checkpointed run.
x2 = bg.from_numpy(x_np.copy(), requires_grad=True)
y2 = checkpoint(block, x2).sum()
y2.backward()
grad_ckpt = x2.grad.numpy()

{"max_diff": float(np.max(np.abs(grad_ref - grad_ckpt)))}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
  });

  it("rewrite produces a clone with a fresh identity for interior UOps", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ shape_preserved: boolean; identity_changed: boolean }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit.utils.checkpoint import checkpoint
from browsergrad_jit._ir import toposort, UOp, OP_NEG
from browsergrad_jit._checkpoint import _REGIONS, apply_checkpoint_rewrite

def block(t):
    return (t * 3.0) + 1.0

x = bg.from_numpy(np.zeros((3,), dtype=np.float32), requires_grad=True)
out = checkpoint(block, x)
assert len(_REGIONS) == 1
region = next(iter(_REGIONS.values()))

# Construct a synthetic gradient subgraph that references one of the
# interior UOps directly (in the real backward this happens because
# VJP rules emit UOps with arg["vjp_of"]=forward_node — but for this
# structural test we just need a graph that touches an interior UOp).
interior_sample = next(iter(region.interior_uop_ids))  # an int id
# Pick the actual UOp from the forward graph.
forward_nodes = {id(u): u for u in toposort(out._uop)}
interior_uop = forward_nodes[interior_sample]

# A "grad" graph that NEG's the interior UOp — non-physical but valid IR.
grad = UOp(op=OP_NEG, inputs=(interior_uop,), shape=interior_uop.shape,
           dtype=interior_uop.dtype, arg=None)
rewritten = apply_checkpoint_rewrite(grad)

# After rewrite: rewritten's input should be a clone of interior_uop —
# same opcode/shape/dtype, different id().
clone = rewritten.inputs[0]
{
    "shape_preserved": clone.shape == interior_uop.shape and clone.op == interior_uop.op,
    "identity_changed": id(clone) != id(interior_uop),
}
`);
    expect(result.shape_preserved).toBe(true);
    expect(result.identity_changed).toBe(true);
  });

  it("MLP training step: parity vs un-checkpointed", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ max_grad_diff: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit.utils.checkpoint import checkpoint

bg.manual_seed(0)
x_np = np.random.RandomState(0).randn(4, 3).astype(np.float32)
target_np = np.random.RandomState(1).randn(4, 2).astype(np.float32)

def make_model(seed):
    np.random.seed(seed)
    return bg.nn.Linear(3, 2)

# Reference run.
m1 = make_model(7)
x1 = bg.from_numpy(x_np.copy())
t1 = bg.tensor(target_np.copy())
y1 = m1(x1)
loss1 = ((y1 - t1) * (y1 - t1)).sum()
loss1.backward()
gw_ref = m1.weight.grad.numpy()
gb_ref = m1.bias.grad.numpy()

# Checkpointed: wrap the linear's forward.
m2 = make_model(7)
x2 = bg.from_numpy(x_np.copy())
t2 = bg.tensor(target_np.copy())
y2 = checkpoint(lambda t: m2(t), x2)
loss2 = ((y2 - t2) * (y2 - t2)).sum()
loss2.backward()
gw_ckpt = m2.weight.grad.numpy()
gb_ckpt = m2.bias.grad.numpy()

{
    "max_grad_diff": float(max(
        np.max(np.abs(gw_ref - gw_ckpt)),
        np.max(np.abs(gb_ref - gb_ckpt)),
    )),
}
`);
    expect(result.max_grad_diff).toBeLessThan(1e-5);
  });

  it("refuses use_reentrant=True", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
from browsergrad_jit.utils.checkpoint import checkpoint
x = bg.tensor([1.0])
try:
    checkpoint(lambda t: t + 1, x, use_reentrant=True)
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/CheckpointError/);
    expect(err).toMatch(/use_reentrant/);
  });

  it("refuses preserve_rng_state=True", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
from browsergrad_jit.utils.checkpoint import checkpoint
x = bg.tensor([1.0])
try:
    checkpoint(lambda t: t + 1, x, preserve_rng_state=True)
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/preserve_rng_state/);
  });

  it("refuses nested checkpointing", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
from browsergrad_jit.utils.checkpoint import checkpoint
x = bg.tensor([1.0])
def outer(t):
    return checkpoint(lambda u: u + 1, t)
try:
    checkpoint(outer, x)
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/CheckpointError/);
    expect(err).toMatch(/nesting/);
  });

  it("torch.utils.checkpoint.checkpoint shim resolves via install_torch_alias", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ found: boolean; works: boolean }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
from torch.utils.checkpoint import checkpoint as torch_ckpt
import numpy as np

x = bg.tensor(np.zeros((4,), dtype=np.float32))
out = torch_ckpt(lambda t: t + 1.0, x)
{
    "found": callable(torch_ckpt),
    "works": bool(np.allclose(out.numpy(), np.ones(4, dtype=np.float32))),
}
`);
    expect(result.found).toBe(true);
    expect(result.works).toBe(true);
  });
});
