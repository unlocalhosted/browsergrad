/**
 * Functional transforms tests (PRD-014).
 *
 * Verifies bg.func.grad, bg.func.vjp, bg.func.functional_call against
 * the existing closure-write .backward() — they should produce
 * numerically identical gradients without mutating .grad.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-014 functional transforms", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("grad(fn) returns the gradient without mutating .grad", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      grad_was_none_after: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np

x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = bg.from_numpy(x_np.copy(), requires_grad=True)

def fn(t):
    return (t * t).sum()

# Reference via .backward() — mutates x.grad.
x_ref = bg.from_numpy(x_np.copy(), requires_grad=True)
fn(x_ref).backward()
ref_grad = x_ref.grad.numpy()

# Functional path: doesn't touch x.grad.
g = bg.func.grad(fn)(x)
out_grad = g.numpy()

{
    "max_diff": float(np.max(np.abs(ref_grad - out_grad))),
    "grad_was_none_after": x.grad is None,
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.grad_was_none_after).toBe(true); // critical for vmap composition
  });

  it("grad with argnums=(0, 1) returns a tuple", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff_x: number;
      max_diff_w: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

x_np = np.array([[1.0, 2.0]], dtype=np.float32)
w_np = np.array([[0.5], [0.5]], dtype=np.float32)

def fn(x, w):
    return (x @ w).sum()

# Reference via .backward().
x_ref = bg.from_numpy(x_np.copy(), requires_grad=True)
w_ref = bg.from_numpy(w_np.copy(), requires_grad=True)
fn(x_ref, w_ref).backward()
gx_ref = x_ref.grad.numpy()
gw_ref = w_ref.grad.numpy()

# Functional.
x = bg.from_numpy(x_np.copy(), requires_grad=True)
w = bg.from_numpy(w_np.copy(), requires_grad=True)
gx, gw = bg.func.grad(fn, argnums=(0, 1))(x, w)

{
    "max_diff_x": float(np.max(np.abs(gx_ref - gx.numpy()))),
    "max_diff_w": float(np.max(np.abs(gw_ref - gw.numpy()))),
}
`);
    expect(result.max_diff_x).toBeLessThan(1e-6);
    expect(result.max_diff_w).toBeLessThan(1e-6);
  });

  it("vjp returns outputs + vjp_fn that produces matching cotangents", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      output_match: boolean;
      grad_max_diff: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = bg.from_numpy(x_np.copy(), requires_grad=True)

def fn(t):
    return t * t  # vector output

out, vjp_fn = bg.func.vjp(fn, x)
cot = bg.from_numpy(np.ones(3, dtype=np.float32))
(g_x,) = vjp_fn(cot)

# Expected gradient: d(t*t).sum() / dt = 2*t
expected = 2 * x_np
{
    "output_match": bool(np.allclose(out.numpy(), x_np * x_np)),
    "grad_max_diff": float(np.max(np.abs(g_x.numpy() - expected))),
}
`);
    expect(result.output_match).toBe(true);
    expect(result.grad_max_diff).toBeLessThan(1e-5);
  });

  it("functional_call substitutes module parameters statelessly", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      original_unchanged: boolean;
      override_correct: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np

# A simple Linear with known weights.
np.random.seed(0)
layer = bg.nn.Linear(2, 1)
orig_w = layer.weight.numpy().copy()

# Override with all-ones; the call should use ones, then revert.
ones_w = bg.from_numpy(np.ones_like(orig_w))
ones_b = bg.from_numpy(np.zeros(1, dtype=np.float32))
x = bg.from_numpy(np.array([[1.0, 1.0]], dtype=np.float32))

out = bg.func.functional_call(
    layer,
    {"weight": ones_w, "bias": ones_b},
    (x,),
)
override_correct = bool(np.allclose(out.numpy(), [[2.0]]))

# After the call, layer.weight should equal orig_w again.
post_w = layer.weight.numpy()
original_unchanged = bool(np.allclose(orig_w, post_w))

{"original_unchanged": original_unchanged, "override_correct": override_correct}
`);
    expect(result.override_correct).toBe(true);
    expect(result.original_unchanged).toBe(true);
  });

  it("vmap maps a scalar fn over the batch dimension", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      shape: number[];
      values: number[];
    }>(`
import browsergrad_jit as bg
import numpy as np

batched_x = bg.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))

def per_sample_sum(x):
    return x.sum()

mapped = bg.func.vmap(per_sample_sum)
out = mapped(batched_x)
arr = out.numpy()
{"shape": list(arr.shape), "values": arr.tolist()}
`);
    expect(result.shape).toEqual([3]);
    // sum of [0..4) + [4..8) + [8..12) = 6, 22, 38
    expect(result.values).toEqual([6, 22, 38]);
  });

  it("vmap maps an elementwise+matmul fn correctly", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      shape: number[];
      max_diff: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

batched_x = bg.from_numpy(
    np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
W = bg.from_numpy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))

def fn(x):
    # x: shape (2,); W: shape (2, 2). Result: shape (2,).
    return (x @ W) * 2.0

mapped = bg.func.vmap(fn)(batched_x)
arr = mapped.numpy()

expected = (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
            @ np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)) * 2.0
{
    "shape": list(arr.shape),
    "max_diff": float(np.max(np.abs(arr - expected))),
}
`);
    expect(result.shape).toEqual([3, 2]);
    expect(result.max_diff).toBeLessThan(1e-5);
  });

  it("vmap composes with grad for per-sample gradients", async () => {
    // PRD-014b — composition now works after two fixes:
    //  1. bg.func.grad force-sets requires_grad=True on argnums inputs
    //     so the autograd chain builds even when vmap passes plain leaves.
    //  2. _vmap.{reshape,broadcast_to} check whether the input was
    //     actually batched (ndim grew by 1) before prepending B —
    //     un-batched pass-throughs keep their original target shape.
    const target = await getJitTarget();
    const result = await target.run<{ shape: number[]; max_diff: number }>(`
import browsergrad_jit as bg
import numpy as np

batched_x = bg.from_numpy(np.array(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))

def loss(x):
    return (x * x).sum()

per_sample = bg.func.vmap(bg.func.grad(loss))(batched_x)
arr = per_sample.numpy()
expected = 2 * np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
{
    "shape": list(arr.shape),
    "max_diff": float(np.max(np.abs(arr - expected))),
}
`);
    expect(result.shape).toEqual([3, 2]);
    expect(result.max_diff).toBeLessThan(1e-5);
  });

  it("jacrev is deferred with a clear pointer", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
try:
    bg.func.jacrev(lambda x: x.sum())
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/jacrev/);
    expect(err).toMatch(/PRD-014b/);
  });

  it("torch.func shim resolves to bg.func via install_torch_alias", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      grad_works: boolean;
      vjp_works: boolean;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import numpy as np

x = torch.from_numpy(np.array([3.0], dtype=np.float32))
x.requires_grad = True
g = torch.func.grad(lambda t: (t * t).sum())(x)

out, vjp_fn = torch.func.vjp(lambda t: t * 2.0, x)
(g2,) = vjp_fn(torch.from_numpy(np.ones(1, dtype=np.float32)))

{
    "grad_works": bool(np.allclose(g.numpy(), [6.0])),
    "vjp_works": bool(np.allclose(g2.numpy(), [2.0])),
}
`);
    expect(result.grad_works).toBe(true);
    expect(result.vjp_works).toBe(true);
  });
});
