/**
 * Tensor indexing (slice / fancy / boolean) + comparison ops.
 *
 * Oracles are NumPy directly: same call on a numpy array vs `.data` of
 * the indexed tensor. Backward verified via finite differences.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { clearNamespace, getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

async function reset(): Promise<void> {
  await clearNamespace(target);
}

const PRELUDE = `
import browsergrad_grad as grad
import numpy as np
`;

describe("Tensor.__getitem__", () => {
  beforeAll(reset);

  it("supports basic slicing (forward)", async () => {
    const result = await target.run<{ values: number[]; shape: number[] }>(`
${PRELUDE}
x = grad.Tensor(np.arange(20, dtype=np.float32).reshape(4, 5))
sliced = x[1:3, 0:3]
{"values": sliced.tolist(), "shape": list(sliced.shape)}
`);
    expect(result.shape).toEqual([2, 3]);
    expect(result.values).toEqual([
      [5, 6, 7],
      [10, 11, 12],
    ]);
  });

  it("supports integer indexing (forward)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor(np.array([10.0, 20.0, 30.0, 40.0]))
[float(x[0]), float(x[2]), float(x[-1])]
`);
    expect(result).toEqual([10, 30, 40]);
  });

  it("supports boolean mask indexing (forward)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
mask = np.array([True, False, True, False, True])
x[mask].tolist()
`);
    expect(result).toEqual([1, 3, 5]);
  });

  it("supports fancy (int-array) indexing (forward)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([10.0, 20.0, 30.0, 40.0])
idx = np.array([3, 0, 2])
x[idx].tolist()
`);
    expect(result).toEqual([40, 10, 30]);
  });

  it("backward of slicing scatters gradient back to original positions", async () => {
    // Hand-derived: for x = [..6.., ..8..], y = x[1:3], loss = y.sum()
    // → dx = [0, 1, 1, 0, ..., 0]
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([10.0, 20.0, 30.0, 40.0, 50.0], requires_grad=True)
y = x[1:3]
y.sum().backward()
x.grad.tolist()
`);
    expect(result).toEqual([0, 1, 1, 0, 0]);
  });

  it("backward of fancy indexing accumulates gradient at indexed positions", async () => {
    // Picking x[2] twice → x.grad[2] += 1 twice
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
idx = np.array([2, 0, 2])
y = x[idx]
y.sum().backward()
x.grad.tolist()
`);
    expect(result).toEqual([1, 0, 2, 0]);
  });

  it("backward of boolean masking routes gradient to True positions", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
mask = np.array([True, False, True, False, True])
y = x[mask]
y.sum().backward()
x.grad.tolist()
`);
    expect(result).toEqual([1, 0, 1, 0, 1]);
  });

  it("backward via finite differences (general slice case)", async () => {
    const result = await target.run<{ analytic: number[]; fd: number[] }>(`
${PRELUDE}
np.random.seed(0)
X = np.random.randn(4, 5).astype(np.float32)
x_t = grad.Tensor(X, requires_grad=True)
# loss = (x[1:3, 0:3] * x[1:3, 0:3]).sum()
y = x_t[1:3, 0:3]
loss = (y * y).sum()
loss.backward()
analytic = x_t.grad.data.flatten().tolist()

# Finite differences
eps = 1e-3
fd = np.zeros_like(X)
def f(Xv):
    t = grad.Tensor(Xv)
    return float(((t[1:3, 0:3] * t[1:3, 0:3]).sum()).item())
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[i, j] += eps
        Xm = X.copy(); Xm[i, j] -= eps
        fd[i, j] = (f(Xp) - f(Xm)) / (2 * eps)

{"analytic": analytic, "fd": fd.flatten().tolist()}
`);
    expect(result.analytic.length).toBe(result.fd.length);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.fd[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("comparison ops", () => {
  beforeAll(reset);

  it("== returns 1/0 float tensor (PyTorch returns bool; we use f32 for op compatibility)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
a = grad.Tensor([1.0, 2.0, 3.0, 2.0])
b = grad.Tensor([1.0, 1.0, 3.0, 0.0])
(a == b).tolist()
`);
    expect(result).toEqual([1, 0, 1, 0]);
  });

  it("!= / < / > / <= / >= behave correctly", async () => {
    const result = await target.run<{ ne: number[]; lt: number[]; gt: number[]; le: number[]; ge: number[] }>(`
${PRELUDE}
a = grad.Tensor([1.0, 2.0, 3.0])
b = grad.Tensor([2.0, 2.0, 2.0])
{
  "ne": (a != b).tolist(),
  "lt": (a <  b).tolist(),
  "gt": (a >  b).tolist(),
  "le": (a <= b).tolist(),
  "ge": (a >= b).tolist(),
}
`);
    expect(result.ne).toEqual([1, 0, 1]);
    expect(result.lt).toEqual([1, 0, 0]);
    expect(result.gt).toEqual([0, 0, 1]);
    expect(result.le).toEqual([1, 1, 0]);
    expect(result.ge).toEqual([0, 1, 1]);
  });

  it("accuracy idiom: (pred == target).float().mean() works (combined with sum/mean)", async () => {
    const result = await target.run<number>(`
${PRELUDE}
pred = grad.Tensor([0, 1, 2, 1, 2])  # already class indices, float-encoded
target = grad.Tensor([0, 1, 1, 1, 2])
float((pred == target).mean().item())
`);
    // 4/5 = 0.8
    expect(Math.abs(result - 0.8)).toBeLessThan(1e-5);
  });
});
