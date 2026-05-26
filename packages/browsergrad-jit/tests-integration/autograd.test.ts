/**
 * Autograd integration tests.
 *
 * Verifies that .backward() populates .grad on leaf tensors that have
 * requires_grad=True, matching standard analytical gradients. Each test
 * asserts both the gradient values and that requires_grad=False tensors
 * are skipped.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("TensorProxy.backward — leaf gradient accumulation", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("d/dx of x*x at x=3 is 6", async () => {
    const target = await getJitTarget();
    const grad = await target.run<number>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([3.0], dtype=np.float32), requires_grad=True)
y = (x * x).sum()
y.backward()
x.grad.item()
`);
    expect(grad).toBeCloseTo(6.0, 5);
  });

  it("d/dx of sum(x) is ones-like", async () => {
    const target = await getJitTarget();
    const grad = await target.run<number[]>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
y = x.sum()
y.backward()
x.grad.tolist()
`);
    expect(grad).toEqual([1.0, 1.0, 1.0]);
  });

  it("d/dx of mean(x) is 1/n everywhere", async () => {
    const target = await getJitTarget();
    const grad = await target.run<number[]>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), requires_grad=True)
y = x.mean()
y.backward()
x.grad.tolist()
`);
    expect(grad.length).toBe(4);
    for (const g of grad) {
      expect(g).toBeCloseTo(0.25, 5);
    }
  });

  it("d/dx of (x + y).sum() is ones for both", async () => {
    const target = await getJitTarget();
    const grads = await target.run<{ dx: number[]; dy: number[] }>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
y = bg.from_numpy(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
out = (x + y).sum()
out.backward()
{"dx": x.grad.tolist(), "dy": y.grad.tolist()}
`);
    expect(grads.dx).toEqual([1.0, 1.0]);
    expect(grads.dy).toEqual([1.0, 1.0]);
  });

  it("matmul backward: d/dA of (A @ B).sum() is rowsum(B) broadcast", async () => {
    const target = await getJitTarget();
    const grads = await target.run<{ dA: number[][]; dB: number[][] }>(`
import browsergrad_jit as bg
import numpy as np
A = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
B = bg.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), requires_grad=True)
out = (A @ B).sum()
out.backward()
{"dA": A.grad.tolist(), "dB": B.grad.tolist()}
`);
    // dA = ones @ B.T = [[5+6, 7+8], [5+6, 7+8]] = [[11, 15], [11, 15]]
    expect(grads.dA).toEqual([
      [11.0, 15.0],
      [11.0, 15.0],
    ]);
    // dB = A.T @ ones = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
    expect(grads.dB).toEqual([
      [4.0, 4.0],
      [6.0, 6.0],
    ]);
  });

  it("non-grad tensor's .grad stays None after backward()", async () => {
    const target = await getJitTarget();
    const is_none = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32), requires_grad=False)
y = bg.from_numpy(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
out = (x + y).sum()
out.backward()
x.grad is None
`);
    expect(is_none).toBe(true);
  });

  it("grads accumulate across backward() calls (PyTorch semantics)", async () => {
    const target = await getJitTarget();
    const grad = await target.run<number[]>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
(x.sum()).backward()
(x.sum()).backward()
x.grad.tolist()
`);
    expect(grad).toEqual([2.0, 2.0, 2.0]);
  });
});
