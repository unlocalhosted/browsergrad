/**
 * Arithmetic + realization integration tests.
 *
 * Asserts that the public arithmetic surface (+, -, *, /, @, comparison
 * dunders, reductions, shape ops) builds the right IR and realizes to
 * the expected NumPy result.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("TensorProxy arithmetic", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("addition broadcasts and realizes", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
a = bg.tensor([1.0, 2.0, 3.0])
b = bg.tensor([10.0, 20.0, 30.0])
(a + b).tolist()
`);
    expect(result).toEqual([11.0, 22.0, 33.0]);
  });

  it("scalar + tensor coerces and realizes", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
a = bg.tensor([1.0, 2.0, 3.0])
(a + 5.0).tolist()
`);
    expect(result).toEqual([6.0, 7.0, 8.0]);
  });

  it("subtraction realizes correctly", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
a = bg.tensor([5.0, 5.0, 5.0])
b = bg.tensor([1.0, 2.0, 3.0])
(a - b).tolist()
`);
    expect(result).toEqual([4.0, 3.0, 2.0]);
  });

  it("multiplication is elementwise", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
a = bg.tensor([2.0, 3.0, 4.0])
(a * a).tolist()
`);
    expect(result).toEqual([4.0, 9.0, 16.0]);
  });

  it("matmul on 2-D matrices matches NumPy", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[][]>(`
import browsergrad_jit as bg
a = bg.tensor([[1.0, 2.0], [3.0, 4.0]])
b = bg.tensor([[5.0, 6.0], [7.0, 8.0]])
(a @ b).tolist()
`);
    expect(result).toEqual([
      [19.0, 22.0],
      [43.0, 50.0],
    ]);
  });

  it("negation is in-graph", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
a = bg.tensor([1.0, -2.0, 3.0])
(-a).tolist()
`);
    expect(result).toEqual([-1.0, 2.0, -3.0]);
  });

  it("division handles non-integer dtypes", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
a = bg.tensor([10.0, 20.0, 30.0])
(a / 2.0).tolist()
`);
    expect(result).toEqual([5.0, 10.0, 15.0]);
  });

  it("sum reduces to scalar", async () => {
    const target = await getJitTarget();
    const result = await target.run<number>(`
import browsergrad_jit as bg
bg.tensor([1.0, 2.0, 3.0, 4.0]).sum().item()
`);
    expect(result).toBe(10.0);
  });

  it("mean reduces to scalar", async () => {
    const target = await getJitTarget();
    const result = await target.run<number>(`
import browsergrad_jit as bg
bg.tensor([2.0, 4.0, 6.0, 8.0]).mean().item()
`);
    expect(result).toBe(5.0);
  });

  it("reshape preserves data ordering", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[][]>(`
import browsergrad_jit as bg
bg.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2).tolist()
`);
    expect(result).toEqual([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
  });

  it("transpose swaps the right dims", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[][]>(`
import browsergrad_jit as bg
m = bg.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
m.T.tolist()
`);
    expect(result).toEqual([
      [1.0, 4.0],
      [2.0, 5.0],
      [3.0, 6.0],
    ]);
  });

  it("comparison dunders produce bool tensors", async () => {
    const target = await getJitTarget();
    const result = await target.run<boolean[]>(`
import browsergrad_jit as bg
a = bg.tensor([1.0, 2.0, 3.0])
b = bg.tensor([2.0, 2.0, 2.0])
(a > b).tolist()
`);
    expect(result).toEqual([false, false, true]);
  });

  it("composite expression: x @ W + b", async () => {
    const target = await getJitTarget();
    const result = await target.run<number[]>(`
import browsergrad_jit as bg
x = bg.tensor([1.0, 2.0])
W = bg.tensor([[1.0, 0.0], [0.0, 1.0]])
b = bg.tensor([10.0, 20.0])
(x @ W + b).tolist()
`);
    expect(result).toEqual([11.0, 22.0]);
  });
});

describe("Python protocol realization", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("__bool__ realizes a 1-element tensor", async () => {
    const target = await getJitTarget();
    const result = await target.run<boolean>(`
import browsergrad_jit as bg
t = bg.tensor([0.5])
bool(t)
`);
    expect(result).toBe(true);
  });

  it("__float__ realizes a scalar tensor", async () => {
    const target = await getJitTarget();
    const result = await target.run<number>(`
import browsergrad_jit as bg
t = bg.tensor([1.0, 2.0, 3.0]).sum()
float(t)
`);
    expect(result).toBe(6.0);
  });

  it("__bool__ refuses multi-element ambiguity", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
t = bg.tensor([1.0, 2.0])
try:
    bool(t)
    result = "no_error"
except RuntimeError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/ambiguous/);
  });

  it("__pow__ supports integer exponents (MSE idiom)", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      sq: number[];
      cube: number[];
      mse: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
{
    "sq": (x ** 2).numpy().tolist(),
    "cube": (x ** 3).numpy().tolist(),
    "mse": float(((x - 1.0) ** 2).mean().numpy()),
}
`);
    expect(result.sq).toEqual([1, 4, 9]);
    expect(result.cube).toEqual([1, 8, 27]);
    // (0**2 + 1**2 + 2**2) / 3 = 5/3
    expect(result.mse).toBeCloseTo(5 / 3, 5);
  });

  it("__pow__ with float exponent uses exp(log(x) * n)", async () => {
    const target = await getJitTarget();
    const r = await target.run<{ values: number[] }>(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 4.0, 9.0], dtype=np.float32))
{"values": (x ** 0.5).numpy().tolist()}
`);
    // sqrt of [1, 4, 9] = [1, 2, 3]
    expect(r.values[0]).toBeCloseTo(1, 4);
    expect(r.values[1]).toBeCloseTo(2, 4);
    expect(r.values[2]).toBeCloseTo(3, 4);
  });

  it("__pow__ backward through MSE works (the canonical training-loop idiom)", async () => {
    const target = await getJitTarget();
    const r = await target.run<{ max_diff: number }>(`
import browsergrad_jit as bg
import numpy as np

x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
loss = ((x - 1.0) ** 2).mean()
loss.backward()

# d/dx mean((x-1)^2) = (2/N)*(x-1)
expected = (2.0 / 3) * (np.array([1.0, 2.0, 3.0]) - 1.0)
{"max_diff": float(np.max(np.abs(x.grad.numpy() - expected)))}
`);
    expect(r.max_diff).toBeLessThan(1e-5);
  });
});
