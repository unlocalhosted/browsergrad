/**
 * Pile A #6 (nn.init), #13 (more tensor math), #14 (more shape ops).
 *
 * Independent oracles: NumPy directly, hand-derived for in-place inits.
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
import browsergrad_grad.nn as nn
import numpy as np
`;

describe("nn.init", () => {
  beforeAll(reset);

  it("zeros_ fills the tensor with zeros in place", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
t = grad.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
nn.init.zeros_(t)
t.tolist()
`);
    expect(result).toEqual([0, 0, 0]);
  });

  it("ones_ fills the tensor with ones in place", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
t = grad.Tensor(np.zeros(4))
nn.init.ones_(t)
t.tolist()
`);
    expect(result).toEqual([1, 1, 1, 1]);
  });

  it("uniform_ draws values inside [a, b]", async () => {
    const result = await target.run<{ in_range: boolean; size: number }>(`
${PRELUDE}
t = grad.Tensor(np.zeros(1000))
nn.init.uniform_(t, a=-1.0, b=1.0)
{"in_range": bool(((t.data >= -1.0) & (t.data <= 1.0)).all()), "size": int(t.data.size)}
`);
    expect(result.size).toBe(1000);
    expect(result.in_range).toBe(true);
  });

  it("normal_ produces values whose mean ≈ mean param, std ≈ std param", async () => {
    const result = await target.run<{ m: number; s: number }>(`
${PRELUDE}
t = grad.Tensor(np.zeros(10000))
nn.init.normal_(t, mean=2.0, std=0.5)
{"m": float(t.data.mean()), "s": float(t.data.std())}
`);
    expect(Math.abs(result.m - 2.0)).toBeLessThan(0.05);
    expect(Math.abs(result.s - 0.5)).toBeLessThan(0.05);
  });

  it("kaiming_uniform_ fills with values in the Kaiming bound for given fan_in", async () => {
    // f32 cast can put a value 1 ulp past the float64 bound; tol 1e-3.
    const result = await target.run<boolean>(`
${PRELUDE}
fan_in = 100
bound = float(np.sqrt(6.0 / fan_in))
t = grad.Tensor(np.zeros((100, 100)))
nn.init.kaiming_uniform_(t, a=0)
bool(((t.data >= -bound - 1e-3) & (t.data <= bound + 1e-3)).all())
`);
    expect(result).toBe(true);
  });

  it("xavier_uniform_ fills within Xavier bound for given fan_in/fan_out", async () => {
    const result = await target.run<boolean>(`
${PRELUDE}
fan_in, fan_out = 50, 80
bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
t = grad.Tensor(np.zeros((fan_out, fan_in)))
nn.init.xavier_uniform_(t)
bool(((t.data >= -bound - 1e-3) & (t.data <= bound + 1e-3)).all())
`);
    expect(result).toBe(true);
  });
});

describe("tensor math grab-bag", () => {
  beforeAll(reset);

  it("abs / sign", async () => {
    const result = await target.run<{ abs: number[]; sign: number[] }>(`
${PRELUDE}
x = grad.Tensor([-2.0, 0.0, 3.0, -0.5])
{"abs": x.abs().tolist(), "sign": x.sign().tolist()}
`);
    expect(result.abs).toEqual([2, 0, 3, 0.5]);
    expect(result.sign).toEqual([-1, 0, 1, -1]);
  });

  it("clamp / clip with min and max", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([-2.0, 0.0, 3.0, 7.0])
x.clamp(0.0, 5.0).tolist()
`);
    expect(result).toEqual([0, 0, 3, 5]);
  });

  it("sqrt / pow", async () => {
    const result = await target.run<{ sqrt: number[]; pow3: number[] }>(`
${PRELUDE}
x = grad.Tensor([1.0, 4.0, 9.0, 16.0])
{"sqrt": x.sqrt().tolist(), "pow3": x.pow(3).tolist()}
`);
    expect(result.sqrt).toEqual([1, 2, 3, 4]);
    expect(result.pow3).toEqual([1, 64, 729, 4096]);
  });

  it("where: torch.where(cond, a, b)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
cond = grad.Tensor([1.0, 0.0, 1.0, 0.0])
a = grad.Tensor([10.0, 20.0, 30.0, 40.0])
b = grad.Tensor([100.0, 200.0, 300.0, 400.0])
grad.where(cond, a, b).tolist()
`);
    expect(result).toEqual([10, 200, 30, 400]);
  });

  it("topk returns (values, indices) of the k largest", async () => {
    const result = await target.run<{ values: number[]; indices: number[] }>(`
${PRELUDE}
x = grad.Tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
vals, idx = x.topk(3)
{"values": vals.tolist(), "indices": idx.tolist()}
`);
    // top-3 values are 9, 6, 5 (in descending order)
    expect(result.values).toEqual([9, 6, 5]);
    expect(result.indices).toEqual([5, 7, 4]);
  });
});

describe("shape ops grab-bag", () => {
  beforeAll(reset);

  it("expand broadcasts size-1 dims to a target size", async () => {
    const result = await target.run<{ shape: number[]; values: number[][] }>(`
${PRELUDE}
x = grad.Tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
y = x.expand(3, 4)  # (3, 4)
{"shape": list(y.shape), "values": y.tolist()}
`);
    expect(result.shape).toEqual([3, 4]);
    expect(result.values).toEqual([
      [1, 1, 1, 1],
      [2, 2, 2, 2],
      [3, 3, 3, 3],
    ]);
  });

  it("repeat tiles the tensor", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0])
x.repeat(3).tolist()
`);
    expect(result).toEqual([1, 2, 1, 2, 1, 2]);
  });

  it("flip reverses along a dim", async () => {
    const result = await target.run<{ flipped_0: number[][]; flipped_1: number[][] }>(`
${PRELUDE}
x = grad.Tensor([[1.0, 2.0], [3.0, 4.0]])
{
  "flipped_0": x.flip(0).tolist(),
  "flipped_1": x.flip(1).tolist(),
}
`);
    expect(result.flipped_0).toEqual([[3, 4], [1, 2]]);
    expect(result.flipped_1).toEqual([[2, 1], [4, 3]]);
  });
});
