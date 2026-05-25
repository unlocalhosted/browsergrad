/**
 * TDD'd nn.AdaptiveAvgPool2d.
 *
 * Reference: PyTorch's AdaptiveAvgPool2d. For target output (H_out, W_out)
 * and input (H_in, W_in), bin boundaries on each axis are:
 *   start_a(i) = floor(i * dim_in / dim_out)
 *   end_a(i)   = ceil((i + 1) * dim_in / dim_out)
 * Each output cell = mean over its (variable-sized) bin region.
 *
 * Test oracles: hand-derived (global pooling = per-channel mean), numpy loop
 * for arbitrary sizes, finite differences for backward.
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

describe("AdaptiveAvgPool2d — forward", () => {
  beforeAll(reset);

  it("tracer: output_size=1 produces per-channel mean (global avg pool)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
np.random.seed(0)
X = np.random.randn(1, 4, 5, 7).astype(np.float32)
pool = nn.AdaptiveAvgPool2d(1)
y = pool(grad.Tensor(X))
# Expected: (1, 4, 1, 1) of per-channel means
expected = X.mean(axis=(2, 3))[0].tolist()
out = y.data[0, :, 0, 0].tolist()
[float(abs(o - e)) for o, e in zip(out, expected)]
`);
    for (const diff of result) expect(diff).toBeLessThan(1e-5);
  });

  it("output_size matches the requested (H_out, W_out)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
X = np.zeros((2, 3, 8, 12), dtype=np.float32)
pool = nn.AdaptiveAvgPool2d((4, 6))
y = pool(grad.Tensor(X))
list(y.shape)
`);
    expect(result).toEqual([2, 3, 4, 6]);
  });

  it("output matches numpy-loop reference for non-evenly-divisible shapes", async () => {
    // 7x7 → 3x3 has unequal bin widths. Verify against the exact PyTorch formula.
    const result = await target.run<{
      ours: number[][];
      ref: number[][];
    }>(`
${PRELUDE}
np.random.seed(1)
X = np.random.randn(1, 1, 7, 7).astype(np.float32)
pool = nn.AdaptiveAvgPool2d(3)
y = pool(grad.Tensor(X))

H_in, W_in, H_out, W_out = 7, 7, 3, 3
ref = np.zeros((H_out, W_out), dtype=np.float32)
for i in range(H_out):
    sh = (i * H_in) // H_out
    eh = -(-((i + 1) * H_in) // H_out)  # ceil
    for j in range(W_out):
        sw = (j * W_in) // W_out
        ew = -(-((j + 1) * W_in) // W_out)
        ref[i, j] = X[0, 0, sh:eh, sw:ew].mean()

{"ours": y.data[0, 0].tolist(), "ref": ref.tolist()}
`);
    expect(result.ours.length).toBe(result.ref.length);
    for (let i = 0; i < result.ours.length; i++) {
      for (let j = 0; j < result.ours[i]!.length; j++) {
        expect(Math.abs(result.ours[i]![j]! - result.ref[i]![j]!)).toBeLessThan(1e-5);
      }
    }
  });
});

describe("AdaptiveAvgPool2d — backward", () => {
  beforeAll(reset);

  it("d(sum(y))/d(x) matches finite differences", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(2)
X = np.random.randn(1, 1, 5, 5).astype(np.float32)
pool = nn.AdaptiveAvgPool2d(2)
x_t = grad.Tensor(X, requires_grad=True)
pool(x_t).sum().backward()
analytic = x_t.grad.data.flatten().tolist()

eps = 1e-3
fd = np.zeros_like(X)
for i in range(X.shape[2]):
    for j in range(X.shape[3]):
        Xp = X.copy(); Xp[0, 0, i, j] += eps
        Xm = X.copy(); Xm[0, 0, i, j] -= eps
        lp = float(pool(grad.Tensor(Xp)).sum().item())
        lm = float(pool(grad.Tensor(Xm)).sum().item())
        fd[0, 0, i, j] = (lp - lm) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});
