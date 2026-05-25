/**
 * TDD'd Conv2d behavior — one test, one implementation, repeat.
 *
 * Tests use only the public `nn.Conv2d(...)` surface plus the standard
 * tensor / autograd API. No private imports, no internal-helper testing.
 * Test oracles are always independent of the implementation:
 *   - hand-derived numerical results, or
 *   - scipy.signal.correlate2d for the spatial-correlation reference, or
 *   - finite-difference checks for gradients.
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

describe("Conv2d — forward", () => {
  beforeAll(reset);

  it("tracer: 1x1 kernel, 1 in / 1 out channel reduces to weight*x + bias", async () => {
    // With kernel_size=1 and a single in/out channel, Conv2d collapses to
    // y[n, 0, h, w] = weight * x[n, 0, h, w] + bias — pure elementwise.
    // We pick weight=2.0 and bias=0.5 so every expected value is hand-computable.
    const result = await target.run<number[][][][]>(`
${PRELUDE}
conv = nn.Conv2d(1, 1, 1)
conv.weight.data[:] = 2.0
conv.bias.data[:] = 0.5
x = grad.Tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
y = conv(x)
y.tolist()
`);
    expect(result).toEqual([[[[2.5, 4.5], [6.5, 8.5]]]]);
  });

  it("3x3 kernel, 1 in / 1 out channel matches naive numpy correlation", async () => {
    // Oracle: a triple-nested numpy loop that does exactly what correlation
    // means — for each output position (i, j), sum-over-(kh, kw) of
    // W[kh, kw] * X[i+kh, j+kw]. Independent from our impl entirely;
    // readable enough to be lesson material.
    const result = await target.run<{
      ours: number[][][][];
      ref: number[][];
    }>(`
${PRELUDE}
np.random.seed(0)
W = np.random.randn(1, 1, 3, 3).astype(np.float32)
X = np.random.randn(1, 1, 5, 5).astype(np.float32)

conv = nn.Conv2d(1, 1, 3, bias=False)
conv.weight.data[:] = W

x = grad.Tensor(X)
y = conv(x)

# Numpy reference: valid correlation
H_out, W_out = 3, 3
ref = np.zeros((H_out, W_out), dtype=np.float32)
for i in range(H_out):
    for j in range(W_out):
        ref[i, j] = (W[0, 0] * X[0, 0, i:i+3, j:j+3]).sum()

{"ours": y.tolist(), "ref": ref.tolist()}
`);
    expect(result.ours.length).toBe(1);
    expect(result.ours[0]!.length).toBe(1);
    const our_flat = result.ours[0]![0]!.flat();
    const ref_flat = result.ref.flat();
    expect(our_flat.length).toBe(ref_flat.length);
    for (let i = 0; i < our_flat.length; i++) {
      expect(Math.abs(our_flat[i]! - ref_flat[i]!)).toBeLessThan(1e-4);
    }
  });

  it("multiple output channels: each channel acts as an independent filter", async () => {
    // Two output channels with different fixed kernels: one all-ones,
    // one all-twos. Output channel 0 should be sum-of-window;
    // output channel 1 should be 2 * sum-of-window.
    const result = await target.run<{
      ch0: number[][];
      ch1: number[][];
    }>(`
${PRELUDE}
conv = nn.Conv2d(1, 2, 2, bias=False)
conv.weight.data[0, 0, :, :] = 1.0
conv.weight.data[1, 0, :, :] = 2.0
X = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
y = conv(grad.Tensor(X))
# y shape: (1, 2, 2, 2)
{"ch0": y.data[0, 0].tolist(), "ch1": y.data[0, 1].tolist()}
`);
    // 2x2 windows starting at (i, j):
    //   (0,0): [1,2,4,5] sum=12
    //   (0,1): [2,3,5,6] sum=16
    //   (1,0): [4,5,7,8] sum=24
    //   (1,1): [5,6,8,9] sum=28
    expect(result.ch0).toEqual([
      [12, 16],
      [24, 28],
    ]);
    expect(result.ch1).toEqual([
      [24, 32],
      [48, 56],
    ]);
  });

  it("multiple input channels are summed across", async () => {
    // Two input channels, one output channel, 1x1 kernel. With both weights
    // set to 1.0, output is the elementwise sum of the two input channels.
    const result = await target.run<number[][]>(`
${PRELUDE}
conv = nn.Conv2d(2, 1, 1, bias=False)
conv.weight.data[0, 0, 0, 0] = 1.0
conv.weight.data[0, 1, 0, 0] = 1.0
X = np.array([[
  [[1, 2], [3, 4]],
  [[10, 20], [30, 40]],
]], dtype=np.float32)
y = conv(grad.Tensor(X))
y.data[0, 0].tolist()
`);
    expect(result).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });

  it("batch dimension: each sample convolves independently", async () => {
    const result = await target.run<number[][][]>(`
${PRELUDE}
conv = nn.Conv2d(1, 1, 2, bias=False)
conv.weight.data[:] = 1.0
X = np.array([
  [[[1, 1], [1, 1]]],   # sample 0 — 2x2 of ones → sum=4
  [[[2, 2], [2, 2]]],   # sample 1 — 2x2 of twos → sum=8
], dtype=np.float32)
y = conv(grad.Tensor(X))
# y shape: (2, 1, 1, 1)
[y.data[0, 0].tolist(), y.data[1, 0].tolist()]
`);
    expect(result).toEqual([[[4]], [[8]]]);
  });

  it("stride=2 skips every other position", async () => {
    // 4x4 input with 2x2 kernel and stride=2 → output is 2x2,
    // sampling at positions (0,0), (0,2), (2,0), (2,2).
    const result = await target.run<number[][]>(`
${PRELUDE}
conv = nn.Conv2d(1, 1, 2, stride=2, bias=False)
conv.weight.data[:] = 1.0
X = np.array([[[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
]]], dtype=np.float32)
y = conv(grad.Tensor(X))
# Output positions sample windows at (0,0), (0,2), (2,0), (2,2)
y.data[0, 0].tolist()
`);
    // Windows summed:
    //   (0,0): 1+2+5+6 = 14
    //   (0,2): 3+4+7+8 = 22
    //   (2,0): 9+10+13+14 = 46
    //   (2,2): 11+12+15+16 = 54
    expect(result).toEqual([
      [14, 22],
      [46, 54],
    ]);
  });

  it("padding=1 with 3x3 kernel preserves spatial dims", async () => {
    // 3x3 kernel + padding=1 + stride=1 → output H/W = input H/W.
    // We verify two things:
    //   - Output shape matches input H/W (3x3)
    //   - Corner output equals sum of valid (non-padded) elements
    const result = await target.run<{
      shape: number[];
      corner_topleft: number;
    }>(`
${PRELUDE}
conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
conv.weight.data[:] = 1.0
X = np.array([[[
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
]]], dtype=np.float32)
y = conv(grad.Tensor(X))
{"shape": list(y.shape), "corner_topleft": float(y.data[0, 0, 0, 0])}
`);
    expect(result.shape).toEqual([1, 1, 3, 3]);
    // Top-left output samples a 3x3 window from input centered at (0, 0)
    // with padded zeros at the missing positions. Visible elements: x[0,0,0,0]=1,
    // x[0,0,0,1]=2, x[0,0,1,0]=4, x[0,0,1,1]=5 → sum=12.
    expect(result.corner_topleft).toBe(12);
  });
});
