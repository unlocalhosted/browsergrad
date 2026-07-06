/**
 * Pile A #16 + #17 + #18 — Conv3d / ConvTranspose1d/2d / GroupNorm /
 * InstanceNorm{1,2,3}d / BatchNorm3d / Module hooks.
 *
 * Oracles:
 *  - Conv3d: hand-written triple-loop convolution in NumPy.
 *  - ConvTranspose: shape oracle (full impl deferred but shape matches).
 *  - GroupNorm: standard formula per-group.
 *  - InstanceNorm: GroupNorm with num_groups = C.
 *  - BatchNorm3d: same as BatchNorm2d generalized.
 *  - Hooks: observe forward output, observe backward grad.
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

describe("nn.Conv3d", () => {
  beforeAll(reset);

  it("tuple kernel/stride/padding matches naive numpy correlation", async () => {
    const result = await target.run<{
      ours: number[][][][][];
      ref: number[][][][][];
      shape: number[];
    }>(`
${PRELUDE}
X = (np.arange(1 * 2 * 4 * 4 * 5, dtype=np.float32).reshape(1, 2, 4, 4, 5) - 20.0) / 17.0
W = (np.arange(2 * 2 * 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 2, 3) - 7.0) / 19.0
b = np.array([0.2, -0.4], dtype=np.float32)
conv = nn.Conv3d(2, 2, kernel_size=(2, 2, 3), stride=(1, 2, 1), padding=(1, 0, 1))
conv.weight.data[:] = W
conv.bias.data[:] = b
y = conv(grad.Tensor(X))

pd, ph, pw = 1, 0, 1
sd, sh, sw = 1, 2, 1
kd, kh, kw = 2, 2, 3
xp = np.pad(X, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)), mode="constant")
D_out = (xp.shape[2] - kd) // sd + 1
H_out = (xp.shape[3] - kh) // sh + 1
W_out = (xp.shape[4] - kw) // sw + 1
ref = np.zeros((1, 2, D_out, H_out, W_out), dtype=np.float32)
for n in range(1):
  for co in range(2):
    for od in range(D_out):
      for oh in range(H_out):
        for ow in range(W_out):
          acc = b[co]
          for ci in range(2):
            for dz in range(kd):
              for r in range(kh):
                for c in range(kw):
                  acc += W[co, ci, dz, r, c] * xp[n, ci, od * sd + dz, oh * sh + r, ow * sw + c]
          ref[n, co, od, oh, ow] = acc
{"ours": y.tolist(), "ref": ref.tolist(), "shape": list(y.shape)}
`);
    expect(result.shape).toEqual([1, 2, 5, 2, 5]);
    const ours = result.ours.flat(4);
    const ref = result.ref.flat(4);
    for (let i = 0; i < ours.length; i++) {
      expect(Math.abs(ours[i]! - ref[i]!)).toBeLessThan(1e-4);
    }
  });

  it("groups isolate channel partitions", async () => {
    const result = await target.run<number[][][][]>(`
${PRELUDE}
conv = nn.Conv3d(4, 4, kernel_size=1, groups=2, bias=False)
conv.weight.data[:] = 0.0
conv.weight.data[0, 0, 0, 0, 0] = 1.0
conv.weight.data[1, 1, 0, 0, 0] = 1.0
conv.weight.data[2, 0, 0, 0, 0] = 10.0
conv.weight.data[3, 1, 0, 0, 0] = 10.0
X = np.array([[
  [[[1, 1], [1, 1]]],
  [[[2, 2], [2, 2]]],
  [[[3, 3], [3, 3]]],
  [[[4, 4], [4, 4]]],
]], dtype=np.float32)
y = conv(grad.Tensor(X))
y.data[0].tolist()
`);
    expect(result).toEqual([
      [
        [
          [1, 1],
          [1, 1],
        ],
      ],
      [
        [
          [2, 2],
          [2, 2],
        ],
      ],
      [
        [
          [30, 30],
          [30, 30],
        ],
      ],
      [
        [
          [40, 40],
          [40, 40],
        ],
      ],
    ]);
  });

  it("input gradient matches finite differences", async () => {
    const result = await target.run<{ analytic: number[]; finite: number[] }>(`
${PRELUDE}
np.random.seed(13)
x0 = np.random.randn(1, 1, 3, 2, 2).astype(np.float32) * 0.25
up = np.random.randn(1, 1, 2, 2, 2).astype(np.float32) * 0.2
conv = nn.Conv3d(1, 1, kernel_size=(2, 1, 1), bias=False)
conv.weight.data[:] = np.array([[[[[0.7]], [[-0.3]]]]], dtype=np.float32)
x = grad.Tensor(x0.copy(), requires_grad=True)
loss = (conv(x) * grad.Tensor(up)).sum()
loss.backward()
analytic = x.grad.data.copy()
finite = np.zeros_like(x0)
eps = 1e-3
for idx in np.ndindex(x0.shape):
    xp = x0.copy(); xm = x0.copy()
    xp[idx] += eps; xm[idx] -= eps
    fp = float((conv(grad.Tensor(xp)).data * up).sum())
    fm = float((conv(grad.Tensor(xm)).data * up).sum())
    finite[idx] = (fp - fm) / (2 * eps)
{"analytic": analytic.flatten().tolist(), "finite": finite.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(result.analytic[i]!).toBeCloseTo(result.finite[i]!, 3);
    }
  });
});

describe("nn.ConvTranspose2d", () => {
  beforeAll(reset);

  it("tuple stride/padding/output_padding matches scatter oracle", async () => {
    const result = await target.run<{
      ours: number[][][][];
      ref: number[][][][];
      shape: number[];
    }>(`
${PRELUDE}
X = (np.arange(1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3) - 3.0) / 5.0
W = (np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) - 4.0) / 7.0
b = np.array([0.1, -0.2, 0.3], dtype=np.float32)
conv = nn.ConvTranspose2d(2, 3, kernel_size=(2, 2), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
conv.weight.data[:] = W
conv.bias.data[:] = b
y = conv(grad.Tensor(X))

N, C_in, H, Wi = X.shape
C_out = 3
kh, kw = 2, 2
sh, sw = 2, 1
ph, pw = 1, 0
oph, opw = 1, 0
H_out = (H - 1) * sh - 2 * ph + (kh - 1) + oph + 1
W_out = (Wi - 1) * sw - 2 * pw + (kw - 1) + opw + 1
ref = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
for n in range(N):
  for ci in range(C_in):
    for ih in range(H):
      for iw in range(Wi):
        for r in range(kh):
          oh = ih * sh - ph + r
          if oh < 0 or oh >= H_out:
            continue
          for c in range(kw):
            ow = iw * sw - pw + c
            if 0 <= ow < W_out:
              ref[n, :, oh, ow] += X[n, ci, ih, iw] * W[ci, :, r, c]
ref += b.reshape(1, C_out, 1, 1)
{"ours": y.tolist(), "ref": ref.tolist(), "shape": list(y.shape)}
`);
    expect(result.shape).toEqual([1, 3, 3, 4]);
    const ours = result.ours.flat(3);
    const ref = result.ref.flat(3);
    for (let i = 0; i < ours.length; i++) {
      expect(Math.abs(ours[i]! - ref[i]!)).toBeLessThan(1e-4);
    }
  });

  it("input gradient matches finite differences", async () => {
    const result = await target.run<{ analytic: number[]; finite: number[] }>(`
${PRELUDE}
np.random.seed(14)
x0 = np.random.randn(1, 1, 2, 2).astype(np.float32) * 0.25
up = np.random.randn(1, 1, 3, 3).astype(np.float32) * 0.2
conv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
conv.weight.data[:] = np.array([[[[0.4, -0.2], [0.7, 0.1]]]], dtype=np.float32)
x = grad.Tensor(x0.copy(), requires_grad=True)
loss = (conv(x) * grad.Tensor(up)).sum()
loss.backward()
analytic = x.grad.data.copy()
finite = np.zeros_like(x0)
eps = 1e-3
for idx in np.ndindex(x0.shape):
    xp = x0.copy(); xm = x0.copy()
    xp[idx] += eps; xm[idx] -= eps
    fp = float((conv(grad.Tensor(xp)).data * up).sum())
    fm = float((conv(grad.Tensor(xm)).data * up).sum())
    finite[idx] = (fp - fm) / (2 * eps)
{"analytic": analytic.flatten().tolist(), "finite": finite.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(result.analytic[i]!).toBeCloseTo(result.finite[i]!, 3);
    }
  });
});

describe("nn.GroupNorm", () => {
  beforeAll(reset);

  it("normalizes per group: mean ≈ 0, var ≈ 1 across each group after affine off", async () => {
    const result = await target.run<{ means: number[]; stds: number[] }>(`
${PRELUDE}
np.random.seed(0)
N, C, H, W = 2, 6, 4, 4
gn = nn.GroupNorm(num_groups=3, num_channels=C, affine=False)
x = grad.Tensor(np.random.randn(N, C, H, W).astype(np.float32))
y = gn(x)
y_np = np.asarray(y.tolist())
# Reshape to (N, G, C/G, H, W) and compute per-group means/stds
G = 3
yr = y_np.reshape(N, G, C // G, H, W)
means = yr.mean(axis=(2, 3, 4)).flatten().tolist()
stds = yr.std(axis=(2, 3, 4)).flatten().tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const s of result.stds) expect(s).toBeCloseTo(1.0, 3);
  });

  it("input gradient matches finite differences through group statistics", async () => {
    const result = await target.run<{ analytic: number[]; finite: number[] }>(`
${PRELUDE}
np.random.seed(10)
N, C, H, W = 1, 4, 2, 3
x0 = np.random.randn(N, C, H, W).astype(np.float32) * 0.4
up = np.random.randn(N, C, H, W).astype(np.float32) * 0.3
gn = nn.GroupNorm(num_groups=2, num_channels=C, affine=True)
gn.weight.data[:] = np.array([0.7, 1.1, -0.9, 0.5], dtype=np.float32)
gn.bias.data[:] = np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32)
x = grad.Tensor(x0.copy(), requires_grad=True)
loss = (gn(x) * grad.Tensor(up)).sum()
loss.backward()
analytic = x.grad.data.copy()
finite = np.zeros_like(x0)
eps = 1e-3
for idx in np.ndindex(x0.shape):
    xp = x0.copy(); xm = x0.copy()
    xp[idx] += eps; xm[idx] -= eps
    fp = float((gn(grad.Tensor(xp)).data * up).sum())
    fm = float((gn(grad.Tensor(xm)).data * up).sum())
    finite[idx] = (fp - fm) / (2 * eps)
{"analytic": analytic.flatten().tolist(), "finite": finite.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(result.analytic[i]!).toBeCloseTo(result.finite[i]!, 2);
    }
  });
});

describe("nn.InstanceNorm2d", () => {
  beforeAll(reset);

  it("normalizes each (N, C) plane independently", async () => {
    const result = await target.run<{ means: number[]; stds: number[] }>(`
${PRELUDE}
np.random.seed(1)
N, C, H, W = 2, 4, 5, 5
inst = nn.InstanceNorm2d(C, affine=False)
x = grad.Tensor(np.random.randn(N, C, H, W).astype(np.float32))
y = inst(x)
y_np = np.asarray(y.tolist())
means = y_np.mean(axis=(2, 3)).flatten().tolist()
stds  = y_np.std(axis=(2, 3)).flatten().tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const s of result.stds) expect(s).toBeCloseTo(1.0, 3);
  });

  it("input gradient matches finite differences through per-instance statistics", async () => {
    const result = await target.run<{ analytic: number[]; finite: number[] }>(`
${PRELUDE}
np.random.seed(11)
N, C, H, W = 1, 3, 2, 3
x0 = np.random.randn(N, C, H, W).astype(np.float32) * 0.4
up = np.random.randn(N, C, H, W).astype(np.float32) * 0.25
inst = nn.InstanceNorm2d(C, affine=True)
inst.weight.data[:] = np.array([0.6, -1.2, 0.9], dtype=np.float32)
inst.bias.data[:] = np.array([0.1, 0.2, -0.3], dtype=np.float32)
x = grad.Tensor(x0.copy(), requires_grad=True)
loss = (inst(x) * grad.Tensor(up)).sum()
loss.backward()
analytic = x.grad.data.copy()
finite = np.zeros_like(x0)
eps = 1e-3
for idx in np.ndindex(x0.shape):
    xp = x0.copy(); xm = x0.copy()
    xp[idx] += eps; xm[idx] -= eps
    fp = float((inst(grad.Tensor(xp)).data * up).sum())
    fm = float((inst(grad.Tensor(xm)).data * up).sum())
    finite[idx] = (fp - fm) / (2 * eps)
{"analytic": analytic.flatten().tolist(), "finite": finite.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(result.analytic[i]!).toBeCloseTo(result.finite[i]!, 2);
    }
  });
});

describe("nn.BatchNorm3d", () => {
  beforeAll(reset);

  it("normalizes per channel across (N, D, H, W) in training mode", async () => {
    const result = await target.run<{ means: number[]; stds: number[] }>(`
${PRELUDE}
np.random.seed(2)
N, C, D, H, W = 2, 3, 3, 4, 4
bn = nn.BatchNorm3d(C, affine=False)
bn.train()
x = grad.Tensor(np.random.randn(N, C, D, H, W).astype(np.float32))
y = bn(x)
y_np = np.asarray(y.tolist())
means = y_np.mean(axis=(0, 2, 3, 4)).tolist()
stds  = y_np.std(axis=(0, 2, 3, 4)).tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const s of result.stds) expect(s).toBeCloseTo(1.0, 3);
  });

  it("training-mode input gradient matches finite differences through batch statistics", async () => {
    const result = await target.run<{ analytic: number[]; finite: number[] }>(`
${PRELUDE}
np.random.seed(12)
N, C, D, H, W = 2, 2, 2, 2, 2
x0 = np.random.randn(N, C, D, H, W).astype(np.float32) * 0.35
up = np.random.randn(N, C, D, H, W).astype(np.float32) * 0.2
bn = nn.BatchNorm3d(C, affine=True, track_running_stats=False)
bn.weight.data[:] = np.array([0.8, -0.6], dtype=np.float32)
bn.bias.data[:] = np.array([0.05, 0.15], dtype=np.float32)
bn.train()
x = grad.Tensor(x0.copy(), requires_grad=True)
loss = (bn(x) * grad.Tensor(up)).sum()
loss.backward()
analytic = x.grad.data.copy()
finite = np.zeros_like(x0)
eps = 1e-3
for idx in np.ndindex(x0.shape):
    xp = x0.copy(); xm = x0.copy()
    xp[idx] += eps; xm[idx] -= eps
    fp = float((bn(grad.Tensor(xp)).data * up).sum())
    fm = float((bn(grad.Tensor(xm)).data * up).sum())
    finite[idx] = (fp - fm) / (2 * eps)
{"analytic": analytic.flatten().tolist(), "finite": finite.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(result.analytic[i]!).toBeCloseTo(result.finite[i]!, 2);
    }
  });
});

describe("Module hooks", () => {
  beforeAll(reset);

  it("register_forward_hook fires after forward with (module, input, output)", async () => {
    const result = await target.run<{ fired: boolean; out_shape: number[] }>(`
${PRELUDE}
fc = nn.Linear(3, 5)
captured = {}
def hook(module, input, output):
    captured["module_is_fc"] = module is fc
    captured["output_shape"] = list(output.data.shape)
fc.register_forward_hook(hook)
x = grad.Tensor(np.zeros((2, 3), dtype=np.float32))
y = fc(x)
{"fired": captured.get("module_is_fc", False), "out_shape": captured.get("output_shape", [])}
`);
    expect(result.fired).toBe(true);
    expect(result.out_shape).toEqual([2, 5]);
  });

  it("multiple forward hooks fire in registration order", async () => {
    const result = await target.run<string[]>(`
${PRELUDE}
fc = nn.Linear(2, 2)
order = []
def h1(m, i, o): order.append("h1")
def h2(m, i, o): order.append("h2")
fc.register_forward_hook(h1)
fc.register_forward_hook(h2)
_ = fc(grad.Tensor(np.zeros((1, 2), dtype=np.float32)))
order
`);
    expect(result).toEqual(["h1", "h2"]);
  });
});
