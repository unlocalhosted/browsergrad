/**
 * TDD'd MaxPool2d + AvgPool2d.
 *
 * Public-interface-only tests. Oracles are NumPy (forward) and finite
 * differences (gradients). One behavior per test, one impl change per cycle.
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

describe("MaxPool2d — forward", () => {
  beforeAll(reset);

  it("tracer: 2x2 pool on a 2x2 input → single max value", async () => {
    // With kernel=2 over a 2x2 input, the whole input is one window;
    // output is its max. Hand-derivable from any 4-element input.
    const result = await target.run<number>(`
${PRELUDE}
pool = nn.MaxPool2d(2)
x = grad.Tensor(np.array([[[[3.0, 1.0], [4.0, 2.0]]]], dtype=np.float32))
y = pool(x)
float(y.data[0, 0, 0, 0])
`);
    expect(result).toBe(4.0);
  });

  it("2x2 pool stride=2 on 4x4 input → 2x2 max-of-window output", async () => {
    // Oracle: explicit numpy slicing — for each output position (i, j),
    // take the max of x[i*2:(i+1)*2, j*2:(j+1)*2]. Independent of our impl.
    const result = await target.run<{
      ours: number[][];
      ref: number[][];
    }>(`
${PRELUDE}
np.random.seed(0)
X = np.random.randn(1, 1, 4, 4).astype(np.float32)
pool = nn.MaxPool2d(2)
y = pool(grad.Tensor(X))

ref = np.zeros((2, 2), dtype=np.float32)
for i in range(2):
    for j in range(2):
        ref[i, j] = X[0, 0, i*2:(i+1)*2, j*2:(j+1)*2].max()

{"ours": y.data[0, 0].tolist(), "ref": ref.tolist()}
`);
    expect(result.ours).toEqual(result.ref);
  });
});

describe("AvgPool2d — forward", () => {
  beforeAll(reset);

  it("2x2 pool stride=2 on 4x4 input → 2x2 mean-of-window output", async () => {
    const result = await target.run<{
      ours: number[][];
      ref: number[][];
    }>(`
${PRELUDE}
np.random.seed(1)
X = np.random.randn(1, 1, 4, 4).astype(np.float32)
pool = nn.AvgPool2d(2)
y = pool(grad.Tensor(X))

ref = np.zeros((2, 2), dtype=np.float32)
for i in range(2):
    for j in range(2):
        ref[i, j] = X[0, 0, i*2:(i+1)*2, j*2:(j+1)*2].mean()

{"ours": y.data[0, 0].tolist(), "ref": ref.tolist()}
`);
    expect(result.ours.length).toBe(result.ref.length);
    for (let i = 0; i < result.ours.length; i++) {
      for (let j = 0; j < result.ours[i]!.length; j++) {
        expect(Math.abs(result.ours[i]![j]! - result.ref[i]![j]!)).toBeLessThan(1e-6);
      }
    }
  });
});

describe("pooling — independence properties", () => {
  beforeAll(reset);

  it("max-pools each channel independently", async () => {
    // Two channels with different magnitudes — their pool maxes must
    // come from within their own channel, never cross-channel.
    const result = await target.run<{
      ch0: number;
      ch1: number;
    }>(`
${PRELUDE}
X = np.array([[
  [[1, 2], [3, 4]],
  [[100, 200], [300, 400]],
]], dtype=np.float32)
pool = nn.MaxPool2d(2)
y = pool(grad.Tensor(X))
{"ch0": float(y.data[0, 0, 0, 0]), "ch1": float(y.data[0, 1, 0, 0])}
`);
    expect(result.ch0).toBe(4);
    expect(result.ch1).toBe(400);
  });

  it("pools each sample in a batch independently", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
X = np.array([
  [[[1, 2], [3, 4]]],
  [[[10, 20], [30, 40]]],
  [[[100, 200], [300, 400]]],
], dtype=np.float32)
pool = nn.MaxPool2d(2)
y = pool(grad.Tensor(X))
[float(y.data[n, 0, 0, 0]) for n in range(3)]
`);
    expect(result).toEqual([4, 40, 400]);
  });

  it("supports overlapping windows when stride < kernel_size", async () => {
    // 4-wide input, 3-wide kernel, stride=1 → 2 windows that overlap.
    const result = await target.run<number[][]>(`
${PRELUDE}
pool = nn.MaxPool2d(3, stride=1)
X = np.array([[[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
]]], dtype=np.float32)
y = pool(grad.Tensor(X))
y.data[0, 0].tolist()
`);
    // Windows at (0,0), (0,1), (1,0), (1,1) — each takes max of 3x3 region.
    //   (0,0): max of [1..11]   = 11
    //   (0,1): max of [2..12]   = 12
    //   (1,0): max of [5..15]   = 15
    //   (1,1): max of [6..16]   = 16
    expect(result).toEqual([
      [11, 12],
      [15, 16],
    ]);
  });
});

describe("MaxPool2d — backward", () => {
  beforeAll(reset);

  it("d(sum(y))/d(x) places 1 only at each window's argmax", async () => {
    // Carefully chosen input — every 2x2 window has a unique max.
    // Hand-derived oracle: for loss = y.sum(), grad_x[r,c] = 1 iff (r,c)
    // is the argmax of the window that produced one of y's elements.
    const result = await target.run<number[][]>(`
${PRELUDE}
X = np.array([[[
  [1.0, 9.0, 2.0, 8.0],
  [3.0, 4.0, 5.0, 6.0],
  [11.0, 12.0, 15.0, 14.0],
  [10.0, 7.0, 13.0, 16.0],
]]], dtype=np.float32)
pool = nn.MaxPool2d(2)
x_t = grad.Tensor(X, requires_grad=True)
y = pool(x_t)
y.sum().backward()
x_t.grad.data[0, 0].tolist()
`);
    // Window argmaxes:
    //   (0..2, 0..2): max = 9 at (0, 1)
    //   (0..2, 2..4): max = 8 at (0, 3)
    //   (2..4, 0..2): max = 12 at (2, 1)
    //   (2..4, 2..4): max = 16 at (3, 3)
    const expected = [
      [0, 1, 0, 1],
      [0, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 1],
    ];
    expect(result).toEqual(expected);
  });

  it("d(sum(y))/d(x) matches finite differences on a random input", async () => {
    // Independent numerical oracle. Max is non-smooth at ties, so we use
    // a Gaussian random input (probability zero of two equal values).
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(2)
X = np.random.randn(1, 1, 4, 4).astype(np.float32)

pool = nn.MaxPool2d(2)
x_t = grad.Tensor(X, requires_grad=True)
pool(x_t).sum().backward()
analytic = x_t.grad.data.flatten().tolist()

eps = 1e-3
fd = np.zeros_like(X)
for i in range(X.shape[2]):
    for j in range(X.shape[3]):
        Xp = X.copy(); Xp[0, 0, i, j] += eps
        lp = float(pool(grad.Tensor(Xp)).sum().item())
        Xm = X.copy(); Xm[0, 0, i, j] -= eps
        lm = float(pool(grad.Tensor(Xm)).sum().item())
        fd[0, 0, i, j] = (lp - lm) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    expect(result.analytic.length).toBe(result.finite_diff.length);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("AvgPool2d — backward", () => {
  beforeAll(reset);

  it("d(sum(y))/d(x) = 1/(K*K) at every position covered by a window", async () => {
    // Non-overlapping 2x2 pool over a 4x4 input: every input cell is in
    // exactly one window, so its gradient is 1/(2*2) = 0.25.
    const result = await target.run<number[][]>(`
${PRELUDE}
pool = nn.AvgPool2d(2)
X = np.zeros((1, 1, 4, 4), dtype=np.float32)
x_t = grad.Tensor(X, requires_grad=True)
pool(x_t).sum().backward()
x_t.grad.data[0, 0].tolist()
`);
    const expected = Array.from({ length: 4 }, () => [0.25, 0.25, 0.25, 0.25]);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(Math.abs(result[i]![j]! - expected[i]![j]!)).toBeLessThan(1e-6);
      }
    }
  });

  it("d(sum(y))/d(x) matches finite differences on a random input", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(3)
X = np.random.randn(1, 1, 4, 4).astype(np.float32)
pool = nn.AvgPool2d(2)
x_t = grad.Tensor(X, requires_grad=True)
pool(x_t).sum().backward()
analytic = x_t.grad.data.flatten().tolist()

eps = 1e-3
fd = np.zeros_like(X)
for i in range(X.shape[2]):
    for j in range(X.shape[3]):
        Xp = X.copy(); Xp[0, 0, i, j] += eps
        lp = float(pool(grad.Tensor(Xp)).sum().item())
        Xm = X.copy(); Xm[0, 0, i, j] -= eps
        lm = float(pool(grad.Tensor(Xm)).sum().item())
        fd[0, 0, i, j] = (lp - lm) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    expect(result.analytic.length).toBe(result.finite_diff.length);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("end-to-end: tiny CNN with Conv + MaxPool + Linear", () => {
  beforeAll(reset);

  it("trains a 2-class image classifier on synthetic shapes", async () => {
    // Two classes of 6x6 images:
    //   class 0: bright top half, dark bottom
    //   class 1: dark top, bright bottom
    // A 3x3 conv + 2x2 max-pool + linear should learn this easily.
    const result = await target.run<{
      initial_acc: number;
      final_acc: number;
      final_loss: number;
    }>(`
${PRELUDE}
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
np.random.seed(11)

def gen_class_0(n):
    out = np.zeros((n, 1, 6, 6), dtype=np.float32)
    out[:, 0, :3, :] = 1.0
    return out + 0.1 * np.random.randn(n, 1, 6, 6).astype(np.float32)
def gen_class_1(n):
    out = np.zeros((n, 1, 6, 6), dtype=np.float32)
    out[:, 0, 3:, :] = 1.0
    return out + 0.1 * np.random.randn(n, 1, 6, 6).astype(np.float32)

X = np.concatenate([gen_class_0(32), gen_class_1(32)], axis=0)
y_labels = np.array([0] * 32 + [1] * 32, dtype=np.int64)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(2 * 3 * 3, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv(x))
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        return self.fc(h)

model = TinyCNN()
opt = optim.Adam(model.parameters(), lr=0.05)

def accuracy(model, X, y):
    logits = model(grad.Tensor(X))
    preds = np.argmax(logits.data, axis=1)
    return float((preds == y).mean())

initial_acc = accuracy(model, X, y_labels)
for _ in range(60):
    opt.zero_grad()
    logits = model(grad.Tensor(X))
    loss = F.cross_entropy_loss(logits, y_labels)
    loss.backward()
    opt.step()

{"initial_acc": initial_acc, "final_acc": accuracy(model, X, y_labels), "final_loss": float(loss.item())}
`);
    // Random init gives ~50% on a 2-class problem; a correctly-wired CNN
    // (conv fwd/bwd + max-pool fwd/bwd + reshape + linear + xent + Adam)
    // gets to >95% in 60 steps. Anything less means the pipeline is broken.
    expect(result.final_acc).toBeGreaterThan(0.95);
    expect(result.final_loss).toBeLessThan(0.5);
  });
});
