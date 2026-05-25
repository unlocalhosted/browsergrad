/**
 * TDD'd nn.Conv1d, nn.BatchNorm1d, nn.Flatten.
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

describe("Conv1d", () => {
  beforeAll(reset);

  it("tracer: kernel_size=1, 1 in/1 out channel → y = w*x + b", async () => {
    const result = await target.run<number[][][]>(`
${PRELUDE}
conv = nn.Conv1d(1, 1, 1)
conv.weight.data[:] = 3.0
conv.bias.data[:] = -1.0
X = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
y = conv(grad.Tensor(X))
y.tolist()
`);
    // 3 * x - 1 per element
    expect(result).toEqual([[[2, 5, 8, 11]]]);
  });

  it("kernel_size=3 stride=1 padding=0 matches numpy 1D correlation", async () => {
    const result = await target.run<{
      ours: number[];
      ref: number[];
    }>(`
${PRELUDE}
np.random.seed(0)
W = np.random.randn(1, 1, 3).astype(np.float32)
X = np.random.randn(1, 1, 8).astype(np.float32)
conv = nn.Conv1d(1, 1, 3, bias=False)
conv.weight.data[:] = W
y = conv(grad.Tensor(X))
# Reference: correlation
L_out = 8 - 3 + 1
ref = np.zeros(L_out, dtype=np.float32)
for i in range(L_out):
    ref[i] = (W[0, 0] * X[0, 0, i:i+3]).sum()
{"ours": y.data[0, 0].tolist(), "ref": ref.tolist()}
`);
    expect(result.ours.length).toBe(result.ref.length);
    for (let i = 0; i < result.ours.length; i++) {
      expect(Math.abs(result.ours[i]! - result.ref[i]!)).toBeLessThan(1e-4);
    }
  });

  it("stride=2 and padding=1 produce correct output length", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
conv = nn.Conv1d(2, 4, 3, stride=2, padding=1, bias=False)
X = np.random.randn(2, 2, 10).astype(np.float32)
y = conv(grad.Tensor(X))
list(y.shape)
`);
    // L_out = (10 + 2*1 - 3) // 2 + 1 = 9 // 2 + 1 = 5
    expect(result).toEqual([2, 4, 5]);
  });

  it("backward d/d(weight) matches finite differences", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(2)
conv = nn.Conv1d(1, 1, 3, bias=False)
W_orig = conv.weight.data.copy()
X = np.random.randn(1, 1, 6).astype(np.float32) * 0.5

y = conv(grad.Tensor(X))
y.sum().backward()
analytic = conv.weight.grad.data.flatten().tolist()

eps = 1e-3
fd = np.zeros_like(W_orig)
for k in range(W_orig.shape[2]):
    conv.weight.data[:] = W_orig
    conv.weight.data[0, 0, k] += eps
    lp = float(conv(grad.Tensor(X)).sum().item())
    conv.weight.data[:] = W_orig
    conv.weight.data[0, 0, k] -= eps
    lm = float(conv(grad.Tensor(X)).sum().item())
    fd[0, 0, k] = (lp - lm) / (2 * eps)
conv.weight.data[:] = W_orig

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });

  it("backward d/d(input) matches finite differences", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(3)
conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
X = np.random.randn(1, 1, 5).astype(np.float32) * 0.3
x_t = grad.Tensor(X, requires_grad=True)
y = conv(x_t)
y.sum().backward()
analytic = x_t.grad.data.flatten().tolist()

eps = 1e-3
fd = np.zeros_like(X)
for i in range(X.shape[2]):
    Xp = X.copy(); Xp[0, 0, i] += eps
    Xm = X.copy(); Xm[0, 0, i] -= eps
    fd[0, 0, i] = (float(conv(grad.Tensor(Xp)).sum().item()) - float(conv(grad.Tensor(Xm)).sum().item())) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("BatchNorm1d", () => {
  beforeAll(reset);

  it("normalizes 3D input (N, C, L) per channel over (N, L)", async () => {
    const result = await target.run<{
      means: number[];
      vars: number[];
    }>(`
${PRELUDE}
np.random.seed(0)
X = np.random.randn(4, 3, 6).astype(np.float32) * 5.0 + 2.0
bn = nn.BatchNorm1d(3, affine=False)
y = bn(grad.Tensor(X))
{
  "means": y.data.mean(axis=(0, 2)).tolist(),
  "vars": y.data.var(axis=(0, 2)).tolist(),
}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const v of result.vars) expect(Math.abs(v - 1)).toBeLessThan(1e-3);
  });

  it("normalizes 2D input (N, C) per channel over N", async () => {
    const result = await target.run<{
      means: number[];
      vars: number[];
    }>(`
${PRELUDE}
np.random.seed(1)
X = np.random.randn(32, 4).astype(np.float32) * 3.0 + 1.0
bn = nn.BatchNorm1d(4, affine=False)
y = bn(grad.Tensor(X))
{
  "means": y.data.mean(axis=0).tolist(),
  "vars": y.data.var(axis=0).tolist(),
}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const v of result.vars) expect(Math.abs(v - 1)).toBeLessThan(1e-3);
  });

  it("backward d/d(input) matches finite differences (3D)", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(2)
X = np.random.randn(3, 2, 5).astype(np.float32)
bn = nn.BatchNorm1d(2, affine=False)

x_t = grad.Tensor(X, requires_grad=True)
bn(x_t).sum().backward()
analytic = x_t.grad.data.flatten().tolist()

def loss(X_):
    bn_local = nn.BatchNorm1d(2, affine=False, track_running_stats=False)
    return float(bn_local(grad.Tensor(X_)).sum().item())

eps = 1e-3
fd = np.zeros_like(X)
for n in range(X.shape[0]):
    for c in range(X.shape[1]):
        for l in range(X.shape[2]):
            Xp = X.copy(); Xp[n, c, l] += eps
            Xm = X.copy(); Xm[n, c, l] -= eps
            fd[n, c, l] = (loss(Xp) - loss(Xm)) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("Flatten", () => {
  beforeAll(reset);

  it("flattens (N, C, H, W) → (N, C*H*W) preserving order", async () => {
    const result = await target.run<{
      shape: number[];
      values_match: boolean;
    }>(`
${PRELUDE}
X = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
flat = nn.Flatten()
y = flat(grad.Tensor(X))
{
  "shape": list(y.shape),
  "values_match": bool((y.data == X.reshape(2, -1)).all()),
}
`);
    expect(result.shape).toEqual([2, 12]);
    expect(result.values_match).toBe(true);
  });

  it("backward passes gradient through reshape", async () => {
    const result = await target.run<number[][][][]>(`
${PRELUDE}
X = np.ones((2, 3, 2, 2), dtype=np.float32)
flat = nn.Flatten()
x_t = grad.Tensor(X, requires_grad=True)
y = flat(x_t)
y.sum().backward()
x_t.grad.data.tolist()
`);
    // d(sum)/dx = 1 everywhere
    expect(result.length).toBe(2);
    for (const sample of result) {
      for (const channel of sample) {
        for (const row of channel) {
          for (const v of row) expect(v).toBe(1);
        }
      }
    }
  });
});

describe("end-to-end Conv1d sequence classifier", () => {
  beforeAll(reset);

  it("Conv1d + BN1d + Flatten + Linear classifier trains on synthetic seqs", async () => {
    const result = await target.run<{
      final_acc: number;
    }>(`
${PRELUDE}
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
np.random.seed(19)

# Two-class 1D-sequence problem: class 0 has bias on left, class 1 on right
def gen_class(n, left_bias):
    base = np.random.randn(n, 1, 8).astype(np.float32) * 0.3
    if left_bias:
        base[:, 0, :4] += 1.0
    else:
        base[:, 0, 4:] += 1.0
    return base

X = np.concatenate([gen_class(32, True), gen_class(32, False)], axis=0)
y_labels = np.array([0] * 32 + [1] * 32, dtype=np.int64)

class SeqCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 3, padding=1)
        self.bn = nn.BatchNorm1d(4)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(4 * 8, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc(self.flat(self.relu(self.bn(self.conv(x)))))

model = SeqCNN()
opt = optim.Adam(model.parameters(), lr=0.05)

for _ in range(80):
    opt.zero_grad()
    loss = F.cross_entropy_loss(model(grad.Tensor(X)), y_labels)
    loss.backward()
    opt.step()

model.eval()
preds = np.argmax(model(grad.Tensor(X)).data, axis=1)
{"final_acc": float((preds == y_labels).mean())}
`);
    expect(result.final_acc).toBeGreaterThan(0.95);
  });
});
