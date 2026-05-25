/**
 * TDD'd BatchNorm2d + Module.training.
 *
 * Public-interface-only tests. Oracles: hand-derived statistics for forward,
 * finite differences for backward, behavioral checks for train/eval mode.
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

describe("BatchNorm2d — forward (train)", () => {
  beforeAll(reset);

  it("tracer: affine=False normalizes each channel to mean≈0, var≈1", async () => {
    // Hand-derived: BatchNorm with affine off subtracts per-channel batch
    // mean and divides by sqrt(batch var + eps). Output statistics per
    // channel (over N, H, W) must be approximately zero-mean, unit-var.
    const result = await target.run<{
      means: number[];
      vars: number[];
    }>(`
${PRELUDE}
np.random.seed(0)
X = np.random.randn(4, 3, 5, 5).astype(np.float32) * 7.0 + 13.0
bn = nn.BatchNorm2d(3, affine=False)
y = bn(grad.Tensor(X))
# Per-channel mean/var over (N, H, W)
means = y.data.mean(axis=(0, 2, 3)).tolist()
vars_ = y.data.var(axis=(0, 2, 3)).tolist()
{"means": means, "vars": vars_}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    // With eps=1e-5 added inside sqrt, var is very close to 1.
    for (const v of result.vars) expect(Math.abs(v - 1)).toBeLessThan(1e-3);
  });

  it("affine=True applies gamma and beta after normalization", async () => {
    // After normalize, the result has mean≈0 std≈1 per channel. Applying
    // y = gamma * x_hat + beta then has mean=beta, std=gamma per channel.
    const result = await target.run<{
      means: number[];
      stds: number[];
    }>(`
${PRELUDE}
np.random.seed(1)
X = np.random.randn(4, 3, 5, 5).astype(np.float32) * 3.0 - 2.0
bn = nn.BatchNorm2d(3)  # affine=True by default
bn.weight.data[:] = 2.0
bn.bias.data[:] = 10.0
y = bn(grad.Tensor(X))
means = y.data.mean(axis=(0, 2, 3)).tolist()
stds = y.data.std(axis=(0, 2, 3)).tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m - 10)).toBeLessThan(1e-3);
    for (const s of result.stds) expect(Math.abs(s - 2)).toBeLessThan(1e-3);
  });

  it("running_mean / running_var update after each forward (momentum=1.0)", async () => {
    // momentum=1.0 means: running = 1.0 * batch + 0.0 * old. After one
    // forward, the running stats exactly equal the batch stats.
    const result = await target.run<{
      batch_mean: number[];
      running_mean: number[];
      batch_var: number[];
      running_var: number[];
    }>(`
${PRELUDE}
np.random.seed(2)
X = np.random.randn(4, 3, 5, 5).astype(np.float32) * 3.0 + 1.5
bn = nn.BatchNorm2d(3, momentum=1.0, affine=False)
batch_mean = X.mean(axis=(0, 2, 3))
batch_var = X.var(axis=(0, 2, 3))
_ = bn(grad.Tensor(X))
{
  "batch_mean": batch_mean.tolist(),
  "running_mean": bn.running_mean.tolist(),
  "batch_var": batch_var.tolist(),
  "running_var": bn.running_var.tolist(),
}
`);
    expect(result.running_mean.length).toBe(result.batch_mean.length);
    for (let i = 0; i < result.batch_mean.length; i++) {
      expect(Math.abs(result.running_mean[i]! - result.batch_mean[i]!)).toBeLessThan(1e-5);
      expect(Math.abs(result.running_var[i]! - result.batch_var[i]!)).toBeLessThan(1e-4);
    }
  });
});

describe("BatchNorm2d — train/eval mode", () => {
  beforeAll(reset);

  it("eval mode uses running stats and does not update them", async () => {
    // Steer running stats to a known value, then check that:
    //   - eval mode normalizes by those running stats (not the batch's)
    //   - eval mode does NOT update running stats further
    const result = await target.run<{
      train_out_first: number;
      eval_out_first: number;
      running_mean_before: number[];
      running_mean_after_eval: number[];
    }>(`
${PRELUDE}
np.random.seed(3)
bn = nn.BatchNorm2d(2, affine=False, momentum=1.0)

# 1) Train pass to seed running stats from a known batch.
X_train = np.array([[
  [[1, 1], [1, 1]],
  [[5, 5], [5, 5]],
]], dtype=np.float32)
_ = bn(grad.Tensor(X_train))
running_mean_before = bn.running_mean.tolist()

# 2) Switch to eval and pass a DIFFERENT batch.
bn.eval()
X_eval = np.array([[
  [[3, 3], [3, 3]],
  [[3, 3], [3, 3]],
]], dtype=np.float32)
y_eval = bn(grad.Tensor(X_eval))
# In eval, channel 0 normalize uses running_mean[0]=1, running_var[0]=0;
# so y[0,0,0,0] = (3 - 1) / sqrt(0 + eps), which is huge — that confirms
# running stats are used (NOT the batch's perfect-constant stats).
running_mean_after_eval = bn.running_mean.tolist()

# 3) Switch back to train and verify mode flips again — a train pass with
#    momentum=1.0 should overwrite running stats from this new batch.
bn.train()
_ = bn(grad.Tensor(X_eval))
# Now running_mean[0] should be 3, not 1 (because train mode updated it).
running_mean_after_train = bn.running_mean.tolist()

# Compute the expected eval output for first channel/first cell to assert.
# In eval, normed = (3 - 1) / sqrt(0 + 1e-5); without affine.
expected_eval_first = (3.0 - 1.0) / float(np.sqrt(0.0 + bn.eps))

{
  "train_out_first": float(expected_eval_first),
  "eval_out_first": float(y_eval.data[0, 0, 0, 0]),
  "running_mean_before": running_mean_before,
  "running_mean_after_eval": running_mean_after_eval,
}
`);
    // 1. Eval used running stats (huge value, not 0).
    expect(Math.abs(result.eval_out_first - result.train_out_first)).toBeLessThan(1e-1);
    // 2. Eval did NOT update running stats.
    expect(result.running_mean_after_eval).toEqual(result.running_mean_before);
  });
});

describe("BatchNorm2d — backward (train mode)", () => {
  beforeAll(reset);

  it("d(sum(y))/d(input) matches finite differences", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(5)
X = np.random.randn(3, 2, 4, 4).astype(np.float32)
bn = nn.BatchNorm2d(2, affine=False)

# Analytic
x_t = grad.Tensor(X, requires_grad=True)
y = bn(x_t)
y.sum().backward()
analytic = x_t.grad.data.flatten().tolist()

# Finite differences. We MUST recreate bn each call because the forward
# updates running stats (mutates state). To isolate the test, set
# track_running_stats=False so each call is stateless.
def loss_of(X_):
    bn_local = nn.BatchNorm2d(2, affine=False, track_running_stats=False)
    return float(bn_local(grad.Tensor(X_)).sum().item())

eps = 1e-3
fd = np.zeros_like(X)
for n in range(X.shape[0]):
    for c in range(X.shape[1]):
        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                Xp = X.copy(); Xp[n, c, i, j] += eps
                Xm = X.copy(); Xm[n, c, i, j] -= eps
                fd[n, c, i, j] = (loss_of(Xp) - loss_of(Xm)) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    expect(result.analytic.length).toBe(result.finite_diff.length);
    for (let i = 0; i < result.analytic.length; i++) {
      // BN's stable backward + f32 + eps=1e-3 → tighter tol than pool, looser than naive elementwise.
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });

  it("d(sum(y))/d(beta) = grad_out summed over (N, H, W)", async () => {
    // For loss = y.sum(), grad_out is all-ones; d(beta)[c] = N * H * W.
    const result = await target.run<number[]>(`
${PRELUDE}
np.random.seed(7)
X = np.random.randn(2, 3, 4, 4).astype(np.float32)
bn = nn.BatchNorm2d(3)
y = bn(grad.Tensor(X))
y.sum().backward()
bn.bias.grad.tolist()
`);
    // 2 * 4 * 4 = 32 elements per channel summed.
    expect(result).toEqual([32, 32, 32]);
  });

  it("d(sum(y))/d(gamma) matches finite differences", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(11)
X = np.random.randn(3, 2, 4, 4).astype(np.float32)
bn = nn.BatchNorm2d(2)
bn.weight.data[:] = np.array([0.7, 1.3], dtype=np.float32)
bn.bias.data[:] = np.array([0.1, -0.2], dtype=np.float32)

x_t = grad.Tensor(X)
y = bn(x_t)
y.sum().backward()
analytic = bn.weight.grad.tolist()

def loss_at_weight(w):
    bn_local = nn.BatchNorm2d(2, track_running_stats=False)
    bn_local.weight.data[:] = w
    bn_local.bias.data[:] = bn.bias.data
    return float(bn_local(grad.Tensor(X)).sum().item())

eps = 1e-3
fd = np.zeros(2, dtype=np.float32)
w0 = bn.weight.data.copy()
for c in range(2):
    w_plus = w0.copy(); w_plus[c] += eps
    w_minus = w0.copy(); w_minus[c] -= eps
    fd[c] = (loss_at_weight(w_plus) - loss_at_weight(w_minus)) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("end-to-end: CNN with BatchNorm trains", () => {
  beforeAll(reset);

  it("Conv → BN → ReLU → Pool → Linear classifier converges", async () => {
    // Same top-vs-bottom synthetic task as the pool test, but with BatchNorm
    // after the conv layer. If the BN backward is wrong, this stalls.
    const result = await target.run<{
      initial_acc: number;
      final_acc: number;
    }>(`
${PRELUDE}
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
np.random.seed(13)

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

class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4 * 3 * 3, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn(self.conv(x)))
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        return self.fc(h)

model = CNNWithBN()
opt = optim.Adam(model.parameters(), lr=0.05)

def accuracy(model, X, y):
    model.eval()
    logits = model(grad.Tensor(X))
    model.train()
    return float((np.argmax(logits.data, axis=1) == y).mean())

initial_acc = accuracy(model, X, y_labels)
for _ in range(60):
    opt.zero_grad()
    loss = F.cross_entropy_loss(model(grad.Tensor(X)), y_labels)
    loss.backward()
    opt.step()

{"initial_acc": initial_acc, "final_acc": accuracy(model, X, y_labels)}
`);
    // Random init ~50%; correct full pipeline (incl. BN forward+backward
    // AND the eval-mode-during-accuracy-check path) reaches >95%.
    expect(result.final_acc).toBeGreaterThan(0.95);
  });
});
