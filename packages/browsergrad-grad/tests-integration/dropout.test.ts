/**
 * TDD'd nn.Dropout. PyTorch-conformant inverted dropout:
 *   train: keep element with prob (1-p), scale kept by 1/(1-p) so E[y]=x
 *   eval:  identity
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

describe("Dropout — forward", () => {
  beforeAll(reset);

  it("tracer: p=0 returns the input unchanged in train mode", async () => {
    const result = await target.run<{
      diff_max: number;
    }>(`
${PRELUDE}
np.random.seed(0)
X = np.random.randn(8, 8).astype(np.float32)
d = nn.Dropout(0.0)
y = d(grad.Tensor(X))
{"diff_max": float(np.abs(y.data - X).max())}
`);
    expect(result.diff_max).toBe(0);
  });

  it("eval mode is identity even with p > 0", async () => {
    const result = await target.run<number>(`
${PRELUDE}
np.random.seed(1)
X = np.random.randn(8, 8).astype(np.float32)
d = nn.Dropout(0.5)
d.eval()
y = d(grad.Tensor(X))
float(np.abs(y.data - X).max())
`);
    expect(result).toBe(0);
  });

  it("train mode with p=0.5 zeros roughly half of a large input (stat check)", async () => {
    // Statistical check — drop rate should be ~p with low variance for a
    // large enough tensor. Tight enough to catch a constant-p bug; loose
    // enough not to be flaky.
    const result = await target.run<{
      zero_frac: number;
      mean_kept_close_to_one: boolean;
    }>(`
${PRELUDE}
np.random.seed(2)
N = 10000
X = np.ones((N,), dtype=np.float32)
d = nn.Dropout(0.5)
y = d(grad.Tensor(X))
zero_frac = float((y.data == 0).sum() / N)
# Each kept element should equal 1.0 / (1-0.5) = 2.0
nonzero = y.data[y.data != 0]
mean_kept = float(nonzero.mean()) if nonzero.size > 0 else 0.0
{"zero_frac": zero_frac, "mean_kept_close_to_one": abs(mean_kept - 2.0) < 1e-5}
`);
    expect(result.zero_frac).toBeGreaterThan(0.45);
    expect(result.zero_frac).toBeLessThan(0.55);
    expect(result.mean_kept_close_to_one).toBe(true);
  });

  it("inverted scaling preserves expected value: E[y] ≈ x", async () => {
    // Run many trials, average. Mean of dropout output should converge to input.
    const result = await target.run<number>(`
${PRELUDE}
np.random.seed(3)
X = np.ones((1000,), dtype=np.float32) * 7.0
d = nn.Dropout(0.3)
trials = 100
acc = np.zeros_like(X)
for _ in range(trials):
    acc += d(grad.Tensor(X)).data
mean = acc / trials
float(abs(mean.mean() - 7.0))
`);
    // 100 trials of 1000 elements is enough to get within ~0.2 of the true mean.
    expect(result).toBeLessThan(0.3);
  });
});

describe("Dropout — backward", () => {
  beforeAll(reset);

  it("backward routes gradient only through kept elements (scaled)", async () => {
    // For y = dropout(x), d(y_sum)/d(x_i) = mask[i] / (1-p) where mask is 1 if
    // the element was kept, 0 otherwise. We don't know the mask from outside,
    // but we can assert: grad equals exactly y/x where x was kept, and is 0
    // where x was dropped — for x of all-equal values, this is observable.
    const result = await target.run<{
      grad_max: number;
      grad_min_nonzero: number;
      zeros_match: boolean;
    }>(`
${PRELUDE}
np.random.seed(5)
X = np.ones((1000,), dtype=np.float32)
d = nn.Dropout(0.3)
x_t = grad.Tensor(X, requires_grad=True)
y = d(x_t)
y.sum().backward()
g = x_t.grad.data
y_d = y.data
# Where y is zero, grad must also be zero.
zeros_match = bool(((y_d == 0) == (g == 0)).all())
# Where kept, grad should be exactly 1 / (1-p) = 1/0.7
nonzero_g = g[g != 0]
{
  "grad_max": float(g.max()),
  "grad_min_nonzero": float(nonzero_g.min()) if nonzero_g.size > 0 else 0.0,
  "zeros_match": zeros_match,
}
`);
    expect(result.zeros_match).toBe(true);
    // All nonzero gradients should be exactly 1 / 0.7 ≈ 1.428571
    expect(Math.abs(result.grad_max - 1 / 0.7)).toBeLessThan(1e-5);
    expect(Math.abs(result.grad_min_nonzero - 1 / 0.7)).toBeLessThan(1e-5);
  });
});

describe("Dropout — end-to-end", () => {
  beforeAll(reset);

  it("MLP with Dropout still trains and accuracy holds up after eval()", async () => {
    const result = await target.run<{
      final_acc_train_mode: number;
      final_acc_eval_mode: number;
    }>(`
${PRELUDE}
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
np.random.seed(13)

# Linearly separable 2-class problem
X = np.concatenate([
    np.random.randn(64, 4).astype(np.float32) + 1.0,
    np.random.randn(64, 4).astype(np.float32) - 1.0,
], axis=0)
y_labels = np.array([0] * 64 + [1] * 64, dtype=np.int64)

class MLPWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(x))))

model = MLPWithDropout()
opt = optim.Adam(model.parameters(), lr=0.05)

for _ in range(100):
    opt.zero_grad()
    logits = model(grad.Tensor(X))
    loss = F.cross_entropy_loss(logits, y_labels)
    loss.backward()
    opt.step()

# Evaluate in BOTH modes — both should be high accuracy.
def acc(X, y):
    return float((np.argmax(model(grad.Tensor(X)).data, axis=1) == y).mean())

acc_train = acc(X, y_labels)
model.eval()
acc_eval = acc(X, y_labels)

{"final_acc_train_mode": acc_train, "final_acc_eval_mode": acc_eval}
`);
    // Dropout shouldn't prevent the model from learning a linearly separable problem.
    expect(result.final_acc_eval_mode).toBeGreaterThan(0.9);
    // Train-mode accuracy is noisier (one stochastic forward) but still high.
    expect(result.final_acc_train_mode).toBeGreaterThan(0.8);
  });
});

describe("Dropout2d — channel-wise dropout", () => {
  beforeAll(reset);

  it("eval mode is identity", async () => {
    const result = await target.run<number>(`
${PRELUDE}
X = np.ones((2, 8, 4, 4), dtype=np.float32)
d = nn.Dropout2d(0.5)
d.eval()
float(np.abs(d(grad.Tensor(X)).data - X).max())
`);
    expect(result).toBe(0);
  });

  it("train mode drops whole channels (per-channel pattern)", async () => {
    // Channel-wise: if any cell in a channel is zero, the whole channel
    // is zero. So per-channel std should be 0 for the dropped ones.
    const result = await target.run<{
      n_dropped: number;
      per_channel_unique_counts: number[];
    }>(`
${PRELUDE}
np.random.seed(7)
X = np.random.randn(4, 16, 3, 3).astype(np.float32) + 5.0  # ensure non-zero
d = nn.Dropout2d(0.5)
y = d(grad.Tensor(X))
# Per (n, c), count unique values: dropped channels have all 0s (1 unique),
# kept channels have many unique scaled values.
unique_counts = []
for n in range(4):
    for c in range(16):
        unique_counts.append(int(len(np.unique(y.data[n, c]))))
n_dropped = int(sum(1 for u in unique_counts if u == 1 and y.data[unique_counts.index(u) // 16, unique_counts.index(u) % 16, 0, 0] == 0))
{"n_dropped": n_dropped, "per_channel_unique_counts": unique_counts}
`);
    // 64 channels total; ~50% should be all-zero. Approximate window: 22..42.
    const fullyZeroed = result.per_channel_unique_counts.filter((u) => u === 1).length;
    expect(fullyZeroed).toBeGreaterThan(15);
    expect(fullyZeroed).toBeLessThan(50);
  });

  it("backward routes per-channel just like forward", async () => {
    const result = await target.run<{
      grad_matches_mask: boolean;
    }>(`
${PRELUDE}
np.random.seed(11)
X = np.ones((2, 8, 3, 3), dtype=np.float32) * 3.0
d = nn.Dropout2d(0.5)
x_t = grad.Tensor(X, requires_grad=True)
y = d(x_t)
y.sum().backward()
# For Dropout2d, grad pattern must match the per-channel zero/non-zero pattern.
y_channel_zero = (y.data == 0).all(axis=(2, 3))  # (N, C)
g_channel_zero = (x_t.grad.data == 0).all(axis=(2, 3))
{"grad_matches_mask": bool((y_channel_zero == g_channel_zero).all())}
`);
    expect(result.grad_matches_mask).toBe(true);
  });
});
