/**
 * Real Python execution tests for the autograd core.
 *
 * Every test boots Pyodide once (via the shared host), installs the grad
 * library, then runs Python that exercises specific autograd behavior and
 * returns numeric results we assert on in JS.
 *
 * These are the tests that would catch a wrong gradient formula. The
 * surface tests in tests/ verify the *contract* (right files, right exports);
 * these verify the *math*.
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
import browsergrad_grad.functional as F
import browsergrad_grad.nn as nn
import browsergrad_grad.optim as optim
import numpy as np
`;

function closeTo(actual: number, expected: number, tol = 1e-4): void {
  expect(Math.abs(actual - expected)).toBeLessThan(tol);
}

function arrayClose(actual: readonly number[], expected: readonly number[], tol = 1e-4): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

/* ────────────────────────────────────────────────────────────
 * Tensor basics + simple gradients
 * ──────────────────────────────────────────────────────────── */

describe("Tensor autograd — basics", () => {
  beforeAll(reset);

  it("d(sum(x*x))/dx = 2x", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
x.grad.tolist()
`);
    arrayClose(result, [2, 4, 6]);
  });

  it("d(mean(x*x))/dx = 2x/N", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = (x * x).mean()
y.backward()
x.grad.tolist()
`);
    arrayClose(result, [0.5, 1.0, 1.5, 2.0]); // 2*x / 4
  });

  it("d(x**3)/dx = 3x²", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 3.0).sum()
y.backward()
x.grad.tolist()
`);
    arrayClose(result, [3, 12, 27]);
  });

  it("chain rule: d(exp(x))/dx = exp(x)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([0.0, 1.0, 2.0], requires_grad=True)
y = x.exp().sum()
y.backward()
x.grad.tolist()
`);
    arrayClose(result, [Math.exp(0), Math.exp(1), Math.exp(2)]);
  });

  it("chain rule: d(log(x))/dx = 1/x", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 4.0], requires_grad=True)
y = x.log().sum()
y.backward()
x.grad.tolist()
`);
    arrayClose(result, [1, 0.5, 0.25]);
  });
});

/* ────────────────────────────────────────────────────────────
 * Broadcasting in backward (the v0.2 marquee feature)
 * ──────────────────────────────────────────────────────────── */

describe("Broadcasting backward", () => {
  beforeAll(reset);

  it("d(sum(x + b))/db sums over batch (bias gradient)", async () => {
    // x: (3, 4) — batch of 3 rows, 4 features
    // b: (4,)   — bias per feature
    // y = sum(x + b) → db_i = 3 (each of 3 rows contributes 1 to each feature)
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor(np.ones((3, 4), dtype=np.float32))
b = grad.Tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
y = (x + b).sum()
y.backward()
b.grad.tolist()
`);
    arrayClose(result, [3, 3, 3, 3]);
  });

  it("d(sum(x + b))/dx is 1s broadcast back to x's shape", async () => {
    const result = await target.run<number[][]>(`
${PRELUDE}
x = grad.Tensor(np.zeros((2, 3), dtype=np.float32), requires_grad=True)
b = grad.Tensor([1.0, 2.0, 3.0])
y = (x + b).sum()
y.backward()
x.grad.tolist()
`);
    expect(result).toEqual([
      [1, 1, 1],
      [1, 1, 1],
    ]);
  });

  it("d(sum(x * w))/dw with w shape (1,4) sums over batch", async () => {
    // x: (3, 4), w: (1, 4) — broadcast over batch
    // y = sum(x*w) → dw_j = sum_i(x_ij), keepdim=1 in dim 0
    const result = await target.run<number[][]>(`
${PRELUDE}
x_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
x = grad.Tensor(x_data)
w = grad.Tensor(np.array([[1, 1, 1, 1]], dtype=np.float32), requires_grad=True)
y = (x * w).sum()
y.backward()
w.grad.tolist()
`);
    // sum over rows: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
    expect(result).toEqual([[15, 18, 21, 24]]);
  });
});

/* ────────────────────────────────────────────────────────────
 * Matmul (incl. higher rank)
 * ──────────────────────────────────────────────────────────── */

describe("matmul gradients", () => {
  beforeAll(reset);

  it("2D matmul forward + backward matches the analytic formula", async () => {
    // A: (2,3), B: (3,2), C = A@B (2,2). Take loss = C.sum().
    // dA = grad_out @ B.T = ones(2,2) @ B.T
    // dB = A.T @ grad_out = A.T @ ones(2,2)
    const result = await target.run<{
      A_grad: number[][];
      B_grad: number[][];
    }>(`
${PRELUDE}
A = grad.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True)
B = grad.Tensor(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32), requires_grad=True)
C = A @ B
loss = C.sum()
loss.backward()
{"A_grad": A.grad.tolist(), "B_grad": B.grad.tolist()}
`);
    // ones(2,2) @ B.T = [[1+2, 3+4, 5+6], [1+2, 3+4, 5+6]] = [[3,7,11],[3,7,11]]
    expect(result.A_grad).toEqual([
      [3, 7, 11],
      [3, 7, 11],
    ]);
    // A.T @ ones(2,2) = [[1+4, 1+4], [2+5, 2+5], [3+6, 3+6]] = [[5,5],[7,7],[9,9]]
    expect(result.B_grad).toEqual([
      [5, 5],
      [7, 7],
      [9, 9],
    ]);
  });

  it("batched 3D matmul: (B,M,K) @ (B,K,N) → (B,M,N)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
A = grad.Tensor(np.ones((2, 3, 4), dtype=np.float32), requires_grad=True)
B_ = grad.Tensor(np.ones((2, 4, 5), dtype=np.float32))
C = A @ B_
list(C.shape)
`);
    expect(result).toEqual([2, 3, 5]);
  });
});

/* ────────────────────────────────────────────────────────────
 * Functional ops
 * ──────────────────────────────────────────────────────────── */

describe("functional ops", () => {
  beforeAll(reset);

  it("relu gradient zeros out negatives", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = F.relu(x).sum()
y.backward()
x.grad.tolist()
`);
    expect(result).toEqual([0, 0, 0, 1, 1]);
  });

  it("softmax rows sum to 1 and gradient passes through", async () => {
    const result = await target.run<{
      probs: number[][];
      grad_sum_per_row: number[];
    }>(`
${PRELUDE}
x = grad.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
y = F.softmax(x, dim=-1)
loss = y.sum()
loss.backward()
{
  "probs": y.tolist(),
  # When loss = sum(softmax(x)), dloss/dx_i should be ~0 for each row
  # (softmax sums to 1; derivative is s_i*(g_i - sum_j(s_j*g_j)) where g=1 → s_i*(1-1)=0).
  "grad_sum_per_row": x.grad.data.sum(axis=-1).tolist(),
}
`);
    // Row 0 should sum to 1.
    closeTo(result.probs[0]!.reduce((a, b) => a + b, 0), 1.0);
    closeTo(result.probs[1]!.reduce((a, b) => a + b, 0), 1.0);
    // sum(grad) per row should be exactly 0 (or extremely close).
    closeTo(result.grad_sum_per_row[0]!, 0, 1e-5);
    closeTo(result.grad_sum_per_row[1]!, 0, 1e-5);
  });

  it("cross_entropy_loss: gradient = (softmax - one_hot)/N", async () => {
    const result = await target.run<{
      loss: number;
      grad: number[][];
    }>(`
${PRELUDE}
logits = grad.Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
targets = np.array([2, 0], dtype=np.int64)
loss = F.cross_entropy_loss(logits, targets)
loss.backward()

# Hand-compute expected gradient: (softmax(logits) - one_hot) / N
import numpy as _np
sm = _np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
sm = sm / sm.sum(axis=1, keepdims=True)
one_hot = _np.zeros_like(sm)
one_hot[0, 2] = 1
one_hot[1, 0] = 1
expected = (sm - one_hot) / 2.0  # N=2

# We return both for the JS test to compare.
{"loss": float(loss.item()), "grad": logits.grad.tolist()}
`);
    // Loss should be positive (we'd need to recompute exact value, just sanity).
    expect(result.loss).toBeGreaterThan(0);
    // Gradient row 0: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652], target=2 → -onehot[2]
    //   so grad/N = ([0.0900, 0.2447, -0.3348]) / 2 ≈ [0.0450, 0.1224, -0.1674]
    arrayClose(result.grad[0]!, [0.0450, 0.1224, -0.1674], 1e-3);
    // Row 1: softmax([1,1,1]) = [1/3, 1/3, 1/3], target=0 → ([-2/3, 1/3, 1/3])/2 = [-1/3, 1/6, 1/6]
    arrayClose(result.grad[1]!, [-1 / 3, 1 / 6, 1 / 6], 1e-3);
  });

  it("mse_loss: scalar output, gradient = 2*(yhat-y)/N", async () => {
    const result = await target.run<{
      loss: number;
      grad: number[];
    }>(`
${PRELUDE}
yhat = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = grad.Tensor([0.0, 0.0, 0.0])
loss = F.mse_loss(yhat, y)
loss.backward()
{"loss": float(loss.item()), "grad": yhat.grad.tolist()}
`);
    // MSE = (1+4+9)/3 = 14/3 ≈ 4.6667
    closeTo(result.loss, 14 / 3);
    // grad = 2 * [1,2,3] / 3 = [2/3, 4/3, 2]
    arrayClose(result.grad, [2 / 3, 4 / 3, 2]);
  });
});

/* ────────────────────────────────────────────────────────────
 * nn — Linear, LayerNorm, Embedding
 * ──────────────────────────────────────────────────────────── */

describe("nn modules", () => {
  beforeAll(reset);

  it("Linear forward + backward shapes match", async () => {
    const result = await target.run<{
      out_shape: number[];
      grad_weight_shape: number[];
      grad_bias_shape: number[];
    }>(`
${PRELUDE}
lin = nn.Linear(4, 3)
x = grad.Tensor(np.ones((5, 4), dtype=np.float32))
y = lin(x)
y.sum().backward()
{
  "out_shape": list(y.shape),
  "grad_weight_shape": list(lin.weight.grad.shape),
  "grad_bias_shape": list(lin.bias.grad.shape),
}
`);
    expect(result.out_shape).toEqual([5, 3]);
    expect(result.grad_weight_shape).toEqual([3, 4]);
    expect(result.grad_bias_shape).toEqual([3]);
  });

  it("LayerNorm normalizes per-row to mean≈0, var≈1", async () => {
    const result = await target.run<{
      mean: number[];
      var_: number[];
    }>(`
${PRELUDE}
ln = nn.LayerNorm(4)
x = grad.Tensor(np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32))
y = ln(x)
m = y.data.mean(axis=-1)
v = y.data.var(axis=-1)
{"mean": m.tolist(), "var_": v.tolist()}
`);
    // With default gamma=1, beta=0, normalized rows have mean=0, var≈1.
    closeTo(result.mean[0]!, 0, 1e-5);
    closeTo(result.mean[1]!, 0, 1e-5);
    closeTo(result.var_[0]!, 1, 1e-3);
    closeTo(result.var_[1]!, 1, 1e-3);
  });

  it("Embedding lookup grad uses scatter-add", async () => {
    const result = await target.run<number[][]>(`
${PRELUDE}
emb = nn.Embedding(5, 3)
emb.weight.data[:] = 0.0  # zero out for deterministic test
# Look up [0, 2, 0] — index 0 appears twice; its gradient should accumulate.
y = emb(np.array([0, 2, 0], dtype=np.int64))
y.sum().backward()
emb.weight.grad.tolist()
`);
    // grad[0] should be [2, 2, 2] (two lookups, each contributing 1 to each dim)
    // grad[2] should be [1, 1, 1]
    // grad[1, 3, 4] should be [0, 0, 0]
    expect(result[0]).toEqual([2, 2, 2]);
    expect(result[1]).toEqual([0, 0, 0]);
    expect(result[2]).toEqual([1, 1, 1]);
    expect(result[3]).toEqual([0, 0, 0]);
    expect(result[4]).toEqual([0, 0, 0]);
  });
});

/* ────────────────────────────────────────────────────────────
 * Optimizers — verify they actually train
 * ──────────────────────────────────────────────────────────── */

describe("optim — end-to-end training", () => {
  beforeAll(reset);

  it("SGD on linear regression converges to y = 3x + 1", async () => {
    const result = await target.run<{
      final_w: number;
      final_b: number;
      final_loss: number;
    }>(`
${PRELUDE}
np.random.seed(0)
X = grad.Tensor(np.random.randn(64, 1).astype(np.float32))
y_true_data = X.data * 3.0 + 1.0
y_true = grad.Tensor(y_true_data)
model = nn.Linear(1, 1)
opt = optim.SGD(model.parameters(), lr=0.1)
for _ in range(500):
    opt.zero_grad()
    yhat = model(X)
    loss = F.mse_loss(yhat, y_true)
    loss.backward()
    opt.step()
{
  "final_w": float(model.weight.item()),
  "final_b": float(model.bias.item()),
  "final_loss": float(loss.item()),
}
`);
    closeTo(result.final_w, 3, 0.05);
    closeTo(result.final_b, 1, 0.05);
    expect(result.final_loss).toBeLessThan(0.01);
  });

  it("Adam converges to a low loss on linear regression", async () => {
    // We don't compare Adam vs SGD because Adam's bias correction
    // makes its early-step behavior different — it can be slower than
    // SGD on simple convex problems for the first ~20 steps. We just
    // check Adam ends up close to the answer with enough steps.
    const result = await target.run<{
      final_loss: number;
    }>(`
${PRELUDE}
np.random.seed(0)
X = grad.Tensor(np.random.randn(64, 1).astype(np.float32))
y_true = grad.Tensor(X.data * 3.0 + 1.0)
model = nn.Linear(1, 1)
opt = optim.Adam(model.parameters(), lr=0.05)
for _ in range(300):
    opt.zero_grad()
    yhat = model(X)
    loss = F.mse_loss(yhat, y_true)
    loss.backward()
    opt.step()
{"final_loss": float(loss.item())}
`);
    expect(result.final_loss).toBeLessThan(0.01);
  });

  it("Trains a 2-layer MLP on a noisy regression with cross_entropy-like loss", async () => {
    const result = await target.run<{
      initial_loss: number;
      final_loss: number;
    }>(`
${PRELUDE}
np.random.seed(42)
X = grad.Tensor(np.random.randn(128, 4).astype(np.float32))
# Generate targets via a noisy linear combo
W_true = np.array([[1.0, -1.0, 0.5, 0.0]], dtype=np.float32).T
y_true_np = (X.data @ W_true).flatten() + 0.1 * np.random.randn(128).astype(np.float32)
y_true = grad.Tensor(y_true_np.reshape(-1, 1))

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
opt = optim.Adam(model.parameters(), lr=0.01)

initial_loss = float(F.mse_loss(model(X), y_true).item())
for _ in range(200):
    opt.zero_grad()
    loss = F.mse_loss(model(X), y_true)
    loss.backward()
    opt.step()
{"initial_loss": initial_loss, "final_loss": float(loss.item())}
`);
    // Loss should drop substantially.
    expect(result.final_loss).toBeLessThan(result.initial_loss);
    expect(result.final_loss).toBeLessThan(0.5);
  });
});
