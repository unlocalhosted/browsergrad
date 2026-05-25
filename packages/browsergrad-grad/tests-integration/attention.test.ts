/**
 * TDD'd nn.MultiHeadAttention.
 *
 * Convention: batch-first inputs of shape (N, S, D), where
 *   N = batch size, S = sequence length, D = embed_dim.
 * embed_dim must be divisible by num_heads.
 *
 * Reference oracle: numpy implementation of scaled dot-product attention.
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

describe("MultiHeadAttention — forward", () => {
  beforeAll(reset);

  it("tracer: num_heads=1, identity projections → output equals softmax(QK^T/sqrt(D)) V", async () => {
    // With Q/K/V/out projections all set to identity and bias=False, the layer
    // collapses to pure scaled-dot-product attention on the raw inputs.
    const result = await target.run<{
      ours: number[][][];
      ref: number[][];
    }>(`
${PRELUDE}
np.random.seed(0)
N, S, D = 1, 4, 8
mha = nn.MultiHeadAttention(D, num_heads=1, bias=False)
# Override all projection weights to identity so the layer becomes
# vanilla scaled-dot-product attention on the inputs themselves.
mha.q_proj.weight.data[:] = np.eye(D, dtype=np.float32)
mha.k_proj.weight.data[:] = np.eye(D, dtype=np.float32)
mha.v_proj.weight.data[:] = np.eye(D, dtype=np.float32)
mha.out_proj.weight.data[:] = np.eye(D, dtype=np.float32)

X = np.random.randn(N, S, D).astype(np.float32)
y = mha(grad.Tensor(X), grad.Tensor(X), grad.Tensor(X))

# Reference: scaled-dot-product attention via numpy
def softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
scores = X[0] @ X[0].T / np.sqrt(D)  # (S, S)
weights = softmax(scores, axis=-1)
ref = weights @ X[0]  # (S, D)

{"ours": y.tolist(), "ref": ref.tolist()}
`);
    // Compare element by element with f32 tolerance.
    const our = result.ours[0]!;
    const ref = result.ref;
    expect(our.length).toBe(ref.length);
    for (let s = 0; s < our.length; s++) {
      for (let d = 0; d < our[s]!.length; d++) {
        expect(Math.abs(our[s]![d]! - ref[s]![d]!)).toBeLessThan(1e-4);
      }
    }
  });

  it("output shape is (N, S, embed_dim) regardless of num_heads", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
mha = nn.MultiHeadAttention(64, num_heads=8)
X = grad.Tensor(np.random.randn(3, 7, 64).astype(np.float32))
list(mha(X, X, X).shape)
`);
    expect(result).toEqual([3, 7, 64]);
  });

  it("multi-head output matches numpy SDP-per-head + concat reference", async () => {
    const result = await target.run<{
      ours: number[];
      ref: number[];
    }>(`
${PRELUDE}
np.random.seed(2)
N, S, D, H = 2, 5, 16, 4
mha = nn.MultiHeadAttention(D, num_heads=H, bias=False)
mha.q_proj.weight.data[:] = np.eye(D, dtype=np.float32)
mha.k_proj.weight.data[:] = np.eye(D, dtype=np.float32)
mha.v_proj.weight.data[:] = np.eye(D, dtype=np.float32)
mha.out_proj.weight.data[:] = np.eye(D, dtype=np.float32)

X = np.random.randn(N, S, D).astype(np.float32)
y = mha(grad.Tensor(X), grad.Tensor(X), grad.Tensor(X))

# Reference: split D into H heads, do SDP per head, concat.
def softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
d_k = D // H
# Reshape X: (N, S, D) → (N, S, H, d_k) → (N, H, S, d_k)
Xh = X.reshape(N, S, H, d_k).transpose(0, 2, 1, 3)
scores = Xh @ Xh.swapaxes(-1, -2) / np.sqrt(d_k)
weights = softmax(scores, axis=-1)
attn = weights @ Xh   # (N, H, S, d_k)
# Concat heads: (N, H, S, d_k) → (N, S, H, d_k) → (N, S, D)
ref = attn.transpose(0, 2, 1, 3).reshape(N, S, D)

{"ours": np.asarray(y.data).flatten().tolist(),
 "ref": ref.flatten().tolist()}
`);
    expect(result.ours.length).toBe(result.ref.length);
    for (let i = 0; i < result.ours.length; i++) {
      expect(Math.abs(result.ours[i]! - result.ref[i]!)).toBeLessThan(1e-4);
    }
  });
});

describe("MultiHeadAttention — backward", () => {
  beforeAll(reset);

  it("d(sum(y))/d(Q) matches finite differences", async () => {
    const result = await target.run<{
      analytic: number[];
      finite_diff: number[];
    }>(`
${PRELUDE}
np.random.seed(5)
N, S, D, H = 1, 4, 8, 2
mha = nn.MultiHeadAttention(D, num_heads=H, bias=False)
np.random.seed(5)  # for reproducible initialization

Q = np.random.randn(N, S, D).astype(np.float32) * 0.5
K = np.random.randn(N, S, D).astype(np.float32) * 0.5
V = np.random.randn(N, S, D).astype(np.float32) * 0.5

q_t = grad.Tensor(Q, requires_grad=True)
y = mha(q_t, grad.Tensor(K), grad.Tensor(V))
y.sum().backward()
analytic = q_t.grad.data.flatten().tolist()

# Finite differences w.r.t. Q
def loss_at(Q_):
    return float(mha(grad.Tensor(Q_), grad.Tensor(K), grad.Tensor(V)).sum().item())

eps = 1e-3
fd = np.zeros_like(Q)
for n in range(N):
    for s in range(S):
        for d in range(D):
            Qp = Q.copy(); Qp[n, s, d] += eps
            Qm = Q.copy(); Qm[n, s, d] -= eps
            fd[n, s, d] = (loss_at(Qp) - loss_at(Qm)) / (2 * eps)

{"analytic": analytic, "finite_diff": fd.flatten().tolist()}
`);
    for (let i = 0; i < result.analytic.length; i++) {
      expect(Math.abs(result.analytic[i]! - result.finite_diff[i]!)).toBeLessThan(5e-3);
    }
  });
});

describe("Transformer block — end-to-end", () => {
  beforeAll(reset);

  it("self-attention + residual + LayerNorm + FFN trains on a copy task", async () => {
    // Copy task: predict the sequence's own one-hot — trivial enough that a
    // tiny transformer block with attention should reach near-zero loss.
    // The strict requirement is that loss DROPS materially with training.
    const result = await target.run<{
      initial_loss: number;
      final_loss: number;
    }>(`
${PRELUDE}
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
np.random.seed(17)

vocab = 8
seq_len = 4
d_model = 16

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.attn = nn.MultiHeadAttention(d_model, num_heads=2)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
    def forward(self, tokens):
        x = self.emb(tokens)
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.ff(x))
        return self.head(x)

model = TransformerBlock()
opt = optim.Adam(model.parameters(), lr=0.05)

# Build a batch of 32 random token sequences, target = same sequence.
batch = np.random.randint(0, vocab, size=(32, seq_len)).astype(np.int64)
def step():
    logits = model(batch)  # (32, seq_len, vocab)
    flat_logits = logits.reshape(32 * seq_len, vocab)
    flat_targets = batch.reshape(-1)
    return F.cross_entropy_loss(flat_logits, flat_targets)

initial = float(step().item())
for _ in range(200):
    opt.zero_grad()
    loss = step()
    loss.backward()
    opt.step()
final = float(loss.item())

{"initial_loss": initial, "final_loss": final}
`);
    // The block has enough capacity to memorize a 32-sample copy task.
    expect(result.final_loss).toBeLessThan(result.initial_loss);
    expect(result.final_loss).toBeLessThan(0.5);
  });
});
