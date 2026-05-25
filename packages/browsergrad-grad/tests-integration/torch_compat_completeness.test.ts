/**
 * Audit: every PyTorch idiom a deep-ml-style problem typically uses.
 *
 * Each test exercises a single API surface area. The end-to-end test
 * combines many features into a full PyTorch lab (data prep, model with
 * loss-module, training loop, inference) — the goldilocks check.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { clearNamespace, getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

async function reset(): Promise<void> {
  await clearNamespace(target);
  await target.run(`
import sys as _sys
for _mod in ["torch", "torch.nn", "torch.nn.functional", "torch.optim"]:
    _sys.modules.pop(_mod, None)
`);
}

const PRELUDE = `
import browsergrad_grad as grad
import numpy as np
grad.install_torch_alias()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
`;

describe("torch shape ops", () => {
  beforeAll(reset);

  it("unsqueeze adds a size-1 dim at the given position", async () => {
    const result = await target.run<{ shapes: number[][] }>(`
${PRELUDE}
x = torch.tensor([1.0, 2.0, 3.0])  # shape (3,)
{
  "shapes": [
    list(x.unsqueeze(0).shape),  # (1, 3)
    list(x.unsqueeze(1).shape),  # (3, 1)
    list(x.unsqueeze(-1).shape), # (3, 1)
  ]
}
`);
    expect(result.shapes).toEqual([[1, 3], [3, 1], [3, 1]]);
  });

  it("squeeze removes size-1 dims", async () => {
    const result = await target.run<{ shapes: number[][] }>(`
${PRELUDE}
x = torch.zeros(1, 3, 1, 5)
{
  "shapes": [
    list(x.squeeze().shape),     # all size-1 dims → (3, 5)
    list(x.squeeze(0).shape),    # only dim 0    → (3, 1, 5)
    list(x.squeeze(2).shape),    # only dim 2    → (1, 3, 5)
  ]
}
`);
    expect(result.shapes).toEqual([[3, 5], [3, 1, 5], [1, 3, 5]]);
  });

  it("permute reorders axes (general transpose)", async () => {
    const result = await target.run<{ shape: number[]; checks: number[] }>(`
${PRELUDE}
x = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
y = x.permute(2, 0, 1)  # (4, 2, 3)
# Sanity: y[k, i, j] should equal x[i, j, k]
checks = []
for i in range(2):
    for j in range(3):
        for k in range(4):
            checks.append(int(y.data[k, i, j] == x.data[i, j, k]))
{"shape": list(y.shape), "checks": checks}
`);
    expect(result.shape).toEqual([4, 2, 3]);
    expect(result.checks.every((c) => c === 1)).toBe(true);
  });

  it("size() returns shape as a tuple-like; size(dim) returns an int", async () => {
    const result = await target.run<{ all_shape: number[]; dim0: number; dim_neg1: number }>(`
${PRELUDE}
x = torch.zeros(3, 4, 5)
{
  "all_shape": list(x.size()),
  "dim0":      int(x.size(0)),
  "dim_neg1":  int(x.size(-1)),
}
`);
    expect(result.all_shape).toEqual([3, 4, 5]);
    expect(result.dim0).toBe(3);
    expect(result.dim_neg1).toBe(5);
  });
});

describe("device manipulation (no-op stubs for browsergrad)", () => {
  beforeAll(reset);

  it("tensor.to(device) and .cpu() / .cuda() return the same tensor (no-op)", async () => {
    // PyTorch user code calls these unconditionally. Since browsergrad has one
    // notional device, they should be no-ops that return self for chaining.
    const result = await target.run<{
      identity_to: boolean;
      identity_cpu: boolean;
      identity_cuda: boolean;
      chains_with_method: boolean;
    }>(`
${PRELUDE}
x = torch.tensor([1.0, 2.0])
# Same data after each call?
identity_to   = (x.to("cpu").data == x.data).all()
identity_cpu  = (x.cpu().data == x.data).all()
identity_cuda = (x.cuda().data == x.data).all()
# Should also chain — e.g. tensor(...).to(device).requires_grad_()
y = torch.tensor([1.0, 2.0]).to("cuda")
chains = isinstance(y, torch.Tensor)
{"identity_to": bool(identity_to), "identity_cpu": bool(identity_cpu),
 "identity_cuda": bool(identity_cuda), "chains_with_method": bool(chains)}
`);
    expect(result.identity_to).toBe(true);
    expect(result.identity_cpu).toBe(true);
    expect(result.identity_cuda).toBe(true);
    expect(result.chains_with_method).toBe(true);
  });
});

describe("numpy interop + reproducibility", () => {
  beforeAll(reset);

  it("torch.from_numpy converts a numpy array to a Tensor", async () => {
    const result = await target.run<{ shape: number[]; values: number[] }>(`
${PRELUDE}
arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
t = torch.from_numpy(arr)
{"shape": list(t.shape), "values": t.tolist()}
`);
    expect(result.shape).toEqual([4]);
    expect(result.values).toEqual([1, 2, 3, 4]);
  });

  it("tensor.numpy() returns a numpy array", async () => {
    const result = await target.run<{ is_ndarray: boolean; shape: number[] }>(`
${PRELUDE}
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
arr = t.numpy()
{"is_ndarray": isinstance(arr, np.ndarray), "shape": list(arr.shape)}
`);
    expect(result.is_ndarray).toBe(true);
    expect(result.shape).toEqual([2, 2]);
  });

  it("torch.manual_seed makes randn reproducible", async () => {
    const result = await target.run<{ run1: number[]; run2: number[] }>(`
${PRELUDE}
torch.manual_seed(42)
r1 = torch.randn(5).tolist()
torch.manual_seed(42)
r2 = torch.randn(5).tolist()
{"run1": r1, "run2": r2}
`);
    expect(result.run1).toEqual(result.run2);
  });
});

describe("top-level math functions on torch namespace", () => {
  beforeAll(reset);

  it("torch.matmul / mm / bmm", async () => {
    const result = await target.run<{
      matmul: number[][];
      mm: number[][];
      bmm_shape: number[];
    }>(`
${PRELUDE}
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
batched_A = torch.tensor(np.ones((4, 2, 3), dtype=np.float32))
batched_B = torch.tensor(np.ones((4, 3, 5), dtype=np.float32))
{
  "matmul": torch.matmul(A, B).tolist(),
  "mm":     torch.mm(A, B).tolist(),
  "bmm_shape": list(torch.bmm(batched_A, batched_B).shape),
}
`);
    expect(result.matmul).toEqual([
      [19, 22],
      [43, 50],
    ]);
    expect(result.mm).toEqual([
      [19, 22],
      [43, 50],
    ]);
    expect(result.bmm_shape).toEqual([4, 2, 5]);
  });

  it("torch.exp / log / sum / mean / argmax as top-level functions", async () => {
    const result = await target.run<{
      exp_first: number;
      log_first: number;
      sum: number;
      mean: number;
      argmax: number;
    }>(`
${PRELUDE}
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
{
  "exp_first": float(torch.exp(x).data[0]),
  "log_first": float(torch.log(x).data[0]),
  "sum":       float(torch.sum(x).item()),
  "mean":      float(torch.mean(x).item()),
  "argmax":    int(torch.argmax(x)),
}
`);
    expect(Math.abs(result.exp_first - Math.E)).toBeLessThan(1e-5);
    expect(result.log_first).toBe(0);  // log(1) = 0
    expect(result.sum).toBe(10);
    expect(result.mean).toBe(2.5);
    expect(result.argmax).toBe(3);  // value 4 at index 3
  });
});

describe("nn loss modules (callable like nn.Linear)", () => {
  beforeAll(reset);

  it("nn.CrossEntropyLoss is a Module-style callable", async () => {
    const result = await target.run<{ loss: number }>(`
${PRELUDE}
logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
targets = np.array([1, 2], dtype=np.int64)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
{"loss": float(loss.item())}
`);
    expect(result.loss).toBeGreaterThan(0);
    expect(result.loss).toBeLessThan(1.0);  // small loss because predictions roughly correct
  });

  it("nn.MSELoss is a Module-style callable", async () => {
    const result = await target.run<{ loss: number }>(`
${PRELUDE}
y_hat = torch.tensor([1.0, 2.0, 3.0])
y     = torch.tensor([1.0, 2.5, 2.5])
criterion = nn.MSELoss()
loss = criterion(y_hat, y)
# Expected: ((0)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.5 / 3 ≈ 0.1666...
{"loss": float(loss.item())}
`);
    expect(Math.abs(result.loss - 0.5 / 3)).toBeLessThan(1e-5);
  });

  it("nn.BCEWithLogitsLoss handles raw logits", async () => {
    const result = await target.run<{ loss: number }>(`
${PRELUDE}
logits = torch.tensor([0.0, 100.0, -100.0])
targets = torch.tensor([0.0, 1.0, 0.0])
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)
# Expected: confident-correct logits → ~0; mid-confidence (0) → log(2).
# Total: (log(2) + ~0 + ~0) / 3 ≈ 0.231
{"loss": float(loss.item())}
`);
    expect(result.loss).toBeGreaterThan(0.2);
    expect(result.loss).toBeLessThan(0.3);
  });

  it("nn.MultiheadAttention (PyTorch lowercase) is the same as our MultiHeadAttention", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
mha = nn.MultiheadAttention(16, num_heads=2)
X = torch.tensor(np.random.randn(2, 4, 16).astype(np.float32))
out = mha(X, X, X)
list(out.shape)
`);
    expect(result).toEqual([2, 4, 16]);
  });
});

describe("F.one_hot + F.dropout functional", () => {
  beforeAll(reset);

  it("F.one_hot encodes class indices", async () => {
    const result = await target.run<{ shape: number[]; values: number[][] }>(`
${PRELUDE}
idx = np.array([0, 2, 1], dtype=np.int64)
out = F.one_hot(idx, num_classes=3)
{"shape": list(out.shape), "values": out.tolist()}
`);
    expect(result.shape).toEqual([3, 3]);
    expect(result.values).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]);
  });

  it("F.dropout(training=False) is identity", async () => {
    const result = await target.run<number>(`
${PRELUDE}
x = torch.tensor([1.0, 2.0, 3.0])
y = F.dropout(x, p=0.5, training=False)
float((y.data == x.data).all())
`);
    expect(result).toBe(1);
  });
});

describe("end-to-end realistic PyTorch lab", () => {
  beforeAll(reset);

  it("complete classifier lab — data prep, BCE loss, train/eval mode", async () => {
    // Idiomatic PyTorch problem: synthetic data, model with Loss Module,
    // training loop, accuracy check via no_grad + argmax.
    const result = await target.run<{ initial: number; final: number }>(`
${PRELUDE}
torch.manual_seed(0)

# Data
X_np = np.random.randn(128, 6).astype(np.float32)
y_np = (X_np[:, 0] + X_np[:, 1] > 0).astype(np.int64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

def accuracy():
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.from_numpy(X_np)), dim=-1)
    model.train()
    return float((preds.numpy() == y_np).mean())

initial = accuracy()
for _ in range(60):
    optimizer.zero_grad()
    logits = model(torch.from_numpy(X_np))
    loss = criterion(logits, y_np)
    loss.backward()
    optimizer.step()
final = accuracy()

{"initial": initial, "final": final}
`);
    expect(result.initial).toBeLessThan(0.7);
    expect(result.final).toBeGreaterThan(0.95);
  });
});
