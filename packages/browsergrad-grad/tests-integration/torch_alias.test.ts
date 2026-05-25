/**
 * `install_torch_alias()` registers a `torch`-namespace shim so PyTorch
 * user code (`import torch; import torch.nn as nn`) runs against
 * browsergrad_grad without modification.
 *
 * Tests cover both the namespace plumbing and a deep-ml-style end-to-end
 * training loop — the latter is the goldilocks check: if a copy-pasted
 * PyTorch snippet trains, the shim is doing its job.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { clearNamespace, getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

async function reset(): Promise<void> {
  await clearNamespace(target);
  // Also un-register the torch shim between tests for hygiene.
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
`;

describe("install_torch_alias — namespace plumbing", () => {
  beforeAll(reset);

  it("makes `import torch` resolve to the shim module", async () => {
    const result = await target.run<{
      has_tensor: boolean;
      has_nn: boolean;
      has_optim: boolean;
      has_no_grad: boolean;
    }>(`
${PRELUDE}
{
  "has_tensor": hasattr(torch, "tensor"),
  "has_nn": hasattr(torch, "nn"),
  "has_optim": hasattr(torch, "optim"),
  "has_no_grad": hasattr(torch, "no_grad"),
}
`);
    expect(result.has_tensor).toBe(true);
    expect(result.has_nn).toBe(true);
    expect(result.has_optim).toBe(true);
    expect(result.has_no_grad).toBe(true);
  });

  it("makes `import torch.nn as nn` work", async () => {
    const result = await target.run<string>(`
${PRELUDE}
import torch.nn as nn
type(nn.Linear(2, 2)).__name__
`);
    expect(result).toBe("Linear");
  });

  it("makes `import torch.nn.functional as F` work with PyTorch-style names", async () => {
    const result = await target.run<{
      relu_works: boolean;
      cross_entropy_aliased: boolean;
    }>(`
${PRELUDE}
import torch.nn.functional as F

# relu maps to functional.relu
x = torch.tensor([-1.0, 2.0, -3.0])
y = F.relu(x)
relu_works = (y.data == [0.0, 2.0, 0.0]).all()

# cross_entropy should be aliased (we name it cross_entropy_loss internally)
ce_aliased = callable(getattr(F, "cross_entropy", None))

{"relu_works": bool(relu_works), "cross_entropy_aliased": ce_aliased}
`);
    expect(result.relu_works).toBe(true);
    expect(result.cross_entropy_aliased).toBe(true);
  });

  it("makes `import torch.optim as optim` work", async () => {
    const result = await target.run<string>(`
${PRELUDE}
import torch.optim as optim
type(optim.Adam([torch.zeros(2, requires_grad=True)], lr=0.01)).__name__
`);
    expect(result).toBe("Adam");
  });

  it("torch.tensor(...) creates a Tensor with same API as grad.Tensor", async () => {
    const result = await target.run<{
      shape: number[];
      requires_grad: boolean;
      same_class: boolean;
    }>(`
${PRELUDE}
t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# torch.Tensor should also be the same type as the constructor
{
  "shape": list(t.shape),
  "requires_grad": bool(t.requires_grad),
  "same_class": isinstance(t, torch.Tensor),
}
`);
    expect(result.shape).toEqual([3]);
    expect(result.requires_grad).toBe(true);
    expect(result.same_class).toBe(true);
  });

  it("torch.no_grad disables autograd graph building", async () => {
    const result = await target.run<{
      inside: boolean;
      outside: boolean;
    }>(`
${PRELUDE}
x = torch.tensor([1.0], requires_grad=True)
with torch.no_grad():
    inside = (x * x).requires_grad
outside = (x * x).requires_grad
{"inside": bool(inside), "outside": bool(outside)}
`);
    expect(result.inside).toBe(false);
    expect(result.outside).toBe(true);
  });

  it("torch.zeros / ones / randn match the browsergrad_grad helpers", async () => {
    const result = await target.run<{
      zeros_sum: number;
      ones_sum: number;
      randn_shape: number[];
    }>(`
${PRELUDE}
z = torch.zeros(3, 4)
o = torch.ones(2, 5)
r = torch.randn(4, 7, seed=0)
{
  "zeros_sum": float(z.data.sum()),
  "ones_sum": float(o.data.sum()),
  "randn_shape": list(r.shape),
}
`);
    expect(result.zeros_sum).toBe(0);
    expect(result.ones_sum).toBe(10);
    expect(result.randn_shape).toEqual([4, 7]);
  });
});

describe("install_torch_alias — end-to-end PyTorch-style code runs", () => {
  beforeAll(reset);

  it("deep-ml-style classifier training loop trains to >95% accuracy", async () => {
    // This is the goldilocks test: idiomatic PyTorch code written by
    // copy-paste from a deep-ml problem. If it converges, the shim is
    // working end-to-end across imports, model construction, training,
    // and inference (with no_grad).
    const result = await target.run<{
      initial_acc: number;
      final_acc: number;
    }>(`
${PRELUDE}
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)

# Linearly separable 2D toy problem.
X = np.concatenate([
    np.random.randn(64, 4).astype(np.float32) + 1.0,
    np.random.randn(64, 4).astype(np.float32) - 1.0,
], axis=0)
y_labels = np.array([0] * 64 + [1] * 64, dtype=np.int64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.05)

def accuracy():
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        preds = np.argmax(logits.data, axis=1)
    model.train()
    return float((preds == y_labels).mean())

initial = accuracy()
for _ in range(80):
    optimizer.zero_grad()
    logits = model(torch.tensor(X))
    loss = F.cross_entropy(logits, y_labels)
    loss.backward()
    optimizer.step()
final = accuracy()

{"initial_acc": initial, "final_acc": final}
`);
    expect(result.initial_acc).toBeLessThan(0.65);
    expect(result.final_acc).toBeGreaterThan(0.95);
  });
});
