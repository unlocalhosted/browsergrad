/**
 * TDD'd `grad.cat([...], dim)` and `grad.stack([...], dim)`.
 *
 * PyTorch behavior:
 *   cat:   concatenate along an *existing* axis. All inputs must agree on
 *          every dim except `dim`.
 *   stack: concatenate along a *new* axis inserted at `dim`. All inputs
 *          must have identical shape.
 *
 * Backward: split the gradient back along the join axis and distribute
 * to each parent.
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
import numpy as np
`;

describe("grad.cat", () => {
  beforeAll(reset);

  it("concatenates two 2D tensors along axis 0", async () => {
    const result = await target.run<{
      shape: number[];
      values: number[][];
    }>(`
${PRELUDE}
a = grad.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = grad.Tensor([[5.0, 6.0]])
c = grad.cat([a, b], dim=0)
{"shape": list(c.shape), "values": c.tolist()}
`);
    expect(result.shape).toEqual([3, 2]);
    expect(result.values).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });

  it("concatenates along axis 1", async () => {
    const result = await target.run<number[][]>(`
${PRELUDE}
a = grad.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = grad.Tensor([[10.0], [20.0]])
grad.cat([a, b], dim=1).tolist()
`);
    expect(result).toEqual([
      [1, 2, 10],
      [3, 4, 20],
    ]);
  });

  it("backward splits gradient along the cat axis to each parent", async () => {
    const result = await target.run<{
      a_grad: number[][];
      b_grad: number[][];
    }>(`
${PRELUDE}
a = grad.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = grad.Tensor([[5.0, 6.0]], requires_grad=True)
c = grad.cat([a, b], dim=0)
c.sum().backward()
{"a_grad": a.grad.tolist(), "b_grad": b.grad.tolist()}
`);
    // For loss = sum, gradient is all ones with same shape as output.
    // Split back: a gets the (2, 2) ones; b gets the (1, 2) ones.
    expect(result.a_grad).toEqual([
      [1, 1],
      [1, 1],
    ]);
    expect(result.b_grad).toEqual([[1, 1]]);
  });
});

describe("grad.stack", () => {
  beforeAll(reset);

  it("stacks tensors along a new leading axis (dim=0)", async () => {
    const result = await target.run<{
      shape: number[];
      values: number[][][];
    }>(`
${PRELUDE}
a = grad.Tensor([1.0, 2.0])
b = grad.Tensor([3.0, 4.0])
c = grad.stack([a, b], dim=0)
{"shape": list(c.shape), "values": c.tolist()}
`);
    expect(result.shape).toEqual([2, 2]);
    expect(result.values).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("stacks along an inner axis (dim=1)", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
a = grad.Tensor([1.0, 2.0])
b = grad.Tensor([3.0, 4.0])
c = grad.stack([a, b], dim=1)  # (2, 2): rows = original elements, cols = which input
list(c.shape)
`);
    expect(result).toEqual([2, 2]);
  });

  it("stack backward routes gradient to each parent independently", async () => {
    const result = await target.run<{
      a_grad: number[];
      b_grad: number[];
    }>(`
${PRELUDE}
a = grad.Tensor([1.0, 2.0], requires_grad=True)
b = grad.Tensor([3.0, 4.0], requires_grad=True)
c = grad.stack([a, b], dim=0)
c.sum().backward()
{"a_grad": a.grad.tolist(), "b_grad": b.grad.tolist()}
`);
    expect(result.a_grad).toEqual([1, 1]);
    expect(result.b_grad).toEqual([1, 1]);
  });
});

describe("compositional check", () => {
  beforeAll(reset);

  it("Linear(cat([h1, h2])) trains end-to-end (gradient flows through cat)", async () => {
    const result = await target.run<{ initial_loss: number; final_loss: number }>(`
${PRELUDE}
import browsergrad_grad.nn as nn
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
np.random.seed(0)

# Two linear branches whose outputs are concatenated and then projected.
# If cat backward isn't routing properly, the loss won't drop.
class TwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Linear(4, 4)
        self.b2 = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        h = grad.cat([self.b1(x), self.b2(x)], dim=-1)
        return self.head(h)

X = np.random.randn(32, 4).astype(np.float32)
y_labels = (X[:, 0] > 0).astype(np.int64)

model = TwoBranch()
opt = optim.Adam(model.parameters(), lr=0.05)

def loss_now():
    return float(F.cross_entropy_loss(model(grad.Tensor(X)), y_labels).item())

initial = loss_now()
for _ in range(80):
    opt.zero_grad()
    loss = F.cross_entropy_loss(model(grad.Tensor(X)), y_labels)
    loss.backward()
    opt.step()
final = float(loss.item())

{"initial_loss": initial, "final_loss": final}
`);
    expect(result.final_loss).toBeLessThan(result.initial_loss);
    expect(result.final_loss).toBeLessThan(0.3);
  });
});
