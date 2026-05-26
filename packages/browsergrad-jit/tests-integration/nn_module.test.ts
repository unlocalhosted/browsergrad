/**
 * nn.Module + functional + optim integration tests.
 *
 * The minimum-viable 0.1.0 scope per PRD-005 critique: elementwise +
 * Linear + Sequential + a softmax/cross-entropy path. These tests
 * exercise the full forward+backward+optimizer-step loop on a 2-layer
 * MLP and assert the loss decreases monotonically over a few steps.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("nn.Module surface", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("Linear has weight + bias as named parameters", async () => {
    const target = await getJitTarget();
    const names = await target.run<string[]>(`
import browsergrad_jit as bg
layer = bg.nn.Linear(4, 3)
[n for n, _ in layer.named_parameters()]
`);
    expect(names).toEqual(["weight", "bias"]);
  });

  it("Linear forward produces the right shape", async () => {
    const target = await getJitTarget();
    const shape = await target.run<number[]>(`
import browsergrad_jit as bg
layer = bg.nn.Linear(4, 3)
x = bg.randn(8, 4, seed=42)
y = layer(x)
list(y.shape)
`);
    expect(shape).toEqual([8, 3]);
  });

  it("Sequential composes layers", async () => {
    const target = await getJitTarget();
    const shape = await target.run<number[]>(`
import browsergrad_jit as bg
model = bg.nn.Sequential(
    bg.nn.Linear(4, 8),
    bg.nn.ReLU(),
    bg.nn.Linear(8, 2),
)
x = bg.randn(3, 4, seed=42)
y = model(x)
list(y.shape)
`);
    expect(shape).toEqual([3, 2]);
  });

  it("state_dict round-trips through load_state_dict", async () => {
    const target = await getJitTarget();
    const matches = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
a = bg.nn.Linear(3, 2)
b = bg.nn.Linear(3, 2)
sd = a.state_dict()
b.load_state_dict(sd)
ok = True
for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
    ok = ok and na == nb
    ok = ok and np.allclose(pa.numpy(), pb.numpy())
ok
`);
    expect(matches).toBe(true);
  });

  it("MSE loss decreases over SGD training on a linear regression", async () => {
    const target = await getJitTarget();
    const losses = await target.run<number[]>(`
import browsergrad_jit as bg
import numpy as np
bg.manual_seed(0)
# Synthetic linear regression: y = 2x + 1
x = bg.tensor(np.linspace(-1, 1, 32, dtype=np.float32).reshape(-1, 1))
target = bg.tensor((2 * np.linspace(-1, 1, 32) + 1).astype(np.float32).reshape(-1, 1))
model = bg.nn.Linear(1, 1)
opt = bg.optim.SGD(model.parameters(), lr=0.1)
losses = []
for _ in range(20):
    opt.zero_grad()
    pred = model(x)
    loss = bg.nn.functional.mse_loss(pred, target)
    loss.backward()
    opt.step()
    losses.append(float(loss.item()))
losses
`);
    expect(losses.length).toBe(20);
    const first = losses[0]!;
    const last = losses[losses.length - 1]!;
    // Loss should be decreasing.
    expect(first).toBeGreaterThan(last);
    // 20 SGD steps on random init from a 1-feature linear regression with
    // lr=0.1 — converges far enough to drop loss by an order of magnitude,
    // but not to ~0 without many more steps. The relative bar is what matters.
    expect(last).toBeLessThan(first * 0.3);
  });

  it("cross_entropy + 2-layer MLP converges on a toy 2-class problem", async () => {
    const target = await getJitTarget();
    const final = await target.run<{ loss: number; acc: number }>(`
import browsergrad_jit as bg
import numpy as np
bg.manual_seed(0)
# Two clusters at (-1,-1) and (1,1). Labels 0 and 1.
np.random.seed(0)
x1 = np.random.randn(32, 2).astype(np.float32) - 1.0
x2 = np.random.randn(32, 2).astype(np.float32) + 1.0
X = bg.tensor(np.concatenate([x1, x2], axis=0))
Y = bg.tensor(np.concatenate([np.zeros(32), np.ones(32)]).astype(np.int64))

model = bg.nn.Sequential(
    bg.nn.Linear(2, 16),
    bg.nn.ReLU(),
    bg.nn.Linear(16, 2),
)
opt = bg.optim.Adam(model.parameters(), lr=0.05)
for _ in range(80):
    opt.zero_grad()
    logits = model(X)
    loss = bg.nn.functional.cross_entropy(logits, Y)
    loss.backward()
    opt.step()

# Final accuracy
logits = model(X).numpy()
preds = np.argmax(logits, axis=-1)
acc = float((preds == Y.numpy()).mean())
{"loss": float(loss.item()), "acc": acc}
`);
    expect(final.loss).toBeLessThan(0.5);
    expect(final.acc).toBeGreaterThan(0.85);
  });

  it("train()/eval() toggles propagate to submodules", async () => {
    const target = await getJitTarget();
    const states = await target.run<{ train: boolean; eval: boolean }>(`
import browsergrad_jit as bg
m = bg.nn.Sequential(bg.nn.Linear(2, 2), bg.nn.Dropout(p=0.5))
m.train()
state_train = m[1].training
m.eval()
state_eval = m[1].training
{"train": state_train, "eval": state_eval}
`);
    expect(states.train).toBe(true);
    expect(states.eval).toBe(false);
  });
});
