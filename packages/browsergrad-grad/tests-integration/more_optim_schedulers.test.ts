/**
 * Pile A #10 + #11 — more optimizers + schedulers.
 *
 * Optimizers: RMSprop, Adagrad, Adadelta.
 * Schedulers: MultiStepLR, ExponentialLR, ReduceLROnPlateau, OneCycleLR.
 *
 * Independent oracle: each update rule written out longhand in NumPy.
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
import browsergrad_grad.optim as optim
import numpy as np
`;

describe("optim.RMSprop", () => {
  beforeAll(reset);

  it("matches hand-derived RMSprop update over 3 steps", async () => {
    const result = await target.run<{ p: number[]; oracle: number[] }>(`
${PRELUDE}
alpha = 0.99
eps = 1e-8
lr = 0.01

p = grad.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
opt = optim.RMSprop([p], lr=lr, alpha=alpha, eps=eps)

# Oracle state
p_oracle = np.array([1.0, 2.0, 3.0], dtype=np.float64)
v_oracle = np.zeros(3, dtype=np.float64)

grads = [np.array([0.1, -0.2, 0.5], dtype=np.float64),
         np.array([0.0, 0.3, -0.1], dtype=np.float64),
         np.array([-0.4, 0.1, 0.2], dtype=np.float64)]

for g in grads:
    p.grad = grad.Tensor(g.astype(np.float32))
    opt.step()
    v_oracle = alpha * v_oracle + (1 - alpha) * g * g
    p_oracle -= lr * g / (np.sqrt(v_oracle) + eps)

{"p": p.tolist(), "oracle": p_oracle.tolist()}
`);
    for (let i = 0; i < result.p.length; i++) {
      expect(result.p[i]!).toBeCloseTo(result.oracle[i]!, 4);
    }
  });
});

describe("optim.Adagrad", () => {
  beforeAll(reset);

  it("matches hand-derived Adagrad update over 3 steps", async () => {
    const result = await target.run<{ p: number[]; oracle: number[] }>(`
${PRELUDE}
eps = 1e-10
lr = 0.05

p = grad.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
opt = optim.Adagrad([p], lr=lr, eps=eps)

p_oracle = np.array([1.0, 2.0, 3.0], dtype=np.float64)
G = np.zeros(3, dtype=np.float64)

grads = [np.array([0.1, -0.2, 0.5], dtype=np.float64),
         np.array([0.0, 0.3, -0.1], dtype=np.float64),
         np.array([-0.4, 0.1, 0.2], dtype=np.float64)]

for g in grads:
    p.grad = grad.Tensor(g.astype(np.float32))
    opt.step()
    G += g * g
    p_oracle -= lr * g / (np.sqrt(G) + eps)

{"p": p.tolist(), "oracle": p_oracle.tolist()}
`);
    for (let i = 0; i < result.p.length; i++) {
      expect(result.p[i]!).toBeCloseTo(result.oracle[i]!, 4);
    }
  });
});

describe("optim.Adadelta", () => {
  beforeAll(reset);

  it("matches hand-derived Adadelta over 3 steps", async () => {
    const result = await target.run<{ p: number[]; oracle: number[] }>(`
${PRELUDE}
rho = 0.9
eps = 1e-6
lr = 1.0

p = grad.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
opt = optim.Adadelta([p], lr=lr, rho=rho, eps=eps)

p_oracle = np.array([1.0, 2.0, 3.0], dtype=np.float64)
Eg = np.zeros(3, dtype=np.float64)
Ex = np.zeros(3, dtype=np.float64)

grads = [np.array([0.1, -0.2, 0.5], dtype=np.float64),
         np.array([0.0, 0.3, -0.1], dtype=np.float64),
         np.array([-0.4, 0.1, 0.2], dtype=np.float64)]

for g in grads:
    p.grad = grad.Tensor(g.astype(np.float32))
    opt.step()
    Eg = rho * Eg + (1 - rho) * g * g
    dx = -(np.sqrt(Ex + eps) / np.sqrt(Eg + eps)) * g
    Ex = rho * Ex + (1 - rho) * dx * dx
    p_oracle += lr * dx

{"p": p.tolist(), "oracle": p_oracle.tolist()}
`);
    for (let i = 0; i < result.p.length; i++) {
      expect(result.p[i]!).toBeCloseTo(result.oracle[i]!, 4);
    }
  });
});

describe("optim.lr_scheduler.MultiStepLR", () => {
  beforeAll(reset);

  it("drops lr by gamma at each milestone", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
p = grad.Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
opt = optim.SGD([p], lr=1.0)
sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[3, 5], gamma=0.1)
lrs = []
for _ in range(7):
    sched.step()
    lrs.append(opt.lr)
lrs
`);
    // step 1: lr=1.0 (no milestone yet), step 2: 1.0, step 3: 0.1, step 4: 0.1, step 5: 0.01, step 6/7: 0.01
    expect(result[0]).toBeCloseTo(1.0, 6);
    expect(result[1]).toBeCloseTo(1.0, 6);
    expect(result[2]).toBeCloseTo(0.1, 6);
    expect(result[3]).toBeCloseTo(0.1, 6);
    expect(result[4]).toBeCloseTo(0.01, 6);
    expect(result[5]).toBeCloseTo(0.01, 6);
    expect(result[6]).toBeCloseTo(0.01, 6);
  });
});

describe("optim.lr_scheduler.ExponentialLR", () => {
  beforeAll(reset);

  it("multiplies lr by gamma each step", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
p = grad.Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
opt = optim.SGD([p], lr=1.0)
sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
lrs = []
for _ in range(4):
    sched.step()
    lrs.append(opt.lr)
lrs
`);
    expect(result[0]).toBeCloseTo(0.5, 6);
    expect(result[1]).toBeCloseTo(0.25, 6);
    expect(result[2]).toBeCloseTo(0.125, 6);
    expect(result[3]).toBeCloseTo(0.0625, 6);
  });
});

describe("optim.lr_scheduler.ReduceLROnPlateau", () => {
  beforeAll(reset);

  it("reduces lr after `patience` epochs without improvement", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
p = grad.Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
opt = optim.SGD([p], lr=1.0)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
# Feed losses: improve, then plateau for 3 epochs (should trigger reduction)
losses = [10.0, 5.0, 5.0, 5.0, 5.0, 5.0]
lrs = []
for L in losses:
    sched.step(L)
    lrs.append(opt.lr)
lrs
`);
    // PyTorch's rule: reduce when num_bad_epochs > patience (NOT >=).
    // With patience=2:
    //  E1 loss=10: best=10, counter=0
    //  E2 loss=5:  is_better=True, best=5, counter=0
    //  E3 loss=5:  counter=1
    //  E4 loss=5:  counter=2 (still ≤ patience, no reduce)
    //  E5 loss=5:  counter=3 (> patience) → REDUCE → counter resets to 0
    //  E6 loss=5:  counter=1 (no reduce yet)
    expect(result[0]).toBeCloseTo(1.0, 6);
    expect(result[1]).toBeCloseTo(1.0, 6);
    expect(result[2]).toBeCloseTo(1.0, 6);
    expect(result[3]).toBeCloseTo(1.0, 6);
    expect(result[4]).toBeCloseTo(0.5, 6);
    expect(result[5]).toBeCloseTo(0.5, 6);
  });
});

describe("optim.lr_scheduler.OneCycleLR", () => {
  beforeAll(reset);

  it("traces a single triangular cycle: warmup then anneal", async () => {
    const result = await target.run<{ lrs: number[]; max_idx: number; max_lr: number }>(`
${PRELUDE}
p = grad.Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
opt = optim.SGD([p], lr=0.0)  # OneCycleLR overrides
sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=1.0, total_steps=10, pct_start=0.3)
lrs = []
for _ in range(10):
    sched.step()
    lrs.append(opt.lr)
{"lrs": lrs, "max_idx": int(np.argmax(lrs)), "max_lr": float(max(lrs))}
`);
    // Peak should land near step pct_start * total_steps = 3 (zero-indexed: 2 or 3).
    expect(result.max_lr).toBeCloseTo(1.0, 4);
    expect(result.max_idx).toBeGreaterThanOrEqual(1);
    expect(result.max_idx).toBeLessThanOrEqual(3);
    // Final lr should be much smaller than peak (anneal phase finished).
    expect(result.lrs[9]!).toBeLessThan(0.1);
  });
});
