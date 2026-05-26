/**
 * Pile A #9 — more losses: BCELoss, L1Loss, SmoothL1Loss, KLDivLoss.
 *
 * Independent oracles:
 *  - forward: hand-derived NumPy directly (closed form per loss).
 *  - backward: central finite differences on a scalar perturbation.
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
import browsergrad_grad.functional as F
import numpy as np
`;

describe("L1Loss", () => {
  beforeAll(reset);

  it("forward matches |x-y|.mean()", async () => {
    const result = await target.run<{ ours: number; oracle: number }>(`
${PRELUDE}
x_np = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
y_np = np.array([[1.5,  0.0], [2.0, 5.0]], dtype=np.float32)
x = grad.Tensor(x_np, requires_grad=True)
y = grad.Tensor(y_np)
loss = nn.L1Loss()(x, y)
oracle = float(np.abs(x_np - y_np).mean())
{"ours": float(loss.data), "oracle": oracle}
`);
    expect(result.ours).toBeCloseTo(result.oracle, 5);
  });

  it("backward gives sign(x-y) / N", async () => {
    const result = await target.run<{ grad: number[][]; oracle: number[][] }>(`
${PRELUDE}
x_np = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
y_np = np.array([[1.5,  0.0], [2.0, 5.0]], dtype=np.float32)
x = grad.Tensor(x_np, requires_grad=True)
y = grad.Tensor(y_np)
loss = nn.L1Loss()(x, y)
loss.backward()
oracle = np.sign(x_np - y_np) / x_np.size
{"grad": x.grad.tolist(), "oracle": oracle.tolist()}
`);
    expect(result.grad).toEqual(result.oracle);
  });
});

describe("BCELoss", () => {
  beforeAll(reset);

  it("forward matches -(y*log(p) + (1-y)*log(1-p)).mean() for probs in (0,1)", async () => {
    const result = await target.run<{ ours: number; oracle: number }>(`
${PRELUDE}
p_np = np.array([0.2, 0.7, 0.9, 0.4], dtype=np.float32)
y_np = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
p = grad.Tensor(p_np, requires_grad=True)
y = grad.Tensor(y_np)
loss = nn.BCELoss()(p, y)
oracle = float(-(y_np * np.log(p_np) + (1.0 - y_np) * np.log(1.0 - p_np)).mean())
{"ours": float(loss.data), "oracle": oracle}
`);
    expect(result.ours).toBeCloseTo(result.oracle, 5);
  });

  it("backward matches finite differences", async () => {
    const result = await target.run<{ ours: number[]; fd: number[] }>(`
${PRELUDE}
p_np = np.array([0.2, 0.7, 0.9, 0.4], dtype=np.float64)
y_np = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
p = grad.Tensor(p_np.astype(np.float32), requires_grad=True)
y = grad.Tensor(y_np.astype(np.float32))
loss = nn.BCELoss()(p, y)
loss.backward()
ours = np.asarray(p.grad).copy()

# central finite differences on float64
def fwd(pp):
    return float(-(y_np * np.log(pp) + (1.0 - y_np) * np.log(1.0 - pp)).mean())
eps = 1e-3
fd = np.zeros_like(p_np)
for i in range(p_np.size):
    pp_plus = p_np.copy(); pp_plus[i] += eps
    pp_minus = p_np.copy(); pp_minus[i] -= eps
    fd[i] = (fwd(pp_plus) - fwd(pp_minus)) / (2 * eps)
{"ours": ours.tolist(), "fd": fd.tolist()}
`);
    for (let i = 0; i < result.ours.length; i++) {
      expect(result.ours[i]!).toBeCloseTo(result.fd[i]!, 3);
    }
  });
});

describe("SmoothL1Loss", () => {
  beforeAll(reset);

  it("forward uses quadratic region for |x-y| < beta", async () => {
    const result = await target.run<{ ours: number; oracle: number }>(`
${PRELUDE}
# beta=1.0; pick diffs all under 1 to land in the quadratic region
x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
y_np = np.array([1.2, 2.5, 2.9], dtype=np.float32)
x = grad.Tensor(x_np, requires_grad=True)
y = grad.Tensor(y_np)
loss = nn.SmoothL1Loss(beta=1.0)(x, y)
diff = x_np - y_np
oracle = float((0.5 * diff * diff).mean())
{"ours": float(loss.data), "oracle": oracle}
`);
    expect(result.ours).toBeCloseTo(result.oracle, 5);
  });

  it("forward uses linear region for |x-y| >= beta", async () => {
    const result = await target.run<{ ours: number; oracle: number }>(`
${PRELUDE}
# all diffs > beta=1.0
x_np = np.array([5.0, 6.0, 7.0], dtype=np.float32)
y_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
x = grad.Tensor(x_np, requires_grad=True)
y = grad.Tensor(y_np)
loss = nn.SmoothL1Loss(beta=1.0)(x, y)
diff = np.abs(x_np - y_np)
oracle = float((diff - 0.5).mean())  # beta=1, so beta*(|d|-0.5*beta)=|d|-0.5
{"ours": float(loss.data), "oracle": oracle}
`);
    expect(result.ours).toBeCloseTo(result.oracle, 5);
  });

  it("backward matches finite differences (mixed regions)", async () => {
    const result = await target.run<{ ours: number[]; fd: number[] }>(`
${PRELUDE}
beta = 1.0
x_np = np.array([0.5, 2.0, -3.0, 0.9], dtype=np.float64)
y_np = np.array([0.0, 0.0,  0.0, 0.0], dtype=np.float64)
x = grad.Tensor(x_np.astype(np.float32), requires_grad=True)
y = grad.Tensor(y_np.astype(np.float32))
loss = nn.SmoothL1Loss(beta=beta)(x, y)
loss.backward()
ours = np.asarray(x.grad).copy()

def fwd(xx):
    d = xx - y_np
    a = np.abs(d)
    return float(np.where(a < beta, 0.5 * d * d / beta, a - 0.5 * beta).mean())
eps = 1e-3
fd = np.zeros_like(x_np)
for i in range(x_np.size):
    xp = x_np.copy(); xp[i] += eps
    xm = x_np.copy(); xm[i] -= eps
    fd[i] = (fwd(xp) - fwd(xm)) / (2 * eps)
{"ours": ours.tolist(), "fd": fd.tolist()}
`);
    for (let i = 0; i < result.ours.length; i++) {
      expect(result.ours[i]!).toBeCloseTo(result.fd[i]!, 3);
    }
  });
});

describe("KLDivLoss", () => {
  beforeAll(reset);

  it("forward matches (target * (log(target) - input)).sum(dim=1).mean() with log_target=False (default), input is log-prob", async () => {
    const result = await target.run<{ ours: number; oracle: number }>(`
${PRELUDE}
# Two probability distributions over 3 classes
logp_np = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float32))
t_np = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype=np.float32)
logp = grad.Tensor(logp_np, requires_grad=True)
t = grad.Tensor(t_np)
# default in torch.nn.KLDivLoss is reduction='mean' which divides by total elements
loss = nn.KLDivLoss(reduction='batchmean')(logp, t)
oracle = float((t_np * (np.log(t_np) - logp_np)).sum() / t_np.shape[0])
{"ours": float(loss.data), "oracle": oracle}
`);
    expect(result.ours).toBeCloseTo(result.oracle, 5);
  });

  it("backward of batchmean KLDivLoss is -target / N over inputs (log-probs)", async () => {
    const result = await target.run<{ ours: number[][]; oracle: number[][] }>(`
${PRELUDE}
logp_np = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float32))
t_np = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype=np.float32)
logp = grad.Tensor(logp_np, requires_grad=True)
t = grad.Tensor(t_np)
loss = nn.KLDivLoss(reduction='batchmean')(logp, t)
loss.backward()
oracle = -t_np / t_np.shape[0]
{"ours": logp.grad.tolist(), "oracle": oracle.tolist()}
`);
    for (let i = 0; i < result.ours.length; i++) {
      const oursRow = result.ours[i]!;
      const oracleRow = result.oracle[i]!;
      for (let j = 0; j < oursRow.length; j++) {
        expect(oursRow[j]!).toBeCloseTo(oracleRow[j]!, 5);
      }
    }
  });
});
