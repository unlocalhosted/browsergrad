/**
 * Pile A #7 + #8 + #12 — F.pad, F.interpolate, F.normalize,
 * F.cosine_similarity, F.scaled_dot_product_attention, grad.einsum.
 *
 * Independent oracles:
 *  - F.pad: NumPy np.pad with the same mode.
 *  - F.interpolate: hand-computed nearest / bilinear from first principles.
 *  - F.normalize, cosine_similarity: explicit formula in NumPy.
 *  - scaled_dot_product_attention: softmax(QK^T/sqrt(d))V written out in NumPy.
 *  - einsum: forward against np.einsum directly; backward via finite differences.
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

describe("F.pad", () => {
  beforeAll(reset);

  it("constant pad on a 2D tensor matches np.pad", async () => {
    const result = await target.run<{ ours: number[][]; oracle: number[][] }>(`
${PRELUDE}
x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
x = grad.Tensor(x_np)
# torch.nn.functional.pad uses last-dim-first convention: (left, right, top, bottom)
y = F.pad(x, (1, 2, 0, 1), mode='constant', value=0.0)
oracle = np.pad(x_np, ((0, 1), (1, 2)), mode='constant')
{"ours": y.tolist(), "oracle": oracle.tolist()}
`);
    expect(result.ours).toEqual(result.oracle);
  });

  it("pad backward routes grads only through unpadded entries", async () => {
    const result = await target.run<{ grad: number[][] }>(`
${PRELUDE}
x = grad.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
y = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
y.sum().backward()
{"grad": np.asarray(x.grad).tolist()}
`);
    expect(result.grad).toEqual([[1, 1], [1, 1]]);
  });
});

describe("F.interpolate", () => {
  beforeAll(reset);

  it("nearest-neighbor 2x upsample picks the floor source index", async () => {
    const result = await target.run<{ ours: number[][][][]; oracle: number[][][][] }>(`
${PRELUDE}
x_np = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # (1,1,2,2)
x = grad.Tensor(x_np)
y = F.interpolate(x, scale_factor=2.0, mode='nearest')
# Expected: each value tiled into a 2x2 block.
oracle = np.array([[[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]]], dtype=np.float32)
{"ours": y.tolist(), "oracle": oracle.tolist()}
`);
    expect(result.ours).toEqual(result.oracle);
  });

  it("bilinear 2x upsample at align_corners=False matches the formula", async () => {
    const result = await target.run<{ ours: number[][][][]; oracle: number[][][][] }>(`
${PRELUDE}
x_np = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=np.float32)  # (1,1,2,2)
x = grad.Tensor(x_np)
y = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
# Hand-compute by mapping output (h,w) → source via half-pixel center rule:
#   src = (out + 0.5) / scale - 0.5
def lin(out, H_in, scale):
    return (out + 0.5) / scale - 0.5
H_out, W_out = 4, 4
oracle = np.zeros((1,1,4,4), dtype=np.float32)
for i in range(H_out):
    si = lin(i, 2, 2.0)
    i0 = int(np.floor(si)); i1 = i0 + 1
    a = si - i0
    i0c = max(0, min(1, i0)); i1c = max(0, min(1, i1))
    for j in range(W_out):
        sj = lin(j, 2, 2.0)
        j0 = int(np.floor(sj)); j1 = j0 + 1
        b = sj - j0
        j0c = max(0, min(1, j0)); j1c = max(0, min(1, j1))
        v00 = x_np[0,0,i0c,j0c]; v01 = x_np[0,0,i0c,j1c]
        v10 = x_np[0,0,i1c,j0c]; v11 = x_np[0,0,i1c,j1c]
        oracle[0,0,i,j] = (1-a)*((1-b)*v00 + b*v01) + a*((1-b)*v10 + b*v11)
ours = np.asarray(y.tolist())
oracle_l = oracle.tolist()
ours_l = ours.tolist()
{"ours": ours_l, "oracle": oracle_l}
`);
    // close compare with element-wise tolerance
    for (let n = 0; n < 1; n++) {
      for (let c = 0; c < 1; c++) {
        for (let i = 0; i < 4; i++) {
          for (let j = 0; j < 4; j++) {
            expect(result.ours[n]![c]![i]![j]!).toBeCloseTo(result.oracle[n]![c]![i]![j]!, 5);
          }
        }
      }
    }
  });
});

describe("F.normalize and F.cosine_similarity", () => {
  beforeAll(reset);

  it("normalize divides by L2 along dim", async () => {
    const result = await target.run<{ ours: number[][]; norms: number[] }>(`
${PRELUDE}
x = grad.Tensor(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32), requires_grad=True)
y = F.normalize(x, dim=1)
ours = y.tolist()
norms = np.linalg.norm(np.asarray(ours), axis=1).tolist()
{"ours": ours, "norms": norms}
`);
    // Each row should now have unit norm.
    for (const n of result.norms) expect(n).toBeCloseTo(1.0, 5);
  });

  it("cosine_similarity matches dot / (|a||b|)", async () => {
    const result = await target.run<{ ours: number[]; oracle: number[] }>(`
${PRELUDE}
a_np = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
b_np = np.array([[1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0]], dtype=np.float32)
a = grad.Tensor(a_np); b = grad.Tensor(b_np)
out = F.cosine_similarity(a, b, dim=1)
oracle = (a_np * b_np).sum(axis=1) / (np.linalg.norm(a_np, axis=1) * np.linalg.norm(b_np, axis=1))
{"ours": out.tolist(), "oracle": oracle.tolist()}
`);
    for (let i = 0; i < result.ours.length; i++) {
      expect(result.ours[i]!).toBeCloseTo(result.oracle[i]!, 5);
    }
  });
});

describe("F.scaled_dot_product_attention", () => {
  beforeAll(reset);

  it("matches softmax(QK^T/sqrt(d_k)) V on a small case", async () => {
    const result = await target.run<{ ours: number[][]; oracle: number[][] }>(`
${PRELUDE}
np.random.seed(0)
Q_np = np.random.randn(2, 4).astype(np.float32)
K_np = np.random.randn(3, 4).astype(np.float32)
V_np = np.random.randn(3, 5).astype(np.float32)
Q, K, V = grad.Tensor(Q_np), grad.Tensor(K_np), grad.Tensor(V_np)
out = F.scaled_dot_product_attention(Q, K, V)
scores = Q_np @ K_np.T / np.sqrt(4.0)
e = np.exp(scores - scores.max(axis=-1, keepdims=True))
attn = e / e.sum(axis=-1, keepdims=True)
oracle = attn @ V_np
{"ours": out.tolist(), "oracle": oracle.tolist()}
`);
    for (let i = 0; i < result.ours.length; i++) {
      for (let j = 0; j < result.ours[i]!.length; j++) {
        expect(result.ours[i]![j]!).toBeCloseTo(result.oracle[i]![j]!, 4);
      }
    }
  });

  it("respects a boolean attn_mask: masked-true positions are zeroed in softmax", async () => {
    const result = await target.run<{ row0_sum: number; masked_position_attn: number }>(`
${PRELUDE}
Q = grad.Tensor(np.eye(2, 3, dtype=np.float32))
K = grad.Tensor(np.eye(2, 3, dtype=np.float32))
V = grad.Tensor(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
# Mask out (row=0, col=1) — torch convention: True means "do NOT attend"
mask = np.array([[False, True], [False, False]])
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
# Position (row=0, col=1) attention weight should be ~0 — i.e. row 0 of out
# should equal V[0] (= [1, 0]) since attention all goes to position 0.
{"row0_sum": float(np.asarray(out.tolist())[0].sum()), "masked_position_attn": float(np.asarray(out.tolist())[0][1])}
`);
    expect(result.masked_position_attn).toBeCloseTo(0, 4);
    expect(result.row0_sum).toBeCloseTo(1.0, 4);
  });
});

describe("grad.einsum", () => {
  beforeAll(reset);

  it("forward: ij,jk->ik matches np.einsum (matmul)", async () => {
    const result = await target.run<{ ours: number[][]; oracle: number[][] }>(`
${PRELUDE}
a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
b_np = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
a = grad.Tensor(a_np); b = grad.Tensor(b_np)
out = grad.einsum("ij,jk->ik", a, b)
oracle = np.einsum("ij,jk->ik", a_np, b_np)
{"ours": out.tolist(), "oracle": oracle.tolist()}
`);
    expect(result.ours).toEqual(result.oracle);
  });

  it("forward: bij,bjk->bik (batched matmul) matches np.einsum", async () => {
    const result = await target.run<{ ours: number[][][]; oracle: number[][][] }>(`
${PRELUDE}
np.random.seed(7)
a_np = np.random.randn(2, 3, 4).astype(np.float32)
b_np = np.random.randn(2, 4, 5).astype(np.float32)
out = grad.einsum("bij,bjk->bik", grad.Tensor(a_np), grad.Tensor(b_np))
oracle = np.einsum("bij,bjk->bik", a_np, b_np)
{"ours": out.tolist(), "oracle": oracle.tolist()}
`);
    for (let bi = 0; bi < result.ours.length; bi++) {
      for (let i = 0; i < result.ours[bi]!.length; i++) {
        for (let j = 0; j < result.ours[bi]![i]!.length; j++) {
          expect(result.ours[bi]![i]![j]!).toBeCloseTo(result.oracle[bi]![i]![j]!, 4);
        }
      }
    }
  });

  it("backward: einsum gradient matches finite differences (ij,jk->ik)", async () => {
    const result = await target.run<{ grad_a: number[][]; fd_a: number[][] }>(`
${PRELUDE}
np.random.seed(11)
a_np = np.random.randn(2, 3).astype(np.float64)
b_np = np.random.randn(3, 4).astype(np.float64)
a = grad.Tensor(a_np.astype(np.float32), requires_grad=True)
b = grad.Tensor(b_np.astype(np.float32))
out = grad.einsum("ij,jk->ik", a, b)
out.sum().backward()
grad_a = np.asarray(a.grad).copy().astype(np.float64)

# Finite-diff oracle: scalar loss = sum(einsum(a,b))
def fwd(aa):
    return float(np.einsum("ij,jk->ik", aa, b_np).sum())
eps = 1e-3
fd = np.zeros_like(a_np)
for i in range(a_np.shape[0]):
    for j in range(a_np.shape[1]):
        ap = a_np.copy(); ap[i,j] += eps
        am = a_np.copy(); am[i,j] -= eps
        fd[i,j] = (fwd(ap) - fwd(am)) / (2 * eps)
{"grad_a": grad_a.tolist(), "fd_a": fd.tolist()}
`);
    for (let i = 0; i < result.grad_a.length; i++) {
      for (let j = 0; j < result.grad_a[i]!.length; j++) {
        expect(result.grad_a[i]![j]!).toBeCloseTo(result.fd_a[i]![j]!, 3);
      }
    }
  });
});
