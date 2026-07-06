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
import numpy as np

class MockKernelDevice:
    def __init__(self):
        self.calls = []
    def matmul(self, a, a_shape, b, b_shape):
        self.calls.append("matmul")
        A = np.asarray(a, dtype=np.float32).reshape(tuple(a_shape))
        B = np.asarray(b, dtype=np.float32).reshape(tuple(b_shape))
        return (A @ B).astype(np.float32).reshape(-1).tolist()
    def softmax(self, x, shape):
        self.calls.append("softmax")
        X = np.asarray(x, dtype=np.float32).reshape(tuple(shape))
        shifted = X - X.max(axis=-1, keepdims=True)
        E = np.exp(shifted)
        return (E / E.sum(axis=-1, keepdims=True)).astype(np.float32).reshape(-1).tolist()
    def layernorm(self, x, shape, gamma, beta, eps):
        self.calls.append("layernorm")
        X = np.asarray(x, dtype=np.float32).reshape(tuple(shape))
        G = np.ones((shape[-1],), dtype=np.float32) if gamma is None else np.asarray(gamma, dtype=np.float32)
        B = np.zeros((shape[-1],), dtype=np.float32) if beta is None else np.asarray(beta, dtype=np.float32)
        mean = X.mean(axis=-1, keepdims=True)
        var = ((X - mean) * (X - mean)).mean(axis=-1, keepdims=True)
        return (((X - mean) / np.sqrt(var + eps)) * G + B).astype(np.float32).reshape(-1).tolist()
    def attention(self, q, q_shape, k, k_shape, v, v_shape):
        self.calls.append("attention")
        Q = np.asarray(q, dtype=np.float32).reshape(tuple(q_shape))
        K = np.asarray(k, dtype=np.float32).reshape(tuple(k_shape))
        V = np.asarray(v, dtype=np.float32).reshape(tuple(v_shape))
        scores = Q @ K.T / np.sqrt(Q.shape[-1])
        shifted = scores - scores.max(axis=-1, keepdims=True)
        E = np.exp(shifted)
        W = E / E.sum(axis=-1, keepdims=True)
        return (W @ V).astype(np.float32).reshape(-1).tolist()
`;

describe("KernelDevice bridge", () => {
  beforeAll(reset);

  it("grad.matmul(device=...) routes forward through bridge and keeps backward", async () => {
    const result = await target.run<{
      calls: string[];
      out: number[][];
      grad_a: number[][];
      grad_b: number[][];
    }>(`
${PRELUDE}
dev = MockKernelDevice()
a = grad.Tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = grad.Tensor([[5., 6.], [7., 8.]], requires_grad=True)
y = grad.matmul(a, b, device=dev)
y.sum().backward()
{"calls": dev.calls, "out": y.tolist(), "grad_a": a.grad.tolist(), "grad_b": b.grad.tolist()}
`);
    expect(result.calls).toEqual(["matmul"]);
    expect(result.out).toEqual([
      [19, 22],
      [43, 50],
    ]);
    expect(result.grad_a).toEqual([
      [11, 15],
      [11, 15],
    ]);
    expect(result.grad_b).toEqual([
      [4, 4],
      [6, 6],
    ]);
  });

  it("F.softmax(device=...) routes last-dim softmax through bridge", async () => {
    const result = await target.run<{ calls: string[]; rows: number[] }>(`
${PRELUDE}
dev = MockKernelDevice()
x = grad.Tensor([[1., 2., 3.], [1., 1., 1.]], requires_grad=True)
y = F.softmax(x, dim=-1, device=dev)
{"calls": dev.calls, "rows": y.data.sum(axis=-1).tolist()}
`);
    expect(result.calls).toEqual(["softmax"]);
    for (const rowSum of result.rows) expect(rowSum).toBeCloseTo(1, 5);
  });

  it("nn.LayerNorm(device=...) routes forward through bridge and keeps affine grads", async () => {
    const result = await target.run<{ calls: string[]; shape: number[]; gw: number[]; gb: number[] }>(`
${PRELUDE}
dev = MockKernelDevice()
ln = nn.LayerNorm(3, device=dev)
ln.weight.data[:] = np.array([0.5, 1.5, -1.0], dtype=np.float32)
ln.bias.data[:] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
x = grad.Tensor([[1., 2., 4.], [0., 1., 0.]], requires_grad=True)
y = ln(x)
y.sum().backward()
{"calls": dev.calls, "shape": list(y.shape), "gw": ln.weight.grad.tolist(), "gb": ln.bias.grad.tolist()}
`);
    expect(result.calls).toEqual(["layernorm"]);
    expect(result.shape).toEqual([2, 3]);
    expect(result.gw.length).toBe(3);
    expect(result.gb).toEqual([2, 2, 2]);
  });

  it("F.scaled_dot_product_attention(device=...) routes unmasked 2D attention", async () => {
    const result = await target.run<{ calls: string[]; shape: number[]; out: number[][] }>(`
${PRELUDE}
dev = MockKernelDevice()
q = grad.Tensor([[1., 0.], [0., 1.]])
k = grad.Tensor([[1., 0.], [0., 1.]])
v = grad.Tensor([[2., 0.], [0., 4.]])
y = F.scaled_dot_product_attention(q, k, v, device=dev)
{"calls": dev.calls, "shape": list(y.shape), "out": y.tolist()}
`);
    expect(result.calls).toEqual(["attention"]);
    expect(result.shape).toEqual([2, 2]);
    expect(result.out[0]![0]!).toBeGreaterThan(result.out[0]![1]!);
    expect(result.out[1]![1]!).toBeGreaterThan(result.out[1]![0]!);
  });
});
