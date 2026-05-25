/**
 * Cross-package conformance: the kernel reference impls in
 * @unlocalhosted/browsergrad-kernels must agree numerically with the
 * equivalent operations in @unlocalhosted/browsergrad-grad's NumPy-backed
 * autograd. If one package drifts (or has a bug), this test catches it.
 *
 * We compare on identical f32 inputs and assert tolerance 1e-4 (slack
 * because each path can accumulate rounding differently).
 */

import { beforeAll, describe, expect, it } from "vitest";
import { reference, tensor as kTensor } from "../../browsergrad-kernels/src/index";
import { getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

const PRELUDE = `
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import numpy as np
`;

function close(actual: readonly number[], expected: readonly number[], tol = 1e-4): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

describe("cross-package: kernels reference impls vs grad NumPy ops", () => {
  it("matmul reference (JS) matches grad's @ operator (Python)", async () => {
    // Same data on both sides.
    const Adata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const Bdata = [7, 8, 9, 10, 11, 12, 13, 14, 15];
    // JS reference path
    const A = kTensor([4, 3], new Float32Array(Adata));
    const B = kTensor([3, 3], new Float32Array(Bdata));
    const kernelsResult = Array.from(reference.matmul(A, B).data);

    // grad path through Pyodide
    const gradResult = await target.run<number[]>(`
${PRELUDE}
A = grad.Tensor(np.array([${Adata.join(",")}], dtype=np.float32).reshape(4, 3))
B = grad.Tensor(np.array([${Bdata.join(",")}], dtype=np.float32).reshape(3, 3))
(A @ B).data.flatten().tolist()
`);
    close(kernelsResult, gradResult);
  });

  it("softmax reference (JS) matches grad's F.softmax (Python)", async () => {
    const xData = [-2.3, 1.1, 0.5, 4.0, -1.5, 0.0, 2.7, -0.8];
    const x = kTensor([2, 4], new Float32Array(xData));
    const kernelsResult = Array.from(reference.softmax(x).data);

    const gradResult = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor(np.array([${xData.join(",")}], dtype=np.float32).reshape(2, 4))
F.softmax(x, dim=-1).data.flatten().tolist()
`);
    close(kernelsResult, gradResult);
  });

  it("relu reference (JS) matches grad's F.relu (Python)", async () => {
    const xData = [-2, -1, 0, 1, 2, -0.5, 0.5, 3.7];
    const x = kTensor([8], new Float32Array(xData));
    const kernelsResult = Array.from(reference.relu(x).data);

    const gradResult = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([${xData.join(",")}])
F.relu(x).data.tolist()
`);
    close(kernelsResult, gradResult);
  });

  it("gelu reference (JS) matches grad's F.gelu (Python)", async () => {
    const xData = [-2, -1, 0, 1, 2, 0.7, -0.3];
    const x = kTensor([7], new Float32Array(xData));
    const kernelsResult = Array.from(reference.gelu(x).data);

    const gradResult = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([${xData.join(",")}])
F.gelu(x).data.tolist()
`);
    close(kernelsResult, gradResult);
  });

  it("layernorm reference (JS) matches grad's nn.LayerNorm (Python)", async () => {
    const xData = [1, 2, 3, 4, 5, 6, 7, 8];  // (2, 4)
    const x = kTensor([2, 4], new Float32Array(xData));
    const kernelsResult = Array.from(reference.layernorm(x).data);

    const gradResult = await target.run<number[]>(`
${PRELUDE}
import browsergrad_grad.nn as nn
x = grad.Tensor(np.array([${xData.join(",")}], dtype=np.float32).reshape(2, 4))
ln = nn.LayerNorm(4, elementwise_affine=False)
ln(x).data.flatten().tolist()
`);
    close(kernelsResult, gradResult);
  });

  it("attention reference (JS) matches the same SDP attention via grad ops (Python)", async () => {
    // (S, D) = (4, 8). Same Q=K=V on both sides → identity-like attention.
    const data = Array.from({ length: 4 * 8 }, (_, i) => Math.sin(i * 0.7));
    const Q = kTensor([4, 8], new Float32Array(data));
    const kernelsResult = Array.from(reference.attention(Q, Q, Q).data);

    const gradResult = await target.run<number[]>(`
${PRELUDE}
X = np.array([${data.join(",")}], dtype=np.float32).reshape(4, 8)
Q = grad.Tensor(X); K = grad.Tensor(X); V = grad.Tensor(X)
# Hand-build SDP attention via grad ops to compare against kernel reference
scores = (Q @ K.transpose(0, 1)) * (1.0 / np.sqrt(8))
weights = F.softmax(scores, dim=-1)
(weights @ V).data.flatten().tolist()
`);
    close(kernelsResult, gradResult, 1e-3);
  });
});
