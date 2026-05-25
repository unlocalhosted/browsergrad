/**
 * Pure-JS reference implementations of every WGSL kernel.
 *
 * Three roles, three audiences:
 *   1. Conformance oracle — tests compare WGSL output to these.
 *   2. CPU fallback — consumers without WebGPU can `import { reference }`
 *      and get the same surface that runs on the CPU. Slower; still correct.
 *   3. Lesson material — these are deliberately readable. A student should
 *      be able to read `referenceAttention` and understand attention.
 *
 * No micro-optimizations. Clarity over speed. The WGSL versions are where
 * speed lives.
 */

import { numel, assertDataLength } from "./runner";
import { KernelError, type Tensor } from "./types";

/* ────────────────────────────────────────────────────────────
 * matmul: A [M, K] × B [K, N] → C [M, N]
 * ──────────────────────────────────────────────────────────── */

export function referenceMatmul(A: Tensor, B: Tensor): Tensor {
  if (A.shape.length !== 2 || B.shape.length !== 2) {
    throw new KernelError(
      `matmul: expected 2D tensors, got ${A.shape.length}D and ${B.shape.length}D`,
    );
  }
  const [M, KA] = A.shape as [number, number];
  const [KB, N] = B.shape as [number, number];
  if (KA !== KB) {
    throw new KernelError(`matmul: inner dimensions don't match (${KA} vs ${KB})`);
  }
  assertDataLength("matmul.A", A.data, A.shape);
  assertDataLength("matmul.B", B.data, B.shape);

  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < KA; k++) {
        sum += A.data[i * KA + k]! * B.data[k * N + j]!;
      }
      out[i * N + j] = sum;
    }
  }
  return { shape: [M, N], data: out };
}

/* ────────────────────────────────────────────────────────────
 * softmax: along the last axis. Stable variant (subtracts max).
 * Accepts any rank; treats every row of size = lastDim independently.
 * ──────────────────────────────────────────────────────────── */

export function referenceSoftmax(x: Tensor): Tensor {
  if (x.shape.length === 0) {
    throw new KernelError("softmax: scalar tensor not supported");
  }
  const D = x.shape[x.shape.length - 1]!;
  const rows = numel(x.shape) / D;
  assertDataLength("softmax.x", x.data, x.shape);

  const out = new Float32Array(x.data.length);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    let maxVal = -Infinity;
    for (let i = 0; i < D; i++) {
      const v = x.data[base + i]!;
      if (v > maxVal) maxVal = v;
    }
    let sum = 0;
    for (let i = 0; i < D; i++) {
      const e = Math.exp(x.data[base + i]! - maxVal);
      out[base + i] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let i = 0; i < D; i++) {
      out[base + i]! *= inv;
    }
  }
  return { shape: x.shape, data: out };
}

/* ────────────────────────────────────────────────────────────
 * relu: elementwise max(0, x)
 * ──────────────────────────────────────────────────────────── */

export function referenceRelu(x: Tensor): Tensor {
  assertDataLength("relu.x", x.data, x.shape);
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i]!;
    out[i] = v > 0 ? v : 0;
  }
  return { shape: x.shape, data: out };
}

/* ────────────────────────────────────────────────────────────
 * gelu: Gaussian Error Linear Unit (tanh approximation).
 *
 *   gelu(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
 *
 * This is the variant used in GPT-2/BERT. We do NOT use the exact erf form
 * — the tanh approximation is standard in transformer impls and runs in
 * a single fused WGSL shader.
 * ──────────────────────────────────────────────────────────── */

const GELU_C = Math.sqrt(2 / Math.PI);

export function referenceGelu(x: Tensor): Tensor {
  assertDataLength("gelu.x", x.data, x.shape);
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i]!;
    const inner = GELU_C * (v + 0.044715 * v * v * v);
    out[i] = 0.5 * v * (1 + Math.tanh(inner));
  }
  return { shape: x.shape, data: out };
}

/* ────────────────────────────────────────────────────────────
 * layernorm: along the last axis.
 *
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * gamma and beta are optional; if omitted they default to identity (gamma=1, beta=0).
 * Shapes for gamma/beta when provided: [last_dim].
 * ──────────────────────────────────────────────────────────── */

export interface LayerNormOptions {
  gamma?: Tensor;
  beta?: Tensor;
  eps?: number;
}

export function referenceLayerNorm(x: Tensor, opts: LayerNormOptions = {}): Tensor {
  if (x.shape.length === 0) {
    throw new KernelError("layernorm: scalar tensor not supported");
  }
  const eps = opts.eps ?? 1e-5;
  const D = x.shape[x.shape.length - 1]!;
  const rows = numel(x.shape) / D;
  assertDataLength("layernorm.x", x.data, x.shape);
  if (opts.gamma) {
    if (opts.gamma.data.length !== D) {
      throw new KernelError(`layernorm: gamma length ${opts.gamma.data.length} doesn't match last dim ${D}`);
    }
  }
  if (opts.beta) {
    if (opts.beta.data.length !== D) {
      throw new KernelError(`layernorm: beta length ${opts.beta.data.length} doesn't match last dim ${D}`);
    }
  }

  const out = new Float32Array(x.data.length);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    // Mean
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x.data[base + i]!;
    mean /= D;
    // Variance
    let varSum = 0;
    for (let i = 0; i < D; i++) {
      const d = x.data[base + i]! - mean;
      varSum += d * d;
    }
    const variance = varSum / D;
    const invStd = 1 / Math.sqrt(variance + eps);
    // Normalize + affine
    for (let i = 0; i < D; i++) {
      let v = (x.data[base + i]! - mean) * invStd;
      if (opts.gamma) v *= opts.gamma.data[i]!;
      if (opts.beta) v += opts.beta.data[i]!;
      out[base + i] = v;
    }
  }
  return { shape: x.shape, data: out };
}

/* ────────────────────────────────────────────────────────────
 * attention: scaled dot-product (Vaswani et al. 2017).
 *
 *   scores = Q @ K^T / sqrt(d_k)
 *   weights = softmax(scores)   (along last axis)
 *   out = weights @ V
 *
 * v0 shapes: Q, K, V all [S, D]. Single head, no batching. The headline
 * lesson is to *see* attention compose from matmul and softmax — extending
 * to multi-head + batching is a future minor.
 * ──────────────────────────────────────────────────────────── */

export function referenceAttention(Q: Tensor, K: Tensor, V: Tensor): Tensor {
  if (Q.shape.length !== 2 || K.shape.length !== 2 || V.shape.length !== 2) {
    throw new KernelError(
      `attention: expected 2D Q, K, V, got ranks ${Q.shape.length}/${K.shape.length}/${V.shape.length}`,
    );
  }
  const [S, DQ] = Q.shape as [number, number];
  const [SK, DK] = K.shape as [number, number];
  const [SV] = V.shape as [number, number];
  if (DQ !== DK) throw new KernelError(`attention: Q dim ${DQ} ≠ K dim ${DK}`);
  if (S !== SK || S !== SV) {
    throw new KernelError(`attention: sequence lengths must match (Q: ${S}, K: ${SK}, V: ${SV})`);
  }

  // K^T has shape [DK, SK] = [DK, S]
  const KT: Tensor = { shape: [DK, SK], data: new Float32Array(DK * SK) };
  for (let i = 0; i < SK; i++) {
    for (let j = 0; j < DK; j++) {
      KT.data[j * SK + i] = K.data[i * DK + j]!;
    }
  }

  // scores = Q @ K^T → [S, S]
  const scores = referenceMatmul(Q, KT);

  // scale by 1/sqrt(d_k)
  const scale = 1 / Math.sqrt(DK);
  for (let i = 0; i < scores.data.length; i++) {
    scores.data[i]! *= scale;
  }

  // weights = softmax(scores)
  const weights = referenceSoftmax(scores);

  // out = weights @ V → [S, DV]
  return referenceMatmul(weights, V);
}

/* ────────────────────────────────────────────────────────────
 * Bundle for ergonomic `reference.matmul(...)` access.
 * ──────────────────────────────────────────────────────────── */

export const reference = {
  matmul: referenceMatmul,
  softmax: referenceSoftmax,
  relu: referenceRelu,
  gelu: referenceGelu,
  layernorm: referenceLayerNorm,
  attention: referenceAttention,
};
