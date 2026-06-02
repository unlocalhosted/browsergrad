import { describe, expect, it } from "vitest";
import {
  reference,
  referenceAttention,
  referenceGelu,
  referenceLayerNorm,
  referenceMatmul,
  referenceRelu,
  referenceSoftmax,
} from "../src/reference";
import { tensor, KernelError, type Tensor } from "../src/index";

/**
 * Tests for the JS reference implementations.
 *
 * These ARE the conformance oracles — the same input/output pairs will be
 * reused later to test the WGSL kernels in a browser environment.
 *
 * Numerical tolerance: 1e-5 for everything. All math is f32-equivalent in
 * JS (TypedArray storage), but JS internal computation is f64 so the
 * reference is slightly more accurate than the eventual WGSL. The WGSL
 * conformance pass will use 1e-4 to absorb f32 accumulation drift.
 */

const TOL = 1e-5;

function expectClose(actual: Float32Array, expected: number[], tol = TOL): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

/* ────────────────────────────────────────────────────────────
 * matmul
 * ──────────────────────────────────────────────────────────── */

describe("referenceMatmul", () => {
  it("multiplies 2×3 by 3×2", () => {
    // A = [[1,2,3],[4,5,6]]   B = [[7,8],[9,10],[11,12]]
    // C = [[58, 64], [139, 154]]
    const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
    const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));
    const C = referenceMatmul(A, B);
    expect(Array.from(C.shape)).toEqual([2, 2]);
    expectClose(C.data, [58, 64, 139, 154]);
  });

  it("multiplies a 1×N row by an N×1 column → scalar", () => {
    const A = tensor([1, 4], new Float32Array([1, 2, 3, 4]));
    const B = tensor([4, 1], new Float32Array([5, 6, 7, 8]));
    const C = referenceMatmul(A, B);
    expect(Array.from(C.shape)).toEqual([1, 1]);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    expectClose(C.data, [70]);
  });

  it("rejects rank mismatch", () => {
    const A: Tensor = { shape: [2, 3, 4], data: new Float32Array(24) };
    const B: Tensor = { shape: [4, 5], data: new Float32Array(20) };
    expect(() => referenceMatmul(A, B)).toThrow(KernelError);
  });

  it("rejects inner-dim mismatch", () => {
    const A = tensor([2, 3], new Float32Array(6));
    const B = tensor([4, 2], new Float32Array(8));
    expect(() => referenceMatmul(A, B)).toThrow(/inner dimensions/);
  });
});

/* ────────────────────────────────────────────────────────────
 * softmax
 * ──────────────────────────────────────────────────────────── */

describe("referenceSoftmax", () => {
  it("returns uniform distribution on equal inputs", () => {
    const x = tensor([1, 4], new Float32Array([1, 1, 1, 1]));
    const y = referenceSoftmax(x);
    expectClose(y.data, [0.25, 0.25, 0.25, 0.25]);
  });

  it("each row sums to 1", () => {
    const x = tensor(
      [3, 5],
      new Float32Array([1, 2, 3, 4, 5, -1, -2, -3, -4, -5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    );
    const y = referenceSoftmax(x);
    for (let r = 0; r < 3; r++) {
      let s = 0;
      for (let i = 0; i < 5; i++) s += y.data[r * 5 + i]!;
      expect(Math.abs(s - 1)).toBeLessThan(TOL);
    }
  });

  it("handles large values without overflow (numerical stability)", () => {
    const x = tensor([1, 3], new Float32Array([1000, 1001, 1002]));
    const y = referenceSoftmax(x);
    // No NaN/Infinity, sum is 1.
    expect(y.data.every((v) => Number.isFinite(v))).toBe(true);
    let s = 0;
    for (const v of y.data) s += v;
    expect(Math.abs(s - 1)).toBeLessThan(TOL);
  });

  it("matches the bundle accessor", () => {
    const x = tensor([1, 3], new Float32Array([1, 2, 3]));
    const a = referenceSoftmax(x);
    const b = reference.softmax(x);
    expectClose(a.data, Array.from(b.data));
  });
});

/* ────────────────────────────────────────────────────────────
 * relu
 * ──────────────────────────────────────────────────────────── */

describe("referenceRelu", () => {
  it("zeros out negatives, preserves positives", () => {
    const x = tensor([5], new Float32Array([-2, -0.5, 0, 0.5, 2]));
    const y = referenceRelu(x);
    expectClose(y.data, [0, 0, 0, 0.5, 2]);
  });
});

/* ────────────────────────────────────────────────────────────
 * gelu
 * ──────────────────────────────────────────────────────────── */

describe("referenceGelu", () => {
  it("approximately reproduces known gelu values", () => {
    // Reference values for the tanh-approx GELU (GPT-2/BERT) at a few inputs.
    // Computed in Python: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))
    const x = tensor([5], new Float32Array([-2, -1, 0, 1, 2]));
    const y = referenceGelu(x);
    // expected values:
    //   gelu(-2) ≈ -0.04540231
    //   gelu(-1) ≈ -0.15880801
    //   gelu( 0) =  0
    //   gelu( 1) ≈  0.8411920
    //   gelu( 2) ≈  1.9545977
    expectClose(y.data, [-0.04540231, -0.15880801, 0, 0.8411920, 1.9545977], 1e-4);
  });

  it("gelu(0) = 0", () => {
    const x = tensor([1], new Float32Array([0]));
    const y = referenceGelu(x);
    expect(y.data[0]).toBe(0);
  });
});

/* ────────────────────────────────────────────────────────────
 * layernorm
 * ──────────────────────────────────────────────────────────── */

describe("referenceLayerNorm", () => {
  it("normalizes per-row to mean≈0, var≈1 with default gamma=1, beta=0", () => {
    const x = tensor([2, 4], new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]));
    const y = referenceLayerNorm(x);
    for (let r = 0; r < 2; r++) {
      let mean = 0;
      for (let i = 0; i < 4; i++) mean += y.data[r * 4 + i]!;
      mean /= 4;
      let varSum = 0;
      for (let i = 0; i < 4; i++) {
        const d = y.data[r * 4 + i]! - mean;
        varSum += d * d;
      }
      const variance = varSum / 4;
      expect(Math.abs(mean)).toBeLessThan(1e-4);
      // Var = 1 - eps/((var+eps)); with eps=1e-5 and var=O(1), close to 1.
      expect(Math.abs(variance - 1)).toBeLessThan(1e-3);
    }
  });

  it("applies gamma and beta", () => {
    const x = tensor([1, 4], new Float32Array([1, 2, 3, 4]));
    const gamma = tensor([4], new Float32Array([2, 2, 2, 2]));
    const beta = tensor([4], new Float32Array([10, 10, 10, 10]));
    const y = referenceLayerNorm(x, { gamma, beta });
    // After normalize, gamma=2 scales magnitude by 2, beta=10 shifts.
    // Mean of result = 10. Var of (result - 10) = 4 * Var(normed) ≈ 4.
    let mean = 0;
    for (const v of y.data) mean += v;
    mean /= 4;
    expect(Math.abs(mean - 10)).toBeLessThan(1e-3);
  });

  it("rejects gamma/beta with wrong shape", () => {
    const x = tensor([1, 4], new Float32Array(4));
    const wrongGamma = tensor([3], new Float32Array(3));
    expect(() => referenceLayerNorm(x, { gamma: wrongGamma })).toThrow(KernelError);
  });
});

/* ────────────────────────────────────────────────────────────
 * attention
 * ──────────────────────────────────────────────────────────── */

describe("referenceAttention", () => {
  it("reduces to V when Q=0 (uniform weights)", () => {
    // With Q=0, scores=0, softmax(scores)=uniform 1/S → out = mean(V) per output position.
    const S = 3;
    const D = 2;
    const Q = tensor([S, D], new Float32Array(S * D)); // all zeros
    const K = tensor([S, D], new Float32Array([1, 2, 3, 4, 5, 6]));
    const V = tensor([S, D], new Float32Array([10, 20, 30, 40, 50, 60]));
    const out = referenceAttention(Q, K, V);
    expect(Array.from(out.shape)).toEqual([S, D]);
    // Each output row = mean of V's rows = [(10+30+50)/3, (20+40+60)/3] = [30, 40]
    for (let r = 0; r < S; r++) {
      expect(Math.abs(out.data[r * D]! - 30)).toBeLessThan(1e-4);
      expect(Math.abs(out.data[r * D + 1]! - 40)).toBeLessThan(1e-4);
    }
  });

  it("output shape matches Q's sequence length and V's feature dim", () => {
    const Q = tensor([4, 8], new Float32Array(32));
    const K = tensor([4, 8], new Float32Array(32));
    const V = tensor([4, 16], new Float32Array(64));
    const out = referenceAttention(Q, K, V);
    expect(Array.from(out.shape)).toEqual([4, 16]);
  });

  it("rejects mismatched sequence lengths", () => {
    const Q = tensor([3, 4], new Float32Array(12));
    const K = tensor([5, 4], new Float32Array(20));
    const V = tensor([5, 4], new Float32Array(20));
    expect(() => referenceAttention(Q, K, V)).toThrow(/seq/);
  });

  it("rejects Q/K feature dim mismatch", () => {
    const Q = tensor([3, 4], new Float32Array(12));
    const K = tensor([3, 6], new Float32Array(18));
    const V = tensor([3, 4], new Float32Array(12));
    expect(() => referenceAttention(Q, K, V)).toThrow(/Q dim/);
  });
});
