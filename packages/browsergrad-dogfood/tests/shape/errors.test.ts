/**
 * Shape error adversarial tests.
 *
 * Hypotheses:
 *   H16 — wrong-rank input → throws
 *   H17 — inner-dim mismatch → throws
 *   H40 — tensor shape lies about data length
 */

import { describe, expect, it } from "vitest";
import { tensor, KernelError } from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";

describe("shape errors — adversarial", () => {
  it("matmul rejects rank-1 input (H16)", () => {
    const A = tensor([3] as unknown as readonly [number, number], new Float32Array(3));
    const B = tensor([3, 2], new Float32Array(6));
    expect(() => reference.matmul(A as never, B)).toThrow();
  });

  it("matmul rejects inner-dim mismatch (H17)", () => {
    const A = tensor([2, 3], new Float32Array(6));
    const B = tensor([4, 2], new Float32Array(8));
    expect(() => reference.matmul(A, B)).toThrow();
  });

  it("matmul error is KernelError, not generic Error", () => {
    const A = tensor([2, 3], new Float32Array(6));
    const B = tensor([4, 2], new Float32Array(8));
    expect(() => reference.matmul(A, B)).toThrow(KernelError);
  });

  it("tensor shape mismatch with data length: behavior (H40)", () => {
    // Spec ambiguity: does tensor() validate? Document actual behavior.
    try {
      const t = tensor([100, 100], new Float32Array(4));
      // If construction allowed, downstream ops will detect mismatch.
      console.warn(`[H40] tensor() accepts shape [100,100] with data.length=4 — data.length=${t.data.length}`);
      expect(t.data.length).toBe(4); // data is what was given
    } catch (e) {
      expect(e).toBeInstanceOf(Error);
    }
  });
});
