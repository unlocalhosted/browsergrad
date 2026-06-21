import { describe, expect, it } from "vitest";
import {
  referenceFlashAttention,
  referenceFlashAttentionBackward,
} from "../src/reference";
import { tensor } from "../src/index";

function expectCloseArray(actual: Float32Array, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(actual[index]).toBeCloseTo(expected[index]!, 5);
  }
}

describe("referenceFlashAttention", () => {
  it("returns output and log-sum-exp for upstream CS336 A2-style fixtures", () => {
    const query = tensor([1, 2, 2], new Float32Array([1, 0, 0, 1]));
    const key = tensor([1, 2, 2], new Float32Array([1, 0, 0, 1]));
    const value = tensor([1, 2, 2], new Float32Array([10, 20, 30, 40]));

    const nonCausal = referenceFlashAttention(query, key, value, { scale: 1 });
    expect(nonCausal.output.shape).toEqual([1, 2, 2]);
    expect(nonCausal.logSumExp.shape).toEqual([1, 2]);
    expectCloseArray(nonCausal.output.data, [
      15.378828,
      25.378828,
      24.621172,
      34.621172,
    ]);
    expectCloseArray(nonCausal.logSumExp.data, [1.313262, 1.313262]);

    const causal = referenceFlashAttention(query, key, value, {
      causal: true,
      scale: 1,
    });
    expectCloseArray(causal.output.data, [10, 20, 24.621172, 34.621172]);
    expectCloseArray(causal.logSumExp.data, [1, 1.313262]);
  });

  it("treats causal masking as a hard boundary, even for very negative valid scores", () => {
    const query = tensor([1, 1, 1], new Float32Array([1]));
    const key = tensor([1, 2, 1], new Float32Array([-2_000_000, 0]));
    const value = tensor([1, 2, 1], new Float32Array([7, 99]));

    const causal = referenceFlashAttention(query, key, value, {
      causal: true,
      scale: 1,
    });

    expectCloseArray(causal.output.data, [7]);
    expectCloseArray(causal.logSumExp.data, [-2_000_000]);
  });

  it("recomputes Q/K/V gradients for upstream CS336 A2-style backward checks", () => {
    const query = tensor([1, 2, 2], new Float32Array([1, 0, 0, 1]));
    const key = tensor([1, 2, 2], new Float32Array([1, 0, 0, 1]));
    const value = tensor([1, 2, 2], new Float32Array([10, 20, 30, 40]));
    const outputGradient = tensor([1, 2, 2], new Float32Array([1, 1, 1, 1]));

    const gradients = referenceFlashAttentionBackward(
      query,
      key,
      value,
      outputGradient,
      { scale: 1 },
    );

    expect(gradients.queryGradient.shape).toEqual([1, 2, 2]);
    expect(gradients.keyGradient.shape).toEqual([1, 2, 2]);
    expect(gradients.valueGradient.shape).toEqual([1, 2, 2]);
    expectCloseArray(gradients.queryGradient.data, [
      -7.864477,
      7.864477,
      -7.864477,
      7.864477,
    ]);
    expectCloseArray(gradients.keyGradient.data, [
      -7.864477,
      -7.864477,
      7.864477,
      7.864477,
    ]);
    expectCloseArray(gradients.valueGradient.data, [1, 1, 1, 1]);
  });
});
