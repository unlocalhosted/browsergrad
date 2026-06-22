import { describe, expect, it } from "vitest";
import {
  aggregateLossAcrossMicrobatch,
  computePolicyGradientLoss,
} from "../../src/rl";

describe("policy gradient loss helpers", () => {
  it("computes on-policy and clipped GRPO per-token losses", () => {
    const onPolicy = computePolicyGradientLoss({
      advantages: [1, -0.5],
      policyLogProbs: [
        [-0.2, -0.4],
        [-1.2, -0.8],
      ],
      importanceReweightingMethod: "none",
    });

    expect(onPolicy.perTokenLoss).toEqual([
      [0.2, 0.4],
      [-0.6, -0.4],
    ]);

    const clipped = computePolicyGradientLoss({
      advantages: [1, -1],
      policyLogProbs: [
        [-0.1, -0.2],
        [-0.7, -0.8],
      ],
      oldLogProbs: [
        [-0.4, -0.2],
        [-0.6, -1.2],
      ],
      importanceReweightingMethod: "grpo",
      cliprange: 0.1,
    });

    expect(clipped.perTokenLoss[0]).toEqual([
      expect.closeTo(-1.1, 6),
      -1,
    ]);
    expect(clipped.perTokenLoss[1]).toEqual([
      expect.closeTo(0.904837, 6),
      1.1,
    ]);
    expect(clipped.metadata.clipFraction).toBe(0.5);
  });

  it("aggregates masked token losses by sequence or constant normalization", () => {
    const perTokenLoss = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const mask = [
      [1, 1, 0],
      [0, 1, 1],
    ];

    expect(
      aggregateLossAcrossMicrobatch({
        perTokenLoss,
        mask,
        lossNormalization: "sequence",
      }),
    ).toBe(3.5);
    expect(
      aggregateLossAcrossMicrobatch({
        perTokenLoss,
        mask,
        lossNormalization: "constant",
        normalizationConstant: 8,
      }),
    ).toBe(1.75);
  });
});
