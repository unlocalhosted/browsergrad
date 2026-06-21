import { describe, expect, it } from "vitest";
import { computeGroupNormalizedRewards, computeRolloutRewards } from "../src/index";

describe("GRPO reward helpers", () => {
  it("computes rollout reward tensors and aggregate metadata", () => {
    const result = computeRolloutRewards({
      rolloutResponses: ["world", "test", "a test", "another test"],
      repeatedGroundTruths: ["42", "42", "42", "42"],
      rewardFn: (response) => {
        const reward = { world: 1, test: 0, "a test": 0.5, "another test": 0 }[
          response
        ];
        return {
          reward: reward ?? 0,
          formatReward: reward && reward > 0 ? 1 : 0,
          answerReward: reward ?? 0,
        };
      },
    });

    expect(result.rawRewards).toEqual([1, 0, 0.5, 0]);
    expect(result.metadata).toEqual({
      meanReward: 0.375,
      meanFormatReward: 0.5,
      meanAnswerReward: 0.375,
    });
  });

  it("normalizes rewards per prompt group", () => {
    const result = computeGroupNormalizedRewards({
      rawRewards: [1, 0, 0, 1],
      groupSize: 2,
      baseline: "mean",
      advantageNormalizer: "std",
      advantageEps: 1e-6,
    });

    expect(result.advantages).toEqual([
      expect.closeTo(0.999998, 6),
      expect.closeTo(-0.999998, 6),
      expect.closeTo(-0.999998, 6),
      expect.closeTo(0.999998, 6),
    ]);
    expect(result.metadata).toMatchObject({
      meanReward: 0.5,
      meanAdvantage: 0,
      groupSize: 2,
    });
  });
});
