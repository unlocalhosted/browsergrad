import { describe, expect, it } from "vitest";
import * as alignment from "../src/index";
import type {
  DpoLossInput,
  GroupNormalizedRewardsResult,
  PolicyGradientLossResult,
  RolloutRewardsResult,
} from "../src/index";

describe("@unlocalhosted/browsergrad-alignment public surface", () => {
  it("exports browser-safe A5 alignment primitives", () => {
    expect(Object.keys(alignment).sort()).toEqual([
      "aggregateLossAcrossMicrobatch",
      "computeGroupNormalizedRewards",
      "computePerInstanceDpoLoss",
      "computePolicyGradientLoss",
      "computeRolloutRewards",
      "parseGsm8kResponse",
      "parseMmluResponse",
    ]);
  });

  it("types alignment result objects for compile-time consumers", () => {
    const dpo: DpoLossInput = {
      beta: 0.1,
      policyChosenLogProbability: -1,
      policyRejectedLogProbability: -2,
      referenceChosenLogProbability: -1.5,
      referenceRejectedLogProbability: -2.5,
    };
    const rollout: RolloutRewardsResult = alignment.computeRolloutRewards({
      rolloutResponses: ["ok"],
      repeatedGroundTruths: ["ok"],
      rewardFn: () => ({ reward: 1, formatReward: 1, answerReward: 1 }),
    });
    const normalized: GroupNormalizedRewardsResult =
      alignment.computeGroupNormalizedRewards({
        rawRewards: [1, 0],
        groupSize: 2,
      });
    const policy: PolicyGradientLossResult = alignment.computePolicyGradientLoss({
      advantages: [1],
      policyLogProbs: [[-1]],
    });

    expect(alignment.computePerInstanceDpoLoss(dpo)).toBeGreaterThan(0);
    expect(rollout.rawRewards).toEqual([1]);
    expect(normalized.advantages).toHaveLength(2);
    expect(policy.perTokenLoss).toEqual([[1]]);
  });
});
