# @unlocalhosted/browsergrad-alignment

Browser-safe alignment and RL loss oracles for BrowserGrad assignment rubrics.

This package targets fixture-scale alignment labs such as CS336 Assignment 5.
It gives rubrics deterministic math helpers before full vLLM, flash-attn,
large-model inference, or native training loops enter the path.

## Public Surface

```ts
import {
  aggregateLossAcrossMicrobatch,
  computeGroupNormalizedRewards,
  computePerInstanceDpoLoss,
  computePolicyGradientLoss,
  parseGsm8kResponse,
} from "@unlocalhosted/browsergrad-alignment";

const dpo = computePerInstanceDpoLoss({
  beta: 0.5,
  policyChosenLogProbability: -3.2,
  policyRejectedLogProbability: -4.1,
  referenceChosenLogProbability: -3.7,
  referenceRejectedLogProbability: -4.0,
});

const rewards = computeGroupNormalizedRewards({
  rawRewards: [1, 0, 0, 1],
  groupSize: 2,
});

const loss = computePolicyGradientLoss({
  advantages: rewards.advantages,
  policyLogProbs: [
    [-0.2, -0.4],
    [-1.2, -0.8],
    [-0.3, -0.5],
    [-0.7, -0.9],
  ],
});

console.log(dpo, loss.perTokenLoss);
console.log(aggregateLossAcrossMicrobatch({
  perTokenLoss: loss.perTokenLoss,
  mask: loss.perTokenLoss.map((row) => row.map(() => 1)),
}));
console.log(parseGsm8kResponse("48 + 24 = 72"));
```
