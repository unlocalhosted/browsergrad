import { describe, expect, it } from "vitest";
import { computePerInstanceDpoLoss } from "../src/index";

describe("computePerInstanceDpoLoss", () => {
  it("computes the DPO logistic loss from policy and reference log-prob sums", () => {
    const loss = computePerInstanceDpoLoss({
      beta: 0.5,
      policyChosenLogProbability: -3.2,
      policyRejectedLogProbability: -4.1,
      referenceChosenLogProbability: -3.7,
      referenceRejectedLogProbability: -4.0,
    });

    expect(loss).toBeCloseTo(0.554355, 6);
  });
});
