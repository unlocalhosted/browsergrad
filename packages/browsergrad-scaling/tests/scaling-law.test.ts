import { describe, expect, it } from "vitest";
import { fitPowerLawScalingLaw } from "../src/index";

describe("fitPowerLawScalingLaw", () => {
  it("fits an exact power law in log-space and predicts held-out compute", () => {
    const fit = fitPowerLawScalingLaw(
      [
        { compute: 1, loss: 4 },
        { compute: 4, loss: 2 },
        { compute: 16, loss: 1 },
      ],
      { x: "compute", y: "loss" },
    );

    expect(fit.slope).toBeCloseTo(-0.5, 12);
    expect(fit.intercept).toBeCloseTo(Math.log(4), 12);
    expect(fit.multiplier).toBeCloseTo(4, 12);
    expect(fit.exponent).toBeCloseTo(-0.5, 12);
    expect(fit.rSquared).toBeCloseTo(1, 12);
    expect(fit.predict(64)).toBeCloseTo(0.5, 12);
  });

  it("rejects non-positive samples because log-space fitting would lie", () => {
    expect(() => {
      fitPowerLawScalingLaw(
        [
          { compute: 1, loss: 4 },
          { compute: 0, loss: 2 },
        ],
        { x: "compute", y: "loss" },
      );
    }).toThrow("samples[1].compute must be positive");
  });
});
