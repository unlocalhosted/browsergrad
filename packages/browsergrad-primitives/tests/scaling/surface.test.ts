import { describe, expect, it } from "vitest";
import * as scaling from "../../src/scaling";

describe("@unlocalhosted/browsergrad-primitives public surface", () => {
  it("exports browser-safe A3 scaling primitives", () => {
    expect(Object.keys(scaling).sort()).toEqual([
      "ScalingApiError",
      "createHostedTrainingApiFixture",
      "fitPowerLawScalingLaw",
      "selectExperimentsForDispatch",
    ]);
  });
});
