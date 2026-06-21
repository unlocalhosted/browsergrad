import { describe, expect, it } from "vitest";
import * as scaling from "../src/index";

describe("@unlocalhosted/browsergrad-scaling public surface", () => {
  it("exports browser-safe A3 scaling primitives", () => {
    expect(Object.keys(scaling).sort()).toEqual([
      "ScalingApiError",
      "createHostedScalingApiMock",
      "fitPowerLawScalingLaw",
      "selectExperimentsForDispatch",
    ]);
  });
});
