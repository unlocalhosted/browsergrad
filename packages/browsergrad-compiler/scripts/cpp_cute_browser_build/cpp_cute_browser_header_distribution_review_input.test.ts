import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderDistributionReviewInputError,
  canonicalCppCuteBrowserHeaderDistributionReviewInputBytes,
  materializeCppCuteBrowserHeaderDistributionReviewInput,
  requireCppCuteBrowserHeaderDistributionReviewInputAuthority,
} from "./cpp_cute_browser_header_distribution_review_input.mjs";

describe("header distribution review input", () => {
  it("rejects forged or copied authority projections before filesystem effects", async () => {
    const forged = Object.freeze({});
    await expect(materializeCppCuteBrowserHeaderDistributionReviewInput({
      cudaRedistributionIndex: forged,
      extraction: forged,
      inventory: forged,
      materialization: forged,
      notices: forged,
    } as never)).rejects.toBeInstanceOf(CppCuteBrowserHeaderDistributionReviewInputError);
    expect(() => requireCppCuteBrowserHeaderDistributionReviewInputAuthority(forged as never))
      .toThrow(CppCuteBrowserHeaderDistributionReviewInputError);
    expect(() => canonicalCppCuteBrowserHeaderDistributionReviewInputBytes(forged as never))
      .toThrow(CppCuteBrowserHeaderDistributionReviewInputError);
  });
});
