import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderNoticeMaterializationError,
  canonicalCppCuteBrowserHeaderNoticeMaterializationBytes,
  materializeCppCuteBrowserHeaderDistributionNotices,
  requireCppCuteBrowserHeaderNoticeMaterializationAuthority,
} from "./cpp_cute_browser_header_notice_materialization.mjs";

describe("header distribution notice materialization", () => {
  it("rejects forged or copied authority projections before filesystem effects", async () => {
    const forged = Object.freeze({});
    await expect(materializeCppCuteBrowserHeaderDistributionNotices({
      distributionReviewInput: forged,
      materialization: forged,
      notices: forged,
    } as never)).rejects.toBeInstanceOf(CppCuteBrowserHeaderNoticeMaterializationError);
    expect(() => requireCppCuteBrowserHeaderNoticeMaterializationAuthority(forged as never))
      .toThrow(CppCuteBrowserHeaderNoticeMaterializationError);
    expect(() => canonicalCppCuteBrowserHeaderNoticeMaterializationBytes(forged as never))
      .toThrow(CppCuteBrowserHeaderNoticeMaterializationError);
  });
});
