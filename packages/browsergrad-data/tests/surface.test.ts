import { describe, expect, it } from "vitest";
import {
  exactLineDeduplicate,
  extractVisibleTextFromHtml,
  maskPii,
  type DeduplicationResult,
  type PiiMaskResult,
} from "../src/index";

describe("public surface", () => {
  it("exports data oracle primitives", () => {
    expect(typeof maskPii).toBe("function");
    expect(typeof exactLineDeduplicate).toBe("function");
    expect(typeof extractVisibleTextFromHtml).toBe("function");
  });

  it("types PII and dedupe results for compile-time consumers", () => {
    const pii: PiiMaskResult = maskPii("a@b.com");
    const dedupe: DeduplicationResult = exactLineDeduplicate(["a"]);
    expect(pii.spans[0]?.kind).toBe("email");
    expect(dedupe.keptLines).toEqual(["a"]);
  });
});
