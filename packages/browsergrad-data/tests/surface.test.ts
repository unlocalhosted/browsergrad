import { describe, expect, it } from "vitest";
import {
  evaluateGopherQuality,
  exactLineDeduplicate,
  extractVisibleTextFromHtml,
  gopherQualityFilter,
  maskPii,
  minhashDeduplicateDocuments,
  type DeduplicationResult,
  type GopherQualityReport,
  type MinHashDeduplicationResult,
  type PiiMaskResult,
} from "../src/index";

describe("public surface", () => {
  it("exports data oracle primitives", () => {
    expect(typeof maskPii).toBe("function");
    expect(typeof exactLineDeduplicate).toBe("function");
    expect(typeof minhashDeduplicateDocuments).toBe("function");
    expect(typeof extractVisibleTextFromHtml).toBe("function");
    expect(typeof evaluateGopherQuality).toBe("function");
    expect(typeof gopherQualityFilter).toBe("function");
  });

  it("types PII and dedupe results for compile-time consumers", () => {
    const pii: PiiMaskResult = maskPii("a@b.com");
    const dedupe: DeduplicationResult = exactLineDeduplicate(["a"]);
    const minhash: MinHashDeduplicationResult = minhashDeduplicateDocuments([
      { id: "a", text: "alpha beta gamma delta epsilon" },
    ]);
    const quality: GopherQualityReport = evaluateGopherQuality("be ".repeat(50));
    expect(pii.spans[0]?.kind).toBe("email");
    expect(dedupe.keptLines).toEqual(["a"]);
    expect(minhash.keptDocuments[0]?.id).toBe("a");
    expect(quality.failedRules).toContain("average_word_length");
  });
});
