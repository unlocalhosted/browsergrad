import { describe, expect, it } from "vitest";
import { evaluateGopherQuality, gopherQualityFilter } from "../src/index";

describe("gopherQualityFilter", () => {
  it("applies deterministic CS336 A4 fixture-scale quality rules", () => {
    expect(gopherQualityFilter("The string you are reading is a short snippet of text.")).toBe(
      false,
    );
    expect(gopherQualityFilter("The string you are reading is a long snippet of text. ".repeat(100))).toBe(
      true,
    );
    expect(gopherQualityFilter("the be ".repeat(100))).toBe(false);
    expect(gopherQualityFilter("the with ".repeat(100))).toBe(true);
    expect(
      gopherQualityFilter(
        [
          ...Array.from(
            { length: 70 },
            () => "The line here is an example of line ending with an ellipsis...",
          ),
          ...Array.from({ length: 30 }, () => "This is a normal line."),
        ].join("\n"),
      ),
    ).toBe(false);

    const report = evaluateGopherQuality("123 ".repeat(80) + "word ".repeat(20));
    expect(report.passed).toBe(false);
    expect(report.failedRules).toContain("alphabetic_word_ratio");
  });
});
