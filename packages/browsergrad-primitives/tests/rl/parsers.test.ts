import { describe, expect, it } from "vitest";
import { parseGsm8kResponse, parseMmluResponse } from "../../src/rl";

describe("alignment response parsers", () => {
  it("parses upstream-style MMLU letter answers", () => {
    const mmluExample = {
      subject: "virology",
      question: "How many human polyomaviruses are known at present?",
      options: ["100", "1", "10", "unknown"],
      answer: "A",
    };

    expect(
      parseMmluResponse(
        mmluExample,
        "The correct answer is B.\nThere is only one human polyomavirus known.",
      ),
    ).toBe("B");
    expect(parseMmluResponse(mmluExample, "The answer is 10000 viruses.")).toBeNull();
  });

  it("parses GSM8K by returning the last numeric answer", () => {
    expect(
      parseGsm8kResponse(
        "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.",
      ),
    ).toBe("72");
    expect(parseGsm8kResponse("Natalia sold seventy-two clips altogether.")).toBeNull();
  });
});
