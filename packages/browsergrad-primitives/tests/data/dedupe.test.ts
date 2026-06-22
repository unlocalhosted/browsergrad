import { describe, expect, it } from "vitest";
import { exactLineDeduplicate } from "../../src/data";

describe("exactLineDeduplicate", () => {
  it("keeps first occurrences and reports duplicate source indexes", () => {
    const result = exactLineDeduplicate(
      [" Alpha  beta ", "gamma", "alpha beta", "", "GAMMA"],
      { trim: true, collapseWhitespace: true, caseSensitive: false },
    );

    expect(result.keptLines).toEqual([" Alpha  beta ", "gamma", ""]);
    expect(result.duplicates).toEqual([
      {
        line: "alpha beta",
        index: 2,
        firstIndex: 0,
        key: "alpha beta",
      },
      {
        line: "GAMMA",
        index: 4,
        firstIndex: 1,
        key: "gamma",
      },
    ]);
  });
});
