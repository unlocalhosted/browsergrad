import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_REAL_COMPILE_CASES,
  CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS,
  CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS,
  cppCuteBrowserRealCompileCase,
} from "../../src/cpp_cute_browser_real_compile_cases.js";

describe("package-owned real browser compile cases", () => {
  it("pins one immutable ordered source and layout matrix", async () => {
    expect(CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS).toEqual([
      "rank2",
      "rank3",
      "rank1",
      "rank4",
      "strided-slice",
      "broadcast",
      "i32-rank2",
      "u32-broadcast",
      "signed-rank2",
    ]);
    expect(CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS).toEqual(
      CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS.slice(0, -1),
    );
    expect(Object.isFrozen(
      CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS,
    )).toBe(true);
    expect(Object.isFrozen(CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS)).toBe(true);
    expect(Object.isFrozen(CPP_CUTE_BROWSER_REAL_COMPILE_CASES)).toBe(true);

    const encoder = new TextEncoder();
    for (const caseId of CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS) {
      const compileCase = cppCuteBrowserRealCompileCase(caseId);
      expect(compileCase).toBe(CPP_CUTE_BROWSER_REAL_COMPILE_CASES[caseId]);
      expect(Object.isFrozen(compileCase)).toBe(true);
      expect(Object.isFrozen(compileCase.sourceLayout)).toBe(true);
      expect(Object.isFrozen(compileCase.sourceLayout.shape)).toBe(true);
      expect(Object.isFrozen(compileCase.sourceLayout.strides)).toBe(true);
      expect(Object.isFrozen(compileCase.destinationLayout)).toBe(true);
      expect(Object.isFrozen(compileCase.destinationLayout.shape)).toBe(true);
      expect(Object.isFrozen(compileCase.destinationLayout.strides)).toBe(true);
      expect(compileCase.source.endsWith("\n")).toBe(true);
      expect(await sha256Hex(encoder.encode(compileCase.source))).toBe(
        compileCase.sourceSha256,
      );
      expect(compileCase.sourceLayout.shape).toHaveLength(
        compileCase.coordinateRank,
      );
      expect(compileCase.sourceLayout.strides).toHaveLength(
        compileCase.coordinateRank,
      );
      expect(compileCase.destinationLayout.shape).toHaveLength(
        compileCase.coordinateRank,
      );
      expect(compileCase.destinationLayout.strides).toHaveLength(
        compileCase.coordinateRank,
      );
    }
  });

  it("covers the promoted word32 dtype and rank profile exactly", () => {
    const cases = CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS.map(
      (caseId) => cppCuteBrowserRealCompileCase(caseId),
    );
    expect(new Set(cases.map((entry) => entry.dtype))).toEqual(
      new Set(["f32", "i32", "u32"]),
    );
    expect(new Set(cases.map((entry) => entry.coordinateRank))).toEqual(
      new Set([1, 2, 3, 4]),
    );
    expect(cases.filter((entry) => entry.dtype === "i32")).toHaveLength(1);
    expect(cases.filter((entry) => entry.dtype === "u32")).toHaveLength(1);
  });

  it("preserves an unchanged signed-stride CuTe source case", () => {
    const compileCase = cppCuteBrowserRealCompileCase("signed-rank2");
    expect(compileCase.source).toContain(
      "cute::Stride<cute::Int<-3>, cute::Int<1>>",
    );
    expect(compileCase.sourceLayout).toEqual({
      shape: ["2", "3"],
      strides: ["-3", "1"],
    });
    expect(compileCase.destinationLayout).toEqual({
      shape: ["2", "3"],
      strides: ["3", "1"],
    });
    expect(compileCase.sourceSpanElements).toBe(6n);
    expect(compileCase.destinationSpanElements).toBe(6n);
  });
});
