import { describe, expect, it } from "vitest";
import {
  mapCudaLiteDiagnosticToSourceProvenance,
  parseCudaLite,
  prepareCudaLiteCompilationUnit,
} from "../../src/index.js";

describe("CUDA-lite source preparation", () => {
  it("assembles caller-normalized context in order with provenance segments", () => {
    const unit = prepareCudaLiteCompilationUnit({
      appliedTransforms: [
        { name: "prune-inactive-preprocessor-branches" },
        { name: "specialize-template", detail: "saxpy<float>" },
      ],
      fragments: [
        {
          source: "#define SCALE 2",
          kind: "define",
          label: "scale",
          provenance: { sourceName: "kernels.cu", sourceOffset: 12, sourceLine: 2, sourceColumn: 1 },
        },
        {
          source: "__device__ float twice(float value) { return value * SCALE; }",
          kind: "device-function",
          provenance: { sourceName: "kernels.cu", sourceOffset: 40, sourceLine: 4, sourceColumn: 1 },
        },
        {
          source: "__global__ void scale(float *out) { out[0] = twice(3.0f); }",
          kind: "kernel",
          label: "scale",
          provenance: { sourceName: "kernels.cu", sourceOffset: 106, sourceLine: 6, sourceColumn: 1 },
        },
      ],
    });

    expect(unit.source).toBe([
      "#define SCALE 2",
      "__device__ float twice(float value) { return value * SCALE; }",
      "__global__ void scale(float *out) { out[0] = twice(3.0f); }",
    ].join("\n"));
    expect(unit.appliedTransforms).toEqual([
      { name: "prune-inactive-preprocessor-branches" },
      { name: "specialize-template", detail: "saxpy<float>" },
    ]);
    expect(unit.segments).toEqual([
      {
        fragmentIndex: 0,
        outputStart: { offset: 0, line: 1, column: 1 },
        outputEnd: { offset: 15, line: 1, column: 16 },
        kind: "define",
        label: "scale",
        provenance: { sourceName: "kernels.cu", sourceOffset: 12, sourceLine: 2, sourceColumn: 1 },
      },
      {
        fragmentIndex: 1,
        outputStart: { offset: 16, line: 2, column: 1 },
        outputEnd: { offset: 77, line: 2, column: 62 },
        kind: "device-function",
        provenance: { sourceName: "kernels.cu", sourceOffset: 40, sourceLine: 4, sourceColumn: 1 },
      },
      {
        fragmentIndex: 2,
        outputStart: { offset: 78, line: 3, column: 1 },
        outputEnd: { offset: 137, line: 3, column: 60 },
        kind: "kernel",
        label: "scale",
        provenance: { sourceName: "kernels.cu", sourceOffset: 106, sourceLine: 6, sourceColumn: 1 },
      },
    ]);

    expect(parseCudaLite(unit.source).kernels[0]?.name).toBe("scale");
  });

  it("preserves explicit separators and tracks CRLF source positions", () => {
    const unit = prepareCudaLiteCompilationUnit({
      separator: "\n// context boundary\n",
      fragments: [
        { source: "first\r\nline", kind: "context" },
        { source: "last", kind: "kernel" },
      ],
    });

    expect(unit.source).toBe("first\r\nline\n// context boundary\nlast");
    expect(unit.segments).toEqual([
      {
        fragmentIndex: 0,
        outputStart: { offset: 0, line: 1, column: 1 },
        outputEnd: { offset: 11, line: 2, column: 5 },
        kind: "context",
      },
      {
        fragmentIndex: 1,
        outputStart: { offset: 32, line: 4, column: 1 },
        outputEnd: { offset: 36, line: 4, column: 5 },
        kind: "kernel",
      },
    ]);
  });

  it("maps a diagnostic inside a CRLF fragment back to caller provenance", () => {
    const unit = prepareCudaLiteCompilationUnit({
      fragments: [
        { source: "#define SCALE 2", kind: "define" },
        {
          source: "__global__ void scale() {\r\n  nope;\r\n}",
          kind: "kernel",
          label: "scale",
          provenance: { sourceName: "kernels.cu", sourceOffset: 200, sourceLine: 40, sourceColumn: 1 },
        },
      ],
    });
    const start = unit.source.indexOf("nope");

    expect(mapCudaLiteDiagnosticToSourceProvenance(unit, {
      code: "unknown-symbol",
      severity: "error",
      message: "unknown CUDA-lite symbol 'nope'",
      span: { start, end: start + 4, line: 3, column: 3 },
    })).toEqual([
      {
        fragmentIndex: 1,
        outputStart: { offset: start, line: 3, column: 3 },
        outputEnd: { offset: start + 4, line: 3, column: 7 },
        kind: "kernel",
        label: "scale",
        sourceStart: { sourceName: "kernels.cu", offset: 229, line: 41, column: 3 },
        sourceEnd: { sourceName: "kernels.cu", offset: 233, line: 41, column: 7 },
      },
    ]);
  });

  it("splits a cross-fragment diagnostic and leaves separator-only spans unmapped", () => {
    const adjacent = prepareCudaLiteCompilationUnit({
      separator: "",
      fragments: [
        { source: "abc", provenance: { sourceName: "first.cu", sourceOffset: 10, sourceLine: 1, sourceColumn: 1 } },
        { source: "def", provenance: { sourceName: "second.cu", sourceOffset: 20, sourceLine: 7, sourceColumn: 4 } },
      ],
    });

    expect(mapCudaLiteDiagnosticToSourceProvenance(adjacent, {
      code: "parse-error",
      severity: "error",
      message: "cross-fragment issue",
      span: { start: 2, end: 4, line: 1, column: 3 },
    })).toEqual([
      {
        fragmentIndex: 0,
        outputStart: { offset: 2, line: 1, column: 3 },
        outputEnd: { offset: 3, line: 1, column: 4 },
        sourceStart: { sourceName: "first.cu", offset: 12, line: 1, column: 3 },
        sourceEnd: { sourceName: "first.cu", offset: 13, line: 1, column: 4 },
      },
      {
        fragmentIndex: 1,
        outputStart: { offset: 3, line: 1, column: 4 },
        outputEnd: { offset: 4, line: 1, column: 5 },
        sourceStart: { sourceName: "second.cu", offset: 20, line: 7, column: 4 },
        sourceEnd: { sourceName: "second.cu", offset: 21, line: 7, column: 5 },
      },
    ]);

    const separated = prepareCudaLiteCompilationUnit({
      fragments: [{ source: "abc" }, { source: "def" }],
    });
    expect(mapCudaLiteDiagnosticToSourceProvenance(separated, {
      code: "parse-error",
      severity: "error",
      message: "separator issue",
      span: { start: 3, end: 4, line: 1, column: 4 },
    })).toEqual([]);
  });

  it("rejects non-string fragment source supplied by untyped callers", () => {
    expect(() => prepareCudaLiteCompilationUnit({
      fragments: [{ source: 1 } as unknown as { source: string }],
    })).toThrow("CUDA-lite source fragment 0 must contain a string source");
  });
});
