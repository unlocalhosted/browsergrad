import { describe, expect, it } from "vitest";
import {
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

  it("rejects non-string fragment source supplied by untyped callers", () => {
    expect(() => prepareCudaLiteCompilationUnit({
      fragments: [{ source: 1 } as unknown as { source: string }],
    })).toThrow("CUDA-lite source fragment 0 must contain a string source");
  });
});
