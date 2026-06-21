import { describe, expect, it } from "vitest";
import { createKernelRubric, tensor } from "../src/index";

describe("KernelRubric", () => {
  it("records a pass when tensors match within tolerance", () => {
    const passed: string[] = [];
    const rubric = createKernelRubric({
      assertPass: (name) => passed.push(name),
    });

    rubric.assertCloseTensor(
      "matmul_tiny",
      tensor([2], new Float32Array([1, 2.00001])),
      tensor([2], new Float32Array([1, 2])),
      { atol: 1e-4 },
    );

    expect(passed).toEqual(["matmul_tiny"]);
    expect(rubric.assertions).toEqual([
      {
        kind: "pass",
        name: "matmul_tiny",
      },
    ]);
  });

  it("records shape mismatches with compact previews", () => {
    const failures: Array<{ name: string; message: string; details?: unknown }> = [];
    const rubric = createKernelRubric({
      assertFail: (name, message, details) => {
        failures.push({ name, message, details });
      },
    });

    rubric.assertCloseTensor(
      "softmax_shape",
      tensor([3], new Float32Array([1, 2, 3])),
      tensor([1, 3], new Float32Array([1, 2, 3])),
    );

    expect(rubric.assertions).toEqual([
      {
        kind: "fail",
        name: "softmax_shape",
        message: "tensor shape mismatch",
        details: {
          actualShape: [3],
          expectedShape: [1, 3],
          actualPreview: [1, 2, 3],
          expectedPreview: [1, 2, 3],
        },
      },
    ]);
    expect(failures).toEqual([
      {
        name: "softmax_shape",
        message: "tensor shape mismatch",
        details: {
          actualShape: [3],
          expectedShape: [1, 3],
          actualPreview: [1, 2, 3],
          expectedPreview: [1, 2, 3],
        },
      },
    ]);
  });

  it("records numerical mismatches with first failing index and max error", () => {
    const rubric = createKernelRubric();

    rubric.assertCloseTensor(
      "gelu_values",
      tensor([4], new Float32Array([0, 1, 2.25, 4])),
      tensor([4], new Float32Array([0, 1, 2, 4])),
      { atol: 1e-3, maxPreview: 3 },
    );

    expect(rubric.assertions).toEqual([
      {
        kind: "fail",
        name: "gelu_values",
        message: "tensor values differ beyond tolerance",
        details: {
          actualShape: [4],
          expectedShape: [4],
          actualPreview: [0, 1, 2.25],
          expectedPreview: [0, 1, 2],
          mismatchIndex: 2,
          actualValue: 2.25,
          expectedValue: 2,
          maxAbsDiff: 0.25,
          atol: 0.001,
          rtol: 0.000001,
        },
      },
    ]);
  });

  it("treats non-finite actual values as mismatches", () => {
    const rubric = createKernelRubric();

    rubric.assertCloseTensor(
      "softmax_nan",
      tensor([2], new Float32Array([0.5, Number.NaN])),
      tensor([2], new Float32Array([0.5, 0.5])),
    );

    expect(rubric.assertions).toMatchObject([
      {
        kind: "fail",
        name: "softmax_nan",
        details: {
          mismatchIndex: 1,
          actualValue: Number.NaN,
          expectedValue: 0.5,
        },
      },
    ]);
  });
});
