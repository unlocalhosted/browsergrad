import { describe, expect, it } from "vitest";
import {
  CudaLiteCompilerError,
  compileCudaLiteKernel,
  parseCudaLite,
} from "../../src/index.js";

function captureCompilerError(run: () => unknown): CudaLiteCompilerError {
  try {
    run();
  } catch (error) {
    if (error instanceof CudaLiteCompilerError) return error;
    throw error;
  }
  throw new Error("Expected CUDA-lite compiler error");
}

describe("CUDA-lite compiler diagnostics", () => {
  it("renders malformed source with the parser diagnostic snippet and caret", () => {
    const source = [
      "__global__ void malformed(float* out) {",
      "  out[0] = ;",
      "}",
    ].join("\n");

    const error = captureCompilerError(() => parseCudaLite(source));

    expect(error.source).toBe(source);
    expect(error.diagnostics).toEqual([expect.objectContaining({
      code: "parse-error",
      severity: "error",
      message: "expected expression",
      span: expect.objectContaining({ line: 2, column: 12 }),
    })]);
    expect(error.message).toContain("ERROR parse-error 2:12 expected expression");
    expect(error.message).toContain("  out[0] = ;\n           ^");
  });

  it("keeps analyzer diagnostics stable while rendering the original source", () => {
    const source = [
      "__global__ void const_write(const float* input) {",
      "  input[0] = 1.0f;",
      "}",
    ].join("\n");

    const error = captureCompilerError(() => compileCudaLiteKernel(source));

    expect(error.source).toBe(source);
    expect(error.diagnostics).toContainEqual(expect.objectContaining({
      code: "const-pointer-write",
      severity: "error",
      message: "cannot write through const pointer 'input'",
      span: expect.objectContaining({ line: 2, column: 3 }),
    }));
    expect(error.message).toContain("CUDA-lite compile failed");
    expect(error.message).toContain("ERROR const-pointer-write 2:3 cannot write through const pointer 'input'");
    expect(error.message).toContain("  input[0] = 1.0f;\n  ^^^^^");
  });
});
