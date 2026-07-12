import { describe, expect, it } from "vitest";
import {
  CudaLiteCompilerError,
  validateSemanticKernelIr,
  compileCudaLiteKernel,
  verifySemanticKernelIr,
  type SemanticKernelIrModule,
} from "../../src/index";

describe("semantic Kernel IR verifier", () => {
  it("accepts compiler-produced IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void copy_value(const float* input, float* output) {
  int index = threadIdx.x;
  output[index] = input[index];
}`, { workgroupSize: [2, 1, 1] });

    expect(verifySemanticKernelIr(compiled.kernelIr)).toEqual([]);
    expect(() => validateSemanticKernelIr(compiled.kernelIr)).not.toThrow();
  });

  it("rejects malformed operations with source-spanned internal diagnostics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void empty_kernel() {}
`);
    const invalid = {
      ...compiled.kernelIr,
      operations: [{
        kind: "break",
        span: compiled.kernelIr.span,
      }],
    } as SemanticKernelIrModule;

    const issues = verifySemanticKernelIr(invalid);
    expect(issues).toEqual([expect.objectContaining({
      code: "internal-lowering-invariant",
      message: "IR break must be nested in a loop",
      span: compiled.kernelIr.span,
    })]);
    expect(() => validateSemanticKernelIr(invalid)).toThrowError(CudaLiteCompilerError);
    try {
      validateSemanticKernelIr(invalid);
    } catch (error) {
      expect(error).toBeInstanceOf(CudaLiteCompilerError);
      expect((error as CudaLiteCompilerError).diagnostics[0]).toMatchObject({
        code: "internal-lowering-invariant",
        span: compiled.kernelIr.span,
      });
    }
  });

  it("rejects dangling memory identities even when display names look valid", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void identity_kernel(float *out) { out[0] = 1.0f; }
`);
    const store = compiled.kernelIr.operations.find((operation) => operation.kind === "store");
    if (store?.kind !== "store") throw new Error("expected store operation");
    const invalid = {
      ...compiled.kernelIr,
      operations: [{
        ...store,
        target: {
          ...store.target,
          baseId: { key: "memory:forged:out" } as unknown as typeof store.target.baseId,
        },
      }],
    } as SemanticKernelIrModule;

    expect(verifySemanticKernelIr(invalid)).toContainEqual(expect.objectContaining({
      code: "internal-lowering-invariant",
      message: "IR store has dangling memory identity 'memory:forged:out'",
      span: store.target.span,
    }));
  });
});
