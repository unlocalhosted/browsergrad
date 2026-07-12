import { describe, expect, it } from "vitest";
import {
  analyzeCudaLite,
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  parseCudaLite,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
} from "../../src/index";

describe("CUDA-lite compiler: Pointer rebasing", () => {
  it("supports mutable CUDA pointer rebasing", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void pointer_rebase(uint* x, uint* out, int offset) {
    x += offset;
    if (threadIdx.x == 0) {
      out[0] = x[0];
      x -= 1;
      out[1] = x[0];
      x++;
      out[2] = *x;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Uint32Array([10, 20, 30, 40]), out: new Uint32Array(3) }, scalars: { offset: 2 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Uint32Array]).toEqual([30, 20, 30]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var x__bg_ptr_offset: i32 = 0;");
      expect(compiled.wgsl).toContain("x__bg_ptr_offset = (x__bg_ptr_offset + bg_uniforms.offset);");
      expect(compiled.wgsl).toContain("x[u32((x__bg_ptr_offset + 0))]");

      const constPointee = compileCudaLiteKernel(`
  __global__ void const_pointer_rebase(const uint* x, uint* out, int offset) {
    x += offset;
    if (threadIdx.x == 0) out[0] = *x;
  }`, { workgroupSize: [1, 1, 1] });
      const constPointeeResult = runCompiledKernelReference(
        constPointee,
        { buffers: { x: new Uint32Array([10, 20, 30]), out: new Uint32Array(1) }, scalars: { offset: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...constPointeeResult.buffers.out as Uint32Array]).toEqual([20]);

      const assignmentRebase = compileCudaLiteKernel(`
  __global__ void assign_pointer_rebase(uint* x, uint* out, int offset) {
    x = &x[offset];
    out[0] = x[0];
    x = x + 1;
    out[1] = *x;
  }`, { workgroupSize: [1, 1, 1] });
      const assignmentRebaseResult = runCompiledKernelReference(
        assignmentRebase,
        { buffers: { x: new Uint32Array([10, 20, 30]), out: new Uint32Array(2) }, scalars: { offset: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...assignmentRebaseResult.buffers.out as Uint32Array]).toEqual([20, 30]);
      expect(assignmentRebase.wgsl).toContain("x__bg_ptr_offset = (x__bg_ptr_offset + bg_uniforms.offset);");
      expect(assignmentRebase.wgsl).toContain("x__bg_ptr_offset = (x__bg_ptr_offset + 1);");

      const nullGuard = compileCudaLiteKernel(`
  __global__ void pointer_null_guard(const uint* x, uint* out) {
    if (x != NULL) out[0] = x[0];
    if (x == nullptr) out[1] = 99u;
    if (x) out[2] = x[0];
  }`, { workgroupSize: [1, 1, 1] });
      const nullGuardResult = runCompiledKernelReference(
        nullGuard,
        { buffers: { x: new Uint32Array([42]), out: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const zeroPointeeResult = runCompiledKernelSemanticReference(
        nullGuard,
        { buffers: { x: new Uint32Array([0]), out: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...nullGuardResult.buffers.out as Uint32Array]).toEqual([42, 0, 42]);
      expect([...zeroPointeeResult.buffers.out as Uint32Array]).toEqual([0, 0, 0]);
      expect(canEmitSemanticKernelIrWgsl(nullGuard.kernelIr)).toBe(true);
      expect(nullGuard.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(nullGuard.wgsl).not.toContain("x[0u] != 0u");

      const pointerIdentity = compileCudaLiteKernel(`
  __global__ void pointer_identity(uint* x, uint* y, uint* out) {
    if (x != y) out[0] = 1u;
    if (x == x) out[1] = 2u;
  }`, { workgroupSize: [1, 1, 1] });
      const pointerIdentityResult = runCompiledKernelReference(
        pointerIdentity,
        { buffers: { x: new Uint32Array([1]), y: new Uint32Array([1]), out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...pointerIdentityResult.buffers.out as Uint32Array]).toEqual([1, 2]);
      expect(canEmitSemanticKernelIrWgsl(pointerIdentity.kernelIr)).toBe(true);
      expect(pointerIdentity.wgsl).toContain("!((0u) == (1u) && (0u) == (0u))");
      expect(pointerIdentity.wgsl).toContain("out[1u] = 2u;");

      const helperPointerIdentity = compileCudaLiteKernel(`
  __device__ int same_pointer(uint* left, uint* right) { return left == right; }
  __global__ void helper_pointer_identity(uint* x, uint* y, int* out) {
    out[0] = same_pointer(x, x);
    out[1] = same_pointer(x, x + 1);
    out[2] = same_pointer(x, y);
  }`, { workgroupSize: [1, 1, 1] });
      const helperPointerIdentityResult = runCompiledKernelSemanticReference(
        helperPointerIdentity,
        { buffers: { x: new Uint32Array(2), y: new Uint32Array(2), out: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(canRunCompiledKernelSemanticReference(helperPointerIdentity)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(helperPointerIdentity.kernelIr)).toBe(true);
      expect([...helperPointerIdentityResult.buffers.out as Int32Array]).toEqual([1, 0, 0]);
      expect(helperPointerIdentity.wgsl).toContain("left_buffer");
      expect(helperPointerIdentity.wgsl).toContain("left_base");

      const pointerDistance = compileCudaLiteKernel(`
  __global__ void pointer_distance(uint* data, int* out, int left) {
    uint* lptr = data + left;
    uint* rptr = &data[3];
    int nright = rptr - data;
    int width = rptr - lptr;
    out[0] = nright;
    out[1] = width;
  }`, { workgroupSize: [1, 1, 1] });
      const pointerDistanceResult = runCompiledKernelReference(
        pointerDistance,
        { buffers: { data: new Uint32Array([10, 20, 30, 40]), out: new Int32Array(2) }, scalars: { left: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...pointerDistanceResult.buffers.out as Int32Array]).toEqual([3, 2]);
      expect(pointerDistance.wgsl).toContain("i32(");
      expect(pointerDistance.wgsl).toContain("var width: i32 = (3 - bg_uniforms.left);");

      const sharedScalarDistance = compileCudaLiteKernel(`
  __global__ void shared_scalar_distance(uint* blocks, uint* out) {
    __shared__ uint start;
    __shared__ uint end;
    __shared__ uint active;
    if (threadIdx.x == 0) {
      start = blocks[0];
      end = blocks[1];
      active = end - start;
      out[0] = active;
    }
  }`, { workgroupSize: [1, 1, 1] });
      expect(sharedScalarDistance.wgsl).toContain("bg_active = (end - start);");
      expect(sharedScalarDistance.wgsl).not.toContain("select(0, (i32(end) - i32(start))");

      const mismatchedPointerDistance = analyzeCudaLite(parseCudaLite(`
  __global__ void bad_pointer_distance(uint* a, float* b, int* out) {
    out[0] = a - b;
  }`));
      expect(mismatchedPointerDistance.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-pointer-difference");

      const constWrite = analyzeCudaLite(parseCudaLite(`
  __global__ void bad_const_write(const uint* x) {
    if (threadIdx.x == 0) x[0] = 1u;
  }`));
      expect(constWrite.diagnostics.map((diagnostic) => diagnostic.code)).toContain("const-pointer-write");
    });
});
