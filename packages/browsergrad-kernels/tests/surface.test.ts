import { describe, expect, it } from "vitest";
import {
  createDevice,
  kernels,
  KernelError,
  reference,
  referenceFlashAttention,
  referenceFlashAttentionBackward,
  tensor,
  type Kernels,
  type KernelDevice,
  type Tensor,
} from "../src/index";

/**
 * Surface + argument-validation tests for the kernels package.
 * Reference-impl correctness lives in reference.test.ts.
 * WGSL conformance tests require a real WebGPU device (planned).
 */

describe("public surface", () => {
  it("exports the documented top-level names", () => {
    expect(typeof createDevice).toBe("function");
    expect(typeof tensor).toBe("function");
    expect(typeof KernelError).toBe("function");
    expect(typeof reference).toBe("object");
    expect(typeof kernels).toBe("object");
    expect(typeof referenceFlashAttention).toBe("function");
    expect(typeof referenceFlashAttentionBackward).toBe("function");
  });

  it("kernels bundle exposes the v0 op set", () => {
    const expected: (keyof Kernels)[] = [
      "matmul",
      "softmax",
      "relu",
      "gelu",
      "layernorm",
      "attention",
    ];
    for (const name of expected) {
      expect(typeof kernels[name]).toBe("function");
    }
  });

  it("reference bundle exposes the same op set", () => {
    expect(typeof reference.matmul).toBe("function");
    expect(typeof reference.softmax).toBe("function");
    expect(typeof reference.relu).toBe("function");
    expect(typeof reference.gelu).toBe("function");
    expect(typeof reference.layernorm).toBe("function");
    expect(typeof reference.attention).toBe("function");
    expect(typeof reference.flashAttention).toBe("function");
    expect(typeof reference.flashAttentionBackward).toBe("function");
  });
});

describe("tensor literal helper", () => {
  it("returns the input shape and data unchanged", () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const t = tensor([2, 2], data);
    expect(t.shape).toEqual([2, 2]);
    expect(t.data).toBe(data);
  });
});

describe("createDevice error path", () => {
  it("throws KernelError in node without navigator.gpu", async () => {
    await expect(createDevice()).rejects.toThrow(KernelError);
  });
});

describe("KernelError", () => {
  it("is a real Error subclass", () => {
    const e = new KernelError("nope");
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(KernelError);
    expect(e.name).toBe("KernelError");
  });
});

/* Type-level compatibility — uncompilable if the public types drift. */
describe("type surface (compile-time only)", () => {
  it("Tensor shape is row-major Float32Array + readonly shape", () => {
    const t: Tensor = { shape: [3, 4], data: new Float32Array(12) };
    expect(t.shape.length).toBe(2);
  });

  it("KernelDevice has gpu, getStats, clearCache", () => {
    const _check = (d: KernelDevice): void => {
      expect(typeof d.getStats).toBe("function");
      expect(typeof d.clearCache).toBe("function");
      // d.gpu is a GPUDevice — typed but not instantiable in node.
      void d;
    };
    expect(typeof _check).toBe("function");
  });
});
