/**
 * Public surface — what an external consumer can import from the tarball.
 * Catches: missing exports, .d.ts/.js drift, files stripped by .npmignore.
 */

import { describe, expect, it } from "vitest";
import {
  createDevice, tensor, kernels,
  matmulDirect, matmulTiledDirect, fusedElementwiseDirect,
  createWebGpuRealizerBridge,
  uploadFloat32, materializeFloat32, runDirect,
  generateFusedWgsl, KernelError,
} from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";

describe("public surface — top-level exports", () => {
  it("declared exports are all functions/objects", () => {
    expect(createDevice).toBeTypeOf("function");
    expect(tensor).toBeTypeOf("function");
    expect(kernels).toBeTypeOf("object");
    expect(reference).toBeTypeOf("object");
    expect(createWebGpuRealizerBridge).toBeTypeOf("function");
    expect(matmulDirect).toBeTypeOf("function");
    expect(matmulTiledDirect).toBeTypeOf("function");
    expect(fusedElementwiseDirect).toBeTypeOf("function");
    expect(generateFusedWgsl).toBeTypeOf("function");
    expect(uploadFloat32).toBeTypeOf("function");
    expect(materializeFloat32).toBeTypeOf("function");
    expect(runDirect).toBeTypeOf("function");
    expect(KernelError).toBeTypeOf("function");
  });

  it("kernels bundle has the v0 op set", () => {
    for (const op of ["matmul", "softmax", "relu", "gelu", "layernorm", "attention"] as const) {
      expect(kernels[op], `kernels.${op}`).toBeTypeOf("function");
    }
  });

  it("reference bundle mirrors the kernels bundle (one-to-one)", () => {
    const kKeys = Object.keys(kernels).filter((k) => typeof kernels[k as keyof typeof kernels] === "function").sort();
    const rKeys = Object.keys(reference).filter((k) => typeof reference[k as keyof typeof reference] === "function").sort();
    expect(rKeys).toEqual(expect.arrayContaining(kKeys));
  });

  it("KernelError is a real Error subclass with correct name", () => {
    const e = new KernelError("test");
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe("KernelError");
    expect(e.message).toBe("test");
  });

  it("tensor() returns an object with shape + data fields", () => {
    const t = tensor([2, 3], new Float32Array(6));
    expect(t.shape).toEqual([2, 3]);
    expect(t.data).toBeInstanceOf(Float32Array);
    expect(t.data.length).toBe(6);
  });
});
