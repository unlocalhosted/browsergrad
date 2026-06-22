import { describe, expect, it } from "vitest";
import {
  createWgslFloat16Array,
  defineWgslKernelProgram,
  detectKernelFeatures,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  getWgslFloat16ArrayConstructor,
  installWgslFloat16ArrayPolyfill,
} from "../src/index";

describe("generic WGSL kernel programs", () => {
  it("normalizes binding indices and storage access", () => {
    const program = defineWgslKernelProgram({
      name: "copy_kernel",
      wgsl: "@compute @workgroup_size(1) fn main() {}",
      workgroupSize: [1, 1, 1],
      bindings: [
        { kind: "storage", name: "input", valueType: "f32", access: "read" },
        { kind: "storage", name: "output", valueType: "f32" },
        { kind: "uniform", name: "params", byteLength: 16 },
      ],
    });

    expect(program.bindings.map((binding) => binding.binding)).toEqual([0, 1, 2]);
    expect(program.bindings[1]).toMatchObject({ kind: "storage", access: "read_write" });
  });

  it("supports f16 storage bindings explicitly", () => {
    const program = defineWgslKernelProgram({
      name: "half_kernel",
      wgsl: "enable f16;\n@compute @workgroup_size(1) fn main() {}",
      workgroupSize: [1, 1, 1],
      bindings: [{ kind: "storage", name: "x", valueType: "f16" }],
    });

    expect(program.bindings[0]).toMatchObject({ kind: "storage", valueType: "f16" });
  });

  it("provides a float16 backing array when the JS runtime lacks a native one", () => {
    const half = createWgslFloat16Array([1.5, 3]);
    const fromBytes = createWgslFloat16Array(half.buffer);

    expect(half instanceof getWgslFloat16ArrayConstructor()).toBe(true);
    expect(half.BYTES_PER_ELEMENT).toBe(2);
    expect(half.byteLength).toBe(4);
    expect(float32ToFloat16Bits(1.5)).toBe(0x3e00);
    expect(float16BitsToFloat32(0x3e00)).toBe(1.5);
    expect([...fromBytes]).toEqual([1.5, 3]);
  });

  it("rounds float32 to float16 bits with IEEE edge behavior", () => {
    expect(float32ToFloat16Bits(0)).toBe(0x0000);
    expect(float32ToFloat16Bits(-0)).toBe(0x8000);
    expect(float32ToFloat16Bits(1)).toBe(0x3c00);
    expect(float32ToFloat16Bits(65504)).toBe(0x7bff);
    expect(float32ToFloat16Bits(Infinity)).toBe(0x7c00);
    expect(float32ToFloat16Bits(-Infinity)).toBe(0xfc00);
    expect(float32ToFloat16Bits(2 ** -24)).toBe(0x0001);
    expect(float32ToFloat16Bits(2 ** -25)).toBe(0x0000);
    expect(float32ToFloat16Bits(2 ** -25 + 2 ** -35)).toBe(0x0001);
    expect(float32ToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(float32ToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
    expect(Number.isNaN(float16BitsToFloat32(float32ToFloat16Bits(NaN)))).toBe(true);
    expect(Object.is(float16BitsToFloat32(0x8000), -0)).toBe(true);
  });

  it("keeps global Float16Array install explicit", () => {
    const target = globalThis as typeof globalThis & { Float16Array?: unknown };
    const original = Object.getOwnPropertyDescriptor(target, "Float16Array");
    if (original && !original.configurable) return;

    try {
      Reflect.deleteProperty(target, "Float16Array");
      expect(target.Float16Array).toBeUndefined();
      expect([...createWgslFloat16Array([2])]).toEqual([2]);
      expect(target.Float16Array).toBeUndefined();

      const Constructor = installWgslFloat16ArrayPolyfill();
      expect(target.Float16Array).toBe(Constructor);
      expect([...new Constructor([3])]).toEqual([3]);
    } finally {
      if (original) Object.defineProperty(target, "Float16Array", original);
      else Reflect.deleteProperty(target, "Float16Array");
    }
  });

  it("supports texture2d bindings explicitly", () => {
    const program = defineWgslKernelProgram({
      name: "texture_kernel",
      wgsl: "@group(0) @binding(0) var image: texture_2d<f32>;\n@compute @workgroup_size(1) fn main() {}",
      workgroupSize: [1, 1, 1],
      bindings: [{ kind: "texture2d", name: "image", valueType: "f32" }],
    });

    expect(program.bindings[0]).toMatchObject({ kind: "texture2d", valueType: "f32" });
  });

  it("allows compute programs with no bindings", () => {
    const program = defineWgslKernelProgram({
      name: "control_only",
      wgsl: "@compute @workgroup_size(1) fn main() { return; }",
      workgroupSize: [1, 1, 1],
      bindings: [],
    });

    expect(program.bindings).toEqual([]);
  });

  it("rejects duplicate names and invalid workgroups", () => {
    expect(() =>
      defineWgslKernelProgram({
        name: "bad",
        wgsl: "@compute @workgroup_size(1) fn main() {}",
        workgroupSize: [0, 1, 1],
        bindings: [{ kind: "storage", name: "x", valueType: "f32" }],
      }),
    ).toThrow(/workgroupSize/);

    expect(() =>
      defineWgslKernelProgram({
        name: "bad",
        wgsl: "@compute @workgroup_size(1) fn main() {}",
        workgroupSize: [1, 1, 1],
        bindings: [
          { kind: "storage", name: "x", valueType: "f32" },
          { kind: "storage", name: "x", valueType: "f32" },
        ],
      }),
    ).toThrow(/duplicate WGSL binding name/);
  });

  it("detects adapter/device features from a provided feature set", async () => {
    const features = new Set(["shader-f16", "subgroups"]) as unknown as GPUSupportedFeatures;
    const result = await detectKernelFeatures({ features } as GPUAdapter);

    expect(result.webgpu).toBe(true);
    expect(result.shaderF16).toBe(true);
    expect(result.float16Array).toBe(true);
    expect(result.subgroups).toBe(true);
    expect(result.features).toEqual(["shader-f16", "subgroups"]);
  });
});
