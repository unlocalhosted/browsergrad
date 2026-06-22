import { describe, expect, it } from "vitest";
import {
  defineWgslKernelProgram,
  detectKernelFeatures,
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
    expect(result.float16Array).toBe(typeof Float16Array !== "undefined");
    expect(result.subgroups).toBe(true);
    expect(result.features).toEqual(["shader-f16", "subgroups"]);
  });
});
