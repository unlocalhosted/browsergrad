import { beforeAll, describe, expect, it } from "vitest";
import {
  createDevice,
  defineWgslKernelProgram,
  runWgslKernelProgram,
} from "../src/index";

interface DeviceCheck {
  readonly available: boolean;
  readonly reason?: string;
}

async function checkDevice(): Promise<DeviceCheck> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu undefined" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: "no GPU adapter" };
    return { available: true };
  } catch (error) {
    return {
      available: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

describe("real WebGPU — generic WGSL program runner", () => {
  let deviceCheck: DeviceCheck;

  beforeAll(async () => {
    deviceCheck = await checkDevice();
    if (!deviceCheck.available) {
      console.warn(`[skip] WebGPU not available: ${deviceCheck.reason}`);
    }
  });

  it("runs typed storage buffers and uniform params", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const params = new Float32Array([2]);
    const program = defineWgslKernelProgram({
      name: "generic_saxpy",
      workgroupSize: [8, 1, 1],
      bindings: [
        { kind: "storage", name: "x", valueType: "f32", access: "read", binding: 0 },
        { kind: "storage", name: "y", valueType: "f32", access: "read_write", binding: 1 },
        { kind: "uniform", name: "params", byteLength: 16, binding: 2 },
      ],
      wgsl: `
struct Params { a: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&y)) {
    y[i] = params.a * x[i] + y[i];
  }
}`,
    });

    const result = await runWgslKernelProgram(
      device,
      program,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array([10, 20, 30, 40]),
        },
        uniforms: { params },
      },
      { dispatchCount: [4, 1, 1] },
    );

    expect([...result.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);
  });
});
