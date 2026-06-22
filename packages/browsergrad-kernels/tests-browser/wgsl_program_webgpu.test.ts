import { beforeAll, describe, expect, it } from "vitest";
import {
  createDevice,
  createWgslStorageBuffer,
  defineWgslKernelProgram,
  destroyWgslStorageBuffer,
  prepareWgslKernelProgramSequence,
  readWgslStorageBuffer,
  runWgslKernelProgram,
  runWgslKernelProgramSequence,
  writeWgslStorageBuffer,
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

  it("runs f32 texture2d bindings", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const program = defineWgslKernelProgram({
      name: "texture_copy",
      workgroupSize: [2, 2, 1],
      bindings: [
        { kind: "texture2d", name: "image", valueType: "f32", binding: 0 },
        { kind: "storage", name: "out", valueType: "f32", access: "read_write", binding: 1 },
      ],
      wgsl: `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(2, 2, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < 2u && gid.y < 2u) {
    let i = gid.y * 2u + gid.x;
    out[i] = textureLoad(image, vec2<i32>(i32(gid.x), i32(gid.y)), 0).r;
  }
}`,
    });

    const result = await runWgslKernelProgram(
      device,
      program,
      {
        textures: {
          image: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4]) },
        },
        buffers: { out: new Float32Array(4) },
      },
      { dispatchCount: [2, 2, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("runs a sequence over shared GPU buffers without intermediate readback", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const addOne = defineWgslKernelProgram({
      name: "seq_add_one",
      workgroupSize: [4, 1, 1],
      bindings: [{ kind: "storage", name: "x", valueType: "f32", access: "read_write" }],
      wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] + 1.0; }
}`,
    });
    const scale = defineWgslKernelProgram({
      name: "seq_scale",
      workgroupSize: [4, 1, 1],
      bindings: [{ kind: "storage", name: "x", valueType: "f32", access: "read_write" }],
      wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] * 2.0; }
}`,
    });

    const result = await runWgslKernelProgramSequence(
      device,
      [
        { program: addOne, launch: { dispatchCount: [4, 1, 1] } },
        { program: scale, launch: { dispatchCount: [4, 1, 1] } },
      ],
      { buffers: { x: new Float32Array([1, 2, 3, 4]) } },
    );

    expect([...result.buffers.x as Float32Array]).toEqual([4, 6, 8, 10]);
  });

  it("runs sequence steps with same uniform binding name but different values", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const fill = defineWgslKernelProgram({
      name: "seq_uniform_fill",
      workgroupSize: [4, 1, 1],
      bindings: [
        { kind: "storage", name: "x", valueType: "f32", access: "read_write", binding: 0 },
        { kind: "uniform", name: "params", byteLength: 16, binding: 1 },
      ],
      wgsl: `
struct Params { value: f32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = params.value; }
}`,
    });
    const affine = defineWgslKernelProgram({
      name: "seq_uniform_affine",
      workgroupSize: [4, 1, 1],
      bindings: [
        { kind: "storage", name: "x", valueType: "f32", access: "read_write", binding: 0 },
        { kind: "uniform", name: "params", byteLength: 16, binding: 1 },
      ],
      wgsl: `
struct Params { scale: f32, bias: f32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] * params.scale + params.bias; }
}`,
    });

    const result = await runWgslKernelProgramSequence(
      device,
      [
        { program: fill, launch: { dispatchCount: [4, 1, 1] }, uniforms: { params: new Float32Array([3]) } },
        { program: affine, launch: { dispatchCount: [4, 1, 1] }, uniforms: { params: new Float32Array([2, 1]) } },
      ],
      { buffers: { x: new Float32Array(4) } },
    );

    expect([...result.buffers.x as Float32Array]).toEqual([7, 7, 7, 7]);
  });

  it("runs sequence steps with storage binding aliases", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const writeX = defineWgslKernelProgram({
      name: "seq_alias_write_x",
      workgroupSize: [4, 1, 1],
      bindings: [{ kind: "storage", name: "x", valueType: "f32", access: "read_write", binding: 0 }],
      wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = f32(gid.x + 1u); }
}`,
    });
    const scaleY = defineWgslKernelProgram({
      name: "seq_alias_scale_y",
      workgroupSize: [4, 1, 1],
      bindings: [{ kind: "storage", name: "y", valueType: "f32", access: "read_write", binding: 0 }],
      wgsl: `
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&y)) { y[gid.x] = y[gid.x] * 3.0; }
}`,
    });

    const result = await runWgslKernelProgramSequence(
      device,
      [
        { program: writeX, launch: { dispatchCount: [4, 1, 1] } },
        { program: scaleY, launch: { dispatchCount: [4, 1, 1] }, storageAliases: { y: "x" } },
      ],
      { buffers: { x: new Float32Array(4) } },
    );

    expect([...result.buffers.x as Float32Array]).toEqual([3, 6, 9, 12]);
  });

  it("runs over caller-owned resident GPU buffers without forced readback", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const program = defineWgslKernelProgram({
      name: "resident_saxpy",
      workgroupSize: [4, 1, 1],
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
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&y)) {
    y[i] = params.a * x[i] + y[i];
  }
}`,
    });
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([1, 2, 3, 4]),
      label: "resident-x",
    });
    const y = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([10, 20, 30, 40]),
      label: "resident-y",
    });

    try {
      const result = await runWgslKernelProgram(
        device,
        program,
        {
          buffers: {},
          residentBuffers: { x, y },
          uniforms: { params: new Float32Array([2]) },
          readback: [],
        },
        { dispatchCount: [4, 1, 1] },
      );
      expect(result.buffers).toEqual({});

      const readback = await readWgslStorageBuffer(device, y);
      expect([...readback as Float32Array]).toEqual([12, 24, 36, 48]);
    } finally {
      destroyWgslStorageBuffer(x);
      destroyWgslStorageBuffer(y);
    }
  });

  it("rewrites resident GPU buffers in place between dispatches", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const program = defineWgslKernelProgram({
      name: "resident_scale",
      workgroupSize: [4, 1, 1],
      bindings: [{ kind: "storage", name: "x", valueType: "f32", access: "read_write" }],
      wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] * 2.0; }
}`,
    });
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      byteLength: Float32Array.BYTES_PER_ELEMENT * 4,
      label: "resident-rewrite-x",
    });

    try {
      writeWgslStorageBuffer(device, x, new Float32Array([1, 2, 3, 4]));
      await runWgslKernelProgram(
        device,
        program,
        { buffers: {}, residentBuffers: { x }, readback: [] },
        { dispatchCount: [4, 1, 1] },
      );
      const first = await readWgslStorageBuffer(device, x);
      expect([...first as Float32Array]).toEqual([2, 4, 6, 8]);

      writeWgslStorageBuffer(device, x, new Float32Array([5, 6]), Float32Array.BYTES_PER_ELEMENT);
      await runWgslKernelProgram(
        device,
        program,
        { buffers: {}, residentBuffers: { x }, readback: [] },
        { dispatchCount: [4, 1, 1] },
      );
      const second = await readWgslStorageBuffer(device, x);
      expect([...second as Float32Array]).toEqual([4, 10, 12, 16]);
    } finally {
      destroyWgslStorageBuffer(x);
    }
  });

  it("reuses a prepared WGSL sequence across resident-buffer runs", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const program = defineWgslKernelProgram({
      name: "prepared_resident_scale",
      workgroupSize: [4, 1, 1],
      bindings: [{ kind: "storage", name: "x", valueType: "f32", access: "read_write" }],
      wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] * 3.0; }
}`,
    });
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([1, 2, 3, 4]),
      label: "prepared-resident-x",
    });
    const prepared = await prepareWgslKernelProgramSequence(
      device,
      [{ program, launch: { dispatchCount: [4, 1, 1] } }],
      { buffers: {}, residentBuffers: { x }, readback: [] },
    );

    try {
      expect(prepared.stepCount).toBe(1);

      const first = await prepared.run();
      expect(first.buffers).toEqual({});
      const firstReadback = await readWgslStorageBuffer(device, x);
      expect([...firstReadback as Float32Array]).toEqual([3, 6, 9, 12]);

      writeWgslStorageBuffer(device, x, new Float32Array([2, 4, 6, 8]));
      await prepared.run({ readback: [] });
      const secondReadback = await readWgslStorageBuffer(device, x);
      expect([...secondReadback as Float32Array]).toEqual([6, 12, 18, 24]);
    } finally {
      prepared.destroy();
      destroyWgslStorageBuffer(x);
    }
  });
});
