/**
 * Shared GPU dispatch helper used by every kernel.
 *
 * Each kernel describes:
 *   - The WGSL source (already compiled into the cache by key)
 *   - The input buffers (Float32Array data + shape)
 *   - The uniform "params" struct (a Uint32Array, packed by the kernel)
 *   - The output shape (so we know how big the output buffer is)
 *   - The workgroup count
 *
 * We bind everything, dispatch, copy to a MAP_READ-able buffer, await read,
 * and return a Float32Array.
 *
 * Buffer lifetime: every call allocates fresh buffers and destroys them after.
 * That's not perf-optimal for hot loops — `runDirect` (planned) will let a
 * grad-style caller manage its own GPU buffers. v0 is correctness-first.
 */

import { asImpl } from "./device.js";
import { KernelError, type KernelDevice } from "./types.js";

export interface KernelDescriptor {
  /** Unique kernel name (used as cache key prefix). */
  readonly name: string;
  /** WGSL source. The entry point must be named `main`. */
  readonly wgsl: string;
  /** Workgroup size declared in the WGSL — must match `@workgroup_size(...)`. */
  readonly workgroupSize: readonly [number, number, number];
}

export interface DispatchOptions {
  readonly inputs: readonly Float32Array[];
  /** Length is the number of f32 elements in the output. */
  readonly outputLength: number;
  /** Uniform params packed as raw bytes by the caller. May be empty. */
  readonly params: Uint32Array | Float32Array;
  /** Optional initial output contents for kernels that read/write one buffer. */
  readonly initialOutput?: Float32Array;
  /** Global dispatch count `[x, y, z]`. The runner divides by workgroupSize internally. */
  readonly dispatchCount: readonly [number, number, number];
  /** A unique-per-parameter-shape suffix appended to the cache key. */
  readonly cacheKeySuffix: string;
}

export async function dispatch(
  device: KernelDevice,
  desc: KernelDescriptor,
  opts: DispatchOptions,
): Promise<Float32Array> {
  const impl = asImpl(device);
  const gpu = impl.gpu;
  validatePositiveInteger(opts.outputLength, "outputLength");

  const numInputs = opts.inputs.length;
  const hasParams = opts.params.length > 0;

  /* Pipeline (cached) ─────────────────────────────────── */
  const cacheKey = `${desc.name}::${opts.cacheKeySuffix}::${numInputs}::${hasParams ? "u" : "n"}`;
  const { pipeline, bindGroupLayout } = impl.acquirePipeline(cacheKey, () => {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < numInputs; i++) {
      entries.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      });
    }
    // Output binding
    entries.push({
      binding: numInputs,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    });
    if (hasParams) {
      entries.push({
        binding: numInputs + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      });
    }
    const bgl = gpu.createBindGroupLayout({ entries });
    const layout = gpu.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const module = gpu.createShaderModule({ code: desc.wgsl });
    const pipe = gpu.createComputePipeline({
      layout,
      compute: { module, entryPoint: "main" },
    });
    return { pipeline: pipe, bindGroupLayout: bgl };
  });

  /* Buffers ───────────────────────────────────────────── */
  const inputBuffers: GPUBuffer[] = [];
  let outputBuffer: GPUBuffer | undefined;
  let paramsBuffer: GPUBuffer | undefined;
  let readBuffer: GPUBuffer | undefined;

  try {
    for (const input of opts.inputs) {
      const buf = gpu.createBuffer({
        size: validateStorageByteLength(input.byteLength, "input"),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      gpu.queue.writeBuffer(buf, 0, input.buffer, input.byteOffset, input.byteLength);
      inputBuffers.push(buf);
    }

    const outputByteLength = opts.outputLength * 4;
    outputBuffer = gpu.createBuffer({
      size: alignTo(outputByteLength, 4),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    if (opts.initialOutput) {
      if (opts.initialOutput.length !== opts.outputLength) {
        throw new KernelError("initialOutput length must match outputLength");
      }
      gpu.queue.writeBuffer(
        outputBuffer,
        0,
        opts.initialOutput.buffer,
        opts.initialOutput.byteOffset,
        opts.initialOutput.byteLength,
      );
    }

    if (hasParams) {
      paramsBuffer = gpu.createBuffer({
        size: alignTo(opts.params.byteLength, 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.queue.writeBuffer(
        paramsBuffer,
        0,
        opts.params.buffer,
        opts.params.byteOffset,
        opts.params.byteLength,
      );
    }

    /* Bind group ────────────────────────────────────────── */
    const bgEntries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < numInputs; i++) {
      bgEntries.push({ binding: i, resource: { buffer: inputBuffers[i]! } });
    }
    bgEntries.push({ binding: numInputs, resource: { buffer: outputBuffer } });
    if (paramsBuffer) {
      bgEntries.push({ binding: numInputs + 1, resource: { buffer: paramsBuffer } });
    }
    const bindGroup = gpu.createBindGroup({
      layout: bindGroupLayout,
      entries: bgEntries,
    });

    /* Dispatch ──────────────────────────────────────────── */
    const encoder = gpu.createCommandEncoder({ label: `bg-${desc.name}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wgX = Math.ceil(opts.dispatchCount[0] / desc.workgroupSize[0]);
    const wgY = Math.ceil(opts.dispatchCount[1] / desc.workgroupSize[1]);
    const wgZ = Math.ceil(opts.dispatchCount[2] / desc.workgroupSize[2]);
    pass.dispatchWorkgroups(Math.max(wgX, 1), Math.max(wgY, 1), Math.max(wgZ, 1));
    pass.end();

    /* Readback ─────────────────────────────────────────── */
    readBuffer = gpu.createBuffer({
      size: alignTo(outputByteLength, 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, alignTo(outputByteLength, 4));
    gpu.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    try {
      impl.recordInvocation();
      return new Float32Array(readBuffer.getMappedRange(0, outputByteLength).slice(0));
    } finally {
      readBuffer.unmap();
    }
  } finally {
    for (const b of inputBuffers) b.destroy();
    outputBuffer?.destroy();
    paramsBuffer?.destroy();
    readBuffer?.destroy();
  }
}

function alignTo(value: number, align: number): number {
  return Math.ceil(value / align) * align;
}

function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`${name} must be a positive integer`);
  }
  return value;
}

function validateStorageByteLength(byteLength: number, name: string): number {
  if (!Number.isInteger(byteLength) || byteLength <= 0) {
    throw new KernelError(`${name} must not be empty`);
  }
  return alignTo(byteLength, 4);
}

function validateFloat32ByteLength(byteLength: number, name: string): number {
  validatePositiveInteger(byteLength, name);
  if (byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new KernelError(`${name} must be a multiple of ${Float32Array.BYTES_PER_ELEMENT}`);
  }
  return byteLength;
}

/* ────────────────────────────────────────────────────────────
 * runDirect — GPUBuffer-in, GPUBuffer-out (PRD-011.5).
 *
 * The realizer-tier dispatch path. `dispatch()` (above) round-trips
 * host on every call — fine for one-shot kernel use, lethal for
 * chained ops. `runDirect()` keeps intermediates resident: inputs
 * are existing GPUBuffers, output is a freshly-allocated GPUBuffer
 * the caller will pass to the next kernel.
 *
 * Caller owns the output buffer's lifetime — pair every successful
 * runDirect with a later `.destroy()`.
 * ──────────────────────────────────────────────────────────── */

export interface DirectDispatchOptions {
  readonly inputBuffers: readonly GPUBuffer[];
  /** Length in f32 elements of the output. */
  readonly outputLength: number;
  /** Caller-allocated output GPUBuffer with STORAGE + COPY_SRC usage.
   *  If null/undefined, runDirect allocates one with default usage. */
  readonly outputBuffer?: GPUBuffer | null;
  readonly params: Uint32Array;
  readonly dispatchCount: readonly [number, number, number];
  readonly cacheKeySuffix: string;
}

export interface DirectDispatchResult {
  /** The output GPUBuffer. The caller owns it and must destroy it. */
  readonly buffer: GPUBuffer;
  /** Byte length corresponding to outputLength * 4. */
  readonly byteLength: number;
}

export function runDirect(
  device: KernelDevice,
  desc: KernelDescriptor,
  opts: DirectDispatchOptions,
): DirectDispatchResult {
  const impl = asImpl(device);
  const gpu = impl.gpu;
  validatePositiveInteger(opts.outputLength, "outputLength");

  const numInputs = opts.inputBuffers.length;
  const hasParams = opts.params.length > 0;

  /* Pipeline (cached) — same cache key shape as dispatch(). */
  const cacheKey = `${desc.name}::${opts.cacheKeySuffix}::${numInputs}::${hasParams ? "u" : "n"}`;
  const { pipeline, bindGroupLayout } = impl.acquirePipeline(cacheKey, () => {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < numInputs; i++) {
      entries.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      });
    }
    entries.push({
      binding: numInputs,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    });
    if (hasParams) {
      entries.push({
        binding: numInputs + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      });
    }
    const bgl = gpu.createBindGroupLayout({ entries });
    const layout = gpu.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const module = gpu.createShaderModule({ code: desc.wgsl });
    const pipe = gpu.createComputePipeline({
      layout,
      compute: { module, entryPoint: "main" },
    });
    return { pipeline: pipe, bindGroupLayout: bgl };
  });

  const outputByteLength = opts.outputLength * 4;
  const ownsOutput = opts.outputBuffer === undefined || opts.outputBuffer === null;
  let completed = false;
  let outputBuffer: GPUBuffer | undefined;
  let paramsBuffer: GPUBuffer | undefined;
  try {
    outputBuffer =
      opts.outputBuffer ??
      gpu.createBuffer({
        size: alignTo(outputByteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

    if (hasParams) {
      paramsBuffer = gpu.createBuffer({
        size: alignTo(opts.params.byteLength, 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.queue.writeBuffer(
        paramsBuffer,
        0,
        opts.params.buffer,
        opts.params.byteOffset,
        opts.params.byteLength,
      );
    }

    const bgEntries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < numInputs; i++) {
      bgEntries.push({ binding: i, resource: { buffer: opts.inputBuffers[i]! } });
    }
    bgEntries.push({ binding: numInputs, resource: { buffer: outputBuffer } });
    if (paramsBuffer) {
      bgEntries.push({ binding: numInputs + 1, resource: { buffer: paramsBuffer } });
    }
    const bindGroup = gpu.createBindGroup({
      layout: bindGroupLayout,
      entries: bgEntries,
    });

    const encoder = gpu.createCommandEncoder({ label: `bg-direct-${desc.name}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wgX = Math.ceil(opts.dispatchCount[0] / desc.workgroupSize[0]);
    const wgY = Math.ceil(opts.dispatchCount[1] / desc.workgroupSize[1]);
    const wgZ = Math.ceil(opts.dispatchCount[2] / desc.workgroupSize[2]);
    pass.dispatchWorkgroups(Math.max(wgX, 1), Math.max(wgY, 1), Math.max(wgZ, 1));
    pass.end();
    gpu.queue.submit([encoder.finish()]);

    impl.recordInvocation();
    completed = true;
    return { buffer: outputBuffer, byteLength: outputByteLength };
  } finally {
    paramsBuffer?.destroy();
    if (!completed && ownsOutput) outputBuffer?.destroy();
  }
}

/**
 * Read a GPUBuffer back to a Float32Array. Used at the realize boundary
 * to materialize the final result. Async because GPUBuffer.mapAsync is.
 */
export async function materializeFloat32(
  device: KernelDevice,
  buffer: GPUBuffer,
  byteLength: number,
): Promise<Float32Array> {
  const impl = asImpl(device);
  const gpu = impl.gpu;
  validateFloat32ByteLength(byteLength, "byteLength");
  const aligned = alignTo(byteLength, 4);
  const readBuffer = gpu.createBuffer({
    size: aligned,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const encoder = gpu.createCommandEncoder({ label: "bg-materialize" });
    encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, aligned);
    gpu.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    try {
      return new Float32Array(readBuffer.getMappedRange(0, byteLength).slice(0));
    } finally {
      readBuffer.unmap();
    }
  } finally {
    readBuffer.destroy();
  }
}

/** Upload typed array data into a freshly-allocated GPUBuffer. */
export function uploadFloat32(
  device: KernelDevice,
  data: Float32Array,
): GPUBuffer {
  const impl = asImpl(device);
  const gpu = impl.gpu;
  const buf = gpu.createBuffer({
    size: validateStorageByteLength(data.byteLength, "uploadFloat32 data"),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  gpu.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

/* ────────────────────────────────────────────────────────────
 * Shape / element-count helpers used by every kernel
 * ──────────────────────────────────────────────────────────── */

export function numel(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function assertShape(
  name: string,
  actual: readonly number[],
  expected: readonly number[],
): void {
  if (actual.length !== expected.length) {
    throw new KernelError(
      `${name}: expected ${expected.length}D tensor, got shape [${actual.join(", ")}]`,
    );
  }
  for (let i = 0; i < actual.length; i++) {
    if (expected[i] !== -1 && actual[i] !== expected[i]) {
      throw new KernelError(
        `${name}: shape mismatch at dim ${i} — expected ${expected[i]}, got ${actual[i]} (full shape [${actual.join(", ")}])`,
      );
    }
  }
}

export function assertDataLength(name: string, data: Float32Array, shape: readonly number[]): void {
  const expected = numel(shape);
  if (data.length !== expected) {
    throw new KernelError(
      `${name}: data length ${data.length} doesn't match shape [${shape.join(", ")}] (expected ${expected})`,
    );
  }
}
