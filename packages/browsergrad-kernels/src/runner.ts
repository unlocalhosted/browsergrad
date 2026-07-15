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
import { issueWithWebGpuErrorScopes } from "./webgpu_error_scope.js";

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
  readonly profile?: DirectDispatchProfileOptions | undefined;
}

export interface DirectDispatchResult {
  /** The output GPUBuffer. The caller owns it and must destroy it. */
  readonly buffer: GPUBuffer;
  /** Byte length corresponding to outputLength * 4. */
  readonly byteLength: number;
  /** Optional timing evidence for this dispatch. */
  readonly profile?: Promise<DirectDispatchProfile>;
}

export interface DirectDispatchProfileOptions {
  /** Defaults to true. Set false only for callers that explicitly do not want timing evidence. */
  readonly enabled?: boolean;
  readonly label?: string;
}

export type DirectDispatchTimingMode =
  | "timestamp-query"
  | "queue-completion"
  | "unavailable";

export type DirectDispatchTimingConfidence =
  | "exact"
  | "coarse"
  | "unavailable";

export interface DirectDispatchProfile {
  readonly label: string;
  readonly timingMode: DirectDispatchTimingMode;
  readonly confidence: DirectDispatchTimingConfidence;
  readonly gpuElapsedMs?: number;
  readonly queueElapsedMs?: number;
  readonly unavailableReason?: string;
  readonly dispatchCount: readonly [number, number, number];
  readonly workgroupSize: readonly [number, number, number];
}

interface DirectDispatchProfiler {
  readonly passDescriptor: GPUComputePassDescriptor;
  resolve(encoder: GPUCommandEncoder): void;
  finish(): Promise<DirectDispatchProfile>;
  cleanup(): void;
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
  let profiler: DirectDispatchProfiler | null = null;
  try {
    outputBuffer =
      opts.outputBuffer ??
      impl.acquireOutputBuffer(alignTo(outputByteLength, 4));

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

    const workgroupDispatchCount = workgroupDispatchCountFor(desc, opts.dispatchCount);
    profiler = createDirectDispatchProfiler(gpu, desc, opts, workgroupDispatchCount);
    const encoder = gpu.createCommandEncoder({ label: `bg-direct-${desc.name}` });
    const pass = encoder.beginComputePass(profiler?.passDescriptor ?? {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      workgroupDispatchCount[0],
      workgroupDispatchCount[1],
      workgroupDispatchCount[2],
    );
    pass.end();
    profiler?.resolve(encoder);
    gpu.queue.submit([encoder.finish()]);

    const profile = profiler?.finish();
    impl.recordInvocation();
    completed = true;
    return {
      buffer: outputBuffer,
      byteLength: outputByteLength,
      ...(profile ? { profile } : {}),
    };
  } finally {
    paramsBuffer?.destroy();
    if (!completed) profiler?.cleanup();
    if (!completed && ownsOutput) outputBuffer?.destroy();
  }
}

function workgroupDispatchCountFor(
  desc: KernelDescriptor,
  dispatchCount: readonly [number, number, number],
): readonly [number, number, number] {
  const wgX = Math.ceil(dispatchCount[0] / desc.workgroupSize[0]);
  const wgY = Math.ceil(dispatchCount[1] / desc.workgroupSize[1]);
  const wgZ = Math.ceil(dispatchCount[2] / desc.workgroupSize[2]);
  return [Math.max(wgX, 1), Math.max(wgY, 1), Math.max(wgZ, 1)];
}

function createDirectDispatchProfiler(
  gpu: GPUDevice,
  desc: KernelDescriptor,
  opts: DirectDispatchOptions,
  workgroupDispatchCount: readonly [number, number, number],
): DirectDispatchProfiler | null {
  if (opts.profile?.enabled === false) return null;
  const label = opts.profile?.label ?? desc.name;

  if (gpu.features.has("timestamp-query")) {
    const querySet = gpu.createQuerySet({ type: "timestamp", count: 2 });
    const queryBuffer = gpu.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = gpu.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    return {
      passDescriptor: {
        label: `bg-profile-${label}`,
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      },
      resolve(encoder): void {
        encoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
        encoder.copyBufferToBuffer(queryBuffer, 0, readBuffer, 0, 16);
      },
      async finish(): Promise<DirectDispatchProfile> {
        try {
          await readBuffer.mapAsync(GPUMapMode.READ);
          try {
            const timestamps = new BigUint64Array(readBuffer.getMappedRange(0, 16).slice(0));
            const deltaNs = timestamps[1]! > timestamps[0]!
              ? timestamps[1]! - timestamps[0]!
              : 0n;
            return {
              label,
              timingMode: "timestamp-query",
              confidence: "exact",
              gpuElapsedMs: Number(deltaNs) / 1_000_000,
              dispatchCount: workgroupDispatchCount,
              workgroupSize: desc.workgroupSize,
            };
          } finally {
            readBuffer.unmap();
          }
        } catch (err) {
          return unavailableProfile(
            label,
            desc,
            workgroupDispatchCount,
            err instanceof Error ? err.message : String(err),
          );
        } finally {
          querySet.destroy();
          queryBuffer.destroy();
          readBuffer.destroy();
        }
      },
      cleanup(): void {
        querySet.destroy();
        queryBuffer.destroy();
        readBuffer.destroy();
      },
    };
  }

  const t0 = performance.now();
  return {
    passDescriptor: { label: `bg-profile-${label}` },
    resolve(): void {},
    async finish(): Promise<DirectDispatchProfile> {
      try {
        await gpu.queue.onSubmittedWorkDone();
        return {
          label,
          timingMode: "queue-completion",
          confidence: "coarse",
          queueElapsedMs: performance.now() - t0,
          dispatchCount: workgroupDispatchCount,
          workgroupSize: desc.workgroupSize,
        };
      } catch (err) {
        return unavailableProfile(
          label,
          desc,
          workgroupDispatchCount,
          err instanceof Error ? err.message : String(err),
        );
      }
    },
    cleanup(): void {},
  };
}

function unavailableProfile(
  label: string,
  desc: KernelDescriptor,
  workgroupDispatchCount: readonly [number, number, number],
  reason: string,
): DirectDispatchProfile {
  return {
    label,
    timingMode: "unavailable",
    confidence: "unavailable",
    unavailableReason: reason,
    dispatchCount: workgroupDispatchCount,
    workgroupSize: desc.workgroupSize,
  };
}

export function releaseDirectBuffer(
  device: KernelDevice,
  buffer: GPUBuffer,
  byteLength: number,
): void {
  asImpl(device).releaseOutputBuffer(buffer, alignTo(byteLength, 4));
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
  let readBuffer: GPUBuffer | undefined;
  try {
    const issued = await issueWithWebGpuErrorScopes(
      gpu,
      "$.materialize",
      () => {
        const scopedReadBuffer = gpu.createBuffer({
          size: aligned,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        readBuffer = scopedReadBuffer;
        const encoder = gpu.createCommandEncoder({ label: "bg-materialize" });
        encoder.copyBufferToBuffer(buffer, 0, scopedReadBuffer, 0, aligned);
        gpu.queue.submit([encoder.finish()]);
        // mapAsync must be initiated before the scope pops. Its settlement is
        // awaited together with the pops and device-loss race by the helper.
        const mapping = scopedReadBuffer.mapAsync(GPUMapMode.READ);
        return Object.freeze({ readBuffer: scopedReadBuffer, mapping });
      },
      {
        completion: ({ mapping }) => mapping,
        cleanup: ({ readBuffer: failedReadBuffer }) => {
          failedReadBuffer.destroy();
          if (readBuffer === failedReadBuffer) readBuffer = undefined;
        },
      },
    );
    try {
      return new Float32Array(issued.readBuffer.getMappedRange(0, byteLength).slice(0));
    } finally {
      issued.readBuffer.unmap();
    }
  } finally {
    readBuffer?.destroy();
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
