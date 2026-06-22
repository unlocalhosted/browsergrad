/**
 * createWebGpuRealizerBridge — the production WebGPU bridge for the
 * browsergrad-jit Python realizer (PRD-011.5).
 *
 * The bridge satisfies the `WebGpuBridge` Protocol declared on the
 * Python side (see `browsergrad_jit/_bridge.py`). Each method:
 *
 *   - Crosses the Pyodide JS boundary as a synchronous-from-Python call
 *     via JSPI (Pyodide 0.27+ on Chrome 144+). Caller asynchrony is
 *     hidden behind Python's normal call stack.
 *   - Operates on opaque integer handles. Python never sees a GPUBuffer
 *     directly; the bridge owns the lifetime via its internal handle
 *     map.
 *   - Returns either a handle (compute ops) or a Uint8Array (materialise),
 *     which Pyodide marshals back to Python.
 *
 * v0 scope per the DL/GPU review:
 *   - f32 only. f16/AMP path is PRD-012a.
 *   - 2-D matmul only. Batched matmul is PRD-012a.
 *   - Flash Attention v2 forward. Backward is its own follow-on PRD.
 *   - Naive (un-tiled) matmul kernel — the existing `kernels/matmul.ts`.
 *     Replacing with a tiled GEMM is the first deliverable of PRD-012a.
 *
 * What this is NOT:
 *   - A pattern matcher. The Python `bg.kernels.flash_attention(...)`
 *     wrapper constructs the CUSTOM UOp explicitly; the bridge has no
 *     "is this an attention block?" logic.
 *   - A cost model. The realizer just dispatches what it gets.
 *   - An OPFS pipeline cache. The device's in-memory pipeline cache
 *     covers session-scope reuse; cross-session caching is PRD-008.2.
 */

import {
  materializeFloat32,
  uploadFloat32,
  runDirect,
  type KernelDescriptor,
} from "./runner.js";
import { matmulTiledDirect } from "./kernels/matmul_tiled.js";
import {
  fusedElementwiseDirect,
  type FusedOp,
} from "./kernels/fused_elementwise.js";
import { flashAttentionDirect } from "./kernels/flash_attention.js";
import { KernelError, type KernelDevice } from "./types.js";

type Handle = number;

interface BufferRecord {
  buffer: GPUBuffer;
  byteLength: number;
  shape: readonly number[];
  dtype: string;
}

export interface WebGpuRealizerBridge {
  upload(data: Uint8Array, shape: readonly number[], dtype: string): Handle;
  /**
   * Read a handle's contents back to a Uint8Array of f32 byte content.
   *
   * Returns a Promise because GPU readback is asynchronous (mapAsync). Pyodide
   * JSPI consumers (Python's `bridge.materialize(...)` call) hide the await
   * transparently — the Promise is unwrapped at the JS↔Python boundary, so
   * Python sees a synchronous Uint8Array return as the Protocol declares.
   *
   * JS/TS consumers must `await` this. (Previous releases declared the return
   * type as `Uint8Array` directly to mirror the Python Protocol; that was a
   * type contract violation — runtime always returned a Promise.)
   */
  materialize(handle: Handle, shape: readonly number[], dtype: string): Promise<Uint8Array>;
  release(handle: Handle): void;
  matmul(a: Handle, b: Handle, m: number, k: number, n: number, dtype: string): Handle;
  fused_elementwise(
    inputs: readonly Handle[],
    ops: readonly (readonly [string, number, number])[],
    shape: readonly number[],
    dtype: string,
  ): Handle;
  cast(handle: Handle, srcDtype: string, dstDtype: string, shape: readonly number[]): Handle;
  flash_attention(
    q: Handle,
    k: Handle,
    v: Handle,
    mask: Handle | null,
    b: number,
    h: number,
    sq: number,
    sk: number,
    d: number,
    scale: number,
    dtype: string,
  ): Promise<Handle>;
  run_user_kernel(
    inputs: readonly Handle[],
    wgsl: string,
    name: string,
    hash: string,
    workgroupSize: readonly [number, number, number],
    dispatchShape: readonly [number, number, number],
    outputLength: number,
    outputShape: readonly number[],
    dtype: string,
  ): Handle;
  /** Diagnostic — number of GPU buffers currently alive. */
  aliveHandleCount(): number;
}

function assertF32(dtype: string, op: string): void {
  if (dtype !== "float32") {
    throw new KernelError(
      `WebGPU bridge: ${op} only supports float32 in v0 (got ${dtype}). ` +
        `f16/AMP path is PRD-012a.`,
    );
  }
}

/**
 * Construct a WebGpuRealizerBridge bound to the given device. Holds a
 * mutable handle map internally; never thread-shared.
 */
export function createWebGpuRealizerBridge(
  device: KernelDevice,
): WebGpuRealizerBridge {
  const handles = new Map<Handle, BufferRecord>();
  let nextId = 1;

  const mint = (
    buffer: GPUBuffer,
    byteLength: number,
    shape: readonly number[],
    dtype: string,
  ): Handle => {
    const id = nextId++;
    handles.set(id, { buffer, byteLength, shape, dtype });
    return id;
  };

  const get = (handle: Handle, op: string): BufferRecord => {
    const rec = handles.get(handle);
    if (rec === undefined) {
      throw new KernelError(
        `WebGPU bridge: ${op} called with unknown handle ${handle}. ` +
          `It may have been released already.`,
      );
    }
    return rec;
  };

  return {
    upload(data: Uint8Array, shape: readonly number[], dtype: string): Handle {
      assertF32(dtype, "upload");
      // Reinterpret the input bytes as f32. Tolerate misaligned byteOffset
      // by copying into a fresh aligned ArrayBuffer when necessary.
      let f32: Float32Array;
      if ((data.byteOffset & 3) === 0 && data.byteLength % 4 === 0) {
        f32 = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
      } else {
        const copy = new Uint8Array(data.byteLength);
        copy.set(data);
        f32 = new Float32Array(copy.buffer);
      }
      const buf = uploadFloat32(device, f32);
      return mint(buf, data.byteLength, shape, dtype);
    },

    async materialize(handle: Handle, shape: readonly number[], dtype: string): Promise<Uint8Array> {
      assertF32(dtype, "materialize");
      // Shape is informational here; the buffer carries its own byte length.
      void shape;
      const rec = get(handle, "materialize");
      // materializeFloat32 returns Float32Array; reinterpret the bytes as
      // a Uint8Array view so the Python side reads raw f32 bytes. Pyodide
      // JSPI unwraps this Promise transparently — the Python Protocol
      // declares a sync return, JS callers must await.
      const f32 = await materializeFloat32(device, rec.buffer, rec.byteLength);
      return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    },

    release(handle: Handle): void {
      const rec = handles.get(handle);
      if (rec === undefined) return; // idempotent
      rec.buffer.destroy();
      handles.delete(handle);
    },

    matmul(a: Handle, b: Handle, m: number, k: number, n: number, dtype: string): Handle {
      assertF32(dtype, "matmul");
      const aRec = get(a, "matmul[A]");
      const bRec = get(b, "matmul[B]");
      // PRD-012a: tiled GEMM (16×16 workgroup-shared tiles) replaces the
      // naive triple-loop. Reduces DRAM reads from 2*M*N*K to ~M*N*K/8 —
      // the load-bearing perf win the megakernel-PRD was claiming.
      const result = matmulTiledDirect(device, aRec.buffer, bRec.buffer, m, k, n);
      return mint(result.buffer, result.byteLength, [m, n], dtype);
    },

    fused_elementwise(
      inputs: readonly Handle[],
      ops: readonly (readonly [string, number, number])[],
      shape: readonly number[],
      dtype: string,
    ): Handle {
      assertF32(dtype, "fused_elementwise");
      // PRD-012a: WGSL codegen for arbitrary elementwise chains. The ops
      // list is the same shape the Python fusion pass produces; we walk
      // it, emit a single compute shader, and pipeline-cache by hash.
      const inputBufs = inputs.map((h, i) => get(h, `fused_elementwise[in${i}]`).buffer);
      let total = 1;
      for (const d of shape) total *= d;
      if (total === 0) total = 1;
      const fusedOps: FusedOp[] = ops.map((o) => [o[0], o[1], o[2]] as FusedOp);
      const result = fusedElementwiseDirect(device, inputBufs, fusedOps, total);
      return mint(result.buffer, result.byteLength, shape, dtype);
    },

    cast(
      handle: Handle,
      srcDtype: string,
      dstDtype: string,
      shape: readonly number[],
    ): Handle {
      // v0: f32→f32 is a no-op (returns same buffer alias). Any other
      // cast is unsupported until f16/AMP lands in PRD-012a.
      if (srcDtype === "float32" && dstDtype === "float32") {
        // GPU-only copy via CopyBufferToBuffer (PRD-012a). The source
        // buffer's lifetime stays with the caller; the new handle owns
        // a freshly-allocated GPUBuffer with STORAGE | COPY_SRC usage so
        // downstream ops can read it and a future materialize() can
        // copy from it.
        const src = get(handle, "cast[f32→f32]");
        const impl = device.gpu;
        let n = 1;
        for (const d of shape) n *= d;
        const byteLength = (n || 1) * 4;
        const aligned = Math.ceil(byteLength / 4) * 4;
        const dst = impl.createBuffer({
          size: aligned,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const encoder = impl.createCommandEncoder({ label: "bg-cast-f32-f32" });
        encoder.copyBufferToBuffer(src.buffer, 0, dst, 0, aligned);
        impl.queue.submit([encoder.finish()]);
        return mint(dst, byteLength, shape, dstDtype);
      }
      throw new KernelError(
        `WebGPU bridge: cast ${srcDtype}→${dstDtype} not supported in v0. ` +
          `f16/bf16 cast kernels land in PRD-012b.`,
      );
    },

    async flash_attention(
      q: Handle,
      k: Handle,
      v: Handle,
      mask: Handle | null,
      b: number,
      h: number,
      sq: number,
      sk: number,
      d: number,
      scale: number,
      dtype: string,
    ): Promise<Handle> {
      assertF32(dtype, "flash_attention");
      const qRec = get(q, "flash_attention[Q]");
      const kRec = get(k, "flash_attention[K]");
      const vRec = get(v, "flash_attention[V]");
      const maskRec = mask !== null ? get(mask, "flash_attention[mask]") : null;
      const result = flashAttentionDirect(
        device,
        qRec.buffer,
        kRec.buffer,
        vRec.buffer,
        maskRec?.buffer ?? null,
        { B: b, H: h, Sq: sq, Sk: sk, D: d },
        scale,
      );
      return mint(result.buffer, result.byteLength, [b, h, sq, d], dtype);
    },

    run_user_kernel(
      inputs: readonly Handle[],
      wgsl: string,
      name: string,
      hash: string,
      workgroupSize: readonly [number, number, number],
      dispatchShape: readonly [number, number, number],
      outputLength: number,
      outputShape: readonly number[],
      dtype: string,
    ): Handle {
      assertF32(dtype, "run_user_kernel");
      const inputBufs = inputs.map(
        (h, i) => get(h, `run_user_kernel[in${i}]`).buffer,
      );
      const desc: KernelDescriptor = {
        // Cache key prefix carries the first 8 hash chars so kernels with
        // the same `name` but different WGSL get distinct cache entries.
        name: `user_${hash.slice(0, 8)}_${name}`,
        wgsl,
        workgroupSize: [workgroupSize[0], workgroupSize[1], workgroupSize[2]],
      };
      // v0: no uniform params. Users bake constants into WGSL.
      const result = runDirect(device, desc, {
        inputBuffers: inputBufs,
        outputLength,
        params: new Uint32Array(0),
        dispatchCount: [dispatchShape[0], dispatchShape[1], dispatchShape[2]],
        cacheKeySuffix: hash,
      });
      return mint(result.buffer, result.byteLength, outputShape, dtype);
    },

    aliveHandleCount(): number {
      return handles.size;
    },
  };
}
