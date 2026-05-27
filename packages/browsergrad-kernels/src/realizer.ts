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
} from "./runner.js";
import { matmulDirect } from "./kernels/matmul.js";
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
  materialize(handle: Handle, shape: readonly number[], dtype: string): Uint8Array;
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

    materialize(handle: Handle, shape: readonly number[], dtype: string): Uint8Array {
      assertF32(dtype, "materialize");
      // Shape is informational here; the buffer carries its own byte length.
      void shape;
      const rec = get(handle, "materialize");
      // materializeFloat32 is async; Python crosses the boundary via JSPI
      // so this Promise can be returned and awaited transparently.
      // For v0 we expose the Promise — Python's JSPI shim awaits it.
      // The Protocol on the Python side is sync from Python's POV.
      const promise = materializeFloat32(device, rec.buffer, rec.byteLength);
      // Wrap to return bytes view (Uint8Array of the f32 byte content).
      // We can't synchronously block here, so this is a Promise<Uint8Array>
      // that Pyodide JSPI will unwrap. TypeScript signature stays sync to
      // match the Python Protocol; the runtime hides the await.
      return promise as unknown as Uint8Array;
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
      const result = matmulDirect(device, aRec.buffer, bRec.buffer, m, k, n);
      return mint(result.buffer, result.byteLength, [m, n], dtype);
    },

    fused_elementwise(
      inputs: readonly Handle[],
      ops: readonly (readonly [string, number, number])[],
      shape: readonly number[],
      dtype: string,
    ): Handle {
      assertF32(dtype, "fused_elementwise");
      void inputs;
      void ops;
      void shape;
      throw new KernelError(
        `WebGPU bridge: fused_elementwise is not implemented in v0. ` +
          `Either decompose the chain back into primitive ops with their ` +
          `own GPU handlers, or fall back to bg.realize() (NumPy) for ` +
          `subgraphs containing OP_FUSED_ELEMENTWISE. PRD-012a will land ` +
          `a generic codegen path here.`,
      );
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
        void handle;
        void shape;
        // A true GPU-only copy via CopyBufferToBuffer lands in PRD-012a
        // alongside the f16 cast kernel. For v0, surface a clear refusal.
        throw new KernelError(
          `WebGPU bridge: cast even for float32→float32 needs a real ` +
            `CopyBufferToBuffer path or shared-handle semantics. Lands ` +
            `in PRD-012a; for v0, avoid OP_CAST in your IR or use ` +
            `bg.realize() (NumPy).`,
        );
      }
      throw new KernelError(
        `WebGPU bridge: cast ${srcDtype}→${dstDtype} not supported in v0. ` +
          `f16 paths land in PRD-012a.`,
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

    aliveHandleCount(): number {
      return handles.size;
    },
  };
}
