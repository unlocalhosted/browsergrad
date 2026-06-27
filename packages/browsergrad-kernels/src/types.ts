/**
 * Public types for @unlocalhosted/browsergrad-kernels.
 *
 * Stability contract: every field here is part of the public API.
 * - Adding a new optional field is non-breaking.
 * - Removing fields or narrowing types is breaking.
 * - Anything not exported from `./index.ts` is private.
 */

/**
 * A tensor is a flat `Float32Array` paired with a shape. Row-major order
 * (last axis is contiguous in memory), matching NumPy's default and matching
 * how the WGSL kernels are written.
 *
 * v0 is f32-only. f16 support is planned but additive.
 */
export interface Tensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

/**
 * Construct a `Tensor` literal. The number of elements in `data` must equal
 * the product of `shape`, otherwise the kernels will throw at dispatch time.
 */
export function tensor(shape: readonly number[], data: Float32Array): Tensor {
  return { shape, data };
}

export interface KernelDeviceOptions {
  /**
   * The `GPUDevice` to wrap. If omitted, `createDevice` calls
   * `navigator.gpu.requestAdapter()` + `requestDevice()` with sensible defaults.
   */
  device?: GPUDevice;

  /** Forwarded to `requestAdapter`. Default `"high-performance"`. */
  powerPreference?: GPUPowerPreference;

  /** Optional WebGPU features requested when BrowserGrad creates the device. */
  requiredFeatures?: readonly GPUFeatureName[];

  /** Max compute pipelines kept warm in cache. Default 32. */
  pipelineCacheSize?: number;
}

/**
 * Wraps a `GPUDevice` with a pipeline cache shared across kernel calls.
 * Construct via {@link createDevice}.
 */
export interface KernelDevice {
  /** The underlying WebGPU device. Safe to use directly if you know what you're doing. */
  readonly gpu: GPUDevice;

  /**
   * Stats about cache usage and kernel invocations — useful for debugging
   * but not part of any performance contract.
   */
  getStats(): KernelDeviceStats;

  /** Drop cached pipelines and release shader modules. The `GPUDevice` itself is not destroyed. */
  clearCache(): void;
}

export interface KernelDeviceStats {
  readonly pipelineCacheSize: number;
  readonly pipelineCacheHits: number;
  readonly pipelineCacheMisses: number;
  readonly kernelInvocations: number;
}

/**
 * Thrown when a kernel input is shape-incompatible, the device is unavailable,
 * or WebGPU itself errors. Kernels never silently produce wrong shapes — they
 * throw and let the caller fix it.
 */
export class KernelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "KernelError";
  }
}
