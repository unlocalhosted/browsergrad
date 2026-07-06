/**
 * Device acquisition + a thin pipeline cache.
 *
 * Each kernel gets its own ComputePipeline; we key the cache by
 * `(kernelId, paramSignature)` so a single device is shared across all kernel
 * calls and we don't re-compile the same WGSL for the same input shape twice.
 */

import { KernelError, type KernelDevice, type KernelDeviceOptions, type KernelDeviceStats } from "./types.js";

interface PipelineEntry {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

class KernelDeviceImpl implements KernelDevice {
  readonly gpu: GPUDevice;

  private cache = new Map<string, PipelineEntry>();
  private cacheLimit: number;
  private hits = 0;
  private misses = 0;
  private invocations = 0;
  private outputBufferPool = new Map<number, GPUBuffer[]>();
  private outputBufferPoolLimit: number;
  private outputBufferPoolHits = 0;
  private outputBufferPoolMisses = 0;

  constructor(gpu: GPUDevice, cacheLimit: number, outputBufferPoolLimit: number) {
    this.gpu = gpu;
    this.cacheLimit = cacheLimit;
    this.outputBufferPoolLimit = outputBufferPoolLimit;
  }

  /** @internal Used by runner. Public API stable, internal access opaque. */
  acquirePipeline(
    cacheKey: string,
    build: () => { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  ): PipelineEntry {
    const existing = this.cache.get(cacheKey);
    if (existing) {
      this.hits++;
      return existing;
    }
    this.misses++;
    const entry = build();
    if (this.cache.size >= this.cacheLimit) {
      // Drop the oldest — Map iteration order is insertion order.
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(cacheKey, entry);
    return entry;
  }

  /** @internal */
  recordInvocation(): void {
    this.invocations++;
  }

  /** @internal */
  acquireOutputBuffer(byteLength: number): GPUBuffer {
    const pool = this.outputBufferPool.get(byteLength);
    const existing = pool?.pop();
    if (existing) {
      this.outputBufferPoolHits++;
      return existing;
    }
    this.outputBufferPoolMisses++;
    return this.gpu.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  /** @internal */
  releaseOutputBuffer(buffer: GPUBuffer, byteLength: number): void {
    const pool = this.outputBufferPool.get(byteLength) ?? [];
    if (pool.length >= this.outputBufferPoolLimit) {
      buffer.destroy();
      return;
    }
    pool.push(buffer);
    this.outputBufferPool.set(byteLength, pool);
  }

  getStats(): KernelDeviceStats {
    let outputBufferPoolBuffers = 0;
    let outputBufferPoolBytes = 0;
    for (const [byteLength, buffers] of this.outputBufferPool.entries()) {
      outputBufferPoolBuffers += buffers.length;
      outputBufferPoolBytes += byteLength * buffers.length;
    }
    return {
      pipelineCacheSize: this.cache.size,
      pipelineCacheHits: this.hits,
      pipelineCacheMisses: this.misses,
      kernelInvocations: this.invocations,
      outputBufferPoolBuffers,
      outputBufferPoolBytes,
      outputBufferPoolHits: this.outputBufferPoolHits,
      outputBufferPoolMisses: this.outputBufferPoolMisses,
    };
  }

  clearCache(): void {
    this.cache.clear();
    for (const buffers of this.outputBufferPool.values()) {
      for (const buffer of buffers) buffer.destroy();
    }
    this.outputBufferPool.clear();
  }
}

export async function createDevice(
  options: KernelDeviceOptions = {},
): Promise<KernelDevice> {
  let gpu: GPUDevice;

  if (options.device) {
    gpu = options.device;
  } else {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new KernelError(
        "WebGPU not available. `navigator.gpu` is undefined — call `createDevice({ device })` with your own GPUDevice, or run in a browser with WebGPU enabled.",
      );
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: options.powerPreference ?? "high-performance",
    });
    if (!adapter) {
      throw new KernelError("Failed to obtain a WebGPU adapter");
    }
    const requiredFeatures = options.requiredFeatures ?? [];
    const missingFeatures = requiredFeatures.filter((feature) => !adapter.features.has(feature));
    if (missingFeatures.length > 0) {
      throw new KernelError(`WebGPU adapter missing required features: ${missingFeatures.join(", ")}`);
    }
    const requiredLimits = options.requiredLimits ? { ...options.requiredLimits } : undefined;
    if (requiredLimits) {
      for (const [name, value] of Object.entries(requiredLimits)) {
        if (value === undefined) continue;
        const supported = adapter.limits[name as keyof GPUSupportedLimits];
        if (typeof supported === "number" && value > supported) {
          throw new KernelError(`WebGPU adapter limit ${name}=${supported} is below required ${value}`);
        }
      }
    }
    const descriptor: GPUDeviceDescriptor = {};
    if (requiredFeatures.length > 0) descriptor.requiredFeatures = [...requiredFeatures];
    if (requiredLimits && Object.keys(requiredLimits).length > 0) descriptor.requiredLimits = requiredLimits;
    gpu = await adapter.requestDevice(Object.keys(descriptor).length > 0 ? descriptor : undefined);
  }

  return new KernelDeviceImpl(
    gpu,
    options.pipelineCacheSize ?? 32,
    options.outputBufferPoolSize ?? 8,
  );
}

/**
 * Type-narrowed accessor for the internal device impl. Internal use only.
 * Kernel implementations call this to reach pipeline caching without leaking
 * the impl class through the public type.
 */
export function asImpl(device: KernelDevice): KernelDeviceImpl {
  if (!(device instanceof KernelDeviceImpl)) {
    throw new KernelError(
      "KernelDevice was not produced by createDevice — pass the same KernelDevice through.",
    );
  }
  return device;
}
