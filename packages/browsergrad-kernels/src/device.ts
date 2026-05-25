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

  constructor(gpu: GPUDevice, cacheLimit: number) {
    this.gpu = gpu;
    this.cacheLimit = cacheLimit;
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

  getStats(): KernelDeviceStats {
    return {
      pipelineCacheSize: this.cache.size,
      pipelineCacheHits: this.hits,
      pipelineCacheMisses: this.misses,
      kernelInvocations: this.invocations,
    };
  }

  clearCache(): void {
    this.cache.clear();
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
    gpu = await adapter.requestDevice();
  }

  return new KernelDeviceImpl(gpu, options.pipelineCacheSize ?? 32);
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
