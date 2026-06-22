import { compileCudaLiteKernel } from "./runner.js";
import type {
  CompiledCudaLiteKernel,
  CompileCudaLiteOptions,
} from "./types.js";

export interface CudaLiteCompilerCacheStats {
  readonly hits: number;
  readonly misses: number;
  readonly evictions: number;
  readonly entries: number;
}

export interface CudaLiteCompilerCacheOptions {
  readonly maxEntries?: number;
  readonly compileOptions?: CompileCudaLiteOptions;
  readonly compile?: (source: string, options?: CompileCudaLiteOptions) => CompiledCudaLiteKernel;
}

export interface CudaLiteCompilerCache {
  readonly maxEntries: number;
  readonly size: number;
  readonly stats: CudaLiteCompilerCacheStats;
  compile(source: string, options?: CompileCudaLiteOptions): CompiledCudaLiteKernel;
  get(source: string, options?: CompileCudaLiteOptions): CompiledCudaLiteKernel | undefined;
  delete(source: string, options?: CompileCudaLiteOptions): boolean;
  clear(): void;
}

const DEFAULT_MAX_ENTRIES = 128;

export function createCudaLiteCompilerCache(
  options: CudaLiteCompilerCacheOptions = {},
): CudaLiteCompilerCache {
  return new CudaLiteCompilerLruCache(options);
}

export function createCudaLiteCompileCacheKey(
  source: string,
  options: CompileCudaLiteOptions = {},
): string {
  return stableStringify([source, normalizeCompileOptions(options)]);
}

class CudaLiteCompilerLruCache implements CudaLiteCompilerCache {
  readonly maxEntries: number;
  private readonly defaults: CompileCudaLiteOptions;
  private readonly compileImpl: (source: string, options?: CompileCudaLiteOptions) => CompiledCudaLiteKernel;
  private readonly entries = new Map<string, CompiledCudaLiteKernel>();
  private hits = 0;
  private misses = 0;
  private evictions = 0;

  constructor(options: CudaLiteCompilerCacheOptions) {
    this.maxEntries = normalizeMaxEntries(options.maxEntries);
    this.defaults = options.compileOptions ?? {};
    this.compileImpl = options.compile ?? compileCudaLiteKernel;
  }

  get size(): number {
    return this.entries.size;
  }

  get stats(): CudaLiteCompilerCacheStats {
    return {
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      entries: this.entries.size,
    };
  }

  compile(source: string, options: CompileCudaLiteOptions = {}): CompiledCudaLiteKernel {
    const merged = mergeCompileOptions(this.defaults, options);
    const key = createCudaLiteCompileCacheKey(source, merged);
    const cached = this.entries.get(key);
    if (cached) {
      this.hits++;
      this.entries.delete(key);
      this.entries.set(key, cached);
      return cached;
    }

    this.misses++;
    const compiled = this.compileImpl(source, merged);
    if (this.maxEntries === 0) return compiled;
    this.entries.set(key, compiled);
    this.evictOverflow();
    return compiled;
  }

  get(source: string, options: CompileCudaLiteOptions = {}): CompiledCudaLiteKernel | undefined {
    const key = createCudaLiteCompileCacheKey(source, mergeCompileOptions(this.defaults, options));
    const cached = this.entries.get(key);
    if (!cached) return undefined;
    this.entries.delete(key);
    this.entries.set(key, cached);
    return cached;
  }

  delete(source: string, options: CompileCudaLiteOptions = {}): boolean {
    return this.entries.delete(createCudaLiteCompileCacheKey(source, mergeCompileOptions(this.defaults, options)));
  }

  clear(): void {
    this.entries.clear();
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  private evictOverflow(): void {
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) return;
      this.entries.delete(oldest);
      this.evictions++;
    }
  }
}

function mergeCompileOptions(
  defaults: CompileCudaLiteOptions,
  options: CompileCudaLiteOptions,
): CompileCudaLiteOptions {
  const merged: Record<string, unknown> = {
    ...defaults,
    ...options,
  };
  if (defaults.features || options.features) {
    merged.features = { ...defaults.features, ...options.features };
  }
  if (defaults.dynamicSharedMemory || options.dynamicSharedMemory) {
    merged.dynamicSharedMemory = { ...defaults.dynamicSharedMemory, ...options.dynamicSharedMemory };
  }
  if (defaults.pointerBaseOffsets || options.pointerBaseOffsets) {
    merged.pointerBaseOffsets = { ...defaults.pointerBaseOffsets, ...options.pointerBaseOffsets };
  }
  return merged as CompileCudaLiteOptions;
}

function normalizeMaxEntries(value: number | undefined): number {
  if (value === undefined) return DEFAULT_MAX_ENTRIES;
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError("CudaLiteCompilerCache maxEntries must be a non-negative integer");
  }
  return value;
}

function normalizeCompileOptions(options: CompileCudaLiteOptions): unknown {
  return stableNormalize(options);
}

function stableStringify(value: unknown): string {
  return JSON.stringify(stableNormalize(value));
}

function stableNormalize(value: unknown): unknown {
  if (value === undefined) return undefined;
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(stableNormalize);
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(value).sort()) {
    const normalized = stableNormalize((value as Record<string, unknown>)[key]);
    if (normalized !== undefined) out[key] = normalized;
  }
  return out;
}
