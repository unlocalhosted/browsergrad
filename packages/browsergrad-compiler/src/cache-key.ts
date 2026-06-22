import type { CompileCudaLiteOptions } from "./types.js";

export function createCudaLiteCompileCacheKey(
  source: string,
  options: CompileCudaLiteOptions = {},
): string {
  return stableStringify([source, stableNormalize(options)]);
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
