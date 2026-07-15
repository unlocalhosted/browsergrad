import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";

export interface DecodeLimits {
  readonly maxDocumentBytes: number;
  readonly maxDepth: number;
  readonly maxNodes: number;
  readonly maxStringBytes: number;
  readonly maxArrayLength: number;
  readonly maxObjectProperties: number;
  readonly maxRank: number;
  readonly maxIntegerBits: number;
  readonly maxArithmeticOperations: number;
}

export const DEFAULT_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: 8 * 1024 * 1024,
  maxDepth: 64,
  maxNodes: 100_000,
  maxStringBytes: 2 * 1024 * 1024,
  maxArrayLength: 100_000,
  maxObjectProperties: 10_000,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 200_000,
});

export const MAXIMUM_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: 64 * 1024 * 1024,
  maxDepth: 256,
  maxNodes: 1_000_000,
  maxStringBytes: 16 * 1024 * 1024,
  maxArrayLength: 1_000_000,
  maxObjectProperties: 100_000,
  maxRank: 1_024,
  maxIntegerBits: 4_096,
  maxArithmeticOperations: 2_000_000,
});

export function resolveDecodeLimits(overrides: Partial<DecodeLimits> = {}): DecodeLimits {
  const resolved = { ...DEFAULT_DECODE_LIMITS, ...overrides };
  for (const key of Object.keys(DEFAULT_DECODE_LIMITS) as Array<keyof DecodeLimits>) {
    const value = resolved[key];
    const maximum = MAXIMUM_DECODE_LIMITS[key];
    if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
      throw schemaError(
        SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
        `${key} must be a positive safe integer no greater than ${maximum}; got ${String(value)}`,
        { path: `$.limits.${key}` },
      );
    }
  }
  return Object.freeze(resolved);
}
