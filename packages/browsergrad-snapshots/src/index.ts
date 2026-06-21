export type SnapshotMismatchKind =
  | "missing"
  | "unexpected"
  | "type"
  | "shape"
  | "value"
  | "non-finite";

export interface SnapshotMismatch {
  readonly path: string;
  readonly kind: SnapshotMismatchKind;
  readonly expected?: string;
  readonly actual?: string;
  readonly delta?: number;
  readonly tolerance?: number;
}

export interface SnapshotComparison {
  readonly ok: boolean;
  readonly mismatches: readonly SnapshotMismatch[];
}

export interface SnapshotCompareOptions {
  readonly absoluteTolerance?: number;
  readonly relativeTolerance?: number;
  readonly allowExtraKeys?: boolean;
}

export interface SnapshotOracle {
  compare(actual: unknown): SnapshotComparison;
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SnapshotError";
  }
}

export function createSnapshotOracle(
  expected: unknown,
  options: SnapshotCompareOptions = {},
): SnapshotOracle {
  validateOptions(options);
  return {
    compare(actual) {
      return compareSnapshot(actual, expected, options);
    },
  };
}

export function compareSnapshot(
  actual: unknown,
  expected: unknown,
  options: SnapshotCompareOptions = {},
): SnapshotComparison {
  validateOptions(options);
  const mismatches: SnapshotMismatch[] = [];
  compareValue(actual, expected, "$", normalizedOptions(options), mismatches);
  return {
    ok: mismatches.length === 0,
    mismatches,
  };
}

interface NormalizedOptions {
  readonly absoluteTolerance: number;
  readonly relativeTolerance: number;
  readonly allowExtraKeys: boolean;
}

function normalizedOptions(options: SnapshotCompareOptions): NormalizedOptions {
  return {
    absoluteTolerance: options.absoluteTolerance ?? 0,
    relativeTolerance: options.relativeTolerance ?? 0,
    allowExtraKeys: options.allowExtraKeys ?? false,
  };
}

function validateOptions(options: SnapshotCompareOptions): void {
  validateTolerance(options.absoluteTolerance, "absoluteTolerance");
  validateTolerance(options.relativeTolerance, "relativeTolerance");
}

function validateTolerance(value: number | undefined, name: string): void {
  if (value === undefined) return;
  if (!Number.isFinite(value) || value < 0) {
    throw new SnapshotError(`${name} must be a non-negative finite number`);
  }
}

function compareValue(
  actual: unknown,
  expected: unknown,
  path: string,
  options: NormalizedOptions,
  mismatches: SnapshotMismatch[],
): void {
  if (expected === undefined) {
    if (actual !== undefined && !options.allowExtraKeys) {
      mismatches.push({ path, kind: "unexpected", actual: repr(actual) });
    }
    return;
  }
  if (actual === undefined) {
    mismatches.push({ path, kind: "missing", expected: repr(expected) });
    return;
  }

  if (typeof expected === "number" || typeof actual === "number") {
    compareNumber(actual, expected, path, options, mismatches);
    return;
  }

  if (Array.isArray(expected) || Array.isArray(actual)) {
    compareArray(actual, expected, path, options, mismatches);
    return;
  }

  if (isRecord(expected) || isRecord(actual)) {
    compareRecord(actual, expected, path, options, mismatches);
    return;
  }

  if (actual !== expected) {
    mismatches.push({
      path,
      kind: typeof actual === typeof expected ? "value" : "type",
      expected: repr(expected),
      actual: repr(actual),
    });
  }
}

function compareNumber(
  actual: unknown,
  expected: unknown,
  path: string,
  options: NormalizedOptions,
  mismatches: SnapshotMismatch[],
): void {
  if (typeof expected !== "number" || typeof actual !== "number") {
    mismatches.push({
      path,
      kind: "type",
      expected: repr(expected),
      actual: repr(actual),
    });
    return;
  }
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    if (Object.is(actual, expected)) return;
    mismatches.push({
      path,
      kind: "non-finite",
      expected: repr(expected),
      actual: repr(actual),
    });
    return;
  }
  const delta = Math.abs(actual - expected);
  const tolerance = Math.max(
    options.absoluteTolerance,
    Math.abs(expected) * options.relativeTolerance,
  );
  if (delta > tolerance) {
    mismatches.push({
      path,
      kind: "value",
      expected: repr(expected),
      actual: repr(actual),
      delta,
      tolerance,
    });
  }
}

function compareArray(
  actual: unknown,
  expected: unknown,
  path: string,
  options: NormalizedOptions,
  mismatches: SnapshotMismatch[],
): void {
  if (!Array.isArray(actual) || !Array.isArray(expected)) {
    mismatches.push({
      path,
      kind: "type",
      expected: repr(expected),
      actual: repr(actual),
    });
    return;
  }
  if (actual.length !== expected.length) {
    mismatches.push({
      path: `${path}.length`,
      kind: "shape",
      expected: String(expected.length),
      actual: String(actual.length),
    });
  }
  const length = Math.min(actual.length, expected.length);
  for (let i = 0; i < length; i++) {
    compareValue(actual[i], expected[i], `${path}[${i}]`, options, mismatches);
  }
}

function compareRecord(
  actual: unknown,
  expected: unknown,
  path: string,
  options: NormalizedOptions,
  mismatches: SnapshotMismatch[],
): void {
  if (!isRecord(actual) || !isRecord(expected)) {
    mismatches.push({
      path,
      kind: "type",
      expected: repr(expected),
      actual: repr(actual),
    });
    return;
  }

  for (const key of Object.keys(expected).sort()) {
    compareValue(
      actual[key],
      expected[key],
      `${path}.${key}`,
      options,
      mismatches,
    );
  }
  if (!options.allowExtraKeys) {
    for (const key of Object.keys(actual).sort()) {
      if (!(key in expected)) {
        compareValue(actual[key], undefined, `${path}.${key}`, options, mismatches);
      }
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function repr(value: unknown): string {
  if (typeof value === "number" && Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  const json = JSON.stringify(value);
  if (json !== undefined) return json;
  return String(value);
}
