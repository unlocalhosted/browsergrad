import { type Tensor } from "./types.js";

export interface KernelRubricOptions {
  readonly assertPass?: (name: string) => void;
  readonly assertFail?: (
    name: string,
    message: string,
    details?: KernelRubricFailureDetails,
  ) => void;
}

export interface KernelCloseOptions {
  readonly atol?: number;
  readonly rtol?: number;
  readonly maxPreview?: number;
}

export type KernelRubricAssertion =
  | {
      readonly kind: "pass";
      readonly name: string;
    }
  | {
      readonly kind: "fail";
      readonly name: string;
      readonly message: string;
      readonly details?: KernelRubricFailureDetails;
    };

export interface KernelRubricFailureDetails {
  readonly actualShape?: readonly number[];
  readonly expectedShape?: readonly number[];
  readonly actualPreview?: readonly number[];
  readonly expectedPreview?: readonly number[];
  readonly mismatchIndex?: number;
  readonly actualValue?: number;
  readonly expectedValue?: number;
  readonly maxAbsDiff?: number;
  readonly atol?: number;
  readonly rtol?: number;
}

export interface KernelRubric {
  readonly assertions: readonly KernelRubricAssertion[];
  pass(name: string): void;
  fail(
    name: string,
    message: string,
    details?: KernelRubricFailureDetails,
  ): void;
  assertCloseTensor(
    name: string,
    actual: Tensor,
    expected: Tensor,
    options?: KernelCloseOptions,
  ): void;
}

const DEFAULT_ATOL = 1e-5;
const DEFAULT_RTOL = 1e-6;
const DEFAULT_MAX_PREVIEW = 8;

export function createKernelRubric(
  options: KernelRubricOptions = {},
): KernelRubric {
  const assertions: KernelRubricAssertion[] = [];

  function pass(name: string): void {
    assertions.push({ kind: "pass", name });
    options.assertPass?.(name);
  }

  function fail(
    name: string,
    message: string,
    details?: KernelRubricFailureDetails,
  ): void {
    assertions.push({
      kind: "fail",
      name,
      message,
      ...(details ? { details } : {}),
    });
    options.assertFail?.(name, message, details);
  }

  return {
    get assertions() {
      return assertions;
    },
    pass,
    fail,
    assertCloseTensor(name, actual, expected, closeOptions = {}) {
      const previewLength = normalizedPreviewLength(closeOptions.maxPreview);
      const actualPreview = preview(actual.data, previewLength);
      const expectedPreview = preview(expected.data, previewLength);
      if (!sameShape(actual.shape, expected.shape)) {
        fail(name, "tensor shape mismatch", {
          actualShape: [...actual.shape],
          expectedShape: [...expected.shape],
          actualPreview,
          expectedPreview,
        });
        return;
      }
      if (actual.data.length !== expected.data.length) {
        fail(name, "tensor data length mismatch", {
          actualShape: [...actual.shape],
          expectedShape: [...expected.shape],
          actualPreview,
          expectedPreview,
        });
        return;
      }

      const atol = closeOptions.atol ?? DEFAULT_ATOL;
      const rtol = closeOptions.rtol ?? DEFAULT_RTOL;
      const mismatch = firstTensorMismatch(actual.data, expected.data, atol, rtol);
      if (!mismatch) {
        pass(name);
        return;
      }

      fail(name, "tensor values differ beyond tolerance", {
        actualShape: [...actual.shape],
        expectedShape: [...expected.shape],
        actualPreview,
        expectedPreview,
        mismatchIndex: mismatch.index,
        actualValue: mismatch.actual,
        expectedValue: mismatch.expected,
        maxAbsDiff: mismatch.maxAbsDiff,
        atol,
        rtol,
      });
    },
  };
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length &&
    actual.every((dim, index) => dim === expected[index]);
}

function preview(data: Float32Array, maxPreview: number): number[] {
  return Array.from(data.subarray(0, maxPreview));
}

function normalizedPreviewLength(value: number | undefined): number {
  if (value === undefined) return DEFAULT_MAX_PREVIEW;
  return Math.max(0, Math.floor(value));
}

function firstTensorMismatch(
  actual: Float32Array,
  expected: Float32Array,
  atol: number,
  rtol: number,
): {
  readonly index: number;
  readonly actual: number;
  readonly expected: number;
  readonly maxAbsDiff: number;
} | undefined {
  let first: {
    readonly index: number;
    readonly actual: number;
    readonly expected: number;
  } | undefined;
  let maxAbsDiff = 0;

  for (let i = 0; i < actual.length; i++) {
    const actualValue = actual[i]!;
    const expectedValue = expected[i]!;
    const finitePair = Number.isFinite(actualValue) && Number.isFinite(expectedValue);
    const absDiff = finitePair
      ? Math.abs(actualValue - expectedValue)
      : Number.POSITIVE_INFINITY;
    maxAbsDiff = Math.max(maxAbsDiff, absDiff);
    const tolerance = atol + rtol * Math.abs(expectedValue);
    if (!first && (!finitePair || absDiff > tolerance)) {
      first = { index: i, actual: actualValue, expected: expectedValue };
    }
  }

  if (!first) return undefined;
  return { ...first, maxAbsDiff };
}
