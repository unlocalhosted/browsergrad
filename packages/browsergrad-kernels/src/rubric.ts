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

export interface KernelAssertionDetails {
  readonly expected?: string;
  readonly actual?: string;
}

export interface BrowsergradKernelAssertionTarget {
  assertPass(name: string): void;
  assertFail(
    name: string,
    message: string,
    details?: KernelAssertionDetails,
  ): void;
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

export function kernelRubricFailureToAssertionDetails(
  details?: KernelRubricFailureDetails,
): KernelAssertionDetails | undefined {
  if (!details) return undefined;
  return {
    actual: formatKernelSide({
      shape: details.actualShape,
      preview: details.actualPreview,
      mismatchIndex: details.mismatchIndex,
      value: details.actualValue,
      maxAbsDiff: details.maxAbsDiff,
    }),
    expected: formatKernelSide({
      shape: details.expectedShape,
      preview: details.expectedPreview,
      mismatchIndex: details.mismatchIndex,
      value: details.expectedValue,
      atol: details.atol,
      rtol: details.rtol,
    }),
  };
}

export function createBrowsergradKernelRubric(
  target: BrowsergradKernelAssertionTarget,
): KernelRubric {
  return createKernelRubric({
    assertPass: (name) => target.assertPass(name),
    assertFail: (name, message, details) => {
      target.assertFail(
        name,
        message,
        kernelRubricFailureToAssertionDetails(details),
      );
    },
  });
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

function formatKernelSide(input: {
  readonly shape?: readonly number[] | undefined;
  readonly preview?: readonly number[] | undefined;
  readonly mismatchIndex?: number | undefined;
  readonly value?: number | undefined;
  readonly maxAbsDiff?: number | undefined;
  readonly atol?: number | undefined;
  readonly rtol?: number | undefined;
}): string {
  const parts: string[] = [];
  if (input.shape) parts.push(`shape=${formatArray(input.shape)}`);
  if (input.preview) parts.push(`preview=${formatArray(input.preview)}`);
  if (input.mismatchIndex !== undefined && input.value !== undefined) {
    parts.push(`value[${input.mismatchIndex}]=${formatNumber(input.value)}`);
  }
  if (input.maxAbsDiff !== undefined) {
    parts.push(`maxAbsDiff=${formatNumber(input.maxAbsDiff)}`);
  }
  if (input.atol !== undefined) parts.push(`atol=${formatNumber(input.atol)}`);
  if (input.rtol !== undefined) parts.push(`rtol=${formatNumber(input.rtol)}`);
  return parts.join(" ");
}

function formatArray(values: readonly number[]): string {
  return `[${values.map(formatNumber).join(",")}]`;
}

function formatNumber(value: number): string {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  return String(value);
}
