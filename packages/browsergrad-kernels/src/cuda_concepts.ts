import { KernelError } from "./types.js";

export interface Cuda1DGridInput {
  readonly inputLength: number;
  readonly outputLength: number;
  readonly threadsPerBlock: number;
  readonly blocks: number;
  readonly initialInput?: readonly number[];
  readonly initialOutput?: readonly number[];
  readonly kernel: Cuda1DKernel;
}

export type Cuda1DKernel = (context: Cuda1DThreadContext) => void;

export interface Cuda1DThreadContext {
  readonly blockIdxX: number;
  readonly blockDimX: number;
  readonly threadIdxX: number;
  readonly globalThreadId: number;
  readonly inputLength: number;
  readonly outputLength: number;
  read(index: number): number;
  write(index: number, value: number): void;
}

export interface Cuda1DThreadTrace {
  readonly globalThreadId: number;
  readonly blockIdxX: number;
  readonly threadIdxX: number;
  readonly reads: readonly Cuda1DMemoryAccess[];
  readonly writes: readonly Cuda1DMemoryAccess[];
}

export interface Cuda1DMemoryAccess {
  readonly index: number;
  readonly ok: boolean;
  readonly value?: number;
}

export interface Cuda1DViolation {
  readonly kind: "read-oob" | "write-oob";
  readonly globalThreadId: number;
  readonly index: number;
}

export interface Cuda1DGridStats {
  readonly launchedThreads: number;
  readonly activeThreads: number;
  readonly globalReads: number;
  readonly globalWrites: number;
  readonly outOfBoundsReads: number;
  readonly outOfBoundsWrites: number;
}

export interface Cuda1DGridResult {
  readonly output: readonly number[];
  readonly stats: Cuda1DGridStats;
  readonly trace: readonly Cuda1DThreadTrace[];
  readonly violations: readonly Cuda1DViolation[];
}

export interface SaxpyInput {
  readonly a: number;
  readonly x: readonly number[];
  readonly y: readonly number[];
}

export function simulateCuda1DGrid(input: Cuda1DGridInput): Cuda1DGridResult {
  const inputLength = validateNonNegativeInteger(input.inputLength, "inputLength");
  const outputLength = validateNonNegativeInteger(input.outputLength, "outputLength");
  const threadsPerBlock = validatePositiveInteger(
    input.threadsPerBlock,
    "threadsPerBlock",
  );
  const blocks = validatePositiveInteger(input.blocks, "blocks");
  const initialInput = materializeVector(input.initialInput, inputLength, "initialInput");
  const output = materializeVector(input.initialOutput, outputLength, "initialOutput");
  const trace: Cuda1DThreadTrace[] = [];
  const violations: Cuda1DViolation[] = [];

  for (let blockIdxX = 0; blockIdxX < blocks; blockIdxX++) {
    for (let threadIdxX = 0; threadIdxX < threadsPerBlock; threadIdxX++) {
      const globalThreadId = blockIdxX * threadsPerBlock + threadIdxX;
      const reads: Cuda1DMemoryAccess[] = [];
      const writes: Cuda1DMemoryAccess[] = [];
      const context: Cuda1DThreadContext = {
        blockIdxX,
        blockDimX: threadsPerBlock,
        threadIdxX,
        globalThreadId,
        inputLength,
        outputLength,
        read(index) {
          const normalized = validateMemoryIndex(index, "read index");
          if (normalized < 0 || normalized >= inputLength) {
            reads.push({ index: normalized, ok: false });
            violations.push({
              kind: "read-oob",
              globalThreadId,
              index: normalized,
            });
            return 0;
          }
          const value = initialInput[normalized]!;
          reads.push({ index: normalized, ok: true, value });
          return value;
        },
        write(index, value) {
          const normalized = validateMemoryIndex(index, "write index");
          const finiteValue = validateFiniteNumber(value, "write value");
          if (normalized < 0 || normalized >= outputLength) {
            writes.push({ index: normalized, ok: false, value: finiteValue });
            violations.push({
              kind: "write-oob",
              globalThreadId,
              index: normalized,
            });
            return;
          }
          output[normalized] = finiteValue;
          writes.push({ index: normalized, ok: true, value: finiteValue });
        },
      };

      input.kernel(context);
      trace.push({
        globalThreadId,
        blockIdxX,
        threadIdxX,
        reads,
        writes,
      });
    }
  }

  return {
    output: [...output],
    stats: {
      launchedThreads: blocks * threadsPerBlock,
      activeThreads: trace.filter(
        (event) => event.reads.length > 0 || event.writes.length > 0,
      ).length,
      globalReads: trace.reduce(
        (count, event) => count + event.reads.filter((access) => access.ok).length,
        0,
      ),
      globalWrites: trace.reduce(
        (count, event) => count + event.writes.filter((access) => access.ok).length,
        0,
      ),
      outOfBoundsReads: violations.filter((violation) => violation.kind === "read-oob")
        .length,
      outOfBoundsWrites: violations.filter(
        (violation) => violation.kind === "write-oob",
      ).length,
    },
    trace: trace.map(cloneTrace),
    violations: violations.map((violation) => ({ ...violation })),
  };
}

export function referenceSaxpy(input: SaxpyInput): number[] {
  const a = validateFiniteNumber(input.a, "a");
  const x = validateNumberList(input.x, "x");
  const y = validateNumberList(input.y, "y");
  if (x.length !== y.length) {
    throw new KernelError("saxpy: x and y must have the same length");
  }
  return x.map((value, index) => a * value + y[index]!);
}

export function referenceExclusiveScan(values: readonly number[]): number[] {
  const input = validateNumberList(values, "values");
  const out: number[] = [];
  let running = 0;
  for (const value of input) {
    out.push(running);
    running += value;
  }
  return out;
}

function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`${name} must be a positive integer`);
  }
  return value;
}

function validateNonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new KernelError(`${name} must be a non-negative integer`);
  }
  return value;
}

function validateMemoryIndex(value: number, name: string): number {
  if (!Number.isInteger(value)) {
    throw new KernelError(`${name} must be an integer`);
  }
  return value;
}

function validateFiniteNumber(value: number, name: string): number {
  if (!Number.isFinite(value)) {
    throw new KernelError(`${name} must be finite`);
  }
  return value;
}

function validateNumberList(values: readonly number[], name: string): number[] {
  return values.map((value, index) =>
    validateFiniteNumber(value, `${name}[${index}]`),
  );
}

function materializeVector(
  values: readonly number[] | undefined,
  length: number,
  name: string,
): number[] {
  if (values === undefined) return Array.from({ length }, () => 0);
  if (values.length !== length) {
    throw new KernelError(`${name} length must be ${length}`);
  }
  return validateNumberList(values, name);
}

function cloneTrace(event: Cuda1DThreadTrace): Cuda1DThreadTrace {
  return {
    globalThreadId: event.globalThreadId,
    blockIdxX: event.blockIdxX,
    threadIdxX: event.threadIdxX,
    reads: event.reads.map((access) => ({ ...access })),
    writes: event.writes.map((access) => ({ ...access })),
  };
}
