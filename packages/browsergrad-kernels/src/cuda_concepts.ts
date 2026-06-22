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
  readOutput(index: number): number;
  write(index: number, value: number): void;
}

export interface Cuda1DThreadTrace {
  readonly globalThreadId: number;
  readonly blockIdxX: number;
  readonly threadIdxX: number;
  readonly reads: readonly Cuda1DMemoryAccess[];
  readonly outputReads: readonly Cuda1DMemoryAccess[];
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

export type ThreadGridInput = Cuda1DGridInput;
export type ThreadGridKernel = Cuda1DKernel;
export type ThreadGridContext = Cuda1DThreadContext;
export type ThreadGridTrace = Cuda1DThreadTrace;
export type ThreadGridMemoryAccess = Cuda1DMemoryAccess;
export type ThreadGridViolation = Cuda1DViolation;
export type ThreadGridStats = Cuda1DGridStats;
export type ThreadGridResult = Cuda1DGridResult;

export interface SaxpyInput {
  readonly a: number;
  readonly x: readonly number[];
  readonly y: readonly number[];
}

export type RgbColor = readonly [number, number, number];
export type Point2D = readonly [number, number];

export interface CirclePrimitive {
  readonly center: Point2D;
  readonly radius: number;
  readonly color: RgbColor;
  readonly alpha: number;
}

export interface OrderedCircleRenderInput {
  readonly width: number;
  readonly height: number;
  readonly background?: RgbColor;
  readonly circles: readonly CirclePrimitive[];
}

export interface OrderedCircleRenderResult {
  readonly width: number;
  readonly height: number;
  readonly pixels: readonly number[];
}

export function runThreadGrid(input: ThreadGridInput): ThreadGridResult {
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
      const outputReads: Cuda1DMemoryAccess[] = [];
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
        readOutput(index) {
          const normalized = validateMemoryIndex(index, "output read index");
          if (normalized < 0 || normalized >= outputLength) {
            outputReads.push({ index: normalized, ok: false });
            return 0;
          }
          const value = output[normalized]!;
          outputReads.push({ index: normalized, ok: true, value });
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
        outputReads,
        writes,
      });
    }
  }

  return {
    output: [...output],
    stats: {
      launchedThreads: blocks * threadsPerBlock,
      activeThreads: trace.filter(
        (event) =>
          event.reads.length > 0 ||
          event.outputReads.length > 0 ||
          event.writes.length > 0,
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

/**
 * @deprecated Use `runThreadGrid()`. This alias remains for CUDA-shaped
 * teaching labs and older GPU Puzzles rubrics.
 */
export function simulateCuda1DGrid(input: Cuda1DGridInput): Cuda1DGridResult {
  return runThreadGrid(input);
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

export function referenceFindRepeats(values: readonly number[]): number[] {
  const input = validateNumberList(values, "values");
  const repeatedAt: number[] = [];
  for (let index = 0; index < input.length - 1; index++) {
    if (input[index] === input[index + 1]) {
      repeatedAt.push(index);
    }
  }
  return repeatedAt;
}

export function referenceOrderedCircleRender(
  input: OrderedCircleRenderInput,
): OrderedCircleRenderResult {
  const width = validatePositiveInteger(input.width, "width");
  const height = validatePositiveInteger(input.height, "height");
  const background = validateRgbColor(input.background ?? [0, 0, 0], "background");
  const pixels = Array.from({ length: width * height }, () => [...background]);
  const circles = input.circles.map(validateCirclePrimitive);

  for (const circle of circles) {
    const [cx, cy] = circle.center;
    const radiusSquared = circle.radius * circle.radius;
    for (let y = 0; y < height; y++) {
      const py = (y + 0.5) / height;
      for (let x = 0; x < width; x++) {
        const px = (x + 0.5) / width;
        const dx = px - cx;
        const dy = py - cy;
        if (dx * dx + dy * dy > radiusSquared) continue;
        const pixel = pixels[y * width + x]!;
        pixel[0] = blendChannel(pixel[0]!, circle.color[0], circle.alpha);
        pixel[1] = blendChannel(pixel[1]!, circle.color[1], circle.alpha);
        pixel[2] = blendChannel(pixel[2]!, circle.color[2], circle.alpha);
      }
    }
  }

  return {
    width,
    height,
    pixels: pixels.flat(),
  };
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

function validateRgbColor(value: RgbColor, name: string): RgbColor {
  if (value.length !== 3) {
    throw new KernelError(`${name} must have exactly 3 channels`);
  }
  return [
    validateFiniteNumber(value[0], `${name}[0]`),
    validateFiniteNumber(value[1], `${name}[1]`),
    validateFiniteNumber(value[2], `${name}[2]`),
  ];
}

function validatePoint2D(value: Point2D, name: string): Point2D {
  if (value.length !== 2) {
    throw new KernelError(`${name} must have exactly 2 coordinates`);
  }
  return [
    validateFiniteNumber(value[0], `${name}[0]`),
    validateFiniteNumber(value[1], `${name}[1]`),
  ];
}

function validateAlpha(value: number, name: string): number {
  const alpha = validateFiniteNumber(value, name);
  if (alpha < 0 || alpha > 1) {
    throw new KernelError(`${name} must be between 0 and 1`);
  }
  return alpha;
}

function validateCirclePrimitive(circle: CirclePrimitive, index: number): CirclePrimitive {
  const radius = validateFiniteNumber(circle.radius, `circles[${index}].radius`);
  if (radius < 0) {
    throw new KernelError(`circles[${index}].radius must be non-negative`);
  }
  return {
    center: validatePoint2D(circle.center, `circles[${index}].center`),
    radius,
    color: validateRgbColor(circle.color, `circles[${index}].color`),
    alpha: validateAlpha(circle.alpha, `circles[${index}].alpha`),
  };
}

function blendChannel(base: number, over: number, alpha: number): number {
  return alpha * over + (1 - alpha) * base;
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
    outputReads: event.outputReads.map((access) => ({ ...access })),
    writes: event.writes.map((access) => ({ ...access })),
  };
}
