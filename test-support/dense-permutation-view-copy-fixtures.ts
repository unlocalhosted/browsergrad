import fixtureJson from "../packages/browsergrad-semantic-core/fixtures/kernel-v1/dense-permutation-view-copy.cases.json";

export const DENSE_PERMUTATION_FIXTURE_SCHEMA =
  "browsergrad.semantic-core.dense-permutation-view-copy-fixtures" as const;

export interface DensePermutationFixtureRequest {
  readonly kind: "dense-permutation-view-copy";
  readonly inputShape: readonly string[];
  readonly axes: readonly number[];
  readonly dtype: "f32";
}

export interface DensePermutationFixtureCase {
  readonly id: string;
  readonly request: DensePermutationFixtureRequest;
  readonly outputShape: readonly string[];
  readonly sourceWords: readonly string[];
  readonly expectedOutputWords: readonly string[];
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
}

export interface DensePermutationFixtureEnvelope {
  readonly schema: typeof DENSE_PERMUTATION_FIXTURE_SCHEMA;
  readonly version: Readonly<{ readonly major: 1; readonly minor: 0 }>;
  readonly cases: readonly DensePermutationFixtureCase[];
}

export const DENSE_PERMUTATION_VIEW_COPY_FIXTURES =
  decodeDensePermutationViewCopyFixtures(fixtureJson);

export function decodeDensePermutationViewCopyFixtures(
  input: unknown,
): DensePermutationFixtureEnvelope {
  const envelope = record(input, "$fixture");
  exactKeys(envelope, ["cases", "schema", "version"], "$fixture");
  equal(envelope.schema, DENSE_PERMUTATION_FIXTURE_SCHEMA, "$fixture.schema");
  const version = record(envelope.version, "$fixture.version");
  exactKeys(version, ["major", "minor"], "$fixture.version");
  equal(version.major, 1, "$fixture.version.major");
  equal(version.minor, 0, "$fixture.version.minor");
  if (!Array.isArray(envelope.cases) || envelope.cases.length === 0 || envelope.cases.length > 32) {
    fail("$fixture.cases", "must contain 1..32 cases");
  }
  const ids = new Set<string>();
  const cases = envelope.cases.map((value, index) => decodeCase(value, index, ids));
  return Object.freeze({
    schema: DENSE_PERMUTATION_FIXTURE_SCHEMA,
    version: Object.freeze({ major: 1, minor: 0 }),
    cases: Object.freeze(cases),
  });
}

export function fixtureExtentNumbers(extents: readonly string[]): readonly number[] {
  return Object.freeze(extents.map((extent, index) => {
    if (!/^[1-9][0-9]*$/u.test(extent)) fail(`$extents[${index}]`, "must be a canonical positive integer");
    const value = Number(extent);
    if (!Number.isSafeInteger(value)) fail(`$extents[${index}]`, "must fit a safe integer");
    return value;
  }));
}

export function fixtureWords(words: readonly string[]): Uint32Array {
  return Uint32Array.from(words, (word) => Number.parseInt(word, 16));
}

function decodeCase(
  input: unknown,
  index: number,
  ids: Set<string>,
): DensePermutationFixtureCase {
  const path = `$fixture.cases[${index}]`;
  const value = record(input, path);
  exactKeys(value, [
    "expectedOutputWords",
    "id",
    "kernelSemanticHash",
    "layoutSemanticHash",
    "outputShape",
    "request",
    "sourceWords",
  ], path);
  const id = nonemptyString(value.id, `${path}.id`);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/u.test(id)) fail(`${path}.id`, "must be a kebab-case identifier");
  if (ids.has(id)) fail(`${path}.id`, `duplicates ${id}`);
  ids.add(id);
  const request = decodeRequest(value.request, `${path}.request`);
  const outputShape = canonicalShape(value.outputShape, `${path}.outputShape`);
  const derivedOutput = request.axes.map((axis) => request.inputShape[axis]!);
  if (!arrayEqual(outputShape, derivedOutput)) fail(`${path}.outputShape`, "does not match the permutation");
  const sourceWords = hexWords(value.sourceWords, `${path}.sourceWords`);
  const expectedOutputWords = hexWords(value.expectedOutputWords, `${path}.expectedOutputWords`);
  const inputElements = elementCount(request.inputShape, `${path}.request.inputShape`);
  const outputElements = elementCount(outputShape, `${path}.outputShape`);
  if (sourceWords.length !== inputElements) fail(`${path}.sourceWords`, `must contain ${inputElements} words`);
  if (expectedOutputWords.length !== outputElements) fail(`${path}.expectedOutputWords`, `must contain ${outputElements} words`);
  assertExpectedPermutation(request, sourceWords, expectedOutputWords, path);
  return Object.freeze({
    id,
    request,
    outputShape,
    sourceWords,
    expectedOutputWords,
    layoutSemanticHash: digest(value.layoutSemanticHash, `${path}.layoutSemanticHash`),
    kernelSemanticHash: digest(value.kernelSemanticHash, `${path}.kernelSemanticHash`),
  });
}

function decodeRequest(input: unknown, path: string): DensePermutationFixtureRequest {
  const request = record(input, path);
  exactKeys(request, ["axes", "dtype", "inputShape", "kind"], path);
  equal(request.kind, "dense-permutation-view-copy", `${path}.kind`);
  equal(request.dtype, "f32", `${path}.dtype`);
  const inputShape = canonicalShape(request.inputShape, `${path}.inputShape`);
  if (inputShape.length !== 2 && inputShape.length !== 3) fail(`${path}.inputShape`, "must have rank 2 or 3");
  if (!Array.isArray(request.axes) || request.axes.length !== inputShape.length) {
    fail(`${path}.axes`, "must match input rank");
  }
  const axes = request.axes.map((axis, index) => {
    if (!Number.isInteger(axis) || (axis as number) < 0 || (axis as number) >= inputShape.length) {
      fail(`${path}.axes[${index}]`, "must be an in-range integer");
    }
    return axis as number;
  });
  if (new Set(axes).size !== axes.length) fail(`${path}.axes`, "must be an exact permutation");
  return Object.freeze({
    kind: "dense-permutation-view-copy",
    inputShape,
    axes: Object.freeze(axes),
    dtype: "f32",
  });
}

function canonicalShape(input: unknown, path: string): readonly string[] {
  if (!Array.isArray(input) || input.length === 0 || input.length > 8) fail(path, "must contain 1..8 extents");
  return Object.freeze(input.map((extent, index) => {
    if (typeof extent !== "string" || !/^[1-9][0-9]*$/u.test(extent)) {
      fail(`${path}[${index}]`, "must be a canonical positive integer");
    }
    return extent;
  }));
}

function hexWords(input: unknown, path: string): readonly string[] {
  if (!Array.isArray(input) || input.length === 0 || input.length > 1_000_000) {
    fail(path, "must contain 1..1000000 words");
  }
  return Object.freeze(input.map((word, index) => {
    if (typeof word !== "string" || !/^[0-9a-f]{8}$/u.test(word)) {
      fail(`${path}[${index}]`, "must be eight lowercase hexadecimal digits");
    }
    return word;
  }));
}

function assertExpectedPermutation(
  request: DensePermutationFixtureRequest,
  sourceWords: readonly string[],
  expectedWords: readonly string[],
  path: string,
): void {
  const inputShape = fixtureExtentNumbers(request.inputShape);
  const outputShape = request.axes.map((axis) => inputShape[axis]!);
  for (let outputFlat = 0; outputFlat < expectedWords.length; outputFlat += 1) {
    const outputCoordinates = unflatten(outputFlat, outputShape);
    const inputCoordinates = Array<number>(inputShape.length).fill(0);
    request.axes.forEach((inputAxis, outputAxis) => {
      inputCoordinates[inputAxis] = outputCoordinates[outputAxis]!;
    });
    const inputFlat = flatten(inputCoordinates, inputShape);
    if (expectedWords[outputFlat] !== sourceWords[inputFlat]) {
      fail(`${path}.expectedOutputWords[${outputFlat}]`, `must copy source word ${inputFlat}`);
    }
  }
}

function unflatten(flat: number, shape: readonly number[]): readonly number[] {
  const coordinates = Array<number>(shape.length);
  let remaining = flat;
  for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
    coordinates[axis] = remaining % shape[axis]!;
    remaining = Math.floor(remaining / shape[axis]!);
  }
  return coordinates;
}

function flatten(coordinates: readonly number[], shape: readonly number[]): number {
  let flat = 0;
  for (let axis = 0; axis < shape.length; axis += 1) flat = flat * shape[axis]! + coordinates[axis]!;
  return flat;
}

function elementCount(shape: readonly string[], path: string): number {
  let count = 1;
  for (const [index, extent] of fixtureExtentNumbers(shape).entries()) {
    count *= extent;
    if (!Number.isSafeInteger(count) || count > 1_000_000) fail(`${path}[${index}]`, "exceeds fixture element budget");
  }
  return count;
}

function digest(input: unknown, path: string): string {
  if (typeof input !== "string" || !/^[0-9a-f]{64}$/u.test(input)) fail(path, "must be a SHA-256 digest");
  return input;
}

function nonemptyString(input: unknown, path: string): string {
  if (typeof input !== "string" || input.length === 0 || input.length > 128) fail(path, "must be a nonempty bounded string");
  return input;
}

function record(input: unknown, path: string): Record<string, unknown> {
  if (input === null || typeof input !== "object" || Array.isArray(input)) fail(path, "must be an object");
  return input as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[], path: string): void {
  const actual = Object.keys(value).sort();
  if (!arrayEqual(actual, [...expected].sort())) fail(path, `expected closed fields ${[...expected].sort().join(", ")}`);
}

function arrayEqual(left: readonly unknown[], right: readonly unknown[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function equal(actual: unknown, expected: unknown, path: string): void {
  if (actual !== expected) fail(path, `must equal ${String(expected)}`);
}

function fail(path: string, message: string): never {
  throw new Error(`invalid dense-permutation fixture at ${path}: ${message}`);
}
