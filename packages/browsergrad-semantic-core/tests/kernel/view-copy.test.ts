import { describe, expect, it } from "vitest";

import {
  KERNEL_ARTIFACT_SCHEMA,
  kernelArtifactPayload,
  prepareViewCopyCpu,
  prepareViewCopySpecialization,
  verifyKernelArtifact,
  type VerifiedKernelArtifact,
} from "../../src/kernel";
import { layoutArtifactPayload, verifyLayoutArtifact, type IndexExpr, type PredicateExpr, type VerifiedLayoutArtifact } from "../../src/layout";
import {
  KERNEL_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  hashSemanticArtifact,
  parseWireI64,
  type WireI64,
} from "../../src/schema";

const TRUE: PredicateExpr = { kind: "bool", value: true };

interface LayoutCase {
  readonly shape: readonly (string | Record<string, unknown>)[];
  readonly sourceLocation: IndexExpr;
  readonly sourcePredicate?: PredicateExpr;
  readonly destinationLocation?: IndexExpr;
  readonly sourceLocationUnit?: "element" | "byte";
  readonly destinationLocationUnit?: "element" | "byte";
  readonly sourceByteOffset?: string | Record<string, unknown>;
  readonly destinationByteOffset?: string | Record<string, unknown>;
  readonly sourceBytes: string | Record<string, unknown>;
  readonly destinationBytes: string | Record<string, unknown>;
  readonly sourceAlias?: string;
  readonly destinationAlias?: string;
  readonly sourceSpace?: Record<string, unknown>;
  readonly destinationSpace?: Record<string, unknown>;
}

async function verifiedLayout(input: LayoutCase): Promise<VerifiedLayoutArtifact> {
  const shape = input.shape.map(dim);
  const envelope = {
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "kernel-tests", version: "1" },
    artifactId: "layout-transport",
    requiredExtensions: [],
    payload: {
      symbols: input.shape.filter((entry): entry is Record<string, unknown> => typeof entry === "object")
        .flatMap((entry) => entry.kind === "symbol" && typeof entry.id === "string"
          ? [{ id: entry.id, domain: { min: "0", max: "1024" } }]
          : []),
      constraints: [],
      allocations: [
        {
          allocationId: "sourceAllocation",
          byteLength: dim(input.sourceBytes),
          memorySpace: input.sourceSpace ?? { kind: "global" },
          alignmentBytes: 4,
          aliasSetId: input.sourceAlias ?? "sourceAlias",
        },
        {
          allocationId: "destinationAllocation",
          byteLength: dim(input.destinationBytes),
          memorySpace: input.destinationSpace ?? { kind: "global" },
          alignmentBytes: 4,
          aliasSetId: input.destinationAlias ?? "destinationAlias",
        },
      ],
      indexMaps: [
        {
          indexMapId: "sourceMap",
          coordinateRank: shape.length,
          locationUnit: input.sourceLocationUnit ?? "element",
          location: input.sourceLocation,
          inBounds: input.sourcePredicate ?? TRUE,
        },
        {
          indexMapId: "destinationMap",
          coordinateRank: shape.length,
          locationUnit: input.destinationLocationUnit ?? "element",
          location: input.destinationLocation ?? rowMajorLocation(input.shape),
          inBounds: TRUE,
        },
      ],
      views: [
        {
          viewId: "sourceView",
          allocationId: "sourceAllocation",
          dtype: "f32",
          byteOffset: dim(input.sourceByteOffset ?? "0"),
          shape,
          indexMapId: "sourceMap",
          requiredAlignmentBytes: 4,
        },
        {
          viewId: "destinationView",
          allocationId: "destinationAllocation",
          dtype: "f32",
          byteOffset: dim(input.destinationByteOffset ?? "0"),
          shape,
          indexMapId: "destinationMap",
          requiredAlignmentBytes: 4,
        },
      ],
    },
  };
  return verifyLayoutArtifact(JSON.parse(JSON.stringify(envelope)) as unknown);
}

async function verifiedKernel(
  layout: VerifiedLayoutArtifact,
  policy: Record<string, unknown> = { kind: "reject" },
  operationId = "copy",
): Promise<VerifiedKernelArtifact> {
  const payload = layoutArtifactPayload(layout);
  const sourceView = payload.views[0];
  const destinationView = payload.views[1];
  if (sourceView === undefined || destinationView === undefined) throw new Error("fixture views missing");
  return verifyKernelArtifact({
    schema: KERNEL_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: { id: "kernel-tests", version: "1" },
    artifactId: "kernel-transport",
    requiredExtensions: [],
    payload: {
      layoutSemanticHash: await hashSemanticArtifact(layout),
      operations: [{
        operationId,
        kind: "view-copy",
        version: { major: 1, minor: 0 },
        dtype: "f32",
        source: { viewId: sourceView.viewId, access: "read", invalidSource: policy },
        destination: { viewId: destinationView.viewId, access: "write" },
        overlap: { kind: "forbid" },
      }],
    },
  }, { layout });
}

async function prepare(
  layout: VerifiedLayoutArtifact,
  kernel: VerifiedKernelArtifact,
  options: {
    readonly bindings?: Readonly<Record<string, WireI64>>;
    readonly maxElements?: number;
    readonly maxEvaluationSteps?: number;
    readonly maxPreparedBytes?: number;
    readonly maxPreparationMs?: number;
    readonly signal?: AbortSignal;
  } = {},
) {
  const operationId = kernelArtifactPayload(kernel).operations[0]?.operationId;
  if (operationId === undefined) throw new Error("fixture operation missing");
  return prepareViewCopyCpu(layout, kernel, { operationId, ...options });
}

function dim(value: string | Record<string, unknown>): Record<string, unknown> {
  return typeof value === "string" ? { kind: "const", value } : value;
}

function c(axis: number): IndexExpr {
  return { kind: "coordinate", axis };
}

function k(value: string): IndexExpr {
  return { kind: "const", value: parseWireI64(value) };
}

function add(...terms: IndexExpr[]): IndexExpr {
  return { kind: "add", terms };
}

function mul(lhs: IndexExpr, rhs: IndexExpr): IndexExpr {
  return { kind: "mul", lhs, rhs };
}

function rowMajorLocation(shape: LayoutCase["shape"]): IndexExpr {
  const terms: IndexExpr[] = [];
  for (let axis = 0; axis < shape.length; axis += 1) {
    let stride: IndexExpr = k("1");
    for (let next = axis + 1; next < shape.length; next += 1) {
      const extent = shape[next];
      if (extent === undefined) throw new Error("shape fixture missing extent");
      stride = mul(stride, typeof extent === "string" ? k(extent) : { kind: "dimension", symbolId: String(extent.id) });
    }
    terms.push(mul(c(axis), stride));
  }
  return add(...terms);
}

function f32Bytes(values: readonly number[], leadingBytes = 0): Uint8Array {
  const bytes = new Uint8Array(leadingBytes + (values.length * 4));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  values.forEach((value, index) => view.setFloat32(leadingBytes + (index * 4), value, true));
  return bytes;
}

function f32Values(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true));
}

async function diagnostic(run: () => Promise<unknown> | unknown): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected semantic failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

describe("verified materializing view-copy", () => {
  it("normalizes operation identity and binds the exact verified layout hash", async () => {
    const layout = await verifiedLayout({
      shape: ["3", "2"],
      sourceLocation: add(mul(c(1), k("3")), c(0)),
      sourceBytes: "24",
      destinationBytes: "24",
    });
    const first = await verifiedKernel(layout, { kind: "reject" }, "firstName");
    const second = await verifiedKernel(layout, { kind: "reject" }, "renamedOperation");
    const payload = kernelArtifactPayload(first);
    expect(payload.operations[0]?.operationId).toMatch(/^bg\.entity\.kernel-operation\.scope-sha256\.[0-9a-f]{64}\.ordinal\.0$/u);
    expect(Object.isFrozen(payload)).toBe(true);
    expect(await hashSemanticArtifact(first)).toBe(await hashSemanticArtifact(second));
  });

  it("materializes rank-2 transpose with a nonzero source byte offset", async () => {
    const layout = await verifiedLayout({
      shape: ["3", "2"],
      sourceLocation: add(mul(c(1), k("3")), c(0)),
      sourceByteOffset: "4",
      sourceBytes: "28",
      destinationBytes: "24",
    });
    const plan = await prepare(layout, await verifiedKernel(layout));
    const source = f32Bytes([1, 2, 3, 4, 5, 6], 4);
    const destination = new Uint8Array(24);
    const trace = plan.execute({ source, destination });
    expect(f32Values(destination)).toEqual([1, 4, 2, 5, 3, 6]);
    expect(trace).toMatchObject({ elementCount: "6", readElements: "6", filledElements: "0", bytesRead: "24", bytesWritten: "24" });
    expect(f32Values(source.subarray(4))).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("uses the unchanged operation for rank-3 permutation, strided slice, and broadcast", async () => {
    const cases = [
      {
        layout: {
          shape: ["3", "2", "2"],
          sourceLocation: add(mul(c(1), k("6")), mul(c(0), k("2")), c(2)),
          sourceBytes: "48",
          destinationBytes: "48",
        },
        input: Array.from({ length: 12 }, (_, index) => index + 1),
        expected: [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12],
      },
      {
        layout: {
          shape: ["2", "2"],
          sourceLocation: add(mul(c(0), k("8")), mul(c(1), k("2")), k("1")),
          sourceBytes: "48",
          destinationBytes: "16",
        },
        input: Array.from({ length: 12 }, (_, index) => index + 1),
        expected: [2, 4, 10, 12],
      },
      {
        layout: {
          shape: ["2", "3"],
          sourceLocation: c(1),
          sourceBytes: "12",
          destinationBytes: "24",
        },
        input: [7, 8, 9],
        expected: [7, 8, 9, 7, 8, 9],
      },
    ] as const;
    for (const testCase of cases) {
      const layout = await verifiedLayout(testCase.layout);
      const plan = await prepare(layout, await verifiedKernel(layout));
      const destination = new Uint8Array(Number(BigInt(testCase.layout.destinationBytes)));
      plan.execute({ source: f32Bytes(testCase.input), destination });
      expect(f32Values(destination)).toEqual(testCase.expected);
    }
  });

  it("executes exact rank-2 and rank-3 rectangular prefixes", async () => {
    const cases = [
      {
        shape: ["3", "4"] as const,
        extents: [2n, 3n] as const,
        expectedIndexes: [0, 1, 2, 4, 5, 6],
      },
      {
        shape: ["2", "3", "4"] as const,
        extents: [2n, 2n, 3n] as const,
        expectedIndexes: [0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18],
      },
    ];
    for (const testCase of cases) {
      const elementCount = testCase.shape.reduce(
        (product, extent) => product * Number(extent),
        1,
      );
      const layout = await verifiedLayout({
        shape: testCase.shape,
        sourceLocation: rowMajorLocation(testCase.shape),
        sourceBytes: String(elementCount * 4),
        destinationBytes: String(elementCount * 4),
      });
      const plan = await prepare(layout, await verifiedKernel(layout));
      const sourceValues = Array.from(
        { length: elementCount },
        (_, index) => index + 1,
      );
      const destination = new Uint8Array(elementCount * 4);
      const trace = plan.executeRectangularPrefix(
        { source: f32Bytes(sourceValues), destination },
        [...testCase.extents],
      );
      const expected = Array.from({ length: elementCount }, (_, index) =>
        testCase.expectedIndexes.includes(index)
          ? sourceValues[index] as number
          : 0);

      expect(f32Values(destination)).toEqual(expected);
      expect(trace).toMatchObject({
        elementCount: String(testCase.expectedIndexes.length),
        readElements: String(testCase.expectedIndexes.length),
        bytesWritten: String(testCase.expectedIndexes.length * 4),
      });
    }
  });

  it("rejects hostile rectangular extents before destination mutation", async () => {
    const layout = await verifiedLayout({
      shape: ["3", "4"],
      sourceLocation: rowMajorLocation(["3", "4"]),
      sourceBytes: "48",
      destinationBytes: "48",
    });
    const plan = await prepare(layout, await verifiedKernel(layout));
    const source = f32Bytes(Array.from({ length: 12 }, (_, index) => index));
    const destination = new Uint8Array(48);
    destination.fill(0x5a);
    const before = new Uint8Array(destination);

    for (const extents of [[0n, 3n], [4n, 3n], [2n]] as const) {
      expect((await diagnostic(() => plan.executeRectangularPrefix(
        { source, destination },
        [...extents],
      ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
      expect(destination).toEqual(before);
    }
    let getterReads = 0;
    const accessorExtents = [2n, 3n];
    Object.defineProperty(accessorExtents, "0", {
      enumerable: true,
      get() {
        getterReads += 1;
        return 2n;
      },
    });
    expect((await diagnostic(() => plan.executeRectangularPrefix(
      { source, destination },
      accessorExtents,
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    expect(getterReads).toBe(0);
    expect(destination).toEqual(before);
  });

  it("executes rank-1 and rank-4 positive-affine views under the edge-rank profile", async () => {
    const cases = [
      {
        layout: {
          shape: ["4"],
          sourceLocation: mul(c(0), k("2")),
          sourceBytes: "28",
          destinationBytes: "16",
        },
        input: [1, 2, 3, 4, 5, 6, 7],
        expected: [1, 3, 5, 7],
      },
      {
        layout: {
          shape: ["2", "2", "2", "2"],
          sourceLocation: add(
            c(0),
            mul(c(1), k("2")),
            mul(c(2), k("4")),
            mul(c(3), k("8")),
          ),
          sourceBytes: "64",
          destinationBytes: "64",
        },
        input: Array.from({ length: 16 }, (_, index) => index + 1),
        expected: [1, 9, 5, 13, 3, 11, 7, 15, 2, 10, 6, 14, 4, 12, 8, 16],
      },
    ] as const;
    for (const testCase of cases) {
      const layout = await verifiedLayout(testCase.layout);
      const kernel = await verifiedKernel(layout);
      const plan = await prepare(layout, kernel);
      const specialization = await prepareViewCopySpecialization(
        layout,
        kernel,
        { operationId: plan.operationId },
      );
      const destination = new Uint8Array(Number(BigInt(testCase.layout.destinationBytes)));
      plan.execute({ source: f32Bytes(testCase.input), destination });
      expect(specialization.portableProfile).toMatchObject({
        profileId:
          "browsergrad.view-copy.positive-affine-rank1-rank4-word32@1",
        rank: testCase.layout.shape.length,
        dtype: "f32",
      });
      expect(f32Values(destination)).toEqual(testCase.expected);
    }
  });

  it("guards padded reads and preserves exact f32 fill bits", async () => {
    const predicate: PredicateExpr = {
      kind: "and",
      values: [
        { kind: "lessEqual", lhs: k("1"), rhs: c(0) },
        { kind: "lessEqual", lhs: c(0), rhs: k("2") },
        { kind: "lessEqual", lhs: k("1"), rhs: c(1) },
        { kind: "lessEqual", lhs: c(1), rhs: k("3") },
      ],
    };
    const layout = await verifiedLayout({
      shape: ["4", "5"],
      sourceLocation: add(mul(add(c(0), k("-1")), k("3")), add(c(1), k("-1"))),
      sourcePredicate: predicate,
      sourceBytes: "24",
      destinationBytes: "80",
    });
    const kernel = await verifiedKernel(layout, { kind: "fill", value: { kind: "float-bits", dtype: "f32", bits: "7fc01234" } });
    const plan = await prepare(layout, kernel);
    const destination = new Uint8Array(80);
    const trace = plan.execute({ source: f32Bytes([1, 2, 3, 4, 5, 6]), destination });
    expect(trace).toMatchObject({ elementCount: "20", readElements: "6", filledElements: "14", bytesRead: "24", bytesWritten: "80" });
    const words = new DataView(destination.buffer);
    expect(words.getUint32(0, true)).toBe(0x7fc01234);
    expect(words.getFloat32((1 * 5 + 1) * 4, true)).toBe(1);
    expect(words.getFloat32((2 * 5 + 3) * 4, true)).toBe(6);
    expect(words.getUint32(19 * 4, true)).toBe(0x7fc01234);

    expect((await diagnostic(async () => prepare(layout, await verifiedKernel(layout)))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidAccess);
  });

  it("resolves dynamic rank-2 shapes once and handles zero extent without access", async () => {
    const n = { kind: "symbol", id: "n" };
    const byteLength = { kind: "mul", lhs: n, rhs: { kind: "const", value: "8" } };
    const layout = await verifiedLayout({
      shape: [n, "2"],
      sourceLocation: rowMajorLocation([n, "2"]),
      sourceBytes: byteLength,
      destinationBytes: byteLength,
    });
    const kernel = await verifiedKernel(layout);
    const normal = await prepare(layout, kernel, { bindings: { n: parseWireI64("2") } });
    const output = new Uint8Array(16);
    expect(normal.execute({ source: f32Bytes([1, 2, 3, 4]), destination: output }).elementCount).toBe("4");
    expect(f32Values(output)).toEqual([1, 2, 3, 4]);

    const empty = await prepare(layout, kernel, { bindings: { n: parseWireI64("0") } });
    expect(empty.execute({ source: new Uint8Array(0), destination: new Uint8Array(0) })).toMatchObject({
      elementCount: "0",
      bytesRead: "0",
      bytesWritten: "0",
    });
    const aborted = new AbortController();
    aborted.abort();
    expect((await diagnostic(() => prepare(layout, kernel, {
      bindings: { n: parseWireI64("0") },
      signal: aborted.signal,
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("fails closed on mismatched layout, fields, references, shape, aliasing, profile, and fill", async () => {
    const layout = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: rowMajorLocation(["2", "2"]),
      sourceBytes: "16",
      destinationBytes: "16",
    });
    const payload = layoutArtifactPayload(layout);
    const sourceView = payload.views[0];
    const destinationView = payload.views[1];
    if (sourceView === undefined || destinationView === undefined) throw new Error("fixture views missing");
    const base = {
      schema: KERNEL_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "kernel-tests", version: "1" },
      artifactId: "kernel-negative",
      requiredExtensions: [],
      payload: {
        layoutSemanticHash: await hashSemanticArtifact(layout),
        operations: [{
          operationId: "copy",
          kind: "view-copy",
          version: { major: 1, minor: 0 },
          dtype: "f32",
          source: { viewId: sourceView.viewId, access: "read", invalidSource: { kind: "reject" } },
          destination: { viewId: destinationView.viewId, access: "write" },
          overlap: { kind: "forbid" },
        }],
      },
    };
    const mutations: Array<[string, (value: typeof base) => void, string]> = [
      ["hash", (value) => { value.payload.layoutSemanticHash = "0".repeat(64); }, KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch],
      ["unknown", (value) => { Object.assign(value.payload.operations[0] as object, { backend: "wgsl" }); }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      ["duplicate", (value) => { value.payload.operations.push(structuredClone(value.payload.operations[0] as never)); }, KERNEL_DIAGNOSTIC_CODES.invalidArtifact],
      ["dangling", (value) => { value.payload.operations[0]!.source.viewId = "missing"; }, KERNEL_DIAGNOSTIC_CODES.danglingReference],
      ["version", (value) => { value.payload.operations[0]!.version.minor = 1; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      ["dtype", (value) => { value.payload.operations[0]!.dtype = "f16"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      ["fill", (value) => { Object.assign(value.payload.operations[0]!.source, { invalidSource: { kind: "fill", value: { kind: "float-bits", dtype: "f16", bits: "0000" } } }); }, KERNEL_DIAGNOSTIC_CODES.invalidFill],
    ];
    for (const [, mutate, code] of mutations) {
      const value = structuredClone(base);
      mutate(value);
      expect((await diagnostic(() => verifyKernelArtifact(value, { layout }))).diagnostic.code).toBe(code);
    }

    const aliasedLayout = await verifiedLayout({
      shape: ["2", "2"], sourceLocation: rowMajorLocation(["2", "2"]), sourceBytes: "16", destinationBytes: "16",
      sourceAlias: "same", destinationAlias: "same",
    });
    expect((await diagnostic(() => verifiedKernel(aliasedLayout))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.aliasConflict);

    const rankFive = await verifiedLayout({
      shape: ["1", "1", "1", "1", "1"],
      sourceLocation: rowMajorLocation(["1", "1", "1", "1", "1"]),
      sourceBytes: "4",
      destinationBytes: "4",
    });
    expect((await diagnostic(async () =>
      prepare(rankFive, await verifiedKernel(rankFive))
    )).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);

    const hostSource = await verifiedLayout({
      shape: ["2", "2"], sourceLocation: rowMajorLocation(["2", "2"]), sourceBytes: "16", destinationBytes: "16", sourceSpace: { kind: "host" },
    });
    expect((await diagnostic(async () => prepare(hostSource, await verifiedKernel(hostSource)))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("rejects short and overlapping bindings, excessive work, and cross-layout execution", async () => {
    const layout = await verifiedLayout({
      shape: ["2", "2"], sourceLocation: rowMajorLocation(["2", "2"]), sourceBytes: "16", destinationBytes: "16",
    });
    const kernel = await verifiedKernel(layout);
    const plan = await prepare(layout, kernel);
    expect((await diagnostic(() => plan.execute({ source: new Uint8Array(12), destination: new Uint8Array(16) }))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const shared = new Uint8Array(32);
    expect((await diagnostic(() => plan.execute({ source: shared.subarray(0, 16), destination: shared.subarray(8, 24) }))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.aliasConflict);
    const sharedSourceBuffer = new SharedArrayBuffer(16);
    const sharedDestinationBuffer = structuredClone(sharedSourceBuffer);
    expect(sharedSourceBuffer).not.toBe(sharedDestinationBuffer);
    expect((await diagnostic(() => plan.execute({
      source: new Uint8Array(sharedSourceBuffer),
      destination: new Uint8Array(sharedDestinationBuffer),
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    expect((await diagnostic(() => prepare(layout, kernel, { maxElements: 3 }))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);

    const otherLayout = await verifiedLayout({
      shape: ["2", "2"], sourceLocation: add(rowMajorLocation(["2", "2"]), k("1")), sourceBytes: "20", destinationBytes: "16",
    });
    expect((await diagnostic(() => prepare(otherLayout, kernel))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch);
  });

  it("rejects destination self-overlap before any execution buffer can be mutated", async () => {
    const layout = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: rowMajorLocation(["2", "2"]),
      destinationLocation: k("0"),
      sourceBytes: "16",
      destinationBytes: "16",
    });
    expect((await diagnostic(async () => prepare(layout, await verifiedKernel(layout)))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.aliasConflict);
  });

  it("rejects deferred signed-stride and division profiles at CPU legalization", async () => {
    const negativeStride = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: add(mul(c(0), k("-2")), mul(c(1), k("-1"))),
      sourceByteOffset: "12",
      sourceBytes: "16",
      destinationBytes: "16",
    });
    expect((await diagnostic(async () => prepare(negativeStride, await verifiedKernel(negativeStride)))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);

    const divided = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: { kind: "floorDiv", value: rowMajorLocation(["2", "2"]), divisor: k("1") },
      sourceBytes: "16",
      destinationBytes: "16",
    });
    expect((await diagnostic(async () => prepare(divided, await verifiedKernel(divided)))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("rejects dynamic geometry and address failures during preparation", async () => {
    const n = { kind: "symbol", id: "n" };
    const invalidOffset = await verifiedLayout({
      shape: [n, "0"],
      sourceLocation: rowMajorLocation([n, "0"]),
      sourceByteOffset: { kind: "mul", lhs: n, rhs: { kind: "const", value: "4" } },
      sourceBytes: "0",
      destinationBytes: "0",
    });
    expect((await diagnostic(async () => prepare(invalidOffset, await verifiedKernel(invalidOffset), {
      bindings: { n: parseWireI64("1") },
    }))).diagnostic.code).toBe("BG-LAYOUT-FIELD-RANGE");

    const sourceOob = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: add(rowMajorLocation(["2", "2"]), k("1")),
      sourceBytes: "16",
      destinationBytes: "16",
    });
    expect((await diagnostic(async () => prepare(sourceOob, await verifiedKernel(sourceOob)))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidAccess);

    const destinationOob = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: rowMajorLocation(["2", "2"]),
      sourceBytes: "16",
      destinationBytes: "12",
    });
    expect((await diagnostic(async () => prepare(destinationOob, await verifiedKernel(destinationOob)))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidAccess);
  });

  it("uses intrinsic typed-array slots and methods and enforces binding alignment", async () => {
    const layout = await verifiedLayout({
      shape: ["2", "2"], sourceLocation: rowMajorLocation(["2", "2"]), sourceBytes: "16", destinationBytes: "16",
    });
    const plan = await prepare(layout, await verifiedKernel(layout));
    const misalignedSource = new Uint8Array(new ArrayBuffer(17), 1, 16);
    expect((await diagnostic(() => plan.execute({ source: misalignedSource, destination: new Uint8Array(16) }))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    const misalignedDestination = new Uint8Array(new ArrayBuffer(17), 1, 16);
    expect((await diagnostic(() => plan.execute({ source: new Uint8Array(16), destination: misalignedDestination }))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const spoofed = new Uint8Array(4);
    Object.defineProperties(spoofed, {
      byteLength: { get: () => 16 },
      byteOffset: { get: () => 0 },
      buffer: { get: () => new ArrayBuffer(16) },
    });
    expect((await diagnostic(() => plan.execute({ source: spoofed, destination: new Uint8Array(16) }))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const source = f32Bytes([1, 2, 3, 4]);
    const destination = new Uint8Array(16);
    Object.defineProperty(source, "subarray", { value: () => new Uint8Array(0) });
    Object.defineProperty(destination, "set", { value: () => undefined });
    plan.execute({ source, destination });
    expect(f32Values(destination)).toEqual([1, 2, 3, 4]);
  });

  it("derives binding-sensitive, key-order-stable specialization hashes and enforces work budgets", async () => {
    const n = { kind: "symbol", id: "n" };
    const m = { kind: "symbol", id: "m" };
    const byteLength = { kind: "mul", lhs: { kind: "mul", lhs: n, rhs: m }, rhs: { kind: "const", value: "4" } };
    const layout = await verifiedLayout({
      shape: [n, m],
      sourceLocation: rowMajorLocation([n, m]),
      sourceBytes: byteLength,
      destinationBytes: byteLength,
    });
    const kernel = await verifiedKernel(layout);
    const first = await prepare(layout, kernel, { bindings: { n: parseWireI64("2"), m: parseWireI64("2") } });
    const reordered = await prepare(layout, kernel, { bindings: { m: parseWireI64("2"), n: parseWireI64("2") } });
    const resized = await prepare(layout, kernel, { bindings: { n: parseWireI64("3"), m: parseWireI64("2") } });
    const backendNeutral = await prepareViewCopySpecialization(layout, kernel, {
      operationId: first.operationId,
      bindings: { n: parseWireI64("2"), m: parseWireI64("2") },
    });
    expect(first.specializationHash).toBe(reordered.specializationHash);
    expect(first.specializationHash).toBe(backendNeutral.specializationHash);
    expect(backendNeutral.sourceByteOffsets).toBeUndefined();
    expect(first.specializationHash).not.toBe(resized.specializationHash);
    expect(first.specializationHash).toMatch(/^[0-9a-f]{64}$/u);

    expect((await diagnostic(() => prepare(layout, kernel, {
      bindings: { n: parseWireI64("2"), m: parseWireI64("2") },
      maxEvaluationSteps: 1,
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
    expect((await diagnostic(() => prepare(layout, kernel, {
      bindings: { n: parseWireI64("2"), m: parseWireI64("2") },
      maxPreparedBytes: 1,
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
    const controller = new AbortController();
    controller.abort();
    expect((await diagnostic(() => prepare(layout, kernel, {
      bindings: { n: parseWireI64("2"), m: parseWireI64("2") },
      signal: controller.signal,
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("supports byte-unit maps and a nonzero dense destination offset", async () => {
    const layout = await verifiedLayout({
      shape: ["2", "2"],
      sourceLocation: mul(rowMajorLocation(["2", "2"]), k("4")),
      sourceLocationUnit: "byte",
      destinationLocation: mul(rowMajorLocation(["2", "2"]), k("4")),
      destinationLocationUnit: "byte",
      destinationByteOffset: "4",
      sourceBytes: "16",
      destinationBytes: "20",
    });
    const destination = new Uint8Array(20);
    const plan = await prepare(layout, await verifiedKernel(layout));
    plan.execute({ source: f32Bytes([1, 2, 3, 4]), destination });
    expect(new DataView(destination.buffer).getUint32(0, true)).toBe(0);
    expect(f32Values(destination.subarray(4))).toEqual([1, 2, 3, 4]);
  });
});
