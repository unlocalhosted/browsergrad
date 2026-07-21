import { describe, expect, it } from "vitest";

import {
  createVerifiedDenseAttentionForwardArtifacts,
  prepareAttentionForwardCpu,
  prepareAttentionForwardSpecialization,
  type AttentionForwardCpuBuffers,
} from "../../src/kernel";
import { layoutArtifactPayload } from "../../src/layout";
import {
  KERNEL_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  parseWireU64,
  type WireU64,
} from "../../src/schema";

const wire = (value: string): WireU64 => parseWireU64(value);

async function diagnostic(run: () => Promise<unknown> | unknown): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected semantic failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

async function prepared(causal: boolean, shape = {
  batch: 1,
  heads: 1,
  queryLength: 3,
  keyLength: 2,
  queryDepth: 1,
  valueDepth: 1,
}) {
  const artifacts = await createVerifiedDenseAttentionForwardArtifacts({
    batch: wire(String(shape.batch)),
    heads: wire(String(shape.heads)),
    queryLength: wire(String(shape.queryLength)),
    keyLength: wire(String(shape.keyLength)),
    queryDepth: wire(String(shape.queryDepth)),
    valueDepth: wire(String(shape.valueDepth)),
    causal,
  });
  const cpu = await prepareAttentionForwardCpu(artifacts.layout, artifacts.kernel, {
    operationId: artifacts.operationId,
  });
  const allocations = layoutArtifactPayload(artifacts.layout).allocations;
  const buffers = Object.fromEntries(
    ["query", "key", "value", "destination"].map((name, index) => {
      const byteLength = allocations[index]?.byteLength;
      if (byteLength?.kind !== "const") throw new Error("expected static allocation");
      return [name, new Uint8Array(Number(BigInt(byteLength.value)))] as const;
    }),
  ) as unknown as AttentionForwardCpuBuffers;
  return { artifacts, cpu, buffers };
}

function writeF32(bytes: Uint8Array, values: readonly number[]): void {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
}

function readF32(bytes: Uint8Array, count = bytes.byteLength / 4): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: count }, (_, index) => view.getFloat32(index * 4, true));
}

describe("attention-forward CPU reference", () => {
  it("executes non-causal stable softmax independently of a physical schedule", async () => {
    const { cpu, buffers } = await prepared(false);
    writeF32(buffers.query, [0, 0, 0]);
    writeF32(buffers.key, [0, 0]);
    writeF32(buffers.value, [2, 6]);

    const trace = await cpu.execute(buffers);
    expect(readF32(buffers.destination)).toEqual([4, 4, 4]);
    expect(trace).toMatchObject({
      operationId: cpu.operationId,
      specializationHash: cpu.specializationHash,
      referenceProfileId: "browsergrad.attention-forward.cpu-stable-softmax-f32@1",
      comparisonPolicyId: "browsergrad.attention-forward.f32-abs-relative@1",
      mask: "none",
      validScoreElements: "6",
      scoreMultiplyAdds: "6",
      weightedValueMultiplyAdds: "6",
      outputElements: "3",
      bytesRead: "72",
      bytesWritten: "12",
    });
  });

  it("applies upper-left causal exclusion before softmax state", async () => {
    const { cpu, buffers } = await prepared(true);
    writeF32(buffers.query, [0, 0, 0]);
    writeF32(buffers.key, [0, 0]);
    writeF32(buffers.value, [2, 6]);

    const trace = await cpu.execute(buffers);
    expect(readF32(buffers.destination)).toEqual([2, 4, 4]);
    expect(trace).toMatchObject({
      mask: "causal-upper-left",
      validScoreElements: "5",
      scoreMultiplyAdds: "5",
      weightedValueMultiplyAdds: "5",
      bytesRead: "60",
    });
  });

  it("matches an independent composed reference within the declared policy", async () => {
    const shape = {
      batch: 1,
      heads: 1,
      queryLength: 2,
      keyLength: 3,
      queryDepth: 2,
      valueDepth: 2,
    };
    const { cpu, buffers } = await prepared(false, shape);
    const query = [1, 0, 0, 1];
    const key = [1, 0, 0, 1, 1, 1];
    const value = [1, 2, 3, 4, 5, 6];
    writeF32(buffers.query, query);
    writeF32(buffers.key, key);
    writeF32(buffers.value, value);
    await cpu.execute(buffers);

    const expected = new Uint8Array(buffers.destination.byteLength);
    const expectedValues: number[] = [];
    const scale = Math.fround(1 / Math.sqrt(shape.queryDepth));
    for (let q = 0; q < shape.queryLength; q += 1) {
      const scores = Array.from({ length: shape.keyLength }, (_, k) => {
        let score = 0;
        for (let d = 0; d < shape.queryDepth; d += 1) {
          score += query[(q * shape.queryDepth) + d] as number
            * (key[(k * shape.queryDepth) + d] as number);
        }
        return score * scale;
      });
      const maximum = Math.max(...scores);
      const weights = scores.map((score) => Math.exp(score - maximum));
      const denominator = weights.reduce((sum, weight) => sum + weight, 0);
      for (let dv = 0; dv < shape.valueDepth; dv += 1) {
        expectedValues.push(weights.reduce((sum, weight, k) => (
          sum + ((weight / denominator) * (value[(k * shape.valueDepth) + dv] as number))
        ), 0));
      }
    }
    writeF32(expected, expectedValues);
    expect(cpu.compare(buffers.destination, expected)).toMatchObject({
      passed: true,
      mismatchCount: "0",
      comparedElements: "4",
      firstMismatchIndex: null,
    });
  });

  it("implements absolute-or-relative comparison and rejects non-finite outputs", async () => {
    const { cpu, buffers } = await prepared(false, {
      batch: 1, heads: 1, queryLength: 1, keyLength: 1, queryDepth: 1, valueDepth: 2,
    });
    const expected = new Uint8Array(buffers.destination.byteLength);
    const actual = new Uint8Array(buffers.destination.byteLength);
    writeF32(expected, [0, 100]);
    writeF32(actual, [0.00005, 100.005]);
    expect(cpu.compare(actual, expected).passed).toBe(true);

    writeF32(actual, [0.0002, 100]);
    expect(cpu.compare(actual, expected)).toMatchObject({
      passed: false,
      mismatchCount: "1",
      firstMismatchIndex: "0",
    });
    writeF32(actual, [Number.NaN, 100]);
    expect(cpu.compare(actual, expected)).toMatchObject({
      passed: false,
      mismatchCount: "1",
      firstMismatchIndex: "0",
    });
  });

  it("rejects non-finite inputs before changing destination bytes", async () => {
    const { cpu, buffers } = await prepared(false);
    writeF32(buffers.query, [Number.NaN, 0, 0]);
    writeF32(buffers.key, [0, 0]);
    writeF32(buffers.value, [2, 6]);
    buffers.destination.fill(0x5a);
    const before = [...buffers.destination];

    const error = await diagnostic(() => cpu.execute(buffers));
    expect(error.diagnostic).toMatchObject({
      code: KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      path: "$.buffers.query",
    });
    expect([...buffers.destination]).toEqual(before);
  });

  it("uses module-captured numerical and native-buffer intrinsics", async () => {
    const { cpu, buffers } = await prepared(false);
    writeF32(buffers.query, [0, 0, 0]);
    writeF32(buffers.key, [0, 0]);
    writeF32(buffers.value, [2, 6]);
    const originalFround = Math.fround;
    const originalFinite = Number.isFinite;
    const originalGetFloat32 = DataView.prototype.getFloat32;
    const originalSetFloat32 = DataView.prototype.setFloat32;
    const originalUint8Set = Uint8Array.prototype.set;
    try {
      Math.fround = () => 123;
      Number.isFinite = () => false;
      DataView.prototype.getFloat32 = () => { throw new Error("poisoned getFloat32"); };
      DataView.prototype.setFloat32 = () => { throw new Error("poisoned setFloat32"); };
      Uint8Array.prototype.set = () => { throw new Error("poisoned Uint8Array.set"); };
      await cpu.execute(buffers);
    } finally {
      Math.fround = originalFround;
      Number.isFinite = originalFinite;
      DataView.prototype.getFloat32 = originalGetFloat32;
      DataView.prototype.setFloat32 = originalSetFloat32;
      Uint8Array.prototype.set = originalUint8Set;
    }
    expect(readF32(buffers.destination)).toEqual([4, 4, 4]);
  });

  it("rejects accessors, subclasses, shared memory, wrong lengths, and overlap", async () => {
    const { cpu, buffers } = await prepared(false, {
      batch: 1, heads: 1, queryLength: 2, keyLength: 2, queryDepth: 2, valueDepth: 2,
    });
    const accessor = {
      get query() { return buffers.query; },
      key: buffers.key,
      value: buffers.value,
      destination: buffers.destination,
    };
    expect((await diagnostic(() => cpu.execute(accessor))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    class Derived extends Uint8Array {}
    expect((await diagnostic(() => cpu.execute({
      ...buffers,
      query: new Derived(buffers.query.byteLength),
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    expect((await diagnostic(() => cpu.execute({
      ...buffers,
      query: new Uint8Array(buffers.query.byteLength - 1),
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    if (typeof SharedArrayBuffer !== "undefined") {
      expect((await diagnostic(() => cpu.execute({
        ...buffers,
        query: new Uint8Array(new SharedArrayBuffer(buffers.query.byteLength)),
      }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    }

    if (Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, "resizable")?.get !== undefined) {
      const ResizableArrayBuffer = ArrayBuffer as unknown as new (
        length: number,
        options: { maxByteLength: number },
      ) => ArrayBuffer;
      const resizable = new ResizableArrayBuffer(buffers.query.byteLength, {
        maxByteLength: buffers.query.byteLength * 2,
      });
      expect((await diagnostic(() => cpu.execute({
        ...buffers,
        query: new Uint8Array(resizable),
      }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    }

    const root = new ArrayBuffer(56);
    const overlapping = {
      query: new Uint8Array(root, 0, 16),
      key: new Uint8Array(root, 8, 16),
      value: new Uint8Array(root, 24, 16),
      destination: new Uint8Array(root, 40, 16),
    };
    expect((await diagnostic(() => cpu.execute(overlapping))).diagnostic.code)
      .toBe(KERNEL_DIAGNOSTIC_CODES.aliasConflict);
  });

  it("bounds preparation and execution without partial output", async () => {
    const { artifacts, cpu, buffers } = await prepared(false);
    expect((await diagnostic(() => prepareAttentionForwardSpecialization(
      artifacts.layout,
      artifacts.kernel,
      { operationId: artifacts.operationId, maxElements: 1 },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
    expect((await diagnostic(() => prepareAttentionForwardSpecialization(
      artifacts.layout,
      artifacts.kernel,
      { operationId: artifacts.operationId, maxScalarOperations: 1 },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
    expect((await diagnostic(() => prepareAttentionForwardSpecialization(
      artifacts.layout,
      artifacts.kernel,
      { operationId: artifacts.operationId, maxAllocationBytes: 1 },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);

    buffers.destination.fill(0x3c);
    const before = [...buffers.destination];
    const controller = new AbortController();
    controller.abort();
    expect((await diagnostic(() => cpu.execute(buffers, {
      signal: controller.signal,
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
    expect([...buffers.destination]).toEqual(before);
  });
});
