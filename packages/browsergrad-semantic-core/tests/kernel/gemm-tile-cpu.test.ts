import { describe, expect, it, vi } from "vitest";

import {
  createVerifiedDenseLogicalGemmTileArtifacts,
  prepareLogicalGemmTileCpu,
  prepareLogicalGemmTileSpecialization,
} from "../../src/kernel";
import {
  KERNEL_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  parseWireU64,
  type WireU64,
} from "../../src/schema";

const wire = (value: string): WireU64 => parseWireU64(value);

function f32Bytes(values: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
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

describe("logical GEMM tile CPU reference", () => {
  it("executes canonical source-ordered f32 GEMM and reports exact work", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("3"),
      logicalTile: { m: wire("8"), n: wire("8"), k: wire("4") },
    });
    const prepared = await prepareLogicalGemmTileCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    const lhs = f32Bytes([1, 2, 3, 4, 5, 6]);
    const rhs = f32Bytes([7, 8, 9, 10, 11, 12]);
    const lhsBefore = [...lhs];
    const rhsBefore = [...rhs];
    const destination = f32Bytes([-1, -1, -1, -1]);
    expect(prepared.execute({ lhs, rhs, destination })).toEqual({
      operationId: artifacts.operationId,
      specializationHash: prepared.specializationHash,
      m: "2",
      n: "2",
      k: "3",
      tileM: "8",
      tileN: "8",
      tileK: "4",
      outputElements: "4",
      multiplyAdds: "12",
      bytesRead: "96",
      bytesWritten: "16",
    });
    expect(f32Values(destination)).toEqual([58, 64, 139, 154]);
    expect([...lhs]).toEqual(lhsBefore);
    expect([...rhs]).toEqual(rhsBefore);
  });

  it("applies f32 rounding after every product and increasing-k addition", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("1"), n: wire("1"), k: wire("3"),
      logicalTile: { m: wire("1"), n: wire("1"), k: wire("2") },
    });
    const prepared = await prepareLogicalGemmTileCpu(artifacts.layout, artifacts.kernel, { operationId: artifacts.operationId });
    const destination = f32Bytes([123]);
    prepared.execute({
      lhs: f32Bytes([16_777_216, 1, -16_777_216]),
      rhs: f32Bytes([1, 1, 1]),
      destination,
    });
    expect(f32Values(destination)).toEqual([0]);
  });

  it("handles logical tiles larger than the domain and zero output extents", async () => {
    const boundary = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("1"), n: wire("2"), k: wire("1"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const boundaryCpu = await prepareLogicalGemmTileCpu(boundary.layout, boundary.kernel, { operationId: boundary.operationId });
    const boundaryDestination = f32Bytes([-1, -1]);
    boundaryCpu.execute({ lhs: f32Bytes([3]), rhs: f32Bytes([4, 5]), destination: boundaryDestination });
    expect(f32Values(boundaryDestination)).toEqual([12, 15]);

    const empty = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("0"), n: wire("7"), k: wire("3"),
      logicalTile: { m: wire("4"), n: wire("4"), k: wire("4") },
    });
    const emptyCpu = await prepareLogicalGemmTileCpu(empty.layout, empty.kernel, { operationId: empty.operationId });
    expect(emptyCpu.execute({
      lhs: new Uint8Array(0),
      rhs: new Uint8Array(84),
      destination: new Uint8Array(0),
    })).toMatchObject({ outputElements: "0", multiplyAdds: "0", bytesRead: "0", bytesWritten: "0" });
  });

  it("enforces operation, aggregate-element, multiply-add, evaluation, time, and cancellation limits", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("3"),
      logicalTile: { m: wire("2"), n: wire("2"), k: wire("2") },
    });
    const cases: readonly [Record<string, unknown>, string][] = [
      [{ operationId: "missing" }, KERNEL_DIAGNOSTIC_CODES.danglingReference],
      [{ operationId: artifacts.operationId, maxElements: 15 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
      [{ operationId: artifacts.operationId, maxMultiplyAdds: 11 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
      [{ operationId: artifacts.operationId, maxEvaluationSteps: 1 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
      [{ operationId: artifacts.operationId, maxPreparationMs: 0 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
    ];
    for (const [request, code] of cases) {
      expect((await diagnostic(() => prepareLogicalGemmTileSpecialization(
        artifacts.layout,
        artifacts.kernel,
        request as never,
      ))).diagnostic.code).toBe(code);
    }
    const controller = new AbortController();
    controller.abort();
    expect((await diagnostic(() => prepareLogicalGemmTileSpecialization(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
      signal: controller.signal,
    }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("enforces the preparation deadline after the final specialization-hash tail", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("0"), n: wire("0"), k: wire("0"),
      logicalTile: { m: wire("1"), n: wire("1"), k: wire("1") },
    });
    const now = vi.spyOn(performance, "now");
    let calls = 0;
    now.mockImplementation(() => {
      calls += 1;
      return calls < 7 ? 0 : 2;
    });
    try {
      const error = await diagnostic(() => prepareLogicalGemmTileSpecialization(
        artifacts.layout,
        artifacts.kernel,
        { operationId: artifacts.operationId, maxPreparationMs: 1 },
      ));
      expect(error.diagnostic).toMatchObject({
        code: KERNEL_DIAGNOSTIC_CODES.resourceLimit,
        path: "$.maxPreparationMs",
      });
      expect(calls).toBe(7);
    } finally {
      now.mockRestore();
    }
  });

  it("requires exact, aligned, unshared, direct, pairwise-disjoint CPU buffers", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("3"),
      logicalTile: { m: wire("2"), n: wire("2"), k: wire("2") },
    });
    const prepared = await prepareLogicalGemmTileCpu(artifacts.layout, artifacts.kernel, { operationId: artifacts.operationId });
    const valid = () => ({ lhs: new Uint8Array(24), rhs: new Uint8Array(24), destination: new Uint8Array(16) });

    const short = valid();
    short.lhs = new Uint8Array(20);
    expect((await diagnostic(() => prepared.execute(short))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const backing = new ArrayBuffer(72);
    const overlapping = {
      lhs: new Uint8Array(backing, 0, 24),
      rhs: new Uint8Array(backing, 16, 24),
      destination: new Uint8Array(backing, 56, 16),
    };
    expect((await diagnostic(() => prepared.execute(overlapping))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.aliasConflict);

    const misalignedBacking = new ArrayBuffer(25);
    const misaligned = valid();
    misaligned.lhs = new Uint8Array(misalignedBacking, 1, 24);
    expect((await diagnostic(() => prepared.execute(misaligned))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    class HostileBytes extends Uint8Array {}
    const subclass = valid();
    subclass.lhs = new HostileBytes(24);
    expect((await diagnostic(() => prepared.execute(subclass))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    if (typeof SharedArrayBuffer !== "undefined") {
      const shared = valid();
      shared.lhs = new Uint8Array(new SharedArrayBuffer(24)) as unknown as Uint8Array<ArrayBuffer>;
      expect((await diagnostic(() => prepared.execute(shared))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    }
  });

  it("captures an exact enumerable own-data binding record without invoking getters", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("3"),
      logicalTile: { m: wire("2"), n: wire("2"), k: wire("2") },
    });
    const prepared = await prepareLogicalGemmTileCpu(artifacts.layout, artifacts.kernel, { operationId: artifacts.operationId });
    let getterReads = 0;
    const hostile = Object.create(null) as Record<string, unknown>;
    Object.defineProperties(hostile, {
      lhs: {
        enumerable: true,
        get() {
          getterReads += 1;
          throw new Error("binding getter must not run");
        },
      },
      rhs: { enumerable: true, value: new Uint8Array(24) },
      destination: { enumerable: true, value: new Uint8Array(16) },
    });
    expect((await diagnostic(() => prepared.execute(hostile as never))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    expect(getterReads).toBe(0);

    const extra = {
      lhs: new Uint8Array(24),
      rhs: new Uint8Array(24),
      destination: new Uint8Array(16),
      scratch: new Uint8Array(0),
    };
    expect((await diagnostic(() => prepared.execute(extra))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const hidden = Object.create(null) as Record<string, Uint8Array>;
    Object.defineProperties(hidden, {
      lhs: { enumerable: false, value: new Uint8Array(24) },
      rhs: { enumerable: true, value: new Uint8Array(24) },
      destination: { enumerable: true, value: new Uint8Array(16) },
    });
    expect((await diagnostic(() => prepared.execute(hidden as never))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
  });

  it("rejects detached zero-length ArrayBuffers even when the binding length is zero", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("0"), n: wire("7"), k: wire("3"),
      logicalTile: { m: wire("4"), n: wire("4"), k: wire("4") },
    });
    const prepared = await prepareLogicalGemmTileCpu(artifacts.layout, artifacts.kernel, { operationId: artifacts.operationId });
    const detachedBuffer = new ArrayBuffer(0);
    const detachedLhs = new Uint8Array(detachedBuffer);
    structuredClone(detachedBuffer, { transfer: [detachedBuffer] });
    const error = await diagnostic(() => prepared.execute({
      lhs: detachedLhs,
      rhs: new Uint8Array(84),
      destination: new Uint8Array(0),
    }));
    expect(error.diagnostic).toMatchObject({
      code: KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      path: "$.buffers.lhs",
    });
  });
});
