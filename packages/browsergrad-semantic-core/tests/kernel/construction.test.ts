import { describe, expect, it } from "vitest";

import {
  createVerifiedDensePermutationViewCopyArtifacts,
  createVerifiedViewCopyArtifacts,
  kernelArtifactPayload,
  prepareViewCopyCpu,
  type CreateVerifiedViewCopyArtifactsRequest,
} from "../../src/kernel";
import {
  evaluateDimExpr,
  layoutArtifactPayload,
  type DimExpr,
} from "../../src/layout";
import {
  LAYOUT_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  parseWireI64,
  type WireI64,
} from "../../src/schema";

const wire = (value: string): WireI64 => parseWireI64(value);
const constant = (value: string): DimExpr => ({ kind: "const", value: wire(value) });

function bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return result;
}

function values(buffer: Uint8Array): number[] {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return Array.from({ length: buffer.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true));
}

async function error(run: () => Promise<unknown>): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected semantic failure");
  } catch (caught) {
    expect(caught).toBeInstanceOf(SemanticSchemaError);
    return caught as SemanticSchemaError;
  }
}

describe("canonical view-copy construction", () => {
  it("derives one rank-2 transpose artifact and executes its canonical maps", async () => {
    const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2"), wire("3")],
      axes: [1, 0],
      dtype: "f32",
    });
    const layout = layoutArtifactPayload(artifacts.layout);
    const operation = kernelArtifactPayload(artifacts.kernel).operations[0];

    expect(artifacts.layoutSemanticHash)
      .toBe("f204e22acb50681d6a52703131d9c13d0b1424da476dfd004b5a5bc3db25c1a2");
    expect(artifacts.kernelSemanticHash)
      .toBe("d189f64cb8d148fe242978e0657f1ecd3747383908b3316ee6d8aa31a65d699a");
    expect(layout.views.map((view) => view.shape.map((dimension) => (
      dimension.kind === "const" ? dimension.value : "dynamic"
    )))).toEqual([["3", "2"], ["3", "2"]]);
    expect(artifacts.source).toEqual({
      allocationId: layout.allocations[0]?.allocationId,
      indexMapId: layout.indexMaps[0]?.indexMapId,
      viewId: layout.views[0]?.viewId,
    });
    expect(artifacts.destination).toEqual({
      allocationId: layout.allocations[1]?.allocationId,
      indexMapId: layout.indexMaps[1]?.indexMapId,
      viewId: layout.views[1]?.viewId,
    });
    expect(operation).toMatchObject({
      operationId: artifacts.operationId,
      dtype: "f32",
      source: { viewId: artifacts.source.viewId, access: "read", invalidSource: { kind: "reject" } },
      destination: { viewId: artifacts.destination.viewId, access: "write" },
      overlap: { kind: "forbid" },
    });
    expect(artifacts.source.allocationId).not.toBe(artifacts.destination.allocationId);

    const prepared = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    const destination = new Uint8Array(24);
    expect(prepared.execute({ source: bytes([1, 2, 3, 4, 5, 6]), destination })).toMatchObject({
      logicalShape: ["3", "2"],
      elementCount: "6",
      readElements: "6",
      bytesWritten: "24",
    });
    expect(values(destination)).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("keeps semantic hashes and canonical IDs independent of transport metadata", async () => {
    const request = {
      inputShape: [wire("2"), wire("3")],
      axes: [1, 0],
      dtype: "f32" as const,
    };
    const first = await createVerifiedDensePermutationViewCopyArtifacts(request, {
      producer: { id: "frontend-a", version: "7" },
      layoutArtifactId: "random-layout-a",
      kernelArtifactId: "random-kernel-a",
    });
    const second = await createVerifiedDensePermutationViewCopyArtifacts(request, {
      producer: { id: "frontend-b", version: "99" },
      layoutArtifactId: "random-layout-b",
      kernelArtifactId: "random-kernel-b",
    });

    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.kernelSemanticHash).toBe(second.kernelSemanticHash);
    expect(first.source).toEqual(second.source);
    expect(first.destination).toEqual(second.destination);
    expect(first.operationId).toBe(second.operationId);
  });

  it("constructs rank-3 permutations and zero extents without legacy max(dim, 1) storage", async () => {
    const rank3 = await createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2"), wire("3"), wire("4")],
      axes: [2, 0, 1],
      dtype: "f32",
    });
    const rank3Plan = await prepareViewCopyCpu(rank3.layout, rank3.kernel, { operationId: rank3.operationId });
    expect(rank3Plan.logicalShape).toEqual([4n, 2n, 3n]);

    const empty = await createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("0"), wire("3")],
      axes: [1, 0],
      dtype: "f32",
    });
    const payload = layoutArtifactPayload(empty.layout);
    expect(payload.allocations.map((allocation) => evaluateDimExpr(allocation.byteLength)))
      .toEqual([{ kind: "resolved", value: 0n }, { kind: "resolved", value: 0n }]);
    const emptyPlan = await prepareViewCopyCpu(empty.layout, empty.kernel, { operationId: empty.operationId });
    expect(emptyPlan.elementCount).toBe(0n);
    expect(emptyPlan.execute({ source: new Uint8Array(0), destination: new Uint8Array(0) })).toMatchObject({
      elementCount: "0",
      readElements: "0",
      bytesWritten: "0",
    });
  });

  it("keeps semantic construction general while portable-profile limits stay separate", async () => {
    const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2"), wire("3"), wire("4"), wire("5")],
      axes: [3, 2, 1, 0],
      dtype: "f64",
    });
    expect(kernelArtifactPayload(artifacts.kernel).operations[0]?.dtype).toBe("f64");
    expect((await error(() => prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    }))).diagnostic.code).toBe("BG-KERNEL-UNSUPPORTED-PROFILE");
  });

  it("sorts symbol and constraint sets before hashing", async () => {
    const n = { id: "n", domain: { min: wire("1"), max: wire("8") } };
    const m = { id: "m", domain: { min: wire("1"), max: wire("8") } };
    const symbol = (id: string): DimExpr => ({ kind: "symbol", id });
    const side = () => ({
      layout: {
        kind: "strided" as const,
        shape: [symbol("n"), symbol("m")],
        strides: [symbol("m"), constant("1")],
      },
      allocation: {
        byteLength: {
          kind: "mul" as const,
          lhs: { kind: "mul" as const, lhs: symbol("n"), rhs: symbol("m") },
          rhs: constant("4"),
        },
        memorySpace: { kind: "global" as const },
        alignmentBytes: 4,
      },
      byteOffset: constant("0"),
      requiredAlignmentBytes: 4,
    });
    const draft = (reverse: boolean): CreateVerifiedViewCopyArtifactsRequest => ({
      dtype: "f32",
      symbols: reverse ? [n, m] : [m, n],
      constraints: reverse
        ? [{ kind: "positive", value: symbol("n") }, { kind: "positive", value: symbol("m") }]
        : [{ kind: "positive", value: symbol("m") }, { kind: "positive", value: symbol("n") }],
      source: side(),
      destination: side(),
      invalidSource: { kind: "reject" },
    });
    const first = await createVerifiedViewCopyArtifacts(draft(false));
    const second = await createVerifiedViewCopyArtifacts(draft(true));
    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.kernelSemanticHash).toBe(second.kernelSemanticHash);
  });

  it("snapshots input and returns an immutable authority-bound bundle", async () => {
    const request = {
      inputShape: [wire("2"), wire("3")],
      axes: [1, 0],
      dtype: "f32" as const,
    };
    const pending = createVerifiedDensePermutationViewCopyArtifacts(request);
    request.inputShape[0] = wire("9");
    request.axes[0] = 0;
    const artifacts = await pending;
    expect(layoutArtifactPayload(artifacts.layout).views[0]?.shape).toMatchObject([
      { kind: "const", value: "3" },
      { kind: "const", value: "2" },
    ]);
    expect(Object.isFrozen(artifacts)).toBe(true);
    expect(Object.isFrozen(artifacts.source)).toBe(true);
    expect(Object.isFrozen(layoutArtifactPayload(artifacts.layout))).toBe(true);
    expect(() => Object.assign(artifacts.source, { viewId: "forged" })).toThrow(TypeError);
  });

  it("rejects invalid axes and non-canonical or negative extents before indexing shape", async () => {
    const invalidAxes = [
      [0],
      [0, 0],
      [0, 2],
      [1, -1],
    ];
    for (const axes of invalidAxes) {
      const caught = await error(() => createVerifiedDensePermutationViewCopyArtifacts({
        inputShape: [wire("2"), wire("3")], axes, dtype: "f32",
      }));
      expect(caught.diagnostic).toMatchObject({ code: LAYOUT_DIAGNOSTIC_CODES.invalidLayoutExpr, path: "$.axes" });
    }
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2"), wire("3")], axes: [1, 0.5], dtype: "f32",
    }))).diagnostic).toMatchObject({ code: SCHEMA_DIAGNOSTIC_CODES.unsafeNumber, path: "$.axes[1]" });

    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("-1"), wire("3")], axes: [1, 0], dtype: "f32",
    }))).diagnostic).toMatchObject({ code: LAYOUT_DIAGNOSTIC_CODES.invalidLayoutExpr, path: "$.inputShape[0]" });
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: ["01" as WireI64, wire("3")], axes: [1, 0], dtype: "f32",
    }))).diagnostic).toMatchObject({ code: SCHEMA_DIAGNOSTIC_CODES.nonCanonicalInteger, path: "$.inputShape[0]" });
  });

  it("uses bigint expressions for large shapes and rejects allocation overflow", async () => {
    const aboveI64Bytes = await createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2305843009213693952")],
      axes: [0],
      dtype: "f32",
    });
    expect(evaluateDimExpr(layoutArtifactPayload(aboveI64Bytes.layout).allocations[0]!.byteLength))
      .toEqual({ kind: "resolved", value: 9223372036854775808n });

    const overflow = await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("9223372036854775807"), wire("3")],
      axes: [1, 0],
      dtype: "f32",
    }));
    expect(overflow.diagnostic.code).toBe(LAYOUT_DIAGNOSTIC_CODES.fieldRange);
    expect(overflow.diagnostic.path).toBe("$.payload.allocations[0].byteLength");
  });

  it("fails closed on unknown fields, accessors, class instances, cycles, sparse arrays, and unsafe numbers", async () => {
    const base = { inputShape: [wire("2"), wire("3")], axes: [1, 0], dtype: "f32" as const };
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts({ ...base, outputShape: [3, 2] } as never)))
      .diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    const accessor = { ...base } as Record<string, unknown>;
    Object.defineProperty(accessor, "dtype", { enumerable: true, get: () => "f32" });
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts(accessor as never))).diagnostic.code)
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    class Request {
      inputShape = [wire("2"), wire("3")];
      axes = [1, 0];
      dtype = "f32" as const;
    }
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts(new Request()))).diagnostic.code)
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    const cyclic = { ...base } as typeof base & { self?: unknown };
    cyclic.self = cyclic;
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts(cyclic as never))).diagnostic.code)
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    const sparse = Array<WireI64>(2);
    sparse[1] = wire("3");
    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts({ ...base, inputShape: sparse }))).diagnostic.code)
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    expect((await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      ...base,
      inputShape: [2 as unknown as WireI64, wire("3")],
    }))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalInteger);
  });

  it("rejects unknown or misspelled construction limits instead of silently weakening budgets", async () => {
    const caught = await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2"), wire("3")], axes: [1, 0], dtype: "f32",
    }, { limits: { maxNode: 1 } } as never));
    expect(caught.diagnostic).toMatchObject({
      code: SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
      path: "$options.limits",
    });
  });

  it("rejects dense ranks before constructing derived stride trees", async () => {
    const caught = await error(() => createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [wire("2"), wire("3"), wire("4")],
      axes: [0, 1, 2],
      dtype: "f32",
    }, { limits: { maxRank: 2 } }));
    expect(caught.diagnostic).toMatchObject({
      code: LAYOUT_DIAGNOSTIC_CODES.resourceLimit,
      path: "$.inputShape",
    });
  });

  it("reports malformed generic symbols as structured construction errors", async () => {
    const side = () => ({
      layout: { kind: "strided" as const, shape: [constant("1")], strides: [constant("1")] },
      allocation: {
        byteLength: constant("4"),
        memorySpace: { kind: "global" as const },
        alignmentBytes: 4,
      },
      byteOffset: constant("0"),
      requiredAlignmentBytes: 4,
    });
    const caught = await error(() => createVerifiedViewCopyArtifacts({
      dtype: "f32",
      symbols: [null] as never,
      constraints: [],
      source: side(),
      destination: side(),
      invalidSource: { kind: "reject" },
    }));
    expect(caught.diagnostic).toMatchObject({
      code: SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
      path: "$.symbols[0].id",
    });
  });
});
