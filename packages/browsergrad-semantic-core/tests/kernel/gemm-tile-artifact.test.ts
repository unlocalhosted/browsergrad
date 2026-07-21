import { describe, expect, it } from "vitest";

import {
  LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
  createVerifiedDenseLogicalGemmTileArtifacts,
  decodeLogicalGemmTileArtifact,
  logicalGemmTileArtifactPayload,
  prepareLogicalGemmTileSpecialization,
  verifyLogicalGemmTileArtifact,
  type LogicalGemmTileArtifactPayloadV1,
} from "../../src/kernel";
import {
  layoutArtifactPayload,
  verifyLayoutArtifact,
  type DimExpr,
  type IndexExpr,
  type MemorySpace,
  type PredicateExpr,
  type VerifiedLayoutArtifact,
} from "../../src/layout";
import {
  KERNEL_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  hashSemanticArtifact,
  parseWireI64,
  parseWireU64,
  type JsonValue,
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

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

interface MutableLayoutPayload {
  readonly symbols: unknown[];
  readonly constraints: unknown[];
  readonly allocations: Array<{ memorySpace: MemorySpace; aliasSetId: string }>;
  readonly indexMaps: Array<{ coordinateRank: number; location: IndexExpr; inBounds: PredicateExpr }>;
  readonly views: Array<{ dtype: string; shape: DimExpr[] }>;
}

async function mutatedLayout(
  base: VerifiedLayoutArtifact,
  mutate: (payload: MutableLayoutPayload) => void,
): Promise<VerifiedLayoutArtifact> {
  const payload = clone(layoutArtifactPayload(base)) as unknown as MutableLayoutPayload;
  mutate(payload);
  return verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "gemm-test-layout", version: "1" },
    artifactId: "mutated-layout",
    requiredExtensions: [],
    payload,
  });
}

async function verifyForLayout(
  layout: VerifiedLayoutArtifact,
  operationTemplate: LogicalGemmTileArtifactPayloadV1["operation"],
) {
  const payload = layoutArtifactPayload(layout);
  const operation = clone(operationTemplate) as unknown as {
    lhs: { viewId: string };
    rhs: { viewId: string };
    destination: { viewId: string };
  };
  operation.lhs.viewId = payload.views[0]?.viewId ?? "missing";
  operation.rhs.viewId = payload.views[1]?.viewId ?? "missing";
  operation.destination.viewId = payload.views[2]?.viewId ?? "missing";
  return verifyLogicalGemmTileArtifact({
    schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: { id: "gemm-test-kernel", version: "1" },
    artifactId: "mutated-kernel",
    requiredExtensions: [],
    payload: {
      layoutSemanticHash: await hashSemanticArtifact(layout),
      operation,
    },
  }, { layout });
}

describe("logical GEMM tile artifact", () => {
  it("constructs closed backend-neutral rank-2 f32 meaning", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("4"), k: wire("3"),
      logicalTile: { m: wire("8"), n: wire("8"), k: wire("4") },
    });
    const layout = layoutArtifactPayload(artifacts.layout);
    const payload = logicalGemmTileArtifactPayload(artifacts.kernel);

    expect(layout.views.map((view) => view.shape.map((extent) => extent.kind === "const" ? extent.value : "dynamic")))
      .toEqual([["2", "3"], ["3", "4"], ["2", "4"]]);
    expect(layout.allocations.map((allocation) => allocation.aliasSetId)).toHaveLength(3);
    expect(new Set(layout.allocations.map((allocation) => allocation.aliasSetId)).size).toBe(3);
    expect(payload.operation).toEqual({
      operationId: artifacts.operationId,
      kind: "logical-gemm-tile",
      version: { major: 1, minor: 0 },
      lhs: { viewId: artifacts.lhs.viewId, access: "read" },
      rhs: { viewId: artifacts.rhs.viewId, access: "read" },
      destination: { viewId: artifacts.destination.viewId, access: "write" },
      logicalTile: { m: "8", n: "8", k: "4" },
      boundary: { lhs: "zero-fill", rhs: "zero-fill", destination: "mask-outside-logical-shape" },
      accumulation: {
        inputDType: "f32",
        accumulatorDType: "f32",
        outputDType: "f32",
        product: "multiply",
        reduction: "sum",
        reductionOrder: "increasing-k",
        rounding: "toward-nearest-ties-even",
        contraction: "forbid",
        reassociation: "forbid",
      },
      phases: { order: ["load", "accumulate", "store"], participation: "masked-full-logical-tile" },
      overlap: { kind: "forbid-all" },
    });
    const forbidden = JSON.stringify(payload);
    for (const word of ["workgroupSize", "subgroup", "vectorWidth", "staging", "wgsl", "backendId", "cuda", "webgpu"]) {
      expect(forbidden).not.toContain(word);
    }
  });

  it("keeps semantic hashes and canonical IDs independent of transport metadata", async () => {
    const request = {
      m: wire("17"), n: wire("19"), k: wire("23"),
      logicalTile: { m: wire("8"), n: wire("4"), k: wire("2") },
    };
    const first = await createVerifiedDenseLogicalGemmTileArtifacts(request, {
      producer: { id: "frontend-a", version: "7" },
      layoutArtifactId: "layout-a",
      kernelArtifactId: "kernel-a",
    });
    const second = await createVerifiedDenseLogicalGemmTileArtifacts(request, {
      producer: { id: "frontend-b", version: "99" },
      layoutArtifactId: "layout-b",
      kernelArtifactId: "kernel-b",
    });
    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.kernelSemanticHash).toBe(second.kernelSemanticHash);
    expect(first.operationId).toBe(second.operationId);
    expect(first.lhs).toEqual(second.lhs);
    expect(first.rhs).toEqual(second.rhs);
    expect(first.destination).toEqual(second.destination);

    const changedTile = await createVerifiedDenseLogicalGemmTileArtifacts({
      ...request,
      logicalTile: { m: wire("16"), n: wire("8"), k: wire("4") },
    });
    expect(changedTile.layoutSemanticHash).toBe(first.layoutSemanticHash);
    expect(changedTile.kernelSemanticHash).not.toBe(first.kernelSemanticHash);
  });

  it("round-trips canonical bytes and rejects structural artifact forgery", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("2"),
      logicalTile: { m: wire("2"), n: wire("2"), k: wire("2") },
    });
    const envelope = {
      schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "roundtrip", version: "1" },
      artifactId: "roundtrip",
      requiredExtensions: [],
      payload: logicalGemmTileArtifactPayload(artifacts.kernel),
    };
    const decoded = await decodeLogicalGemmTileArtifact(canonicalJsonBytes(envelope), { layout: artifacts.layout });
    expect(await hashSemanticArtifact(decoded)).toBe(artifacts.kernelSemanticHash);
    expect((await diagnostic(() => logicalGemmTileArtifactPayload(
      logicalGemmTileArtifactPayload(artifacts.kernel) as never,
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
  });

  it("fails closed on schedule, backend, numerical, mask, phase, and tile mutations", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("2"),
      logicalTile: { m: wire("2"), n: wire("2"), k: wire("2") },
    });
    const base = clone(logicalGemmTileArtifactPayload(artifacts.kernel));
    const cases: readonly [(payload: Record<string, unknown>) => void, string][] = [
      [(payload) => { payload.workgroupSize = [8, 8, 1]; }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { payload.backendId = "browsergrad.backend.webgpu.core"; }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { (payload.operation as Record<string, unknown>).wgsl = "@compute fn main() {}"; }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { ((payload.operation as Record<string, unknown>).logicalTile as Record<string, unknown>).m = "0"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { ((payload.operation as Record<string, unknown>).boundary as Record<string, unknown>).lhs = "clamp"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { ((payload.operation as Record<string, unknown>).accumulation as Record<string, unknown>).contraction = "allow"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { ((payload.operation as Record<string, unknown>).accumulation as Record<string, unknown>).reductionOrder = "backend-declared"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { ((payload.operation as Record<string, unknown>).phases as Record<string, unknown>).order = ["load", "store", "accumulate"]; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { ((payload.operation as Record<string, unknown>).overlap as Record<string, unknown>).kind = "allow"; }, KERNEL_DIAGNOSTIC_CODES.aliasConflict],
    ];
    for (const [mutate, code] of cases) {
      const payload = clone(base) as unknown as Record<string, unknown>;
      mutate(payload);
      const caught = await diagnostic(() => verifyLogicalGemmTileArtifact({
        schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
        version: { major: 1, minor: 0 },
        producer: { id: "mutation", version: "1" },
        artifactId: "mutation",
        requiredExtensions: [],
        payload,
      }, { layout: artifacts.layout }));
      expect(caught.diagnostic.code).toBe(code);
    }
  });

  it("rejects wrong dtype, rank, shape, memory space, aliases, layout hash, and view IDs", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("4"), k: wire("3"),
      logicalTile: { m: wire("2"), n: wire("2"), k: wire("2") },
    });
    const operation = logicalGemmTileArtifactPayload(artifacts.kernel).operation;
    const mutations: readonly [(payload: MutableLayoutPayload) => void, string][] = [
      [(payload) => { if (payload.views[0]) payload.views[0].dtype = "f16"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        if (payload.views[0] && payload.indexMaps[0]) {
          payload.views[0].shape = [payload.views[0].shape[0]!];
          payload.indexMaps[0].coordinateRank = 1;
          payload.indexMaps[0].location = { kind: "coordinate", axis: 0 };
          payload.indexMaps[0].inBounds = { kind: "bool", value: true };
        }
      }, KERNEL_DIAGNOSTIC_CODES.shapeMismatch],
      [(payload) => { if (payload.views[0]) payload.views[0].shape[1] = { kind: "const", value: parseWireI64("4") }; }, KERNEL_DIAGNOSTIC_CODES.shapeMismatch],
      [(payload) => { if (payload.allocations[0]) payload.allocations[0].memorySpace = { kind: "host" }; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { if (payload.allocations[0] && payload.allocations[1]) payload.allocations[1].aliasSetId = payload.allocations[0].aliasSetId; }, KERNEL_DIAGNOSTIC_CODES.aliasConflict],
    ];
    for (const [mutate, code] of mutations) {
      const layout = await mutatedLayout(artifacts.layout, mutate);
      expect((await diagnostic(() => verifyForLayout(layout, operation))).diagnostic.code).toBe(code);
    }

    const payload = clone(logicalGemmTileArtifactPayload(artifacts.kernel)) as unknown as {
      layoutSemanticHash: string;
      operation: { lhs: { viewId: string } };
    };
    payload.layoutSemanticHash = "0".repeat(64);
    expect((await diagnostic(() => verifyLogicalGemmTileArtifact({
      schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "bad-hash", version: "1" },
      artifactId: "bad-hash",
      requiredExtensions: [],
      payload,
    }, { layout: artifacts.layout }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch);

    const missingViewPayload = clone(logicalGemmTileArtifactPayload(artifacts.kernel)) as unknown as {
      layoutSemanticHash: string;
      operation: { lhs: { viewId: string } };
    };
    missingViewPayload.operation.lhs.viewId = "missingView";
    expect((await diagnostic(() => verifyLogicalGemmTileArtifact({
      schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "bad-view", version: "1" },
      artifactId: "bad-view",
      requiredExtensions: [],
      payload: missingViewPayload,
    }, { layout: artifacts.layout }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.danglingReference);
  });

  it("validates construction requests before creating semantic entities", async () => {
    const invalid: readonly [unknown, string][] = [
      [{ m: "2", n: "2", k: "2", logicalTile: { m: "0", n: "2", k: "2" } }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [{ m: "02", n: "2", k: "2", logicalTile: { m: "2", n: "2", k: "2" } }, SCHEMA_DIAGNOSTIC_CODES.nonCanonicalInteger],
      [{ m: "-1", n: "2", k: "2", logicalTile: { m: "2", n: "2", k: "2" } }, SCHEMA_DIAGNOSTIC_CODES.nonCanonicalInteger],
      [{ m: "9223372036854775808", n: "0", k: "0", logicalTile: { m: "2", n: "2", k: "2" } }, SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue],
      [{ m: "2", n: "2", k: "2", logicalTile: { m: "2", n: "2", k: "2" }, backend: "webgpu" }, SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue],
      [{ m: "2", n: "2", k: "2", logicalTile: { m: "2", n: "2", k: "2", workgroup: 8 } }, SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue],
    ];
    for (const [request, code] of invalid) {
      expect((await diagnostic(() => createVerifiedDenseLogicalGemmTileArtifacts(request as never))).diagnostic.code).toBe(code);
    }
  });

  it("prepares no physical schedule facts and derives binding-sensitive identity", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("3"), n: wire("2"), k: wire("4"),
      logicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const prepared = await prepareLogicalGemmTileSpecialization(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    expect(prepared).toMatchObject({ m: 3n, n: 2n, k: 4n, tileM: 8n, tileN: 8n, tileK: 8n, multiplyAdds: 24n });
    const record = prepared as unknown as Record<string, JsonValue>;
    expect(record.workgroupSize).toBeUndefined();
    expect(record.wgsl).toBeUndefined();
    expect(record.backendId).toBeUndefined();
    expect(record.staging).toBeUndefined();
  });
});
