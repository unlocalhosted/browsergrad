import { describe, expect, it } from "vitest";
import {
  ATTENTION_FORWARD_ARTIFACT_SCHEMA,
  INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY,
  INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY,
  attentionForwardArtifactPayload,
  createVerifiedDenseAttentionForwardArtifacts,
  decodeAttentionForwardArtifact,
  verifyAttentionForwardArtifact,
  type AttentionForwardArtifactPayloadV1,
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
  readonly indexMaps: Array<{
    coordinateRank: number;
    location: IndexExpr;
    inBounds: PredicateExpr;
  }>;
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
    producer: { id: "attention-test-layout", version: "1" },
    artifactId: "mutated-layout",
    requiredExtensions: [],
    payload,
  });
}

async function verifyForLayout(
  layout: VerifiedLayoutArtifact,
  operationTemplate: AttentionForwardArtifactPayloadV1["operation"],
) {
  const payload = layoutArtifactPayload(layout);
  const operation = clone(operationTemplate) as unknown as {
    query: { viewId: string };
    key: { viewId: string };
    value: { viewId: string };
    destination: { viewId: string };
  };
  operation.query.viewId = payload.views[0]?.viewId ?? "missing";
  operation.key.viewId = payload.views[1]?.viewId ?? "missing";
  operation.value.viewId = payload.views[2]?.viewId ?? "missing";
  operation.destination.viewId = payload.views[3]?.viewId ?? "missing";
  return verifyAttentionForwardArtifact({
    schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: { id: "attention-test-kernel", version: "1" },
    artifactId: "mutated-kernel",
    requiredExtensions: [],
    payload: {
      layoutSemanticHash: await hashSemanticArtifact(layout),
      operation,
    },
  }, { layout });
}

function request(causal = false) {
  return {
    batch: wire("2"),
    heads: wire("3"),
    queryLength: wire("5"),
    keyLength: wire("7"),
    queryDepth: wire("4"),
    valueDepth: wire("6"),
    causal,
  };
}

describe("attention-forward artifact", () => {
  it("constructs closed backend-neutral rank-4 f32 attention meaning", async () => {
    const artifacts = await createVerifiedDenseAttentionForwardArtifacts(request());
    const layout = layoutArtifactPayload(artifacts.layout);
    const payload = attentionForwardArtifactPayload(artifacts.kernel);

    expect(layout.views.map((view) => view.shape.map((extent) =>
      extent.kind === "const" ? extent.value : "dynamic"))).toEqual([
      ["2", "3", "5", "4"],
      ["2", "3", "7", "4"],
      ["2", "3", "7", "6"],
      ["2", "3", "5", "6"],
    ]);
    expect(new Set(layout.allocations.map((allocation) => allocation.aliasSetId)).size).toBe(4);
    expect(INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY).toEqual({
      policyId: "browsergrad.attention-forward.f32-abs-relative@1",
      rule: "absolute-or-relative",
      absoluteTolerance: 0.0001,
      relativeTolerance: 0.0001,
      nonFinite: "reject",
      signedZero: "ignore-sign",
    });
    expect(Object.isFrozen(INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY)).toBe(true);
    expect(Object.isFrozen(INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY.inputDTypes)).toBe(true);
    expect(payload.operation).toMatchObject({
      operationId: artifacts.operationId,
      kind: "scaled-dot-product-attention-forward",
      version: { major: 1, minor: 0 },
      query: { viewId: artifacts.query.viewId, access: "read" },
      key: { viewId: artifacts.key.viewId, access: "read" },
      value: { viewId: artifacts.value.viewId, access: "read" },
      destination: { viewId: artifacts.destination.viewId, access: "write" },
      mask: { kind: "none" },
      scale: {
        source: "inverse-square-root-query-depth-rounded-to-f32",
        value: { kind: "float-bits", dtype: "f32", bits: "3f000000" },
      },
      softmax: {
        kind: "stable-max-subtracted",
        scope: "complete-logical-key-range",
        fullyMaskedRows: "forbidden",
      },
      numerical: {
        policyId: "browsergrad.attention-forward.f32-stable-softmax@1",
        comparisonPolicyId: "browsergrad.attention-forward.f32-abs-relative@1",
        contraction: "allow",
        reassociation: "allow",
        denormals: "backend-declared",
      },
      autodiff: {
        vjp: "not-defined",
        diagnosticId: "browsergrad.attention-forward-vjp-unavailable",
      },
      phases: { order: ["load", "score", "softmax", "weighted-value", "store"] },
      overlap: { kind: "forbid-all" },
    });
    const serialized = JSON.stringify(payload);
    for (const forbidden of [
      "workgroup", "subgroup", "staging", "barrier", "wgsl", "webgpu", "cuda", "flash",
    ]) {
      expect(serialized.toLowerCase()).not.toContain(forbidden);
    }
  });

  it("supports explicit upper-left causal meaning for rectangular sequences", async () => {
    const artifacts = await createVerifiedDenseAttentionForwardArtifacts(request(true));
    expect(attentionForwardArtifactPayload(artifacts.kernel).operation.mask).toEqual({
      kind: "causal",
      orientation: "upper-left",
      predicate: "key-index-less-equal-query-index",
    });
  });

  it("keeps canonical identity independent of transport metadata and sensitive to semantics", async () => {
    const first = await createVerifiedDenseAttentionForwardArtifacts(request(), {
      producer: { id: "frontend-a", version: "7" },
      layoutArtifactId: "layout-a",
      kernelArtifactId: "kernel-a",
    });
    const second = await createVerifiedDenseAttentionForwardArtifacts(request(), {
      producer: { id: "frontend-b", version: "99" },
      layoutArtifactId: "layout-b",
      kernelArtifactId: "kernel-b",
    });
    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.kernelSemanticHash).toBe(second.kernelSemanticHash);
    expect(first.operationId).toBe(second.operationId);

    const causal = await createVerifiedDenseAttentionForwardArtifacts(request(true));
    const changedDepth = await createVerifiedDenseAttentionForwardArtifacts({
      ...request(),
      queryDepth: wire("16"),
    });
    expect(causal.layoutSemanticHash).toBe(first.layoutSemanticHash);
    expect(causal.kernelSemanticHash).not.toBe(first.kernelSemanticHash);
    expect(changedDepth.layoutSemanticHash).not.toBe(first.layoutSemanticHash);
    expect(changedDepth.kernelSemanticHash).not.toBe(first.kernelSemanticHash);
    expect(attentionForwardArtifactPayload(changedDepth.kernel).operation.scale.value.bits)
      .toBe("3e800000");
  });

  it("round-trips canonical bytes and rejects structural artifact forgery", async () => {
    const artifacts = await createVerifiedDenseAttentionForwardArtifacts(request());
    const envelope = {
      schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "roundtrip", version: "1" },
      artifactId: "roundtrip",
      requiredExtensions: [],
      payload: attentionForwardArtifactPayload(artifacts.kernel),
    };
    const decoded = await decodeAttentionForwardArtifact(
      canonicalJsonBytes(envelope),
      { layout: artifacts.layout },
    );
    expect(await hashSemanticArtifact(decoded)).toBe(artifacts.kernelSemanticHash);
    expect((await diagnostic(() => attentionForwardArtifactPayload(
      attentionForwardArtifactPayload(artifacts.kernel) as never,
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
  });

  it("fails closed on backend, schedule, scale, mask, numerical, phase, and VJP mutations", async () => {
    const artifacts = await createVerifiedDenseAttentionForwardArtifacts(request());
    const base = clone(attentionForwardArtifactPayload(artifacts.kernel));
    const cases: readonly [(payload: Record<string, unknown>) => void, string][] = [
      [(payload) => { payload.workgroupSize = [8, 1, 1]; }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { payload.backendId = "browsergrad.backend.webgpu.core"; }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { (payload.operation as Record<string, unknown>).wgsl = "@compute fn main() {}"; }, KERNEL_DIAGNOSTIC_CODES.unknownField],
      [(payload) => {
        (((payload.operation as Record<string, unknown>).scale as Record<string, unknown>).value as Record<string, unknown>).bits = "00000000";
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        (payload.operation as Record<string, unknown>).mask = {
          kind: "causal",
          orientation: "lower-right",
          predicate: "key-index-less-equal-query-index",
        };
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        (((payload.operation as Record<string, unknown>).scale as Record<string, unknown>).value as Record<string, unknown>).bits = "3e800000";
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.operation as Record<string, unknown>).score as Record<string, unknown>).reductionOrder = "tree";
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.operation as Record<string, unknown>).numerical as Record<string, unknown>).comparisonPolicyId = "loose";
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.operation as Record<string, unknown>).autodiff as Record<string, unknown>).vjp = "available";
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.operation as Record<string, unknown>).phases as Record<string, unknown>).order = ["load", "store"];
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.operation as Record<string, unknown>).overlap as Record<string, unknown>).kind = "allow";
      }, KERNEL_DIAGNOSTIC_CODES.aliasConflict],
    ];
    for (const [mutate, code] of cases) {
      const payload = clone(base) as unknown as Record<string, unknown>;
      mutate(payload);
      const caught = await diagnostic(() => verifyAttentionForwardArtifact({
        schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
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
    const artifacts = await createVerifiedDenseAttentionForwardArtifacts(request());
    const operation = attentionForwardArtifactPayload(artifacts.kernel).operation;
    const mutations: readonly [(payload: MutableLayoutPayload) => void, string][] = [
      [(payload) => { if (payload.views[0]) payload.views[0].dtype = "f16"; }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        if (payload.views[0] && payload.indexMaps[0]) {
          payload.views[0].shape = payload.views[0].shape.slice(0, 3);
          payload.indexMaps[0].coordinateRank = 3;
          payload.indexMaps[0].location = { kind: "coordinate", axis: 0 };
          payload.indexMaps[0].inBounds = { kind: "bool", value: true };
        }
      }, KERNEL_DIAGNOSTIC_CODES.shapeMismatch],
      [(payload) => {
        if (payload.views[1]) payload.views[1].shape[0] = { kind: "const", value: parseWireI64("3") };
      }, KERNEL_DIAGNOSTIC_CODES.shapeMismatch],
      [(payload) => {
        if (payload.views[0]) payload.views[0].shape[3] = { kind: "const", value: parseWireI64("5") };
      }, KERNEL_DIAGNOSTIC_CODES.shapeMismatch],
      [(payload) => {
        if (payload.views[2]) payload.views[2].shape[2] = { kind: "const", value: parseWireI64("8") };
      }, KERNEL_DIAGNOSTIC_CODES.shapeMismatch],
      [(payload) => {
        if (payload.views[0]) payload.views[0].shape[0] = { kind: "const", value: parseWireI64("0") };
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        if (payload.allocations[0]) payload.allocations[0].memorySpace = { kind: "host" };
      }, KERNEL_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        if (payload.allocations[0] && payload.allocations[1]) {
          payload.allocations[1].aliasSetId = payload.allocations[0].aliasSetId;
        }
      }, KERNEL_DIAGNOSTIC_CODES.aliasConflict],
    ];
    for (const [mutate, code] of mutations) {
      const layout = await mutatedLayout(artifacts.layout, mutate);
      expect((await diagnostic(() => verifyForLayout(layout, operation))).diagnostic.code).toBe(code);
    }

    const payload = clone(attentionForwardArtifactPayload(artifacts.kernel)) as unknown as {
      layoutSemanticHash: string;
      operation: { query: { viewId: string } };
    };
    payload.layoutSemanticHash = "0".repeat(64);
    expect((await diagnostic(() => verifyAttentionForwardArtifact({
      schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "bad-hash", version: "1" },
      artifactId: "bad-hash",
      requiredExtensions: [],
      payload,
    }, { layout: artifacts.layout }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch);

    const missingView = clone(attentionForwardArtifactPayload(artifacts.kernel)) as unknown as {
      operation: { query: { viewId: string } };
    };
    missingView.operation.query.viewId = "missingView";
    expect((await diagnostic(() => verifyAttentionForwardArtifact({
      schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "bad-view", version: "1" },
      artifactId: "bad-view",
      requiredExtensions: [],
      payload: missingView,
    }, { layout: artifacts.layout }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.danglingReference);
  });

  it("validates construction requests before creating semantic entities", async () => {
    const invalid: readonly unknown[] = [
      { ...request(), batch: "0" },
      { ...request(), queryDepth: "257" },
      { ...request(), valueDepth: "4294967296" },
      { ...request(), queryLength: "05" },
      { ...request(), causal: "yes" },
      { ...request(), backend: "webgpu" },
      { ...request(), schedule: { keyRows: "8" } },
    ];
    for (const value of invalid) {
      expect((await diagnostic(() => createVerifiedDenseAttentionForwardArtifacts(
        value as never,
      ))).diagnostic.code).toMatch(/^BG-SCHEMA-/u);
    }
  });

  it("rejects nonpositive and nonfinite exact scale bit records", async () => {
    const artifacts = await createVerifiedDenseAttentionForwardArtifacts(request());
    for (const bits of ["00000000", "80000000", "7f800000", "7fc00000"] as const) {
      const payload = clone(attentionForwardArtifactPayload(artifacts.kernel)) as unknown as {
        operation: { scale: { value: { bits: string } } };
      };
      payload.operation.scale.value.bits = bits;
      expect((await diagnostic(() => verifyAttentionForwardArtifact({
        schema: ATTENTION_FORWARD_ARTIFACT_SCHEMA,
        version: { major: 1, minor: 0 },
        producer: { id: "bad-scale", version: "1" },
        artifactId: "bad-scale",
        requiredExtensions: [],
        payload,
      }, { layout: artifacts.layout }))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);
    }
  });

  it("does not accept caller-authored numerical or execution options", async () => {
    const invalidOptions: readonly unknown[] = [
      { backend: "webgpu" },
      { numerical: { tolerance: 1 } },
      { producer: { id: "x", version: "1", trust: true } },
      { limits: { maxDepth: 0 } },
    ];
    for (const options of invalidOptions) {
      expect((await diagnostic(() => createVerifiedDenseAttentionForwardArtifacts(
        request(),
        options as never,
      ))).diagnostic.code).toMatch(/^BG-SCHEMA-/u);
    }
  });
});
