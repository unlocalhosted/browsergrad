import { describe, expect, it } from "vitest";

import {
  createVerifiedDenseAttentionForwardArtifacts,
  type VerifiedAttentionForwardArtifacts,
} from "../../src/kernel";
import {
  ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
  attentionOnlineKvTileScheduleArtifactPayload,
  createVerifiedAttentionOnlineKvTileSchedule,
  decodeAttentionOnlineKvTileScheduleArtifact,
  verifyAttentionOnlineKvTileScheduleArtifact,
  type AttentionOnlineKvTileScheduleArtifactPayloadV1,
} from "../../src/schedule";
import {
  SCHEDULE_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  hashSemanticArtifact,
  parseWireU64,
  type WireU64,
} from "../../src/schema";

const wire = (value: string): WireU64 => parseWireU64(value);

async function attention(causal = false): Promise<VerifiedAttentionForwardArtifacts> {
  return createVerifiedDenseAttentionForwardArtifacts({
    batch: wire("2"),
    heads: wire("3"),
    queryLength: wire("17"),
    keyLength: wire("23"),
    queryDepth: wire("16"),
    valueDepth: wire("12"),
    causal,
  });
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

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

describe("attention online K/V tile schedule artifact", () => {
  it("constructs distinct 8x8 and 8x16 schedules for unchanged logical attention", async () => {
    const logical = await attention();
    const logicalHashBefore = await hashSemanticArtifact(logical.kernel);
    const schedule8 = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("8") },
    }, { producer: { id: "scheduler-a", version: "7" }, artifactId: "schedule-a" });
    const schedule16 = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("16") },
    });

    expect(schedule8.attentionForwardSemanticHash).toBe(logical.kernelSemanticHash);
    expect(schedule16.attentionForwardSemanticHash).toBe(logical.kernelSemanticHash);
    expect(schedule8.scheduleSemanticHash).not.toBe(schedule16.scheduleSemanticHash);
    expect(await hashSemanticArtifact(logical.kernel)).toBe(logicalHashBefore);
    expect(attentionOnlineKvTileScheduleArtifactPayload(schedule8.artifact)).toEqual({
      attentionForwardSemanticHash: logical.kernelSemanticHash,
      schedule: {
        kind: "attention-online-kv-tile-schedule",
        version: { major: 1, minor: 0 },
        physicalTile: { queryRows: "8", keyRows: "8" },
        workgroup: {
          size: { x: "8", y: "1", z: "1" },
          dispatchX: "query-tile",
          dispatchY: "head",
          dispatchZ: "batch",
        },
        invocation: {
          localX: "query-row-within-tile",
          localY: "unused",
          localZ: "unused",
          privateQuery: "one-logical-query-row",
          privateOutput: "one-logical-output-row",
        },
        traversal: {
          keyTiles: "increasing-key-index",
          keysWithinTile: "increasing-key-index",
          coverage: "complete-logical-key-range",
          tail: "masked-final-tile",
        },
        staging: {
          space: "workgroup",
          key: "cooperative",
          value: "cooperative",
          layout: "key-major-contiguous-depth",
          buffering: "single",
        },
        onlineSoftmax: {
          state: "running-maximum-denominator-and-weighted-value",
          tileScores: "scaled-query-key-dot-products",
          tileMaximum: "maximum-over-valid-tile-scores",
          tileReductionOrder: "increasing-key-index",
          update: "rescale-prior-state-then-accumulate-current-tile",
          priorRescale: "exp-previous-maximum-minus-new-maximum",
          currentWeight: "exp-score-minus-new-maximum",
          finalize: "divide-weighted-value-by-denominator-after-all-key-tiles",
        },
        participation: {
          workgroup: "all-invocations",
          boundaryQueryLanes: "participate",
          earlyExit: "forbidden",
        },
        uniformity: {
          barrierControl: "workgroup-uniform",
          activeMaskScope: "memory-effects-and-online-state-only",
        },
        vectorization: { keyLoad: "1", valueLoad: "1", destinationStore: "1" },
        barriers: {
          afterCooperativeLoad: {
            scope: "workgroup",
            memory: "workgroup",
            semantics: "acquire-release",
          },
          beforeStagingReuse: {
            scope: "workgroup",
            memory: "workgroup",
            semantics: "acquire-release",
          },
        },
        masks: {
          queryLane: "suppress-logical-state-and-store",
          keyLoad: "zero-fill",
          valueLoad: "zero-fill",
          invalidKeyScore: "exclude-before-online-state-update",
          logicalMask: "exclude-before-online-state-update",
          destinationStore: "suppress",
        },
      },
    });
  });

  it("binds the same physical recurrence separately to causal and non-causal meaning", async () => {
    const nonCausal = await attention(false);
    const causal = await attention(true);
    const request = { physicalTile: { queryRows: wire("8"), keyRows: wire("16") } };
    const first = await createVerifiedAttentionOnlineKvTileSchedule(nonCausal.kernel, request);
    const second = await createVerifiedAttentionOnlineKvTileSchedule(causal.kernel, request);
    const firstPayload = attentionOnlineKvTileScheduleArtifactPayload(first.artifact);
    const secondPayload = attentionOnlineKvTileScheduleArtifactPayload(second.artifact);

    expect(firstPayload.schedule).toEqual(secondPayload.schedule);
    expect(first.attentionForwardSemanticHash).not.toBe(second.attentionForwardSemanticHash);
    expect(first.scheduleSemanticHash).not.toBe(second.scheduleSemanticHash);
    expect(firstPayload.schedule.masks.logicalMask).toBe("exclude-before-online-state-update");
  });

  it("keeps identity independent of transport metadata and logical fields out of schedule", async () => {
    const logical = await attention();
    const request = { physicalTile: { queryRows: wire("8"), keyRows: wire("16") } };
    const first = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, request, {
      producer: { id: "scheduler-a", version: "1" },
      artifactId: "transport-a",
    });
    const second = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, request, {
      producer: { id: "scheduler-b", version: "99" },
      artifactId: "transport-b",
    });
    expect(first.scheduleSemanticHash).toBe(second.scheduleSemanticHash);

    const text = JSON.stringify(attentionOnlineKvTileScheduleArtifactPayload(first.artifact));
    for (const forbidden of [
      "layoutSemanticHash", "operationId", "viewId", "dtype", "comparisonPolicy",
      "autodiff", "preservation", "backendId", "webgpu", "wgsl", "cuda", "subgroup", "flash",
    ]) {
      expect(text.toLowerCase()).not.toContain(forbidden.toLowerCase());
    }
  });

  it("round-trips canonical bytes and rejects copied artifact authority", async () => {
    const logical = await attention();
    const constructed = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("16") },
    });
    const envelope = {
      schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "roundtrip", version: "1" },
      artifactId: "roundtrip",
      requiredExtensions: [],
      payload: attentionOnlineKvTileScheduleArtifactPayload(constructed.artifact),
    };
    const decoded = await decodeAttentionOnlineKvTileScheduleArtifact(
      canonicalJsonBytes(envelope),
      { attentionForward: logical.kernel },
    );
    expect(await hashSemanticArtifact(decoded)).toBe(constructed.scheduleSemanticHash);
    expect((await diagnostic(() => attentionOnlineKvTileScheduleArtifactPayload(
      attentionOnlineKvTileScheduleArtifactPayload(constructed.artifact) as never,
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
  });

  it("requires the exact verified attention-forward semantic hash", async () => {
    const first = await attention(false);
    const second = await attention(true);
    const constructed = await createVerifiedAttentionOnlineKvTileSchedule(first.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("8") },
    });
    const error = await diagnostic(() => verifyAttentionOnlineKvTileScheduleArtifact({
      schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "wrong-logical", version: "1" },
      artifactId: "wrong-logical",
      requiredExtensions: [],
      payload: attentionOnlineKvTileScheduleArtifactPayload(constructed.artifact),
    }, { attentionForward: second.kernel }));
    expect(error.diagnostic).toMatchObject({
      code: SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      path: "$.payload.attentionForwardSemanticHash",
    });
  });

  it("keeps boundary lanes uniform and invalid or masked keys out of online state", async () => {
    const logical = await attention(true);
    const constructed = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("16") },
    });
    const schedule = attentionOnlineKvTileScheduleArtifactPayload(constructed.artifact).schedule;
    expect(schedule.participation).toEqual({
      workgroup: "all-invocations",
      boundaryQueryLanes: "participate",
      earlyExit: "forbidden",
    });
    expect(schedule.uniformity).toEqual({
      barrierControl: "workgroup-uniform",
      activeMaskScope: "memory-effects-and-online-state-only",
    });
    expect(schedule.masks).toMatchObject({
      invalidKeyScore: "exclude-before-online-state-update",
      logicalMask: "exclude-before-online-state-update",
    });
  });

  it("fails closed on backend, recurrence, tile, mapping, vector, barrier, and mask changes", async () => {
    const logical = await attention();
    const constructed = await createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("16") },
    });
    const base = attentionOnlineKvTileScheduleArtifactPayload(constructed.artifact);
    const cases: readonly [(payload: Record<string, unknown>) => void, string][] = [
      [(payload) => { payload.backendId = "webgpu"; }, SCHEDULE_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { (payload.schedule as Record<string, unknown>).dtype = "f16"; }, SCHEDULE_DIAGNOSTIC_CODES.unknownField],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).physicalTile as Record<string, unknown>).queryRows = "257";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((((payload.schedule as Record<string, unknown>).workgroup as Record<string, unknown>).size) as Record<string, unknown>).x = "4";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).traversal as Record<string, unknown>).keyTiles = "reverse";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).onlineSoftmax as Record<string, unknown>).update = "independent-tiles";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).participation as Record<string, unknown>).earlyExit = "inactive-lanes";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).uniformity as Record<string, unknown>).activeMaskScope = "control-flow";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).vectorization as Record<string, unknown>).keyLoad = "4";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        (((((payload.schedule as Record<string, unknown>).barriers as Record<string, unknown>)
          .afterCooperativeLoad) as Record<string, unknown>).scope) = "subgroup";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).masks as Record<string, unknown>).invalidKeyScore = "zero-score";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => {
        ((payload.schedule as Record<string, unknown>).masks as Record<string, unknown>).logicalMask = "after-exp";
      }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
    ];
    for (const [mutate, code] of cases) {
      const payload = clone(base) as unknown as Record<string, unknown>;
      mutate(payload);
      const error = await diagnostic(() => verifyAttentionOnlineKvTileScheduleArtifact({
        schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
        version: { major: 1, minor: 0 },
        producer: { id: "mutation", version: "1" },
        artifactId: "mutation",
        requiredExtensions: [],
        payload: payload as unknown as AttentionOnlineKvTileScheduleArtifactPayloadV1,
      }, { attentionForward: logical.kernel }));
      expect(error.diagnostic.code).toBe(code);
    }
  });

  it("validates construction requests before schedule authority", async () => {
    const logical = await attention();
    for (const request of [
      { physicalTile: { queryRows: "0", keyRows: "8" } },
      { physicalTile: { queryRows: "8", keyRows: "257" } },
    ]) {
      expect((await diagnostic(() => createVerifiedAttentionOnlineKvTileSchedule(
        logical.kernel,
        request as never,
      ))).diagnostic.code).toBe(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile);
    }
    expect((await diagnostic(() => createVerifiedAttentionOnlineKvTileSchedule(logical.kernel, {
      physicalTile: { queryRows: wire("8"), keyRows: wire("8") },
      backend: "webgpu",
    } as never))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });
});
