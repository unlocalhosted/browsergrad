import { describe, expect, it } from "vitest";

import {
  LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
  createVerifiedDenseLogicalGemmTileArtifacts,
  logicalGemmTileArtifactPayload,
  prepareLogicalGemmTileSpecialization,
} from "../../src/kernel";
import {
  LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
  createVerifiedLogicalGemmTileSchedule,
  decodeLogicalGemmTileScheduleArtifact,
  logicalGemmTileScheduleArtifactPayload,
  prepareLogicalGemmTileSchedule,
  verifyLogicalGemmTileScheduleArtifact,
  type LogicalGemmTileScheduleArtifactPayloadV1,
} from "../../src/schedule";
import {
  SCHEDULE_DIAGNOSTIC_CODES,
  KERNEL_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  hashSemanticArtifact,
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

describe("logical GEMM tile schedule artifact", () => {
  it("constructs distinct legal 8x8x8 and 16x16x16 schedules for unchanged logical semantics", async () => {
    const logical = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("31"), n: wire("33"), k: wire("35"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const logicalHashBefore = await hashSemanticArtifact(logical.kernel);
    const schedule8 = await createVerifiedLogicalGemmTileSchedule(logical.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    }, { producer: { id: "scheduler-a", version: "7" }, artifactId: "schedule-a" });
    const schedule16 = await createVerifiedLogicalGemmTileSchedule(logical.kernel, {
      physicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });

    expect(schedule8.logicalGemmSemanticHash).toBe(logical.kernelSemanticHash);
    expect(schedule16.logicalGemmSemanticHash).toBe(logical.kernelSemanticHash);
    expect(schedule8.scheduleSemanticHash).not.toBe(schedule16.scheduleSemanticHash);
    expect(await hashSemanticArtifact(logical.kernel)).toBe(logicalHashBefore);
    expect(logicalGemmTileScheduleArtifactPayload(schedule8.artifact)).toEqual({
      logicalGemmSemanticHash: logical.kernelSemanticHash,
      schedule: {
        kind: "logical-gemm-tile-schedule",
        version: { major: 1, minor: 0 },
        physicalTile: { m: "8", n: "8", k: "8" },
        workgroup: {
          size: { x: "8", y: "8", z: "1" },
          x: "physical-tile-column",
          y: "physical-tile-row",
          z: "singleton",
        },
        invocation: {
          output: "one-element",
          localX: "output-column",
          localY: "output-row",
          localZ: "unused",
        },
        staging: { space: "workgroup", lhs: "cooperative", rhs: "cooperative", buffering: "single" },
        participation: { workgroup: "all-invocations", boundaryLanes: "participate", earlyExit: "forbidden" },
        uniformity: { barrierControl: "workgroup-uniform", activeMaskScope: "memory-effects-only" },
        vectorization: { lhsLoad: "1", rhsLoad: "1", destinationStore: "1" },
        barriers: {
          afterCooperativeLoad: { scope: "workgroup", memory: "workgroup", semantics: "acquire-release" },
          beforeStagingReuse: { scope: "workgroup", memory: "workgroup", semantics: "acquire-release" },
        },
        masks: { lhsLoad: "zero-fill", rhsLoad: "zero-fill", destinationStore: "suppress" },
      },
    });
  });

  it("specializes two schedules from one authorized logical proof without reconstructing logical meaning", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("17"), n: wire("19"), k: wire("23"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("8") },
    });
    const logical = await prepareLogicalGemmTileSpecialization(
      artifacts.layout,
      artifacts.kernel,
      { operationId: artifacts.operationId },
    );
    const schedule8Artifact = await createVerifiedLogicalGemmTileSchedule(artifacts.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const schedule16Artifact = await createVerifiedLogicalGemmTileSchedule(artifacts.kernel, {
      physicalTile: { m: wire("16"), n: wire("16"), k: wire("8") },
    });
    const schedule8 = await prepareLogicalGemmTileSchedule(
      logical,
      artifacts.kernel,
      schedule8Artifact.artifact,
    );
    const schedule16 = await prepareLogicalGemmTileSchedule(
      logical,
      artifacts.kernel,
      schedule16Artifact.artifact,
    );

    expect(schedule8.logical).toBe(logical);
    expect(schedule16.logical).toBe(logical);
    expect(schedule8.logicalGemmSemanticHash).toBe(logical.kernelSemanticHash);
    expect(schedule16.logicalGemmSemanticHash).toBe(logical.kernelSemanticHash);
    expect(schedule8).toMatchObject({
      physicalM: 8n,
      physicalN: 8n,
      physicalK: 8n,
      workgroupInvocations: 64n,
      lhsStagingElements: 64n,
      rhsStagingElements: 64n,
      aggregateStagingElements: 128n,
      dispatchX: 3n,
      dispatchY: 3n,
      dispatchZ: 1n,
      dispatchWorkgroups: 9n,
    });
    expect(schedule16).toMatchObject({
      physicalM: 16n,
      physicalN: 16n,
      physicalK: 8n,
      workgroupInvocations: 256n,
      lhsStagingElements: 128n,
      rhsStagingElements: 128n,
      aggregateStagingElements: 256n,
      dispatchX: 2n,
      dispatchY: 2n,
      dispatchZ: 1n,
      dispatchWorkgroups: 4n,
    });
    expect(schedule8.scheduleSemanticHash).not.toBe(schedule16.scheduleSemanticHash);
    expect(schedule8.scheduleSpecializationHash).not.toBe(schedule16.scheduleSpecializationHash);
    expect(JSON.stringify(schedule8.schedule)).not.toContain("rounding");
  });

  it("rejects copied logical proofs, artifact mismatches, and schedule resource overflow", async () => {
    const first = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const second = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const logical = await prepareLogicalGemmTileSpecialization(
      first.layout,
      first.kernel,
      { operationId: first.operationId },
    );
    const firstSchedule = await createVerifiedLogicalGemmTileSchedule(first.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const secondSchedule = await createVerifiedLogicalGemmTileSchedule(second.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });

    expect((await diagnostic(() => prepareLogicalGemmTileSchedule(
      { ...logical },
      first.kernel,
      firstSchedule.artifact,
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    expect((await diagnostic(() => prepareLogicalGemmTileSchedule(
      logical,
      second.kernel,
      secondSchedule.artifact,
    ))).diagnostic.code).toBe(SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch);
    expect((await diagnostic(() => prepareLogicalGemmTileSchedule(
      logical,
      first.kernel,
      firstSchedule.artifact,
      { maxWorkgroupInvocations: 63 },
    ))).diagnostic.code).toBe(SCHEDULE_DIAGNOSTIC_CODES.resourceLimit);
    expect((await diagnostic(() => prepareLogicalGemmTileSchedule(
      logical,
      first.kernel,
      firstSchedule.artifact,
      { maxStagingElements: 127 },
    ))).diagnostic.code).toBe(SCHEDULE_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("keeps schedule hashes independent of transport metadata and separate from logical meaning", async () => {
    const logical = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const request = { physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") } };
    const first = await createVerifiedLogicalGemmTileSchedule(logical.kernel, request, {
      producer: { id: "scheduler-a", version: "1" }, artifactId: "transport-a",
    });
    const second = await createVerifiedLogicalGemmTileSchedule(logical.kernel, request, {
      producer: { id: "scheduler-b", version: "99" }, artifactId: "transport-b",
    });
    expect(first.scheduleSemanticHash).toBe(second.scheduleSemanticHash);

    const scheduleText = JSON.stringify(logicalGemmTileScheduleArtifactPayload(first.artifact));
    for (const forbidden of [
      "layoutSemanticHash", "operationId", "viewId", "accumulation", "rounding", "dtype",
      "preservation", "exactInputDomain", "backendId", "webgpu", "wgsl", "cuda", "subgroup",
    ]) {
      expect(scheduleText).not.toContain(forbidden);
    }
    const logicalText = JSON.stringify(logicalGemmTileArtifactPayload(logical.kernel));
    for (const physical of [
      "physicalTile", "workgroup", "invocation", "staging", "uniformity",
      "vectorization", "barriers",
    ]) {
      expect(logicalText).not.toContain(physical);
    }
  });

  it("requires irregular boundary lanes to remain active through every workgroup barrier", async () => {
    const logical = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("17"), n: wire("19"), k: wire("23"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("8") },
    });
    const constructed = await createVerifiedLogicalGemmTileSchedule(logical.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const base = logicalGemmTileScheduleArtifactPayload(constructed.artifact);
    expect(base.schedule.participation).toEqual({
      workgroup: "all-invocations",
      boundaryLanes: "participate",
      earlyExit: "forbidden",
    });
    expect(base.schedule.uniformity).toEqual({
      barrierControl: "workgroup-uniform",
      activeMaskScope: "memory-effects-only",
    });

    for (const mutate of [
      (payload: Record<string, unknown>) => {
        (((payload.schedule as Record<string, unknown>).participation) as Record<string, unknown>).boundaryLanes = "deactivate";
      },
      (payload: Record<string, unknown>) => {
        (((payload.schedule as Record<string, unknown>).uniformity) as Record<string, unknown>).activeMaskScope = "control-flow";
      },
      (payload: Record<string, unknown>) => {
        (((((payload.schedule as Record<string, unknown>).barriers) as Record<string, unknown>).afterCooperativeLoad) as Record<string, unknown>).memory = "none";
      },
    ]) {
      const payload = clone(base) as unknown as Record<string, unknown>;
      mutate(payload);
      const error = await diagnostic(() => verifyLogicalGemmTileScheduleArtifact({
        schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
        version: { major: 1, minor: 0 },
        producer: { id: "boundary-hostile", version: "1" },
        artifactId: "boundary-hostile",
        requiredExtensions: [],
        payload: payload as unknown as LogicalGemmTileScheduleArtifactPayloadV1,
      }, { logicalGemm: logical.kernel }));
      expect(error.diagnostic.code).toBe(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile);
    }
  });

  it("round-trips canonical bytes and rejects artifact forgery", async () => {
    const logical = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const constructed = await createVerifiedLogicalGemmTileSchedule(logical.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const envelope = {
      schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "roundtrip", version: "1" },
      artifactId: "roundtrip",
      requiredExtensions: [],
      payload: logicalGemmTileScheduleArtifactPayload(constructed.artifact),
    };
    const decoded = await decodeLogicalGemmTileScheduleArtifact(canonicalJsonBytes(envelope), {
      logicalGemm: logical.kernel,
    });
    expect(await hashSemanticArtifact(decoded)).toBe(constructed.scheduleSemanticHash);
    expect((await diagnostic(() => logicalGemmTileScheduleArtifactPayload(
      logicalGemmTileScheduleArtifactPayload(constructed.artifact) as never,
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
  });

  it("requires the exact verified logical GEMM semantic hash", async () => {
    const first = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const second = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const constructed = await createVerifiedLogicalGemmTileSchedule(first.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const error = await diagnostic(() => verifyLogicalGemmTileScheduleArtifact({
      schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "wrong-logical", version: "1" },
      artifactId: "wrong-logical",
      requiredExtensions: [],
      payload: logicalGemmTileScheduleArtifactPayload(constructed.artifact),
    }, { logicalGemm: second.kernel }));
    expect(error.diagnostic).toMatchObject({
      code: SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      path: "$.payload.logicalGemmSemanticHash",
    });
    expect(LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA).toBe("browsergrad.kernel.gemm-tile");
  });

  it("fails closed on backend, logical, illegal tile, mapping, vector, barrier, participation, uniformity, and mask changes", async () => {
    const logical = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("16"), n: wire("16"), k: wire("16"),
      logicalTile: { m: wire("16"), n: wire("16"), k: wire("16") },
    });
    const constructed = await createVerifiedLogicalGemmTileSchedule(logical.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
    });
    const base = logicalGemmTileScheduleArtifactPayload(constructed.artifact);
    const cases: readonly [(payload: Record<string, unknown>) => void, string][] = [
      [(payload) => { payload.backendId = "webgpu"; }, SCHEDULE_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { (payload.schedule as Record<string, unknown>).accumulation = "fast"; }, SCHEDULE_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { (payload.schedule as Record<string, unknown>).numericalPreservation = "general-f32"; }, SCHEDULE_DIAGNOSTIC_CODES.unknownField],
      [(payload) => { ((payload.schedule as Record<string, unknown>).physicalTile as Record<string, unknown>).m = "3"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { ((((payload.schedule as Record<string, unknown>).workgroup as Record<string, unknown>).size) as Record<string, unknown>).x = "4"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((payload.schedule as Record<string, unknown>).vectorization) as Record<string, unknown>).lhsLoad = "2"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((payload.schedule as Record<string, unknown>).participation) as Record<string, unknown>).boundaryLanes = "deactivate"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((payload.schedule as Record<string, unknown>).participation) as Record<string, unknown>).earlyExit = "masked-lanes"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((payload.schedule as Record<string, unknown>).uniformity) as Record<string, unknown>).activeMaskScope = "control-flow"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((((payload.schedule as Record<string, unknown>).barriers) as Record<string, unknown>).afterCooperativeLoad) as Record<string, unknown>).scope = "subgroup"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((((payload.schedule as Record<string, unknown>).barriers) as Record<string, unknown>).beforeStagingReuse) as Record<string, unknown>).semantics = "none"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
      [(payload) => { (((payload.schedule as Record<string, unknown>).masks) as Record<string, unknown>).lhsLoad = "clamp"; }, SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile],
    ];
    for (const [mutate, code] of cases) {
      const payload = clone(base) as unknown as Record<string, unknown>;
      mutate(payload);
      const error = await diagnostic(() => verifyLogicalGemmTileScheduleArtifact({
        schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
        version: { major: 1, minor: 0 },
        producer: { id: "mutation", version: "1" },
        artifactId: "mutation",
        requiredExtensions: [],
        payload: payload as unknown as LogicalGemmTileScheduleArtifactPayloadV1,
      }, { logicalGemm: logical.kernel }));
      expect(error.diagnostic.code).toBe(code);
    }

    expect((await diagnostic(() => createVerifiedLogicalGemmTileSchedule(logical.kernel, {
      physicalTile: { m: wire("8"), n: wire("8"), k: wire("8") },
      backendId: "webgpu",
    } as never))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });
});
