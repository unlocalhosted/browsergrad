import { describe, expect, it } from "vitest";
import {
  copyCppCuteAotOfflineResultBytes,
  copyCppCuteAotOfflineRunSourceSnapshots,
  copyCppCuteAotOfflineRunStagingInputs,
  decodeCppCuteAotResultFrame,
  encodeCppCuteAotResultFrame,
  prepareCppCuteAotOfflineRun,
  unwrapPreparedCppCuteAotOfflineRun,
  unwrapVerifiedCppCuteAotOfflineResult,
} from "../../src/cpp_cute_aot_runner_plan.js";
import { canonicalJsonBytes, decodeWireJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createCppCuteAotRunnerFixture } from "./support/cpp_cute_aot_runner_fixtures.js";
import { createCppCuteAotExecutionEnvironmentFixture } from "./support/cpp_cute_aot_environment_fixtures.js";

describe("AOT runner plan request authority", () => {
  it("derives source snapshots and staging only from run metadata request", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const record = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    expect(fixture.plan).toMatchObject({
      runMetadataId: record.metadata.runMetadataId,
      requestId: record.metadata.requestId,
      profileHash: fixture.profile.profileHash,
      sourceFileCount: 1,
    });
    const sources = copyCppCuteAotOfflineRunSourceSnapshots(fixture.plan);
    const staging = copyCppCuteAotOfflineRunStagingInputs(fixture.plan);
    expect(staging.sourceSnapshots).toHaveLength(1);
    expect(decodeWireJson(staging.requestBytes)).toMatchObject({ requestId: fixture.plan.requestId });
    expect(decodeWireJson(staging.runMetadataBytes)).toMatchObject({ runMetadataId: fixture.plan.runMetadataId });
    expect(staging).not.toHaveProperty("jobBytes");
    sources[0]!.bytes.fill(0);
    staging.sourceSnapshots[0]!.bytes.fill(0);
    expect(copyCppCuteAotOfflineRunSourceSnapshots(fixture.plan)[0]!.bytes.some((value) => value !== 0)).toBe(true);
  });

  it("rejects metadata/environment profile cross-wiring", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const other = await createCppCuteAotExecutionEnvironmentFixture();
    const metadata = unwrapPreparedCppCuteAotOfflineRun(fixture.plan).metadata;
    await expect(prepareCppCuteAotOfflineRun(metadata, other.environment)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-POLICY-MISMATCH",
      path: "$.executionEnvironment",
    });
  });

  it.each(["accepted", "rejected"] as const)("decodes an exact %s artifact-binding-receipt frame", async (outcome) => {
    const fixture = await createCppCuteAotRunnerFixture({}, { outcome });
    const frame = encodeCppCuteAotResultFrame(fixture.artifactBytes, fixture.receiptBytes);
    const result = await decodeCppCuteAotResultFrame(fixture.plan, frame);
    expect(result).toMatchObject({
      runMetadataId: fixture.plan.runMetadataId,
      requestId: fixture.plan.requestId,
      frontendOutcome: outcome,
    });
    expect(result.requestBinding.outcome).toBe(outcome);
    expect(unwrapVerifiedCppCuteAotOfflineResult(result).plan).toBe(fixture.plan);
    const copies = copyCppCuteAotOfflineResultBytes(result);
    expect(copies.artifactBytes).toEqual(fixture.artifactBytes);
    expect(copies.receiptBytes).toEqual(fixture.receiptBytes);
  });

  it("rejects malformed, trailing, and cross-wired result frames", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const other = await createCppCuteAotRunnerFixture({ trustStoreSha256: "e".repeat(64) });
    await expect(decodeCppCuteAotResultFrame(fixture.plan, new Uint8Array())).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID",
    });
    const valid = encodeCppCuteAotResultFrame(fixture.artifactBytes, fixture.receiptBytes);
    const trailing = new Uint8Array(valid.byteLength + 1);
    trailing.set(valid);
    await expect(decodeCppCuteAotResultFrame(fixture.plan, trailing)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID",
    });
    const otherReceipt = canonicalJsonBytes(decodeWireJson(other.receiptBytes));
    await expect(decodeCppCuteAotResultFrame(
      fixture.plan,
      encodeCppCuteAotResultFrame(fixture.artifactBytes, otherReceipt),
    )).rejects.toBeDefined();
  });
});
