import { describe, expect, it } from "vitest";
import {
  canonicalCppCuteAotRunnerReceiptBytes,
  decodeCppCuteAotRunnerReceipt,
  deriveCppCuteAotRunnerReceiptId,
  unwrapVerifiedCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceiptResource,
  verifyCppCuteAotRunnerReceipt,
} from "../../src/cpp_cute_aot_receipt.js";
import { createCppCuteAotExecutionEnvironmentFixture } from "./support/cpp_cute_aot_environment_fixtures.js";
import { createCppCuteAotReceiptFixture } from "./support/cpp_cute_aot_receipt_fixtures.js";
import { artifactCompatibleProfileOptions } from "./support/cpp_cute_frontend_fixtures.js";

async function fixture(outcome: "accepted" | "rejected" = "accepted") {
  const environment = await createCppCuteAotExecutionEnvironmentFixture({
    profile: artifactCompatibleProfileOptions("d".repeat(64)),
  });
  return createCppCuteAotReceiptFixture(environment.profile, environment.environment, outcome);
}

describe("AOT runner receipt v3", () => {
  it.each(["accepted", "rejected"] as const)("verifies an exact %s metadata-request-binding chain", async (outcome) => {
    const value = await fixture(outcome);
    const verified = await verifyCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      value.requestBinding,
      value.receipt,
    );
    expect(verified).toMatchObject({
      runMetadataId: value.metadata.runMetadataId,
      requestId: value.request.requestId,
      requestBindingId: value.requestBinding.bindingId,
      artifactId: value.artifact.artifactId,
    });
    const record = unwrapVerifiedCppCuteAotRunnerReceipt(verified);
    expect(record.metadata).toBe(value.metadata);
    expect(record.requestBinding).toBe(value.requestBinding);
    expect(record.artifactResource).toBe(value.artifactResource);
  });

  it("mints byte authority only for exact canonical bytes", async () => {
    const value = await fixture();
    const verified = await verifyCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      value.requestBinding,
      value.receipt,
    );
    const bytes = canonicalCppCuteAotRunnerReceiptBytes(verified);
    const resource = await decodeCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      value.requestBinding,
      bytes,
    );
    expect(unwrapVerifiedCppCuteAotRunnerReceiptResource(resource)).toBeDefined();
    const noncanonical = new Uint8Array(bytes.byteLength + 1);
    noncanonical.set(bytes);
    noncanonical[bytes.byteLength] = 0x20;
    await expect(decodeCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      value.requestBinding,
      noncanonical,
    )).rejects.toBeDefined();
  });

  it("rejects cross-wired metadata, binding, and observed closure", async () => {
    const value = await fixture();
    const other = await fixture();
    await expect(verifyCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      other.requestBinding,
      value.receipt,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH" });

    const drift = structuredClone(value.receipt);
    (drift.openedInputs as { inputClosureSha256: string }).inputClosureSha256 = "f".repeat(64);
    (drift as { receiptId: string }).receiptId = await deriveCppCuteAotRunnerReceiptId(drift);
    await expect(verifyCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      value.requestBinding,
      drift,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
      path: "$.openedInputs",
    });
  });

  it("rejects legacy job, duplicated files, and duplicated selection fields", async () => {
    const value = await fixture();
    for (const legacy of ["jobId", "selection"] as const) {
      const drift = structuredClone(value.receipt) as Record<string, unknown>;
      drift[legacy] = legacy === "jobId" ? `bg.cpp.aot-job.sha256.${"0".repeat(64)}` : {};
      await expect(verifyCppCuteAotRunnerReceipt(
        value.metadata,
        value.executionEnvironment,
        value.requestBinding,
        drift,
      )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID" });
    }
    const files = structuredClone(value.receipt) as unknown as { openedInputs: Record<string, unknown> };
    files.openedInputs.files = [];
    await expect(verifyCppCuteAotRunnerReceipt(
      value.metadata,
      value.executionEnvironment,
      value.requestBinding,
      files,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID" });
  });
});
