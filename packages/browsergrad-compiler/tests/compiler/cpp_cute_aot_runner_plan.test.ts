import { describe, expect, it } from "vitest";
import { parseWireU64, sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  deriveCppCuteAotEntryRequestId,
  deriveCppCuteAotJobId,
  deriveCppCuteAotSourceFileId,
  prepareCppCuteAotJob,
  unwrapPreparedCppCuteAotJob,
} from "../../src/cpp_cute_aot_job.js";
import {
  CPP_CUTE_AOT_RESULT_FRAME_MAGIC,
  CppCuteAotOfflineRunnerError,
  copyCppCuteAotOfflineResultBytes,
  copyCppCuteAotOfflineRunSourceBlobs,
  decodeCppCuteAotResultFrame,
  encodeCppCuteAotResultFrame,
  prepareCppCuteAotOfflineRun,
  unwrapPreparedCppCuteAotOfflineRun,
  unwrapVerifiedCppCuteAotOfflineResult,
  type PreparedCppCuteAotOfflineRun,
} from "../../src/cpp_cute_aot_runner_plan.js";
import {
  CPP_CUTE_AOT_SANDBOX_POLICY_SHA256,
} from "../../src/cpp_cute_aot_policy.js";
import { createCppCuteAotRunnerFixture } from "./support/cpp_cute_aot_runner_fixtures.js";

describe("C++/CuTe offline AOT runner plan", () => {
  it("snapshots exact source bytes behind opaque plan authority", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const record = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    expect(fixture.plan).toMatchObject({
      imageReference: expect.stringMatching(/@sha256:[0-9a-f]{64}$/u),
      imageConfigDigest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u),
      sourceFileCount: 1,
      sourceBytes: "100",
      executionPlanSha256: expect.stringMatching(/^[0-9a-f]{64}$/u),
    });
    const firstSourceCopy = copyCppCuteAotOfflineRunSourceBlobs(fixture.plan);
    expect(firstSourceCopy[0]?.bytes).toEqual(fixture.sourceBlob.bytes);
    firstSourceCopy[0]?.bytes.fill(0xee);
    expect(copyCppCuteAotOfflineRunSourceBlobs(fixture.plan)[0]?.bytes).toEqual(fixture.sourceBlob.bytes);
    expect(Object.isFrozen(fixture.plan)).toBe(true);
    expect(() => unwrapPreparedCppCuteAotOfflineRun({ ...fixture.plan } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-UNVERIFIED" }),
    );

    const callerBytes = new Uint8Array(fixture.sourceBlob.bytes);
    const pending = prepareCppCuteAotOfflineRun(record.job, [{
      fileId: fixture.sourceBlob.fileId,
      bytes: callerBytes,
    }]);
    callerBytes.fill(0xff);
    const snapshotted = await pending;
    expect(copyCppCuteAotOfflineRunSourceBlobs(snapshotted)[0]?.bytes).toEqual(
      fixture.sourceBlob.bytes,
    );
  });

  it("rejects missing, extra, mutated, shared, and accessor source blobs", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const { job } = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    await expect(prepareCppCuteAotOfflineRun(job, [])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      path: "$sourceBlobs",
    });
    await expect(prepareCppCuteAotOfflineRun(job, [fixture.sourceBlob, fixture.sourceBlob])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
    });
    const mutated = new Uint8Array(fixture.sourceBlob.bytes);
    mutated[1] = (mutated[1] ?? 0) ^ 1;
    await expect(prepareCppCuteAotOfflineRun(job, [{
      fileId: fixture.sourceBlob.fileId,
      bytes: mutated,
    }])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      path: "$sourceBlobs[0].bytes",
    });
    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(prepareCppCuteAotOfflineRun(job, [{
        fileId: fixture.sourceBlob.fileId,
        bytes: new Uint8Array(new SharedArrayBuffer(100)),
      }])).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-INVALID" });
    }
    const disguisedWords = new Uint16Array(50);
    Object.setPrototypeOf(disguisedWords, Uint8Array.prototype);
    await expect(prepareCppCuteAotOfflineRun(job, [{
      fileId: fixture.sourceBlob.fileId,
      bytes: disguisedWords as unknown as Uint8Array,
    }])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-INVALID",
      path: "$sourceBlobs[0].bytes",
    });
    const accessor = { fileId: fixture.sourceBlob.fileId } as Record<string, unknown>;
    Object.defineProperty(accessor, "bytes", {
      enumerable: true,
      get: () => fixture.sourceBlob.bytes,
    });
    await expect(prepareCppCuteAotOfflineRun(job, [accessor as never])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-INVALID",
      path: "$sourceBlobs[0].bytes",
    });
    const accessorArray: unknown[] = [];
    Object.defineProperty(accessorArray, "0", {
      enumerable: true,
      get: () => fixture.sourceBlob,
    });
    await expect(prepareCppCuteAotOfflineRun(job, accessorArray as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-INVALID",
      path: "$sourceBlobs[0]",
    });

    const oversized = new Uint8Array(fixture.sourceBlob.bytes.byteLength + 1_048_576);
    await expect(prepareCppCuteAotOfflineRun(job, [{
      fileId: fixture.sourceBlob.fileId,
      bytes: oversized,
    }])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      path: "$sourceBlobs[0].bytes",
    });
  });

  it("uses captured descriptors instead of reopening proxy getters", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const { job } = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    const sourceBytes = new Uint8Array(fixture.sourceBlob.bytes);
    Object.defineProperty(sourceBytes, "constructor", {
      get: () => {
        throw new Error("typed-array species access must not run");
      },
    });
    let entryOwnKeysCalls = 0;
    const entry = new Proxy({
      fileId: fixture.sourceBlob.fileId,
      bytes: sourceBytes,
    }, {
      get: () => {
        throw new Error("ordinary entry get must not run");
      },
      ownKeys: (target) => {
        entryOwnKeysCalls += 1;
        if (entryOwnKeysCalls > 1) throw new Error("entry ownKeys reopened");
        return Reflect.ownKeys(target);
      },
    });
    let arrayOwnKeysCalls = 0;
    const sourceBlobs = new Proxy([entry], {
      get: () => {
        throw new Error("ordinary array get must not run");
      },
      ownKeys: (target) => {
        arrayOwnKeysCalls += 1;
        if (arrayOwnKeysCalls > 1) throw new Error("array ownKeys reopened");
        return Reflect.ownKeys(target);
      },
    });
    let optionsOwnKeysCalls = 0;
    const options = new Proxy({}, {
      get: () => {
        throw new Error("ordinary options get must not run");
      },
      ownKeys: (target) => {
        optionsOwnKeysCalls += 1;
        if (optionsOwnKeysCalls > 1) throw new Error("options ownKeys reopened");
        return Reflect.ownKeys(target);
      },
    });

    await expect(prepareCppCuteAotOfflineRun(job, sourceBlobs, options)).resolves.toMatchObject({
      jobId: job.jobId,
    });
    expect({ arrayOwnKeysCalls, entryOwnKeysCalls, optionsOwnKeysCalls }).toEqual({
      arrayOwnKeysCalls: 1,
      entryOwnKeysCalls: 1,
      optionsOwnKeysCalls: 1,
    });
  });

  it("rejects true duplicate and unknown IDs before copying a multi-file request", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const runRecord = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    const jobRecord = unwrapPreparedCppCuteAotJob(runRecord.job);
    const jobValue = structuredClone(jobRecord.job);
    const headerBytes = new Uint8Array([1, 2, 3]);
    const headerBody = {
      fileId: `bg.cpp.file.sha256.${"0".repeat(64)}`,
      role: "project-header" as const,
      virtualPath: "/src/project.hpp",
      contentSha256: await sha256Hex(headerBytes),
      byteLength: parseWireU64(String(headerBytes.byteLength)),
    };
    const header = { ...headerBody, fileId: await deriveCppCuteAotSourceFileId(headerBody) };
    (jobValue.files as unknown as Array<typeof header>).push(header);
    (jobValue.files as unknown as Array<typeof header>).sort((left, right) =>
      left.virtualPath.localeCompare(right.virtualPath));
    (jobValue as { jobId: string }).jobId = await deriveCppCuteAotJobId(jobValue);
    const twoFileJob = await prepareCppCuteAotJob(jobRecord.profile, jobValue);

    await expect(prepareCppCuteAotOfflineRun(twoFileJob, [
      fixture.sourceBlob,
      { fileId: fixture.sourceBlob.fileId, bytes: headerBytes },
    ])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      path: "$sourceBlobs[1].fileId",
    });
    await expect(prepareCppCuteAotOfflineRun(twoFileJob, [
      fixture.sourceBlob,
      { fileId: `bg.cpp.file.sha256.${"f".repeat(64)}`, bytes: headerBytes },
    ])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      path: "$sourceBlobs[1].fileId",
    });
  });

  it("rechecks the declaration anchor against the snapshotted source", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const runRecord = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    const jobRecord = unwrapPreparedCppCuteAotJob(runRecord.job);
    const jobValue = structuredClone(jobRecord.job);
    const request = jobValue.entryRequests[0];
    if (request === undefined) throw new Error("runner fixture lost entry request");
    (request.anchor as { tokenSha256: string }).tokenSha256 = "0".repeat(64);
    (request as { requestId: string }).requestId = await deriveCppCuteAotEntryRequestId(request);
    (jobValue as { jobId: string }).jobId = await deriveCppCuteAotJobId(jobValue);
    const anchorMismatchJob = await prepareCppCuteAotJob(jobRecord.profile, jobValue);

    await expect(prepareCppCuteAotOfflineRun(anchorMismatchJob, [fixture.sourceBlob])).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      path: "$.job.entryRequests[0].anchor.tokenSha256",
    });
  });

  it("strict-decodes exactly one bounded artifact/receipt frame", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const frame = encodeCppCuteAotResultFrame(fixture.artifactBytes, fixture.receiptBytes);
    const result = await decodeCppCuteAotResultFrame(fixture.plan, frame);
    const record = unwrapVerifiedCppCuteAotOfflineResult(result);
    const resultBytes = copyCppCuteAotOfflineResultBytes(result);
    const originalArtifactBytes = new Uint8Array(fixture.artifactBytes);
    const originalReceiptBytes = new Uint8Array(fixture.receiptBytes);
    expect(result).toMatchObject({
      jobId: fixture.plan.jobId,
      profileHash: fixture.plan.profileHash,
      executionPlanSha256: fixture.plan.executionPlanSha256,
      artifactByteLength: String(fixture.artifactBytes.byteLength),
      receiptByteLength: String(fixture.receiptBytes.byteLength),
      frontendOutcome: "accepted",
    });
    expect(resultBytes.artifactBytes).toEqual(fixture.artifactBytes);
    expect(resultBytes.receiptBytes).toEqual(fixture.receiptBytes);
    expect(record.plan).toBe(fixture.plan);
    expect(Object.isFrozen(result)).toBe(true);

    resultBytes.artifactBytes.fill(0xaa);
    resultBytes.receiptBytes.fill(0xbb);
    const secondCopy = copyCppCuteAotOfflineResultBytes(result);
    expect(secondCopy.artifactBytes).toEqual(originalArtifactBytes);
    expect(secondCopy.receiptBytes).toEqual(originalReceiptBytes);

    fixture.artifactBytes.fill(0);
    fixture.receiptBytes.fill(0);
    expect(secondCopy.artifactBytes).not.toEqual(fixture.artifactBytes);
    expect(secondCopy.receiptBytes).not.toEqual(fixture.receiptBytes);
  });

  it("preserves a successful producer run with a rejected frontend outcome", async () => {
    const fixture = await createCppCuteAotRunnerFixture({}, { outcome: "rejected" });
    const result = await decodeCppCuteAotResultFrame(
      fixture.plan,
      encodeCppCuteAotResultFrame(fixture.artifactBytes, fixture.receiptBytes),
    );

    expect(result.frontendOutcome).toBe("rejected");
    expect(copyCppCuteAotOfflineResultBytes(result)).toMatchObject({
      artifactBytes: fixture.artifactBytes,
      receiptBytes: fixture.receiptBytes,
    });
  });

  it("rejects malformed, truncated, trailing, oversized, and noncanonical frames", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const frame = encodeCppCuteAotResultFrame(fixture.artifactBytes, fixture.receiptBytes);
    const magicOffset = new TextEncoder().encode(CPP_CUTE_AOT_RESULT_FRAME_MAGIC).byteLength;
    const malformedMagic = new Uint8Array(frame);
    malformedMagic[0] = (malformedMagic[0] ?? 0) ^ 1;
    const truncated = frame.slice(0, frame.byteLength - 1);
    const trailing = new Uint8Array(frame.byteLength + 1);
    trailing.set(frame);
    const oversized = new Uint8Array(frame);
    new DataView(oversized.buffer).setBigUint64(
      magicOffset,
      BigInt(fixture.plan.artifactByteLimit + 1),
      false,
    );
    const oversizedReceipt = new Uint8Array(frame);
    new DataView(oversizedReceipt.buffer).setBigUint64(
      magicOffset + 8,
      BigInt(fixture.plan.receiptByteLimit + 1),
      false,
    );
    const maximumLength = new Uint8Array(frame);
    new DataView(maximumLength.buffer).setBigUint64(magicOffset, 0xffff_ffff_ffff_ffffn, false);
    const zeroLength = new Uint8Array(frame);
    new DataView(zeroLength.buffer).setBigUint64(magicOffset, 0n, false);
    for (const candidate of [malformedMagic, truncated, trailing]) {
      await expect(decodeCppCuteAotResultFrame(fixture.plan, candidate)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID",
      });
    }
    await expect(decodeCppCuteAotResultFrame(fixture.plan, oversized)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-RESOURCE-LIMIT",
      path: "$bytes.artifact",
    });
    await expect(decodeCppCuteAotResultFrame(fixture.plan, oversizedReceipt)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-RESOURCE-LIMIT",
      path: "$bytes.receipt",
    });
    await expect(decodeCppCuteAotResultFrame(fixture.plan, maximumLength)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-RESOURCE-LIMIT",
      path: "$bytes.artifact",
    });
    await expect(decodeCppCuteAotResultFrame(fixture.plan, zeroLength)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID",
    });

    const noncanonicalArtifact = new TextEncoder().encode(JSON.stringify(
      JSON.parse(new TextDecoder().decode(fixture.artifactBytes)),
      null,
      2,
    ));
    await expect(decodeCppCuteAotResultFrame(
      fixture.plan,
      encodeCppCuteAotResultFrame(noncanonicalArtifact, fixture.receiptBytes),
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-ARTIFACT-NONCANONICAL-BYTES" });
    const noncanonicalReceipt = new TextEncoder().encode(JSON.stringify(
      JSON.parse(new TextDecoder().decode(fixture.receiptBytes)),
      null,
      2,
    ));
    await expect(decodeCppCuteAotResultFrame(
      fixture.plan,
      encodeCppCuteAotResultFrame(fixture.artifactBytes, noncanonicalReceipt),
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-NONCANONICAL-BYTES" });
  });

  it("copies byte-offset views and rejects proxy/shared frame buffers", async () => {
    const fixture = await createCppCuteAotRunnerFixture();
    const artifactBacking = new Uint8Array(fixture.artifactBytes.byteLength + 2);
    artifactBacking.set(fixture.artifactBytes, 1);
    const receiptBacking = new Uint8Array(fixture.receiptBytes.byteLength + 4);
    receiptBacking.set(fixture.receiptBytes, 2);
    const encoded = encodeCppCuteAotResultFrame(
      artifactBacking.subarray(1, 1 + fixture.artifactBytes.byteLength),
      receiptBacking.subarray(2, 2 + fixture.receiptBytes.byteLength),
    );
    const backing = new Uint8Array(encoded.byteLength + 8);
    backing.set(encoded, 4);
    await expect(decodeCppCuteAotResultFrame(
      fixture.plan,
      backing.subarray(4, 4 + encoded.byteLength),
    )).resolves.toMatchObject({ frontendOutcome: "accepted" });
    await expect(decodeCppCuteAotResultFrame(
      fixture.plan,
      new Proxy(encoded, {}) as Uint8Array,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID" });
    const disguisedWords = new Uint16Array(Math.ceil(encoded.byteLength / 2));
    Object.setPrototypeOf(disguisedWords, Uint8Array.prototype);
    await expect(decodeCppCuteAotResultFrame(
      fixture.plan,
      disguisedWords as unknown as Uint8Array,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID" });
    expect(() => encodeCppCuteAotResultFrame(
      disguisedWords as unknown as Uint8Array,
      new Uint8Array([1]),
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID",
    }));

    if (typeof SharedArrayBuffer !== "undefined") {
      expect(() => encodeCppCuteAotResultFrame(
        new Uint8Array(new SharedArrayBuffer(1)),
        new Uint8Array([1]),
      )).toThrowError(expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID",
      }));
    }
  });

  it("rejects policy substitution, cancellation, and structural plan copies", async () => {
    await expect(createCppCuteAotRunnerFixture({
      sandboxPolicySha256: "f".repeat(64),
    })).rejects.toThrowError(/POLICY-UNSUPPORTED|POLICY-MISMATCH/u);

    const fixture = await createCppCuteAotRunnerFixture();
    const { job } = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteAotOfflineRun(job, [fixture.sourceBlob], {
      signal: controller.signal,
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-CANCELLED" });
    await expect(decodeCppCuteAotResultFrame(
      { ...fixture.plan } as unknown as PreparedCppCuteAotOfflineRun,
      encodeCppCuteAotResultFrame(fixture.artifactBytes, fixture.receiptBytes),
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-UNVERIFIED" });
    expect(CPP_CUTE_AOT_SANDBOX_POLICY_SHA256).toMatch(/^[0-9a-f]{64}$/u);
    expect(CppCuteAotOfflineRunnerError).toBeDefined();
  });
});
