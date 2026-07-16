import {
  parseWireU64,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_AOT_JOB_SCHEMA,
  deriveCppCuteAotEntryRequestId,
  deriveCppCuteAotJobId,
  deriveCppCuteAotSourceFileId,
  prepareCppCuteAotJob,
  unwrapPreparedCppCuteAotJob,
  type CppCuteAotEntryRequestV1,
  type CppCuteAotJobBodyV1,
  type CppCuteAotJobV1,
  type CppCuteAotSourceFileV1,
} from "../../src/cpp_cute_aot_job.js";
import {
  prepareCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FIXTURE_ENTRY_ID,
  CPP_CUTE_FIXTURE_HEADER_SET_HASH,
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

const SOURCE = [
  "#include <cute/layout.hpp>",
  "using namespace cute;",
  "constexpr auto traced_layout = make_layout(",
  "  make_shape(Int<3>{}, Int<2>{}),",
  "  make_stride(Int<1>{}, Int<3>{}));",
  "static_assert(size(traced_layout) == 6);",
  "static_assert(cosize(traced_layout) == 6);",
  "",
].join("\n");
const SOURCE_BYTES = new TextEncoder().encode(SOURCE);
const TOKEN = "traced_layout";
const TOKEN_BEGIN = SOURCE.indexOf(TOKEN);
const wire = (value: number): WireU64 => parseWireU64(String(value));

interface JobFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly job: CppCuteAotJobV1;
}

async function createJobFixture(): Promise<JobFixture> {
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
  const requestWithoutId = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    expectedEntryId: CPP_CUTE_FIXTURE_ENTRY_ID,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor: {
      virtualPath: "/workspace/src/layout.cu",
      beginByte: wire(TOKEN_BEGIN),
      endByte: wire(TOKEN_BEGIN + TOKEN.length),
      tokenSha256: await sha256Hex(new TextEncoder().encode(TOKEN)),
    },
  } satisfies CppCuteAotEntryRequestV1;
  const request: CppCuteAotEntryRequestV1 = {
    ...requestWithoutId,
    requestId: await deriveCppCuteAotEntryRequestId(requestWithoutId),
  };
  const sourceFileWithoutId = {
    fileId: `bg.cpp.file.sha256.${"0".repeat(64)}`,
    role: "main-source" as const,
    virtualPath: "/workspace/src/layout.cu",
    contentSha256: await sha256Hex(SOURCE_BYTES),
    byteLength: wire(SOURCE_BYTES.byteLength),
  } satisfies CppCuteAotSourceFileV1;
  const sourceFile: CppCuteAotSourceFileV1 = {
    ...sourceFileWithoutId,
    fileId: await deriveCppCuteAotSourceFileId(sourceFileWithoutId),
  };
  const body = {
    schema: CPP_CUTE_AOT_JOB_SCHEMA,
    version: { major: 1 as const, minor: 0 as const },
    profileHash: profile.profileHash,
    source: {
      repository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
      revision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    },
    mainVirtualPath: "/workspace/src/layout.cu",
    files: [sourceFile],
    entryRequests: [request],
    expectedOutput: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact" as const,
      version: { major: 1 as const, minor: 0 as const },
      sourceSetSha256: "a".repeat(64),
      headerSetSha256: CPP_CUTE_FIXTURE_HEADER_SET_HASH,
      inputClosureSha256: "b".repeat(64),
    },
  } satisfies CppCuteAotJobBodyV1;
  const job: CppCuteAotJobV1 = {
    ...body,
    jobId: await deriveCppCuteAotJobId(body),
  };
  return { profile, job };
}

describe("C++/CuTe AOT producer request", () => {
  it("prepares one content-addressed profile-bound source and declaration request", async () => {
    const fixture = await createJobFixture();
    const prepared = await prepareCppCuteAotJob(fixture.profile, fixture.job);
    const record = unwrapPreparedCppCuteAotJob(prepared);

    expect(prepared).toEqual({
      jobId: fixture.job.jobId,
      profileHash: fixture.profile.profileHash,
      mainVirtualPath: "/workspace/src/layout.cu",
      sourceFileCount: 1,
      sourceBytes: String(SOURCE_BYTES.byteLength),
      entryRequestId: fixture.job.entryRequests[0]?.requestId,
      expectedEntryId: CPP_CUTE_FIXTURE_ENTRY_ID,
    });
    expect(prepared.jobId).toBe(
      "bg.cpp.aot-job.sha256.1800d5b8da0ac964665364e006a0d19fcfb0868878afaf1cd4bd0275d4f72961",
    );
    expect(prepared.entryRequestId).toBe(
      "bg.cpp.entry-request.sha256.b3bcfcdba40d2d0980480de85525f5ec9e30370d404b8fec4815f5b2ecaa9af2",
    );
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(record.profile).toBe(fixture.profile);
    expect(record.job).not.toBe(fixture.job);
    expect(Object.isFrozen(record.job)).toBe(true);
    expect(record.job.files[0]?.fileId).toBe(
      "bg.cpp.file.sha256.73153ae2fbc60cd447e711b94af5005399a520da45fdb6140ede80bd636047a1",
    );
    expect(record.job).not.toHaveProperty("command");
    expect(record.job).not.toHaveProperty("compilerFlags");
    expect(record.job).not.toHaveProperty("environment");
    expect(record.job).not.toHaveProperty("outputPath");
  });

  it("rejects profile and content-address drift before authority", async () => {
    const fixture = await createJobFixture();
    const profileDrift = structuredClone(fixture.job) as CppCuteAotJobV1;
    (profileDrift as { profileHash: string }).profileHash = "f".repeat(64);
    await expect(prepareCppCuteAotJob(fixture.profile, profileDrift)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-PROFILE-MISMATCH",
      path: "$.profileHash",
    });

    const headerDrift = structuredClone(fixture.job) as CppCuteAotJobV1;
    (headerDrift.expectedOutput as { headerSetSha256: string }).headerSetSha256 = "f".repeat(64);
    await expect(prepareCppCuteAotJob(fixture.profile, headerDrift)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-PROFILE-MISMATCH",
      path: "$.expectedOutput.headerSetSha256",
    });

    const requestDrift = structuredClone(fixture.job) as CppCuteAotJobV1;
    (requestDrift.entryRequests[0]?.anchor as { tokenSha256: string }).tokenSha256 = "f".repeat(64);
    await expect(prepareCppCuteAotJob(fixture.profile, requestDrift)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH",
      path: "$.entryRequests[0].requestId",
    });

    const fileDrift = structuredClone(fixture.job) as CppCuteAotJobV1;
    (fileDrift.files[0] as { contentSha256: string }).contentSha256 = "f".repeat(64);
    await expect(prepareCppCuteAotJob(fixture.profile, fileDrift)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH",
      path: "$.files[0].fileId",
    });

    const jobDrift = structuredClone(fixture.job) as CppCuteAotJobV1;
    (jobDrift.expectedOutput as { sourceSetSha256: string }).sourceSetSha256 = "f".repeat(64);
    await expect(prepareCppCuteAotJob(fixture.profile, jobDrift)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH",
      path: "$.jobId",
    });
  });

  it("rejects operational escape hatches, accessors, and structural authority copies", async () => {
    const fixture = await createJobFixture();
    await expect(prepareCppCuteAotJob(fixture.profile, {
      ...fixture.job,
      command: ["clang++", "@response-file"],
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID" });

    const hostile = structuredClone(fixture.job) as unknown as Record<string, unknown>;
    Object.defineProperty(hostile, "files", { enumerable: true, get: () => [] });
    await expect(prepareCppCuteAotJob(fixture.profile, hostile)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID",
    });

    const prepared = await prepareCppCuteAotJob(fixture.profile, fixture.job);
    expect(() => unwrapPreparedCppCuteAotJob({ ...prepared } as never)).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-UNVERIFIED",
    }));
  });

  it("enforces source VFS, ordering, main ownership, and declaration-anchor bounds", async () => {
    const fixture = await createJobFixture();
    const cases: Array<(job: Record<string, unknown>) => void> = [
      (job) => {
        const files = job["files"] as Record<string, unknown>[];
        if (files[0] === undefined) throw new Error("fixture lost file");
        files[0]["virtualPath"] = "/workspace/src2/layout.cu";
      },
      (job) => { job["mainVirtualPath"] = "/workspace/src/missing.cu"; },
      (job) => {
        const requests = job["entryRequests"] as Array<{ anchor: Record<string, unknown> }>;
        if (requests[0] === undefined) throw new Error("fixture lost request");
        requests[0].anchor["endByte"] = String(SOURCE_BYTES.byteLength + 1);
      },
      (job) => {
        const requests = job["entryRequests"] as Array<{ anchor: Record<string, unknown> }>;
        if (requests[0] === undefined) throw new Error("fixture lost request");
        requests[0].anchor["virtualPath"] = "/workspace/src/header.cuh";
      },
    ];
    for (const mutate of cases) {
      const value = structuredClone(fixture.job) as unknown as Record<string, unknown>;
      mutate(value);
      await expect(prepareCppCuteAotJob(fixture.profile, value)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID",
      });
    }
  });

  it("applies profile file/byte ceilings and semantic decode budgets", async () => {
    const fixture = await createJobFixture();
    const constrainedInput = createCppCuteProfileInput();
    (constrainedInput.extractionLimits as { maxSourceBytes: number }).maxSourceBytes = SOURCE_BYTES.byteLength - 1;
    const constrained = await prepareCppCuteFrontendProfile(constrainedInput);
    const constrainedJob = structuredClone(fixture.job) as CppCuteAotJobV1;
    (constrainedJob as { profileHash: string }).profileHash = constrained.profileHash;
    await expect(prepareCppCuteAotJob(constrained, constrainedJob)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-RESOURCE-LIMIT",
      path: "$.files",
    });

    await expect(prepareCppCuteAotJob(fixture.profile, fixture.job, {
      limits: { maxNodes: 4 },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-JOB-RESOURCE-LIMIT",
    });
  });

  it("rejects unsupported versions and cancellation", async () => {
    const fixture = await createJobFixture();
    await expect(prepareCppCuteAotJob(fixture.profile, {
      ...fixture.job,
      version: { major: 2, minor: 0 },
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-JOB-UNSUPPORTED-VERSION" });

    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteAotJob(fixture.profile, fixture.job, {
      signal: controller.signal,
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-JOB-CANCELLED" });
  });
});
