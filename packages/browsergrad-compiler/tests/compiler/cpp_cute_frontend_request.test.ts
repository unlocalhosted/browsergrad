import { describe, expect, it } from "vitest";
import { encodeWireU64, sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  CppCuteFrontendRequestError,
  copyPreparedCppCuteFrontendSourceBytes,
  copyPreparedCppCuteFrontendSourceSnapshots,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  unwrapPreparedCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestSourceFileV1,
  type CppCuteFrontendRequestV1,
  type CppCuteFrontendSourceSnapshotInput,
  type PreparedCppCuteFrontendRequest,
} from "../../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../src/cpp_cute_frontend_types.js";
import {
  createCppCuteBrowserProfileInput,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

const ENCODER = new TextEncoder();
const MAIN_PATH = "/workspace/src/main.cu";
const HEADER_PATH = "/workspace/src/project.hpp";
const MAIN_TEXT = '#include "project.hpp"\nauto layout = make_layout(Int<2>{});\n';
const HEADER_TEXT = "constexpr int project_value = 2;\n";
const TOKEN = "layout";

interface RequestFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly input: CppCuteFrontendRequestV1;
  readonly snapshots: readonly CppCuteFrontendSourceSnapshotInput[];
}

async function createRequestFixture(
  suppliedProfile?: PreparedCppCuteFrontendProfile,
): Promise<RequestFixture> {
  const profile = suppliedProfile ?? await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
  const mainBytes = ENCODER.encode(MAIN_TEXT);
  const headerBytes = ENCODER.encode(HEADER_TEXT);
  const files = await Promise.all([
    createFile("main-source", MAIN_PATH, null, mainBytes),
    createFile("project-header", HEADER_PATH, "workspace-source", headerBytes),
  ]);
  files.sort((left, right) => left.virtualPath.localeCompare(right.virtualPath));
  const tokenStart = mainBytesIndexOf(mainBytes, ENCODER.encode(TOKEN));
  const anchor = {
    virtualPath: MAIN_PATH,
    beginByte: encodeWireU64(BigInt(tokenStart)),
    endByte: encodeWireU64(BigInt(tokenStart + TOKEN.length)),
    tokenSha256: await sha256Hex(mainBytes.subarray(tokenStart, tokenStart + TOKEN.length)),
  };
  const entryRequestBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor,
  };
  const entryRequest: CppCuteFrontendEntryRequestV1 = {
    ...entryRequestBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryRequestBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: { major: 1, minor: 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: MAIN_PATH,
    files,
    entryRequests: [entryRequest],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: semanticLimits(profile),
  };
  const input: CppCuteFrontendRequestV1 = {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
  const bytes = new Map([[MAIN_PATH, mainBytes], [HEADER_PATH, headerBytes]]);
  const snapshots = files.map((file) => ({ virtualPath: file.virtualPath, bytes: bytes.get(file.virtualPath)! }));
  return { profile, input, snapshots };
}

function semanticLimits(profile: PreparedCppCuteFrontendProfile): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    maxHeaderFiles: limits.maxHeaderFiles,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
    maxTemplateDepth: limits.maxTemplateDepth,
    maxDeclarations: limits.maxDeclarations,
    maxTypes: limits.maxTypes,
    maxConstants: limits.maxConstants,
    maxLayouts: limits.maxLayouts,
    maxTensors: limits.maxTensors,
    maxOperations: limits.maxOperations,
    maxTargetIntrinsics: limits.maxTargetIntrinsics,
    maxDiagnostics: limits.maxDiagnostics,
    maxOutputBytes: limits.maxOutputBytes,
  };
}

async function createFile(
  role: CppCuteFrontendRequestSourceFileV1["role"],
  virtualPath: string,
  includeRootId: string | null,
  bytes: Uint8Array,
): Promise<CppCuteFrontendRequestSourceFileV1> {
  const body = {
    role,
    virtualPath,
    contentSha256: await sha256Hex(bytes),
    byteLength: encodeWireU64(BigInt(bytes.byteLength)),
    includeRootId,
  };
  return { ...body, fileId: await deriveCppCuteFrontendSourceFileId(body) };
}

function cloneRequest(input: CppCuteFrontendRequestV1): CppCuteFrontendRequestV1 {
  return structuredClone(input);
}

function expectRequestError(
  promise: Promise<unknown>,
  code: CppCuteFrontendRequestError["code"],
  path: string,
): Promise<void> {
  return expect(promise).rejects.toMatchObject({ code, path });
}

function mainBytesIndexOf(haystack: Uint8Array, needle: Uint8Array): number {
  outer: for (let start = 0; start <= haystack.byteLength - needle.byteLength; start += 1) {
    for (let offset = 0; offset < needle.byteLength; offset += 1) {
      if (haystack[start + offset] !== needle[offset]) continue outer;
    }
    return start;
  }
  throw new Error("token missing");
}

describe("producer-neutral C++/CUDA/CuTe frontend request", () => {
  it("prepares deterministic opaque authority over copied caller source bytes", async () => {
    const fixture = await createRequestFixture();
    const first = await prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots);
    const second = await prepareCppCuteFrontendRequest(fixture.profile, cloneRequest(fixture.input), fixture.snapshots);
    const record = unwrapPreparedCppCuteFrontendRequest(first);

    expect(first).toEqual(second);
    expect(first.requestId).toBe(fixture.input.requestId);
    expect(first.requestHash).toBe(fixture.input.requestId.slice("bg.cpp.frontend-request.sha256.".length));
    expect(first.profileHash).toBe(fixture.profile.profileHash);
    expect(first.compilationContractHash).toBe(fixture.profile.compilationContractHash);
    expect(first.sourceFileCount).toBe(2);
    expect(first.sourceByteLength).toBe(String(ENCODER.encode(MAIN_TEXT).byteLength + ENCODER.encode(HEADER_TEXT).byteLength));
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(record.request)).toBe(true);
    expect(record.profile).toBe(fixture.profile);

    fixture.snapshots[1]!.bytes.fill(0);
    const firstCopy = copyPreparedCppCuteFrontendSourceBytes(first, HEADER_PATH);
    expect(new TextDecoder().decode(firstCopy)).toBe(HEADER_TEXT);
    firstCopy.fill(0);
    expect(new TextDecoder().decode(copyPreparedCppCuteFrontendSourceBytes(first, HEADER_PATH))).toBe(HEADER_TEXT);

    const batch = copyPreparedCppCuteFrontendSourceSnapshots(first);
    expect(batch.map((snapshot) => snapshot.virtualPath)).toEqual(record.request.files.map((file) => file.virtualPath));
    batch[0]!.bytes.fill(0);
    expect(copyPreparedCppCuteFrontendSourceSnapshots(first)[0]!.bytes).not.toEqual(batch[0]!.bytes);
    expect(Object.isFrozen(batch)).toBe(true);
    expect(Object.isFrozen(batch[0])).toBe(true);
  });

  it("keeps deployment producer identity and detached expectations outside request hash", async () => {
    const aotProfile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
    const browserProfile = await prepareCppCuteFrontendProfile(createCppCuteBrowserProfileInput());
    expect(aotProfile.compilationContractHash).toBe(browserProfile.compilationContractHash);
    const aotFixture = await createRequestFixture(aotProfile);
    const browserFixture = await createRequestFixture(browserProfile);
    const plain = await prepareCppCuteFrontendRequest(aotProfile, aotFixture.input, aotFixture.snapshots);
    const detached = await prepareCppCuteFrontendRequest(browserProfile, browserFixture.input, browserFixture.snapshots, {
      detached: {
        declaredSourceReference: { statementSha256: "a".repeat(64) },
        conformance: {
          expectedArtifactSha256: "b".repeat(64),
          expectedOpenedHeaderSetSha256: "c".repeat(64),
          expectedInputClosureSha256: "d".repeat(64),
        },
      },
    });

    expect(detached.requestId).toBe(plain.requestId);
    expect(detached.requestHash).toBe(plain.requestHash);
    expect(detached.profileHash).not.toBe(plain.profileHash);
    expect(detached.declaredSourceReferenceStatementSha256).toBe("a".repeat(64));
    expect(detached.conformanceAssertionSha256).toMatch(/^[0-9a-f]{64}$/u);
    const wireKeys = JSON.stringify(unwrapPreparedCppCuteFrontendRequest(detached).request);
    expect(wireKeys).not.toContain("repository");
    expect(wireKeys).not.toContain("revision");
    expect(wireKeys).not.toContain("container");
    expect(wireKeys).not.toContain("worker");
    expect(wireKeys).not.toContain("argv");
    expect(wireKeys).not.toContain("expectedOpenedHeaderSetSha256");
    expect(wireKeys).not.toContain("expectedInputClosureSha256");
    const requestLimits = unwrapPreparedCppCuteFrontendRequest(detached).request.limits;
    for (const key of ["maxWallTimeMs", "maxCpuTimeMs", "maxMemoryBytes", "maxProcesses"]) {
      expect(requestLimits).not.toHaveProperty(key);
    }
  });

  it("supports layout-variable and view-copy-function entry families without changing request schema", async () => {
    const fixture = await createRequestFixture();
    const input = cloneRequest(fixture.input);
    const current = input.entryRequests[0]!;
    const viewCopyCandidate: CppCuteFrontendEntryRequestV1 = {
      requestId: current.requestId,
      kind: "view-copy",
      declarationKind: "function",
      anchor: current.anchor,
    };
    const viewCopy: CppCuteFrontendEntryRequestV1 = {
      ...viewCopyCandidate,
      requestId: await deriveCppCuteFrontendEntryRequestId(viewCopyCandidate),
    };
    (input.entryRequests as CppCuteFrontendEntryRequestV1[])[0] = viewCopy;
    (input as { requestId: string }).requestId =
      `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(input)}`;

    const prepared = await prepareCppCuteFrontendRequest(fixture.profile, input, fixture.snapshots);
    expect(unwrapPreparedCppCuteFrontendRequest(prepared).request.entryRequests[0]).toMatchObject({
      kind: "view-copy",
      declarationKind: "function",
    });
  });

  it("preserves profile-owned option/include order while separating available and observed headers", async () => {
    const fixture = await createRequestFixture();
    const prepared = await prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots);
    const record = unwrapPreparedCppCuteFrontendRequest(prepared);
    const profile = unwrapPreparedCppCuteFrontendProfile(fixture.profile).profile;

    expect(record.orderedInputs.compilerOptions).toEqual(profile.language.options);
    expect(record.orderedInputs.availableIncludeRoots).toEqual(profile.virtualFileSystem.includeRoots);
    expect(record.orderedInputs.availableIncludeRoots.map((root) => root.includeRootId)).toEqual([
      "workspace-source", "clang-resource", "cuda", "cutlass", "cxx-stdlib", "linux-sysroot",
    ]);
    expect("observedOpenedHeaders" in record.orderedInputs).toBe(false);
    expect("headerSetSha256" in record.request).toBe(false);
    expect("inputClosureSha256" in record.request).toBe(false);
  });

  it("rejects producer and operational escape fields as unknown closed JSON", async () => {
    const fixture = await createRequestFixture();
    for (const key of ["repository", "revision", "container", "worker", "argv", "environment", "outputPath"]) {
      const input = cloneRequest(fixture.input) as unknown as Record<string, unknown>;
      input[key] = key;
      await expectRequestError(
        prepareCppCuteFrontendRequest(fixture.profile, input, fixture.snapshots),
        "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
        "$",
      );
    }
  });

  it("binds compilation identity and independently bounded request ceilings", async () => {
    const fixture = await createRequestFixture();
    const profileMismatch = cloneRequest(fixture.input);
    (profileMismatch as { compilationContractHash: string }).compilationContractHash = "f".repeat(64);
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, profileMismatch, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-PROFILE-MISMATCH",
      "$.compilationContractHash",
    );

    const widened = cloneRequest(fixture.input);
    (widened.limits as { maxSourceBytes: number }).maxSourceBytes = fixture.profile.extractionLimits.maxSourceBytes + 1;
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, widened, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-RESOURCE-LIMIT",
      "$.limits.maxSourceBytes",
    );

    const narrowed = cloneRequest(fixture.input);
    (narrowed.limits as { maxDiagnostics: number }).maxDiagnostics = 1;
    (narrowed as { requestId: string }).requestId =
      `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(narrowed)}`;
    const prepared = await prepareCppCuteFrontendRequest(fixture.profile, narrowed, fixture.snapshots);
    expect(unwrapPreparedCppCuteFrontendRequest(prepared).request.limits.maxDiagnostics).toBe(1);

    const sourceCount = cloneRequest(fixture.input);
    (sourceCount.limits as { maxSourceFiles: number }).maxSourceFiles = 1;
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, sourceCount, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-RESOURCE-LIMIT",
      "$.files",
    );

    const sourceBytes = cloneRequest(fixture.input);
    (sourceBytes.limits as { maxSourceBytes: number }).maxSourceBytes =
      ENCODER.encode(MAIN_TEXT).byteLength + ENCODER.encode(HEADER_TEXT).byteLength - 1;
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, sourceBytes, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-RESOURCE-LIMIT",
      "$.files",
    );

    const unknownLimit = cloneRequest(fixture.input) as unknown as { limits: Record<string, unknown> };
    unknownLimit.limits["maxGpuTimeMs"] = 1;
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, unknownLimit, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      "$.limits",
    );
  });

  it("verifies source content, length, file ID, anchor bytes, entry ID, and request ID", async () => {
    const fixture = await createRequestFixture();
    const cases: readonly [
      (input: CppCuteFrontendRequestV1) => void,
      CppCuteFrontendRequestError["code"],
      string,
    ][] = [
      [(input) => { (input.files[0] as { contentSha256: string }).contentSha256 = "f".repeat(64); }, "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", "$.files[0].contentSha256"],
      [(input) => { (input.files[1] as { byteLength: string }).byteLength = "1"; }, "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", "$.files[1].byteLength"],
      [(input) => { (input.files[0] as { fileId: string }).fileId = `bg.cpp.file.sha256.${"f".repeat(64)}`; }, "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", "$.files[0].fileId"],
      [(input) => { (input.entryRequests[0]!.anchor as { tokenSha256: string }).tokenSha256 = "f".repeat(64); }, "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", "$.entryRequests[0].anchor.tokenSha256"],
      [(input) => { (input.entryRequests[0] as { requestId: string }).requestId = `bg.cpp.entry-request.sha256.${"f".repeat(64)}`; }, "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", "$.entryRequests[0].requestId"],
      [(input) => { (input as { requestId: string }).requestId = `bg.cpp.frontend-request.sha256.${"f".repeat(64)}`; }, "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", "$.requestId"],
    ];
    for (const [mutate, code, path] of cases) {
      const input = cloneRequest(fixture.input);
      mutate(input);
      if (path !== "$.requestId") {
        (input as { requestId: string }).requestId =
          `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(input)}`;
      }
      await expectRequestError(prepareCppCuteFrontendRequest(fixture.profile, input, fixture.snapshots), code, path);
    }
  });

  it("rejects malformed snapshots, shared/proxied bytes, accessors, and post-parse structural copies", async () => {
    const fixture = await createRequestFixture();
    const accessorSnapshots = fixture.snapshots.map((entry) => ({ ...entry }));
    let getterCalls = 0;
    Object.defineProperty(accessorSnapshots[0], "bytes", {
      enumerable: true,
      get() { getterCalls += 1; return fixture.snapshots[0]!.bytes; },
    });
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, fixture.input, accessorSnapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      "$.sourceSnapshots[0].bytes",
    );
    expect(getterCalls).toBe(0);

    const proxied = fixture.snapshots.map((entry) => ({ ...entry }));
    proxied[0]!.bytes = new Proxy(fixture.snapshots[0]!.bytes, {});
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, fixture.input, proxied),
      "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      "$.sourceSnapshots[0].bytes",
    );

    if (typeof SharedArrayBuffer !== "undefined") {
      const shared = fixture.snapshots.map((entry) => ({ ...entry }));
      const bytes = new Uint8Array(new SharedArrayBuffer(fixture.snapshots[0]!.bytes.byteLength));
      bytes.set(fixture.snapshots[0]!.bytes);
      shared[0]!.bytes = bytes;
      await expectRequestError(
        prepareCppCuteFrontendRequest(fixture.profile, fixture.input, shared),
        "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
        "$.sourceSnapshots[0].bytes",
      );
    }

    const prepared = await prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots);
    expect(() => unwrapPreparedCppCuteFrontendRequest({ ...prepared } as PreparedCppCuteFrontendRequest)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-REQUEST-UNVERIFIED", path: "$" }),
    );
  });

  it("rejects accessors and polluted prototypes without invoking getters", async () => {
    const fixture = await createRequestFixture();
    const accessor = cloneRequest(fixture.input) as unknown as Record<string, unknown>;
    let getterCalls = 0;
    Object.defineProperty(accessor, "mainVirtualPath", {
      enumerable: true,
      get() { getterCalls += 1; return MAIN_PATH; },
    });
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, accessor, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      "$",
    );
    expect(getterCalls).toBe(0);

    const polluted = Object.assign(Object.create({ inherited: true }), cloneRequest(fixture.input));
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, polluted, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      "$",
    );
  });

  it("rejects detached producer records and binds conformance only out of band", async () => {
    const fixture = await createRequestFixture();
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots, {
        detached: {
          declaredSourceReference: { statementSha256: "a".repeat(64), repository: "https://example.test/repo" } as never,
          conformance: null,
        },
      }),
      "BG-COMPILER-CPP-CUTE-REQUEST-INVALID",
      "$.options.detached.declaredSourceReference",
    );

    const first = await prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots, {
      detached: {
        declaredSourceReference: null,
        conformance: {
          expectedArtifactSha256: null,
          expectedOpenedHeaderSetSha256: "1".repeat(64),
          expectedInputClosureSha256: null,
        },
      },
    });
    const second = await prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots, {
      detached: {
        declaredSourceReference: null,
        conformance: {
          expectedArtifactSha256: null,
          expectedOpenedHeaderSetSha256: "2".repeat(64),
          expectedInputClosureSha256: null,
        },
      },
    });
    expect(first.requestId).toBe(second.requestId);
    expect(first.conformanceAssertionSha256).not.toBe(second.conformanceAssertionSha256);
  });

  it("fails closed on version, cancellation, and decode budgets", async () => {
    const fixture = await createRequestFixture();
    const version = cloneRequest(fixture.input);
    (version.version as { minor: number }).minor = 1;
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, version, fixture.snapshots),
      "BG-COMPILER-CPP-CUTE-REQUEST-UNSUPPORTED-VERSION",
      "$.version",
    );

    const controller = new AbortController();
    controller.abort();
    await expectRequestError(
      prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots, { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-REQUEST-CANCELLED",
      "$.signal",
    );

    await expect(prepareCppCuteFrontendRequest(fixture.profile, fixture.input, fixture.snapshots, {
      limits: { maxArrayLength: 1 },
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-REQUEST-RESOURCE-LIMIT" });
  });
});
