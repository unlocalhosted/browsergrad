import { encodeWireU64, sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_AOT_RUN_METADATA_SCHEMA,
  CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA,
  CppCuteAotRunMetadataError,
  copyPreparedCppCuteAotRunSourceSnapshots,
  deriveCppCuteAotRunMetadataHash,
  deriveCppCuteGitSourceReferenceStatementSha256,
  prepareCppCuteAotRunMetadata,
  unwrapPreparedCppCuteAotRunMetadata,
  type CppCuteAotRunMetadataBodyV1,
  type CppCuteAotRunMetadataV1,
  type CppCuteGitSourceReferenceStatementV1,
  type PreparedCppCuteAotRunMetadata,
} from "../../src/cpp_cute_aot_run_metadata.js";
import {
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestV1,
  type PreparedCppCuteFrontendRequest,
} from "../../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../src/cpp_cute_frontend_types.js";
import { createCppCuteProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const ENCODER = new TextEncoder();
const MAIN_PATH = "/src/layout.cu";
const SOURCE_TEXT = "auto layout = make_layout(Int<2>{});\n";
const TOKEN = "layout";

interface MetadataFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly statement: CppCuteGitSourceReferenceStatementV1;
  readonly input: CppCuteAotRunMetadataV1;
}

async function createMetadataFixture(): Promise<MetadataFixture> {
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput({ sourceRoots: ["/src"] }));
  const statement: CppCuteGitSourceReferenceStatementV1 = {
    schema: CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA,
    version: { major: 1, minor: 0 },
    repository: "https://github.com/unlocalhosted/browsergrad",
    revision: { algorithm: "git-sha1", value: "1".repeat(40) },
  };
  const statementSha256 = await deriveCppCuteGitSourceReferenceStatementSha256(statement);
  const request = await createRequest(profile, statementSha256);
  const body: CppCuteAotRunMetadataBodyV1 = {
    schema: CPP_CUTE_AOT_RUN_METADATA_SCHEMA,
    version: { major: 1, minor: 0 },
    profileHash: profile.profileHash,
    requestId: request.requestId,
    declaredSourceReference: { statementSha256, statement },
  };
  const input: CppCuteAotRunMetadataV1 = {
    ...body,
    runMetadataId: `bg.cpp.aot-run-metadata.sha256.${await deriveCppCuteAotRunMetadataHash(body)}`,
  };
  return { profile, request, statement, input };
}

async function createRequest(
  profile: PreparedCppCuteFrontendProfile,
  statementSha256: string | null,
): Promise<PreparedCppCuteFrontendRequest> {
  const bytes = ENCODER.encode(SOURCE_TEXT);
  const fileBody = {
    role: "main-source" as const,
    virtualPath: MAIN_PATH,
    contentSha256: await sha256Hex(bytes),
    byteLength: encodeWireU64(BigInt(bytes.byteLength)),
    includeRootId: null,
  };
  const file = { ...fileBody, fileId: await deriveCppCuteFrontendSourceFileId(fileBody) };
  const begin = SOURCE_TEXT.indexOf(TOKEN);
  const anchor = {
    virtualPath: MAIN_PATH,
    beginByte: encodeWireU64(BigInt(begin)),
    endByte: encodeWireU64(BigInt(begin + TOKEN.length)),
    tokenSha256: await sha256Hex(bytes.subarray(begin, begin + TOKEN.length)),
  };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor,
  };
  const entry: CppCuteFrontendEntryRequestV1 = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: { major: 1, minor: 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: MAIN_PATH,
    files: [file],
    entryRequests: [entry],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR },
    },
    limits: semanticLimits(profile),
  };
  const input: CppCuteFrontendRequestV1 = {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
  return prepareCppCuteFrontendRequest(profile, input, [{ virtualPath: MAIN_PATH, bytes }], {
    detached: {
      declaredSourceReference: statementSha256 === null ? null : { statementSha256 },
      conformance: null,
    },
  });
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

function expectMetadataError(
  promise: Promise<unknown>,
  code: CppCuteAotRunMetadataError["code"],
  path: unknown,
): Promise<void> {
  return expect(promise).rejects.toMatchObject({ code, path });
}

describe("AOT run metadata around producer-neutral request", () => {
  it("binds exact AOT profile, request, strict Git statement, and copied request source", async () => {
    const fixture = await createMetadataFixture();
    const prepared = await prepareCppCuteAotRunMetadata(fixture.request, fixture.input);
    const record = unwrapPreparedCppCuteAotRunMetadata(prepared);

    expect(prepared.runMetadataId).toBe(fixture.input.runMetadataId);
    expect(prepared.requestId).toBe(fixture.request.requestId);
    expect(record.profile).toBe(fixture.profile);
    expect(record.request).toBe(fixture.request);
    expect(record.metadata.declaredSourceReference.statement).toEqual(fixture.statement);
    expect(Object.isFrozen(prepared)).toBe(true);

    const first = copyPreparedCppCuteAotRunSourceSnapshots(prepared);
    first[0]!.bytes.fill(0);
    expect(new TextDecoder().decode(copyPreparedCppCuteAotRunSourceSnapshots(prepared)[0]!.bytes)).toBe(SOURCE_TEXT);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first[0])).toBe(true);

    const callerOwned = structuredClone(fixture.input);
    const pending = prepareCppCuteAotRunMetadata(fixture.request, callerOwned);
    (callerOwned.declaredSourceReference.statement as unknown as { repository: string }).repository =
      "https://example.com/mutated";
    await expect(pending).resolves.toMatchObject({ runMetadataId: fixture.input.runMetadataId });
  });

  it("derives exact profile from request and requires detached declared-source hash", async () => {
    const fixture = await createMetadataFixture();
    const noProvenanceRequest = await createRequest(fixture.profile, null);
    const noProvenanceInput = structuredClone(fixture.input);
    (noProvenanceInput as { requestId: string }).requestId = noProvenanceRequest.requestId;
    const body = { ...noProvenanceInput } as CppCuteAotRunMetadataV1;
    (noProvenanceInput as { runMetadataId: string }).runMetadataId =
      `bg.cpp.aot-run-metadata.sha256.${await deriveCppCuteAotRunMetadataHash(body)}`;
    await expectMetadataError(
      prepareCppCuteAotRunMetadata(noProvenanceRequest, noProvenanceInput),
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-SOURCE-REFERENCE-MISMATCH",
      "$.declaredSourceReference.statementSha256",
    );

    const hash = structuredClone(fixture.input);
    (hash.declaredSourceReference as unknown as { statementSha256: string }).statementSha256 = "f".repeat(64);
    await expectMetadataError(
      prepareCppCuteAotRunMetadata(fixture.request, hash),
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-HASH-MISMATCH",
      "$.declaredSourceReference.statementSha256",
    );

    const repository = structuredClone(fixture.input);
    (repository.declaredSourceReference.statement as unknown as { repository: string }).repository =
      "https://user@example.com/repo";
    await expectMetadataError(
      prepareCppCuteAotRunMetadata(fixture.request, repository),
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-INVALID",
      "$.declaredSourceReference.statement.repository",
    );
  });

  it("rejects forged authorities, accessors, cancellation, and decode-budget exhaustion", async () => {
    const fixture = await createMetadataFixture();
    const prepared = await prepareCppCuteAotRunMetadata(fixture.request, fixture.input);
    expect(() => unwrapPreparedCppCuteAotRunMetadata({ ...prepared } as PreparedCppCuteAotRunMetadata)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-UNVERIFIED", path: "$" }),
    );
    await expect(prepareCppCuteAotRunMetadata(
      { ...fixture.request } as PreparedCppCuteFrontendRequest,
      fixture.input,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-REQUEST-UNVERIFIED", path: "$" });

    let getterCalls = 0;
    const accessor = structuredClone(fixture.input) as unknown as Record<string, unknown>;
    Object.defineProperty(accessor, "requestId", {
      enumerable: true,
      get() { getterCalls += 1; return fixture.request.requestId; },
    });
    await expectMetadataError(
      prepareCppCuteAotRunMetadata(fixture.request, accessor),
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-INVALID",
      "$",
    );
    expect(getterCalls).toBe(0);

    const controller = new AbortController();
    controller.abort();
    await expectMetadataError(
      prepareCppCuteAotRunMetadata(fixture.request, fixture.input, { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-CANCELLED",
      "$.signal",
    );
    const midflightController = new AbortController();
    const pending = prepareCppCuteAotRunMetadata(
      fixture.request,
      fixture.input,
      { signal: midflightController.signal },
    );
    midflightController.abort();
    await expectMetadataError(
      pending,
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-CANCELLED",
      "$.signal",
    );
    await expectMetadataError(
      prepareCppCuteAotRunMetadata(fixture.request, fixture.input, { limits: { maxNodes: 1 } }),
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-RESOURCE-LIMIT",
      expect.any(String),
    );
  });
});
