import {
  encodeWireU64,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  canonicalCppCuteFrontendArtifactBytes,
  decodeCppCuteFrontendArtifact,
  deriveCppCuteFrontendArtifactId,
  unwrapVerifiedCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "../../src/cpp_cute_frontend_artifact.js";
import {
  CppCuteFrontendRequestBindingError,
  deriveCppCuteFrontendRequestBindingHash,
  prepareCppCuteFrontendRequestBinding,
  unwrapPreparedCppCuteFrontendRequestBinding,
  type PreparedCppCuteFrontendRequestBinding,
} from "../../src/cpp_cute_frontend_request_binding.js";
import {
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
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
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
  type CppCuteFrontendPayloadV3,
} from "../../src/cpp_cute_frontend_types.js";
import {
  computeCppCuteInputHashes,
  computeCppCuteSemanticPassInputClosureHash,
  computeCppCuteSharedSurfaceHash,
} from "../../src/cpp_cute_frontend_verify.js";
import {
  createCppCuteArtifactInput,
  createCppCuteBrowserProfileInput,
  createCppCuteProfileInput,
  rebindCppCuteFixtureSourceEntityIds,
} from "./support/cpp_cute_frontend_fixtures.js";

const ENCODER = new TextEncoder();
const MAIN_PATH = "/src/layout.cu";
const HEADER_PATH = "/src/project.hpp";
const TOKEN = "layout";

interface BindingFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly requestInput: CppCuteFrontendRequestV1;
  readonly snapshots: readonly CppCuteFrontendSourceSnapshotInput[];
  readonly request: PreparedCppCuteFrontendRequest;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
}

async function createBindingFixture(): Promise<BindingFixture> {
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput({ sourceRoots: ["/src"] }));
  const mainBytes = paddedBytes("auto layout = make_layout(Int<2>{});\n", 100);
  const headerBytes = paddedBytes("constexpr int project_value = 2;\n", 200);
  const files = await Promise.all([
    requestFile("main-source", MAIN_PATH, null, mainBytes),
    requestFile("project-header", HEADER_PATH, "workspace-source", headerBytes),
  ]);
  files.sort((left, right) => left.virtualPath.localeCompare(right.virtualPath));
  const snapshots = files.map((file) => ({
    virtualPath: file.virtualPath,
    bytes: file.virtualPath === MAIN_PATH ? mainBytes : headerBytes,
  }));
  const requestInput = await requestInputFor(profile, files, mainBytes);
  const requestEntry = requestInput.entryRequests[0];
  if (requestEntry === undefined) throw new Error("fixture request entry missing");
  const artifactResource = await artifactResourceFor(profile, files, requestEntry);
  const request = await preparedRequest(profile, requestInput, snapshots, artifactResource);
  return { profile, requestInput, snapshots, request, artifactResource };
}

async function requestFile(
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

async function requestInputFor(
  profile: PreparedCppCuteFrontendProfile,
  files: readonly CppCuteFrontendRequestSourceFileV1[],
  mainBytes: Uint8Array,
  token = TOKEN,
  versions: {
    readonly requestMinor?: 0 | typeof CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR;
    readonly artifactMinor?: typeof CPP_CUTE_FRONTEND_ARTIFACT_MINOR |
      typeof CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR;
  } = {},
): Promise<CppCuteFrontendRequestV1> {
  const tokenBytes = ENCODER.encode(token);
  const begin = bytesIndexOf(mainBytes, tokenBytes);
  const anchor = {
    virtualPath: MAIN_PATH,
    beginByte: encodeWireU64(BigInt(begin)),
    endByte: encodeWireU64(BigInt(begin + tokenBytes.byteLength)),
    tokenSha256: await sha256Hex(mainBytes.subarray(begin, begin + tokenBytes.byteLength)),
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
    version: { major: 1, minor: versions.requestMinor ?? 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: MAIN_PATH,
    files,
    entryRequests: [entry],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: versions.artifactMinor ?? CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: semanticLimits(profile),
  };
  return {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
}

async function artifactResourceFor(
  profile: PreparedCppCuteFrontendProfile,
  requestFiles: readonly CppCuteFrontendRequestSourceFileV1[],
  requestEntry: CppCuteFrontendEntryRequestV1,
  options: {
    readonly injectUnrequestedSource?: boolean;
    readonly omitProjectHeader?: boolean;
    readonly omitRequestedMain?: boolean;
    readonly artifactMinor?: typeof CPP_CUTE_FRONTEND_ARTIFACT_MINOR |
      typeof CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR;
  } = {},
): Promise<VerifiedCppCuteFrontendArtifactResource> {
  const artifact = await createCppCuteArtifactInput(profile.compilationContractHash);
  const payload = artifact.payload as CppCuteFrontendPayloadV3;
  const oldMain = payload.inputs.files.find((file) => file.role === "main-source");
  const oldHeader = payload.inputs.files.find((file) => file.role === "dependency-header");
  const main = requestFiles.find((file) => file.role === "main-source");
  const header = requestFiles.find((file) => file.role === "project-header");
  if (oldMain === undefined || oldHeader === undefined || main === undefined || header === undefined) {
    throw new Error("fixture lost source files");
  }
  const oldMainFileId = oldMain.fileId;
  const oldHeaderFileId = oldHeader.fileId;
  const replacements = new Map<string, string>();
  if (options.omitRequestedMain !== true) replacements.set(oldMainFileId, main.fileId);
  if (options.omitProjectHeader !== true) replacements.set(oldHeaderFileId, header.fileId);
  replaceStrings(artifact as unknown as JsonValue, replacements);
  const mutablePayload = artifact.payload as unknown as Record<string, unknown>;
  const inputs = mutablePayload["inputs"] as Record<string, unknown>;
  const artifactFiles = inputs["files"] as Record<string, unknown>[];
  const artifactMainFileId = options.omitRequestedMain === true ? oldMainFileId : main.fileId;
  const artifactHeaderFileId = options.omitProjectHeader === true ? oldHeaderFileId : header.fileId;
  const mutableMain = artifactFiles.find((file) => file["fileId"] === artifactMainFileId);
  const mutableHeader = artifactFiles.find((file) => file["fileId"] === artifactHeaderFileId);
  if (mutableMain === undefined || mutableHeader === undefined) throw new Error("fixture replacement failed");
  if (options.omitRequestedMain !== true) {
    Object.assign(mutableMain, {
      role: main.role,
      virtualPath: main.virtualPath,
      contentSha256: main.contentSha256,
      byteLength: main.byteLength,
      owner: { kind: "source" },
      includeRootId: main.includeRootId,
    });
  }
  if (options.omitProjectHeader !== true) {
    Object.assign(mutableHeader, {
      role: header.role,
      virtualPath: header.virtualPath,
      contentSha256: header.contentSha256,
      byteLength: header.byteLength,
      owner: { kind: "source" },
      includeRootId: header.includeRootId,
    });
  }
  const spans = mutablePayload["spans"] as Record<string, unknown>[];
  const identitySpanId = `bg.cpp.span.sha256.${"f".repeat(64)}`;
  spans.push({
    spanId: identitySpanId,
    spelling: {
      fileId: artifactMainFileId,
      startByte: requestEntry.anchor.beginByte,
      endByte: requestEntry.anchor.endByte,
    },
    expansion: {
      fileId: artifactMainFileId,
      startByte: requestEntry.anchor.beginByte,
      endByte: requestEntry.anchor.endByte,
    },
    macroExpansionId: null,
  });
  spans.sort((left, right) => String(left["spanId"]).localeCompare(String(right["spanId"])));
  const entries = mutablePayload["entries"] as Record<string, unknown>[];
  const outcome = mutablePayload["outcome"] as Record<string, unknown>;
  const selectedEntryId = (outcome["selectedEntryIds"] as string[])[0];
  const selectedEntry = entries.find((entry) => entry["entryId"] === selectedEntryId);
  const selectedRootDeclarationId = (selectedEntry?.["selectedRootDeclarationIds"] as string[] | undefined)?.[0];
  const declarations = mutablePayload["declarations"] as Record<string, unknown>[];
  const selectedRoot = declarations.find((declaration) =>
    declaration["declarationId"] === selectedRootDeclarationId && declaration["kind"] === requestEntry.declarationKind);
  if (selectedRoot === undefined) throw new Error("fixture lost selected root declaration");
  selectedRoot["identitySpanId"] = identitySpanId;
  const includeEdges = inputs["includeEdges"] as Record<string, unknown>[];
  const sourceEdge = includeEdges.find((edge) => edge["kind"] === "source-directive");
  if (sourceEdge === undefined) throw new Error("fixture lost source include edge");
  const resolution = sourceEdge["resolution"] as Record<string, unknown>;
  if (options.omitProjectHeader !== true) resolution["includeRootId"] = "workspace-source";
  if (options.injectUnrequestedSource === true) {
    const injectedFileId = `bg.cpp.file.sha256.${"9".repeat(64)}`;
    const injectedEdgeId = `bg.cpp.include-edge.sha256.${"9".repeat(64)}`;
    artifactFiles.push({
      fileId: injectedFileId,
      role: "generated-header",
      virtualPath: "/src/generated.hpp",
      contentSha256: "9".repeat(64),
      byteLength: "20",
      owner: { kind: "source" },
      includeRootId: "workspace-source",
    });
    includeEdges.push({
      kind: "source-directive",
      includeEdgeId: injectedEdgeId,
      includingFileId: main.fileId,
      directiveSpanId: spans[0]?.["spanId"],
      spelling: "generated.hpp",
      mode: "quote",
      resolution: {
        kind: "resolved",
        fileId: injectedFileId,
        includeRootId: "workspace-source",
      },
    });
    for (const pass of mutablePayload["semanticPasses"] as Record<string, unknown>[]) {
      (pass["openedFileIds"] as string[]).push(injectedFileId);
      (pass["includeEdgeIds"] as string[]).push(injectedEdgeId);
    }
  }
  artifactFiles.sort((left, right) => String(left["fileId"]).localeCompare(String(right["fileId"])));
  includeEdges.sort((left, right) =>
    String(left["includeEdgeId"]).localeCompare(String(right["includeEdgeId"])));

  await rebindCppCuteFixtureSourceEntityIds(artifact.payload);
  const hashes = await computeCppCuteInputHashes(artifact.payload);
  inputs["sourceSetSha256"] = hashes.sourceSetSha256;
  inputs["headerSetSha256"] = hashes.headerSetSha256;
  inputs["closureSha256"] = hashes.closureSha256;
  for (const [index, pass] of (mutablePayload["semanticPasses"] as Record<string, unknown>[]).entries()) {
    (pass["openedFileIds"] as string[]).sort();
    (pass["includeEdgeIds"] as string[]).sort();
    pass["observedInputClosureSha256"] = await computeCppCuteSemanticPassInputClosureHash(artifact.payload, index);
    pass["sharedSurfaceSha256"] = await computeCppCuteSharedSurfaceHash(
      artifact.payload,
      pass["domain"] as "host" | "device",
    );
  }
  (mutablePayload["extraction"] as Record<string, unknown>)["inputClosureSha256"] = hashes.closureSha256;
  const artifactMinor = options.artifactMinor ?? CPP_CUTE_FRONTEND_ARTIFACT_MINOR;
  (artifact.version as { minor: number }).minor = artifactMinor;
  (artifact as { artifactId: string }).artifactId = await deriveCppCuteFrontendArtifactId(
    artifact.payload,
    { minor: artifactMinor },
  );
  const verified = await verifyCppCuteFrontendArtifact(artifact);
  return decodeCppCuteFrontendArtifact(canonicalCppCuteFrontendArtifactBytes(verified));
}

async function preparedRequest(
  profile: PreparedCppCuteFrontendProfile,
  input: CppCuteFrontendRequestV1,
  snapshots: readonly CppCuteFrontendSourceSnapshotInput[],
  artifactResource: VerifiedCppCuteFrontendArtifactResource,
  expectedArtifactSha256?: string,
): Promise<PreparedCppCuteFrontendRequest> {
  const artifact = unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource);
  return prepareCppCuteFrontendRequest(profile, input, snapshots, {
    detached: {
      declaredSourceReference: null,
      conformance: {
        expectedArtifactSha256: expectedArtifactSha256 ?? artifact.artifactBytesSha256,
        expectedOpenedHeaderSetSha256: artifact.headerSetSha256,
        expectedInputClosureSha256: artifact.inputClosureSha256,
      },
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

function paddedBytes(text: string, byteLength: number): Uint8Array {
  const prefix = ENCODER.encode(text);
  if (prefix.byteLength > byteLength) throw new Error("fixture text exceeds requested byte length");
  const bytes = new Uint8Array(byteLength);
  bytes.fill(32);
  bytes.set(prefix);
  return bytes;
}

function bytesIndexOf(haystack: Uint8Array, needle: Uint8Array): number {
  outer: for (let start = 0; start <= haystack.byteLength - needle.byteLength; start += 1) {
    for (let offset = 0; offset < needle.byteLength; offset += 1) {
      if (haystack[start + offset] !== needle[offset]) continue outer;
    }
    return start;
  }
  throw new Error("fixture token missing");
}

function replaceStrings(value: JsonValue, replacements: ReadonlyMap<string, string>): void {
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      const entry = value[index];
      if (typeof entry === "string" && replacements.has(entry)) {
        (value as JsonValue[])[index] = replacements.get(entry)!;
      } else if (entry !== undefined) {
        replaceStrings(entry, replacements);
      }
    }
    return;
  }
  if (typeof value !== "object" || value === null) return;
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === "string" && replacements.has(entry)) {
      (value as Record<string, JsonValue>)[key] = replacements.get(entry)!;
    } else {
      replaceStrings(entry, replacements);
    }
  }
}

function expectBindingError(
  promise: Promise<unknown>,
  code: CppCuteFrontendRequestBindingError["code"],
  path: unknown,
): Promise<void> {
  return expect(promise).rejects.toMatchObject({ code, path });
}

describe("producer-neutral frontend request binding", () => {
  it("derives exact request, artifact, selection, and conformance binding host-side", async () => {
    const fixture = await createBindingFixture();
    const prepared = await prepareCppCuteFrontendRequestBinding(fixture.request, fixture.artifactResource);
    const record = unwrapPreparedCppCuteFrontendRequestBinding(prepared);

    expect(prepared.bindingId).toBe(
      `bg.cpp.frontend-request-binding.sha256.${await deriveCppCuteFrontendRequestBindingHash(record.binding)}`,
    );
    expect(prepared.outcome).toBe("accepted");
    expect(record.request).toBe(fixture.request);
    expect(record.artifactResource).toBe(fixture.artifactResource);
    expect(record.binding.selection).toMatchObject({
      kind: "resolved",
      requestId: fixture.requestInput.entryRequests[0]?.requestId,
      anchorTokenSha256: fixture.requestInput.entryRequests[0]?.anchor.tokenSha256,
      anchorMatch: "spelling",
    });
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(Object.isFrozen(record.binding)).toBe(true);
    expect(Object.isFrozen(record.binding.selection)).toBe(true);
  });

  it("requires the verified artifact to match expectedArtifact major and minor exactly", async () => {
    const fixture = await createBindingFixture();
    const requestEntry = fixture.requestInput.entryRequests[0];
    const mainSnapshot = fixture.snapshots.find((snapshot) => snapshot.virtualPath === MAIN_PATH);
    if (requestEntry === undefined || mainSnapshot === undefined) throw new Error("fixture input missing");

    const artifact31 = await artifactResourceFor(
      fixture.profile,
      fixture.requestInput.files,
      requestEntry,
      { artifactMinor: CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR },
    );
    const request10Expecting30 = await preparedRequest(
      fixture.profile,
      fixture.requestInput,
      fixture.snapshots,
      artifact31,
    );
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(request10Expecting30, artifact31),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-ARTIFACT-MISMATCH",
      "$.artifact.version",
    );

    const request11Input = await requestInputFor(
      fixture.profile,
      fixture.requestInput.files,
      mainSnapshot.bytes,
      TOKEN,
      {
        requestMinor: CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR,
        artifactMinor: CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR,
      },
    );
    const request11Expecting31 = await preparedRequest(
      fixture.profile,
      request11Input,
      fixture.snapshots,
      fixture.artifactResource,
    );
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(request11Expecting31, fixture.artifactResource),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-ARTIFACT-MISMATCH",
      "$.artifact.version",
    );
  });

  it("accepts an unopened request project header omitted from artifact source inputs", async () => {
    const fixture = await createBindingFixture();
    const requestEntry = fixture.requestInput.entryRequests[0];
    if (requestEntry === undefined) throw new Error("fixture request entry missing");
    const artifactResource = await artifactResourceFor(
      fixture.profile,
      fixture.requestInput.files,
      requestEntry,
      { omitProjectHeader: true },
    );
    const request = await preparedRequest(
      fixture.profile,
      fixture.requestInput,
      fixture.snapshots,
      artifactResource,
    );

    await expect(prepareCppCuteFrontendRequestBinding(request, artifactResource))
      .resolves.toMatchObject({ requestId: request.requestId });
    const artifact = unwrapVerifiedCppCuteFrontendArtifact(
      unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource),
    );
    expect(artifact.envelope.payload.inputs.files
      .filter((file) => file.owner.kind === "source")
      .map((file) => file.virtualPath)).toEqual([MAIN_PATH]);
    expect(fixture.requestInput.files.some((file) => file.virtualPath === HEADER_PATH)).toBe(true);
  });

  it("keeps common binding valid for equivalent AOT and browser compilation contracts", async () => {
    const fixture = await createBindingFixture();
    await expect(prepareCppCuteFrontendRequestBinding(
      fixture.request,
      fixture.artifactResource,
    )).resolves.toMatchObject({ profileHash: fixture.profile.profileHash });

    const browserProfile = await prepareCppCuteFrontendProfile(
      createCppCuteBrowserProfileInput({ sourceRoots: ["/src"] }),
    );
    const browserRequest = await preparedRequest(
      browserProfile,
      fixture.requestInput,
      fixture.snapshots,
      fixture.artifactResource,
    );
    await expect(prepareCppCuteFrontendRequestBinding(
      browserRequest,
      fixture.artifactResource,
    )).resolves.toMatchObject({ profileHash: browserProfile.profileHash });
  });

  it("rejects detached-conformance mismatch", async () => {
    const fixture = await createBindingFixture();
    const wrongConformance = await preparedRequest(
      fixture.profile,
      fixture.requestInput,
      fixture.snapshots,
      fixture.artifactResource,
      "f".repeat(64),
    );
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(wrongConformance, fixture.artifactResource),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CONFORMANCE-MISMATCH",
      "$.detached.conformance.expectedArtifactSha256",
    );
  });

  it("rejects any source-owned artifact input absent from request authority", async () => {
    const fixture = await createBindingFixture();
    const requestEntry = fixture.requestInput.entryRequests[0];
    if (requestEntry === undefined) throw new Error("fixture request entry missing");
    const injected = await artifactResourceFor(
      fixture.profile,
      fixture.requestInput.files,
      requestEntry,
      { injectUnrequestedSource: true },
    );
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(fixture.request, injected),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
      expect.stringMatching(/^\$\.artifact\.inputs\.files\[\d+\]$/u),
    );
  });

  it("rejects an artifact source descriptor and bytes mutated from request authority", async () => {
    const fixture = await createBindingFixture();
    const requestEntry = fixture.requestInput.entryRequests[0];
    if (requestEntry === undefined) throw new Error("fixture request entry missing");
    const mutatedHeader = await requestFile(
      "project-header",
      HEADER_PATH,
      "workspace-source",
      paddedBytes("constexpr int project_value = 3;\n", 200),
    );
    const mutatedFiles = fixture.requestInput.files.map((file) =>
      file.virtualPath === HEADER_PATH ? mutatedHeader : file);
    const artifactResource = await artifactResourceFor(
      fixture.profile,
      mutatedFiles,
      requestEntry,
    );

    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(fixture.request, artifactResource),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
      expect.stringMatching(/^\$\.artifact\.inputs\.files\[\d+\]$/u),
    );
  });

  it("rejects an artifact that omits the exact requested main descriptor", async () => {
    const fixture = await createBindingFixture();
    const requestEntry = fixture.requestInput.entryRequests[0];
    if (requestEntry === undefined) throw new Error("fixture request entry missing");
    const artifactResource = await artifactResourceFor(
      fixture.profile,
      fixture.requestInput.files,
      requestEntry,
      { omitRequestedMain: true },
    );

    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(fixture.request, artifactResource),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
      "$.artifact.inputs.mainFileId",
    );
  });

  it("rejects a same-family artifact whose selected root does not contain the request anchor", async () => {
    const fixture = await createBindingFixture();
    const mainSnapshot = fixture.snapshots.find((snapshot) => snapshot.virtualPath === MAIN_PATH);
    if (mainSnapshot === undefined) throw new Error("fixture lost main source snapshot");
    const wrongInput = await requestInputFor(
      fixture.profile,
      fixture.requestInput.files,
      mainSnapshot.bytes,
      "make_layout",
    );
    const wrongRequest = await preparedRequest(
      fixture.profile,
      wrongInput,
      fixture.snapshots,
      fixture.artifactResource,
    );
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(wrongRequest, fixture.artifactResource),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-SELECTION-MISMATCH",
      "$.request.entryRequests[0].anchor",
    );
  });

  it("rejects forged authorities, accessors, cancellation, and decode-budget exhaustion", async () => {
    const fixture = await createBindingFixture();
    const prepared = await prepareCppCuteFrontendRequestBinding(fixture.request, fixture.artifactResource);
    expect(() => unwrapPreparedCppCuteFrontendRequestBinding(
      { ...prepared } as PreparedCppCuteFrontendRequestBinding,
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-UNVERIFIED",
      path: "$",
    }));
    await expect(prepareCppCuteFrontendRequestBinding(
      { ...fixture.request } as PreparedCppCuteFrontendRequest,
      fixture.artifactResource,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-REQUEST-UNVERIFIED", path: "$" });

    await expect(prepareCppCuteFrontendRequestBinding(
      fixture.request,
      { ...fixture.artifactResource } as VerifiedCppCuteFrontendArtifactResource,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED", path: "$" });

    let getterCalls = 0;
    const accessor = {} as Record<string, unknown>;
    Object.defineProperty(accessor, "signal", {
      enumerable: true,
      get() { getterCalls += 1; return undefined; },
    });
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(fixture.request, fixture.artifactResource, accessor),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INVALID",
      "$.options.signal",
    );
    expect(getterCalls).toBe(0);

    const controller = new AbortController();
    controller.abort();
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(
        fixture.request,
        fixture.artifactResource,
        { signal: controller.signal },
      ),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CANCELLED",
      "$.signal",
    );
    const midflightController = new AbortController();
    const pending = prepareCppCuteFrontendRequestBinding(
      fixture.request,
      fixture.artifactResource,
      { signal: midflightController.signal },
    );
    midflightController.abort();
    await expectBindingError(
      pending,
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CANCELLED",
      "$.signal",
    );
    await expectBindingError(
      prepareCppCuteFrontendRequestBinding(
        fixture.request,
        fixture.artifactResource,
        { limits: { maxNodes: 1 } },
      ),
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-RESOURCE-LIMIT",
      expect.any(String),
    );
  });
});
