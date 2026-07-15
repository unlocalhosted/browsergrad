import { describe, expect, it } from "vitest";
import {
  decodeCppCuteFrontendArtifact,
  deriveCppCuteFrontendArtifactId,
  deriveCppCuteStableId,
  unwrapVerifiedCppCuteFrontendArtifact,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import { computeCppCuteInputHashes } from "../../src/cpp_cute_frontend_verify.js";
import type {
  CppCuteFrontendArtifactV1,
  CppCuteFrontendPayloadV1,
} from "../../src/cpp_cute_frontend_types.js";

const PROFILE_HASH = "a".repeat(64);
const MAIN_FILE_ID = stableId("file", "1");
const HEADER_FILE_ID = stableId("file", "e");
const INCLUDE_ROOT_ID = stableId("include-root", "f");
const INCLUDE_EDGE_ID = stableId("include-edge", "0");
const SPAN_ID = stableId("span", "2");
const INT_TYPE_ID = stableId("type", "3");
const LAYOUT_TYPE_ID = stableId("type", "4");
const TEMPLATE_DECLARATION_ID = stableId("declaration", "5");
const RECORD_DECLARATION_ID = stableId("declaration", "6");
const VARIABLE_DECLARATION_ID = stableId("declaration", "7");
const INSTANTIATION_ID = stableId("template-instantiation", "8");
const LAYOUT_FACT_ID = stableId("fact", "9");
const INTRINSIC_FACT_ID = stableId("fact", "a");
const ENTRY_ID = stableId("entry", "b");
const DIAGNOSTIC_ID = stableId("diagnostic", "c");
const ZERO_HASH = "0".repeat(64);

function stableId(kind: string, digit: string): string {
  return `bg.cpp.${kind}.sha256.${digit.repeat(64)}`;
}

function sourceOrigin(): { readonly kind: "source"; readonly spanId: string } {
  return { kind: "source", spanId: SPAN_ID };
}

function qualifiers(): { readonly const: boolean; readonly volatile: boolean; readonly restrict: boolean } {
  return { const: false, volatile: false, restrict: false };
}

async function payloadFixture(): Promise<CppCuteFrontendPayloadV1> {
  const payload: CppCuteFrontendPayloadV1 = {
    profileHash: PROFILE_HASH,
    inputs: {
      mainFileId: MAIN_FILE_ID,
      includeRoots: [
        {
          includeRootId: INCLUDE_ROOT_ID,
          ordinal: 0,
          mode: "system",
          virtualPath: "/toolchain/cutlass/include",
          manifestSha256: "f".repeat(64),
        },
      ],
      files: [
        {
          fileId: MAIN_FILE_ID,
          role: "main-source",
          virtualPath: "/src/layout.cu",
          contentSha256: "d".repeat(64),
          byteLength: "100" as never,
          profileDependency: "none",
        },
        {
          fileId: HEADER_FILE_ID,
          role: "cute-header",
          virtualPath: "/toolchain/cutlass/include/cute/layout.hpp",
          contentSha256: "e".repeat(64),
          byteLength: "200" as never,
          profileDependency: "cute",
        },
      ],
      includeEdges: [
        {
          includeEdgeId: INCLUDE_EDGE_ID,
          includingFileId: MAIN_FILE_ID,
          directiveSpanId: SPAN_ID,
          spelling: "cute/layout.hpp",
          mode: "angle",
          resolution: {
            kind: "resolved",
            fileId: HEADER_FILE_ID,
            includeRootId: INCLUDE_ROOT_ID,
          },
        },
      ],
      sourceSetSha256: ZERO_HASH,
      headerSetSha256: ZERO_HASH,
      closureSha256: ZERO_HASH,
    },
    spans: [
      {
        spanId: SPAN_ID,
        spelling: { fileId: MAIN_FILE_ID, startByte: "0" as never, endByte: "100" as never },
        expansion: { fileId: MAIN_FILE_ID, startByte: "0" as never, endByte: "100" as never },
        macroExpansionId: null,
      },
    ],
    macroExpansions: [],
    types: [
      {
        typeId: INT_TYPE_ID,
        kind: "builtin",
        canonicalName: "int",
        qualifiers: qualifiers(),
        origin: sourceOrigin(),
        builtin: "int",
      },
      {
        typeId: LAYOUT_TYPE_ID,
        kind: "template-specialization",
        canonicalName: "cute::Layout<cute::Shape<cute::Int<3>, cute::Int<2>>, cute::Stride<cute::Int<1>, cute::Int<3>>>",
        qualifiers: qualifiers(),
        origin: sourceOrigin(),
        templateDeclarationId: TEMPLATE_DECLARATION_ID,
        arguments: [],
      },
    ],
    constants: [],
    declarations: [
      {
        declarationId: TEMPLATE_DECLARATION_ID,
        kind: "template",
        canonicalUsr: "c:@N@cute@ST>1#T@Layout",
        canonicalName: "cute::Layout",
        lexicalParentId: null,
        semanticParentId: null,
        typeId: INT_TYPE_ID,
        targetTypeId: null,
        origin: sourceOrigin(),
        definitionKind: "external",
        linkage: "external",
        storageDuration: "none",
        memorySpace: "host",
        mangledName: null,
        cudaAttributes: { host: true, device: true, global: false, forceInline: true },
      },
      {
        declarationId: RECORD_DECLARATION_ID,
        kind: "record",
        canonicalUsr: "c:@N@cute@S@Layout>#I#I",
        canonicalName: "cute::Layout<cute::Shape<cute::Int<3>, cute::Int<2>>, cute::Stride<cute::Int<1>, cute::Int<3>>>",
        lexicalParentId: null,
        semanticParentId: null,
        typeId: LAYOUT_TYPE_ID,
        targetTypeId: null,
        origin: sourceOrigin(),
        definitionKind: "external",
        linkage: "external",
        storageDuration: "none",
        memorySpace: "host",
        mangledName: null,
        cudaAttributes: { host: true, device: true, global: false, forceInline: false },
      },
      {
        declarationId: VARIABLE_DECLARATION_ID,
        kind: "variable",
        canonicalUsr: "c:@layout",
        canonicalName: "layout",
        lexicalParentId: null,
        semanticParentId: null,
        typeId: LAYOUT_TYPE_ID,
        targetTypeId: null,
        origin: sourceOrigin(),
        definitionKind: "definition",
        linkage: "internal",
        storageDuration: "static",
        memorySpace: "host",
        mangledName: "_ZL6layout",
        cudaAttributes: { host: true, device: false, global: false, forceInline: false },
      },
    ],
    templateInstantiations: [
      {
        instantiationId: INSTANTIATION_ID,
        templateDeclarationId: TEMPLATE_DECLARATION_ID,
        specializationDeclarationId: RECORD_DECLARATION_ID,
        arguments: [],
        pointOfInstantiationSpanId: SPAN_ID,
        parentInstantiationId: null,
        depth: 0,
      },
    ],
    overloadResolutions: [],
    sourceAbi: { types: [], functions: [] },
    functionBodies: [],
    facts: [
      {
        factId: LAYOUT_FACT_ID,
        kind: "affine-layout",
        origin: sourceOrigin(),
        resultDeclarationId: VARIABLE_DECLARATION_ID,
        shape: {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: { kind: "integer", value: "3" as never } },
            { kind: "scalar", value: { kind: "integer", value: "2" as never } },
          ],
        },
        stride: {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: { kind: "integer", value: "1" as never } },
            { kind: "scalar", value: { kind: "integer", value: "3" as never } },
          ],
        },
        rank: 2,
        leafRank: 2,
        size: { kind: "integer", value: "6" as never },
        cosize: { kind: "integer", value: "6" as never },
      },
      {
        factId: INTRINSIC_FACT_ID,
        kind: "target-intrinsic",
        origin: sourceOrigin(),
        familyId: "nvidia:wgmma@1",
        operation: { kind: "capability", capabilityId: "nvidia:wgmma@1" },
        operandExpressionIds: [],
        resultTypeId: null,
        effects: { readsMemory: false, writesMemory: false, synchronizes: true, convergent: true },
        availability: { kind: "recognized-unsupported", diagnosticId: DIAGNOSTIC_ID },
      },
    ],
    entries: [
      {
        entryId: ENTRY_ID,
        kind: "layout",
        layoutFactId: LAYOUT_FACT_ID,
        selectedRootDeclarationIds: [VARIABLE_DECLARATION_ID],
      },
    ],
    diagnostics: [
      {
        diagnosticId: DIAGNOSTIC_ID,
        phase: "artifact-extraction",
        severity: "warning",
        code: "browsergrad.cpp-cute:recognized-unsupported-intrinsic",
        renderedMessage: "WGMMA preserved as a typed unsupported target capability.",
        primarySpanId: SPAN_ID,
        subject: { kind: "fact", factId: INTRINSIC_FACT_ID },
        parentDiagnosticId: null,
        related: [],
      },
    ],
    outcome: { kind: "accepted", selectedEntryIds: [ENTRY_ID] },
    extraction: {
      profileHash: PROFILE_HASH,
      inputClosureSha256: ZERO_HASH,
      appliedTransforms: [],
    },
  };
  const hashes = await computeCppCuteInputHashes(payload);
  return {
    ...payload,
    inputs: {
      ...payload.inputs,
      sourceSetSha256: hashes.sourceSetSha256,
      headerSetSha256: hashes.headerSetSha256,
      closureSha256: hashes.closureSha256,
    },
    extraction: {
      ...payload.extraction,
      inputClosureSha256: hashes.closureSha256,
    },
  };
}

async function artifactFixture(): Promise<CppCuteFrontendArtifactV1> {
  const payload = await payloadFixture();
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
    version: { major: 1, minor: 0 },
    producer: { id: "browsergrad-tools/cpp-cute-frontend", version: "0.1.0" },
    artifactId: await deriveCppCuteFrontendArtifactId(payload),
    payload,
    requiredExtensions: [],
  };
}

async function cloneArtifact(): Promise<Record<string, unknown>> {
  return structuredClone(await artifactFixture()) as unknown as Record<string, unknown>;
}

describe("C++/CuTe frontend artifact", () => {
  it("verifies one closed layout artifact with typed unsupported target fact", async () => {
    const verified = await verifyCppCuteFrontendArtifact(await artifactFixture());
    const record = unwrapVerifiedCppCuteFrontendArtifact(verified);

    expect(verified.artifactId).toBe(`bg.artifact.cpp-cute-frontend.sha256.${verified.artifactHash}`);
    expect({
      artifactHash: verified.artifactHash,
      sourceSetSha256: verified.sourceSetSha256,
      headerSetSha256: verified.headerSetSha256,
      inputClosureSha256: verified.inputClosureSha256,
    }).toEqual({
      artifactHash: "a89655452625dadf8c489e3640dfa98a21e78af742cf593fa974e387c8d3ce65",
      sourceSetSha256: "9a122d8462fc232451f6e758bcda17f48dcf2614d67aa641e951a5886fac6975",
      headerSetSha256: "a6b1ad5036810001364f1c3062577e7c0d3089bb6c84297dfb6ba59a0b02437c",
      inputClosureSha256: "36b44d4e6ef78c9696f9864407dcebbec7ff988394653a0cae1600a2662ab847",
    });
    expect(verified.transportHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(verified.profileHash).toBe(PROFILE_HASH);
    expect(verified.outcome).toBe("accepted");
    expect(record.envelope.payload.facts).toContainEqual(expect.objectContaining({
      kind: "target-intrinsic",
      familyId: "nvidia:wgmma@1",
      availability: { kind: "recognized-unsupported", diagnosticId: DIAGNOSTIC_ID },
    }));
    expect(Object.isFrozen(verified)).toBe(true);
    expect(Object.isFrozen(record.envelope.payload)).toBe(true);
  });

  it("decodes bounded UTF-8 bytes through the untrusted wire parser", async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(await artifactFixture()));
    const verified = await decodeCppCuteFrontendArtifact(bytes);
    expect(verified.outcome).toBe("accepted");
  });

  it("normalizes set-like record order before deriving identity", async () => {
    const canonical = await artifactFixture();
    const permuted = structuredClone(canonical);
    const mutablePayload = permuted.payload as unknown as Record<string, unknown>;
    (mutablePayload["types"] as unknown[]).reverse();
    (mutablePayload["declarations"] as unknown[]).reverse();
    (mutablePayload["facts"] as unknown[]).reverse();

    const first = await verifyCppCuteFrontendArtifact(canonical);
    const second = await verifyCppCuteFrontendArtifact(permuted);
    expect(second.artifactHash).toBe(first.artifactHash);
    expect(second.transportHash).toBe(first.transportHash);
  });

  it("keeps transport producer outside semantic identity and trust", async () => {
    const firstArtifact = await artifactFixture();
    const secondArtifact = structuredClone(firstArtifact);
    (secondArtifact as unknown as Record<string, unknown>)["producer"] = {
      id: "untrusted/other-extractor",
      version: "99.0.0",
    };

    const first = await verifyCppCuteFrontendArtifact(firstArtifact);
    const second = await verifyCppCuteFrontendArtifact(secondArtifact);
    expect(second.artifactHash).toBe(first.artifactHash);
    expect(second.transportHash).not.toBe(first.transportHash);
  });

  it("rejects structural copies without artifact authority", async () => {
    const verified = await verifyCppCuteFrontendArtifact(await artifactFixture());
    const forged = { ...verified } as VerifiedCppCuteFrontendArtifact;
    expect(() => unwrapVerifiedCppCuteFrontendArtifact(forged)).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED",
      path: "$",
    }));
  });

  it("rejects unknown closed payload fields and optional metadata", async () => {
    const unknown = await cloneArtifact();
    (unknown["payload"] as Record<string, unknown>)["backendResult"] = { trusted: true };
    await expect(verifyCppCuteFrontendArtifact(unknown)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload",
    });

    const metadata = await cloneArtifact();
    metadata["optionalMetadata"] = { trusted: true };
    await expect(verifyCppCuteFrontendArtifact(metadata)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.optionalMetadata",
    });
  });

  it("binds artifactId to normalized semantic content", async () => {
    const value = await cloneArtifact();
    value["artifactId"] = `bg.artifact.cpp-cute-frontend.sha256.${"f".repeat(64)}`;
    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.artifactId",
    });
  });

  it("recomputes source, header, and closure hashes", async () => {
    const value = await cloneArtifact();
    const payload = value["payload"] as Record<string, unknown>;
    const inputs = payload["inputs"] as Record<string, unknown>;
    const files = inputs["files"] as Record<string, unknown>[];
    if (files[0] === undefined) throw new Error("fixture lost main source");
    files[0]["contentSha256"] = "b".repeat(64);
    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.inputs.sourceSetSha256",
    });
  });

  it("rejects host paths, traversal, unreachable files, and dangling include roots", async () => {
    const traversal = await cloneArtifact();
    const inputs = ((traversal["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>);
    const files = inputs["files"] as Record<string, unknown>[];
    if (files[0] === undefined) throw new Error("fixture lost main source");
    files[0]["virtualPath"] = "/src/../private.cu";
    await expect(verifyCppCuteFrontendArtifact(traversal)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.inputs.files[0].virtualPath",
    });

    const dangling = await cloneArtifact();
    const danglingInputs = ((dangling["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>);
    const edges = danglingInputs["includeEdges"] as Record<string, unknown>[];
    const resolution = edges[0]?.["resolution"] as Record<string, unknown> | undefined;
    if (resolution === undefined) throw new Error("fixture lost include resolution");
    resolution["includeRootId"] = stableId("include-root", "9");
    await expect(verifyCppCuteFrontendArtifact(dangling)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE",
      path: "$.payload.inputs.includeEdges[0].resolution.includeRootId",
    });
  });

  it("rejects out-of-bounds spans and macro provenance cycles", async () => {
    const range = await cloneArtifact();
    const spans = (range["payload"] as Record<string, unknown>)["spans"] as Record<string, unknown>[];
    const spelling = spans[0]?.["spelling"] as Record<string, unknown> | undefined;
    if (spelling === undefined) throw new Error("fixture lost span");
    spelling["endByte"] = "101";
    await expect(verifyCppCuteFrontendArtifact(range)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.spans[0].spelling",
    });

    const macro = await cloneArtifact();
    const macroPayload = macro["payload"] as Record<string, unknown>;
    const macroId = stableId("macro", "d");
    macroPayload["macroExpansions"] = [{
      macroExpansionId: macroId,
      macroName: "LAYOUT",
      definitionSpanId: SPAN_ID,
      invocationSpanId: SPAN_ID,
      parentMacroExpansionId: macroId,
    }];
    const macroSpans = macroPayload["spans"] as Record<string, unknown>[];
    if (macroSpans[0] === undefined) throw new Error("fixture lost span");
    macroSpans[0]["macroExpansionId"] = macroId;
    await expect(verifyCppCuteFrontendArtifact(macro)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.macroExpansions",
    });
  });

  it("rejects dangling resolved facts and malformed static layout queries", async () => {
    const dangling = await cloneArtifact();
    const facts = (dangling["payload"] as Record<string, unknown>)["facts"] as Record<string, unknown>[];
    const layout = facts.find((fact) => fact["kind"] === "affine-layout");
    if (layout === undefined) throw new Error("fixture lost layout fact");
    layout["resultDeclarationId"] = stableId("declaration", "d");
    await expect(verifyCppCuteFrontendArtifact(dangling)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE",
    });

    const cosize = await cloneArtifact();
    const cosizeFacts = (cosize["payload"] as Record<string, unknown>)["facts"] as Record<string, unknown>[];
    const cosizeLayout = cosizeFacts.find((fact) => fact["kind"] === "affine-layout");
    if (cosizeLayout === undefined) throw new Error("fixture lost layout fact");
    cosizeLayout["cosize"] = { kind: "integer", value: "5" };
    await expect(verifyCppCuteFrontendArtifact(cosize)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".cosize"),
    });
  });

  it("rejects accepted outcomes with blocking diagnostics", async () => {
    const value = await cloneArtifact();
    const diagnostics = (value["payload"] as Record<string, unknown>)["diagnostics"] as Record<string, unknown>[];
    if (diagnostics[0] === undefined) throw new Error("fixture lost diagnostic");
    diagnostics[0]["severity"] = "error";
    const facts = (value["payload"] as Record<string, unknown>)["facts"] as Record<string, unknown>[];
    const intrinsic = facts.find((fact) => fact["kind"] === "target-intrinsic");
    if (intrinsic === undefined) throw new Error("fixture lost target intrinsic");
    intrinsic["availability"] = { kind: "requires-capability", capabilityIds: ["nvidia:wgmma@1"] };
    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.outcome",
    });
  });

  it("only permits artifact-specific resource limits to lower ceilings", async () => {
    await expect(verifyCppCuteFrontendArtifact(await artifactFixture(), {
      artifactLimits: { maxFiles: 1 },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT",
      path: "$.payload.inputs.files",
    });

    await expect(verifyCppCuteFrontendArtifact(await artifactFixture(), {
      artifactLimits: { maxFiles: 4_097 },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT",
      path: "$.options.artifactLimits.maxFiles",
    });
  });

  it("honors cancellation and derives domain-separated stable IDs", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(verifyCppCuteFrontendArtifact(await artifactFixture(), { signal: controller.signal })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-CANCELLED",
      path: "$.signal",
    });

    const fileId = await deriveCppCuteStableId("file", { virtualPath: "/src/layout.cu" });
    const spanId = await deriveCppCuteStableId("span", { virtualPath: "/src/layout.cu" });
    expect(fileId).toMatch(/^bg\.cpp\.file\.sha256\.[0-9a-f]{64}$/u);
    expect(spanId).not.toBe(fileId);
  });
});
