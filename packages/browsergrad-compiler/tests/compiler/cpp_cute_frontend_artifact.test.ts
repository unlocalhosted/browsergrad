import { describe, expect, it } from "vitest";
import {
  decodeCppCuteFrontendArtifact,
  deriveCppCuteStableId,
  unwrapVerifiedCppCuteFrontendArtifact,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import {
  cloneCppCuteArtifactInput,
  CPP_CUTE_FIXTURE_DIAGNOSTIC_ID,
  CPP_CUTE_FIXTURE_PROFILE_HASH,
  CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID,
  CPP_CUTE_FIXTURE_SPAN_ID,
  createCppCuteArtifactInput,
} from "./support/cpp_cute_frontend_fixtures.js";

function stableId(kind: string, digit: string): string {
  return `bg.cpp.${kind}.sha256.${digit.repeat(64)}`;
}

describe("C++/CuTe frontend artifact", () => {
  it("verifies one closed layout artifact with typed unsupported target fact", async () => {
    const verified = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput());
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
    expect(verified.profileHash).toBe(CPP_CUTE_FIXTURE_PROFILE_HASH);
    expect(verified.outcome).toBe("accepted");
    expect(record.envelope.payload.facts).toContainEqual(expect.objectContaining({
      kind: "target-intrinsic",
      familyId: "nvidia:wgmma@1",
      availability: { kind: "recognized-unsupported", diagnosticId: CPP_CUTE_FIXTURE_DIAGNOSTIC_ID },
    }));
    expect(Object.isFrozen(verified)).toBe(true);
    expect(Object.isFrozen(record.envelope.payload)).toBe(true);
  });

  it("decodes bounded UTF-8 bytes through the untrusted wire parser", async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(await createCppCuteArtifactInput()));
    const verified = await decodeCppCuteFrontendArtifact(bytes);
    expect(verified.outcome).toBe("accepted");
  });

  it("normalizes set-like record order before deriving identity", async () => {
    const canonical = await createCppCuteArtifactInput();
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
    const firstArtifact = await createCppCuteArtifactInput();
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
    const verified = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput());
    const forged = { ...verified } as VerifiedCppCuteFrontendArtifact;
    expect(() => unwrapVerifiedCppCuteFrontendArtifact(forged)).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED",
      path: "$",
    }));
  });

  it("rejects unknown closed payload fields and optional metadata", async () => {
    const unknown = await cloneCppCuteArtifactInput();
    (unknown["payload"] as Record<string, unknown>)["backendResult"] = { trusted: true };
    await expect(verifyCppCuteFrontendArtifact(unknown)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload",
    });

    const metadata = await cloneCppCuteArtifactInput();
    metadata["optionalMetadata"] = { trusted: true };
    await expect(verifyCppCuteFrontendArtifact(metadata)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.optionalMetadata",
    });
  });

  it("binds artifactId to normalized semantic content", async () => {
    const value = await cloneCppCuteArtifactInput();
    value["artifactId"] = `bg.artifact.cpp-cute-frontend.sha256.${"f".repeat(64)}`;
    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.artifactId",
    });
  });

  it("recomputes source, header, and closure hashes", async () => {
    const value = await cloneCppCuteArtifactInput();
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
    const traversal = await cloneCppCuteArtifactInput();
    const inputs = ((traversal["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>);
    const files = inputs["files"] as Record<string, unknown>[];
    if (files[0] === undefined) throw new Error("fixture lost main source");
    files[0]["virtualPath"] = "/src/../private.cu";
    await expect(verifyCppCuteFrontendArtifact(traversal)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.inputs.files[0].virtualPath",
    });

    const dangling = await cloneCppCuteArtifactInput();
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
    const range = await cloneCppCuteArtifactInput();
    const spans = (range["payload"] as Record<string, unknown>)["spans"] as Record<string, unknown>[];
    const spelling = spans[0]?.["spelling"] as Record<string, unknown> | undefined;
    if (spelling === undefined) throw new Error("fixture lost span");
    spelling["endByte"] = "101";
    await expect(verifyCppCuteFrontendArtifact(range)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.spans[0].spelling",
    });

    const macro = await cloneCppCuteArtifactInput();
    const macroPayload = macro["payload"] as Record<string, unknown>;
    const macroId = stableId("macro", "d");
    macroPayload["macroExpansions"] = [{
      macroExpansionId: macroId,
      macroName: "LAYOUT",
      definitionSpanId: CPP_CUTE_FIXTURE_SPAN_ID,
      invocationSpanId: CPP_CUTE_FIXTURE_SPAN_ID,
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
    const dangling = await cloneCppCuteArtifactInput();
    const facts = (dangling["payload"] as Record<string, unknown>)["facts"] as Record<string, unknown>[];
    const layout = facts.find((fact) => fact["kind"] === "affine-layout");
    if (layout === undefined) throw new Error("fixture lost layout fact");
    layout["resultDeclarationId"] = stableId("declaration", "d");
    await expect(verifyCppCuteFrontendArtifact(dangling)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE",
    });

    const cosize = await cloneCppCuteArtifactInput();
    const cosizeFacts = (cosize["payload"] as Record<string, unknown>)["facts"] as Record<string, unknown>[];
    const cosizeLayout = cosizeFacts.find((fact) => fact["kind"] === "affine-layout");
    if (cosizeLayout === undefined) throw new Error("fixture lost layout fact");
    cosizeLayout["cosize"] = { kind: "integer", value: "5" };
    await expect(verifyCppCuteFrontendArtifact(cosize)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".cosize"),
    });

    const wrongRoot = await cloneCppCuteArtifactInput();
    const wrongRootPayload = wrongRoot["payload"] as Record<string, unknown>;
    const entries = wrongRootPayload["entries"] as Record<string, unknown>[];
    if (entries[0] === undefined) throw new Error("fixture lost layout entry");
    entries[0]["selectedRootDeclarationIds"] = [CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID];
    await expect(verifyCppCuteFrontendArtifact(wrongRoot)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.entries[0].selectedRootDeclarationIds",
    });
  });

  it("rejects accepted outcomes with blocking diagnostics", async () => {
    const value = await cloneCppCuteArtifactInput();
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
    await expect(verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput(), {
      artifactLimits: { maxFiles: 1 },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT",
      path: "$.payload.inputs.files",
    });

    await expect(verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput(), {
      artifactLimits: { maxFiles: 4_097 },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT",
      path: "$.options.artifactLimits.maxFiles",
    });

    await expect(verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput(), {
      limits: { maxIntegerBits: 2 },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT",
      path: "$.payload.facts[0].size",
    });
  });

  it("honors cancellation and derives domain-separated stable IDs", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput(), { signal: controller.signal })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-CANCELLED",
      path: "$.signal",
    });

    const fileId = await deriveCppCuteStableId("file", { virtualPath: "/src/layout.cu" });
    const spanId = await deriveCppCuteStableId("span", { virtualPath: "/src/layout.cu" });
    expect(fileId).toMatch(/^bg\.cpp\.file\.sha256\.[0-9a-f]{64}$/u);
    expect(spanId).not.toBe(fileId);
  });
});
