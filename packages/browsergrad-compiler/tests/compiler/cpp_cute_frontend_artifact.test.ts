import { describe, expect, it } from "vitest";
import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  canonicalCppCuteFrontendArtifactBytes,
  decodeCppCuteFrontendArtifact,
  deriveCppCuteFrontendArtifactId,
  deriveCppCuteStableId,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  unwrapVerifiedCppCuteFrontendArtifact,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import { computeCppCuteInputHashes } from "../../src/cpp_cute_frontend_verify.js";
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

async function rebindArtifactId(value: Record<string, unknown>): Promise<void> {
  value["artifactId"] = await deriveCppCuteFrontendArtifactId(value["payload"] as never);
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
      artifactHash: "b007e902edf81d64bf1508ce71cee635fba9bc11d1ab4914ceb3288c7f82b2e2",
      sourceSetSha256: "1c6c78df750362ea1a78dd0513be899140c4b6bbcc7986e476c916c718270a46",
      headerSetSha256: "a2974167b9230f04b7cf95e0d2e2d1304b9974ba90398fadd93d087b12d44b91",
      inputClosureSha256: "4df918262f32e5655e26fc72c7f9053e707c9612216733146d035d75870e2f7b",
    });
    expect(verified.transportHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(verified.artifactBytesSha256).toBe("8f53eb66109db78973ebd082033741e9a00a4430a81c6ac224169acd7bfd4680");
    expect(verified.artifactByteLength).toBe("11064");
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
    const prepared = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput());
    const bytes = canonicalCppCuteFrontendArtifactBytes(prepared);
    const resource = await decodeCppCuteFrontendArtifact(bytes);
    const verified = unwrapVerifiedCppCuteFrontendArtifactResource(resource);
    expect(await sha256Hex(bytes)).toBe(prepared.artifactBytesSha256);
    expect(resource).toMatchObject({
      artifactBytesSha256: prepared.artifactBytesSha256,
      artifactByteLength: String(bytes.byteLength),
    });
    expect(verified.outcome).toBe("accepted");
    expect(() => unwrapVerifiedCppCuteFrontendArtifactResource({ ...resource } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED" }),
    );

    const callerOwnedBytes = new Uint8Array(bytes);
    const pending = decodeCppCuteFrontendArtifact(callerOwnedBytes);
    callerOwnedBytes.fill(0);
    await expect(pending).resolves.toMatchObject({ artifactBytesSha256: prepared.artifactBytesSha256 });

    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as unknown;
    const noncanonical = new TextEncoder().encode(JSON.stringify(parsed, null, 2));
    await expect(decodeCppCuteFrontendArtifact(noncanonical)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-NONCANONICAL-BYTES",
      path: "$bytes",
    });
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
    expect(second.artifactBytesSha256).toBe(first.artifactBytesSha256);
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
    expect(second.artifactBytesSha256).not.toBe(first.artifactBytesSha256);
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
    resolution["includeRootId"] = "unknown-root";
    await expect(verifyCppCuteFrontendArtifact(dangling)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE",
      path: "$.payload.inputs.includeEdges[0].resolution.includeRootId",
    });
  });

  it("treats the virtual root as containing normalized absolute child paths", async () => {
    const input = await createCppCuteArtifactInput();
    const root = input.payload.inputs.includeRoots.find((candidate) => candidate.includeRootId === "cutlass");
    if (root === undefined) throw new Error("fixture lost dependency include root");
    (root as { virtualPath: string }).virtualPath = "/";
    const hashes = await computeCppCuteInputHashes(input.payload);
    (input.payload.inputs as { sourceSetSha256: string }).sourceSetSha256 = hashes.sourceSetSha256;
    (input.payload.inputs as { headerSetSha256: string }).headerSetSha256 = hashes.headerSetSha256;
    (input.payload.inputs as { closureSha256: string }).closureSha256 = hashes.closureSha256;
    (input.payload.extraction as { inputClosureSha256: string }).inputClosureSha256 = hashes.closureSha256;
    (input as { artifactId: string }).artifactId = await deriveCppCuteFrontendArtifactId(input.payload);

    await expect(verifyCppCuteFrontendArtifact(input)).resolves.toMatchObject({ outcome: "accepted" });
  });

  it("models compiler-forced includes without fabricated source directives", async () => {
    const unreachable = await cloneCppCuteArtifactInput();
    const unreachableInputs = (unreachable["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>;
    const unreachableEdges = unreachableInputs["includeEdges"] as Record<string, unknown>[];
    unreachableInputs["includeEdges"] = unreachableEdges.filter((edge) => edge["kind"] !== "compiler-forced");
    await expect(verifyCppCuteFrontendArtifact(unreachable)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("$.payload.inputs.files"),
    });

    const fabricated = await cloneCppCuteArtifactInput();
    const fabricatedInputs = (fabricated["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>;
    const forced = (fabricatedInputs["includeEdges"] as Record<string, unknown>[])
      .find((edge) => edge["kind"] === "compiler-forced");
    if (forced === undefined) throw new Error("fixture lost compiler-forced include");
    forced["directiveSpanId"] = CPP_CUTE_FIXTURE_SPAN_ID;
    await expect(verifyCppCuteFrontendArtifact(fabricated)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("$.payload.inputs.includeEdges"),
    });

    const negativeOrdinal = await cloneCppCuteArtifactInput();
    const ordinalInputs = (negativeOrdinal["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>;
    const ordinalForced = (ordinalInputs["includeEdges"] as Record<string, unknown>[])
      .find((edge) => edge["kind"] === "compiler-forced");
    if (ordinalForced === undefined) throw new Error("fixture lost compiler-forced include");
    ordinalForced["compilerOptionOrdinal"] = -1;
    await expect(verifyCppCuteFrontendArtifact(negativeOrdinal)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".compilerOptionOrdinal"),
    });
  });

  it("enforces exact source, compiler, and dependency ownership", async () => {
    const value = await cloneCppCuteArtifactInput();
    const inputs = (value["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>;
    const compilerHeader = (inputs["files"] as Record<string, unknown>[])
      .find((file) => file["role"] === "compiler-header");
    if (compilerHeader === undefined) throw new Error("fixture lost compiler header");
    compilerHeader["owner"] = { kind: "source" };
    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".owner"),
    });
  });

  it("binds variable definitions to closed initializer expression trees", async () => {
    const valid = await cloneCppCuteArtifactInput();
    const validPayload = valid["payload"] as Record<string, unknown>;
    const declarations = validPayload["declarations"] as Record<string, unknown>[];
    const variable = declarations.find((declaration) => declaration["kind"] === "variable");
    if (variable === undefined) throw new Error("fixture lost variable declaration");
    const expressionId = stableId("expression", "8");
    variable["initializerExpressionId"] = expressionId;
    validPayload["initializerExpressions"] = [{
      expressionId,
      typeId: variable["typeId"],
      valueCategory: "lvalue",
      origin: { kind: "source", spanId: CPP_CUTE_FIXTURE_SPAN_ID },
      kind: "declaration-reference",
      declarationId: variable["declarationId"],
    }];
    await rebindArtifactId(valid);
    await expect(verifyCppCuteFrontendArtifact(valid)).resolves.toMatchObject({ outcome: "accepted" });

    const dangling = await cloneCppCuteArtifactInput();
    const danglingDeclarations = (dangling["payload"] as Record<string, unknown>)["declarations"] as Record<string, unknown>[];
    const danglingVariable = danglingDeclarations.find((declaration) => declaration["kind"] === "variable");
    if (danglingVariable === undefined) throw new Error("fixture lost variable declaration");
    danglingVariable["initializerExpressionId"] = stableId("expression", "9");
    await expect(verifyCppCuteFrontendArtifact(dangling)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-DANGLING-REFERENCE",
      path: expect.stringContaining(".initializerExpressionId"),
    });

    const wrongKind = await cloneCppCuteArtifactInput();
    const wrongDeclarations = (wrongKind["payload"] as Record<string, unknown>)["declarations"] as Record<string, unknown>[];
    const record = wrongDeclarations.find((declaration) => declaration["kind"] === "record");
    if (record === undefined) throw new Error("fixture lost record declaration");
    record["initializerExpressionId"] = stableId("expression", "a");
    await expect(verifyCppCuteFrontendArtifact(wrongKind)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".initializerExpressionId"),
    });
  });

  it("permits honest locationless compiler diagnostics only for non-source subjects", async () => {
    const compiler = await cloneCppCuteArtifactInput();
    const compilerPayload = compiler["payload"] as Record<string, unknown>;
    const diagnostics = compilerPayload["diagnostics"] as Record<string, unknown>[];
    if (diagnostics[0] === undefined) throw new Error("fixture lost diagnostic");
    diagnostics[0]["location"] = { kind: "none" };
    diagnostics[0]["subject"] = { kind: "compiler" };
    await rebindArtifactId(compiler);
    await expect(verifyCppCuteFrontendArtifact(compiler)).resolves.toMatchObject({ outcome: "accepted" });

    const fabricated = await cloneCppCuteArtifactInput();
    const fabricatedDiagnostics = (fabricated["payload"] as Record<string, unknown>)["diagnostics"] as Record<string, unknown>[];
    if (fabricatedDiagnostics[0] === undefined) throw new Error("fixture lost diagnostic");
    fabricatedDiagnostics[0]["location"] = { kind: "none" };
    await expect(verifyCppCuteFrontendArtifact(fabricated)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".location"),
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
