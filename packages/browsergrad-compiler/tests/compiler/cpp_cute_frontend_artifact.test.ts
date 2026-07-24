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
import {
  computeCppCuteInputHashes,
  computeCppCuteSemanticPassInputClosureHash,
  computeCppCuteSharedSurfaceHash,
  deriveCppCuteSourceEntityId,
} from "../../src/cpp_cute_frontend_verify.js";
import { findCppCuteFrontendProfileBindingMismatch } from "../../src/cpp_cute_frontend_profile_binding.js";
import {
  cloneCppCuteArtifactInput,
  CPP_CUTE_FIXTURE_DIAGNOSTIC_ID,
  CPP_CUTE_FIXTURE_COMPILATION_CONTRACT_HASH,
  CPP_CUTE_FIXTURE_MAIN_FILE_ID,
  CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID,
  CPP_CUTE_FIXTURE_SPAN_ID,
  createCppCuteArtifactInput,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";
import {
  CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID,
  CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
  CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID,
  CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_EXPRESSION_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
  CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID,
  CPP_CUTE_VIEW_COPY_VOID_TYPE_ID,
  createCppCuteViewCopyArtifactInput,
} from "./support/cpp_cute_frontend_view_copy_fixtures.js";

function stableId(kind: string, digit: string): string {
  return `bg.cpp.${kind}.sha256.${digit.repeat(64)}`;
}

async function rebindArtifactId(value: Record<string, unknown>): Promise<void> {
  value["artifactId"] = await deriveCppCuteFrontendArtifactId(value["payload"] as never);
}

async function rebindInputAndPassEvidence(value: Record<string, unknown>): Promise<void> {
  const payload = value["payload"] as Record<string, unknown>;
  const inputs = payload["inputs"] as Record<string, unknown>;
  const hashes = await computeCppCuteInputHashes(payload as never);
  inputs["sourceSetSha256"] = hashes.sourceSetSha256;
  inputs["headerSetSha256"] = hashes.headerSetSha256;
  inputs["closureSha256"] = hashes.closureSha256;
  (payload["extraction"] as Record<string, unknown>)["inputClosureSha256"] = hashes.closureSha256;
  const passes = payload["semanticPasses"] as Record<string, unknown>[];
  for (const [index, pass] of passes.entries()) {
    if (pass["status"] === "not-run") continue;
    pass["observedInputClosureSha256"] = await computeCppCuteSemanticPassInputClosureHash(payload as never, index);
    pass["sharedSurfaceSha256"] = await computeCppCuteSharedSurfaceHash(
      payload as never,
      pass["domain"] as "host" | "device",
    );
  }
  await rebindArtifactId(value);
}

function clearNotRunPass(pass: Record<string, unknown>): void {
  pass["status"] = "not-run";
  pass["openedFileIds"] = [];
  pass["includeEdgeIds"] = [];
  pass["observedInputClosureSha256"] = null;
  pass["sharedSurfaceSha256"] = null;
  pass["selectedSourceRootEntityIds"] = [];
  pass["factIds"] = [];
  pass["diagnosticIds"] = [];
}

function payloadRecord(artifact: Record<string, unknown>): Record<string, unknown> {
  return artifact["payload"] as Record<string, unknown>;
}

function recordById(
  payload: Record<string, unknown>,
  collection: string,
  idField: string,
  id: string,
): Record<string, unknown> {
  const record = (payload[collection] as Record<string, unknown>[])
    .find((candidate) => candidate[idField] === id);
  if (record === undefined) throw new Error(`view-copy fixture lost ${collection} record ${id}`);
  return record;
}

async function expectInvalidViewCopy(
  mutate: (payload: Record<string, unknown>) => void,
  path: string | RegExp,
): Promise<void> {
  const value = await createCppCuteViewCopyArtifactInput();
  mutate(payloadRecord(value));
  await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
    code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
    path: typeof path === "string" ? path : expect.stringMatching(path),
  });
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
      artifactHash: "9c9523b934fa5904df8ba047fc183a6d7116cd98d1a139a9c90cb6a20544bd79",
      sourceSetSha256: "1c6c78df750362ea1a78dd0513be899140c4b6bbcc7986e476c916c718270a46",
      headerSetSha256: "b737489faf070a5b30e0664ba0aaeb3a209caa7b7cc8380d74a3a54683729649",
      inputClosureSha256: "70acc82f15520f388eafbc75a2c75952d5d34d067da2c21fcb98f47d489e588f",
    });
    expect(verified.transportHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(verified.artifactBytesSha256).toBe("3e06e4b9881bb6d51c45e5e40e038cf9e8fb1778ee0381374a573aae31091754");
    expect(verified.artifactByteLength).toBe("15081");
    expect(verified.compilationContractHash).toBe(CPP_CUTE_FIXTURE_COMPILATION_CONTRACT_HASH);
    expect(verified.outcome).toBe("accepted");
    expect(record.envelope.payload.semanticPasses).toEqual([
      expect.objectContaining({ ordinal: 0, passId: "cuda-device-sema", domain: "device", status: "succeeded" }),
      expect.objectContaining({ ordinal: 1, passId: "cuda-host-sema", domain: "host", status: "succeeded" }),
    ]);
    expect(record.envelope.payload.sourceAbi.types.map(({ domain, sourceTypeEntityId }) => ({ domain, sourceTypeEntityId }))).toEqual([
      { domain: "device", sourceTypeEntityId: expect.stringMatching(/^bg\.cpp\.source-entity\.sha256\./u) },
      { domain: "host", sourceTypeEntityId: expect.stringMatching(/^bg\.cpp\.source-entity\.sha256\./u) },
    ]);
    expect(record.envelope.payload.facts).toContainEqual(expect.objectContaining({
      kind: "target-intrinsic",
      familyId: "nvidia:wgmma@1",
      availability: { kind: "recognized-unsupported", diagnosticId: CPP_CUTE_FIXTURE_DIAGNOSTIC_ID },
    }));
    expect(Object.isFrozen(verified)).toBe(true);
    expect(Object.isFrozen(record.envelope.payload)).toBe(true);
  });

  it("binds source-scoped Clang USRs to their exact spelling file and offset", async () => {
    const scoped = await cloneCppCuteArtifactInput();
    const payload = scoped["payload"] as Record<string, unknown>;
    const declaration = recordById(
      payload,
      "declarations",
      "declarationId",
      CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID,
    );
    declaration["canonicalUsr"] = "c:layout.cu@0@N@cute@S@Layout>#I#I";
    await rebindArtifactId(scoped);
    await expect(verifyCppCuteFrontendArtifact(scoped)).resolves.toMatchObject({
      outcome: "accepted",
    });

    const mismatched = await cloneCppCuteArtifactInput();
    const mismatchedDeclaration = recordById(
      mismatched["payload"] as Record<string, unknown>,
      "declarations",
      "declarationId",
      CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID,
    );
    mismatchedDeclaration["canonicalUsr"] =
      "c:other.cu@0@N@cute@S@Layout>#I#I";
    await rebindArtifactId(mismatched);
    await expect(verifyCppCuteFrontendArtifact(mismatched)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringMatching(/\.canonicalUsr$/u),
    });
  });

  it("verifies one function-owned typed view-copy graph with compatible strided layouts", async () => {
    const verified = await verifyCppCuteFrontendArtifact(await createCppCuteViewCopyArtifactInput());
    const payload = unwrapVerifiedCppCuteFrontendArtifact(verified).envelope.payload;
    expect(payload.entries).toEqual([{
      entryId: expect.any(String),
      kind: "view-copy",
      sourceTensorFactId: CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_FACT_ID,
      destinationTensorFactId: CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID,
      operationExpressionId: CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID,
      selectedRootDeclarationIds: [CPP_CUTE_VIEW_COPY_FUNCTION_DECLARATION_ID],
    }]);
    expect(payload.facts).toContainEqual(expect.objectContaining({
      factId: CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID,
      operation: expect.objectContaining({
        kind: "copy",
        sourceSpace: "global",
        destinationSpace: "global",
        transferBits: 32,
      }),
      effects: {
        readsMemory: true,
        writesMemory: true,
        synchronizes: false,
        convergent: false,
      },
    }));
  });

  it("rejects cross-wired view-copy roots, operands, tensor contracts, engines, and intrinsic contracts", async () => {
    await expectInvalidViewCopy((payload) => {
      const entry = (payload["entries"] as Record<string, unknown>[])[0];
      if (entry === undefined) throw new Error("view-copy fixture lost entry");
      entry["selectedRootDeclarationIds"] = [CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID];
    }, "$.payload.entries[0].selectedRootDeclarationIds[0]");

    await expectInvalidViewCopy((payload) => {
      const entry = (payload["entries"] as Record<string, unknown>[])[0];
      if (entry === undefined) throw new Error("view-copy fixture lost entry");
      entry["operationExpressionId"] = CPP_CUTE_VIEW_COPY_SOURCE_EXPRESSION_ID;
    }, "$.payload.entries[0].operationExpressionId");

    await expectInvalidViewCopy((payload) => {
      const intrinsic = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID);
      intrinsic["operandExpressionIds"] = [
        (intrinsic["operandExpressionIds"] as string[])[1],
        (intrinsic["operandExpressionIds"] as string[])[0],
      ];
    }, "$.payload.entries[0].operationExpressionId.intrinsicFactId.operandExpressionIds[0]");

    await expectInvalidViewCopy((payload) => {
      const intrinsic = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID);
      intrinsic["operandExpressionIds"] = [
        CPP_CUTE_VIEW_COPY_SOURCE_EXPRESSION_ID,
        CPP_CUTE_VIEW_COPY_OPERATION_EXPRESSION_ID,
      ];
    }, /operandExpressionIds\[1\]$/u);

    await expectInvalidViewCopy((payload) => {
      const sourceDeclaration = recordById(
        payload,
        "declarations",
        "declarationId",
        CPP_CUTE_VIEW_COPY_SOURCE_TENSOR_DECLARATION_ID,
      );
      sourceDeclaration["lexicalParentId"] = null;
      sourceDeclaration["semanticParentId"] = null;
    }, "$.payload.entries[0]");

    await expectInvalidViewCopy((payload) => {
      const destination = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID);
      destination["elementTypeId"] = CPP_CUTE_VIEW_COPY_VOID_TYPE_ID;
    }, "$.payload.entries[0]");

    await expectInvalidViewCopy((payload) => {
      const layout = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_DESTINATION_LAYOUT_FACT_ID);
      const shape = layout["shape"] as Record<string, unknown>;
      const first = (shape["elements"] as Record<string, unknown>[])[0];
      if (first === undefined) throw new Error("view-copy fixture lost layout shape");
      (first["value"] as Record<string, unknown>)["value"] = "4";
      layout["size"] = { kind: "integer", value: "8" };
      layout["cosize"] = { kind: "integer", value: "8" };
    }, "$.payload.entries[0]");

    await expectInvalidViewCopy((payload) => {
      const destination = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_DESTINATION_TENSOR_FACT_ID);
      destination["engine"] = {
        kind: "global-pointer",
        pointerDeclarationId: CPP_CUTE_VIEW_COPY_SOURCE_ENGINE_DECLARATION_ID,
        nullable: false,
      };
    }, "$.payload.entries[0]");

    await expectInvalidViewCopy((payload) => {
      const intrinsic = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID);
      (intrinsic["operation"] as Record<string, unknown>)["sourceSpace"] = "shared";
    }, "$.payload.entries[0].operationExpressionId.intrinsicFactId.operation");

    await expectInvalidViewCopy((payload) => {
      const intrinsic = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID);
      (intrinsic["effects"] as Record<string, unknown>)["writesMemory"] = false;
    }, "$.payload.entries[0].operationExpressionId.intrinsicFactId.effects");

    await expectInvalidViewCopy((payload) => {
      const intrinsic = recordById(payload, "facts", "factId", CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID);
      (intrinsic["operation"] as Record<string, unknown>)["transferBits"] = 16;
    }, "$.payload.entries[0].operationExpressionId.intrinsicFactId.operation.transferBits");
  });

  it("requires every recognized-unsupported intrinsic diagnostic to name its exact fact", async () => {
    await expectInvalidViewCopy((payload) => {
      const diagnostic = recordById(payload, "diagnostics", "diagnosticId", CPP_CUTE_FIXTURE_DIAGNOSTIC_ID);
      diagnostic["subject"] = { kind: "fact", factId: CPP_CUTE_VIEW_COPY_INTRINSIC_FACT_ID };
    }, /\.availability\.diagnosticId$/u);
  });

  it("closes semantic-pass evidence and rejects missing, duplicate, reordered, or cross-domain facts", async () => {
    const missing = await cloneCppCuteArtifactInput();
    const missingPayload = missing["payload"] as Record<string, unknown>;
    const missingPasses = missingPayload["semanticPasses"] as Record<string, unknown>[];
    if (missingPasses[0] === undefined) throw new Error("fixture lost device semantic pass");
    missingPasses[0]["factIds"] = [];
    await expect(verifyCppCuteFrontendArtifact(missing)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[0].factIds",
    });

    const duplicate = await cloneCppCuteArtifactInput();
    const duplicatePayload = duplicate["payload"] as Record<string, unknown>;
    const duplicatePasses = duplicatePayload["semanticPasses"] as Record<string, unknown>[];
    const duplicateFacts = duplicatePayload["facts"] as Record<string, unknown>[];
    if (duplicatePasses[1] === undefined || duplicateFacts[0] === undefined) {
      throw new Error("fixture lost semantic pass facts");
    }
    duplicatePasses[1]["factIds"] = [duplicateFacts[0]["factId"], ...(duplicatePasses[1]["factIds"] as unknown[])];
    await expect(verifyCppCuteFrontendArtifact(duplicate)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[1].factIds[0]",
    });

    const reordered = await cloneCppCuteArtifactInput();
    const reorderedPasses = (reordered["payload"] as Record<string, unknown>)["semanticPasses"] as unknown[];
    reorderedPasses.reverse();
    await expect(verifyCppCuteFrontendArtifact(reordered)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[0].ordinal",
    });

    const crossDomain = await cloneCppCuteArtifactInput();
    const crossPayload = crossDomain["payload"] as Record<string, unknown>;
    const crossPasses = crossPayload["semanticPasses"] as Record<string, unknown>[];
    const crossFacts = crossPayload["facts"] as Record<string, unknown>[];
    if (crossPasses[0] === undefined || crossPasses[1] === undefined ||
        crossFacts[0] === undefined || crossFacts[1] === undefined) {
      throw new Error("fixture lost cross-domain fact");
    }
    crossPasses[0]["factIds"] = [crossFacts[1]["factId"]];
    crossPasses[1]["factIds"] = [crossFacts[0]["factId"]];
    await expect(verifyCppCuteFrontendArtifact(crossDomain)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[0].factIds",
    });
  });

  it("permits target-conditional pass closures while requiring forced includes in every executed pass", async () => {
    const divergent = await cloneCppCuteArtifactInput();
    const payload = divergent["payload"] as Record<string, unknown>;
    const passes = payload["semanticPasses"] as Record<string, unknown>[];
    const hostPass = passes[1];
    if (hostPass === undefined) throw new Error("fixture lost host semantic pass");
    const inputs = payload["inputs"] as Record<string, unknown>;
    const dependencyHeader = (inputs["files"] as Record<string, unknown>[])
      .find((file) => file["role"] === "dependency-header");
    if (dependencyHeader === undefined) throw new Error("fixture lost dependency header");
    hostPass["openedFileIds"] = (hostPass["openedFileIds"] as string[])
      .filter((id) => id !== dependencyHeader["fileId"]);
    const sourceEdge = (inputs["includeEdges"] as Record<string, unknown>[])
      .find((edge) => edge["kind"] === "source-directive");
    if (sourceEdge === undefined) throw new Error("fixture lost source include edge");
    hostPass["includeEdgeIds"] = (hostPass["includeEdgeIds"] as string[])
      .filter((id) => id !== sourceEdge["includeEdgeId"]);
    await rebindInputAndPassEvidence(divergent);
    await expect(verifyCppCuteFrontendArtifact(divergent)).resolves.toMatchObject({ outcome: "accepted" });

    const missingForced = structuredClone(divergent) as Record<string, unknown>;
    const missingPayload = missingForced["payload"] as Record<string, unknown>;
    const missingHost = (missingPayload["semanticPasses"] as Record<string, unknown>[])[1];
    if (missingHost === undefined) throw new Error("fixture lost host semantic pass");
    const forcedEdge = ((missingPayload["inputs"] as Record<string, unknown>)["includeEdges"] as Record<string, unknown>[])
      .find((edge) => edge["kind"] === "compiler-forced");
    if (forcedEdge === undefined) throw new Error("fixture lost forced include edge");
    missingHost["includeEdgeIds"] = (missingHost["includeEdgeIds"] as string[])
      .filter((id) => id !== forcedEdge["includeEdgeId"]);
    missingHost["openedFileIds"] = (missingHost["openedFileIds"] as string[])
      .filter((id) => id !== forcedEdge["fileId"]);
    await expect(verifyCppCuteFrontendArtifact(missingForced)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[1].includeEdgeIds",
    });
  });

  it("binds unresolved-include diagnostics to the pass that observed the edge", async () => {
    const value = await cloneCppCuteArtifactInput();
    const payload = value["payload"] as Record<string, unknown>;
    const inputs = payload["inputs"] as Record<string, unknown>;
    const files = inputs["files"] as Record<string, unknown>[];
    const dependencyHeader = files.find((file) => file["role"] === "dependency-header");
    const sourceEdge = (inputs["includeEdges"] as Record<string, unknown>[])
      .find((edge) => edge["kind"] === "source-directive");
    if (dependencyHeader === undefined || sourceEdge === undefined) throw new Error("fixture lost source include closure");
    const blockingId = stableId("diagnostic", "d");
    sourceEdge["resolution"] = { kind: "unresolved", diagnosticId: blockingId };
    inputs["files"] = files.filter((file) => file !== dependencyHeader);

    const passes = payload["semanticPasses"] as Record<string, unknown>[];
    const devicePass = passes[0];
    const hostPass = passes[1];
    if (devicePass === undefined || hostPass === undefined) throw new Error("fixture lost semantic passes");
    for (const pass of passes) {
      pass["openedFileIds"] = (pass["openedFileIds"] as string[])
        .filter((id) => id !== dependencyHeader["fileId"]);
    }
    hostPass["includeEdgeIds"] = (hostPass["includeEdgeIds"] as string[])
      .filter((id) => id !== sourceEdge["includeEdgeId"]);
    hostPass["status"] = "failed";
    hostPass["diagnosticIds"] = [blockingId];
    (payload["diagnostics"] as unknown[]).push({
      diagnosticId: blockingId,
      phase: "preprocessing",
      severity: "error",
      code: "browsergrad.cpp-cute:unresolved-include",
      renderedMessage: "Header is unavailable in the host semantic pass.",
      location: { kind: "none" },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    });
    payload["outcome"] = { kind: "rejected", blockingDiagnosticIds: [blockingId] };
    await rebindInputAndPassEvidence(value);

    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".resolution.diagnosticId"),
    });
  });

  it("rejects shared-source drift, unowned diagnostics, and cross-pass notes", async () => {
    const forgedIdentity = await cloneCppCuteArtifactInput();
    const forgedPayload = forgedIdentity["payload"] as Record<string, unknown>;
    const forgedEntity = (forgedPayload["sourceEntities"] as Record<string, unknown>[])[0];
    if (forgedEntity === undefined) throw new Error("fixture lost source entity");
    forgedEntity["canonicalIdentity"] = "signed int";
    await expect(verifyCppCuteFrontendArtifact(forgedIdentity)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.sourceEntities[0].sourceEntityId",
    });

    const hiddenShared = await cloneCppCuteArtifactInput();
    const hiddenPayload = hiddenShared["payload"] as Record<string, unknown>;
    const hiddenAbi = hiddenPayload["sourceAbi"] as Record<string, unknown>;
    for (const entry of hiddenAbi["types"] as Record<string, unknown>[]) entry["shared"] = false;
    await rebindInputAndPassEvidence(hiddenShared);
    await expect(verifyCppCuteFrontendArtifact(hiddenShared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("$.payload.sourceAbi.types"),
    });

    const drift = await cloneCppCuteArtifactInput();
    const driftPayload = drift["payload"] as Record<string, unknown>;
    const hostPass = (driftPayload["semanticPasses"] as Record<string, unknown>[])[1];
    if (hostPass === undefined) throw new Error("fixture lost host ABI surface");
    hostPass["sharedSurfaceSha256"] = "0".repeat(64);
    await expect(verifyCppCuteFrontendArtifact(drift)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.semanticPasses[1].sharedSurfaceSha256",
    });

    const unowned = await cloneCppCuteArtifactInput();
    const unownedPayload = unowned["payload"] as Record<string, unknown>;
    (unownedPayload["diagnostics"] as unknown[]).push({
      diagnosticId: stableId("diagnostic", "d"),
      phase: "parsing",
      severity: "warning",
      code: "browsergrad.cpp-cute:unowned-parser-warning",
      renderedMessage: "Parser warning must belong to one semantic pass.",
      location: { kind: "none" },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    });
    await expect(verifyCppCuteFrontendArtifact(unowned)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("$.payload.diagnostics"),
    });

    const crossNote = await cloneCppCuteArtifactInput();
    const crossPayload = crossNote["payload"] as Record<string, unknown>;
    const noteId = stableId("diagnostic", "d");
    (crossPayload["diagnostics"] as unknown[]).push({
      diagnosticId: noteId,
      phase: "artifact-extraction",
      severity: "note",
      code: "browsergrad.cpp-cute:cross-pass-note",
      renderedMessage: "Notes cannot cross semantic-pass ownership.",
      location: { kind: "none" },
      subject: { kind: "compiler" },
      parentDiagnosticId: CPP_CUTE_FIXTURE_DIAGNOSTIC_ID,
    });
    const crossHost = (crossPayload["semanticPasses"] as Record<string, unknown>[])[1];
    if (crossHost === undefined) throw new Error("fixture lost host semantic pass");
    crossHost["diagnosticIds"] = [noteId];
    await expect(verifyCppCuteFrontendArtifact(crossNote)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".parentDiagnosticId"),
    });

    const hostGraphReference = await cloneCppCuteArtifactInput();
    const hostGraphPayload = hostGraphReference["payload"] as Record<string, unknown>;
    const hostDiagnosticId = stableId("diagnostic", "d");
    (hostGraphPayload["diagnostics"] as unknown[]).push({
      diagnosticId: hostDiagnosticId,
      phase: "cuda-sema",
      severity: "warning",
      code: "browsergrad.cpp-cute:host-device-graph-reference",
      renderedMessage: "Host diagnostics cannot reference device graph declarations.",
      location: { kind: "source", primarySpanId: CPP_CUTE_FIXTURE_SPAN_ID, related: [] },
      subject: { kind: "declaration", declarationId: CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID },
      parentDiagnosticId: null,
    });
    const hostGraphPass = (hostGraphPayload["semanticPasses"] as Record<string, unknown>[])[1];
    if (hostGraphPass === undefined) throw new Error("fixture lost host semantic pass");
    hostGraphPass["diagnosticIds"] = [hostDiagnosticId];
    await expect(verifyCppCuteFrontendArtifact(hostGraphReference)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".subject"),
    });
  });

  it("accepts an unannotated CuTe layout variable selected by the device-owned entry graph", async () => {
    const unannotated = await cloneCppCuteArtifactInput();
    const payload = unannotated["payload"] as Record<string, unknown>;
    const declarations = payload["declarations"] as Record<string, unknown>[];
    const entries = payload["entries"] as Record<string, unknown>[];
    const selectedDeclarationId = (entries[0]?.["selectedRootDeclarationIds"] as string[] | undefined)?.[0];
    const selected = declarations.find(
      (declaration) => declaration["declarationId"] === selectedDeclarationId,
    );
    if (selected === undefined) throw new Error("fixture lost selected layout declaration");
    selected["cudaAttributes"] = {
      host: false,
      device: false,
      global: false,
      forceInline: false,
    };
    await rebindInputAndPassEvidence(unannotated);

    const verified = await verifyCppCuteFrontendArtifact(unannotated);
    const record = unwrapVerifiedCppCuteFrontendArtifact(verified);
    expect(record.envelope.payload.entries[0]?.selectedRootDeclarationIds).toEqual([
      selectedDeclarationId,
    ]);
    expect(record.envelope.payload.semanticPasses[0]?.factIds).toEqual(
      record.envelope.payload.facts.map((fact) => fact.factId),
    );
  });

  it("does not let a CUDA attribute manufacture selected-root membership", async () => {
    const attributeOnly = await cloneCppCuteArtifactInput();
    const payload = attributeOnly["payload"] as Record<string, unknown>;
    const declarations = payload["declarations"] as Record<string, unknown>[];
    const selected = declarations.find((declaration) => declaration["kind"] === "variable");
    if (selected === undefined) throw new Error("fixture lost selected layout declaration");
    const candidateDeclarationId = stableId("declaration", "e");
    const candidateCanonicalIdentity = "c:@attribute_only_layout";
    declarations.push({
      ...structuredClone(selected),
      declarationId: candidateDeclarationId,
      canonicalUsr: candidateCanonicalIdentity,
      canonicalName: "attribute_only_layout",
      mangledName: "_ZL21attribute_only_layout",
      cudaAttributes: {
        host: false,
        device: true,
        global: false,
        forceInline: false,
      },
    });
    declarations.sort((left, right) =>
      String(left["declarationId"]).localeCompare(String(right["declarationId"])));
    const candidateBody = {
      entityKind: "variable" as const,
      canonicalIdentity: candidateCanonicalIdentity,
      origin: structuredClone(selected["origin"]),
      domains: ["device"] as const,
    };
    const candidateSourceEntityId = await deriveCppCuteSourceEntityId(
      payload as never,
      candidateBody as never,
    );
    const sourceEntities = payload["sourceEntities"] as Record<string, unknown>[];
    sourceEntities.push({ sourceEntityId: candidateSourceEntityId, ...candidateBody });
    sourceEntities.sort((left, right) =>
      String(left["sourceEntityId"]).localeCompare(String(right["sourceEntityId"])));
    const devicePass = (payload["semanticPasses"] as Record<string, unknown>[])[0];
    if (devicePass === undefined) throw new Error("fixture lost device semantic pass");
    devicePass["selectedSourceRootEntityIds"] = [
      ...(devicePass["selectedSourceRootEntityIds"] as string[]),
      candidateSourceEntityId,
    ].sort();
    await rebindInputAndPassEvidence(attributeOnly);

    await expect(verifyCppCuteFrontendArtifact(attributeOnly)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.semanticPasses[0].selectedSourceRootEntityIds",
    });
  });

  it("rejects split host/device identities for one selected source root", async () => {
    const split = await cloneCppCuteArtifactInput();
    const payload = split["payload"] as Record<string, unknown>;
    const declarations = payload["declarations"] as Record<string, unknown>[];
    const selected = declarations.find((declaration) => declaration["kind"] === "variable");
    if (selected === undefined) throw new Error("fixture lost selected layout declaration");
    selected["cudaAttributes"] = {
      host: false,
      device: false,
      global: false,
      forceInline: false,
    };
    const entities = payload["sourceEntities"] as Record<string, unknown>[];
    const realRoot = entities.find((entity) => entity["entityKind"] === "variable");
    if (realRoot === undefined) throw new Error("fixture lost selected root source entity");
    realRoot["domains"] = ["device"];
    const forgedBody = {
      entityKind: "variable" as const,
      canonicalIdentity: "c:@forged_layout",
      origin: structuredClone(realRoot["origin"]),
      domains: ["host"] as const,
    };
    const forgedId = await deriveCppCuteSourceEntityId(payload as never, forgedBody as never);
    entities.push({ sourceEntityId: forgedId, ...forgedBody });
    entities.sort((left, right) => String(left["sourceEntityId"]).localeCompare(String(right["sourceEntityId"])));
    const hostPass = (payload["semanticPasses"] as Record<string, unknown>[])[1];
    if (hostPass === undefined) throw new Error("fixture lost host semantic pass");
    hostPass["selectedSourceRootEntityIds"] = [forgedId];
    await rebindInputAndPassEvidence(split);

    await expect(verifyCppCuteFrontendArtifact(split)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.semanticPasses[1].selectedSourceRootEntityIds",
    });
  });

  it("enforces the device-first pass state matrix", async () => {
    const skippedHost = await cloneCppCuteArtifactInput();
    const skippedPayload = skippedHost["payload"] as Record<string, unknown>;
    const skippedPass = (skippedPayload["semanticPasses"] as Record<string, unknown>[])[1];
    if (skippedPass === undefined) throw new Error("fixture lost host semantic pass");
    clearNotRunPass(skippedPass);
    const skippedAbi = skippedPayload["sourceAbi"] as Record<string, unknown>;
    skippedAbi["types"] = (skippedAbi["types"] as Record<string, unknown>[])
      .filter((entry) => entry["domain"] !== "host");
    skippedAbi["functions"] = (skippedAbi["functions"] as Record<string, unknown>[])
      .filter((entry) => entry["domain"] !== "host");
    for (const entity of skippedPayload["sourceEntities"] as Record<string, unknown>[]) entity["domains"] = ["device"];
    for (const entry of skippedAbi["types"] as Record<string, unknown>[]) entry["shared"] = false;
    for (const entry of skippedAbi["functions"] as Record<string, unknown>[]) entry["shared"] = false;
    const skippedDevice = (skippedPayload["semanticPasses"] as Record<string, unknown>[])[0];
    if (skippedDevice === undefined) throw new Error("fixture lost device semantic pass");
    skippedDevice["sharedSurfaceSha256"] = await computeCppCuteSharedSurfaceHash(skippedPayload as never, "device");
    await expect(verifyCppCuteFrontendArtifact(skippedHost)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[1].status",
    });

    const deviceFailure = await cloneCppCuteArtifactInput();
    const failedPayload = deviceFailure["payload"] as Record<string, unknown>;
    const failedPasses = failedPayload["semanticPasses"] as Record<string, unknown>[];
    const devicePass = failedPasses[0];
    const notRunHost = failedPasses[1];
    if (devicePass === undefined || notRunHost === undefined) throw new Error("fixture lost semantic passes");
    const blockingId = stableId("diagnostic", "d");
    (failedPayload["diagnostics"] as unknown[]).push({
      diagnosticId: blockingId,
      phase: "cuda-sema",
      severity: "error",
      code: "browsergrad.cpp-cute:device-sema-failed",
      renderedMessage: "Device semantic extraction failed.",
      location: { kind: "none" },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    });
    devicePass["status"] = "failed";
    devicePass["diagnosticIds"] = [...(devicePass["diagnosticIds"] as string[]), blockingId];
    devicePass["selectedSourceRootEntityIds"] = [];
    clearNotRunPass(notRunHost);
    const failedAbi = failedPayload["sourceAbi"] as Record<string, unknown>;
    failedAbi["types"] = (failedAbi["types"] as Record<string, unknown>[])
      .filter((entry) => entry["domain"] !== "host");
    failedAbi["functions"] = (failedAbi["functions"] as Record<string, unknown>[])
      .filter((entry) => entry["domain"] !== "host");
    const failedEntities = failedPayload["sourceEntities"] as Record<string, unknown>[];
    failedPayload["sourceEntities"] = failedEntities
      .filter((entity) => entity["entityKind"] !== "variable")
      .map((entity) => ({ ...entity, domains: ["device"] }));
    for (const entry of failedAbi["types"] as Record<string, unknown>[]) entry["shared"] = false;
    for (const entry of failedAbi["functions"] as Record<string, unknown>[]) entry["shared"] = false;
    devicePass["sharedSurfaceSha256"] = null;
    failedPayload["entries"] = [];
    failedPayload["outcome"] = { kind: "rejected", blockingDiagnosticIds: [blockingId] };
    await rebindArtifactId(deviceFailure);
    await expect(verifyCppCuteFrontendArtifact(deviceFailure)).resolves.toMatchObject({ outcome: "rejected" });
  });

  it("binds semantic-pass diagnostics, input closure, status, target, and ABI domain", async () => {
    const missingDiagnostic = await cloneCppCuteArtifactInput();
    const missingDiagnosticPasses = (
      (missingDiagnostic["payload"] as Record<string, unknown>)["semanticPasses"] as Record<string, unknown>[]
    );
    if (missingDiagnosticPasses[0] === undefined) throw new Error("fixture lost device semantic pass");
    missingDiagnosticPasses[0]["diagnosticIds"] = [];
    await expect(verifyCppCuteFrontendArtifact(missingDiagnostic)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".availability.diagnosticId"),
    });

    const closure = await cloneCppCuteArtifactInput();
    const closurePasses = (closure["payload"] as Record<string, unknown>)["semanticPasses"] as Record<string, unknown>[];
    if (closurePasses[0] === undefined) throw new Error("fixture lost device semantic pass");
    closurePasses[0]["observedInputClosureSha256"] = "0".repeat(64);
    await expect(verifyCppCuteFrontendArtifact(closure)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.semanticPasses[0].observedInputClosureSha256",
    });

    const status = await cloneCppCuteArtifactInput();
    const statusPasses = (status["payload"] as Record<string, unknown>)["semanticPasses"] as Record<string, unknown>[];
    if (statusPasses[0] === undefined) throw new Error("fixture lost device semantic pass");
    statusPasses[0]["status"] = "not-run";
    await expect(verifyCppCuteFrontendArtifact(status)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.semanticPasses[0].status",
    });

    const abi = await cloneCppCuteArtifactInput();
    const sourceAbi = (abi["payload"] as Record<string, unknown>)["sourceAbi"] as Record<string, unknown>;
    const abiTypes = sourceAbi["types"] as Record<string, unknown>[];
    if (abiTypes[0] === undefined) throw new Error("fixture lost device ABI");
    delete abiTypes[0]["domain"];
    await expect(verifyCppCuteFrontendArtifact(abi)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.sourceAbi.types[0]",
    });

    const target = await createCppCuteArtifactInput();
    (target.payload.semanticPasses[0] as { deviceArchitecture: string }).deviceArchitecture = "sm_90";
    expect(findCppCuteFrontendProfileBindingMismatch(target.payload, createCppCuteProfileInput())).toEqual({
      path: "$.artifact.semanticPasses[0]",
      message: "artifact semantic-pass domain or target differs from prepared profile",
    });
  });

  it("rejects function ABI facts attributed to the wrong CUDA domain", async () => {
    const valid = await cloneCppCuteArtifactInput();
    const payload = valid["payload"] as Record<string, unknown>;
    const types = payload["types"] as Record<string, unknown>[];
    const declarations = payload["declarations"] as Record<string, unknown>[];
    const intType = types.find((type) => type["kind"] === "builtin");
    if (intType === undefined) throw new Error("fixture lost integer type");
    const functionTypeId = stableId("type", "f");
    const functionDeclarationId = stableId("declaration", "f");
    types.push({
      typeId: functionTypeId,
      kind: "function",
      canonicalName: "int device_only()",
      qualifiers: structuredClone(intType["qualifiers"]),
      origin: structuredClone(intType["origin"]),
      returnTypeId: intType["typeId"],
      parameterTypeIds: [],
      variadic: false,
      callingConvention: "cuda-device",
    });
    declarations.push({
      declarationId: functionDeclarationId,
      kind: "function",
      canonicalUsr: "c:@F@device_only#",
      canonicalName: "device_only",
      lexicalParentId: null,
      semanticParentId: null,
      typeId: functionTypeId,
      targetTypeId: null,
      initializerExpressionId: null,
      origin: structuredClone(intType["origin"]),
      identitySpanId: CPP_CUTE_FIXTURE_SPAN_ID,
      definitionKind: "declaration-only",
      linkage: "external",
      storageDuration: "none",
      memorySpace: "generic",
      mangledName: "_Z11device_onlyv",
      cudaAttributes: { host: false, device: true, global: false, forceInline: false },
    });
    const sourceAbi = payload["sourceAbi"] as Record<string, unknown>;
    const intSourceAbi = (sourceAbi["types"] as Record<string, unknown>[])
      .find((entry) => entry["domain"] === "device" && entry["deviceTypeId"] === intType["typeId"]);
    if (intSourceAbi === undefined) throw new Error("fixture lost integer source ABI");
    const functionSourceEntityBody = {
      entityKind: "function" as const,
      canonicalIdentity: "c:@F@device_only#",
      origin: structuredClone(intType["origin"]),
      domains: ["device"] as const,
    };
    const functionSourceEntityId = await deriveCppCuteSourceEntityId(payload as never, functionSourceEntityBody as never);
    (payload["sourceEntities"] as unknown[]).push({
      sourceEntityId: functionSourceEntityId,
      ...functionSourceEntityBody,
    });
    (sourceAbi["functions"] as unknown[]).push({
      domain: "device",
      shared: false,
      sourceEntityId: functionSourceEntityId,
      deviceDeclarationId: functionDeclarationId,
      loweredCallingConvention: "nvptx-device",
      returnSourceTypeEntityId: intSourceAbi["sourceTypeEntityId"],
      returnPassing: "direct",
      parameters: [],
    });
    await rebindArtifactId(valid);
    await expect(verifyCppCuteFrontendArtifact(valid)).resolves.toMatchObject({ outcome: "accepted" });

    const wrongReturn = structuredClone(valid) as Record<string, unknown>;
    const wrongReturnPayload = wrongReturn["payload"] as Record<string, unknown>;
    const wrongReturnTypes = wrongReturnPayload["types"] as Record<string, unknown>[];
    const layoutType = wrongReturnTypes.find((type) => type["kind"] === "template-specialization");
    const wrongReturnSourceAbi = wrongReturnPayload["sourceAbi"] as Record<string, unknown>;
    if (layoutType === undefined) throw new Error("fixture lost return-type inputs");
    const layoutSourceBody = {
      entityKind: "type" as const,
      canonicalIdentity: String(layoutType["canonicalName"]),
      origin: structuredClone(layoutType["origin"]),
      domains: ["device"] as const,
    };
    const layoutSourceEntityId = await deriveCppCuteSourceEntityId(wrongReturnPayload as never, layoutSourceBody as never);
    (wrongReturnPayload["sourceEntities"] as unknown[]).push({ sourceEntityId: layoutSourceEntityId, ...layoutSourceBody });
    (wrongReturnSourceAbi["types"] as unknown[]).push({
      domain: "device",
      shared: false,
      sourceTypeEntityId: layoutSourceEntityId,
      deviceTypeId: layoutType["typeId"],
      sizeBits: "64",
      alignmentBits: "32",
      fields: [],
      bases: [],
    });
    const wrongFunction = (wrongReturnSourceAbi["functions"] as Record<string, unknown>[])[0];
    if (wrongFunction === undefined) throw new Error("fixture lost function ABI");
    wrongFunction["returnSourceTypeEntityId"] = layoutSourceEntityId;
    await expect(verifyCppCuteFrontendArtifact(wrongReturn)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.sourceAbi.functions[0].returnSourceTypeEntityId",
    });

    const wrongDomain = structuredClone(valid) as Record<string, unknown>;
    const wrongSourceAbi = (wrongDomain["payload"] as Record<string, unknown>)["sourceAbi"] as Record<string, unknown>;
    const functions = wrongSourceAbi["functions"] as Record<string, unknown>[];
    if (functions[0] === undefined) throw new Error("fixture lost function ABI");
    functions[0]["loweredCallingConvention"] = "c";
    await expect(verifyCppCuteFrontendArtifact(wrongDomain)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: "$.payload.sourceAbi.functions[0].loweredCallingConvention",
    });
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
    const semanticPasses = payload["semanticPasses"] as Record<string, unknown>[];
    for (const [index, pass] of semanticPasses.entries()) {
      pass["observedInputClosureSha256"] = await computeCppCuteSemanticPassInputClosureHash(payload as never, index);
    }
    await expect(verifyCppCuteFrontendArtifact(value)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      path: "$.payload.sourceEntities[0].sourceEntityId",
    });

    const sourceSet = await cloneCppCuteArtifactInput();
    const sourceSetInputs = (sourceSet["payload"] as Record<string, unknown>)["inputs"] as Record<string, unknown>;
    sourceSetInputs["sourceSetSha256"] = "b".repeat(64);
    await expect(verifyCppCuteFrontendArtifact(sourceSet)).rejects.toMatchObject({
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
    for (const [index, pass] of input.payload.semanticPasses.entries()) {
      (pass as { observedInputClosureSha256: string }).observedInputClosureSha256 =
        await computeCppCuteSemanticPassInputClosureHash(input.payload, index);
    }
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
    const compilerDiagnosticId = stableId("diagnostic", "f");
    diagnostics.push({
      diagnosticId: compilerDiagnosticId,
      phase: "artifact-extraction",
      severity: "warning",
      code: "browsergrad.cpp-cute:compiler-observation",
      renderedMessage: "Compiler observation has no source location.",
      location: { kind: "none" },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    });
    const devicePass = (compilerPayload["semanticPasses"] as Record<string, unknown>[])[0];
    if (devicePass === undefined) throw new Error("fixture lost device pass");
    devicePass["diagnosticIds"] = [
      ...(devicePass["diagnosticIds"] as string[]),
      compilerDiagnosticId,
    ].sort();
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

  it("requires exact nonempty declaration identity spans inside source origins", async () => {
    const missing = await cloneCppCuteArtifactInput();
    const missingDeclarations = (missing["payload"] as Record<string, unknown>)["declarations"] as Record<string, unknown>[];
    const missingVariable = missingDeclarations.find((declaration) => declaration["kind"] === "variable");
    if (missingVariable === undefined) throw new Error("fixture lost variable declaration");
    missingVariable["identitySpanId"] = null;
    await expect(verifyCppCuteFrontendArtifact(missing)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".identitySpanId"),
    });

    const empty = await cloneCppCuteArtifactInput();
    const emptyPayload = empty["payload"] as Record<string, unknown>;
    const emptyDeclarations = emptyPayload["declarations"] as Record<string, unknown>[];
    const emptyVariable = emptyDeclarations.find((declaration) => declaration["kind"] === "variable");
    const emptySpans = emptyPayload["spans"] as Record<string, unknown>[];
    if (emptyVariable === undefined) throw new Error("fixture lost variable declaration");
    const emptySpanId = stableId("span", "e");
    emptySpans.push({
      spanId: emptySpanId,
      spelling: { fileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID, startByte: "1", endByte: "1" },
      expansion: { fileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID, startByte: "1", endByte: "1" },
      macroExpansionId: null,
    });
    emptyVariable["identitySpanId"] = emptySpanId;
    await expect(verifyCppCuteFrontendArtifact(empty)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".identitySpanId"),
    });

    const outside = await cloneCppCuteArtifactInput();
    const outsidePayload = outside["payload"] as Record<string, unknown>;
    const outsideDeclarations = outsidePayload["declarations"] as Record<string, unknown>[];
    const outsideVariable = outsideDeclarations.find((declaration) => declaration["kind"] === "variable");
    const outsideSpans = outsidePayload["spans"] as Record<string, unknown>[];
    const inputs = outsidePayload["inputs"] as Record<string, unknown>;
    const files = inputs["files"] as Record<string, unknown>[];
    const header = files.find((file) => file["role"] === "dependency-header");
    if (outsideVariable === undefined || header === undefined) throw new Error("fixture lost identity-span inputs");
    const outsideSpanId = stableId("span", "f");
    outsideSpans.push({
      spanId: outsideSpanId,
      spelling: { fileId: header["fileId"], startByte: "0", endByte: "1" },
      expansion: { fileId: header["fileId"], startByte: "0", endByte: "1" },
      macroExpansionId: null,
    });
    outsideVariable["identitySpanId"] = outsideSpanId;
    await expect(verifyCppCuteFrontendArtifact(outside)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining(".identitySpanId"),
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
