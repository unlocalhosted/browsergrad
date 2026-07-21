import { layoutArtifactPayload } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { logicalGemmTileArtifactPayload } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { beforeAll, describe, expect, it } from "vitest";
import { prepareCppCuteAotOfflineRun } from "../../src/cpp_cute_aot_runner_plan.js";
import {
  deriveCppCuteFrontendArtifactId,
  unwrapVerifiedCppCuteFrontendArtifact,
  verifyCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import type { AuthorizedCppCuteFrontendArtifact } from "../../src/cpp_cute_frontend_authorization.js";
import {
  lowerAuthorizedCppCuteLogicalGemmTileEntry,
  prepareVerifiedCppCuteLogicalGemmTileSemantics,
} from "../../src/cpp_cute_logical_gemm_lowering.js";
import type {
  CppCuteFrontendPayloadV3,
  CppCuteLogicalGemmTileFactV1,
  CppCuteTensorFactV1,
} from "../../src/cpp_cute_frontend_types.js";
import {
  CPP_CUTE_LOGICAL_GEMM_FACT_ID,
  CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID,
  createCppCuteLogicalGemmArtifactInput,
  mutateCppCutePayloadToLogicalGemm,
} from "./support/cpp_cute_logical_gemm_fixtures.js";
import {
  createAuthorizedCppCuteProvenanceFixture,
  type AuthorizedCppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";

let fixture: AuthorizedCppCuteProvenanceFixture;

beforeAll(async () => {
  fixture = await createAuthorizedCppCuteProvenanceFixture({
    mutatePayload: mutateCppCutePayloadToLogicalGemm,
  });
});

function selectedEntryId(): string {
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected logical GEMM entry");
  }
  return payload.outcome.selectedEntryIds[0];
}

async function lower(authorization = fixture.authorization) {
  return lowerAuthorizedCppCuteLogicalGemmTileEntry(authorization, { entryId: selectedEntryId() });
}

function mutablePayload(input: Record<string, unknown>): CppCuteFrontendPayloadV3 {
  return input["payload"] as CppCuteFrontendPayloadV3;
}

function mutableLogicalFact(payload: CppCuteFrontendPayloadV3): CppCuteLogicalGemmTileFactV1 {
  const fact = payload.facts.find((candidate) => candidate.factId === CPP_CUTE_LOGICAL_GEMM_FACT_ID);
  if (fact?.kind !== "logical-gemm-tile") throw new Error("fixture lost logical GEMM fact");
  return fact;
}

function mutableTensor(payload: CppCuteFrontendPayloadV3, factId: string): CppCuteTensorFactV1 {
  const fact = payload.facts.find((candidate) => candidate.factId === factId);
  if (fact?.kind !== "tensor") throw new Error(`fixture lost tensor ${factId}`);
  return fact;
}

describe("authorized C++/CuTe typed-artifact logical GEMM lowering", () => {
  it("lowers one exact authorized 3.1 fact into canonical backend-neutral semantics", async () => {
    const frontend = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope;
    const artifacts = await lower();
    const layout = layoutArtifactPayload(artifacts.layout);
    const kernel = logicalGemmTileArtifactPayload(artifacts.kernel);

    expect(frontend.version).toEqual({ major: 3, minor: 1 });
    expect(frontend.payload.entries).toEqual([expect.objectContaining({
      kind: "logical-gemm-tile",
      logicalGemmTileFactId: CPP_CUTE_LOGICAL_GEMM_FACT_ID,
    })]);
    expect(layout.views.map((view) => view.shape.map((extent) => (
      extent.kind === "const" ? extent.value : "dynamic"
    )))).toEqual([
      ["17", "23"],
      ["23", "19"],
      ["17", "19"],
    ]);
    expect(kernel.operation).toMatchObject({
      kind: "logical-gemm-tile",
      logicalTile: { m: "16", n: "16", k: "16" },
      boundary: {
        lhs: "zero-fill",
        rhs: "zero-fill",
        destination: "mask-outside-logical-shape",
      },
      accumulation: {
        inputDType: "f32",
        accumulatorDType: "f32",
        outputDType: "f32",
        reductionOrder: "increasing-k",
        rounding: "toward-nearest-ties-even",
        contraction: "forbid",
        reassociation: "forbid",
      },
      overlap: { kind: "forbid-all" },
    });
    expect(kernel.operation).not.toHaveProperty("schedule");
    expect(kernel.operation).not.toHaveProperty("backend");
    expect(kernel.operation).not.toHaveProperty("mma");
  });

  it("prepares structural meaning without minting authority or claiming body parity/native extraction", async () => {
    const prepared = await prepareVerifiedCppCuteLogicalGemmTileSemantics(
      fixture.artifact,
      { entryId: selectedEntryId() },
    );
    expect(prepared).toMatchObject({
      entry: { kind: "logical-gemm-tile" },
      fact: { kind: "logical-gemm-tile" },
      m: "17",
      n: "19",
      k: "23",
      loweringAuthorityMinted: false,
      nativeExtractorEvidenceClaimed: false,
      nativeFpControlEvidenceClaimed: false,
      sourceBodyParityClaimed: false,
      backendExecutionAuthorized: false,
    });
    expect(prepared).not.toHaveProperty("authorization");
  });

  it("requires the opaque authorization for the exact artifact", async () => {
    await expect(lower({ ...fixture.authorization } as AuthorizedCppCuteFrontendArtifact))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AUTHORIZATION-UNVERIFIED" });
  });

  it("does not let the production AOT extractor turn request acceptance into source compatibility", async () => {
    await expect(prepareCppCuteAotOfflineRun(fixture.metadata, fixture.executionEnvironment))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-RUNNER-UNSUPPORTED-ENTRY",
        path: "$.request.entryRequests[0].kind",
      });
  });

  it("uses a deliberate 3.1 wire transition and rejects logical facts in 3.0", async () => {
    const input = await createCppCuteLogicalGemmArtifactInput();
    (input["version"] as { minor: number }).minor = 0;
    await expect(verifyCppCuteFrontendArtifact(input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNSUPPORTED-VERSION",
      path: "$.version.minor",
    });
    await expect(deriveCppCuteFrontendArtifactId(
      mutablePayload(await createCppCuteLogicalGemmArtifactInput()),
      { minor: 2 as never },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNSUPPORTED-VERSION",
      path: "$.options.minor",
    });
  });

  it("keeps facts closed against schedule, backend, target-MMA, and weakened numerical claims", async () => {
    const unknownFields = ["schedule", "backend", "mma"];
    for (const field of unknownFields) {
      const input = await createCppCuteLogicalGemmArtifactInput();
      (mutableLogicalFact(mutablePayload(input)) as unknown as Record<string, unknown>)[field] = {};
      await expect(verifyCppCuteFrontendArtifact(input)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      });
    }
    const weakened = await createCppCuteLogicalGemmArtifactInput();
    (mutableLogicalFact(mutablePayload(weakened)).accumulation as { contraction: string }).contraction = "allow";
    await expect(verifyCppCuteFrontendArtifact(weakened)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("accumulation.contraction"),
    });
  });

  it("rejects pointer aliasing, nullable inputs, non-dense layouts, zero tiles, and extra target facts", async () => {
    const alias = await createCppCuteLogicalGemmArtifactInput();
    const aliasPayload = mutablePayload(alias);
    const lhs = mutableTensor(aliasPayload, mutableLogicalFact(aliasPayload).lhsTensorFactId);
    const rhs = mutableTensor(aliasPayload, CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID);
    if (lhs.engine.kind !== "global-pointer" || rhs.engine.kind !== "global-pointer") {
      throw new Error("fixture lost pointer engines");
    }
    (rhs.engine as { pointerDeclarationId: string }).pointerDeclarationId = lhs.engine.pointerDeclarationId;
    await expect(verifyCppCuteFrontendArtifact(alias)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
    });

    const nullable = await createCppCuteLogicalGemmArtifactInput();
    const nullableRhs = mutableTensor(mutablePayload(nullable), CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID);
    if (nullableRhs.engine.kind !== "global-pointer") throw new Error("fixture lost rhs pointer");
    (nullableRhs.engine as { nullable: boolean }).nullable = true;
    await expect(verifyCppCuteFrontendArtifact(nullable)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
    });

    const strided = await createCppCuteLogicalGemmArtifactInput();
    const stridedPayload = mutablePayload(strided);
    const rhsTensor = mutableTensor(stridedPayload, CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID);
    const rhsLayout = stridedPayload.facts.find((candidate) => candidate.factId === rhsTensor.layoutFactId);
    if (rhsLayout?.kind !== "affine-layout" || rhsLayout.stride.kind !== "tuple" ||
        rhsLayout.stride.elements[0]?.kind !== "scalar") throw new Error("fixture lost rhs layout");
    (rhsLayout.stride.elements[0].value as { value: string }).value = "20";
    (rhsLayout as { cosize: unknown }).cosize = { kind: "integer", value: "459" };
    await expect(verifyCppCuteFrontendArtifact(strided)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
    });

    const zeroTile = await createCppCuteLogicalGemmArtifactInput();
    (mutableLogicalFact(mutablePayload(zeroTile)).logicalTile as { m: string }).m = "0";
    await expect(verifyCppCuteFrontendArtifact(zeroTile)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("logicalTile.m"),
    });

    const extraTarget = await createCppCuteLogicalGemmArtifactInput();
    const extraPayload = mutablePayload(extraTarget);
    const targetFactId = `bg.cpp.fact.sha256.${"e".repeat(64)}`;
    (extraPayload.facts as CppCuteFrontendPayloadV3["facts"] & Array<Record<string, unknown>>).push({
      factId: targetFactId,
      kind: "target-intrinsic",
      origin: structuredClone(mutableLogicalFact(extraPayload).origin),
      familyId: "nvidia:mma@1",
      operation: {
        kind: "mma",
        m: 16,
        n: 8,
        k: 16,
        aTypeId: mutableTensor(extraPayload, CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID).elementTypeId,
        bTypeId: mutableTensor(extraPayload, CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID).elementTypeId,
        accumulatorTypeId: mutableTensor(extraPayload, CPP_CUTE_LOGICAL_GEMM_RHS_TENSOR_FACT_ID).elementTypeId,
      },
      operandExpressionIds: [],
      resultTypeId: null,
      effects: { readsMemory: true, writesMemory: true, synchronizes: false, convergent: true },
      availability: { kind: "portable-candidate" },
    });
    const devicePass = extraPayload.semanticPasses.find((pass) => pass.domain === "device");
    if (devicePass === undefined) throw new Error("fixture lost device pass");
    (devicePass.factIds as string[]).push(targetFactId);
    (devicePass.factIds as string[]).sort();
    await expect(verifyCppCuteFrontendArtifact(extraTarget)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      path: expect.stringContaining("logicalGemmTileFactId"),
    });
  });

  it("fails closed on cancellation, open requests, hostile options, and decode budgets", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(lowerAuthorizedCppCuteLogicalGemmTileEntry(
      fixture.authorization,
      { entryId: selectedEntryId() },
      { signal: controller.signal },
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-CANCELLED" });

    await expect(lowerAuthorizedCppCuteLogicalGemmTileEntry(
      fixture.authorization,
      { entryId: selectedEntryId(), extra: true } as never,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INVALID-REQUEST" });

    const hostile: Record<string, unknown> = {};
    Object.defineProperty(hostile, "limits", { enumerable: true, get: () => ({ maxNodes: 10 }) });
    await expect(lowerAuthorizedCppCuteLogicalGemmTileEntry(
      fixture.authorization,
      { entryId: selectedEntryId() },
      hostile,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-INVALID-REQUEST" });

    await expect(lowerAuthorizedCppCuteLogicalGemmTileEntry(
      fixture.authorization,
      { entryId: selectedEntryId() },
      { limits: { maxNodes: 4 } },
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-RESOURCE-LIMIT" });
  });
});
