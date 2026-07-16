import { canonicalJsonBytes, parseWireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_AOT_RECEIPT_SCHEMA,
  canonicalCppCuteAotRunnerReceiptBytes,
  decodeCppCuteAotRunnerReceipt,
  deriveCppCuteAotRunnerReceiptId,
  unwrapVerifiedCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceiptResource,
  verifyCppCuteAotRunnerReceipt,
  type CppCuteAotRunnerReceiptV2,
} from "../../src/cpp_cute_aot_receipt.js";
import { unwrapPreparedCppCuteAotFrontendProfile } from "../../src/cpp_cute_frontend_profile.js";
import type { CppCuteFrontendPayloadV2 } from "../../src/cpp_cute_frontend_types.js";
import {
  createCppCuteAotReceiptFixture,
  PINNED_CPP_CUTE_AOT_INVOCATION_ID,
  PINNED_CPP_CUTE_AOT_RECEIPT_BYTE_LENGTH,
  PINNED_CPP_CUTE_AOT_RECEIPT_BYTES_SHA256,
  PINNED_CPP_CUTE_AOT_RECEIPT_ID,
} from "./support/cpp_cute_aot_receipt_fixtures.js";
import {
  createCppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";

async function createFixture() {
  const base = await createCppCuteProvenanceFixture();
  return createCppCuteAotReceiptFixture(
    base.profile,
    base.executionEnvironment,
    base.artifact,
  );
}

describe("C++/CuTe AOT runner receipt", () => {
  it("verifies one content-addressed intent/observation/output record", async () => {
    const fixture = await createFixture();
    const verified = await verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      fixture.receipt,
    );
    const record = unwrapVerifiedCppCuteAotRunnerReceipt(verified);

    expect(verified).toMatchObject({
      receiptId: fixture.receipt.receiptId,
      jobId: fixture.job.jobId,
      profileHash: fixture.profile.profileHash,
      artifactId: fixture.artifact.artifactId,
      artifactHash: fixture.artifact.artifactHash,
      artifactBytesSha256: fixture.artifact.artifactBytesSha256,
      executionPlanSha256: fixture.receipt.invocation.executionPlanSha256,
      executionEnvironmentManifestSha256:
        fixture.receipt.invocation.executionEnvironmentManifestSha256,
    });
    expect(verified.receiptId).toMatch(/^bg\.cpp\.aot-receipt\.sha256\.[0-9a-f]{64}$/u);
    expect(verified.invocationId).toBe(
      `bg.cpp.aot-invocation.sha256.${verified.invocationManifestSha256}`,
    );
    expect({
      receiptId: verified.receiptId,
      receiptBytesSha256: verified.receiptBytesSha256,
      receiptByteLength: verified.receiptByteLength,
      invocationId: verified.invocationId,
    }).toEqual({
      receiptId: PINNED_CPP_CUTE_AOT_RECEIPT_ID,
      receiptBytesSha256: PINNED_CPP_CUTE_AOT_RECEIPT_BYTES_SHA256,
      receiptByteLength: PINNED_CPP_CUTE_AOT_RECEIPT_BYTE_LENGTH,
      invocationId: PINNED_CPP_CUTE_AOT_INVOCATION_ID,
    });
    expect(record.job).toBe(fixture.job);
    expect(record.profile).toBe(fixture.profile);
    expect(record.executionEnvironment).toBe(fixture.executionEnvironment);
    expect(record.artifactResource).toBe(fixture.artifactResource);
    expect(record.artifact).toEqual(fixture.artifact);
    expect(Object.isFrozen(verified)).toBe(true);
    expect(Object.isFrozen(record.receipt)).toBe(true);
    expect(record.receipt).not.toHaveProperty("command");
    expect(record.receipt).not.toHaveProperty("environment");
    expect(record.receipt).not.toHaveProperty("hostPath");
    expect(record.receipt.resources).toMatchObject({
      observedInputs: { accountingKind: "observed-exact" },
      processMeasurements: { accountingKind: "observed-exact" },
      emittedArtifact: { accountingKind: "emitted-artifact-exact" },
      enforcedCeilings: { accountingKind: "enforced-upper-bound" },
    });
    expect(record.receipt.resources.observedInputs.values).not.toHaveProperty("constexprSteps");
    expect(record.receipt.resources.emittedArtifact.values).not.toHaveProperty("constexprSteps");
    expect(record.receipt.resources.enforcedCeilings.values.maxConstexprSteps).toBe(
      String(unwrapPreparedCppCuteAotFrontendProfile(fixture.profile).profile.extractionLimits.maxConstexprSteps),
    );
  });

  it("mints byte-origin authority only from strict canonical receipt bytes", async () => {
    const fixture = await createFixture();
    const structural = await verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      fixture.receipt,
    );
    const bytes = canonicalCppCuteAotRunnerReceiptBytes(structural);
    const resource = await decodeCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      bytes,
    );
    const decoded = unwrapVerifiedCppCuteAotRunnerReceiptResource(resource);
    expect(resource).toMatchObject({
      receiptId: structural.receiptId,
      receiptBytesSha256: structural.receiptBytesSha256,
      receiptByteLength: String(bytes.byteLength),
    });
    expect(decoded).toBeDefined();
    expect(() => unwrapVerifiedCppCuteAotRunnerReceiptResource({ ...resource } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNVERIFIED" }),
    );

    const callerOwnedBytes = new Uint8Array(bytes);
    const pending = decodeCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      callerOwnedBytes,
    );
    callerOwnedBytes.fill(0);
    await expect(pending).resolves.toMatchObject({ receiptId: structural.receiptId });

    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as unknown;
    const noncanonical = new TextEncoder().encode(JSON.stringify(parsed, null, 2));
    await expect(decodeCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      noncanonical,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-NONCANONICAL-BYTES",
      path: "$bytes",
    });
  });

  it("requires strict artifact-byte authority and rejects instance substitution", async () => {
    const first = await createFixture();
    await expect(verifyCppCuteAotRunnerReceipt(
      first.job,
      first.executionEnvironment,
      first.artifact as never,
      first.receipt,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED" });

    const verified = await verifyCppCuteAotRunnerReceipt(
      first.job,
      first.executionEnvironment,
      first.artifactResource,
      first.receipt,
    );
    expect(() => unwrapVerifiedCppCuteAotRunnerReceipt({ ...verified } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNVERIFIED" }),
    );
  });

  it("binds the exact pre-run job, invocation, opened inputs, selection, and output", async () => {
    const fixture = await createFixture();
    const cases: Array<{
      readonly code: string;
      readonly path: string;
      readonly mutate: (receipt: Record<string, unknown>) => void;
    }> = [
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-JOB-MISMATCH",
        path: "$.jobId",
        mutate: (receipt) => { receipt["jobId"] = `bg.cpp.aot-job.sha256.${"0".repeat(64)}`; },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
        path: "$.invocation",
        mutate: (receipt) => {
          const invocation = receipt["invocation"] as Record<string, unknown>;
          const runner = invocation["runner"] as Record<string, unknown>;
          runner["binarySha256"] = "0".repeat(64);
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
        path: "$.invocation",
        mutate: (receipt) => {
          const invocation = receipt["invocation"] as Record<string, unknown>;
          const sandbox = invocation["sandbox"] as Record<string, unknown>;
          sandbox["policySha256"] = "0".repeat(64);
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
        path: "$.invocation",
        mutate: (receipt) => {
          const invocation = receipt["invocation"] as Record<string, unknown>;
          invocation["executionPlanSha256"] = "0".repeat(64);
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
        path: "$.invocation",
        mutate: (receipt) => {
          const invocation = receipt["invocation"] as Record<string, unknown>;
          invocation["executionEnvironmentManifestSha256"] = "f".repeat(64);
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
        path: "$.invocation",
        mutate: (receipt) => {
          const invocation = receipt["invocation"] as Record<string, unknown>;
          const container = invocation["container"] as Record<string, unknown>;
          container["configDigest"] = `sha256:${"f".repeat(64)}`;
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
        path: "$.openedInputs",
        mutate: (receipt) => {
          const inputs = receipt["openedInputs"] as Record<string, unknown>;
          inputs["sourceSetSha256"] = "0".repeat(64);
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
        path: "$.selection",
        mutate: (receipt) => {
          const selection = receipt["selection"] as Record<string, unknown>;
          selection["resolvedEntryId"] = `bg.cpp.entry.sha256.${"0".repeat(64)}`;
        },
      },
      {
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
        path: "$.output",
        mutate: (receipt) => {
          const output = receipt["output"] as Record<string, unknown>;
          output["artifactBytesSha256"] = "0".repeat(64);
        },
      },
    ];
    for (const testCase of cases) {
      const receipt = structuredClone(fixture.receipt) as unknown as Record<string, unknown>;
      testCase.mutate(receipt);
      await expect(verifyCppCuteAotRunnerReceipt(
        fixture.job,
        fixture.executionEnvironment,
        fixture.artifactResource,
        receipt,
      )).rejects.toMatchObject({ code: testCase.code, path: testCase.path });
    }
  });

  it("allows a successful process receipt for a structurally valid rejected artifact", async () => {
    const base = await createCppCuteProvenanceFixture({
      mutatePayload: (payload: CppCuteFrontendPayloadV2) => {
        const blockingDiagnosticId = `bg.cpp.diagnostic.sha256.${"1".repeat(64)}`;
        (payload.diagnostics as unknown as Array<unknown>).push({
          diagnosticId: blockingDiagnosticId,
          phase: "artifact-extraction",
          severity: "error",
          code: "browsergrad.cpp-cute:fixture-rejected",
          renderedMessage: "Fixture rejection for receipt coverage.",
          location: { kind: "none" },
          subject: { kind: "compiler" },
          parentDiagnosticId: null,
        });
        (payload.diagnostics as unknown as Array<{ diagnosticId: string }>).sort((left, right) =>
          left.diagnosticId.localeCompare(right.diagnosticId));
        const hostPass = payload.semanticPasses[1];
        if (hostPass === undefined) throw new Error("fixture lost host validation pass");
        (hostPass as { status: string }).status = "failed";
        (hostPass as { diagnosticIds: readonly string[] }).diagnosticIds = [blockingDiagnosticId];
        (payload as { outcome: unknown }).outcome = {
          kind: "rejected",
          blockingDiagnosticIds: [blockingDiagnosticId],
        };
      },
    });
    const fixture = await createCppCuteAotReceiptFixture(
      base.profile,
      base.executionEnvironment,
      base.artifact,
    );
    const verified = await verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      fixture.receipt,
    );
    expect(unwrapVerifiedCppCuteAotRunnerReceipt(verified).receipt.selection).toMatchObject({
      kind: "rejected",
      blockingDiagnosticIds: [`bg.cpp.diagnostic.sha256.${"1".repeat(64)}`],
    });
  });

  it("rejects producer-invented forced includes outside the exact profile option", async () => {
    await expect(createCppCuteProvenanceFixture({
      mutatePayload: (payload) => {
        const edge = payload.inputs.includeEdges.find((candidate) => candidate.kind === "compiler-forced");
        if (edge?.kind !== "compiler-forced") throw new Error("fixture lost compiler-forced include");
        (edge as { compilerOptionOrdinal: number }).compilerOptionOrdinal = 0;
      },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
      path: expect.stringContaining("includeEdges"),
    });
  });

  it("separates exact observations and emitted counts from enforced compiler-work ceilings", async () => {
    const fixture = await createFixture();
    const limits = unwrapPreparedCppCuteAotFrontendProfile(fixture.profile).profile.extractionLimits;
    const boundedGroups: Readonly<Record<string, Readonly<Record<string, number>>>> = {
      observedInputs: {
        openedSourceFiles: limits.maxSourceFiles,
        openedSourceBytes: limits.maxSourceBytes,
        openedHeaderFiles: limits.maxHeaderFiles,
        openedHeaderBytes: limits.maxHeaderBytes,
      },
      emittedArtifact: {
        macroExpansionFacts: limits.maxMacroExpansions,
        templateInstantiationFacts: limits.maxTemplateInstantiations,
        declarations: limits.maxDeclarations,
        types: limits.maxTypes,
        constants: limits.maxConstants,
        layoutFacts: limits.maxLayouts,
        tensorFacts: limits.maxTensors,
        operationFacts: limits.maxOperations,
        targetIntrinsicFacts: limits.maxTargetIntrinsics,
        diagnostics: limits.maxDiagnostics,
        outputBytes: limits.maxOutputBytes,
      },
      processMeasurements: {
        wallTimeMs: limits.maxWallTimeMs,
        cpuTimeMs: limits.maxCpuTimeMs,
        peakMemoryBytes: limits.maxMemoryBytes,
        peakProcesses: limits.maxProcesses,
      },
    };
    for (const [groupName, maximums] of Object.entries(boundedGroups)) {
      for (const [field, maximum] of Object.entries(maximums)) {
        const receipt = structuredClone(fixture.receipt) as unknown as Record<string, unknown>;
        const resources = receipt["resources"] as Record<string, unknown>;
        const group = resources[groupName] as Record<string, unknown>;
        const values = group["values"] as Record<string, unknown>;
        values[field] = parseWireU64(String(BigInt(maximum) + 1n));
        await expect(verifyCppCuteAotRunnerReceipt(
          fixture.job,
          fixture.executionEnvironment,
          fixture.artifactResource,
          receipt,
        )).rejects.toMatchObject({
          code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-RESOURCE-LIMIT",
          path: `$.resources.${groupName}.values.${field}`,
        });
      }
    }

    const counterDrift = structuredClone(fixture.receipt) as unknown as Record<string, unknown>;
    const resources = counterDrift["resources"] as Record<string, unknown>;
    const observed = resources["observedInputs"] as Record<string, unknown>;
    const observedValues = observed["values"] as Record<string, unknown>;
    observedValues["openedSourceFiles"] = "2";
    await expect(verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      counterDrift,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
      path: "$.resources.observedInputs.values",
    });

    const emittedDrift = structuredClone(fixture.receipt) as unknown as Record<string, unknown>;
    const emittedResources = emittedDrift["resources"] as Record<string, unknown>;
    const emitted = emittedResources["emittedArtifact"] as Record<string, unknown>;
    const emittedValues = emitted["values"] as Record<string, unknown>;
    emittedValues["declarations"] = "0";
    await expect(verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      emittedDrift,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
      path: "$.resources.emittedArtifact.values",
    });

    const configuredCeilings = fixture.receipt.resources.enforcedCeilings.values;
    for (const field of Object.keys(configuredCeilings)) {
      const ceilingDrift = structuredClone(fixture.receipt) as unknown as Record<string, unknown>;
      const ceilingResources = ceilingDrift["resources"] as Record<string, unknown>;
      const ceilings = ceilingResources["enforcedCeilings"] as Record<string, unknown>;
      const ceilingValues = ceilings["values"] as Record<string, unknown>;
      ceilingValues[field] = String(BigInt(ceilingValues[field] as string) - 1n);
      await expect(verifyCppCuteAotRunnerReceipt(
        fixture.job,
        fixture.executionEnvironment,
        fixture.artifactResource,
        ceilingDrift,
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
        path: "$.resources.enforcedCeilings.values",
      });
    }
  });

  it("rejects hostile accounting kinds, values, and category substitution", async () => {
    const fixture = await createFixture();
    const cases: Array<{
      readonly path: string;
      readonly mutate: (resources: Record<string, unknown>) => void;
    }> = [
      {
        path: "$.resources.observedInputs.accountingKind",
        mutate: (resources) => {
          const group = resources["observedInputs"] as Record<string, unknown>;
          group["accountingKind"] = "enforced-upper-bound";
        },
      },
      {
        path: "$.resources.emittedArtifact.accountingKind",
        mutate: (resources) => {
          const group = resources["emittedArtifact"] as Record<string, unknown>;
          group["accountingKind"] = "observed-exact";
        },
      },
      {
        path: "$.resources.enforcedCeilings.accountingKind",
        mutate: (resources) => {
          const group = resources["enforcedCeilings"] as Record<string, unknown>;
          group["accountingKind"] = "emitted-artifact-exact";
        },
      },
      {
        path: "$",
        mutate: (resources) => {
          const group = resources["processMeasurements"] as Record<string, unknown>;
          const values = group["values"] as Record<string, unknown>;
          values["peakProcesses"] = "-1";
        },
      },
      {
        path: "$.resources.observedInputs.values",
        mutate: (resources) => {
          const group = resources["observedInputs"] as Record<string, unknown>;
          const values = group["values"] as Record<string, unknown>;
          values["maxConstexprSteps"] = values["openedSourceFiles"];
        },
      },
    ];
    for (const testCase of cases) {
      const receipt = structuredClone(fixture.receipt) as unknown as Record<string, unknown>;
      testCase.mutate(receipt["resources"] as Record<string, unknown>);
      await expect(verifyCppCuteAotRunnerReceipt(
        fixture.job,
        fixture.executionEnvironment,
        fixture.artifactResource,
        receipt,
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID",
        path: testCase.path,
      });
    }
  });

  it("rejects receipt hash/version/outcome drift, operational fields, cancellation, and hostile inputs", async () => {
    const fixture = await createFixture();
    const hashDrift = structuredClone(fixture.receipt) as CppCuteAotRunnerReceiptV2;
    (hashDrift as { receiptId: string }).receiptId = `bg.cpp.aot-receipt.sha256.${"0".repeat(64)}`;
    await expect(verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      hashDrift,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-HASH-MISMATCH" });

    await expect(verifyCppCuteAotRunnerReceipt(fixture.job, fixture.executionEnvironment, fixture.artifactResource, {
      ...fixture.receipt,
      version: { major: 1, minor: 1 },
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNSUPPORTED-VERSION" });
    await expect(verifyCppCuteAotRunnerReceipt(fixture.job, fixture.executionEnvironment, fixture.artifactResource, {
      ...fixture.receipt,
      outcome: "failed",
      exitCode: 1,
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID" });
    await expect(verifyCppCuteAotRunnerReceipt(fixture.job, fixture.executionEnvironment, fixture.artifactResource, {
      ...fixture.receipt,
      command: ["clang++"],
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID" });

    const controller = new AbortController();
    controller.abort();
    await expect(verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      fixture.receipt,
      { signal: controller.signal },
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-CANCELLED" });

    const hostile = {};
    Object.defineProperty(hostile, "schema", { enumerable: true, get: () => CPP_CUTE_AOT_RECEIPT_SCHEMA });
    await expect(verifyCppCuteAotRunnerReceipt(
      fixture.job,
      fixture.executionEnvironment,
      fixture.artifactResource,
      hostile,
    )).rejects.toThrow();
    expect(canonicalJsonBytes(fixture.receipt).byteLength).toBeGreaterThan(0);
    expect(await deriveCppCuteAotRunnerReceiptId(fixture.receipt)).toBe(fixture.receipt.receiptId);
  });
});
