import {
  canonicalJsonBytes,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_BYTE_LIMIT,
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_DECODE_LIMITS,
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_MANIFEST_ID,
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  canonicalCppCuteDiagnosticNormalizationBytes,
  cppCuteClangDiagnosticCode,
  cppCuteDiagnosticNormalizationCategoryRule,
  cppCuteDiagnosticNormalizationResourceBytes,
  cppCuteDiagnosticNormalizationSeverityRule,
  cppCuteDiagnosticNormalizationStageRule,
  decodeCppCuteDiagnosticNormalization,
  deriveCppCuteNormalizedDiagnosticId,
  deriveCppCuteDiagnosticNormalizationManifestId,
  unwrapPreparedCppCuteDiagnosticNormalization,
  type CppCuteSerializedDiagnosticIdMaterial,
  type PreparedCppCuteDiagnosticNormalization,
} from "../../src/cpp_cute_diagnostic_normalization.js";

type MutableJson = Record<string, unknown>;

function mutableResource(): MutableJson {
  return JSON.parse(new TextDecoder().decode(
    cppCuteDiagnosticNormalizationResourceBytes(),
  )) as MutableJson;
}

function objectField(value: MutableJson, name: string): MutableJson {
  return value[name] as MutableJson;
}

function arrayField(value: MutableJson, name: string): MutableJson[] {
  return value[name] as MutableJson[];
}

function canonicalBytes(value: MutableJson): Uint8Array {
  return canonicalJsonBytes(value as JsonValue, {
    limits: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_DECODE_LIMITS,
  });
}

async function expectDecodeError(
  value: Uint8Array,
  code: string,
  path?: string,
): Promise<void> {
  await expect(decodeCppCuteDiagnosticNormalization(value)).rejects.toMatchObject({
    code,
    ...(path === undefined ? {} : { path }),
  });
}

describe("C++/CuTe diagnostic-normalization authority", () => {
  it("strict-decodes one pinned Clang 22.1.8 design authority without execution claims", async () => {
    const resource = cppCuteDiagnosticNormalizationResourceBytes();
    const prepared = await decodeCppCuteDiagnosticNormalization(resource);

    expect(prepared).toEqual({
      manifestId: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_MANIFEST_ID,
      policyId: "browsergrad.compiler.cpp-cute.clang-diagnostic-normalization@1",
      clangVersion: "22.1.8",
      resourceSha256: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
      resourceByteLength: 8_439,
      designAuthority: true,
      clangInvocationAuthorized: false,
      diagnosticNormalizationPerformed: false,
      artifactProductionAuthorized: false,
    });
    expect(await sha256Hex(resource)).toBe(
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
    );
    expect(canonicalCppCuteDiagnosticNormalizationBytes(prepared)).toEqual(resource);

    const normalization = unwrapPreparedCppCuteDiagnosticNormalization(prepared).normalization;
    expect(await deriveCppCuteDiagnosticNormalizationManifestId(normalization.body)).toBe(
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_MANIFEST_ID,
    );
    expect(normalization.body.clang).toEqual({ compilerId: "clang", version: "22.1.8" });
    expect(normalization.body.artifactBinding).toEqual({
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      major: 3,
      diagnosticShape: "browsergrad.compiler.cpp-cute.frontend-diagnostic@3",
      fixItsRepresented: false,
    });
  });

  it("closes producer stages and every Clang severity disposition", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const body = unwrapPreparedCppCuteDiagnosticNormalization(prepared).normalization.body;

    expect(body.stageMappings).toEqual([
      { producerStage: "preprocessor", artifactPhase: "preprocessing" },
      { producerStage: "parser", artifactPhase: "parsing" },
      { producerStage: "sema-name-lookup", artifactPhase: "name-lookup" },
      { producerStage: "sema-overload-resolution", artifactPhase: "overload-resolution" },
      { producerStage: "sema-template-instantiation", artifactPhase: "template-instantiation" },
      { producerStage: "sema-cuda", artifactPhase: "cuda-sema" },
      { producerStage: "artifact-extractor", artifactPhase: "artifact-extraction" },
    ]);
    expect(body.severityMappings.map((entry) => [
      entry.clangLevel,
      entry.disposition,
      entry.artifactSeverity,
      entry.blocking,
      entry.parentRequired,
    ])).toEqual([
      ["ignored", "omit-before-normalization", null, false, false],
      ["remark", "emit", "remark", false, false],
      ["note", "emit", "note", false, true],
      ["warning", "emit", "warning", false, false],
      ["error", "emit", "error", true, false],
      ["fatal", "emit", "fatal", true, false],
    ]);
    expect(cppCuteDiagnosticNormalizationStageRule(prepared, "sema-cuda")).toEqual({
      producerStage: "sema-cuda",
      artifactPhase: "cuda-sema",
    });
    expect(cppCuteDiagnosticNormalizationSeverityRule(prepared, "note")).toMatchObject({
      artifactSeverity: "note",
      blocking: false,
      parentRequired: true,
    });
  });

  it("pins stage-default categories, raw-ID Clang codes, and explicit custom diagnostics", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const body = unwrapPreparedCppCuteDiagnosticNormalization(prepared).normalization.body;

    expect(body.classification).toMatchObject({
      rawDiagnosticIdEncoding: "unsigned-decimal-no-leading-zeroes",
      clangDiagnosticCodeFormat: "clang:diag-<raw-diagnostic-id>",
      categorySource: "producer-stage-default-with-closed-custom-overrides",
      messageInference: "forbidden",
      diagnosticGroupInference: "forbidden",
    });
    expect(body.categoryMappings.map((entry) => [entry.category, entry.codeStrategy])).toEqual([
      ["preprocessing", "clang-raw-diagnostic-id"],
      ["parsing", "clang-raw-diagnostic-id"],
      ["name-lookup", "clang-raw-diagnostic-id"],
      ["overload-resolution", "clang-raw-diagnostic-id"],
      ["template-instantiation", "clang-raw-diagnostic-id"],
      ["cuda-sema", "clang-raw-diagnostic-id"],
      ["artifact-extraction", "closed-browsergrad-custom-code"],
      ["policy", "closed-browsergrad-custom-code"],
      ["resource-limit", "closed-browsergrad-custom-code"],
    ]);
    expect(cppCuteDiagnosticNormalizationCategoryRule(prepared, "preprocessing")).toEqual({
      category: "preprocessing",
      permittedProducerStages: ["preprocessor"],
      codeStrategy: "clang-raw-diagnostic-id",
    });
    expect(cppCuteClangDiagnosticCode(prepared, 0)).toBe("clang:diag-0");
    expect(cppCuteClangDiagnosticCode(prepared, 4_294_967_295)).toBe(
      "clang:diag-4294967295",
    );
    for (const invalidId of [-1, 1.5, 4_294_967_296, Number.NaN]) {
      expect(() => cppCuteClangDiagnosticCode(prepared, invalidId)).toThrow(
        expect.objectContaining({
          code: "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
          path: "$rawDiagnosticId",
        }),
      );
    }
    expect(body.customMappings).toEqual([
      {
        customCode: "browsergrad.cpp-cute:temporal-macro-forbidden",
        producerStage: "preprocessor",
        category: "policy",
        artifactSeverity: "error",
        blocking: true,
      },
      {
        customCode: "browsergrad.cpp-cute:temporal-macro-mutation-forbidden",
        producerStage: "preprocessor",
        category: "policy",
        artifactSeverity: "error",
        blocking: true,
      },
      {
        customCode: "browsergrad.cpp-cute:diagnostic-resource-limit",
        producerStage: "artifact-extractor",
        category: "resource-limit",
        artifactSeverity: "fatal",
        blocking: true,
      },
      {
        customCode: "browsergrad.cpp-cute:diagnostic-normalization-failed",
        producerStage: "artifact-extractor",
        category: "artifact-extraction",
        artifactSeverity: "fatal",
        blocking: true,
      },
    ]);
  });

  it("freezes deterministic diagnostic-ID inputs without host or runtime entropy", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const stable = unwrapPreparedCppCuteDiagnosticNormalization(prepared)
      .normalization.body.stableDiagnosticId;

    expect(stable).toEqual({
      algorithm: "sha256-canonical-json",
      domain: "browsergrad.compiler.cpp-cute.frontend-diagnostic-id.v1",
      outputPrefix: "bg.cpp.diagnostic.sha256.",
      projectionSchema: "browsergrad.compiler.cpp-cute.frontend-diagnostic-id-input@1",
      canonicalProjectionKeys: [
        "domain",
        "compilationContractHash",
        "ownerPassId",
        "diagnostic",
      ],
      compilationContractHashSource: "payload.compilationContractHash",
      ownerPassIdSource: "semantic-pass-diagnosticIds-membership.passId",
      serializedDiagnosticFields: [
        "phase",
        "severity",
        "code",
        "renderedMessage",
        "location",
        "subject",
        "parentDiagnosticId",
      ],
      parentDependency: "serialized-parentDiagnosticId-root-first",
      recomputationOrder: "root-diagnostics-then-direct-note-children",
      duplicatePolicy: "collapse-exact-canonical-projection-within-owner-pass",
      duplicateScope: "compilation-contract-and-owner-pass",
      forbiddenInputs: [
        "hostPath",
        "presumedPath",
        "wallClock",
        "pointerAddress",
        "threadId",
        "renderedColor",
        "fixIts",
        "rawClangDiagnosticId",
        "producerStage",
        "category",
        "emissionOrdinal",
        "sourceManagerAddress",
      ],
    });
  });

  it("recomputes IDs from Artifact V3 evidence and collapses exact normalized duplicates", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const diagnostic: CppCuteSerializedDiagnosticIdMaterial = {
      phase: "parsing",
      severity: "error",
      code: "clang:diag-1234",
      renderedMessage: "expected expression",
      location: {
        kind: "source",
        primarySpanId: `bg.cpp.span.sha256.${"1".repeat(64)}`,
        related: [],
      },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    };
    const input = {
      compilationContractHash: "2".repeat(64),
      ownerPassId: "cuda-device-sema" as const,
      diagnostic,
    };

    const first = await deriveCppCuteNormalizedDiagnosticId(prepared, input);
    const exactDuplicate = await deriveCppCuteNormalizedDiagnosticId(prepared, {
      ...input,
      diagnostic: {
        ...diagnostic,
        location: { ...diagnostic.location },
        subject: { ...diagnostic.subject },
      },
    });
    const withProducerOnlyMetadata = await deriveCppCuteNormalizedDiagnosticId(prepared, {
      ...input,
      diagnostic: {
        ...diagnostic,
        rawClangDiagnosticId: 1_234,
        producerStage: "parser",
        emissionOrdinal: 99,
      } as CppCuteSerializedDiagnosticIdMaterial,
    });

    expect(first).toMatch(/^bg\.cpp\.diagnostic\.sha256\.[0-9a-f]{64}$/u);
    expect(exactDuplicate).toBe(first);
    expect(withProducerOnlyMetadata).toBe(first);
    await expect(deriveCppCuteNormalizedDiagnosticId(prepared, {
      ...input,
      ownerPassId: "cuda-host-sema",
    })).resolves.not.toBe(first);
    await expect(deriveCppCuteNormalizedDiagnosticId(prepared, {
      ...input,
      compilationContractHash: "3".repeat(64),
    })).resolves.not.toBe(first);

    for (const invalidHash of [
      `bg.cpp.contract.sha256.${"2".repeat(64)}`,
      "2".repeat(63),
      "G".repeat(64),
    ]) {
      await expect(deriveCppCuteNormalizedDiagnosticId(prepared, {
        ...input,
        compilationContractHash: invalidHash,
      })).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
        path: "$diagnosticId.compilationContractHash",
      });
    }
    await expect(deriveCppCuteNormalizedDiagnosticId(prepared, {
      ...input,
      ownerPassId: "forged-pass" as "cuda-device-sema",
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      path: "$diagnosticId.ownerPassId",
    });
    await expect(deriveCppCuteNormalizedDiagnosticId(prepared, {
      ...input,
      diagnostic: { ...diagnostic, renderedMessage: "expected identifier" },
    })).resolves.not.toBe(first);
  });

  it("uses physical VFS byte ranges for source and macro identity", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const body = unwrapPreparedCppCuteDiagnosticNormalization(prepared).normalization.body;

    expect(body.sourceLocationPolicy).toEqual({
      coordinateUnit: "utf8-byte-offset",
      rangeConvention: "half-open",
      pointConvention: "zero-width-half-open-range",
      pathIdentity: "physical-opened-vfs-file-only",
      presumedLocationUse: "rendering-only-never-identity",
      macroPolicy: "retain-physical-spelling-and-outermost-expansion",
      nonMacroPolicy: "spelling-and-expansion-must-match",
      crossFileRangePolicy: "reject-normalization",
      invalidLocationPolicy: "location-none-only-for-compiler-subject",
      maxRelatedLocations: 32,
      maxRenderedBytesPerRelatedLocation: 2_048,
      maxAggregateRenderedRelatedLocationBytes: 16_384,
    });
    expect(body.pathNormalizationPolicy).toMatchObject({
      grammar: "canonical-posix-absolute-v1",
      requireOpenedFileMembership: true,
      rejectBackslash: true,
      rejectEmptyDotAndDotDotSegments: true,
      resolveSymlinks: false,
      useCurrentWorkingDirectory: false,
      useHostFilesystemFallback: false,
      lineDirectiveFilenameAuthority: false,
    });
  });

  it("bounds messages, child notes, and omitted Artifact V3 fix-its without truncation", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const body = unwrapPreparedCppCuteDiagnosticNormalization(prepared).normalization.body;

    expect(body.renderedTextPolicy).toMatchObject({
      encoding: "utf-8",
      lineEndings: "lf",
      maxRenderedMessageBytes: 4_096,
      overflowDisposition: "emit-resource-limit-and-stop-pass-without-truncation",
    });
    expect(body.notePolicy).toEqual({
      representation: "child-diagnostic-with-parent-diagnostic-id",
      ordering: "clang-emission-order",
      parentPolicy: "direct-child-of-emitted-root-only",
      normalizationOrder: "root-before-direct-note-children",
      maxNotesPerPrimaryDiagnostic: 16,
      maxRenderedBytesPerNote: 2_048,
      maxAggregateRenderedNoteBytesPerPrimaryDiagnostic: 16_384,
      orphanDisposition: "emit-normalization-failed-and-stop-pass",
      overflowDisposition: "emit-resource-limit-and-stop-pass-without-truncation",
    });
    expect(body.fixItPolicy).toMatchObject({
      artifactV3Disposition: "bounded-validate-then-omit",
      maxFixItsPerDiagnostic: 16,
      maxReplacementBytesPerFixIt: 4_096,
      maxAggregateReplacementBytesPerDiagnostic: 16_384,
      requireOpenedFileMembership: true,
      requireSingleFileHalfOpenRange: true,
      stableDiagnosticIdContribution: false,
    });
  });

  it("fails closed for unknown structure while preserving bounded unknown Clang diagnostics", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const policy = unwrapPreparedCppCuteDiagnosticNormalization(prepared)
      .normalization.body.unknownDiagnosticPolicy;

    expect(policy).toEqual({
      unknownRawId: "emit-versioned-raw-id-code-at-active-stage",
      malformedRawId: "emit-normalization-failed-and-stop-pass",
      unknownCategory: "use-producer-stage-default-category",
      unknownCustomCode: "emit-normalization-failed-and-stop-pass",
      unknownProducerStage: "emit-normalization-failed-and-stop-pass",
      unknownClangLevel: "emit-normalization-failed-and-stop-pass",
      preserveEffectiveSeverity: true,
      preserveBoundedRenderedMessage: true,
      rawIdIdentityContribution: "serialized-code-only",
      silentlyDrop: false,
    });
    expect(policy.unknownRawId).toContain("raw-id-code");
    expect(policy.unknownCategory).toContain("producer-stage-default");
  });

  it("rejects noncanonical, duplicate-key, trailing, malformed, and oversized bytes", async () => {
    const resource = cppCuteDiagnosticNormalizationResourceBytes();
    await expectDecodeError(
      new TextEncoder().encode(JSON.stringify(mutableResource(), null, 2)),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-NONCANONICAL-BYTES",
      "$bytes",
    );

    const text = new TextDecoder().decode(resource);
    await expectDecodeError(
      new TextEncoder().encode(text.replace("{\"body\":", "{\"body\":null,\"body\":")),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$bytes",
    );

    const trailing = new Uint8Array(resource.byteLength + 1);
    trailing.set(resource);
    trailing[resource.byteLength] = 0x20;
    await expectDecodeError(
      trailing,
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-NONCANONICAL-BYTES",
      "$bytes",
    );
    await expectDecodeError(
      new Uint8Array([0xff]),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$bytes",
    );
    await expectDecodeError(
      new Uint8Array(CPP_CUTE_DIAGNOSTIC_NORMALIZATION_BYTE_LIMIT + 1),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-RESOURCE-LIMIT",
      "$bytes",
    );
  });

  it("rejects unknown fields, version or Clang drift, and reordered closed mappings", async () => {
    const unknown = mutableResource();
    objectField(unknown, "body").unknown = true;
    await expectDecodeError(
      canonicalBytes(unknown),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$.body",
    );

    const version = mutableResource();
    objectField(version, "version").minor = 1;
    await expectDecodeError(
      canonicalBytes(version),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-UNSUPPORTED-VERSION",
      "$.version.minor",
    );

    const clang = mutableResource();
    objectField(objectField(clang, "body"), "clang").version = "22.1.7";
    await expectDecodeError(
      canonicalBytes(clang),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$.body.clang.version",
    );

    const reordered = mutableResource();
    arrayField(objectField(reordered, "body"), "stageMappings").reverse();
    await expectDecodeError(
      canonicalBytes(reordered),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$.body.stageMappings[0].producerStage",
    );
  });

  it("rejects manifest identity drift before it can mint authority", async () => {
    const value = mutableResource();
    value.manifestId = `bg.cpp.diagnostic-normalization.sha256.${"0".repeat(64)}`;
    await expectDecodeError(
      canonicalBytes(value),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-HASH-MISMATCH",
      "$.manifestId",
    );
  });

  it("snapshots bytes and rejects hostile byte containers", async () => {
    const bytes = cppCuteDiagnosticNormalizationResourceBytes();
    const pending = decodeCppCuteDiagnosticNormalization(bytes);
    bytes.fill(0);
    await expect(pending).resolves.toMatchObject({ designAuthority: true });

    class ByteSubclass extends Uint8Array {}
    await expectDecodeError(
      new ByteSubclass(cppCuteDiagnosticNormalizationResourceBytes()),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$bytes",
    );
    await expectDecodeError(
      new Proxy(cppCuteDiagnosticNormalizationResourceBytes(), {}),
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      "$bytes",
    );
    if (typeof SharedArrayBuffer !== "undefined") {
      await expectDecodeError(
        new Uint8Array(new SharedArrayBuffer(8)),
        "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
        "$bytes",
      );
    }
  });

  it("honors cancellation and rejects accessor-bearing options without invoking them", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-CANCELLED",
      path: "$options.signal",
    });

    let getterCalls = 0;
    const hostile = Object.defineProperty({}, "signal", {
      enumerable: true,
      get() {
        getterCalls += 1;
        throw new Error("must not run");
      },
    });
    await expect(decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
      hostile,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID",
      path: "$options.signal",
    });
    expect(getterCalls).toBe(0);
  });

  it("keeps prepared authority opaque, immutable, and realm-local", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    const record = unwrapPreparedCppCuteDiagnosticNormalization(prepared);
    expect(() => {
      (record.normalization.body.clang as { version: string }).version = "forged";
    }).toThrow(TypeError);

    const copy = canonicalCppCuteDiagnosticNormalizationBytes(prepared);
    copy.fill(0);
    expect(canonicalCppCuteDiagnosticNormalizationBytes(prepared)[0]).not.toBe(0);

    expect(() => unwrapPreparedCppCuteDiagnosticNormalization({
      ...prepared,
    } as PreparedCppCuteDiagnosticNormalization)).toThrow(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-UNVERIFIED",
      path: "$prepared",
    }));
  });
});
