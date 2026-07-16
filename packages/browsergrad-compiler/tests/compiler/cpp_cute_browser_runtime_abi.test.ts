import {
  canonicalJsonBytes,
  decodeWireJson,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_BYTE_LIMIT,
  CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  canonicalCppCuteBrowserRuntimeAbiManifestBytes,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
  deriveCppCuteBrowserRuntimeAbiManifestId,
  deriveCppCuteBrowserGeneratedImportAllowlistSha256,
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
  type PreparedCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";

type MutableJson = Record<string, unknown>;

function mutableResource(): MutableJson {
  return JSON.parse(new TextDecoder().decode(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  )) as MutableJson;
}

function body(resource: MutableJson): MutableJson {
  return resource.body as MutableJson;
}

function objectField(value: MutableJson, name: string): MutableJson {
  return value[name] as MutableJson;
}

function arrayField(value: MutableJson, name: string): MutableJson[] {
  return value[name] as MutableJson[];
}

function canonicalBytes(value: MutableJson): Uint8Array {
  return canonicalJsonBytes(value as JsonValue, {
    limits: CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS,
  });
}

async function expectDecodeError(
  value: Uint8Array,
  code: string,
  path?: string,
): Promise<void> {
  await expect(decodeCppCuteBrowserRuntimeAbiManifest(value)).rejects.toMatchObject({
    code,
    ...(path === undefined ? {} : { path: expect.stringContaining(path) }),
  });
}

describe("browser Clang-WASM runtime ABI manifest", () => {
  it("strict-decodes one stable design authority without claiming observation or release", async () => {
    const resource = cppCuteBrowserRuntimeAbiManifestResourceBytes();
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(resource);

    expect(prepared).toEqual({
      manifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
      runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
      resourceSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      contractSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
      generatedImportAllowlistSha256:
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256,
      resourceByteLength: 13_255,
      designAuthority: true,
      interfaceReviewReady: false,
      observedWasmVerified: false,
      releaseReady: false,
    });
    expect(canonicalCppCuteBrowserRuntimeAbiManifestBytes(prepared)).toEqual(resource);
    const record = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared);
    expect(await deriveCppCuteBrowserRuntimeAbiManifestId(record.manifest.body)).toBe(
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    );
    expect(await deriveCppCuteBrowserGeneratedImportAllowlistSha256(record.manifest.body)).toBe(
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256,
    );
    expect(record.manifest.body.authority).toEqual({
      kind: "design-contract-only",
      observedWasm: "detached-verification-required",
      workerExecution: "not-authorized",
      releaseReadiness: "not-authorized",
    });
  });

  it("pins the eight exported C signatures and no generic execution surface", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const { body: manifest } = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest;

    expect(manifest.wasm.moduleRole).toBe(
      "compiler-extractor-only-user-programs-never-linked-or-executed",
    );
    expect(manifest.wasm.unlistedCExports).toBe("forbidden");
    expect(manifest.cExports.map((entry) => [
      entry.ordinal,
      entry.cSymbol,
      entry.cSignature,
      entry.wasmParameters,
      entry.wasmResults,
    ])).toEqual([
      [0, "bg_cpp_cute_abi_version", "uint32_t bg_cpp_cute_abi_version(void)", [], ["i32"]],
      [1, "bg_cpp_cute_alloc", "uint32_t bg_cpp_cute_alloc(uint32_t byte_length)", ["i32"], ["i32"]],
      [2, "bg_cpp_cute_compile", "int32_t bg_cpp_cute_compile(uint32_t input_pointer, uint32_t input_length)", ["i32", "i32"], ["i32"]],
      [3, "bg_cpp_cute_free", "void bg_cpp_cute_free(uint32_t pointer, uint32_t byte_length)", ["i32", "i32"], []],
      [4, "bg_cpp_cute_reset", "void bg_cpp_cute_reset(void)", [], []],
      [5, "bg_cpp_cute_result_length", "uint32_t bg_cpp_cute_result_length(void)", [], ["i32"]],
      [6, "bg_cpp_cute_result_pointer", "uint32_t bg_cpp_cute_result_pointer(void)", [], ["i32"]],
      [7, "bg_cpp_cute_status", "int32_t bg_cpp_cute_status(void)", [], ["i32"]],
    ]);
  });

  it("pins one unshared wasm32 memory and reserves every coexisting byte budget", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const memory = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body.wasm.memory;

    expect(memory).toMatchObject({
      count: 1,
      addressType: "i32",
      imported: false,
      exported: true,
      sharing: "unshared",
      initialPages: 4_096,
      maximumPages: 16_384,
      growthLinearStepPages: 1_024,
      stackByteLength: 16_777_216,
      maxCompilerWorkingByteLength: 536_870_912,
      maxAggregateOpenedVfsByteLength: 402_653_184,
      maxInputFrameByteLength: 4_194_304,
      maxResultByteLength: 8_388_608,
    });
    const reserved = memory.stackByteLength + memory.maxCompilerWorkingByteLength +
      memory.maxAggregateOpenedVfsByteLength + memory.maxInputFrameByteLength +
      memory.maxResultByteLength;
    expect(reserved).toBeLessThanOrEqual(memory.maximumPages * memory.pageByteLength);
  });

  it("closes the Wasm baseline and rejects every unlisted extension", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const wasm = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body.wasm;

    expect(wasm.featurePolicy).toEqual({
      instructionSetBaseline: "webassembly-mvp",
      allowedExtensions: wasm.requiredFeatures,
      unlistedExtensions: "forbidden",
      staticOpcodeAndSectionInspection: "required",
      targetFeaturesCrossCheck: "required-but-not-authoritative",
    });
  });

  it("pins synchronous lazy-VFS calls, binary records, and fail-closed semantics", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const manifest = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body;

    expect(manifest.hostImports).toMatchObject({
      moduleName: "browsergrad_vfs_v1",
      invocation: "synchronous-non-reentrant",
      pointerLifetime: "only-for-import-call-duration",
      unlistedApplicationImports: "forbidden",
      memoryAccess: {
        rangeArithmetic: "checked-u32-no-wrap",
        completeRangeValidation: "required-before-first-input-read-or-output-write",
        pathInputSnapshot: "after-complete-range-validation-before-any-output-write",
        inputOutputOverlap: "allowed-only-after-complete-input-snapshot",
        outputOutputOverlap: "forbidden",
        memoryGrowthDuringImport: "forbidden",
        invalidRangeMemoryMutation: "forbidden",
        alignmentByteLength: {
          byteOutput: 1,
          u32Output: 4,
          u64ContainingRecord: 8,
        },
      },
      generatedImportAllowlist: {
        policyId: "browsergrad.compiler.cpp-cute.emscripten-generated-imports@1",
        status: "unresolved-first-build-review-required",
        allowlistSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256,
        exactFunctions: [],
        unlistedGeneratedImports: "forbidden",
        observedModuleCannotExtendAllowlist: true,
        capabilityCeiling: "no-clock-random-network-process-or-ambient-filesystem",
        releaseConformance: "forbidden-until-independent-review-and-manifest-repin",
      },
    });
    expect(manifest.hostImports.functions.map((entry) => [entry.fieldName, entry.wasmParameters.length]))
      .toEqual([
        ["bg_vfs_status", 3],
        ["bg_vfs_open", 3],
        ["bg_vfs_read", 5],
        ["bg_vfs_close", 1],
        ["bg_vfs_directory_count", 3],
        ["bg_vfs_directory_entry", 6],
      ]);
    expect(manifest.vfs).toMatchObject({
      storage: "host-backed-lazy-verified-pack-files-only",
      physicalFilesystemFallback: "forbidden",
      networkFallback: "forbidden",
      pathEncoding: "utf8",
      maxPathByteLength: 4_096,
      maxLiveFileHandles: 65_536,
      maxSessionCalls: 1_000_000,
      directoryOrder: "strict-ascending-utf8-byte-order",
      failureAtomicity: "nonzero-status-writes-no-output-except-required-name-length-in-metadata",
    });
    expect(manifest.vfs.metadataRecord.byteLength).toBe(32);
    expect(manifest.vfs.openResultRecord.byteLength).toBe(16);
    expect(manifest.vfs.statuses.map((entry) => entry.code)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });

  it("keeps generated imports and support exports empty until independent first-build review", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const wasm = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body.wasm;
    const generated = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared)
      .manifest.body.hostImports.generatedImportAllowlist;

    expect(generated).toMatchObject({
      status: "unresolved-first-build-review-required",
      exactFunctions: [],
      unlistedGeneratedImports: "forbidden",
      observedModuleCannotExtendAllowlist: true,
      releaseConformance: "forbidden-until-independent-review-and-manifest-repin",
    });
    expect(wasm.supportExports).toEqual({
      status: "unresolved-first-build-review-required",
      exactFunctionAllowlist: [],
      exactGlobalAllowlist: [],
      exactTableAllowlist: [],
      unlistedExports: "forbidden",
      observedModuleCannotExtendAllowlist: true,
      releaseConformance: "forbidden-until-independent-review-and-manifest-repin",
    });
  });

  it("bounds unresolved table, global, tag, and custom-section projections", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const policy = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared)
      .manifest.body.wasm.structuralPolicy;

    expect(policy).toMatchObject({
      status: "unresolved-first-build-review-required",
      releaseConformance: "forbidden-until-exact-first-build-projection-is-reviewed-and-repinned",
      tables: {
        maximumCount: 1,
        allowedElementTypes: ["funcref"],
        imported: "forbidden",
        declaredMaximumRequired: true,
        maximumElementsCeiling: 65_536,
        exactReviewedProjection: [],
      },
      globals: {
        maximumCount: 4_096,
        allowedValueTypes: ["f32", "f64", "i32", "i64"],
        imported: "forbidden",
        exactReviewedExports: [],
      },
      tags: { exactCount: 0, imported: "forbidden", exported: "forbidden" },
      customSections: {
        maximumCount: 4,
        maximumSectionByteLength: 524_288,
        maximumTotalByteLength: 1_048_576,
        duplicateNames: "forbidden",
        exactReviewedNameAllowlist: [],
        unlistedNames: "forbidden",
        explicitlyForbiddenNames: ["dylink.0", "producers", "sourceMappingURL"],
        targetFeatures: {
          sectionName: "target_features",
          status: "unresolved-first-build-review-required",
          requiredDeclarations: [
            "bulk-memory", "mutable-globals", "nontrapping-fptoint", "sign-ext",
          ],
          forbiddenDeclarations: [
            "atomics", "exception-handling", "memory64", "multimemory", "simd128",
          ],
          exactRawSectionProjection: [],
        },
      },
    });
  });

  it("pins the exact input frame and keeps source bytes under the Worker VFS session", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const frame = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body.inputFrame;

    expect(frame).toMatchObject({
      magicAscii: "BGCCABI1",
      major: 1,
      minor: 0,
      headerByteLength: 64,
      alignmentByteLength: 8,
      maxFrameByteLength: 4_194_304,
      profileRegion: "exact-canonical-prepared-frontend-profile-json",
      requestRegion: "exact-canonical-producer-neutral-frontend-request-json",
      sourceBytes: "out-of-band-through-worker-owned-vfs-session",
      compileReadRule: "synchronous-complete-frame-validation-before-vfs-access",
    });
    expect(frame.fields.map((entry) => [entry.name, entry.offset])).toEqual([
      ["magic", 0],
      ["major", 8],
      ["minor", 10],
      ["headerByteLength", 12],
      ["totalByteLength", 16],
      ["flags", 20],
      ["profileOffset", 24],
      ["profileByteLength", 28],
      ["requestOffset", 32],
      ["requestByteLength", 36],
      ["reserved", 40],
    ]);
  });

  it("closes alloc, compile, free, result, status, reset, and failure transitions", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const manifest = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body;
    const rules = new Set<string>(manifest.lifecycle.rules);

    expect(manifest.lifecycle.initialState).toBe("idle");
    expect(manifest.lifecycle.states).toEqual([
      "idle", "input-allocated", "compiling-internal", "artifact-ready", "failed",
    ]);
    for (const rule of [
      "alloc-is-valid-only-in-idle-with-byte-length-from-1-through-max-input-frame-byte-length",
      "only-one-live-input-allocation-is-permitted",
      "compile-is-valid-only-for-the-exact-live-input-pointer-and-length",
      "compile-is-synchronous-and-not-reentrant",
      "free-is-valid-only-for-the-exact-live-input-pointer-and-length-and-never-for-result-memory",
      "free-after-success-preserves-the-artifact-ready-result",
      "result-getters-return-zero-unless-state-is-artifact-ready",
      "result-bytes-are-immutable-until-reset",
      "reset-releases-input-result-and-module-side-vfs-state-and-returns-to-idle",
      "infrastructure-failure-never-masquerades-as-a-rejected-source-artifact",
      "wasm-trap-abort-or-out-of-memory-is-a-worker-infrastructure-failure-with-no-readable-status-guarantee",
    ]) {
      expect(rules.has(rule), rule).toBe(true);
    }
    expect(manifest.compileStatuses).toEqual([
      { code: 0, name: "artifact-ready", retry: "reset-then-new-invocation" },
      { code: 1, name: "idle", retry: "allocate-input" },
      { code: 2, name: "input-allocated", retry: "compile-or-free" },
      { code: 100, name: "invalid-state", retry: "reset-required" },
      { code: 101, name: "invalid-argument", retry: "reset-required" },
      { code: 102, name: "invalid-frame", retry: "reset-required" },
      { code: 103, name: "abi-mismatch", retry: "module-must-not-be-reused" },
      { code: 104, name: "vfs-error", retry: "reset-required" },
      { code: 105, name: "resource-limit", retry: "reset-required" },
      { code: 106, name: "internal-error", retry: "module-must-not-be-reused" },
    ]);
    expect(manifest.result).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: { major: 3, minor: 0 },
      encoding: "canonical-json-bytes",
      maximumByteLength: 8_388_608,
      ownership: "module-owned-worker-must-copy-before-reset",
      lifetime: "from-artifact-ready-until-reset-or-worker-termination",
    });
  });

  it("makes cancellation external, destructive, and non-cooperative", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    expect(unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body.cancellation).toEqual({
      mechanism: "terminate-dedicated-worker",
      cooperativeImport: "forbidden",
      reason: "synchronous-unshared-wasm-cannot-service-worker-messages-during-compile",
      effect: "invalidates-module-memory-input-result-vfs-handles-and-session-authority",
      workerReuseAfterCancellation: "forbidden",
    });
  });

  it("requires exact canonical bytes and a self-consistent manifest hash", async () => {
    const resource = cppCuteBrowserRuntimeAbiManifestResourceBytes();
    const noncanonical = new Uint8Array(resource.byteLength + 1);
    noncanonical.set(resource);
    noncanonical[resource.byteLength] = 0x0a;
    await expectDecodeError(
      noncanonical,
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-NONCANONICAL-BYTES",
      "$bytes",
    );

    const wrongId = mutableResource();
    wrongId.manifestId = `bg.cpp.browser-runtime-abi.sha256.${"0".repeat(64)}`;
    await expectDecodeError(
      canonicalBytes(wrongId),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-HASH-MISMATCH",
      "manifestId",
    );
  });

  it.each([
    ["extra top-level field", (value: MutableJson) => { value.extra = true; }, "$"],
    ["observed-Wasm claim", (value: MutableJson) => {
      objectField(body(value), "authority").observedWasm = "verified";
    }, "authority"],
    ["second memory", (value: MutableJson) => {
      objectField(objectField(body(value), "wasm"), "memory").count = 2;
    }, "memory"],
    ["shared memory", (value: MutableJson) => {
      objectField(objectField(body(value), "wasm"), "memory").sharing = "shared";
    }, "memory"],
    ["unlisted Wasm extension permitted", (value: MutableJson) => {
      const policy = objectField(objectField(body(value), "wasm"), "featurePolicy");
      policy.unlistedExtensions = "allowed";
    }, "featurePolicy"],
    ["ninth C export", (value: MutableJson) => {
      const exports = arrayField(body(value), "cExports");
      const copied = structuredClone(exports[0]);
      exports.push({ ...copied, ordinal: 8, cSymbol: "bg_escape" });
    }, "cExports"],
    ["changed C signature", (value: MutableJson) => {
      arrayField(body(value), "cExports")[2]!.wasmParameters = ["i32"];
    }, "cExports[2]"],
    ["async VFS", (value: MutableJson) => {
      objectField(body(value), "hostImports").invocation = "asynchronous";
    }, "hostImports"],
    ["wrapping VFS pointer arithmetic", (value: MutableJson) => {
      const access = objectField(objectField(body(value), "hostImports"), "memoryAccess");
      access.rangeArithmetic = "wrapping-u32";
    }, "memoryAccess"],
    ["overlapping VFS outputs", (value: MutableJson) => {
      const access = objectField(objectField(body(value), "hostImports"), "memoryAccess");
      access.outputOutputOverlap = "allowed";
    }, "memoryAccess"],
    ["unbounded live VFS handles", (value: MutableJson) => {
      objectField(body(value), "vfs").maxLiveFileHandles = Number.MAX_SAFE_INTEGER;
    }, "vfs"],
    ["unbounded VFS calls", (value: MutableJson) => {
      objectField(body(value), "vfs").maxSessionCalls = Number.MAX_SAFE_INTEGER;
    }, "vfs"],
    ["self-authorized generated import", (value: MutableJson) => {
      const generated = objectField(objectField(body(value), "hostImports"), "generatedImportAllowlist");
      (generated.exactFunctions as unknown[]).push({
        moduleName: "env", fieldName: "attacker", wasmParameters: [], wasmResults: [],
      });
    }, "generatedImportAllowlist"],
    ["self-authorized support export", (value: MutableJson) => {
      const wasm = objectField(body(value), "wasm");
      const support = objectField(wasm, "supportExports");
      (support.exactFunctionAllowlist as unknown[]).push("attacker");
    }, "supportExports"],
    ["unbounded table", (value: MutableJson) => {
      const structural = objectField(objectField(body(value), "wasm"), "structuralPolicy");
      objectField(structural, "tables").declaredMaximumRequired = false;
    }, "tables"],
    ["observed target_features self-approval", (value: MutableJson) => {
      const structural = objectField(objectField(body(value), "wasm"), "structuralPolicy");
      const custom = objectField(structural, "customSections");
      const target = objectField(custom, "targetFeatures");
      (target.exactRawSectionProjection as unknown[]).push("+bulk-memory");
    }, "targetFeatures"],
    ["extra VFS parameter", (value: MutableJson) => {
      arrayField(objectField(body(value), "hostImports"), "functions")[0]!.wasmParameters = [
        "i32", "i32", "i32", "i32",
      ];
    }, "hostImports.functions[0]"],
    ["frame magic", (value: MutableJson) => {
      objectField(body(value), "inputFrame").magicAscii = "EVILABI1";
    }, "inputFrame"],
    ["result version", (value: MutableJson) => {
      objectField(objectField(body(value), "result"), "version").major = 4;
    }, "result"],
    ["cooperative cancellation", (value: MutableJson) => {
      objectField(body(value), "cancellation").cooperativeImport = "allowed";
    }, "cancellation"],
  ])("rejects hostile contract mutation: %s", async (_name, mutate, path) => {
    const resource = mutableResource();
    mutate(resource);
    await expectDecodeError(
      canonicalBytes(resource),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID",
      path,
    );
  });

  it("rejects unsupported versions before accepting a closed contract", async () => {
    for (const [field, value] of [["major", 2], ["minor", 1]] as const) {
      const resource = mutableResource();
      objectField(resource, "version")[field] = value;
      await expectDecodeError(
        canonicalBytes(resource),
        "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-UNSUPPORTED-VERSION",
        field,
      );
    }
  });

  it("rejects oversized, shared, subclassed, proxied, and malformed bytes", async () => {
    await expectDecodeError(
      new Uint8Array(CPP_CUTE_BROWSER_RUNTIME_ABI_BYTE_LIMIT + 1),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-RESOURCE-LIMIT",
      "$bytes",
    );
    await expectDecodeError(
      new Uint8Array(new SharedArrayBuffer(16)),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID",
      "$bytes",
    );
    class DerivedBytes extends Uint8Array {}
    await expectDecodeError(
      new DerivedBytes(cppCuteBrowserRuntimeAbiManifestResourceBytes()),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID",
      "$bytes",
    );
    await expectDecodeError(
      new Proxy(cppCuteBrowserRuntimeAbiManifestResourceBytes(), {}),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID",
      "$bytes",
    );
    await expectDecodeError(
      new TextEncoder().encode("{\"schema\":"),
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID",
      "$bytes",
    );
  });

  it("honors cancellation and rejects hostile option surfaces", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-CANCELLED",
      path: "$options.signal",
    });

    const options = Object.defineProperty({}, "signal", {
      enumerable: true,
      get: () => controller.signal,
    });
    await expect(decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
      options,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID",
      path: "$options.signal",
    });
  });

  it("keeps prepared authorities opaque and returned bytes isolated", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const copied = { ...prepared } as PreparedCppCuteBrowserRuntimeAbiManifest;
    expect(() => unwrapPreparedCppCuteBrowserRuntimeAbiManifest(copied)).toThrow(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-UNVERIFIED",
    }));

    const first = canonicalCppCuteBrowserRuntimeAbiManifestBytes(prepared);
    first.fill(0);
    const second = canonicalCppCuteBrowserRuntimeAbiManifestBytes(prepared);
    expect(second).toEqual(cppCuteBrowserRuntimeAbiManifestResourceBytes());
    expect(second).not.toEqual(first);
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(Object.isFrozen(unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest)).toBe(true);
  });

  it("uses the dedicated bounded decoder rather than workspace-wide defaults", () => {
    expect(CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS).toEqual({
      maxDocumentBytes: 65_536,
      maxDepth: 16,
      maxNodes: 4_096,
      maxStringBytes: 49_152,
      maxArrayLength: 128,
      maxObjectProperties: 32,
      maxRank: 1,
      maxIntegerBits: 64,
      maxArithmeticOperations: 8_192,
    });
    const decoded = decodeWireJson(cppCuteBrowserRuntimeAbiManifestResourceBytes(), {
      limits: CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS,
    });
    expect((decoded as JsonObject).schema).toBe(
      "browsergrad.compiler.cpp-cute.browser-runtime-abi-manifest",
    );
  });
});
