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
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256,
  canonicalCppCuteBrowserRuntimeAbiManifestBytes,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
  deriveCppCuteBrowserRuntimeAbiManifestId,
  deriveCppCuteBrowserGeneratedImportAllowlistSha256,
  deriveCppCuteBrowserSupportFunctionAllowlistSha256,
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
      supportFunctionAllowlistSha256:
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256,
      resourceByteLength: 33_423,
      designAuthority: true,
      interfaceReviewReady: false,
      observedWasmVerified: false,
      releaseReady: false,
    });
    expect(canonicalCppCuteBrowserRuntimeAbiManifestBytes(prepared)).toEqual(resource);
    const record = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared);
    expect(record.manifest.version).toEqual({ major: 1, minor: 5 });
    expect(record.manifest.body.wasm.cAbiVersion).toBe(65_537);
    expect(await deriveCppCuteBrowserRuntimeAbiManifestId(record.manifest.body)).toBe(
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    );
    expect(await deriveCppCuteBrowserGeneratedImportAllowlistSha256(record.manifest.body)).toBe(
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256,
    );
    expect(await deriveCppCuteBrowserSupportFunctionAllowlistSha256(record.manifest.body)).toBe(
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256,
    );
    expect(record.manifest.body.authority).toEqual({
      kind: "design-contract-only",
      observedWasm: "detached-verification-required",
      workerExecution: "not-authorized",
      releaseReadiness: "not-authorized",
    });
  });

  it("pins the nine exported C signatures and no generic execution surface", async () => {
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
      [2, "bg_cpp_cute_allocator_metrics_pointer", "uint32_t bg_cpp_cute_allocator_metrics_pointer(void)", [], ["i32"]],
      [3, "bg_cpp_cute_compile", "int32_t bg_cpp_cute_compile(uint32_t input_pointer, uint32_t input_length)", ["i32", "i32"], ["i32"]],
      [4, "bg_cpp_cute_free", "void bg_cpp_cute_free(uint32_t pointer, uint32_t byte_length)", ["i32", "i32"], []],
      [5, "bg_cpp_cute_reset", "void bg_cpp_cute_reset(void)", [], []],
      [6, "bg_cpp_cute_result_length", "uint32_t bg_cpp_cute_result_length(void)", [], ["i32"]],
      [7, "bg_cpp_cute_result_pointer", "uint32_t bg_cpp_cute_result_pointer(void)", [], ["i32"]],
      [8, "bg_cpp_cute_status", "int32_t bg_cpp_cute_status(void)", [], ["i32"]],
    ]);
  });

  it("pins one nonzero stable allocator-metrics pointer and exact v1 record", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const contract = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared)
      .manifest.body.allocatorMetricsRecord;

    expect(contract).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.allocator-metrics-record",
      version: { major: 1, minor: 0 },
      magicAscii: "BGRTMET1",
      byteLength: 72,
      alignmentByteLength: 8,
      pointerExport: "bg_cpp_cute_allocator_metrics_pointer",
      pointerContract: {
        zero: "forbidden-for-conforming-live-module-instance",
        stability: "constant-for-module-instance-lifetime",
        mutability: "module-writes-host-read-only",
      },
      accounting: {
        requestedByteBasis:
          "caller-requested-byte-length-before-alignment-or-allocator-rounding",
        interception: {
          exactEntrypoints: [
            "aligned_alloc", "calloc", "free", "__libc_calloc", "__libc_free",
            "__libc_malloc", "__libc_realloc", "malloc", "memalign", "posix_memalign",
            "pvalloc", "realloc", "reallocarray", "valloc",
          ],
          forbiddenEntrypoints: [
            "bulk_free", "independent_calloc", "independent_comalloc", "realloc_in_place",
          ],
          underlyingBypassEntrypoints: [
            "emscripten_builtin_calloc", "emscripten_builtin_free",
            "emscripten_builtin_malloc", "emscripten_builtin_memalign",
            "emscripten_builtin_realloc",
          ],
          directBypassReferences:
            "forbidden-outside-BrowserGradCppCuteMetrics.cpp",
          linkClosureProof:
            "pinned-object-and-final-wasm-call-graph-evidence-required",
        },
        failedInvalidRequestSemantics:
          "invalid-or-size-overflowing-nonzero-request-increments-failed-once-zero-request-does-not",
        zeroByteCreationSemantics:
          "nonnull-result-counts-one-tracked-zero-byte-allocation-null-result-is-permitted-no-op-neither-success-nor-failure",
        freeNullSemantics: "no-op-with-no-counter-change",
        reallocNullPointerSemantics: "same-as-creation-at-requested-size",
        reallocNonzeroSuccessSemantics:
          "allocated-adds-new-requested-size-freed-adds-old-requested-size-success-and-free-counts-each-increment-once-even-in-place",
        reallocNonzeroFailureSemantics:
          "failed-count-increments-once-and-all-live-byte-and-success-free-counters-remain-unchanged",
        reallocZeroSizeSemantics:
          "nonnull-old-pointer-is-released-freed-adds-old-requested-size-free-count-increments-once-result-is-null-and-failure-count-does-not-change",
      },
      authority: {
        workerExecution: "not-authorized-by-record-values",
        lowering: "not-authorized-by-record-values",
      },
    });
    expect(contract.fields.map((field) => [
      field.ordinal,
      field.name,
      field.offset,
      field.byteLength,
      field.encoding,
    ])).toEqual([
      [0, "magic", 0, 8, "ascii[8]"],
      [1, "version", 8, 4, "u32le"],
      [2, "byteLength", 12, 4, "u32le"],
      [3, "currentLiveGlobalRequestedByteLength", 16, 8, "u64le"],
      [4, "peakLiveGlobalRequestedByteLength", 24, 8, "u64le"],
      [5, "cumulativeGlobalAllocatedRequestedByteLength", 32, 8, "u64le"],
      [6, "cumulativeGlobalFreedRequestedByteLength", 40, 8, "u64le"],
      [7, "successfulAllocationCount", 48, 8, "u64le"],
      [8, "freeCount", 56, 8, "u64le"],
      [9, "failedAllocationCount", 64, 8, "u64le"],
    ]);
  });

  it("pins one unshared wasm32 memory and reserves every coexisting linear-memory budget", async () => {
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
      maxInputFrameByteLength: 4_194_304,
      maxResultByteLength: 8_388_608,
    });
    const reserved = memory.stackByteLength + memory.maxCompilerWorkingByteLength +
      memory.maxInputFrameByteLength + memory.maxResultByteLength;
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
      targetFeaturesCrossCheck: "optional-advisory-when-present",
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
        status: "independently-reviewed-hash-pinned",
        allowlistSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256,
        independentReview: {
          basis: "pinned-emscripten-runtime-sources-and-locked-link-flags",
          emscriptenVersion: "6.0.3",
          emscriptenCommit: "283e2d130132859fde6a4e4c87fd254b38127651",
        },
        unlistedGeneratedImports: "forbidden",
        observedModuleCannotExtendAllowlist: true,
        capabilityCeiling: "no-clock-random-network-process-or-ambient-filesystem",
        releaseConformance: "allowed-only-for-exact-hash-pinned-signatures",
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
      maxIndexedNodes: 262_144,
      maxIndexLogicalByteLength: 134_217_728,
      indexLogicalByteAccounting:
        "sum-per-node-metadata-record-plus-canonical-path-utf8-plus-immediate-basename-utf8",
      maxAggregateLiveOpenByteLength: 402_653_184,
      liveOpenByteAccounting:
        "logical-full-file-per-live-handle-reservation-not-wasm-residency",
      maxLiveFileHandles: 65_536,
      maxSessionCalls: 1_000_000,
      directoryOrder: "strict-ascending-utf8-byte-order",
      failureAtomicity: "nonzero-status-writes-no-output-except-required-name-length-in-metadata",
    });
    expect(manifest.vfs.metadataRecord.byteLength).toBe(32);
    expect(manifest.vfs.openResultRecord.byteLength).toBe(16);
    expect(manifest.vfs.statuses.map((entry) => entry.code)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });

  it("pins reviewed generated imports and worker-internal support functions", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const wasm = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared).manifest.body.wasm;
    const generated = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared)
      .manifest.body.hostImports.generatedImportAllowlist;

    expect(generated).toMatchObject({
      status: "independently-reviewed-hash-pinned",
      unlistedGeneratedImports: "forbidden",
      observedModuleCannotExtendAllowlist: true,
      releaseConformance: "allowed-only-for-exact-hash-pinned-signatures",
    });
    expect(generated.exactFunctions).toHaveLength(52);
    expect(generated.independentReview.runtimeRoles).toEqual([
      { name: "javascript-exception-control-flow", exactFunctionCount: 48, ambientCapability: "none" },
      { name: "bounded-memory-growth", exactFunctionCount: 2, ambientCapability: "none" },
      { name: "stack-overflow-trap", exactFunctionCount: 1, ambientCapability: "none" },
      { name: "stdout-stderr-only", exactFunctionCount: 1, ambientCapability: "caller-provided-output-hooks-only" },
    ]);
    expect(generated.exactFunctions.filter((entry) => entry.runtimeRole === "stdout-stderr-only"))
      .toEqual([expect.objectContaining({
        moduleName: "wasi_snapshot_preview1",
        fieldName: "fd_write",
        wasmParameters: ["i32", "i32", "i32", "i32"],
        wasmResults: ["i32"],
      })]);
    expect(wasm.supportExports).toMatchObject({
      status: "independently-reviewed-hash-pinned",
      functionAllowlistSha256:
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256,
      exactGlobalAllowlist: [],
      exactTableAllowlist: [{
        name: "__indirect_function_table",
        index: 0,
        runtimeRole: "javascript-exception-dispatch-table",
      }],
      unlistedExports: "forbidden",
      observedModuleCannotExtendAllowlist: true,
      releaseConformance: "allowed-only-for-exact-reviewed-support-exports",
    });
    expect(wasm.supportExports.exactFunctionAllowlist).toHaveLength(29);
    expect(wasm.supportExports.functionReview).toMatchObject({
      emscriptenVersion: "6.0.3",
      emscriptenCommit: "283e2d130132859fde6a4e4c87fd254b38127651",
      visibility: "worker-internal-not-browsergrad-c-api",
      runtimeRoles: [
        { name: "allocator-runtime", exactFunctionCount: 14 },
        { name: "javascript-exception-bridge", exactFunctionCount: 6 },
        { name: "module-initialization", exactFunctionCount: 1 },
        { name: "stack-runtime", exactFunctionCount: 8 },
      ],
    });
    expect(wasm.supportExports.exactFunctionAllowlist.some((entry) =>
      entry.name.startsWith("bg_cpp_cute_"))).toBe(false);
    expect(wasm.supportExports.tableReview).toEqual({
      basis: "detached-raw-wasm-inspection-and-javascript-exception-dispatch-requirement",
      visibility: "worker-internal-not-browsergrad-c-api",
      exactExportCount: 1,
      runtimeRole: "javascript-exception-dispatch-table",
    });
  });

  it("pins table/global projections and treats absent target metadata as advisory", async () => {
    const prepared = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const policy = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared)
      .manifest.body.wasm.structuralPolicy;

    expect(policy).toMatchObject({
      status: "independently-reviewed-hash-pinned",
      releaseConformance: "allowed-only-for-exact-reviewed-structural-projection",
      tables: {
        maximumCount: 1,
        allowedElementTypes: ["funcref"],
        imported: "forbidden",
        declaredMaximumRequired: true,
        maximumElementsCeiling: 65_536,
        exactReviewedProjection: [
          { elementType: "funcref", minimum: 14_549, maximum: 14_549 },
        ],
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
          status: "independently-reviewed-absent",
          authority: "advisory-only-static-opcode-and-section-inspection-is-authoritative",
          requiredDeclarations: [],
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
    expect(frame.decodeLimits).toEqual({
      maxDocumentByteLength: frame.maxFrameByteLength,
      maxNestingDepth: 128,
      maxNodeCount: 1_000_000,
      maxCumulativeStringByteLength: frame.maxFrameByteLength,
      maxArrayElementCount: 65_536,
      maxObjectPropertyCount: 512,
      maxScratchByteLength: frame.maxFrameByteLength * 4,
      accounting: {
        documentBytes: "per-region-before-utf8-decode",
        nestingNodesAndStrings:
          "per-region-root-depth-one-strings-include-object-keys-and-values",
        containers: "per-array-or-object",
        scratchBytes:
          "peak-live-decoder-owned-bytes-per-compile-session-excluding-input-frame-vfs-and-producer-state",
      },
      numberPolicy:
        "safe-integer-lexemes-only-no-negative-zero-fraction-or-exponent",
      canonicalValidationPolicy:
        "byte-exact-browsergrad-canonical-json-validation-per-region-rejecting-duplicate-keys",
    });
    expect(frame.decodeLimits.maxScratchByteLength)
      .toBeLessThanOrEqual(unwrapPreparedCppCuteBrowserRuntimeAbiManifest(prepared)
        .manifest.body.wasm.memory.maxCompilerWorkingByteLength);
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
    ["tenth C export", (value: MutableJson) => {
      const exports = arrayField(body(value), "cExports");
      const copied = structuredClone(exports[0]);
      exports.push({ ...copied, ordinal: 9, cSymbol: "bg_escape" });
    }, "cExports"],
    ["changed C signature", (value: MutableJson) => {
      arrayField(body(value), "cExports")[3]!.wasmParameters = ["i32"];
    }, "cExports[3]"],
    ["nullable allocator metrics pointer", (value: MutableJson) => {
      const metrics = objectField(body(value), "allocatorMetricsRecord");
      objectField(metrics, "pointerContract").zero = "allowed";
    }, "allocatorMetricsRecord.pointerContract"],
    ["shifted allocator metrics field", (value: MutableJson) => {
      const metrics = objectField(body(value), "allocatorMetricsRecord");
      arrayField(metrics, "fields")[3]!.offset = 24;
    }, "allocatorMetricsRecord.fields"],
    ["wrapping allocator metrics counter", (value: MutableJson) => {
      const metrics = objectField(body(value), "allocatorMetricsRecord");
      objectField(metrics, "accounting").overflowPolicy = "wrap-u64";
    }, "allocatorMetricsRecord.accounting"],
    ["ambiguous allocator realloc accounting", (value: MutableJson) => {
      const metrics = objectField(body(value), "allocatorMetricsRecord");
      objectField(metrics, "accounting").reallocNonzeroSuccessSemantics = "implementation-defined";
    }, "allocatorMetricsRecord.accounting"],
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
    ["unbounded VFS index nodes", (value: MutableJson) => {
      objectField(body(value), "vfs").maxIndexedNodes = Number.MAX_SAFE_INTEGER;
    }, "vfs"],
    ["unbounded VFS logical index bytes", (value: MutableJson) => {
      objectField(body(value), "vfs").maxIndexLogicalByteLength = Number.MAX_SAFE_INTEGER;
    }, "vfs"],
    ["unbounded aggregate live-open VFS bytes", (value: MutableJson) => {
      objectField(body(value), "vfs").maxAggregateLiveOpenByteLength = Number.MAX_SAFE_INTEGER;
    }, "vfs"],
    ["Wasm-resident live-open accounting", (value: MutableJson) => {
      objectField(body(value), "vfs").liveOpenByteAccounting = "wasm-resident-bytes";
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
    ["widened frame decoder document budget", (value: MutableJson) => {
      objectField(objectField(body(value), "inputFrame"), "decodeLimits")
        .maxDocumentByteLength = 4_194_305;
    }, "inputFrame.decodeLimits"],
    ["widened frame decoder scratch budget", (value: MutableJson) => {
      objectField(objectField(body(value), "inputFrame"), "decodeLimits")
        .maxScratchByteLength = 16_777_217;
    }, "inputFrame.decodeLimits"],
    ["weakened frame decoder number policy", (value: MutableJson) => {
      objectField(objectField(body(value), "inputFrame"), "decodeLimits")
        .numberPolicy = "any-json-number";
    }, "inputFrame.decodeLimits"],
    ["weakened frame decoder canonical policy", (value: MutableJson) => {
      objectField(objectField(body(value), "inputFrame"), "decodeLimits")
        .canonicalValidationPolicy = "parse-only";
    }, "inputFrame.decodeLimits"],
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
    for (const [field, value] of [["major", 2], ["minor", 6]] as const) {
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
