import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  decodeWireJson,
  hashCanonicalJson,
  sha256Hex,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE,
  type CppCuteBrowserRuntimeAbiManifestV1Resource,
} from "./resources/cpp_cute_browser_runtime_abi_v1.js";

export const CPP_CUTE_BROWSER_RUNTIME_ABI_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-runtime-abi-manifest";
export const CPP_CUTE_BROWSER_RUNTIME_ABI_MAJOR = 1;
export const CPP_CUTE_BROWSER_RUNTIME_ABI_MINOR = 4;
export const CPP_CUTE_BROWSER_RUNTIME_ABI_BYTE_LIMIT = 64 * 1024;
export const CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID =
  "bg.cpp.browser-runtime-abi.sha256.84e8320ae85e3f49dba5adc729fe07544aa7fcb9d0f72f18c604fc5d840d0bf2";
export const CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256 =
  "4d6de469eb287dabfdc8ed4d1c057c5aef2af915a651a8d9c6eee2e9e9c57c69";
export const CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256 =
  "34173da34810c0f0cb80af84b36bb800007c05bb3f9f759943e2c6d10e5d2226";
export const CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256 =
  "8b48a9e038fc9c2b3ed677d6df99e7d0803da9083db19c41a3017f844fa10f48";
export const CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256 =
  "54fcf849f006f656162394c9feaeec801af059cbc7b0d61e612bcdebc6abb361";

const MANIFEST_ID = /^bg\.cpp\.browser-runtime-abi\.sha256\.[0-9a-f]{64}$/u;
const PREPARED_MANIFESTS = new WeakMap<object, StoredCppCuteBrowserRuntimeAbiManifest>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export const CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_RUNTIME_ABI_BYTE_LIMIT,
  maxDepth: 16,
  maxNodes: 4_096,
  maxStringBytes: 48 * 1024,
  maxArrayLength: 128,
  maxObjectProperties: 32,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 8_192,
});

export type CppCuteBrowserRuntimeAbiManifestV1 = CppCuteBrowserRuntimeAbiManifestV1Resource;
export type CppCuteBrowserRuntimeAbiBodyV1 = CppCuteBrowserRuntimeAbiManifestV1["body"];

declare const preparedCppCuteBrowserRuntimeAbiManifestBrand: unique symbol;

/**
 * Opaque authority over the exact canonical runtime-ABI design contract.
 *
 * It deliberately proves neither conformance of any Wasm bytes nor release or
 * Worker-execution readiness. Those require separate observed authorities.
 */
export interface PreparedCppCuteBrowserRuntimeAbiManifest {
  readonly [preparedCppCuteBrowserRuntimeAbiManifestBrand]: true;
  readonly manifestId: string;
  readonly runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1";
  readonly resourceSha256: string;
  readonly contractSha256: string;
  readonly generatedImportAllowlistSha256: string;
  readonly supportFunctionAllowlistSha256: string;
  readonly resourceByteLength: number;
  readonly designAuthority: true;
  readonly interfaceReviewReady: false;
  readonly observedWasmVerified: false;
  readonly releaseReady: false;
}

export interface PreparedCppCuteBrowserRuntimeAbiManifestRecord {
  readonly manifest: CppCuteBrowserRuntimeAbiManifestV1;
}

interface StoredCppCuteBrowserRuntimeAbiManifest
  extends PreparedCppCuteBrowserRuntimeAbiManifestRecord {
  readonly bytes: Uint8Array;
}

export interface DecodeCppCuteBrowserRuntimeAbiManifestOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserRuntimeAbiManifestErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-UNVERIFIED";

export class CppCuteBrowserRuntimeAbiManifestError extends Error {
  constructor(
    readonly code: CppCuteBrowserRuntimeAbiManifestErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserRuntimeAbiManifestError";
  }
}

validateBodyInvariants(CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body);
if (CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.manifestId !==
    CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID) {
  invalid("$.manifestId", "built-in manifest ID does not equal the pinned v1 identity");
}

const BUILTIN_RESOURCE_BYTES = canonicalResourceBytes(CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE);
const BUILTIN_BODY_BYTES = canonicalResourceBytes(CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body);

/** Returns a disposable copy of the exact canonical ABI-manifest resource. */
export function cppCuteBrowserRuntimeAbiManifestResourceBytes(): Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

export async function decodeCppCuteBrowserRuntimeAbiManifest(
  bytes: Uint8Array,
  options: DecodeCppCuteBrowserRuntimeAbiManifestOptions = {},
): Promise<PreparedCppCuteBrowserRuntimeAbiManifest> {
  const signal = normalizeOptions(options);
  const snapshot = snapshotBytes(bytes);
  throwIfAborted(signal);
  let value: JsonValue;
  try {
    value = decodeWireJson(snapshot, { limits: CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$bytes", "runtime-ABI manifest decoding exceeded fixed resource limits", { cause });
    }
    invalid("$bytes", "runtime-ABI manifest bytes are not bounded strict JSON", { cause });
  }
  const manifest = parseManifest(value);
  const canonical = canonicalResourceBytes(manifest);
  if (!equalBytes(snapshot, canonical)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-NONCANONICAL-BYTES",
      "$bytes",
      "runtime-ABI manifest bytes must exactly equal canonical JSON bytes",
    );
  }
  throwIfAborted(signal);
  const expectedManifestId = await deriveCppCuteBrowserRuntimeAbiManifestId(manifest.body);
  if (manifest.manifestId !== expectedManifestId) {
    hashMismatch("$.manifestId", `manifest ID must equal ${expectedManifestId}`);
  }
  throwIfAborted(signal);
  const [
    resourceSha256,
    contractSha256,
    generatedImportAllowlistSha256,
    supportFunctionAllowlistSha256,
  ] = await Promise.all([
    hashBytes(snapshot, "$bytes"),
    hashJson({
      domain: "browsergrad.compiler.cpp-cute.browser-runtime-abi-contract.v1",
      body: manifest.body,
    }, "$.body"),
    deriveGeneratedImportAllowlistSha256(manifest.body),
    deriveSupportFunctionAllowlistSha256(manifest.body),
  ]);
  if (resourceSha256 !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256) {
    hashMismatch("$bytes", "canonical resource SHA-256 does not equal the pinned v1 resource identity");
  }
  if (contractSha256 !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256) {
    hashMismatch("$.body", "runtime-ABI contract SHA-256 does not equal the pinned v1 contract identity");
  }
  if (generatedImportAllowlistSha256 !==
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256 ||
      manifest.body.hostImports.generatedImportAllowlist.allowlistSha256 !==
        generatedImportAllowlistSha256) {
    hashMismatch(
      "$.body.hostImports.generatedImportAllowlist.allowlistSha256",
      "generated-import allowlist SHA-256 does not equal the closed v1 policy projection",
    );
  }
  if (supportFunctionAllowlistSha256 !==
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256 ||
      manifest.body.wasm.supportExports.functionAllowlistSha256 !==
        supportFunctionAllowlistSha256) {
    hashMismatch(
      "$.body.wasm.supportExports.functionAllowlistSha256",
      "support-function export allowlist SHA-256 does not equal the closed v1 policy projection",
    );
  }
  throwIfAborted(signal);
  const prepared = Object.freeze({
    manifestId: expectedManifestId,
    runtimeAbiId: manifest.body.runtimeAbiId,
    resourceSha256,
    contractSha256,
    generatedImportAllowlistSha256,
    supportFunctionAllowlistSha256,
    resourceByteLength: snapshot.byteLength,
    designAuthority: true,
    interfaceReviewReady: false,
    observedWasmVerified: false,
    releaseReady: false,
  }) as PreparedCppCuteBrowserRuntimeAbiManifest;
  PREPARED_MANIFESTS.set(prepared, Object.freeze({
    manifest,
    bytes: new Uint8Array(snapshot),
  }));
  return prepared;
}

export async function deriveCppCuteBrowserRuntimeAbiManifestId(
  body: CppCuteBrowserRuntimeAbiBodyV1,
): Promise<string> {
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.browser-runtime-abi-manifest-id.v1",
    body,
  }, "$.manifestId");
  return `bg.cpp.browser-runtime-abi.sha256.${digest}`;
}

export async function deriveCppCuteBrowserGeneratedImportAllowlistSha256(
  body: CppCuteBrowserRuntimeAbiBodyV1,
): Promise<string> {
  return deriveGeneratedImportAllowlistSha256(body);
}

export async function deriveCppCuteBrowserSupportFunctionAllowlistSha256(
  body: CppCuteBrowserRuntimeAbiBodyV1,
): Promise<string> {
  return deriveSupportFunctionAllowlistSha256(body);
}

export function unwrapPreparedCppCuteBrowserRuntimeAbiManifest(
  prepared: PreparedCppCuteBrowserRuntimeAbiManifest,
): PreparedCppCuteBrowserRuntimeAbiManifestRecord {
  return Object.freeze({ manifest: storedManifest(prepared).manifest });
}

export function canonicalCppCuteBrowserRuntimeAbiManifestBytes(
  prepared: PreparedCppCuteBrowserRuntimeAbiManifest,
): Uint8Array {
  return new Uint8Array(storedManifest(prepared).bytes);
}

function parseManifest(value: JsonValue): CppCuteBrowserRuntimeAbiManifestV1 {
  const object = closedObject(value, ["schema", "version", "manifestId", "body"], "$", true);
  literal(field(object, "schema", "$"), CPP_CUTE_BROWSER_RUNTIME_ABI_SCHEMA, "$.schema");
  const version = closedObject(field(object, "version", "$"), ["major", "minor"], "$.version", true);
  if (version.major !== CPP_CUTE_BROWSER_RUNTIME_ABI_MAJOR) {
    unsupported("$.version.major", `reader supports major ${CPP_CUTE_BROWSER_RUNTIME_ABI_MAJOR}`);
  }
  if (version.minor !== CPP_CUTE_BROWSER_RUNTIME_ABI_MINOR) {
    unsupported(
      "$.version.minor",
      `closed reader supports ${CPP_CUTE_BROWSER_RUNTIME_ABI_MAJOR}.${CPP_CUTE_BROWSER_RUNTIME_ABI_MINOR} only`,
    );
  }
  const manifestId = boundedPattern(field(object, "manifestId", "$"), "$.manifestId", MANIFEST_ID);
  const body = field(object, "body", "$" );
  if (typeof body !== "object" || body === null || Array.isArray(body)) invalid("$.body", "expected object");
  validateBodyInvariants(body as JsonObject);
  if (!equalBytes(canonicalResourceBytes(body), BUILTIN_BODY_BYTES)) {
    invalid("$.body", "body does not equal the single supported runtime-ABI design contract");
  }
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_RUNTIME_ABI_SCHEMA,
    version: Object.freeze({
      major: CPP_CUTE_BROWSER_RUNTIME_ABI_MAJOR,
      minor: CPP_CUTE_BROWSER_RUNTIME_ABI_MINOR,
    }),
    manifestId,
    body: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body,
  }) as CppCuteBrowserRuntimeAbiManifestV1;
}

function validateBodyInvariants(value: JsonObject): void {
  const body = value as unknown as CppCuteBrowserRuntimeAbiBodyV1;
  try {
    if (body.runtimeAbiId !== "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1") {
      invalid("$.body.runtimeAbiId", "unknown runtime ABI");
    }
    if (body.authority.kind !== "design-contract-only" ||
        body.authority.observedWasm !== "detached-verification-required" ||
        body.authority.workerExecution !== "not-authorized" ||
        body.authority.releaseReadiness !== "not-authorized") {
      invalid("$.body.authority", "manifest must not claim Wasm observation, execution, or release authority");
    }
    if (body.wasm.moduleRole !== "compiler-extractor-only-user-programs-never-linked-or-executed" ||
        body.wasm.cAbiVersion !== 65_537 ||
        body.wasm.cAbiVersionEncoding !== "major-shift-left-16-bitwise-or-minor" ||
        body.wasm.startSection !== "forbidden" || body.wasm.unlistedCExports !== "forbidden") {
      invalid("$.body.wasm", "module role, ABI version, start, or export closure differs from runtime v1");
    }
    const supportExports = body.wasm.supportExports;
    if (supportExports.status !== "independently-reviewed-hash-pinned" ||
        supportExports.functionAllowlistSha256 !==
          CPP_CUTE_BROWSER_RUNTIME_ABI_V1_SUPPORT_FUNCTION_ALLOWLIST_SHA256 ||
        supportExports.exactFunctionAllowlist.length !== 29 ||
        supportExports.exactGlobalAllowlist.length !== 0 ||
        supportExports.exactTableAllowlist.length !== 1 ||
        supportExports.unlistedExports !== "forbidden" ||
        supportExports.observedModuleCannotExtendAllowlist !== true ||
        supportExports.releaseConformance !==
          "allowed-only-for-exact-reviewed-support-exports") {
      invalid(
        "$.body.wasm.supportExports",
        "support exports must equal the independently reviewed closed policy",
      );
    }
    const supportReview = supportExports.functionReview;
    if (supportReview.basis !==
          "pinned-emscripten-runtime-sources-locked-link-flags-and-detached-raw-wasm-inspection" ||
        supportReview.emscriptenVersion !== "6.0.3" ||
        supportReview.emscriptenCommit !== "283e2d130132859fde6a4e4c87fd254b38127651" ||
        supportReview.visibility !== "worker-internal-not-browsergrad-c-api") {
      invalid(
        "$.body.wasm.supportExports.functionReview",
        "support-function review must bind the selected Emscripten runtime and worker-only visibility",
      );
    }
    const expectedSupportRoles = [
      ["allocator-runtime", 14],
      ["javascript-exception-bridge", 6],
      ["module-initialization", 1],
      ["stack-runtime", 8],
    ] as const;
    if (supportReview.runtimeRoles.length !== expectedSupportRoles.length ||
        supportReview.runtimeRoles.some((role, index) => {
          const expected = expectedSupportRoles[index];
          return expected === undefined || role.name !== expected[0] ||
            role.exactFunctionCount !== expected[1];
        })) {
      invalid(
        "$.body.wasm.supportExports.functionReview.runtimeRoles",
        "support-function runtime-role inventory differs from the independent review",
      );
    }
    const supportNames = new Set<string>();
    const supportRoleCounts = new Map<string, number>();
    for (const [index, entry] of supportExports.exactFunctionAllowlist.entries()) {
      if (entry.name.length === 0 || entry.name.startsWith("bg_cpp_cute_") ||
          supportNames.has(entry.name) || entry.wasmResults.length > 1 ||
          [...entry.wasmParameters, ...entry.wasmResults].some((value) =>
            value !== "f32" && value !== "f64" && value !== "i32" && value !== "i64")) {
        invalid(
          `$.body.wasm.supportExports.exactFunctionAllowlist[${index}]`,
          "support export has an invalid, duplicate, public-API, or non-core-Wasm signature",
        );
      }
      supportNames.add(entry.name);
      supportRoleCounts.set(entry.runtimeRole, (supportRoleCounts.get(entry.runtimeRole) ?? 0) + 1);
    }
    for (const [name, count] of expectedSupportRoles) {
      if (supportRoleCounts.get(name) !== count) {
        invalid(
          "$.body.wasm.supportExports.exactFunctionAllowlist",
          `support export runtime role ${name} must contain exactly ${count} functions`,
        );
      }
    }
    const tableReview = supportExports.tableReview;
    const tableExport = supportExports.exactTableAllowlist[0];
    if (tableReview.basis !==
          "detached-raw-wasm-inspection-and-javascript-exception-dispatch-requirement" ||
        tableReview.visibility !== "worker-internal-not-browsergrad-c-api" ||
        tableReview.exactExportCount !== 1 ||
        tableReview.runtimeRole !== "javascript-exception-dispatch-table" ||
        tableExport?.name !== "__indirect_function_table" || tableExport.index !== 0 ||
        tableExport.runtimeRole !== "javascript-exception-dispatch-table") {
      invalid(
        "$.body.wasm.supportExports.tableReview",
        "support table must be the single worker-internal JavaScript exception dispatch table",
      );
    }
    const structural = body.wasm.structuralPolicy;
    if (structural.status !== "table-and-global-projections-reviewed-target-features-pending" ||
        structural.releaseConformance !==
          "forbidden-until-exact-first-build-projection-is-reviewed-and-repinned") {
      invalid("$.body.wasm.structuralPolicy", "Wasm structural projection must remain release-blocked pending review");
    }
    if (structural.tables.maximumCount !== 1 ||
        structural.tables.imported !== "forbidden" ||
        structural.tables.declaredMaximumRequired !== true ||
        structural.tables.maximumElementsCeiling !== 65_536 ||
        structural.tables.exactReviewedProjection.length !== 1 ||
        structural.tables.exactReviewedProjection[0]?.elementType !== "funcref" ||
        structural.tables.exactReviewedProjection[0].minimum !== 14_549 ||
        structural.tables.exactReviewedProjection[0].maximum !== 14_549) {
      invalid("$.body.wasm.structuralPolicy.tables", "table policy differs from the exact reviewed v1 projection");
    }
    assertExactStrings(
      structural.tables.allowedElementTypes,
      ["funcref"],
      "$.body.wasm.structuralPolicy.tables.allowedElementTypes",
    );
    if (structural.globals.maximumCount !== 4_096 || structural.globals.imported !== "forbidden" ||
        structural.globals.exactReviewedExports.length !== 0) {
      invalid("$.body.wasm.structuralPolicy.globals", "global policy differs from the bounded unresolved v1 policy");
    }
    assertExactStrings(structural.globals.allowedValueTypes, [
      "f32", "f64", "i32", "i64",
    ], "$.body.wasm.structuralPolicy.globals.allowedValueTypes");
    if (structural.tags.exactCount !== 0 || structural.tags.imported !== "forbidden" ||
        structural.tags.exported !== "forbidden") {
      invalid("$.body.wasm.structuralPolicy.tags", "runtime v1 forbids Wasm exception tags");
    }
    const customSections = structural.customSections;
    if (customSections.maximumCount !== 4 || customSections.maximumSectionByteLength !== 524_288 ||
        customSections.maximumTotalByteLength !== 1_048_576 ||
        customSections.duplicateNames !== "forbidden" ||
        customSections.exactReviewedNameAllowlist.length !== 0 ||
        customSections.unlistedNames !== "forbidden") {
      invalid("$.body.wasm.structuralPolicy.customSections", "custom-section policy differs from bounded unresolved v1");
    }
    assertExactStrings(customSections.explicitlyForbiddenNames, [
      "dylink.0", "producers", "sourceMappingURL",
    ], "$.body.wasm.structuralPolicy.customSections.explicitlyForbiddenNames");
    if (customSections.targetFeatures.sectionName !== "target_features" ||
        customSections.targetFeatures.status !== "unresolved-first-build-review-required" ||
        customSections.targetFeatures.exactRawSectionProjection.length !== 0) {
      invalid("$.body.wasm.structuralPolicy.customSections.targetFeatures", "target_features review remains unresolved");
    }
    // Tool-conventions names are wire vocabulary, not BrowserGrad feature
    // vocabulary (`sign-ext`/`multimemory` differ intentionally).
    assertExactStrings(customSections.targetFeatures.requiredDeclarations, [
      "bulk-memory", "mutable-globals", "nontrapping-fptoint", "sign-ext",
    ], "$.body.wasm.structuralPolicy.customSections.targetFeatures.requiredDeclarations");
    assertExactStrings(customSections.targetFeatures.forbiddenDeclarations, [
      "atomics", "exception-handling", "memory64", "multimemory", "simd128",
    ], "$.body.wasm.structuralPolicy.customSections.targetFeatures.forbiddenDeclarations");
    assertExactStrings(body.wasm.requiredFeatures, [
      "bulk-memory", "mutable-globals", "nontrapping-fptoint", "sign-extension",
    ], "$.body.wasm.requiredFeatures");
    assertExactStrings(body.wasm.forbiddenFeatures, [
      "atomics", "exception-handling", "memory64", "multi-memory", "simd128", "threads",
    ], "$.body.wasm.forbiddenFeatures");
    const featurePolicy = body.wasm.featurePolicy;
    if (featurePolicy.instructionSetBaseline !== "webassembly-mvp" ||
        featurePolicy.unlistedExtensions !== "forbidden" ||
        featurePolicy.staticOpcodeAndSectionInspection !== "required" ||
        featurePolicy.targetFeaturesCrossCheck !== "required-but-not-authoritative") {
      invalid("$.body.wasm.featurePolicy", "Wasm extension policy must reject every undeclared extension");
    }
    assertExactStrings(
      featurePolicy.allowedExtensions,
      body.wasm.requiredFeatures,
      "$.body.wasm.featurePolicy.allowedExtensions",
    );
    const memory = body.wasm.memory;
    if (body.wasm.addressBits !== 32 || memory.count !== 1 || memory.addressType !== "i32" ||
        memory.imported !== false || memory.exported !== true || memory.exportName !== "memory" ||
        memory.sharing !== "unshared" ||
        memory.ownership !== "module-instance-owned-by-dedicated-worker" ||
        memory.growth !== "allowed-to-maximum") {
      invalid("$.body.wasm.memory", "runtime v1 requires one module-exported unshared wasm32 memory");
    }
    if (memory.initialPages * memory.pageByteLength !== 268_435_456 ||
        memory.maximumPages * memory.pageByteLength !== 1_073_741_824 ||
        memory.growthLinearStepPages * memory.pageByteLength !== 67_108_864) {
      invalid("$.body.wasm.memory", "memory pages do not match the fixed runtime-v1 byte contract");
    }
    const reservedBytes = memory.stackByteLength + memory.maxCompilerWorkingByteLength +
      memory.maxInputFrameByteLength + memory.maxResultByteLength;
    if (reservedBytes > memory.maximumPages * memory.pageByteLength) {
      invalid("$.body.wasm.memory", "declared reservations exceed maximum linear memory");
    }
    const expectedExports = [
      ["bg_cpp_cute_abi_version", "uint32_t bg_cpp_cute_abi_version(void)", 0, 1, "u32-packed-abi-version"],
      ["bg_cpp_cute_alloc", "uint32_t bg_cpp_cute_alloc(uint32_t byte_length)", 1, 1, "u32-input-pointer-zero-on-failure"],
      ["bg_cpp_cute_allocator_metrics_pointer", "uint32_t bg_cpp_cute_allocator_metrics_pointer(void)", 0, 1, "u32-nonzero-stable-read-only-allocator-metrics-record-v1-pointer"],
      ["bg_cpp_cute_compile", "int32_t bg_cpp_cute_compile(uint32_t input_pointer, uint32_t input_length)", 2, 1, "typed-compile-status"],
      ["bg_cpp_cute_free", "void bg_cpp_cute_free(uint32_t pointer, uint32_t byte_length)", 2, 0, "void-status-readable-separately"],
      ["bg_cpp_cute_reset", "void bg_cpp_cute_reset(void)", 0, 0, "void-infallible-for-live-instance"],
      ["bg_cpp_cute_result_length", "uint32_t bg_cpp_cute_result_length(void)", 0, 1, "u32-result-length-zero-unless-artifact-ready"],
      ["bg_cpp_cute_result_pointer", "uint32_t bg_cpp_cute_result_pointer(void)", 0, 1, "u32-result-pointer-zero-unless-artifact-ready"],
      ["bg_cpp_cute_status", "int32_t bg_cpp_cute_status(void)", 0, 1, "current-typed-status-without-state-mutation"],
    ] as const;
    if (body.cExports.length !== expectedExports.length) invalid("$.body.cExports", "expected exactly nine C exports");
    for (const [index, expected] of expectedExports.entries()) {
      const actual = body.cExports[index];
      if (actual?.ordinal !== index || actual.cSymbol !== expected[0] || actual.wasmExportName !== expected[0] ||
          actual.cSignature !== expected[1] || actual.wasmParameters.length !== expected[2] ||
          actual.wasmResults.length !== expected[3] || actual.resultSemantics !== expected[4] ||
          [...actual.wasmParameters, ...actual.wasmResults].some((type) => type !== "i32")) {
        invalid(`$.body.cExports[${index}]`, "C export does not match the exact runtime-v1 signature inventory");
      }
    }
    const metrics = body.allocatorMetricsRecord;
    if (metrics.schema !== "browsergrad.compiler.cpp-cute.allocator-metrics-record" ||
        metrics.version.major !== 1 || metrics.version.minor !== 0 ||
        metrics.magicAscii !== "BGRTMET1" || metrics.byteLength !== 72 ||
        metrics.alignmentByteLength !== 8 || metrics.encoding !== "little-endian-fixed-width" ||
        metrics.storage !== "module-global-linear-memory-record" ||
        metrics.pointerExport !== "bg_cpp_cute_allocator_metrics_pointer") {
      invalid("$.body.allocatorMetricsRecord", "allocator metrics record identity or layout differs from runtime v1");
    }
    assertExactNumbers(metrics.magicBytes, [66, 71, 82, 84, 77, 69, 84, 49],
      "$.body.allocatorMetricsRecord.magicBytes");
    if (metrics.pointerContract.resultEncoding !== "u32-wasm32-linear-memory-offset" ||
        metrics.pointerContract.zero !== "forbidden-for-conforming-live-module-instance" ||
        metrics.pointerContract.stability !== "constant-for-module-instance-lifetime" ||
        metrics.pointerContract.completeRange !== "must-fit-current-exported-memory" ||
        metrics.pointerContract.mutability !== "module-writes-host-read-only") {
      invalid("$.body.allocatorMetricsRecord.pointerContract", "allocator metrics pointer contract differs from runtime v1");
    }
    if (metrics.snapshotContract.allowedPhases !== "between-synchronous-runtime-calls-only" ||
        metrics.snapshotContract.hostRead !== "copy-exact-record-before-decoding" ||
        metrics.snapshotContract.memoryGrowthDuringCopy !== "forbidden" ||
        metrics.snapshotContract.consistency !==
          "module-calls-are-synchronous-and-memory-is-unshared") {
      invalid("$.body.allocatorMetricsRecord.snapshotContract", "allocator metrics snapshot contract differs from runtime v1");
    }
    const accounting = metrics.accounting;
    if (accounting.unit !== "requested-bytes" ||
        accounting.scope !== "all-instrumented-module-global-allocator-events" ||
        accounting.requestedByteBasis !==
          "caller-requested-byte-length-before-alignment-or-allocator-rounding" ||
        accounting.currentFormula !== "cumulative-allocated-minus-cumulative-freed" ||
        accounting.peakFormula !== "maximum-current-live-since-module-instantiation" ||
        accounting.resetPolicy !==
          "counters-persist-and-reset-frees-are-accounted-until-worker-termination" ||
        accounting.allocationCountSemantics !==
          "successful-creation-or-resize-of-one-tracked-live-allocation" ||
        accounting.freeCountSemantics !== "successful-release-of-one-tracked-live-allocation" ||
        accounting.failedAllocationCountSemantics !==
          "failed-nonzero-request-that-preserves-all-prior-live-allocations" ||
        accounting.failedInvalidRequestSemantics !==
          "invalid-or-size-overflowing-nonzero-request-increments-failed-once-zero-request-does-not" ||
        accounting.zeroByteCreationSemantics !==
          "nonnull-result-counts-one-tracked-zero-byte-allocation-null-result-is-permitted-no-op-neither-success-nor-failure" ||
        accounting.freeNullSemantics !== "no-op-with-no-counter-change" ||
        accounting.reallocNullPointerSemantics !== "same-as-creation-at-requested-size" ||
        accounting.reallocNonzeroSuccessSemantics !==
          "allocated-adds-new-requested-size-freed-adds-old-requested-size-success-and-free-counts-each-increment-once-even-in-place" ||
        accounting.reallocNonzeroFailureSemantics !==
          "failed-count-increments-once-and-all-live-byte-and-success-free-counters-remain-unchanged" ||
        accounting.reallocZeroSizeSemantics !==
          "nonnull-old-pointer-is-released-freed-adds-old-requested-size-free-count-increments-once-result-is-null-and-failure-count-does-not-change" ||
        accounting.overflowPolicy !==
          "counter-overflow-must-fail-closed-before-wrap-and-forbids-artifact-ready") {
      invalid("$.body.allocatorMetricsRecord.accounting", "allocator metrics accounting differs from runtime v1");
    }
    const interception = accounting.interception;
    assertExactStrings(interception.exactEntrypoints, [
      "aligned_alloc", "calloc", "free", "__libc_calloc", "__libc_free",
      "__libc_malloc", "__libc_realloc", "malloc", "memalign", "posix_memalign",
      "pvalloc", "realloc", "reallocarray", "valloc",
    ], "$.body.allocatorMetricsRecord.accounting.interception.exactEntrypoints");
    assertExactStrings(interception.forbiddenEntrypoints, [
      "bulk_free", "independent_calloc", "independent_comalloc", "realloc_in_place",
    ], "$.body.allocatorMetricsRecord.accounting.interception.forbiddenEntrypoints");
    assertExactStrings(interception.underlyingBypassEntrypoints, [
      "emscripten_builtin_calloc", "emscripten_builtin_free",
      "emscripten_builtin_malloc", "emscripten_builtin_memalign",
      "emscripten_builtin_realloc",
    ], "$.body.allocatorMetricsRecord.accounting.interception.underlyingBypassEntrypoints");
    if (interception.directBypassReferences !==
          "forbidden-outside-BrowserGradCppCuteMetrics.cpp" ||
        interception.linkClosureProof !==
          "pinned-object-and-final-wasm-call-graph-evidence-required") {
      invalid(
        "$.body.allocatorMetricsRecord.accounting.interception",
        "allocator interception must remain source-closed and observed-link blocked",
      );
    }
    assertExactStrings(accounting.excludes, [
      "allocator-metadata-alignment-and-size-class-rounding",
      "javascript-heap-and-worker-host-memory",
      "static-data-and-module-globals",
      "vfs-logical-reservations-not-resident-in-wasm-memory",
      "wasm-stack",
    ], "$.body.allocatorMetricsRecord.accounting.excludes");
    assertExactStrings(accounting.invariants, [
      "cumulative-freed-is-less-than-or-equal-to-cumulative-allocated",
      "current-equals-cumulative-allocated-minus-cumulative-freed",
      "free-count-is-less-than-or-equal-to-successful-allocation-count",
      "peak-is-between-current-and-cumulative-allocated-inclusive",
      "peak-cumulative-and-count-counters-are-monotonic-for-module-instance-lifetime",
    ], "$.body.allocatorMetricsRecord.accounting.invariants");
    assertExactNumbers(metrics.fields.map((field) => field.ordinal), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      "$.body.allocatorMetricsRecord.fields[*].ordinal");
    assertExactStrings(metrics.fields.map((field) => field.name), [
      "magic", "version", "byteLength", "currentLiveGlobalRequestedByteLength",
      "peakLiveGlobalRequestedByteLength", "cumulativeGlobalAllocatedRequestedByteLength",
      "cumulativeGlobalFreedRequestedByteLength", "successfulAllocationCount", "freeCount",
      "failedAllocationCount",
    ], "$.body.allocatorMetricsRecord.fields[*].name");
    assertExactNumbers(metrics.fields.map((field) => field.offset), [0, 8, 12, 16, 24, 32, 40, 48, 56, 64],
      "$.body.allocatorMetricsRecord.fields[*].offset");
    assertExactNumbers(metrics.fields.map((field) => field.byteLength), [8, 4, 4, 8, 8, 8, 8, 8, 8, 8],
      "$.body.allocatorMetricsRecord.fields[*].byteLength");
    assertExactStrings(metrics.fields.map((field) => field.encoding), [
      "ascii[8]", "u32le", "u32le", "u64le", "u64le", "u64le", "u64le", "u64le",
      "u64le", "u64le",
    ], "$.body.allocatorMetricsRecord.fields[*].encoding");
    assertExactStrings(metrics.fields.map((field) => field.semantics), [
      "must-equal-BGRTMET1", "must-equal-1", "must-equal-72", "current-live-requested-bytes",
      "peak-live-requested-bytes", "cumulative-successfully-allocated-requested-bytes",
      "cumulative-successfully-freed-requested-bytes", "successful-creation-or-resize-event-count",
      "successful-tracked-release-event-count", "failed-request-event-count",
    ], "$.body.allocatorMetricsRecord.fields[*].semantics");
    if (metrics.authority.values !== "module-self-reported-local-observation-only" ||
        metrics.authority.producerConformance !==
          "detached-observed-wasm-verification-required" ||
        metrics.authority.workerExecution !== "not-authorized-by-record-values" ||
        metrics.authority.lowering !== "not-authorized-by-record-values") {
      invalid("$.body.allocatorMetricsRecord.authority", "allocator metrics record must not grant execution or lowering authority");
    }
    const expectedImports = [
      ["bg_vfs_status", "int32_t bg_vfs_status(uint32_t path_pointer, uint32_t path_length, uint32_t metadata_pointer)", 3, "write-one-32-byte-metadata-record-for-an-existing-file-or-directory"],
      ["bg_vfs_open", "int32_t bg_vfs_open(uint32_t path_pointer, uint32_t path_length, uint32_t open_result_pointer)", 3, "open-one-existing-file-and-write-one-16-byte-open-result"],
      ["bg_vfs_read", "int32_t bg_vfs_read(uint32_t handle, uint32_t offset_low, uint32_t offset_high, uint32_t destination_pointer, uint32_t byte_length)", 5, "copy-the-exact-requested-range-or-copy-nothing"],
      ["bg_vfs_close", "int32_t bg_vfs_close(uint32_t handle)", 1, "close-one-live-file-handle-exactly-once"],
      ["bg_vfs_directory_count", "int32_t bg_vfs_directory_count(uint32_t path_pointer, uint32_t path_length, uint32_t count_pointer)", 3, "write-the-stable-immediate-child-count-for-one-directory"],
      ["bg_vfs_directory_entry", "int32_t bg_vfs_directory_entry(uint32_t path_pointer, uint32_t path_length, uint32_t index, uint32_t name_pointer, uint32_t name_capacity, uint32_t metadata_pointer)", 6, "write-one-byte-sorted-child-basename-and-one-32-byte-metadata-record"],
    ] as const;
    if (body.hostImports.moduleName !== "browsergrad_vfs_v1" ||
        body.hostImports.invocation !== "synchronous-non-reentrant" ||
        body.hostImports.pointerLifetime !== "only-for-import-call-duration" ||
        body.hostImports.unlistedApplicationImports !== "forbidden" ||
        body.hostImports.functions.length !== expectedImports.length) {
      invalid("$.body.hostImports", "host import surface does not match runtime v1");
    }
    const generatedImports = body.hostImports.generatedImportAllowlist;
    if (generatedImports.policyId !== "browsergrad.compiler.cpp-cute.emscripten-generated-imports@1" ||
        generatedImports.status !== "independently-reviewed-hash-pinned" ||
        generatedImports.allowlistSha256 !==
          CPP_CUTE_BROWSER_RUNTIME_ABI_V1_GENERATED_IMPORT_ALLOWLIST_SHA256 ||
        generatedImports.exactFunctions.length !== 52 ||
        generatedImports.unlistedGeneratedImports !== "forbidden" ||
        generatedImports.observedModuleCannotExtendAllowlist !== true ||
        generatedImports.capabilityCeiling !==
          "no-clock-random-network-process-or-ambient-filesystem" ||
        generatedImports.releaseConformance !==
          "allowed-only-for-exact-hash-pinned-signatures") {
      invalid(
        "$.body.hostImports.generatedImportAllowlist",
        "generated imports must equal the independently reviewed closed runtime policy",
      );
    }
    const generatedReview = generatedImports.independentReview;
    if (generatedReview.basis !== "pinned-emscripten-runtime-sources-and-locked-link-flags" ||
        generatedReview.emscriptenVersion !== "6.0.3" ||
        generatedReview.emscriptenCommit !== "283e2d130132859fde6a4e4c87fd254b38127651") {
      invalid(
        "$.body.hostImports.generatedImportAllowlist.independentReview",
        "generated-import review must bind the exact selected Emscripten runtime",
      );
    }
    assertExactStrings(generatedReview.lockedFlags, [
      "-fexceptions",
      "-sSTACK_OVERFLOW_CHECK=2",
      "-sALLOW_MEMORY_GROWTH=1",
      "-sFILESYSTEM=0",
      "-sINCOMING_MODULE_JS_API=['instantiateWasm','onAbort','print','printErr']",
    ], "$.body.hostImports.generatedImportAllowlist.independentReview.lockedFlags");
    const expectedGeneratedRoles = [
      ["javascript-exception-control-flow", 48, "none"],
      ["bounded-memory-growth", 2, "none"],
      ["stack-overflow-trap", 1, "none"],
      ["stdout-stderr-only", 1, "caller-provided-output-hooks-only"],
    ] as const;
    if (generatedReview.runtimeRoles.length !== expectedGeneratedRoles.length ||
        generatedReview.runtimeRoles.some((role, index) => {
          const expected = expectedGeneratedRoles[index];
          return expected === undefined || role.name !== expected[0] ||
            role.exactFunctionCount !== expected[1] || role.ambientCapability !== expected[2];
        })) {
      invalid(
        "$.body.hostImports.generatedImportAllowlist.independentReview.runtimeRoles",
        "generated-import runtime-role inventory differs from the independent review",
      );
    }
    const generatedRoleCounts = new Map<string, number>();
    for (const [index, entry] of generatedImports.exactFunctions.entries()) {
      if ((entry.moduleName !== "env" && entry.moduleName !== "wasi_snapshot_preview1") ||
          entry.fieldName.length === 0 || entry.wasmResults.length > 1 ||
          [...entry.wasmParameters, ...entry.wasmResults].some((value) =>
            value !== "f32" && value !== "f64" && value !== "i32" && value !== "i64")) {
        invalid(
          `$.body.hostImports.generatedImportAllowlist.exactFunctions[${index}]`,
          "generated import has an invalid module, name, or core Wasm signature",
        );
      }
      generatedRoleCounts.set(entry.runtimeRole, (generatedRoleCounts.get(entry.runtimeRole) ?? 0) + 1);
    }
    for (const [name, count] of expectedGeneratedRoles.map((entry) => [entry[0], entry[1]] as const)) {
      if (generatedRoleCounts.get(name) !== count) {
        invalid(
          "$.body.hostImports.generatedImportAllowlist.exactFunctions",
          `generated import runtime role ${name} must contain exactly ${count} functions`,
        );
      }
    }
    const outputImports = generatedImports.exactFunctions.filter((entry) =>
      entry.runtimeRole === "stdout-stderr-only");
    if (outputImports.length !== 1 || outputImports[0]?.moduleName !== "wasi_snapshot_preview1" ||
        outputImports[0].fieldName !== "fd_write" ||
        outputImports[0].wasmParameters.join(",") !== "i32,i32,i32,i32" ||
        outputImports[0].wasmResults.join(",") !== "i32") {
      invalid(
        "$.body.hostImports.generatedImportAllowlist.exactFunctions",
        "the only reviewed output import must be the exact WASI fd_write signature",
      );
    }
    const memoryAccess = body.hostImports.memoryAccess;
    if (memoryAccess.rangeArithmetic !== "checked-u32-no-wrap" ||
        memoryAccess.completeRangeValidation !==
          "required-before-first-input-read-or-output-write" ||
        memoryAccess.pathInputSnapshot !==
          "after-complete-range-validation-before-any-output-write" ||
        memoryAccess.inputOutputOverlap !==
          "allowed-only-after-complete-input-snapshot" ||
        memoryAccess.outputOutputOverlap !== "forbidden" ||
        memoryAccess.memoryGrowthDuringImport !== "forbidden" ||
        memoryAccess.invalidRangeMemoryMutation !== "forbidden" ||
        memoryAccess.alignmentByteLength.byteOutput !== 1 ||
        memoryAccess.alignmentByteLength.u32Output !== 4 ||
        memoryAccess.alignmentByteLength.u64ContainingRecord !== 8) {
      invalid(
        "$.body.hostImports.memoryAccess",
        "host imports require checked ranges, stable snapshots, aligned disjoint outputs, and failure atomicity",
      );
    }
    for (const [index, expected] of expectedImports.entries()) {
      const actual = body.hostImports.functions[index];
      if (actual?.ordinal !== index || actual.fieldName !== expected[0] ||
          actual.cSignature !== expected[1] || actual.wasmParameters.length !== expected[2] ||
          actual.wasmResults.length !== 1 || actual.semantics !== expected[3] ||
          [...actual.wasmParameters, ...actual.wasmResults].some((type) => type !== "i32")) {
        invalid(`$.body.hostImports.functions[${index}]`, "VFS import signature does not match runtime v1");
      }
    }
    if (body.vfs.storage !== "host-backed-lazy-verified-pack-files-only" ||
        body.vfs.physicalFilesystemFallback !== "forbidden" ||
        body.vfs.networkFallback !== "forbidden" || body.vfs.pathEncoding !== "utf8" ||
        body.vfs.pathForm !== "canonical-absolute-forward-slash-no-nul-dot-or-parent-segments" ||
        body.vfs.maxPathByteLength !== 4_096 || body.vfs.maxIndexedNodes !== 262_144 ||
        body.vfs.maxIndexLogicalByteLength !== 134_217_728 ||
        body.vfs.indexLogicalByteAccounting !==
          "sum-per-node-metadata-record-plus-canonical-path-utf8-plus-immediate-basename-utf8" ||
        body.vfs.maxAggregateLiveOpenByteLength !== 402_653_184 ||
        body.vfs.liveOpenByteAccounting !==
          "logical-full-file-per-live-handle-reservation-not-wasm-residency" ||
        body.vfs.maxLiveFileHandles !== 65_536 ||
        body.vfs.maxSessionCalls !== 1_000_000 ||
        body.vfs.directoryOrder !== "strict-ascending-utf8-byte-order" ||
        body.vfs.failureAtomicity !==
          "nonzero-status-writes-no-output-except-required-name-length-in-metadata") {
      invalid("$.body.vfs", "VFS storage, path, ordering, or failure semantics differ from runtime v1");
    }
    if (body.vfs.metadataRecord.byteLength !== 32 || body.vfs.openResultRecord.byteLength !== 16) {
      invalid("$.body.vfs", "VFS binary records have incorrect byte lengths");
    }
    assertExactStrings(body.vfs.metadataRecord.fields.map((entry) => entry.name), [
      "kind", "nameByteLength", "fileByteLength", "uniqueIdDevice", "uniqueIdFile",
    ], "$.body.vfs.metadataRecord.fields[*].name");
    assertExactNumbers(body.vfs.metadataRecord.fields.map((entry) => entry.offset), [
      0, 4, 8, 16, 24,
    ], "$.body.vfs.metadataRecord.fields[*].offset");
    assertExactStrings(body.vfs.metadataRecord.fields.map((entry) => entry.encoding), [
      "u32le", "u32le", "u64le", "u64le", "u64le",
    ], "$.body.vfs.metadataRecord.fields[*].encoding");
    assertExactStrings(body.vfs.metadataRecord.fields.flatMap((entry) => entry.values), [
      "file=1", "directory=2", "zero-for-status", "zero-for-directory", "session-stable",
      "session-stable",
    ], "$.body.vfs.metadataRecord.fields[*].values");
    assertExactStrings(body.vfs.openResultRecord.fields.map((entry) => entry.name), [
      "handle", "reserved", "fileByteLength",
    ], "$.body.vfs.openResultRecord.fields[*].name");
    assertExactNumbers(body.vfs.openResultRecord.fields.map((entry) => entry.offset), [
      0, 4, 8,
    ], "$.body.vfs.openResultRecord.fields[*].offset");
    assertExactStrings(body.vfs.openResultRecord.fields.map((entry) => entry.encoding), [
      "u32le", "u32le-zero", "u64le",
    ], "$.body.vfs.openResultRecord.fields[*].encoding");
    assertContiguousCodes(body.vfs.statuses, 0, "$.body.vfs.statuses");
    assertExactStrings(body.vfs.statuses.map((entry) => entry.name), [
      "ok", "not-found", "not-directory", "is-directory", "invalid-path", "buffer-too-small",
      "out-of-range", "invalid-handle", "resource-limit", "session-closed", "internal-error",
    ], "$.body.vfs.statuses[*].name");
    assertExactStrings(body.vfs.noAmbientInputs, [
      "clock", "current-directory", "environment", "locale", "random", "timezone",
    ], "$.body.vfs.noAmbientInputs");
    assertUniqueCodes(body.compileStatuses, "$.body.compileStatuses");
    if (body.inputFrame.magicAscii !== "BGCCABI1" || body.inputFrame.headerByteLength !== 64 ||
        body.inputFrame.alignmentByteLength !== 8 ||
        body.inputFrame.maxFrameByteLength !== memory.maxInputFrameByteLength) {
      invalid("$.body.inputFrame", "input frame does not match the fixed runtime-v1 framing contract");
    }
    const decodeLimits = body.inputFrame.decodeLimits;
    if (decodeLimits.maxDocumentByteLength !== body.inputFrame.maxFrameByteLength ||
        decodeLimits.maxNestingDepth !== 128 ||
        decodeLimits.maxNodeCount !== 1_000_000 ||
        decodeLimits.maxCumulativeStringByteLength !== body.inputFrame.maxFrameByteLength ||
        decodeLimits.maxArrayElementCount !== 65_536 ||
        decodeLimits.maxObjectPropertyCount !== 512 ||
        decodeLimits.maxScratchByteLength !== body.inputFrame.maxFrameByteLength * 4 ||
        decodeLimits.maxScratchByteLength > memory.maxCompilerWorkingByteLength) {
      invalid(
        "$.body.inputFrame.decodeLimits",
        "input-frame decoder resource ceilings differ from the fixed runtime-v1 contract",
      );
    }
    if (decodeLimits.accounting.documentBytes !== "per-region-before-utf8-decode" ||
        decodeLimits.accounting.nestingNodesAndStrings !==
          "per-region-root-depth-one-strings-include-object-keys-and-values" ||
        decodeLimits.accounting.containers !== "per-array-or-object" ||
        decodeLimits.accounting.scratchBytes !==
          "peak-live-decoder-owned-bytes-per-compile-session-excluding-input-frame-vfs-and-producer-state" ||
        decodeLimits.numberPolicy !==
          "safe-integer-lexemes-only-no-negative-zero-fraction-or-exponent" ||
        decodeLimits.canonicalValidationPolicy !==
          "byte-exact-browsergrad-canonical-json-validation-per-region-rejecting-duplicate-keys") {
      invalid(
        "$.body.inputFrame.decodeLimits",
        "input-frame decoder accounting or canonical JSON policy differs from runtime v1",
      );
    }
    const expectedFieldNames = [
      "magic", "major", "minor", "headerByteLength", "totalByteLength", "flags",
      "profileOffset", "profileByteLength", "requestOffset", "requestByteLength", "reserved",
    ];
    const expectedOffsets = [0, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40];
    if (body.inputFrame.fields.length !== expectedOffsets.length ||
        body.inputFrame.fields.some((entry, index) =>
          entry.name !== expectedFieldNames[index] || entry.offset !== expectedOffsets[index])) {
      invalid("$.body.inputFrame.fields", "input-frame fields do not match the fixed header layout");
    }
    assertExactStrings(body.inputFrame.fields.map((entry) => entry.encoding), [
      "ascii[8]", "u16le", "u16le", "u32le", "u32le", "u32le-zero",
      "u32le-must-equal-64", "u32le", "u32le-aligned-8", "u32le", "zero[24]",
    ], "$.body.inputFrame.fields[*].encoding");
    if (body.inputFrame.encoding !== "little-endian-binary-header-with-canonical-json-regions" ||
        body.inputFrame.profileRegion !== "exact-canonical-prepared-frontend-profile-json" ||
        body.inputFrame.requestRegion !== "exact-canonical-producer-neutral-frontend-request-json" ||
        body.inputFrame.regionOrder !== "profile-then-zero-padding-then-request-then-zero-padding" ||
        body.inputFrame.totalLengthRule !==
          "aligned-request-end-equals-total-byte-length-and-all-regions-are-in-bounds" ||
        body.inputFrame.sourceBytes !== "out-of-band-through-worker-owned-vfs-session" ||
        body.inputFrame.compileReadRule !==
          "synchronous-complete-frame-validation-before-vfs-access") {
      invalid("$.body.inputFrame", "input-frame region or validation semantics differ from runtime v1");
    }
    assertExactNumbers(body.compileStatuses.map((entry) => entry.code), [
      0, 1, 2, 100, 101, 102, 103, 104, 105, 106,
    ], "$.body.compileStatuses[*].code");
    assertExactStrings(body.compileStatuses.map((entry) => entry.name), [
      "artifact-ready", "idle", "input-allocated", "invalid-state", "invalid-argument",
      "invalid-frame", "abi-mismatch", "vfs-error", "resource-limit", "internal-error",
    ], "$.body.compileStatuses[*].name");
    assertExactStrings(body.compileStatuses.map((entry) => entry.retry), [
      "reset-then-new-invocation", "allocate-input", "compile-or-free", "reset-required",
      "reset-required", "reset-required", "module-must-not-be-reused", "reset-required",
      "reset-required", "module-must-not-be-reused",
    ], "$.body.compileStatuses[*].retry");
    if (body.lifecycle.initialState !== "idle") invalid("$.body.lifecycle.initialState", "initial state must be idle");
    assertExactStrings(body.lifecycle.states, [
      "idle", "input-allocated", "compiling-internal", "artifact-ready", "failed",
    ], "$.body.lifecycle.states");
    assertExactStrings(body.lifecycle.rules, [
      "alloc-is-valid-only-in-idle-with-byte-length-from-1-through-max-input-frame-byte-length",
      "only-one-live-input-allocation-is-permitted",
      "compile-is-valid-only-for-the-exact-live-input-pointer-and-length",
      "compile-is-synchronous-and-not-reentrant",
      "compile-status-zero-means-one-complete-canonical-artifact-is-ready-even-when-that-artifact-rejects-source",
      "infrastructure-failure-never-masquerades-as-a-rejected-source-artifact",
      "free-is-valid-only-for-the-exact-live-input-pointer-and-length-and-never-for-result-memory",
      "free-after-success-preserves-the-artifact-ready-result",
      "result-getters-return-zero-unless-state-is-artifact-ready",
      "result-bytes-are-immutable-until-reset",
      "reset-releases-input-result-and-module-side-vfs-state-and-returns-to-idle",
      "after-abi-mismatch-or-internal-error-the-worker-discards-the-module-instance-after-reading-status",
      "wasm-trap-abort-or-out-of-memory-is-a-worker-infrastructure-failure-with-no-readable-status-guarantee",
    ], "$.body.lifecycle.rules");
    if (body.result.maximumByteLength !== memory.maxResultByteLength ||
        body.result.schema !== "browsergrad.compiler.cpp-cute.frontend-artifact" ||
        body.result.version.major !== 3 || body.result.version.minor !== 0 ||
        body.result.encoding !== "canonical-json-bytes" ||
        body.result.ownership !== "module-owned-worker-must-copy-before-reset" ||
        body.result.lifetime !== "from-artifact-ready-until-reset-or-worker-termination" ||
        body.result.emptyResult !== "forbidden-when-status-is-artifact-ready") {
      invalid("$.body.result", "result contract does not match canonical artifact v3.0");
    }
    if (body.cancellation.mechanism !== "terminate-dedicated-worker" ||
        body.cancellation.cooperativeImport !== "forbidden" ||
        body.cancellation.reason !==
          "synchronous-unshared-wasm-cannot-service-worker-messages-during-compile" ||
        body.cancellation.effect !==
          "invalidates-module-memory-input-result-vfs-handles-and-session-authority" ||
        body.cancellation.workerReuseAfterCancellation !== "forbidden") {
      invalid("$.body.cancellation", "runtime v1 cancellation must terminate and discard the Worker");
    }
  } catch (cause) {
    if (cause instanceof CppCuteBrowserRuntimeAbiManifestError) throw cause;
    invalid("$.body", "runtime-ABI body violates required invariants", { cause });
  }
}

function assertContiguousCodes(
  values: readonly { readonly code: number }[],
  first: number,
  path: string,
): void {
  if (values.some((entry, index) => entry.code !== first + index)) {
    invalid(path, "status codes must be contiguous and ordered");
  }
}

function assertUniqueCodes(values: readonly { readonly code: number }[], path: string): void {
  const codes = values.map((entry) => entry.code);
  if (new Set(codes).size !== codes.length || codes.some((code) => !Number.isSafeInteger(code) || code < 0)) {
    invalid(path, "status codes must be unique nonnegative safe integers");
  }
}

function assertExactStrings(actual: readonly string[], expected: readonly string[], path: string): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    invalid(path, "values do not equal the exact ordered runtime-v1 inventory");
  }
}

function assertExactNumbers(actual: readonly number[], expected: readonly number[], path: string): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    invalid(path, "numbers do not equal the exact ordered runtime-v1 inventory");
  }
}

async function deriveGeneratedImportAllowlistSha256(
  body: CppCuteBrowserRuntimeAbiBodyV1,
): Promise<string> {
  const policy = body.hostImports.generatedImportAllowlist;
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.emscripten-generated-import-allowlist.v1",
    policyId: policy.policyId,
    exactFunctions: policy.exactFunctions,
  }, "$.body.hostImports.generatedImportAllowlist.allowlistSha256");
}

async function deriveSupportFunctionAllowlistSha256(
  body: CppCuteBrowserRuntimeAbiBodyV1,
): Promise<string> {
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.emscripten-support-function-export-allowlist.v1",
    exactFunctions: body.wasm.supportExports.exactFunctionAllowlist,
  }, "$.body.wasm.supportExports.functionAllowlistSha256");
}

function storedManifest(
  prepared: PreparedCppCuteBrowserRuntimeAbiManifest,
): StoredCppCuteBrowserRuntimeAbiManifest {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_MANIFESTS.get(prepared as object);
  if (stored === undefined) unverified();
  return stored;
}

function snapshotBytes(value: unknown): Uint8Array {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$bytes", "runtime-ABI manifest must be an unshared plain Uint8Array", { cause });
  }
  if (inspected.byteLength === 0) invalid("$bytes", "runtime-ABI manifest bytes must be nonempty");
  if (inspected.byteLength > CPP_CUTE_BROWSER_RUNTIME_ABI_BYTE_LIMIT) {
    resource("$bytes", `runtime-ABI manifest exceeds ${CPP_CUTE_BROWSER_RUNTIME_ABI_BYTE_LIMIT} bytes`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid("$bytes", "runtime-ABI manifest bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(options: DecodeCppCuteBrowserRuntimeAbiManifestOptions): AbortSignal | undefined {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(options);
    descriptors = Object.getOwnPropertyDescriptors(options);
  } catch (cause) {
    invalid("$options", "options must be an inspectable plain object", { cause });
  }
  if (typeof options !== "object" || options === null || prototype !== Object.prototype) {
    invalid("$options", "options must be a plain object");
  }
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length > 1 || keys.some((key) => key !== "signal")) invalid("$options", "options contain unknown fields");
  const descriptor = descriptors.signal;
  if (descriptor !== undefined && (descriptor.enumerable !== true || !("value" in descriptor))) {
    invalid("$options.signal", "signal must be an enumerable data property");
  }
  const signal = descriptor?.value as unknown;
  if (signal !== undefined && !isAbortSignal(signal)) invalid("$options.signal", "signal must be an AbortSignal");
  return signal;
}

function isAbortSignal(value: unknown): value is AbortSignal {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) return false;
  try {
    return typeof ABORT_SIGNAL_ABORTED_GETTER.call(value) === "boolean";
  } catch {
    return false;
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  let aborted: unknown;
  try {
    aborted = ABORT_SIGNAL_ABORTED_GETTER?.call(signal);
  } catch (cause) {
    invalid("$options.signal", "signal is not a readable AbortSignal", { cause });
  }
  if (aborted === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-CANCELLED",
      "$options.signal",
      "runtime-ABI manifest preparation was cancelled",
    );
  }
}

function closedObject(value: JsonValue, keys: readonly string[], path: string, requireAll: boolean): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected object");
  const object = value as JsonObject;
  for (const key of Object.keys(object)) if (!keys.includes(key)) invalid(path, `unknown field ${key}`);
  if (requireAll) {
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(object, key)) invalid(`${path}.${key}`, "required field is missing");
    }
  }
  return object;
}

function field(value: JsonObject, key: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(value, key)) invalid(`${path}.${key}`, "required field is missing");
  return value[key] as JsonValue;
}

function boundedPattern(value: JsonValue, path: string, pattern: RegExp): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 128 || !pattern.test(value)) {
    invalid(path, "string does not match required closed format");
  }
  return value;
}

function literal<T extends string>(value: JsonValue, expected: T, path: string): asserts value is T {
  if (value !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
}

function canonicalResourceBytes(value: JsonValue): Uint8Array {
  try {
    return canonicalJsonBytes(value, { limits: CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$", "canonical runtime-ABI manifest exceeds fixed limits", { cause });
    invalid("$", "runtime-ABI manifest cannot be canonically encoded", { cause });
  }
}

async function hashJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, { limits: CPP_CUTE_BROWSER_RUNTIME_ABI_DECODE_LIMITS });
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable(path, cause);
    if (isSchemaResourceLimit(cause)) resource(path, "hash projection exceeds fixed limits", { cause });
    invalid(path, "hash projection is invalid", { cause });
  }
}

async function hashBytes(value: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(value);
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable(path, cause);
    invalid(path, "SHA-256 calculation failed", { cause });
  }
}

function isHashUnavailable(cause: unknown): boolean {
  return cause instanceof Error && /Web Crypto|crypto\.subtle|SHA-256 unavailable/iu.test(cause.message);
}

function isSchemaResourceLimit(cause: unknown): boolean {
  return cause instanceof SemanticSchemaError && cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-INVALID", path, message, options);
}

function unsupported(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-UNSUPPORTED-VERSION", path, message);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-RESOURCE-LIMIT", path, message, options);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-HASH-MISMATCH", path, message);
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable",
    { cause },
  );
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-RUNTIME-ABI-UNVERIFIED",
    "$prepared",
    "runtime-ABI manifest authority was not created by this module instance",
  );
}

function fail(
  code: CppCuteBrowserRuntimeAbiManifestErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserRuntimeAbiManifestError(code, path, message, options);
}
