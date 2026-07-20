import {
  canonicalJsonBytes,
  decodeWireJson,
  sha256Hex,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteBrowserAssetSet,
  unwrapVerifiedCppCuteBrowserRuntimeAbiAsset,
  type VerifiedCppCuteBrowserAssetSet,
  type VerifiedCppCuteBrowserRuntimeAbiAsset,
} from "./cpp_cute_browser_asset_installation.js";
import {
  unwrapPreparedCppCuteBrowserAssetManifest,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
  type PreparedCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  inspectObservedCppCuteBrowserPackageWasmConformance,
  unwrapObservedCppCuteBrowserPackageWasmConformance,
  type ObservedCppCuteBrowserPackageWasmConformance,
} from "./cpp_cute_browser_wasm_verifier_controller.js";
import { CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID } from
  "./cpp_cute_browser_wasm_verifier_bundle.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
} from "./resources/cpp_cute_browser_wasm_verifier_bundle_v1.js";

export const CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_SCHEMA =
  "browsergrad.compiler.cpp-cute.package-wasm-verifier-evidence";
export const CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MAJOR = 1;
export const CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MINOR = 0;
export const CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_BYTE_LIMIT = 16 * 1024;

const SHA256 = /^[0-9a-f]{64}$/u;
const EVIDENCE_ID = /^bg\.cpp\.browser-wasm-verifier-conformance\.sha256\.[0-9a-f]{64}$/u;
const VERIFIER_REQUEST_ID = /^bg\.cpp\.browser-wasm-verifier-request\.sha256\.[0-9a-f]{64}$/u;
const ASSET_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const WIRE_U64 = /^(?:0|[1-9][0-9]*)$/u;
const U64_MAX = (1n << 64n) - 1n;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_OBJECT_KEYS = Object.keys;
const NATIVE_OBJECT_CREATE = Object.create;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const CAPTURED_OBJECT = Object;
const CAPTURED_OBJECT_PROTOTYPE = Object.prototype;
const CAPTURED_REFLECT = Reflect;
const NATIVE_ARRAY_IS_ARRAY = Array.isArray;
const NATIVE_REGEXP_TEST = RegExp.prototype.test;
const NATIVE_NUMBER_IS_SAFE_INTEGER = Number.isSafeInteger;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const CAPTURED_BIG_INT = BigInt;
const CAPTURED_NUMBER = Number;
const CAPTURED_STRING = String;
const CAPTURED_UINT8_ARRAY = Uint8Array;
const CAPTURED_WEAK_MAP = WeakMap;
const EVIDENCES = new CAPTURED_WEAK_MAP<object, StoredEvidence>();
const EVIDENCE_LIMITS: DecodeLimits = NATIVE_OBJECT_FREEZE({
  maxDocumentBytes: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_BYTE_LIMIT,
  maxDepth: 4,
  maxNodes: 64,
  maxStringBytes: 8 * 1024,
  maxArrayLength: 1,
  maxObjectProperties: 40,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 100_000,
});

export interface CppCuteBrowserWasmVerifierEvidenceV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MAJOR;
    readonly minor: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MINOR;
  };
  readonly sourceEvidenceId: string;
  readonly verifierBundleId: string;
  readonly verifierRequestId: string;
  readonly verifierInvocationNonceSha256: string;
  readonly verifierModuleSha256: string;
  readonly verifierModuleByteLength: string;
  readonly assetManifestId: string;
  readonly assetManifestSha256: string;
  readonly assetSetSha256: string;
  readonly wasmAssetId: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly runtimeAbiResourceSha256: string;
  readonly observedProjectionSha256: string;
  readonly reportSha256: string;
  readonly reportByteLength: string;
  readonly acceptedTerminalMessages: "1";
  readonly verifierWorkerExecutionObserved: true;
  readonly rawWasmVerified: true;
  readonly exactInterfaceConformanceObserved: true;
  readonly packageOwnedVerifier: true;
  readonly sourceProductionConformanceAuthorityMinted: true;
  readonly compilerWorkerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
  readonly releaseReady: false;
}

declare const verifierEvidenceBrand: unique symbol;

/**
 * Opaque binding to canonical copy-safe evidence. It is derivative input
 * authority only and cannot stand in for the host-realm conformance object.
 */
export interface PreparedCppCuteBrowserWasmVerifierEvidence {
  readonly [verifierEvidenceBrand]: true;
  readonly authority:
    | "host-observed-verifier-evidence-region-binding"
    | "worker-reconstructed-verifier-evidence-region-binding";
  readonly sourceEvidenceId: string;
  readonly regionSha256: string;
  readonly regionByteLength: number;
  readonly wasmAssetId: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly runtimeAbiResourceSha256: string;
  readonly observedProjectionSha256: string;
  readonly sourceHostVerifierExecutionReported: true;
  readonly hostVerifierExecutionLocallyObserved: false;
  readonly workerLocalVerifierExecutionObserved: false;
  readonly productionConformanceAuthorityMinted: false;
  readonly releaseReady: false;
}

export interface PreparedCppCuteBrowserWasmVerifierEvidenceRecord {
  readonly evidence: CppCuteBrowserWasmVerifierEvidenceV1;
  readonly assetSet: VerifiedCppCuteBrowserAssetSet;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
  readonly runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest;
  readonly sourceObservedConformance: ObservedCppCuteBrowserPackageWasmConformance | null;
  readonly workerReconstructed: boolean;
  readonly productionAuthority: false;
  readonly releaseReady: false;
}

interface StoredEvidence {
  readonly canonicalBytes: Uint8Array;
  readonly record: PreparedCppCuteBrowserWasmVerifierEvidenceRecord;
}

export interface CppCuteBrowserWasmVerifierEvidenceBindingInput {
  readonly assetSet: VerifiedCppCuteBrowserAssetSet;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
}

export async function prepareCppCuteBrowserWasmVerifierEvidence(
  observed: ObservedCppCuteBrowserPackageWasmConformance,
  binding: CppCuteBrowserWasmVerifierEvidenceBindingInput,
): Promise<PreparedCppCuteBrowserWasmVerifierEvidence> {
  const exact = exactBinding(binding);
  const inspection = inspectObservedCppCuteBrowserPackageWasmConformance(observed);
  const observedRecord = unwrapObservedCppCuteBrowserPackageWasmConformance(observed);
  if (observedRecord.assetSet !== exact.assetSet ||
      observedRecord.assetManifest !== exact.assetManifest ||
      observedRecord.runtimeAbiAsset !== exact.runtimeAbiAsset ||
      observedRecord.runtimeAbi !== exact.runtimeAbi) {
    mismatch("$.observed", "host verifier authority belongs to different exact asset authorities");
  }
  const evidence = freezeEvidence({
    schema: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MAJOR,
      minor: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MINOR,
    },
    sourceEvidenceId: inspection.evidenceId,
    verifierBundleId: inspection.verifierBundleId,
    verifierRequestId: inspection.requestId,
    verifierInvocationNonceSha256: inspection.invocationNonceSha256,
    verifierModuleSha256: inspection.verifierModuleSha256,
    verifierModuleByteLength: wireFromNumber(inspection.verifierModuleByteLength),
    assetManifestId: inspection.assetManifestId,
    assetManifestSha256: exact.assetManifest.manifestSha256,
    assetSetSha256: inspection.assetSetSha256,
    wasmAssetId: inspection.wasmAssetId,
    wasmSha256: inspection.wasmSha256,
    wasmByteLength: wireFromNumber(inspection.wasmByteLength),
    runtimeAbiManifestId: inspection.runtimeAbiManifestId,
    runtimeAbiContractSha256: inspection.runtimeAbiContractSha256,
    runtimeAbiResourceSha256: inspection.runtimeAbiResourceSha256,
    observedProjectionSha256: inspection.observedProjectionSha256,
    reportSha256: inspection.reportSha256,
    reportByteLength: wireFromNumber(inspection.reportByteLength),
    acceptedTerminalMessages: "1",
    verifierWorkerExecutionObserved: true,
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    packageOwnedVerifier: true,
    sourceProductionConformanceAuthorityMinted: true,
    compilerWorkerExecutionObserved: false,
    loweringAuthorityMinted: false,
    releaseReady: false,
  });
  validateEvidenceBindings(evidence, exact);
  return issueEvidence(evidence, exact, observed, false);
}

export async function decodeCppCuteBrowserWasmVerifierEvidence(
  bytes: Uint8Array,
  binding: CppCuteBrowserWasmVerifierEvidenceBindingInput,
  expectedRegionSha256: string,
): Promise<PreparedCppCuteBrowserWasmVerifierEvidence> {
  const exact = exactBinding(binding);
  const snapshot = new CAPTURED_UINT8_ARRAY(bytes);
  if (snapshot.byteLength === 0 ||
      snapshot.byteLength > CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_BYTE_LIMIT) {
    invalid("$.verifierEvidenceRegionBytes", "verifier evidence region exceeds its byte ceiling");
  }
  let value: JsonValue;
  try {
    value = decodeWireJson(snapshot, { limits: EVIDENCE_LIMITS });
  } catch (cause) {
    invalid("$.verifierEvidenceRegionBytes", "verifier evidence is not bounded strict JSON", {
      cause,
    });
  }
  const evidence = parseEvidence(value);
  const canonical = canonicalJsonBytes(evidence, { limits: EVIDENCE_LIMITS });
  if (!equalBytes(snapshot, canonical)) {
    invalid("$.verifierEvidenceRegionBytes", "verifier evidence bytes are not canonical JSON");
  }
  validateEvidenceBindings(evidence, exact);
  const prepared = await issueEvidence(evidence, exact, null, true);
  if (!NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, SHA256, [expectedRegionSha256]) ||
      prepared.regionSha256 !== expectedRegionSha256) {
    mismatch(
      "$.verifierEvidenceRegionBytes",
      "canonical verifier evidence differs from the host invocation region binding",
    );
  }
  return prepared;
}

export function canonicalCppCuteBrowserWasmVerifierEvidenceBytes(
  prepared: PreparedCppCuteBrowserWasmVerifierEvidence,
): Uint8Array {
  return new CAPTURED_UINT8_ARRAY(stored(prepared).canonicalBytes);
}

export function unwrapPreparedCppCuteBrowserWasmVerifierEvidence(
  prepared: PreparedCppCuteBrowserWasmVerifierEvidence,
): PreparedCppCuteBrowserWasmVerifierEvidenceRecord {
  return stored(prepared).record;
}

async function issueEvidence(
  evidence: CppCuteBrowserWasmVerifierEvidenceV1,
  binding: ExactBinding,
  observed: ObservedCppCuteBrowserPackageWasmConformance | null,
  workerReconstructed: boolean,
): Promise<PreparedCppCuteBrowserWasmVerifierEvidence> {
  const canonicalBytes = canonicalJsonBytes(evidence, { limits: EVIDENCE_LIMITS });
  const regionSha256 = await sha256Hex(canonicalBytes);
  const prepared = NATIVE_OBJECT_FREEZE({
    authority: workerReconstructed
      ? "worker-reconstructed-verifier-evidence-region-binding"
      : "host-observed-verifier-evidence-region-binding",
    sourceEvidenceId: evidence.sourceEvidenceId,
    regionSha256,
    regionByteLength: canonicalBytes.byteLength,
    wasmAssetId: evidence.wasmAssetId,
    wasmSha256: evidence.wasmSha256,
    wasmByteLength: CAPTURED_NUMBER(parseWire(evidence.wasmByteLength, "$.verifierEvidence.wasmByteLength")),
    runtimeAbiManifestId: evidence.runtimeAbiManifestId,
    runtimeAbiContractSha256: evidence.runtimeAbiContractSha256,
    runtimeAbiResourceSha256: evidence.runtimeAbiResourceSha256,
    observedProjectionSha256: evidence.observedProjectionSha256,
    sourceHostVerifierExecutionReported: true,
    hostVerifierExecutionLocallyObserved: false,
    workerLocalVerifierExecutionObserved: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  }) as PreparedCppCuteBrowserWasmVerifierEvidence;
  const record = NATIVE_OBJECT_FREEZE({
    evidence,
    assetSet: binding.assetSet,
    assetManifest: binding.assetManifest,
    runtimeAbiAsset: binding.runtimeAbiAsset,
    runtimeAbi: binding.runtimeAbi,
    sourceObservedConformance: observed,
    workerReconstructed,
    productionAuthority: false,
    releaseReady: false,
  });
  weakMapSet(EVIDENCES, prepared, { canonicalBytes, record });
  return prepared;
}

interface ExactBinding extends CppCuteBrowserWasmVerifierEvidenceBindingInput {
  readonly runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest;
  readonly clangAsset: ReturnType<typeof unwrapPreparedCppCuteBrowserAssetManifest>["manifest"]["body"]["assets"][number];
}

function exactBinding(input: CppCuteBrowserWasmVerifierEvidenceBindingInput): ExactBinding {
  const values = exactDataRecord(input, "$.binding", [
    "assetSet", "assetManifest", "runtimeAbiAsset",
  ]);
  const assetSet = values["assetSet"] as VerifiedCppCuteBrowserAssetSet;
  const assetManifest = values["assetManifest"] as PreparedCppCuteBrowserAssetManifest;
  const runtimeAbiAsset = values["runtimeAbiAsset"] as VerifiedCppCuteBrowserRuntimeAbiAsset;
  const assetSetRecord = unwrapVerifiedCppCuteBrowserAssetSet(assetSet);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(assetManifest);
  const runtimeAbiRecord = unwrapVerifiedCppCuteBrowserRuntimeAbiAsset(runtimeAbiAsset);
  const runtimeAbi = runtimeAbiRecord.runtimeAbi;
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi);
  if (assetSetRecord.manifest !== assetManifest || runtimeAbiRecord.assetSet !== assetSet) {
    mismatch("$.binding", "verifier evidence authorities do not share one exact asset set");
  }
  let clangAsset: ExactBinding["clangAsset"] | undefined;
  for (let index = 0; index < manifestRecord.manifest.body.assets.length; index += 1) {
    const asset = manifestRecord.manifest.body.assets[index]!;
    if (asset.kind !== "clang-extractor-wasm") continue;
    if (clangAsset !== undefined) mismatch("$.binding.assetManifest", "Clang Wasm asset is not unique");
    clangAsset = asset;
  }
  if (clangAsset === undefined) mismatch("$.binding.assetManifest", "Clang Wasm asset is missing");
  return NATIVE_OBJECT_FREEZE({
    assetSet,
    assetManifest,
    runtimeAbiAsset,
    runtimeAbi,
    clangAsset,
  });
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null ||
      (NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_PROTOTYPE_OF, CAPTURED_OBJECT, [value]) !==
        CAPTURED_OBJECT_PROTOTYPE &&
       NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_PROTOTYPE_OF, CAPTURED_OBJECT, [value]) !== null)) {
    invalid(path, "expected plain data object");
  }
  const ownKeys = NATIVE_REFLECT_APPLY(
    NATIVE_REFLECT_OWN_KEYS,
    CAPTURED_REFLECT,
    [value],
  ) as PropertyKey[];
  if (ownKeys.length !== keys.length ||
      hasUnexpectedPropertyKey(ownKeys, keys)) {
    invalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  const descriptors = NATIVE_REFLECT_APPLY(
    NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS,
    CAPTURED_OBJECT,
    [value],
  ) as Record<PropertyKey, PropertyDescriptor>;
  const result = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_CREATE, CAPTURED_OBJECT, [null]) as Record<
    string,
    unknown
  >;
  for (let index = 0; index < keys.length; index += 1) {
    const key = keys[index]!;
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be one enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return NATIVE_OBJECT_FREEZE(result);
}

function hasUnexpectedPropertyKey(
  actual: readonly PropertyKey[],
  expected: readonly string[],
): boolean {
  for (let index = 0; index < actual.length; index += 1) {
    const key = actual[index]!;
    if (typeof key !== "string" || hasUnexpectedKey([key], expected)) return true;
  }
  return false;
}

function validateEvidenceBindings(evidence: CppCuteBrowserWasmVerifierEvidenceV1, binding: ExactBinding): void {
  if (evidence.verifierBundleId !== CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID ||
      evidence.verifierModuleSha256 !== CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256 ||
      parseWire(evidence.verifierModuleByteLength, "$.verifierEvidence.verifierModuleByteLength") !==
        CAPTURED_BIG_INT(CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH)) {
    mismatch("$.verifierEvidence.verifierBundle", "evidence does not name the exact package verifier bundle");
  }
  if (evidence.assetManifestId !== binding.assetManifest.manifestId ||
      evidence.assetManifestSha256 !== binding.assetManifest.manifestSha256 ||
      evidence.assetSetSha256 !== binding.assetManifest.assetSetSha256) {
    mismatch("$.verifierEvidence.assetManifest", "evidence differs from the exact local asset manifest");
  }
  if (evidence.wasmAssetId !== binding.clangAsset.assetId ||
      evidence.wasmSha256 !== binding.clangAsset.sha256 ||
      evidence.wasmByteLength !== binding.clangAsset.byteLength) {
    mismatch("$.verifierEvidence.clangWasm", "evidence differs from the exact local Clang Wasm asset");
  }
  if (evidence.runtimeAbiManifestId !== binding.runtimeAbi.manifestId ||
      evidence.runtimeAbiContractSha256 !== binding.runtimeAbi.contractSha256 ||
      evidence.runtimeAbiResourceSha256 !== binding.runtimeAbi.resourceSha256) {
    mismatch("$.verifierEvidence.runtimeAbi", "evidence differs from the exact local runtime ABI");
  }
}

function parseEvidence(value: JsonValue): CppCuteBrowserWasmVerifierEvidenceV1 {
  const keys = [
    "schema", "version", "sourceEvidenceId", "verifierBundleId", "verifierRequestId",
    "verifierInvocationNonceSha256", "verifierModuleSha256", "verifierModuleByteLength",
    "assetManifestId", "assetManifestSha256", "assetSetSha256", "wasmAssetId", "wasmSha256",
    "wasmByteLength", "runtimeAbiManifestId", "runtimeAbiContractSha256",
    "runtimeAbiResourceSha256", "observedProjectionSha256", "reportSha256",
    "reportByteLength", "acceptedTerminalMessages", "verifierWorkerExecutionObserved",
    "rawWasmVerified", "exactInterfaceConformanceObserved", "packageOwnedVerifier",
    "sourceProductionConformanceAuthorityMinted", "compilerWorkerExecutionObserved",
    "loweringAuthorityMinted", "releaseReady",
  ] as const;
  const root = closedObject(value, keys, "$.verifierEvidence");
  literal(root.schema, CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_SCHEMA, "$.verifierEvidence.schema");
  const version = closedObject(root.version, ["major", "minor"] as const, "$.verifierEvidence.version");
  literal(version.major, CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MAJOR, "$.verifierEvidence.version.major");
  literal(version.minor, CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MINOR, "$.verifierEvidence.version.minor");
  literal(root.verifierBundleId, CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID, "$.verifierEvidence.verifierBundleId");
  literal(root.acceptedTerminalMessages, "1", "$.verifierEvidence.acceptedTerminalMessages");
  literal(root.verifierWorkerExecutionObserved, true, "$.verifierEvidence.verifierWorkerExecutionObserved");
  literal(root.rawWasmVerified, true, "$.verifierEvidence.rawWasmVerified");
  literal(root.exactInterfaceConformanceObserved, true, "$.verifierEvidence.exactInterfaceConformanceObserved");
  literal(root.packageOwnedVerifier, true, "$.verifierEvidence.packageOwnedVerifier");
  literal(root.sourceProductionConformanceAuthorityMinted, true, "$.verifierEvidence.sourceProductionConformanceAuthorityMinted");
  literal(root.compilerWorkerExecutionObserved, false, "$.verifierEvidence.compilerWorkerExecutionObserved");
  literal(root.loweringAuthorityMinted, false, "$.verifierEvidence.loweringAuthorityMinted");
  literal(root.releaseReady, false, "$.verifierEvidence.releaseReady");
  return NATIVE_OBJECT_FREEZE({
    schema: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_SCHEMA,
    version: NATIVE_OBJECT_FREEZE({
      major: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MAJOR,
      minor: CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_MINOR,
    }),
    sourceEvidenceId: pattern(root.sourceEvidenceId, EVIDENCE_ID, "$.verifierEvidence.sourceEvidenceId"),
    verifierBundleId: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
    verifierRequestId: pattern(root.verifierRequestId, VERIFIER_REQUEST_ID, "$.verifierEvidence.verifierRequestId"),
    verifierInvocationNonceSha256: pattern(root.verifierInvocationNonceSha256, SHA256, "$.verifierEvidence.verifierInvocationNonceSha256"),
    verifierModuleSha256: pattern(root.verifierModuleSha256, SHA256, "$.verifierEvidence.verifierModuleSha256"),
    verifierModuleByteLength: wire(root.verifierModuleByteLength, "$.verifierEvidence.verifierModuleByteLength"),
    assetManifestId: boundedString(root.assetManifestId, "$.verifierEvidence.assetManifestId"),
    assetManifestSha256: pattern(root.assetManifestSha256, SHA256, "$.verifierEvidence.assetManifestSha256"),
    assetSetSha256: pattern(root.assetSetSha256, SHA256, "$.verifierEvidence.assetSetSha256"),
    wasmAssetId: pattern(root.wasmAssetId, ASSET_ID, "$.verifierEvidence.wasmAssetId"),
    wasmSha256: pattern(root.wasmSha256, SHA256, "$.verifierEvidence.wasmSha256"),
    wasmByteLength: wire(root.wasmByteLength, "$.verifierEvidence.wasmByteLength"),
    runtimeAbiManifestId: boundedString(root.runtimeAbiManifestId, "$.verifierEvidence.runtimeAbiManifestId"),
    runtimeAbiContractSha256: pattern(root.runtimeAbiContractSha256, SHA256, "$.verifierEvidence.runtimeAbiContractSha256"),
    runtimeAbiResourceSha256: pattern(root.runtimeAbiResourceSha256, SHA256, "$.verifierEvidence.runtimeAbiResourceSha256"),
    observedProjectionSha256: pattern(root.observedProjectionSha256, SHA256, "$.verifierEvidence.observedProjectionSha256"),
    reportSha256: pattern(root.reportSha256, SHA256, "$.verifierEvidence.reportSha256"),
    reportByteLength: wire(root.reportByteLength, "$.verifierEvidence.reportByteLength"),
    acceptedTerminalMessages: "1",
    verifierWorkerExecutionObserved: true,
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    packageOwnedVerifier: true,
    sourceProductionConformanceAuthorityMinted: true,
    compilerWorkerExecutionObserved: false,
    loweringAuthorityMinted: false,
    releaseReady: false,
  });
}

function freezeEvidence(value: CppCuteBrowserWasmVerifierEvidenceV1): CppCuteBrowserWasmVerifierEvidenceV1 {
  return NATIVE_OBJECT_FREEZE({ ...value, version: NATIVE_OBJECT_FREEZE({ ...value.version }) });
}

function closedObject<const K extends readonly string[]>(
  value: JsonValue,
  keys: K,
  path: string,
): Record<K[number], JsonValue> {
  if (typeof value !== "object" || value === null || NATIVE_ARRAY_IS_ARRAY(value)) invalid(path, "expected JSON object");
  const actual = NATIVE_OBJECT_KEYS(value);
  if (actual.length !== keys.length || hasUnexpectedKey(actual, keys)) {
    invalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  return value as Record<K[number], JsonValue>;
}

function hasUnexpectedKey(actual: readonly string[], expected: readonly string[]): boolean {
  for (let index = 0; index < actual.length; index += 1) {
    let found = false;
    for (let expectedIndex = 0; expectedIndex < expected.length; expectedIndex += 1) {
      if (actual[index] === expected[expectedIndex]) {
        found = true;
        break;
      }
    }
    if (!found) return true;
  }
  return false;
}

function wire(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected canonical unsigned wire integer");
  parseWire(value, path);
  return value;
}

function parseWire(value: string, path: string): bigint {
  if (!NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, WIRE_U64, [value])) {
    invalid(path, "expected canonical unsigned wire integer");
  }
  const parsed = CAPTURED_BIG_INT(value);
  if (parsed > U64_MAX) invalid(path, "wire integer exceeds u64");
  return parsed;
}

function wireFromNumber(value: number): string {
  if (!NATIVE_NUMBER_IS_SAFE_INTEGER(value) || value < 0) {
    invalid("$.verifierEvidence", "evidence byte length is not a safe unsigned integer");
  }
  return CAPTURED_STRING(value);
}

function pattern(value: JsonValue, expression: RegExp, path: string): string {
  if (typeof value !== "string" ||
      !NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, expression, [value])) {
    invalid(path, "string identity is invalid");
  }
  return value;
}

function boundedString(value: JsonValue, path: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 256) invalid(path, "expected bounded identity string");
  return value;
}

function literal(value: JsonValue, expected: JsonValue, path: string): void {
  if (value !== expected) mismatch(path, `expected exact literal ${CAPTURED_STRING(expected)}`);
}

function stored(prepared: PreparedCppCuteBrowserWasmVerifierEvidence): StoredEvidence {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const value = weakMapGet(EVIDENCES, prepared as object);
  if (value === undefined || prepared.productionConformanceAuthorityMinted !== false ||
      prepared.workerLocalVerifierExecutionObserved !== false || prepared.releaseReady !== false) {
    unverified();
  }
  return value;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) difference |= left[index]! ^ right[index]!;
  return difference === 0;
}

export class CppCuteBrowserWasmVerifierEvidenceError extends Error {
  constructor(readonly path: string, message: string, options?: ErrorOptions) {
    super(`BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-EVIDENCE: ${message}`, options);
    this.name = "CppCuteBrowserWasmVerifierEvidenceError";
  }
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierEvidenceError(path, message, options);
}

function mismatch(path: string, message: string): never {
  throw new CppCuteBrowserWasmVerifierEvidenceError(path, message);
}

function unverified(): never {
  throw new CppCuteBrowserWasmVerifierEvidenceError(
    "$.verifierEvidence",
    "expected opaque prepared verifier evidence authority",
  );
}
