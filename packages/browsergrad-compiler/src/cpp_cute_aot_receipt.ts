import {
  assertJsonValue,
  canonicalJsonBytes,
  canonicalizeJson,
  decodeWireJson,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  isJsonObject,
  parseWireU64,
  resolveDecodeLimits,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  sha256Hex,
  wireIntegerToBigInt,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  computeCppCuteAotDependencyManifestHash,
  computeCppCuteAotInvocationManifestHash,
  computeCppCuteAotLimitsManifestHash,
  computeCppCuteAotOutputManifestHash,
} from "./cpp_cute_aot_manifests.js";
import {
  unwrapPreparedCppCuteAotExecutionEnvironment,
  type PreparedCppCuteAotExecutionEnvironment,
} from "./cpp_cute_aot_environment.js";
import { computeCppCuteAotExecutionPlanHash } from "./cpp_cute_aot_policy.js";
import {
  unwrapPreparedCppCuteAotRunMetadata,
  type PreparedCppCuteAotRunMetadata,
} from "./cpp_cute_aot_run_metadata.js";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "./cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteFrontendRequestBinding,
  type PreparedCppCuteFrontendRequestBinding,
} from "./cpp_cute_frontend_request_binding.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  type CppCuteFrontendCompilerProfile,
  type CppCuteFrontendContainerProfile,
  type CppCuteFrontendExtractorProfile,
  type CppCuteFrontendRunnerProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_AOT_RECEIPT_SCHEMA = "browsergrad.compiler.cpp-cute.aot-runner-receipt";
export const CPP_CUTE_AOT_RECEIPT_MAJOR = 3;
export const CPP_CUTE_AOT_RECEIPT_MINOR = 0;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const OCI_SHA256 = /^sha256:[0-9a-f]{64}$/u;
const RECEIPT_ID = /^bg\.cpp\.aot-receipt\.sha256\.[0-9a-f]{64}$/u;
const INVOCATION_ID = /^bg\.cpp\.aot-invocation\.sha256\.[0-9a-f]{64}$/u;
const ARTIFACT_ID = /^bg\.artifact\.cpp-cute-frontend\.sha256\.[0-9a-f]{64}$/u;
const VERIFIED_RECEIPTS = new WeakMap<object, VerifiedCppCuteAotRunnerReceiptRecord>();
const VERIFIED_RECEIPT_RESOURCES = new WeakMap<object, VerifiedCppCuteAotRunnerReceipt>();

export interface CppCuteAotReceiptVersionV3 extends JsonObject {
  readonly major: typeof CPP_CUTE_AOT_RECEIPT_MAJOR;
  readonly minor: typeof CPP_CUTE_AOT_RECEIPT_MINOR;
}

export interface CppCuteAotReceiptSandboxV3 extends JsonObject {
  readonly contractId: "browsergrad.compiler.cpp-cute.aot@1";
  readonly policySha256: string;
  readonly limitsSha256: string;
  readonly network: "none";
  readonly readOnlyRoot: true;
  readonly noNewPrivileges: true;
  readonly linking: "forbidden";
  readonly userProducedNativeExecution: "forbidden";
}

export interface CppCuteAotReceiptInvocationV3 extends JsonObject {
  readonly invocationId: string;
  readonly invocationManifestSha256: string;
  readonly executionPlanSha256: string;
  readonly executionEnvironmentManifestSha256: string;
  readonly runner: CppCuteFrontendRunnerProfile;
  readonly container: CppCuteFrontendContainerProfile;
  readonly extractor: CppCuteFrontendExtractorProfile;
  readonly compiler: CppCuteFrontendCompilerProfile;
  readonly dependencyManifestSha256: string;
  readonly sandbox: CppCuteAotReceiptSandboxV3;
}

export interface CppCuteAotReceiptOpenedInputsV3 extends JsonObject {
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
}

export interface CppCuteAotReceiptOutputV3 extends JsonObject {
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly transportHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly outputManifestSha256: string;
}

/** Exact files/bytes present in verified opened-input closure. */
export interface CppCuteAotReceiptObservedInputValuesV3 extends JsonObject {
  readonly openedSourceFiles: WireU64;
  readonly openedSourceBytes: WireU64;
  readonly openedHeaderFiles: WireU64;
  readonly openedHeaderBytes: WireU64;
}

/** Values sampled by supervisor/OS accounting for this completed process. */
export interface CppCuteAotReceiptProcessMeasurementValuesV3 extends JsonObject {
  readonly wallTimeMs: WireU64;
  readonly cpuTimeMs: WireU64;
  readonly peakMemoryBytes: WireU64;
  readonly peakProcesses: WireU64;
}

/** Exact counts of records serialized in verified frontend artifact, not total Clang work. */
export interface CppCuteAotReceiptEmittedArtifactValuesV3 extends JsonObject {
  readonly macroExpansionFacts: WireU64;
  readonly templateInstantiationFacts: WireU64;
  readonly declarations: WireU64;
  readonly types: WireU64;
  readonly constants: WireU64;
  readonly layoutFacts: WireU64;
  readonly tensorFacts: WireU64;
  readonly operationFacts: WireU64;
  readonly targetIntrinsicFacts: WireU64;
  readonly diagnostics: WireU64;
  readonly outputBytes: WireU64;
}

/** Controls configured before extraction. These are ceilings, never observations. */
export interface CppCuteAotReceiptEnforcedCeilingValuesV3 extends JsonObject {
  readonly maxSourceFiles: WireU64;
  readonly maxSourceBytes: WireU64;
  readonly maxHeaderFiles: WireU64;
  readonly maxHeaderBytes: WireU64;
  readonly maxIncludeDepth: WireU64;
  readonly maxMacroExpansions: WireU64;
  readonly maxPreprocessedTokens: WireU64;
  readonly maxAstNodes: WireU64;
  readonly maxConstexprSteps: WireU64;
  readonly maxTemplateInstantiations: WireU64;
  readonly maxTemplateDepth: WireU64;
  readonly maxDeclarations: WireU64;
  readonly maxTypes: WireU64;
  readonly maxConstants: WireU64;
  readonly maxLayouts: WireU64;
  readonly maxTensors: WireU64;
  readonly maxOperations: WireU64;
  readonly maxTargetIntrinsics: WireU64;
  readonly maxDiagnostics: WireU64;
  readonly maxOutputBytes: WireU64;
  readonly maxWallTimeMs: WireU64;
  readonly maxCpuTimeMs: WireU64;
  readonly maxMemoryBytes: WireU64;
  readonly maxProcesses: WireU64;
}

export type CppCuteAotReceiptAccountingKindV3 =
  | "observed-exact"
  | "emitted-artifact-exact"
  | "enforced-upper-bound";

export interface CppCuteAotReceiptAccountingV3<
  K extends CppCuteAotReceiptAccountingKindV3,
  T extends JsonObject,
> extends JsonObject {
  readonly accountingKind: K;
  readonly values: T;
}

export interface CppCuteAotReceiptResourcesV3 extends JsonObject {
  readonly observedInputs: CppCuteAotReceiptAccountingV3<"observed-exact", CppCuteAotReceiptObservedInputValuesV3>;
  readonly processMeasurements: CppCuteAotReceiptAccountingV3<"observed-exact", CppCuteAotReceiptProcessMeasurementValuesV3>;
  readonly emittedArtifact: CppCuteAotReceiptAccountingV3<"emitted-artifact-exact", CppCuteAotReceiptEmittedArtifactValuesV3>;
  readonly enforcedCeilings: CppCuteAotReceiptAccountingV3<"enforced-upper-bound", CppCuteAotReceiptEnforcedCeilingValuesV3>;
}

export interface CppCuteAotRunnerReceiptV3 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_RECEIPT_SCHEMA;
  readonly version: CppCuteAotReceiptVersionV3;
  readonly receiptId: string;
  readonly runMetadataId: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly profileHash: string;
  readonly invocation: CppCuteAotReceiptInvocationV3;
  readonly openedInputs: CppCuteAotReceiptOpenedInputsV3;
  readonly output: CppCuteAotReceiptOutputV3;
  readonly resources: CppCuteAotReceiptResourcesV3;
  readonly outcome: "succeeded";
  readonly exitCode: 0;
}

export interface CppCuteAotRunnerReceiptBodyV3 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_RECEIPT_SCHEMA;
  readonly version: CppCuteAotReceiptVersionV3;
  readonly runMetadataId: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly profileHash: string;
  readonly invocation: CppCuteAotReceiptInvocationV3;
  readonly openedInputs: CppCuteAotReceiptOpenedInputsV3;
  readonly output: CppCuteAotReceiptOutputV3;
  readonly resources: CppCuteAotReceiptResourcesV3;
  readonly outcome: "succeeded";
  readonly exitCode: 0;
}

declare const verifiedCppCuteAotRunnerReceiptBrand: unique symbol;

export interface VerifiedCppCuteAotRunnerReceipt {
  readonly [verifiedCppCuteAotRunnerReceiptBrand]: true;
  readonly receiptId: string;
  readonly receiptBytesSha256: string;
  readonly receiptByteLength: WireU64;
  readonly runMetadataId: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly profileHash: string;
  readonly invocationId: string;
  readonly invocationManifestSha256: string;
  readonly executionPlanSha256: string;
  readonly executionEnvironmentManifestSha256: string;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
}

declare const verifiedCppCuteAotRunnerReceiptResourceBrand: unique symbol;

/** Opaque proof that exact canonical receipt bytes passed strict decoding. */
export interface VerifiedCppCuteAotRunnerReceiptResource {
  readonly [verifiedCppCuteAotRunnerReceiptResourceBrand]: true;
  readonly receiptId: string;
  readonly receiptBytesSha256: string;
  readonly receiptByteLength: WireU64;
}

export interface VerifiedCppCuteAotRunnerReceiptRecord {
  readonly receipt: CppCuteAotRunnerReceiptV3;
  readonly metadata: PreparedCppCuteAotRunMetadata;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
}

export interface VerifyCppCuteAotRunnerReceiptOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

export type CppCuteAotRunnerReceiptErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-METADATA-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNVERIFIED";

export class CppCuteAotRunnerReceiptError extends Error {
  constructor(
    readonly code: CppCuteAotRunnerReceiptErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotRunnerReceiptError";
  }
}

/**
 * Verifies a successful runner report against exact prepared intent and exact
 * verified output. This establishes structural authority only. Producer trust
 * requires a detached attestation that authenticates this receipt resource.
 */
export async function verifyCppCuteAotRunnerReceipt(
  metadata: PreparedCppCuteAotRunMetadata,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  requestBinding: PreparedCppCuteFrontendRequestBinding,
  value: unknown,
  options: VerifyCppCuteAotRunnerReceiptOptions = {},
): Promise<VerifiedCppCuteAotRunnerReceipt> {
  const metadataRecord = unwrapPreparedCppCuteAotRunMetadata(metadata);
  const bindingRecord = unwrapPreparedCppCuteFrontendRequestBinding(requestBinding);
  if (bindingRecord.request !== metadataRecord.request) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
      "$.requestBindingId",
      "request binding belongs to a different prepared request",
    );
  }
  const profile = metadataRecord.profile;
  const profileRecord = unwrapPreparedCppCuteAotFrontendProfile(profile);
  const environmentRecord = unwrapPreparedCppCuteAotExecutionEnvironment(executionEnvironment);
  if (environmentRecord.profile !== profile || executionEnvironment.profileHash !== profile.profileHash) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
      "$.invocation.executionEnvironmentManifestSha256",
      "execution environment belongs to a different prepared profile",
    );
  }
  const artifactResource = bindingRecord.artifactResource;
  const artifact = unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource);
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const limits = normalizeOptions(options);
  throwIfAborted(options.signal);
  let receipt: CppCuteAotRunnerReceiptV3;
  try {
    assertJsonValue(value, { limits });
    receipt = parseReceipt(value);
    canonicalizeJson(receipt, { limits });
  } catch (error) {
    if (error instanceof CppCuteAotRunnerReceiptError) throw error;
    if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
      resource(error.diagnostic.path ?? "$", error.message, { cause: error });
    }
    invalid("$", "runner receipt is not closed canonical JSON", { cause: error });
  }

  verifyMetadataAndArtifactBindings(receipt, metadata, requestBinding, artifact, artifactRecord, profileRecord);
  await verifyInvocationBindings(receipt, metadata, executionEnvironment, profile, profileRecord);
  await verifyOutputBindings(receipt, artifact);
  verifyResourceBindings(receipt.resources, profile, artifact, artifactRecord);
  const expectedReceiptId = await deriveCppCuteAotRunnerReceiptId(receipt, { limits });
  if (receipt.receiptId !== expectedReceiptId) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-HASH-MISMATCH",
      "$.receiptId",
      `receipt ID must equal ${expectedReceiptId}`,
    );
  }
  throwIfAborted(options.signal);
  const canonicalBytes = canonicalJsonBytes(receipt, { limits });
  const receiptBytesSha256 = await sha256Hex(canonicalBytes);
  const receiptByteLength = encodeWireU64(BigInt(canonicalBytes.byteLength));
  const verified = Object.freeze({
    receiptId: expectedReceiptId,
    receiptBytesSha256,
    receiptByteLength,
    runMetadataId: receipt.runMetadataId,
    requestId: receipt.requestId,
    requestBindingId: receipt.requestBindingId,
    profileHash: receipt.profileHash,
    invocationId: receipt.invocation.invocationId,
    invocationManifestSha256: receipt.invocation.invocationManifestSha256,
    executionPlanSha256: receipt.invocation.executionPlanSha256,
    executionEnvironmentManifestSha256: receipt.invocation.executionEnvironmentManifestSha256,
    artifactId: receipt.output.artifactId,
    artifactHash: receipt.output.artifactHash,
    artifactBytesSha256: receipt.output.artifactBytesSha256,
    artifactByteLength: receipt.output.artifactByteLength,
  }) as VerifiedCppCuteAotRunnerReceipt;
  VERIFIED_RECEIPTS.set(verified, Object.freeze({
    receipt,
    metadata,
    requestBinding,
    profile,
    executionEnvironment,
    artifactResource,
    artifact,
  }));
  return verified;
}

export async function decodeCppCuteAotRunnerReceipt(
  metadata: PreparedCppCuteAotRunMetadata,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  requestBinding: PreparedCppCuteFrontendRequestBinding,
  bytes: Uint8Array,
  options: VerifyCppCuteAotRunnerReceiptOptions = {},
): Promise<VerifiedCppCuteAotRunnerReceiptResource> {
  throwIfAborted(options.signal);
  const snapshot = new Uint8Array(bytes);
  const verified = await verifyCppCuteAotRunnerReceipt(
    metadata,
    executionEnvironment,
    requestBinding,
    decodeWireJson(snapshot, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
  const canonical = canonicalCppCuteAotRunnerReceiptBytes(
    verified,
    options.limits === undefined ? {} : { limits: options.limits },
  );
  if (!equalBytes(snapshot, canonical)) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-NONCANONICAL-BYTES",
      "$bytes",
      "runner receipt bytes must exactly equal the canonical normalized record",
    );
  }
  const resource = Object.freeze({
    receiptId: verified.receiptId,
    receiptBytesSha256: verified.receiptBytesSha256,
    receiptByteLength: verified.receiptByteLength,
  }) as VerifiedCppCuteAotRunnerReceiptResource;
  VERIFIED_RECEIPT_RESOURCES.set(resource, verified);
  return resource;
}

export function unwrapVerifiedCppCuteAotRunnerReceiptResource(
  resource: VerifiedCppCuteAotRunnerReceiptResource,
): VerifiedCppCuteAotRunnerReceipt {
  if (typeof resource !== "object" || resource === null) unverified();
  const verified = VERIFIED_RECEIPT_RESOURCES.get(resource as object);
  if (verified === undefined) unverified();
  return verified;
}

export function canonicalCppCuteAotRunnerReceiptResourceBytes(
  resource: VerifiedCppCuteAotRunnerReceiptResource,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Uint8Array {
  return canonicalCppCuteAotRunnerReceiptBytes(unwrapVerifiedCppCuteAotRunnerReceiptResource(resource), options);
}

export function unwrapVerifiedCppCuteAotRunnerReceipt(
  verified: VerifiedCppCuteAotRunnerReceipt,
): VerifiedCppCuteAotRunnerReceiptRecord {
  if (typeof verified !== "object" || verified === null) unverified();
  const record = VERIFIED_RECEIPTS.get(verified as object);
  if (record === undefined) unverified();
  return record;
}

export function canonicalCppCuteAotRunnerReceiptBytes(
  verified: VerifiedCppCuteAotRunnerReceipt,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Uint8Array {
  return canonicalJsonBytes(unwrapVerifiedCppCuteAotRunnerReceipt(verified).receipt, options);
}

export async function deriveCppCuteAotRunnerReceiptId(
  receipt: CppCuteAotRunnerReceiptV3 | CppCuteAotRunnerReceiptBodyV3,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-runner-receipt.v3",
    receipt: {
      schema: receipt.schema,
      version: receipt.version,
      runMetadataId: receipt.runMetadataId,
      requestId: receipt.requestId,
      requestBindingId: receipt.requestBindingId,
      profileHash: receipt.profileHash,
      invocation: receipt.invocation,
      openedInputs: receipt.openedInputs,
      output: receipt.output,
      resources: receipt.resources,
      outcome: receipt.outcome,
      exitCode: receipt.exitCode,
    },
  }, options);
  return `bg.cpp.aot-receipt.sha256.${digest}`;
}

function parseReceipt(value: JsonValue): CppCuteAotRunnerReceiptV3 {
  const object = closedObject(value, [
    "schema", "version", "receiptId", "runMetadataId", "requestId", "requestBindingId", "profileHash", "invocation", "openedInputs",
    "output", "resources", "outcome", "exitCode",
  ], "$");
  if (object.schema !== CPP_CUTE_AOT_RECEIPT_SCHEMA) invalid("$.schema", `expected ${CPP_CUTE_AOT_RECEIPT_SCHEMA}`);
  if (object.outcome !== "succeeded" || object.exitCode !== 0) {
    invalid("$.outcome", "artifact receipts represent successful zero-exit runs only");
  }
  return deepFreezeJson({
    schema: CPP_CUTE_AOT_RECEIPT_SCHEMA,
    version: parseVersion(field(object, "version", "$"), "$.version"),
    receiptId: patterned(field(object, "receiptId", "$"), "$.receiptId", RECEIPT_ID, "receipt ID"),
    runMetadataId: patterned(field(object, "runMetadataId", "$"), "$.runMetadataId", /^bg\.cpp\.aot-run-metadata\.sha256\.[0-9a-f]{64}$/u, "run metadata ID"),
    requestId: patterned(field(object, "requestId", "$"), "$.requestId", /^bg\.cpp\.frontend-request\.sha256\.[0-9a-f]{64}$/u, "request ID"),
    requestBindingId: patterned(field(object, "requestBindingId", "$"), "$.requestBindingId", /^bg\.cpp\.frontend-request-binding\.sha256\.[0-9a-f]{64}$/u, "request binding ID"),
    profileHash: sha256(field(object, "profileHash", "$"), "$.profileHash"),
    invocation: parseInvocation(field(object, "invocation", "$"), "$.invocation"),
    openedInputs: parseOpenedInputs(field(object, "openedInputs", "$"), "$.openedInputs"),
    output: parseOutput(field(object, "output", "$"), "$.output"),
    resources: parseResources(field(object, "resources", "$"), "$.resources"),
    outcome: "succeeded",
    exitCode: 0,
  });
}

function parseVersion(value: JsonValue, path: string): CppCuteAotReceiptVersionV3 {
  const object = closedObject(value, ["major", "minor"], path);
  if (object.major !== CPP_CUTE_AOT_RECEIPT_MAJOR || object.minor !== CPP_CUTE_AOT_RECEIPT_MINOR) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNSUPPORTED-VERSION",
      path,
      `closed runner receipt reader supports ${CPP_CUTE_AOT_RECEIPT_MAJOR}.${CPP_CUTE_AOT_RECEIPT_MINOR} only`,
    );
  }
  return { major: CPP_CUTE_AOT_RECEIPT_MAJOR, minor: CPP_CUTE_AOT_RECEIPT_MINOR };
}

function parseInvocation(value: JsonValue, path: string): CppCuteAotReceiptInvocationV3 {
  const object = closedObject(value, [
    "invocationId", "invocationManifestSha256", "executionPlanSha256", "executionEnvironmentManifestSha256",
    "runner", "container", "extractor", "compiler", "dependencyManifestSha256", "sandbox",
  ], path);
  return {
    invocationId: patterned(field(object, "invocationId", path), `${path}.invocationId`, INVOCATION_ID, "invocation ID"),
    invocationManifestSha256: sha256(field(object, "invocationManifestSha256", path), `${path}.invocationManifestSha256`),
    executionPlanSha256: sha256(field(object, "executionPlanSha256", path), `${path}.executionPlanSha256`),
    executionEnvironmentManifestSha256: sha256(
      field(object, "executionEnvironmentManifestSha256", path),
      `${path}.executionEnvironmentManifestSha256`,
    ),
    runner: parseRunner(field(object, "runner", path), `${path}.runner`),
    container: parseContainer(field(object, "container", path), `${path}.container`),
    extractor: parseExtractor(field(object, "extractor", path), `${path}.extractor`),
    compiler: parseCompiler(field(object, "compiler", path), `${path}.compiler`),
    dependencyManifestSha256: sha256(
      field(object, "dependencyManifestSha256", path),
      `${path}.dependencyManifestSha256`,
    ),
    sandbox: parseSandbox(field(object, "sandbox", path), `${path}.sandbox`),
  };
}

function parseRunner(value: JsonValue, path: string): CppCuteFrontendRunnerProfile {
  const object = closedObject(value, ["id", "version", "binarySha256"], path);
  return {
    id: boundedString(field(object, "id", path), `${path}.id`, 256),
    version: boundedString(field(object, "version", path), `${path}.version`, 128),
    binarySha256: sha256(field(object, "binarySha256", path), `${path}.binarySha256`),
  };
}

function parseContainer(value: JsonValue, path: string): CppCuteFrontendContainerProfile {
  const object = closedObject(value, ["runtime", "repository", "platform", "manifestDigest", "configDigest"], path);
  if (object.runtime !== "docker" || object.platform !== "linux/amd64") {
    invalid(path, "runner receipt requires Docker on resolved linux/amd64");
  }
  return {
    runtime: "docker",
    repository: boundedString(field(object, "repository", path), `${path}.repository`, 512),
    platform: "linux/amd64",
    manifestDigest: patterned(field(object, "manifestDigest", path), `${path}.manifestDigest`, OCI_SHA256, "OCI digest"),
    configDigest: patterned(field(object, "configDigest", path), `${path}.configDigest`, OCI_SHA256, "OCI digest"),
  };
}

function parseExtractor(value: JsonValue, path: string): CppCuteFrontendExtractorProfile {
  const object = closedObject(
    value,
    ["id", "version", "buildId", "binarySha256", "semanticAdapterManifestSha256"],
    path,
  );
  return {
    id: boundedString(field(object, "id", path), `${path}.id`, 256),
    version: boundedString(field(object, "version", path), `${path}.version`, 128),
    buildId: boundedString(field(object, "buildId", path), `${path}.buildId`, 256),
    binarySha256: sha256(field(object, "binarySha256", path), `${path}.binarySha256`),
    semanticAdapterManifestSha256: sha256(
      field(object, "semanticAdapterManifestSha256", path),
      `${path}.semanticAdapterManifestSha256`,
    ),
  };
}

function parseCompiler(value: JsonValue, path: string): CppCuteFrontendCompilerProfile {
  const object = closedObject(value, ["id", "version", "buildId", "binarySha256", "resourceDirectorySha256"], path);
  return {
    id: boundedString(field(object, "id", path), `${path}.id`, 256),
    version: boundedString(field(object, "version", path), `${path}.version`, 128),
    buildId: boundedString(field(object, "buildId", path), `${path}.buildId`, 512),
    binarySha256: sha256(field(object, "binarySha256", path), `${path}.binarySha256`),
    resourceDirectorySha256: sha256(
      field(object, "resourceDirectorySha256", path),
      `${path}.resourceDirectorySha256`,
    ),
  };
}

function parseSandbox(value: JsonValue, path: string): CppCuteAotReceiptSandboxV3 {
  const object = closedObject(value, [
    "contractId", "policySha256", "limitsSha256", "network", "readOnlyRoot", "noNewPrivileges", "linking",
    "userProducedNativeExecution",
  ], path);
  if (
    object.contractId !== "browsergrad.compiler.cpp-cute.aot@1"
    || object.network !== "none"
    || object.readOnlyRoot !== true
    || object.noNewPrivileges !== true
    || object.linking !== "forbidden"
    || object.userProducedNativeExecution !== "forbidden"
  ) {
    invalid(path, "runner receipt does not satisfy the closed AOT sandbox contract");
  }
  return {
    contractId: "browsergrad.compiler.cpp-cute.aot@1",
    policySha256: sha256(field(object, "policySha256", path), `${path}.policySha256`),
    limitsSha256: sha256(field(object, "limitsSha256", path), `${path}.limitsSha256`),
    network: "none",
    readOnlyRoot: true,
    noNewPrivileges: true,
    linking: "forbidden",
    userProducedNativeExecution: "forbidden",
  };
}

function parseOpenedInputs(value: JsonValue, path: string): CppCuteAotReceiptOpenedInputsV3 {
  const object = closedObject(value, ["sourceSetSha256", "headerSetSha256", "inputClosureSha256"], path);
  return {
    sourceSetSha256: sha256(field(object, "sourceSetSha256", path), `${path}.sourceSetSha256`),
    headerSetSha256: sha256(field(object, "headerSetSha256", path), `${path}.headerSetSha256`),
    inputClosureSha256: sha256(field(object, "inputClosureSha256", path), `${path}.inputClosureSha256`),
  };
}

function parseOutput(value: JsonValue, path: string): CppCuteAotReceiptOutputV3 {
  const object = closedObject(value, [
    "artifactId", "artifactHash", "transportHash", "artifactBytesSha256", "artifactByteLength",
    "outputManifestSha256",
  ], path);
  return {
    artifactId: patterned(field(object, "artifactId", path), `${path}.artifactId`, ARTIFACT_ID, "artifact ID"),
    artifactHash: sha256(field(object, "artifactHash", path), `${path}.artifactHash`),
    transportHash: sha256(field(object, "transportHash", path), `${path}.transportHash`),
    artifactBytesSha256: sha256(field(object, "artifactBytesSha256", path), `${path}.artifactBytesSha256`),
    artifactByteLength: parseWireU64(field(object, "artifactByteLength", path), `${path}.artifactByteLength`),
    outputManifestSha256: sha256(field(object, "outputManifestSha256", path), `${path}.outputManifestSha256`),
  };
}

function parseResources(value: JsonValue, path: string): CppCuteAotReceiptResourcesV3 {
  const object = closedObject(
    value,
    ["observedInputs", "processMeasurements", "emittedArtifact", "enforcedCeilings"],
    path,
  );
  const observedInputs = parseAccounting(
    field(object, "observedInputs", path),
    `${path}.observedInputs`,
    "observed-exact",
    ["openedSourceFiles", "openedSourceBytes", "openedHeaderFiles", "openedHeaderBytes"],
  ) as CppCuteAotReceiptResourcesV3["observedInputs"];
  const processMeasurements = parseAccounting(
    field(object, "processMeasurements", path),
    `${path}.processMeasurements`,
    "observed-exact",
    ["wallTimeMs", "cpuTimeMs", "peakMemoryBytes", "peakProcesses"],
  ) as CppCuteAotReceiptResourcesV3["processMeasurements"];
  const emittedArtifact = parseAccounting(
    field(object, "emittedArtifact", path),
    `${path}.emittedArtifact`,
    "emitted-artifact-exact",
    [
      "macroExpansionFacts", "templateInstantiationFacts", "declarations", "types", "constants",
      "layoutFacts", "tensorFacts", "operationFacts", "targetIntrinsicFacts", "diagnostics", "outputBytes",
    ],
  ) as CppCuteAotReceiptResourcesV3["emittedArtifact"];
  const enforcedCeilings = parseAccounting(
    field(object, "enforcedCeilings", path),
    `${path}.enforcedCeilings`,
    "enforced-upper-bound",
    [
      "maxSourceFiles", "maxSourceBytes", "maxHeaderFiles", "maxHeaderBytes", "maxIncludeDepth",
      "maxMacroExpansions", "maxPreprocessedTokens", "maxAstNodes", "maxConstexprSteps",
      "maxTemplateInstantiations", "maxTemplateDepth", "maxDeclarations", "maxTypes", "maxConstants",
      "maxLayouts", "maxTensors", "maxOperations", "maxTargetIntrinsics", "maxDiagnostics",
      "maxOutputBytes", "maxWallTimeMs", "maxCpuTimeMs", "maxMemoryBytes", "maxProcesses",
    ],
  ) as CppCuteAotReceiptResourcesV3["enforcedCeilings"];
  if (
    wireIntegerToBigInt(processMeasurements.values.peakMemoryBytes) === 0n
    || wireIntegerToBigInt(processMeasurements.values.peakProcesses) === 0n
  ) {
    invalid(`${path}.processMeasurements.values`, "successful receipt must report nonzero peak memory and process observations");
  }
  return { observedInputs, processMeasurements, emittedArtifact, enforcedCeilings };
}

function parseAccounting(
  value: JsonValue,
  path: string,
  expectedKind: CppCuteAotReceiptAccountingKindV3,
  valueFields: readonly string[],
): CppCuteAotReceiptAccountingV3<CppCuteAotReceiptAccountingKindV3, JsonObject> {
  const object = closedObject(value, ["accountingKind", "values"], path);
  if (object.accountingKind !== expectedKind) {
    invalid(`${path}.accountingKind`, `accounting kind must be ${expectedKind}`);
  }
  const values = closedObject(field(object, "values", path), valueFields, `${path}.values`);
  return {
    accountingKind: expectedKind,
    values: Object.fromEntries(valueFields.map((name) => [
      name,
      parseWireU64(field(values, name, `${path}.values`), `${path}.values.${name}`),
    ])),
  };
}

function verifyMetadataAndArtifactBindings(
  receipt: CppCuteAotRunnerReceiptV3,
  metadata: PreparedCppCuteAotRunMetadata,
  requestBinding: PreparedCppCuteFrontendRequestBinding,
  artifact: VerifiedCppCuteFrontendArtifact,
  artifactRecord: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>,
  profileRecord: ReturnType<typeof unwrapPreparedCppCuteAotFrontendProfile>,
): void {
  if (
    receipt.runMetadataId !== metadata.runMetadataId
    || receipt.requestId !== metadata.requestId
    || receipt.requestBindingId !== requestBinding.bindingId
    || receipt.profileHash !== metadata.profileHash
  ) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-METADATA-MISMATCH",
      "$.runMetadataId",
      "receipt differs from exact run metadata, request, or request binding authority",
    );
  }
  const expectedInputs: CppCuteAotReceiptOpenedInputsV3 = {
    sourceSetSha256: artifact.sourceSetSha256,
    headerSetSha256: artifact.headerSetSha256,
    inputClosureSha256: artifact.inputClosureSha256,
  };
  if (canonicalText(receipt.openedInputs) !== canonicalText(expectedInputs)) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
      "$.openedInputs",
      "runner observations differ from the exact verified artifact closure",
    );
  }
  if (
    requestBinding.artifactId !== artifact.artifactId
    || requestBinding.artifactBytesSha256 !== artifact.artifactBytesSha256
    || artifact.compilationContractHash !== profileRecord.compilationContractHash
  ) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
      "$.output.artifactId",
      "request binding and verified artifact differ",
    );
  }
  const extractor = profileRecord.profile.deployment.extractor;
  if (
    artifactRecord.envelope.producer.id !== extractor.id
    || artifactRecord.envelope.producer.version !== extractor.version
  ) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
      "$.artifact.producer",
      "artifact producer differs from prepared extractor profile",
    );
  }
}

async function verifyInvocationBindings(
  receipt: CppCuteAotRunnerReceiptV3,
  metadata: PreparedCppCuteAotRunMetadata,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  profile: PreparedCppCuteFrontendProfile,
  profileRecord: ReturnType<typeof unwrapPreparedCppCuteAotFrontendProfile>,
): Promise<void> {
  const configured = profileRecord.profile;
  const invocationManifestSha256 = await computeCppCuteAotInvocationManifestHash(metadata);
  const executionPlanSha256 = await computeCppCuteAotExecutionPlanHash(metadata, executionEnvironment);
  const expected: CppCuteAotReceiptInvocationV3 = {
    invocationId: `bg.cpp.aot-invocation.sha256.${invocationManifestSha256}`,
    invocationManifestSha256,
    executionPlanSha256,
    executionEnvironmentManifestSha256: executionEnvironment.manifestSha256,
    runner: configured.deployment.runner,
    container: configured.deployment.container,
    extractor: configured.deployment.extractor,
    compiler: configured.toolchain.compiler,
    dependencyManifestSha256: await computeCppCuteAotDependencyManifestHash(profile),
    sandbox: {
      contractId: configured.deployment.contractId,
      policySha256: configured.deployment.sandboxPolicySha256,
      limitsSha256: await computeCppCuteAotLimitsManifestHash(configured.extractionLimits),
      network: "none",
      readOnlyRoot: true,
      noNewPrivileges: true,
      linking: "forbidden",
      userProducedNativeExecution: "forbidden",
    },
  };
  if (canonicalText(receipt.invocation) !== canonicalText(expected)) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH", "$.invocation", "reported invocation differs from exact prepared profile and run metadata");
  }
}

async function verifyOutputBindings(
  receipt: CppCuteAotRunnerReceiptV3,
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<void> {
  const expected: CppCuteAotReceiptOutputV3 = {
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    transportHash: artifact.transportHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    outputManifestSha256: await computeCppCuteAotOutputManifestHash(artifact),
  };
  if (canonicalText(receipt.output) !== canonicalText(expected)) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH", "$.output", "reported output differs from exact verified artifact resource");
  }
}

function verifyResourceBindings(
  resources: CppCuteAotReceiptResourcesV3,
  profile: PreparedCppCuteFrontendProfile,
  artifact: VerifiedCppCuteFrontendArtifact,
  artifactRecord: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>,
): void {
  const configured = unwrapPreparedCppCuteAotFrontendProfile(profile).profile.extractionLimits;
  const payload = artifactRecord.envelope.payload;
  const sources = payload.inputs.files.filter(
    (file) => file.owner.kind === "source",
  );
  const headers = payload.inputs.files.filter(
    (file) => file.owner.kind !== "source",
  );
  const expectedObservedInputs: CppCuteAotReceiptObservedInputValuesV3 = {
    openedSourceFiles: encodeWireU64(BigInt(sources.length)),
    openedSourceBytes: encodeWireU64(sumFileBytes(sources)),
    openedHeaderFiles: encodeWireU64(BigInt(headers.length)),
    openedHeaderBytes: encodeWireU64(sumFileBytes(headers)),
  };
  const expectedEmittedArtifact: CppCuteAotReceiptEmittedArtifactValuesV3 = {
    macroExpansionFacts: encodeWireU64(BigInt(payload.macroExpansions.length)),
    templateInstantiationFacts: encodeWireU64(BigInt(payload.templateInstantiations.length)),
    declarations: encodeWireU64(BigInt(payload.declarations.length)),
    types: encodeWireU64(BigInt(payload.types.length)),
    constants: encodeWireU64(BigInt(payload.constants.length)),
    layoutFacts: encodeWireU64(BigInt(payload.facts.filter((fact) => fact.kind === "affine-layout").length)),
    tensorFacts: encodeWireU64(BigInt(payload.facts.filter((fact) => fact.kind === "tensor").length)),
    operationFacts: encodeWireU64(BigInt(payload.facts.filter((fact) => (
      fact.kind !== "affine-layout" && fact.kind !== "tensor" && fact.kind !== "target-intrinsic"
    )).length)),
    targetIntrinsicFacts: encodeWireU64(BigInt(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length)),
    diagnostics: encodeWireU64(BigInt(payload.diagnostics.length)),
    outputBytes: artifact.artifactByteLength,
  };
  const expectedCeilings: CppCuteAotReceiptEnforcedCeilingValuesV3 = {
    maxSourceFiles: encodeWireU64(BigInt(configured.maxSourceFiles)),
    maxSourceBytes: encodeWireU64(BigInt(configured.maxSourceBytes)),
    maxHeaderFiles: encodeWireU64(BigInt(configured.maxHeaderFiles)),
    maxHeaderBytes: encodeWireU64(BigInt(configured.maxHeaderBytes)),
    maxIncludeDepth: encodeWireU64(BigInt(configured.maxIncludeDepth)),
    maxMacroExpansions: encodeWireU64(BigInt(configured.maxMacroExpansions)),
    maxPreprocessedTokens: encodeWireU64(BigInt(configured.maxPreprocessedTokens)),
    maxAstNodes: encodeWireU64(BigInt(configured.maxAstNodes)),
    maxConstexprSteps: encodeWireU64(BigInt(configured.maxConstexprSteps)),
    maxTemplateInstantiations: encodeWireU64(BigInt(configured.maxTemplateInstantiations)),
    maxTemplateDepth: encodeWireU64(BigInt(configured.maxTemplateDepth)),
    maxDeclarations: encodeWireU64(BigInt(configured.maxDeclarations)),
    maxTypes: encodeWireU64(BigInt(configured.maxTypes)),
    maxConstants: encodeWireU64(BigInt(configured.maxConstants)),
    maxLayouts: encodeWireU64(BigInt(configured.maxLayouts)),
    maxTensors: encodeWireU64(BigInt(configured.maxTensors)),
    maxOperations: encodeWireU64(BigInt(configured.maxOperations)),
    maxTargetIntrinsics: encodeWireU64(BigInt(configured.maxTargetIntrinsics)),
    maxDiagnostics: encodeWireU64(BigInt(configured.maxDiagnostics)),
    maxOutputBytes: encodeWireU64(BigInt(configured.maxOutputBytes)),
    maxWallTimeMs: encodeWireU64(BigInt(configured.maxWallTimeMs)),
    maxCpuTimeMs: encodeWireU64(BigInt(configured.maxCpuTimeMs)),
    maxMemoryBytes: encodeWireU64(BigInt(configured.maxMemoryBytes)),
    maxProcesses: encodeWireU64(BigInt(configured.maxProcesses)),
  };
  if (canonicalText(resources.enforcedCeilings.values) !== canonicalText(expectedCeilings)) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH",
      "$.resources.enforcedCeilings.values",
      "enforced upper bounds differ from exact prepared profile configuration",
    );
  }
  const observed = resources.observedInputs.values;
  const process = resources.processMeasurements.values;
  const emitted = resources.emittedArtifact.values;
  const bounded: readonly [WireU64, number, string][] = [
    [observed.openedSourceFiles, configured.maxSourceFiles, "$.resources.observedInputs.values.openedSourceFiles"],
    [observed.openedSourceBytes, configured.maxSourceBytes, "$.resources.observedInputs.values.openedSourceBytes"],
    [observed.openedHeaderFiles, configured.maxHeaderFiles, "$.resources.observedInputs.values.openedHeaderFiles"],
    [observed.openedHeaderBytes, configured.maxHeaderBytes, "$.resources.observedInputs.values.openedHeaderBytes"],
    [emitted.macroExpansionFacts, configured.maxMacroExpansions, "$.resources.emittedArtifact.values.macroExpansionFacts"],
    [emitted.templateInstantiationFacts, configured.maxTemplateInstantiations, "$.resources.emittedArtifact.values.templateInstantiationFacts"],
    [emitted.declarations, configured.maxDeclarations, "$.resources.emittedArtifact.values.declarations"],
    [emitted.types, configured.maxTypes, "$.resources.emittedArtifact.values.types"],
    [emitted.constants, configured.maxConstants, "$.resources.emittedArtifact.values.constants"],
    [emitted.layoutFacts, configured.maxLayouts, "$.resources.emittedArtifact.values.layoutFacts"],
    [emitted.tensorFacts, configured.maxTensors, "$.resources.emittedArtifact.values.tensorFacts"],
    [emitted.operationFacts, configured.maxOperations, "$.resources.emittedArtifact.values.operationFacts"],
    [emitted.targetIntrinsicFacts, configured.maxTargetIntrinsics, "$.resources.emittedArtifact.values.targetIntrinsicFacts"],
    [emitted.diagnostics, configured.maxDiagnostics, "$.resources.emittedArtifact.values.diagnostics"],
    [emitted.outputBytes, configured.maxOutputBytes, "$.resources.emittedArtifact.values.outputBytes"],
    [process.wallTimeMs, configured.maxWallTimeMs, "$.resources.processMeasurements.values.wallTimeMs"],
    [process.cpuTimeMs, configured.maxCpuTimeMs, "$.resources.processMeasurements.values.cpuTimeMs"],
    [process.peakMemoryBytes, configured.maxMemoryBytes, "$.resources.processMeasurements.values.peakMemoryBytes"],
    [process.peakProcesses, configured.maxProcesses, "$.resources.processMeasurements.values.peakProcesses"],
  ];
  for (const [value, maximum, path] of bounded) {
    if (wireIntegerToBigInt(value) > BigInt(maximum)) {
      resource(path, `observed exact value exceeds prepared profile maximum ${maximum}`);
    }
  }
  if (canonicalText(resources.observedInputs.values) !== canonicalText(expectedObservedInputs)) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH",
      "$.resources.observedInputs.values",
      "exact opened-input accounting differs from verified artifact inputs",
    );
  }
  if (canonicalText(resources.emittedArtifact.values) !== canonicalText(expectedEmittedArtifact)) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
      "$.resources.emittedArtifact.values",
      "exact emitted-artifact accounting differs from verified artifact",
    );
  }
}

function sumFileBytes(files: readonly { readonly byteLength: WireU64 }[]): bigint {
  return files.reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
}

function canonicalText(value: JsonValue): string {
  return new TextDecoder().decode(canonicalJsonBytes(value));
}

function normalizeOptions(options: VerifyCppCuteAotRunnerReceiptOptions): DecodeLimits {
  if (typeof options !== "object" || options === null || Array.isArray(options)) invalid("$options", "options must be a plain object");
  const prototype = Object.getPrototypeOf(options);
  if (prototype !== Object.prototype && prototype !== null) invalid("$options", "options must be a plain object");
  const descriptors = Object.getOwnPropertyDescriptors(options);
  for (const key of Reflect.ownKeys(options)) {
    if (typeof key !== "string" || (key !== "limits" && key !== "signal")) invalid("$options", "options contain unknown fields");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`$options.${key}`, "options require enumerable data properties without accessors");
    }
  }
  try {
    return resolveDecodeLimits(options.limits);
  } catch (error) {
    resource("$options.limits", "invalid semantic decode limits", { cause: error });
  }
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const unknown = Object.keys(value).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length > 0) invalid(path, `unknown closed-record fields: ${unknown.sort().join(", ")}`);
  if (missing.length > 0) invalid(path, `missing required fields: ${missing.sort().join(", ")}`);
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  if (typeof value !== "string") invalid(path, "expected string");
  if (value.length === 0 || value.includes("\0") || new TextEncoder().encode(value).byteLength > maximumBytes) {
    invalid(path, `string must be nonempty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return value;
}

function patterned(value: JsonValue, path: string, pattern: RegExp, name: string): string {
  const result = boundedString(value, path, 1_024);
  if (!pattern.test(result)) invalid(path, `${name} has invalid syntax`);
  return result;
}

function sha256(value: JsonValue, path: string): string {
  return patterned(value, path, SHA256_HEX, "SHA-256");
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-CANCELLED", "$.signal", "runner receipt verification was aborted");
}

function unverified(): never {
  mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-UNVERIFIED", "$", "expected an instance-authorized runner receipt");
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVALID", path, message, options);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-RESOURCE-LIMIT", path, message, options);
}

function mismatch(
  code: CppCuteAotRunnerReceiptErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteAotRunnerReceiptError(code, path, message, options);
}
