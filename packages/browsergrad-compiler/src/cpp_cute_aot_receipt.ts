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
  unwrapPreparedCppCuteAotJob,
  type CppCuteAotSourceFileV1,
  type PreparedCppCuteAotJob,
} from "./cpp_cute_aot_job.js";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "./cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteFrontendProfile,
  type CppCuteFrontendCompilerProfile,
  type CppCuteFrontendContainerProfile,
  type CppCuteFrontendExtractorProfile,
  type CppCuteFrontendRunnerProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_AOT_RECEIPT_SCHEMA = "browsergrad.compiler.cpp-cute.aot-runner-receipt";
export const CPP_CUTE_AOT_RECEIPT_MAJOR = 1;
export const CPP_CUTE_AOT_RECEIPT_MINOR = 0;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const OCI_SHA256 = /^sha256:[0-9a-f]{64}$/u;
const RECEIPT_ID = /^bg\.cpp\.aot-receipt\.sha256\.[0-9a-f]{64}$/u;
const INVOCATION_ID = /^bg\.cpp\.aot-invocation\.sha256\.[0-9a-f]{64}$/u;
const STABLE_ID = /^bg\.cpp\.[a-z0-9-]+\.sha256\.[0-9a-f]{64}$/u;
const ARTIFACT_ID = /^bg\.artifact\.cpp-cute-frontend\.sha256\.[0-9a-f]{64}$/u;
const VERIFIED_RECEIPTS = new WeakMap<object, VerifiedCppCuteAotRunnerReceiptRecord>();
const VERIFIED_RECEIPT_RESOURCES = new WeakMap<object, VerifiedCppCuteAotRunnerReceipt>();

export interface CppCuteAotReceiptVersionV1 extends JsonObject {
  readonly major: typeof CPP_CUTE_AOT_RECEIPT_MAJOR;
  readonly minor: typeof CPP_CUTE_AOT_RECEIPT_MINOR;
}

export interface CppCuteAotReceiptSandboxV1 extends JsonObject {
  readonly contractId: "browsergrad.compiler.cpp-cute.aot@1";
  readonly policySha256: string;
  readonly limitsSha256: string;
  readonly network: "none";
  readonly readOnlyRoot: true;
  readonly noNewPrivileges: true;
  readonly linking: "forbidden";
  readonly nativeExecution: "forbidden";
}

export interface CppCuteAotReceiptInvocationV1 extends JsonObject {
  readonly invocationId: string;
  readonly invocationManifestSha256: string;
  readonly runner: CppCuteFrontendRunnerProfile;
  readonly container: CppCuteFrontendContainerProfile;
  readonly extractor: CppCuteFrontendExtractorProfile;
  readonly compiler: CppCuteFrontendCompilerProfile;
  readonly dependencyManifestSha256: string;
  readonly sandbox: CppCuteAotReceiptSandboxV1;
}

export interface CppCuteAotReceiptOpenedInputsV1 extends JsonObject {
  readonly files: readonly CppCuteAotSourceFileV1[];
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
}

export type CppCuteAotReceiptSelectionV1 =
  | (JsonObject & {
      readonly kind: "resolved";
      readonly requestId: string;
      readonly anchorTokenSha256: string;
      readonly expectedEntryId: string;
      readonly resolvedEntryId: string;
    })
  | (JsonObject & {
      readonly kind: "rejected";
      readonly requestId: string;
      readonly anchorTokenSha256: string;
      readonly expectedEntryId: string;
      readonly blockingDiagnosticIds: readonly string[];
    });

export interface CppCuteAotReceiptOutputV1 extends JsonObject {
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly transportHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly outputManifestSha256: string;
}

export interface CppCuteAotReceiptResourcesV1 extends JsonObject {
  readonly sourceFiles: WireU64;
  readonly sourceBytes: WireU64;
  readonly headerFiles: WireU64;
  readonly headerBytes: WireU64;
  readonly includeDepth: WireU64;
  readonly macroExpansions: WireU64;
  readonly preprocessedTokens: WireU64;
  readonly astNodes: WireU64;
  readonly constexprSteps: WireU64;
  readonly templateInstantiations: WireU64;
  readonly templateDepth: WireU64;
  readonly declarations: WireU64;
  readonly types: WireU64;
  readonly constants: WireU64;
  readonly layouts: WireU64;
  readonly tensors: WireU64;
  readonly operations: WireU64;
  readonly targetIntrinsics: WireU64;
  readonly diagnostics: WireU64;
  readonly outputBytes: WireU64;
  readonly wallTimeMs: WireU64;
  readonly cpuTimeMs: WireU64;
  readonly peakMemoryBytes: WireU64;
  readonly peakProcesses: WireU64;
}

export interface CppCuteAotRunnerReceiptV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_RECEIPT_SCHEMA;
  readonly version: CppCuteAotReceiptVersionV1;
  readonly receiptId: string;
  readonly jobId: string;
  readonly profileHash: string;
  readonly invocation: CppCuteAotReceiptInvocationV1;
  readonly openedInputs: CppCuteAotReceiptOpenedInputsV1;
  readonly selection: CppCuteAotReceiptSelectionV1;
  readonly output: CppCuteAotReceiptOutputV1;
  readonly resources: CppCuteAotReceiptResourcesV1;
  readonly outcome: "succeeded";
  readonly exitCode: 0;
}

export interface CppCuteAotRunnerReceiptBodyV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_RECEIPT_SCHEMA;
  readonly version: CppCuteAotReceiptVersionV1;
  readonly jobId: string;
  readonly profileHash: string;
  readonly invocation: CppCuteAotReceiptInvocationV1;
  readonly openedInputs: CppCuteAotReceiptOpenedInputsV1;
  readonly selection: CppCuteAotReceiptSelectionV1;
  readonly output: CppCuteAotReceiptOutputV1;
  readonly resources: CppCuteAotReceiptResourcesV1;
  readonly outcome: "succeeded";
  readonly exitCode: 0;
}

declare const verifiedCppCuteAotRunnerReceiptBrand: unique symbol;

export interface VerifiedCppCuteAotRunnerReceipt {
  readonly [verifiedCppCuteAotRunnerReceiptBrand]: true;
  readonly receiptId: string;
  readonly receiptBytesSha256: string;
  readonly receiptByteLength: WireU64;
  readonly jobId: string;
  readonly profileHash: string;
  readonly invocationId: string;
  readonly invocationManifestSha256: string;
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
  readonly receipt: CppCuteAotRunnerReceiptV1;
  readonly job: PreparedCppCuteAotJob;
  readonly profile: PreparedCppCuteFrontendProfile;
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
  | "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-JOB-MISMATCH"
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
  job: PreparedCppCuteAotJob,
  artifactResource: VerifiedCppCuteFrontendArtifactResource,
  value: unknown,
  options: VerifyCppCuteAotRunnerReceiptOptions = {},
): Promise<VerifiedCppCuteAotRunnerReceipt> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profile = jobRecord.profile;
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(profile);
  const artifact = unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource);
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const limits = normalizeOptions(options);
  throwIfAborted(options.signal);
  let receipt: CppCuteAotRunnerReceiptV1;
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

  verifyJobAndArtifactBindings(receipt, job, artifact, artifactRecord, profileRecord);
  await verifyInvocationBindings(receipt, job, profile, profileRecord);
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
    jobId: receipt.jobId,
    profileHash: receipt.profileHash,
    invocationId: receipt.invocation.invocationId,
    invocationManifestSha256: receipt.invocation.invocationManifestSha256,
    artifactId: receipt.output.artifactId,
    artifactHash: receipt.output.artifactHash,
    artifactBytesSha256: receipt.output.artifactBytesSha256,
    artifactByteLength: receipt.output.artifactByteLength,
  }) as VerifiedCppCuteAotRunnerReceipt;
  VERIFIED_RECEIPTS.set(verified, Object.freeze({ receipt, job, profile, artifactResource, artifact }));
  return verified;
}

export async function decodeCppCuteAotRunnerReceipt(
  job: PreparedCppCuteAotJob,
  artifactResource: VerifiedCppCuteFrontendArtifactResource,
  bytes: Uint8Array,
  options: VerifyCppCuteAotRunnerReceiptOptions = {},
): Promise<VerifiedCppCuteAotRunnerReceiptResource> {
  throwIfAborted(options.signal);
  const snapshot = new Uint8Array(bytes);
  const verified = await verifyCppCuteAotRunnerReceipt(
    job,
    artifactResource,
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
  receipt: CppCuteAotRunnerReceiptV1 | CppCuteAotRunnerReceiptBodyV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-runner-receipt.v1",
    receipt: {
      schema: receipt.schema,
      version: receipt.version,
      jobId: receipt.jobId,
      profileHash: receipt.profileHash,
      invocation: receipt.invocation,
      openedInputs: receipt.openedInputs,
      selection: receipt.selection,
      output: receipt.output,
      resources: receipt.resources,
      outcome: receipt.outcome,
      exitCode: receipt.exitCode,
    },
  }, options);
  return `bg.cpp.aot-receipt.sha256.${digest}`;
}

function parseReceipt(value: JsonValue): CppCuteAotRunnerReceiptV1 {
  const object = closedObject(value, [
    "schema", "version", "receiptId", "jobId", "profileHash", "invocation", "openedInputs", "selection",
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
    jobId: patterned(field(object, "jobId", "$"), "$.jobId", /^bg\.cpp\.aot-job\.sha256\.[0-9a-f]{64}$/u, "job ID"),
    profileHash: sha256(field(object, "profileHash", "$"), "$.profileHash"),
    invocation: parseInvocation(field(object, "invocation", "$"), "$.invocation"),
    openedInputs: parseOpenedInputs(field(object, "openedInputs", "$"), "$.openedInputs"),
    selection: parseSelection(field(object, "selection", "$"), "$.selection"),
    output: parseOutput(field(object, "output", "$"), "$.output"),
    resources: parseResources(field(object, "resources", "$"), "$.resources"),
    outcome: "succeeded",
    exitCode: 0,
  });
}

function parseVersion(value: JsonValue, path: string): CppCuteAotReceiptVersionV1 {
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

function parseInvocation(value: JsonValue, path: string): CppCuteAotReceiptInvocationV1 {
  const object = closedObject(value, [
    "invocationId", "invocationManifestSha256", "runner", "container", "extractor", "compiler",
    "dependencyManifestSha256", "sandbox",
  ], path);
  return {
    invocationId: patterned(field(object, "invocationId", path), `${path}.invocationId`, INVOCATION_ID, "invocation ID"),
    invocationManifestSha256: sha256(field(object, "invocationManifestSha256", path), `${path}.invocationManifestSha256`),
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
  const object = closedObject(value, ["runtime", "repository", "platform", "manifestDigest"], path);
  if (object.runtime !== "docker" || object.platform !== "linux/amd64") {
    invalid(path, "runner receipt requires Docker on resolved linux/amd64");
  }
  return {
    runtime: "docker",
    repository: boundedString(field(object, "repository", path), `${path}.repository`, 512),
    platform: "linux/amd64",
    manifestDigest: patterned(field(object, "manifestDigest", path), `${path}.manifestDigest`, OCI_SHA256, "OCI digest"),
  };
}

function parseExtractor(value: JsonValue, path: string): CppCuteFrontendExtractorProfile {
  const object = closedObject(value, ["id", "version", "buildId", "binarySha256"], path);
  return {
    id: boundedString(field(object, "id", path), `${path}.id`, 256),
    version: boundedString(field(object, "version", path), `${path}.version`, 128),
    buildId: boundedString(field(object, "buildId", path), `${path}.buildId`, 256),
    binarySha256: sha256(field(object, "binarySha256", path), `${path}.binarySha256`),
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

function parseSandbox(value: JsonValue, path: string): CppCuteAotReceiptSandboxV1 {
  const object = closedObject(value, [
    "contractId", "policySha256", "limitsSha256", "network", "readOnlyRoot", "noNewPrivileges", "linking",
    "nativeExecution",
  ], path);
  if (
    object.contractId !== "browsergrad.compiler.cpp-cute.aot@1"
    || object.network !== "none"
    || object.readOnlyRoot !== true
    || object.noNewPrivileges !== true
    || object.linking !== "forbidden"
    || object.nativeExecution !== "forbidden"
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
    nativeExecution: "forbidden",
  };
}

function parseOpenedInputs(value: JsonValue, path: string): CppCuteAotReceiptOpenedInputsV1 {
  const object = closedObject(value, ["files", "sourceSetSha256", "headerSetSha256", "inputClosureSha256"], path);
  const rawFiles = arrayValue(field(object, "files", path), `${path}.files`);
  if (rawFiles.length === 0) invalid(`${path}.files`, "receipt must report at least one opened source file");
  return {
    files: rawFiles.map((file, index) => parseOpenedFile(file, `${path}.files[${index}]`)),
    sourceSetSha256: sha256(field(object, "sourceSetSha256", path), `${path}.sourceSetSha256`),
    headerSetSha256: sha256(field(object, "headerSetSha256", path), `${path}.headerSetSha256`),
    inputClosureSha256: sha256(field(object, "inputClosureSha256", path), `${path}.inputClosureSha256`),
  };
}

function parseOpenedFile(value: JsonValue, path: string): CppCuteAotSourceFileV1 {
  const object = closedObject(value, ["fileId", "role", "virtualPath", "contentSha256", "byteLength"], path);
  if (object.role !== "main-source" && object.role !== "project-header") {
    invalid(`${path}.role`, "opened job source role must be main-source or project-header");
  }
  return {
    fileId: patterned(field(object, "fileId", path), `${path}.fileId`, STABLE_ID, "file ID"),
    role: object.role,
    virtualPath: boundedString(field(object, "virtualPath", path), `${path}.virtualPath`, 1_024),
    contentSha256: sha256(field(object, "contentSha256", path), `${path}.contentSha256`),
    byteLength: parseWireU64(field(object, "byteLength", path), `${path}.byteLength`),
  };
}

function parseSelection(value: JsonValue, path: string): CppCuteAotReceiptSelectionV1 {
  if (!isJsonObject(value)) invalid(path, "expected object");
  if (value.kind === "resolved") {
    const object = closedObject(value, ["kind", "requestId", "anchorTokenSha256", "expectedEntryId", "resolvedEntryId"], path);
    return {
      kind: "resolved",
      requestId: patterned(field(object, "requestId", path), `${path}.requestId`, STABLE_ID, "request ID"),
      anchorTokenSha256: sha256(field(object, "anchorTokenSha256", path), `${path}.anchorTokenSha256`),
      expectedEntryId: patterned(field(object, "expectedEntryId", path), `${path}.expectedEntryId`, STABLE_ID, "entry ID"),
      resolvedEntryId: patterned(field(object, "resolvedEntryId", path), `${path}.resolvedEntryId`, STABLE_ID, "entry ID"),
    };
  }
  if (value.kind === "rejected") {
    const object = closedObject(value, [
      "kind", "requestId", "anchorTokenSha256", "expectedEntryId", "blockingDiagnosticIds",
    ], path);
    const blockingDiagnosticIds = arrayValue(
      field(object, "blockingDiagnosticIds", path),
      `${path}.blockingDiagnosticIds`,
    ).map((entry, index) => patterned(
      entry,
      `${path}.blockingDiagnosticIds[${index}]`,
      STABLE_ID,
      "diagnostic ID",
    ));
    requireSortedUnique(blockingDiagnosticIds, `${path}.blockingDiagnosticIds`);
    if (blockingDiagnosticIds.length === 0) invalid(`${path}.blockingDiagnosticIds`, "rejected selection requires diagnostics");
    return {
      kind: "rejected",
      requestId: patterned(field(object, "requestId", path), `${path}.requestId`, STABLE_ID, "request ID"),
      anchorTokenSha256: sha256(field(object, "anchorTokenSha256", path), `${path}.anchorTokenSha256`),
      expectedEntryId: patterned(field(object, "expectedEntryId", path), `${path}.expectedEntryId`, STABLE_ID, "entry ID"),
      blockingDiagnosticIds,
    };
  }
  invalid(`${path}.kind`, "selection kind must be resolved or rejected");
}

function parseOutput(value: JsonValue, path: string): CppCuteAotReceiptOutputV1 {
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

function parseResources(value: JsonValue, path: string): CppCuteAotReceiptResourcesV1 {
  const fields = [
    "sourceFiles", "sourceBytes", "headerFiles", "headerBytes", "includeDepth", "macroExpansions",
    "preprocessedTokens", "astNodes", "constexprSteps", "templateInstantiations", "templateDepth",
    "declarations", "types", "constants", "layouts", "tensors", "operations", "targetIntrinsics",
    "diagnostics", "outputBytes", "wallTimeMs", "cpuTimeMs", "peakMemoryBytes", "peakProcesses",
  ] as const;
  const object = closedObject(value, fields, path);
  const resources = Object.fromEntries(
    fields.map((name) => [name, parseWireU64(field(object, name, path), `${path}.${name}`)]),
  ) as unknown as CppCuteAotReceiptResourcesV1;
  if (wireIntegerToBigInt(resources.peakMemoryBytes) === 0n || wireIntegerToBigInt(resources.peakProcesses) === 0n) {
    invalid(path, "successful receipt must report nonzero peak memory and process observations");
  }
  return resources;
}

function verifyJobAndArtifactBindings(
  receipt: CppCuteAotRunnerReceiptV1,
  job: PreparedCppCuteAotJob,
  artifact: VerifiedCppCuteFrontendArtifact,
  artifactRecord: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>,
  profileRecord: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>,
): void {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  if (receipt.jobId !== job.jobId || receipt.profileHash !== job.profileHash) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-JOB-MISMATCH", "$.jobId", "receipt differs from prepared job authority");
  }
  const expectedInputs: CppCuteAotReceiptOpenedInputsV1 = {
    files: jobRecord.job.files,
    sourceSetSha256: jobRecord.job.expectedOutput.sourceSetSha256,
    headerSetSha256: jobRecord.job.expectedOutput.headerSetSha256,
    inputClosureSha256: jobRecord.job.expectedOutput.inputClosureSha256,
  };
  if (canonicalText(receipt.openedInputs) !== canonicalText(expectedInputs)) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH", "$.openedInputs", "opened input report differs from the prepared request");
  }
  if (
    artifact.profileHash !== job.profileHash
    || artifact.sourceSetSha256 !== expectedInputs.sourceSetSha256
    || artifact.headerSetSha256 !== expectedInputs.headerSetSha256
    || artifact.inputClosureSha256 !== expectedInputs.inputClosureSha256
  ) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH", "$.artifact.inputs", "verified artifact closure differs from the prepared request");
  }
  const payload = artifactRecord.envelope.payload;
  const sourceOwnedFiles = payload.inputs.files.filter((file) => file.profileDependency === "none");
  if (sourceOwnedFiles.length !== jobRecord.job.files.length) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH", "$.artifact.inputs.files", "verified artifact has extra or missing source-owned files");
  }
  for (const [index, expected] of jobRecord.job.files.entries()) {
    const actual = sourceOwnedFiles.find((file) => file.virtualPath === expected.virtualPath);
    if (
      actual === undefined
      || actual.role !== expected.role
      || actual.virtualPath !== expected.virtualPath
      || actual.contentSha256 !== expected.contentSha256
      || actual.byteLength !== expected.byteLength
      || actual.profileDependency !== "none"
    ) {
      mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH", `$.openedInputs.files[${index}]`, "opened source differs from the verified artifact input closure");
    }
  }
  const mainFile = payload.inputs.files.find((file) => file.fileId === payload.inputs.mainFileId);
  if (mainFile === undefined || mainFile.virtualPath !== jobRecord.job.mainVirtualPath || mainFile.role !== "main-source") {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH", "$.artifact.inputs.mainFileId", "artifact main file differs from prepared job ownership");
  }
  const expectedRoots = profileRecord.profile.virtualFileSystem.includeRoots;
  if (payload.inputs.includeRoots.length !== expectedRoots.length) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH", "$.artifact.inputs.includeRoots", "artifact include-root count differs from prepared profile");
  }
  for (const [index, expected] of expectedRoots.entries()) {
    const actual = payload.inputs.includeRoots[index];
    if (
      actual === undefined
      || actual.ordinal !== index
      || actual.mode !== expected.mode
      || actual.virtualPath !== expected.virtualPath
      || actual.manifestSha256 !== expected.manifestSha256
    ) {
      mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INPUT-MISMATCH", `$.artifact.inputs.includeRoots[${index}]`, "artifact include-root precedence or manifest differs from prepared profile");
    }
  }
  const extractor = profileRecord.profile.deployment.extractor;
  if (
    artifactRecord.envelope.producer.id !== extractor.id
    || artifactRecord.envelope.producer.version !== extractor.version
  ) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH", "$.artifact.producer", "artifact producer differs from prepared extractor profile");
  }
  const request = jobRecord.job.entryRequests[0];
  if (request === undefined) mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-JOB-MISMATCH", "$.selection", "prepared job lost its entry request");
  const expectedSelection: CppCuteAotReceiptSelectionV1 = payload.outcome.kind === "accepted"
    ? {
        kind: "resolved",
        requestId: request.requestId,
        anchorTokenSha256: request.anchor.tokenSha256,
        expectedEntryId: request.expectedEntryId,
        resolvedEntryId: request.expectedEntryId,
      }
    : {
        kind: "rejected",
        requestId: request.requestId,
        anchorTokenSha256: request.anchor.tokenSha256,
        expectedEntryId: request.expectedEntryId,
        blockingDiagnosticIds: payload.outcome.blockingDiagnosticIds,
      };
  if (canonicalText(receipt.selection) !== canonicalText(expectedSelection)) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH", "$.selection", "resolved declaration differs from the prepared entry request");
  }
  if (payload.outcome.kind === "accepted" && (
    payload.outcome.selectedEntryIds.length !== 1
    || payload.outcome.selectedEntryIds[0] !== request.expectedEntryId
    || !payload.entries.some((entry) => entry.entryId === request.expectedEntryId)
  )) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH", "$.artifact.payload.outcome", "accepted artifact did not resolve exactly the requested entry");
  }
}

async function verifyInvocationBindings(
  receipt: CppCuteAotRunnerReceiptV1,
  job: PreparedCppCuteAotJob,
  profile: PreparedCppCuteFrontendProfile,
  profileRecord: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>,
): Promise<void> {
  const configured = profileRecord.profile;
  const invocationManifestSha256 = await computeCppCuteAotInvocationManifestHash(job);
  const expected: CppCuteAotReceiptInvocationV1 = {
    invocationId: `bg.cpp.aot-invocation.sha256.${invocationManifestSha256}`,
    invocationManifestSha256,
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
      nativeExecution: "forbidden",
    },
  };
  if (canonicalText(receipt.invocation) !== canonicalText(expected)) {
    mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-INVOCATION-MISMATCH", "$.invocation", "reported invocation differs from exact prepared profile and job");
  }
}

async function verifyOutputBindings(
  receipt: CppCuteAotRunnerReceiptV1,
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<void> {
  const expected: CppCuteAotReceiptOutputV1 = {
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
  resources: CppCuteAotReceiptResourcesV1,
  profile: PreparedCppCuteFrontendProfile,
  artifact: VerifiedCppCuteFrontendArtifact,
  artifactRecord: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>,
): void {
  const configured = unwrapPreparedCppCuteFrontendProfile(profile).profile.extractionLimits;
  const cases: readonly [keyof CppCuteAotReceiptResourcesV1, number][] = [
    ["sourceFiles", configured.maxSourceFiles],
    ["sourceBytes", configured.maxSourceBytes],
    ["headerFiles", configured.maxHeaderFiles],
    ["headerBytes", configured.maxHeaderBytes],
    ["includeDepth", configured.maxIncludeDepth],
    ["macroExpansions", configured.maxMacroExpansions],
    ["preprocessedTokens", configured.maxPreprocessedTokens],
    ["astNodes", configured.maxAstNodes],
    ["constexprSteps", configured.maxConstexprSteps],
    ["templateInstantiations", configured.maxTemplateInstantiations],
    ["templateDepth", configured.maxTemplateDepth],
    ["declarations", configured.maxDeclarations],
    ["types", configured.maxTypes],
    ["constants", configured.maxConstants],
    ["layouts", configured.maxLayouts],
    ["tensors", configured.maxTensors],
    ["operations", configured.maxOperations],
    ["targetIntrinsics", configured.maxTargetIntrinsics],
    ["diagnostics", configured.maxDiagnostics],
    ["outputBytes", configured.maxOutputBytes],
    ["wallTimeMs", configured.maxWallTimeMs],
    ["cpuTimeMs", configured.maxCpuTimeMs],
    ["peakMemoryBytes", configured.maxMemoryBytes],
    ["peakProcesses", configured.maxProcesses],
  ];
  for (const [fieldName, maximum] of cases) {
    if (wireIntegerToBigInt(resources[fieldName] as WireU64) > BigInt(maximum)) {
      resource(`$.resources.${fieldName}`, `observed resource use exceeds profile maximum ${maximum}`);
    }
  }
  const payload = artifactRecord.envelope.payload;
  const sources = payload.inputs.files.filter((file) => file.profileDependency === "none");
  const headers = payload.inputs.files.filter((file) => file.profileDependency !== "none");
  const exact: Readonly<Partial<Record<keyof CppCuteAotReceiptResourcesV1, bigint>>> = {
    sourceFiles: BigInt(sources.length),
    sourceBytes: sumFileBytes(sources),
    headerFiles: BigInt(headers.length),
    headerBytes: sumFileBytes(headers),
    macroExpansions: BigInt(payload.macroExpansions.length),
    templateInstantiations: BigInt(payload.templateInstantiations.length),
    declarations: BigInt(payload.declarations.length),
    types: BigInt(payload.types.length),
    constants: BigInt(payload.constants.length),
    layouts: BigInt(payload.facts.filter((fact) => fact.kind === "affine-layout").length),
    tensors: BigInt(payload.facts.filter((fact) => fact.kind === "tensor").length),
    operations: BigInt(payload.facts.filter((fact) => (
      fact.kind !== "affine-layout" && fact.kind !== "tensor" && fact.kind !== "target-intrinsic"
    )).length),
    targetIntrinsics: BigInt(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length),
    diagnostics: BigInt(payload.diagnostics.length),
    outputBytes: wireIntegerToBigInt(artifact.artifactByteLength),
  };
  for (const [fieldName, expected] of Object.entries(exact) as Array<[keyof CppCuteAotReceiptResourcesV1, bigint]>) {
    if (wireIntegerToBigInt(resources[fieldName] as WireU64) !== expected) {
      mismatch("BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH", `$.resources.${fieldName}`, "reported extraction count differs from verified artifact");
    }
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

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
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

function requireSortedUnique(values: readonly string[], path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined || previous.localeCompare(current) >= 0) {
      invalid(path, "set-like identifiers must be strictly sorted and unique");
    }
  }
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
