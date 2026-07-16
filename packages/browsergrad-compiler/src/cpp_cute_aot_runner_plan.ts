import {
  canonicalJsonBytes,
  encodeWireU64,
  sha256Hex,
  wireIntegerToBigInt,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteAotJob,
  type CppCuteAotSourceFileV2,
  type PreparedCppCuteAotJob,
} from "./cpp_cute_aot_job.js";
import {
  CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS,
  CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS,
  CPP_CUTE_AOT_SANDBOX_POLICY_SHA256,
  computeCppCuteAotExecutionPlanHash,
  verifyCppCuteAotSandboxPolicyIdentity,
} from "./cpp_cute_aot_policy.js";
import {
  copyPreparedCppCuteAotExecutionEnvironmentBytes,
  unwrapPreparedCppCuteAotExecutionEnvironment,
  type PreparedCppCuteAotExecutionEnvironment,
} from "./cpp_cute_aot_environment.js";
import {
  decodeCppCuteAotRunnerReceipt,
  type VerifiedCppCuteAotRunnerReceiptResource,
} from "./cpp_cute_aot_receipt.js";
import {
  decodeCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  type VerifiedCppCuteFrontendArtifactResource,
} from "./cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
  type InspectedUnsharedUint8Array,
} from "./cpp_cute_aot_bytes.js";

export const CPP_CUTE_AOT_RESULT_FRAME_MAGIC = "BGCUTE-AOT-R1\n";
const FRAME_MAGIC_BYTES = new TextEncoder().encode(CPP_CUTE_AOT_RESULT_FRAME_MAGIC);
const FRAME_LENGTH_BYTES = 16;
const PREPARED_RUNS = new WeakMap<object, StoredCppCuteAotOfflineRunRecord>();
const VERIFIED_RESULTS = new WeakMap<object, StoredCppCuteAotOfflineResultRecord>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export interface CppCuteAotSourceBlob {
  readonly fileId: string;
  readonly bytes: Uint8Array;
}

interface SnapshottedCppCuteAotSourceBlob {
  readonly file: CppCuteAotSourceFileV2;
  readonly bytes: Uint8Array;
}

declare const preparedCppCuteAotOfflineRunBrand: unique symbol;

/** Opaque, output-independent authority over exact snapshotted source bytes. */
export interface PreparedCppCuteAotOfflineRun {
  readonly [preparedCppCuteAotOfflineRunBrand]: true;
  readonly jobId: string;
  readonly profileHash: string;
  readonly executionPlanSha256: string;
  readonly imageReference: string;
  readonly imageConfigDigest: string;
  readonly executionEnvironmentManifestSha256: string;
  readonly sourceFileCount: number;
  readonly sourceBytes: WireU64;
  readonly artifactByteLimit: number;
  readonly receiptByteLimit: number;
  readonly frameByteLimit: number;
}

export interface PreparedCppCuteAotOfflineRunRecord {
  readonly job: PreparedCppCuteAotJob;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly sourceFiles: readonly CppCuteAotSourceFileV2[];
  readonly executionPlanSha256: string;
  readonly artifactByteLimit: number;
  readonly receiptByteLimit: number;
  readonly frameByteLimit: number;
}

interface StoredCppCuteAotOfflineRunRecord extends PreparedCppCuteAotOfflineRunRecord {
  readonly sourceBlobs: readonly SnapshottedCppCuteAotSourceBlob[];
}

declare const verifiedCppCuteAotOfflineResultBrand: unique symbol;

/**
 * Canonical frame/job consistency only. This is not container-execution,
 * sandbox-enforcement, producer-trust, provenance, or attestation authority.
 */
export interface VerifiedCppCuteAotOfflineResult {
  readonly [verifiedCppCuteAotOfflineResultBrand]: true;
  readonly jobId: string;
  readonly profileHash: string;
  readonly executionPlanSha256: string;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly receiptResource: VerifiedCppCuteAotRunnerReceiptResource;
  readonly artifactByteLength: WireU64;
  readonly receiptByteLength: WireU64;
  readonly frontendOutcome: "accepted" | "rejected";
}

export interface VerifiedCppCuteAotOfflineResultRecord {
  readonly plan: PreparedCppCuteAotOfflineRun;
}

interface StoredCppCuteAotOfflineResultRecord extends VerifiedCppCuteAotOfflineResultRecord {
  readonly artifactBytes: Uint8Array;
  readonly receiptBytes: Uint8Array;
}

export interface CppCuteAotOfflineResultBytes {
  readonly artifactBytes: Uint8Array;
  readonly receiptBytes: Uint8Array;
}

export interface CppCuteAotOfflineStagingSourceBlob extends CppCuteAotSourceBlob {
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
}

/** Disposable canonical control/source bytes for the private Node staging shell. */
export interface CppCuteAotOfflineStagingInputs {
  readonly profileBytes: Uint8Array;
  readonly jobBytes: Uint8Array;
  readonly environmentBytes: Uint8Array;
  readonly sourceBlobs: readonly CppCuteAotOfflineStagingSourceBlob[];
}

export interface PrepareCppCuteAotOfflineRunOptions {
  readonly signal?: AbortSignal;
}

export interface DecodeCppCuteAotResultFrameOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteAotOfflineRunnerErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-POLICY-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-RUNNER-UNVERIFIED";

export class CppCuteAotOfflineRunnerError extends Error {
  constructor(
    readonly code: CppCuteAotOfflineRunnerErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotOfflineRunnerError";
  }
}

/**
 * Snapshots caller bytes synchronously, then verifies exact job ownership.
 * The returned authority contains no caller paths, commands, environment, or
 * output. Node staging later consumes only the private byte snapshots.
 */
export async function prepareCppCuteAotOfflineRun(
  job: PreparedCppCuteAotJob,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  sourceBlobs: readonly CppCuteAotSourceBlob[],
  options: PrepareCppCuteAotOfflineRunOptions = {},
): Promise<PreparedCppCuteAotOfflineRun> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profile = jobRecord.profile;
  const profileRecord = unwrapPreparedCppCuteAotFrontendProfile(profile);
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const snapshots = snapshotSourceBlobs(sourceBlobs, jobRecord.job.files);
  throwIfAborted(signal);
  const environmentRecord = unwrapPreparedCppCuteAotExecutionEnvironment(executionEnvironment);
  if (environmentRecord.profile !== profile || executionEnvironment.profileHash !== profile.profileHash) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUNNER-POLICY-MISMATCH",
      "$.executionEnvironment",
      "execution environment belongs to a different prepared profile",
    );
  }
  if (profileRecord.profile.deployment.sandboxPolicySha256 !== CPP_CUTE_AOT_SANDBOX_POLICY_SHA256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUNNER-POLICY-MISMATCH",
      "$.profile.deployment.sandboxPolicySha256",
      "prepared profile does not name the built-in offline-runner sandbox policy",
    );
  }
  await verifyCppCuteAotSandboxPolicyIdentity();
  const verifiedSources = await verifySourceSnapshots(jobRecord.job.files, snapshots, jobRecord.job.entryRequests[0]);
  throwIfAborted(signal);
  const executionPlanSha256 = await computeCppCuteAotExecutionPlanHash(job, executionEnvironment);
  throwIfAborted(signal);
  const artifactByteLimit = Math.min(
    profile.extractionLimits.maxOutputBytes,
    CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS.maxDocumentBytes,
  );
  const receiptByteLimit = CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS.maxDocumentBytes;
  const frameByteLimit = FRAME_MAGIC_BYTES.byteLength + FRAME_LENGTH_BYTES + artifactByteLimit + receiptByteLimit;
  const sourceBytes = jobRecord.job.files.reduce(
    (total, file) => total + wireIntegerToBigInt(file.byteLength),
    0n,
  );
  const container = profileRecord.profile.deployment.container;
  const prepared = Object.freeze({
    jobId: job.jobId,
    profileHash: profile.profileHash,
    executionPlanSha256,
    imageReference: `${container.repository}@${container.manifestDigest}`,
    imageConfigDigest: container.configDigest,
    executionEnvironmentManifestSha256:
      executionEnvironment.manifestSha256,
    sourceFileCount: verifiedSources.length,
    sourceBytes: encodeWireU64(sourceBytes),
    artifactByteLimit,
    receiptByteLimit,
    frameByteLimit,
  }) as PreparedCppCuteAotOfflineRun;
  PREPARED_RUNS.set(prepared, Object.freeze({
    job,
    profile,
    executionEnvironment,
    sourceFiles: jobRecord.job.files,
    sourceBlobs: verifiedSources,
    executionPlanSha256,
    artifactByteLimit,
    receiptByteLimit,
    frameByteLimit,
  }));
  return prepared;
}

export function unwrapPreparedCppCuteAotOfflineRun(
  prepared: PreparedCppCuteAotOfflineRun,
): PreparedCppCuteAotOfflineRunRecord {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_RUNS.get(prepared as object);
  if (record === undefined) unverified();
  return Object.freeze({
    job: record.job,
    profile: record.profile,
    executionEnvironment: record.executionEnvironment,
    sourceFiles: record.sourceFiles,
    executionPlanSha256: record.executionPlanSha256,
    artifactByteLimit: record.artifactByteLimit,
    receiptByteLimit: record.receiptByteLimit,
    frameByteLimit: record.frameByteLimit,
  });
}

/** Returns disposable copies for the Node staging shell; copies hold no authority. */
export function copyCppCuteAotOfflineRunSourceBlobs(
  prepared: PreparedCppCuteAotOfflineRun,
): readonly CppCuteAotSourceBlob[] {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_RUNS.get(prepared as object);
  if (record === undefined) unverified();
  return Object.freeze(record.sourceBlobs.map(({ file, bytes }) => Object.freeze({
    fileId: file.fileId,
    bytes: new Uint8Array(bytes),
  })));
}

/**
 * Returns canonical profile/job bytes plus exact source snapshots. Returned
 * arrays hold no authority and may be discarded or mutated by the caller.
 */
export function copyCppCuteAotOfflineRunStagingInputs(
  prepared: PreparedCppCuteAotOfflineRun,
): CppCuteAotOfflineStagingInputs {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_RUNS.get(prepared as object);
  if (record === undefined) unverified();
  const profile = unwrapPreparedCppCuteAotFrontendProfile(record.profile).profile;
  const job = unwrapPreparedCppCuteAotJob(record.job).job;
  return Object.freeze({
    profileBytes: new Uint8Array(canonicalJsonBytes(profile)),
    jobBytes: new Uint8Array(canonicalJsonBytes(job)),
    environmentBytes: copyPreparedCppCuteAotExecutionEnvironmentBytes(record.executionEnvironment),
    sourceBlobs: Object.freeze(record.sourceBlobs.map(({ file, bytes }) => Object.freeze({
      fileId: file.fileId,
      virtualPath: file.virtualPath,
      contentSha256: file.contentSha256,
      byteLength: file.byteLength,
      bytes: new Uint8Array(bytes),
    }))),
  });
}

export function unwrapVerifiedCppCuteAotOfflineResult(
  result: VerifiedCppCuteAotOfflineResult,
): VerifiedCppCuteAotOfflineResultRecord {
  if (typeof result !== "object" || result === null) unverified();
  const record = VERIFIED_RESULTS.get(result as object);
  if (record === undefined) unverified();
  return Object.freeze({ plan: record.plan });
}

/** Returns disposable exact-byte copies; mutation cannot affect verified authority. */
export function copyCppCuteAotOfflineResultBytes(
  result: VerifiedCppCuteAotOfflineResult,
): CppCuteAotOfflineResultBytes {
  if (typeof result !== "object" || result === null) unverified();
  const record = VERIFIED_RESULTS.get(result as object);
  if (record === undefined) unverified();
  return Object.freeze({
    artifactBytes: new Uint8Array(record.artifactBytes),
    receiptBytes: new Uint8Array(record.receiptBytes),
  });
}

/** Strictly parses one frame and mints artifact/receipt byte authorities. */
export async function decodeCppCuteAotResultFrame(
  plan: PreparedCppCuteAotOfflineRun,
  bytes: Uint8Array,
  options: DecodeCppCuteAotResultFrameOptions = {},
): Promise<VerifiedCppCuteAotOfflineResult> {
  const record = unwrapPreparedCppCuteAotOfflineRun(plan);
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const inspected = inspectFrameBytes(bytes, "$bytes");
  if (inspected.byteLength > record.frameByteLimit) {
    resource("$bytes", `result frame exceeds ${record.frameByteLimit} bytes`);
  }
  const snapshot = copyFrameBytes(bytes, inspected, "$bytes");
  const { artifactBytes, receiptBytes } = parseResultFrame(snapshot, record);
  throwIfAborted(signal);
  const artifactDecodeLimits = {
    ...CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS,
    maxDocumentBytes: record.artifactByteLimit,
  };
  const artifactResource = await decodeCppCuteFrontendArtifact(
    artifactBytes,
    signal === undefined
      ? { limits: artifactDecodeLimits }
      : { limits: artifactDecodeLimits, signal },
  );
  const artifact = unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource);
  const receiptResource = await decodeCppCuteAotRunnerReceipt(
    record.job,
    record.executionEnvironment,
    artifactResource,
    receiptBytes,
    signal === undefined
      ? { limits: CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS }
      : { limits: CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS, signal },
  );
  throwIfAborted(signal);
  const result = Object.freeze({
    jobId: plan.jobId,
    profileHash: plan.profileHash,
    executionPlanSha256: plan.executionPlanSha256,
    artifactResource,
    receiptResource,
    artifactByteLength: String(artifactBytes.byteLength) as WireU64,
    receiptByteLength: String(receiptBytes.byteLength) as WireU64,
    frontendOutcome: artifact.outcome,
  }) as VerifiedCppCuteAotOfflineResult;
  VERIFIED_RESULTS.set(result, Object.freeze({
    plan,
    artifactBytes: new Uint8Array(artifactBytes),
    receiptBytes: new Uint8Array(receiptBytes),
  }));
  return result;
}

/** Deterministic producer-frame encoder; it grants no verification authority. */
export function encodeCppCuteAotResultFrame(
  artifactBytes: Uint8Array,
  receiptBytes: Uint8Array,
): Uint8Array {
  const artifact = inspectFrameBytes(artifactBytes, "$artifactBytes");
  const receipt = inspectFrameBytes(receiptBytes, "$receiptBytes");
  if (artifact.byteLength === 0 || receipt.byteLength === 0) {
    frameInvalid("$bytes", "frame payloads must be nonempty");
  }
  if (artifact.byteLength > CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS.maxDocumentBytes) {
    resource("$artifactBytes", "artifact payload exceeds the hard frame encoder ceiling");
  }
  if (receipt.byteLength > CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS.maxDocumentBytes) {
    resource("$receiptBytes", "receipt payload exceeds the hard frame encoder ceiling");
  }
  const artifactSnapshot = copyFrameBytes(artifactBytes, artifact, "$artifactBytes");
  const receiptSnapshot = copyFrameBytes(receiptBytes, receipt, "$receiptBytes");
  const total = FRAME_MAGIC_BYTES.byteLength + FRAME_LENGTH_BYTES
    + artifact.byteLength + receipt.byteLength;
  const hardFrameLimit = FRAME_MAGIC_BYTES.byteLength + FRAME_LENGTH_BYTES
    + CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS.maxDocumentBytes
    + CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS.maxDocumentBytes;
  if (!Number.isSafeInteger(total) || total > hardFrameLimit) {
    resource("$bytes", "result frame exceeds the hard aggregate encoder ceiling");
  }
  const frame = new Uint8Array(total);
  frame.set(FRAME_MAGIC_BYTES, 0);
  const view = new DataView(frame.buffer);
  view.setBigUint64(FRAME_MAGIC_BYTES.byteLength, BigInt(artifact.byteLength), false);
  view.setBigUint64(FRAME_MAGIC_BYTES.byteLength + 8, BigInt(receipt.byteLength), false);
  frame.set(artifactSnapshot, FRAME_MAGIC_BYTES.byteLength + FRAME_LENGTH_BYTES);
  frame.set(
    receiptSnapshot,
    FRAME_MAGIC_BYTES.byteLength + FRAME_LENGTH_BYTES + artifact.byteLength,
  );
  return frame;
}

function snapshotSourceBlobs(
  value: readonly CppCuteAotSourceBlob[],
  expectedFiles: readonly CppCuteAotSourceFileV2[],
): readonly { readonly fileId: string; readonly bytes: Uint8Array }[] {
  let arrayDescriptors: Record<string, PropertyDescriptor>;
  let arrayKeys: readonly PropertyKey[];
  let arrayPrototype: object | null;
  try {
    arrayPrototype = Object.getPrototypeOf(value);
    arrayDescriptors = Object.getOwnPropertyDescriptors(value) as Record<string, PropertyDescriptor>;
    arrayKeys = Reflect.ownKeys(arrayDescriptors);
  } catch (cause) {
    invalid("$sourceBlobs", "source blobs must be an inspectable plain dense array", { cause });
  }
  if (!Array.isArray(value) || arrayPrototype !== Array.prototype) {
    invalid("$sourceBlobs", "source blobs must be a plain dense array");
  }
  const length = arrayDescriptors.length?.value;
  if (typeof length !== "number" || length !== expectedFiles.length) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      "$sourceBlobs",
      `source blob count must equal prepared job count ${expectedFiles.length}`,
    );
  }
  for (let index = 0; index < length; index += 1) {
    const descriptor = arrayDescriptors[String(index)];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`$sourceBlobs[${index}]`, "source arrays must contain enumerable data elements");
    }
  }
  for (const key of arrayKeys) {
    if (key === "length") continue;
    if (typeof key !== "string" || !/^(?:0|[1-9][0-9]*)$/u.test(key) || Number(key) >= length) {
      invalid("$sourceBlobs", "source array contains extra properties");
    }
  }
  const expectedById = new Map(expectedFiles.map((file) => [file.fileId, file] as const));
  if (expectedById.size !== expectedFiles.length) {
    invalid("$.job.files", "prepared job contains duplicate source file IDs");
  }
  const seen = new Set<string>();
  const snapshots: Array<{ readonly fileId: string; readonly bytes: Uint8Array }> = [];
  for (let index = 0; index < length; index += 1) {
    const entry = arrayDescriptors[String(index)]?.value as unknown;
    let entryPrototype: object | null;
    let descriptors: PropertyDescriptorMap;
    let keys: readonly PropertyKey[];
    try {
      entryPrototype = typeof entry === "object" && entry !== null
        ? Object.getPrototypeOf(entry)
        : null;
      descriptors = typeof entry === "object" && entry !== null
        ? Object.getOwnPropertyDescriptors(entry)
        : {};
      keys = Reflect.ownKeys(descriptors);
    } catch (cause) {
      invalid(`$sourceBlobs[${index}]`, "source blob must be an inspectable plain object", { cause });
    }
    if (typeof entry !== "object" || entry === null || entryPrototype !== Object.prototype) {
      invalid(`$sourceBlobs[${index}]`, "source blob must be a plain object");
    }
    if (keys.length !== 2 || !keys.includes("fileId") || !keys.includes("bytes")) {
      invalid(`$sourceBlobs[${index}]`, "source blob fields must be exactly fileId and bytes");
    }
    for (const key of ["fileId", "bytes"] as const) {
      const descriptor = descriptors[key];
      if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
        invalid(`$sourceBlobs[${index}].${key}`, "source blob fields must be enumerable data properties");
      }
    }
    const fileId = descriptors.fileId?.value as unknown;
    const bytesValue = descriptors.bytes?.value as unknown;
    if (typeof fileId !== "string") invalid(`$sourceBlobs[${index}].fileId`, "fileId must be a string");
    const expected = expectedById.get(fileId);
    if (expected === undefined) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].fileId`,
        "source blob ID is not owned by the prepared job",
      );
    }
    if (seen.has(fileId)) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].fileId`,
        "source blob IDs must be unique",
      );
    }
    const inspected = inspectSourceBytes(bytesValue, `$sourceBlobs[${index}].bytes`);
    if (BigInt(inspected.byteLength) !== wireIntegerToBigInt(expected.byteLength)) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].bytes`,
        "source byte length differs from prepared job",
      );
    }
    seen.add(fileId);
    snapshots.push(Object.freeze({
      fileId,
      bytes: copySourceBytes(bytesValue, inspected, `$sourceBlobs[${index}].bytes`),
    }));
  }
  return Object.freeze(snapshots);
}

async function verifySourceSnapshots(
  expectedFiles: readonly CppCuteAotSourceFileV2[],
  snapshots: readonly { readonly fileId: string; readonly bytes: Uint8Array }[],
  request: ReturnType<typeof unwrapPreparedCppCuteAotJob>["job"]["entryRequests"][number] | undefined,
): Promise<readonly SnapshottedCppCuteAotSourceBlob[]> {
  const byId = new Map<string, Uint8Array>();
  for (const [index, snapshot] of snapshots.entries()) {
    if (byId.has(snapshot.fileId)) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].fileId`,
        "source blob IDs must be unique",
      );
    }
    byId.set(snapshot.fileId, snapshot.bytes);
  }
  const result: SnapshottedCppCuteAotSourceBlob[] = [];
  for (const [index, file] of expectedFiles.entries()) {
    const bytes = byId.get(file.fileId);
    if (bytes === undefined) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].fileId`,
        `missing source bytes for ${file.fileId}`,
      );
    }
    if (BigInt(bytes.byteLength) !== wireIntegerToBigInt(file.byteLength)) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].bytes`,
        "source byte length differs from prepared job",
      );
    }
    if (await sha256Hex(bytes) !== file.contentSha256) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
        `$sourceBlobs[${index}].bytes`,
        "source digest differs from prepared job",
      );
    }
    result.push(Object.freeze({ file, bytes }));
  }
  if (request === undefined) invalid("$.job.entryRequests", "prepared job lost its entry request");
  const anchorFile = result.find(({ file }) => file.virtualPath === request.anchor.virtualPath);
  if (anchorFile === undefined) invalid("$.job.entryRequests[0].anchor", "entry anchor source disappeared");
  const begin = Number(wireIntegerToBigInt(request.anchor.beginByte));
  const end = Number(wireIntegerToBigInt(request.anchor.endByte));
  if (await sha256Hex(anchorFile.bytes.subarray(begin, end)) !== request.anchor.tokenSha256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUNNER-SOURCE-MISMATCH",
      "$.job.entryRequests[0].anchor.tokenSha256",
      "source anchor token differs from prepared request",
    );
  }
  return Object.freeze(result);
}

function parseResultFrame(
  bytes: Uint8Array,
  record: PreparedCppCuteAotOfflineRunRecord,
): { readonly artifactBytes: Uint8Array; readonly receiptBytes: Uint8Array } {
  const headerBytes = FRAME_MAGIC_BYTES.byteLength + FRAME_LENGTH_BYTES;
  if (bytes.byteLength < headerBytes) frameInvalid("$bytes", "result frame is truncated before lengths");
  for (let index = 0; index < FRAME_MAGIC_BYTES.byteLength; index += 1) {
    if (bytes[index] !== FRAME_MAGIC_BYTES[index]) frameInvalid("$bytes", "result frame magic is invalid");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const artifactLength = view.getBigUint64(FRAME_MAGIC_BYTES.byteLength, false);
  const receiptLength = view.getBigUint64(FRAME_MAGIC_BYTES.byteLength + 8, false);
  if (artifactLength === 0n || receiptLength === 0n) frameInvalid("$bytes", "result payloads must be nonempty");
  if (artifactLength > BigInt(record.artifactByteLimit)) {
    resource("$bytes.artifact", `artifact payload exceeds ${record.artifactByteLimit} bytes`);
  }
  if (receiptLength > BigInt(record.receiptByteLimit)) {
    resource("$bytes.receipt", `receipt payload exceeds ${record.receiptByteLimit} bytes`);
  }
  const total = BigInt(headerBytes) + artifactLength + receiptLength;
  if (total !== BigInt(bytes.byteLength)) {
    frameInvalid("$bytes", "result frame is truncated or contains trailing/multiple output");
  }
  const artifactEnd = headerBytes + Number(artifactLength);
  return {
    artifactBytes: bytes.slice(headerBytes, artifactEnd),
    receiptBytes: bytes.slice(artifactEnd),
  };
}

function inspectSourceBytes(value: unknown, path: string): InspectedUnsharedUint8Array {
  try {
    return inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "source bytes must be an unshared plain Uint8Array", { cause });
  }
}

function copySourceBytes(
  value: unknown,
  inspected: InspectedUnsharedUint8Array,
  path: string,
): Uint8Array {
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid(path, "source bytes became unreadable while snapshotting", { cause });
  }
}

function inspectFrameBytes(value: unknown, path: string): InspectedUnsharedUint8Array {
  try {
    return inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    frameInvalid(path, "frame bytes must be an unshared plain Uint8Array", { cause });
  }
}

function copyFrameBytes(
  value: unknown,
  inspected: InspectedUnsharedUint8Array,
  path: string,
): Uint8Array {
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    frameInvalid(path, "frame bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(
  options: PrepareCppCuteAotOfflineRunOptions | DecodeCppCuteAotResultFrameOptions,
): AbortSignal | undefined {
  if (typeof options !== "object" || options === null || Array.isArray(options)) {
    invalid("$options", "options must be a plain object");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = Object.getPrototypeOf(options);
    descriptors = Object.getOwnPropertyDescriptors(options);
    keys = Reflect.ownKeys(descriptors);
  } catch (cause) {
    invalid("$options", "options must be an inspectable plain object", { cause });
  }
  if (prototype !== Object.prototype && prototype !== null) invalid("$options", "options must be a plain object");
  for (const key of keys) {
    if (key !== "signal") invalid("$options", "options contain unknown fields");
    const descriptor = descriptors.signal;
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid("$options.signal", "options require enumerable data properties without accessors");
    }
  }
  const signal = descriptors.signal?.value as unknown;
  if (signal !== undefined && !isAbortSignal(signal)) {
    invalid("$options.signal", "signal must be an AbortSignal");
  }
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
    invalid("$.signal", "signal is not a readable AbortSignal", { cause });
  }
  if (aborted === true) {
    fail("BG-COMPILER-CPP-CUTE-AOT-RUNNER-CANCELLED", "$.signal", "offline AOT run was aborted");
  }
}

function unverified(): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-RUNNER-UNVERIFIED", "$", "expected opaque offline-runner authority");
}

function frameInvalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-RUNNER-FRAME-INVALID", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-RUNNER-RESOURCE-LIMIT", path, message);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-RUNNER-INVALID", path, message, options);
}

function mismatch(code: CppCuteAotOfflineRunnerErrorCode, path: string, message: string): never {
  fail(code, path, message);
}

function fail(
  code: CppCuteAotOfflineRunnerErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteAotOfflineRunnerError(code, path, message, options);
}
