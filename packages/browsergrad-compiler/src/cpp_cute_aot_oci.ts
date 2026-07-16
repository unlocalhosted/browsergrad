import {
  decodeWireJson,
  sha256Hex,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
  type InspectedUnsharedUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_AOT_OCI_CONFIG_DECODE_LIMITS,
  CPP_CUTE_AOT_OCI_MANIFEST_DECODE_LIMITS,
  CPP_CUTE_AOT_OCI_RESOURCE_LIMITS,
} from "./cpp_cute_aot_policy.js";
import {
  unwrapPreparedCppCuteAotExecutionEnvironment,
  type CppCuteAotExecutionEnvironmentLayer,
} from "./cpp_cute_aot_environment.js";
import {
  unwrapPreparedCppCuteAotOfflineRun,
  type PreparedCppCuteAotOfflineRun,
} from "./cpp_cute_aot_runner_plan.js";

const OCI_IMAGE_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json";
const OCI_IMAGE_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json";
const OCI_LAYER_MEDIA_TYPES = new Set([
  "application/vnd.oci.image.layer.v1.tar",
  "application/vnd.oci.image.layer.v1.tar+gzip",
  "application/vnd.oci.image.layer.v1.tar+zstd",
]);
const OCI_DIGEST = /^sha256:[0-9a-f]{64}$/u;
const RFC_3339 = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))$/u;
const VERIFIED_OCI_METADATA = new WeakMap<object, StoredCppCuteAotOciMetadata>();
const AUTHORIZED_OCI_METADATA = new WeakMap<object, StoredAuthorizedCppCuteAotOciMetadata>();
const TEXT_ENCODER = new TextEncoder();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export interface CppCuteAotOciMetadataBytes {
  readonly manifestBytes: Uint8Array;
  readonly configBytes: Uint8Array;
}

export interface VerifyCppCuteAotOciMetadataOptions {
  readonly signal?: AbortSignal;
}

export interface CppCuteAotOciDescriptor {
  readonly mediaType: string;
  readonly digest: string;
  readonly size: number;
}

declare const verifiedCppCuteAotOciMetadataBrand: unique symbol;

/**
 * Exact immutable/cacheable OCI manifest/config metadata. Layer descriptors
 * are verified, but layer blob bytes are not supplied or rehashed. This
 * authority does not claim plan authorization, acquisition, local Docker
 * presence, container creation, or execution.
 */
export interface VerifiedCppCuteAotOciMetadata {
  readonly [verifiedCppCuteAotOciMetadataBrand]: true;
  readonly manifest: CppCuteAotOciDescriptor;
  readonly config: CppCuteAotOciDescriptor;
  readonly layerCount: number;
  readonly totalLayerBytes: number;
}

interface StoredCppCuteAotOciMetadata {
  readonly layers: readonly CppCuteAotOciDescriptor[];
  readonly diffIds: readonly string[];
}

declare const authorizedCppCuteAotOciMetadataBrand: unique symbol;

/** Exact OCI manifest/config metadata authorized for one prepared plan. */
export interface AuthorizedCppCuteAotOciMetadata {
  readonly [authorizedCppCuteAotOciMetadataBrand]: true;
  readonly jobId: string;
  readonly profileHash: string;
  readonly executionPlanSha256: string;
  readonly imageReference: string;
  readonly manifestDigest: string;
  readonly configDigest: string;
  readonly layerCount: number;
  readonly totalLayerBytes: number;
}

export interface AuthorizedCppCuteAotOciMetadataRecord {
  readonly plan: PreparedCppCuteAotOfflineRun;
  readonly metadata: VerifiedCppCuteAotOciMetadata;
}

interface StoredAuthorizedCppCuteAotOciMetadata extends AuthorizedCppCuteAotOciMetadataRecord {
  readonly layers: readonly CppCuteAotOciDescriptor[];
  readonly diffIds: readonly string[];
}

export type CppCuteAotOciErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-PLATFORM-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-OCI-UNVERIFIED";

export class CppCuteAotOciError extends Error {
  constructor(
    readonly code: CppCuteAotOciErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotOciError";
  }
}

export async function verifyCppCuteAotOciMetadata(
  evidence: CppCuteAotOciMetadataBytes,
  options: VerifyCppCuteAotOciMetadataOptions = {},
): Promise<VerifiedCppCuteAotOciMetadata> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const snapshots = snapshotEvidence(evidence);
  throwIfAborted(signal);

  const manifestBytesSha256 = await sha256Hex(snapshots.manifestBytes);
  throwIfAborted(signal);
  const configBytesSha256 = await sha256Hex(snapshots.configBytes);
  throwIfAborted(signal);
  const manifestDigest = `sha256:${manifestBytesSha256}`;
  const configDigest = `sha256:${configBytesSha256}`;

  const manifestValue = decodeEvidenceJson(
    snapshots.manifestBytes,
    CPP_CUTE_AOT_OCI_MANIFEST_DECODE_LIMITS,
    "$.manifestBytes",
  );
  const manifest = verifyManifest(
    manifestValue,
    configDigest,
    snapshots.configBytes.byteLength,
  );
  throwIfAborted(signal);
  const configValue = decodeEvidenceJson(
    snapshots.configBytes,
    CPP_CUTE_AOT_OCI_CONFIG_DECODE_LIMITS,
    "$.configBytes",
  );
  const config = verifyConfig(configValue, manifest.layers.length);
  throwIfAborted(signal);

  const manifestDescriptor = freezeDescriptor({
    mediaType: OCI_IMAGE_MANIFEST_MEDIA_TYPE,
    digest: manifestDigest,
    size: snapshots.manifestBytes.byteLength,
  });
  const configDescriptor = freezeDescriptor({
    mediaType: OCI_IMAGE_CONFIG_MEDIA_TYPE,
    digest: configDigest,
    size: snapshots.configBytes.byteLength,
  });
  const verified = Object.freeze({
    manifest: manifestDescriptor,
    config: configDescriptor,
    layerCount: manifest.layers.length,
    totalLayerBytes: manifest.totalLayerBytes,
  }) as VerifiedCppCuteAotOciMetadata;
  VERIFIED_OCI_METADATA.set(verified, Object.freeze({
    layers: manifest.layers,
    diffIds: config.diffIds,
  }));
  return verified;
}

export function authorizeCppCuteAotOciMetadata(
  plan: PreparedCppCuteAotOfflineRun,
  metadata: VerifiedCppCuteAotOciMetadata,
): AuthorizedCppCuteAotOciMetadata {
  const planRecord = unwrapPreparedCppCuteAotOfflineRun(plan);
  const metadataRecord = storedMetadata(metadata);
  const expectedManifestDigest = digestFromImageReference(plan.imageReference);
  if (metadata.manifest.digest !== expectedManifestDigest) {
    digestMismatch("$.plan.imageReference", "prepared plan names a different raw OCI manifest");
  }
  if (metadata.config.digest !== plan.imageConfigDigest) {
    digestMismatch("$.plan.imageConfigDigest", "prepared plan names a different raw OCI config");
  }
  const environment = unwrapPreparedCppCuteAotExecutionEnvironment(
    planRecord.executionEnvironment,
  );
  verifyEnvironmentImageClosure(
    metadataRecord,
    environment.manifest.body.image.layers,
  );
  const authorized = Object.freeze({
    jobId: plan.jobId,
    profileHash: plan.profileHash,
    executionPlanSha256: plan.executionPlanSha256,
    imageReference: plan.imageReference,
    manifestDigest: metadata.manifest.digest,
    configDigest: metadata.config.digest,
    layerCount: metadata.layerCount,
    totalLayerBytes: metadata.totalLayerBytes,
  }) as AuthorizedCppCuteAotOciMetadata;
  AUTHORIZED_OCI_METADATA.set(authorized, Object.freeze({
    plan,
    metadata,
    layers: metadataRecord.layers,
    diffIds: metadataRecord.diffIds,
  }));
  return authorized;
}

function verifyEnvironmentImageClosure(
  metadata: StoredCppCuteAotOciMetadata,
  expectedLayers: readonly CppCuteAotExecutionEnvironmentLayer[],
): void {
  if (metadata.layers.length !== expectedLayers.length) {
    imageMismatch(
      "$.manifest.layers",
      "OCI layer count differs from the prepared execution environment",
    );
  }
  if (metadata.diffIds.length !== expectedLayers.length) {
    imageMismatch(
      "$.config.rootfs.diff_ids",
      "OCI diff-ID count differs from the prepared execution environment",
    );
  }
  for (let index = 0; index < expectedLayers.length; index += 1) {
    const expected = expectedLayers[index];
    const layer = metadata.layers[index];
    const diffId = metadata.diffIds[index];
    if (expected === undefined || layer === undefined || diffId === undefined) {
      imageMismatch("$.manifest.layers", "OCI layer closure is incomplete");
    }
    const path = `$.manifest.layers[${index}]`;
    if (layer.mediaType !== expected.mediaType) {
      imageMismatch(
        `${path}.mediaType`,
        "OCI layer media type differs from the prepared execution environment",
      );
    }
    if (layer.digest !== expected.digest) {
      imageMismatch(
        `${path}.digest`,
        "OCI layer digest differs from the prepared execution environment",
      );
    }
    if (BigInt(layer.size) !== BigInt(expected.size)) {
      imageMismatch(
        `${path}.size`,
        "OCI layer size differs from the prepared execution environment",
      );
    }
    if (diffId !== expected.diffId) {
      imageMismatch(
        `$.config.rootfs.diff_ids[${index}]`,
        "OCI diff ID differs from the prepared execution environment",
      );
    }
  }
}

export function unwrapAuthorizedCppCuteAotOciMetadata(
  image: AuthorizedCppCuteAotOciMetadata,
): AuthorizedCppCuteAotOciMetadataRecord {
  const record = storedAuthorizedImage(image);
  return Object.freeze({ plan: record.plan, metadata: record.metadata });
}

/** Internal runner view; nested arrays and descriptors are deeply frozen. */
export function inspectAuthorizedCppCuteAotOciMetadata(
  image: AuthorizedCppCuteAotOciMetadata,
): Readonly<{
  plan: PreparedCppCuteAotOfflineRun;
  metadata: VerifiedCppCuteAotOciMetadata;
  layers: readonly CppCuteAotOciDescriptor[];
  diffIds: readonly string[];
}> {
  const record = storedAuthorizedImage(image);
  return Object.freeze({
    plan: record.plan,
    metadata: record.metadata,
    layers: record.layers,
    diffIds: record.diffIds,
  });
}

function storedMetadata(image: VerifiedCppCuteAotOciMetadata): StoredCppCuteAotOciMetadata {
  if (typeof image !== "object" || image === null) unverified();
  const record = VERIFIED_OCI_METADATA.get(image as object);
  if (record === undefined) unverified();
  return record;
}

function storedAuthorizedImage(image: AuthorizedCppCuteAotOciMetadata): StoredAuthorizedCppCuteAotOciMetadata {
  if (typeof image !== "object" || image === null) unverified();
  const record = AUTHORIZED_OCI_METADATA.get(image as object);
  if (record === undefined) unverified();
  return record;
}

function snapshotEvidence(value: CppCuteAotOciMetadataBytes): CppCuteAotOciMetadataBytes {
  const descriptors = exactDataObject(
    value,
    ["configBytes", "manifestBytes"],
    "$",
    "raw OCI evidence",
  );
  return Object.freeze({
    manifestBytes: snapshotBytes(
      descriptors.manifestBytes?.value,
      "$.manifestBytes",
      CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.manifestBytes,
    ),
    configBytes: snapshotBytes(
      descriptors.configBytes?.value,
      "$.configBytes",
      CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.configBytes,
    ),
  });
}

function snapshotBytes(value: unknown, path: string, limit: number): Uint8Array {
  let inspection: InspectedUnsharedUint8Array;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "evidence bytes must be an unshared plain Uint8Array", { cause });
  }
  if (inspection.byteLength === 0) invalid(path, "evidence bytes must be nonempty");
  if (inspection.byteLength > limit) resource(path, `evidence bytes exceed ${limit}`);
  try {
    return copyInspectedUnsharedUint8Array(value, inspection);
  } catch (cause) {
    invalid(path, "evidence bytes became unreadable while snapshotting", { cause });
  }
}

function decodeEvidenceJson(
  bytes: Uint8Array,
  limits: typeof CPP_CUTE_AOT_OCI_MANIFEST_DECODE_LIMITS
    | typeof CPP_CUTE_AOT_OCI_CONFIG_DECODE_LIMITS,
  path: string,
): JsonValue {
  try {
    return decodeWireJson(bytes, { limits });
  } catch (cause) {
    invalid(path, "evidence is not bounded strict JSON", { cause });
  }
}

function verifyManifest(
  value: JsonValue,
  expectedConfigDigest: string,
  expectedConfigSize: number,
): {
  readonly layers: readonly CppCuteAotOciDescriptor[];
  readonly totalLayerBytes: number;
} {
  const manifest = closedObject(
    value,
    ["schemaVersion", "mediaType", "config", "layers", "annotations"],
    ["schemaVersion", "mediaType", "config", "layers"],
    "$.manifest",
  );
  if (manifest.schemaVersion !== 2) invalid("$.manifest.schemaVersion", "OCI manifest schemaVersion must be 2");
  if (manifest.mediaType !== OCI_IMAGE_MANIFEST_MEDIA_TYPE) {
    invalid("$.manifest.mediaType", "expected an OCI platform image manifest, not an index or Docker manifest");
  }
  const config = verifyDescriptor(
    field(manifest, "config", "$.manifest"),
    "$.manifest.config",
    new Set([OCI_IMAGE_CONFIG_MEDIA_TYPE]),
    Number.MAX_SAFE_INTEGER,
  );
  if (config.digest !== expectedConfigDigest) {
    digestMismatch("$.manifest.config.digest", "manifest config descriptor differs from exact raw config bytes");
  }
  if (config.size !== expectedConfigSize) {
    digestMismatch("$.manifest.config.size", "manifest config size differs from exact raw config bytes");
  }
  const layersValue = array(field(manifest, "layers", "$.manifest"), "$.manifest.layers");
  if (layersValue.length === 0 || layersValue.length > CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layers) {
    resource(
      "$.manifest.layers",
      `OCI image must contain 1..${CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layers} layers`,
    );
  }
  const layers = Object.freeze(layersValue.map((layer, index) => verifyDescriptor(
    layer,
    `$.manifest.layers[${index}]`,
    OCI_LAYER_MEDIA_TYPES,
    CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layerBytes,
  )));
  const totalLayerBytes = layers.reduce((total, layer) => total + BigInt(layer.size), 0n);
  if (totalLayerBytes > BigInt(CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.totalLayerBytes)) {
    resource("$.manifest.layers", "aggregate OCI layer bytes exceed policy");
  }
  verifyStringAnnotations(manifest.annotations, "$.manifest.annotations");
  return Object.freeze({ layers, totalLayerBytes: Number(totalLayerBytes) });
}

function verifyDescriptor(
  value: JsonValue,
  path: string,
  mediaTypes: ReadonlySet<string>,
  maxSize: number,
): CppCuteAotOciDescriptor {
  const descriptor = closedObject(
    value,
    ["mediaType", "digest", "size"],
    ["mediaType", "digest", "size"],
    path,
  );
  const mediaType = boundedString(field(descriptor, "mediaType", path), `${path}.mediaType`, 256);
  if (!mediaTypes.has(mediaType)) invalid(`${path}.mediaType`, "descriptor media type is not allowed");
  const digest = ociDigest(field(descriptor, "digest", path), `${path}.digest`);
  const size = safePositiveInteger(field(descriptor, "size", path), `${path}.size`);
  if (size > maxSize) resource(`${path}.size`, `descriptor size exceeds ${maxSize}`);
  return freezeDescriptor({ mediaType, digest, size });
}

function verifyConfig(
  value: JsonValue,
  expectedLayerCount: number,
): { readonly diffIds: readonly string[] } {
  const config = closedObject(
    value,
    ["created", "author", "architecture", "os", "config", "rootfs", "history"],
    ["architecture", "os", "rootfs"],
    "$.config",
  );
  if (config.architecture !== "amd64" || config.os !== "linux") {
    platformMismatch("$.config", "OCI image config must be exactly linux/amd64");
  }
  if (config.created !== undefined) rfc3339(config.created, "$.config.created");
  if (config.author !== undefined) boundedString(config.author, "$.config.author", 4_096, true);
  if (config.config !== undefined && config.config !== null) {
    const runtime = object(config.config, "$.config.config");
    if (Object.keys(runtime).length !== 0) {
      imageMismatch("$.config.config", "OCI image execution config must be absent, null, or empty");
    }
  }
  const rootfs = closedObject(
    field(config, "rootfs", "$.config"),
    ["type", "diff_ids"],
    ["type", "diff_ids"],
    "$.config.rootfs",
  );
  if (rootfs.type !== "layers") invalid("$.config.rootfs.type", "OCI rootfs type must be layers");
  const diffIdsValue = array(field(rootfs, "diff_ids", "$.config.rootfs"), "$.config.rootfs.diff_ids");
  if (diffIdsValue.length === 0 || diffIdsValue.length > CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layers) {
    resource("$.config.rootfs.diff_ids", "OCI rootfs diff-id count is outside policy");
  }
  const diffIds = Object.freeze(diffIdsValue.map(
    (digest, index) => ociDigest(digest, `$.config.rootfs.diff_ids[${index}]`),
  ));
  if (diffIds.length !== expectedLayerCount) {
    imageMismatch("$.config.rootfs.diff_ids", "config diff-id count differs from manifest layer count");
  }
  verifyHistory(config.history, diffIds.length);
  return Object.freeze({ diffIds });
}

function verifyHistory(value: JsonValue | undefined, expectedNonemptyLayers: number): void {
  if (value === undefined) return;
  const entries = array(value, "$.config.history");
  if (entries.length > CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.historyEntries) {
    resource("$.config.history", "OCI history exceeds policy entry count");
  }
  let nonemptyLayers = 0;
  for (const [index, entry] of entries.entries()) {
    const path = `$.config.history[${index}]`;
    const history = closedObject(
      entry,
      ["created", "author", "created_by", "comment", "empty_layer"],
      [],
      path,
    );
    if (history.created !== undefined) rfc3339(history.created, `${path}.created`);
    for (const key of ["author", "created_by", "comment"] as const) {
      if (history[key] !== undefined) boundedString(history[key], `${path}.${key}`, 16_384, true);
    }
    if (history.empty_layer !== undefined && typeof history.empty_layer !== "boolean") {
      invalid(`${path}.empty_layer`, "history empty_layer must be boolean");
    }
    if (history.empty_layer !== true) nonemptyLayers += 1;
  }
  if (nonemptyLayers !== expectedNonemptyLayers) {
    imageMismatch("$.config.history", "nonempty history count differs from rootfs diff-id count");
  }
}

function verifyStringAnnotations(value: JsonValue | undefined, path: string): void {
  if (value === undefined) return;
  const annotations = object(value, path);
  const entries = Object.entries(annotations);
  if (entries.length > CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.annotations) {
    resource(path, "OCI annotations exceed policy entry count");
  }
  for (const [key, entry] of entries) {
    if (
      key.length === 0
      || utf8Length(key) > CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.annotationKeyBytes
      || typeof entry !== "string"
      || utf8Length(entry) > CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.annotationValueBytes
    ) {
      invalid(path, "annotations must contain bounded nonempty UTF-8 keys and string values");
    }
  }
}

function freezeDescriptor(value: CppCuteAotOciDescriptor): CppCuteAotOciDescriptor {
  return Object.freeze({
    mediaType: value.mediaType,
    digest: value.digest,
    size: value.size,
  });
}

function exactDataObject(
  value: unknown,
  expectedKeys: readonly string[],
  path: string,
  name: string,
): PropertyDescriptorMap {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch (cause) {
    invalid(path, `${name} must be an inspectable plain object`, { cause });
  }
  if (typeof value !== "object" || value === null || prototype !== Object.prototype) {
    invalid(path, `${name} must be a plain object`);
  }
  const keys = Reflect.ownKeys(descriptors);
  if (
    keys.length !== expectedKeys.length
    || keys.some((key) => typeof key !== "string" || !expectedKeys.includes(key))
  ) {
    invalid(path, `${name} fields must be exactly ${expectedKeys.join(", ")}`);
  }
  for (const key of expectedKeys) {
    const descriptor = descriptors[key];
    if (
      descriptor === undefined
      || descriptor.enumerable !== true
      || !("value" in descriptor)
    ) {
      invalid(`${path}.${key}`, `${name} fields must be ordinary enumerable data properties`);
    }
  }
  return descriptors;
}

function normalizeOptions(options: VerifyCppCuteAotOciMetadataOptions): AbortSignal | undefined {
  const descriptors = exactOptionsObject(options);
  const signal = descriptors.signal?.value as unknown;
  if (signal !== undefined && !isAbortSignal(signal)) {
    invalid("$options.signal", "signal must be an AbortSignal");
  }
  return signal;
}

function exactOptionsObject(options: VerifyCppCuteAotOciMetadataOptions): PropertyDescriptorMap {
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
  if (keys.length > 1 || keys.some((key) => key !== "signal")) {
    invalid("$options", "options contain unknown fields");
  }
  if (descriptors.signal !== undefined) {
    const descriptor = descriptors.signal;
    if (descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid("$options.signal", "signal must be an enumerable data property");
    }
  }
  return descriptors;
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
  if (aborted === true) cancelled();
}

function closedObject(
  value: JsonValue,
  allowed: readonly string[],
  required: readonly string[],
  path: string,
): JsonObject {
  const candidate = object(value, path);
  for (const key of Object.keys(candidate)) {
    if (!allowed.includes(key)) invalid(path, `unknown field ${key}`);
  }
  for (const key of required) {
    if (!Object.prototype.hasOwnProperty.call(candidate, key)) invalid(`${path}.${key}`, "required field is missing");
  }
  return candidate;
}

function object(value: JsonValue, path: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected object");
  return value as JsonObject;
}

function array(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function field(value: JsonObject, key: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(value, key)) invalid(`${path}.${key}`, "required field is missing");
  return value[key] as JsonValue;
}

function boundedString(value: JsonValue, path: string, maxBytes: number, allowEmpty = false): string {
  if (
    typeof value !== "string"
    || (!allowEmpty && value.length === 0)
    || utf8Length(value) > maxBytes
  ) {
    invalid(path, "expected bounded string");
  }
  return value;
}

function safePositiveInteger(value: JsonValue, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    invalid(path, "expected positive safe integer");
  }
  return value;
}

function rfc3339(value: JsonValue, path: string): void {
  const timestamp = boundedString(value, path, 256);
  const match = RFC_3339.exec(timestamp);
  if (match === null) invalid(path, "expected RFC 3339 timestamp");
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);
  const second = Number(match[6]);
  const offsetHour = match[7] === undefined ? 0 : Number(match[7]);
  const offsetMinute = match[8] === undefined ? 0 : Number(match[8]);
  const daysInMonth = month === 2
    ? (isLeapYear(year) ? 29 : 28)
    : [4, 6, 9, 11].includes(month) ? 30 : 31;
  if (
    month < 1
    || month > 12
    || day < 1
    || day > daysInMonth
    || hour > 23
    || minute > 59
    || second > 59
    || offsetHour > 23
    || offsetMinute > 59
  ) {
    invalid(path, "expected RFC 3339 timestamp");
  }
}

function isLeapYear(year: number): boolean {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
}

function utf8Length(value: string): number {
  return TEXT_ENCODER.encode(value).byteLength;
}

function ociDigest(value: JsonValue, path: string): string {
  const digest = boundedString(value, path, 128);
  if (!OCI_DIGEST.test(digest)) invalid(path, "expected lowercase sha256 OCI digest");
  return digest;
}

function digestFromImageReference(reference: string): string {
  const separator = reference.lastIndexOf("@");
  const digest = separator < 0 ? "" : reference.slice(separator + 1);
  if (!OCI_DIGEST.test(digest)) invalid("$.plan.imageReference", "prepared image reference lost its OCI digest");
  return digest;
}

function cancelled(): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-CANCELLED", "$options.signal", "OCI verification was aborted");
}

function unverified(): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-UNVERIFIED", "$", "expected opaque verified OCI metadata authority");
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT", path, message);
}

function digestMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH", path, message);
}

function platformMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-PLATFORM-MISMATCH", path, message);
}

function imageMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH", path, message);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID", path, message, options);
}

function fail(
  code: CppCuteAotOciErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteAotOciError(code, path, message, options);
}
