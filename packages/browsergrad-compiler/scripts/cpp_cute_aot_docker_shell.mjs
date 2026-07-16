import { chmod, lstat, mkdir, mkdtemp, readdir, rm } from "node:fs/promises";
import { join } from "node:path";

import { decodeWireJson } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  inspectAuthorizedCppCuteAotOciMetadata,
} from "../dist/cpp_cute_aot_oci.js";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "../dist/cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_AOT_DOCKER_API_VERSION,
  CPP_CUTE_AOT_DOCKER_CLIENT_DEFAULT_API_VERSION,
  CPP_CUTE_AOT_DOCKER_CLIENT_VERSION,
  CPP_CUTE_AOT_DOCKER_ENGINE_API_VERSION,
  CPP_CUTE_AOT_DOCKER_ENGINE_MIN_API_VERSION,
  CPP_CUTE_AOT_DOCKER_ENGINE_VERSION,
  CPP_CUTE_AOT_DOCKER_INFO_DECODE_LIMITS,
  CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
  CPP_CUTE_AOT_DOCKER_INFO_SCHEMA,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_DECODE_LIMITS,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA,
  CPP_CUTE_AOT_DOCKER_VERSION_DECODE_LIMITS,
  CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
  CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA,
} from "../dist/cpp_cute_aot_policy.js";
import {
  buildCppCuteAotDockerInfoRequest,
  buildCppCuteAotDockerImageInspectRequest,
  buildCppCuteAotDockerVersionRequest,
  CppCuteAotDockerProcessError,
  runBoundedChildProcess,
} from "./cpp_cute_aot_docker_process.mjs";

const OCI_DIGEST = /^sha256:[0-9a-f]{64}$/u;
const DOCKER_REPO_DIGEST = /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[1-9][0-9]*)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)*@sha256:[0-9a-f]{64}$/u;
const RUN_ROOT_TEMPLATE = "/tmp/browsergrad-cpp-cute-docker-";
const ABORTED_GETTER = Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const LIVE_OBSERVATIONS = new WeakMap();
const TEST_OBSERVATIONS = new WeakMap();

/** @typedef {"BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CANCELLED" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-INVALID" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-PROCESS" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-EXIT" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-MISMATCH" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CLEANUP" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-UNVERIFIED"} CppCuteAotDockerImageErrorCode */

export class CppCuteAotDockerImageError extends Error {
  /**
   * @param {CppCuteAotDockerImageErrorCode} code
   * @param {string} path
   * @param {string} message
   */
  constructor(code, path, message) {
    super(`${code}: ${message}`);
    this.name = "CppCuteAotDockerImageError";
    this.code = code;
    this.path = path;
  }
}

/**
 * @typedef {Readonly<{
 *   jobId: string;
 *   profileHash: string;
 *   executionPlanSha256: string;
 *   imageReference: string;
 *   manifestDigest: string;
 *   imageId: string;
 *   configDigest: string;
 *   platform: "linux/amd64";
 *   dockerClientVersion: "29.6.1";
 *   dockerEngineVersion: "29.6.1";
 *   dockerRequestApiVersion: "1.49";
 *   dockerEngineApiVersion: "1.55";
 *   dockerEngineMinApiVersion: "1.40";
 *   dockerImageStore: "containerd";
 *   layerCount: number;
 *   totalLayerBytes: number;
 * }>} ObservedCppCuteAotLocalDockerImage
 */

/**
 * @typedef {Readonly<{
 *   authorizedMetadata: import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata;
 *   shellSession: object;
 *   repoDigests: readonly string[];
 * }>} StoredDockerImageObservation
 */

/** @typedef {(request: import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessRequest) => Promise<import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessResult>} DockerObservationProcess */

/**
 * Production point-in-time observation. No caller can supply Docker process,
 * executable, argv, environment, endpoint, or result bytes.
 *
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {{ readonly signal?: AbortSignal }} [options]
 * @returns {Promise<ObservedCppCuteAotLocalDockerImage>}
 */
export async function observeCppCuteAotLocalDockerImage(authorizedMetadata, options = {}) {
  return observeWithProcess(
    authorizedMetadata,
    runBoundedChildProcess,
    options,
    LIVE_OBSERVATIONS,
  );
}

/**
 * Test-only adapter seam. Results live in a disjoint WeakMap and can never be
 * consumed as production live-image authority.
 *
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {DockerObservationProcess} processAdapter
 * @param {{ readonly signal?: AbortSignal }} [options]
 * @returns {Promise<ObservedCppCuteAotLocalDockerImage>}
 */
export async function __observeCppCuteAotLocalDockerImageWithProcessForTest(
  authorizedMetadata,
  processAdapter,
  options = {},
) {
  if (typeof processAdapter !== "function") invalid("$processAdapter", "test process adapter must be a function");
  return observeWithProcess(authorizedMetadata, processAdapter, options, TEST_OBSERVATIONS);
}

/**
 * Production lifecycle view. Structural copies and test-adapter observations
 * are rejected.
 *
 * @param {ObservedCppCuteAotLocalDockerImage} observed
 */
export function unwrapObservedCppCuteAotLocalDockerImage(observed) {
  return unwrapStored(observed, LIVE_OBSERVATIONS);
}

/** @param {ObservedCppCuteAotLocalDockerImage} observed */
export function __unwrapObservedCppCuteAotLocalDockerImageForTest(observed) {
  return unwrapStored(observed, TEST_OBSERVATIONS);
}

/**
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {DockerObservationProcess} processAdapter
 * @param {{ readonly signal?: AbortSignal }} options
 * @param {WeakMap<object, StoredDockerImageObservation>} authorityStore
 * @returns {Promise<ObservedCppCuteAotLocalDockerImage>}
 */
async function observeWithProcess(authorizedMetadata, processAdapter, options, authorityStore) {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const authorized = inspectAuthorizedCppCuteAotOciMetadata(authorizedMetadata);
  throwIfAborted(signal);

  /** @type {string | undefined} */
  let runRoot;
  /** @type {Readonly<{ imageId: string; repoDigests: readonly string[] }> | undefined} */
  let verifiedImage;
  /** @type {Readonly<{ clientVersion: "29.6.1"; engineVersion: "29.6.1"; requestApiVersion: "1.49"; engineApiVersion: "1.55"; engineMinApiVersion: "1.40"; imageStore: "containerd" }> | undefined} */
  let verifiedRuntime;
  /** @type {unknown} */
  let failure;
  try {
    runRoot = await mkdtemp(RUN_ROOT_TEMPLATE);
    throwIfAborted(signal);
    await chmod(runRoot, 0o700);
    throwIfAborted(signal);
    const configDirectory = join(runRoot, "docker-config");
    const homeDirectory = join(runRoot, "home");
    await mkdir(configDirectory, { mode: 0o700 });
    throwIfAborted(signal);
    await mkdir(homeDirectory, { mode: 0o700 });
    throwIfAborted(signal);
    await Promise.all([chmod(configDirectory, 0o700), chmod(homeDirectory, 0o700)]);
    throwIfAborted(signal);
    await assertPrivateDirectory(runRoot, "$runRoot");
    throwIfAborted(signal);
    await assertPrivateDirectory(configDirectory, "$runRoot/docker-config");
    throwIfAborted(signal);
    await assertPrivateDirectory(homeDirectory, "$runRoot/home");
    throwIfAborted(signal);
    if ((await readdir(configDirectory)).length !== 0) {
      invalid("$runRoot/docker-config", "private Docker configuration directory must start empty");
    }
    throwIfAborted(signal);

    const commonRequest = {
      runRoot,
      configDirectory,
      homeDirectory,
      ...(signal === undefined ? {} : { signal }),
    };

    const versionProjection = await executeProbe(
      processAdapter,
      buildCppCuteAotDockerVersionRequest(commonRequest),
      configDirectory,
      signal,
      CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
      CPP_CUTE_AOT_DOCKER_VERSION_DECODE_LIMITS,
      "version",
    );
    const version = verifyVersionProjection(versionProjection);

    const infoProjection = await executeProbe(
      processAdapter,
      buildCppCuteAotDockerInfoRequest(commonRequest),
      configDirectory,
      signal,
      CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
      CPP_CUTE_AOT_DOCKER_INFO_DECODE_LIMITS,
      "info",
    );
    verifyInfoProjection(infoProjection, version);

    const imageProjection = await executeProbe(
      processAdapter,
      buildCppCuteAotDockerImageInspectRequest({
        ...commonRequest,
        imageReference: authorizedMetadata.imageReference,
      }),
      configDirectory,
      signal,
      CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS,
      CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_DECODE_LIMITS,
      "image-inspect",
    );
    verifiedImage = verifyImageProjection(imageProjection, authorizedMetadata, authorized);
    verifiedRuntime = Object.freeze({
      clientVersion: CPP_CUTE_AOT_DOCKER_CLIENT_VERSION,
      engineVersion: CPP_CUTE_AOT_DOCKER_ENGINE_VERSION,
      requestApiVersion: CPP_CUTE_AOT_DOCKER_API_VERSION,
      engineApiVersion: CPP_CUTE_AOT_DOCKER_ENGINE_API_VERSION,
      engineMinApiVersion: CPP_CUTE_AOT_DOCKER_ENGINE_MIN_API_VERSION,
      imageStore: "containerd",
    });
  } catch (error) {
    failure = normalizeObservationFailure(error);
  }

  if (runRoot !== undefined) {
    try {
      await rm(runRoot, { recursive: true, force: true, maxRetries: 0 });
      await assertRemoved(runRoot);
    } catch {
      cleanupFailure(failure);
    }
  }
  if (failure !== undefined) throw failure;
  throwIfAborted(signal);
  if (verifiedImage === undefined || verifiedRuntime === undefined) processFailure();

  const shellSession = Object.freeze({
    runtime: verifiedRuntime,
    childCompletion: "closed",
    runRootCleanup: "removed",
  });
  const observed = Object.freeze({
    jobId: authorizedMetadata.jobId,
    profileHash: authorizedMetadata.profileHash,
    executionPlanSha256: authorizedMetadata.executionPlanSha256,
    imageReference: authorizedMetadata.imageReference,
    manifestDigest: authorizedMetadata.manifestDigest,
    imageId: verifiedImage.imageId,
    configDigest: authorizedMetadata.configDigest,
    platform: "linux/amd64",
    dockerClientVersion: verifiedRuntime.clientVersion,
    dockerEngineVersion: verifiedRuntime.engineVersion,
    dockerRequestApiVersion: verifiedRuntime.requestApiVersion,
    dockerEngineApiVersion: verifiedRuntime.engineApiVersion,
    dockerEngineMinApiVersion: verifiedRuntime.engineMinApiVersion,
    dockerImageStore: verifiedRuntime.imageStore,
    layerCount: authorizedMetadata.layerCount,
    totalLayerBytes: authorizedMetadata.totalLayerBytes,
  });
  authorityStore.set(observed, Object.freeze({
    authorizedMetadata,
    shellSession,
    repoDigests: verifiedImage.repoDigests,
  }));
  return observed;
}

/** @param {ObservedCppCuteAotLocalDockerImage} observed @param {WeakMap<object, StoredDockerImageObservation>} store @returns {StoredDockerImageObservation} */
function unwrapStored(observed, store) {
  if (typeof observed !== "object" || observed === null) unverified();
  const record = store.get(observed);
  if (record === undefined) unverified();
  return Object.freeze({
    authorizedMetadata: record.authorizedMetadata,
    shellSession: record.shellSession,
    repoDigests: record.repoDigests,
  });
}

/**
 * @param {DockerObservationProcess} processAdapter
 * @param {import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessRequest} request
 * @param {string} configDirectory
 * @param {AbortSignal | undefined} signal
 * @param {{ readonly stdoutBytes: number; readonly stderrBytes: number }} limits
 * @param {import("@unlocalhosted/browsergrad-semantic-core/schema").DecodeLimits} decodeLimits
 * @param {string} probe
 * @returns {Promise<import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue>}
 */
async function executeProbe(
  processAdapter,
  request,
  configDirectory,
  signal,
  limits,
  decodeLimits,
  probe,
) {
  throwIfAborted(signal);
  let result;
  try {
    result = await processAdapter(request);
  } catch (error) {
    if (error instanceof CppCuteAotDockerProcessError) throw error;
    processFailure();
  }
  throwIfAborted(signal);
  const processResult = snapshotProcessResult(result, limits);
  if (processResult.exitCode !== 0 || processResult.signal !== null) exitFailure(probe);
  if (processResult.stderr.byteLength !== 0) {
    outputInvalid("$process.stderr", `successful Docker ${probe} probe must emit no stderr`);
  }
  if ((await readdir(configDirectory)).length !== 0) {
    outputInvalid("$runRoot/docker-config", `Docker ${probe} probe modified its private empty configuration`);
  }
  throwIfAborted(signal);
  return decodeProjection(processResult.stdout, decodeLimits, probe);
}

/**
 * @param {unknown} result
 * @param {{ readonly stdoutBytes: number; readonly stderrBytes: number }} limits
 * @returns {import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessResult}
 */
function snapshotProcessResult(result, limits) {
  const object = exactDataObject(
    result,
    ["exitCode", "signal", "stderr", "stdout"],
    "$process",
  );
  const exitCode = dataValue(object, "exitCode", "$process");
  const exitSignal = dataValue(object, "signal", "$process");
  if (exitCode !== null && (typeof exitCode !== "number" || !Number.isSafeInteger(exitCode) || exitCode < 0 || exitCode > 255)) {
    outputInvalid("$process.exitCode", "process exit code is invalid");
  }
  if (exitSignal !== null && (typeof exitSignal !== "string" || !/^SIG[A-Z0-9]+$/u.test(exitSignal))) {
    outputInvalid("$process.signal", "process signal is invalid");
  }
  return Object.freeze({
    exitCode,
    signal: exitSignal,
    stdout: snapshotBytes(dataValue(object, "stdout", "$process"), "$process.stdout", limits.stdoutBytes),
    stderr: snapshotBytes(dataValue(object, "stderr", "$process"), "$process.stderr", limits.stderrBytes),
  });
}

/** @param {unknown} value @param {string} path @param {number} limit @returns {Uint8Array} */
function snapshotBytes(value, path, limit) {
  let inspection;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch {
    outputInvalid(path, "process output must be an unshared plain Uint8Array");
  }
  if (inspection.byteLength > limit) outputInvalid(path, "process output exceeds policy");
  try {
    return copyInspectedUnsharedUint8Array(value, inspection);
  } catch {
    outputInvalid(path, "process output became unreadable while snapshotting");
  }
}

/**
 * @param {Uint8Array} stdout
 * @param {import("@unlocalhosted/browsergrad-semantic-core/schema").DecodeLimits} decodeLimits
 * @param {string} probe
 * @returns {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue}
 */
function decodeProjection(stdout, decodeLimits, probe) {
  if (
    stdout.byteLength < 3
    || stdout[0] !== 0x7b
    || stdout[stdout.byteLength - 1] !== 0x0a
    || stdout[stdout.byteLength - 2] === 0x0d
  ) {
    outputInvalid("$process.stdout", `Docker ${probe} probe must emit one compact JSON object followed by LF`);
  }
  const document = stdout.subarray(0, stdout.byteLength - 1);
  if (document.includes(0x0a) || document.includes(0x0d) || document.includes(0x00)) {
    outputInvalid("$process.stdout", `Docker ${probe} output contains an unexpected control separator`);
  }
  try {
    return decodeWireJson(document, { limits: decodeLimits });
  } catch {
    outputInvalid("$process.stdout", `Docker ${probe} output is not bounded strict JSON`);
  }
}

/**
 * @param {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue} value
 * @returns {Readonly<{ clientVersion: "29.6.1"; engineVersion: "29.6.1"; requestApiVersion: "1.49"; engineApiVersion: "1.55"; engineMinApiVersion: "1.40" }>}
 */
function verifyVersionProjection(value) {
  const projection = closedObject(value, ["client", "schema", "server", "version"], "$version");
  if (projection.schema !== CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA || projection.version !== 1) {
    outputInvalid("$version", "Docker version projection schema/version mismatch");
  }
  const client = closedObject(
    projection.client,
    ["apiVersion", "defaultApiVersion", "version"],
    "$version.client",
  );
  if (
    client.version !== CPP_CUTE_AOT_DOCKER_CLIENT_VERSION
    || client.apiVersion !== CPP_CUTE_AOT_DOCKER_API_VERSION
    || client.defaultApiVersion !== CPP_CUTE_AOT_DOCKER_CLIENT_DEFAULT_API_VERSION
  ) {
    mismatch("$version.client", "Docker client identity or selected API differs from policy");
  }
  const server = closedObject(
    projection.server,
    ["apiVersion", "arch", "minApiVersion", "os", "version"],
    "$version.server",
  );
  if (
    server.version !== CPP_CUTE_AOT_DOCKER_ENGINE_VERSION
    || server.apiVersion !== CPP_CUTE_AOT_DOCKER_ENGINE_API_VERSION
    || server.minApiVersion !== CPP_CUTE_AOT_DOCKER_ENGINE_MIN_API_VERSION
    || server.os !== "linux"
    || server.arch !== "amd64"
  ) {
    mismatch("$version.server", "Docker engine identity, API range, or platform differs from policy");
  }
  return Object.freeze({
    clientVersion: CPP_CUTE_AOT_DOCKER_CLIENT_VERSION,
    engineVersion: CPP_CUTE_AOT_DOCKER_ENGINE_VERSION,
    requestApiVersion: CPP_CUTE_AOT_DOCKER_API_VERSION,
    engineApiVersion: CPP_CUTE_AOT_DOCKER_ENGINE_API_VERSION,
    engineMinApiVersion: CPP_CUTE_AOT_DOCKER_ENGINE_MIN_API_VERSION,
  });
}

/**
 * @param {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue} value
 * @param {ReturnType<typeof verifyVersionProjection>} version
 */
function verifyInfoProjection(value, version) {
  const projection = closedObject(
    value,
    ["architecture", "driverStatus", "osType", "schema", "serverVersion", "version"],
    "$info",
  );
  if (projection.schema !== CPP_CUTE_AOT_DOCKER_INFO_SCHEMA || projection.version !== 1) {
    outputInvalid("$info", "Docker info projection schema/version mismatch");
  }
  if (
    projection.serverVersion !== version.engineVersion
    || projection.osType !== "linux"
    || projection.architecture !== "x86_64"
  ) {
    mismatch("$info", "Docker info engine identity or host platform differs from version evidence");
  }
  const status = array(projection.driverStatus, "$info.driverStatus");
  if (status.length !== CPP_CUTE_AOT_DOCKER_INFO_LIMITS.driverStatusEntries) {
    mismatch("$info.driverStatus", "Docker image store is not the exact pinned containerd store");
  }
  const entry = status[0];
  if (
    !Array.isArray(entry)
    || entry.length !== 2
    || entry[0] !== "driver-type"
    || entry[1] !== "io.containerd.snapshotter.v1"
  ) {
    mismatch("$info.driverStatus[0]", "Docker image store is not the exact pinned containerd store");
  }
}

/**
 * @param {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue} value
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {ReturnType<typeof inspectAuthorizedCppCuteAotOciMetadata>} authorized
 * @returns {Readonly<{ imageId: string; repoDigests: readonly string[] }>}
 */
function verifyImageProjection(value, authorizedMetadata, authorized) {
  const projection = closedObject(value, [
    "config", "descriptor", "id", "platform", "repoDigests", "rootfs", "schema", "version",
  ], "$");
  if (projection.schema !== CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA || projection.version !== 1) {
    outputInvalid("$", "Docker image inspect projection schema/version mismatch");
  }
  const descriptor = closedObject(projection.descriptor, ["digest", "mediaType", "size"], "$.descriptor");
  if (
    descriptor.digest !== authorizedMetadata.manifestDigest
    || descriptor.mediaType !== authorized.metadata.manifest.mediaType
    || descriptor.size !== authorized.metadata.manifest.size
  ) {
    mismatch("$.descriptor", "local image descriptor differs from exact authorized OCI manifest");
  }
  const imageId = ociDigest(projection.id, "$.id");
  if (imageId !== authorizedMetadata.manifestDigest) {
    mismatch("$.id", "pinned containerd image-store ID differs from authorized manifest identity");
  }
  const platform = closedObject(
    projection.platform,
    ["architecture", "os", "osVersion", "variant"],
    "$.platform",
  );
  if (
    platform.architecture !== "amd64"
    || platform.os !== "linux"
    || platform.osVersion !== ""
    || platform.variant !== ""
  ) {
    mismatch("$.platform", "local image platform must be exactly linux/amd64 without variant or OS version");
  }
  const repoDigestsValue = array(projection.repoDigests, "$.repoDigests");
  if (repoDigestsValue.length === 0 || repoDigestsValue.length > CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS.repoDigests) {
    outputInvalid("$.repoDigests", "repository digest count is outside policy");
  }
  const repoDigests = repoDigestsValue.map((entry, index) => dockerRepoDigest(entry, `$.repoDigests[${index}]`));
  if (new Set(repoDigests).size !== repoDigests.length) {
    outputInvalid("$.repoDigests", "repository digests must be unique");
  }
  const normalizedAuthorizedReference = normalizeDockerRepositoryReference(authorizedMetadata.imageReference);
  if (!repoDigests.some((entry) => normalizeDockerRepositoryReference(entry) === normalizedAuthorizedReference)) {
    mismatch("$.repoDigests", "local image lacks exact authorized repository manifest reference");
  }
  const rootfs = closedObject(projection.rootfs, ["diffIds", "type"], "$.rootfs");
  if (rootfs.type !== "layers") mismatch("$.rootfs.type", "local image rootfs type must be layers");
  const diffIdsValue = array(rootfs.diffIds, "$.rootfs.diffIds");
  if (diffIdsValue.length !== authorized.diffIds.length) {
    mismatch("$.rootfs.diffIds", "local image diff-ID count differs from authorized OCI config");
  }
  for (const [index, expected] of authorized.diffIds.entries()) {
    const actual = diffIdsValue[index];
    if (actual === undefined || ociDigest(actual, `$.rootfs.diffIds[${index}]`) !== expected) {
      mismatch(`$.rootfs.diffIds[${index}]`, "local image diff-ID order differs from authorized OCI config");
    }
  }
  verifySemanticallyEmptyConfig(field(projection, "config", "$"));
  return Object.freeze({
    imageId,
    repoDigests: Object.freeze([...repoDigests].sort()),
  });
}

/** @param {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue} value */
function verifySemanticallyEmptyConfig(value) {
  const config = object(value, "$.config");
  for (const [key, entry] of Object.entries(config)) {
    if (!isEmptyConfigLeaf(entry)) {
      mismatch(`$.config.${key}`, "local image execution config is not semantically empty");
    }
  }
}

/** @param {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue} value */
function isEmptyConfigLeaf(value) {
  if (value === null || value === false || value === 0 || value === "") return true;
  if (Array.isArray(value)) return value.length === 0;
  return typeof value === "object" && value !== null && Object.keys(value).length === 0;
}

/** @param {unknown} options @returns {AbortSignal | undefined} */
function normalizeOptions(options) {
  let descriptors;
  try {
    if (!isPlainObject(options)) invalid("$options", "options must be a plain object");
    descriptors = Object.getOwnPropertyDescriptors(options);
  } catch (error) {
    if (error instanceof CppCuteAotDockerImageError) throw error;
    invalid("$options", "options must be safely inspectable");
  }
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length > 1 || keys.some((key) => key !== "signal")) invalid("$options", "options contain unknown fields");
  const descriptor = descriptors.signal;
  if (descriptor === undefined) return undefined;
  if (descriptor.enumerable !== true || !("value" in descriptor)) invalid("$options.signal", "signal must be an enumerable data property");
  if (descriptor.value === undefined) return undefined;
  const getter = ABORTED_GETTER;
  if (getter === undefined) invalid("$options.signal", "AbortSignal is unavailable");
  try {
    getter.call(descriptor.value);
  } catch {
    invalid("$options.signal", "expected AbortSignal");
  }
  return /** @type {AbortSignal} */ (descriptor.value);
}

/** @param {AbortSignal | undefined} signal */
function throwIfAborted(signal) {
  if (signal === undefined) return;
  let aborted;
  try {
    aborted = ABORTED_GETTER?.call(signal);
  } catch {
    invalid("$options.signal", "AbortSignal became unreadable");
  }
  if (aborted === true) cancelled();
}

/** @param {string} path @param {string} diagnosticPath */
async function assertPrivateDirectory(path, diagnosticPath) {
  const status = await lstat(path);
  if (!status.isDirectory() || status.isSymbolicLink() || (status.mode & 0o777) !== 0o700) {
    invalid(diagnosticPath, "runner-owned Docker directory must be a non-symlink mode-0700 directory");
  }
}

/** @param {string} path */
async function assertRemoved(path) {
  try {
    await lstat(path);
  } catch (error) {
    if (
      typeof error === "object"
      && error !== null
      && "code" in error
      && error.code === "ENOENT"
    ) return;
    throw error;
  }
  throw new Error("private Docker observation directory still exists after cleanup");
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} path @returns {Record<string, import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue>} */
function closedObject(value, keys, path) {
  const result = object(value, path);
  const actual = Object.keys(result);
  if (actual.length !== keys.length || actual.some((key) => !keys.includes(key))) {
    outputInvalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  return result;
}

/** @param {unknown} value @param {string} path @returns {Record<string, import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue>} */
function object(value, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) outputInvalid(path, "expected object");
  return /** @type {Record<string, import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue>} */ (value);
}

/** @param {unknown} value @param {string} path @returns {readonly import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue[]} */
function array(value, path) {
  if (!Array.isArray(value)) outputInvalid(path, "expected array");
  return value;
}

/** @param {unknown} value @param {string} path @returns {string} */
function ociDigest(value, path) {
  if (typeof value !== "string" || !OCI_DIGEST.test(value)) outputInvalid(path, "expected lowercase sha256 OCI digest");
  return value;
}

/** @param {unknown} value @param {string} path @returns {string} */
function dockerRepoDigest(value, path) {
  if (typeof value !== "string" || !DOCKER_REPO_DIGEST.test(value)) outputInvalid(path, "expected bounded lowercase Docker repository digest");
  return value;
}

/** @param {string} reference @returns {string} */
function normalizeDockerRepositoryReference(reference) {
  const separator = reference.lastIndexOf("@");
  const repository = reference.slice(0, separator);
  const digest = reference.slice(separator + 1);
  const components = repository.split("/");
  const first = components[0] ?? "";
  const explicitDomain = first.includes(".") || first.includes(":") || first === "localhost";
  let domain;
  let path;
  if (explicitDomain) {
    domain = first === "index.docker.io" ? "docker.io" : first;
    path = components.slice(1);
  } else {
    domain = "docker.io";
    path = components;
  }
  if (domain === "docker.io" && path.length === 1) path = ["library", ...path];
  return `${domain}/${path.join("/")}@${digest}`;
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} path @returns {Record<string, PropertyDescriptor>} */
function exactDataObject(value, keys, path) {
  if (!isPlainObject(value)) outputInvalid(path, "process result must be a plain object");
  let descriptors;
  try {
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch {
    outputInvalid(path, "process result must be safely inspectable");
  }
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length || actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    outputInvalid(path, `process result must contain exactly ${keys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      outputInvalid(`${path}.${key}`, "process result fields must be enumerable data properties");
    }
  }
  return Object.fromEntries(keys.map((key) => [key, /** @type {PropertyDescriptor} */ (descriptors[key])]));
}

/** @param {Record<string, PropertyDescriptor>} descriptors @param {string} key @param {string} path @returns {unknown} */
function dataValue(descriptors, key, path) {
  const descriptor = descriptors[key];
  if (descriptor === undefined || !("value" in descriptor)) outputInvalid(`${path}.${key}`, "required data property is missing");
  return descriptor.value;
}

/** @param {Record<string, import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue>} value @param {string} key @param {string} path @returns {import("@unlocalhosted/browsergrad-semantic-core/schema").JsonValue} */
function field(value, key, path) {
  const result = value[key];
  if (result === undefined) outputInvalid(`${path}.${key}`, "required field is missing");
  return result;
}

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function isPlainObject(value) {
  if (typeof value !== "object" || value === null) return false;
  try {
    return Object.getPrototypeOf(value) === Object.prototype;
  } catch {
    return false;
  }
}

/** @param {unknown} error @returns {unknown} */
function normalizeObservationFailure(error) {
  if (error instanceof CppCuteAotDockerImageError || error instanceof CppCuteAotDockerProcessError) {
    return error;
  }
  return new CppCuteAotDockerImageError(
    "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-PROCESS",
    "$runRoot",
    "Docker image observation staging or process failed",
  );
}

/** @returns {never} */
function cancelled() {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CANCELLED", "$options.signal", "Docker image observation was aborted");
}

/** @returns {never} */
function processFailure() {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-PROCESS", "$process", "Docker observation process failed");
}

/** @param {string} probe @returns {never} */
function exitFailure(probe) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-EXIT", "$process", `Docker ${probe} probe did not exit cleanly`);
}

/** @param {unknown} primary @returns {never} */
function cleanupFailure(primary) {
  const error = new CppCuteAotDockerImageError(
    "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CLEANUP",
    "$runRoot",
    "failed to remove private Docker observation directory",
  );
  if (primary !== undefined) {
    Object.defineProperty(error, "cause", {
      configurable: false,
      enumerable: false,
      value: primary,
      writable: false,
    });
  }
  throw error;
}

/** @returns {never} */
function unverified() {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-UNVERIFIED", "$", "expected opaque production local-Docker image observation");
}

/** @param {string} path @param {string} message @returns {never} */
function invalid(path, message) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-INVALID", path, message);
}

/** @param {string} path @param {string} message @returns {never} */
function outputInvalid(path, message) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT", path, message);
}

/** @param {string} path @param {string} message @returns {never} */
function mismatch(path, message) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-MISMATCH", path, message);
}

/** @param {CppCuteAotDockerImageErrorCode} code @param {string} path @param {string} message @returns {never} */
function fail(code, path, message) {
  throw new CppCuteAotDockerImageError(code, path, message);
}
