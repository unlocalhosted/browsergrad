import { randomBytes } from "node:crypto";
import { constants } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  open,
  readdir,
  readFile,
} from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";

import {
  decodeWireJson,
  sha256Hex,
  wireIntegerToBigInt,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  inspectAuthorizedCppCuteAotOciMetadata,
} from "../dist/cpp_cute_aot_oci.js";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "../dist/cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT,
  CPP_CUTE_AOT_CONTAINER_HOSTNAME,
  CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT,
  CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS,
  CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_DECODE_LIMITS,
  CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
  CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA,
  CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA,
  CPP_CUTE_AOT_DOCKER_CREATE_LIMITS,
  CPP_CUTE_AOT_DOCKER_LIFECYCLE_LIMITS,
  CPP_CUTE_AOT_DOCKER_REMOVE_LIMITS,
  CPP_CUTE_AOT_DOCKER_START_LIMITS,
  CPP_CUTE_AOT_SANDBOX_POLICY_V1,
} from "../dist/cpp_cute_aot_policy.js";
import {
  copyCppCuteAotOfflineRunStagingInputs,
  decodeCppCuteAotResultFrame,
  unwrapPreparedCppCuteAotOfflineRun,
} from "../dist/cpp_cute_aot_runner_plan.js";
import { unwrapPreparedCppCuteFrontendProfile } from "../dist/cpp_cute_frontend_profile.js";
import {
  buildCppCuteAotDockerAbsenceRequest,
  buildCppCuteAotDockerContainerInspectRequest,
  buildCppCuteAotDockerContainerRecoveryRequest,
  buildCppCuteAotDockerCreateRequest,
  buildCppCuteAotDockerRemoveRequest,
  buildCppCuteAotDockerStartAttachedRequest,
  CppCuteAotDockerProcessError,
} from "./cpp_cute_aot_docker_process.mjs";
import {
  __runCppCuteAotLocalDockerImageSession,
  __runCppCuteAotLocalDockerImageSessionWithProcessForTest,
} from "./cpp_cute_aot_docker_shell.mjs";

const CONTAINER_ID = /^[0-9a-f]{64}$/u;
const ABORTED_GETTER = Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const LIVE_RUNS = new WeakMap();
const TEST_RUNS = new WeakMap();

/** @typedef {"BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CANCELLED" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-HOST" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-INVALID" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-STAGING" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CREATE" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CONTAINER-MISMATCH" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-START" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-TERMINAL" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-UNVERIFIED"} CppCuteAotDockerRunErrorCode */

export class CppCuteAotDockerRunError extends Error {
  /** @param {CppCuteAotDockerRunErrorCode} code @param {string} path @param {string} message @param {ErrorOptions} [options] */
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotDockerRunError";
    this.code = code;
    this.path = path;
  }
}

/**
 * @typedef {Readonly<{
 *   jobId: string;
 *   profileHash: string;
 *   executionPlanSha256: string;
 *   containerId: string;
 *   imageId: string;
 *   frontendOutcome: "accepted" | "rejected";
 * }>} CompletedCppCuteAotDockerRun
 */

/** @typedef {(request: import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessRequest) => Promise<import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessResult>} DockerProcess */

/**
 * @typedef {Readonly<{
 *   authorized: ReturnType<typeof inspectAuthorizedCppCuteAotOciMetadata>;
 *   containerName: string;
 *   sessionNonce: string;
 *   sourceDirectory: string;
 *   controlDirectory: string;
 *   containerIdFile: string;
 *   memoryBytes: number;
 *   maxProcesses: number;
 *   temporaryBytes: number;
 * }>} ExpectedContainer
 */

/**
 * Production lifecycle. Requires Linux process-group termination semantics and
 * accepts no caller Docker values, paths, container identity, or output bytes.
 *
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {{ readonly signal?: AbortSignal }} [options]
 * @returns {Promise<CompletedCppCuteAotDockerRun>}
 */
export async function executeCppCuteAotDockerRun(authorizedMetadata, options = {}) {
  if (process.platform !== "linux") {
    fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-HOST", "$host", "production Docker AOT runner requires Linux");
  }
  return executeWithSession(authorizedMetadata, undefined, options, LIVE_RUNS, true);
}

/**
 * Test-only effect adapter. Results are stored in a disjoint authority map.
 *
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {DockerProcess} processAdapter
 * @param {{ readonly signal?: AbortSignal }} [options]
 * @returns {Promise<CompletedCppCuteAotDockerRun>}
 */
export async function __executeCppCuteAotDockerRunWithProcessForTest(
  authorizedMetadata,
  processAdapter,
  options = {},
) {
  if (typeof processAdapter !== "function") invalid("$processAdapter", "test process adapter must be a function");
  return executeWithSession(authorizedMetadata, processAdapter, options, TEST_RUNS, false);
}

/** @param {CompletedCppCuteAotDockerRun} completed */
export function unwrapCompletedCppCuteAotDockerRun(completed) {
  return unwrapCompleted(completed, LIVE_RUNS);
}

/** @param {CompletedCppCuteAotDockerRun} completed */
export function __unwrapCompletedCppCuteAotDockerRunForTest(completed) {
  return unwrapCompleted(completed, TEST_RUNS);
}

/**
 * @param {import("../dist/cpp_cute_aot_oci.js").AuthorizedCppCuteAotOciMetadata} authorizedMetadata
 * @param {DockerProcess | undefined} processAdapter
 * @param {{ readonly signal?: AbortSignal }} options
 * @param {WeakMap<object, object>} authorityStore
 * @param {boolean} production
 * @returns {Promise<CompletedCppCuteAotDockerRun>}
 */
async function executeWithSession(
  authorizedMetadata,
  processAdapter,
  options,
  authorityStore,
  production,
) {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const authorized = inspectAuthorizedCppCuteAotOciMetadata(authorizedMetadata);
  const plan = authorized.plan;
  const planRecord = unwrapPreparedCppCuteAotOfflineRun(plan);
  const profile = unwrapPreparedCppCuteFrontendProfile(planRecord.profile).profile;
  /** @param {import("./cpp_cute_aot_docker_shell.mjs").DockerObservationSession} session */
  const continuation = (session) => runContainerLifecycle(
    session,
    authorized,
    profile.extractionLimits,
    production,
  );
  const sessionOptions = signal === undefined ? {} : { signal };
  const sessionResult = processAdapter === undefined
    ? await __runCppCuteAotLocalDockerImageSession(
      authorizedMetadata,
      continuation,
      sessionOptions,
    )
    : await __runCppCuteAotLocalDockerImageSessionWithProcessForTest(
      authorizedMetadata,
      processAdapter,
      continuation,
      sessionOptions,
    );
  throwIfAborted(signal);
  const verifiedResult = await decodeCppCuteAotResultFrame(
    plan,
    sessionResult.value.frameBytes,
    sessionOptions,
  );
  throwIfAborted(signal);
  const completed = Object.freeze({
    jobId: plan.jobId,
    profileHash: plan.profileHash,
    executionPlanSha256: plan.executionPlanSha256,
    containerId: sessionResult.value.containerId,
    imageId: sessionResult.observed.imageId,
    frontendOutcome: verifiedResult.frontendOutcome,
  });
  authorityStore.set(completed, Object.freeze({
    observedImage: sessionResult.observed,
    result: verifiedResult,
    evidence: sessionResult.value.evidence,
  }));
  return completed;
}

/**
 * @param {import("./cpp_cute_aot_docker_shell.mjs").DockerObservationSession} session
 * @param {ReturnType<typeof inspectAuthorizedCppCuteAotOciMetadata>} authorized
 * @param {import("../dist/cpp_cute_frontend_profile.js").CppCuteFrontendExtractionLimits} limits
 * @param {boolean} production
 */
async function runContainerLifecycle(session, authorized, limits, production) {
  const signal = session.signal;
  const deadline = performance.now() + limits.maxWallTimeMs + CPP_CUTE_AOT_DOCKER_LIFECYCLE_LIMITS.overheadMs;
  throwIfAborted(signal);
  const sessionNonce = randomBytes(16).toString("hex");
  const containerName = `browsergrad-cpp-cute-aot-${sessionNonce}`;
  const staged = await stageInputs(session.runRoot, authorized.plan, signal, deadline);
  const containerIdFile = join(session.runRoot, "container.cid");
  /** @type {ExpectedContainer} */
  const expected = Object.freeze({
    authorized,
    containerName,
    sessionNonce,
    sourceDirectory: staged.sourceDirectory,
    controlDirectory: staged.controlDirectory,
    containerIdFile,
    memoryBytes: limits.maxMemoryBytes,
    maxProcesses: limits.maxProcesses,
    temporaryBytes: Math.max(
      1,
      Math.min(Math.floor(limits.maxMemoryBytes / 4), 536_870_912),
    ),
  });
  /** @type {string | undefined} */
  let containerId;
  /** @type {unknown} */
  let primaryFailure;
  /** @type {Uint8Array | undefined} */
  let frameBytes;
  let startAttempted = false;
  let startClean = false;
  let createMayExist = false;
  let terminalVerified = false;
  try {
    const createRequest = withRemainingDeadline(buildCppCuteAotDockerCreateRequest({
      runRoot: session.runRoot,
      configDirectory: session.configDirectory,
      homeDirectory: session.homeDirectory,
      controlDirectory: staged.controlDirectory,
      sourceDirectory: staged.sourceDirectory,
      containerIdFile,
      containerName,
      sessionNonce,
      imageReference: authorized.plan.imageReference,
      jobId: authorized.plan.jobId,
      executionPlanSha256: authorized.plan.executionPlanSha256,
      memoryBytes: limits.maxMemoryBytes,
      maxProcesses: limits.maxProcesses,
      ...(signal === undefined ? {} : { signal }),
    }), deadline);
    let createResult;
    try {
      createMayExist = true;
      createResult = await executeProcess(
        session.processAdapter,
        createRequest,
        session.configDirectory,
        session.homeDirectory,
        CPP_CUTE_AOT_DOCKER_CREATE_LIMITS,
      );
      throwIfAborted(signal);
    } catch (error) {
      primaryFailure = normalizeProcessFailure(error, "container create failed");
    }
    const stdoutId = createResult === undefined ? undefined : parseCreateResult(createResult);
    const cidFileId = await readContainerIdFile(containerIdFile);
    if (stdoutId !== undefined && cidFileId !== undefined && stdoutId === cidFileId) {
      containerId = stdoutId;
    } else {
      if (primaryFailure === undefined) {
        primaryFailure = runError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CREATE",
          "$create",
          "Docker create output and cidfile did not bind one exact container ID",
        );
      }
      try {
        containerId = await recoverContainerId(session, expected, deadline);
      } catch (recoveryFailure) {
        unsafeCleanup(
          session,
          production,
          combineFailures(primaryFailure, recoveryFailure, "container identity recovery failed"),
        );
      }
      if (containerId === undefined) unsafeCleanup(session, production, primaryFailure);
    }
    const created = await inspectContainer(session, containerId, deadline, signal);
    verifyContainerProjection(created, expected, containerId, "created");
    if (primaryFailure !== undefined) throw primaryFailure;

    startAttempted = true;
    const startRequest = buildCppCuteAotDockerStartAttachedRequest({
      runRoot: session.runRoot,
      configDirectory: session.configDirectory,
      homeDirectory: session.homeDirectory,
      containerId,
      timeoutMs: Math.max(1, Math.min(limits.maxWallTimeMs, remainingMs(deadline))),
      stdoutByteLimit: authorized.plan.frameByteLimit,
      ...(signal === undefined ? {} : { signal }),
    });
    let startResult;
    try {
      startResult = await executeProcess(
        session.processAdapter,
        startRequest,
        session.configDirectory,
        session.homeDirectory,
        CPP_CUTE_AOT_DOCKER_START_LIMITS,
      );
      frameBytes = startResult.stdout;
      if (startResult.exitCode !== 0 || startResult.signal !== null) {
        throw runError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-START",
          "$start",
          "attached container execution did not exit cleanly",
        );
      }
      startClean = true;
    } catch (error) {
      primaryFailure = normalizeProcessFailure(error, "attached container execution failed");
    }
    const terminal = await inspectContainer(session, containerId, deadline, undefined);
    verifyContainerProjection(terminal, expected, containerId, "exited", startClean);
    terminalVerified = true;
    if (primaryFailure !== undefined) throw primaryFailure;
    if (frameBytes === undefined) {
      fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-START", "$start.stdout", "container produced no bounded result frame");
    }
  } catch (error) {
    primaryFailure = combineFailures(primaryFailure, error, "Docker lifecycle produced multiple failures");
  }

  let containerCleanupFailure;
  if (containerId !== undefined) {
    containerCleanupFailure = await removeAndProveAbsent(
      session,
      containerId,
      terminalVerified && primaryFailure === undefined,
    );
    if (containerCleanupFailure !== undefined && !containerCleanupFailure.absenceProved) {
      unsafeCleanup(session, production, primaryFailure ?? containerCleanupFailure.error);
    }
  } else if (createMayExist || startAttempted) {
    unsafeCleanup(session, production, primaryFailure);
  }
  try {
    await restoreStagingDirectoryModes(staged.directories);
  } catch (restoreFailure) {
    session.preserveRunRoot();
    throw cleanupError(
      combineFailures(
        primaryFailure,
        containerCleanupFailure?.error,
        "Docker lifecycle and container cleanup failed",
      ),
      restoreFailure,
    );
  }
  if (containerCleanupFailure !== undefined) {
    throw cleanupError(primaryFailure, containerCleanupFailure.error);
  }
  if (primaryFailure !== undefined) throw primaryFailure;
  if (containerId === undefined || frameBytes === undefined || !terminalVerified) {
    fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-TERMINAL", "$", "Docker lifecycle did not reach one verified terminal result");
  }
  return Object.freeze({
    containerId,
    frameBytes: new Uint8Array(frameBytes),
    evidence: Object.freeze({
      sequence: Object.freeze([
        "version", "info", "image-inspect", "create", "inspect-created",
        "start-attached", "inspect-terminal", "remove", "prove-absent",
      ]),
      containerAbsent: true,
      stagingCleanup: "owned-by-observation-session",
    }),
  });
}

/** @param {string} runRoot @param {import("../dist/cpp_cute_aot_runner_plan.js").PreparedCppCuteAotOfflineRun} plan @param {AbortSignal | undefined} signal @param {number} deadline */
async function stageInputs(runRoot, plan, signal, deadline) {
  checkLifecycleDeadline(deadline, signal);
  const inputs = copyCppCuteAotOfflineRunStagingInputs(plan);
  const sourceDirectory = join(runRoot, "source");
  const controlDirectory = join(runRoot, "control");
  await mkdir(sourceDirectory, { mode: 0o700 });
  checkLifecycleDeadline(deadline, signal);
  await mkdir(controlDirectory, { mode: 0o700 });
  checkLifecycleDeadline(deadline, signal);
  const expectedFiles = [];
  expectedFiles.push(await stageFile(
    join(controlDirectory, "profile.json"),
    inputs.profileBytes,
    await sha256Hex(inputs.profileBytes),
    signal,
    deadline,
  ));
  checkLifecycleDeadline(deadline, signal);
  expectedFiles.push(await stageFile(
    join(controlDirectory, "job.json"),
    inputs.jobBytes,
    await sha256Hex(inputs.jobBytes),
    signal,
    deadline,
  ));
  checkLifecycleDeadline(deadline, signal);
  expectedFiles.push(await stageFile(
    join(controlDirectory, "execution-environment.json"),
    inputs.environmentBytes,
    await sha256Hex(inputs.environmentBytes),
    signal,
    deadline,
  ));
  const createdDirectories = new Set([sourceDirectory, controlDirectory]);
  for (const source of inputs.sourceBlobs) {
    checkLifecycleDeadline(deadline, signal);
    const target = sourceStagePath(sourceDirectory, source.virtualPath);
    await mkdir(dirname(target), { recursive: true, mode: 0o700 });
    for (let current = dirname(target); current.startsWith(`${sourceDirectory}${sep}`); current = dirname(current)) {
      createdDirectories.add(current);
      if (current === sourceDirectory) break;
    }
    if (BigInt(source.bytes.byteLength) !== wireIntegerToBigInt(source.byteLength)) {
      stagingFailure(target, "source byte length changed before staging");
    }
    expectedFiles.push(await stageFile(
      target,
      source.bytes,
      source.contentSha256,
      signal,
      deadline,
    ));
  }
  try {
    for (const directory of [...createdDirectories].sort((left, right) => right.length - left.length)) {
      await chmod(directory, 0o555);
      checkLifecycleDeadline(deadline, signal);
    }
    await verifyStagedTree(runRoot, expectedFiles, createdDirectories, deadline, signal);
  } catch (failure) {
    try {
      await restoreStagingDirectoryModes([...createdDirectories]);
    } catch (restoreFailure) {
      throw cleanupError(failure, restoreFailure);
    }
    throw failure;
  }
  return Object.freeze({
    sourceDirectory,
    controlDirectory,
    directories: Object.freeze([...createdDirectories].sort()),
  });
}

/** @param {readonly string[]} directories */
async function restoreStagingDirectoryModes(directories) {
  for (const path of directories) {
    const handle = await open(
      path,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    try {
      const status = await handle.stat();
      if (!status.isDirectory()) stagingFailure(path, "staging cleanup path is not a directory");
      await handle.chmod(0o700);
    } finally {
      await handle.close();
    }
  }
}

/** @param {string} path @param {Uint8Array} bytes @param {string} digest @param {AbortSignal | undefined} signal @param {number} deadline */
async function stageFile(path, bytes, digest, signal, deadline) {
  checkLifecycleDeadline(deadline, signal);
  const handle = await open(
    path,
    constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
    0o400,
  );
  try {
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.chmod(0o444);
  } finally {
    await handle.close();
  }
  checkLifecycleDeadline(deadline, signal);
  const status = await lstat(path);
  if (
    !status.isFile()
    || status.isSymbolicLink()
    || status.nlink !== 1
    || status.size !== bytes.byteLength
    || (status.mode & 0o777) !== 0o444
  ) {
    stagingFailure(path, "staged file identity differs from the private snapshot");
  }
  const readback = await readFile(path);
  if (await sha256Hex(readback) !== digest) stagingFailure(path, "staged file digest differs from the private snapshot");
  checkLifecycleDeadline(deadline, signal);
  return Object.freeze({ path, digest, bytes: bytes.byteLength });
}

/** @param {string} sourceDirectory @param {string} virtualPath */
function sourceStagePath(sourceDirectory, virtualPath) {
  if (typeof virtualPath !== "string" || !virtualPath.startsWith("/") || virtualPath.includes("\0")) {
    stagingFailure("$source.virtualPath", "source virtual path is not normalized absolute POSIX syntax");
  }
  const target = resolve(sourceDirectory, virtualPath.slice(1));
  if (!target.startsWith(`${sourceDirectory}${sep}`)) {
    stagingFailure("$source.virtualPath", "source virtual path escapes the private staging root");
  }
  return target;
}

/** @param {string} runRoot @param {readonly {path: string; digest: string; bytes: number}[]} expectedFiles @param {ReadonlySet<string>} expectedDirectories @param {number} deadline @param {AbortSignal | undefined} signal */
async function verifyStagedTree(runRoot, expectedFiles, expectedDirectories, deadline, signal) {
  const expectedPaths = new Set([
    ...expectedFiles.map((entry) => entry.path),
    ...expectedDirectories,
    join(runRoot, "docker-config"),
    join(runRoot, "home"),
  ]);
  const pending = [...expectedDirectories, join(runRoot, "docker-config"), join(runRoot, "home")];
  for (const directory of pending) {
    const status = await lstat(directory);
    const expectedMode = expectedDirectories.has(directory) ? 0o555 : 0o700;
    if (
      !status.isDirectory()
      || status.isSymbolicLink()
      || status.nlink < 1
      || (status.mode & 0o777) !== expectedMode
    ) {
      stagingFailure(directory, "staging tree contains a non-directory or symlink");
    }
    for (const name of await readdir(directory)) {
      const child = join(directory, name);
      if (!expectedPaths.has(child)) stagingFailure(child, "staging tree contains an unexpected path");
    }
    checkLifecycleDeadline(deadline, signal);
  }
}

/** @param {import("./cpp_cute_aot_docker_shell.mjs").DockerObservationSession} session @param {ExpectedContainer} expected @param {number} deadline */
async function recoverContainerId(session, expected, deadline) {
  for (let attempt = 0; attempt < CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS.recoveryAttempts; attempt += 1) {
    const request = withRemainingDeadline(buildCppCuteAotDockerContainerRecoveryRequest({
      runRoot: session.runRoot,
      configDirectory: session.configDirectory,
      homeDirectory: session.homeDirectory,
      containerName: expected.containerName,
    }), deadline);
    let result;
    try {
      result = await executeProcess(
        session.processAdapter,
        request,
        session.configDirectory,
        session.homeDirectory,
        CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
      );
    } catch {
      result = undefined;
    }
    if (result !== undefined && result.exitCode === 0 && result.signal === null && result.stderr.byteLength === 0) {
      const projection = decodeJsonLine(result.stdout, "container recovery");
      const object = exactObject(projection, ["id", "labels", "name", "schema", "version"], "$recovery");
      if (object.schema !== CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA || object.version !== 1) {
        containerMismatch("$recovery", "container recovery schema/version mismatch");
      }
      const id = containerId(object.id, "$recovery.id");
      verifyNameAndLabels(object.name, object.labels, expected, "$recovery");
      return id;
    }
    if (attempt + 1 < CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS.recoveryAttempts) {
      await delay(Math.min(
        CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS.recoveryIntervalMs,
        remainingMs(deadline),
      ));
    }
  }
  return undefined;
}

/** @param {import("./cpp_cute_aot_docker_shell.mjs").DockerObservationSession} session @param {string} id @param {number} deadline @param {AbortSignal | undefined} signal */
async function inspectContainer(session, id, deadline, signal) {
  const request = withRemainingDeadline(buildCppCuteAotDockerContainerInspectRequest({
    runRoot: session.runRoot,
    configDirectory: session.configDirectory,
    homeDirectory: session.homeDirectory,
    containerId: id,
    ...(signal === undefined ? {} : { signal }),
  }), deadline);
  const result = await executeProcess(
    session.processAdapter,
    request,
    session.configDirectory,
    session.homeDirectory,
    CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
  );
  if (result.exitCode !== 0 || result.signal !== null || result.stderr.byteLength !== 0) {
    containerMismatch("$inspect", "Docker container inspect did not exit cleanly");
  }
  return decodeJsonLine(result.stdout, "container inspect");
}

/** @param {unknown} value @param {ExpectedContainer} expected @param {string} id @param {"created" | "exited"} state @param {boolean} [requireZeroExit] */
function verifyContainerProjection(value, expected, id, state, requireZeroExit = true) {
  const root = exactObject(
    value,
    [
      "args", "config", "hostConfig", "id", "image", "imageManifestDescriptor",
      "mounts", "name", "path", "restartCount", "schema", "state", "version",
    ],
    "$container",
  );
  if (root.schema !== CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA || root.version !== 1) {
    containerMismatch("$container", "container projection schema/version mismatch");
  }
  if (containerId(root.id, "$container.id") !== id) containerMismatch("$container.id", "container ID changed");
  if (root.image !== expected.authorized.plan.imageReference.split("@")[1]) {
    containerMismatch("$container.image", "container image ID differs from authorized manifest");
  }
  const descriptor = objectValue(root.imageManifestDescriptor, "$container.imageManifestDescriptor");
  if (
    descriptor.digest !== expected.authorized.metadata.manifest.digest
    || descriptor.mediaType !== expected.authorized.metadata.manifest.mediaType
    || descriptor.size !== expected.authorized.metadata.manifest.size
  ) {
    containerMismatch("$container.imageManifestDescriptor", "created container manifest descriptor differs from authorized OCI metadata");
  }
  const platform = objectValue(descriptor.platform, "$container.imageManifestDescriptor.platform");
  if (platform.os !== "linux" || platform.architecture !== "amd64") {
    containerMismatch("$container.imageManifestDescriptor.platform", "created container manifest platform differs from policy");
  }
  for (const field of ["variant", "os.version"]) {
    if (platform[field] !== undefined && platform[field] !== "") {
      containerMismatch(`$container.imageManifestDescriptor.platform.${field}`, "container manifest platform qualifier is unsupported");
    }
  }
  if (platform["os.features"] !== undefined) {
    expectEmptyArray(platform["os.features"], "$container.imageManifestDescriptor.platform.os.features");
  }
  if (root.path !== CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.entrypoint) {
    containerMismatch("$container.path", "container entrypoint path differs from policy");
  }
  expectExactStringArray(root.args, CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments, "$container.args");
  if (root.restartCount !== 0) containerMismatch("$container.restartCount", "container restart count must remain zero");
  verifyNameAndLabels(root.name, objectValue(root.config, "$container.config").Labels, expected, "$container");
  verifyContainerConfig(objectValue(root.config, "$container.config"), expected);
  verifyHostConfig(objectValue(root.hostConfig, "$container.hostConfig"), expected);
  verifyMounts(root.mounts, expected);
  verifyState(objectValue(root.state, "$container.state"), state, requireZeroExit);
}

/** @param {Record<string, unknown>} config @param {ExpectedContainer} expected */
function verifyContainerConfig(config, expected) {
  const expectedUser = `${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.uid}:${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.gid}`;
  /** @type {readonly (readonly [string, unknown])[]} */
  const checks = [
    ["Hostname", CPP_CUTE_AOT_CONTAINER_HOSTNAME],
    ["Domainname", ""],
    ["User", expectedUser],
    ["Image", expected.authorized.plan.imageReference],
    ["WorkingDir", CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.workingDirectory],
  ];
  for (const [field, expectedValue] of checks) {
    if (config[field] !== expectedValue) containerMismatch(`$container.config.${field}`, "container config differs from policy");
  }
  for (const field of ["AttachStdin", "Tty", "OpenStdin", "StdinOnce"]) {
    if (config[field] !== false) containerMismatch(`$container.config.${field}`, "container input/TTY must be disabled");
  }
  for (const field of ["AttachStdout", "AttachStderr"]) {
    if (config[field] !== true) containerMismatch(`$container.config.${field}`, "container output streams must be attached");
  }
  expectExactStringArray(config.Entrypoint, [CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.entrypoint], "$container.config.Entrypoint");
  expectExactStringArray(config.Cmd, CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments, "$container.config.Cmd");
  expectEmptyArray(config.Env, "$container.config.Env");
  expectEmptyObject(config.ExposedPorts, "$container.config.ExposedPorts");
  expectEmptyObject(config.Volumes, "$container.config.Volumes");
  const healthcheck = objectValue(config.Healthcheck, "$container.config.Healthcheck");
  expectExactStringArray(healthcheck.Test, ["NONE"], "$container.config.Healthcheck.Test");
}

/** @param {Record<string, unknown>} host @param {ExpectedContainer} expected */
function verifyHostConfig(host, expected) {
  /** @type {readonly (readonly [string, unknown])[]} */
  const exact = [
    ["NetworkMode", "none"], ["IpcMode", "none"], ["CgroupnsMode", "private"],
    ["PidMode", ""], ["UTSMode", ""], ["Runtime", CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.runtime],
    ["Memory", expected.memoryBytes], ["MemorySwap", expected.memoryBytes],
    ["PidsLimit", expected.maxProcesses], ["MemoryReservation", 0], ["NanoCpus", 0],
    ["CpuShares", 0], ["CpuPeriod", 0], ["CpuQuota", 0],
    ["CpuRealtimePeriod", 0], ["CpuRealtimeRuntime", 0],
    ["CpusetCpus", ""], ["CpusetMems", ""], ["UsernsMode", ""],
    ["CgroupParent", ""], ["ContainerIDFile", expected.containerIdFile],
    ["OomScoreAdj", 0], ["ShmSize", 67_108_864],
  ];
  for (const [field, expectedValue] of exact) {
    if (host[field] !== expectedValue) containerMismatch(`$container.hostConfig.${field}`, "container host config differs from policy");
  }
  for (const field of ["AutoRemove", "Privileged", "PublishAllPorts"]) {
    if (host[field] !== false) containerMismatch(`$container.hostConfig.${field}`, "container host privilege/lifecycle flag must be false");
  }
  if (host.ReadonlyRootfs !== true) containerMismatch("$container.hostConfig.ReadonlyRootfs", "container root must be read-only");
  expectEmptyArray(host.CapAdd, "$container.hostConfig.CapAdd");
  expectExactStringArray(host.CapDrop, ["ALL"], "$container.hostConfig.CapDrop");
  expectExactStringArray(host.SecurityOpt, ["no-new-privileges=true"], "$container.hostConfig.SecurityOpt");
  for (const field of [
    "Binds", "Devices", "DeviceRequests", "DeviceCgroupRules", "Dns", "DnsOptions",
    "DnsSearch", "ExtraHosts", "GroupAdd", "Links", "PortBindings", "Ulimits",
    "VolumesFrom",
  ]) {
    if (field === "PortBindings") expectEmptyObject(host[field], `$container.hostConfig.${field}`);
    else expectEmptyArray(host[field], `$container.hostConfig.${field}`);
  }
  const logging = objectValue(host.LogConfig, "$container.hostConfig.LogConfig");
  if (logging.Type !== "none") containerMismatch("$container.hostConfig.LogConfig.Type", "container logging must be disabled");
  expectEmptyObject(logging.Config, "$container.hostConfig.LogConfig.Config");
  const restart = objectValue(host.RestartPolicy, "$container.hostConfig.RestartPolicy");
  if (restart.Name !== "no" || restart.MaximumRetryCount !== 0) {
    containerMismatch("$container.hostConfig.RestartPolicy", "container restart policy must be disabled");
  }
  const tmpfs = objectValue(host.Tmpfs, "$container.hostConfig.Tmpfs");
  const expectedTmpfs = `rw,noexec,nosuid,nodev,size=${expected.temporaryBytes},mode=1777`;
  if (Reflect.ownKeys(tmpfs).length !== 1 || tmpfs["/tmp"] !== expectedTmpfs) {
    containerMismatch("$container.hostConfig.Tmpfs", "container tmpfs differs from policy");
  }
  verifyHostMounts(host.Mounts, expected);
}

/** @param {unknown} mounts @param {ExpectedContainer} expected */
function verifyHostMounts(mounts, expected) {
  const entries = arrayValue(mounts, "$container.hostConfig.Mounts");
  if (entries.length !== 2) containerMismatch("$container.hostConfig.Mounts", "container requires exactly two host mounts");
  const byTarget = new Map();
  for (const [index, raw] of entries.entries()) {
    const mount = exactObject(
      raw,
      ["BindOptions", "ReadOnly", "Source", "Target", "Type"],
      `$container.hostConfig.Mounts[${index}]`,
    );
    byTarget.set(mount.Target, mount);
  }
  for (const [target, source] of [
    [CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT, expected.sourceDirectory],
    [CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT, expected.controlDirectory],
  ]) {
    const mount = byTarget.get(target);
    if (
      mount === undefined
      || mount.Type !== "bind"
      || mount.Source !== source
      || mount.ReadOnly !== true
    ) containerMismatch("$container.hostConfig.Mounts", "container bind mount differs from policy");
    const bind = exactObject(
      mount.BindOptions,
      ["Propagation"],
      "$container.hostConfig.Mounts.BindOptions",
    );
    if (bind.Propagation !== "rprivate") containerMismatch("$container.hostConfig.Mounts.BindOptions", "bind propagation must be rprivate");
  }
}

/** @param {unknown} mounts @param {ExpectedContainer} expected */
function verifyMounts(mounts, expected) {
  const entries = arrayValue(mounts, "$container.mounts");
  if (entries.length !== 2) containerMismatch("$container.mounts", "container requires exactly two registered mountpoints");
  const byDestination = new Map();
  for (const [index, raw] of entries.entries()) {
    const mount = exactObject(
      raw,
      ["Destination", "Mode", "Propagation", "RW", "Source", "Type"],
      `$container.mounts[${index}]`,
    );
    byDestination.set(mount.Destination, mount);
  }
  for (const [destination, source] of [
    [CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT, expected.sourceDirectory],
    [CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT, expected.controlDirectory],
  ]) {
    const mount = byDestination.get(destination);
    if (
      mount === undefined
      || mount.Type !== "bind"
      || mount.Source !== source
      || mount.Mode !== ""
      || mount.RW !== false
      || mount.Propagation !== "rprivate"
    ) containerMismatch("$container.mounts", "registered bind mountpoint differs from policy");
  }
}

/** @param {Record<string, unknown>} state @param {"created" | "exited"} expectedStatus @param {boolean} requireZeroExit */
function verifyState(state, expectedStatus, requireZeroExit) {
  if (state.Status !== expectedStatus) containerMismatch("$container.state.Status", `container state must be ${expectedStatus}`);
  for (const field of ["Running", "Paused", "Restarting", "OOMKilled", "Dead"]) {
    if (state[field] !== false) containerMismatch(`$container.state.${field}`, "container state flag must be false");
  }
  const exitCode = state.ExitCode;
  if (
    state.Pid !== 0
    || typeof exitCode !== "number"
    || !Number.isSafeInteger(exitCode)
    || exitCode < 0
    || (requireZeroExit && exitCode !== 0)
    || state.Error !== ""
  ) {
    containerMismatch("$container.state", "container PID/exit/error state is invalid");
  }
  if (state.Health !== undefined) containerMismatch("$container.state.Health", "container health state must be absent");
  if (expectedStatus === "created") {
    if (state.StartedAt !== "0001-01-01T00:00:00Z" || state.FinishedAt !== "0001-01-01T00:00:00Z") {
      containerMismatch("$container.state", "created container timestamps must be zero");
    }
  } else {
    const startedAt = timestampMilliseconds(state.StartedAt, "$container.state.StartedAt");
    const finishedAt = timestampMilliseconds(state.FinishedAt, "$container.state.FinishedAt");
    if (finishedAt < startedAt) containerMismatch("$container.state.FinishedAt", "container finish time precedes start time");
  }
}

/** @param {unknown} value @param {string} path */
function timestampMilliseconds(value, path) {
  if (typeof value !== "string" || value === "0001-01-01T00:00:00Z") containerMismatch(path, "expected nonzero RFC 3339 timestamp");
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) containerMismatch(path, "expected RFC 3339 timestamp");
  return parsed;
}

/** @param {unknown} name @param {unknown} labelsValue @param {ExpectedContainer} expected @param {string} path */
function verifyNameAndLabels(name, labelsValue, expected, path) {
  if (name !== `/${expected.containerName}`) containerMismatch(`${path}.name`, "container name differs from private session");
  const labels = objectValue(labelsValue, `${path}.labels`);
  const expectedLabels = {
    "browsergrad.job": expected.authorized.plan.jobId,
    "browsergrad.owner": "cpp-cute-aot",
    "browsergrad.plan": expected.authorized.plan.executionPlanSha256,
    "browsergrad.session": expected.sessionNonce,
  };
  const keys = Object.keys(labels).sort();
  if (keys.length !== 4 || keys.some((key, index) => key !== Object.keys(expectedLabels).sort()[index])) {
    containerMismatch(`${path}.labels`, "container labels differ from closed session labels");
  }
  for (const [key, expectedValue] of Object.entries(expectedLabels)) {
    if (labels[key] !== expectedValue) containerMismatch(`${path}.labels.${key}`, "container label differs from session authority");
  }
}

/** @param {import("./cpp_cute_aot_docker_shell.mjs").DockerObservationSession} session @param {string} id @param {boolean} cleanTerminal */
async function removeAndProveAbsent(session, id, cleanTerminal) {
  const cleanupDeadline = performance.now() + CPP_CUTE_AOT_DOCKER_LIFECYCLE_LIMITS.cleanupMs;
  /** @type {unknown} */
  let error;
  try {
    const remove = await executeProcess(
      session.processAdapter,
      withRemainingDeadline(buildCppCuteAotDockerRemoveRequest({
        runRoot: session.runRoot,
        configDirectory: session.configDirectory,
        homeDirectory: session.homeDirectory,
        containerId: id,
        force: !cleanTerminal,
      }), cleanupDeadline),
      session.configDirectory,
      session.homeDirectory,
      CPP_CUTE_AOT_DOCKER_REMOVE_LIMITS,
    );
    if (
      remove.exitCode !== 0
      || remove.signal !== null
      || remove.stderr.byteLength !== 0
      || new TextDecoder().decode(remove.stdout) !== `${id}\n`
    ) throw runError("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP", "$remove", "Docker container removal was not exact");
  } catch (cause) {
    error = normalizeProcessFailure(cause, "Docker container removal failed");
  }
  try {
    const absence = await executeProcess(
      session.processAdapter,
      withRemainingDeadline(buildCppCuteAotDockerAbsenceRequest({
        runRoot: session.runRoot,
        configDirectory: session.configDirectory,
        homeDirectory: session.homeDirectory,
        containerId: id,
      }), cleanupDeadline),
      session.configDirectory,
      session.homeDirectory,
      CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS,
    );
    if (
      absence.exitCode !== 0
      || absence.signal !== null
      || absence.stderr.byteLength !== 0
      || absence.stdout.byteLength !== 0
    ) throw runError("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP", "$absence", "container absence could not be proven");
    return error === undefined ? undefined : Object.freeze({ error, absenceProved: true });
  } catch (cause) {
    return Object.freeze({
      error: normalizeProcessFailure(cause, "container absence proof failed"),
      absenceProved: false,
    });
  }
}

/** @param {DockerProcess} adapter @param {import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessRequest} request @param {string} configDirectory @param {string} homeDirectory @param {{stdoutBytes: number; stderrBytes: number}} limits */
async function executeProcess(adapter, request, configDirectory, homeDirectory, limits) {
  let raw;
  try {
    raw = await adapter(request);
  } catch (cause) {
    if (cause instanceof CppCuteAotDockerProcessError) throw cause;
    throw runError("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS", "$process", "Docker lifecycle process adapter failed");
  }
  const result = snapshotProcessResult(raw, limits);
  if ((await readdir(configDirectory)).length !== 0 || (await readdir(homeDirectory)).length !== 0) {
    fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS", "$dockerClientState", "Docker command modified private empty client state");
  }
  return result;
}

/** @param {unknown} raw @param {{stdoutBytes: number; stderrBytes: number}} limits */
function snapshotProcessResult(raw, limits) {
  const object = exactDataObject(raw, ["exitCode", "signal", "stderr", "stdout"], [], "$process");
  const exitCode = object.exitCode;
  const signal = object.signal;
  if (exitCode !== null && (typeof exitCode !== "number" || !Number.isSafeInteger(exitCode) || exitCode < 0 || exitCode > 255)) {
    invalid("$process.exitCode", "process exit code is invalid");
  }
  if (signal !== null && (typeof signal !== "string" || !/^SIG[A-Z0-9]+$/u.test(signal))) {
    invalid("$process.signal", "process signal is invalid");
  }
  return Object.freeze({
    exitCode,
    signal,
    stdout: snapshotBytes(object.stdout, "$process.stdout", limits.stdoutBytes),
    stderr: snapshotBytes(object.stderr, "$process.stderr", limits.stderrBytes),
  });
}

/** @param {unknown} value @param {string} path @param {number} limit */
function snapshotBytes(value, path, limit) {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch {
    invalid(path, "process output must be an unshared plain Uint8Array");
  }
  if (inspected.byteLength > limit) invalid(path, "process output exceeds lifecycle policy");
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch {
    invalid(path, "process output became unreadable while snapshotting");
  }
}

/** @param {import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessResult} result */
function parseCreateResult(result) {
  if (result.exitCode !== 0 || result.signal !== null || result.stderr.byteLength !== 0) return undefined;
  const text = new TextDecoder().decode(result.stdout);
  if (!/^[0-9a-f]{64}\n$/u.test(text)) return undefined;
  return text.slice(0, -1);
}

/** @param {string} path */
async function readContainerIdFile(path) {
  try {
    const status = await lstat(path);
    if (!status.isFile() || status.isSymbolicLink() || status.nlink !== 1 || status.size !== CPP_CUTE_AOT_DOCKER_CREATE_LIMITS.cidFileBytes) {
      return undefined;
    }
    const text = await readFile(path, "utf8");
    return CONTAINER_ID.test(text) ? text : undefined;
  } catch {
    return undefined;
  }
}

/** @param {Uint8Array} stdout @param {string} stage */
function decodeJsonLine(stdout, stage) {
  if (
    stdout.byteLength < 3
    || stdout[0] !== 0x7b
    || stdout[stdout.byteLength - 1] !== 0x0a
    || stdout[stdout.byteLength - 2] === 0x0d
  ) containerMismatch("$process.stdout", `Docker ${stage} must emit one compact JSON object plus LF`);
  const document = stdout.subarray(0, -1);
  if (document.includes(0x0a) || document.includes(0x0d) || document.includes(0x00)) {
    containerMismatch("$process.stdout", `Docker ${stage} output contains a control separator`);
  }
  try {
    return decodeWireJson(document, { limits: CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_DECODE_LIMITS });
  } catch {
    containerMismatch("$process.stdout", `Docker ${stage} output is not bounded strict JSON`);
  }
}

/** @param {import("./cpp_cute_aot_docker_process.mjs").BoundedChildProcessRequest} request @param {number} deadline */
function withRemainingDeadline(request, deadline) {
  return Object.freeze({ ...request, timeoutMs: Math.max(1, Math.min(request.timeoutMs, remainingMs(deadline))) });
}

/** @param {number} deadline */
function remainingMs(deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS", "$deadline", "aggregate Docker lifecycle deadline expired");
  return Math.floor(remaining);
}

/** @param {number} deadline @param {AbortSignal | undefined} signal */
function checkLifecycleDeadline(deadline, signal) {
  throwIfAborted(signal);
  remainingMs(deadline);
}

/** @param {number} milliseconds */
function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

/** @param {import("./cpp_cute_aot_docker_shell.mjs").DockerObservationSession} session @param {boolean} production @param {unknown} cause @returns {never} */
function unsafeCleanup(session, production, cause) {
  session.preserveRunRoot();
  if (production) process.abort();
  throw runError(
    "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP",
    "$cleanup",
    "container absence is unproved; private staging was preserved",
    cause instanceof Error ? { cause } : undefined,
  );
}

/** @param {unknown} primary @param {unknown} cleanup */
function cleanupError(primary, cleanup) {
  const failures = [cleanup, primary].filter((failure) => failure instanceof Error);
  return runError(
    "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP",
    "$cleanup",
    "Docker container cleanup failed",
    failures.length === 0
      ? undefined
      : { cause: failures.length === 1 ? failures[0] : new AggregateError(failures, "cleanup and lifecycle failures") },
  );
}

/** @param {unknown} primary @param {unknown} secondary @param {string} message */
function combineFailures(primary, secondary, message) {
  if (primary === undefined || primary === secondary) return secondary;
  if (secondary === undefined) return primary;
  const failures = [primary, secondary].filter((failure) => failure instanceof Error);
  return runError(
    "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS",
    "$lifecycle",
    message,
    failures.length === 0 ? undefined : { cause: new AggregateError(failures, message) },
  );
}

/** @param {unknown} error @param {string} message */
function normalizeProcessFailure(error, message) {
  if (error instanceof CppCuteAotDockerRunError || error instanceof CppCuteAotDockerProcessError) return error;
  return runError("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS", "$process", message);
}

/** @param {CompletedCppCuteAotDockerRun} completed @param {WeakMap<object, object>} store */
function unwrapCompleted(completed, store) {
  if (typeof completed !== "object" || completed === null) unverified();
  const record = store.get(completed);
  if (record === undefined) unverified();
  return Object.freeze(record);
}

/** @param {{ readonly signal?: AbortSignal }} options */
function normalizeOptions(options) {
  const object = exactDataObject(options, [], ["signal"], "$options");
  const signal = object.signal;
  if (signal === undefined) return undefined;
  try {
    ABORTED_GETTER?.call(signal);
  } catch {
    invalid("$options.signal", "signal must be an AbortSignal");
  }
  return /** @type {AbortSignal} */ (signal);
}

/** @param {AbortSignal | undefined} signal */
function throwIfAborted(signal) {
  if (signal === undefined) return;
  let aborted;
  try {
    aborted = ABORTED_GETTER?.call(signal);
  } catch {
    invalid("$options.signal", "signal became unreadable");
  }
  if (aborted === true) cancelled();
}

/** @param {unknown} value @param {readonly string[]} required @param {string} path @returns {Record<string, unknown>} */
function exactObject(value, required, path) {
  const object = objectValue(value, path);
  const keys = Reflect.ownKeys(object);
  if (keys.length !== required.length || keys.some((key) => typeof key !== "string" || !required.includes(key))) {
    containerMismatch(path, "projection fields differ from closed schema");
  }
  return object;
}

/** @param {unknown} value @param {readonly string[]} required @param {readonly string[]} optional @param {string} path @returns {Record<string, unknown>} */
function exactDataObject(value, required, optional, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected plain data object");
  let prototype;
  let descriptors;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch {
    invalid(path, "data object must be inspectable");
  }
  if (prototype !== Object.prototype && prototype !== null) invalid(path, "expected plain data object");
  const allowed = new Set([...required, ...optional]);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.some((key) => typeof key !== "string" || !allowed.has(key)) || required.some((key) => !(key in descriptors))) {
    invalid(path, "data object fields differ from closed schema");
  }
  const result = Object.create(null);
  for (const key of keys) {
    if (typeof key !== "string") continue;
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) invalid(`${path}.${key}`, "field must be enumerable data property");
    result[key] = descriptor.value;
  }
  return result;
}

/** @param {unknown} value @param {string} path @returns {Record<string, unknown>} */
function objectValue(value, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) containerMismatch(path, "expected JSON object");
  return /** @type {Record<string, unknown>} */ (value);
}

/** @param {unknown} value @param {string} path */
function arrayValue(value, path) {
  if (!Array.isArray(value)) containerMismatch(path, "expected JSON array");
  return value;
}

/** @param {unknown} actual @param {readonly string[]} expected @param {string} path */
function expectExactStringArray(actual, expected, path) {
  const array = arrayValue(actual, path);
  if (array.length !== expected.length || array.some((entry, index) => entry !== expected[index])) {
    containerMismatch(path, "string array differs from policy");
  }
}

/** @param {unknown} actual @param {string} path */
function expectEmptyArray(actual, path) {
  if (actual === null || actual === undefined) return;
  if (!Array.isArray(actual) || actual.length !== 0) containerMismatch(path, "expected empty array");
}

/** @param {unknown} actual @param {string} path */
function expectEmptyObject(actual, path) {
  if (actual === null || actual === undefined) return;
  const object = objectValue(actual, path);
  if (Reflect.ownKeys(object).length !== 0) containerMismatch(path, "expected empty object");
}

/** @param {unknown} value @param {string} path */
function containerId(value, path) {
  if (typeof value !== "string" || !CONTAINER_ID.test(value)) containerMismatch(path, "expected full lowercase Docker container ID");
  return value;
}

/** @returns {never} */
function cancelled() {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CANCELLED", "$options.signal", "Docker AOT lifecycle was aborted");
}

/** @param {string} path @param {string} message @returns {never} */
function stagingFailure(path, message) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-STAGING", path, message);
}

/** @param {string} path @param {string} message @returns {never} */
function containerMismatch(path, message) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CONTAINER-MISMATCH", path, message);
}

/** @returns {never} */
function unverified() {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-UNVERIFIED", "$", "expected opaque production Docker AOT run authority");
}

/** @param {string} path @param {string} message @returns {never} */
function invalid(path, message) {
  fail("BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-INVALID", path, message);
}

/** @param {CppCuteAotDockerRunErrorCode} code @param {string} path @param {string} message @param {ErrorOptions} [options] */
function runError(code, path, message, options) {
  return new CppCuteAotDockerRunError(code, path, message, options);
}

/** @param {CppCuteAotDockerRunErrorCode} code @param {string} path @param {string} message @param {ErrorOptions} [options] @returns {never} */
function fail(code, path, message, options) {
  throw runError(code, path, message, options);
}
