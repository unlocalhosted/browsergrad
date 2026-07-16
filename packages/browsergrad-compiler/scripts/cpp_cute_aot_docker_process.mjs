import { spawn } from "node:child_process";
import { isAbsolute, join, resolve } from "node:path";

import {
  CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT,
  CPP_CUTE_AOT_CONTAINER_HOSTNAME,
  CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT,
  CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS,
  CPP_CUTE_AOT_DOCKER_API_VERSION,
  CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_FORMAT,
  CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
  CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_FORMAT,
  CPP_CUTE_AOT_DOCKER_CREATE_LIMITS,
  CPP_CUTE_AOT_DOCKER_EXECUTABLE,
  CPP_CUTE_AOT_DOCKER_INFO_FORMAT,
  CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_FORMAT,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS,
  CPP_CUTE_AOT_DOCKER_REMOVE_LIMITS,
  CPP_CUTE_AOT_DOCKER_START_LIMITS,
  CPP_CUTE_AOT_DOCKER_VERSION_FORMAT,
  CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
  CPP_CUTE_AOT_HARD_FRAME_BYTE_LIMIT,
  CPP_CUTE_AOT_PRIVATE_SECCOMP_FILE,
  CPP_CUTE_AOT_SANDBOX_POLICY_V1,
} from "../dist/cpp_cute_aot_policy.js";

const OCI_IMAGE_REFERENCE = /^[a-z0-9.-]+(?::[1-9][0-9]*)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$/u;
const CONTAINER_ID = /^[0-9a-f]{64}$/u;
const CONTAINER_NAME = /^browsergrad-cpp-cute-aot-[0-9a-f]{32}$/u;
const SESSION_NONCE = /^[0-9a-f]{32}$/u;
const JOB_ID = /^bg\.cpp\.aot-job\.sha256\.[0-9a-f]{64}$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const ABORTED_GETTER = Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const RUN_ROOT_PREFIX = "/tmp/browsergrad-cpp-cute-docker-";

/** @typedef {"BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-INVALID" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-TIMEOUT" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-CANCELLED" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDOUT-LIMIT" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDERR-LIMIT" | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-KILL"} CppCuteAotDockerProcessErrorCode */

export class CppCuteAotDockerProcessError extends Error {
  /**
   * @param {CppCuteAotDockerProcessErrorCode} code
   * @param {string} path
   * @param {string} message
   */
  constructor(code, path, message) {
    super(`${code}: ${message}`);
    this.name = "CppCuteAotDockerProcessError";
    this.code = code;
    this.path = path;
  }
}

/**
 * @typedef {Readonly<{
 *   executable: string;
 *   arguments: readonly string[];
 *   cwd: string;
 *   environment: Readonly<Record<string, string>>;
 *   timeoutMs: number;
 *   killGraceMs: number;
 *   stdoutByteLimit: number;
 *   stderrByteLimit: number;
 *   signal?: AbortSignal;
 * }>} BoundedChildProcessRequest
 */

/**
 * @typedef {Readonly<{
 *   exitCode: number | null;
 *   signal: string | null;
 *   stdout: Uint8Array;
 *   stderr: Uint8Array;
 * }>} BoundedChildProcessResult
 */

/**
 * @typedef {Readonly<{
 *   runRoot: string;
 *   configDirectory: string;
 *   homeDirectory: string;
 *   signal?: AbortSignal;
 * }>} CppCuteAotPrivateDockerRequestInput
 */

/**
 * @typedef {Readonly<{
 *   timeoutMs: number;
 *   killGraceMs: number;
 *   stdoutBytes: number;
 *   stderrBytes: number;
 * }>} CppCuteAotDockerProcessLimits
 */

/**
 * Builds the closed Docker client/engine version observation request.
 *
 * @param {CppCuteAotPrivateDockerRequestInput} input
 * @returns {BoundedChildProcessRequest}
 */
export function buildCppCuteAotDockerVersionRequest(input) {
  return buildPrivateDockerRequest(
    input,
    ["version", "--format", CPP_CUTE_AOT_DOCKER_VERSION_FORMAT],
    CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
  );
}

/**
 * Builds the closed Docker engine-info observation request.
 *
 * @param {CppCuteAotPrivateDockerRequestInput} input
 * @returns {BoundedChildProcessRequest}
 */
export function buildCppCuteAotDockerInfoRequest(input) {
  return buildPrivateDockerRequest(
    input,
    ["info", "--format", CPP_CUTE_AOT_DOCKER_INFO_FORMAT],
    CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
  );
}

/**
 * Builds the only Docker image-inspect request accepted by the production
 * shell. Every Docker-affecting value comes from the built policy or exact
 * authorized image reference; no ambient environment is inherited.
 *
 * @param {Readonly<{
 *   runRoot: string;
 *   configDirectory: string;
 *   homeDirectory: string;
 *   imageReference: string;
 *   signal?: AbortSignal;
 * }>} input
 * @returns {BoundedChildProcessRequest}
 */
export function buildCppCuteAotDockerImageInspectRequest(input) {
  const snapshot = closedDataObject(
    input,
    ["configDirectory", "homeDirectory", "imageReference", "runRoot"],
    ["signal"],
    "$input",
  );
  const context = privateDockerContext(snapshot);
  const imageReference = boundedString(snapshot.imageReference, "$input.imageReference", 512);
  if (!OCI_IMAGE_REFERENCE.test(imageReference)) invalid("$input.imageReference", "expected canonical digest-qualified OCI image reference");
  return dockerRequest(context, [
    "image", "inspect",
    `--platform=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.runtime.platform}`,
    "--format", CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_FORMAT,
    imageReference,
  ], CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS);
}

/**
 * Builds the sole policy-authorized container creation request.
 *
 * @param {Readonly<{
 *   runRoot: string;
 *   configDirectory: string;
 *   homeDirectory: string;
 *   controlDirectory: string;
 *   sourceDirectory: string;
 *   containerIdFile: string;
 *   containerName: string;
 *   sessionNonce: string;
 *   imageReference: string;
 *   jobId: string;
 *   executionPlanSha256: string;
 *   memoryBytes: number;
 *   maxProcesses: number;
 *   seccompProfilePath: string;
 *   signal?: AbortSignal;
 * }>} input
 * @returns {BoundedChildProcessRequest}
 */
export function buildCppCuteAotDockerCreateRequest(input) {
  const snapshot = closedDataObject(
    input,
    [
      "configDirectory", "containerIdFile", "containerName", "controlDirectory",
      "executionPlanSha256", "homeDirectory", "imageReference", "jobId",
      "maxProcesses", "memoryBytes", "runRoot", "sessionNonce", "sourceDirectory",
      "seccompProfilePath",
    ],
    ["signal"],
    "$input",
  );
  const context = privateDockerContext(snapshot);
  const controlDirectory = exactChildPath(
    snapshot.controlDirectory,
    context.runRoot,
    "control",
    "$input.controlDirectory",
  );
  const sourceDirectory = exactChildPath(
    snapshot.sourceDirectory,
    context.runRoot,
    "source",
    "$input.sourceDirectory",
  );
  const containerIdFile = exactChildPath(
    snapshot.containerIdFile,
    context.runRoot,
    "container.cid",
    "$input.containerIdFile",
  );
  const seccompProfilePath = exactChildPath(
    snapshot.seccompProfilePath,
    context.runRoot,
    CPP_CUTE_AOT_PRIVATE_SECCOMP_FILE,
    "$input.seccompProfilePath",
  );
  const containerName = matchingString(
    snapshot.containerName,
    "$input.containerName",
    CONTAINER_NAME,
    "expected BrowserGrad container name",
  );
  const sessionNonce = matchingString(
    snapshot.sessionNonce,
    "$input.sessionNonce",
    SESSION_NONCE,
    "expected 128-bit lowercase-hex session nonce",
  );
  const imageReference = matchingString(
    snapshot.imageReference,
    "$input.imageReference",
    OCI_IMAGE_REFERENCE,
    "expected canonical digest-qualified OCI image reference",
  );
  const jobId = matchingString(
    snapshot.jobId,
    "$input.jobId",
    JOB_ID,
    "expected prepared AOT job ID",
  );
  const executionPlanSha256 = matchingString(
    snapshot.executionPlanSha256,
    "$input.executionPlanSha256",
    SHA256_HEX,
    "expected lowercase SHA-256 execution-plan digest",
  );
  const memoryBytes = boundedPositiveInteger(
    snapshot.memoryBytes,
    "$input.memoryBytes",
    17_179_869_184,
  );
  const maxProcesses = boundedPositiveInteger(
    snapshot.maxProcesses,
    "$input.maxProcesses",
    1_024,
  );
  const temporaryBytes = Math.max(
    1,
    Math.min(Math.floor(memoryBytes / 4), 536_870_912),
  );
  return dockerRequest(context, [
    "container", "create",
    "--pull=never",
    `--platform=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.runtime.platform}`,
    `--name=${containerName}`,
    `--cidfile=${containerIdFile}`,
    `--label=browsergrad.owner=cpp-cute-aot`,
    `--label=browsergrad.session=${sessionNonce}`,
    `--label=browsergrad.job=${jobId}`,
    `--label=browsergrad.plan=${executionPlanSha256}`,
    `--hostname=${CPP_CUTE_AOT_CONTAINER_HOSTNAME}`,
    "--attach=stdout",
    "--attach=stderr",
    `--user=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.uid}:${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.gid}`,
    `--workdir=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.workingDirectory}`,
    `--entrypoint=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.entrypoint}`,
    "--network=none",
    "--ipc=none",
    "--cgroupns=private",
    "--read-only",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges=true",
    `--security-opt=seccomp=${seccompProfilePath}`,
    `--memory=${memoryBytes}`,
    `--memory-swap=${memoryBytes}`,
    `--pids-limit=${maxProcesses}`,
    "--restart=no",
    "--log-driver=none",
    "--no-healthcheck",
    `--runtime=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.runtime}`,
    "--mount", `type=bind,source=${sourceDirectory},target=${CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT},readonly,bind-propagation=rprivate`,
    "--mount", `type=bind,source=${controlDirectory},target=${CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT},readonly,bind-propagation=rprivate`,
    `--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=${temporaryBytes},mode=1777`,
    imageReference,
    ...CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments,
  ], CPP_CUTE_AOT_DOCKER_CREATE_LIMITS);
}

/** @param {Readonly<CppCuteAotPrivateDockerRequestInput & { containerId: string }>} input */
export function buildCppCuteAotDockerContainerInspectRequest(input) {
  return buildContainerReferenceRequest(
    input,
    "containerId",
    CONTAINER_ID,
    ["container", "inspect", "--format", CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_FORMAT],
    CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
  );
}

/** @param {Readonly<CppCuteAotPrivateDockerRequestInput & { containerName: string }>} input */
export function buildCppCuteAotDockerContainerRecoveryRequest(input) {
  return buildContainerReferenceRequest(
    input,
    "containerName",
    CONTAINER_NAME,
    ["container", "inspect", "--format", CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_FORMAT],
    CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
  );
}

/**
 * @param {Readonly<CppCuteAotPrivateDockerRequestInput & {
 *   containerId: string;
 *   timeoutMs: number;
 *   stdoutByteLimit: number;
 * }>} input
 */
export function buildCppCuteAotDockerStartAttachedRequest(input) {
  const snapshot = closedDataObject(
    input,
    [
      "configDirectory", "containerId", "homeDirectory", "runRoot",
      "stdoutByteLimit", "timeoutMs",
    ],
    ["signal"],
    "$input",
  );
  const context = privateDockerContext(snapshot);
  const containerId = matchingString(
    snapshot.containerId,
    "$input.containerId",
    CONTAINER_ID,
    "expected full lowercase Docker container ID",
  );
  const timeoutMs = boundedPositiveInteger(snapshot.timeoutMs, "$input.timeoutMs", 1_800_000);
  const stdoutBytes = boundedPositiveInteger(
    snapshot.stdoutByteLimit,
    "$input.stdoutByteLimit",
    CPP_CUTE_AOT_DOCKER_START_LIMITS.stdoutBytes,
  );
  return dockerRequest(context, [
    "container", "start", "--attach", containerId,
  ], {
    timeoutMs,
    killGraceMs: CPP_CUTE_AOT_DOCKER_START_LIMITS.killGraceMs,
    stdoutBytes,
    stderrBytes: CPP_CUTE_AOT_DOCKER_START_LIMITS.stderrBytes,
  });
}

/**
 * @param {Readonly<CppCuteAotPrivateDockerRequestInput & {
 *   containerId: string;
 *   force: boolean;
 * }>} input
 */
export function buildCppCuteAotDockerRemoveRequest(input) {
  const snapshot = closedDataObject(
    input,
    ["configDirectory", "containerId", "force", "homeDirectory", "runRoot"],
    [],
    "$input",
  );
  const context = privateDockerContext(snapshot);
  const containerId = matchingString(
    snapshot.containerId,
    "$input.containerId",
    CONTAINER_ID,
    "expected full lowercase Docker container ID",
  );
  if (typeof snapshot.force !== "boolean") invalid("$input.force", "force must be a boolean");
  return dockerRequest(context, [
    "container", "rm",
    ...(snapshot.force ? ["--force"] : []),
    "--volumes",
    containerId,
  ], CPP_CUTE_AOT_DOCKER_REMOVE_LIMITS);
}

/** @param {Readonly<CppCuteAotPrivateDockerRequestInput & { containerId: string }>} input */
export function buildCppCuteAotDockerAbsenceRequest(input) {
  return buildContainerReferenceRequest(
    input,
    "containerId",
    CONTAINER_ID,
    ["container", "ls", "--all", "--quiet", "--no-trunc", "--filter"],
    CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS,
    (containerId) => `id=${containerId}`,
  );
}

/**
 * @param {unknown} input
 * @param {string} referenceField
 * @param {RegExp} referencePattern
 * @param {readonly string[]} operationArguments
 * @param {CppCuteAotDockerProcessLimits} limits
 * @param {(reference: string) => string} [renderReference]
 * @returns {BoundedChildProcessRequest}
 */
function buildContainerReferenceRequest(
  input,
  referenceField,
  referencePattern,
  operationArguments,
  limits,
  renderReference = (reference) => reference,
) {
  const snapshot = closedDataObject(
    input,
    ["configDirectory", "homeDirectory", referenceField, "runRoot"],
    ["signal"],
    "$input",
  );
  const context = privateDockerContext(snapshot);
  const reference = matchingString(
    snapshot[referenceField],
    `$input.${referenceField}`,
    referencePattern,
    "expected canonical Docker container reference",
  );
  return dockerRequest(
    context,
    [...operationArguments, renderReference(reference)],
    limits,
  );
}

/**
 * Bounded binary process primitive. Production callers pass only requests
 * built above. Export exists for direct adversarial process-contract tests;
 * package publication excludes this scripts directory.
 *
 * @param {BoundedChildProcessRequest} request
 * @returns {Promise<BoundedChildProcessResult>}
 */
export function runBoundedChildProcess(request) {
  const normalized = normalizeRequest(request);
  return new Promise((resolvePromise, rejectPromise) => {
    /** @type {import("node:child_process").ChildProcess | undefined} */
    let child;
    /** @type {NodeJS.Timeout | undefined} */
    let deadline;
    /** @type {NodeJS.Timeout | undefined} */
    let killDeadline;
    /** @type {CppCuteAotDockerProcessError | undefined} */
    let pendingFailure;
    /** @type {Uint8Array[]} */
    const stdoutChunks = [];
    /** @type {Uint8Array[]} */
    const stderrChunks = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let settled = false;
    let spawnObserved = false;
    let closeObserved = false;
    let terminationStarted = false;

    const cleanup = () => {
      if (deadline !== undefined) clearTimeout(deadline);
      if (killDeadline !== undefined) clearTimeout(killDeadline);
      normalized.signal?.removeEventListener("abort", onAbort);
    };
    /** @param {CppCuteAotDockerProcessError} error */
    const reject = (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      rejectPromise(error);
    };
    /** @param {BoundedChildProcessResult} result */
    const resolve = (result) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolvePromise(result);
    };
    /** @param {CppCuteAotDockerProcessError} error @param {boolean} [stronger] */
    const recordFailure = (error, stronger = false) => {
      if (pendingFailure === undefined || stronger) pendingFailure = error;
    };
    /** @param {CppCuteAotDockerProcessError} error */
    const terminate = (error) => {
      if (settled) return;
      recordFailure(error);
      if (child === undefined) {
        reject(pendingFailure ?? error);
        return;
      }
      if (terminationStarted) return;
      terminationStarted = true;
      if (!killChildProcessGroup(child)) {
        recordFailure(processError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-KILL",
          "$process",
          "failed to terminate bounded child process group",
        ), true);
      }
      if (closeObserved || settled) return;
      killDeadline = setTimeout(() => failStopUnreapedChild(), normalized.killGraceMs);
    };
    function onAbort() {
      terminate(processError(
        "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-CANCELLED",
        "$request.signal",
        "bounded child process was aborted",
      ));
    }

    if (normalized.signal !== undefined && signalAborted(normalized.signal)) {
      reject(processError(
        "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-CANCELLED",
        "$request.signal",
        "bounded child process was aborted",
      ));
      return;
    }
    if (normalized.signal !== undefined) {
      normalized.signal.addEventListener("abort", onAbort, { once: true });
      if (signalAborted(normalized.signal)) onAbort();
      if (settled) return;
    }

    try {
      child = spawn(normalized.executable, [...normalized.arguments], {
        cwd: normalized.cwd,
        env: { ...normalized.environment },
        detached: process.platform === "linux",
        shell: false,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch {
      reject(processError(
        "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN",
        "$request.executable",
        "failed to spawn bounded child process",
      ));
      return;
    }
    const activeChild = child;
    if (activeChild === undefined) {
      reject(processError(
        "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN",
        "$request.executable",
        "bounded child process was not created",
      ));
      return;
    }
    spawnObserved = typeof activeChild.pid === "number";
    activeChild.once("spawn", () => {
      spawnObserved = true;
    });
    activeChild.once("error", () => {
      const error = processError(
        "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN",
        "$request.executable",
        "bounded child process failed before completion",
      );
      if (!spawnObserved && activeChild.pid === undefined) {
        recordFailure(error);
        reject(pendingFailure ?? error);
        return;
      }
      terminate(error);
    });
    activeChild.once("close", (exitCode, exitSignal) => {
      closeObserved = true;
      if (!spawnObserved && activeChild.pid === undefined && pendingFailure === undefined) {
        recordFailure(processError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN",
          "$request.executable",
          "bounded child process closed without a successful spawn",
        ));
      }
      if (pendingFailure !== undefined) {
        reject(pendingFailure);
        return;
      }
      resolve(Object.freeze({
        exitCode,
        signal: exitSignal,
        stdout: concatenate(stdoutChunks, stdoutBytes),
        stderr: concatenate(stderrChunks, stderrBytes),
      }));
    });
    const stdout = activeChild.stdout;
    const stderr = activeChild.stderr;
    if (stdout === null || stderr === null) {
      terminate(processError(
        "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN",
        "$request.executable",
        "bounded child process did not expose piped output",
      ));
      return;
    }

    deadline = setTimeout(() => terminate(processError(
      "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-TIMEOUT",
      "$request.timeoutMs",
      "bounded child process exceeded its deadline",
    )), normalized.timeoutMs);
    deadline.unref();
    if (normalized.signal !== undefined && signalAborted(normalized.signal)) onAbort();

    stdout.on("data", (chunk) => {
      if (!(chunk instanceof Uint8Array)) {
        terminate(processError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-INVALID",
          "$process.stdout",
          "bounded child emitted a non-byte stdout chunk",
        ));
        return;
      }
      if (pendingFailure !== undefined) return;
      stdoutBytes += chunk.byteLength;
      if (stdoutBytes > normalized.stdoutByteLimit) {
        terminate(processError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDOUT-LIMIT",
          "$process.stdout",
          "bounded child stdout exceeded policy",
        ));
        return;
      }
      stdoutChunks.push(Uint8Array.from(chunk));
    });
    stderr.on("data", (chunk) => {
      if (!(chunk instanceof Uint8Array)) {
        terminate(processError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-INVALID",
          "$process.stderr",
          "bounded child emitted a non-byte stderr chunk",
        ));
        return;
      }
      if (pendingFailure !== undefined) return;
      stderrBytes += chunk.byteLength;
      if (stderrBytes > normalized.stderrByteLimit) {
        terminate(processError(
          "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDERR-LIMIT",
          "$process.stderr",
          "bounded child stderr exceeded policy",
        ));
        return;
      }
      stderrChunks.push(Uint8Array.from(chunk));
    });
  });
}

/**
 * @param {CppCuteAotPrivateDockerRequestInput} input
 * @param {readonly string[]} operationArguments
 * @param {CppCuteAotDockerProcessLimits} limits
 * @returns {BoundedChildProcessRequest}
 */
function buildPrivateDockerRequest(input, operationArguments, limits) {
  const snapshot = closedDataObject(
    input,
    ["configDirectory", "homeDirectory", "runRoot"],
    ["signal"],
    "$input",
  );
  return dockerRequest(privateDockerContext(snapshot), operationArguments, limits);
}

/**
 * @param {Record<string, unknown>} input
 * @returns {Readonly<{
 *   runRoot: string;
 *   configDirectory: string;
 *   environment: Readonly<Record<string, string>>;
 *   signal?: AbortSignal;
 * }>}
 */
function privateDockerContext(input) {
  const runRoot = privateRunRoot(input.runRoot, "$input.runRoot");
  const configDirectory = exactChildPath(input.configDirectory, runRoot, "docker-config", "$input.configDirectory");
  const homeDirectory = exactChildPath(input.homeDirectory, runRoot, "home", "$input.homeDirectory");
  const signal = optionalAbortSignal(input.signal, "$input.signal");
  const environment = Object.freeze(Object.assign(Object.create(null), {
    DOCKER_API_VERSION: CPP_CUTE_AOT_DOCKER_API_VERSION,
    HOME: homeDirectory,
    LANG: "C",
    LC_ALL: "C",
    TZ: "UTC",
  }));
  return Object.freeze({
    runRoot,
    configDirectory,
    environment,
    ...(signal === undefined ? {} : { signal }),
  });
}

/**
 * @param {Readonly<{
 *   runRoot: string;
 *   configDirectory: string;
 *   environment: Readonly<Record<string, string>>;
 *   signal?: AbortSignal;
 * }>} context
 * @param {readonly string[]} operationArguments
 * @param {CppCuteAotDockerProcessLimits} limits
 * @returns {BoundedChildProcessRequest}
 */
function dockerRequest(context, operationArguments, limits) {
  return Object.freeze({
    executable: CPP_CUTE_AOT_DOCKER_EXECUTABLE,
    arguments: Object.freeze([
      "--config", context.configDirectory,
      `--host=${CPP_CUTE_AOT_SANDBOX_POLICY_V1.runtime.endpointUri}`,
      "--log-level=error",
      ...operationArguments,
    ]),
    cwd: context.runRoot,
    environment: context.environment,
    timeoutMs: limits.timeoutMs,
    killGraceMs: limits.killGraceMs,
    stdoutByteLimit: limits.stdoutBytes,
    stderrByteLimit: limits.stderrBytes,
    ...(context.signal === undefined ? {} : { signal: context.signal }),
  });
}

/** @param {import("node:child_process").ChildProcess} child @returns {boolean} */
function killChildProcessGroup(child) {
  const pid = child.pid;
  if (process.platform === "linux" && typeof pid === "number") {
    try {
      process.kill(-pid, "SIGKILL");
      return true;
    } catch {
      // Fall back to the direct child. The grace deadline remains fail-closed
      // if a descendant retained either output pipe.
    }
  }
  try {
    return child.kill("SIGKILL");
  } catch {
    return false;
  }
}

/** @returns {never} */
function failStopUnreapedChild() {
  // Returning or ordinarily rejecting here would let callers clean staging
  // state while an untrusted process could still be using it.
  process.abort();
}

/** @param {BoundedChildProcessRequest} request @returns {BoundedChildProcessRequest} */
function normalizeRequest(request) {
  const snapshot = closedDataObject(
    request,
    [
      "arguments", "cwd", "environment", "executable", "killGraceMs",
      "stderrByteLimit", "stdoutByteLimit", "timeoutMs",
    ],
    ["signal"],
    "$request",
  );
  const executable = absolutePath(snapshot.executable, "$request.executable");
  const rawArguments = snapshotDenseArray(snapshot.arguments, "$request.arguments", 128);
  const args = Object.freeze(rawArguments.map((arg, index) => boundedString(
    arg,
    `$request.arguments[${index}]`,
    262_144,
    true,
  )));
  const cwd = absolutePath(snapshot.cwd, "$request.cwd");
  const environment = normalizeEnvironment(snapshot.environment);
  const timeoutMs = boundedPositiveInteger(snapshot.timeoutMs, "$request.timeoutMs", 1_800_000);
  const killGraceMs = boundedPositiveInteger(snapshot.killGraceMs, "$request.killGraceMs", 10_000);
  const stdoutByteLimit = boundedPositiveInteger(
    snapshot.stdoutByteLimit,
    "$request.stdoutByteLimit",
    CPP_CUTE_AOT_HARD_FRAME_BYTE_LIMIT,
  );
  const stderrByteLimit = boundedPositiveInteger(snapshot.stderrByteLimit, "$request.stderrByteLimit", 67_108_864);
  const signal = optionalAbortSignal(snapshot.signal, "$request.signal");
  return Object.freeze({
    executable,
    arguments: args,
    cwd,
    environment,
    timeoutMs,
    killGraceMs,
    stdoutByteLimit,
    stderrByteLimit,
    ...(signal === undefined ? {} : { signal }),
  });
}

/** @param {unknown} value @returns {Readonly<Record<string, string>>} */
function normalizeEnvironment(value) {
  if (!isPlainOrNullObject(value)) invalid("$request.environment", "environment must be a plain data object");
  /** @type {Record<PropertyKey, PropertyDescriptor>} */
  let descriptors;
  try {
    descriptors = /** @type {Record<PropertyKey, PropertyDescriptor>} */ (
      /** @type {unknown} */ (Object.getOwnPropertyDescriptors(value))
    );
  } catch {
    invalid("$request.environment", "environment must be safely inspectable");
  }
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length > 32 || keys.some((key) => typeof key !== "string" || !/^[A-Z_][A-Z0-9_]*$/u.test(key))) {
    invalid("$request.environment", "environment names are invalid or exceed policy");
  }
  const environment = Object.create(null);
  for (const key of keys) {
    if (typeof key !== "string") continue;
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`$request.environment.${key}`, "environment fields must be enumerable data properties");
    }
    environment[key] = boundedString(descriptor.value, `$request.environment.${key}`, 8_192, true);
  }
  return Object.freeze(environment);
}

/**
 * @param {unknown} value
 * @param {readonly string[]} required
 * @param {readonly string[]} optional
 * @param {string} path
 * @returns {Record<string, unknown>}
 */
function closedDataObject(value, required, optional, path) {
  if (!isPlainObject(value)) invalid(path, "expected a plain data object");
  let descriptors;
  try {
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch {
    invalid(path, "data object must be safely inspectable");
  }
  const allowed = new Set([...required, ...optional]);
  const keys = Reflect.ownKeys(descriptors);
  if (
    keys.some((key) => typeof key !== "string" || !allowed.has(key))
    || required.some((key) => !(key in descriptors))
  ) {
    invalid(path, "data object fields differ from the closed schema");
  }
  const result = Object.create(null);
  for (const key of keys) {
    if (typeof key !== "string") continue;
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "fields must be enumerable data properties");
    }
    result[key] = descriptor.value;
  }
  return result;
}

/** @param {unknown} value @param {string} path @param {number} max @returns {unknown[]} */
function snapshotDenseArray(value, path, max) {
  try {
    if (!Array.isArray(value)) invalid(path, "expected a dense array");
  } catch (error) {
    if (error instanceof CppCuteAotDockerProcessError) throw error;
    invalid(path, "array must be safely inspectable");
  }
  /** @type {Record<PropertyKey, PropertyDescriptor>} */
  let descriptors;
  try {
    descriptors = /** @type {Record<PropertyKey, PropertyDescriptor>} */ (
      /** @type {unknown} */ (Object.getOwnPropertyDescriptors(value))
    );
  } catch {
    invalid(path, "array must be safely inspectable");
  }
  const lengthDescriptor = descriptors.length;
  if (
    lengthDescriptor === undefined
    || !("value" in lengthDescriptor)
    || typeof lengthDescriptor.value !== "number"
    || !Number.isSafeInteger(lengthDescriptor.value)
    || lengthDescriptor.value < 0
    || lengthDescriptor.value > max
  ) {
    invalid(path, `array length must be at most ${max}`);
  }
  const length = lengthDescriptor.value;
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length !== length + 1) invalid(path, "array must be dense and contain no extra fields");
  const result = [];
  for (let index = 0; index < length; index += 1) {
    const descriptor = descriptors[String(index)];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}[${index}]`, "array entries must be enumerable data properties");
    }
    result.push(descriptor.value);
  }
  return result;
}

/** @param {readonly Uint8Array[]} chunks @param {number} total */
function concatenate(chunks, total) {
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

/** @param {unknown} value @param {string} path @returns {AbortSignal | undefined} */
function optionalAbortSignal(value, path) {
  if (value === undefined) return undefined;
  const getter = ABORTED_GETTER;
  if (getter === undefined) invalid(path, "AbortSignal is unavailable");
  try {
    getter.call(value);
  } catch {
    invalid(path, "expected AbortSignal");
  }
  return /** @type {AbortSignal} */ (value);
}

/** @param {AbortSignal} signal @returns {boolean} */
function signalAborted(signal) {
  try {
    return ABORTED_GETTER?.call(signal) === true;
  } catch {
    invalid("$request.signal", "AbortSignal became unreadable");
  }
}

/** @param {unknown} value @param {string} path @returns {string} */
function privateRunRoot(value, path) {
  const result = absolutePath(value, path);
  if (!result.startsWith(RUN_ROOT_PREFIX) || result === RUN_ROOT_PREFIX) {
    invalid(path, "run root must be a private BrowserGrad Docker directory under /tmp");
  }
  return result;
}

/** @param {unknown} value @param {string} root @param {string} child @param {string} path @returns {string} */
function exactChildPath(value, root, child, path) {
  const result = absolutePath(value, path);
  if (result !== join(root, child)) invalid(path, `path must be the private ${child} directory`);
  return result;
}

/** @param {unknown} value @param {string} path @returns {string} */
function absolutePath(value, path) {
  const result = boundedString(value, path, 4_096);
  if (!isAbsolute(result) || resolve(result) !== result) invalid(path, "expected normalized absolute path");
  return result;
}

/** @param {unknown} value @param {string} path @param {number} max @param {boolean} [allowEmpty] @returns {string} */
function boundedString(value, path, max, allowEmpty = false) {
  if (typeof value !== "string" || value.includes("\0") || (!allowEmpty && value.length === 0) || Buffer.byteLength(value, "utf8") > max) {
    invalid(path, "expected bounded UTF-8 string without NUL");
  }
  return value;
}

/** @param {unknown} value @param {string} path @param {RegExp} pattern @param {string} message @returns {string} */
function matchingString(value, path, pattern, message) {
  const result = boundedString(value, path, 4_096);
  if (!pattern.test(result)) invalid(path, message);
  return result;
}

/** @param {unknown} value @param {string} path @param {number} max @returns {number} */
function boundedPositiveInteger(value, path, max) {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0 || value > max) {
    invalid(path, `expected positive safe integer at most ${max}`);
  }
  return value;
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

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function isPlainOrNullObject(value) {
  if (typeof value !== "object" || value === null) return false;
  try {
    const prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
  } catch {
    return false;
  }
}

/** @param {CppCuteAotDockerProcessErrorCode} code @param {string} path @param {string} message @returns {CppCuteAotDockerProcessError} */
function processError(code, path, message) {
  return new CppCuteAotDockerProcessError(code, path, message);
}

/** @param {string} path @param {string} message @returns {never} */
function invalid(path, message) {
  throw processError("BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-INVALID", path, message);
}
