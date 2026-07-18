import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import { dirname, join } from "node:path/posix";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

const INVALID =
  "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-INVALID";
const CONFLICT =
  "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-CONFLICT";
const IO = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-IO";
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const LOCK_ID = /^bg\.cpp\.browser-build-input-lock\.sha256\.[0-9a-f]{64}$/u;
const MAX_LOG_BYTE_LENGTH = 16 * 1024 * 1024;
const MAX_FAILURE_STRING_LENGTH = 4_096;
const MAX_FAILURE_CAUSE_DEPTH = 4;
const STEP_IDS = Object.freeze([
  "native-tablegen-configure",
  "native-tablegen-build",
  "clang-extractor-wasm-configure",
  "clang-extractor-wasm-build",
]);

export class CppCuteBrowserBuildFailureObservationError extends Error {
  /** @param {string} code @param {string} path @param {string} message @param {ErrorOptions} [options] */
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildFailureObservationError";
    this.code = code;
    this.path = path;
  }
}

/**
 * Persists a failure-only receipt after the runner has verified the lock,
 * builder, isolation, and source archive. Partial logs remain the underlying
 * evidence; this receipt binds their bytes to one typed failure without
 * granting any successful-build authority.
 *
 * @param {Readonly<{
 *   outputRoot: string;
 *   stateRoot: string;
 *   lockId: string;
 *   sourceSetSha256: string;
 *   cause: unknown;
 * }>} input
 */
export async function persistCppCuteBrowserBuildFailureObservation(input) {
  const lockId = pattern(input.lockId, LOCK_ID, "$.lockId");
  const sourceSetSha256 = pattern(
    input.sourceSetSha256,
    SHA256_HEX,
    "$.sourceSetSha256",
  );
  await admitPrivateDirectory(input.outputRoot, "$.outputRoot");
  const logs = await collectPartialLogs(input.stateRoot);
  const failure = projectCppCuteBrowserBuildFailure(input.cause);
  const observation = Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.clang-wasm-build-failure-observation",
    version: 2,
    authority: "build-failure-observation-only",
    lockId,
    sourceSetSha256,
    failure,
    partialLogs: Object.freeze(logs),
    claims: Object.freeze({
      successfulBuildReceiptWritten: false,
      outputIdentityAuthorized: false,
      abiConformanceVerified: false,
      reproducibilityVerified: false,
      producerAttested: false,
      releaseReady: false,
    }),
  });
  const bytes = canonicalJsonBytes(observation);
  const outputPath = join(
    input.outputRoot,
    "build-failure-observation.v2.json",
  );
  await writeExclusiveReadOnlyFile(outputPath, bytes);
  return Object.freeze({
    outputPath,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    byteLength: bytes.byteLength,
    partialLogCount: logs.length,
    successfulBuildReceiptWritten: false,
  });
}

/** @param {string} stateRoot */
async function collectPartialLogs(stateRoot) {
  const logRoot = join(stateRoot, "evidence", "build-logs");
  const logs = [];
  for (const stepId of STEP_IDS) {
    for (const stream of ["stdout", "stderr"]) {
      const fileName = `${stepId}.${stream}.log`;
      const observed = await readOptionalBoundedFile(
        join(logRoot, fileName),
        `$.partialLogs.${fileName}`,
      );
      if (observed !== undefined) {
        logs.push(Object.freeze({
          stepId,
          stream,
          relativePath: `state/evidence/build-logs/${fileName}`,
          sha256: observed.sha256,
          byteLength: observed.byteLength,
        }));
      }
    }
  }
  return logs;
}

/** @param {string} path @param {string} diagnosticPath */
async function readOptionalBoundedFile(path, diagnosticPath) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) return undefined;
    if (isNodeError(cause, "ELOOP")) {
      invalid(diagnosticPath, "partial build log must not be a symbolic link", { cause });
    }
    io(diagnosticPath, "failed to open partial build log", { cause });
  }
  try {
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size < 0n ||
        before.size > BigInt(MAX_LOG_BYTE_LENGTH)) {
      invalid(diagnosticPath, "partial build log is not a bounded regular file");
    }
    const hash = createHash("sha256");
    const buffer = new Uint8Array(1024 * 1024);
    let offset = 0n;
    while (offset < before.size) {
      const remaining = before.size - offset;
      const length = Number(
        remaining > BigInt(buffer.byteLength)
          ? BigInt(buffer.byteLength)
          : remaining,
      );
      const read = await handle.read(buffer, 0, length, Number(offset));
      if (read.bytesRead === 0) conflict(diagnosticPath, "partial build log became shorter while hashing");
      hash.update(buffer.subarray(0, read.bytesRead));
      offset += BigInt(read.bytesRead);
    }
    const after = await handle.stat({ bigint: true });
    if (!sameSnapshot(before, after)) {
      conflict(diagnosticPath, "partial build log changed while hashing");
    }
    return Object.freeze({
      sha256: hash.digest("hex"),
      byteLength: Number(after.size),
    });
  } finally {
    await handle.close();
  }
}

/** @param {unknown} cause */
export function projectCppCuteBrowserBuildFailure(cause) {
  const root = singleFailureProjection(cause);
  if (!(cause instanceof Error)) {
    return Object.freeze({
      ...root,
      causes: Object.freeze([]),
      causeChainComplete: true,
    });
  }
  const causes = [];
  const seen = new WeakSet([cause]);
  let current = cause;
  let causeChainComplete = true;
  while (true) {
    const descriptor = Object.getOwnPropertyDescriptor(current, "cause");
    if (descriptor === undefined || ("value" in descriptor && descriptor.value === undefined)) break;
    if (!("value" in descriptor) || !(descriptor.value instanceof Error) ||
        seen.has(descriptor.value) || causes.length === MAX_FAILURE_CAUSE_DEPTH) {
      causeChainComplete = false;
      break;
    }
    current = descriptor.value;
    seen.add(current);
    causes.push(singleFailureProjection(current));
  }
  return Object.freeze({
    ...root,
    causes: Object.freeze(causes),
    causeChainComplete,
  });
}

/** @param {unknown} cause */
function singleFailureProjection(cause) {
  if (!(cause instanceof Error)) {
    return Object.freeze({
      name: "Error",
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-UNKNOWN",
      path: "$build",
      message: "unknown build failure",
    });
  }
  const descriptors = Object.getOwnPropertyDescriptors(cause);
  return Object.freeze({
    name: boundedString(descriptors["name"]?.value, "Error"),
    code: boundedString(
      descriptors["code"]?.value,
      "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-UNKNOWN",
    ),
    path: boundedString(descriptors["path"]?.value, "$build"),
    message: boundedString(descriptors["message"]?.value, "build failed"),
  });
}

/** @param {unknown} value @param {string} fallback */
function boundedString(value, fallback) {
  return typeof value === "string" && value.length > 0
    ? value.slice(0, MAX_FAILURE_STRING_LENGTH)
    : fallback;
}

/** @param {string} path @param {string} diagnosticPath */
async function admitPrivateDirectory(path, diagnosticPath) {
  let stat;
  try {
    stat = await lstat(path, { bigint: true });
  } catch (cause) {
    io(diagnosticPath, "failed to inspect failure-observation directory", { cause });
  }
  if (!stat.isDirectory() || stat.isSymbolicLink() ||
      typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid()) ||
      (stat.mode & 0o022n) !== 0n) {
    invalid(diagnosticPath, "failure-observation directory must be private and current-user-owned");
  }
}

/** @param {string} path @param {Uint8Array} bytes */
async function writeExclusiveReadOnlyFile(path, bytes) {
  let handle;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.chmod(0o444);
  } catch (cause) {
    if (isNodeError(cause, "EEXIST") || isNodeError(cause, "ELOOP")) {
      conflict("$.output", "failure observation output must not already exist", { cause });
    }
    io("$.output", "failed to persist failure observation", { cause });
  } finally {
    await handle?.close();
  }
  const directory = await open(dirname(path), constants.O_RDONLY | constants.O_DIRECTORY);
  try {
    await directory.sync();
  } finally {
    await directory.close();
  }
}

/** @param {import("node:fs").BigIntStats} left @param {import("node:fs").BigIntStats} right */
function sameSnapshot(left, right) {
  return left.dev === right.dev && left.ino === right.ino &&
    left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

/** @param {unknown} value @param {RegExp} expression @param {string} path */
function pattern(value, expression, path) {
  if (typeof value !== "string" || !expression.test(value)) {
    invalid(path, "value does not match the required closed pattern");
  }
  return value;
}

/** @param {unknown} value @param {string} code */
function isNodeError(value, code) {
  return typeof value === "object" && value !== null && "code" in value && value.code === code;
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] */
function invalid(path, message, options) {
  throw new CppCuteBrowserBuildFailureObservationError(INVALID, path, message, options);
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] */
function conflict(path, message, options) {
  throw new CppCuteBrowserBuildFailureObservationError(CONFLICT, path, message, options);
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] */
function io(path, message, options) {
  throw new CppCuteBrowserBuildFailureObservationError(IO, path, message, options);
}
