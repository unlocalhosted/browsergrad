import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, opendir, realpath, rmdir, unlink } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path/posix";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

export const CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_FILES_SCHEMA =
  "browsergrad.compiler.cpp-cute.distribution-output-file-materialization";
export const CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_VERIFICATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.distribution-output-file-verification";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-OUTPUT-FILES";
const MATERIALIZATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.distribution-output-file-materialization.v1";
const VERIFICATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.distribution-output-file-verification.v1";
const SHA256 = /^[0-9a-f]{64}$/u;
const WIRE_U64 = /^(0|[1-9][0-9]*)$/u;
const PORTABLE_SEGMENT = /^[A-Za-z0-9._+@=-]+$/u;
const MAX_EXISTING_FILE_BYTES = 128 * 1024 * 1024;
const MAX_EXISTING_TOTAL_BYTES = 512 * 1024 * 1024;
const MAX_NEW_FILE_BYTES = 32 * 1024 * 1024;
const MAX_NEW_TOTAL_BYTES = 64 * 1024 * 1024;
const MAX_TREE_ENTRIES = 256;
const MAX_TREE_DEPTH = 16;
const MAX_PATH_BYTES = 4_096;
const READ_BUFFER_BYTES = 256 * 1024;

export class CppCuteBrowserDistributionOutputFilesError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserDistributionOutputFilesError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Rehashes every expected immutable file while proving the complete private
 * file and directory tree stayed exact before and after verification. Caller
 * expectations grant no policy, provenance, approval, or release authority.
 */
export async function verifyCppCuteBrowserDistributionOutputFiles(input) {
  const object = exactObject(input, ["expectedOutputs", "outputRoot"], "$.input");
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  const expectedOutputs = parseExistingOutputs(
    object.expectedOutputs,
    "$.input.expectedOutputs",
  );
  await verifyExpectedOutputTree(
    outputRoot,
    expectedOutputs,
    "$.input.expectedOutputs",
  );
  const verificationHash = sha256(canonicalJsonBytes({
    domain: VERIFICATION_HASH_DOMAIN,
    outputs: expectedOutputs,
  }));
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_VERIFICATION_SCHEMA,
    version: 1,
    verificationId: `bg.cpp.distribution-output-file-verification.sha256.${verificationHash}`,
    authority: "caller-expected-private-distribution-output-verification-only",
    outputRoot,
    outputs: Object.freeze(expectedOutputs),
    totals: Object.freeze({
      fileCount: expectedOutputs.length,
      byteLength: expectedOutputs.reduce(
        (total, output) => total + BigInt(output.byteLength),
        0n,
      ).toString(),
    }),
    claims: Object.freeze({
      exactTreeVerifiedBeforeAndAfter: true,
      exactFileBytesReverified: true,
      callerPolicyBound: false,
      reproducibilityVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
}

/**
 * Verifies one exact private output tree, transactionally adds a bounded set of
 * caller-selected immutable files, independently rereads them, and verifies
 * the final exact tree. Caller expectations grant no license, provenance,
 * reproducibility, asset, or release authority.
 */
export async function materializeCppCuteBrowserDistributionOutputFiles(input) {
  const object = exactObject(input, ["existingOutputs", "outputRoot", "outputs"], "$.input");
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  const existingOutputs = parseExistingOutputs(
    object.existingOutputs,
    "$.input.existingOutputs",
  );
  const outputs = parseNewOutputs(object.outputs);
  validateCombinedPaths(existingOutputs, outputs);
  const initialVerification = await verifyExpectedOutputTree(
    outputRoot,
    existingOutputs,
    "$.input.existingOutputs",
  );
  const rootIdentity = initialVerification.rootIdentity;
  const createdDirectories = [];
  const createdFiles = [];
  try {
    for (const output of outputs) {
      await createPrivateParents(outputRoot, output.outputPath, createdDirectories);
      const absoluteOutputPath = join(outputRoot, output.outputPath);
      const identity = await writeExclusiveFile(absoluteOutputPath, output.bytes, output.outputPath);
      createdFiles.push(Object.freeze({ path: absoluteOutputPath, identity }));
      const persisted = await readExactFile(
        absoluteOutputPath,
        output.bytes.byteLength,
        identity,
        `$.input.outputs.${output.outputPath}`,
      );
      const digest = sha256(persisted);
      if (digest !== output.sha256 || !sameBytes(persisted, output.bytes)) {
        invalid(`$.input.outputs.${output.outputPath}`, "persisted bytes differ from the input snapshot");
      }
    }
    await syncAffectedDirectories(outputRoot, outputs);
    const finalOutputs = [
      ...existingOutputs,
      ...outputs.map((output) => Object.freeze({
        outputPath: output.outputPath,
        sha256: output.sha256,
        byteLength: String(output.bytes.byteLength),
      })),
    ].sort((left, right) => compareUtf8(left.outputPath, right.outputPath));
    const finalVerification = await verifyExpectedOutputTree(
      outputRoot,
      finalOutputs,
      "$.output.final",
    );
    if (!sameDirectoryIdentity(rootIdentity, finalVerification.rootIdentity)) {
      invalid("$.input.outputRoot", "output root identity changed during materialization");
    }
  } catch (cause) {
    await cleanupCreated(createdFiles, createdDirectories, cause);
    if (cause instanceof CppCuteBrowserDistributionOutputFilesError) throw cause;
    invalid("$.outputs", "distribution output materialization failed", { cause });
  }
  const materializedOutputs = Object.freeze(outputs.map((output, ordinal) => Object.freeze({
    ordinal,
    outputPath: output.outputPath,
    sha256: output.sha256,
    byteLength: String(output.bytes.byteLength),
  })));
  const materializationHash = sha256(canonicalJsonBytes({
    domain: MATERIALIZATION_HASH_DOMAIN,
    existingOutputs,
    outputs: materializedOutputs,
  }));
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_OUTPUT_FILES_SCHEMA,
    version: 1,
    materializationId:
      `bg.cpp.distribution-output-file-materialization.sha256.${materializationHash}`,
    authority: "caller-expected-private-distribution-output-materialization-only",
    outputRoot,
    existingOutputs: Object.freeze(existingOutputs),
    outputs: materializedOutputs,
    totals: Object.freeze({
      existingFileCount: existingOutputs.length,
      existingByteLength: existingOutputs.reduce(
        (total, output) => total + BigInt(output.byteLength),
        0n,
      ).toString(),
      materializedFileCount: outputs.length,
      materializedByteLength: outputs.reduce(
        (total, output) => total + BigInt(output.bytes.byteLength),
        0n,
      ).toString(),
    }),
    claims: Object.freeze({
      exactInitialTreeVerified: true,
      exactExistingFileBytesReverified: true,
      newFilesWrittenWithoutClobber: true,
      newFilesIndependentlyReread: true,
      exactFinalTreeVerified: true,
      callerPolicyBound: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
}

function parseExistingOutputs(value, diagnosticPath) {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_TREE_ENTRIES) {
    invalid(diagnosticPath, "expected one bounded nonempty output list");
  }
  let total = 0;
  const outputs = value.map((item, index) => {
    const path = `${diagnosticPath}[${index}]`;
    const object = exactObject(item, ["byteLength", "outputPath", "sha256"], path);
    const outputPath = portablePath(object.outputPath, `${path}.outputPath`);
    if (typeof object.sha256 !== "string" || !SHA256.test(object.sha256) ||
        typeof object.byteLength !== "string" || !WIRE_U64.test(object.byteLength)) {
      invalid(path, "existing output identity is malformed");
    }
    const byteLength = Number(object.byteLength);
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength > MAX_EXISTING_FILE_BYTES) {
      resource(path, "existing file exceeds byte bound");
    }
    total += byteLength;
    if (total > MAX_EXISTING_TOTAL_BYTES) resource(path, "existing files exceed aggregate byte bound");
    return Object.freeze({ outputPath, sha256: object.sha256, byteLength: object.byteLength });
  }).sort((left, right) => compareUtf8(left.outputPath, right.outputPath));
  assertUniquePaths(outputs, diagnosticPath);
  return outputs;
}

function parseNewOutputs(value) {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_TREE_ENTRIES) {
    invalid("$.input.outputs", "expected one bounded nonempty output list");
  }
  let total = 0;
  const outputs = value.map((item, index) => {
    const path = `$.input.outputs[${index}]`;
    const object = exactObject(item, ["bytes", "outputPath"], path);
    const outputPath = portablePath(object.outputPath, `${path}.outputPath`);
    const bytes = snapshotBytes(object.bytes, `${path}.bytes`);
    if (bytes.byteLength === 0 || bytes.byteLength > MAX_NEW_FILE_BYTES) {
      resource(path, "new file exceeds byte bound");
    }
    total += bytes.byteLength;
    if (total > MAX_NEW_TOTAL_BYTES) resource(path, "new files exceed aggregate byte bound");
    return Object.freeze({ outputPath, bytes, sha256: sha256(bytes) });
  }).sort((left, right) => compareUtf8(left.outputPath, right.outputPath));
  assertUniquePaths(outputs, "$.input.outputs");
  return outputs;
}

function validateCombinedPaths(existingOutputs, outputs) {
  const combined = [...existingOutputs, ...outputs]
    .sort((left, right) => compareUtf8(left.outputPath, right.outputPath));
  assertUniquePaths(combined, "$.input");
  const filePaths = new Set(combined.map((output) => output.outputPath));
  for (const path of filePaths) {
    let parent = dirname(path);
    while (parent !== ".") {
      if (filePaths.has(parent)) invalid("$.input", "distribution file path is also a parent directory");
      parent = dirname(parent);
    }
  }
}

async function verifyExistingFiles(outputRoot, existingOutputs, diagnosticPath) {
  const identities = [];
  for (const [index, output] of existingOutputs.entries()) {
    const path = join(outputRoot, output.outputPath);
    const discovered = await lstat(path, { bigint: true }).catch((cause) =>
      invalid(`${diagnosticPath}[${index}]`, "existing output is unavailable", { cause }));
    const hashed = await hashExactFile(
      path,
      Number(output.byteLength),
      discovered,
      `${diagnosticPath}[${index}]`,
    );
    if (hashed.sha256 !== output.sha256) {
      invalid(`${diagnosticPath}[${index}]`, "existing output bytes differ from expectation");
    }
    identities.push(Object.freeze({ path, identity: hashed.identity }));
  }
  return identities;
}

async function verifyExpectedOutputTree(outputRoot, expectedOutputs, diagnosticPath) {
  const expectedPaths = new Set(expectedOutputs.map((output) => output.outputPath));
  const before = await assertExactTree(outputRoot, expectedPaths);
  const identities = await verifyExistingFiles(outputRoot, expectedOutputs, diagnosticPath);
  const after = await assertExactTree(outputRoot, expectedPaths);
  if (!sameDirectoryIdentity(before, after)) {
    invalid("$.input.outputRoot", "output root identity changed during verification");
  }
  for (const [index, verified] of identities.entries()) {
    const observed = await lstat(verified.path, { bigint: true }).catch((cause) =>
      invalid(`${diagnosticPath}[${index}]`, "verified output disappeared after tree inspection", { cause }));
    if (!sameFileIdentity(verified.identity, observed)) {
      invalid(`${diagnosticPath}[${index}]`, "verified output identity changed after tree inspection");
    }
  }
  return Object.freeze({ rootIdentity: after });
}

async function createPrivateParents(outputRoot, outputPath, createdDirectories) {
  const segments = dirname(outputPath).split("/");
  let current = outputRoot;
  for (const segment of segments) {
    if (segment === ".") continue;
    current = join(current, segment);
    let stat = await lstat(current, { bigint: true }).catch(() => undefined);
    if (stat === undefined) {
      await mkdir(current, { mode: 0o700 });
      stat = await lstat(current, { bigint: true });
      createdDirectories.push(Object.freeze({
        path: current,
        identity: Object.freeze({ dev: stat.dev, ino: stat.ino, mode: stat.mode }),
      }));
    }
    if (!stat.isDirectory() || stat.isSymbolicLink() || Number(stat.mode & 0o077n) !== 0 ||
        await realpath(current) !== current) {
      invalid("$.input.outputs", "output parent must remain one private canonical directory");
    }
  }
}

async function writeExclusiveFile(path, bytes, outputPath) {
  const parent = dirname(path);
  const parentBefore = await lstat(parent, { bigint: true });
  let handle;
  let identity;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    identity = await handle.stat({ bigint: true });
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesWritten } = await handle.write(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesWritten <= 0) invalid(`$.input.outputs.${outputPath}`, "write made no progress");
      offset += bytesWritten;
    }
    await handle.sync();
    const after = await handle.stat({ bigint: true });
    if (!sameStableFileIdentity(identity, after) || after.size !== BigInt(bytes.byteLength)) {
      invalid(`$.input.outputs.${outputPath}`, "output identity changed while written");
    }
    const parentAfter = await lstat(parent, { bigint: true });
    if (!sameDirectoryIdentity(parentBefore, parentAfter) || await realpath(parent) !== parent) {
      invalid(`$.input.outputs.${outputPath}`, "output parent changed while written");
    }
    return Object.freeze({
      dev: after.dev,
      ino: after.ino,
      mode: after.mode,
      nlink: after.nlink,
      size: after.size,
      mtimeNs: after.mtimeNs,
      ctimeNs: after.ctimeNs,
    });
  } catch (cause) {
    if (identity !== undefined) await unlinkOwnedFile(path, identity);
    if (cause instanceof CppCuteBrowserDistributionOutputFilesError) throw cause;
    invalid(`$.input.outputs.${outputPath}`, "failed to write exact output", { cause });
  } finally {
    await handle?.close();
  }
}

async function readExactFile(path, expectedByteLength, identity, diagnosticPath) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(identity, before) || before.size !== BigInt(expectedByteLength)) {
      invalid(diagnosticPath, "persisted output identity differs before reread");
    }
    const bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "persisted output changed while reread");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "persisted output changed while reread");
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserDistributionOutputFilesError) throw cause;
    invalid(diagnosticPath, "failed to reread exact output", { cause });
  } finally {
    await handle?.close();
  }
}

async function hashExactFile(path, expectedByteLength, discovered, diagnosticPath) {
  const uid = typeof process.getuid === "function" ? BigInt(process.getuid()) : undefined;
  if (!discovered.isFile() || discovered.isSymbolicLink() || discovered.nlink !== 1n ||
      discovered.size !== BigInt(expectedByteLength) || Number(discovered.mode & 0o222n) !== 0 ||
      (uid !== undefined && discovered.uid !== uid) || await realpath(path) !== path) {
    invalid(diagnosticPath, "expected one exact canonical non-writable regular file");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(discovered, before)) invalid(diagnosticPath, "file identity changed before read");
    const digest = createHash("sha256");
    const buffer = new Uint8Array(READ_BUFFER_BYTES);
    let offset = 0;
    while (offset < expectedByteLength) {
      const length = Math.min(buffer.byteLength, expectedByteLength - offset);
      const { bytesRead } = await handle.read(buffer, 0, length, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "file changed while read");
      digest.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "file identity changed while read");
    return Object.freeze({
      sha256: digest.digest("hex"),
      identity: after,
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserDistributionOutputFilesError) throw cause;
    invalid(diagnosticPath, "failed to hash exact existing output", { cause });
  } finally {
    await handle?.close();
  }
}

async function assertExactTree(root, expectedFiles) {
  const canonical = await realpath(root).catch((cause) =>
    invalid("$.input.outputRoot", "output root is unavailable", { cause }));
  const rootStat = await lstat(root, { bigint: true }).catch((cause) =>
    invalid("$.input.outputRoot", "output root identity is unavailable", { cause }));
  if (canonical !== root || !rootStat.isDirectory() || rootStat.isSymbolicLink() ||
      Number(rootStat.mode & 0o077n) !== 0) {
    invalid("$.input.outputRoot", "output root must be one private canonical directory");
  }
  const observedFiles = [];
  const observedDirectories = [];
  await walkTree(root, "", observedFiles, observedDirectories, { entries: 0 }, 0);
  observedFiles.sort(compareUtf8);
  observedDirectories.sort(compareUtf8);
  const expected = [...expectedFiles].sort(compareUtf8);
  const expectedDirectories = expectedDirectoriesForFiles(expected);
  if (!sameStrings(observedFiles, expected) ||
      !sameStrings(observedDirectories, expectedDirectories)) {
    invalid("$.input.outputRoot", "output tree differs from the exact expected file and directory set");
  }
  return Object.freeze({ dev: rootStat.dev, ino: rootStat.ino, mode: rootStat.mode });
}

async function walkTree(root, relativeDirectory, files, directories, budget, depth) {
  if (depth > MAX_TREE_DEPTH) resource("$.input.outputRoot", "tree exceeds depth bound");
  const path = relativeDirectory === "" ? root : join(root, relativeDirectory);
  let directory;
  try {
    directory = await opendir(path);
    for await (const entry of directory) {
      budget.entries += 1;
      if (budget.entries > MAX_TREE_ENTRIES) resource("$.input.outputRoot", "tree exceeds entry bound");
      const relativePath = relativeDirectory === "" ? entry.name : `${relativeDirectory}/${entry.name}`;
      if (Buffer.byteLength(relativePath, "utf8") > MAX_PATH_BYTES) {
        resource("$.input.outputRoot", "tree path exceeds byte bound");
      }
      const entryPath = join(root, relativePath);
      const stat = await lstat(entryPath, { bigint: true });
      if (entry.isSymbolicLink() || stat.isSymbolicLink()) {
        invalid("$.input.outputRoot", "output tree contains a symbolic link");
      }
      if (stat.isDirectory()) {
        if (Number(stat.mode & 0o077n) !== 0) {
          invalid("$.input.outputRoot", "output tree contains a non-private directory");
        }
        directories.push(relativePath);
        await walkTree(root, relativePath, files, directories, budget, depth + 1);
      } else if (stat.isFile() && stat.nlink === 1n && Number(stat.mode & 0o222n) === 0) {
        files.push(relativePath);
      } else {
        invalid("$.input.outputRoot", "output tree contains an unsafe entry");
      }
    }
  } catch (cause) {
    if (cause instanceof CppCuteBrowserDistributionOutputFilesError) throw cause;
    invalid("$.input.outputRoot", "failed to inspect output tree", { cause });
  } finally {
    await directory?.close().catch(() => undefined);
  }
}

function expectedDirectoriesForFiles(files) {
  const directories = new Set();
  for (const file of files) {
    let parent = dirname(file);
    while (parent !== ".") {
      directories.add(parent);
      parent = dirname(parent);
    }
  }
  return [...directories].sort(compareUtf8);
}

async function syncAffectedDirectories(outputRoot, outputs) {
  const directories = new Set([outputRoot]);
  for (const output of outputs) {
    let relative = dirname(output.outputPath);
    while (relative !== ".") {
      directories.add(join(outputRoot, relative));
      relative = dirname(relative);
    }
  }
  const ordered = [...directories].sort((left, right) => right.length - left.length);
  for (const directory of ordered) {
    let handle;
    try {
      handle = await open(directory, constants.O_RDONLY | constants.O_NOFOLLOW);
      await handle.sync();
    } catch (cause) {
      invalid("$.outputs", "failed to sync an affected output directory", { cause });
    } finally {
      await handle?.close();
    }
  }
}

async function cleanupCreated(createdFiles, createdDirectories, primaryCause) {
  const cleanupFailures = [];
  for (const created of [...createdFiles].reverse()) {
    try {
      await unlinkOwnedFile(created.path, created.identity);
    } catch (cause) {
      cleanupFailures.push(cause);
    }
  }
  for (const created of [...createdDirectories].reverse()) {
    try {
      const observed = await lstat(created.path, { bigint: true }).catch(() => undefined);
      if (observed !== undefined && sameDirectoryIdentity(observed, created.identity)) {
        await rmdir(created.path);
      }
    } catch (cause) {
      cleanupFailures.push(cause);
    }
  }
  if (cleanupFailures.length !== 0) {
    invalid("$.outputs.cleanup", "output cleanup failed after materialization error", {
      cause: new AggregateError([primaryCause, ...cleanupFailures], "distribution output cleanup failure"),
    });
  }
}

async function unlinkOwnedFile(path, identity) {
  const observed = await lstat(path, { bigint: true }).catch(() => undefined);
  if (observed !== undefined && observed.dev === identity.dev && observed.ino === identity.ino) {
    await unlink(path);
  }
}

function assertUniquePaths(outputs, diagnosticPath) {
  for (let index = 1; index < outputs.length; index += 1) {
    if (outputs[index - 1].outputPath === outputs[index].outputPath) {
      invalid(diagnosticPath, "distribution output paths must be unique");
    }
  }
}

function portablePath(value, diagnosticPath) {
  if (typeof value !== "string" || value === "" || value.startsWith("/") ||
      value.includes("\\") || value.includes("\0") || Buffer.byteLength(value, "utf8") > MAX_PATH_BYTES) {
    invalid(diagnosticPath, "expected one bounded portable relative path");
  }
  const segments = value.split("/");
  if (segments.some((segment) => segment === "." || segment === ".." || !PORTABLE_SEGMENT.test(segment))) {
    invalid(diagnosticPath, "distribution output path contains an invalid segment");
  }
  return value;
}

function snapshotBytes(value, diagnosticPath) {
  if (!(value instanceof Uint8Array) || Object.getPrototypeOf(value) !== Uint8Array.prototype ||
      !(value.buffer instanceof ArrayBuffer)) {
    invalid(diagnosticPath, "expected one ordinary Uint8Array");
  }
  try {
    const snapshot = new Uint8Array(value.byteLength);
    snapshot.set(value);
    return snapshot;
  } catch (cause) {
    invalid(diagnosticPath, "failed to snapshot output bytes", { cause });
  }
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${diagnosticPath}.${key}`, "expected data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !isAbsolute(value) || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink && left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function sameStableFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink;
}

function sameDirectoryIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode;
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  return Buffer.from(left.buffer, left.byteOffset, left.byteLength)
    .equals(Buffer.from(right.buffer, right.byteOffset, right.byteLength));
}

function sameStrings(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function resource(path, message) {
  invalid(path, `resource limit: ${message}`);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserDistributionOutputFilesError(path, message, options);
}
