import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, unlink } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

export const CPP_CUTE_BROWSER_SOURCE_ARCHIVE_INSPECTION_SCHEMA =
  "browsergrad.compiler.cpp-cute.source-archive-inspection";
export const CPP_CUTE_BROWSER_REGULAR_ARCHIVE_INSPECTION_SCHEMA =
  "browsergrad.compiler.cpp-cute.regular-archive-inspection";
export const CPP_CUTE_BROWSER_SOURCE_ARCHIVE_ADMISSION_SCHEMA =
  "browsergrad.compiler.cpp-cute.current-source-archive-admission";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-SOURCE-ARCHIVE-ADMISSION";
const INSPECTION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.source-archive-inspection.v1";
const ADMISSION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.current-source-archive-admission.v1";
const SOURCE_SET_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.source-archive-expectations.v1";
const REGULAR_ARCHIVE_SET_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.regular-archive-expectations.v1";
const SOURCE_ID = /^[a-z][a-z0-9-]*$/u;
const SHA1 = /^[0-9a-f]{40}$/u;
const SHA256 = /^[0-9a-f]{64}$/u;
const MAX_ARCHIVE_BYTES = 512n * 1024n * 1024n;
const MAX_TOTAL_ARCHIVE_BYTES = 1024n * 1024n * 1024n;
const READ_BUFFER_BYTES = 1024 * 1024;
const EXPECTED_CURRENT_SOURCE_IDS = Object.freeze(["cutlass", "llvm-project"]);
const INSPECTED_REGULAR_ARCHIVE_FILES = new WeakMap();
const INSPECTED_ARCHIVE_FILES = new WeakMap();
const CURRENT_ARCHIVE_FILES = new WeakMap();

export class CppCuteBrowserSourceArchiveAdmissionError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserSourceArchiveAdmissionError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Returns the exact source-archive expectations selected by the package build
 * lock. This reads no archive and grants no local-file or acquisition authority.
 */
export async function cppCuteBrowserCurrentSourceArchiveExpectations() {
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const sources = parseExpectedSources(
    unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body.sources,
    "$.buildInputLock.body.sources",
  );
  requireCurrentSourceSet(sources, "$.buildInputLock.body.sources");
  return Object.freeze({
    buildInputLockId: buildInputLock.lockId,
    buildInputLockResourceSha256: buildInputLock.resourceSha256,
    sources,
  });
}

/**
 * Inspects regular files against caller-supplied expectations. This lower-level
 * seam exists for focused testing and reusable file verification. Its result
 * explicitly does not bind the package build lock and cannot be promoted by
 * serialization or object copying.
 */
export async function inspectCppCuteBrowserSourceArchives(input) {
  const object = exactObject(input, ["archives", "expectedSources"], "$.input");
  const expectedSources = parseExpectedSources(object.expectedSources, "$.input.expectedSources");
  const archives = parseArchiveInputs(object.archives, "$.input.archives", expectedSources.length);
  bindArchiveInputs(archives, expectedSources, "$.input.archives");
  const regularInspection = await inspectCppCuteBrowserRegularArchiveFiles({
    archives: archives.map(({ sourceId, archivePath }) => ({ sourceId, archivePath })),
    expectedArchives: expectedSources.map(({ sourceId, archiveSha256, archiveByteLength }) => ({
      sourceId,
      archiveSha256,
      archiveByteLength,
    })),
  });
  const regularStored = INSPECTED_REGULAR_ARCHIVE_FILES.get(regularInspection);
  if (regularStored === undefined) invalid("$.inspection", "regular archive inspection lost live authority");
  const sourceSetSha256 = sha256(canonicalJsonBytes({
    domain: SOURCE_SET_HASH_DOMAIN,
    sources: expectedSources,
  }));
  const observed = expectedSources.map((source, index) => {
    const regular = regularInspection.archives[index];
    if (regular === undefined || regular.sourceId !== source.sourceId) {
      invalid("$.inspection", "regular archive observation differs from source expectations");
    }
    return Object.freeze({
      ...source,
      observedArchiveSha256: regular.observedArchiveSha256,
      observedArchiveByteLength: regular.observedArchiveByteLength,
    });
  });
  const inspectionHash = sha256(canonicalJsonBytes({
    domain: INSPECTION_HASH_DOMAIN,
    sourceSetSha256,
    archives: observed,
  }));
  const inspection = Object.freeze({
    schema: CPP_CUTE_BROWSER_SOURCE_ARCHIVE_INSPECTION_SCHEMA,
    version: 1,
    inspectionId: `bg.cpp.source-archive-inspection.sha256.${inspectionHash}`,
    authority: "caller-supplied-archive-expectations-only",
    expectedSourceSetSha256: sourceSetSha256,
    archives: Object.freeze(observed),
    totals: Object.freeze({
      archiveCount: observed.length,
      archiveByteLength: regularInspection.totals.archiveByteLength,
    }),
    claims: Object.freeze({
      exactCallerExpectedArchiveBytesVerified: true,
      currentBuildInputLockBound: false,
      networkAccessed: false,
      archivesExtracted: false,
      sourceTreesVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      buildExecuted: false,
      releaseReady: false,
    }),
  });
  INSPECTED_ARCHIVE_FILES.set(inspection, regularStored);
  return inspection;
}

/**
 * Verifies an arbitrary bounded set of regular archive files against minimal
 * caller-supplied byte identities. This grants no package-lock or source-plan
 * authority; higher-level wrappers must bind their own exact policy.
 */
export async function inspectCppCuteBrowserRegularArchiveFiles(input) {
  const object = exactObject(input, ["archives", "expectedArchives"], "$.input");
  const expectedArchives = parseArchiveExpectations(
    object.expectedArchives,
    "$.input.expectedArchives",
  );
  const archives = parseArchiveInputs(
    object.archives,
    "$.input.archives",
    expectedArchives.length,
  );
  bindArchiveInputs(archives, expectedArchives, "$.input.archives");
  const observations = [];
  const sourceFiles = new Map();
  let totalBytes = 0n;
  for (const [index, expected] of expectedArchives.entries()) {
    const archive = archives[index];
    if (archive === undefined || archive.sourceId !== expected.sourceId) {
      invalid("$.input.archives", "archive set differs from expected archive set");
    }
    const admitted = await inspectExactArchiveFile(
      archive.archivePath,
      expected,
      `$.input.archives[${archive.inputIndex}].archivePath`,
    );
    totalBytes += BigInt(admitted.archiveByteLength);
    if (totalBytes > MAX_TOTAL_ARCHIVE_BYTES) {
      resource("$.input.archives", "archive set exceeds the fixed total-byte ceiling");
    }
    for (const prior of sourceFiles.values()) {
      if (prior.identity.dev === admitted.identity.dev && prior.identity.ino === admitted.identity.ino) {
        invalid("$.input.archives", "distinct sources must use distinct archive inodes");
      }
    }
    observations.push(Object.freeze({
      ...expected,
      observedArchiveSha256: admitted.archiveSha256,
      observedArchiveByteLength: admitted.archiveByteLength,
    }));
    sourceFiles.set(expected.sourceId, admitted);
  }
  const expectedArchiveSetSha256 = sha256(canonicalJsonBytes({
    domain: REGULAR_ARCHIVE_SET_HASH_DOMAIN,
    archives: expectedArchives,
  }));
  const inspectionHash = sha256(canonicalJsonBytes({
    domain: `${REGULAR_ARCHIVE_SET_HASH_DOMAIN}.inspection`,
    expectedArchiveSetSha256,
    archives: observations,
  }));
  const inspection = Object.freeze({
    schema: CPP_CUTE_BROWSER_REGULAR_ARCHIVE_INSPECTION_SCHEMA,
    version: 1,
    inspectionId: `bg.cpp.regular-archive-inspection.sha256.${inspectionHash}`,
    authority: "caller-supplied-regular-archive-byte-expectations-only",
    expectedArchiveSetSha256,
    archives: Object.freeze(observations),
    totals: Object.freeze({
      archiveCount: observations.length,
      archiveByteLength: String(totalBytes),
    }),
    claims: Object.freeze({
      exactCallerExpectedArchiveBytesVerified: true,
      packagePolicyBound: false,
      networkAccessed: false,
      archivesExtracted: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
  INSPECTED_REGULAR_ARCHIVE_FILES.set(inspection, Object.freeze({ sourceFiles }));
  return inspection;
}

export async function copyCppCuteBrowserInspectedRegularArchive(
  inspection,
  sourceId,
  outputPath,
) {
  const stored = INSPECTED_REGULAR_ARCHIVE_FILES.get(inspection);
  if (stored === undefined) invalid("$.inspection", "expected verifier-issued regular-archive authority");
  return copyStoredArchive(stored, sourceId, outputPath, "$.inspection");
}

/**
 * Admits only the two exact archive files selected by the package build lock.
 * The evidence intentionally excludes local paths and retains them only behind
 * a same-process opaque authority.
 */
export async function admitCppCuteBrowserCurrentSourceArchives(input) {
  const object = exactObject(input, ["archives"], "$.input");
  const current = await cppCuteBrowserCurrentSourceArchiveExpectations();
  const inspection = await inspectCppCuteBrowserSourceArchives({
    archives: object.archives,
    expectedSources: current.sources,
  });
  const stored = INSPECTED_ARCHIVE_FILES.get(inspection);
  if (stored === undefined) invalid("$.inspection", "source-archive inspection lost live authority");
  const admissionHash = sha256(canonicalJsonBytes({
    domain: ADMISSION_HASH_DOMAIN,
    buildInputLockId: current.buildInputLockId,
    buildInputLockResourceSha256: current.buildInputLockResourceSha256,
    expectedSourceSetSha256: inspection.expectedSourceSetSha256,
    archives: inspection.archives,
  }));
  const admission = Object.freeze({
    schema: CPP_CUTE_BROWSER_SOURCE_ARCHIVE_ADMISSION_SCHEMA,
    version: 1,
    admissionId: `bg.cpp.current-source-archive-admission.sha256.${admissionHash}`,
    authority: "current-build-lock-local-source-archive-admission-only",
    buildInputLockId: current.buildInputLockId,
    buildInputLockResourceSha256: current.buildInputLockResourceSha256,
    expectedSourceSetSha256: inspection.expectedSourceSetSha256,
    archives: inspection.archives,
    totals: inspection.totals,
    claims: Object.freeze({
      exactCurrentBuildInputLockArchiveBytesVerified: true,
      currentBuildInputLockBound: true,
      localArchivePathsRetainedOpaquely: true,
      networkAccessed: false,
      archivesExtracted: false,
      sourceTreesVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      buildExecuted: false,
      releaseReady: false,
    }),
  });
  CURRENT_ARCHIVE_FILES.set(admission, stored);
  return admission;
}

export function requireCppCuteBrowserCurrentSourceArchiveAuthority(admission) {
  if (typeof admission !== "object" || admission === null ||
      CURRENT_ARCHIVE_FILES.get(admission) === undefined) {
    invalid("$.admission", "expected verifier-issued current source-archive authority");
  }
}

export function canonicalCppCuteBrowserCurrentSourceArchiveAdmissionBytes(admission) {
  requireCppCuteBrowserCurrentSourceArchiveAuthority(admission);
  return canonicalJsonBytes(admission);
}

/**
 * Copies one still-exact admitted archive into a private canonical directory.
 * The destination is no-clobber, identity-bound, synced, and independently
 * rehashed before success is returned.
 */
export async function copyCppCuteBrowserCurrentSourceArchive(
  admission,
  sourceId,
  outputPath,
) {
  const stored = CURRENT_ARCHIVE_FILES.get(admission);
  if (stored === undefined) invalid("$.admission", "expected verifier-issued current source-archive authority");
  return copyStoredArchive(stored, sourceId, outputPath, "$.admission");
}

async function copyStoredArchive(stored, sourceId, outputPath, authorityPath) {
  const id = sourceIdentifier(sourceId, "$.sourceId");
  const source = stored.sourceFiles.get(id);
  if (source === undefined) invalid("$.sourceId", "source is absent from the inspected archive set");
  const destination = absolutePath(outputPath, "$.outputPath");
  if (destination === source.archivePath) invalid("$.outputPath", "source and output paths must differ");
  await admitPrivateCanonicalDirectory(dirname(destination), "$.outputPath.parent");
  await copyExactArchiveToExclusiveFile(source, destination, "$.outputPath", authorityPath);
  return Object.freeze({
    sourceId: id,
    outputPath: destination,
    archiveSha256: source.archiveSha256,
    archiveByteLength: source.archiveByteLength,
    releaseReady: false,
  });
}

export function parseCppCuteBrowserSourceArchiveArguments(argv) {
  if (!Array.isArray(argv) || argv.length !== EXPECTED_CURRENT_SOURCE_IDS.length) {
    invalid("$arguments", "expected exactly --cutlass=/absolute/path and --llvm-project=/absolute/path");
  }
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--(cutlass|llvm-project)=(.+)$/u.exec(argument);
    if (match === null) invalid(`$arguments[${index}]`, "expected --cutlass= or --llvm-project=");
    const [, sourceId, archivePath] = match;
    if (values.has(sourceId)) invalid(`$arguments[${index}]`, `duplicate --${sourceId}`);
    values.set(sourceId, absolutePath(archivePath, `$arguments.${sourceId}`));
  }
  for (const sourceId of EXPECTED_CURRENT_SOURCE_IDS) {
    if (!values.has(sourceId)) invalid("$arguments", `missing --${sourceId}`);
  }
  return Object.freeze({
    archives: Object.freeze(EXPECTED_CURRENT_SOURCE_IDS.map((sourceId) => Object.freeze({
      sourceId,
      archivePath: values.get(sourceId),
    }))),
  });
}

function parseExpectedSources(value, diagnosticPath) {
  const sources = denseArray(value, diagnosticPath, 1, 16).map((entry, index) => {
    const path = `${diagnosticPath}[${index}]`;
    const object = exactObject(
      entry,
      [
        "sourceId",
        "repository",
        "acquisitionUrl",
        "tag",
        "commit",
        "treeSha1",
        "archiveSha256",
        "archiveByteLength",
        "attestationUrl",
        "attestationSha256",
        "attestationByteLength",
      ],
      path,
      true,
    );
    const source = {
      sourceId: sourceIdentifier(object.sourceId, `${path}.sourceId`),
      repository: httpsUrl(object.repository, `${path}.repository`),
      acquisitionUrl: httpsUrl(object.acquisitionUrl, `${path}.acquisitionUrl`),
      tag: boundedString(object.tag, `${path}.tag`, 1, 128),
      commit: patternString(object.commit, `${path}.commit`, SHA1),
      treeSha1: patternString(object.treeSha1, `${path}.treeSha1`, SHA1),
      archiveSha256: patternString(object.archiveSha256, `${path}.archiveSha256`, SHA256),
      archiveByteLength: positiveBoundedU64(
        object.archiveByteLength,
        `${path}.archiveByteLength`,
        MAX_ARCHIVE_BYTES,
      ),
    };
    const optionalNames = ["attestationUrl", "attestationSha256", "attestationByteLength"];
    const optionalCount = optionalNames.filter((name) => object[name] !== undefined).length;
    if (optionalCount !== 0 && optionalCount !== optionalNames.length) {
      invalid(path, "attestation URL, hash, and length must be present together");
    }
    if (optionalCount === optionalNames.length) {
      source.attestationUrl = httpsUrl(object.attestationUrl, `${path}.attestationUrl`);
      source.attestationSha256 = patternString(
        object.attestationSha256,
        `${path}.attestationSha256`,
        SHA256,
      );
      source.attestationByteLength = positiveBoundedU64(
        object.attestationByteLength,
        `${path}.attestationByteLength`,
        16n * 1024n * 1024n,
      );
    }
    return Object.freeze(source);
  });
  sources.sort((left, right) => compareUtf8(left.sourceId, right.sourceId));
  rejectDuplicateIds(sources, diagnosticPath);
  return Object.freeze(sources);
}

function parseArchiveExpectations(value, diagnosticPath) {
  const expected = denseArray(value, diagnosticPath, 1, 16).map((entry, index) => {
    const path = `${diagnosticPath}[${index}]`;
    const object = exactObject(
      entry,
      ["sourceId", "archiveSha256", "archiveByteLength"],
      path,
    );
    return Object.freeze({
      sourceId: sourceIdentifier(object.sourceId, `${path}.sourceId`),
      archiveSha256: patternString(object.archiveSha256, `${path}.archiveSha256`, SHA256),
      archiveByteLength: positiveBoundedU64(
        object.archiveByteLength,
        `${path}.archiveByteLength`,
        MAX_ARCHIVE_BYTES,
      ),
    });
  });
  expected.sort((left, right) => compareUtf8(left.sourceId, right.sourceId));
  rejectDuplicateIds(expected, diagnosticPath);
  return Object.freeze(expected);
}

function parseArchiveInputs(value, diagnosticPath, expectedLength) {
  const archives = denseArray(value, diagnosticPath, expectedLength, expectedLength)
    .map((entry, index) => {
      const path = `${diagnosticPath}[${index}]`;
      const object = exactObject(entry, ["sourceId", "archivePath"], path);
      return Object.freeze({
        inputIndex: index,
        sourceId: sourceIdentifier(object.sourceId, `${path}.sourceId`),
        archivePath: absolutePath(object.archivePath, `${path}.archivePath`),
      });
    });
  archives.sort((left, right) => compareUtf8(left.sourceId, right.sourceId));
  rejectDuplicateIds(archives, diagnosticPath);
  return Object.freeze(archives);
}

function bindArchiveInputs(archives, expectedSources, diagnosticPath) {
  if (archives.length !== expectedSources.length) {
    invalid(diagnosticPath, "archive set differs from expected source set");
  }
  for (const [index, source] of expectedSources.entries()) {
    if (archives[index]?.sourceId !== source.sourceId) {
      invalid(diagnosticPath, "archive set differs from expected source set");
    }
  }
}

function requireCurrentSourceSet(sources, diagnosticPath) {
  if (sources.length !== EXPECTED_CURRENT_SOURCE_IDS.length ||
      sources.some((source, index) => source.sourceId !== EXPECTED_CURRENT_SOURCE_IDS[index])) {
    invalid(diagnosticPath, "current build lock must contain the exact supported source-archive set");
  }
}

async function inspectExactArchiveFile(archivePath, expected, diagnosticPath) {
  await admitPrivateCanonicalDirectory(dirname(archivePath), `${diagnosticPath}.parent`);
  let discovered;
  try {
    discovered = await lstat(archivePath, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "source archive is unavailable", { cause });
  }
  requireOwnedRegularArchive(discovered, BigInt(expected.archiveByteLength), diagnosticPath);
  let resolved;
  try {
    resolved = await realpath(archivePath);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve source archive", { cause });
  }
  if (resolved !== archivePath) {
    invalid(diagnosticPath, "source archive path must already be canonical and contain no symlinks");
  }
  let handle;
  try {
    handle = await open(archivePath, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(discovered, before)) {
      invalid(diagnosticPath, "source archive identity changed before hashing");
    }
    requireOwnedRegularArchive(before, BigInt(expected.archiveByteLength), diagnosticPath);
    const digest = await hashOpenFile(handle, Number(before.size), diagnosticPath);
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) {
      invalid(diagnosticPath, "source archive changed while it was hashed");
    }
    if (digest !== expected.archiveSha256) {
      invalid(diagnosticPath, "source archive SHA-256 differs from the expected source record");
    }
    return Object.freeze({
      archivePath,
      archiveSha256: digest,
      archiveByteLength: String(before.size),
      identity: snapshotIdentity(before),
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserSourceArchiveAdmissionError) throw cause;
    invalid(diagnosticPath, "failed to inspect source archive", { cause });
  } finally {
    await handle?.close();
  }
}

async function copyExactArchiveToExclusiveFile(source, outputPath, diagnosticPath, authorityPath) {
  let sourceDiscovered;
  try {
    sourceDiscovered = await lstat(source.archivePath, { bigint: true });
  } catch (cause) {
    invalid(authorityPath, "inspected source archive is unavailable before copy", { cause });
  }
  if (!sameFileIdentity(source.identity, sourceDiscovered)) {
    invalid(authorityPath, "inspected source archive identity changed before copy");
  }
  let input;
  let output;
  let outputIdentity;
  try {
    input = await open(source.archivePath, constants.O_RDONLY | constants.O_NOFOLLOW);
    const inputBefore = await input.stat({ bigint: true });
    if (!sameFileIdentity(source.identity, inputBefore)) {
      invalid(authorityPath, "inspected source archive identity changed before copy");
    }
    output = await open(
      outputPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    const outputBefore = await output.stat({ bigint: true });
    outputIdentity = snapshotIdentity(outputBefore);
    if (!outputBefore.isFile() || outputBefore.nlink !== 1n || outputBefore.size !== 0n) {
      invalid(diagnosticPath, "new archive output is not one empty regular file");
    }
    const digest = createHash("sha256");
    const buffer = Buffer.allocUnsafe(READ_BUFFER_BYTES);
    let offset = 0;
    const expectedBytes = Number(inputBefore.size);
    while (offset < expectedBytes) {
      const length = Math.min(buffer.byteLength, expectedBytes - offset);
      const { bytesRead } = await input.read(buffer, 0, length, offset);
      if (bytesRead <= 0) invalid(authorityPath, "inspected source archive changed while copied");
      digest.update(buffer.subarray(0, bytesRead));
      await writeAll(output, buffer, bytesRead, offset, diagnosticPath);
      offset += bytesRead;
    }
    await output.sync();
    const [inputAfter, outputAfter] = await Promise.all([
      input.stat({ bigint: true }),
      output.stat({ bigint: true }),
    ]);
    const copiedSha256 = digest.digest("hex");
    if (!sameFileIdentity(inputBefore, inputAfter) || copiedSha256 !== source.archiveSha256) {
      invalid(authorityPath, "inspected source archive changed while copied");
    }
    if (outputAfter.dev !== outputBefore.dev || outputAfter.ino !== outputBefore.ino ||
        outputAfter.nlink !== 1n || outputAfter.size !== inputBefore.size) {
      invalid(diagnosticPath, "archive output identity changed while written");
    }
    await input.close();
    input = undefined;
    await output.close();
    output = undefined;
    const persisted = await inspectExactArchiveFile(outputPath, {
      archiveByteLength: source.archiveByteLength,
      archiveSha256: source.archiveSha256,
    }, diagnosticPath);
    if (persisted.identity.dev !== outputIdentity.dev || persisted.identity.ino !== outputIdentity.ino) {
      invalid(diagnosticPath, "persisted archive output no longer names the owned inode");
    }
  } catch (cause) {
    await input?.close().catch(() => {});
    await output?.close().catch(() => {});
    if (outputIdentity !== undefined) await unlinkOwnedFile(outputPath, outputIdentity);
    if (cause instanceof CppCuteBrowserSourceArchiveAdmissionError) throw cause;
    invalid(diagnosticPath, "failed to copy admitted source archive", { cause });
  }
}

async function hashOpenFile(handle, byteLength, diagnosticPath) {
  const digest = createHash("sha256");
  const buffer = Buffer.allocUnsafe(Math.min(READ_BUFFER_BYTES, byteLength || 1));
  let offset = 0;
  while (offset < byteLength) {
    const length = Math.min(buffer.byteLength, byteLength - offset);
    const { bytesRead } = await handle.read(buffer, 0, length, offset);
    if (bytesRead <= 0) invalid(diagnosticPath, "source archive changed while it was hashed");
    digest.update(buffer.subarray(0, bytesRead));
    offset += bytesRead;
  }
  return digest.digest("hex");
}

async function writeAll(handle, buffer, byteLength, position, diagnosticPath) {
  let written = 0;
  while (written < byteLength) {
    const result = await handle.write(buffer, written, byteLength - written, position + written);
    if (result.bytesWritten <= 0) invalid(diagnosticPath, "archive output stopped accepting bytes");
    written += result.bytesWritten;
  }
}

async function unlinkOwnedFile(path, identity) {
  try {
    const current = await lstat(path, { bigint: true });
    if (current.dev === identity.dev && current.ino === identity.ino) await unlink(path);
  } catch {
    // Never remove a path that no longer names the inode created here.
  }
}

async function admitPrivateCanonicalDirectory(path, diagnosticPath) {
  let before;
  try {
    before = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "archive directory is unavailable", { cause });
  }
  if (!before.isDirectory() || before.isSymbolicLink() || before.nlink < 1n) {
    invalid(diagnosticPath, "expected one non-symlink directory");
  }
  requireCurrentUser(before, diagnosticPath);
  if ((Number(before.mode) & 0o022) !== 0) {
    invalid(diagnosticPath, "archive directory must not be group- or world-writable");
  }
  let resolved;
  try {
    resolved = await realpath(path);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve archive directory", { cause });
  }
  if (resolved !== path) invalid(diagnosticPath, "archive directory path must already be canonical");
  const after = await lstat(path, { bigint: true });
  if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "archive directory identity changed");
}

function requireOwnedRegularArchive(entry, expectedByteLength, diagnosticPath) {
  if (!entry.isFile() || entry.isSymbolicLink() || entry.nlink !== 1n) {
    invalid(diagnosticPath, "source archive must be one non-symlink regular file with one hard link");
  }
  requireCurrentUser(entry, diagnosticPath);
  if ((Number(entry.mode) & 0o022) !== 0) {
    invalid(diagnosticPath, "source archive must not be group- or world-writable");
  }
  if (entry.size !== expectedByteLength) {
    invalid(diagnosticPath, "source archive byte length differs from the expected source record");
  }
}

function requireCurrentUser(entry, diagnosticPath) {
  const getuid = process.getuid;
  if (typeof getuid !== "function") invalid(diagnosticPath, "POSIX current-user identity is unavailable");
  if (entry.uid !== BigInt(getuid.call(process))) {
    invalid(diagnosticPath, "archive path must be owned by the current user");
  }
}

function snapshotIdentity(entry) {
  return Object.freeze({
    dev: entry.dev,
    ino: entry.ino,
    mode: entry.mode,
    nlink: entry.nlink,
    uid: entry.uid,
    gid: entry.gid,
    size: entry.size,
    mtimeNs: entry.mtimeNs,
    ctimeNs: entry.ctimeNs,
  });
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink && left.uid === right.uid && left.gid === right.gid &&
    left.size === right.size && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

function exactObject(value, keys, diagnosticPath, optional = false) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.some((key) => typeof key !== "string" || !keys.includes(key)) ||
      (!optional && actual.length !== keys.length)) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined) continue;
    if (!("value" in descriptor)) invalid(`${diagnosticPath}.${key}`, "expected data property");
    result[key] = descriptor.value;
  }
  return result;
}

function denseArray(value, diagnosticPath, minimum, maximum) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype ||
      value.length < minimum || value.length > maximum) {
    invalid(diagnosticPath, `expected a dense array with ${minimum} to ${maximum} entries`);
  }
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.hasOwn(value, index)) invalid(`${diagnosticPath}[${index}]`, "sparse arrays are forbidden");
  }
  return [...value];
}

function rejectDuplicateIds(values, diagnosticPath) {
  for (let index = 1; index < values.length; index += 1) {
    if (values[index - 1].sourceId === values[index].sourceId) {
      invalid(diagnosticPath, `duplicate source ID ${JSON.stringify(values[index].sourceId)}`);
    }
  }
}

function sourceIdentifier(value, diagnosticPath) {
  return patternString(value, diagnosticPath, SOURCE_ID);
}

function httpsUrl(value, diagnosticPath) {
  const text = boundedString(value, diagnosticPath, 1, 2048);
  let url;
  try {
    url = new URL(text);
  } catch (cause) {
    invalid(diagnosticPath, "expected one absolute HTTPS URL", { cause });
  }
  if (url.protocol !== "https:" || url.username !== "" || url.password !== "" || url.hash !== "" ||
      url.href !== text) {
    invalid(diagnosticPath, "expected one canonical credential-free HTTPS URL without a fragment");
  }
  return text;
}

function boundedString(value, diagnosticPath, minimum, maximum) {
  if (typeof value !== "string" || value.length < minimum || value.length > maximum || value.includes("\0")) {
    invalid(diagnosticPath, `expected a NUL-free string with ${minimum} to ${maximum} code units`);
  }
  return value;
}

function patternString(value, diagnosticPath, pattern) {
  const text = boundedString(value, diagnosticPath, 1, 2048);
  if (!pattern.test(text)) invalid(diagnosticPath, `expected ${pattern}`);
  return text;
}

function positiveBoundedU64(value, diagnosticPath, maximum) {
  if (typeof value !== "string" || !/^[1-9][0-9]*$/u.test(value)) {
    invalid(diagnosticPath, "expected one positive canonical decimal integer string");
  }
  const parsed = BigInt(value);
  if (parsed > maximum) resource(diagnosticPath, `value exceeds ${maximum}`);
  return value;
}

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !isAbsolute(value) || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function invalid(path, message, options) {
  throw new CppCuteBrowserSourceArchiveAdmissionError(path, message, options);
}

function resource(path, message) {
  invalid(path, `resource limit: ${message}`);
}

async function main() {
  try {
    const input = parseCppCuteBrowserSourceArchiveArguments(process.argv.slice(2));
    const admission = await admitCppCuteBrowserCurrentSourceArchives(input);
    process.stdout.write(`${new TextDecoder().decode(
      canonicalCppCuteBrowserCurrentSourceArchiveAdmissionBytes(admission),
    )}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown source-archive admission failure");
    const path = typeof cause === "object" && cause !== null && "path" in cause &&
      typeof cause.path === "string"
      ? ` at ${cause.path}`
      : "";
    process.stderr.write(`${error.name}${path}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
