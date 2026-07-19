import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, opendir, realpath } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve } from "node:path/posix";
import { fileURLToPath, pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

export const CPP_CUTE_BROWSER_HEADER_NOTICE_VERIFICATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-notice-verification";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-NOTICE-VERIFICATION";
const SCRIPT_ROOT = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(SCRIPT_ROOT, "..", "..");
const DEFAULT_RESOURCE_ROOT = join(PACKAGE_ROOT, "src", "resources", "licenses");
const MAX_NOTICE_BYTES = 64 * 1024;
const HEADER_ASSETS = Object.freeze([
  "compiler-resource-pack",
  "dependency-header-pack:cuda",
  "dependency-header-pack:cutlass",
  "dependency-header-pack:cxx-stdlib",
  "dependency-header-pack:linux-sysroot",
]);
const VERIFIED_NOTICES = new WeakSet();

export class CppCuteBrowserHeaderNoticeVerificationError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderNoticeVerificationError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Verifies package-owned bytes for every currently approved header-pack
 * notice. CUDA and Linux-sysroot review remain explicit unresolved blockers;
 * this authority cannot approve distribution or release.
 */
export async function verifyCppCuteBrowserHeaderPackNotices(input = {}) {
  const fields = exactObject(input, ["resourceRoot"], "$.input", true);
  const resourceRoot = await admitCanonicalDirectory(
    fields.resourceRoot === undefined
      ? DEFAULT_RESOURCE_ROOT
      : absolutePath(fields.resourceRoot, "$.input.resourceRoot"),
    "$.input.resourceRoot",
  );
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body;
  const approved = body.notices.approvedComponents
    .filter((notice) => notice.appliesTo.some((asset) => HEADER_ASSETS.includes(asset)))
    .sort((left, right) => compareUtf8(left.componentId, right.componentId));
  const unresolved = body.notices.unresolvedComponents
    .filter((notice) => HEADER_ASSETS.includes(notice.intendedAsset))
    .sort((left, right) => compareUtf8(left.componentId, right.componentId));
  if (approved.length === 0 || unresolved.length === 0) {
    invalid("$.buildInputLock.notices", "current header notice policy lost approved or unresolved components");
  }
  const expectedNames = new Set(approved.map((notice) => basename(notice.noticeOutputPath)));
  if (expectedNames.size !== approved.length) {
    invalid("$.buildInputLock.notices", "approved header notice output basenames are ambiguous");
  }
  await assertExactResourceDirectory(resourceRoot, expectedNames);

  const notices = [];
  for (const [index, notice] of approved.entries()) {
    const resourceFileName = basename(notice.noticeOutputPath);
    const bytes = await readExactNotice(
      join(resourceRoot, resourceFileName),
      Number(notice.noticeByteLength),
      `$.notices[${index}]`,
    );
    const digest = sha256(bytes);
    if (digest !== notice.noticeSha256) {
      invalid(`$.notices[${index}]`, "package notice bytes differ from the current build lock");
    }
    notices.push(Object.freeze({
      componentId: notice.componentId,
      licenseExpression: notice.licenseExpression,
      upstreamSourcePath: notice.sourcePath,
      noticeOutputPath: notice.noticeOutputPath,
      packageResourceFileName: resourceFileName,
      noticeSha256: digest,
      noticeByteLength: notice.noticeByteLength,
      appliesTo: Object.freeze([...notice.appliesTo]),
    }));
  }
  const evidence = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_NOTICE_VERIFICATION_SCHEMA,
    version: 1,
    authority: "approved-header-notice-byte-verification-only",
    buildInputLockId: buildInputLock.lockId,
    buildInputLockResourceSha256: buildInputLock.resourceSha256,
    notices: Object.freeze(notices),
    unresolvedNotices: Object.freeze(unresolved.map((notice) => Object.freeze({
      componentId: notice.componentId,
      intendedAsset: notice.intendedAsset,
      reasonCode: notice.reasonCode,
      disposition: notice.disposition,
    }))),
    claims: Object.freeze({
      exactApprovedHeaderNoticeBytesVerified: true,
      unresolvedHeaderNoticeComponentCount: unresolved.length,
      allHeaderNoticesResolved: false,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
  VERIFIED_NOTICES.add(evidence);
  return evidence;
}

export function requireCppCuteBrowserHeaderNoticeVerificationAuthority(evidence) {
  if (typeof evidence !== "object" || evidence === null || !VERIFIED_NOTICES.has(evidence)) {
    invalid("$.evidence", "expected verifier-issued approved header-notice authority");
  }
}

export function canonicalCppCuteBrowserHeaderNoticeVerificationBytes(evidence) {
  requireCppCuteBrowserHeaderNoticeVerificationAuthority(evidence);
  return canonicalJsonBytes(evidence);
}

async function assertExactResourceDirectory(resourceRoot, expectedNames) {
  let directory;
  const names = [];
  try {
    directory = await opendir(resourceRoot);
    for await (const entry of directory) names.push(entry.name);
  } catch (cause) {
    invalid("$.input.resourceRoot", "failed to enumerate package notice resources", { cause });
  }
  names.sort(compareUtf8);
  const expected = [...expectedNames].sort(compareUtf8);
  if (names.length !== expected.length || names.some((name, index) => name !== expected[index])) {
    invalid("$.input.resourceRoot", "package notice directory must contain the exact approved header notice set");
  }
}

async function readExactNotice(path, expectedByteLength, diagnosticPath) {
  if (!Number.isSafeInteger(expectedByteLength) || expectedByteLength <= 0 ||
      expectedByteLength > MAX_NOTICE_BYTES) {
    invalid(diagnosticPath, "build-lock notice length is outside the verification bound");
  }
  let discovered;
  try {
    discovered = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "package notice resource is unavailable", { cause });
  }
  if (!discovered.isFile() || discovered.isSymbolicLink() || discovered.nlink !== 1n ||
      discovered.size !== BigInt(expectedByteLength)) {
    invalid(diagnosticPath, "package notice must be one exact non-symlink regular file");
  }
  if (await realpath(path) !== path) {
    invalid(diagnosticPath, "package notice path must already be canonical and contain no symlinks");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(discovered, before)) invalid(diagnosticPath, "notice identity changed before read");
    const bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "notice changed while read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "notice changed while read");
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderNoticeVerificationError) throw cause;
    invalid(diagnosticPath, "failed to read package notice resource", { cause });
  } finally {
    await handle?.close();
  }
}

async function admitCanonicalDirectory(path, diagnosticPath) {
  let before;
  try {
    before = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "package notice resource directory is unavailable", { cause });
  }
  if (!before.isDirectory() || before.isSymbolicLink() || before.nlink < 1n) {
    invalid(diagnosticPath, "expected one non-symlink directory");
  }
  let resolved;
  try {
    resolved = await realpath(path);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve package notice resource directory", { cause });
  }
  if (resolved !== path) invalid(diagnosticPath, "package notice resource directory must already be canonical");
  const after = await lstat(path, { bigint: true });
  if (before.dev !== after.dev || before.ino !== after.ino) {
    invalid(diagnosticPath, "package notice resource directory identity changed during admission");
  }
  return path;
}

function exactObject(value, keys, diagnosticPath, optional) {
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

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderNoticeVerificationError(path, message, options);
}

async function main() {
  try {
    if (process.argv.length !== 2) invalid("$arguments", "this command accepts no arguments");
    const evidence = await verifyCppCuteBrowserHeaderPackNotices();
    process.stdout.write(`${new TextDecoder().decode(canonicalCppCuteBrowserHeaderNoticeVerificationBytes(evidence))}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-notice verification failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
