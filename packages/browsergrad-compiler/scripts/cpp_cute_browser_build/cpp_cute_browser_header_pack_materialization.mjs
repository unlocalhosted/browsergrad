import { constants } from "node:fs";
import { lstat, mkdir, open, opendir, realpath } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  encodeCppCuteBrowserVfsPack,
  inspectCppCuteBrowserVfsPack,
} from "../../dist/cpp_cute_browser_vfs_pack.js";
import {
  copyCppCuteBrowserHeaderPackInventorySourceFile,
  inventoryCppCuteBrowserHeaderPackSources,
  readCppCuteBrowserHeaderPackInventorySpecification,
  requireCppCuteBrowserHeaderPackInventorySourceAuthority,
} from "./cpp_cute_browser_header_pack_inventory.mjs";

export const CPP_CUTE_BROWSER_HEADER_PACK_MATERIALIZATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-pack-materialization";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MATERIALIZATION";
const SAFE_OUTPUT_PATH = /^(?!\/)(?!.*\\)(?!.*(?:^|\/)\.\.?(?:\/|$))[A-Za-z0-9._+@=/-]+$/u;
const MAX_PACK_BYTES = 512 * 1024 * 1024;
const MATERIALIZATIONS = new WeakSet();

export class CppCuteBrowserHeaderPackMaterializationError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderPackMaterializationError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Materializes exact canonical VFS packs from one live source inventory. The
 * source files are reread and rehashed, and persisted bytes are independently
 * inspected. This remains output observation only: license, build,
 * reproducibility, asset-manifest, and release authority are all false.
 */
export async function materializeCppCuteBrowserHeaderPacks(input) {
  const fields = exactObject(input, ["inventory", "outputRoot"], "$.input");
  const inventory = fields.inventory;
  try {
    requireCppCuteBrowserHeaderPackInventorySourceAuthority(inventory);
  } catch (cause) {
    invalid("$.input.inventory", "expected one live exact source inventory", { cause });
  }
  const outputRoot = absolutePath(fields.outputRoot, "$.input.outputRoot");
  const rootIdentity = await admitEmptyCanonicalDirectory(outputRoot, "$.input.outputRoot");
  const expectedOutputPaths = inventory.packs.map((pack) => {
    safeOutputPath(pack.outputPath, "$.input.inventory.packs.outputPath");
    return pack.outputPath;
  });
  if (new Set(expectedOutputPaths).size !== expectedOutputPaths.length) {
    invalid("$.input.inventory.packs", "output paths must be unique");
  }
  await createOutputDirectories(outputRoot, expectedOutputPaths);

  const outputs = [];
  let totalPackBytes = 0;
  for (const [packIndex, pack] of inventory.packs.entries()) {
    const files = [];
    for (const file of pack.files) {
      const bytes = await copyCppCuteBrowserHeaderPackInventorySourceFile(
        inventory,
        pack.includeRootId,
        file.virtualPath,
      );
      files.push(Object.freeze({ virtualPath: file.virtualPath, bytes }));
    }
    let packBytes;
    let inspected;
    try {
      packBytes = await encodeCppCuteBrowserVfsPack(files);
      inspected = await inspectCppCuteBrowserVfsPack(packBytes);
    } catch (cause) {
      invalid(`$.input.inventory.packs[${packIndex}]`, "failed to encode and inspect canonical VFS pack", { cause });
    } finally {
      files.length = 0;
    }
    assertPackIdentity(pack, inspected, `$.input.inventory.packs[${packIndex}]`);
    totalPackBytes += packBytes.byteLength;
    if (totalPackBytes > MAX_PACK_BYTES) {
      resource("$.outputs", `aggregate pack bytes exceed ${MAX_PACK_BYTES}`);
    }
    const outputPath = join(outputRoot, pack.outputPath);
    const outputIdentity = await writeExclusivePack(
      outputPath,
      packBytes,
      `$.outputs[${packIndex}]`,
    );
    const persistedBytes = await readExactPack(
      outputPath,
      packBytes.byteLength,
      outputIdentity,
      `$.outputs[${packIndex}]`,
    );
    let persisted;
    try {
      persisted = await inspectCppCuteBrowserVfsPack(persistedBytes);
    } catch (cause) {
      invalid(`$.outputs[${packIndex}]`, "persisted bytes failed independent VFS inspection", { cause });
    }
    assertSameInspection(inspected, persisted, `$.outputs[${packIndex}]`);
    outputs.push(Object.freeze({
      ordinal: packIndex,
      includeRootId: pack.includeRootId,
      intendedAsset: pack.intendedAsset,
      outputRole: pack.outputRole,
      outputPath: pack.outputPath,
      packSha256: persisted.packSha256,
      packByteLength: persisted.packByteLength,
      fileContentByteLength: persisted.fileContentByteLength,
      contentSetSha256: persisted.contentSetSha256,
      fileCount: persisted.fileCount,
    }));
  }
  await assertExactOutputTree(outputRoot, new Set(expectedOutputPaths));
  const finalRoot = await lstatDirectory(outputRoot, "$.input.outputRoot");
  if (finalRoot.dev !== rootIdentity.dev || finalRoot.ino !== rootIdentity.ino) {
    invalid("$.input.outputRoot", "output root identity changed during materialization");
  }
  const materialization = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_PACK_MATERIALIZATION_SCHEMA,
    version: 1,
    authority: "deterministic-vfs-pack-materialization-only",
    inventoryId: inventory.inventoryId,
    buildInputLockId: inventory.buildInputLockId,
    buildInputLockResourceSha256: inventory.buildInputLockResourceSha256,
    outputRoot,
    outputs: Object.freeze(outputs),
    totalPackByteLength: String(totalPackBytes),
    claims: Object.freeze({
      exactSourceBytesReverified: true,
      canonicalVfsPacksIndependentlyInspected: true,
      networkAccessed: false,
      licenseReviewComplete: false,
      assetManifestBound: false,
      buildExecuted: false,
      reproducibilityObserved: false,
      releaseReady: false,
    }),
  });
  MATERIALIZATIONS.add(materialization);
  return materialization;
}

export function requireCppCuteBrowserHeaderPackMaterializationAuthority(materialization) {
  if (typeof materialization !== "object" || materialization === null ||
      !MATERIALIZATIONS.has(materialization)) {
    invalid("$.materialization", "expected verifier-issued header-pack materialization authority");
  }
}

export function canonicalCppCuteBrowserHeaderPackMaterializationBytes(materialization) {
  requireCppCuteBrowserHeaderPackMaterializationAuthority(materialization);
  return canonicalJsonBytes(materialization);
}

export function parseCppCuteBrowserHeaderPackMaterializationArguments(argv) {
  if (!Array.isArray(argv) || argv.length !== 2) {
    invalid("$arguments", "expected exactly --input=/absolute/path and --output-root=/absolute/path");
  }
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--(input|output-root)=(.+)$/u.exec(argument);
    if (match === null) invalid(`$arguments[${index}]`, "expected --input= or --output-root=");
    const [, name, value] = match;
    if (values.has(name)) invalid(`$arguments[${index}]`, `duplicate --${name}`);
    values.set(name, absolutePath(value, `$arguments.${name}`));
  }
  if (!values.has("input") || !values.has("output-root")) {
    invalid("$arguments", "both --input and --output-root are required");
  }
  return Object.freeze({ inputPath: values.get("input"), outputRoot: values.get("output-root") });
}

export async function materializeCppCuteBrowserHeaderPacksFromSpecification(input) {
  const fields = exactObject(input, ["inputPath", "outputRoot"], "$.input");
  const inputPath = absolutePath(fields.inputPath, "$.input.inputPath");
  const outputRoot = absolutePath(fields.outputRoot, "$.input.outputRoot");
  const specification = await readCppCuteBrowserHeaderPackInventorySpecification(inputPath);
  const inventory = await inventoryCppCuteBrowserHeaderPackSources(specification);
  return materializeCppCuteBrowserHeaderPacks({ inventory, outputRoot });
}

function assertPackIdentity(pack, inspected, diagnosticPath) {
  if (inspected.contentSetSha256 !== pack.contentSetSha256 ||
      inspected.fileContentByteLength !== pack.fileContentByteLength ||
      inspected.fileCount !== pack.fileCount) {
    invalid(diagnosticPath, "encoded VFS pack differs from the exact source inventory");
  }
}

function assertSameInspection(expected, actual, diagnosticPath) {
  for (const field of [
    "packSha256",
    "packByteLength",
    "fileContentByteLength",
    "contentSetSha256",
    "fileCount",
  ]) {
    if (expected[field] !== actual[field]) {
      invalid(diagnosticPath, `persisted pack differs at ${field}`);
    }
  }
}

async function admitEmptyCanonicalDirectory(path, diagnosticPath) {
  const before = await lstatDirectory(path, diagnosticPath);
  const effectiveUid = typeof process.getuid === "function" ? BigInt(process.getuid()) : undefined;
  if ((effectiveUid !== undefined && before.uid !== effectiveUid) || (before.mode & 0o077n) !== 0n) {
    invalid(diagnosticPath, "output root must be a current-user-owned private directory");
  }
  let resolved;
  try {
    resolved = await realpath(path);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve output root", { cause });
  }
  if (resolved !== path) invalid(diagnosticPath, "output root must already be canonical and contain no symlinks");
  let directory;
  try {
    directory = await opendir(path);
    const first = await directory.read();
    if (first !== null) invalid(diagnosticPath, "output root must be empty");
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackMaterializationError) throw cause;
    invalid(diagnosticPath, "failed to inspect output root", { cause });
  } finally {
    await directory?.close();
  }
  const after = await lstatDirectory(path, diagnosticPath);
  if (before.dev !== after.dev || before.ino !== after.ino || before.mtimeNs !== after.mtimeNs ||
      before.ctimeNs !== after.ctimeNs) {
    invalid(diagnosticPath, "output root changed during admission");
  }
  return before;
}

async function createOutputDirectories(outputRoot, outputPaths) {
  const directories = new Set();
  for (const outputPath of outputPaths) {
    let current = dirname(outputPath);
    while (current !== ".") {
      directories.add(current);
      current = dirname(current);
    }
  }
  const sorted = [...directories].sort((left, right) => {
    const depth = left.split("/").length - right.split("/").length;
    return depth === 0 ? compareUtf8(left, right) : depth;
  });
  for (const directory of sorted) {
    try {
      await mkdir(join(outputRoot, directory), { mode: 0o700 });
    } catch (cause) {
      invalid("$.input.outputRoot", "failed to create private output directory", { cause });
    }
  }
}

async function writeExclusivePack(path, bytes, diagnosticPath) {
  let handle;
  let identity;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o444,
    );
    const opened = await handle.stat({ bigint: true });
    if (!opened.isFile() || opened.size !== 0n || opened.nlink !== 1n) {
      invalid(diagnosticPath, "new pack output is not one empty regular file");
    }
    identity = Object.freeze({ dev: opened.dev, ino: opened.ino });
    await handle.writeFile(bytes);
    await handle.sync();
    const written = await handle.stat({ bigint: true });
    if (!written.isFile() || written.dev !== opened.dev || written.ino !== opened.ino ||
        written.nlink !== 1n || written.size !== BigInt(bytes.byteLength)) {
      invalid(diagnosticPath, "pack output identity changed while written");
    }
    await handle.close();
    handle = undefined;
    const persisted = await lstat(path, { bigint: true });
    if (!persisted.isFile() || persisted.isSymbolicLink() || persisted.nlink !== 1n ||
        persisted.dev !== identity.dev || persisted.ino !== identity.ino ||
        persisted.size !== BigInt(bytes.byteLength)) {
      invalid(diagnosticPath, "pack path no longer names the owned output inode");
    }
    return identity;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackMaterializationError) throw cause;
    invalid(diagnosticPath, "failed to write exclusive pack output", { cause });
  } finally {
    await handle?.close();
  }
}

async function readExactPack(path, expectedByteLength, expectedIdentity, diagnosticPath) {
  if (expectedByteLength < 96 || expectedByteLength > MAX_PACK_BYTES) {
    resource(diagnosticPath, "pack byte length is outside the closed materialization bounds");
  }
  let discovered;
  try {
    discovered = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "pack output is unavailable", { cause });
  }
  if (!discovered.isFile() || discovered.isSymbolicLink() || discovered.nlink !== 1n ||
      discovered.size !== BigInt(expectedByteLength) || discovered.dev !== expectedIdentity.dev ||
      discovered.ino !== expectedIdentity.ino) {
    invalid(diagnosticPath, "pack output identity or length changed");
  }
  let handle;
  let bytes;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(discovered, before)) invalid(diagnosticPath, "pack output changed before read");
    bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "pack output changed while read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "pack output changed while read");
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackMaterializationError) throw cause;
    invalid(diagnosticPath, "failed to read exact pack output", { cause });
  } finally {
    await handle?.close();
  }
  const persisted = await lstat(path, { bigint: true });
  if (!persisted.isFile() || persisted.isSymbolicLink() || persisted.nlink !== 1n ||
      persisted.dev !== expectedIdentity.dev || persisted.ino !== expectedIdentity.ino ||
      persisted.size !== BigInt(expectedByteLength)) {
    invalid(diagnosticPath, "pack path changed after persisted-byte verification");
  }
  return bytes;
}

async function assertExactOutputTree(root, expectedFiles) {
  const observed = new Set();
  await enumerate(root, "", observed);
  if (observed.size !== expectedFiles.size || [...observed].some((path) => !expectedFiles.has(path))) {
    invalid("$.input.outputRoot", "materialized tree contains missing or unexpected files");
  }
}

async function enumerate(root, relativePath, observed) {
  const directoryPath = relativePath === "" ? root : join(root, relativePath);
  const before = await lstatDirectory(directoryPath, "$.input.outputRoot");
  let directory;
  try {
    directory = await opendir(directoryPath);
    const names = [];
    for await (const entry of directory) names.push(entry.name);
    names.sort(compareUtf8);
    for (const name of names) {
      const childRelative = relativePath === "" ? name : `${relativePath}/${name}`;
      const childPath = join(root, childRelative);
      const entry = await lstat(childPath, { bigint: true });
      if (entry.isSymbolicLink()) invalid("$.input.outputRoot", "materialized tree contains a symbolic link");
      if (entry.isDirectory()) await enumerate(root, childRelative, observed);
      else if (entry.isFile() && entry.nlink === 1n) observed.add(childRelative);
      else invalid("$.input.outputRoot", "materialized tree contains an unsupported entry");
    }
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackMaterializationError) throw cause;
    invalid("$.input.outputRoot", "failed to enumerate materialized output tree", { cause });
  }
  const after = await lstatDirectory(directoryPath, "$.input.outputRoot");
  if (before.dev !== after.dev || before.ino !== after.ino) {
    invalid("$.input.outputRoot", "materialized directory identity changed during enumeration");
  }
}

async function lstatDirectory(path, diagnosticPath) {
  let entry;
  try {
    entry = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "directory is unavailable", { cause });
  }
  if (!entry.isDirectory() || entry.isSymbolicLink() || entry.nlink < 1n) {
    invalid(diagnosticPath, "expected one non-symlink directory");
  }
  return entry;
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length || actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected exactly ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) invalid(`${diagnosticPath}.${key}`, "expected data property");
    result[key] = descriptor.value;
  }
  return result;
}

function safeOutputPath(value, diagnosticPath) {
  if (typeof value !== "string" || !SAFE_OUTPUT_PATH.test(value)) {
    invalid(diagnosticPath, "expected one safe relative output path");
  }
  return value;
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

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderPackMaterializationError(path, message, options);
}

function resource(path, message, options) {
  throw new CppCuteBrowserHeaderPackMaterializationError(path, `resource limit: ${message}`, options);
}

async function main() {
  try {
    const options = parseCppCuteBrowserHeaderPackMaterializationArguments(process.argv.slice(2));
    const report = await materializeCppCuteBrowserHeaderPacksFromSpecification(options);
    process.stdout.write(`${JSON.stringify(report)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-pack materialization failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
