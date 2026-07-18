import { createHash, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  open,
  opendir,
  realpath,
} from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, relative } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-RUNTIME-CLOSURE";
const OBSERVATION_SCHEMA = "browsergrad.compiler.cpp-cute.build-runtime-closure";
const CLOSURE_HASH_DOMAIN = "browsergrad.compiler.cpp-cute.build-runtime-closure.v1";
const PORTABLE_ABSOLUTE_PATH = /^\/[A-Za-z0-9._+/-]+$/u;
const SAFE_RELATIVE_PATH = /^(?!\/)(?!.*\\)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._@/-]+$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const MAX_FILE_BYTE_LENGTH = 16 * 1024 * 1024;
const MAX_AGGREGATE_BYTE_LENGTH = 64 * 1024 * 1024;
const MAX_FILE_COUNT = 256;
const MAX_OBSERVATION_BYTE_LENGTH = 256 * 1024;
const COPY_BUFFER_BYTE_LENGTH = 256 * 1024;

export const CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME =
  ".browsergrad-cpp-cute-build-runtime-closure.v1.json";

export const CPP_CUTE_BROWSER_BUILD_RUNTIME_SOURCE_PATHS = Object.freeze([
  "packages/browsergrad-compiler/package.json",
  "packages/browsergrad-compiler/dist/cpp_cute_aot_bytes.js",
  "packages/browsergrad-compiler/dist/cpp_cute_browser_build_lock.js",
  "packages/browsergrad-compiler/dist/cpp_cute_browser_runtime_abi.js",
  "packages/browsergrad-compiler/dist/resources/cpp_cute_browser_build_lock_v1.js",
  "packages/browsergrad-compiler/dist/resources/cpp_cute_browser_runtime_abi_v1.js",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_executor.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_executor_fs.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_executor_options.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_executor_process.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_failure_observation.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_log_sink.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_plan.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_runner.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_build_runtime_closure.mjs",
  "packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_configured_target_review.mjs",
  "packages/browsergrad-semantic-core/package.json",
  "packages/browsergrad-semantic-core/dist/schema.js",
  "packages/browsergrad-semantic-core/dist/schema/canonical-json.js",
  "packages/browsergrad-semantic-core/dist/schema/diagnostics.js",
  "packages/browsergrad-semantic-core/dist/schema/envelope.js",
  "packages/browsergrad-semantic-core/dist/schema/float-bits.js",
  "packages/browsergrad-semantic-core/dist/schema/hash.js",
  "packages/browsergrad-semantic-core/dist/schema/integers.js",
  "packages/browsergrad-semantic-core/dist/schema/json.js",
  "packages/browsergrad-semantic-core/dist/schema/limits.js",
]);

export class CppCuteBrowserBuildRuntimeClosureError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserBuildRuntimeClosureError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserBuildRuntimeClosureArguments(argv) {
  if (!Array.isArray(argv) || argv.length !== 2) {
    invalid("$argv", "expected exactly --source-root and --output-root");
  }
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string" || !argument.startsWith("--")) {
      invalid(`$argv[${index}]`, "expected --name=/absolute/path");
    }
    const equals = argument.indexOf("=");
    if (equals <= 2) invalid(`$argv[${index}]`, "expected --name=/absolute/path");
    const name = argument.slice(2, equals);
    if (name !== "source-root" && name !== "output-root") {
      invalid(`$argv[${index}]`, `unknown argument ${name}`);
    }
    if (values.has(name)) invalid(`$argv[${index}]`, `duplicate argument ${name}`);
    values.set(name, portableAbsolutePath(argument.slice(equals + 1), `$argv.${name}`));
  }
  if (!values.has("source-root") || !values.has("output-root")) {
    invalid("$argv", "both --source-root and --output-root are required");
  }
  return Object.freeze({
    outputRoot: values.get("output-root"),
    sourceRoot: values.get("source-root"),
  });
}

export async function stageCppCuteBrowserBuildRuntimeClosure(input) {
  const object = exactObject(input, ["outputRoot", "sourceRoot"], "$input");
  const sourceRoot = await admitExistingDirectory(
    portableAbsolutePath(object.sourceRoot, "$.sourceRoot"),
    "$.sourceRoot",
  );
  const outputRoot = portableAbsolutePath(object.outputRoot, "$.outputRoot");
  const outputParent = await admitExistingDirectory(dirname(outputRoot), "$.outputRoot.parent");
  if (dirname(outputRoot) !== outputParent) {
    invalid("$.outputRoot.parent", "output parent must already be a canonical real path");
  }
  await admitPrivateOutputParent(outputParent);
  if (pathsOverlap(sourceRoot, outputRoot) || pathsOverlap(outputRoot, sourceRoot)) {
    invalid("$input", "source and output roots must be distinct and non-overlapping");
  }
  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
  const mappings = runtimeMappings();
  for (const file of body.recipe.extractorSource.files) {
    mappings.push(Object.freeze({
      kind: "extractor",
      sourcePath: `packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/${file.path}`,
      targetPath: `packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/${file.path}`,
    }));
  }
  validateMappings(mappings);

  try {
    await mkdir(outputRoot, { mode: 0o700 });
  } catch (cause) {
    conflict("$.outputRoot", "output root must not already exist", { cause });
  }

  const directories = new Set([outputRoot]);
  const files = [];
  let aggregateByteLength = 0;
  for (const [index, mapping] of mappings.entries()) {
    const sourcePath = join(sourceRoot, mapping.sourcePath);
    const targetPath = join(outputRoot, mapping.targetPath);
    await makePrivateParents(outputRoot, dirname(targetPath), directories);
    const copied = await copyBoundedRegularFile(
      sourcePath,
      targetPath,
      `${mapping.kind === "runtime" ? "$runtime" : "$extractor"}[${index}]`,
    );
    aggregateByteLength += copied.byteLength;
    if (aggregateByteLength > MAX_AGGREGATE_BYTE_LENGTH) {
      resource("$files", "runtime closure exceeds the aggregate byte budget");
    }
    files.push(Object.freeze({
      kind: mapping.kind,
      path: mapping.targetPath,
      sha256: copied.sha256,
      byteLength: copied.byteLength,
    }));
  }
  files.sort(compareClosureFiles);
  const closureSha256 = hashCanonical({ domain: CLOSURE_HASH_DOMAIN, files });
  const observation = Object.freeze({
    schema: OBSERVATION_SCHEMA,
    version: 1,
    authority: "staged-build-runtime-closure-observation-only",
    lockId: lock.lockId,
    extractorSourceSetSha256: lock.extractorSourceSetSha256,
    closureSha256,
    fileCount: files.length,
    files: Object.freeze(files),
    claims: Object.freeze({
      exactReadableWorkspaceClosureVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
    }),
  });
  const observationBytes = canonicalJsonBytes(observation);
  if (observationBytes.byteLength > MAX_OBSERVATION_BYTE_LENGTH) {
    resource("$observation", "runtime closure observation exceeds its byte budget");
  }
  const observationPath = join(
    outputRoot,
    CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME,
  );
  await writeExclusiveFile(observationPath, observationBytes);
  await sealDirectories(directories);
  const verified = await verifyCppCuteBrowserBuildRuntimeClosure({ workspaceRoot: outputRoot });
  return Object.freeze({
    workspaceRoot: outputRoot,
    observationPath,
    observationSha256: verified.observationSha256,
    observationByteLength: verified.observationByteLength,
    closureSha256: verified.observation.closureSha256,
    fileCount: verified.observation.fileCount,
  });
}

export async function verifyCppCuteBrowserBuildRuntimeClosure(input) {
  const object = exactObject(input, ["workspaceRoot"], "$input");
  const workspaceRoot = await admitExistingDirectory(
    portableAbsolutePath(object.workspaceRoot, "$.workspaceRoot"),
    "$.workspaceRoot",
  );
  const observationPath = join(
    workspaceRoot,
    CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME,
  );
  const observationBytes = await readBoundedRegularFile(
    observationPath,
    MAX_OBSERVATION_BYTE_LENGTH,
    "$observation",
  );
  const observation = decodeObservation(observationBytes);
  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  exactValue(observation.lockId, lock.lockId, "$.observation.lockId");
  exactValue(
    observation.extractorSourceSetSha256,
    lock.extractorSourceSetSha256,
    "$.observation.extractorSourceSetSha256",
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
  const expectedMappings = runtimeMappings();
  for (const file of body.recipe.extractorSource.files) {
    expectedMappings.push(Object.freeze({
      kind: "extractor",
      sourcePath: "",
      targetPath: `packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/${file.path}`,
    }));
  }
  validateMappings(expectedMappings);
  const expected = expectedMappings
    .map(({ kind, targetPath }) => `${kind}:${targetPath}`)
    .sort();
  const observed = observation.files.map(({ kind, path }) => `${kind}:${path}`).sort();
  if (expected.length !== observed.length || expected.some((value, index) => value !== observed[index])) {
    invalid("$.observation.files", "observation does not name the exact runtime and extractor closure");
  }

  const expectedEntries = new Set([
    CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME,
    ...observation.files.map((file) => file.path),
  ]);
  const entries = await enumerateWorkspace(workspaceRoot, expectedEntries);
  if (entries.size !== expectedEntries.size ||
      [...entries].some((path) => !expectedEntries.has(path))) {
    invalid("$.workspaceRoot", "staged workspace contains entries outside the exact closure");
  }

  let aggregateByteLength = 0;
  for (const [index, file] of observation.files.entries()) {
    const actual = await readBoundedRegularFile(
      join(workspaceRoot, file.path),
      MAX_FILE_BYTE_LENGTH,
      `$.observation.files[${index}]`,
    );
    aggregateByteLength += actual.byteLength;
    if (aggregateByteLength > MAX_AGGREGATE_BYTE_LENGTH) {
      resource("$.observation.files", "runtime closure exceeds the aggregate byte budget");
    }
    exactValue(actual.byteLength, file.byteLength, `$.observation.files[${index}].byteLength`);
    if (!safeDigestEqual(sha256(actual), file.sha256)) {
      invalid(`$.observation.files[${index}].sha256`, "staged file digest differs from observation");
    }
  }
  const closureSha256 = hashCanonical({
    domain: CLOSURE_HASH_DOMAIN,
    files: observation.files,
  });
  exactValue(closureSha256, observation.closureSha256, "$.observation.closureSha256");
  return Object.freeze({
    observationPath,
    observationSha256: sha256(observationBytes),
    observationByteLength: observationBytes.byteLength,
    observation,
  });
}

function runtimeMappings() {
  return CPP_CUTE_BROWSER_BUILD_RUNTIME_SOURCE_PATHS.map((sourcePath) => {
    const semanticPrefix = "packages/browsergrad-semantic-core/";
    const targetPath = sourcePath.startsWith(semanticPrefix)
      ? `packages/browsergrad-compiler/node_modules/@unlocalhosted/browsergrad-semantic-core/${sourcePath.slice(semanticPrefix.length)}`
      : sourcePath;
    return Object.freeze({ kind: "runtime", sourcePath, targetPath });
  });
}

function validateMappings(mappings) {
  if (mappings.length === 0 || mappings.length > MAX_FILE_COUNT) {
    resource("$files", "runtime closure file count is outside the admitted range");
  }
  const targets = new Set();
  for (const [index, mapping] of mappings.entries()) {
    safeRelativePath(mapping.sourcePath || mapping.targetPath, `$files[${index}].sourcePath`);
    safeRelativePath(mapping.targetPath, `$files[${index}].targetPath`);
    if (targets.has(mapping.targetPath)) invalid(`$files[${index}].targetPath`, "duplicate target path");
    targets.add(mapping.targetPath);
  }
}

function decodeObservation(bytes) {
  let value;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (cause) {
    invalid("$observation", "observation must be strict UTF-8 JSON", { cause });
  }
  let canonical;
  try {
    canonical = canonicalJsonBytes(value);
  } catch (cause) {
    invalid("$observation", "observation is outside canonical JSON", { cause });
  }
  if (!equalBytes(bytes, canonical)) invalid("$observation", "observation bytes must be canonical JSON");
  const object = exactObject(value, [
    "schema", "version", "authority", "lockId", "extractorSourceSetSha256",
    "closureSha256", "fileCount", "files", "claims",
  ], "$observation");
  exactValue(object.schema, OBSERVATION_SCHEMA, "$.observation.schema");
  exactValue(object.version, 1, "$.observation.version");
  exactValue(
    object.authority,
    "staged-build-runtime-closure-observation-only",
    "$.observation.authority",
  );
  exactString(object.lockId, "$.observation.lockId");
  exactSha(object.extractorSourceSetSha256, "$.observation.extractorSourceSetSha256");
  exactSha(object.closureSha256, "$.observation.closureSha256");
  const fileCount = boundedInteger(object.fileCount, 1, MAX_FILE_COUNT, "$.observation.fileCount");
  if (!Array.isArray(object.files) || object.files.length !== fileCount) {
    invalid("$.observation.files", "files must match fileCount");
  }
  const files = object.files.map((value, index) => {
    const path = `$.observation.files[${index}]`;
    const file = exactObject(value, ["kind", "path", "sha256", "byteLength"], path);
    if (file.kind !== "runtime" && file.kind !== "extractor") {
      invalid(`${path}.kind`, "kind must be runtime or extractor");
    }
    return Object.freeze({
      kind: file.kind,
      path: safeRelativePath(file.path, `${path}.path`),
      sha256: exactSha(file.sha256, `${path}.sha256`),
      byteLength: boundedInteger(file.byteLength, 0, MAX_FILE_BYTE_LENGTH, `${path}.byteLength`),
    });
  });
  for (let index = 1; index < files.length; index += 1) {
    if (compareClosureFiles(files[index - 1], files[index]) >= 0) {
      invalid("$.observation.files", "files must be strictly sorted and unique");
    }
  }
  const claims = exactObject(object.claims, [
    "exactReadableWorkspaceClosureVerified", "buildExecuted", "outputIdentityAuthorized",
    "reproducibilityVerified", "releaseReady",
  ], "$.observation.claims");
  exactValue(claims.exactReadableWorkspaceClosureVerified, true, "$.observation.claims.exactReadableWorkspaceClosureVerified");
  for (const name of ["buildExecuted", "outputIdentityAuthorized", "reproducibilityVerified", "releaseReady"]) {
    exactValue(claims[name], false, `$.observation.claims.${name}`);
  }
  return Object.freeze({
    ...object,
    fileCount,
    files: Object.freeze(files),
    claims: Object.freeze({ ...claims }),
  });
}

async function copyBoundedRegularFile(sourcePath, targetPath, diagnosticPath) {
  let resolvedSourcePath;
  try {
    resolvedSourcePath = await realpath(sourcePath);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve source file", { cause });
  }
  if (resolvedSourcePath !== sourcePath) {
    invalid(diagnosticPath, "source file path must not traverse symbolic links");
  }
  const source = await openNoFollow(sourcePath, constants.O_RDONLY, diagnosticPath);
  let target;
  try {
    const before = await source.stat({ bigint: true });
    admitRegularStat(before, MAX_FILE_BYTE_LENGTH, diagnosticPath);
    target = await open(
      targetPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    const hash = createHash("sha256");
    const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTE_LENGTH);
    let offset = 0;
    while (offset < Number(before.size)) {
      const length = Math.min(buffer.byteLength, Number(before.size) - offset);
      const { bytesRead } = await source.read(buffer, 0, length, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "source file changed while it was copied");
      let written = 0;
      while (written < bytesRead) {
        const result = await target.write(
          buffer.subarray(written, bytesRead),
          0,
          bytesRead - written,
          offset + written,
        );
        if (result.bytesWritten <= 0) invalid(diagnosticPath, "target file accepted a short write");
        written += result.bytesWritten;
      }
      hash.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = await source.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "source file changed while it was copied");
    await target.sync();
    await target.chmod(0o444);
    return Object.freeze({ sha256: hash.digest("hex"), byteLength: offset });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserBuildRuntimeClosureError) throw cause;
    invalid(diagnosticPath, "failed to copy bounded regular file", { cause });
  } finally {
    await target?.close();
    await source.close();
  }
}

async function readBoundedRegularFile(path, maximum, diagnosticPath) {
  const handle = await openNoFollow(path, constants.O_RDONLY, diagnosticPath);
  try {
    const before = await handle.stat({ bigint: true });
    admitRegularStat(before, maximum, diagnosticPath);
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "file changed while it was read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "file changed while it was read");
    return bytes;
  } finally {
    await handle.close();
  }
}

async function writeExclusiveFile(path, bytes) {
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
    conflict("$observation", "failed to persist closure observation exclusively", { cause });
  } finally {
    await handle?.close();
  }
}

async function enumerateWorkspace(root, expectedFiles) {
  const files = new Set();
  const expectedDirectories = new Set();
  for (const file of expectedFiles) {
    const segments = file.split("/");
    for (let index = 1; index < segments.length; index += 1) {
      expectedDirectories.add(segments.slice(0, index).join("/"));
    }
  }
  const maximumNodes = expectedFiles.size + expectedDirectories.size;
  let nodes = 0;
  const directories = [root];
  while (directories.length > 0) {
    const directory = directories.pop();
    let stream;
    try {
      stream = await opendir(directory);
      for await (const entry of stream) {
        const path = join(directory, entry.name);
        const relativePath = relative(root, path);
        nodes += 1;
        if (nodes > maximumNodes) invalid("$.workspaceRoot", "staged workspace exceeds its exact node budget");
        if (entry.isDirectory()) {
          if (!expectedDirectories.has(relativePath)) {
            invalid("$.workspaceRoot", "staged workspace contains an undeclared directory");
          }
          directories.push(path);
        } else if (entry.isFile() && expectedFiles.has(relativePath)) {
          files.add(relativePath);
        } else {
          invalid("$.workspaceRoot", "staged workspace contains an undeclared node");
        }
      }
    } catch (cause) {
      if (cause instanceof CppCuteBrowserBuildRuntimeClosureError) throw cause;
      invalid("$.workspaceRoot", "failed to enumerate staged workspace", { cause });
    } finally {
      await stream?.close().catch(() => {});
    }
  }
  return files;
}

async function makePrivateParents(root, target, directories) {
  const relativeTarget = relative(root, target);
  if (relativeTarget.startsWith("..") || isAbsolute(relativeTarget)) {
    invalid("$outputRoot", "target parent escaped output root");
  }
  let current = root;
  for (const segment of relativeTarget.split("/").filter(Boolean)) {
    current = join(current, segment);
    if (directories.has(current)) continue;
    await mkdir(current, { mode: 0o700 });
    directories.add(current);
  }
}

async function sealDirectories(directories) {
  const paths = [...directories].sort((left, right) => right.length - left.length);
  for (const path of paths) await chmod(path, 0o555);
}

async function admitExistingDirectory(path, diagnosticPath) {
  let metadata;
  try {
    metadata = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "directory is unavailable", { cause });
  }
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    invalid(diagnosticPath, "expected a real directory, not a symbolic link");
  }
  let resolved;
  try {
    resolved = await realpath(path);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve directory", { cause });
  }
  return portableAbsolutePath(resolved, diagnosticPath);
}

async function admitPrivateOutputParent(path) {
  let metadata;
  try {
    metadata = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid("$.outputRoot.parent", "failed to inspect output parent", { cause });
  }
  if (typeof process.getuid !== "function" || metadata.uid !== BigInt(process.getuid())) {
    invalid("$.outputRoot.parent", "output parent must be owned by the current user");
  }
  if ((metadata.mode & 0o077n) !== 0n) {
    invalid("$.outputRoot.parent", "output parent must not grant group or other permissions");
  }
}

async function openNoFollow(path, flags, diagnosticPath) {
  try {
    return await open(path, flags | constants.O_NOFOLLOW);
  } catch (cause) {
    invalid(diagnosticPath, "failed to open regular file without following links", { cause });
  }
}

function admitRegularStat(metadata, maximum, path) {
  if (!metadata.isFile() || metadata.isSymbolicLink()) invalid(path, "expected a regular file");
  if (metadata.size < 0n || metadata.size > BigInt(maximum)) {
    resource(path, "file byte length is outside the admitted range");
  }
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size &&
    left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

function compareClosureFiles(left, right) {
  return left.path < right.path ? -1 : left.path > right.path ? 1 : 0;
}

function portableAbsolutePath(value, path) {
  const string = exactString(value, path);
  if (string === "/" || string.endsWith("/") || string.length > 4_096 ||
      !PORTABLE_ABSOLUTE_PATH.test(string) || !isAbsolute(string) || normalize(string) !== string) {
    invalid(path, "expected a normalized portable absolute path");
  }
  return string;
}

function safeRelativePath(value, path) {
  const string = exactString(value, path);
  if (string.length > 4_096 || !SAFE_RELATIVE_PATH.test(string) ||
      normalize(string) !== string || string === ".") {
    invalid(path, "expected a normalized safe relative path");
  }
  return string;
}

function pathsOverlap(left, right) {
  return left === right || right.startsWith(`${left}/`);
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value) || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected a plain object");
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    invalid(path, `expected exactly keys ${expected.join(",")}`);
  }
  return value;
}

function exactString(value, path) {
  if (typeof value !== "string" || value.length === 0) invalid(path, "expected a non-empty string");
  return value;
}

function boundedInteger(value, minimum, maximum, path) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    invalid(path, `expected an integer in [${minimum}, ${maximum}]`);
  }
  return value;
}

function exactSha(value, path) {
  const string = exactString(value, path);
  if (!SHA256_HEX.test(string)) invalid(path, "expected a lowercase SHA-256 digest");
  return string;
}

function exactValue(value, expected, path) {
  if (value !== expected) invalid(path, `expected ${JSON.stringify(expected)}`);
}

function hashCanonical(value) {
  return sha256(canonicalJsonBytes(value));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function safeDigestEqual(left, right) {
  if (!SHA256_HEX.test(left) || !SHA256_HEX.test(right)) return false;
  return timingSafeEqual(Buffer.from(left, "hex"), Buffer.from(right, "hex"));
}

function equalBytes(left, right) {
  return left.byteLength === right.byteLength && timingSafeEqual(left, right);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserBuildRuntimeClosureError(path, message, options);
}

function conflict(path, message, options) {
  throw new CppCuteBrowserBuildRuntimeClosureError(path, message, options);
}

function resource(path, message, options) {
  throw new CppCuteBrowserBuildRuntimeClosureError(path, message, options);
}

async function main() {
  try {
    const arguments_ = parseCppCuteBrowserBuildRuntimeClosureArguments(process.argv.slice(2));
    const result = await stageCppCuteBrowserBuildRuntimeClosure(arguments_);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown runtime closure failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
