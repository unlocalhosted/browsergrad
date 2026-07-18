import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  mkdir,
  open,
  readdir,
  readFile,
  realpath,
} from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, resolve } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  executeCppCuteClangWasmBuild,
  materializeCppCuteClangWasmSidecar,
  prepareCppCuteClangWasmBuildSource,
} from "./cpp_cute_browser_build_executor.mjs";

const ARGUMENT_NAMES = Object.freeze([
  "builder-observation",
  "llvm-archive",
  "llvm-source-root",
  "work-root",
  "workspace-root",
]);
const PORTABLE_ABSOLUTE_PATH = /^\/[A-Za-z0-9._+/-]+$/u;
const BUILDER_OBSERVATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.builder-container-observation";
const BUILD_EXECUTION_SCHEMA =
  "browsergrad.compiler.cpp-cute.clang-wasm-build-execution-observation";
const MAX_BUILDER_OBSERVATION_BYTE_LENGTH = 8 * 1024;

export class CppCuteBrowserBuildRunnerError extends Error {
  /** @param {string} path @param {string} message @param {ErrorOptions} [options] */
  constructor(path, message, options) {
    super(`BG-COMPILER-CPP-CUTE-BROWSER-BUILD-RUNNER: ${message}`, options);
    this.name = "CppCuteBrowserBuildRunnerError";
    this.path = path;
  }
}

/** @param {readonly string[]} argv */
export function parseCppCuteBrowserBuildRunnerArguments(argv) {
  if (argv.length !== ARGUMENT_NAMES.length) {
    fail("$argv", `expected exactly ${ARGUMENT_NAMES.length} named arguments`);
  }
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string" || !argument.startsWith("--")) {
      fail(`$argv[${index}]`, "expected --name=/absolute/path");
    }
    const equals = argument.indexOf("=");
    if (equals <= 2) fail(`$argv[${index}]`, "expected --name=/absolute/path");
    const name = argument.slice(2, equals);
    const value = argument.slice(equals + 1);
    if (!ARGUMENT_NAMES.includes(name)) fail(`$argv[${index}]`, `unknown argument ${name}`);
    if (values.has(name)) fail(`$argv[${index}]`, `duplicate argument ${name}`);
    values.set(name, portableAbsolutePath(value, `$argv.${name}`));
  }
  return Object.freeze(Object.fromEntries(ARGUMENT_NAMES.map((name) => [name, values.get(name)])));
}

/**
 * @param {Uint8Array} bytes
 * @param {Readonly<{ platformManifestDigest: string; imageConfigDigest: string }>} expected
 */
export function decodeCppCuteBuilderContainerObservation(bytes, expected) {
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_BUILDER_OBSERVATION_BYTE_LENGTH) {
    fail("$builderObservation", "builder observation length is outside the admitted range");
  }
  let value;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (cause) {
    fail("$builderObservation", "builder observation must be strict UTF-8 JSON", { cause });
  }
  const object = exactObject(value, [
    "schema",
    "version",
    "platform",
    "platformManifestDigest",
    "imageConfigDigest",
  ], "$builderObservation");
  exactValue(object.schema, BUILDER_OBSERVATION_SCHEMA, "$builderObservation.schema");
  exactValue(object.version, 1, "$builderObservation.version");
  exactValue(object.platform, "linux/amd64", "$builderObservation.platform");
  exactValue(
    object.platformManifestDigest,
    expected.platformManifestDigest,
    "$builderObservation.platformManifestDigest",
  );
  exactValue(
    object.imageConfigDigest,
    expected.imageConfigDigest,
    "$builderObservation.imageConfigDigest",
  );
  return Object.freeze({
    schema: BUILDER_OBSERVATION_SCHEMA,
    version: 1,
    platform: "linux/amd64",
    platformManifestDigest: expected.platformManifestDigest,
    imageConfigDigest: expected.imageConfigDigest,
  });
}

/** @param {readonly string[]} argv */
export async function runCppCuteBrowserBuild(argv) {
  const arguments_ = parseCppCuteBrowserBuildRunnerArguments(argv);
  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
  const llvm = body.sources.find((source) => source.sourceId === "llvm-project");
  if (llvm === undefined) fail("$buildLock.sources", "LLVM source selection is missing");

  const builderBytes = await readBoundedRegularFile(
    arguments_["builder-observation"],
    MAX_BUILDER_OBSERVATION_BYTE_LENGTH,
    "$builderObservation",
  );
  const builder = decodeCppCuteBuilderContainerObservation(builderBytes, body.builder);
  const isolation = await observeContainerIsolation([
    arguments_["workspace-root"],
    arguments_["llvm-archive"],
    arguments_["llvm-source-root"],
  ], arguments_["work-root"]);
  const archive = await hashRegularFile(
    arguments_["llvm-archive"],
    BigInt(llvm.archiveByteLength),
    llvm.archiveSha256,
  );
  await assertPrivateEmptyDirectory(arguments_["work-root"], "$workRoot");
  const outputRoot = join(arguments_["work-root"], "output");
  await mkdir(outputRoot, { mode: 0o700 });

  const tools = await discoverPinnedBuilderTools();
  const roots = Object.freeze({
    llvmProjectSourceRoot: arguments_["llvm-source-root"],
    extractorSourceRoot: join(arguments_["work-root"], "staged-extractor-source"),
    nativeBuildRoot: join(arguments_["work-root"], "native-tablegen"),
    wasmBuildRoot: join(arguments_["work-root"], "clang-extractor-wasm"),
    outputRoot,
    stateRoot: join(arguments_["work-root"], "state"),
  });
  const extractorSourceInputRoot = join(
    arguments_["workspace-root"],
    "packages",
    "browsergrad-compiler",
    "scripts",
    "cpp_cute_browser_build",
    "extractor",
  );
  const prepared = await prepareCppCuteClangWasmBuildSource({
    lock,
    tools,
    roots,
    extractorSourceInputRoot,
  });
  const executed = await executeCppCuteClangWasmBuild(prepared);
  const materialized = await materializeCppCuteClangWasmSidecar(prepared);
  const evidence = Object.freeze({
    schema: BUILD_EXECUTION_SCHEMA,
    version: 1,
    authority: "build-execution-observation-only",
    lockId: lock.lockId,
    builder,
    isolation,
    llvmSourceArchive: archive,
    execution: executed,
    sidecarMaterialization: materialized,
    claims: Object.freeze({
      sourceArchiveVerified: true,
      buildExecuted: true,
      networkDuringBuildObservedDisabled: true,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      producerAttested: false,
      releaseReady: false,
    }),
  });
  const evidenceBytes = canonicalJsonBytes(evidence);
  const evidencePath = join(outputRoot, "build-execution-observation.v1.json");
  await writeExclusiveReadOnlyFile(evidencePath, evidenceBytes);
  return Object.freeze({
    evidencePath,
    evidenceSha256: createHash("sha256").update(evidenceBytes).digest("hex"),
    evidenceByteLength: evidenceBytes.byteLength,
    wasmSha256: executed.wasmSha256,
    wasmByteLength: executed.wasmByteLength,
    factoryModuleSha256: executed.factoryModuleSha256,
    factoryModuleByteLength: executed.factoryModuleByteLength,
  });
}

async function discoverPinnedBuilderTools() {
  const emsdkRoot = "/emsdk";
  const cmakeExecutable = await firstRegularFile([
    "/usr/local/bin/cmake",
    "/usr/bin/cmake",
  ], "$tools.cmakeExecutable");
  const buildToolExecutable = "/usr/bin/make";
  await requireRegularFile(buildToolExecutable, "$tools.buildToolExecutable");
  const emscriptenToolchainFile =
    "/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake";
  await requireRegularFile(emscriptenToolchainFile, "$tools.emscriptenToolchainFile");
  const emscriptenConfigFile = await firstRegularFile([
    "/emsdk/.emscripten",
    "/emsdk/upstream/emscripten/.emscripten",
  ], "$tools.emscriptenConfigFile");
  const searchCandidates = [
    dirname(cmakeExecutable),
    "/emsdk/upstream/bin",
    "/emsdk/upstream/emscripten",
    "/usr/bin",
  ];
  /** @type {string[]} */
  const searchPath = [];
  for (const candidate of searchCandidates) {
    const resolved = await realpath(candidate);
    const stat = await lstat(resolved, { bigint: true });
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      fail("$tools.searchPath", `expected regular directory ${candidate}`);
    }
    if (!searchPath.includes(resolved)) searchPath.push(resolved);
  }
  return Object.freeze({
    cmakeExecutable,
    buildToolExecutable,
    emsdkRoot,
    emscriptenToolchainFile,
    emscriptenConfigFile,
    searchPath: Object.freeze(searchPath),
  });
}

/** @param {readonly string[]} readOnlyPaths @param {string} writablePath */
async function observeContainerIsolation(readOnlyPaths, writablePath) {
  const interfaces = (await readdir("/sys/class/net")).sort();
  if (interfaces.length !== 1 || interfaces[0] !== "lo") {
    fail("$isolation.network", "build container must expose only the loopback interface");
  }
  const status = await readFile("/proc/self/status", "utf8");
  const capability = /^CapEff:\s*([0-9A-Fa-f]+)$/mu.exec(status)?.[1];
  const noNewPrivileges = /^NoNewPrivs:\s*(\d+)$/mu.exec(status)?.[1];
  if (capability === undefined || !/^0+$/u.test(capability) || noNewPrivileges !== "1") {
    fail("$isolation.process", "effective capabilities must be zero and no-new-privileges enabled");
  }
  const mounts = parseMountInfo(await readFile("/proc/self/mountinfo", "utf8"));
  assertPathMountMode("/", mounts, "ro", "$isolation.rootFilesystem");
  for (const [index, path] of readOnlyPaths.entries()) {
    assertPathMountMode(path, mounts, "ro", `$isolation.readOnlyInputs[${index}]`);
  }
  assertPathMountMode(writablePath, mounts, "rw", "$isolation.workRoot");
  return Object.freeze({
    networkInterfaces: Object.freeze(interfaces),
    effectiveCapabilities: capability,
    noNewPrivileges: true,
    rootFilesystemReadOnly: true,
    inputMountsReadOnly: true,
    workMountReadWrite: true,
  });
}

/** @param {string} text */
function parseMountInfo(text) {
  return text.trim().split("\n").map((line, index) => {
    const fields = line.split(" ");
    const separator = fields.indexOf("-");
    if (separator < 6 || fields[4] === undefined || fields[5] === undefined) {
      fail(`$isolation.mountInfo[${index}]`, "malformed Linux mountinfo record");
    }
    return Object.freeze({
      mountPoint: decodeMountInfoPath(fields[4]),
      options: Object.freeze(fields[5].split(",")),
    });
  });
}

/** @param {string} path @param {readonly { mountPoint: string; options: readonly string[] }[]} mounts @param {"ro" | "rw"} mode @param {string} diagnosticPath */
function assertPathMountMode(path, mounts, mode, diagnosticPath) {
  const matching = mounts.filter((mount) => (
    path === mount.mountPoint || path.startsWith(`${mount.mountPoint === "/" ? "" : mount.mountPoint}/`)
  )).sort((left, right) => right.mountPoint.length - left.mountPoint.length)[0];
  if (matching === undefined || !matching.options.includes(mode)) {
    fail(diagnosticPath, `path ${path} must resolve through a ${mode} mount`);
  }
}

/** @param {string} value */
function decodeMountInfoPath(value) {
  return value
    .replaceAll("\\040", " ")
    .replaceAll("\\011", "\t")
    .replaceAll("\\012", "\n")
    .replaceAll("\\134", "\\");
}

/** @param {string} path @param {bigint} expectedByteLength @param {string} expectedSha256 */
async function hashRegularFile(path, expectedByteLength, expectedSha256) {
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size !== expectedByteLength || before.nlink !== 1n) {
      fail("$llvmArchive", "LLVM archive type, length, or link count differs from the build lock");
    }
    const hash = createHash("sha256");
    const buffer = new Uint8Array(1024 * 1024);
    let offset = 0n;
    while (offset < before.size) {
      const maximum = Number(before.size - offset > BigInt(buffer.byteLength)
        ? BigInt(buffer.byteLength)
        : before.size - offset);
      const { bytesRead } = await handle.read(buffer, 0, maximum, Number(offset));
      if (bytesRead === 0) fail("$llvmArchive", "LLVM archive became shorter while hashing");
      hash.update(buffer.subarray(0, bytesRead));
      offset += BigInt(bytesRead);
    }
    const after = await handle.stat({ bigint: true });
    if (!sameSnapshot(before, after)) {
      fail("$llvmArchive", "LLVM archive changed while hashing");
    }
    const observedSha256 = hash.digest("hex");
    if (observedSha256 !== expectedSha256) {
      fail("$llvmArchive.sha256", "LLVM archive digest differs from the build lock");
    }
    return Object.freeze({
      sourceId: "llvm-project",
      sha256: observedSha256,
      byteLength: String(after.size),
      verified: true,
    });
  } finally {
    await handle.close();
  }
}

/** @param {import("node:fs").BigIntStats} left @param {import("node:fs").BigIntStats} right */
function sameSnapshot(left, right) {
  return left.dev === right.dev && left.ino === right.ino &&
    left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

/** @param {string} path @param {number} maximumByteLength @param {string} diagnosticPath */
async function readBoundedRegularFile(path, maximumByteLength, diagnosticPath) {
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = await handle.stat({ bigint: true });
    if (!stat.isFile() || stat.size <= 0n || stat.size > BigInt(maximumByteLength)) {
      fail(diagnosticPath, "file type or length is outside the admitted range");
    }
    const bytes = new Uint8Array(Number(stat.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead === 0) fail(diagnosticPath, "file became shorter while reading");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameSnapshot(stat, after)) fail(diagnosticPath, "file changed while reading");
    return bytes;
  } finally {
    await handle.close();
  }
}

/** @param {string} path @param {string} diagnosticPath */
async function assertPrivateEmptyDirectory(path, diagnosticPath) {
  const stat = await lstat(path, { bigint: true });
  if (!stat.isDirectory() || stat.isSymbolicLink() ||
      typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid()) ||
      (stat.mode & 0o022n) !== 0n) {
    fail(diagnosticPath, "work root must be a private current-user-owned directory");
  }
  if ((await readdir(path)).length !== 0) fail(diagnosticPath, "work root must be empty");
}

/** @param {readonly string[]} candidates @param {string} diagnosticPath */
async function firstRegularFile(candidates, diagnosticPath) {
  for (const candidate of candidates) {
    try {
      const resolved = await realpath(candidate);
      await requireRegularFile(resolved, diagnosticPath);
      return resolved;
    } catch (cause) {
      if (!isNodeError(cause, "ENOENT")) throw cause;
    }
  }
  fail(diagnosticPath, `none of the fixed candidates exist: ${candidates.join(", ")}`);
}

/** @param {string} path @param {string} diagnosticPath */
async function requireRegularFile(path, diagnosticPath) {
  const stat = await lstat(path, { bigint: true });
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0n) {
    fail(diagnosticPath, `expected nonempty regular file ${path}`);
  }
}

/** @param {string} path @param {Uint8Array} bytes */
async function writeExclusiveReadOnlyFile(path, bytes) {
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
  const directory = await open(dirname(path), constants.O_RDONLY | constants.O_DIRECTORY);
  try {
    await directory.sync();
  } finally {
    await directory.close();
  }
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} path */
function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    fail(path, "expected a plain object");
  }
  const object = /** @type {Record<string, unknown>} */ (value);
  const observedKeys = Object.keys(object);
  if (observedKeys.length !== keys.length || observedKeys.some((key) => !keys.includes(key))) {
    fail(path, `expected exactly fields ${keys.join(", ")}`);
  }
  return object;
}

/** @param {unknown} observed @param {unknown} expected @param {string} path */
function exactValue(observed, expected, path) {
  if (observed !== expected) fail(path, `expected ${String(expected)}`);
}

/** @param {string} value @param {string} path */
function portableAbsolutePath(value, path) {
  if (!isAbsolute(value) || normalize(value) !== value || value === "/" ||
      value.endsWith("/") || value.length > 4_096 || !PORTABLE_ABSOLUTE_PATH.test(value)) {
    fail(path, "expected a normalized portable absolute POSIX path");
  }
  return value;
}

/** @param {unknown} value @param {string} code */
function isNodeError(value, code) {
  return typeof value === "object" && value !== null && "code" in value && value.code === code;
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] @returns {never} */
function fail(path, message, options) {
  throw new CppCuteBrowserBuildRunnerError(path, message, options);
}

const mainUrl = process.argv[1] === undefined
  ? undefined
  : pathToFileURL(resolve(process.argv[1])).href;
if (mainUrl === import.meta.url) {
  try {
    const result = await runCppCuteBrowserBuild(process.argv.slice(2));
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown build-runner failure");
    process.stderr.write(`${JSON.stringify({
      name: error.name,
      message: error.message,
      ...(error instanceof CppCuteBrowserBuildRunnerError ? { path: error.path } : {}),
    })}\n`);
    process.exitCode = 1;
  }
}
