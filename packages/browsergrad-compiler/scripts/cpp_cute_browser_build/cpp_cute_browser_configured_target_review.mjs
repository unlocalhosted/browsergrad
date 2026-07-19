import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import { isAbsolute, join, normalize, relative } from "node:path/posix";
import { TextDecoder } from "node:util";

const INVALID =
  "BG-COMPILER-CPP-CUTE-BROWSER-CONFIGURED-TARGET-REVIEW-INVALID";
const MAX_CONFIGURATION_FILE_BYTE_LENGTH = 1024 * 1024;
const TARGET_DIRECTORY_PARTS = Object.freeze([
  "tools",
  "browsergrad_extractor",
  "CMakeFiles",
  "browsergrad-cpp-cute-extractor.dir",
]);
const REQUIRED_EXCEPTION_FLAG = /(?:^|\s)-fexceptions(?:\s|$)/u;
const FORBIDDEN_COMPILE_FLAGS = Object.freeze([
  /(?:^|\s)-fno-exceptions(?:\s|$)/u,
  /(?:^|\s)-fwasm-exceptions(?:\s|$)/u,
  /(?:^|\s)-fno-rtti(?:\s|$)/u,
]);
const FORBIDDEN_LINK_FLAGS = Object.freeze([
  /(?:^|\s)-fno-exceptions(?:\s|$)/u,
  /(?:^|\s)-fwasm-exceptions(?:\s|$)/u,
]);

export class CppCuteBrowserConfiguredTargetReviewError extends Error {
  /** @param {string} path @param {string} message @param {ErrorOptions} [options] */
  constructor(path, message, options) {
    super(`${INVALID}: ${message}`, options);
    this.name = "CppCuteBrowserConfiguredTargetReviewError";
    this.code = INVALID;
    this.path = path;
  }
}

/**
 * Reviews CMake's generated target-specific compile and link commands after
 * configuration and before the expensive LLVM/Clang build. This is a narrow
 * configuration observation: it proves neither compilation nor output ABI.
 *
 * @param {Readonly<{
 *   wasmBuildRoot: string;
 *   llvmProjectSourceRoot: string;
 *   factoryModulePath: string;
 * }>} input
 */
export async function reviewCppCuteBrowserConfiguredTarget(input) {
  const wasmBuildRoot = exactAbsolutePath(
    input.wasmBuildRoot,
    "$.wasmBuildRoot",
  );
  const llvmProjectSourceRoot = exactAbsolutePath(
    input.llvmProjectSourceRoot,
    "$.llvmProjectSourceRoot",
  );
  const factoryModulePath = exactAbsolutePath(
    input.factoryModulePath,
    "$.factoryModulePath",
  );
  const targetDirectory = join(wasmBuildRoot, ...TARGET_DIRECTORY_PARTS);
  await assertClosedDirectoryAncestry(wasmBuildRoot, targetDirectory);

  const compileFlags = await readObservedUtf8File(
    join(targetDirectory, "flags.make"),
    "$.compileFlags",
  );
  const linkCommand = await readObservedUtf8File(
    join(targetDirectory, "link.txt"),
    "$.linkCommand",
  );
  const cmakeCache = await readObservedUtf8File(
    join(wasmBuildRoot, "CMakeCache.txt"),
    "$.cmakeCache",
  );

  requireCacheBoolean(
    cmakeCache.text,
    "LLVM_ENABLE_RTTI",
    true,
    "$.cmakeCache.LLVM_ENABLE_RTTI",
  );

  const cxxFlags = exactMakeVariable(
    compileFlags.text,
    "CXX_FLAGS",
    "$.compileFlags.CXX_FLAGS",
  );
  requireFlag(cxxFlags, REQUIRED_EXCEPTION_FLAG, "$.compileFlags.CXX_FLAGS", "-fexceptions");
  forbidFlags(cxxFlags, FORBIDDEN_COMPILE_FLAGS, "$.compileFlags.CXX_FLAGS");
  const cxxIncludes = exactMakeVariable(
    compileFlags.text,
    "CXX_INCLUDES",
    "$.compileFlags.CXX_INCLUDES",
  );
  requireIncludeDirectory(
    cxxIncludes,
    join(llvmProjectSourceRoot, "clang", "include"),
    "$.compileFlags.CXX_INCLUDES",
  );
  requireIncludeDirectory(
    cxxIncludes,
    join(wasmBuildRoot, "tools", "clang", "include"),
    "$.compileFlags.CXX_INCLUDES",
  );
  requireIncludeDirectory(
    cxxIncludes,
    join(llvmProjectSourceRoot, "clang", "lib", "AST"),
    "$.compileFlags.CXX_INCLUDES",
  );

  const linkLine = exactNonemptyLine(linkCommand.text, "$.linkCommand");
  requireFlag(linkLine, REQUIRED_EXCEPTION_FLAG, "$.linkCommand", "-fexceptions");
  forbidFlags(linkLine, FORBIDDEN_LINK_FLAGS, "$.linkCommand");
  if (!linkLine.includes(factoryModulePath)) {
    invalid("$.linkCommand", "link command does not name the exact locked factory output");
  }

  return Object.freeze({
    authority: "configured-target-flags-review-only",
    exceptionMode: "emscripten-javascript",
    rttiRequired: true,
    clangIncludeDirectoriesVerified: true,
    llvmLibrariesRttiEnabled: true,
    cmakeCachePath: cmakeCache.path,
    cmakeCacheSha256: cmakeCache.sha256,
    cmakeCacheByteLength: cmakeCache.byteLength,
    compileFlagsPath: compileFlags.path,
    compileFlagsSha256: compileFlags.sha256,
    compileFlagsByteLength: compileFlags.byteLength,
    linkCommandPath: linkCommand.path,
    linkCommandSha256: linkCommand.sha256,
    linkCommandByteLength: linkCommand.byteLength,
    buildExecuted: false,
    abiConformanceVerified: false,
    releaseReady: false,
  });
}

/** @param {string} root @param {string} targetDirectory */
async function assertClosedDirectoryAncestry(root, targetDirectory) {
  let current = root;
  for (const part of TARGET_DIRECTORY_PARTS) {
    current = join(current, part);
    const stat = await lstat(current).catch((cause) => {
      invalid("$.targetDirectory", "configured target directory is missing", { cause });
    });
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      invalid("$.targetDirectory", "configured target ancestry must contain only real directories");
    }
  }
  if (current !== targetDirectory) {
    invalid("$.targetDirectory", "configured target directory escaped its closed ancestry");
  }
}

/** @param {string} path @param {string} diagnosticPath */
async function readObservedUtf8File(path, diagnosticPath) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size <= 0n ||
        before.size > BigInt(MAX_CONFIGURATION_FILE_BYTE_LENGTH)) {
      invalid(diagnosticPath, "expected a bounded nonempty regular file");
    }
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const read = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (read.bytesRead === 0) invalid(diagnosticPath, "configuration file ended before its observed length");
      offset += read.bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (before.dev !== after.dev || before.ino !== after.ino ||
        before.size !== after.size || before.ctimeNs !== after.ctimeNs) {
      invalid(diagnosticPath, "configuration file changed while it was reviewed");
    }
    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch (cause) {
      invalid(diagnosticPath, "configuration file is not valid UTF-8", { cause });
    }
    return Object.freeze({
      path,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      byteLength: bytes.byteLength,
      text,
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserConfiguredTargetReviewError) throw cause;
    invalid(diagnosticPath, "failed to read configured target evidence", { cause });
  } finally {
    await handle?.close();
  }
}

/** @param {string} text @param {string} name @param {string} path */
function exactMakeVariable(text, name, path) {
  const prefix = `${name} = `;
  const matches = text.split("\n").filter((line) => line.startsWith(prefix));
  if (matches.length !== 1) invalid(path, `expected exactly one ${name} assignment`);
  return matches[0].slice(prefix.length);
}

/** @param {string} text @param {string} path */
function exactNonemptyLine(text, path) {
  const lines = text.split("\n").filter((line) => line.length > 0);
  if (lines.length !== 1) invalid(path, "expected exactly one nonempty link command line");
  return lines[0];
}

/** @param {string} text @param {string} name @param {boolean} expected @param {string} path */
function requireCacheBoolean(text, name, expected, path) {
  const prefix = `${name}:BOOL=`;
  const matches = text.split("\n").filter((line) => line.startsWith(prefix));
  const expectedValue = expected ? "ON" : "OFF";
  if (matches.length !== 1 || matches[0] !== `${prefix}${expectedValue}`) {
    invalid(path, `expected exactly ${name}:BOOL=${expectedValue}`);
  }
}

/** @param {string} value @param {RegExp} pattern @param {string} path @param {string} flag */
function requireFlag(value, pattern, path, flag) {
  if (!pattern.test(value)) invalid(path, `configured target is missing required ${flag}`);
}

/** @param {string} value @param {string} directory @param {string} path */
function requireIncludeDirectory(value, directory, path) {
  const tokens = value.split(/\s+/u).filter((token) => token.length > 0);
  if (!tokens.includes(`-I${directory}`)) {
    invalid(path, `configured target is missing required Clang include directory ${directory}`);
  }
}

/** @param {string} value @param {readonly RegExp[]} patterns @param {string} path */
function forbidFlags(value, patterns, path) {
  if (patterns.some((pattern) => pattern.test(value))) {
    invalid(path, "configured target contains a forbidden exception or RTTI mode");
  }
}

/** @param {unknown} value @param {string} path */
function exactAbsolutePath(value, path) {
  if (typeof value !== "string" || !isAbsolute(value) || value.length > 4096 ||
      normalize(value) !== value || relative("/", value).startsWith("..")) {
    invalid(path, "expected an absolute path");
  }
  return value;
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] */
function invalid(path, message, options) {
  throw new CppCuteBrowserConfiguredTargetReviewError(path, message, options);
}
