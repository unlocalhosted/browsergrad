import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import { dirname, isAbsolute, normalize, resolve } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "../../dist/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH,
  inspectCppCuteBrowserWasmAgainstRuntimeAbi,
} from "../../dist/cpp_cute_browser_wasm_inspection.js";

const INVALID = "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-INVALID";
const CONFLICT = "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-CONFLICT";
const IO = "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-IO";
const PORTABLE_ABSOLUTE_PATH = /^\/[A-Za-z0-9._+/-]+$/u;
const ARGUMENT_NAMES = Object.freeze(["output", "wasm"]);
const REVIEW_REPORTS = new WeakSet();

export class CppCuteBrowserWasmReviewError extends Error {
  /** @param {string} code @param {string} path @param {string} message @param {ErrorOptions} [options] */
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWasmReviewError";
    this.code = code;
    this.path = path;
  }
}

/** @param {readonly string[]} argv */
export function parseCppCuteBrowserWasmReviewArguments(argv) {
  if (argv.length !== ARGUMENT_NAMES.length) invalid("$argv", "expected exactly two named arguments");
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string" || !argument.startsWith("--")) {
      invalid(`$argv[${index}]`, "expected --name=/absolute/path");
    }
    const equals = argument.indexOf("=");
    if (equals <= 2) invalid(`$argv[${index}]`, "expected --name=/absolute/path");
    const name = argument.slice(2, equals);
    if (!ARGUMENT_NAMES.includes(name)) invalid(`$argv[${index}]`, `unknown argument ${name}`);
    if (values.has(name)) invalid(`$argv[${index}]`, `duplicate argument ${name}`);
    values.set(name, portableAbsolutePath(argument.slice(equals + 1), `$argv.${name}`));
  }
  return Object.freeze(Object.fromEntries(ARGUMENT_NAMES.map((name) => [name, values.get(name)])));
}

/**
 * The CLI process is the disposable hard-cancellation boundary around the raw
 * inspector's intentionally synchronous parse/validation phases.
 *
 * @param {import("./cpp_cute_browser_wasm_review.mjs").ReviewCppCuteBrowserWasmFileInput} input
 */
export async function reviewCppCuteBrowserWasmFile(input) {
  const object = exactObject(input, ["wasmPath"], "$input");
  const wasmPath = portableAbsolutePath(exactString(object.wasmPath, "$.wasmPath"), "$.wasmPath");
  const bytes = await readExactWasmSnapshot(wasmPath);
  const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  );
  const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, runtimeAbi, {
    maxModuleByteLength: CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH,
  });
  REVIEW_REPORTS.add(report);
  return report;
}

/**
 * @param {string} outputPath
 * @param {import("../../dist/cpp_cute_browser_wasm_inspection.js").CppCuteBrowserWasmInspectionReport} report
 */
export async function writeCppCuteBrowserWasmReviewReport(outputPath, report) {
  if (typeof report !== "object" || report === null || !REVIEW_REPORTS.has(report)) {
    invalid("$report", "expected reviewer-issued raw-Wasm report");
  }
  const path = portableAbsolutePath(outputPath, "$outputPath");
  await admitPrivateDirectory(dirname(path), "$outputPath.parent");
  const bytes = canonicalJsonBytes(report);
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
      conflict("$outputPath", "Wasm review output must not already exist", { cause });
    }
    io("$outputPath", "failed to persist the Wasm review report", { cause });
  } finally {
    await handle?.close();
  }
  const directory = await open(dirname(path), constants.O_RDONLY | constants.O_DIRECTORY);
  try {
    await directory.sync();
  } finally {
    await directory.close();
  }
  return Object.freeze({
    outputPath: path,
    sha256: sha256(bytes),
    byteLength: bytes.byteLength,
    exactInterfaceConformance: report.exactInterfaceConformance,
    mismatchCount: report.mismatches.length,
  });
}

async function readExactWasmSnapshot(path) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size < 8n ||
        before.size > BigInt(CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH)) {
      invalid(
        "$.wasmPath",
        `Wasm file length must be from 8 through ${CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH} bytes`,
      );
    }
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead === 0) conflict("$.wasmPath", "Wasm file became shorter while reading");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileSnapshot(before, after)) conflict("$.wasmPath", "Wasm file changed while reading");
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserWasmReviewError) throw cause;
    if (isNodeError(cause, "ELOOP")) invalid("$.wasmPath", "Wasm input must not be a symbolic link", { cause });
    io("$.wasmPath", "failed to snapshot the Wasm file", { cause });
  } finally {
    await handle?.close();
  }
}

async function admitPrivateDirectory(path, diagnosticPath) {
  let stat;
  try {
    stat = await lstat(path, { bigint: true });
  } catch (cause) {
    io(diagnosticPath, "failed to inspect output directory", { cause });
  }
  if (!stat.isDirectory() || stat.isSymbolicLink()) invalid(diagnosticPath, "expected a non-symlink directory");
  if (typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid())) {
    invalid(diagnosticPath, "directory must be owned by the current user");
  }
  if ((stat.mode & 0o022n) !== 0n) invalid(diagnosticPath, "directory must not be writable by group or other users");
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected a plain object");
  }
  const object = /** @type {Record<string, unknown>} */ (value);
  const observed = Object.keys(object);
  if (observed.length !== keys.length || observed.some((key) => !keys.includes(key))) {
    invalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  return object;
}

function exactString(value, path) {
  if (typeof value !== "string") invalid(path, "expected a string");
  return value;
}

function portableAbsolutePath(value, path) {
  if (!isAbsolute(value) || normalize(value) !== value || value === "/" || value.endsWith("/") ||
      value.length > 4_096 || !PORTABLE_ABSOLUTE_PATH.test(value)) {
    invalid(path, "expected a normalized portable absolute POSIX path");
  }
  return value;
}

function sameFileSnapshot(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size &&
    left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function isNodeError(value, code) {
  return typeof value === "object" && value !== null && "code" in value && value.code === code;
}

function invalid(path, message, options) {
  throw new CppCuteBrowserWasmReviewError(INVALID, path, message, options);
}

function conflict(path, message, options) {
  throw new CppCuteBrowserWasmReviewError(CONFLICT, path, message, options);
}

function io(path, message, options) {
  throw new CppCuteBrowserWasmReviewError(IO, path, message, options);
}

const mainUrl = process.argv[1] === undefined ? undefined : pathToFileURL(resolve(process.argv[1])).href;
if (mainUrl === import.meta.url) {
  try {
    const arguments_ = parseCppCuteBrowserWasmReviewArguments(process.argv.slice(2));
    const report = await reviewCppCuteBrowserWasmFile({ wasmPath: arguments_.wasm });
    const result = await writeCppCuteBrowserWasmReviewReport(arguments_.output, report);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown Wasm review failure");
    process.stderr.write(`${JSON.stringify({
      name: error.name,
      message: error.message,
      ...ownErrorString(error, "code"),
      ...ownErrorString(error, "path"),
    })}\n`);
    process.exitCode = 1;
  }
}

function ownErrorString(error, name) {
  const descriptor = Object.getOwnPropertyDescriptor(error, name);
  return descriptor !== undefined && "value" in descriptor && typeof descriptor.value === "string"
    ? { [name]: descriptor.value }
    : {};
}
