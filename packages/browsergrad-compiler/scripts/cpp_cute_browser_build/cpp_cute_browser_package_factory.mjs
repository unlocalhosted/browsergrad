import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  readFile,
  realpath,
  unlink,
} from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256,
} from "../../dist/resources/cpp_cute_browser_factory_identity.js";

export const CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256 =
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_SHA256;
export const CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH =
  CPP_CUTE_BROWSER_REVIEWED_FACTORY_BYTE_LENGTH;

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-FACTORY";
const SCRIPT_ROOT = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(SCRIPT_ROOT, "..", "..");
const DEFAULT_SOURCE = join(PACKAGE_ROOT, "src", "resources", "clang-extractor.mjs");
const DEFAULT_DESTINATION_ROOT = join(PACKAGE_ROOT, "dist", "resources");

export class CppCuteBrowserPackageFactoryError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserPackageFactoryError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Copies the exact reviewed generated source into a fresh TypeScript output
 * tree. This is package materialization only; it grants no clean-build,
 * reproducibility, Worker-execution, provenance, or release authority.
 */
export async function materializeCppCuteBrowserPackageFactory(input = {}) {
  const values = exactInput(input);
  const sourcePath = values.sourcePath ?? DEFAULT_SOURCE;
  const destinationRoot = values.destinationRoot ?? DEFAULT_DESTINATION_ROOT;
  await exactRegularFile(sourcePath, "$.sourcePath");
  await exactDirectory(destinationRoot, "$.destinationRoot");
  const sourceBytes = await readExactFactory(sourcePath, "$.sourcePath");
  const destinationPath = join(destinationRoot, "clang-extractor.mjs");
  let destinationHandle;
  let destinationIdentity;
  try {
    destinationHandle = await open(
      destinationPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o444,
    );
    const opened = await destinationHandle.stat();
    if (!opened.isFile() || opened.size !== 0) {
      fail("$.destinationPath", "new factory inode is not one empty regular file");
    }
    destinationIdentity = Object.freeze({ dev: opened.dev, ino: opened.ino });
    await destinationHandle.writeFile(sourceBytes);
    await destinationHandle.sync();
    const written = await destinationHandle.stat();
    if (!written.isFile() || written.dev !== destinationIdentity.dev ||
        written.ino !== destinationIdentity.ino ||
        written.size !== CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH) {
      fail("$.destinationPath", "materialized factory inode has an invalid type or byte length");
    }
    await destinationHandle.close();
    destinationHandle = undefined;
    await exactFileIdentity(destinationPath, destinationIdentity, "$.destinationPath");
    const destinationBytes = await readExactFactory(destinationPath, "$.destinationPath");
    await exactFileIdentity(destinationPath, destinationIdentity, "$.destinationPath");
    if (!sourceBytes.equals(destinationBytes)) {
      fail("$.destinationPath", "materialized factory bytes differ from the checked-in source");
    }
  } catch (cause) {
    if (destinationHandle !== undefined) {
      try {
        await destinationHandle.close();
      } catch {
        // Continue to the identity-bound cleanup attempt.
      }
    }
    if (destinationIdentity !== undefined) {
      try {
        await exactFileIdentity(destinationPath, destinationIdentity, "$.destinationPath");
        await unlink(destinationPath);
      } catch {
        // Never unlink a replacement; the primary failure remains authoritative.
      }
    }
    if (cause instanceof CppCuteBrowserPackageFactoryError) throw cause;
    fail("$.destinationPath", "failed to materialize the exact package factory", { cause });
  }
  return Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.package-factory-materialization",
    version: 1,
    authority: "package-materialization-only",
    sourcePath,
    destinationPath,
    factorySha256: CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256,
    factoryByteLength: CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH,
    exactSourceVerified: true,
    packageOwned: true,
    cleanBuildVerified: false,
    reproducibilityVerified: false,
    workerBundleVerified: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
}

function exactInput(value) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    fail("$.input", "expected one plain input record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.some((key) => typeof key !== "string" ||
      (key !== "sourcePath" && key !== "destinationRoot"))) {
    fail("$.input", "expected only sourcePath and destinationRoot");
  }
  const result = {};
  for (const key of ["sourcePath", "destinationRoot"]) {
    const descriptor = descriptors[key];
    if (descriptor === undefined) continue;
    if (!("value" in descriptor) || typeof descriptor.value !== "string" ||
        !descriptor.value.startsWith("/") || descriptor.value.includes("\0")) {
      fail(`$.input.${key}`, "expected one absolute NUL-free path data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

async function readExactFactory(path, field) {
  let bytes;
  try {
    bytes = await readFile(path);
  } catch (cause) {
    fail(field, "failed to read generated factory bytes", { cause });
  }
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (bytes.byteLength !== CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH ||
      digest !== CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256) {
    fail(field, "generated factory identity differs from the reviewed package resource");
  }
  return bytes;
}

async function exactRegularFile(path, field) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    fail(field, "factory path is unavailable", { cause });
  }
  if (!entry.isFile() || entry.isSymbolicLink()) {
    fail(field, "factory path must be one non-symlink regular file");
  }
  const resolved = await realpath(path);
  if (resolved !== path) fail(field, "factory path must already be canonical");
}

async function exactDirectory(path, field) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    fail(field, "package resource directory is unavailable", { cause });
  }
  if (!entry.isDirectory() || entry.isSymbolicLink()) {
    fail(field, "package resource root must be one non-symlink directory");
  }
  if (await realpath(path) !== path) fail(field, "package resource root must already be canonical");
}

async function exactFileIdentity(path, expected, field) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    fail(field, "materialized factory path is unavailable", { cause });
  }
  if (!entry.isFile() || entry.isSymbolicLink() ||
      entry.dev !== expected.dev || entry.ino !== expected.ino) {
    fail(field, "materialized factory path no longer names the owned inode");
  }
}

function fail(path, message, options) {
  throw new CppCuteBrowserPackageFactoryError(path, message, options);
}

async function main() {
  try {
    if (process.argv.length !== 2) fail("$arguments", "this command accepts no arguments");
    const report = await materializeCppCuteBrowserPackageFactory();
    process.stdout.write(`${JSON.stringify(report)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown package-factory failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
