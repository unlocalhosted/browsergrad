import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  deriveCppCuteBrowserBuildInputLockId,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_build_lock_v1.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
} from "../../dist/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_EXTRACTOR_SOURCE_PATHS,
} from "./cpp_cute_browser_build_plan.mjs";

const AUTHORING_ERROR = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-AUTHORING";
const scriptRoot = dirname(fileURLToPath(import.meta.url));
const extractorRoot = join(scriptRoot, "extractor");

export class CppCuteBrowserBuildLockAuthoringError extends Error {
  constructor(path, message, options) {
    super(`${AUTHORING_ERROR}: ${message}`, options);
    this.name = "CppCuteBrowserBuildLockAuthoringError";
    this.code = AUTHORING_ERROR;
    this.path = path;
  }
}

export function parseCppCuteBrowserBuildLockAuthoringArguments(argv) {
  if (argv.length === 0) return Object.freeze({ check: false });
  if (argv.length === 1 && argv[0] === "--check") {
    return Object.freeze({ check: true });
  }
  invalid("$arguments", "expected no arguments or exactly --check");
}

/**
 * Computes every derived source/build-lock identity from the current owned
 * extractor closure. This is an authoring projection only: it never edits the
 * reviewed TypeScript resource and never grants build or release authority.
 */
export async function projectCppCuteBrowserBuildInputLock() {
  const current = CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE;
  const projected = structuredClone(current);
  const source = projected.body.recipe.extractorSource;
  const files = [];
  let compileSessionSource;
  for (const [index, path] of CPP_CUTE_BROWSER_EXTRACTOR_SOURCE_PATHS.entries()) {
    let bytes;
    try {
      bytes = await readFile(join(extractorRoot, path));
    } catch (cause) {
      invalid(`$.files[${index}]`, `failed to read owned source ${path}`, { cause });
    }
    files.push({
      path,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      byteLength: String(bytes.byteLength),
    });
    if (path === "BrowserGradCppCuteCompileSession.cpp") {
      compileSessionSource = bytes.toString("utf8");
    }
  }
  const nativeRuntimeAbiResourceSha256 =
    exactNativeRuntimeAbiResourceSha256(compileSessionSource);
  source.files = files;
  source.sourceSetSha256 = await hashCanonicalJson({
    domain: source.hashDomain,
    files,
  });
  projected.lockId = await deriveCppCuteBrowserBuildInputLockId(projected.body);
  const resourceBytes = canonicalJsonBytes(projected);
  const report = Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.browser-build-lock-authoring-projection",
    version: 1,
    authority: "authoring-projection-only",
    checkedInResourceMatches: current.lockId === projected.lockId &&
      current.body.recipe.extractorSource.sourceSetSha256 === source.sourceSetSha256 &&
      sameFiles(current.body.recipe.extractorSource.files, files),
    lockId: projected.lockId,
    resourceSha256: createHash("sha256").update(resourceBytes).digest("hex"),
    resourceByteLength: resourceBytes.byteLength,
    recipeSha256: await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.browser-build-recipe.v1",
      recipe: projected.body.recipe,
    }),
    extractorSourceSetSha256: source.sourceSetSha256,
    nativeRuntimeAbiResourceSha256,
    files: Object.freeze(files.map((file) => Object.freeze(file))),
  });
  return report;
}

function exactNativeRuntimeAbiResourceSha256(source) {
  if (typeof source !== "string") {
    invalid(
      "$.nativeRuntimeAbiResourceSha256",
      "owned compile-session source is unavailable",
    );
  }
  const matches = [...source.matchAll(
    /constexpr std::string_view kRuntimeAbiManifestSha256 =\s*"([0-9a-f]{64})";/gu,
  )];
  if (matches.length !== 1 ||
      matches[0]?.[1] !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256) {
    invalid(
      "$.nativeRuntimeAbiResourceSha256",
      "native profile admission does not bind the exact package runtime-ABI resource",
    );
  }
  return matches[0][1];
}

function sameFiles(left, right) {
  return left.length === right.length && left.every((file, index) => {
    const candidate = right[index];
    return candidate !== undefined && file.path === candidate.path &&
      file.sha256 === candidate.sha256 && file.byteLength === candidate.byteLength;
  });
}

function invalid(path, message, options) {
  throw new CppCuteBrowserBuildLockAuthoringError(path, message, options);
}

async function main() {
  try {
    const arguments_ = parseCppCuteBrowserBuildLockAuthoringArguments(process.argv.slice(2));
    const report = await projectCppCuteBrowserBuildInputLock();
    if (arguments_.check && report.checkedInResourceMatches) {
      process.stdout.write(`build-lock authoring projection current: ${report.lockId}\n`);
      return;
    }
    const reportBytes = Buffer.from(canonicalJsonBytes(report));
    const stream = arguments_.check ? process.stderr : process.stdout;
    stream.write(reportBytes);
    stream.write("\n");
    if (arguments_.check) process.exitCode = 1;
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown build-lock authoring failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
