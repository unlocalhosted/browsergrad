import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  readFile,
  realpath,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_reproducibility_v3.js";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
  CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
} from "../../dist/resources/cpp_cute_browser_reproducibility_identity_v1.js";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-AUTHORING";
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIRECTORY = dirname(SCRIPT_PATH);
const PACKAGE_ROOT = resolve(SCRIPT_DIRECTORY, "../..");
const RESOURCE_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_reproducibility_v3.ts",
);
const IDENTITY_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_reproducibility_identity_v1.ts",
);
const MAX_INPUT_BYTE_LENGTH = 1024 * 1024;
const SHA256 = /^[0-9a-f]{64}$/u;
const SOURCE_REVISION = /^[0-9a-f]{40}$/u;
const RUN_ID = /^[1-9][0-9]{0,19}$/u;
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

const TOP_LEVEL_KEYS = Object.freeze([
  "authority",
  "builds",
  "claims",
  "cleanBuildCount",
  "comparison",
  "lockId",
  "schema",
  "sourceSetSha256",
  "version",
]);
const BUILD_KEYS = Object.freeze([
  "buildExecutionEvidenceByteLength",
  "buildExecutionEvidenceSha256",
  "factoryModuleByteLength",
  "factoryModuleSha256",
  "linkMapByteLength",
  "linkMapCanonicalByteLength",
  "linkMapCanonicalSha256",
  "linkMapSha256",
  "nativeTools",
  "ordinal",
  "runtimeAbiReviewByteLength",
  "runtimeAbiReviewExactInterfaceConformance",
  "runtimeAbiReviewSha256",
  "runtimeClosureObservationByteLength",
  "runtimeClosureObservationSha256",
  "runtimeClosureSha256",
  "wasmByteLength",
  "wasmSha256",
]);
const NATIVE_TOOLS_KEYS = Object.freeze([
  "clangTablegenByteLength",
  "clangTablegenSha256",
  "llvmTablegenByteLength",
  "llvmTablegenSha256",
]);
const COMPARISON_KEYS = Object.freeze([
  "canonicalCommandsAndEnvironmentMatched",
  "factoryModuleBytesMatched",
  "linkMapCanonicalProjectionMatched",
  "nativeTablegenIdentitiesMatched",
  "runtimeAbiReviewBytesMatched",
  "runtimeClosureMatched",
  "sourceAndBuildPathsDistinct",
  "wasmBytesMatched",
]);
const CLAIM_KEYS = Object.freeze([
  "abiConformanceVerified",
  "extractorOutputsReproducible",
  "fullDistributedOutputSetReproducible",
  "outputIdentityAuthorized",
  "producerAttested",
  "releaseReady",
]);

export class CppCuteBrowserReproducibilityAuthoringError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserReproducibilityAuthoringError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserReproducibilityAuthoringArguments(argv) {
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length === 1 && arguments_[0] === "--check") {
    return Object.freeze({
      check: true,
      inputPath: undefined,
      runId: undefined,
      sourceRevision: undefined,
    });
  }
  if (arguments_.length !== 3) {
    invalid(
      "$.arguments",
      "expected --check or input, run-id, and source-revision",
    );
  }
  const values = new Map();
  for (const [index, argument] of arguments_.entries()) {
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      invalid(`$.arguments[${index}]`, "expected one --name=value option");
    }
    const name = argument.slice(2, separator);
    if (name !== "input" && name !== "run-id" &&
        name !== "source-revision") {
      invalid(`$.arguments[${index}]`, `unsupported option --${name}`);
    }
    if (values.has(name)) {
      invalid(`$.arguments[${index}]`, `duplicate option --${name}`);
    }
    values.set(name, argument.slice(separator + 1));
  }
  const inputPath = values.get("input");
  const runId = values.get("run-id");
  const sourceRevision = values.get("source-revision");
  if (typeof inputPath !== "string" || !isAbsolute(inputPath) ||
      inputPath.includes("\0")) {
    invalid("$.arguments.input", "input requires one absolute NUL-free path");
  }
  if (typeof runId !== "string" || !RUN_ID.test(runId)) {
    invalid("$.arguments.runId", "run-id requires one positive decimal ID");
  }
  if (typeof sourceRevision !== "string" ||
      !SOURCE_REVISION.test(sourceRevision)) {
    invalid(
      "$.arguments.sourceRevision",
      "source-revision requires one lowercase 40-hex revision",
    );
  }
  return Object.freeze({
    check: false,
    inputPath,
    runId,
    sourceRevision,
  });
}

export async function projectCppCuteBrowserReproducibility(
  value,
  metadata,
) {
  validateEvidence(value);
  const metadataObject = exactObject(
    metadata,
    ["runId", "sourceRevision"],
    "$.metadata",
  );
  const runId = exactString(metadataObject.runId, "$.metadata.runId");
  const sourceRevision = exactString(
    metadataObject.sourceRevision,
    "$.metadata.sourceRevision",
  );
  if (!RUN_ID.test(runId)) {
    invalid("$.metadata.runId", "expected one positive decimal run ID");
  }
  if (!SOURCE_REVISION.test(sourceRevision)) {
    invalid(
      "$.metadata.sourceRevision",
      "expected one lowercase 40-hex source revision",
    );
  }

  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  if (value.lockId !== lock.lockId ||
      value.sourceSetSha256 !== lock.extractorSourceSetSha256) {
    invalid(
      "$.evidence",
      "evidence does not bind the current package build-input lock",
    );
  }
  const resourceBytes = canonicalJsonBytes(value);
  const first = value.builds[0];
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.reproducibility-authoring-projection",
    version: 1,
    authority: "package-authoring-projection-only",
    evidence: value,
    resourceSha256: digest(resourceBytes),
    resourceByteLength: resourceBytes.byteLength,
    runId,
    sourceRevision,
    wasmSha256: first.wasmSha256,
    wasmByteLength: first.wasmByteLength,
  });
}

export function renderCppCuteBrowserReproducibilityResource(projection) {
  exactProjection(projection);
  return `import {\n` +
    `  deepFreezeJson,\n` +
    `  type JsonObject,\n` +
    `} from "@unlocalhosted/browsergrad-semantic-core/schema";\n\n` +
    `/** Generated by cpp_cute_browser_reproducibility_authoring.mjs. */\n` +
    `const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE = ${JSON.stringify(projection.evidence, null, 2)} as const satisfies JsonObject;\n\n` +
    `export type CppCuteBrowserReproducibilityV3Resource =\n` +
    `  typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE;\n\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE = deepFreezeJson(\n` +
    `  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_VALUE,\n` +
    `) as unknown as CppCuteBrowserReproducibilityV3Resource;\n`;
}

export function renderCppCuteBrowserReproducibilityIdentity(projection) {
  exactProjection(projection);
  return `/** Generated by cpp_cute_browser_reproducibility_authoring.mjs. */\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256 =\n` +
    `  ${JSON.stringify(projection.resourceSha256)};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH = ${projection.resourceByteLength};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID = ${JSON.stringify(projection.runId)};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION =\n` +
    `  ${JSON.stringify(projection.sourceRevision)};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID = ${JSON.stringify(projection.runId)};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION =\n` +
    `  ${JSON.stringify(projection.sourceRevision)};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_SHA256 =\n` +
    `  ${JSON.stringify(projection.wasmSha256)};\n` +
    `export const CPP_CUTE_BROWSER_REPRODUCIBILITY_WASM_BYTE_LENGTH = ${projection.wasmByteLength};\n`;
}

async function run(argv) {
  const options = parseCppCuteBrowserReproducibilityAuthoringArguments(argv);
  const input = options.check
    ? CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE
    : await readCanonicalEvidence(options.inputPath);
  const projection = await projectCppCuteBrowserReproducibility(input, {
    runId: options.check
      ? exactCurrentRunId()
      : options.runId,
    sourceRevision: options.check
      ? exactCurrentSourceRevision()
      : options.sourceRevision,
  });
  const resource = renderCppCuteBrowserReproducibilityResource(projection);
  const identity = renderCppCuteBrowserReproducibilityIdentity(projection);
  if (options.check) {
    const [currentResource, currentIdentity] = await Promise.all([
      readFile(RESOURCE_PATH, "utf8"),
      readFile(IDENTITY_PATH, "utf8"),
    ]);
    if (currentResource !== resource || currentIdentity !== identity) {
      invalid(
        "$.package",
        "checked-in reproducibility resource differs from authoring projection",
      );
    }
    process.stdout.write(
      `reproducibility authoring projection current: ${projection.resourceSha256}\n`,
    );
    return;
  }
  await Promise.all([
    writeFile(RESOURCE_PATH, resource, "utf8"),
    writeFile(IDENTITY_PATH, identity, "utf8"),
  ]);
  process.stdout.write(JSON.stringify({
    schema:
      "browsergrad.compiler.cpp-cute.reproducibility-authoring-result",
    version: 1,
    authority: "package-authoring-result-only",
    resourceSha256: projection.resourceSha256,
    resourceByteLength: projection.resourceByteLength,
    runId: projection.runId,
    sourceRevision: projection.sourceRevision,
    wasmSha256: projection.wasmSha256,
    wasmByteLength: projection.wasmByteLength,
  }) + "\n");
}

function validateEvidence(value) {
  const evidence = exactObject(value, TOP_LEVEL_KEYS, "$.evidence");
  exactValue(
    evidence.schema,
    "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
    "$.evidence.schema",
  );
  exactValue(evidence.version, 3, "$.evidence.version");
  exactValue(
    evidence.authority,
    "clang-wasm-extractor-reproducibility-observation-only",
    "$.evidence.authority",
  );
  exactValue(evidence.cleanBuildCount, 2, "$.evidence.cleanBuildCount");
  sha256(evidence.sourceSetSha256, "$.evidence.sourceSetSha256");
  if (typeof evidence.lockId !== "string" ||
      !evidence.lockId.startsWith(
        "bg.cpp.browser-build-input-lock.sha256.",
      ) ||
      !SHA256.test(evidence.lockId.slice(
        "bg.cpp.browser-build-input-lock.sha256.".length,
      ))) {
    invalid("$.evidence.lockId", "expected one content-addressed lock ID");
  }
  const comparison = exactObject(
    evidence.comparison,
    COMPARISON_KEYS,
    "$.evidence.comparison",
  );
  for (const key of COMPARISON_KEYS) {
    exactValue(comparison[key], true, `$.evidence.comparison.${key}`);
  }
  const claims = exactObject(
    evidence.claims,
    CLAIM_KEYS,
    "$.evidence.claims",
  );
  exactValue(
    claims.extractorOutputsReproducible,
    true,
    "$.evidence.claims.extractorOutputsReproducible",
  );
  for (const key of CLAIM_KEYS) {
    if (key !== "extractorOutputsReproducible") {
      exactValue(claims[key], false, `$.evidence.claims.${key}`);
    }
  }
  if (!Array.isArray(evidence.builds) || evidence.builds.length !== 2) {
    invalid("$.evidence.builds", "expected exactly two clean builds");
  }
  const builds = evidence.builds.map((build, index) =>
    validateBuild(build, index));
  for (const field of [
    "factoryModuleByteLength",
    "factoryModuleSha256",
    "linkMapByteLength",
    "linkMapCanonicalByteLength",
    "linkMapCanonicalSha256",
    "runtimeAbiReviewByteLength",
    "runtimeAbiReviewSha256",
    "runtimeClosureObservationByteLength",
    "runtimeClosureObservationSha256",
    "runtimeClosureSha256",
    "wasmByteLength",
    "wasmSha256",
  ]) {
    exactValue(
      builds[1][field],
      builds[0][field],
      `$.evidence.builds[1].${field}`,
    );
  }
  exactValue(
    JSON.stringify(builds[1].nativeTools),
    JSON.stringify(builds[0].nativeTools),
    "$.evidence.builds[1].nativeTools",
  );
  canonicalJsonBytes(evidence);
}

function validateBuild(value, index) {
  const path = `$.evidence.builds[${index}]`;
  const build = exactObject(value, BUILD_KEYS, path);
  exactValue(build.ordinal, index + 1, `${path}.ordinal`);
  exactValue(
    build.runtimeAbiReviewExactInterfaceConformance,
    true,
    `${path}.runtimeAbiReviewExactInterfaceConformance`,
  );
  for (const field of BUILD_KEYS.filter((key) =>
    key.endsWith("Sha256"))) {
    sha256(build[field], `${path}.${field}`);
  }
  for (const field of BUILD_KEYS.filter((key) =>
    key.endsWith("ByteLength"))) {
    positiveInteger(build[field], `${path}.${field}`);
  }
  const nativeTools = exactObject(
    build.nativeTools,
    NATIVE_TOOLS_KEYS,
    `${path}.nativeTools`,
  );
  for (const prefix of ["clangTablegen", "llvmTablegen"]) {
    sha256(
      nativeTools[`${prefix}Sha256`],
      `${path}.nativeTools.${prefix}Sha256`,
    );
    positiveInteger(
      nativeTools[`${prefix}ByteLength`],
      `${path}.nativeTools.${prefix}ByteLength`,
    );
  }
  return build;
}

async function readCanonicalEvidence(path) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    invalid("$.input", "input is unavailable", { cause });
  }
  if (!entry.isFile() || entry.isSymbolicLink() ||
      entry.size <= 0 || entry.size > MAX_INPUT_BYTE_LENGTH ||
      await realpath(path) !== path) {
    invalid(
      "$.input",
      "input must be one canonical bounded non-symlink regular file",
    );
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat();
    if (!before.isFile() || before.dev !== entry.dev ||
        before.ino !== entry.ino || before.size !== entry.size) {
      invalid("$.input", "input identity changed before read");
    }
    const bytes = await handle.readFile();
    const after = await handle.stat();
    if (after.dev !== before.dev || after.ino !== before.ino ||
        after.size !== before.size || after.mtimeMs !== before.mtimeMs) {
      invalid("$.input", "input identity changed during read");
    }
    let parsed;
    try {
      parsed = JSON.parse(UTF8_DECODER.decode(bytes));
    } catch (cause) {
      invalid("$.input", "input must be canonical UTF-8 JSON", { cause });
    }
    if (!equalBytes(bytes, canonicalJsonBytes(parsed))) {
      invalid("$.input", "input bytes are not canonical JSON");
    }
    return parsed;
  } finally {
    await handle?.close();
  }
}

function exactCurrentRunId() {
  if (CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID !==
      CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID) {
    invalid("$.package.runId", "package build and verifier run IDs differ");
  }
  return CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID;
}

function exactCurrentSourceRevision() {
  if (CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION !==
      CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION) {
    invalid(
      "$.package.sourceRevision",
      "package build and verifier revisions differ",
    );
  }
  return CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION;
}

function exactProjection(value) {
  const projection = exactObject(value, [
    "authority",
    "evidence",
    "resourceByteLength",
    "resourceSha256",
    "runId",
    "schema",
    "sourceRevision",
    "version",
    "wasmByteLength",
    "wasmSha256",
  ], "$.projection");
  exactValue(
    projection.schema,
    "browsergrad.compiler.cpp-cute.reproducibility-authoring-projection",
    "$.projection.schema",
  );
  exactValue(projection.version, 1, "$.projection.version");
  exactValue(
    projection.authority,
    "package-authoring-projection-only",
    "$.projection.authority",
  );
  validateEvidence(projection.evidence);
  sha256(projection.resourceSha256, "$.projection.resourceSha256");
  positiveInteger(
    projection.resourceByteLength,
    "$.projection.resourceByteLength",
  );
  if (!RUN_ID.test(projection.runId)) {
    invalid("$.projection.runId", "invalid run ID");
  }
  if (!SOURCE_REVISION.test(projection.sourceRevision)) {
    invalid("$.projection.sourceRevision", "invalid source revision");
  }
  sha256(projection.wasmSha256, "$.projection.wasmSha256");
  positiveInteger(
    projection.wasmByteLength,
    "$.projection.wasmByteLength",
  );
  const resourceBytes = canonicalJsonBytes(projection.evidence);
  exactValue(
    digest(resourceBytes),
    projection.resourceSha256,
    "$.projection.resourceSha256",
  );
  exactValue(
    resourceBytes.byteLength,
    projection.resourceByteLength,
    "$.projection.resourceByteLength",
  );
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected one plain object");
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length ||
      actual.some((key, index) => key !== expected[index])) {
    invalid(path, `expected exactly keys ${expected.join(", ")}`);
  }
  return value;
}

function exactString(value, path) {
  if (typeof value !== "string") invalid(path, "expected one string");
  return value;
}

function sha256(value, path) {
  if (typeof value !== "string" || !SHA256.test(value)) {
    invalid(path, "expected one lowercase SHA-256 digest");
  }
}

function positiveInteger(value, path) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    invalid(path, "expected one positive safe integer");
  }
}

function exactValue(actual, expected, path) {
  if (actual !== expected) {
    invalid(path, `expected ${JSON.stringify(expected)}`);
  }
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function equalBytes(left, right) {
  return left.byteLength === right.byteLength &&
    left.every((byte, index) => byte === right[index]);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserReproducibilityAuthoringError(
    path,
    message,
    options,
  );
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  run(process.argv.slice(2)).catch((cause) => {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown reproducibility authoring failure");
    process.stderr.write(`${error.stack ?? error.message}\n`);
    process.exitCode = 1;
  });
}
