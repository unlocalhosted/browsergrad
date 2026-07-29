import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  readFile,
  realpath,
  writeFile,
} from "node:fs/promises";
import {
  dirname,
  isAbsolute,
  resolve,
} from "node:path";
import {
  fileURLToPath,
  pathToFileURL,
} from "node:url";

import {
  canonicalJsonBytes,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  cppCuteBrowserHeaderInputProjectionId,
  decodeCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V2_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_header_distribution_reproducibility_v2.js";
import {
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
} from "../../dist/resources/cpp_cute_browser_header_distribution_reproducibility_identity_v1.js";
import {
  verifyCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-DISTRIBUTION-REPRODUCIBILITY-AUTHORING";
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIRECTORY = dirname(SCRIPT_PATH);
const PACKAGE_ROOT = resolve(SCRIPT_DIRECTORY, "../..");
const RESOURCE_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_header_distribution_reproducibility_v2.ts",
);
const IDENTITY_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_header_distribution_reproducibility_identity_v1.ts",
);
const MAX_INPUT_BYTE_LENGTH = 64 * 1024;
const SOURCE_REVISION = /^[0-9a-f]{40}$/u;
const SHA256 = /^[0-9a-f]{64}$/u;
const WIRE_U64 = /^(?:0|[1-9][0-9]*)$/u;
const RESOURCE_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility";
const LIVE_AUTHORITY =
  "two-root-exact-header-distribution-reproducibility-only";
const RESOURCE_AUTHORITY =
  "two-root-exact-header-distribution-reproducibility-observation-only";
const SCOPE =
  "five-header-packs-license-inventory-and-notice-outputs-only";
const OUTPUT_VERIFICATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.distribution-output-file-verification.v1";
const REPRODUCIBILITY_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility.v2";
const EXPECTED_OUTPUT_COUNT = 17;
const EXPECTED_OUTPUT_BYTE_LENGTH = "71114743";
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

const CLAIMS = Object.freeze({
  twoDistinctPrivateOutputRootsVerified: true,
  exactOutputsRehashedInBothRoots: true,
  exactHeaderDistributionOutputSetReproducible: true,
  fullDistributedOutputSetReproducible: false,
  externalDistributedFileLicenseMapReviewed: false,
  licenseReviewComplete: false,
  distributionAuthorized: false,
  signedProvenanceVerified: false,
  workerExecutionObserved: false,
  releaseReady: false,
});

export class CppCuteBrowserHeaderDistributionReproducibilityAuthoringError
  extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name =
      "CppCuteBrowserHeaderDistributionReproducibilityAuthoringError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserHeaderDistributionReproducibilityAuthoringArguments(
  argv,
) {
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length === 1 && arguments_[0] === "--check") {
    return Object.freeze({
      check: true,
      inputPath: undefined,
      sourceRevision: undefined,
    });
  }
  if (arguments_.length !== 2) {
    invalid(
      "$.arguments",
      "expected --check or one absolute input and one source revision",
    );
  }
  const values = new Map();
  for (const [index, argument] of arguments_.entries()) {
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      invalid(`$.arguments[${index}]`, "expected one --name=value option");
    }
    const name = argument.slice(2, separator);
    if (name !== "input" && name !== "source-revision") {
      invalid(`$.arguments[${index}]`, `unsupported option --${name}`);
    }
    if (values.has(name)) {
      invalid(`$.arguments[${index}]`, `duplicate option --${name}`);
    }
    values.set(name, argument.slice(separator + 1));
  }
  const inputPath = values.get("input");
  const sourceRevision = values.get("source-revision");
  if (typeof inputPath !== "string" || !isAbsolute(inputPath) ||
      inputPath.includes("\0")) {
    invalid("$.arguments.input", "input requires one absolute NUL-free path");
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
    sourceRevision,
  });
}

/**
 * Reauthenticates both private output roots before projecting away their
 * machine-local paths. The resulting resource remains observation-only.
 */
export async function projectCppCuteBrowserHeaderDistributionReproducibility(
  value,
  sourceRevision,
) {
  if (typeof sourceRevision !== "string" ||
      !SOURCE_REVISION.test(sourceRevision)) {
    invalid(
      "$.sourceRevision",
      "expected one lowercase 40-hex source revision",
    );
  }
  const live = validateLiveReport(value);
  const buildLock = await currentBuildLock();
  await validateCurrentHeaderBinding(live, buildLock);

  let first;
  let second;
  try {
    [first, second] = await Promise.all([
      verifyCppCuteBrowserDistributionOutputFiles({
        outputRoot: live.firstOutputRoot,
        expectedOutputs: live.outputs,
      }),
      verifyCppCuteBrowserDistributionOutputFiles({
        outputRoot: live.secondOutputRoot,
        expectedOutputs: live.outputs,
      }),
    ]);
  } catch (cause) {
    invalid(
      "$.roots",
      "failed to rehash both exact private output trees",
      { cause },
    );
  }
  if (first.outputRoot === second.outputRoot ||
      first.verificationId !== live.outputVerificationId ||
      second.verificationId !== live.outputVerificationId ||
      !equalBytes(
        canonicalJsonBytes(first.outputs),
        canonicalJsonBytes(live.outputs),
      ) ||
      !equalBytes(
        canonicalJsonBytes(second.outputs),
        canonicalJsonBytes(live.outputs),
      )) {
    invalid(
      "$.roots",
      "live roots do not reproduce the exact reported output set",
    );
  }

  return projectResource(live, sourceRevision);
}

export function renderCppCuteBrowserHeaderDistributionReproducibilityResource(
  projection,
) {
  exactProjection(projection);
  const value = JSON.stringify(projection.resource, null, 2);
  return `import {\n` +
    `  deepFreezeJson,\n` +
    `  type JsonObject,\n` +
    `} from "@unlocalhosted/browsergrad-semantic-core/schema";\n\n` +
    `export interface CppCuteBrowserHeaderDistributionReproducibilityOutputV2 extends JsonObject {\n` +
    `  readonly outputPath: string;\n` +
    `  readonly sha256: string;\n` +
    `  readonly byteLength: string;\n` +
    `}\n\n` +
    `/** Generated by cpp_cute_browser_header_distribution_reproducibility_authoring.mjs. */\n` +
    `const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V2_VALUE = ${value} as const satisfies JsonObject;\n\n` +
    `export type CppCuteBrowserHeaderDistributionReproducibilityResourceV2 =\n` +
    `  typeof CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V2_VALUE;\n\n` +
    `export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V2_RESOURCE =\n` +
    `  deepFreezeJson(\n` +
    `    CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V2_VALUE,\n` +
    `  ) as unknown as CppCuteBrowserHeaderDistributionReproducibilityResourceV2;\n`;
}

export function renderCppCuteBrowserHeaderDistributionReproducibilityIdentity(
  projection,
) {
  exactProjection(projection);
  return `/** Generated by cpp_cute_browser_header_distribution_reproducibility_authoring.mjs. */\n` +
    `export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256 =\n` +
    `  ${JSON.stringify(projection.resourceSha256)};\n` +
    `export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH = ${projection.resourceByteLength};\n` +
    `export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION =\n` +
    `  ${JSON.stringify(projection.sourceRevision)};\n` +
    `export const CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID =\n` +
    `  ${JSON.stringify(projection.headerInputProjectionId)};\n`;
}

async function currentBuildLock() {
  return decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
}

async function validateCurrentHeaderBinding(value, buildLock) {
  const projectionId = await cppCuteBrowserHeaderInputProjectionId(buildLock);
  if (value.headerInputProjectionId !== projectionId) {
    invalid(
      "$.headerInputProjectionId",
      "report does not bind the current package header-input projection",
    );
  }
  if (value.buildInputLockId !== buildLock.lockId ||
      value.buildInputLockResourceSha256 !== buildLock.resourceSha256) {
    invalid(
      "$.buildInputLockId",
      "live authoring requires the current exact package build-input lock",
    );
  }
}

function validateLiveReport(value) {
  const object = exactObject(value, [
    "authority",
    "buildInputLockId",
    "buildInputLockResourceSha256",
    "claims",
    "firstOutputRoot",
    "headerInputProjectionId",
    "outputVerificationId",
    "outputs",
    "pipelineId",
    "reproducibilityId",
    "schema",
    "scope",
    "secondOutputRoot",
    "totals",
    "version",
  ], "$");
  if (object.authority !== LIVE_AUTHORITY) {
    invalid("$.authority", "expected exact live two-root authority");
  }
  const firstOutputRoot = absolutePath(
    object.firstOutputRoot,
    "$.firstOutputRoot",
  );
  const secondOutputRoot = absolutePath(
    object.secondOutputRoot,
    "$.secondOutputRoot",
  );
  if (firstOutputRoot === secondOutputRoot) {
    invalid("$.secondOutputRoot", "output roots must be distinct");
  }
  return validateCommonReport({
    ...object,
    firstOutputRoot,
    secondOutputRoot,
  });
}

function validateStaticResource(value) {
  const object = exactObject(value, [
    "authority",
    "buildInputLockId",
    "buildInputLockResourceSha256",
    "claims",
    "headerInputProjectionId",
    "outputVerificationId",
    "outputs",
    "pipelineId",
    "reproducibilityId",
    "schema",
    "scope",
    "totals",
    "verifierSourceRevision",
    "version",
  ], "$");
  if (object.authority !== RESOURCE_AUTHORITY) {
    invalid("$.authority", "expected exact package observation authority");
  }
  if (typeof object.verifierSourceRevision !== "string" ||
      !SOURCE_REVISION.test(object.verifierSourceRevision)) {
    invalid("$.verifierSourceRevision", "invalid verifier source revision");
  }
  return validateCommonReport(object);
}

function validateCommonReport(object) {
  if (object.schema !== RESOURCE_SCHEMA || object.version !== 2 ||
      object.scope !== SCOPE) {
    invalid("$", "unexpected schema, version, or scope");
  }
  for (const [field, prefix] of [
    ["buildInputLockId", "bg.cpp.browser-build-input-lock.sha256."],
    [
      "headerInputProjectionId",
      "bg.cpp.browser-header-input-projection.sha256.",
    ],
    ["pipelineId", "bg.cpp.browser-header-pack-pipeline.sha256."],
    [
      "outputVerificationId",
      "bg.cpp.distribution-output-file-verification.sha256.",
    ],
    [
      "reproducibilityId",
      "bg.cpp.browser-header-distribution-reproducibility.sha256.",
    ],
  ]) {
    if (typeof object[field] !== "string" ||
        !object[field].startsWith(prefix) ||
        !SHA256.test(object[field].slice(prefix.length))) {
      invalid(`$.${field}`, "invalid content identity");
    }
  }
  if (typeof object.buildInputLockResourceSha256 !== "string" ||
      !SHA256.test(object.buildInputLockResourceSha256)) {
    invalid(
      "$.buildInputLockResourceSha256",
      "invalid build-lock resource digest",
    );
  }
  const outputs = validateOutputs(object.outputs);
  const totals = exactObject(
    object.totals,
    ["byteLength", "outputCount"],
    "$.totals",
  );
  if (totals.outputCount !== EXPECTED_OUTPUT_COUNT ||
      totals.byteLength !== EXPECTED_OUTPUT_BYTE_LENGTH) {
    invalid("$.totals", "unexpected exact output totals");
  }
  const claims = exactObject(
    object.claims,
    Object.keys(CLAIMS),
    "$.claims",
  );
  if (!equalBytes(canonicalJsonBytes(claims), canonicalJsonBytes(CLAIMS))) {
    invalid("$.claims", "unexpected authority claims");
  }
  const outputVerificationId =
    `bg.cpp.distribution-output-file-verification.sha256.${digest(
      canonicalJsonBytes({
        domain: OUTPUT_VERIFICATION_HASH_DOMAIN,
        outputs,
      }),
    )}`;
  if (object.outputVerificationId !== outputVerificationId) {
    invalid(
      "$.outputVerificationId",
      "identity does not match exact outputs",
    );
  }
  const reproducibilityId =
    `bg.cpp.browser-header-distribution-reproducibility.sha256.${digest(
      canonicalJsonBytes({
        domain: REPRODUCIBILITY_HASH_DOMAIN,
        pipelineId: object.pipelineId,
        outputVerificationId,
        outputs,
      }),
    )}`;
  if (object.reproducibilityId !== reproducibilityId) {
    invalid("$.reproducibilityId", "identity does not match exact outputs");
  }
  return Object.freeze({
    ...object,
    outputs: Object.freeze(outputs),
    totals: Object.freeze({
      outputCount: EXPECTED_OUTPUT_COUNT,
      byteLength: EXPECTED_OUTPUT_BYTE_LENGTH,
    }),
    claims: CLAIMS,
  });
}

function validateOutputs(value) {
  if (!Array.isArray(value) || value.length !== EXPECTED_OUTPUT_COUNT) {
    invalid("$.outputs", "expected exactly 17 outputs");
  }
  let total = 0n;
  const outputs = value.map((item, index) => {
    const path = `$.outputs[${index}]`;
    const output = exactObject(
      item,
      ["byteLength", "outputPath", "sha256"],
      path,
    );
    if (typeof output.outputPath !== "string" ||
        !/^(?:[A-Za-z0-9._+@=-]+\/)*[A-Za-z0-9._+@=-]+$/u
          .test(output.outputPath) ||
        typeof output.sha256 !== "string" ||
        !SHA256.test(output.sha256) ||
        typeof output.byteLength !== "string" ||
        !WIRE_U64.test(output.byteLength)) {
      invalid(path, "invalid output identity");
    }
    if (index > 0 &&
        compareUtf8(value[index - 1]?.outputPath ?? "", output.outputPath) >= 0) {
      invalid(`${path}.outputPath`, "outputs must be unique UTF-8 order");
    }
    total += BigInt(output.byteLength);
    return Object.freeze({
      outputPath: output.outputPath,
      sha256: output.sha256,
      byteLength: output.byteLength,
    });
  });
  if (total.toString() !== EXPECTED_OUTPUT_BYTE_LENGTH) {
    invalid("$.outputs", "output byte total differs from exact observation");
  }
  return outputs;
}

function projectResource(value, sourceRevision) {
  const resource = Object.freeze({
    schema: RESOURCE_SCHEMA,
    version: 2,
    authority: RESOURCE_AUTHORITY,
    scope: SCOPE,
    verifierSourceRevision: sourceRevision,
    buildInputLockId: value.buildInputLockId,
    buildInputLockResourceSha256: value.buildInputLockResourceSha256,
    headerInputProjectionId: value.headerInputProjectionId,
    pipelineId: value.pipelineId,
    outputVerificationId: value.outputVerificationId,
    reproducibilityId: value.reproducibilityId,
    outputs: value.outputs,
    totals: value.totals,
    claims: value.claims,
  });
  const bytes = canonicalJsonBytes(resource);
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility-authoring-projection",
    version: 1,
    authority: "package-authoring-projection-only",
    resource,
    resourceSha256: digest(bytes),
    resourceByteLength: bytes.byteLength,
    sourceRevision,
    headerInputProjectionId: value.headerInputProjectionId,
  });
}

function exactProjection(value) {
  if (value?.schema !==
        "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility-authoring-projection" ||
      value.version !== 1 ||
      value.authority !== "package-authoring-projection-only" ||
      typeof value.resourceSha256 !== "string" ||
      !SHA256.test(value.resourceSha256) ||
      !Number.isSafeInteger(value.resourceByteLength) ||
      value.resourceByteLength <= 0 ||
      typeof value.sourceRevision !== "string" ||
      !SOURCE_REVISION.test(value.sourceRevision) ||
      typeof value.headerInputProjectionId !== "string") {
    invalid("$.projection", "expected one bounded authoring projection");
  }
}

async function readStableJson(path) {
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
    if (!after.isFile() || after.dev !== before.dev ||
        after.ino !== before.ino || after.size !== before.size ||
        after.mtimeMs !== before.mtimeMs ||
        after.ctimeMs !== before.ctimeMs) {
      invalid("$.input", "input changed during read");
    }
    try {
      return JSON.parse(UTF8_DECODER.decode(bytes));
    } catch (cause) {
      invalid("$.input", "input is not strict UTF-8 JSON", { cause });
    }
  } finally {
    await handle?.close();
  }
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected one plain data record");
  }
  const actual = Reflect.ownKeys(value);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected only ${keys.join(", ")}`);
  }
  return value;
}

function absolutePath(value, path) {
  if (typeof value !== "string" || !isAbsolute(value) ||
      value.includes("\0") || resolve(value) !== value) {
    invalid(path, "expected one canonical absolute NUL-free path");
  }
  return value;
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function equalBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderDistributionReproducibilityAuthoringError(
    path,
    message,
    options,
  );
}

async function main() {
  const options =
    parseCppCuteBrowserHeaderDistributionReproducibilityAuthoringArguments(
      process.argv.slice(2),
    );
  let projection;
  if (options.check) {
    const resource = validateStaticResource(
      CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V2_RESOURCE,
    );
    const buildLock = await currentBuildLock();
    const currentHeaderInputProjectionId =
      await cppCuteBrowserHeaderInputProjectionId(buildLock);
    if (resource.headerInputProjectionId !== currentHeaderInputProjectionId) {
      invalid(
        "$.headerInputProjectionId",
        "checked-in resource does not bind current header inputs",
      );
    }
    projection = projectResource(
      resource,
      CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
    );
  } else {
    const value = await readStableJson(options.inputPath);
    projection =
      await projectCppCuteBrowserHeaderDistributionReproducibility(
        value,
        options.sourceRevision,
      );
  }
  const resource =
    renderCppCuteBrowserHeaderDistributionReproducibilityResource(projection);
  const identity =
    renderCppCuteBrowserHeaderDistributionReproducibilityIdentity(projection);
  if (options.check) {
    if (await readFile(RESOURCE_PATH, "utf8") !== resource ||
        await readFile(IDENTITY_PATH, "utf8") !== identity) {
      invalid(
        "$.resources",
        "checked-in header reproducibility resources are stale",
      );
    }
  } else {
    await writeFile(RESOURCE_PATH, resource, { encoding: "utf8", mode: 0o644 });
    await writeFile(IDENTITY_PATH, identity, { encoding: "utf8", mode: 0o644 });
  }
  process.stdout.write(`${JSON.stringify({
    schema: projection.schema,
    version: projection.version,
    authority: projection.authority,
    checked: options.check,
    resourceSha256: projection.resourceSha256,
    resourceByteLength: projection.resourceByteLength,
    sourceRevision: projection.sourceRevision,
    headerInputProjectionId: projection.headerInputProjectionId,
  })}\n`);
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((cause) => {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown header reproducibility authoring failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  });
}
