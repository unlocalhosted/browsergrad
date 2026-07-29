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
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from "../../dist/cpp_cute_browser_full_distribution_reproducibility.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import {
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_full_distribution_reproducibility_v1.js";
import {
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION,
} from "../../dist/resources/cpp_cute_browser_full_distribution_reproducibility_identity_v1.js";
import {
  admitCppCuteBrowserDeterministicDistribution,
} from "./cpp_cute_browser_full_distribution_materialization.mjs";
import {
  observeCppCuteBrowserFullDistributionReproducibility,
} from "./cpp_cute_browser_full_distribution_reproducibility.mjs";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-REPRODUCIBILITY-AUTHORING";
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIRECTORY = dirname(SCRIPT_PATH);
const PACKAGE_ROOT = resolve(SCRIPT_DIRECTORY, "../..");
const RESOURCE_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_full_distribution_reproducibility_v1.ts",
);
const IDENTITY_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_full_distribution_reproducibility_identity_v1.ts",
);
const MAX_INPUT_BYTE_LENGTH = 64 * 1024;
const SOURCE_REVISION = /^[0-9a-f]{40}$/u;
const SHA256 = /^[0-9a-f]{64}$/u;
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });
const RESOURCE_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility-observation";

export class CppCuteBrowserFullDistributionReproducibilityAuthoringError
  extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name =
      "CppCuteBrowserFullDistributionReproducibilityAuthoringError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserFullDistributionReproducibilityAuthoringArguments(
  argv,
) {
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length === 1 && arguments_[0] === "--check") {
    return Object.freeze({
      check: true,
      deterministicRoot: undefined,
      evidencePath: undefined,
      producerPolicyPath: undefined,
      profilePath: undefined,
      sourceRevision: undefined,
    });
  }
  if (arguments_.length !== 5) {
    invalid(
      "$.arguments",
      "expected --check or evidence, deterministic root, profile, producer policy, and source revision",
    );
  }
  const names = new Set([
    "deterministic-root",
    "evidence",
    "producer-policy",
    "profile",
    "source-revision",
  ]);
  const values = new Map();
  for (const [index, argument] of arguments_.entries()) {
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      invalid(`$.arguments[${index}]`, "expected one --name=value option");
    }
    const name = argument.slice(2, separator);
    if (!names.has(name) || values.has(name)) {
      invalid(`$.arguments[${index}]`, "unsupported or duplicate option");
    }
    values.set(name, argument.slice(separator + 1));
  }
  const deterministicRoot = absolutePath(
    values.get("deterministic-root"),
    "$.arguments.deterministicRoot",
  );
  const evidencePath = absolutePath(
    values.get("evidence"),
    "$.arguments.evidence",
  );
  const producerPolicyPath = absolutePath(
    values.get("producer-policy"),
    "$.arguments.producerPolicy",
  );
  const profilePath = absolutePath(
    values.get("profile"),
    "$.arguments.profile",
  );
  const sourceRevision = values.get("source-revision");
  if (typeof sourceRevision !== "string" ||
      !SOURCE_REVISION.test(sourceRevision)) {
    invalid(
      "$.arguments.sourceRevision",
      "source-revision requires one lowercase 40-hex revision",
    );
  }
  return Object.freeze({
    check: false,
    deterministicRoot,
    evidencePath,
    producerPolicyPath,
    profilePath,
    sourceRevision,
  });
}

export async function projectCppCuteBrowserFullDistributionReproducibility(
  evidence,
  inputs,
) {
  const metadata = exactObject(
    inputs,
    [
      "deterministicRoot",
      "producerPolicyBytes",
      "profileBytes",
      "sourceRevision",
    ],
    "$.inputs",
  );
  const sourceRevision = metadata.sourceRevision;
  if (typeof sourceRevision !== "string" ||
      !SOURCE_REVISION.test(sourceRevision)) {
    invalid("$.inputs.sourceRevision", "invalid source revision");
  }
  const firstOutputRoot = absolutePath(
    evidence?.firstOutputRoot,
    "$.evidence.firstOutputRoot",
  );
  const secondOutputRoot = absolutePath(
    evidence?.secondOutputRoot,
    "$.evidence.secondOutputRoot",
  );
  const observed =
    await observeCppCuteBrowserFullDistributionReproducibility({
      firstOutputRoot,
      secondOutputRoot,
    });
  if (!equalBytes(
    canonicalJsonBytes(observed),
    canonicalJsonBytes(evidence),
  )) {
    invalid(
      "$.evidence",
      "input differs from a fresh exact two-root observation",
    );
  }

  let producerPolicy;
  let deterministic;
  try {
    producerPolicy = await admitCppCuteBrowserProducerTrustPolicy(
      metadata.producerPolicyBytes,
    );
    deterministic = await admitCppCuteBrowserDeterministicDistribution({
      outputRoot: absolutePath(
        metadata.deterministicRoot,
        "$.inputs.deterministicRoot",
      ),
      producerTrustPolicy: producerPolicy,
      profileBytes: metadata.profileBytes,
    });
  } catch (cause) {
    invalid(
      "$.inputs.deterministicRoot",
      "failed to reauthenticate current deterministic distribution metadata",
      { cause },
    );
  }
  const deterministicIdentities = deterministic.outputs.map((output) => ({
    outputPath: output.outputPath,
    sha256: output.sha256,
    byteLength: output.byteLength,
  })).sort(compareOutputPath);
  const observedIdentities = observed.deterministicOutputs.map((output) => ({
    outputPath: output.outputPath,
    sha256: output.sha256,
    byteLength: output.byteLength,
  })).sort(compareOutputPath);
  if (!equalBytes(
    canonicalJsonBytes(deterministicIdentities),
    canonicalJsonBytes(observedIdentities),
  ) ||
      deterministic.buildInputLockId !== observed.buildInputLockId ||
      deterministic.buildInputLockResourceSha256 !==
        observed.buildInputLockResourceSha256 ||
      deterministic.buildSubjectId !==
        observed.detachedEvidence.buildSubjectId ||
      deterministic.buildSubjectSha256 !==
        observed.detachedEvidence.buildSubjectSha256) {
    invalid(
      "$.inputs.deterministicRoot",
      "deterministic metadata does not bind the exact full-distribution observation",
    );
  }

  const resource = Object.freeze({
    schema: RESOURCE_SCHEMA,
    version: 1,
    authority:
      "two-root-exact-full-distribution-reproducibility-observation-only",
    verifierSourceRevision: sourceRevision,
    materializerSourceRevision: sourceRevision,
    producerPolicyScope: "local-engineering-reproducibility-only",
    buildInputLockId: observed.buildInputLockId,
    buildInputLockResourceSha256:
      observed.buildInputLockResourceSha256,
    reproducibilityId: observed.reproducibilityId,
    deterministicMetadata: Object.freeze({
      metadataId: deterministic.metadataId,
      profileId: deterministic.profileId,
      profileHash: deterministic.profileHash,
      compilationContractHash: deterministic.compilationContractHash,
      profileSha256: deterministic.profileSha256,
      profileByteLength: deterministic.profileByteLength,
      assetManifestId: deterministic.assetManifestId,
      assetManifestSha256: deterministic.assetManifestSha256,
      assetManifestByteLength: deterministic.assetManifestByteLength,
      assetSetSha256: deterministic.assetSetSha256,
      buildSubjectId: deterministic.buildSubjectId,
      buildSubjectSha256: deterministic.buildSubjectSha256,
      wasmSha256: deterministic.wasmSha256,
      wasmByteLength: deterministic.wasmByteLength,
      workerBundleSha256: deterministic.workerBundleSha256,
      headerDistributionReproducibilityId:
        deterministic.headerDistributionReproducibilityId,
      headerDistributionOutputVerificationId:
        deterministic.headerDistributionOutputVerificationId,
    }),
    deterministicOutputs: observed.deterministicOutputs,
    detachedEvidence: observed.detachedEvidence,
    totals: observed.totals,
    claims: Object.freeze({
      twoDistinctPrivateOutputRootsVerified: true,
      exactBuildLockOutputPlanMatched: true,
      exactOutputsRehashedInBothRoots: true,
      deterministicSubjectsByteIdentical: true,
      detachedEvidenceBuildSubjectMatched: true,
      fullDistributedOutputSetReproducible: true,
      detachedSignatureVerified: false,
      externallyRootedProducerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    }),
  });
  return projection(resource, sourceRevision);
}

export function renderCppCuteBrowserFullDistributionReproducibilityResource(
  value,
) {
  exactProjection(value);
  return `import {\n` +
    `  deepFreezeJson,\n` +
    `  type JsonObject,\n` +
    `} from "@unlocalhosted/browsergrad-semantic-core/schema";\n\n` +
    `export interface CppCuteBrowserFullDistributionReproducibilityOutputV1 extends JsonObject {\n` +
    `  readonly outputPath: string;\n` +
    `  readonly role: string;\n` +
    `  readonly mediaType: string;\n` +
    `  readonly sha256: string;\n` +
    `  readonly byteLength: string;\n` +
    `}\n\n` +
    `/** Generated by cpp_cute_browser_full_distribution_reproducibility_authoring.mjs. */\n` +
    `const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_VALUE = ${JSON.stringify(value.resource, null, 2)} as const satisfies JsonObject;\n\n` +
    `export type CppCuteBrowserFullDistributionReproducibilityResourceV1 =\n` +
    `  typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_VALUE;\n\n` +
    `export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE =\n` +
    `  deepFreezeJson(\n` +
    `    CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_VALUE,\n` +
    `  ) as unknown as CppCuteBrowserFullDistributionReproducibilityResourceV1;\n`;
}

export function renderCppCuteBrowserFullDistributionReproducibilityIdentity(
  value,
) {
  exactProjection(value);
  return `/** Generated by cpp_cute_browser_full_distribution_reproducibility_authoring.mjs. */\n` +
    `export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256 =\n` +
    `  ${JSON.stringify(value.resourceSha256)};\n` +
    `export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH = ${value.resourceByteLength};\n` +
    `export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION =\n` +
    `  ${JSON.stringify(value.sourceRevision)};\n`;
}

function projection(resource, sourceRevision) {
  const bytes = canonicalJsonBytes(resource);
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility-authoring-projection",
    version: 1,
    authority: "package-authoring-projection-only",
    resource,
    resourceSha256: digest(bytes),
    resourceByteLength: bytes.byteLength,
    sourceRevision,
  });
}

function exactProjection(value) {
  if (value?.schema !==
        "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility-authoring-projection" ||
      value.version !== 1 ||
      value.authority !== "package-authoring-projection-only" ||
      typeof value.resourceSha256 !== "string" ||
      !SHA256.test(value.resourceSha256) ||
      !Number.isSafeInteger(value.resourceByteLength) ||
      value.resourceByteLength <= 0 ||
      typeof value.sourceRevision !== "string" ||
      !SOURCE_REVISION.test(value.sourceRevision)) {
    invalid("$.projection", "expected one bounded authoring projection");
  }
}

async function readStableBytes(path, maximumByteLength, diagnosticPath) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    invalid(diagnosticPath, "input is unavailable", { cause });
  }
  if (!entry.isFile() || entry.isSymbolicLink() ||
      entry.size <= 0 || entry.size > maximumByteLength ||
      await realpath(path) !== path) {
    invalid(
      diagnosticPath,
      "input must be one canonical bounded non-symlink regular file",
    );
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat();
    if (!before.isFile() || before.dev !== entry.dev ||
        before.ino !== entry.ino || before.size !== entry.size) {
      invalid(diagnosticPath, "input identity changed before read");
    }
    const bytes = new Uint8Array(await handle.readFile());
    const after = await handle.stat();
    if (!after.isFile() || after.dev !== before.dev ||
        after.ino !== before.ino || after.size !== before.size ||
        after.mtimeMs !== before.mtimeMs ||
        after.ctimeMs !== before.ctimeMs) {
      invalid(diagnosticPath, "input changed during read");
    }
    return bytes;
  } finally {
    await handle?.close();
  }
}

async function readStableJson(path) {
  const bytes = await readStableBytes(
    path,
    MAX_INPUT_BYTE_LENGTH,
    "$.evidence",
  );
  try {
    const value = JSON.parse(UTF8_DECODER.decode(bytes));
    if (!equalBytes(bytes, canonicalJsonBytes(value))) {
      invalid("$.evidence", "evidence must use exact canonical JSON");
    }
    return value;
  } catch (cause) {
    if (cause instanceof
        CppCuteBrowserFullDistributionReproducibilityAuthoringError) {
      throw cause;
    }
    invalid("$.evidence", "evidence is not strict UTF-8 JSON", { cause });
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

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function compareOutputPath(left, right) {
  return Buffer.compare(
    Buffer.from(left.outputPath, "utf8"),
    Buffer.from(right.outputPath, "utf8"),
  );
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
  throw new CppCuteBrowserFullDistributionReproducibilityAuthoringError(
    path,
    message,
    options,
  );
}

async function main() {
  const options =
    parseCppCuteBrowserFullDistributionReproducibilityAuthoringArguments(
      process.argv.slice(2),
    );
  let value;
  if (options.check) {
    await verifyCppCuteBrowserFullDistributionReproducibilityResource(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
    value = projection(
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION,
    );
  } else {
    const [
      evidence,
      producerPolicyBytes,
      profileBytes,
    ] = await Promise.all([
      readStableJson(options.evidencePath),
      readStableBytes(
        options.producerPolicyPath,
        MAX_INPUT_BYTE_LENGTH,
        "$.producerPolicy",
      ),
      readStableBytes(
        options.profilePath,
        MAX_INPUT_BYTE_LENGTH,
        "$.profile",
      ),
    ]);
    value = await projectCppCuteBrowserFullDistributionReproducibility(
      evidence,
      {
        deterministicRoot: options.deterministicRoot,
        producerPolicyBytes,
        profileBytes,
        sourceRevision: options.sourceRevision,
      },
    );
  }
  const resource =
    renderCppCuteBrowserFullDistributionReproducibilityResource(value);
  const identity =
    renderCppCuteBrowserFullDistributionReproducibilityIdentity(value);
  if (options.check) {
    if (await readFile(RESOURCE_PATH, "utf8") !== resource ||
        await readFile(IDENTITY_PATH, "utf8") !== identity) {
      invalid(
        "$.resources",
        "checked-in full-distribution reproducibility resources are stale",
      );
    }
  } else {
    await writeFile(RESOURCE_PATH, resource, { encoding: "utf8", mode: 0o644 });
    await writeFile(IDENTITY_PATH, identity, { encoding: "utf8", mode: 0o644 });
  }
  process.stdout.write(`${JSON.stringify({
    schema: value.schema,
    version: value.version,
    authority: value.authority,
    checked: options.check,
    resourceSha256: value.resourceSha256,
    resourceByteLength: value.resourceByteLength,
    sourceRevision: value.sourceRevision,
    reproducibilityId: value.resource.reproducibilityId,
  })}\n`);
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((cause) => {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown full-distribution authoring failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  });
}
