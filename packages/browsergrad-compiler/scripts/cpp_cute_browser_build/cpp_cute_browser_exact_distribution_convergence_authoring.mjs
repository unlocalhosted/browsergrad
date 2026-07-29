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
  cppCuteBrowserExactDistributionConvergenceResourceBytes,
  verifyCppCuteBrowserExactDistributionConvergenceResource,
} from "../../dist/cpp_cute_browser_exact_distribution_convergence.js";
import {
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from "../../dist/cpp_cute_browser_full_distribution_reproducibility.js";
import {
  CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS,
  cppCuteBrowserRealCompileCase,
} from "../../dist/cpp_cute_browser_real_compile_cases.js";
import {
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_exact_distribution_convergence_v1.js";
import {
  prepareCppCuteBrowserExactDistributionConvergenceMatrix,
} from "./cpp_cute_browser_exact_distribution_convergence.mjs";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-AUTHORING";
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIRECTORY = dirname(SCRIPT_PATH);
const PACKAGE_ROOT = resolve(SCRIPT_DIRECTORY, "../..");
const RESOURCE_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_exact_distribution_convergence_v1.ts",
);
const IDENTITY_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_exact_distribution_convergence_identity_v1.ts",
);
const MAX_INPUT_BYTE_LENGTH = 1_048_576;
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });
const SHA256 = /^[0-9a-f]{64}$/u;
const SOURCE_REVISION = /^[0-9a-f]{40}$/u;
const IDENTIFIER = /^bg\.[a-z0-9.-]+\.sha256\.[0-9a-f]{64}$/u;
const SOFTWARE_ADAPTER =
  /(?:swiftshader|software|llvmpipe|lavapipe|warp)/iu;
const OUTPUT_VERIFICATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.distribution-output-file-verification.v1";

export class CppCuteBrowserExactDistributionConvergenceAuthoringError
  extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name =
      "CppCuteBrowserExactDistributionConvergenceAuthoringError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function
parseCppCuteBrowserExactDistributionConvergenceAuthoringArguments(argv) {
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length === 1 && arguments_[0] === "--check") {
    return Object.freeze({ check: true, inputPath: undefined });
  }
  if (arguments_.length === 1 && arguments_[0].startsWith("--input=")) {
    const inputPath = arguments_[0].slice("--input=".length);
    if (!isAbsolute(inputPath) || inputPath.includes("\0")) {
      invalid(
        "$.arguments[0]",
        "--input requires one absolute NUL-free path",
      );
    }
    return Object.freeze({ check: false, inputPath });
  }
  invalid(
    "$.arguments",
    "expected exactly --check or --input=/absolute/matrix.json",
  );
}

export async function projectCppCuteBrowserExactDistributionConvergence(
  value,
) {
  const sourceRevision = value?.sourceRevision;
  if (typeof sourceRevision !== "string" ||
      !SOURCE_REVISION.test(sourceRevision)) {
    invalid("$.matrix.sourceRevision", "invalid source revision");
  }
  const prepared =
    prepareCppCuteBrowserExactDistributionConvergenceMatrix(
      value?.cases,
      {
        distribution: value?.distribution,
        producer: value?.producer,
      },
      sourceRevision,
    );
  if (!equalBytes(canonicalJsonBytes(value), canonicalJsonBytes(prepared))) {
    invalid(
      "$.matrix",
      "matrix differs from the exact closed convergence projection",
    );
  }

  const fullDistribution =
    await verifyCppCuteBrowserFullDistributionReproducibilityResource(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
  requireCurrentDistribution(prepared, fullDistribution);
  requireObservedAuthority(prepared);

  const resource = Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-observation",
    version: 1,
    authority:
      "package-pinned-local-engineering-exact-payload-convergence-observation-only",
    matrixId: prepared.matrixId,
    sourceRevision,
    caseCount: prepared.caseCount,
    distribution: prepared.distribution,
    producer: prepared.producer,
    webgpu: prepared.webgpu,
    cases: Object.freeze(prepared.cases.map((entry) => Object.freeze({
      caseId: entry.caseId,
      evidenceId: entry.evidenceId,
      sourceSha256: entry.source.sourceSha256,
      dtype: entry.source.dtype,
      coordinateRank: entry.source.coordinateRank,
      candidateId: entry.execution.candidateId,
      artifactId: entry.execution.artifactId,
      authorizationId: entry.execution.authorizationId,
      executionEvidenceId: entry.execution.executionEvidenceId,
      layoutSemanticHash: entry.execution.layoutSemanticHash,
      kernelSemanticHash: entry.execution.kernelSemanticHash,
      cpuDestinationHash: entry.execution.cpuDestinationHash,
      webGpuDestinationHash: entry.execution.webGpuDestinationHash,
      deviceProfileHash: entry.webgpu.deviceProfileHash,
    }))),
    claims: prepared.claims,
  });
  const resourceBytes = canonicalJsonBytes(resource);
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.exact-distribution-convergence-authoring-projection",
    version: 1,
    authority: "package-authoring-projection-only",
    resource,
    resourceSha256: digest(resourceBytes),
    resourceByteLength: resourceBytes.byteLength,
    sourceRevision,
    matrixId: prepared.matrixId,
    caseCount: prepared.caseCount,
  });
}

async function projectPinnedCppCuteBrowserExactDistributionConvergence() {
  const verified =
    await verifyCppCuteBrowserExactDistributionConvergenceResource(
      cppCuteBrowserExactDistributionConvergenceResourceBytes(),
    );
  const resource =
    CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE;
  const resourceBytes = canonicalJsonBytes(resource);
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.exact-distribution-convergence-authoring-projection",
    version: 1,
    authority: "package-authoring-projection-only",
    resource,
    resourceSha256: digest(resourceBytes),
    resourceByteLength: resourceBytes.byteLength,
    sourceRevision: verified.sourceRevision,
    matrixId: verified.matrixId,
    caseCount: verified.cases.length,
  });
}

export function
renderCppCuteBrowserExactDistributionConvergenceResource(projection) {
  exactProjection(projection);
  const value = JSON.stringify(projection.resource, null, 2);
  return `import {\n` +
    `  deepFreezeJson,\n` +
    `  type JsonObject,\n` +
    `} from "@unlocalhosted/browsergrad-semantic-core/schema";\n\n` +
    `/** Generated by cpp_cute_browser_exact_distribution_convergence_authoring.mjs. */\n` +
    `const VALUE = ${value} as const satisfies JsonObject;\n\n` +
    `export type CppCuteBrowserExactDistributionConvergenceV1Resource =\n` +
    `  typeof VALUE;\n\n` +
    `export const CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE =\n` +
    `  deepFreezeJson(VALUE) as unknown as\n` +
    `    CppCuteBrowserExactDistributionConvergenceV1Resource;\n`;
}

export function
renderCppCuteBrowserExactDistributionConvergenceIdentity(projection) {
  exactProjection(projection);
  return `/** Generated by cpp_cute_browser_exact_distribution_convergence_authoring.mjs. */\n` +
    `export const CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256 = ${JSON.stringify(projection.resourceSha256)};\n` +
    `export const CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH = ${projection.resourceByteLength};\n` +
    `export const CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION = ${JSON.stringify(projection.sourceRevision)};\n`;
}

function requireCurrentDistribution(matrix, current) {
  const metadata = current.deterministicMetadata;
  const expected = {
    reproducibilityId: current.reproducibilityId,
    resourceSha256: current.resourceSha256,
    buildInputLockId: current.buildInputLockId,
    buildInputLockResourceSha256: current.buildInputLockResourceSha256,
    profileHash: metadata.profileHash,
    profileSha256: metadata.profileSha256,
    profileByteLength: metadata.profileByteLength,
    assetManifestId: metadata.assetManifestId,
    assetManifestSha256: metadata.assetManifestSha256,
    assetSetSha256: metadata.assetSetSha256,
    buildSubjectId: metadata.buildSubjectId,
    buildSubjectSha256: metadata.buildSubjectSha256,
    workerBundleSha256: metadata.workerBundleSha256,
    exactRootVerificationId: exactRootVerificationId(current),
    exactOutputCount: current.outputCount,
    exactOutputByteLength: current.firstByteLength,
  };
  if (!equalBytes(
    canonicalJsonBytes(matrix.distribution),
    canonicalJsonBytes(expected),
  )) {
    invalid(
      "$.matrix.distribution",
      "matrix does not bind the current exact package distribution",
    );
  }
}

function exactRootVerificationId(current) {
  const outputs = [
    ...current.deterministicOutputs.map((entry) => ({
      outputPath: entry.outputPath,
      sha256: entry.sha256,
      byteLength: entry.byteLength,
    })),
    {
      outputPath: current.detachedEvidence.outputPath,
      sha256: current.detachedEvidence.firstSha256,
      byteLength: current.detachedEvidence.firstByteLength,
    },
  ].sort((left, right) =>
    Buffer.compare(
      Buffer.from(left.outputPath, "utf8"),
      Buffer.from(right.outputPath, "utf8"),
    ));
  return `bg.cpp.distribution-output-file-verification.sha256.${
    digest(canonicalJsonBytes({
      domain: OUTPUT_VERIFICATION_HASH_DOMAIN,
      outputs,
    }))
  }`;
}

function requireObservedAuthority(matrix) {
  if (matrix.caseCount !== CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS.length ||
      matrix.webgpu?.required !== true ||
      matrix.webgpu.actualExecutionObservedForEveryCase !== true ||
      matrix.webgpu.deviceProfileCount !== 1 ||
      matrix.webgpu.deviceProfileHashes?.length !== 1 ||
      !SHA256.test(matrix.webgpu.deviceProfileHashes[0] ?? "") ||
      matrix.claims?.exactPrivateDistributionTreeVerified !== true ||
      matrix.claims.packagePinnedFullDistributionReproducibilityMatched !==
        true ||
      matrix.claims.localEngineeringProducerAuthenticated !== true ||
      matrix.claims.exactNineCaseBrowserWorkerCompilationObserved !== true ||
      matrix.claims.exactCandidatesAuthorizedThroughSharedSeam !== true ||
      matrix.claims.cpuReferenceConvergenceObservedForEveryCase !== true ||
      matrix.claims.requiredRealWebGpuConvergenceObservedForEveryCase !==
        true ||
      matrix.claims.completeDestinationBitComparisonPassedForEveryCase !==
        true ||
      matrix.claims.nonzeroOffsetCanariesPreservedForEveryCase !== true ||
      matrix.claims.externalProducerTrusted !== false ||
      matrix.claims.licenseReviewComplete !== false ||
      matrix.claims.distributionAuthorized !== false ||
      matrix.claims.backendExecutionAuthorityMinted !== false ||
      matrix.claims.releaseReady !== false) {
    invalid(
      "$.matrix",
      "matrix exceeds or falls short of the closed observation authority",
    );
  }
  if (matrix.producer?.builderId !==
        "https://builders.browsergrad.dev/local-engineering-reproducibility" ||
      typeof matrix.producer.keyId !== "string" ||
      !matrix.producer.keyId.startsWith("sha256:") ||
      !IDENTIFIER.test(matrix.producer.policyId ?? "") ||
      !IDENTIFIER.test(matrix.producer.producerEvidenceId ?? "") ||
      !SHA256.test(matrix.producer.policySha256 ?? "") ||
      !SHA256.test(matrix.producer.signatureEvidenceSha256 ?? "") ||
      !SHA256.test(matrix.producer.statementSha256 ?? "") ||
      !SHA256.test(matrix.producer.trustStoreSha256 ?? "")) {
    invalid(
      "$.matrix.producer",
      "matrix producer is not the bounded local-engineering authority",
    );
  }

  const deviceProfileHash = matrix.webgpu.deviceProfileHashes[0];
  for (const [index, caseId] of
    CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS.entries()) {
    const entry = matrix.cases[index];
    const expected = cppCuteBrowserRealCompileCase(caseId);
    const adapter = entry?.webgpu?.environment?.adapter;
    const adapterIdentity = [
      adapter?.vendor,
      adapter?.architecture,
      adapter?.device,
      adapter?.description,
    ].filter((value) => typeof value === "string").join(" ");
    if (entry?.caseId !== caseId ||
        entry.source?.sourceSha256 !== expected.sourceSha256 ||
        entry.source?.dtype !== expected.dtype ||
        entry.source?.coordinateRank !== expected.coordinateRank ||
        !equalBytes(
          canonicalJsonBytes(entry.distribution),
          canonicalJsonBytes(matrix.distribution),
        ) ||
        !equalBytes(
          canonicalJsonBytes(entry.producer),
          canonicalJsonBytes(matrix.producer),
        ) ||
        entry.webgpu?.required !== true ||
        entry.webgpu.actualExecutionObserved !== true ||
        entry.webgpu.deviceProfileHash !== deviceProfileHash ||
        entry.webgpu.environment?.schema !==
          "browsergrad.execution-environment@1" ||
        entry.webgpu.environment.acquisition !==
          "navigator.gpu.requestAdapter/requestDevice" ||
        adapterIdentity.length === 0 ||
        SOFTWARE_ADAPTER.test(adapterIdentity) ||
        entry.execution?.browserWorkerCompiled !== true ||
        entry.execution.localSemanticAuthorizationMinted !== true ||
        entry.execution.cpuReferenceExecuted !== true ||
        entry.execution.actualWebGpuExecuted !== true ||
        entry.execution.completeDestinationBitComparisonPassed !== true ||
        entry.execution.nonzeroOffsetCanariesPreserved !== true ||
        entry.execution.cpuDestinationHash !==
          entry.execution.webGpuDestinationHash ||
        entry.claims?.exactPrivateDistributionTreeVerified !== true ||
        entry.claims.packagePinnedFullDistributionReproducibilityMatched !==
          true ||
        entry.claims.localEngineeringProducerAuthenticated !== true ||
        entry.claims.browserWorkerCompilationObserved !== true ||
        entry.claims.exactCandidateAuthorizedThroughSharedSeam !== true ||
        entry.claims.cpuReferenceConvergenceObserved !== true ||
        entry.claims.requiredRealWebGpuConvergenceObserved !== true ||
        entry.claims.externalProducerTrusted !== false ||
        entry.claims.licenseReviewComplete !== false ||
        entry.claims.distributionAuthorized !== false ||
        entry.claims.backendExecutionAuthorityMinted !== false ||
        entry.claims.releaseReady !== false) {
      invalid(
        `$.matrix.cases[${index}]`,
        `invalid exact hardware convergence observation for ${caseId}`,
      );
    }
  }
}

async function readCanonicalMatrix(path) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    invalid("$.input", "matrix input is unavailable", { cause });
  }
  if (!entry.isFile() || entry.isSymbolicLink() ||
      entry.size <= 0 || entry.size > MAX_INPUT_BYTE_LENGTH ||
      await realpath(path) !== path) {
    invalid(
      "$.input",
      "matrix input must be one canonical bounded non-symlink regular file",
    );
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat();
    if (!before.isFile() || before.dev !== entry.dev ||
        before.ino !== entry.ino || before.size !== entry.size) {
      invalid("$.input", "matrix input identity changed before read");
    }
    const bytes = await handle.readFile();
    const after = await handle.stat();
    if (!after.isFile() || after.dev !== before.dev ||
        after.ino !== before.ino || after.size !== before.size ||
        after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) {
      invalid("$.input", "matrix input changed during read");
    }
    let value;
    try {
      value = JSON.parse(UTF8_DECODER.decode(bytes));
    } catch (cause) {
      invalid("$.input", "matrix input is not strict UTF-8 JSON", { cause });
    }
    if (!equalBytes(bytes, canonicalJsonBytes(value))) {
      invalid("$.input", "matrix input is not canonical JSON");
    }
    return value;
  } finally {
    await handle?.close();
  }
}

function exactProjection(value) {
  if (value?.schema !==
        "browsergrad.compiler.cpp-cute.exact-distribution-convergence-authoring-projection" ||
      value.version !== 1 ||
      value.authority !== "package-authoring-projection-only" ||
      typeof value.resourceSha256 !== "string" ||
      !SHA256.test(value.resourceSha256) ||
      !Number.isSafeInteger(value.resourceByteLength) ||
      value.resourceByteLength <= 0 ||
      typeof value.sourceRevision !== "string" ||
      !SOURCE_REVISION.test(value.sourceRevision) ||
      typeof value.matrixId !== "string" ||
      !IDENTIFIER.test(value.matrixId) ||
      value.caseCount !== CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS.length) {
    invalid(
      "$.projection",
      "expected one authentic bounded authoring projection",
    );
  }
  const resourceBytes = canonicalJsonBytes(value.resource);
  if (resourceBytes.byteLength !== value.resourceByteLength ||
      digest(resourceBytes) !== value.resourceSha256 ||
      value.resource?.sourceRevision !== value.sourceRevision ||
      value.resource?.matrixId !== value.matrixId ||
      value.resource?.caseCount !== value.caseCount) {
    invalid("$.projection.resource", "projection resource identity changed");
  }
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
  throw new CppCuteBrowserExactDistributionConvergenceAuthoringError(
    path,
    message,
    options,
  );
}

async function main() {
  const options =
    parseCppCuteBrowserExactDistributionConvergenceAuthoringArguments(
      process.argv.slice(2),
    );
  const projection = options.check
    ? await projectPinnedCppCuteBrowserExactDistributionConvergence()
    : await projectCppCuteBrowserExactDistributionConvergence(
        await readCanonicalMatrix(options.inputPath),
      );
  const resource =
    renderCppCuteBrowserExactDistributionConvergenceResource(projection);
  const identity =
    renderCppCuteBrowserExactDistributionConvergenceIdentity(projection);
  if (options.check) {
    if (await readFile(RESOURCE_PATH, "utf8") !== resource ||
        await readFile(IDENTITY_PATH, "utf8") !== identity) {
      invalid(
        "$.resources",
        "checked-in exact convergence resources are stale",
      );
    }
  } else {
    await writeFile(RESOURCE_PATH, resource, {
      encoding: "utf8",
      mode: 0o644,
    });
    await writeFile(IDENTITY_PATH, identity, {
      encoding: "utf8",
      mode: 0o644,
    });
  }
  process.stdout.write(`${JSON.stringify({
    schema: projection.schema,
    version: projection.version,
    authority: projection.authority,
    checked: options.check,
    resourceSha256: projection.resourceSha256,
    resourceByteLength: projection.resourceByteLength,
    sourceRevision: projection.sourceRevision,
    matrixId: projection.matrixId,
    caseCount: projection.caseCount,
  })}\n`);
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((cause) => {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown exact convergence authoring failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  });
}
