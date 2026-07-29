import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  canonicalJsonBytes,
  decodeWireJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  decodeCppCuteBrowserAssetManifest,
  unwrapPreparedCppCuteBrowserAssetManifest,
} from "../../dist/cpp_cute_browser_assets.js";
import {
  decodeCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../dist/cpp_cute_browser_build_provenance.js";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";
import {
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from "../../dist/cpp_cute_browser_full_distribution_reproducibility.js";
import {
  verifyCppCuteBrowserBuildProducer,
} from "../../dist/cpp_cute_browser_producer_trust.js";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
  admitCppCuteBrowserProducerTrustPolicy,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import {
  CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS,
  cppCuteBrowserRealCompileCase,
} from "../../dist/cpp_cute_browser_real_compile_cases.js";
import {
  CPP_CUTE_FRONTEND_PROFILE_BYTE_LIMIT,
  decodeCppCuteFrontendProfile,
} from "../../dist/cpp_cute_frontend_profile.js";
import {
  prepareCppCuteAttestationTrustStore,
} from "../../dist/cpp_cute_frontend_provenance.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
} from "../../dist/cpp_cute_browser_worker_bundle.js";
import {
  verifyCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";
import {
  persistCppCuteBrowserRealCompileEvidence,
} from "./cpp_cute_browser_real_compile_runner.mjs";

export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_INPUT_SCHEMA =
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-inputs";
export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_OBSERVATION_SCHEMA =
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-observation";
export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_MATRIX_SCHEMA =
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-matrix";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-DISTRIBUTION-CONVERGENCE";
const EVIDENCE_MARKER =
  "BROWSERGRAD_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_EVIDENCE=";
const LOCAL_ENGINEERING_BUILDER_ID =
  "https://builders.browsergrad.dev/local-engineering-reproducibility";
const ASSET_ROOT = "assets/browsergrad-cpp-cute";
const ASSET_MANIFEST_PATH = `${ASSET_ROOT}/asset-manifest.json`;
const BUILD_INPUT_LOCK_PATH = `${ASSET_ROOT}/build-input-lock.json`;
const ENVELOPE_PATH = `${ASSET_ROOT}/build-provenance.dsse.json`;
const PROFILE_ROUTE =
  "/__browsergrad_cpp_cute_exact_distribution__/frontend-profile.json";
const PRODUCER_POLICY_ROUTE =
  "/__browsergrad_cpp_cute_exact_distribution__/producer-policy.json";
const PRODUCER_TRUST_STORE_ROUTE =
  "/__browsergrad_cpp_cute_exact_distribution__/producer-trust-store.json";
const ANSI_COLOR_PATTERN = new RegExp(
  `${String.fromCharCode(27)}\\[[0-9;]*m`,
  "gu",
);
const MAX_CAPTURED_OUTPUT_BYTES = 16 * 1024 * 1024;
const TRUST_STORE_BYTE_LIMIT = 256 * 1024;
const TRUST_STORE_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: TRUST_STORE_BYTE_LIMIT,
  maxDepth: 8,
  maxNodes: 2_048,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 256,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 32,
  maxArithmeticOperations: 4_096,
});
const SOURCE_REVISION = /^[0-9a-f]{40}$/u;
const scriptRoot = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(scriptRoot, "../..");

export class CppCuteBrowserExactDistributionConvergenceError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserExactDistributionConvergenceError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserExactDistributionConvergenceArguments(
  argv,
) {
  if (!Array.isArray(argv) || argv.length === 0 || argv.length > 8) {
    invalid("$.argv", "expected one bounded nonempty argument list");
  }
  const values = {};
  for (const [index, argument] of argv.entries()) {
    if (argument === "--preflight-only") {
      if (values.preflightOnly === true) {
        invalid(`$.argv[${index}]`, "duplicate --preflight-only");
      }
      values.preflightOnly = true;
      continue;
    }
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      invalid(`$.argv[${index}]`, "expected one supported --name=value");
    }
    const name = argument.slice(2, separator);
    const value = argument.slice(separator + 1);
    if (value.length === 0 || Object.hasOwn(values, name)) {
      invalid(`$.argv[${index}]`, `missing or duplicate --${name}`);
    }
    if (name === "distribution-root") values.distributionRoot = value;
    else if (name === "profile") values.profilePath = value;
    else if (name === "producer-policy") values.producerPolicyPath = value;
    else if (name === "producer-trust-store") {
      values.producerTrustStorePath = value;
    } else if (name === "evidence-output") values.evidenceOutput = value;
    else if (name === "source-revision") values.sourceRevision = value;
    else invalid(`$.argv[${index}]`, `unsupported option --${name}`);
  }
  const requiredPaths = [
    ["distributionRoot", "$.distributionRoot"],
    ["profilePath", "$.profilePath"],
    ["producerPolicyPath", "$.producerPolicyPath"],
    ["producerTrustStorePath", "$.producerTrustStorePath"],
  ];
  for (const [key, path] of requiredPaths) {
    values[key] = requiredAbsolutePath(values[key], path);
  }
  if (values.preflightOnly !== true) {
    values.evidenceOutput = requiredAbsolutePath(
      values.evidenceOutput,
      "$.evidenceOutput",
    );
    if (typeof values.sourceRevision !== "string" ||
        !SOURCE_REVISION.test(values.sourceRevision)) {
      invalid(
        "$.sourceRevision",
        "source revision must be one lowercase 40-hex Git revision",
      );
    }
  } else {
    if (values.evidenceOutput !== undefined ||
        values.sourceRevision !== undefined) {
      invalid(
        "$.argv",
        "preflight-only does not accept evidence-output or source-revision",
      );
    }
  }
  return Object.freeze({
    distributionRoot: values.distributionRoot,
    profilePath: values.profilePath,
    producerPolicyPath: values.producerPolicyPath,
    producerTrustStorePath: values.producerTrustStorePath,
    ...(values.preflightOnly === true
      ? { preflightOnly: true }
      : {
          evidenceOutput: values.evidenceOutput,
          sourceRevision: values.sourceRevision,
          preflightOnly: false,
        }),
  });
}

/**
 * Rehashes the exact 25-file tree, admits the exact profile and metadata, and
 * independently verifies the supplied public producer inputs. No private key
 * or distribution/release approval is accepted by this lane.
 */
export async function preflightCppCuteBrowserExactDistributionConvergence(
  input,
) {
  const options = exactPreflightInput(input);
  const distributionRoot = await canonicalDirectory(
    options.distributionRoot,
    "$.distributionRoot",
  );
  const authority =
    await verifyCppCuteBrowserFullDistributionReproducibilityResource(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
  const expectedOutputs = [
    ...authority.deterministicOutputs.map((output) => ({
      outputPath: output.outputPath,
      sha256: output.sha256,
      byteLength: output.byteLength,
    })),
    {
      outputPath: authority.detachedEvidence.outputPath,
      sha256: authority.detachedEvidence.firstSha256,
      byteLength: authority.detachedEvidence.firstByteLength,
    },
  ];
  const rootVerification =
    await verifyCppCuteBrowserDistributionOutputFiles({
      outputRoot: distributionRoot,
      expectedOutputs,
    });
  if (rootVerification.outputs.length !== 25 ||
      rootVerification.totals.byteLength !== authority.firstByteLength) {
    mismatch(
      "$.distributionRoot",
      "exact root verification differs from the package-pinned 25-file observation",
    );
  }

  const control = await Promise.all([
    readImmutableInput(
      options.profilePath,
      CPP_CUTE_FRONTEND_PROFILE_BYTE_LIMIT,
      "$.profilePath",
    ),
    readImmutableInput(
      join(distributionRoot, ASSET_MANIFEST_PATH),
      256 * 1024,
      "$.distribution.assetManifest",
    ),
    readImmutableInput(
      join(distributionRoot, BUILD_INPUT_LOCK_PATH),
      256 * 1024,
      "$.distribution.buildInputLock",
    ),
    readImmutableInput(
      join(distributionRoot, ENVELOPE_PATH),
      256 * 1024,
      "$.distribution.envelope",
    ),
    readImmutableInput(
      options.producerPolicyPath,
      CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
      "$.producerPolicyPath",
    ),
    readImmutableInput(
      options.producerTrustStorePath,
      TRUST_STORE_BYTE_LIMIT,
      "$.producerTrustStorePath",
    ),
  ]);
  const [
    profileInput,
    manifestInput,
    buildLockInput,
    envelopeInput,
    producerPolicyInput,
    trustStoreInput,
  ] = control;
  const metadata = authority.deterministicMetadata;
  requireIdentity(
    profileInput,
    metadata.profileSha256,
    metadata.profileByteLength,
    "$.profilePath",
  );
  requireIdentity(
    manifestInput,
    metadata.assetManifestSha256,
    metadata.assetManifestByteLength,
    "$.distribution.assetManifest",
  );
  requireIdentity(
    buildLockInput,
    authority.buildInputLockResourceSha256,
    expectedOutput(expectedOutputs, BUILD_INPUT_LOCK_PATH).byteLength,
    "$.distribution.buildInputLock",
  );
  requireIdentity(
    envelopeInput,
    authority.detachedEvidence.firstSha256,
    authority.detachedEvidence.firstByteLength,
    "$.distribution.envelope",
  );

  const profile = await decodeCppCuteFrontendProfile(profileInput.bytes);
  if (profile.profileHash !== metadata.profileHash ||
      profile.compilationContractHash !== metadata.compilationContractHash) {
    mismatch(
      "$.profilePath",
      "profile semantic identity differs from package-pinned distribution metadata",
    );
  }
  const [
    assetManifest,
    buildInputLock,
    workerBundle,
    producerTrustPolicy,
  ] = await Promise.all([
    decodeCppCuteBrowserAssetManifest(manifestInput.bytes, profile),
    decodeCppCuteBrowserBuildInputLock(buildLockInput.bytes),
    verifyCppCuteBrowserWorkerBundle(),
    admitCppCuteBrowserProducerTrustPolicy(producerPolicyInput.bytes),
  ]);
  if (assetManifest.manifestId !== metadata.assetManifestId ||
      assetManifest.assetSetSha256 !== metadata.assetSetSha256 ||
      buildInputLock.resourceSha256 !==
        authority.buildInputLockResourceSha256) {
    mismatch(
      "$.distribution",
      "prepared profile, manifest, or build lock differs from distribution metadata",
    );
  }
  const trustStore = await prepareCppCuteAttestationTrustStore(
    decodeCanonicalJson(
      trustStoreInput.bytes,
      TRUST_STORE_DECODE_LIMITS,
      "$.producerTrustStorePath",
    ),
    { limits: TRUST_STORE_DECODE_LIMITS },
  );
  const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
    decodeCanonicalJson(
      envelopeInput.bytes,
      CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
      "$.distribution.envelope",
    ),
    {
      assetManifest,
      buildInputLock,
      workerBundle,
      trustStore,
    },
  );
  const producer = await verifyCppCuteBrowserBuildProducer(
    signatureBinding,
    producerTrustPolicy,
  );
  if (producer.builderId !== LOCAL_ENGINEERING_BUILDER_ID ||
      producer.profileHash !== metadata.profileHash ||
      producer.manifestId !== metadata.assetManifestId ||
      producer.buildSubjectId !== metadata.buildSubjectId ||
      producer.workerBundleSha256 !== metadata.workerBundleSha256 ||
      producer.exactAssetBytesVerified !== false ||
      producer.fullDistributedOutputSetReproducible !== false ||
      producer.licenseReviewComplete !== false ||
      producer.distributionAuthorized !== false ||
      producer.releaseReady !== false) {
    mismatch(
      "$.producer",
      "producer is not the narrow local-engineering authority for this exact build",
    );
  }
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle);
  const manifest = unwrapPreparedCppCuteBrowserAssetManifest(
    assetManifest,
  ).manifest;
  const expectedByPath = new Map(
    expectedOutputs.map((output) => [output.outputPath, output]),
  );
  const assets = manifest.body.assets.map((asset, index) => {
    if (!asset.url.startsWith("/") ||
        asset.url.includes("?") ||
        asset.url.includes("#") ||
        asset.url.includes("..")) {
      mismatch(
        `$.assetManifest.body.assets[${index}].url`,
        "asset URL is not one canonical root-relative distribution path",
      );
    }
    const outputPath = asset.url.slice(1);
    const output = expectedByPath.get(outputPath);
    if (output === undefined ||
        output.sha256 !== asset.sha256 ||
        output.byteLength !== asset.byteLength) {
      mismatch(
        `$.assetManifest.body.assets[${index}]`,
        "manifest asset differs from the exact distribution output identity",
      );
    }
    return Object.freeze({
      assetId: asset.assetId,
      route: asset.url,
      path: join(distributionRoot, outputPath),
      mediaType: asset.mediaType,
      sha256: asset.sha256,
      byteLength: Number(asset.byteLength),
    });
  });
  if (assets.length !== 9) {
    mismatch("$.assetManifest.body.assets", "expected the closed nine-asset runtime set");
  }
  const controls = Object.freeze({
    profile: servedControl(
      PROFILE_ROUTE,
      profileInput,
      "application/json",
    ),
    assetManifest: servedControl(
      `/${ASSET_MANIFEST_PATH}`,
      manifestInput,
      "application/json",
    ),
    buildInputLock: servedControl(
      `/${BUILD_INPUT_LOCK_PATH}`,
      buildLockInput,
      "application/json",
    ),
    envelope: servedControl(
      `/${ENVELOPE_PATH}`,
      envelopeInput,
      "application/vnd.dsse.envelope.v1+json",
    ),
    producerPolicy: servedControl(
      PRODUCER_POLICY_ROUTE,
      producerPolicyInput,
      "application/json",
    ),
    producerTrustStore: servedControl(
      PRODUCER_TRUST_STORE_ROUTE,
      trustStoreInput,
      "application/json",
    ),
  });
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_INPUT_SCHEMA,
    version: 1,
    authority: "host-preflight-exact-private-distribution-only",
    controls,
    assets: Object.freeze(assets),
    distribution: Object.freeze({
      reproducibilityId: authority.reproducibilityId,
      resourceSha256: authority.resourceSha256,
      buildInputLockId: authority.buildInputLockId,
      buildInputLockResourceSha256:
        authority.buildInputLockResourceSha256,
      profileHash: metadata.profileHash,
      profileSha256: metadata.profileSha256,
      profileByteLength: metadata.profileByteLength,
      assetManifestId: metadata.assetManifestId,
      assetManifestSha256: metadata.assetManifestSha256,
      assetSetSha256: metadata.assetSetSha256,
      buildSubjectId: metadata.buildSubjectId,
      buildSubjectSha256: metadata.buildSubjectSha256,
      workerBundleSha256: worker.sha256,
      exactRootVerificationId: rootVerification.verificationId,
      exactOutputCount: rootVerification.outputs.length,
      exactOutputByteLength: rootVerification.totals.byteLength,
    }),
    producer: Object.freeze({
      producerEvidenceId: producer.producerEvidenceId,
      policyId: producer.policyId,
      policySha256: producer.policySha256,
      builderId: producer.builderId,
      keyId: producer.keyId,
      trustStoreSha256: producer.trustStoreSha256,
      statementSha256: producer.statementSha256,
      signatureEvidenceSha256: producer.signatureEvidenceSha256,
    }),
    claims: Object.freeze({
      exactPrivateDistributionTreeVerified: true,
      packagePinnedFullDistributionReproducibilityMatched: true,
      localEngineeringProducerAuthenticated: true,
      externalProducerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    }),
  });
}

export function prepareCppCuteBrowserExactDistributionConvergenceMatrix(
  observations,
  preflight,
  sourceRevision,
) {
  if (!SOURCE_REVISION.test(sourceRevision)) {
    invalid(
      "$.sourceRevision",
      "matrix source revision must be one lowercase 40-hex revision",
    );
  }
  if (!Array.isArray(observations) ||
      observations.length !==
        CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.length) {
    invalid("$.observations", "expected exactly eight convergence observations");
  }
  const cases = CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.map(
    (caseId, index) => {
      const observation = observations[index];
      const compileCase = cppCuteBrowserRealCompileCase(caseId);
      if (typeof observation !== "object" || observation === null ||
          observation.schema !==
            CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_OBSERVATION_SCHEMA ||
          observation.version !== 1 ||
          observation.caseId !== caseId ||
          observation.source?.sourceSha256 !== compileCase.sourceSha256 ||
          observation.source?.dtype !== compileCase.dtype ||
          observation.source?.coordinateRank !== compileCase.coordinateRank ||
          observation.distribution?.reproducibilityId !==
            preflight.distribution.reproducibilityId ||
          observation.distribution?.buildSubjectId !==
            preflight.distribution.buildSubjectId ||
          observation.producer?.producerEvidenceId !==
            preflight.producer.producerEvidenceId ||
          observation.execution?.browserWorkerCompiled !== true ||
          observation.execution?.localSemanticAuthorizationMinted !== true ||
          observation.execution?.cpuReferenceExecuted !== true ||
          observation.execution?.actualWebGpuExecuted !== true ||
          observation.execution?.completeDestinationBitComparisonPassed !==
            true ||
          observation.execution?.nonzeroOffsetCanariesPreserved !== true ||
          observation.claims?.externalProducerTrusted !== false ||
          observation.claims?.licenseReviewComplete !== false ||
          observation.claims?.distributionAuthorized !== false ||
          observation.claims?.backendExecutionAuthorityMinted !== false ||
          observation.claims?.releaseReady !== false) {
        invalid(
          `$.observations[${index}]`,
          `invalid exact-payload convergence observation for ${caseId}`,
        );
      }
      return observation;
    },
  );
  requireUnique(cases, "candidateId", (entry) =>
    entry.execution.candidateId);
  requireUnique(cases, "artifactId", (entry) =>
    entry.execution.artifactId);
  requireUnique(cases, "authorizationId", (entry) =>
    entry.execution.authorizationId);
  requireUnique(cases, "executionEvidenceId", (entry) =>
    entry.execution.executionEvidenceId);
  const environments = new Set(
    cases.map((entry) => entry.webgpu.deviceProfileHash),
  );
  const matrixId = `bg.cpp.browser-exact-distribution-convergence.sha256.${
    sha256(canonicalJsonBytes({
      domain:
        "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-matrix.v1",
      sourceRevision,
      distribution: preflight.distribution,
      producer: preflight.producer,
      cases: cases.map((entry) => ({
        caseId: entry.caseId,
        evidenceId: entry.evidenceId,
        candidateId: entry.execution.candidateId,
        authorizationId: entry.execution.authorizationId,
        layoutSemanticHash: entry.execution.layoutSemanticHash,
        kernelSemanticHash: entry.execution.kernelSemanticHash,
        cpuDestinationHash: entry.execution.cpuDestinationHash,
        webGpuDestinationHash: entry.execution.webGpuDestinationHash,
        deviceProfileHash: entry.webgpu.deviceProfileHash,
      })),
    }))
  }`;
  return Object.freeze({
    schema:
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_MATRIX_SCHEMA,
    version: 1,
    authority:
      "local-engineering-exact-payload-cpu-webgpu-observation-only",
    matrixId,
    sourceRevision,
    caseCount: cases.length,
    cases: Object.freeze(cases),
    distribution: preflight.distribution,
    producer: preflight.producer,
    webgpu: Object.freeze({
      required: true,
      actualExecutionObservedForEveryCase: true,
      deviceProfileCount: environments.size,
      deviceProfileHashes: Object.freeze([...environments].sort()),
    }),
    claims: Object.freeze({
      exactPrivateDistributionTreeVerified: true,
      packagePinnedFullDistributionReproducibilityMatched: true,
      exactEightCaseBrowserWorkerCompilationObserved: true,
      localEngineeringProducerAuthenticated: true,
      exactCandidatesAuthorizedThroughSharedSeam: true,
      cpuReferenceConvergenceObservedForEveryCase: true,
      requiredRealWebGpuConvergenceObservedForEveryCase: true,
      completeDestinationBitComparisonPassedForEveryCase: true,
      nonzeroOffsetCanariesPreservedForEveryCase: true,
      externalProducerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      backendExecutionAuthorityMinted: false,
      releaseReady: false,
    }),
  });
}

export async function runCppCuteBrowserExactDistributionConvergence(
  argv = process.argv.slice(2),
) {
  const options =
    parseCppCuteBrowserExactDistributionConvergenceArguments(argv);
  const preflight =
    await preflightCppCuteBrowserExactDistributionConvergence(options);
  process.stdout.write(
    `Verified exact ${preflight.distribution.exactOutputCount}-file private distribution ` +
    `${preflight.distribution.reproducibilityId}\n`,
  );
  if (options.preflightOnly) return preflight;

  const observations = [];
  // Each Clang-Wasm Worker reserves a bounded large memory. Sequential cases
  // keep peak memory bounded while preserving case-isolated browser evidence.
  for (const caseId of CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS) {
    observations.push(await runBrowserCase(
      preflight,
      caseId,
      options.sourceRevision,
    ));
  }
  const matrix =
    prepareCppCuteBrowserExactDistributionConvergenceMatrix(
      observations,
      preflight,
      options.sourceRevision,
    );
  const written = await persistCppCuteBrowserRealCompileEvidence(
    options.evidenceOutput,
    matrix,
  );
  process.stdout.write(
    `Exact distribution convergence evidence: ${written.outputPath}\n`,
  );
  return matrix;
}

async function runBrowserCase(preflight, caseId, sourceRevision) {
  const input = Object.freeze({ ...preflight, caseId, sourceRevision });
  const child = spawn(
    "pnpm",
    [
      "exec",
      "vitest",
      "run",
      "--config",
      "vitest.browser.cpp-cute-exact-distribution.config.ts",
    ],
    {
      cwd: packageRoot,
      env: {
        ...process.env,
        // Required WebGPU on macOS uses the same headed-default profile as
        // the established browser evidence lanes. Callers may opt into a
        // proven headless backend explicitly.
        BG_BROWSER_HEADLESS: process.env.BG_BROWSER_HEADLESS ?? "0",
        BG_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_INPUTS:
          JSON.stringify(input),
        NO_COLOR: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let captured = "";
  const capture = (chunk, output) => {
    const text = String(chunk);
    output.write(text);
    if (captured.length < MAX_CAPTURED_OUTPUT_BYTES) {
      captured += text.slice(
        0,
        MAX_CAPTURED_OUTPUT_BYTES - captured.length,
      );
    }
  };
  child.stdout.on("data", (chunk) => capture(chunk, process.stdout));
  child.stderr.on("data", (chunk) => capture(chunk, process.stderr));
  const exitCode = await new Promise((resolveExit, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => {
      if (signal !== null) {
        reject(new CppCuteBrowserExactDistributionConvergenceError(
          "$.browser",
          `${caseId} browser verifier terminated by ${signal}`,
        ));
        return;
      }
      resolveExit(code ?? 1);
    });
  });
  if (exitCode !== 0) {
    invalid(
      "$.browser",
      `${caseId} browser verifier exited with status ${exitCode}`,
    );
  }
  return parseBrowserEvidence(captured, preflight, caseId);
}

function parseBrowserEvidence(output, preflight, caseId) {
  const clean = output.replaceAll(ANSI_COLOR_PATTERN, "");
  const lines = clean.split(/\r?\n/u)
    .filter((line) => line.includes(EVIDENCE_MARKER));
  if (lines.length !== 1) {
    invalid(
      "$.evidence",
      `expected exactly one ${EVIDENCE_MARKER} record for ${caseId}`,
    );
  }
  const line = lines[0];
  const offset = line.indexOf(EVIDENCE_MARKER);
  let evidence;
  try {
    evidence = JSON.parse(line.slice(offset + EVIDENCE_MARKER.length));
  } catch (cause) {
    invalid("$.evidence", "browser evidence is not valid JSON", { cause });
  }
  const compileCase = cppCuteBrowserRealCompileCase(caseId);
  if (evidence?.schema !==
        CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_OBSERVATION_SCHEMA ||
      evidence?.version !== 1 ||
      evidence?.caseId !== caseId ||
      evidence?.source?.sourceSha256 !== compileCase.sourceSha256 ||
      evidence?.distribution?.reproducibilityId !==
        preflight.distribution.reproducibilityId ||
      evidence?.distribution?.buildSubjectId !==
        preflight.distribution.buildSubjectId ||
      evidence?.producer?.producerEvidenceId !==
        preflight.producer.producerEvidenceId ||
      evidence?.execution?.browserWorkerCompiled !== true ||
      evidence?.execution?.localSemanticAuthorizationMinted !== true ||
      evidence?.execution?.cpuReferenceExecuted !== true ||
      evidence?.execution?.actualWebGpuExecuted !== true ||
      evidence?.execution?.completeDestinationBitComparisonPassed !== true ||
      evidence?.execution?.nonzeroOffsetCanariesPreserved !== true ||
      evidence?.claims?.externalProducerTrusted !== false ||
      evidence?.claims?.licenseReviewComplete !== false ||
      evidence?.claims?.distributionAuthorized !== false ||
      evidence?.claims?.backendExecutionAuthorityMinted !== false ||
      evidence?.claims?.releaseReady !== false) {
    invalid(
      "$.evidence",
      `${caseId} browser evidence has an invalid authority boundary`,
    );
  }
  return evidence;
}

function exactPreflightInput(input) {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    invalid("$.input", "expected one plain preflight input record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(input);
  const allowed = new Set([
    "distributionRoot",
    "profilePath",
    "producerPolicyPath",
    "producerTrustStorePath",
    "evidenceOutput",
    "sourceRevision",
    "preflightOnly",
  ]);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== "string" || !allowed.has(key)) {
      invalid("$.input", "preflight input contains an unknown field");
    }
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) ||
        descriptor.enumerable !== true) {
      invalid(
        `$.input.${String(key)}`,
        "preflight fields must be enumerable data properties",
      );
    }
  }
  return Object.freeze({
    distributionRoot: requiredAbsolutePath(
      descriptors.distributionRoot?.value,
      "$.distributionRoot",
    ),
    profilePath: requiredAbsolutePath(
      descriptors.profilePath?.value,
      "$.profilePath",
    ),
    producerPolicyPath: requiredAbsolutePath(
      descriptors.producerPolicyPath?.value,
      "$.producerPolicyPath",
    ),
    producerTrustStorePath: requiredAbsolutePath(
      descriptors.producerTrustStorePath?.value,
      "$.producerTrustStorePath",
    ),
  });
}

function servedControl(route, input, mediaType) {
  return Object.freeze({
    route,
    path: input.path,
    mediaType,
    sha256: input.sha256,
    byteLength: input.byteLength,
  });
}

function expectedOutput(outputs, outputPath) {
  const output = outputs.find((candidate) =>
    candidate.outputPath === outputPath);
  if (output === undefined) {
    mismatch("$.distribution", `missing package output ${outputPath}`);
  }
  return output;
}

function requireIdentity(input, sha256_, byteLength, path) {
  if (input.sha256 !== sha256_ ||
      String(input.byteLength) !== byteLength) {
    mismatch(path, "input bytes differ from their exact package identity");
  }
}

function requireUnique(cases, name, select) {
  const values = new Set(cases.map(select));
  if (values.size !== cases.length) {
    invalid("$.observations", `${name} must be unique for all eight cases`);
  }
}

function requiredAbsolutePath(value, path) {
  if (typeof value !== "string" || value.length === 0 ||
      !isAbsolute(value)) {
    invalid(path, "expected one nonempty absolute path");
  }
  return value;
}

async function canonicalDirectory(path, diagnosticPath) {
  let metadata;
  let canonical;
  try {
    [metadata, canonical] = await Promise.all([
      lstat(path),
      realpath(path),
    ]);
  } catch (cause) {
    invalid(diagnosticPath, "directory is unavailable", { cause });
  }
  if (!metadata.isDirectory() || metadata.isSymbolicLink() ||
      canonical !== path) {
    invalid(
      diagnosticPath,
      "expected one canonical non-symlink directory",
    );
  }
  return canonical;
}

async function readImmutableInput(path, byteLimit, diagnosticPath) {
  const canonical = await canonicalRegularFile(path, diagnosticPath);
  let handle;
  try {
    handle = await open(
      canonical,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
    const before = await handle.stat();
    if (!before.isFile() || before.size <= 0 || before.size > byteLimit) {
      invalid(diagnosticPath, "input file exceeds its fixed byte bounds");
    }
    const bytes = new Uint8Array(await handle.readFile());
    const after = await handle.stat();
    if (!sameFileIdentity(before, after) ||
        bytes.byteLength !== before.size) {
      invalid(diagnosticPath, "input identity changed while it was read");
    }
    return Object.freeze({
      path: canonical,
      bytes,
      sha256: sha256(bytes),
      byteLength: bytes.byteLength,
    });
  } catch (cause) {
    if (cause instanceof
        CppCuteBrowserExactDistributionConvergenceError) {
      throw cause;
    }
    invalid(diagnosticPath, "input could not be read exactly", { cause });
  } finally {
    await handle?.close();
  }
}

async function canonicalRegularFile(path, diagnosticPath) {
  let metadata;
  let canonical;
  try {
    [metadata, canonical] = await Promise.all([
      lstat(path),
      realpath(path),
    ]);
  } catch (cause) {
    invalid(diagnosticPath, "file is unavailable", { cause });
  }
  if (!metadata.isFile() || metadata.isSymbolicLink() ||
      canonical !== path) {
    invalid(
      diagnosticPath,
      "expected one canonical non-symlink regular file",
    );
  }
  return canonical;
}

function decodeCanonicalJson(bytes, limits, path) {
  let value;
  let canonical;
  try {
    value = decodeWireJson(bytes, { limits });
    canonical = canonicalJsonBytes(value, { limits });
  } catch (cause) {
    invalid(path, "input must be bounded strict UTF-8 JSON", { cause });
  }
  if (!sameBytes(bytes, canonical)) {
    mismatch(path, "input bytes must exactly equal canonical JSON");
  }
  return value;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeMs === right.mtimeMs;
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function mismatch(path, message, options) {
  invalid(path, message, options);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserExactDistributionConvergenceError(
    path,
    message,
    options,
  );
}

if (process.argv[1] !== undefined &&
    import.meta.url === new URL(`file://${resolve(process.argv[1])}`).href) {
  runCppCuteBrowserExactDistributionConvergence().catch((error) => {
    process.stderr.write(
      `${error instanceof Error
        ? error.stack ?? error.message
        : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
