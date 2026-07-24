import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open } from "node:fs/promises";
import { join, normalize } from "node:path/posix";

import {
  canonicalJsonBytes,
  decodeWireJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  decodeUntrustedCppCuteBrowserBuildProvenanceSyntax,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";
import {
  verifyCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";

export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-REPRODUCIBILITY";
const HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility.v1";
const EXPECTED_DETACHED_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/build-provenance.dsse.json";
const EXPECTED_OUTPUT_COUNT = 24;
const EXPECTED_DETERMINISTIC_SUBJECT_COUNT = 23;
const REPRODUCIBILITY_AUTHORITIES = new WeakSet();

export class CppCuteBrowserFullDistributionReproducibilityError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserFullDistributionReproducibilityError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Rehashes two exact private distribution trees against the prepared build
 * lock. Deterministic subjects must be byte-identical. The sole detached DSSE
 * envelope may differ only while its strict-decoded build subject remains
 * identical. This grants reproducibility authority only.
 */
export async function verifyCppCuteBrowserFullDistributionReproducibility(
  input,
) {
  const object = exactObject(
    input,
    ["buildInputLock", "first", "second"],
    "$.input",
  );
  let lockRecord;
  try {
    lockRecord = unwrapPreparedCppCuteBrowserBuildInputLock(
      object.buildInputLock,
    );
  } catch (cause) {
    invalid(
      "$.input.buildInputLock",
      "expected one opaque prepared build-input lock",
      { cause },
    );
  }
  const first = exactTreeInput(object.first, "$.input.first");
  const second = exactTreeInput(object.second, "$.input.second");
  if (pathsOverlap(first.outputRoot, second.outputRoot)) {
    invalid(
      "$.input",
      "reproducibility roots must be distinct and non-overlapping",
    );
  }

  let firstVerification;
  let secondVerification;
  try {
    [firstVerification, secondVerification] = await Promise.all([
      verifyCppCuteBrowserDistributionOutputFiles(first),
      verifyCppCuteBrowserDistributionOutputFiles(second),
    ]);
  } catch (cause) {
    invalid(
      "$.input",
      "failed to rehash both exact distribution output trees",
      { cause },
    );
  }

  const plan = exactOutputPlan(lockRecord.lock);
  assertPlanMatches(plan, firstVerification.outputs, "$.input.first");
  assertPlanMatches(plan, secondVerification.outputs, "$.input.second");
  const firstByPath = new Map(
    firstVerification.outputs.map((output) => [output.outputPath, output]),
  );
  const secondByPath = new Map(
    secondVerification.outputs.map((output) => [output.outputPath, output]),
  );
  const deterministicOutputs = [];
  let detachedPlan;
  for (const planned of plan) {
    const firstOutput = requiredOutput(firstByPath, planned.path, "$.input.first");
    const secondOutput =
      requiredOutput(secondByPath, planned.path, "$.input.second");
    if (planned.reproducibilityClass === "deterministic-subject") {
      if (!sameOutput(firstOutput, secondOutput)) {
        mismatch(
          `$.outputs.${planned.path}`,
          "deterministic distribution subject differs between roots",
        );
      }
      deterministicOutputs.push(Object.freeze({
        outputPath: planned.path,
        role: planned.role,
        mediaType: planned.mediaType,
        sha256: firstOutput.sha256,
        byteLength: firstOutput.byteLength,
      }));
    } else {
      detachedPlan = planned;
    }
  }
  if (deterministicOutputs.length !== EXPECTED_DETERMINISTIC_SUBJECT_COUNT ||
      detachedPlan?.path !== EXPECTED_DETACHED_OUTPUT_PATH) {
    mismatch(
      "$.outputs",
      "build lock no longer defines the closed full-distribution class split",
    );
  }

  const firstDetached = requiredOutput(
    firstByPath,
    EXPECTED_DETACHED_OUTPUT_PATH,
    "$.input.first",
  );
  const secondDetached = requiredOutput(
    secondByPath,
    EXPECTED_DETACHED_OUTPUT_PATH,
    "$.input.second",
  );
  const [firstSubject, secondSubject] = await Promise.all([
    readDetachedSubject(first.outputRoot, firstDetached, object.buildInputLock),
    readDetachedSubject(second.outputRoot, secondDetached, object.buildInputLock),
  ]);
  if (firstSubject.buildSubjectId !== secondSubject.buildSubjectId ||
      firstSubject.buildSubjectSha256 !==
        secondSubject.buildSubjectSha256) {
    mismatch(
      `$.outputs.${EXPECTED_DETACHED_OUTPUT_PATH}`,
      "detached provenance envelopes bind different build subjects",
    );
  }

  let finalFirst;
  let finalSecond;
  try {
    [finalFirst, finalSecond] = await Promise.all([
      verifyCppCuteBrowserDistributionOutputFiles(first),
      verifyCppCuteBrowserDistributionOutputFiles(second),
    ]);
  } catch (cause) {
    invalid(
      "$.input",
      "distribution tree changed after detached-subject inspection",
      { cause },
    );
  }
  if (finalFirst.verificationId !== firstVerification.verificationId ||
      finalSecond.verificationId !== secondVerification.verificationId) {
    mismatch(
      "$.input",
      "distribution identities changed across terminal verification",
    );
  }

  const detachedEvidence = Object.freeze({
    outputPath: EXPECTED_DETACHED_OUTPUT_PATH,
    role: detachedPlan.role,
    mediaType: detachedPlan.mediaType,
    firstSha256: firstDetached.sha256,
    firstByteLength: firstDetached.byteLength,
    secondSha256: secondDetached.sha256,
    secondByteLength: secondDetached.byteLength,
    buildSubjectId: firstSubject.buildSubjectId,
    buildSubjectSha256: firstSubject.buildSubjectSha256,
  });
  const reproducibilityHash = sha256(canonicalJsonBytes({
    domain: HASH_DOMAIN,
    buildInputLockId: object.buildInputLock.lockId,
    buildInputLockResourceSha256: object.buildInputLock.resourceSha256,
    deterministicOutputs,
    detachedEvidence,
  }));
  const report = Object.freeze({
    schema: CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_SCHEMA,
    version: 1,
    reproducibilityId:
      `bg.cpp.browser-full-distribution-reproducibility.sha256.${reproducibilityHash}`,
    authority: "two-root-complete-distribution-output-reproducibility-only",
    scope: "complete-build-input-lock-distributed-output-plan",
    buildInputLockId: object.buildInputLock.lockId,
    buildInputLockResourceSha256: object.buildInputLock.resourceSha256,
    firstOutputRoot: first.outputRoot,
    secondOutputRoot: second.outputRoot,
    firstOutputVerificationId: firstVerification.verificationId,
    secondOutputVerificationId: secondVerification.verificationId,
    deterministicOutputs: Object.freeze(deterministicOutputs),
    detachedEvidence,
    totals: Object.freeze({
      outputCount: plan.length,
      deterministicSubjectCount: deterministicOutputs.length,
      detachedEvidenceCount: 1,
      firstByteLength: firstVerification.totals.byteLength,
      secondByteLength: secondVerification.totals.byteLength,
    }),
    claims: Object.freeze({
      twoDistinctPrivateOutputRootsVerified: true,
      exactBuildLockOutputPlanMatched: true,
      exactOutputsRehashedInBothRoots: true,
      deterministicSubjectsByteIdentical: true,
      detachedEvidenceBuildSubjectMatched: true,
      fullDistributedOutputSetReproducible: true,
      detachedSignatureVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      producerTrusted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    }),
  });
  REPRODUCIBILITY_AUTHORITIES.add(report);
  return report;
}

export function requireCppCuteBrowserFullDistributionReproducibilityAuthority(
  report,
) {
  if (typeof report !== "object" || report === null ||
      !REPRODUCIBILITY_AUTHORITIES.has(report)) {
    invalid(
      "$.report",
      "expected verifier-issued full-distribution reproducibility authority",
    );
  }
}

export function canonicalCppCuteBrowserFullDistributionReproducibilityBytes(
  report,
) {
  requireCppCuteBrowserFullDistributionReproducibilityAuthority(report);
  return canonicalJsonBytes(report);
}

function exactTreeInput(value, path) {
  const object = exactObject(
    value,
    ["expectedOutputs", "outputRoot"],
    path,
  );
  return Object.freeze({
    expectedOutputs: object.expectedOutputs,
    outputRoot: absolutePath(object.outputRoot, `${path}.outputRoot`),
  });
}

function exactOutputPlan(lock) {
  const plan = lock.body.recipe.distributedOutputPlan;
  if (plan.closure !== "exact-path-set-no-additional-distributed-files" ||
      plan.outputs.length !== EXPECTED_OUTPUT_COUNT) {
    mismatch(
      "$.input.buildInputLock.body.recipe.distributedOutputPlan",
      "build lock does not define the exact complete output plan",
    );
  }
  const detached = plan.outputs.filter(
    (output) => output.reproducibilityClass === "detached-evidence",
  );
  if (detached.length !== 1 ||
      detached[0]?.path !== EXPECTED_DETACHED_OUTPUT_PATH) {
    mismatch(
      "$.input.buildInputLock.body.recipe.distributedOutputPlan.outputs",
      "build lock must retain one exact detached provenance output",
    );
  }
  return plan.outputs;
}

function assertPlanMatches(plan, outputs, path) {
  if (outputs.length !== plan.length) {
    mismatch(path, "distribution output count differs from the build lock");
  }
  const outputPaths = new Set(outputs.map((output) => output.outputPath));
  for (const planned of plan) {
    if (!outputPaths.has(planned.path)) {
      mismatch(
        `${path}.expectedOutputs`,
        "distribution output path differs from the build lock",
      );
    }
  }
}

async function readDetachedSubject(root, output, buildInputLock) {
  const bytes = await readExactOutput(
    join(root, output.outputPath),
    output,
    `$.outputs.${output.outputPath}`,
  );
  let syntax;
  try {
    const decoded = decodeWireJson(bytes, {
      maxDocumentBytes: CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
      maxDepth: 16,
      maxNodes: 1_024,
      maxStringBytes: 192 * 1024,
      maxArrayLength: 32,
      maxObjectProperties: 64,
      maxRank: 1,
      maxIntegerBits: 64,
      maxArithmeticOperations: 2_048,
    });
    syntax = decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(decoded);
  } catch (cause) {
    invalid(
      `$.outputs.${output.outputPath}`,
      "detached provenance is not strict canonical BrowserGrad DSSE syntax",
      { cause },
    );
  }
  if (syntax.statement.predicate.buildInputLock.lockId !==
        buildInputLock.lockId ||
      syntax.statement.predicate.buildInputLock.resourceSha256 !==
        buildInputLock.resourceSha256 ||
      syntax.statement.predicate.buildInputLock.recipeSha256 !==
        buildInputLock.recipeSha256) {
    mismatch(
      `$.outputs.${output.outputPath}`,
      "detached provenance does not bind the prepared build-input lock",
    );
  }
  return Object.freeze({
    buildSubjectId: syntax.statement.subject[0].name,
    buildSubjectSha256: syntax.statement.subject[0].digest.sha256,
  });
}

async function readExactOutput(path, expected, diagnosticPath) {
  const byteLength = Number(expected.byteLength);
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 ||
      byteLength > CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT) {
    invalid(diagnosticPath, "detached provenance exceeds its byte bound");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n ||
        before.size !== BigInt(byteLength) ||
        Number(before.mode & 0o222n) !== 0) {
      invalid(diagnosticPath, "detached provenance is not one immutable file");
    }
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    while (offset < byteLength) {
      const { bytesRead } = await handle.read(
        bytes,
        offset,
        byteLength - offset,
        offset,
      );
      if (bytesRead <= 0) {
        invalid(diagnosticPath, "detached provenance changed while read");
      }
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after) ||
        sha256(bytes) !== expected.sha256) {
      invalid(diagnosticPath, "detached provenance identity changed while read");
    }
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserFullDistributionReproducibilityError) {
      throw cause;
    }
    invalid(diagnosticPath, "failed to read exact detached provenance", {
      cause,
    });
  } finally {
    await handle?.close();
  }
}

function requiredOutput(outputs, outputPath, path) {
  const output = outputs.get(outputPath);
  if (output === undefined) {
    mismatch(path, `distribution output ${outputPath} is missing`);
  }
  return output;
}

function sameOutput(left, right) {
  return left.outputPath === right.outputPath &&
    left.sha256 === right.sha256 &&
    left.byteLength === right.byteLength;
}

function pathsOverlap(left, right) {
  const first = normalize(left);
  const second = normalize(right);
  return first === second ||
    first.startsWith(`${second}/`) ||
    second.startsWith(`${first}/`);
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.nlink === right.nlink &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "expected one data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function absolutePath(value, path) {
  if (typeof value !== "string" || !value.startsWith("/") ||
      value.includes("\0") || normalize(value) !== value) {
    invalid(path, "expected one canonical absolute POSIX path");
  }
  return value;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function mismatch(path, message) {
  throw new CppCuteBrowserFullDistributionReproducibilityError(path, message);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserFullDistributionReproducibilityError(
    path,
    message,
    options,
  );
}
