import { createHash } from "node:crypto";
import { normalize } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  verifyCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";
import {
  materializeCppCuteBrowserHeaderPacksFromSourceArchives,
  parseCppCuteBrowserHeaderPackPipelineArguments,
  requireCppCuteBrowserHeaderPackPipelineAuthority,
} from "./cpp_cute_browser_header_pack_pipeline.mjs";

export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-DISTRIBUTION-REPRODUCIBILITY";
const REPRODUCIBILITY_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility.v1";
const EXPECTED_PACK_COUNT = 5;
const EXPECTED_COMPONENT_NOTICE_COUNT = 10;
const EXPECTED_OUTPUT_COUNT = 17;
const REPRODUCIBILITY_AUTHORITIES = new WeakSet();
const REPRODUCIBILITY_INPUTS = new WeakSet();

export class CppCuteBrowserHeaderDistributionReproducibilityError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderDistributionReproducibilityError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export async function materializeAndVerifyCppCuteBrowserHeaderDistributionReproducibility(input) {
  if (typeof input !== "object" || input === null || !REPRODUCIBILITY_INPUTS.has(input)) {
    invalid("$.input", "expected parser-issued two-root reproducibility input authority");
  }
  const object = exactObject(input, ["first", "second"], "$.input");
  assertDistinctPipelineInputs(object.first, object.second);
  let first;
  let second;
  try {
    first = await materializeCppCuteBrowserHeaderPacksFromSourceArchives(object.first);
    second = await materializeCppCuteBrowserHeaderPacksFromSourceArchives(object.second);
  } catch (cause) {
    invalid("$.input", "one exact header-distribution materialization failed", { cause });
  }
  return verifyCppCuteBrowserHeaderDistributionReproducibility({ first, second });
}

/**
 * Rehashes and compares two live exact header-pipeline output trees. This
 * proves reproducibility only for the five packs, review input, and eleven
 * notice outputs; it cannot approve licenses or the complete distribution.
 */
export async function verifyCppCuteBrowserHeaderDistributionReproducibility(input) {
  const object = exactObject(input, ["first", "second"], "$.input");
  try {
    requireCppCuteBrowserHeaderPackPipelineAuthority(object.first);
    requireCppCuteBrowserHeaderPackPipelineAuthority(object.second);
  } catch (cause) {
    invalid("$.input", "expected two live exact header-pack pipeline authorities", { cause });
  }
  if (object.first === object.second) {
    invalid("$.input", "reproducibility requires two distinct pipeline authority instances");
  }
  assertPipelineIdentityParity(object.first, object.second);
  const firstRoot = object.first.noticeMaterialization.outputRoot;
  const secondRoot = object.second.noticeMaterialization.outputRoot;
  if (pathsOverlap(firstRoot, secondRoot)) {
    invalid("$.input", "reproducibility output roots must be distinct and non-overlapping");
  }
  const firstOutputs = exactDistributionOutputs(object.first, "$.input.first");
  const secondOutputs = exactDistributionOutputs(object.second, "$.input.second");
  if (!sameOutputs(firstOutputs, secondOutputs)) {
    mismatch("$.input", "pipeline output identities differ before filesystem verification");
  }
  let firstVerification;
  let secondVerification;
  try {
    [firstVerification, secondVerification] = await Promise.all([
      verifyCppCuteBrowserDistributionOutputFiles({
        outputRoot: firstRoot,
        expectedOutputs: firstOutputs,
      }),
      verifyCppCuteBrowserDistributionOutputFiles({
        outputRoot: secondRoot,
        expectedOutputs: secondOutputs,
      }),
    ]);
  } catch (cause) {
    invalid("$.input", "failed to reverify both exact distribution output trees", { cause });
  }
  if (firstVerification.verificationId !== secondVerification.verificationId ||
      !sameOutputs(firstVerification.outputs, secondVerification.outputs) ||
      firstVerification.totals.fileCount !== EXPECTED_OUTPUT_COUNT ||
      firstVerification.totals.byteLength !== secondVerification.totals.byteLength) {
    mismatch("$.input", "independently reverified output trees are not reproducible");
  }
  const reproducibilityHash = sha256(canonicalJsonBytes({
    domain: REPRODUCIBILITY_HASH_DOMAIN,
    pipelineId: object.first.pipelineId,
    outputVerificationId: firstVerification.verificationId,
    outputs: firstVerification.outputs,
  }));
  const report = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_SCHEMA,
    version: 1,
    reproducibilityId:
      `bg.cpp.browser-header-distribution-reproducibility.sha256.${reproducibilityHash}`,
    authority: "two-root-exact-header-distribution-reproducibility-only",
    scope: "five-header-packs-license-inventory-and-notice-outputs-only",
    buildInputLockId: object.first.buildInputLockId,
    pipelineId: object.first.pipelineId,
    outputVerificationId: firstVerification.verificationId,
    firstOutputRoot: firstRoot,
    secondOutputRoot: secondRoot,
    outputs: firstVerification.outputs,
    totals: Object.freeze({
      outputCount: firstVerification.totals.fileCount,
      byteLength: firstVerification.totals.byteLength,
    }),
    claims: Object.freeze({
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
    }),
  });
  REPRODUCIBILITY_AUTHORITIES.add(report);
  return report;
}

export function requireCppCuteBrowserHeaderDistributionReproducibilityAuthority(report) {
  if (typeof report !== "object" || report === null || !REPRODUCIBILITY_AUTHORITIES.has(report)) {
    invalid("$.report", "expected verifier-issued header-distribution reproducibility authority");
  }
}

export function canonicalCppCuteBrowserHeaderDistributionReproducibilityBytes(report) {
  requireCppCuteBrowserHeaderDistributionReproducibilityAuthority(report);
  return canonicalJsonBytes(report);
}

export function parseCppCuteBrowserHeaderDistributionReproducibilityArguments(argv) {
  if (!Array.isArray(argv)) invalid("$arguments", "expected one argument array");
  const arguments_ = argv[0] === "--" ? argv.slice(1) : [...argv];
  const roots = new Map();
  const common = [];
  const rootArguments = new Map([
    ["first-source-output-root", "firstSourceOutputRoot"],
    ["first-pack-output-root", "firstPackOutputRoot"],
    ["second-source-output-root", "secondSourceOutputRoot"],
    ["second-pack-output-root", "secondPackOutputRoot"],
  ]);
  for (const [index, argument] of arguments_.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--([a-z-]+)=(.+)$/u.exec(argument);
    const field = match === null ? undefined : rootArguments.get(match[1]);
    if (field === undefined) {
      if (/^--(?:output-root|pack-output-root)=/u.test(argument)) {
        invalid(`$arguments[${index}]`, "use the explicit first/second output-root arguments");
      }
      common.push(argument);
      continue;
    }
    if (roots.has(field)) invalid(`$arguments[${index}]`, `duplicate --${match[1]}`);
    roots.set(field, match[2]);
  }
  for (const field of rootArguments.values()) {
    if (!roots.has(field)) invalid("$arguments", `missing ${field}`);
  }
  let first;
  let second;
  try {
    first = parseCppCuteBrowserHeaderPackPipelineArguments([
      ...common,
      `--output-root=${roots.get("firstSourceOutputRoot")}`,
      `--pack-output-root=${roots.get("firstPackOutputRoot")}`,
    ]);
    second = parseCppCuteBrowserHeaderPackPipelineArguments([
      ...common,
      `--output-root=${roots.get("secondSourceOutputRoot")}`,
      `--pack-output-root=${roots.get("secondPackOutputRoot")}`,
    ]);
  } catch (cause) {
    invalid("$arguments", "invalid two-root header-distribution arguments", { cause });
  }
  assertDistinctPipelineInputs(first, second);
  const input = Object.freeze({ first, second });
  REPRODUCIBILITY_INPUTS.add(input);
  return input;
}

function assertPipelineIdentityParity(first, second) {
  const fields = [
    "pipelineId",
    "buildInputLockId",
    "buildInputLockResourceSha256",
    "headerSourcePlanId",
    "archiveAdmissionId",
    "extractionId",
    "inventoryId",
  ];
  for (const field of fields) {
    if (first[field] !== second[field]) {
      mismatch(`$.input.second.${field}`, "pipeline identity differs between exact runs");
    }
  }
  if (first.distributionReviewInput.reviewInputId !==
      second.distributionReviewInput.reviewInputId ||
      first.noticeMaterialization.noticeMaterializationId !==
      second.noticeMaterialization.noticeMaterializationId) {
    mismatch("$.input.second", "review or notice materialization identity differs between exact runs");
  }
}

function assertDistinctPipelineInputs(first, second) {
  const firstCommon = canonicalJsonBytes({
    archives: first.archives,
    bsdtarPath: first.bsdtarPath,
    cudaRedistributionIndexPath: first.cudaRedistributionIndexPath,
  });
  const secondCommon = canonicalJsonBytes({
    archives: second.archives,
    bsdtarPath: second.bsdtarPath,
    cudaRedistributionIndexPath: second.cudaRedistributionIndexPath,
  });
  if (!sameBytes(firstCommon, secondCommon)) {
    invalid("$.input", "both materializations must use the same exact archives, index, and host tool");
  }
  const roots = [
    first.sourceOutputRoot,
    first.packOutputRoot,
    second.sourceOutputRoot,
    second.packOutputRoot,
  ];
  for (const [index, root] of roots.entries()) {
    if (normalize(root) !== root || root.endsWith("/")) {
      invalid("$.input", "all source and pack output roots must be canonical absolute path spellings");
    }
    for (let other = 0; other < index; other += 1) {
      if (pathsOverlap(root, roots[other])) {
        invalid("$.input", "all source and pack output roots must be distinct and non-overlapping");
      }
    }
  }
}

function exactDistributionOutputs(report, diagnosticPath) {
  if (report.outputs.length !== EXPECTED_PACK_COUNT ||
      report.noticeMaterialization.componentOutputs.length !== EXPECTED_COMPONENT_NOTICE_COUNT) {
    invalid(diagnosticPath, "pipeline does not contain the exact current distribution-output subset");
  }
  const outputs = [
    ...report.outputs.map((output) => Object.freeze({
      outputPath: output.outputPath,
      sha256: output.packSha256,
      byteLength: output.packByteLength,
    })),
    Object.freeze({
      outputPath: report.distributionReviewInput.outputPath,
      sha256: report.distributionReviewInput.reviewInputSha256,
      byteLength: report.distributionReviewInput.reviewInputByteLength,
    }),
    ...report.noticeMaterialization.componentOutputs.map((output) => Object.freeze({
      outputPath: output.outputPath,
      sha256: output.sha256,
      byteLength: output.byteLength,
    })),
    Object.freeze({
      outputPath: report.noticeMaterialization.aggregateOutput.outputPath,
      sha256: report.noticeMaterialization.aggregateOutput.sha256,
      byteLength: report.noticeMaterialization.aggregateOutput.byteLength,
    }),
  ].sort((left, right) => compareUtf8(left.outputPath, right.outputPath));
  if (outputs.length !== EXPECTED_OUTPUT_COUNT ||
      outputs.some((output, index) => index > 0 &&
        outputs[index - 1]?.outputPath === output.outputPath)) {
    invalid(diagnosticPath, "pipeline distribution-output paths are not the exact unique set");
  }
  return Object.freeze(outputs);
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${diagnosticPath}.${key}`, "expected data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function pathsOverlap(left, right) {
  return left === right || left.startsWith(`${right}/`) || right.startsWith(`${left}/`);
}

function sameOutputs(left, right) {
  return left.length === right.length && left.every((output, index) => {
    const candidate = right[index];
    return candidate !== undefined && output.outputPath === candidate.outputPath &&
      output.sha256 === candidate.sha256 && output.byteLength === candidate.byteLength;
  });
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  return Buffer.from(left.buffer, left.byteOffset, left.byteLength)
    .equals(Buffer.from(right.buffer, right.byteOffset, right.byteLength));
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function mismatch(path, message) {
  invalid(path, `reproducibility mismatch: ${message}`);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderDistributionReproducibilityError(path, message, options);
}

async function main() {
  try {
    const input = parseCppCuteBrowserHeaderDistributionReproducibilityArguments(
      process.argv.slice(2),
    );
    const report = await materializeAndVerifyCppCuteBrowserHeaderDistributionReproducibility(input);
    process.stdout.write(`${JSON.stringify({
      firstSourceOutputRoot: input.first.sourceOutputRoot,
      secondSourceOutputRoot: input.second.sourceOutputRoot,
      ...report,
    })}\n`);
  } catch (cause) {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown header-distribution reproducibility failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
