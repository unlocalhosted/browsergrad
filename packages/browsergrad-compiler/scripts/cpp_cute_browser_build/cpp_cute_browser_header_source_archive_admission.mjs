import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  canonicalCppCuteBrowserHeaderSourcePlanBytes,
  prepareCppCuteBrowserHeaderSourcePlan,
} from "./cpp_cute_browser_header_source_plan.mjs";
import {
  copyCppCuteBrowserInspectedRegularArchive,
  inspectCppCuteBrowserRegularArchiveFiles,
} from "./cpp_cute_browser_source_archive_admission.mjs";

export const CPP_CUTE_BROWSER_HEADER_SOURCE_ARCHIVE_ADMISSION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-source-archive-admission";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-SOURCE-ARCHIVE-ADMISSION";
const ADMISSION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-source-archive-admission.v1";
const SOURCE_ID = /^[a-z][a-z0-9-]*$/u;
const HEADER_SOURCE_ARCHIVE_IDS = Object.freeze([
  "cuda-cccl-linux-x86-64",
  "cuda-cudart-linux-x86-64",
  "cuda-nvcc-linux-x86-64",
  "cutlass",
  "llvm-project",
  "ubuntu-noble-libc6-dev-amd64-cross",
  "ubuntu-noble-linux-libc-dev-amd64-cross",
]);
const ADMITTED_HEADER_ARCHIVES = new WeakMap();

export class CppCuteBrowserHeaderSourceArchiveAdmissionError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderSourceArchiveAdmissionError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Admits exactly the seven archives selected by the current header-source plan.
 * File verification remains in the reusable regular-archive boundary; this
 * wrapper alone binds those observations to package policy.
 */
export async function admitCppCuteBrowserHeaderSourcePlanArchives(input) {
  const object = exactObject(input, ["archives"], "$.input");
  const archives = parseArchiveInputs(object.archives, "$.input.archives");
  const plan = await prepareCppCuteBrowserHeaderSourcePlan();
  const expectedIds = plan.body.archives.map((archive) => archive.sourceId);
  if (!sameStrings(expectedIds, HEADER_SOURCE_ARCHIVE_IDS)) {
    invalid("$.plan.body.archives", "header-source plan differs from the closed admission set");
  }
  if (!sameStrings(archives.map((archive) => archive.sourceId), expectedIds)) {
    invalid("$.input.archives", "input must cover the exact current header-source archive set");
  }
  let inspection;
  try {
    inspection = await inspectCppCuteBrowserRegularArchiveFiles({
      archives,
      expectedArchives: plan.body.archives.map((archive) => ({
        sourceId: archive.sourceId,
        archiveSha256: archive.archiveSha256,
        archiveByteLength: archive.archiveByteLength,
      })),
    });
  } catch (cause) {
    const detail = cause instanceof Error ? `: ${cause.message}` : "";
    invalid("$.input.archives", `archive verification failed${detail}`, { cause });
  }
  const observations = plan.body.archives.map((archive, index) => {
    const observed = inspection.archives[index];
    if (observed === undefined || observed.sourceId !== archive.sourceId ||
        observed.observedArchiveSha256 !== archive.archiveSha256 ||
        observed.observedArchiveByteLength !== archive.archiveByteLength) {
      invalid("$.inspection.archives", "archive observation differs from the exact source plan");
    }
    return Object.freeze({
      sourceId: archive.sourceId,
      sourceKind: archive.sourceKind,
      provider: archive.provider,
      version: archive.version,
      acquisitionUrl: archive.acquisitionUrl,
      archiveFormat: archive.archiveFormat,
      archiveSha256: archive.archiveSha256,
      archiveByteLength: archive.archiveByteLength,
      observedArchiveSha256: observed.observedArchiveSha256,
      observedArchiveByteLength: observed.observedArchiveByteLength,
      licenseComponentId: archive.licenseComponentId,
      licensePolicy: archive.licensePolicy,
      selections: archive.selections,
    });
  });
  const planBytes = canonicalCppCuteBrowserHeaderSourcePlanBytes(plan);
  const planResourceSha256 = sha256(planBytes);
  const admissionHash = sha256(canonicalJsonBytes({
    domain: ADMISSION_HASH_DOMAIN,
    headerSourcePlanId: plan.planId,
    headerSourcePlanSha256: planResourceSha256,
    expectedArchiveSetSha256: inspection.expectedArchiveSetSha256,
    archives: observations,
  }));
  const admission = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_SOURCE_ARCHIVE_ADMISSION_SCHEMA,
    version: 1,
    admissionId: `bg.cpp.browser-header-source-archive-admission.sha256.${admissionHash}`,
    authority: "exact-current-header-source-plan-archive-admission-only",
    buildInputLockId: plan.body.buildInputLockId,
    buildInputLockResourceSha256: plan.body.buildInputLockResourceSha256,
    headerSourcePlanId: plan.planId,
    headerSourcePlanSha256: planResourceSha256,
    headerSourcePlanByteLength: planBytes.byteLength,
    expectedArchiveSetSha256: inspection.expectedArchiveSetSha256,
    archives: Object.freeze(observations),
    totals: inspection.totals,
    unresolvedBlockers: plan.body.unresolvedBlockers,
    claims: Object.freeze({
      exactCurrentHeaderSourcePlanArchiveBytesVerified: true,
      exactBuildInputLockBound: true,
      exactHeaderSourcePlanBound: true,
      localArchivePathsRetainedOpaquely: true,
      networkAccessed: false,
      archiveAttestationsVerified: false,
      sourceSubtreesExtracted: false,
      generatedClangResourceHeadersComplete: false,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      headerUniverseComplete: false,
      headerPacksAssembled: false,
      buildExecuted: false,
      releaseReady: false,
    }),
  });
  ADMITTED_HEADER_ARCHIVES.set(admission, Object.freeze({ inspection }));
  return admission;
}

export function requireCppCuteBrowserHeaderSourceArchiveAuthority(admission) {
  if (typeof admission !== "object" || admission === null ||
      ADMITTED_HEADER_ARCHIVES.get(admission) === undefined) {
    invalid("$.admission", "expected verifier-issued current header-source archive authority");
  }
}

export function canonicalCppCuteBrowserHeaderSourceArchiveAdmissionBytes(admission) {
  requireCppCuteBrowserHeaderSourceArchiveAuthority(admission);
  return canonicalJsonBytes(admission);
}

export async function copyCppCuteBrowserHeaderSourceArchive(
  admission,
  sourceId,
  outputPath,
) {
  const stored = ADMITTED_HEADER_ARCHIVES.get(admission);
  if (stored === undefined) {
    invalid("$.admission", "expected verifier-issued current header-source archive authority");
  }
  try {
    return await copyCppCuteBrowserInspectedRegularArchive(
      stored.inspection,
      sourceId,
      outputPath,
    );
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderSourceArchiveAdmissionError) throw cause;
    invalid("$.archiveCopy", "failed to copy exact header-source archive", { cause });
  }
}

export function parseCppCuteBrowserHeaderSourceArchiveArguments(argv) {
  if (!Array.isArray(argv) || argv.length !== HEADER_SOURCE_ARCHIVE_IDS.length) {
    invalid("$arguments", `expected exactly ${HEADER_SOURCE_ARCHIVE_IDS.length} source-path arguments`);
  }
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--([a-z][a-z0-9-]*)=(.+)$/u.exec(argument);
    if (match === null) invalid(`$arguments[${index}]`, "expected --source-id=/absolute/path");
    const [, sourceId, archivePath] = match;
    if (!HEADER_SOURCE_ARCHIVE_IDS.includes(sourceId)) {
      invalid(`$arguments[${index}]`, "source ID is absent from the current header-source plan");
    }
    if (values.has(sourceId)) invalid(`$arguments[${index}]`, `duplicate --${sourceId}`);
    values.set(sourceId, absolutePath(archivePath, `$arguments.${sourceId}`));
  }
  for (const sourceId of HEADER_SOURCE_ARCHIVE_IDS) {
    if (!values.has(sourceId)) invalid("$arguments", `missing --${sourceId}`);
  }
  return Object.freeze({
    archives: Object.freeze(HEADER_SOURCE_ARCHIVE_IDS.map((sourceId) => Object.freeze({
      sourceId,
      archivePath: values.get(sourceId),
    }))),
  });
}

function parseArchiveInputs(value, diagnosticPath) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype ||
      value.length !== HEADER_SOURCE_ARCHIVE_IDS.length) {
    invalid(diagnosticPath, `expected a dense ${HEADER_SOURCE_ARCHIVE_IDS.length}-archive array`);
  }
  const parsed = [];
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.hasOwn(value, index)) invalid(`${diagnosticPath}[${index}]`, "sparse arrays are forbidden");
    const path = `${diagnosticPath}[${index}]`;
    const object = exactObject(value[index], ["sourceId", "archivePath"], path);
    if (typeof object.sourceId !== "string" || !SOURCE_ID.test(object.sourceId)) {
      invalid(`${path}.sourceId`, "expected one portable source ID");
    }
    parsed.push(Object.freeze({
      sourceId: object.sourceId,
      archivePath: absolutePath(object.archivePath, `${path}.archivePath`),
    }));
  }
  parsed.sort((left, right) => compareUtf8(left.sourceId, right.sourceId));
  for (let index = 1; index < parsed.length; index += 1) {
    if (parsed[index - 1].sourceId === parsed[index].sourceId) {
      invalid(diagnosticPath, "source IDs must be unique");
    }
  }
  return Object.freeze(parsed);
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

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !value.startsWith("/") || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function sameStrings(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderSourceArchiveAdmissionError(path, message, options);
}

async function main() {
  try {
    const input = parseCppCuteBrowserHeaderSourceArchiveArguments(process.argv.slice(2));
    const admission = await admitCppCuteBrowserHeaderSourcePlanArchives(input);
    process.stdout.write(`${new TextDecoder().decode(
      canonicalCppCuteBrowserHeaderSourceArchiveAdmissionBytes(admission),
    )}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-source archive admission failure");
    const path = typeof cause === "object" && cause !== null && "path" in cause &&
      typeof cause.path === "string"
      ? ` at ${cause.path}`
      : "";
    process.stderr.write(`${error.name}${path}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
