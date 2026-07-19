import { createHash } from "node:crypto";
import { lstat, mkdir, realpath, rm } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  admitPinnedCppCuteBrowserArchiveNormalizationEnvironment,
} from "./cpp_cute_browser_archive_normalization.mjs";
import { admitCppCuteBrowserHeaderSourcePlanArchives } from "./cpp_cute_browser_header_source_archive_admission.mjs";
import {
  extractCppCuteBrowserHeaderSourcePlan,
  parseCppCuteBrowserHeaderSourceExtractionArguments,
} from "./cpp_cute_browser_header_source_extraction.mjs";
import {
  inventoryCppCuteBrowserExtractedHeaderSources,
} from "./cpp_cute_browser_header_pack_inventory.mjs";
import { materializeCppCuteBrowserHeaderPacks } from "./cpp_cute_browser_header_pack_materialization.mjs";

export const CPP_CUTE_BROWSER_HEADER_PACK_PIPELINE_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-pack-pipeline";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-PIPELINE";
const PIPELINE_HASH_DOMAIN = "browsergrad.compiler.cpp-cute.browser-header-pack-pipeline.v1";

export class CppCuteBrowserHeaderPackPipelineError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderPackPipelineError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Runs archive admission, collision-free extraction, exact virtual inventory,
 * and canonical VFS pack materialization in one process so opaque file
 * authorities never need serialization.
 */
export async function materializeCppCuteBrowserHeaderPacksFromSourceArchives(input) {
  const object = exactObject(
    input,
    ["archives", "bsdtarPath", "packOutputRoot", "sourceOutputRoot"],
    "$.input",
  );
  const bsdtarPath = absolutePath(object.bsdtarPath, "$.input.bsdtarPath");
  const sourceOutputRoot = absolutePath(object.sourceOutputRoot, "$.input.sourceOutputRoot");
  const packOutputRoot = absolutePath(object.packOutputRoot, "$.input.packOutputRoot");
  if (sourceOutputRoot === packOutputRoot) {
    invalid("$.input", "source and pack output roots must differ");
  }
  let archiveAdmission;
  let bsdtarTool;
  try {
    [archiveAdmission, bsdtarTool] = await Promise.all([
      admitCppCuteBrowserHeaderSourcePlanArchives({ archives: object.archives }),
      admitPinnedCppCuteBrowserArchiveNormalizationEnvironment({ executablePath: bsdtarPath }),
    ]);
  } catch (cause) {
    invalid("$.input", "failed to admit exact archives or host normalization tool", { cause });
  }
  let extraction;
  let inventory;
  let materialization;
  let packOutputIdentity;
  try {
    packOutputIdentity = await createCppCuteBrowserPrivatePackOutputRoot(packOutputRoot);
    extraction = await extractCppCuteBrowserHeaderSourcePlan({
      archiveAdmission,
      bsdtarTool,
      outputRoot: sourceOutputRoot,
    });
    inventory = await inventoryCppCuteBrowserExtractedHeaderSources(extraction);
    materialization = await materializeCppCuteBrowserHeaderPacks({
      inventory,
      outputRoot: packOutputRoot,
    });
  } catch (cause) {
    if (packOutputIdentity !== undefined) {
      try {
        await removeCppCuteBrowserOwnedPackOutputRoot(packOutputRoot, packOutputIdentity);
      } catch (cleanupCause) {
        invalid(
          "$.pipeline.cleanup",
          "pipeline failed and its owned pack-output root could not be removed",
          { cause: cleanupCause },
        );
      }
    }
    invalid("$.pipeline", "exact header-pack pipeline failed", { cause });
  }
  const pipelineHash = sha256(canonicalJsonBytes({
    domain: PIPELINE_HASH_DOMAIN,
    archiveAdmissionId: archiveAdmission.admissionId,
    extractionId: extraction.extractionId,
    inventoryId: inventory.inventoryId,
    outputs: materialization.outputs,
  }));
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_PACK_PIPELINE_SCHEMA,
    version: 1,
    pipelineId: `bg.cpp.browser-header-pack-pipeline.sha256.${pipelineHash}`,
    authority: "exact-source-host-tool-vfs-pack-pipeline-observation-only",
    buildInputLockId: extraction.buildInputLockId,
    buildInputLockResourceSha256: extraction.buildInputLockResourceSha256,
    headerSourcePlanId: extraction.headerSourcePlanId,
    archiveAdmissionId: archiveAdmission.admissionId,
    extractionId: extraction.extractionId,
    inventoryId: inventory.inventoryId,
    bsdtarTool: extraction.bsdtarTool,
    sourceTotals: extraction.totals,
    inventoryTotals: inventory.totals,
    outputs: materialization.outputs,
    totalPackByteLength: materialization.totalPackByteLength,
    unresolvedBlockers: extraction.unresolvedBlockers,
    claims: Object.freeze({
      exactCurrentHeaderSourcePlanArchiveBytesVerified: true,
      collisionFreePortableStorageMaterialized: true,
      exactExtractedSourceBytesInventoried: true,
      canonicalVfsPacksIndependentlyInspected: true,
      allFiveSelectedSourcePacksMaterialized: true,
      exactSelectedSourceSubtreesComplete: true,
      hostToolImplementationAttested: false,
      hostToolPackageIdentityPinned: true,
      nodeZstdDecompressorPackageIdentityPinned: true,
      generatedClangResourceHeadersComplete: true,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      headerUniverseComplete: true,
      buildExecuted: false,
      releaseReady: false,
    }),
  });
}

export async function createCppCuteBrowserPrivatePackOutputRoot(outputRoot) {
  absolutePath(outputRoot, "$.input.packOutputRoot");
  const parent = dirname(outputRoot);
  let resolvedParent;
  let parentStat;
  try {
    resolvedParent = await realpath(parent);
    parentStat = await lstat(parent, { bigint: true });
  } catch (cause) {
    invalid("$.input.packOutputRoot.parent", "parent is unavailable", { cause });
  }
  if (resolvedParent !== parent) invalid("$.input.packOutputRoot.parent", "parent must be canonical");
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink() ||
      Number(parentStat.mode & 0o077n) !== 0) {
    invalid("$.input.packOutputRoot.parent", "parent must be one private directory");
  }
  let root;
  try {
    await mkdir(outputRoot, { mode: 0o700 });
    root = await lstat(outputRoot, { bigint: true });
  } catch (cause) {
    invalid("$.input.packOutputRoot", "pack output root must not already exist", { cause });
  }
  if (!root.isDirectory() || root.isSymbolicLink() || Number(root.mode & 0o077n) !== 0) {
    invalid("$.input.packOutputRoot", "created pack output root is not private");
  }
  return Object.freeze({ dev: root.dev, ino: root.ino });
}

async function removeCppCuteBrowserOwnedPackOutputRoot(outputRoot, identity) {
  const root = await lstat(outputRoot, { bigint: true });
  if (!root.isDirectory() || root.dev !== identity.dev || root.ino !== identity.ino) {
    invalid("$.input.packOutputRoot", "refusing to remove a replaced pack output root");
  }
  await rm(outputRoot, { recursive: true, force: false, maxRetries: 0 });
}

export function parseCppCuteBrowserHeaderPackPipelineArguments(argv) {
  if (!Array.isArray(argv)) invalid("$arguments", "expected one argument array");
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  const pipelineArguments = [];
  let packOutputRoot;
  for (const [index, argument] of arguments_.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--pack-output-root=(.+)$/u.exec(argument);
    if (match === null) {
      pipelineArguments.push(argument);
      continue;
    }
    if (packOutputRoot !== undefined) invalid(`$arguments[${index}]`, "duplicate --pack-output-root");
    packOutputRoot = absolutePath(match[1], "$arguments.pack-output-root");
  }
  if (packOutputRoot === undefined) invalid("$arguments", "missing --pack-output-root");
  let extraction;
  try {
    extraction = parseCppCuteBrowserHeaderSourceExtractionArguments(pipelineArguments);
  } catch (cause) {
    invalid("$arguments", "invalid header-source extraction arguments", { cause });
  }
  return Object.freeze({
    archives: extraction.archives,
    bsdtarPath: extraction.bsdtarPath,
    sourceOutputRoot: extraction.outputRoot,
    packOutputRoot,
  });
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
  if (typeof value !== "string" || !isAbsolute(value) || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderPackPipelineError(path, message, options);
}

async function main() {
  try {
    const options = parseCppCuteBrowserHeaderPackPipelineArguments(process.argv.slice(2));
    const report = await materializeCppCuteBrowserHeaderPacksFromSourceArchives(options);
    process.stdout.write(`${JSON.stringify({
      sourceOutputRoot: options.sourceOutputRoot,
      packOutputRoot: options.packOutputRoot,
      ...report,
    })}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-pack pipeline failure");
    process.stderr.write(`${formatBoundedErrorChain(error)}\n`);
    process.exitCode = 1;
  }
}

function formatBoundedErrorChain(error) {
  const messages = [];
  const seen = new Set();
  let current = error;
  while (current instanceof Error && messages.length < 6 && !seen.has(current)) {
    seen.add(current);
    const path = typeof current.path === "string" ? ` at ${current.path}` : "";
    const message = `${current.name}${path}: ${current.message}`
      .replace(/[\r\n\u2028\u2029]+/gu, " ")
      .slice(0, 2_048);
    messages.push(message);
    current = current.cause;
  }
  return messages.join(" <- caused by ");
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
