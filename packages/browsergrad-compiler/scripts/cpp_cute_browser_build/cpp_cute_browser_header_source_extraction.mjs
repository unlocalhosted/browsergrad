import { createHash } from "node:crypto";
import { lstat, mkdir, realpath, rm, unlink } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  admitCppCuteBrowserBsdtarTool,
  copyCppCuteBrowserArchiveNormalizationFile,
  cppCuteBrowserArchiveNormalizationRoots,
  materializeCppCuteBrowserNormalizedArchive,
  requireCppCuteBrowserBsdtarToolAuthority,
} from "./cpp_cute_browser_archive_normalization.mjs";
import {
  admitCppCuteBrowserHeaderSourcePlanArchives,
  copyCppCuteBrowserHeaderSourceArchive,
  parseCppCuteBrowserHeaderSourceArchiveArguments,
  requireCppCuteBrowserHeaderSourceArchiveAuthority,
} from "./cpp_cute_browser_header_source_archive_admission.mjs";
import { prepareCppCuteBrowserHeaderSourcePlan } from "./cpp_cute_browser_header_source_plan.mjs";

export const CPP_CUTE_BROWSER_HEADER_SOURCE_EXTRACTION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-source-extraction";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-SOURCE-EXTRACTION";
const EXTRACTION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-source-extraction.v1";
const SOURCE_EXTRACTIONS = new WeakMap();

export class CppCuteBrowserHeaderSourceExtractionError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderSourceExtractionError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Copies the seven exact admitted archives into a private staging directory,
 * normalizes their eight selected subtrees, then removes all archive copies.
 * The output is a collision-free virtual-path-backed source store, not a
 * host-filesystem-shaped include tree.
 */
export async function extractCppCuteBrowserHeaderSourcePlan(input) {
  const object = exactObject(
    input,
    ["archiveAdmission", "bsdtarTool", "outputRoot"],
    "$.input",
  );
  try {
    requireCppCuteBrowserHeaderSourceArchiveAuthority(object.archiveAdmission);
  } catch (cause) {
    invalid("$.input.archiveAdmission", "expected exact current header-source archive authority", { cause });
  }
  try {
    requireCppCuteBrowserBsdtarToolAuthority(object.bsdtarTool);
  } catch (cause) {
    invalid("$.input.bsdtarTool", "expected observed bsdtar tool authority", { cause });
  }
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  await admitPrivateCanonicalDirectory(dirname(outputRoot), "$.input.outputRoot.parent");
  const plan = await prepareCppCuteBrowserHeaderSourcePlan();
  bindAdmissionToPlan(object.archiveAdmission, plan);
  let rootIdentity;
  let stagingIdentity;
  try {
    await mkdir(outputRoot, { mode: 0o700 });
    const root = await lstat(outputRoot, { bigint: true });
    requirePrivateDirectory(root, "$.input.outputRoot");
    rootIdentity = Object.freeze({ dev: root.dev, ino: root.ino });
    const stagingRoot = join(outputRoot, "staging");
    const treesRoot = join(outputRoot, "trees");
    await mkdir(stagingRoot, { mode: 0o700 });
    await mkdir(treesRoot, { mode: 0o700 });
    const [staging, trees] = await Promise.all([
      lstat(stagingRoot, { bigint: true }),
      lstat(treesRoot, { bigint: true }),
    ]);
    requirePrivateDirectory(staging, "$.output.staging");
    requirePrivateDirectory(trees, "$.output.trees");
    stagingIdentity = Object.freeze({ dev: staging.dev, ino: staging.ino });

    const archives = [];
    const normalizations = new Map();
    let fileCount = 0;
    let fileContentBytes = 0n;
    for (const source of plan.body.archives) {
      const stagedArchivePath = join(stagingRoot, `${source.sourceId}.archive`);
      let stagedArchiveIdentity;
      try {
        const copy = await copyCppCuteBrowserHeaderSourceArchive(
          object.archiveAdmission,
          source.sourceId,
          stagedArchivePath,
        );
        if (copy.archiveSha256 !== source.archiveSha256 ||
            copy.archiveByteLength !== source.archiveByteLength) {
          invalid(`$.archives.${source.sourceId}`, "staged archive copy differs from the exact plan");
        }
        const staged = await lstat(stagedArchivePath, { bigint: true });
        stagedArchiveIdentity = Object.freeze({ dev: staged.dev, ino: staged.ino });
        const normalization = await materializeCppCuteBrowserNormalizedArchive({
          archiveFormat: source.archiveFormat,
          archivePath: stagedArchivePath,
          outputRoot: join(treesRoot, source.sourceId),
          selections: source.selections.map((selection) => Object.freeze({
            selectionId: selection.includeRootId,
            archiveSubtree: normalizedArchiveSubtree(source.archiveFormat, selection.archiveSubtree),
            outputSubdirectory: selection.includeRootId,
          })),
          tool: object.bsdtarTool,
        });
        if (normalization.observedArchiveSha256 !== source.archiveSha256 ||
            normalization.observedArchiveByteLength !== source.archiveByteLength) {
          invalid(`$.archives.${source.sourceId}`, "normalizer observed bytes outside the exact source plan");
        }
        const selections = source.selections.map((selection) => {
          const observed = normalization.selections.find(
            ({ selectionId }) => selectionId === selection.includeRootId,
          );
          if (observed === undefined) {
            invalid(`$.archives.${source.sourceId}`, "normalizer omitted a selected source subtree");
          }
          return Object.freeze({
            includeRootId: selection.includeRootId,
            archiveSubtree: selection.archiveSubtree,
            virtualPrefix: selection.virtualPrefix,
            intendedAsset: selection.intendedAsset,
            licenseComponentIds: selection.licenseComponentIds,
            contribution: selection.contribution,
            sourceTreeId: observed.sourceTreeId,
            fileCount: observed.fileCount,
            fileContentByteLength: observed.fileContentByteLength,
          });
        });
        fileCount += normalization.totals.fileCount;
        fileContentBytes += BigInt(normalization.totals.fileContentByteLength);
        archives.push(Object.freeze({
          sourceId: source.sourceId,
          archiveFormat: source.archiveFormat,
          archiveSha256: source.archiveSha256,
          archiveByteLength: source.archiveByteLength,
          licenseComponentId: source.licenseComponentId,
          licensePolicy: source.licensePolicy,
          normalizationId: normalization.normalizationId,
          selections: Object.freeze(selections),
        }));
        normalizations.set(source.sourceId, normalization);
      } finally {
        if (stagedArchiveIdentity !== undefined) {
          if (!await unlinkOwnedFile(stagedArchivePath, stagedArchiveIdentity)) {
            invalid(`$.archives.${source.sourceId}`, "failed to remove the owned staged archive copy");
          }
        }
      }
    }
    if (!await removeOwnedRoot(stagingRoot, stagingIdentity)) {
      invalid("$.output.staging", "failed to remove the owned archive staging directory");
    }
    stagingIdentity = undefined;
    const extractionHash = sha256(canonicalJsonBytes({
      domain: EXTRACTION_HASH_DOMAIN,
      archiveAdmissionId: object.archiveAdmission.admissionId,
      headerSourcePlanId: plan.planId,
      bsdtarToolAdmissionId: object.bsdtarTool.toolAdmissionId,
      archives,
      unresolvedBlockers: plan.body.unresolvedBlockers,
    }));
    const extraction = Object.freeze({
      schema: CPP_CUTE_BROWSER_HEADER_SOURCE_EXTRACTION_SCHEMA,
      version: 1,
      extractionId: `bg.cpp.browser-header-source-extraction.sha256.${extractionHash}`,
      authority: "exact-plan-host-tool-source-materialization-observation-only",
      buildInputLockId: plan.body.buildInputLockId,
      buildInputLockResourceSha256: plan.body.buildInputLockResourceSha256,
      headerSourcePlanId: plan.planId,
      archiveAdmissionId: object.archiveAdmission.admissionId,
      bsdtarTool: Object.freeze({
        toolAdmissionId: object.bsdtarTool.toolAdmissionId,
        executableSha256: object.bsdtarTool.executableSha256,
        executableByteLength: object.bsdtarTool.executableByteLength,
        observedVersion: object.bsdtarTool.observedVersion,
      }),
      archives: Object.freeze(archives),
      totals: Object.freeze({
        archiveCount: archives.length,
        selectedSubtreeCount: archives.reduce((total, source) => total + source.selections.length, 0),
        fileCount,
        fileContentByteLength: String(fileContentBytes),
      }),
      unresolvedBlockers: plan.body.unresolvedBlockers,
      claims: Object.freeze({
        exactCurrentHeaderSourcePlanArchiveBytesVerified: true,
        exactBuildInputLockBound: true,
        exactHeaderSourcePlanBound: true,
        sourceSubtreeMaterializationsObserved: true,
        exactSelectedSourceSubtreesComplete: false,
        collisionFreePortableStorageMaterialized: true,
        allFiveIncludeRootsRepresented: true,
        copiedSourceArchivesRemoved: true,
        hostToolImplementationAttested: false,
        generatedClangResourceHeadersComplete: false,
        externalDistributedFileLicenseMapReviewed: false,
        licenseReviewComplete: false,
        headerUniverseComplete: false,
        headerPacksAssembled: false,
        buildExecuted: false,
        releaseReady: false,
      }),
    });
    SOURCE_EXTRACTIONS.set(extraction, Object.freeze({ normalizations, outputRoot, rootIdentity }));
    return extraction;
  } catch (cause) {
    if (rootIdentity !== undefined) await removeOwnedRoot(outputRoot, rootIdentity);
    if (cause instanceof CppCuteBrowserHeaderSourceExtractionError) throw cause;
    invalid("$.input", "failed to extract exact header-source plan", { cause });
  }
}

export function requireCppCuteBrowserHeaderSourceExtractionAuthority(extraction) {
  if (typeof extraction !== "object" || extraction === null ||
      SOURCE_EXTRACTIONS.get(extraction) === undefined) {
    invalid("$.extraction", "expected extractor-issued current header-source authority");
  }
}

export function canonicalCppCuteBrowserHeaderSourceExtractionBytes(extraction) {
  requireCppCuteBrowserHeaderSourceExtractionAuthority(extraction);
  return canonicalJsonBytes(extraction);
}

export function cppCuteBrowserHeaderSourceExtractionRoots(extraction) {
  const stored = SOURCE_EXTRACTIONS.get(extraction);
  if (stored === undefined) invalid("$.extraction", "expected extractor-issued header-source authority");
  const roots = [];
  for (const source of extraction.archives) {
    const normalization = stored.normalizations.get(source.sourceId);
    if (normalization === undefined) invalid("$.extraction", "header-source normalization authority is incomplete");
    const bySelection = new Map(cppCuteBrowserArchiveNormalizationRoots(normalization)
      .map((root) => [root.selectionId, root]));
    for (const selection of source.selections) {
      const root = bySelection.get(selection.includeRootId);
      if (root === undefined || root.sourceTreeId !== selection.sourceTreeId) {
        invalid("$.extraction", "header-source storage root differs from extraction evidence");
      }
      roots.push(Object.freeze({
        sourceId: source.sourceId,
        includeRootId: selection.includeRootId,
        storageRoot: root.storageRoot,
        sourceTreeId: root.sourceTreeId,
      }));
    }
  }
  return Object.freeze(roots);
}

export async function copyCppCuteBrowserExtractedHeaderSourceFile(
  extraction,
  sourceId,
  includeRootId,
  relativePath,
) {
  const stored = SOURCE_EXTRACTIONS.get(extraction);
  if (stored === undefined) invalid("$.extraction", "expected extractor-issued header-source authority");
  const normalization = stored.normalizations.get(sourceId);
  if (normalization === undefined) invalid("$.sourceId", "source is absent from the extraction");
  try {
    return await copyCppCuteBrowserArchiveNormalizationFile(
      normalization,
      includeRootId,
      relativePath,
    );
  } catch (cause) {
    invalid("$.relativePath", "failed to copy exact extracted header source", { cause });
  }
}

export function cppCuteBrowserExtractedHeaderSourceFiles(
  extraction,
  sourceId,
  includeRootId,
) {
  const stored = SOURCE_EXTRACTIONS.get(extraction);
  if (stored === undefined) invalid("$.extraction", "expected extractor-issued header-source authority");
  const normalization = stored.normalizations.get(sourceId);
  if (normalization === undefined) invalid("$.sourceId", "source is absent from the extraction");
  const selection = normalization.selections.find(({ selectionId }) => selectionId === includeRootId);
  if (selection === undefined) invalid("$.includeRootId", "include root is absent from the source");
  return selection.files;
}

export function parseCppCuteBrowserHeaderSourceExtractionArguments(argv) {
  if (!Array.isArray(argv)) invalid("$arguments", "expected one argument array");
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length !== 9) {
    invalid("$arguments", "expected seven source archives, --bsdtar, and --output-root");
  }
  const operational = new Map();
  const archives = [];
  for (const [index, argument] of arguments_.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--(bsdtar|output-root)=(.+)$/u.exec(argument);
    if (match === null) {
      archives.push(argument);
      continue;
    }
    const [, name, value] = match;
    if (operational.has(name)) invalid(`$arguments[${index}]`, `duplicate --${name}`);
    operational.set(name, absolutePath(value, `$arguments.${name}`));
  }
  if (!operational.has("bsdtar") || !operational.has("output-root")) {
    invalid("$arguments", "both --bsdtar and --output-root are required");
  }
  const archiveInput = parseCppCuteBrowserHeaderSourceArchiveArguments(archives);
  return Object.freeze({
    archives: archiveInput.archives,
    bsdtarPath: operational.get("bsdtar"),
    outputRoot: operational.get("output-root"),
  });
}

function bindAdmissionToPlan(admission, plan) {
  if (admission.headerSourcePlanId !== plan.planId ||
      admission.buildInputLockId !== plan.body.buildInputLockId ||
      admission.archives.length !== plan.body.archives.length) {
    invalid("$.input.archiveAdmission", "archive admission differs from the current source plan");
  }
  for (const [index, source] of plan.body.archives.entries()) {
    const admitted = admission.archives[index];
    if (admitted === undefined || admitted.sourceId !== source.sourceId ||
        admitted.archiveSha256 !== source.archiveSha256 ||
        admitted.archiveByteLength !== source.archiveByteLength ||
        admitted.archiveFormat !== source.archiveFormat) {
      invalid("$.input.archiveAdmission", "archive admission source set differs from the current plan");
    }
  }
}

function normalizedArchiveSubtree(archiveFormat, archiveSubtree) {
  if (archiveFormat !== "deb-data-tar-zstd") return archiveSubtree;
  const match = /^data\.tar\.zst:\.\/(.+)$/u.exec(archiveSubtree);
  if (match === null) invalid("$.plan", "Debian selection does not name data.tar.zst");
  return match[1];
}

async function admitPrivateCanonicalDirectory(path, diagnosticPath) {
  const canonical = await realpath(path).catch(() => undefined);
  if (canonical !== path) invalid(diagnosticPath, "expected one canonical private directory");
  const stat = await lstat(path, { bigint: true });
  requirePrivateDirectory(stat, diagnosticPath);
}

function requirePrivateDirectory(stat, diagnosticPath) {
  if (!stat.isDirectory() || stat.isSymbolicLink() ||
      typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid()) ||
      (stat.mode & 0o077n) !== 0n) {
    invalid(diagnosticPath, "expected one current-user-owned private directory");
  }
}

async function unlinkOwnedFile(path, identity) {
  try {
    const current = await lstat(path, { bigint: true });
    if (current.dev !== identity.dev || current.ino !== identity.ino) return false;
    await unlink(path);
    return !await pathExists(path);
  } catch (cause) {
    return isNodeError(cause, "ENOENT");
  }
}

async function removeOwnedRoot(path, identity) {
  try {
    const current = await lstat(path, { bigint: true });
    if (current.dev !== identity.dev || current.ino !== identity.ino) return false;
    await rm(path, { recursive: true });
    return !await pathExists(path);
  } catch (cause) {
    return isNodeError(cause, "ENOENT");
  }
}

async function pathExists(path) {
  try {
    await lstat(path);
    return true;
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) return false;
    throw cause;
  }
}

function isNodeError(value, code) {
  return value instanceof Error && "code" in value && value.code === code;
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
  throw new CppCuteBrowserHeaderSourceExtractionError(path, message, options);
}

async function main() {
  try {
    const options = parseCppCuteBrowserHeaderSourceExtractionArguments(process.argv.slice(2));
    const archiveAdmission = await admitCppCuteBrowserHeaderSourcePlanArchives({
      archives: options.archives,
    });
    const bsdtarTool = await admitCppCuteBrowserBsdtarTool({
      executablePath: options.bsdtarPath,
    });
    const extraction = await extractCppCuteBrowserHeaderSourcePlan({
      archiveAdmission,
      bsdtarTool,
      outputRoot: options.outputRoot,
    });
    process.stdout.write(`${JSON.stringify({ outputRoot: options.outputRoot, ...extraction })}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-source extraction failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
