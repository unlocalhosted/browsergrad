import { createHash } from "node:crypto";
import { lstat, mkdir, realpath, rm, unlink } from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path/posix";
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
  "browsergrad.compiler.cpp-cute.browser-header-source-extraction.v3";
const EVIDENCE_ID = /^[a-z][a-z0-9-]*$/u;
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
 * normalizes their eight selected header subtrees, exact supplemental
 * configured-header inputs, and upstream license/copyright review files, then
 * removes all archive copies. The output is a collision-free
 * virtual-path-backed source store, not a host-filesystem-shaped include tree
 * or a license-review conclusion.
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
          selections: [
            ...source.selections.map((selection) => Object.freeze({
              selectionId: selection.includeRootId,
              selectionKind: "subtree",
              archiveSubtree: normalizedArchiveSubtree(source.archiveFormat, selection.archiveSubtree),
              outputSubdirectory: selection.includeRootId,
            })),
            ...source.licenseEvidence.map((evidence) => Object.freeze({
              selectionId: licenseEvidenceSelectionId(evidence.evidenceId),
              selectionKind: "file",
              archiveSubtree: normalizedArchiveSubtree(source.archiveFormat, evidence.archivePath),
              outputSubdirectory: licenseEvidenceSelectionId(evidence.evidenceId),
            })),
            ...source.supplementalFiles.map((file) => Object.freeze({
              selectionId: supplementalFileSelectionId(file.supplementalFileId),
              selectionKind: "file",
              archiveSubtree: normalizedArchiveSubtree(source.archiveFormat, file.archivePath),
              outputSubdirectory: supplementalFileSelectionId(file.supplementalFileId),
            })),
          ],
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
          if (observed.selectionKind !== "subtree") {
            invalid(`$.archives.${source.sourceId}`, "header selection was not normalized as a subtree");
          }
          verifyConfiguredResourceOutput(
            selection.configuredResourceOutput,
            observed,
            `$.archives.${source.sourceId}.${selection.includeRootId}`,
          );
          return Object.freeze({
            includeRootId: selection.includeRootId,
            archiveSubtree: selection.archiveSubtree,
            virtualPrefix: selection.virtualPrefix,
            intendedAsset: selection.intendedAsset,
            licenseComponentIds: selection.licenseComponentIds,
            contribution: selection.contribution,
            ...(selection.configuredResourceOutput === undefined
              ? {}
              : { configuredResourceOutput: selection.configuredResourceOutput }),
            sourceTreeId: observed.sourceTreeId,
            fileCount: observed.fileCount,
            fileContentByteLength: observed.fileContentByteLength,
          });
        });
        const licenseEvidence = source.licenseEvidence.map((evidence) => {
          const selectionId = licenseEvidenceSelectionId(evidence.evidenceId);
          const observed = normalization.selections.find(
            (selection) => selection.selectionId === selectionId,
          );
          const expectedRelativePath = basename(normalizedArchiveSubtree(
            source.archiveFormat,
            evidence.archivePath,
          ));
          if (observed === undefined || observed.selectionKind !== "file" ||
              observed.fileCount !== 1 || observed.files.length !== 1 ||
              observed.files[0]?.relativePath !== expectedRelativePath ||
              observed.files[0]?.contentSha256 !== evidence.sha256 ||
              observed.files[0]?.byteLength !== evidence.byteLength ||
              observed.fileContentByteLength !== evidence.byteLength) {
            invalid(
              `$.archives.${source.sourceId}.licenseEvidence.${evidence.evidenceId}`,
              "exact upstream license evidence differs from the source plan",
            );
          }
          return Object.freeze({
            ...evidence,
            sourceTreeId: observed.sourceTreeId,
          });
        });
        const supplementalFiles = source.supplementalFiles.map((file) => {
          const selectionId = supplementalFileSelectionId(file.supplementalFileId);
          const observed = normalization.selections.find(
            (selection) => selection.selectionId === selectionId,
          );
          const expectedRelativePath = basename(normalizedArchiveSubtree(
            source.archiveFormat,
            file.archivePath,
          ));
          if (observed === undefined || observed.selectionKind !== "file" ||
              observed.fileCount !== 1 || observed.files.length !== 1 ||
              observed.files[0]?.relativePath !== expectedRelativePath ||
              observed.files[0]?.contentSha256 !== file.sha256 ||
              observed.files[0]?.byteLength !== file.byteLength ||
              observed.fileContentByteLength !== file.byteLength) {
            invalid(
              `$.archives.${source.sourceId}.supplementalFiles.${file.supplementalFileId}`,
              "exact supplemental configured-header input differs from the source plan",
            );
          }
          return Object.freeze({
            ...file,
            sourceTreeId: observed.sourceTreeId,
          });
        });
        fileCount += selections.reduce((total, selection) => total + selection.fileCount, 0) +
          supplementalFiles.length;
        fileContentBytes += selections.reduce(
          (total, selection) => total + BigInt(selection.fileContentByteLength),
          0n,
        ) + supplementalFiles.reduce(
          (total, file) => total + BigInt(file.byteLength),
          0n,
        );
        archives.push(Object.freeze({
          sourceId: source.sourceId,
          archiveFormat: source.archiveFormat,
          archiveSha256: source.archiveSha256,
          archiveByteLength: source.archiveByteLength,
          licenseComponentId: source.licenseComponentId,
          licensePolicy: source.licensePolicy,
          normalizationId: normalization.normalizationId,
          selections: Object.freeze(selections),
          supplementalFiles: Object.freeze(supplementalFiles),
          licenseEvidence: Object.freeze(licenseEvidence),
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
    const unresolvedBlockers = Object.freeze(plan.body.unresolvedBlockers.filter(
      ({ blockerId }) => blockerId !== "clang-resource-generated-headers",
    ));
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
      unresolvedBlockers,
    }));
    const extraction = Object.freeze({
      schema: CPP_CUTE_BROWSER_HEADER_SOURCE_EXTRACTION_SCHEMA,
      version: 3,
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
        packageToolIdentityPinned: object.bsdtarTool.claims.packageToolIdentityPinned,
        ...(object.bsdtarTool.nodeZstdRuntime === undefined
          ? {}
          : { nodeZstdRuntime: object.bsdtarTool.nodeZstdRuntime }),
      }),
      archives: Object.freeze(archives),
      totals: Object.freeze({
        archiveCount: archives.length,
        selectedSubtreeCount: archives.reduce((total, source) => total + source.selections.length, 0),
        supplementalFileCount: archives.reduce(
          (total, source) => total + source.supplementalFiles.length,
          0,
        ),
        supplementalFileByteLength: archives.reduce(
          (total, source) => total + source.supplementalFiles.reduce(
            (sourceTotal, file) => sourceTotal + BigInt(file.byteLength),
            0n,
          ),
          0n,
        ).toString(),
        fileCount,
        fileContentByteLength: String(fileContentBytes),
        licenseEvidenceFileCount: archives.reduce(
          (total, source) => total + source.licenseEvidence.length,
          0,
        ),
        licenseEvidenceByteLength: archives.reduce(
          (total, source) => total + source.licenseEvidence.reduce(
            (sourceTotal, evidence) => sourceTotal + BigInt(evidence.byteLength),
            0n,
          ),
          0n,
        ).toString(),
      }),
      unresolvedBlockers,
      claims: Object.freeze({
        exactCurrentHeaderSourcePlanArchiveBytesVerified: true,
        exactBuildInputLockBound: true,
        exactHeaderSourcePlanBound: true,
        sourceSubtreeMaterializationsObserved: true,
        exactSelectedSourceSubtreesComplete:
          object.bsdtarTool.claims.packageToolIdentityPinned &&
          object.bsdtarTool.nodeZstdRuntime !== undefined,
        collisionFreePortableStorageMaterialized: true,
        allFiveIncludeRootsRepresented: true,
        copiedSourceArchivesRemoved: true,
        hostToolImplementationAttested: false,
        hostToolPackageIdentityPinned: object.bsdtarTool.claims.packageToolIdentityPinned,
        nodeZstdDecompressorPackageIdentityPinned:
          object.bsdtarTool.nodeZstdRuntime !== undefined,
        generatedClangResourceHeadersComplete: true,
        exactUpstreamLicenseEvidenceExtracted: true,
        externalDistributedFileLicenseMapReviewed: false,
        licenseReviewComplete: false,
        headerUniverseComplete:
          object.bsdtarTool.claims.packageToolIdentityPinned &&
          object.bsdtarTool.nodeZstdRuntime !== undefined,
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

function verifyConfiguredResourceOutput(policy, observed, diagnosticPath) {
  if (policy === undefined) return;
  if (policy.buildStageId !== "clang-extractor-wasm" ||
      policy.llvmTargetsToBuild !== "WebAssembly" || policy.clangEnableHlsl !== "OFF" ||
      !Array.isArray(policy.generatedVirtualPaths) || policy.generatedVirtualPaths.length !== 0 ||
      !Array.isArray(policy.omittedSourceVirtualPaths) ||
      policy.omittedSourceVirtualPaths.length !== 1) {
    invalid(diagnosticPath, "configured Clang resource-output policy is malformed");
  }
  const manifest = observed.files.find(
    ({ relativePath }) => relativePath === policy.upstreamBuildManifest.virtualPath,
  );
  if (manifest === undefined ||
      manifest.contentSha256 !== policy.upstreamBuildManifest.sha256 ||
      manifest.byteLength !== policy.upstreamBuildManifest.byteLength ||
      policy.omittedSourceVirtualPaths[0] !== manifest.relativePath) {
    invalid(diagnosticPath, "configured Clang resource build manifest differs from the exact plan");
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

export async function copyCppCuteBrowserExtractedHeaderLicenseEvidence(
  extraction,
  sourceId,
  evidenceId,
) {
  const stored = SOURCE_EXTRACTIONS.get(extraction);
  if (stored === undefined) invalid("$.extraction", "expected extractor-issued header-source authority");
  if (typeof evidenceId !== "string" || !EVIDENCE_ID.test(evidenceId)) {
    invalid("$.evidenceId", "expected one portable evidence ID");
  }
  const source = extraction.archives.find((archive) => archive.sourceId === sourceId);
  const evidence = source?.licenseEvidence.find((item) => item.evidenceId === evidenceId);
  const normalization = stored.normalizations.get(sourceId);
  if (source === undefined || evidence === undefined || normalization === undefined) {
    invalid("$.evidenceId", "license evidence is absent from the extraction");
  }
  const relativePath = basename(normalizedArchiveSubtree(source.archiveFormat, evidence.archivePath));
  let bytes;
  try {
    bytes = await copyCppCuteBrowserArchiveNormalizationFile(
      normalization,
      licenseEvidenceSelectionId(evidenceId),
      relativePath,
    );
  } catch (cause) {
    invalid("$.evidenceId", "failed to copy exact extracted license evidence", { cause });
  }
  if (String(bytes.byteLength) !== evidence.byteLength || sha256(bytes) !== evidence.sha256) {
    invalid("$.evidenceId", "copied license evidence differs from extraction evidence");
  }
  return bytes;
}

export async function copyCppCuteBrowserExtractedHeaderSupplementalFile(
  extraction,
  sourceId,
  supplementalFileId,
) {
  const stored = SOURCE_EXTRACTIONS.get(extraction);
  if (stored === undefined) invalid("$.extraction", "expected extractor-issued header-source authority");
  if (typeof supplementalFileId !== "string" || !EVIDENCE_ID.test(supplementalFileId)) {
    invalid("$.supplementalFileId", "expected one portable supplemental file ID");
  }
  const source = extraction.archives.find((archive) => archive.sourceId === sourceId);
  const file = source?.supplementalFiles.find(
    (item) => item.supplementalFileId === supplementalFileId,
  );
  const normalization = stored.normalizations.get(sourceId);
  if (source === undefined || file === undefined || normalization === undefined) {
    invalid("$.supplementalFileId", "supplemental file is absent from the extraction");
  }
  const relativePath = basename(normalizedArchiveSubtree(source.archiveFormat, file.archivePath));
  let bytes;
  try {
    bytes = await copyCppCuteBrowserArchiveNormalizationFile(
      normalization,
      supplementalFileSelectionId(supplementalFileId),
      relativePath,
    );
  } catch (cause) {
    invalid("$.supplementalFileId", "failed to copy exact supplemental file", { cause });
  }
  if (String(bytes.byteLength) !== file.byteLength || sha256(bytes) !== file.sha256) {
    invalid("$.supplementalFileId", "copied supplemental file differs from extraction evidence");
  }
  return bytes;
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

function licenseEvidenceSelectionId(evidenceId) {
  if (typeof evidenceId !== "string" || !EVIDENCE_ID.test(evidenceId)) {
    invalid("$.plan.licenseEvidence.evidenceId", "expected one portable evidence ID");
  }
  return `license-${evidenceId}`;
}

function supplementalFileSelectionId(supplementalFileId) {
  if (typeof supplementalFileId !== "string" || !EVIDENCE_ID.test(supplementalFileId)) {
    invalid("$.plan.supplementalFiles.supplementalFileId", "expected one portable file ID");
  }
  return `supplemental-${supplementalFileId}`;
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
