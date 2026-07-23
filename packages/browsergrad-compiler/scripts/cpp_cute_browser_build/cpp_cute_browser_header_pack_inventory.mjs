import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, opendir, realpath, unlink } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  deriveCppCuteBrowserVfsContentSetSha256,
} from "../../dist/cpp_cute_browser_vfs_pack.js";
import {
  cppCuteBrowserHeaderInputProjectionId,
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  copyCppCuteBrowserExtractedHeaderSourceFile,
  copyCppCuteBrowserExtractedHeaderSupplementalFile,
  cppCuteBrowserExtractedHeaderSourceFiles,
  requireCppCuteBrowserHeaderSourceExtractionAuthority,
} from "./cpp_cute_browser_header_source_extraction.mjs";
import {
  CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE,
  CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_VIRTUAL_PATH,
  materializeCppCuteBrowserClangCudaRuntimeWrapper,
} from "./cpp_cute_browser_clang_cuda_runtime_wrapper.mjs";
import {
  CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE,
  CPP_CUTE_BROWSER_LIBCXX_MODULE_MAP_PROFILE,
  materializeCppCuteBrowserLibcxxConfigSite,
  materializeCppCuteBrowserLibcxxModuleMap,
} from "./cpp_cute_browser_libcxx_config_site.mjs";

export const CPP_CUTE_BROWSER_HEADER_PACK_INVENTORY_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-pack-source-inventory";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVENTORY";
const INVENTORY_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-pack-source-inventory.v2";
const PORTABLE_SEGMENT = /^[A-Za-z0-9._+@=-]+$/u;
const IDENTIFIER = /^[a-z][a-z0-9._-]*$/u;
const SHA256 = /^[0-9a-f]{64}$/u;
const MAX_INPUT_BYTES = 256 * 1024;
const MAX_OUTPUT_BYTES = 32 * 1024 * 1024;
const MAX_SOURCES = 256;
const MAX_FILES = 100_000;
const MAX_PATH_BYTES = 4_096;
const MAX_FILE_BYTES = 64 * 1024 * 1024;
const MAX_TOTAL_FILE_BYTES = 512 * 1024 * 1024;
const READ_BUFFER_BYTES = 256 * 1024;
const TEXT_ENCODER = new TextEncoder();
const INVENTORY_SOURCE_FILES = new WeakMap();
const PACK_POLICIES = Object.freeze([
  Object.freeze({
    includeRootId: "clang-resource",
    intendedAsset: "compiler-resource-pack",
    outputRole: "clang-resource-header-vfs",
    outputPath: "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cxx-stdlib",
    intendedAsset: "dependency-header-pack:cxx-stdlib",
    outputRole: "libcxx-header-vfs",
    outputPath: "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cuda",
    intendedAsset: "dependency-header-pack:cuda",
    outputRole: "cuda-header-vfs",
    outputPath: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cutlass",
    intendedAsset: "dependency-header-pack:cutlass",
    outputRole: "cutlass-header-vfs",
    outputPath: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "linux-sysroot",
    intendedAsset: "dependency-header-pack:linux-sysroot",
    outputRole: "linux-sysroot-header-vfs",
    outputPath: "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
  }),
]);

export class CppCuteBrowserHeaderPackInventoryError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderPackInventoryError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Inventories caller-supplied, already-extracted regular-file trees. This
 * operation never downloads or extracts an archive and deliberately grants no
 * archive-provenance, license-review, pack-output, build, or release authority.
 */
export async function inventoryCppCuteBrowserHeaderPackSources(input) {
  const root = exactObject(input, ["packs"], "$.input");
  const rawPacks = denseArray(
    root.packs,
    "$.input.packs",
    PACK_POLICIES.length,
    PACK_POLICIES.length,
  );
  const parsedPacks = rawPacks.map((value, index) => parsePack(value, index));
  parsedPacks.sort((left, right) => compareUtf8(left.includeRootId, right.includeRootId));
  for (let index = 1; index < parsedPacks.length; index += 1) {
    if (parsedPacks[index - 1].includeRootId === parsedPacks[index].includeRootId) {
      invalid(`$.input.packs[${parsedPacks[index].inputIndex}].includeRootId`, "duplicate include-root inventory");
    }
  }
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const lockBody = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body;
  const headerInputProjectionId =
    await cppCuteBrowserHeaderInputProjectionId(buildInputLock);
  const packPolicies = bindPackPoliciesToBuildLock(lockBody);
  for (const [index, pack] of parsedPacks.entries()) {
    const policy = packPolicies[index];
    if (policy === undefined || pack.includeRootId !== policy.includeRootId) {
      invalid("$.input.packs", "inventory must cover the exact current build-lock pack set");
    }
    for (const source of pack.sources) {
      if (!sameStrings(source.licenseComponentIds, policy.licenseComponentIds)) {
        invalid(
          `$.input.packs[${pack.inputIndex}].sources[${source.inputIndex}].licenseComponentIds`,
          "source-tree license components differ from the current build-lock notice policy",
        );
      }
    }
  }
  const sourceCount = parsedPacks.reduce((count, pack) => count + pack.sources.length, 0);
  if (sourceCount > MAX_SOURCES) resource("$.input.packs", `inventory exceeds ${MAX_SOURCES} source trees`);

  const budget = { files: 0, bytes: 0 };
  const packs = [];
  const sourceFilesByIncludeRootId = new Map();
  for (const [packIndex, pack] of parsedPacks.entries()) {
    const policy = packPolicies[packIndex];
    if (policy === undefined) invalid("$.input.packs", "current build-lock pack policy is incomplete");
    const filesByPath = new Map();
    const sourceFilesByPath = new Map();
    for (const source of pack.sources) {
      const sourceRoot = await admitCanonicalDirectory(
        source.sourceRoot,
        `$.input.packs[${pack.inputIndex}].sources[${source.inputIndex}].sourceRoot`,
      );
      await inventoryDirectory({
        sourceRoot,
        directoryPath: sourceRoot,
        relativeSegments: [],
        virtualPrefix: source.virtualPrefix,
        licenseComponentIds: source.licenseComponentIds,
        filesByPath,
        sourceFilesByPath,
        budget,
        diagnosticPath: `$.input.packs[${pack.inputIndex}].sources[${source.inputIndex}]`,
      });
    }
    const files = [...filesByPath.values()].sort((left, right) =>
      compareUtf8(left.virtualPath, right.virtualPath));
    if (files.length === 0) {
      invalid(`$.input.packs[${pack.inputIndex}].sources`, "header-pack inventory must contain at least one file");
    }
    rejectFileDirectoryCollisions(files, `$.input.packs[${pack.inputIndex}]`);
    let contentSetSha256;
    try {
      contentSetSha256 = await deriveCppCuteBrowserVfsContentSetSha256(files);
    } catch (cause) {
      invalid(
        `$.input.packs[${pack.inputIndex}]`,
        "inventory is outside the closed VFS content-set contract",
        { cause },
      );
    }
    const fileContentByteLength = files.reduce(
      (total, file) => total + BigInt(file.byteLength),
      0n,
    ).toString();
    packs.push(Object.freeze({
      includeRootId: pack.includeRootId,
      intendedAsset: policy.intendedAsset,
      outputRole: policy.outputRole,
      outputPath: policy.outputPath,
      contentSetSha256,
      fileCount: files.length,
      fileContentByteLength,
      files: Object.freeze(files),
    }));
    sourceFilesByIncludeRootId.set(pack.includeRootId, sourceFilesByPath);
  }

  return finalizeInventory({
    authority: "local-source-tree-inventory-only",
    buildInputLock,
    headerInputProjectionId,
    packs,
    sourceCount,
    fileCount: budget.files,
    fileContentByteLength: String(budget.bytes),
    sourceFilesByIncludeRootId,
  });
}

/**
 * Inventories the collision-free virtual source stores produced from the exact
 * eight-archive header plan. Every file is reread through the extraction's
 * opaque authority, so no host path or case-folding behavior enters VFS paths.
 */
export async function inventoryCppCuteBrowserExtractedHeaderSources(extraction) {
  try {
    requireCppCuteBrowserHeaderSourceExtractionAuthority(extraction);
  } catch (cause) {
    invalid("$.extraction", "expected one live exact header-source extraction", { cause });
  }
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const headerInputProjectionId =
    await cppCuteBrowserHeaderInputProjectionId(buildInputLock);
  if (extraction.buildInputLockId !== buildInputLock.lockId ||
      extraction.buildInputLockResourceSha256 !== buildInputLock.resourceSha256 ||
      extraction.headerInputProjectionId !== headerInputProjectionId) {
    invalid("$.extraction", "header-source extraction differs from the current build lock");
  }
  const lockBody = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body;
  const packPolicies = bindPackPoliciesToBuildLock(lockBody);
  const groups = new Map(packPolicies.map((policy) => [policy.includeRootId, {
    policy,
    filesByPath: new Map(),
    sourceFilesByPath: new Map(),
    sourceCount: 0,
  }]));
  const observed = { files: 0, bytes: 0 };
  for (const source of extraction.archives) {
    identifier(
      source.licenseComponentId,
      `$.extraction.archives.${source.sourceId}.licenseComponentId`,
    );
    for (const selection of source.selections) {
      const group = groups.get(selection.includeRootId);
      if (group === undefined) {
        invalid("$.extraction.archives", "extraction contains an unknown include root");
      }
      if (!sameStrings(selection.licenseComponentIds, group.policy.licenseComponentIds) ||
          selection.intendedAsset !== group.policy.intendedAsset) {
        invalid(
          `$.extraction.archives.${source.sourceId}.selections`,
          "extracted selection license policy differs from build-lock pack policy",
        );
      }
      group.sourceCount += 1;
      const files = cppCuteBrowserExtractedHeaderSourceFiles(
        extraction,
        source.sourceId,
        selection.includeRootId,
      );
      for (const file of files) {
        const virtualPath = selection.virtualPrefix === ""
          ? file.relativePath
          : `${selection.virtualPrefix}/${file.relativePath}`;
        portableRelativePath(virtualPath, "$.extraction.files.virtualPath", false);
        if (!SHA256.test(file.contentSha256) || !/^(0|[1-9][0-9]*)$/u.test(file.byteLength)) {
          invalid("$.extraction.files", "extracted file evidence is malformed");
        }
        const byteLength = Number(file.byteLength);
        if (!Number.isSafeInteger(byteLength) || byteLength > MAX_FILE_BYTES) {
          resource("$.extraction.files", `source file exceeds ${MAX_FILE_BYTES} bytes`);
        }
        const bytes = await copyCppCuteBrowserExtractedHeaderSourceFile(
          extraction,
          source.sourceId,
          selection.includeRootId,
          file.relativePath,
        );
        if (bytes.byteLength !== byteLength || sha256(bytes) !== file.contentSha256) {
          invalid("$.extraction.files", "extracted file bytes differ from source-tree evidence");
        }
        observed.files += 1;
        observed.bytes += byteLength;
        if (observed.files > MAX_FILES || observed.bytes > MAX_TOTAL_FILE_BYTES) {
          resource("$.extraction", "extracted source files exceed inventory ceilings");
        }
        if (isConfiguredResourceOutputOmission(
          selection.configuredResourceOutput,
          file,
          `$.extraction.archives.${source.sourceId}.${selection.includeRootId}`,
        )) {
          continue;
        }
        const configuredClangHeader = materializeConfiguredClangResourceHeader(
          selection.configuredResourceOutput,
          file,
          virtualPath,
          bytes,
          `$.extraction.archives.${source.sourceId}.${selection.includeRootId}`,
        );
        const expected = Object.freeze({
          virtualPath,
          contentSha256:
            configuredClangHeader?.configuredSha256 ?? file.contentSha256,
          byteLength: configuredClangHeader === undefined
            ? file.byteLength
            : String(configuredClangHeader.bytes.byteLength),
          licenseComponentIds: Object.freeze([...selection.licenseComponentIds]),
        });
        const prior = group.filesByPath.get(virtualPath);
        if (prior !== undefined) {
          if (prior.contentSha256 !== expected.contentSha256 ||
              prior.byteLength !== expected.byteLength) {
            invalid(
              "$.extraction.files",
              `conflicting extracted overlay for ${JSON.stringify(virtualPath)}`,
            );
          }
          const mergedLicenses = [...new Set([
            ...prior.licenseComponentIds,
            ...expected.licenseComponentIds,
          ])].sort(compareUtf8);
          group.filesByPath.set(virtualPath, Object.freeze({
            ...prior,
            licenseComponentIds: Object.freeze(mergedLicenses),
          }));
          continue;
        }
        group.filesByPath.set(virtualPath, expected);
        group.sourceFilesByPath.set(virtualPath, Object.freeze({
          extraction,
          sourceId: source.sourceId,
          includeRootId: selection.includeRootId,
          relativePath: file.relativePath,
          ...(configuredClangHeader === undefined
            ? {}
            : { derivedProfile: configuredClangHeader.profile }),
          expected,
        }));
        if (source.sourceId === "llvm-project" &&
            selection.includeRootId === "cxx-stdlib" &&
            virtualPath === "__config_site.in") {
          const configured = materializeCppCuteBrowserLibcxxConfigSite(bytes);
          const configuredVirtualPath = "__config_site";
          if (group.filesByPath.has(configuredVirtualPath) ||
              group.sourceFilesByPath.has(configuredVirtualPath)) {
            invalid(
              "$.extraction.files",
              "configured libc++ header collides with an extracted source path",
            );
          }
          const configuredExpected = Object.freeze({
            virtualPath: configuredVirtualPath,
            contentSha256: configured.configuredSha256,
            byteLength: String(configured.bytes.byteLength),
            licenseComponentIds: Object.freeze([...selection.licenseComponentIds]),
          });
          group.filesByPath.set(configuredVirtualPath, configuredExpected);
          group.sourceFilesByPath.set(configuredVirtualPath, Object.freeze({
            extraction,
            sourceId: source.sourceId,
            includeRootId: selection.includeRootId,
            relativePath: file.relativePath,
            derivedProfile: configured.profile,
            expected: configuredExpected,
          }));
        }
        if (source.sourceId === "llvm-project" &&
            selection.includeRootId === "cxx-stdlib" &&
            virtualPath === "module.modulemap.in") {
          const configured = materializeCppCuteBrowserLibcxxModuleMap(bytes);
          const configuredVirtualPath = "module.modulemap";
          if (group.filesByPath.has(configuredVirtualPath) ||
              group.sourceFilesByPath.has(configuredVirtualPath)) {
            invalid(
              "$.extraction.files",
              "configured libc++ module map collides with an extracted source path",
            );
          }
          const configuredExpected = Object.freeze({
            virtualPath: configuredVirtualPath,
            contentSha256: configured.configuredSha256,
            byteLength: String(configured.bytes.byteLength),
            licenseComponentIds: Object.freeze([...selection.licenseComponentIds]),
          });
          group.filesByPath.set(configuredVirtualPath, configuredExpected);
          group.sourceFilesByPath.set(configuredVirtualPath, Object.freeze({
            extraction,
            sourceId: source.sourceId,
            includeRootId: selection.includeRootId,
            relativePath: file.relativePath,
            derivedProfile: configured.profile,
            expected: configuredExpected,
          }));
        }
      }
    }
    for (const file of source.supplementalFiles) {
      const group = groups.get(file.includeRootId);
      if (group === undefined ||
          !sameStrings(file.licenseComponentIds, group.policy.licenseComponentIds) ||
          file.intendedAsset !== group.policy.intendedAsset) {
        invalid(
          `$.extraction.archives.${source.sourceId}.supplementalFiles`,
          "supplemental file license policy differs from build-lock pack policy",
        );
      }
      portableRelativePath(file.virtualPath, "$.extraction.supplementalFiles.virtualPath", false);
      if (!SHA256.test(file.sha256) || !/^[1-9][0-9]*$/u.test(file.byteLength)) {
        invalid("$.extraction.supplementalFiles", "supplemental file evidence is malformed");
      }
      const byteLength = Number(file.byteLength);
      if (!Number.isSafeInteger(byteLength) || byteLength > MAX_FILE_BYTES) {
        resource("$.extraction.supplementalFiles", `source file exceeds ${MAX_FILE_BYTES} bytes`);
      }
      const bytes = await copyCppCuteBrowserExtractedHeaderSupplementalFile(
        extraction,
        source.sourceId,
        file.supplementalFileId,
      );
      if (bytes.byteLength !== byteLength || sha256(bytes) !== file.sha256) {
        invalid(
          "$.extraction.supplementalFiles",
          "supplemental file bytes differ from source-tree evidence",
        );
      }
      observed.files += 1;
      observed.bytes += byteLength;
      if (observed.files > MAX_FILES || observed.bytes > MAX_TOTAL_FILE_BYTES) {
        resource("$.extraction", "extracted source files exceed inventory ceilings");
      }
      if (group.filesByPath.has(file.virtualPath) ||
          group.sourceFilesByPath.has(file.virtualPath)) {
        invalid(
          "$.extraction.supplementalFiles",
          `supplemental file collides at ${JSON.stringify(file.virtualPath)}`,
        );
      }
      const expected = Object.freeze({
        virtualPath: file.virtualPath,
        contentSha256: file.sha256,
        byteLength: file.byteLength,
        licenseComponentIds: Object.freeze([...file.licenseComponentIds]),
      });
      group.filesByPath.set(file.virtualPath, expected);
      group.sourceFilesByPath.set(file.virtualPath, Object.freeze({
        extraction,
        sourceId: source.sourceId,
        supplementalFileId: file.supplementalFileId,
        expected,
      }));
    }
  }

  const libcxx = groups.get("cxx-stdlib");
  if (libcxx === undefined ||
      !libcxx.filesByPath.has("__assertion_handler") ||
      !libcxx.filesByPath.has("__config_site") ||
      !libcxx.filesByPath.has("module.modulemap")) {
    invalid(
      "$.extraction.cxx-stdlib",
      "exact libc++ source did not produce its complete configured header outputs",
    );
  }

  const packs = [];
  const sourceFilesByIncludeRootId = new Map();
  let uniqueFileCount = 0;
  let uniqueFileContentBytes = 0n;
  let sourceCount = 0;
  for (const policy of packPolicies) {
    const group = groups.get(policy.includeRootId);
    if (group === undefined || group.sourceCount === 0) {
      invalid("$.extraction", `extraction omitted ${JSON.stringify(policy.includeRootId)}`);
    }
    const files = [...group.filesByPath.values()].sort((left, right) =>
      compareUtf8(left.virtualPath, right.virtualPath));
    rejectFileDirectoryCollisions(files, `$.extraction.${policy.includeRootId}`);
    const contentSetSha256 = await deriveContentSet(files, `$.extraction.${policy.includeRootId}`);
    const fileContentByteLength = files.reduce(
      (total, file) => total + BigInt(file.byteLength),
      0n,
    ).toString();
    packs.push(Object.freeze({
      includeRootId: policy.includeRootId,
      intendedAsset: policy.intendedAsset,
      outputRole: policy.outputRole,
      outputPath: policy.outputPath,
      contentSetSha256,
      fileCount: files.length,
      fileContentByteLength,
      files: Object.freeze(files),
    }));
    sourceFilesByIncludeRootId.set(policy.includeRootId, group.sourceFilesByPath);
    sourceCount += group.sourceCount;
    uniqueFileCount += files.length;
    uniqueFileContentBytes += BigInt(fileContentByteLength);
  }
  if (sourceCount !== extraction.totals.selectedSubtreeCount) {
    invalid("$.extraction", "not every extracted source subtree entered the pack inventory");
  }
  return finalizeInventory({
    authority: "exact-extraction-source-inventory-only",
    buildInputLock,
    headerInputProjectionId,
    headerSourceExtractionId: extraction.extractionId,
    packs,
    sourceCount,
    fileCount: uniqueFileCount,
    fileContentByteLength: String(uniqueFileContentBytes),
    sourceFilesByIncludeRootId,
  });
}

function finalizeInventory(input) {
  const inventoryHash = sha256(canonicalJsonBytes({
    domain: INVENTORY_HASH_DOMAIN,
    headerInputProjectionId: input.headerInputProjectionId,
    ...(input.headerSourceExtractionId === undefined
      ? {}
      : { headerSourceExtractionId: input.headerSourceExtractionId }),
    packs: input.packs,
  }));
  const manifest = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_PACK_INVENTORY_SCHEMA,
    version: 2,
    inventoryId: `bg.cpp.browser-header-pack-source-inventory.sha256.${inventoryHash}`,
    authority: input.authority,
    buildInputLockId: input.buildInputLock.lockId,
    buildInputLockResourceSha256: input.buildInputLock.resourceSha256,
    headerInputProjectionId: input.headerInputProjectionId,
    ...(input.headerSourceExtractionId === undefined
      ? {}
      : { headerSourceExtractionId: input.headerSourceExtractionId }),
    packs: Object.freeze(input.packs),
    totals: Object.freeze({
      packCount: input.packs.length,
      sourceCount: input.sourceCount,
      fileCount: input.fileCount,
      fileContentByteLength: input.fileContentByteLength,
    }),
    claims: Object.freeze({
      exactReadableSourceTreesVerified: true,
      buildInputLockBound: true,
      headerInputProjectionBound: true,
      networkAccessed: false,
      archiveProvenanceVerified: false,
      generatedClangResourceHeadersComplete: input.headerSourceExtractionId !== undefined,
      configuredLibcxxHeaderComplete: input.headerSourceExtractionId !== undefined,
      licenseReviewComplete: false,
      headerPackSelectionPrepared: false,
      headerPacksAssembled: false,
      buildExecuted: false,
      releaseReady: false,
    }),
  });
  const bytes = canonicalJsonBytes(manifest);
  if (bytes.byteLength > MAX_OUTPUT_BYTES) {
    resource("$.inventory", `canonical inventory exceeds ${MAX_OUTPUT_BYTES} bytes`);
  }
  INVENTORY_SOURCE_FILES.set(manifest, Object.freeze({
    sourceFilesByIncludeRootId: input.sourceFilesByIncludeRootId,
  }));
  return manifest;
}

function materializeConfiguredClangResourceHeader(
  policy,
  file,
  virtualPath,
  bytes,
  diagnosticPath,
) {
  if (policy === undefined ||
      virtualPath !== CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_VIRTUAL_PATH) {
    return undefined;
  }
  const profile = configuredClangResourceHeaderProfile(policy, diagnosticPath);
  if (file.contentSha256 !== profile.upstreamSha256 ||
      file.byteLength !== profile.upstreamByteLength) {
    invalid(
      diagnosticPath,
      "configured Clang CUDA wrapper input differs from its exact profile",
    );
  }
  let configured;
  try {
    configured = materializeCppCuteBrowserClangCudaRuntimeWrapper(bytes);
  } catch (cause) {
    invalid(
      diagnosticPath,
      "failed to configure the exact Clang CUDA runtime wrapper",
      { cause },
    );
  }
  if (configured.profile !== profile.profile ||
      configured.templateSha256 !== profile.upstreamSha256) {
    invalid(
      diagnosticPath,
      "configured Clang CUDA wrapper transform differs from its exact profile",
    );
  }
  return configured;
}

function configuredClangResourceHeaderProfile(policy, diagnosticPath) {
  if (!Array.isArray(policy.generatedVirtualPaths) ||
      policy.generatedVirtualPaths.length !== 1 ||
      policy.generatedVirtualPaths[0] !==
        CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_VIRTUAL_PATH ||
      !Array.isArray(policy.generatedHeaderProfiles) ||
      policy.generatedHeaderProfiles.length !== 1) {
    invalid(diagnosticPath, "configured Clang resource header set is malformed");
  }
  const profile = policy.generatedHeaderProfiles[0];
  if (profile.virtualPath !==
        CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_VIRTUAL_PATH ||
      profile.profile !== CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE ||
      typeof profile.upstreamSha256 !== "string" ||
      !SHA256.test(profile.upstreamSha256) ||
      typeof profile.upstreamByteLength !== "string" ||
      !/^[1-9][0-9]*$/u.test(profile.upstreamByteLength)) {
    invalid(diagnosticPath, "configured Clang resource header profile is malformed");
  }
  return profile;
}

function isConfiguredResourceOutputOmission(policy, file, diagnosticPath) {
  if (policy === undefined || !policy.omittedSourceVirtualPaths.includes(file.relativePath)) {
    return false;
  }
  const manifest = policy.upstreamBuildManifest;
  configuredClangResourceHeaderProfile(policy, diagnosticPath);
  if (file.relativePath !== manifest.virtualPath || file.contentSha256 !== manifest.sha256 ||
      file.byteLength !== manifest.byteLength ||
      policy.llvmTargetsToBuild !== "WebAssembly" || policy.clangEnableHlsl !== "OFF") {
    invalid(diagnosticPath, "configured Clang resource output omission is not exact");
  }
  return true;
}

async function deriveContentSet(files, diagnosticPath) {
  if (files.length === 0) invalid(diagnosticPath, "header-pack inventory must contain at least one file");
  try {
    return await deriveCppCuteBrowserVfsContentSetSha256(files);
  } catch (cause) {
    invalid(diagnosticPath, "inventory is outside the closed VFS content-set contract", { cause });
  }
}

/**
 * Copies one file from the exact live source-tree authority behind an
 * inventory instance. Serialized or forged inventory records have no access
 * to caller paths, and changed source bytes fail before they are returned.
 */
export async function copyCppCuteBrowserHeaderPackInventorySourceFile(
  inventory,
  includeRootId,
  virtualPath,
) {
  const stored = INVENTORY_SOURCE_FILES.get(inventory);
  if (stored === undefined) invalid("$.inventory", "inventory has no live source-tree authority");
  identifier(includeRootId, "$.includeRootId");
  portableRelativePath(virtualPath, "$.virtualPath", false);
  const source = stored.sourceFilesByIncludeRootId.get(includeRootId)?.get(virtualPath);
  if (source === undefined) invalid("$.virtualPath", "file is absent from the exact inventory");
  if (source.extraction !== undefined) {
    if (source.supplementalFileId !== undefined) {
      let supplementalBytes;
      try {
        supplementalBytes = await copyCppCuteBrowserExtractedHeaderSupplementalFile(
          source.extraction,
          source.sourceId,
          source.supplementalFileId,
        );
      } catch (cause) {
        invalid("$.virtualPath", "failed to reread supplemental inventory source", { cause });
      }
      if (supplementalBytes.byteLength !== Number(source.expected.byteLength) ||
          sha256(supplementalBytes) !== source.expected.contentSha256) {
        invalid("$.virtualPath", "supplemental inventory source bytes changed");
      }
      return supplementalBytes;
    }
    let bytes;
    try {
      bytes = await copyCppCuteBrowserExtractedHeaderSourceFile(
        source.extraction,
        source.sourceId,
        source.includeRootId,
        source.relativePath,
      );
    } catch (cause) {
      invalid("$.virtualPath", "failed to reread extracted inventory source", { cause });
    }
    if (source.derivedProfile !== undefined) {
      const configured = source.derivedProfile === CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE
        ? materializeCppCuteBrowserLibcxxConfigSite(bytes)
        : source.derivedProfile === CPP_CUTE_BROWSER_LIBCXX_MODULE_MAP_PROFILE
          ? materializeCppCuteBrowserLibcxxModuleMap(bytes)
          : source.derivedProfile ===
              CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE
            ? materializeCppCuteBrowserClangCudaRuntimeWrapper(bytes)
            : undefined;
      if (configured === undefined) {
        invalid("$.virtualPath", "configured header source profile changed");
      }
      if (configured.profile !== source.derivedProfile ||
          configured.bytes.byteLength !== Number(source.expected.byteLength) ||
          configured.configuredSha256 !== source.expected.contentSha256) {
        invalid("$.virtualPath", "configured inventory source bytes changed");
      }
      return configured.bytes;
    }
    if (bytes.byteLength !== Number(source.expected.byteLength) ||
        sha256(bytes) !== source.expected.contentSha256) {
      invalid("$.virtualPath", "extracted inventory source bytes changed");
    }
    return bytes;
  }
  return readExactInventorySourceFile(source.sourcePath, source.expected, "$.virtualPath");
}

export function canonicalCppCuteBrowserHeaderPackInventoryBytes(inventory) {
  requireCppCuteBrowserHeaderPackInventorySourceAuthority(inventory);
  const bytes = canonicalJsonBytes(inventory);
  if (bytes.byteLength > MAX_OUTPUT_BYTES) {
    resource("$.inventory", `canonical inventory exceeds ${MAX_OUTPUT_BYTES} bytes`);
  }
  return bytes;
}

export function requireCppCuteBrowserHeaderPackInventorySourceAuthority(inventory) {
  if (INVENTORY_SOURCE_FILES.get(inventory) === undefined) {
    invalid("$.inventory", "inventory has no live source-tree authority");
  }
}

export async function readCppCuteBrowserHeaderPackInventorySpecification(inputPath) {
  const path = absolutePath(inputPath, "$.inputPath");
  const inputBytes = await readBoundedRegularFile(path, MAX_INPUT_BYTES, "$.inputPath");
  let specification;
  try {
    specification = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(inputBytes));
  } catch (cause) {
    invalid("$.inputPath", "input must be strict UTF-8 JSON", { cause });
  }
  return specification;
}

export function parseCppCuteBrowserHeaderPackInventoryArguments(argv) {
  if (!Array.isArray(argv) || argv.length !== 2) {
    invalid("$arguments", "expected exactly --input=/absolute/path and --output=/absolute/path");
  }
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string") invalid(`$arguments[${index}]`, "expected string argument");
    const match = /^--(input|output)=(.+)$/u.exec(argument);
    if (match === null) invalid(`$arguments[${index}]`, "expected --input= or --output=");
    const [, name, value] = match;
    if (values.has(name)) invalid(`$arguments[${index}]`, `duplicate --${name}`);
    values.set(name, absolutePath(value, `$arguments.${name}`));
  }
  if (!values.has("input") || !values.has("output")) {
    invalid("$arguments", "both --input and --output are required");
  }
  return Object.freeze({ inputPath: values.get("input"), outputPath: values.get("output") });
}

export async function authorCppCuteBrowserHeaderPackInventory(input) {
  const object = exactObject(input, ["inputPath", "outputPath"], "$.input");
  const inputPath = absolutePath(object.inputPath, "$.input.inputPath");
  const outputPath = absolutePath(object.outputPath, "$.input.outputPath");
  if (inputPath === outputPath) invalid("$.input", "input and output paths must differ");
  await admitCanonicalDirectory(dirname(outputPath), "$.input.outputPath.parent");
  const specification = await readCppCuteBrowserHeaderPackInventorySpecification(inputPath);
  const inventory = await inventoryCppCuteBrowserHeaderPackSources(specification);
  const outputBytes = canonicalCppCuteBrowserHeaderPackInventoryBytes(inventory);
  await writeExclusiveRegularFile(outputPath, outputBytes, "$.input.outputPath");
  return Object.freeze({
    outputPath,
    inventoryId: inventory.inventoryId,
    inventorySha256: sha256(outputBytes),
    inventoryByteLength: outputBytes.byteLength,
    packCount: inventory.totals.packCount,
    fileCount: inventory.totals.fileCount,
    releaseReady: false,
  });
}

function parsePack(value, inputIndex) {
  const path = `$.input.packs[${inputIndex}]`;
  const object = exactObject(value, ["includeRootId", "sources"], path);
  const includeRootId = identifier(object.includeRootId, `${path}.includeRootId`);
  const rawSources = denseArray(object.sources, `${path}.sources`, 1, MAX_SOURCES);
  const sources = rawSources.map((source, sourceIndex) => {
    const sourcePath = `${path}.sources[${sourceIndex}]`;
    const fields = exactObject(
      source,
      ["sourceRoot", "virtualPrefix", "licenseComponentIds"],
      sourcePath,
    );
    const licenseComponentIds = denseArray(
      fields.licenseComponentIds,
      `${sourcePath}.licenseComponentIds`,
      1,
      16,
    ).map((component, componentIndex) =>
      identifier(component, `${sourcePath}.licenseComponentIds[${componentIndex}]`));
    licenseComponentIds.sort(compareUtf8);
    for (let index = 1; index < licenseComponentIds.length; index += 1) {
      if (licenseComponentIds[index - 1] === licenseComponentIds[index]) {
        invalid(`${sourcePath}.licenseComponentIds`, "license component IDs must be unique");
      }
    }
    return Object.freeze({
      inputIndex: sourceIndex,
      sourceRoot: absolutePath(fields.sourceRoot, `${sourcePath}.sourceRoot`),
      virtualPrefix: portableRelativePath(fields.virtualPrefix, `${sourcePath}.virtualPrefix`, true),
      licenseComponentIds: Object.freeze(licenseComponentIds),
    });
  });
  return Object.freeze({ inputIndex, includeRootId, sources: Object.freeze(sources) });
}

async function inventoryDirectory(context) {
  const before = await lstatDirectory(context.directoryPath, context.diagnosticPath);
  let directory;
  try {
    directory = await opendir(context.directoryPath);
  } catch (cause) {
    invalid(context.diagnosticPath, "failed to open source directory", { cause });
  }
  const names = [];
  try {
    for await (const entry of directory) names.push(entry.name);
  } catch (cause) {
    invalid(context.diagnosticPath, "failed to enumerate source directory", { cause });
  }
  names.sort(compareUtf8);
  for (const name of names) {
    portableSegment(name, `${context.diagnosticPath}.entries`);
    const entryPath = join(context.directoryPath, name);
    const relativeSegments = [...context.relativeSegments, name];
    const entryDiagnosticPath = `${context.diagnosticPath}/${relativeSegments.join("/")}`;
    let entry;
    try {
      entry = await lstat(entryPath, { bigint: true });
    } catch (cause) {
      invalid(entryDiagnosticPath, "source entry disappeared during inventory", { cause });
    }
    if (entry.isSymbolicLink()) invalid(entryDiagnosticPath, "symbolic links are forbidden");
    if (entry.isDirectory()) {
      await inventoryDirectory({ ...context, directoryPath: entryPath, relativeSegments });
      continue;
    }
    if (!entry.isFile()) invalid(entryDiagnosticPath, "only regular files and directories are allowed");
    const relativePath = relativeSegments.join("/");
    const virtualPath = context.virtualPrefix === ""
      ? relativePath
      : `${context.virtualPrefix}/${relativePath}`;
    portableRelativePath(virtualPath, entryDiagnosticPath, false);
    if (context.filesByPath.has(virtualPath)) {
      invalid(entryDiagnosticPath, `duplicate virtual path ${JSON.stringify(virtualPath)}`);
    }
    const file = await hashRegularFile(entryPath, entry, context.budget, entryDiagnosticPath);
    context.filesByPath.set(virtualPath, Object.freeze({
      virtualPath,
      contentSha256: file.sha256,
      byteLength: String(file.byteLength),
      licenseComponentIds: Object.freeze([...context.licenseComponentIds]),
    }));
    context.sourceFilesByPath.set(virtualPath, Object.freeze({
      sourcePath: entryPath,
      expected: context.filesByPath.get(virtualPath),
    }));
  }
  const after = await lstatDirectory(context.directoryPath, context.diagnosticPath);
  if (!sameStatIdentity(before, after)) {
    invalid(context.diagnosticPath, "source directory changed during inventory");
  }
}

async function hashRegularFile(path, discovered, budget, diagnosticPath) {
  if (discovered.nlink !== 1n) invalid(diagnosticPath, "source files must have exactly one hard link");
  admitFileSize(discovered.size, diagnosticPath);
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameStatIdentity(discovered, before) || !before.isFile() || before.nlink !== 1n) {
      invalid(diagnosticPath, "source file identity changed before hashing");
    }
    admitFileSize(before.size, diagnosticPath);
    const hash = createHash("sha256");
    const buffer = Buffer.allocUnsafe(Math.min(READ_BUFFER_BYTES, Number(before.size) || 1));
    let offset = 0;
    while (offset < Number(before.size)) {
      const length = Math.min(buffer.byteLength, Number(before.size) - offset);
      const { bytesRead } = await handle.read(buffer, 0, length, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "source file changed while it was hashed");
      hash.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameStatIdentity(before, after)) invalid(diagnosticPath, "source file changed while it was hashed");
    budget.files += 1;
    budget.bytes += offset;
    if (budget.files > MAX_FILES) resource("$.input.packs", `inventory exceeds ${MAX_FILES} files`);
    if (budget.bytes > MAX_TOTAL_FILE_BYTES) {
      resource("$.input.packs", `inventory exceeds ${MAX_TOTAL_FILE_BYTES} file-content bytes`);
    }
    return Object.freeze({ sha256: hash.digest("hex"), byteLength: offset });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackInventoryError) throw cause;
    invalid(diagnosticPath, "failed to hash source file", { cause });
  } finally {
    await handle?.close();
  }
}

function rejectFileDirectoryCollisions(files, diagnosticPath) {
  const paths = new Set(files.map((file) => file.virtualPath));
  for (const file of files) {
    const segments = file.virtualPath.split("/");
    for (let end = 1; end < segments.length; end += 1) {
      const ancestor = segments.slice(0, end).join("/");
      if (paths.has(ancestor)) {
        invalid(diagnosticPath, `virtual file ${JSON.stringify(ancestor)} collides with a directory`);
      }
    }
  }
}

function bindPackPoliciesToBuildLock(body) {
  const approved = body.notices.approvedComponents;
  const unresolved = body.notices.unresolvedComponents;
  return Object.freeze(PACK_POLICIES
    .map((policy) => {
      const outputs = body.recipe.distributedOutputPlan.outputs.filter(
        (output) => output.role === policy.outputRole,
      );
      if (outputs.length !== 1 || outputs[0].path !== policy.outputPath ||
          outputs[0].reproducibilityClass !== "deterministic-subject") {
        invalid("$.buildInputLock", `current build lock lost ${JSON.stringify(policy.outputRole)}`);
      }
      const licenseComponentIds = [
        ...approved.filter((notice) => notice.appliesTo.includes(policy.intendedAsset))
          .map((notice) => notice.componentId),
        ...unresolved.filter((notice) => notice.intendedAsset === policy.intendedAsset)
          .map((notice) => notice.componentId),
      ].sort(compareUtf8);
      if (licenseComponentIds.length === 0 || new Set(licenseComponentIds).size !== licenseComponentIds.length) {
        invalid("$.buildInputLock", `header asset ${JSON.stringify(policy.intendedAsset)} has no unambiguous notice policy`);
      }
      return Object.freeze({ ...policy, licenseComponentIds: Object.freeze(licenseComponentIds) });
    })
    .sort((left, right) => compareUtf8(left.includeRootId, right.includeRootId)));
}

async function admitCanonicalDirectory(path, diagnosticPath) {
  const before = await lstatDirectory(path, diagnosticPath);
  let resolved;
  try {
    resolved = await realpath(path);
  } catch (cause) {
    invalid(diagnosticPath, "failed to resolve source directory", { cause });
  }
  if (resolved !== path) invalid(diagnosticPath, "directory path must already be canonical and contain no symlinks");
  const after = await lstatDirectory(path, diagnosticPath);
  if (!sameStatIdentity(before, after)) invalid(diagnosticPath, "directory identity changed during admission");
  return path;
}

async function lstatDirectory(path, diagnosticPath) {
  let entry;
  try {
    entry = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "source directory is unavailable", { cause });
  }
  if (!entry.isDirectory() || entry.isSymbolicLink() || entry.nlink < 1n) {
    invalid(diagnosticPath, "expected one non-symlink directory");
  }
  return entry;
}

async function readBoundedRegularFile(path, maximum, diagnosticPath) {
  let discovered;
  try {
    discovered = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "input file is unavailable", { cause });
  }
  if (!discovered.isFile() || discovered.isSymbolicLink() || discovered.nlink !== 1n ||
      discovered.size < 1n || discovered.size > BigInt(maximum)) {
    invalid(diagnosticPath, `expected one nonempty non-symlink regular file no larger than ${maximum} bytes`);
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameStatIdentity(discovered, before)) invalid(diagnosticPath, "input file identity changed before read");
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "input file changed while read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameStatIdentity(before, after)) invalid(diagnosticPath, "input file changed while read");
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackInventoryError) throw cause;
    invalid(diagnosticPath, "failed to read bounded regular file", { cause });
  } finally {
    await handle?.close();
  }
}

async function readExactInventorySourceFile(path, expected, diagnosticPath) {
  let discovered;
  try {
    discovered = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "inventoried source file is unavailable", { cause });
  }
  if (!discovered.isFile() || discovered.isSymbolicLink() || discovered.nlink !== 1n ||
      discovered.size !== BigInt(expected.byteLength)) {
    invalid(diagnosticPath, "inventoried source file identity or length changed");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameStatIdentity(discovered, before)) {
      invalid(diagnosticPath, "inventoried source file changed before read");
    }
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "inventoried source file changed while read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameStatIdentity(before, after) || sha256(bytes) !== expected.contentSha256) {
      invalid(diagnosticPath, "inventoried source file bytes changed");
    }
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderPackInventoryError) throw cause;
    invalid(diagnosticPath, "failed to copy inventoried source file", { cause });
  } finally {
    await handle?.close();
  }
}

async function writeExclusiveRegularFile(path, bytes, diagnosticPath) {
  let handle;
  let identity;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o444,
    );
    const opened = await handle.stat({ bigint: true });
    if (!opened.isFile() || opened.size !== 0n || opened.nlink !== 1n) {
      invalid(diagnosticPath, "new output is not one empty regular file");
    }
    identity = opened;
    await handle.writeFile(bytes);
    await handle.sync();
    const written = await handle.stat({ bigint: true });
    if (written.size !== BigInt(bytes.byteLength) || written.dev !== opened.dev ||
        written.ino !== opened.ino || written.nlink !== 1n) {
      invalid(diagnosticPath, "output identity changed while written");
    }
    await handle.close();
    handle = undefined;
    const persisted = await lstat(path, { bigint: true });
    if (!persisted.isFile() || persisted.isSymbolicLink() || persisted.dev !== opened.dev ||
        persisted.ino !== opened.ino || persisted.nlink !== 1n ||
        persisted.size !== BigInt(bytes.byteLength)) {
      invalid(diagnosticPath, "output path no longer names the owned inventory inode");
    }
    const reread = await readBoundedRegularFile(path, bytes.byteLength, diagnosticPath);
    const finalIdentity = await lstat(path, { bigint: true });
    if (finalIdentity.dev !== opened.dev || finalIdentity.ino !== opened.ino ||
        sha256(reread) !== sha256(bytes)) {
      invalid(diagnosticPath, "persisted inventory differs from the authored canonical bytes");
    }
  } catch (cause) {
    if (handle !== undefined) {
      try {
        await handle.close();
      } catch {
        // Continue with identity-bound cleanup.
      }
      handle = undefined;
    }
    if (identity !== undefined) {
      try {
        const current = await lstat(path, { bigint: true });
        if (current.dev === identity.dev && current.ino === identity.ino) await unlink(path);
      } catch {
        // Never remove a replacement output.
      }
    }
    if (cause instanceof CppCuteBrowserHeaderPackInventoryError) throw cause;
    invalid(diagnosticPath, "failed to write exclusive inventory output", { cause });
  } finally {
    await handle?.close();
  }
}

function sameStatIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink && left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function admitFileSize(size, diagnosticPath) {
  if (size < 0n || size > BigInt(MAX_FILE_BYTES)) {
    resource(diagnosticPath, `source file exceeds ${MAX_FILE_BYTES} bytes`);
  }
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length || actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected exactly ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) invalid(`${diagnosticPath}.${key}`, "expected data property");
    result[key] = descriptor.value;
  }
  return result;
}

function denseArray(value, diagnosticPath, minimum, maximum) {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) {
    invalid(diagnosticPath, `expected ${minimum}..${maximum} array entries`);
  }
  if (Object.getPrototypeOf(value) !== Array.prototype) {
    invalid(diagnosticPath, "expected one ordinary dense array");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length !== value.length + 1 || keys.some((key) =>
    typeof key !== "string" || (key !== "length" && !/^(?:0|[1-9][0-9]*)$/u.test(key)))) {
    invalid(diagnosticPath, "array contains non-index properties");
  }
  const result = [];
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = descriptors[String(index)];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${diagnosticPath}[${index}]`, "sparse arrays and accessors are forbidden");
    }
    result.push(descriptor.value);
  }
  return result;
}

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !isAbsolute(value) || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function identifier(value, diagnosticPath) {
  if (typeof value !== "string" || value.length > 128 || !IDENTIFIER.test(value)) {
    invalid(diagnosticPath, "expected one bounded lowercase identifier");
  }
  return value;
}

function portableSegment(value, diagnosticPath) {
  if (value === "." || value === ".." || !PORTABLE_SEGMENT.test(value)) {
    invalid(diagnosticPath, `non-portable path segment ${JSON.stringify(value)}`);
  }
  return value;
}

function portableRelativePath(value, diagnosticPath, allowEmpty) {
  if (typeof value !== "string" || value.includes("\\") || value.startsWith("/") ||
      (!allowEmpty && value === "")) {
    invalid(diagnosticPath, "expected one portable relative POSIX path");
  }
  if (value !== "") for (const segment of value.split("/")) portableSegment(segment, diagnosticPath);
  if (TEXT_ENCODER.encode(value).byteLength > MAX_PATH_BYTES) {
    resource(diagnosticPath, `path exceeds ${MAX_PATH_BYTES} UTF-8 bytes`);
  }
  return value;
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sameStrings(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderPackInventoryError(path, message, options);
}

function resource(path, message, options) {
  throw new CppCuteBrowserHeaderPackInventoryError(path, `resource limit: ${message}`, options);
}

async function main() {
  try {
    const options = parseCppCuteBrowserHeaderPackInventoryArguments(process.argv.slice(2));
    const report = await authorCppCuteBrowserHeaderPackInventory(options);
    process.stdout.write(`${JSON.stringify(report)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-pack inventory failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
