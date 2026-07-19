import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, opendir, realpath, unlink } from "node:fs/promises";
import { dirname, join } from "node:path/posix";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  requireCppCuteBrowserCudaRedistributionIndexAuthority,
} from "./cpp_cute_browser_cuda_redistribution_index.mjs";
import {
  requireCppCuteBrowserHeaderNoticeVerificationAuthority,
} from "./cpp_cute_browser_header_notice_verification.mjs";
import {
  requireCppCuteBrowserHeaderPackInventorySourceAuthority,
} from "./cpp_cute_browser_header_pack_inventory.mjs";
import {
  requireCppCuteBrowserHeaderPackMaterializationAuthority,
} from "./cpp_cute_browser_header_pack_materialization.mjs";
import {
  requireCppCuteBrowserHeaderSourceExtractionAuthority,
} from "./cpp_cute_browser_header_source_extraction.mjs";

export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REVIEW_INPUT_SCHEMA =
  "browsergrad.compiler.cpp-cute.header-distribution-review-input";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-DISTRIBUTION-REVIEW-INPUT";
const REVIEW_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.header-distribution-review-input.v1";
const MAX_REVIEW_INPUT_BYTES = 32 * 1024 * 1024;
const MAX_PACK_BYTES = 128 * 1024 * 1024;
const READ_BUFFER_BYTES = 256 * 1024;
const MAX_OUTPUT_TREE_ENTRIES = 128;
const MAX_OUTPUT_TREE_DEPTH = 16;
const MAX_OUTPUT_PATH_BYTES = 4_096;
const REVIEW_INPUTS = new WeakMap();

export class CppCuteBrowserHeaderDistributionReviewInputError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderDistributionReviewInputError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Materializes the deterministic license-inventory review input for the exact
 * five header packs. It binds every distributed virtual file to its component,
 * exact pack output, package notice bytes, upstream license/copyright bytes,
 * and the selected CUDA index. External review remains a separate authority.
 */
export async function materializeCppCuteBrowserHeaderDistributionReviewInput(input) {
  const object = exactObject(
    input,
    ["cudaRedistributionIndex", "extraction", "inventory", "materialization", "notices"],
    "$.input",
  );
  requireAuthorities(object);
  assertIdentityChain(object);
  const outputPath = await distributionReviewInputOutputPath(object.notices);
  const manifest = composeReviewInput(object);
  const bytes = canonicalJsonBytes(manifest);
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_REVIEW_INPUT_BYTES) {
    resource("$.reviewInput", `canonical review input exceeds ${MAX_REVIEW_INPUT_BYTES} bytes`);
  }
  const expectedPackPaths = new Set(object.materialization.outputs.map((output) => output.outputPath));
  const rootIdentity = await assertExactOutputTree(
    object.materialization.outputRoot,
    expectedPackPaths,
    "$.materialization.outputRoot",
  );
  await verifyPersistedPacks(object.materialization);
  const absoluteOutputPath = join(object.materialization.outputRoot, outputPath);
  const outputIdentity = await writeExclusiveReviewInput(absoluteOutputPath, bytes);
  const persisted = await readExactFile(
    absoluteOutputPath,
    bytes.byteLength,
    outputIdentity,
    MAX_REVIEW_INPUT_BYTES,
    "$.reviewInput.output",
  );
  if (sha256(persisted) !== sha256(bytes) || !sameBytes(persisted, bytes)) {
    invalid("$.reviewInput.output", "persisted review input differs from canonical bytes");
  }
  const finalPaths = new Set([...expectedPackPaths, outputPath]);
  const finalRootIdentity = await assertExactOutputTree(
    object.materialization.outputRoot,
    finalPaths,
    "$.materialization.outputRoot",
  );
  if (rootIdentity.dev !== finalRootIdentity.dev || rootIdentity.ino !== finalRootIdentity.ino) {
    invalid("$.materialization.outputRoot", "pack output root identity changed during review authoring");
  }
  const report = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REVIEW_INPUT_SCHEMA,
    version: 1,
    reviewInputId: manifest.reviewInputId,
    authority: "materialized-exact-header-distribution-review-input-only",
    buildInputLockId: manifest.buildInputLockId,
    buildInputLockResourceSha256: manifest.buildInputLockResourceSha256,
    headerSourcePlanId: manifest.headerSourcePlanId,
    extractionId: manifest.extractionId,
    inventoryId: manifest.inventoryId,
    cudaRedistributionIndexId: manifest.cudaRedistributionIndex.indexId,
    outputPath,
    reviewInputSha256: sha256(persisted),
    reviewInputByteLength: String(persisted.byteLength),
    totals: manifest.totals,
    unresolvedExternalReviews: manifest.unresolvedExternalReviews,
    claims: manifest.claims,
  });
  REVIEW_INPUTS.set(report, manifest);
  return report;
}

export function requireCppCuteBrowserHeaderDistributionReviewInputAuthority(report) {
  if (typeof report !== "object" || report === null || !REVIEW_INPUTS.has(report)) {
    invalid("$.reviewInput", "expected package-issued header distribution review input");
  }
}

export function canonicalCppCuteBrowserHeaderDistributionReviewInputBytes(report) {
  requireCppCuteBrowserHeaderDistributionReviewInputAuthority(report);
  return canonicalJsonBytes(REVIEW_INPUTS.get(report));
}

function requireAuthorities(object) {
  const checks = [
    ["$.input.extraction", () => requireCppCuteBrowserHeaderSourceExtractionAuthority(object.extraction)],
    ["$.input.inventory", () => requireCppCuteBrowserHeaderPackInventorySourceAuthority(object.inventory)],
    ["$.input.materialization", () => requireCppCuteBrowserHeaderPackMaterializationAuthority(object.materialization)],
    ["$.input.notices", () => requireCppCuteBrowserHeaderNoticeVerificationAuthority(object.notices)],
    ["$.input.cudaRedistributionIndex", () =>
      requireCppCuteBrowserCudaRedistributionIndexAuthority(object.cudaRedistributionIndex)],
  ];
  for (const [path, check] of checks) {
    try {
      check();
    } catch (cause) {
      invalid(path, "review input requires one live verifier-issued authority", { cause });
    }
  }
}

function assertIdentityChain(object) {
  const { extraction, inventory, materialization, notices, cudaRedistributionIndex } = object;
  const lockId = extraction.buildInputLockId;
  const lockSha256 = extraction.buildInputLockResourceSha256;
  if (inventory.headerSourceExtractionId !== extraction.extractionId ||
      inventory.buildInputLockId !== lockId || inventory.buildInputLockResourceSha256 !== lockSha256 ||
      materialization.inventoryId !== inventory.inventoryId ||
      materialization.buildInputLockId !== lockId ||
      materialization.buildInputLockResourceSha256 !== lockSha256 ||
      notices.buildInputLockId !== lockId || notices.buildInputLockResourceSha256 !== lockSha256 ||
      cudaRedistributionIndex.headerSourcePlanId !== extraction.headerSourcePlanId) {
    invalid("$.input", "review-input authorities do not form one exact current identity chain");
  }
  if (inventory.packs.length !== 5 || materialization.outputs.length !== 5) {
    invalid("$.input.inventory", "review input must bind exactly five header packs");
  }
  for (const [index, pack] of inventory.packs.entries()) {
    const output = materialization.outputs[index];
    if (output === undefined || output.ordinal !== index || output.includeRootId !== pack.includeRootId ||
        output.intendedAsset !== pack.intendedAsset || output.outputRole !== pack.outputRole ||
        output.outputPath !== pack.outputPath || output.contentSetSha256 !== pack.contentSetSha256 ||
        output.fileCount !== pack.fileCount ||
        output.fileContentByteLength !== pack.fileContentByteLength) {
      invalid(`$.input.materialization.outputs[${index}]`, "pack output differs from exact file inventory");
    }
  }
}

async function distributionReviewInputOutputPath(notices) {
  const allOutputPaths = new Set(notices.notices.map((notice) => notice.noticeOutputPath));
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  if (buildInputLock.lockId !== notices.buildInputLockId ||
      buildInputLock.resourceSha256 !== notices.buildInputLockResourceSha256) {
    invalid("$.input.notices", "notice authority differs from the current build lock");
  }
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body;
  const outputs = body.recipe.distributedOutputPlan.outputs.filter(
    (output) => output.role === "license-inventory",
  );
  if (outputs.length !== 1 || outputs[0].path !==
      "assets/browsergrad-cpp-cute/license-inventory.json" ||
      outputs[0].mediaType !== "application/json" ||
      outputs[0].reproducibilityClass !== "deterministic-subject") {
    invalid("$.buildInputLock", "current deterministic license-inventory output is not exact");
  }
  const outputPath = outputs[0].path;
  if (allOutputPaths.has(outputPath)) {
    invalid("$.input.notices", "license inventory path collides with a component notice");
  }
  return outputPath;
}

function composeReviewInput(object) {
  const { extraction, inventory, materialization, notices, cudaRedistributionIndex } = object;
  const evidence = extraction.archives.flatMap((archive) => archive.licenseEvidence.map((item) =>
    Object.freeze({ sourceId: archive.sourceId, ...item })))
    .sort((left, right) => compareUtf8(
      `${left.componentId}\0${left.sourceId}\0${left.evidenceId}`,
      `${right.componentId}\0${right.sourceId}\0${right.evidenceId}`,
    ));
  const approvedByComponent = new Map(notices.notices.map((notice) => [notice.componentId, notice]));
  const unresolvedByComponent = new Map(
    notices.unresolvedNotices.map((notice) => [notice.componentId, notice]),
  );
  const fileComponentIds = new Set(inventory.packs.flatMap((pack) =>
    pack.files.flatMap((file) => file.licenseComponentIds)));
  const components = [...fileComponentIds].sort(compareUtf8).map((componentId) => {
    const approvedNotice = approvedByComponent.get(componentId);
    const unresolvedNotice = unresolvedByComponent.get(componentId);
    if ((approvedNotice === undefined) === (unresolvedNotice === undefined)) {
      invalid("$.input.inventory.packs.files.licenseComponentIds", "component has no unique review policy");
    }
    const sourceEvidence = evidence.filter((item) => item.componentId === componentId);
    if (sourceEvidence.length === 0) {
      invalid("$.input.extraction.licenseEvidence", `component ${JSON.stringify(componentId)} has no source evidence`);
    }
    const intendedAssets = [...new Set(inventory.packs
      .filter((pack) => pack.files.some((file) => file.licenseComponentIds.includes(componentId)))
      .map((pack) => pack.intendedAsset))].sort(compareUtf8);
    if (approvedNotice !== undefined &&
        intendedAssets.some((asset) => !approvedNotice.appliesTo.includes(asset))) {
      invalid("$.input.notices", "approved component notice does not cover its header-pack asset");
    }
    return Object.freeze({
      componentId,
      intendedAssets: Object.freeze(intendedAssets),
      reviewState: approvedNotice === undefined
        ? "external-component-and-file-map-review-pending"
        : "package-notice-approved-external-file-map-review-pending",
      ...(approvedNotice === undefined
        ? { unresolvedPolicy: Object.freeze({ ...unresolvedNotice }) }
        : { packageApprovedNotice: cloneNotice(approvedNotice) }),
      sourceEvidence: Object.freeze(sourceEvidence),
      ...(componentId === "cuda-toolkit-12.6.3-headers"
        ? { cudaRedistributionIndexComponents: Object.freeze(
          cudaRedistributionIndex.components.map(cloneCudaIndexComponent),
        ) }
        : {}),
    });
  });
  const expectedComponents = [
    "clang",
    "cuda-toolkit-12.6.3-headers",
    "cutlass",
    "libcxx",
    "linux-sysroot",
  ];
  if (!sameStrings(components.map((component) => component.componentId), expectedComponents)) {
    invalid("$.input.inventory", "header file map does not cover the exact current component set");
  }
  const cudaSourceIds = new Set(cudaRedistributionIndex.components.map((component) => component.sourceId));
  const cudaEvidenceSourceIds = new Set(evidence
    .filter((item) => item.componentId === "cuda-toolkit-12.6.3-headers")
    .map((item) => item.sourceId));
  if (!sameStrings([...cudaSourceIds].sort(compareUtf8), [...cudaEvidenceSourceIds].sort(compareUtf8))) {
    invalid("$.input.cudaRedistributionIndex", "CUDA index and extracted license sources differ");
  }
  let fileOrdinal = 0;
  const packs = inventory.packs.map((pack, packOrdinal) => {
    const output = materialization.outputs[packOrdinal];
    if (output === undefined) invalid("$.input.materialization.outputs", "pack output is absent");
    return Object.freeze({
      ordinal: packOrdinal,
      includeRootId: pack.includeRootId,
      intendedAsset: pack.intendedAsset,
      outputRole: pack.outputRole,
      outputPath: pack.outputPath,
      packSha256: output.packSha256,
      packByteLength: output.packByteLength,
      contentSetSha256: pack.contentSetSha256,
      fileCount: pack.fileCount,
      fileContentByteLength: pack.fileContentByteLength,
      files: Object.freeze(pack.files.map((file) => Object.freeze({
        ordinal: fileOrdinal++,
        virtualPath: file.virtualPath,
        contentSha256: file.contentSha256,
        byteLength: file.byteLength,
        licenseComponentIds: file.licenseComponentIds,
      }))),
    });
  });
  const body = Object.freeze({
    buildInputLockId: extraction.buildInputLockId,
    buildInputLockResourceSha256: extraction.buildInputLockResourceSha256,
    headerSourcePlanId: extraction.headerSourcePlanId,
    archiveAdmissionId: extraction.archiveAdmissionId,
    extractionId: extraction.extractionId,
    inventoryId: inventory.inventoryId,
    scope: "exact-five-browser-header-packs-only",
    cudaRedistributionIndex: Object.freeze({
      indexId: cudaRedistributionIndex.indexId,
      sourceUrl: cudaRedistributionIndex.sourceUrl,
      releaseLabel: cudaRedistributionIndex.releaseLabel,
      releaseProduct: cudaRedistributionIndex.releaseProduct,
      releaseDate: cudaRedistributionIndex.releaseDate,
      indexSha256: cudaRedistributionIndex.indexSha256,
      indexByteLength: cudaRedistributionIndex.indexByteLength,
      components: Object.freeze(cudaRedistributionIndex.components.map(cloneCudaIndexComponent)),
    }),
    packageNoticeSet: Object.freeze(notices.notices.map(cloneNotice)),
    components: Object.freeze(components),
    packs: Object.freeze(packs),
    totals: Object.freeze({
      packCount: packs.length,
      fileMapEntryCount: fileOrdinal,
      fileContentByteLength: inventory.totals.fileContentByteLength,
      packageNoticeCount: notices.notices.length,
      sourceEvidenceFileCount: evidence.length,
      sourceEvidenceByteLength: evidence.reduce(
        (total, item) => total + BigInt(item.byteLength),
        0n,
      ).toString(),
      componentCount: components.length,
    }),
    unresolvedExternalReviews: Object.freeze([
      Object.freeze({
        blockerId: "cuda-header-redistribution",
        requirement: "external-review-of-exact-cuda-header-file-map-index-and-license-evidence",
      }),
      Object.freeze({
        blockerId: "distributed-file-license-manifest",
        requirement: "external-review-of-every-file-to-license-component-mapping",
      }),
      Object.freeze({
        blockerId: "linux-sysroot-redistribution",
        requirement: "external-review-of-exact-linux-sysroot-file-map-and-source-package-copyright-evidence",
      }),
    ]),
    claims: Object.freeze({
      exactPerDistributedFileComponentMapPrepared: true,
      exactMaterializedPackOutputsBound: true,
      exactPackageNoticeBytesBound: true,
      exactUpstreamLicenseEvidenceBound: true,
      exactCudaRedistributionIndexBound: true,
      allHeaderPackFilesCovered: fileOrdinal === inventory.totals.fileCount,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
  if (!body.claims.allHeaderPackFilesCovered) {
    invalid("$.input.inventory", "not every distributed header file entered the review map");
  }
  const reviewHash = sha256(canonicalJsonBytes({ domain: REVIEW_HASH_DOMAIN, body }));
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REVIEW_INPUT_SCHEMA,
    version: 1,
    reviewInputId: `bg.cpp.header-distribution-review-input.sha256.${reviewHash}`,
    authority: "exact-header-distribution-review-input-only",
    ...body,
  });
}

async function verifyPersistedPacks(materialization) {
  for (const [index, output] of materialization.outputs.entries()) {
    const path = join(materialization.outputRoot, output.outputPath);
    const byteLength = Number(output.packByteLength);
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength > MAX_PACK_BYTES) {
      invalid(`$.materialization.outputs[${index}]`, "pack length is outside the reread bound");
    }
    const identity = await lstat(path, { bigint: true }).catch((cause) =>
      invalid(`$.materialization.outputs[${index}]`, "pack output is unavailable", { cause }));
    const digest = await hashExactFile(
      path,
      byteLength,
      identity,
      `$.materialization.outputs[${index}]`,
    );
    if (digest !== output.packSha256) {
      invalid(`$.materialization.outputs[${index}]`, "pack bytes changed before review-input authoring");
    }
  }
}

async function hashExactFile(path, expectedByteLength, discovered, diagnosticPath) {
  const uid = typeof process.getuid === "function" ? BigInt(process.getuid()) : undefined;
  if (!discovered.isFile() || discovered.isSymbolicLink() || discovered.nlink !== 1n ||
      discovered.size !== BigInt(expectedByteLength) || Number(discovered.mode & 0o022n) !== 0 ||
      (uid !== undefined && discovered.uid !== uid) || await realpath(path) !== path) {
    invalid(diagnosticPath, "expected one exact canonical regular file");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(discovered, before)) invalid(diagnosticPath, "file identity changed before read");
    const digest = createHash("sha256");
    const buffer = new Uint8Array(READ_BUFFER_BYTES);
    let offset = 0;
    while (offset < expectedByteLength) {
      const length = Math.min(buffer.byteLength, expectedByteLength - offset);
      const { bytesRead } = await handle.read(buffer, 0, length, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "file changed while read");
      digest.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "file identity changed while read");
    return digest.digest("hex");
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderDistributionReviewInputError) throw cause;
    invalid(diagnosticPath, "failed to reread exact file", { cause });
  } finally {
    await handle?.close();
  }
}

async function writeExclusiveReviewInput(path, bytes) {
  const parent = dirname(path);
  const parentBefore = await lstat(parent, { bigint: true }).catch((cause) =>
    invalid("$.reviewInput.output.parent", "review-input parent is unavailable", { cause }));
  if (!parentBefore.isDirectory() || parentBefore.isSymbolicLink() ||
      Number(parentBefore.mode & 0o077n) !== 0 || await realpath(parent) !== parent) {
    invalid("$.reviewInput.output.parent", "review-input parent must be one private canonical directory");
  }
  let handle;
  let identity;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    identity = await handle.stat({ bigint: true });
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesWritten } = await handle.write(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesWritten <= 0) invalid("$.reviewInput.output", "review-input write made no progress");
      offset += bytesWritten;
    }
    await handle.sync();
    const after = await handle.stat({ bigint: true });
    if (!sameStableFileIdentity(identity, after) || after.size !== BigInt(bytes.byteLength)) {
      invalid("$.reviewInput.output", "review-input identity changed while written");
    }
    const parentAfter = await lstat(parent, { bigint: true });
    if (!sameDirectoryIdentity(parentBefore, parentAfter) || await realpath(parent) !== parent) {
      invalid("$.reviewInput.output.parent", "review-input parent changed while written");
    }
    return Object.freeze({ dev: after.dev, ino: after.ino, mode: after.mode, nlink: after.nlink,
      size: after.size, mtimeNs: after.mtimeNs, ctimeNs: after.ctimeNs });
  } catch (cause) {
    if (identity !== undefined) await unlinkOwnedFile(path, identity);
    if (cause instanceof CppCuteBrowserHeaderDistributionReviewInputError) throw cause;
    invalid("$.reviewInput.output", "failed to persist review input", { cause });
  } finally {
    await handle?.close();
  }
}

async function readExactFile(path, expectedByteLength, identity, maximumBytes, diagnosticPath) {
  if (expectedByteLength <= 0 || expectedByteLength > maximumBytes) {
    resource(diagnosticPath, "file exceeds reread bound");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(identity, before) || before.size !== BigInt(expectedByteLength)) {
      invalid(diagnosticPath, "persisted file identity differs before reread");
    }
    const bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid(diagnosticPath, "persisted file changed while reread");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid(diagnosticPath, "persisted file changed while reread");
    return bytes;
  } finally {
    await handle?.close();
  }
}

async function assertExactOutputTree(root, expectedFiles, diagnosticPath) {
  const canonical = await realpath(root).catch((cause) =>
    invalid(diagnosticPath, "pack output root is unavailable", { cause }));
  if (canonical !== root) invalid(diagnosticPath, "pack output root must be canonical");
  const rootStat = await lstat(root, { bigint: true });
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || Number(rootStat.mode & 0o077n) !== 0) {
    invalid(diagnosticPath, "pack output root must remain one private directory");
  }
  const observed = [];
  await walkOutputTree(root, "", observed, diagnosticPath, { entries: 0 }, 0);
  observed.sort(compareUtf8);
  const expected = [...expectedFiles].sort(compareUtf8);
  if (!sameStrings(observed, expected)) {
    invalid(diagnosticPath, "pack output tree differs from the exact expected file set");
  }
  return Object.freeze({ dev: rootStat.dev, ino: rootStat.ino });
}

async function walkOutputTree(root, relativeDirectory, files, diagnosticPath, budget, depth) {
  if (depth > MAX_OUTPUT_TREE_DEPTH) resource(diagnosticPath, "output tree exceeds depth bound");
  const path = relativeDirectory === "" ? root : join(root, relativeDirectory);
  let directory;
  try {
    directory = await opendir(path);
    for await (const entry of directory) {
      budget.entries += 1;
      if (budget.entries > MAX_OUTPUT_TREE_ENTRIES) {
        resource(diagnosticPath, "output tree exceeds entry bound");
      }
      const relativePath = relativeDirectory === "" ? entry.name : `${relativeDirectory}/${entry.name}`;
      if (Buffer.byteLength(relativePath, "utf8") > MAX_OUTPUT_PATH_BYTES) {
        resource(diagnosticPath, "output tree path exceeds byte bound");
      }
      const entryPath = join(root, relativePath);
      const stat = await lstat(entryPath, { bigint: true });
      if (entry.isSymbolicLink() || stat.isSymbolicLink()) {
        invalid(diagnosticPath, "pack output tree contains a symbolic link");
      }
      if (stat.isDirectory()) {
        if (Number(stat.mode & 0o077n) !== 0) {
          invalid(diagnosticPath, "pack output tree contains a non-private directory");
        }
        await walkOutputTree(root, relativePath, files, diagnosticPath, budget, depth + 1);
      } else if (stat.isFile() && stat.nlink === 1n && Number(stat.mode & 0o022n) === 0) {
        files.push(relativePath);
      } else {
        invalid(diagnosticPath, "pack output tree contains a non-regular entry");
      }
    }
  } catch (cause) {
    if (cause instanceof CppCuteBrowserHeaderDistributionReviewInputError) throw cause;
    invalid(diagnosticPath, "failed to inspect pack output tree", { cause });
  } finally {
    await directory?.close().catch(() => undefined);
  }
}

async function unlinkOwnedFile(path, identity) {
  const observed = await lstat(path, { bigint: true }).catch(() => undefined);
  if (observed !== undefined && observed.dev === identity.dev && observed.ino === identity.ino) {
    await unlink(path);
  }
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

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink && left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function sameStableFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink;
}

function sameDirectoryIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode;
}

function cloneNotice(notice) {
  return Object.freeze({
    ...notice,
    appliesTo: Object.freeze([...notice.appliesTo]),
  });
}

function cloneCudaIndexComponent(component) {
  return Object.freeze({ ...component });
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  return Buffer.from(left.buffer, left.byteOffset, left.byteLength)
    .equals(Buffer.from(right.buffer, right.byteOffset, right.byteLength));
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

function resource(path, message) {
  invalid(path, `resource limit: ${message}`);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderDistributionReviewInputError(path, message, options);
}
