import { createHash } from "node:crypto";

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
  materializeCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";
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
  "browsergrad.compiler.cpp-cute.header-distribution-review-input.v2";
const MAX_REVIEW_INPUT_BYTES = 32 * 1024 * 1024;
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
  let outputMaterialization;
  try {
    outputMaterialization = await materializeCppCuteBrowserDistributionOutputFiles({
      outputRoot: object.materialization.outputRoot,
      existingOutputs: object.materialization.outputs.map((output) => Object.freeze({
        outputPath: output.outputPath,
        sha256: output.packSha256,
        byteLength: output.packByteLength,
      })),
      outputs: [Object.freeze({ outputPath, bytes })],
    });
  } catch (cause) {
    invalid("$.reviewInput.output", "failed to materialize the exact distribution review input", { cause });
  }
  const persisted = outputMaterialization.outputs[0];
  if (persisted === undefined || persisted.outputPath !== outputPath) {
    invalid("$.reviewInput.output", "distribution output materializer omitted the review input");
  }
  const report = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REVIEW_INPUT_SCHEMA,
    version: 2,
    reviewInputId: manifest.reviewInputId,
    authority: "materialized-exact-header-distribution-review-input-only",
    buildInputLockId: object.extraction.buildInputLockId,
    buildInputLockResourceSha256: object.extraction.buildInputLockResourceSha256,
    headerInputProjectionId: manifest.headerInputProjectionId,
    headerSourcePlanId: manifest.headerSourcePlanId,
    extractionId: manifest.extractionId,
    inventoryId: manifest.inventoryId,
    cudaRedistributionIndexId: manifest.cudaRedistributionIndex.indexId,
    outputPath,
    reviewInputSha256: persisted.sha256,
    reviewInputByteLength: persisted.byteLength,
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
      inventory.headerInputProjectionId !== extraction.headerInputProjectionId ||
      materialization.inventoryId !== inventory.inventoryId ||
      materialization.buildInputLockId !== lockId ||
      materialization.buildInputLockResourceSha256 !== lockSha256 ||
      materialization.headerInputProjectionId !== extraction.headerInputProjectionId ||
      notices.buildInputLockId !== lockId || notices.buildInputLockResourceSha256 !== lockSha256 ||
      notices.headerInputProjectionId !== extraction.headerInputProjectionId ||
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
    headerInputProjectionId: extraction.headerInputProjectionId,
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
    version: 2,
    reviewInputId: `bg.cpp.header-distribution-review-input.sha256.${reviewHash}`,
    authority: "exact-header-distribution-review-input-only",
    ...body,
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

function cloneNotice(notice) {
  return Object.freeze({
    ...notice,
    appliesTo: Object.freeze([...notice.appliesTo]),
  });
}

function cloneCudaIndexComponent(component) {
  return Object.freeze({ ...component });
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
