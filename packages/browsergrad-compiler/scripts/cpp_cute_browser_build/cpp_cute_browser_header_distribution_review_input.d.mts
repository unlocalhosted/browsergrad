import type { CppCuteBrowserCudaRedistributionIndexAdmission } from "./cpp_cute_browser_cuda_redistribution_index.mjs";
import type { CppCuteBrowserHeaderNoticeVerification } from "./cpp_cute_browser_header_notice_verification.mjs";
import type { CppCuteBrowserHeaderPackInventory } from "./cpp_cute_browser_header_pack_inventory.mjs";
import type { CppCuteBrowserHeaderPackMaterialization } from "./cpp_cute_browser_header_pack_materialization.mjs";
import type { CppCuteBrowserHeaderSourceExtraction } from "./cpp_cute_browser_header_source_extraction.mjs";

export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REVIEW_INPUT_SCHEMA:
"browsergrad.compiler.cpp-cute.header-distribution-review-input";

export class CppCuteBrowserHeaderDistributionReviewInputError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-DISTRIBUTION-REVIEW-INPUT";
  readonly path: string;
}

export interface CppCuteBrowserHeaderDistributionReviewInputReport {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REVIEW_INPUT_SCHEMA;
  readonly version: 2;
  readonly reviewInputId: string;
  readonly authority: "materialized-exact-header-distribution-review-input-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly headerSourcePlanId: string;
  readonly extractionId: string;
  readonly inventoryId: string;
  readonly cudaRedistributionIndexId: string;
  readonly outputPath: "assets/browsergrad-cpp-cute/license-inventory.json";
  readonly reviewInputSha256: string;
  readonly reviewInputByteLength: string;
  readonly totals: Readonly<{
    packCount: number;
    fileMapEntryCount: number;
    fileContentByteLength: string;
    packageNoticeCount: number;
    sourceEvidenceFileCount: number;
    sourceEvidenceByteLength: string;
    componentCount: number;
  }>;
  readonly unresolvedExternalReviews: readonly Readonly<{
    blockerId: string;
    requirement: string;
  }>[];
  readonly claims: Readonly<{
    exactPerDistributedFileComponentMapPrepared: true;
    exactMaterializedPackOutputsBound: true;
    exactPackageNoticeBytesBound: true;
    exactUpstreamLicenseEvidenceBound: true;
    exactCudaRedistributionIndexBound: true;
    allHeaderPackFilesCovered: true;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export function materializeCppCuteBrowserHeaderDistributionReviewInput(input: Readonly<{
  cudaRedistributionIndex: CppCuteBrowserCudaRedistributionIndexAdmission;
  extraction: CppCuteBrowserHeaderSourceExtraction;
  inventory: CppCuteBrowserHeaderPackInventory;
  materialization: CppCuteBrowserHeaderPackMaterialization;
  notices: CppCuteBrowserHeaderNoticeVerification;
}>): Promise<Readonly<CppCuteBrowserHeaderDistributionReviewInputReport>>;

export function requireCppCuteBrowserHeaderDistributionReviewInputAuthority(
  report: CppCuteBrowserHeaderDistributionReviewInputReport,
): void;

export function canonicalCppCuteBrowserHeaderDistributionReviewInputBytes(
  report: CppCuteBrowserHeaderDistributionReviewInputReport,
): Uint8Array;
