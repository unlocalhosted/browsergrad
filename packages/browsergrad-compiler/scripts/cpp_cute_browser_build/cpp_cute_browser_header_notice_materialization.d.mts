import type {
  CppCuteBrowserHeaderDistributionReviewInputReport,
} from "./cpp_cute_browser_header_distribution_review_input.mjs";
import type {
  CppCuteBrowserHeaderNoticeVerification,
} from "./cpp_cute_browser_header_notice_verification.mjs";
import type {
  CppCuteBrowserHeaderPackMaterialization,
} from "./cpp_cute_browser_header_pack_materialization.mjs";

export const CPP_CUTE_BROWSER_HEADER_NOTICE_MATERIALIZATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-notice-materialization";
export const CPP_CUTE_BROWSER_AGGREGATE_NOTICE_FORMAT:
"browsergrad.compiler.cpp-cute.distribution-notices.v1";

export class CppCuteBrowserHeaderNoticeMaterializationError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-NOTICE-MATERIALIZATION";
  readonly path: string;
}

export interface CppCuteBrowserHeaderNoticeMaterializationReport {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_NOTICE_MATERIALIZATION_SCHEMA;
  readonly version: 2;
  readonly noticeMaterializationId: string;
  readonly authority: "exact-private-distribution-notice-materialization-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly distributionReviewInputId: string;
  readonly inventoryId: string;
  readonly outputFileMaterializationId: string;
  readonly outputRoot: string;
  readonly componentOutputs: readonly Readonly<{
    ordinal: number;
    componentId: string;
    outputPath: string;
    sha256: string;
    byteLength: string;
  }>[];
  readonly aggregateOutput: Readonly<{
    outputPath: "assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt";
    sha256: string;
    byteLength: string;
    format: typeof CPP_CUTE_BROWSER_AGGREGATE_NOTICE_FORMAT;
    componentCount: number;
    thirdPartyComponentCount: number;
  }>;
  readonly totals: Readonly<{
    componentFileCount: number;
    componentByteLength: string;
    aggregateFileCount: 1;
    aggregateByteLength: string;
    materializedFileCount: number;
    materializedByteLength: string;
  }>;
  readonly unresolvedNotices: readonly Readonly<{
    componentId: string;
    intendedAsset: string;
    reasonCode: string;
    disposition: "blocks-release";
  }>[];
  readonly claims: Readonly<{
    exactApprovedDistributionNoticeBytesMaterialized: true;
    exactComponentLicenseOutputSetMaterialized: true;
    deterministicAggregateNoticeMaterialized: true;
    exactFinalDistributionOutputTreeVerified: true;
    allHeaderNoticesResolved: false;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export function materializeCppCuteBrowserHeaderDistributionNotices(input: Readonly<{
  distributionReviewInput: CppCuteBrowserHeaderDistributionReviewInputReport;
  materialization: CppCuteBrowserHeaderPackMaterialization;
  notices: CppCuteBrowserHeaderNoticeVerification;
}>): Promise<Readonly<CppCuteBrowserHeaderNoticeMaterializationReport>>;

export function requireCppCuteBrowserHeaderNoticeMaterializationAuthority(
  report: CppCuteBrowserHeaderNoticeMaterializationReport,
): void;

export function canonicalCppCuteBrowserHeaderNoticeMaterializationBytes(
  report: CppCuteBrowserHeaderNoticeMaterializationReport,
): Uint8Array;
