export const CPP_CUTE_BROWSER_HEADER_NOTICE_VERIFICATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-notice-verification";

export class CppCuteBrowserHeaderNoticeVerificationError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-NOTICE-VERIFICATION";
  readonly path: string;
}

export interface CppCuteBrowserHeaderNoticeVerificationInput {
  readonly resourceRoot?: string;
}

export interface CppCuteBrowserHeaderNoticeVerification {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_NOTICE_VERIFICATION_SCHEMA;
  readonly version: 2;
  readonly authority: "approved-header-notice-byte-verification-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly notices: readonly Readonly<{
    componentId: string;
    licenseExpression: string;
    upstreamSourcePath: string;
    noticeOutputPath: string;
    packageResourceFileName: string;
    noticeSha256: string;
    noticeByteLength: string;
    appliesTo: readonly string[];
  }>[];
  readonly unresolvedNotices: readonly Readonly<{
    componentId: string;
    intendedAsset: string;
    reasonCode: string;
    disposition: "blocks-release";
  }>[];
  readonly claims: Readonly<{
    exactApprovedDistributionNoticeBytesVerified: true;
    unresolvedHeaderNoticeComponentCount: number;
    allHeaderNoticesResolved: false;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export function verifyCppCuteBrowserHeaderPackNotices(
  input?: CppCuteBrowserHeaderNoticeVerificationInput,
): Promise<Readonly<CppCuteBrowserHeaderNoticeVerification>>;

export function requireCppCuteBrowserHeaderNoticeVerificationAuthority(
  evidence: CppCuteBrowserHeaderNoticeVerification,
): void;

export function copyCppCuteBrowserVerifiedHeaderNoticeBytes(
  evidence: CppCuteBrowserHeaderNoticeVerification,
  componentId: string,
): Uint8Array;

export function canonicalCppCuteBrowserHeaderNoticeVerificationBytes(
  evidence: CppCuteBrowserHeaderNoticeVerification,
): Uint8Array;
