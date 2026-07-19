export const CPP_CUTE_BROWSER_CUDA_REDISTRIBUTION_INDEX_SCHEMA:
"browsergrad.compiler.cpp-cute.cuda-redistribution-index-admission";

export class CppCuteBrowserCudaRedistributionIndexError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-CUDA-REDISTRIBUTION-INDEX";
  readonly path: string;
}

export interface CppCuteBrowserCudaRedistributionIndexComponentExpectation {
  readonly sourceId: string;
  readonly componentKey: string;
  readonly version: string;
  readonly relativePath: string;
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserCudaRedistributionIndexExpected {
  readonly headerSourcePlanId?: string;
  readonly url: string;
  readonly releaseLabel: string;
  readonly sha256: string;
  readonly byteLength: string;
  readonly components: readonly CppCuteBrowserCudaRedistributionIndexComponentExpectation[];
}

export interface CppCuteBrowserCudaRedistributionIndexAdmission {
  readonly schema: typeof CPP_CUTE_BROWSER_CUDA_REDISTRIBUTION_INDEX_SCHEMA;
  readonly version: 1;
  readonly indexId: string;
  readonly authority:
    | "caller-expected-cuda-redistribution-index-inspection-only"
    | "exact-current-header-source-plan-cuda-index-admission-only";
  readonly headerSourcePlanId?: string;
  readonly sourceUrl: string;
  readonly releaseLabel: string;
  readonly releaseProduct: "cuda";
  readonly releaseDate: string;
  readonly indexSha256: string;
  readonly indexByteLength: string;
  readonly components: readonly Readonly<{
    ordinal: number;
    sourceId: string;
    componentKey: string;
    name: string;
    version: string;
    license: "CUDA Toolkit";
    licensePath: string;
    platform: "linux-x86_64";
    relativePath: string;
    archiveSha256: string;
    archiveByteLength: string;
  }>[];
  readonly claims: Readonly<{
    exactIndexBytesVerified: true;
    selectedComponentMetadataVerified: true;
    exactCurrentHeaderSourcePlanBound: boolean;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    releaseReady: false;
  }>;
}

export function inspectCppCuteBrowserCudaRedistributionIndexBytes(input: Readonly<{
  bytes: Uint8Array;
  expected: CppCuteBrowserCudaRedistributionIndexExpected;
}>): Readonly<CppCuteBrowserCudaRedistributionIndexAdmission>;

export function admitCppCuteBrowserCudaRedistributionIndex(input: Readonly<{
  indexPath: string;
}>): Promise<Readonly<CppCuteBrowserCudaRedistributionIndexAdmission>>;

export function requireCppCuteBrowserCudaRedistributionIndexAuthority(
  admission: CppCuteBrowserCudaRedistributionIndexAdmission,
): void;
