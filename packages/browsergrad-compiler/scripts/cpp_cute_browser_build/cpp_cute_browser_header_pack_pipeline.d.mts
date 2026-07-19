import type {
  CppCuteBrowserHeaderSourceBlocker,
} from "./cpp_cute_browser_header_source_plan.mjs";
import type {
  CppCuteBrowserBsdtarToolAdmission,
} from "./cpp_cute_browser_archive_normalization.mjs";
import type {
  CppCuteBrowserHeaderPackMaterializationOutput,
} from "./cpp_cute_browser_header_pack_materialization.mjs";
import type {
  CppCuteBrowserHeaderDistributionReviewInputReport,
} from "./cpp_cute_browser_header_distribution_review_input.mjs";
import type {
  CppCuteBrowserHeaderNoticeMaterializationReport,
} from "./cpp_cute_browser_header_notice_materialization.mjs";
import type {
  CppCuteBrowserSourceArchiveInput,
} from "./cpp_cute_browser_source_archive_admission.mjs";

export const CPP_CUTE_BROWSER_HEADER_PACK_PIPELINE_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-pack-pipeline";

export class CppCuteBrowserHeaderPackPipelineError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-PIPELINE";
  readonly path: string;
}

export interface CppCuteBrowserHeaderPackPipelineInput {
  readonly archives: readonly CppCuteBrowserSourceArchiveInput[];
  readonly bsdtarPath: string;
  readonly cudaRedistributionIndexPath: string;
  readonly sourceOutputRoot: string;
  readonly packOutputRoot: string;
}

export interface CppCuteBrowserHeaderPackPipeline {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_PACK_PIPELINE_SCHEMA;
  readonly version: 4;
  readonly pipelineId: string;
  readonly authority: "exact-source-host-tool-vfs-pack-pipeline-observation-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerSourcePlanId: string;
  readonly archiveAdmissionId: string;
  readonly extractionId: string;
  readonly inventoryId: string;
  readonly bsdtarTool: Readonly<{
    toolAdmissionId: string;
    executableSha256: string;
    executableByteLength: string;
    observedVersion: string;
    packageToolIdentityPinned: true;
    nodeZstdRuntime: NonNullable<CppCuteBrowserBsdtarToolAdmission["nodeZstdRuntime"]>;
  }>;
  readonly sourceTotals: Readonly<{
    archiveCount: number;
    selectedSubtreeCount: number;
    fileCount: number;
    fileContentByteLength: string;
    licenseEvidenceFileCount: number;
    licenseEvidenceByteLength: string;
  }>;
  readonly licenseEvidence: readonly Readonly<{
    sourceId: string;
    evidenceId: string;
    archivePath: string;
    componentId: string;
    evidenceRole: "upstream-license-text" | "source-package-copyright";
    sha256: string;
    byteLength: string;
    sourceTreeId: string;
  }>[];
  readonly inventoryTotals: Readonly<{
    packCount: number;
    sourceCount: number;
    fileCount: number;
    fileContentByteLength: string;
  }>;
  readonly outputs: readonly CppCuteBrowserHeaderPackMaterializationOutput[];
  readonly totalPackByteLength: string;
  readonly distributionReviewInput: CppCuteBrowserHeaderDistributionReviewInputReport;
  readonly noticeMaterialization: CppCuteBrowserHeaderNoticeMaterializationReport;
  readonly unresolvedBlockers: readonly CppCuteBrowserHeaderSourceBlocker[];
  readonly claims: Readonly<{
    exactCurrentHeaderSourcePlanArchiveBytesVerified: true;
    collisionFreePortableStorageMaterialized: true;
    exactExtractedSourceBytesInventoried: true;
    canonicalVfsPacksIndependentlyInspected: true;
    allFiveSelectedSourcePacksMaterialized: true;
    exactSelectedSourceSubtreesComplete: true;
    hostToolImplementationAttested: false;
    hostToolPackageIdentityPinned: true;
    nodeZstdDecompressorPackageIdentityPinned: true;
    generatedClangResourceHeadersComplete: true;
    exactUpstreamLicenseEvidenceExtracted: true;
    exactCudaRedistributionIndexBound: true;
    exactPerDistributedFileComponentMapPrepared: true;
    deterministicLicenseInventoryMaterialized: true;
    exactApprovedDistributionNoticeBytesMaterialized: true;
    deterministicAggregateNoticeMaterialized: true;
    exactFinalDistributionOutputTreeVerified: true;
    allHeaderNoticesResolved: false;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    headerUniverseComplete: true;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export function materializeCppCuteBrowserHeaderPacksFromSourceArchives(
  input: CppCuteBrowserHeaderPackPipelineInput,
): Promise<Readonly<CppCuteBrowserHeaderPackPipeline>>;

export function requireCppCuteBrowserHeaderPackPipelineAuthority(
  report: CppCuteBrowserHeaderPackPipeline,
): void;

export function parseCppCuteBrowserHeaderPackPipelineArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserHeaderPackPipelineInput>;

export function createCppCuteBrowserPrivatePackOutputRoot(
  outputRoot: string,
): Promise<Readonly<{ dev: bigint; ino: bigint }>>;
