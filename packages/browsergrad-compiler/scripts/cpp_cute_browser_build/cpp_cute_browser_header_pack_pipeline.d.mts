import type {
  CppCuteBrowserHeaderSourceBlocker,
} from "./cpp_cute_browser_header_source_plan.mjs";
import type {
  CppCuteBrowserHeaderPackMaterializationOutput,
} from "./cpp_cute_browser_header_pack_materialization.mjs";
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
  readonly sourceOutputRoot: string;
  readonly packOutputRoot: string;
}

export interface CppCuteBrowserHeaderPackPipeline {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_PACK_PIPELINE_SCHEMA;
  readonly version: 1;
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
  }>;
  readonly sourceTotals: Readonly<{
    archiveCount: number;
    selectedSubtreeCount: number;
    fileCount: number;
    fileContentByteLength: string;
  }>;
  readonly inventoryTotals: Readonly<{
    packCount: number;
    sourceCount: number;
    fileCount: number;
    fileContentByteLength: string;
  }>;
  readonly outputs: readonly CppCuteBrowserHeaderPackMaterializationOutput[];
  readonly totalPackByteLength: string;
  readonly unresolvedBlockers: readonly CppCuteBrowserHeaderSourceBlocker[];
  readonly claims: Readonly<{
    exactCurrentHeaderSourcePlanArchiveBytesVerified: true;
    collisionFreePortableStorageMaterialized: true;
    exactExtractedSourceBytesInventoried: true;
    canonicalVfsPacksIndependentlyInspected: true;
    allFiveSelectedSourcePacksMaterialized: true;
    exactSelectedSourceSubtreesComplete: false;
    hostToolImplementationAttested: false;
    generatedClangResourceHeadersComplete: true;
    externalDistributedFileLicenseMapReviewed: false;
    licenseReviewComplete: false;
    headerUniverseComplete: false;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export function materializeCppCuteBrowserHeaderPacksFromSourceArchives(
  input: CppCuteBrowserHeaderPackPipelineInput,
): Promise<Readonly<CppCuteBrowserHeaderPackPipeline>>;

export function parseCppCuteBrowserHeaderPackPipelineArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserHeaderPackPipelineInput>;

export function createCppCuteBrowserPrivatePackOutputRoot(
  outputRoot: string,
): Promise<Readonly<{ dev: bigint; ino: bigint }>>;
