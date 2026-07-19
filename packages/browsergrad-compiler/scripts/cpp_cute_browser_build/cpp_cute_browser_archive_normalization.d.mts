import type {
  CppCuteBrowserSelectedTarMaterializationRoot,
  CppCuteBrowserSelectedTarSelection,
  CppCuteBrowserSelectedTarSelectionResult,
} from "./cpp_cute_browser_selected_tar_stream.mjs";

export const CPP_CUTE_BROWSER_BSDTAR_TOOL_ADMISSION_SCHEMA:
"browsergrad.compiler.cpp-cute.bsdtar-tool-admission";
export const CPP_CUTE_BROWSER_ARCHIVE_NORMALIZATION_SCHEMA:
"browsergrad.compiler.cpp-cute.archive-normalization";

export class CppCuteBrowserArchiveNormalizationError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-ARCHIVE-NORMALIZATION";
  readonly path: string;
}

export interface CppCuteBrowserBsdtarToolAdmission {
  readonly schema: typeof CPP_CUTE_BROWSER_BSDTAR_TOOL_ADMISSION_SCHEMA;
  readonly version: 1;
  readonly toolAdmissionId: string;
  readonly authority:
    | "caller-selected-host-bsdtar-observation-only"
    | "package-pinned-archive-normalization-environment";
  readonly executableSha256: string;
  readonly executableByteLength: string;
  readonly observedVersion: string;
  readonly nodeZstdRuntime?: Readonly<{
    platform: "darwin";
    architecture: "arm64";
    runtimeVersion: "v25.9.0";
    executableSha256: string;
    executableByteLength: string;
    zstdVersion: "1.5.7";
    execArgv: readonly [];
    nodeOptions: "absent";
  }>;
  readonly claims: Readonly<{
    executableRegularFileObserved: true;
    executableBytesHashed: true;
    closedEnvironmentVersionObserved: true;
    toolImplementationAttested: false;
    packageToolIdentityPinned: boolean;
    nodeZstdRuntimeIdentityPinned?: true;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserArchiveNormalizationProcess {
  readonly stageId: "deb-data-member-read" | "selected-pax-normalization";
  readonly stderrSha256: string;
  readonly stderrByteLength: string;
}

export interface CppCuteBrowserArchiveNormalization {
  readonly schema: typeof CPP_CUTE_BROWSER_ARCHIVE_NORMALIZATION_SCHEMA;
  readonly version: 2;
  readonly normalizationId: string;
  readonly authority: "caller-expected-host-tool-archive-normalization-only";
  readonly archiveFormat: "tar.gz" | "tar.xz" | "deb-data-tar-zstd";
  readonly observedArchiveSha256: string;
  readonly observedArchiveByteLength: string;
  readonly tool: Readonly<{
    toolAdmissionId: string;
    executableSha256: string;
    executableByteLength: string;
    observedVersion: string;
  }>;
  readonly selections: readonly CppCuteBrowserSelectedTarSelectionResult[];
  readonly totals: Readonly<{
    selectionCount: number;
    fileCount: number;
    fileContentByteLength: string;
    consumedTarByteLength: string;
  }>;
  readonly intermediate?: Readonly<{
    memberName: "data.tar.zst";
    memberSha256: string;
    memberByteLength: string;
    decompressedTarSha256: string;
    decompressedTarByteLength: string;
    decompressor: "node:zlib.createZstdDecompress";
    runtimeVersion: string;
    pinnedRuntime?: NonNullable<CppCuteBrowserBsdtarToolAdmission["nodeZstdRuntime"]>;
  }>;
  readonly processes: readonly CppCuteBrowserArchiveNormalizationProcess[];
  readonly claims: Readonly<{
    observedArchiveBytesHashed: true;
    expectedArchiveIdentityBound: false;
    hostToolExecutableBytesHashed: true;
    hostToolImplementationAttested: false;
    hostToolPackageIdentityPinned: boolean;
    nodeZstdDecompressorObserved: boolean;
    decompressorImplementationAttested: false;
    nodeZstdDecompressorPackageIdentityPinned: boolean;
    strictNormalizedTarParsed: true;
    collisionFreePortableStorageMaterialized: true;
    hierarchicalSourceTreesMaterialized: false;
    allSelectedStreamFilesMaterialized: true;
    callerSelectedPathsComplete: false;
    headerSourcePlanBound: false;
    licenseReviewComplete: false;
    releaseReady: false;
  }>;
}

export function admitCppCuteBrowserBsdtarTool(input: Readonly<{
  executablePath: string;
}>): Promise<Readonly<CppCuteBrowserBsdtarToolAdmission>>;

export function admitPinnedCppCuteBrowserArchiveNormalizationEnvironment(
  input: Readonly<{ executablePath: string }>,
): Promise<Readonly<CppCuteBrowserBsdtarToolAdmission>>;

export function requireCppCuteBrowserBsdtarToolAuthority(
  admission: CppCuteBrowserBsdtarToolAdmission,
): void;

export function materializeCppCuteBrowserNormalizedArchive(input: Readonly<{
  archiveFormat: "tar.gz" | "tar.xz" | "deb-data-tar-zstd";
  archivePath: string;
  outputRoot: string;
  selections: readonly CppCuteBrowserSelectedTarSelection[];
  tool: CppCuteBrowserBsdtarToolAdmission;
}>): Promise<Readonly<CppCuteBrowserArchiveNormalization>>;

export function requireCppCuteBrowserArchiveNormalizationAuthority(
  normalization: CppCuteBrowserArchiveNormalization,
): void;

export function cppCuteBrowserArchiveNormalizationRoots(
  normalization: CppCuteBrowserArchiveNormalization,
): readonly Readonly<CppCuteBrowserSelectedTarMaterializationRoot>[];

export function copyCppCuteBrowserArchiveNormalizationFile(
  normalization: CppCuteBrowserArchiveNormalization,
  selectionId: string,
  relativePath: string,
): Promise<Uint8Array>;
