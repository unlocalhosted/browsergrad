export const CPP_CUTE_BROWSER_SELECTED_TAR_MATERIALIZATION_SCHEMA:
"browsergrad.compiler.cpp-cute.selected-tar-materialization";

export class CppCuteBrowserSelectedTarStreamError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-SELECTED-TAR-STREAM";
  readonly path: string;
}

export interface CppCuteBrowserSelectedTarSelection {
  readonly selectionId: string;
  readonly archiveSubtree: string;
  readonly outputSubdirectory: string;
}

export interface CppCuteBrowserSelectedTarFile {
  readonly relativePath: string;
  readonly contentSha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserSelectedTarSelectionResult
  extends CppCuteBrowserSelectedTarSelection {
  readonly sourceTreeId: string;
  readonly fileCount: number;
  readonly fileContentByteLength: string;
  readonly files: readonly CppCuteBrowserSelectedTarFile[];
}

export interface CppCuteBrowserSelectedTarMaterialization {
  readonly schema: typeof CPP_CUTE_BROWSER_SELECTED_TAR_MATERIALIZATION_SCHEMA;
  readonly version: 1;
  readonly materializationId: string;
  readonly authority: "caller-selected-normalized-tar-materialization-only";
  readonly selections: readonly CppCuteBrowserSelectedTarSelectionResult[];
  readonly totals: Readonly<{
    selectionCount: number;
    fileCount: number;
    fileContentByteLength: string;
    consumedTarByteLength: string;
  }>;
  readonly claims: Readonly<{
    strictNormalizedTarParsed: true;
    onlyRegularFileContentsMaterialized: true;
    collisionFreePortableStorageMaterialized: true;
    hierarchicalSourceTreesMaterialized: false;
    allSelectedStreamFilesMaterialized: true;
    callerSelectedSubtreesComplete: false;
    archiveIdentityVerified: false;
    decompressorVerified: false;
    headerSourcePlanBound: false;
    generatedClangResourceHeadersComplete: false;
    licenseReviewComplete: false;
    headerPacksAssembled: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserSelectedTarMaterializationRoot {
  readonly selectionId: string;
  readonly storageRoot: string;
  readonly sourceTreeId: string;
}

export function materializeCppCuteBrowserSelectedTarStream(input: Readonly<{
  chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>;
  outputRoot: string;
  selections: readonly CppCuteBrowserSelectedTarSelection[];
}>): Promise<Readonly<CppCuteBrowserSelectedTarMaterialization>>;

export function requireCppCuteBrowserSelectedTarMaterializationAuthority(
  manifest: CppCuteBrowserSelectedTarMaterialization,
): void;

export function cppCuteBrowserSelectedTarMaterializationRoots(
  manifest: CppCuteBrowserSelectedTarMaterialization,
): readonly Readonly<CppCuteBrowserSelectedTarMaterializationRoot>[];

export function copyCppCuteBrowserSelectedTarMaterializationFile(
  manifest: CppCuteBrowserSelectedTarMaterialization,
  selectionId: string,
  relativePath: string,
): Promise<Uint8Array>;
