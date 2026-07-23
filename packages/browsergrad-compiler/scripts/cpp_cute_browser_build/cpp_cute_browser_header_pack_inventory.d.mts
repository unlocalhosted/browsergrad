import type {
  CppCuteBrowserHeaderSourceExtraction,
} from "./cpp_cute_browser_header_source_extraction.mjs";

export const CPP_CUTE_BROWSER_HEADER_PACK_INVENTORY_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-pack-source-inventory";

export class CppCuteBrowserHeaderPackInventoryError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVENTORY";
  readonly path: string;
}

export interface CppCuteBrowserHeaderPackInventorySourceInput {
  readonly sourceRoot: string;
  readonly virtualPrefix: string;
  readonly licenseComponentIds: readonly string[];
}

export interface CppCuteBrowserHeaderPackInventoryPackInput {
  readonly includeRootId: string;
  readonly sources: readonly CppCuteBrowserHeaderPackInventorySourceInput[];
}

export interface CppCuteBrowserHeaderPackInventoryInput {
  readonly packs: readonly CppCuteBrowserHeaderPackInventoryPackInput[];
}

export interface CppCuteBrowserHeaderPackInventoryFile {
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: string;
  readonly licenseComponentIds: readonly string[];
}

export interface CppCuteBrowserHeaderPackInventoryPack {
  readonly includeRootId: string;
  readonly intendedAsset: string;
  readonly outputRole: string;
  readonly outputPath: string;
  readonly contentSetSha256: string;
  readonly fileCount: number;
  readonly fileContentByteLength: string;
  readonly files: readonly CppCuteBrowserHeaderPackInventoryFile[];
}

export interface CppCuteBrowserHeaderPackInventory {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_PACK_INVENTORY_SCHEMA;
  readonly version: 1;
  readonly inventoryId: string;
  readonly authority:
    | "local-source-tree-inventory-only"
    | "exact-extraction-source-inventory-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerSourceExtractionId?: string;
  readonly packs: readonly CppCuteBrowserHeaderPackInventoryPack[];
  readonly totals: Readonly<{
    packCount: number;
    sourceCount: number;
    fileCount: number;
    fileContentByteLength: string;
  }>;
  readonly claims: Readonly<{
    exactReadableSourceTreesVerified: true;
    buildInputLockBound: true;
    networkAccessed: false;
    archiveProvenanceVerified: false;
    generatedClangResourceHeadersComplete: boolean;
    configuredLibcxxHeaderComplete: boolean;
    licenseReviewComplete: false;
    headerPackSelectionPrepared: false;
    headerPacksAssembled: false;
    buildExecuted: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserHeaderPackInventoryArguments {
  readonly inputPath: string;
  readonly outputPath: string;
}

export interface CppCuteBrowserHeaderPackInventoryAuthoringReport {
  readonly outputPath: string;
  readonly inventoryId: string;
  readonly inventorySha256: string;
  readonly inventoryByteLength: number;
  readonly packCount: number;
  readonly fileCount: number;
  readonly releaseReady: false;
}

export function inventoryCppCuteBrowserHeaderPackSources(
  input: CppCuteBrowserHeaderPackInventoryInput,
): Promise<Readonly<CppCuteBrowserHeaderPackInventory>>;

export function inventoryCppCuteBrowserExtractedHeaderSources(
  extraction: CppCuteBrowserHeaderSourceExtraction,
): Promise<Readonly<CppCuteBrowserHeaderPackInventory>>;

export function canonicalCppCuteBrowserHeaderPackInventoryBytes(
  inventory: CppCuteBrowserHeaderPackInventory,
): Uint8Array;

export function requireCppCuteBrowserHeaderPackInventorySourceAuthority(
  inventory: CppCuteBrowserHeaderPackInventory,
): void;

export function readCppCuteBrowserHeaderPackInventorySpecification(
  inputPath: string,
): Promise<unknown>;

export function copyCppCuteBrowserHeaderPackInventorySourceFile(
  inventory: CppCuteBrowserHeaderPackInventory,
  includeRootId: string,
  virtualPath: string,
): Promise<Uint8Array>;

export function parseCppCuteBrowserHeaderPackInventoryArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserHeaderPackInventoryArguments>;

export function authorCppCuteBrowserHeaderPackInventory(
  input: CppCuteBrowserHeaderPackInventoryArguments,
): Promise<Readonly<CppCuteBrowserHeaderPackInventoryAuthoringReport>>;
