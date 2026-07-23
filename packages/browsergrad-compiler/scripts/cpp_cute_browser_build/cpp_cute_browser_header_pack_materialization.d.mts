import type {
  CppCuteBrowserHeaderPackInventory,
} from "./cpp_cute_browser_header_pack_inventory.mjs";

export const CPP_CUTE_BROWSER_HEADER_PACK_MATERIALIZATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-header-pack-materialization";

export class CppCuteBrowserHeaderPackMaterializationError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MATERIALIZATION";
  readonly path: string;
}

export interface CppCuteBrowserHeaderPackMaterializationInput {
  readonly inventory: CppCuteBrowserHeaderPackInventory;
  readonly outputRoot: string;
}

export interface CppCuteBrowserHeaderPackMaterializationOutput {
  readonly ordinal: number;
  readonly includeRootId: string;
  readonly intendedAsset: string;
  readonly outputRole: string;
  readonly outputPath: string;
  readonly packSha256: string;
  readonly packByteLength: string;
  readonly fileContentByteLength: string;
  readonly contentSetSha256: string;
  readonly fileCount: number;
}

export interface CppCuteBrowserHeaderPackMaterialization {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_PACK_MATERIALIZATION_SCHEMA;
  readonly version: 2;
  readonly authority: "deterministic-vfs-pack-materialization-only";
  readonly inventoryId: string;
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly outputRoot: string;
  readonly outputs: readonly CppCuteBrowserHeaderPackMaterializationOutput[];
  readonly totalPackByteLength: string;
  readonly claims: Readonly<{
    exactSourceBytesReverified: true;
    canonicalVfsPacksIndependentlyInspected: true;
    networkAccessed: false;
    licenseReviewComplete: false;
    assetManifestBound: false;
    buildExecuted: false;
    reproducibilityObserved: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserHeaderPackMaterializationArguments {
  readonly inputPath: string;
  readonly outputRoot: string;
}

export function materializeCppCuteBrowserHeaderPacks(
  input: CppCuteBrowserHeaderPackMaterializationInput,
): Promise<Readonly<CppCuteBrowserHeaderPackMaterialization>>;

export function requireCppCuteBrowserHeaderPackMaterializationAuthority(
  materialization: CppCuteBrowserHeaderPackMaterialization,
): void;

export function canonicalCppCuteBrowserHeaderPackMaterializationBytes(
  materialization: CppCuteBrowserHeaderPackMaterialization,
): Uint8Array;

export function parseCppCuteBrowserHeaderPackMaterializationArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserHeaderPackMaterializationArguments>;

export function materializeCppCuteBrowserHeaderPacksFromSpecification(
  input: CppCuteBrowserHeaderPackMaterializationArguments,
): Promise<Readonly<CppCuteBrowserHeaderPackMaterialization>>;
