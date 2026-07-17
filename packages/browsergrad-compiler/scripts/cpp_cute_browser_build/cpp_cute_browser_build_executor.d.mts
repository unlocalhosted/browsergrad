import type {
  PreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import type {
  CppCuteClangWasmBuildRoots,
  CppCuteClangWasmMaterializedTools,
} from "./cpp_cute_browser_build_plan.mjs";

export interface PrepareCppCuteClangWasmBuildSourceInput {
  readonly lock: PreparedCppCuteBrowserBuildInputLock;
  readonly tools: CppCuteClangWasmMaterializedTools;
  readonly roots: CppCuteClangWasmBuildRoots;
  readonly extractorSourceInputRoot: string;
}

export interface CppCuteBrowserBuildExecutorOptions {
  readonly signal?: AbortSignal;
}

declare const preparedCppCuteClangWasmBuildSourceBrand: unique symbol;

/**
 * Opaque authority over one exact staged extractor source snapshot.
 * Its parent is current-user-owned and not group/other-writable; same-uid
 * actors must honor the executor's single-writer boundary.
 * It proves no build execution, output identity, reproducibility, or release.
 */
export interface PreparedCppCuteClangWasmBuildSource {
  readonly [preparedCppCuteClangWasmBuildSourceBrand]: true;
  readonly authority: "build-source-snapshot-only";
  readonly lockId: string;
  readonly sourceSetSha256: string;
  readonly fileCount: number;
  readonly totalByteLength: number;
  readonly stagedSourceRoot: string;
  readonly sourceVerified: true;
  readonly buildExecuted: false;
  readonly outputIdentityAuthorized: false;
  readonly reproducibilityVerified: false;
  readonly releaseReady: false;
}

declare const materializedCppCuteClangWasmSidecarBrand: unique symbol;

/**
 * Exact-copy observation. It is not build, provenance, or release authority.
 * The hard-link is the cancellation commit point. Output parents are
 * current-user-owned and not group/other-writable; same-uid actors must honor
 * the executor's single-writer boundary.
 */
export interface MaterializedCppCuteClangWasmSidecar {
  readonly [materializedCppCuteClangWasmSidecarBrand]: true;
  readonly authority: "wasm-sidecar-byte-materialization-observation-only";
  readonly lockId: string;
  readonly sourceSetSha256: string;
  readonly generatedWasmSha256: string;
  readonly distributedWasmSha256: string;
  readonly wasmByteLength: number;
  readonly distributedWasmPath: string;
  readonly sidecarBytesMaterialized: true;
  readonly webAssemblyValidated: false;
  readonly abiConformanceVerified: false;
  readonly sourceVerified: true;
  readonly buildExecuted: false;
  readonly outputIdentityAuthorized: false;
  readonly reproducibilityVerified: false;
  readonly releaseReady: false;
  readonly factoryModuleDistributed: false;
}

export type CppCuteBrowserBuildExecutorErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-IO"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-UNVERIFIED";

export class CppCuteBrowserBuildExecutorError extends Error {
  readonly code: CppCuteBrowserBuildExecutorErrorCode;
  readonly path: string;
}

export function prepareCppCuteClangWasmBuildSource(
  input: PrepareCppCuteClangWasmBuildSourceInput,
  options?: CppCuteBrowserBuildExecutorOptions,
): Promise<PreparedCppCuteClangWasmBuildSource>;

export function materializeCppCuteClangWasmSidecar(
  prepared: PreparedCppCuteClangWasmBuildSource,
  options?: CppCuteBrowserBuildExecutorOptions,
): Promise<MaterializedCppCuteClangWasmSidecar>;
