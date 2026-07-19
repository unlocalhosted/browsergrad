import type {
  CppCuteBrowserBuildInputLockBodyV1,
} from "../../dist/cpp_cute_browser_build_lock.js";

export interface CppCuteBrowserToolchainCacheInputs {
  readonly schema: "browsergrad.compiler.cpp-cute.clang-wasm-toolchain-cache-inputs";
  readonly version: 1;
  readonly llvmSource: Readonly<Record<string, unknown>>;
  readonly builder: Readonly<Record<string, unknown>>;
  readonly recipe: Readonly<Record<string, unknown>>;
  readonly selectedClangLibraries: readonly string[];
}

export interface CppCuteBrowserToolchainCacheProjection {
  readonly schema: "browsergrad.compiler.cpp-cute.clang-wasm-toolchain-cache-projection";
  readonly version: 1;
  readonly authority: "untrusted-diagnostic-cache-selection-only";
  readonly cacheKey: string;
  readonly compatibleLegacyCacheKey: string;
  readonly inputs: CppCuteBrowserToolchainCacheInputs;
  readonly claims: Readonly<{
    cacheContentsTrusted: false;
    cleanBuild: false;
    buildExecuted: false;
    reproducibilityVerified: false;
    releaseReady: false;
  }>;
}

export function selectCppCuteBrowserToolchainCacheInputs(
  body: CppCuteBrowserBuildInputLockBodyV1,
): CppCuteBrowserToolchainCacheInputs;

export function deriveCppCuteBrowserToolchainCacheKey(
  inputs: CppCuteBrowserToolchainCacheInputs,
): Promise<string>;

export function projectCppCuteBrowserToolchainCache():
  Promise<CppCuteBrowserToolchainCacheProjection>;
