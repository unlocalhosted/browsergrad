export interface CppCuteBrowserRealCompileRunnerInput {
  readonly wasmPath: string;
  readonly packRoot: string;
  readonly evidenceOutput?: string;
  readonly preflightOnly?: boolean;
  readonly requireCompiled?: boolean;
}

export interface CppCuteBrowserRealCompileAsset {
  readonly assetId: string;
  readonly path: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: number;
}

export interface CppCuteBrowserRealCompilePreflight {
  readonly schema: "browsergrad.compiler.cpp-cute.browser-real-compile-inputs";
  readonly version: 1;
  readonly authority: "local-exact-byte-preflight-only";
  readonly wasmPath: string;
  readonly packRoot: string;
  readonly assets: readonly CppCuteBrowserRealCompileAsset[];
  readonly pinnedReproducibleWasmMatched: true;
  readonly headerDistributionLicenseApproved: false;
  readonly producerTrusted: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export class CppCuteBrowserRealCompileRunnerError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER";
  readonly path: string;
}

export function preflightCppCuteBrowserRealCompileInputs(
  input: CppCuteBrowserRealCompileRunnerInput,
): Promise<CppCuteBrowserRealCompilePreflight>;

export function runCppCuteBrowserRealCompile(
  argv?: readonly string[],
): Promise<unknown>;
