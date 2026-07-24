export interface CppCuteBrowserRealCompileRunnerInput {
  readonly wasmPath: string;
  readonly packRoot: string;
  readonly evidenceOutput?: string;
  readonly preflightOnly?: boolean;
  readonly requireCompiled?: boolean;
  readonly allowUntrustedDiagnosticWasm?: boolean;
  readonly caseId?:
    | "rank2"
    | "rank3"
    | "strided-slice"
    | "broadcast"
    | "i32-rank2"
    | "u32-broadcast";
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
  readonly version: 3;
  readonly authority: "local-exact-byte-preflight-only";
  readonly caseId:
    | "rank2"
    | "rank3"
    | "strided-slice"
    | "broadcast"
    | "i32-rank2"
    | "u32-broadcast";
  readonly wasmPath: string;
  readonly packRoot: string;
  readonly assets: readonly CppCuteBrowserRealCompileAsset[];
  readonly wasmAuthority:
    | "package-pinned-two-clean-build-output"
    | "untrusted-diagnostic-local-byte-observation-only";
  readonly pinnedReproducibleWasmMatched: boolean;
  readonly untrustedDiagnosticWasm: boolean;
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

export function persistCppCuteBrowserRealCompileEvidence(
  outputPath: string,
  evidence: unknown,
): Promise<Readonly<{
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: number;
}>>;

export function runCppCuteBrowserRealCompile(
  argv?: readonly string[],
): Promise<unknown>;
