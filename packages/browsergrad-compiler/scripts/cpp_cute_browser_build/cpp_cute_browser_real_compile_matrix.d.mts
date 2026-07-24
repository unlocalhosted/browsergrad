export interface CppCuteBrowserRealCompileMatrixOptions {
  readonly wasmPath: string;
  readonly packRoot: string;
  readonly evidenceOutput: string;
  readonly requireCompiled: boolean;
  readonly allowUntrustedDiagnosticWasm: boolean;
}

export class CppCuteBrowserRealCompileMatrixError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-MATRIX";
  readonly path: string;
}

export function parseCppCuteBrowserRealCompileMatrixArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserRealCompileMatrixOptions>;

export function prepareCppCuteBrowserRealCompileMatrix(
  observations: readonly unknown[],
): Readonly<Record<string, unknown>>;

export function runCppCuteBrowserRealCompileMatrix(
  argv?: readonly string[],
): Promise<Readonly<Record<string, unknown>>>;
