import type {
  CppCuteBrowserWasmInspectionReport,
} from "../../dist/cpp_cute_browser_wasm_inspection.js";

export interface ReviewCppCuteBrowserWasmFileInput {
  readonly wasmPath: string;
}

export type CppCuteBrowserWasmReviewErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-CONFLICT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-IO"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-MISMATCH";

export class CppCuteBrowserWasmReviewError extends Error {
  readonly code: CppCuteBrowserWasmReviewErrorCode;
  readonly path: string;
}

/**
 * Reviews one exact build-produced module against the package-owned runtime
 * ABI. A returned mismatch report is discovery evidence, not conformance.
 */
export function reviewCppCuteBrowserWasmFile(
  input: ReviewCppCuteBrowserWasmFileInput,
): Promise<CppCuteBrowserWasmInspectionReport>;

export function writeCppCuteBrowserWasmReviewReport(
  outputPath: string,
  report: CppCuteBrowserWasmInspectionReport,
): Promise<Readonly<{
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: number;
  readonly exactInterfaceConformance: boolean;
  readonly mismatchCount: number;
}>>;

export function requireExactCppCuteBrowserWasmInterface(
  report: CppCuteBrowserWasmInspectionReport,
): void;

export function parseCppCuteBrowserWasmReviewArguments(
  argv: readonly string[],
): Readonly<{
  wasm: string;
  output: string;
  requireExactInterface: boolean;
}>;
