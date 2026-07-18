export interface CppCuteBrowserConfiguredTargetReview {
  readonly authority: "configured-target-flags-review-only";
  readonly exceptionMode: "emscripten-javascript";
  readonly rttiRequired: true;
  readonly compileFlagsPath: string;
  readonly compileFlagsSha256: string;
  readonly compileFlagsByteLength: number;
  readonly linkCommandPath: string;
  readonly linkCommandSha256: string;
  readonly linkCommandByteLength: number;
  readonly buildExecuted: false;
  readonly abiConformanceVerified: false;
  readonly releaseReady: false;
}

export class CppCuteBrowserConfiguredTargetReviewError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-CONFIGURED-TARGET-REVIEW-INVALID";
  readonly path: string;
}

export function reviewCppCuteBrowserConfiguredTarget(input: Readonly<{
  wasmBuildRoot: string;
  factoryModulePath: string;
}>): Promise<CppCuteBrowserConfiguredTargetReview>;
