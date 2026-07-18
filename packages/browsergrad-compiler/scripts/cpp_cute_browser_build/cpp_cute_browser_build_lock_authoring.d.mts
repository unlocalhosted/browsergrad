export interface CppCuteBrowserBuildLockAuthoringReport {
  readonly schema: "browsergrad.compiler.cpp-cute.browser-build-lock-authoring-projection";
  readonly version: 1;
  readonly authority: "authoring-projection-only";
  readonly checkedInResourceMatches: boolean;
  readonly lockId: string;
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly recipeSha256: string;
  readonly extractorSourceSetSha256: string;
  readonly files: readonly {
    readonly path: string;
    readonly sha256: string;
    readonly byteLength: string;
  }[];
}

export class CppCuteBrowserBuildLockAuthoringError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-AUTHORING";
  readonly path: string;
}

export function parseCppCuteBrowserBuildLockAuthoringArguments(
  argv: readonly string[],
): Readonly<{ check: boolean }>;

export function projectCppCuteBrowserBuildInputLock():
  Promise<CppCuteBrowserBuildLockAuthoringReport>;

