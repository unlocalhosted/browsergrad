export class CppCuteBrowserBuildCacheReuseError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-CACHE-REUSE";
  readonly path: string;
  constructor(path: string, message: string, options?: ErrorOptions);
}

export function invalidateCachedCppCuteExtractorObjects(input: Readonly<{
  wasmBuildRoot: string;
  sourcePaths: readonly string[];
}>): Promise<void>;
