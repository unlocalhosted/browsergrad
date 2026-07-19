export interface CppCuteBrowserWorkerBundleProjection {
  readonly schema: "browsergrad.compiler.cpp-cute.package-worker-bundle-resource";
  readonly version: 1;
  readonly authority: "package-worker-bundle-authoring-projection-only";
  readonly entryPath: string;
  readonly outputFileName: "browsergrad-cpp-cute-worker.mjs";
  readonly sha256: string;
  readonly byteLength: number;
  readonly staticImportCount: 0;
  readonly dynamicImportCount: 0;
  readonly factorySha256: string;
  readonly factoryByteLength: number;
  readonly source: string;
}

export class CppCuteBrowserWorkerBundleAuthoringError extends Error {}

export function buildCppCuteBrowserWorkerBundleProjection():
Promise<CppCuteBrowserWorkerBundleProjection>;

export function renderCppCuteBrowserWorkerBundleResource(
  projection: CppCuteBrowserWorkerBundleProjection,
): string;
