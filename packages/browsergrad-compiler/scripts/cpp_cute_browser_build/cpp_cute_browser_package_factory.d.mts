export const CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256: string;
export const CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH: number;

export class CppCuteBrowserPackageFactoryError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-FACTORY";
  readonly path: string;
}

export interface CppCuteBrowserPackageFactoryMaterializationInput {
  readonly sourcePath?: string;
  readonly destinationRoot?: string;
}

export interface CppCuteBrowserPackageFactoryMaterialization {
  readonly schema: "browsergrad.compiler.cpp-cute.package-factory-materialization";
  readonly version: 1;
  readonly authority: "package-materialization-only";
  readonly sourcePath: string;
  readonly destinationPath: string;
  readonly factorySha256: string;
  readonly factoryByteLength: number;
  readonly exactSourceVerified: true;
  readonly packageOwned: true;
  readonly cleanBuildVerified: false;
  readonly reproducibilityVerified: false;
  readonly workerBundleVerified: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export function materializeCppCuteBrowserPackageFactory(
  input?: CppCuteBrowserPackageFactoryMaterializationInput,
): Promise<Readonly<CppCuteBrowserPackageFactoryMaterialization>>;
