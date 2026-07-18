export interface VerifyCppCuteClangWasmReproducibilityInput {
  readonly firstRoot: string;
  readonly secondRoot: string;
}

export interface CppCuteClangWasmReproducibilityBuildIdentity {
  readonly ordinal: 1 | 2;
  readonly buildExecutionEvidenceSha256: string;
  readonly buildExecutionEvidenceByteLength: number;
  readonly runtimeClosureSha256: string;
  readonly runtimeClosureObservationSha256: string;
  readonly runtimeClosureObservationByteLength: number;
  readonly nativeTools: Readonly<{
    readonly clangTablegenSha256: string;
    readonly clangTablegenByteLength: number;
    readonly llvmTablegenSha256: string;
    readonly llvmTablegenByteLength: number;
  }>;
  readonly factoryModuleSha256: string;
  readonly factoryModuleByteLength: number;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly linkMapSha256: string;
  readonly linkMapByteLength: number;
}

declare const cppCuteClangWasmReproducibilityBrand: unique symbol;

/**
 * Two-build byte reproducibility authority for the extractor build only. It
 * does not prove ABI conformance, provenance, the complete package output set,
 * or release readiness.
 */
export interface VerifiedCppCuteClangWasmReproducibility {
  readonly [cppCuteClangWasmReproducibilityBrand]: true;
  readonly schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility";
  readonly version: 2;
  readonly authority: "clang-wasm-extractor-reproducibility-observation-only";
  readonly lockId: string;
  readonly sourceSetSha256: string;
  readonly cleanBuildCount: 2;
  readonly builds: readonly [
    CppCuteClangWasmReproducibilityBuildIdentity,
    CppCuteClangWasmReproducibilityBuildIdentity,
  ];
  readonly comparison: Readonly<{
    readonly sourceAndBuildPathsDistinct: true;
    readonly runtimeClosureMatched: true;
    readonly canonicalCommandsAndEnvironmentMatched: true;
    readonly nativeTablegenIdentitiesMatched: true;
    readonly factoryModuleBytesMatched: true;
    readonly wasmBytesMatched: true;
    readonly linkMapBytesMatched: true;
  }>;
  readonly claims: Readonly<{
    readonly extractorOutputsReproducible: true;
    readonly fullDistributedOutputSetReproducible: false;
    readonly abiConformanceVerified: false;
    readonly outputIdentityAuthorized: false;
    readonly producerAttested: false;
    readonly releaseReady: false;
  }>;
}

export type CppCuteClangWasmReproducibilityErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-CONFLICT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-IO";

export class CppCuteClangWasmReproducibilityError extends Error {
  readonly code: CppCuteClangWasmReproducibilityErrorCode;
  readonly path: string;
}

export function verifyCppCuteClangWasmReproducibility(
  input: VerifyCppCuteClangWasmReproducibilityInput,
): Promise<VerifiedCppCuteClangWasmReproducibility>;

export function writeCppCuteClangWasmReproducibilityEvidence(
  outputPath: string,
  evidence: VerifiedCppCuteClangWasmReproducibility,
): Promise<Readonly<{
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: number;
}>>;

export function parseCppCuteClangWasmReproducibilityArguments(
  argv: readonly string[],
): Readonly<{
  "first-root": string;
  "second-root": string;
  output: string;
}>;
