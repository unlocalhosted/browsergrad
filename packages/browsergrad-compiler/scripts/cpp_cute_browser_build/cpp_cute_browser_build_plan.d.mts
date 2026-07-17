import type {
  PreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

export interface CppCuteClangWasmMaterializedTools {
  readonly cmakeExecutable: string;
  readonly buildToolExecutable: string;
  readonly emsdkRoot: string;
  readonly emscriptenToolchainFile: string;
  readonly emscriptenConfigFile: string;
  readonly searchPath: readonly string[];
}

export interface CppCuteClangWasmBuildRoots {
  readonly llvmProjectSourceRoot: string;
  readonly extractorSourceRoot: string;
  readonly nativeBuildRoot: string;
  readonly wasmBuildRoot: string;
  readonly outputRoot: string;
  readonly stateRoot: string;
}

export interface CppCuteClangWasmBuildStep {
  readonly id: string;
  readonly stageId: "native-tablegen" | "clang-extractor-wasm";
  readonly kind: "configure" | "build";
  readonly executable: string;
  readonly arguments: readonly string[];
  readonly cwd: string;
  readonly environment: Readonly<Record<string, string>>;
}

export interface CppCuteClangWasmExtractorSourceFile {
  readonly path:
    | "BrowserGradCppCuteArtifactV3.cpp"
    | "BrowserGradCppCuteArtifactV3.h"
    | "BrowserGradCppCuteClangAction.cpp"
    | "BrowserGradCppCuteClangAction.h"
    | "BrowserGradCppCuteExtractor.cpp"
    | "BrowserGradCppCuteImportedVfs.cpp"
    | "BrowserGradCppCuteImportedVfs.h"
    | "BrowserGradCppCuteRuntime.cpp"
    | "BrowserGradCppCuteRuntime.h"
    | "CMakeLists.txt";
  readonly sha256: string;
  readonly byteLength: string;
  readonly absolutePath: string;
}

export interface CppCuteClangWasmBuildPlan {
  readonly schema: "browsergrad.compiler.cpp-cute.clang-wasm-build-plan";
  readonly version: 1;
  readonly lockId: string;
  readonly recipeId: string;
  readonly steps: readonly CppCuteClangWasmBuildStep[];
  readonly nativeTools: Readonly<{ clangTablegen: string; llvmTablegen: string }>;
  readonly extractorSource: Readonly<{
    sourceSetSha256: string;
    files: readonly CppCuteClangWasmExtractorSourceFile[];
    buildVerified: false;
    blockerId: "browsergrad-extractor-source-verification";
  }>;
  readonly generatedExtractor: Readonly<{
    factoryModulePath: string;
    wasmSidecarPath: string;
    distributedWasmPath: string;
    distributedMaterializationReady: false;
    materializationBlockerId: "browsergrad-extractor-distributed-materialization";
    workerBundleReady: false;
    blockerId: "browsergrad-worker-emscripten-factory-bundle";
  }>;
  readonly outputRoot: string;
}

export class CppCuteBrowserBuildPlanError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PLAN-INVALID";
  readonly path: string;
}

export function planCppCuteClangWasmBuild(input: Readonly<{
  lock: PreparedCppCuteBrowserBuildInputLock;
  tools: CppCuteClangWasmMaterializedTools;
  roots: CppCuteClangWasmBuildRoots;
}>): CppCuteClangWasmBuildPlan;
