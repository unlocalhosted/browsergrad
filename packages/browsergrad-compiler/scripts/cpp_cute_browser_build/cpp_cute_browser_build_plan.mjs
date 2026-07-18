import { isAbsolute, join, normalize, relative } from "node:path/posix";

import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

const INVALID = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PLAN-INVALID";
const RECIPE_ID = "browsergrad.compiler.cpp-cute.clang-wasm-build@1";
const LOCK_ID = /^bg\.cpp\.browser-build-input-lock\.sha256\.[0-9a-f]{64}$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const PLACEHOLDER = /@[A-Z][A-Z0-9_]*@/gu;
const POSSIBLE_PLACEHOLDER = /@[A-Za-z0-9_]+@/u;
const PORTABLE_ABSOLUTE_PATH = /^\/[A-Za-z0-9._+/-]+$/u;

const RECIPE_ENVIRONMENT_NAMES = Object.freeze([
  "LANG",
  "LC_ALL",
  "SOURCE_DATE_EPOCH",
  "TZ",
  "ZERO_AR_DATE",
]);
const PREFIX_MAP_KINDS = Object.freeze(["debug", "file", "macro"]);
const NATIVE_DEFINITION_NAMES = Object.freeze([
  "CLANG_ENABLE_STATIC_ANALYZER",
  "CLANG_INCLUDE_DOCS",
  "CLANG_INCLUDE_TESTS",
  "CMAKE_BUILD_TYPE",
  "CMAKE_C_COMPILER",
  "CMAKE_CXX_COMPILER",
  "CMAKE_FIND_USE_PACKAGE_REGISTRY",
  "CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH",
  "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY",
  "CMAKE_MAKE_PROGRAM",
  "LLVM_APPEND_VC_REV",
  "LLVM_ENABLE_BINDINGS",
  "LLVM_ENABLE_CURL",
  "LLVM_ENABLE_EH",
  "LLVM_ENABLE_FFI",
  "LLVM_ENABLE_HTTPLIB",
  "LLVM_ENABLE_LIBEDIT",
  "LLVM_ENABLE_LIBXML2",
  "LLVM_ENABLE_PROJECTS",
  "LLVM_ENABLE_RTTI",
  "LLVM_ENABLE_Z3_SOLVER",
  "LLVM_ENABLE_ZLIB",
  "LLVM_ENABLE_ZSTD",
  "LLVM_INCLUDE_BENCHMARKS",
  "LLVM_INCLUDE_DOCS",
  "LLVM_INCLUDE_EXAMPLES",
  "LLVM_INCLUDE_TESTS",
  "LLVM_TARGETS_TO_BUILD",
  "Python3_EXECUTABLE",
]);
const WASM_DEFINITION_NAMES = Object.freeze([
  "BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH",
  "BUILD_SHARED_LIBS",
  "CLANG_BUILD_TOOLS",
  "CLANG_ENABLE_HLSL",
  "CLANG_ENABLE_OBJC_REWRITER",
  "CLANG_ENABLE_STATIC_ANALYZER",
  "CLANG_INCLUDE_DOCS",
  "CLANG_INCLUDE_TESTS",
  "CLANG_TABLEGEN",
  "CMAKE_BUILD_TYPE",
  "CMAKE_FIND_USE_PACKAGE_REGISTRY",
  "CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH",
  "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY",
  "CMAKE_INSTALL_PREFIX",
  "CMAKE_MAKE_PROGRAM",
  "CMAKE_TRY_COMPILE_TARGET_TYPE",
  "LLVM_APPEND_VC_REV",
  "LLVM_BUILD_LLVM_DYLIB",
  "LLVM_BUILD_TOOLS",
  "LLVM_ENABLE_BACKTRACES",
  "LLVM_ENABLE_BINDINGS",
  "LLVM_ENABLE_CRASH_OVERRIDES",
  "LLVM_ENABLE_CURL",
  "LLVM_ENABLE_EH",
  "LLVM_ENABLE_FFI",
  "LLVM_ENABLE_HTTPLIB",
  "LLVM_ENABLE_LIBEDIT",
  "LLVM_ENABLE_LIBXML2",
  "LLVM_ENABLE_LTO",
  "LLVM_ENABLE_PIC",
  "LLVM_ENABLE_PLUGINS",
  "LLVM_ENABLE_PROJECTS",
  "LLVM_ENABLE_RTTI",
  "LLVM_ENABLE_THREADS",
  "LLVM_ENABLE_UNWIND_TABLES",
  "LLVM_ENABLE_Z3_SOLVER",
  "LLVM_ENABLE_ZLIB",
  "LLVM_ENABLE_ZSTD",
  "LLVM_EXTERNAL_BROWSERGRAD_EXTRACTOR_SOURCE_DIR",
  "LLVM_EXTERNAL_PROJECTS",
  "LLVM_INCLUDE_BENCHMARKS",
  "LLVM_INCLUDE_DOCS",
  "LLVM_INCLUDE_EXAMPLES",
  "LLVM_INCLUDE_TESTS",
  "LLVM_LINK_LLVM_DYLIB",
  "LLVM_NATIVE_TOOL_DIR",
  "LLVM_TABLEGEN",
  "LLVM_TARGETS_TO_BUILD",
  "Python3_EXECUTABLE",
]);
const NATIVE_TARGETS = Object.freeze(["clang-tblgen", "llvm-tblgen"]);
const WASM_TARGETS = Object.freeze(["browsergrad-cpp-cute-extractor"]);
const EXTRACTOR_SOURCE_PATHS = Object.freeze([
  "BrowserGradCppCuteArtifactV3.cpp",
  "BrowserGradCppCuteArtifactV3.h",
  "BrowserGradCppCuteArtifactWriter.cpp",
  "BrowserGradCppCuteArtifactWriter.h",
  "BrowserGradCppCuteBrowserHost.cpp",
  "BrowserGradCppCuteCanonicalJson.cpp",
  "BrowserGradCppCuteCanonicalJson.h",
  "BrowserGradCppCuteClangAction.cpp",
  "BrowserGradCppCuteClangAction.h",
  "BrowserGradCppCuteCommandLine.cpp",
  "BrowserGradCppCuteCommandLine.h",
  "BrowserGradCppCuteCommandLinePolicy.inc",
  "BrowserGradCppCuteCompilePlan.cpp",
  "BrowserGradCppCuteCompilePlan.h",
  "BrowserGradCppCuteCompileSession.cpp",
  "BrowserGradCppCuteCompileSession.h",
  "BrowserGradCppCuteDiagnostics.cpp",
  "BrowserGradCppCuteDiagnostics.h",
  "BrowserGradCppCuteDiagnosticsPolicy.inc",
  "BrowserGradCppCuteExtractor.cpp",
  "BrowserGradCppCuteImportedVfs.cpp",
  "BrowserGradCppCuteImportedVfs.h",
  "BrowserGradCppCuteInvocation.cpp",
  "BrowserGradCppCuteInvocation.h",
  "BrowserGradCppCuteMetrics.cpp",
  "BrowserGradCppCuteMetrics.h",
  "BrowserGradCppCutePreprocessorPolicy.cpp",
  "BrowserGradCppCutePreprocessorPolicy.h",
  "BrowserGradCppCuteProducer.cpp",
  "BrowserGradCppCuteProducer.h",
  "BrowserGradCppCuteRuntime.cpp",
  "BrowserGradCppCuteRuntime.h",
  "BrowserGradCppCuteSha256.cpp",
  "BrowserGradCppCuteSha256.h",
  "BrowserGradCppCuteVirtualPath.cpp",
  "BrowserGradCppCuteVirtualPath.h",
  "CMakeLists.txt",
]);

/** @typedef {Readonly<{ name: string; value: string }>} RecipeNameValue */

/** @typedef {Readonly<{ path: string; sha256: string; byteLength: string }>} ExtractorSourceFile */

/**
 * @typedef {Readonly<{
 *   sourceSetSha256: string;
 *   hashDomain: "browsergrad.compiler.cpp-cute.browser-extractor-source-set.v1";
 *   files: readonly ExtractorSourceFile[];
 * }>} ExtractorSourceSet
 */

/**
 * @typedef {Readonly<{
 *   ordinal: number;
 *   stageId: "native-tablegen" | "clang-extractor-wasm";
 *   executionPlatform: string;
 *   cmakeGenerator: string;
 *   sourceSubdirectory: string;
 *   buildDirectoryRole: string;
 *   definitions: readonly RecipeNameValue[];
 *   compilerFlags: readonly string[];
 *   linkerFlags?: readonly string[];
 *   targets: readonly string[];
 * }>} CppCuteClangWasmRecipeStage
 */

/**
 * @typedef {Readonly<{
 *   recipeId: string;
 *   sourceDateEpoch: string;
 *   environment: readonly RecipeNameValue[];
 *   parallelJobs: number;
 *   prefixMapKinds: readonly string[];
 *   stages: readonly CppCuteClangWasmRecipeStage[];
 *   extractorSource: ExtractorSourceSet;
 * }>} CppCuteClangWasmRecipe
 */

/** @typedef {import("../../dist/cpp_cute_browser_build_lock.js").PreparedCppCuteBrowserBuildInputLock} PreparedCppCuteBrowserBuildInputLock */

/**
 * @typedef {Readonly<{
 *   cmakeExecutable: string;
 *   buildToolExecutable: string;
 *   emsdkRoot: string;
 *   emscriptenToolchainFile: string;
 *   emscriptenConfigFile: string;
 *   searchPath: readonly string[];
 * }>} CppCuteClangWasmMaterializedTools
 */

/**
 * @typedef {Readonly<{
 *   llvmProjectSourceRoot: string;
 *   extractorSourceRoot: string;
 *   nativeBuildRoot: string;
 *   wasmBuildRoot: string;
 *   outputRoot: string;
 *   stateRoot: string;
 * }>} CppCuteClangWasmBuildRoots
 */

/**
 * @typedef {Readonly<{
 *   id: string;
 *   stageId: "native-tablegen" | "clang-extractor-wasm";
 *   kind: "configure" | "build";
 *   executable: string;
 *   arguments: readonly string[];
 *   cwd: string;
 *   environment: Readonly<Record<string, string>>;
 * }>} CppCuteClangWasmBuildStep
 */

export class CppCuteBrowserBuildPlanError extends Error {
  /** @param {string} path @param {string} message */
  constructor(path, message) {
    super(`${INVALID}: ${message}`);
    this.name = "CppCuteBrowserBuildPlanError";
    this.code = INVALID;
    this.path = path;
  }
}

/**
 * Materializes the exact recipe selected by an already verified build lock.
 * It only substitutes absolute paths and reproducibility prefix maps. It has
 * no filesystem, process, network, package-manager, container, runtime, or
 * artifact-authorization effects.
 *
 * @param {Readonly<{
 *   lock: PreparedCppCuteBrowserBuildInputLock;
 *   tools: CppCuteClangWasmMaterializedTools;
 *   roots: CppCuteClangWasmBuildRoots;
 * }>} input
 */
export function planCppCuteClangWasmBuild(input) {
  const object = requiredObject(input, "$input");
  const selected = snapshotSelectedRecipe(unwrapPreparedCppCuteBrowserBuildInputLock(
    /** @type {PreparedCppCuteBrowserBuildInputLock} */ (object.lock),
  ));
  const tools = snapshotTools(object.tools);
  const roots = snapshotRoots(object.roots);
  assertSeparatedRoots(roots, tools.emsdkRoot);

  const native = selected.recipe.stages[0];
  const wasm = selected.recipe.stages[1];
  if (native === undefined || wasm === undefined) invalid("$preparedLockRecord.lock.body.recipe.stages", "expected two stages");

  const nativeEnvironment = cleanEnvironment(selected.recipe, tools, roots, false);
  const wasmEnvironment = cleanEnvironment(selected.recipe, tools, roots, true);
  const nativeBuild = roots.nativeBuildRoot;
  const wasmBuild = roots.wasmBuildRoot;
  const nativeTools = {
    clangTablegen: join(nativeBuild, "bin", "clang-tblgen"),
    llvmTablegen: join(nativeBuild, "bin", "llvm-tblgen"),
  };
  const bindings = {
    "@BUILD_EVIDENCE@": join(roots.stateRoot, "evidence"),
    "@EMSDK@": tools.emsdkRoot,
    "@EXTRACTOR_SOURCE@": roots.extractorSourceRoot,
    "@NATIVE_BUILD@": nativeBuild,
    "@OUTPUT@": roots.outputRoot,
  };
  const generatedFactoryModulePath = join(
    roots.stateRoot,
    "evidence",
    "generated",
    "clang-extractor.mjs",
  );

  const nativeSteps = stageSteps({
    stage: native,
    stagePath: "$preparedLockRecord.lock.body.recipe.stages[0]",
    sourceRoot: roots.llvmProjectSourceRoot,
    buildRoot: nativeBuild,
    environment: nativeEnvironment,
    tools,
    bindings: {
      ...bindings,
      "@PREFIX_MAP_FLAGS@": prefixMapFlags(selected.recipe.prefixMapKinds, [
        [roots.llvmProjectSourceRoot, "/browsergrad/source/llvm-project"],
        [nativeBuild, "/browsergrad/build/native-tablegen"],
      ]).join(" "),
    },
    parallelJobs: selected.recipe.parallelJobs,
    emscripten: false,
  });
  const wasmSteps = stageSteps({
    stage: wasm,
    stagePath: "$preparedLockRecord.lock.body.recipe.stages[1]",
    sourceRoot: roots.llvmProjectSourceRoot,
    buildRoot: wasmBuild,
    environment: wasmEnvironment,
    tools,
    bindings: {
      ...bindings,
      "@PREFIX_MAP_FLAGS@": prefixMapFlags(selected.recipe.prefixMapKinds, [
        [roots.llvmProjectSourceRoot, "/browsergrad/source/llvm-project"],
        [roots.extractorSourceRoot, "/browsergrad/source/extractor"],
        [nativeBuild, "/browsergrad/build/native-tablegen"],
        [wasmBuild, "/browsergrad/build/clang-extractor-wasm"],
        [roots.outputRoot, "/browsergrad/output"],
      ]).join(" "),
    },
    parallelJobs: selected.recipe.parallelJobs,
    emscripten: true,
  });

  return deepFreeze({
    schema: "browsergrad.compiler.cpp-cute.clang-wasm-build-plan",
    version: 1,
    lockId: selected.lockId,
    recipeId: selected.recipe.recipeId,
    steps: [...nativeSteps, ...wasmSteps],
    nativeTools,
    extractorSource: {
      sourceSetSha256: selected.recipe.extractorSource.sourceSetSha256,
      files: selected.recipe.extractorSource.files.map((file) => ({
        ...file,
        absolutePath: join(roots.extractorSourceRoot, file.path),
      })),
      buildVerified: false,
      blockerId: "browsergrad-extractor-source-verification",
    },
    generatedExtractor: {
      factoryModulePath: generatedFactoryModulePath,
      wasmSidecarPath: join(
        roots.stateRoot,
        "evidence",
        "generated",
        "clang-extractor.wasm",
      ),
      distributedWasmPath: join(
        roots.outputRoot,
        "browsergrad-cpp-cute",
        "clang-extractor.wasm",
      ),
      distributedMaterializationReady: false,
      materializationBlockerId: "browsergrad-extractor-distributed-materialization",
      workerBundleReady: false,
      blockerId: "browsergrad-worker-emscripten-factory-bundle",
    },
    outputRoot: roots.outputRoot,
  });
}

/**
 * @param {Readonly<{
 *   stage: CppCuteClangWasmRecipeStage;
 *   stagePath: string;
 *   sourceRoot: string;
 *   buildRoot: string;
 *   environment: Readonly<Record<string, string>>;
 *   tools: CppCuteClangWasmMaterializedTools;
 *   bindings: Readonly<Record<string, string>>;
 *   parallelJobs: number;
 *   emscripten: boolean;
 * }>} input
 * @returns {readonly CppCuteClangWasmBuildStep[]}
 */
function stageSteps(input) {
  const source = join(input.sourceRoot, input.stage.sourceSubdirectory);
  const definitions = input.stage.definitions.flatMap((definition, index) => {
    const value = substitute(definition.value, input.bindings, `${input.stagePath}.definitions[${index}].value`);
    if (definition.name === "CMAKE_MAKE_PROGRAM") {
      if (value !== input.tools.buildToolExecutable) {
        invalid(`${input.stagePath}.definitions[${index}].value`, "materialized build tool must equal the lock-selected path");
      }
      return [];
    }
    return [define(definition.name, value)];
  });
  const compilerFlags = input.stage.compilerFlags.map((flag, index) => substitute(
    flag,
    input.bindings,
    `${input.stagePath}.compilerFlags[${index}]`,
  ));
  const configureArguments = [
    "-S", source,
    "-B", input.buildRoot,
    "-G", input.stage.cmakeGenerator,
    define("CMAKE_MAKE_PROGRAM", input.tools.buildToolExecutable),
  ];
  if (input.emscripten) {
    configureArguments.push(define("CMAKE_TOOLCHAIN_FILE", input.tools.emscriptenToolchainFile));
  }
  configureArguments.push(
    ...definitions,
    define("CMAKE_C_FLAGS", compilerFlags.join(" ")),
    define("CMAKE_CXX_FLAGS", compilerFlags.join(" ")),
  );
  if (input.stage.linkerFlags !== undefined) {
    const linkerFlags = input.stage.linkerFlags.map((flag, index) => substitute(
      flag,
      input.bindings,
      `${input.stagePath}.linkerFlags[${index}]`,
    ));
    configureArguments.push(define("CMAKE_EXE_LINKER_FLAGS", linkerFlags.join(" ")));
  }

  return [
    {
      id: `${input.stage.stageId}-configure`,
      stageId: input.stage.stageId,
      kind: "configure",
      executable: input.tools.cmakeExecutable,
      arguments: configureArguments,
      cwd: input.sourceRoot,
      environment: input.environment,
    },
    {
      id: `${input.stage.stageId}-build`,
      stageId: input.stage.stageId,
      kind: "build",
      executable: input.tools.cmakeExecutable,
      arguments: [
        "--build", input.buildRoot,
        "--target", ...input.stage.targets,
        "--parallel", String(input.parallelJobs),
      ],
      cwd: input.buildRoot,
      environment: input.environment,
    },
  ];
}

/** @param {unknown} value @returns {Readonly<{ lockId: string; recipe: CppCuteClangWasmRecipe }>} */
function snapshotSelectedRecipe(value) {
  const record = requiredObject(value, "$preparedLockRecord");
  const lock = requiredObject(record.lock, "$preparedLockRecord.lock");
  const body = requiredObject(lock.body, "$preparedLockRecord.lock.body");
  const lockId = requiredString(lock.lockId, "$preparedLockRecord.lock.lockId", 128);
  if (!LOCK_ID.test(lockId)) invalid("$preparedLockRecord.lock.lockId", "expected a verified browser build-lock ID");
  return { lockId, recipe: parseRecipe(body.recipe) };
}

/** @param {unknown} value @returns {CppCuteClangWasmRecipe} */
function parseRecipe(value) {
  const recipe = requiredObject(value, "$preparedLockRecord.lock.body.recipe");
  const recipeId = requiredString(recipe.recipeId, "$preparedLockRecord.lock.body.recipe.recipeId", 128);
  if (recipeId !== RECIPE_ID) invalid("$preparedLockRecord.lock.body.recipe.recipeId", `expected ${RECIPE_ID}`);
  const sourceDateEpoch = decimalString(recipe.sourceDateEpoch, "$preparedLockRecord.lock.body.recipe.sourceDateEpoch");
  const environment = nameValueArray(
    recipe.environment,
    "$preparedLockRecord.lock.body.recipe.environment",
    RECIPE_ENVIRONMENT_NAMES,
  );
  const environmentEpoch = environment.find((entry) => entry.name === "SOURCE_DATE_EPOCH")?.value;
  if (environmentEpoch !== sourceDateEpoch) {
    invalid("$preparedLockRecord.lock.body.recipe.environment", "SOURCE_DATE_EPOCH must equal recipe sourceDateEpoch");
  }
  const parallelJobs = positiveInteger(recipe.parallelJobs, "$preparedLockRecord.lock.body.recipe.parallelJobs", 256);
  const prefixMapKinds = exactStringArray(
    recipe.prefixMapKinds,
    "$preparedLockRecord.lock.body.recipe.prefixMapKinds",
    PREFIX_MAP_KINDS,
  );
  const stages = requiredArray(recipe.stages, "$preparedLockRecord.lock.body.recipe.stages");
  if (stages.length !== 2) invalid("$preparedLockRecord.lock.body.recipe.stages", "recipe v1 requires exactly two stages");
  return {
    recipeId,
    sourceDateEpoch,
    environment,
    parallelJobs,
    prefixMapKinds,
    extractorSource: parseExtractorSource(recipe.extractorSource),
    stages: [
      parseStage(stages[0], 0, "native-tablegen", NATIVE_DEFINITION_NAMES, NATIVE_TARGETS, false),
      parseStage(stages[1], 1, "clang-extractor-wasm", WASM_DEFINITION_NAMES, WASM_TARGETS, true),
    ],
  };
}

/** @param {unknown} value @returns {ExtractorSourceSet} */
function parseExtractorSource(value) {
  const path = "$preparedLockRecord.lock.body.recipe.extractorSource";
  const source = requiredObject(value, path);
  const hashDomain = requiredString(source.hashDomain, `${path}.hashDomain`, 128);
  if (hashDomain !== "browsergrad.compiler.cpp-cute.browser-extractor-source-set.v1") {
    invalid(`${path}.hashDomain`, "unexpected extractor source hash domain");
  }
  const sourceSetSha256 = requiredString(
    source.sourceSetSha256,
    `${path}.sourceSetSha256`,
    64,
  );
  if (!SHA256_HEX.test(sourceSetSha256)) {
    invalid(`${path}.sourceSetSha256`, "expected lowercase SHA-256 hex");
  }
  const files = requiredArray(source.files, `${path}.files`);
  if (files.length !== EXTRACTOR_SOURCE_PATHS.length) {
    invalid(`${path}.files`, "extractor source set has missing or extra files");
  }
  return {
    sourceSetSha256,
    hashDomain,
    files: files.map((entry, index) => {
      const filePath = `${path}.files[${index}]`;
      const file = requiredObject(entry, filePath);
      const selectedPath = requiredString(file.path, `${filePath}.path`, 256);
      if (selectedPath !== EXTRACTOR_SOURCE_PATHS[index]) {
        invalid(`${filePath}.path`, `expected ${EXTRACTOR_SOURCE_PATHS[index]}`);
      }
      const sha256 = requiredString(file.sha256, `${filePath}.sha256`, 64);
      if (!SHA256_HEX.test(sha256)) {
        invalid(`${filePath}.sha256`, "expected lowercase SHA-256 hex");
      }
      return {
        path: selectedPath,
        sha256,
        byteLength: decimalString(file.byteLength, `${filePath}.byteLength`),
      };
    }),
  };
}

/**
 * @param {unknown} value
 * @param {number} ordinal
 * @param {"native-tablegen" | "clang-extractor-wasm"} stageId
 * @param {readonly string[]} definitionNames
 * @param {readonly string[]} targets
 * @param {boolean} requiresLinkerFlags
 * @returns {CppCuteClangWasmRecipeStage}
 */
function parseStage(value, ordinal, stageId, definitionNames, targets, requiresLinkerFlags) {
  const path = `$preparedLockRecord.lock.body.recipe.stages[${ordinal}]`;
  const stage = requiredObject(value, path);
  if (stage.ordinal !== ordinal) invalid(`${path}.ordinal`, `expected ${ordinal}`);
  if (stage.stageId !== stageId) invalid(`${path}.stageId`, `expected ${stageId}`);
  const executionPlatform = requiredString(stage.executionPlatform, `${path}.executionPlatform`, 128);
  const cmakeGenerator = requiredString(stage.cmakeGenerator, `${path}.cmakeGenerator`, 128);
  const sourceSubdirectory = relativeSubdirectory(stage.sourceSubdirectory, `${path}.sourceSubdirectory`);
  const buildDirectoryRole = requiredString(stage.buildDirectoryRole, `${path}.buildDirectoryRole`, 128);
  if (buildDirectoryRole !== stageId) invalid(`${path}.buildDirectoryRole`, `expected ${stageId}`);
  const definitions = nameValueArray(stage.definitions, `${path}.definitions`, definitionNames);
  const compilerFlags = nonemptyStringArray(stage.compilerFlags, `${path}.compilerFlags`);
  if (compilerFlags.filter((flag) => flag === "@PREFIX_MAP_FLAGS@").length !== 1) {
    invalid(`${path}.compilerFlags`, "expected exactly one prefix-map placeholder");
  }
  const selectedTargets = exactStringArray(stage.targets, `${path}.targets`, targets);
  let linkerFlags;
  if (requiresLinkerFlags) {
    linkerFlags = nonemptyStringArray(stage.linkerFlags, `${path}.linkerFlags`);
    if (!linkerFlags.some((flag) => flag.includes("@BUILD_EVIDENCE@"))) {
      invalid(`${path}.linkerFlags`, "expected build-evidence placeholder");
    }
  } else if (stage.linkerFlags !== undefined) {
    invalid(`${path}.linkerFlags`, "native TableGen stage must not declare linker flags");
  }
  return {
    ordinal,
    stageId,
    executionPlatform,
    cmakeGenerator,
    sourceSubdirectory,
    buildDirectoryRole,
    definitions,
    compilerFlags,
    ...(linkerFlags === undefined ? {} : { linkerFlags }),
    targets: selectedTargets,
  };
}

/** @param {unknown} value @returns {CppCuteClangWasmMaterializedTools} */
function snapshotTools(value) {
  const tools = requiredObject(value, "$input.tools");
  const searchPath = requiredArray(tools.searchPath, "$input.tools.searchPath").map((entry, index) => (
    absolutePath(entry, `$input.tools.searchPath[${index}]`)
  ));
  if (searchPath.length === 0 || new Set(searchPath).size !== searchPath.length) {
    invalid("$input.tools.searchPath", "expected unique verified tool directories");
  }
  const result = {
    cmakeExecutable: absolutePath(tools.cmakeExecutable, "$input.tools.cmakeExecutable"),
    buildToolExecutable: absolutePath(tools.buildToolExecutable, "$input.tools.buildToolExecutable"),
    emsdkRoot: absolutePath(tools.emsdkRoot, "$input.tools.emsdkRoot"),
    emscriptenToolchainFile: absolutePath(tools.emscriptenToolchainFile, "$input.tools.emscriptenToolchainFile"),
    emscriptenConfigFile: absolutePath(tools.emscriptenConfigFile, "$input.tools.emscriptenConfigFile"),
    searchPath,
  };
  if (!contains(result.emsdkRoot, result.emscriptenToolchainFile)
      || !contains(result.emsdkRoot, result.emscriptenConfigFile)) {
    invalid("$input.tools", "Emscripten toolchain and config must be inside the verified emsdk root");
  }
  return result;
}

/** @param {unknown} value @returns {CppCuteClangWasmBuildRoots} */
function snapshotRoots(value) {
  const roots = requiredObject(value, "$input.roots");
  return {
    llvmProjectSourceRoot: absolutePath(roots.llvmProjectSourceRoot, "$input.roots.llvmProjectSourceRoot"),
    extractorSourceRoot: absolutePath(roots.extractorSourceRoot, "$input.roots.extractorSourceRoot"),
    nativeBuildRoot: absolutePath(roots.nativeBuildRoot, "$input.roots.nativeBuildRoot"),
    wasmBuildRoot: absolutePath(roots.wasmBuildRoot, "$input.roots.wasmBuildRoot"),
    outputRoot: absolutePath(roots.outputRoot, "$input.roots.outputRoot"),
    stateRoot: absolutePath(roots.stateRoot, "$input.roots.stateRoot"),
  };
}

/** @param {CppCuteClangWasmBuildRoots} roots @param {string} emsdkRoot */
function assertSeparatedRoots(roots, emsdkRoot) {
  const entries = Object.entries(roots);
  for (let leftIndex = 0; leftIndex < entries.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < entries.length; rightIndex += 1) {
      const [leftName, left] = entries[leftIndex];
      const [rightName, right] = entries[rightIndex];
      if (contains(left, right) || contains(right, left)) {
        invalid("$input.roots", `${leftName} and ${rightName} must be disjoint roots`);
      }
    }
  }
  for (const [name, root] of entries) {
    if (contains(emsdkRoot, root) || contains(root, emsdkRoot)) {
      invalid(`$input.roots.${name}`, "build/source/output roots must be disjoint from the Emscripten toolchain");
    }
  }
}

/**
 * @param {CppCuteClangWasmRecipe} recipe
 * @param {CppCuteClangWasmMaterializedTools} tools
 * @param {CppCuteClangWasmBuildRoots} roots
 * @param {boolean} emscripten
 * @returns {Readonly<Record<string, string>>}
 */
function cleanEnvironment(recipe, tools, roots, emscripten) {
  /** @type {Record<string, string>} */
  const environment = {
    HOME: join(roots.stateRoot, "home"),
    PATH: tools.searchPath.join(":"),
    PKG_CONFIG_LIBDIR: join(roots.stateRoot, "empty-pkgconfig"),
    PKG_CONFIG_PATH: "",
    PYTHONHASHSEED: "0",
    TMPDIR: join(roots.stateRoot, "tmp"),
  };
  for (const entry of recipe.environment) environment[entry.name] = entry.value;
  if (emscripten) {
    environment.EM_CACHE = join(roots.stateRoot, "em-cache");
    environment.EM_CONFIG = tools.emscriptenConfigFile;
  }
  return environment;
}

/**
 * @param {readonly string[]} kinds
 * @param {readonly (readonly [string, string])[]} mappings
 * @returns {string[]}
 */
function prefixMapFlags(kinds, mappings) {
  return mappings.flatMap(([from, to]) => kinds.map((kind) => `-f${kind}-prefix-map=${from}=${to}`));
}

/** @param {string} value @param {Readonly<Record<string, string>>} bindings @param {string} path */
function substitute(value, bindings, path) {
  const substituted = value.replace(PLACEHOLDER, (placeholder) => {
    const replacement = bindings[placeholder];
    if (replacement === undefined) invalid(path, `unknown build-recipe placeholder ${placeholder}`);
    return replacement;
  });
  if (POSSIBLE_PLACEHOLDER.test(substituted)) invalid(path, "unresolved build-recipe placeholder");
  return substituted;
}

/**
 * @param {unknown} value
 * @param {string} path
 * @param {readonly string[]} expectedNames
 * @returns {readonly RecipeNameValue[]}
 */
function nameValueArray(value, path, expectedNames) {
  const entries = requiredArray(value, path);
  if (entries.length !== expectedNames.length) invalid(path, "missing or extra entries are forbidden");
  return entries.map((entry, index) => {
    const object = requiredObject(entry, `${path}[${index}]`);
    const name = requiredString(object.name, `${path}[${index}].name`, 128);
    if (name !== expectedNames[index]) invalid(`${path}[${index}].name`, `expected ${expectedNames[index]}`);
    return { name, value: requiredString(object.value, `${path}[${index}].value`, 16_384) };
  });
}

/** @param {unknown} value @param {string} path @param {readonly string[]} expected @returns {readonly string[]} */
function exactStringArray(value, path, expected) {
  const entries = requiredArray(value, path);
  if (entries.length !== expected.length) invalid(path, "missing or extra entries are forbidden");
  return entries.map((entry, index) => {
    const string = requiredString(entry, `${path}[${index}]`, 16_384);
    if (string !== expected[index]) invalid(`${path}[${index}]`, `expected ${expected[index]}`);
    return string;
  });
}

/** @param {unknown} value @param {string} path @returns {readonly string[]} */
function nonemptyStringArray(value, path) {
  const entries = requiredArray(value, path);
  if (entries.length === 0) invalid(path, "expected a nonempty array");
  return entries.map((entry, index) => requiredString(entry, `${path}[${index}]`, 16_384));
}

/** @param {unknown} value @param {string} path */
function relativeSubdirectory(value, path) {
  const candidate = requiredString(value, path, 128);
  if (candidate === "." || candidate === ".." || candidate.includes("/") || candidate.includes("\\")) {
    invalid(path, "expected one relative source subdirectory segment");
  }
  return candidate;
}

/** @param {unknown} value @param {string} path */
function decimalString(value, path) {
  const candidate = requiredString(value, path, 32);
  if (!/^(?:0|[1-9][0-9]*)$/u.test(candidate)) invalid(path, "expected a canonical decimal integer string");
  return candidate;
}

/** @param {unknown} value @param {string} path @param {number} maximum */
function positiveInteger(value, path, maximum) {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1 || value > maximum) {
    invalid(path, `expected an integer from 1 through ${maximum}`);
  }
  return value;
}

/** @param {string} name @param {string} value */
function define(name, value) {
  return `-D${name}=${value}`;
}

/** @param {string} parent @param {string} child */
function contains(parent, child) {
  const path = relative(parent, child);
  return path === "" || (!path.startsWith("../") && path !== "..");
}

/** @param {unknown} value @param {string} path @returns {Record<string, unknown>} */
function requiredObject(value, path) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) invalid(path, "expected an object");
  return /** @type {Record<string, unknown>} */ (value);
}

/** @param {unknown} value @param {string} path @returns {unknown[]} */
function requiredArray(value, path) {
  if (!Array.isArray(value)) invalid(path, "expected an array");
  return value;
}

/** @param {unknown} value @param {string} path @param {number} maximumLength @returns {string} */
function requiredString(value, path, maximumLength) {
  if (typeof value !== "string" || value.length === 0 || value.length > maximumLength || hasControlCharacter(value)) {
    invalid(path, `expected a nonempty string of at most ${maximumLength} characters without controls`);
  }
  return value;
}

/** @param {string} value */
function hasControlCharacter(value) {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (codePoint !== undefined && (codePoint <= 31 || codePoint === 127)) return true;
  }
  return false;
}

/** @param {unknown} value @param {string} path */
function absolutePath(value, path) {
  const candidate = requiredString(value, path, 4_096);
  if (!isAbsolute(candidate)
      || normalize(candidate) !== candidate
      || candidate === "/"
      || candidate.endsWith("/")
      || !PORTABLE_ABSOLUTE_PATH.test(candidate)) {
    invalid(path, "expected a normalized portable CMake-safe absolute POSIX path");
  }
  return candidate;
}

/** @param {string} path @param {string} message @returns {never} */
function invalid(path, message) {
  throw new CppCuteBrowserBuildPlanError(path, message);
}

/** @template T @param {T} value @returns {Readonly<T>} */
function deepFreeze(value) {
  if (value === null || typeof value !== "object" || Object.isFrozen(value)) {
    return /** @type {Readonly<T>} */ (value);
  }
  for (const entry of Object.values(/** @type {Record<string, unknown>} */ (value))) deepFreeze(entry);
  return /** @type {Readonly<T>} */ (Object.freeze(value));
}
