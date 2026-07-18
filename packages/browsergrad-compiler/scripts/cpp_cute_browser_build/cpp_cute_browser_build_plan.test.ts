import { beforeAll, describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CppCuteBrowserBuildPlanError,
  planCppCuteClangWasmBuild,
} from "./cpp_cute_browser_build_plan.mjs";

interface MutableTools {
  cmakeExecutable: string;
  buildToolExecutable: string;
  emsdkRoot: string;
  emscriptenToolchainFile: string;
  emscriptenConfigFile: string;
  searchPath: string[];
}

interface MutableRoots {
  llvmProjectSourceRoot: string;
  extractorSourceRoot: string;
  nativeBuildRoot: string;
  wasmBuildRoot: string;
  outputRoot: string;
  stateRoot: string;
}

interface MutableInput {
  lock: PreparedCppCuteBrowserBuildInputLock;
  tools: MutableTools;
  roots: MutableRoots;
}

let preparedLock: PreparedCppCuteBrowserBuildInputLock;

beforeAll(async () => {
  preparedLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
});

function input(): MutableInput {
  return {
    lock: preparedLock,
    tools: {
      cmakeExecutable: "/tools/cmake/bin/cmake",
      buildToolExecutable: "/usr/bin/make",
      emsdkRoot: "/tools/emsdk",
      emscriptenToolchainFile: "/tools/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake",
      emscriptenConfigFile: "/tools/emsdk/.emscripten",
      searchPath: [
        "/tools/cmake/bin",
        "/tools/emsdk/upstream/bin",
        "/tools/emsdk/upstream/emscripten",
        "/tools/python/bin",
        "/tools/node/bin",
      ],
    },
    roots: {
      llvmProjectSourceRoot: "/source/llvm-project",
      extractorSourceRoot: "/source/browsergrad-extractor",
      nativeBuildRoot: "/build/native",
      wasmBuildRoot: "/build/wasm",
      outputRoot: "/release/assets",
      stateRoot: "/state/clang-wasm-a",
    },
  };
}

function definition(step: ReturnType<typeof planCppCuteClangWasmBuild>["steps"][number], name: string): string {
  const value = step.arguments.find((entry) => entry.startsWith(`-D${name}=`));
  if (value === undefined) throw new Error(`missing definition ${name}`);
  return value;
}

function expectPlannerInvalid(value: MutableInput, path: string): void {
  expect(() => planCppCuteClangWasmBuild(value)).toThrowError(expect.objectContaining({
    code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PLAN-INVALID",
    path: expect.stringContaining(path),
  }));
}

describe("C++/CuTe opaque Clang-WASM build plan", () => {
  it("unwraps the prepared lock and emits its native TableGen stage before its WASM stage", () => {
    const plan = planCppCuteClangWasmBuild(input());

    expect(plan.lockId).toBe(preparedLock.lockId);
    expect(plan.steps.map((step) => step.id)).toEqual([
      "native-tablegen-configure",
      "native-tablegen-build",
      "clang-extractor-wasm-configure",
      "clang-extractor-wasm-build",
    ]);
    expect(plan.steps[0]?.arguments.slice(0, 7)).toEqual([
      "-S", "/source/llvm-project/llvm",
      "-B", "/build/native",
      "-G", "Unix Makefiles",
      "-DCMAKE_MAKE_PROGRAM=/usr/bin/make",
    ]);
    expect(plan.steps[1]?.arguments).toEqual([
      "--build", "/build/native", "--target", "clang-tblgen", "llvm-tblgen", "--parallel", "1",
    ]);
    expect(plan.steps[3]?.arguments).toEqual([
      "--build", "/build/wasm", "--target", "browsergrad-cpp-cute-extractor", "--parallel", "1",
    ]);
  });

  it("materializes only exact tool/source bindings and keeps native TableGen explicit", () => {
    const plan = planCppCuteClangWasmBuild(input());
    const native = plan.steps[0]!;
    const wasm = plan.steps[2]!;

    expect(definition(native, "CMAKE_C_COMPILER")).toBe("-DCMAKE_C_COMPILER=/tools/emsdk/upstream/bin/clang");
    expect(definition(wasm, "CMAKE_TOOLCHAIN_FILE"))
      .toBe("-DCMAKE_TOOLCHAIN_FILE=/tools/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake");
    expect(definition(wasm, "LLVM_NATIVE_TOOL_DIR")).toBe("-DLLVM_NATIVE_TOOL_DIR=/build/native/bin");
    expect(definition(wasm, "LLVM_TABLEGEN")).toBe("-DLLVM_TABLEGEN=/build/native/bin/llvm-tblgen");
    expect(definition(wasm, "CLANG_TABLEGEN")).toBe("-DCLANG_TABLEGEN=/build/native/bin/clang-tblgen");
    expect(definition(wasm, "BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH"))
      .toBe("-DBROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH=/state/clang-wasm-a/evidence/generated/clang-extractor.mjs");
    expect(plan.nativeTools).toEqual({
      clangTablegen: "/build/native/bin/clang-tblgen",
      llvmTablegen: "/build/native/bin/llvm-tblgen",
    });
  });

  it("writes build evidence under stateRoot rather than the distributed output root", () => {
    const plan = planCppCuteClangWasmBuild(input());
    const linkerFlags = definition(plan.steps[2]!, "CMAKE_EXE_LINKER_FLAGS");

    expect(linkerFlags).toContain("-Wl,--Map=/state/clang-wasm-a/evidence/clang-extractor.link.map");
    expect(linkerFlags).not.toContain("-Wl,--Map=/release/assets/");
    expect(JSON.stringify(plan)).not.toMatch(/@BUILD_EVIDENCE@|@OUTPUT@/u);
    expect(plan.generatedExtractor).toEqual({
      factoryModulePath: "/state/clang-wasm-a/evidence/generated/clang-extractor.mjs",
      wasmSidecarPath: "/state/clang-wasm-a/evidence/generated/clang-extractor.wasm",
      distributedWasmPath: "/release/assets/browsergrad-cpp-cute/clang-extractor.wasm",
      distributedMaterializationReady: false,
      materializationBlockerId: "browsergrad-extractor-distributed-materialization",
      workerBundleReady: false,
      blockerId: "browsergrad-worker-emscripten-factory-bundle",
    });
  });

  it("materializes every lock-pinned extractor source path without granting source verification", () => {
    const plan = planCppCuteClangWasmBuild(input());

    expect(plan.extractorSource).toMatchObject({
      sourceSetSha256: preparedLock.extractorSourceSetSha256,
      buildVerified: false,
      blockerId: "browsergrad-extractor-source-verification",
    });
    expect(plan.extractorSource.files.map(({ path, absolutePath }) => ({
      path,
      absolutePath,
    }))).toEqual([
      "BrowserGradCppCuteArtifactV3.cpp",
      "BrowserGradCppCuteArtifactV3.h",
      "BrowserGradCppCuteArtifactWriter.cpp",
      "BrowserGradCppCuteArtifactWriter.h",
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
    ].map((path) => ({
      path,
      absolutePath: `/source/browsergrad-extractor/${path}`,
    })));
  });

  it("prefix-maps every selected source/build/output root", () => {
    const plan = planCppCuteClangWasmBuild(input());
    const wasm = plan.steps[2]!;
    const flags = `${definition(wasm, "CMAKE_C_FLAGS")} ${definition(wasm, "CMAKE_CXX_FLAGS")}`;

    for (const root of [
      "/source/llvm-project",
      "/source/browsergrad-extractor",
      "/build/native",
      "/build/wasm",
      "/release/assets",
    ]) {
      expect(flags).toContain(`-fdebug-prefix-map=${root}=`);
      expect(flags).toContain(`-ffile-prefix-map=${root}=`);
      expect(flags).toContain(`-fmacro-prefix-map=${root}=`);
    }
  });

  it("constructs a closed environment without ambient credentials, package managers, containers, or network settings", () => {
    process.env.BROWSERGRAD_BUILD_SECRET = "must-not-leak";
    process.env.DOCKER_HOST = "tcp://attacker.invalid";
    process.env.npm_config_registry = "https://attacker.invalid";
    const plan = planCppCuteClangWasmBuild(input());
    const native = plan.steps[0]!.environment;
    const wasm = plan.steps[2]!.environment;

    expect(Object.keys(native).sort()).toEqual([
      "HOME", "LANG", "LC_ALL", "PATH", "PKG_CONFIG_LIBDIR", "PKG_CONFIG_PATH",
      "PYTHONHASHSEED", "SOURCE_DATE_EPOCH", "TMPDIR", "TZ", "ZERO_AR_DATE",
    ]);
    expect(Object.keys(wasm).sort()).toEqual([
      "EM_CACHE", "EM_CONFIG", "HOME", "LANG", "LC_ALL", "PATH", "PKG_CONFIG_LIBDIR",
      "PKG_CONFIG_PATH", "PYTHONHASHSEED", "SOURCE_DATE_EPOCH", "TMPDIR", "TZ", "ZERO_AR_DATE",
    ]);
    expect(wasm).toMatchObject({
      EM_CACHE: "/state/clang-wasm-a/em-cache",
      EM_CONFIG: "/tools/emsdk/.emscripten",
    });
    expect(JSON.stringify(plan)).not.toMatch(/BROWSERGRAD_BUILD_SECRET|attacker\.invalid|DOCKER_HOST|npm_config/u);
  });

  it("rejects copied or caller-fabricated lock lookalikes", () => {
    const selected = input();
    selected.lock = { ...preparedLock } as PreparedCppCuteBrowserBuildInputLock;

    expect(() => planCppCuteClangWasmBuild(selected)).toThrow(/UNVERIFIED/u);
  });

  it.each([
    ["root whitespace", (value: MutableInput) => { value.roots.outputRoot = "/release/bad root"; }, "outputRoot"],
    ["tool semicolon", (value: MutableInput) => { value.tools.cmakeExecutable = "/tools/cmake;inject/bin/cmake"; }, "cmakeExecutable"],
    ["single quote", (value: MutableInput) => { value.tools.buildToolExecutable = "/usr/bin/ma'ke"; }, "buildToolExecutable"],
    ["double quote", (value: MutableInput) => { value.tools.emsdkRoot = "/tools/\"emsdk"; }, "emsdkRoot"],
    ["equals", (value: MutableInput) => { value.tools.emscriptenConfigFile = "/tools/emsdk/config=evil"; }, "emscriptenConfigFile"],
    ["colon", (value: MutableInput) => { value.tools.searchPath[0] = "/tools/cmake:evil/bin"; }, "searchPath[0]"],
    ["dollar expansion", (value: MutableInput) => { value.roots.llvmProjectSourceRoot = "/source/$HOME"; }, "llvmProjectSourceRoot"],
    ["at expansion", (value: MutableInput) => { value.roots.extractorSourceRoot = "/source/@CACHE"; }, "extractorSourceRoot"],
    ["parentheses", (value: MutableInput) => { value.roots.wasmBuildRoot = "/build/(wasm)"; }, "wasmBuildRoot"],
    ["brackets", (value: MutableInput) => { value.roots.stateRoot = "/state/[build]"; }, "stateRoot"],
    ["braces", (value: MutableInput) => { value.roots.nativeBuildRoot = "/build/{native}"; }, "nativeBuildRoot"],
    ["comment marker", (value: MutableInput) => { value.tools.searchPath[0] = "/tools/#cmake/bin"; }, "searchPath[0]"],
  ])("rejects portable-unsafe %s paths", (_name, mutate, path) => {
    const selected = input();
    mutate(selected);
    expectPlannerInvalid(selected, path);
  });

  it.each([
    ["relative source", (value: MutableInput) => { value.roots.llvmProjectSourceRoot = "source/llvm"; }, "llvmProjectSourceRoot"],
    ["source/build overlap", (value: MutableInput) => { value.roots.nativeBuildRoot = "/source/llvm-project/build"; }, "$input.roots"],
    ["tool/build overlap", (value: MutableInput) => { value.roots.wasmBuildRoot = "/tools/emsdk/build"; }, "wasmBuildRoot"],
    ["empty tool path", (value: MutableInput) => { value.tools.searchPath = []; }, "searchPath"],
    ["lock/tool mismatch", (value: MutableInput) => { value.tools.buildToolExecutable = "/tools/make/bin/make"; }, "definitions"],
  ])("rejects %s", (_name, mutate, path) => {
    const selected = input();
    mutate(selected);
    expectPlannerInvalid(selected, path);
  });

  it("is deterministic and returns a deeply frozen plan", () => {
    const first = planCppCuteClangWasmBuild(input());
    const second = planCppCuteClangWasmBuild(input());

    expect(first).toEqual(second);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.steps)).toBe(true);
    expect(Object.isFrozen(first.steps[0]!.arguments)).toBe(true);
    expect(Object.isFrozen(first.steps[0]!.environment)).toBe(true);
  });

  it("uses a typed planning error for path failures", () => {
    const selected = input();
    selected.roots.outputRoot = "/bad path";
    try {
      planCppCuteClangWasmBuild(selected);
      throw new Error("expected planning to fail");
    } catch (error) {
      expect(error).toBeInstanceOf(CppCuteBrowserBuildPlanError);
    }
  });
});
