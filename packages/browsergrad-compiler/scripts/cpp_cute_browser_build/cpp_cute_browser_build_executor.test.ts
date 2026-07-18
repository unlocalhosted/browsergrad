import {
  chmod,
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

const executorFsFaults = vi.hoisted(() => ({
  afterLink: undefined as
    | ((sourcePath: string, destinationPath: string) => Promise<void>)
    | undefined,
  syncDirectoryFailure: undefined as
    | { path: string; remaining: number; error: Error }
    | undefined,
  closeTemporaryFailure: undefined as Error | undefined,
}));

const executorProcessState = vi.hoisted(() => ({
  calls: [] as Array<Readonly<{
    executable: string;
    arguments: readonly string[];
    cwd: string;
    environment: Readonly<Record<string, string>>;
    maximumOutputByteLength: number;
    maximumDurationMs: number;
    signal?: AbortSignal;
    onOutputChunk?: (stream: "stdout" | "stderr", chunk: Uint8Array) => Promise<void>;
  }>>,
  failureIndex: undefined as number | undefined,
  thrownReason: undefined as
    | "cancelled"
    | "output-limit"
    | "output-sink"
    | "spawn"
    | "timeout"
    | undefined,
  afterRun: undefined as ((index: number) => Promise<void>) | undefined,
  factoryPath: undefined as string | undefined,
  linkMapPath: undefined as string | undefined,
  invalidConfiguredTarget: false,
  missingClangInclude: false,
  malformedFactory: false,
  malformedWasm: false,
}));

vi.mock("./cpp_cute_browser_build_executor_fs.mjs", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("./cpp_cute_browser_build_executor_fs.mjs")
  >();
  const base = actual.CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS;
  return {
    CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS: Object.freeze({
      ...base,
      link: async (sourcePath: string, destinationPath: string) => {
        await base.link(sourcePath, destinationPath);
        const afterLink = executorFsFaults.afterLink;
        executorFsFaults.afterLink = undefined;
        await afterLink?.(sourcePath, destinationPath);
      },
      syncDirectory: async (path: string) => {
        const failure = executorFsFaults.syncDirectoryFailure;
        if (failure !== undefined && failure.path === path && failure.remaining > 0) {
          failure.remaining -= 1;
          throw failure.error;
        }
        await base.syncDirectory(path);
      },
      closeFileHandle: async (
        handle: import("node:fs/promises").FileHandle,
        purpose: string,
      ) => {
        await base.closeFileHandle(handle, purpose);
        if (purpose === "temporary-sidecar" &&
            executorFsFaults.closeTemporaryFailure !== undefined) {
          const failure = executorFsFaults.closeTemporaryFailure;
          executorFsFaults.closeTemporaryFailure = undefined;
          throw failure;
        }
      },
    }),
  };
});

vi.mock("./cpp_cute_browser_build_executor_process.mjs", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("./cpp_cute_browser_build_executor_process.mjs")
  >();
  return {
    ...actual,
    CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS: Object.freeze({
      run: async (input: {
        executable: string;
        arguments: readonly string[];
        cwd: string;
        environment: Readonly<Record<string, string>>;
        maximumOutputByteLength: number;
        maximumDurationMs: number;
        signal?: AbortSignal;
        onOutputChunk?: (stream: "stdout" | "stderr", chunk: Uint8Array) => Promise<void>;
      }) => {
        const callIndex = executorProcessState.calls.length;
        executorProcessState.calls.push(Object.freeze({
          ...input,
          arguments: Object.freeze([...input.arguments]),
          environment: Object.freeze({ ...input.environment }),
        }));
        const stdout = new TextEncoder().encode(`stdout ${callIndex}\n`);
        const stderr = new TextEncoder().encode(`stderr ${callIndex}\n`);
        await input.onOutputChunk?.("stdout", stdout);
        await input.onOutputChunk?.("stderr", stderr);
        if (executorProcessState.thrownReason !== undefined) {
          throw new actual.CppCuteBrowserBuildProcessError(
            executorProcessState.thrownReason,
            "injected process-boundary failure",
          );
        }

        const { chmod, mkdir, writeFile } = await import("node:fs/promises");
        const { dirname, join } = await import("node:path");
        if (input.arguments[0] === "--build" &&
            input.arguments.includes("clang-tblgen")) {
          const bin = join(input.cwd, "bin");
          await mkdir(bin, { recursive: true });
          await Promise.all([
            writeFile(join(bin, "clang-tblgen"), "native clang tablegen\n"),
            writeFile(join(bin, "llvm-tblgen"), "native llvm tablegen\n"),
          ]);
          await Promise.all([
            chmod(join(bin, "clang-tblgen"), 0o555),
            chmod(join(bin, "llvm-tblgen"), 0o555),
          ]);
        }
        if (input.arguments[0] === "-S" &&
            input.arguments.some((argument) => argument.includes("BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH"))) {
          executorProcessState.factoryPath = input.arguments
            .find((argument) => argument.startsWith("-DBROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH="))
            ?.split("=").slice(1).join("=");
          const linkerFlags = input.arguments
            .find((argument) => argument.startsWith("-DCMAKE_EXE_LINKER_FLAGS="));
          executorProcessState.linkMapPath = linkerFlags
            ?.match(/--Map=([^ ]+)/u)?.[1];
          const buildRoot = input.arguments[input.arguments.indexOf("-B") + 1];
          const llvmSourceRoot = input.arguments[input.arguments.indexOf("-S") + 1];
          if (buildRoot === undefined || llvmSourceRoot === undefined ||
              executorProcessState.factoryPath === undefined) {
            throw new Error("mock did not observe the Wasm build root and factory path");
          }
          const llvmProjectSourceRoot = dirname(llvmSourceRoot);
          const targetDirectory = join(
            buildRoot,
            "tools",
            "browsergrad_extractor",
            "CMakeFiles",
            "browsergrad-cpp-cute-extractor.dir",
          );
          await mkdir(targetDirectory, { recursive: true });
          await Promise.all([
            writeFile(
              join(buildRoot, "CMakeCache.txt"),
              "LLVM_ENABLE_RTTI:BOOL=ON\n",
            ),
            writeFile(
              join(targetDirectory, "flags.make"),
              [
                executorProcessState.missingClangInclude
                  ? `CXX_INCLUDES = -I${join(buildRoot, "tools", "clang", "include")}`
                  : `CXX_INCLUDES = -I${join(buildRoot, "tools", "clang", "include")} ` +
                    `-I${join(llvmProjectSourceRoot, "clang", "include")}`,
                executorProcessState.invalidConfiguredTarget
                  ? "CXX_FLAGS = -O3 -fno-exceptions -DNDEBUG"
                  : "CXX_FLAGS = -O3 -fexceptions -DNDEBUG",
                "",
              ].join("\n"),
            ),
            writeFile(
              join(targetDirectory, "link.txt"),
              `/emsdk/upstream/emscripten/em++ -O3 -fexceptions objects.o -o ${executorProcessState.factoryPath}\n`,
            ),
          ]);
        }
        if (input.arguments[0] === "--build" &&
            input.arguments.includes("browsergrad-cpp-cute-extractor")) {
          const factoryPath = executorProcessState.factoryPath;
          const linkMapPath = executorProcessState.linkMapPath;
          if (factoryPath === undefined || linkMapPath === undefined) {
            throw new Error("mock did not observe the Wasm configure outputs");
          }
          await Promise.all([
            writeFile(
              factoryPath,
              executorProcessState.malformedFactory
                ? Uint8Array.of(0xff)
                : "const createBrowserGradCppCuteExtractor = () => {}; export default createBrowserGradCppCuteExtractor;\n",
            ),
            writeFile(
              factoryPath.replace(/\.mjs$/u, ".wasm"),
              executorProcessState.malformedWasm
                ? Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8)
                : Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00),
            ),
            writeFile(linkMapPath, "browsergrad link map\n"),
          ]);
        }
        await executorProcessState.afterRun?.(callIndex);
        return Object.freeze({
          exitCode: executorProcessState.failureIndex === callIndex ? 7 : 0,
          terminationSignal: null,
          stdout: input.onOutputChunk === undefined ? stdout : new Uint8Array(),
          stderr: input.onOutputChunk === undefined ? stderr : new Uint8Array(),
          stdoutByteLength: stdout.byteLength,
          stderrByteLength: stderr.byteLength,
        });
      },
    }),
  };
});

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CppCuteBrowserBuildExecutorError,
  executeCppCuteClangWasmBuild,
  materializeCppCuteClangWasmSidecar,
  prepareCppCuteClangWasmBuildSource,
  type PrepareCppCuteClangWasmBuildSourceInput,
} from "./cpp_cute_browser_build_executor.mjs";
import { planCppCuteClangWasmBuild } from "./cpp_cute_browser_build_plan.mjs";

const checkedInExtractorRoot = join(
  dirname(fileURLToPath(import.meta.url)),
  "extractor",
);
const temporaryRoots: string[] = [];
const MINIMAL_WASM = Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);

let lock: PreparedCppCuteBrowserBuildInputLock;

beforeAll(async () => {
  lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
});

afterEach(async () => {
  executorFsFaults.afterLink = undefined;
  executorFsFaults.syncDirectoryFailure = undefined;
  executorFsFaults.closeTemporaryFailure = undefined;
  executorProcessState.calls.splice(0);
  executorProcessState.failureIndex = undefined;
  executorProcessState.thrownReason = undefined;
  executorProcessState.afterRun = undefined;
  executorProcessState.factoryPath = undefined;
  executorProcessState.linkMapPath = undefined;
  executorProcessState.invalidConfiguredTarget = false;
  executorProcessState.missingClangInclude = false;
  executorProcessState.malformedFactory = false;
  executorProcessState.malformedWasm = false;
  await Promise.all(temporaryRoots.splice(0).map(async (root) => {
    await chmod(join(root, "staged-extractor-source"), 0o700).catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }));
});

async function fixture(materializeBuildInputs = false): Promise<{
  readonly root: string;
  readonly input: PrepareCppCuteClangWasmBuildSourceInput;
}> {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-clang-wasm-executor-"));
  temporaryRoots.push(root);
  const sourceRoot = join(root, "source-input");
  const outputRoot = join(root, "output");
  await Promise.all([mkdir(sourceRoot), mkdir(outputRoot)]);
  const sourcePaths = unwrapPreparedCppCuteBrowserBuildInputLock(lock)
    .lock.body.recipe.extractorSource.files.map((file) => file.path);
  await Promise.all(sourcePaths.map((path) => copyFile(
    join(checkedInExtractorRoot, path),
    join(sourceRoot, path),
  )));
  const tools = materializeBuildInputs
    ? await materializedTools(root)
    : {
        cmakeExecutable: "/tools/cmake/bin/cmake",
        buildToolExecutable: "/usr/bin/make",
        emsdkRoot: "/tools/emsdk",
        emscriptenToolchainFile:
          "/tools/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake",
        emscriptenConfigFile: "/tools/emsdk/.emscripten",
        searchPath: [
          "/tools/cmake/bin",
          "/tools/emsdk/upstream/bin",
          "/tools/emsdk/upstream/emscripten",
          "/tools/python/bin",
          "/tools/node/bin",
        ],
      };
  const llvmProjectSourceRoot = join(root, "llvm-source");
  if (materializeBuildInputs) {
    await mkdir(join(llvmProjectSourceRoot, "llvm"), { recursive: true });
  }
  return {
    root,
    input: {
      lock,
      tools,
      roots: {
        llvmProjectSourceRoot,
        extractorSourceRoot: join(root, "staged-extractor-source"),
        nativeBuildRoot: join(root, "native-build"),
        wasmBuildRoot: join(root, "wasm-build"),
        outputRoot,
        stateRoot: join(root, "state"),
      },
      extractorSourceInputRoot: sourceRoot,
    },
  };
}

async function materializedTools(root: string) {
  const toolsRoot = join(root, "tools");
  const cmakeDirectory = join(toolsRoot, "cmake", "bin");
  const emsdkRoot = join(toolsRoot, "emsdk");
  const emscriptenDirectory = join(emsdkRoot, "upstream", "emscripten");
  const upstreamBin = join(emsdkRoot, "upstream", "bin");
  const pythonBin = join(toolsRoot, "python", "bin");
  const nodeBin = join(toolsRoot, "node", "bin");
  const emscriptenToolchainFile = join(
    emscriptenDirectory,
    "cmake",
    "Modules",
    "Platform",
    "Emscripten.cmake",
  );
  await Promise.all([
    mkdir(cmakeDirectory, { recursive: true }),
    mkdir(dirname(emscriptenToolchainFile), { recursive: true }),
    mkdir(upstreamBin, { recursive: true }),
    mkdir(pythonBin, { recursive: true }),
    mkdir(nodeBin, { recursive: true }),
  ]);
  const cmakeExecutable = join(cmakeDirectory, "cmake");
  const buildToolExecutable = "/usr/bin/make";
  const emscriptenConfigFile = join(emsdkRoot, ".emscripten");
  await Promise.all([
    writeFile(cmakeExecutable, "mock cmake\n"),
    writeFile(emscriptenToolchainFile, "mock emscripten toolchain\n"),
    writeFile(emscriptenConfigFile, "mock emscripten config\n"),
  ]);
  await Promise.all([
    chmod(cmakeExecutable, 0o555),
    chmod(emscriptenToolchainFile, 0o444),
    chmod(emscriptenConfigFile, 0o444),
  ]);
  return {
    cmakeExecutable,
    buildToolExecutable,
    emsdkRoot,
    emscriptenToolchainFile,
    emscriptenConfigFile,
    searchPath: [
      cmakeDirectory,
      upstreamBin,
      emscriptenDirectory,
      pythonBin,
      nodeBin,
    ],
  };
}

async function writeGeneratedSidecar(
  input: PrepareCppCuteClangWasmBuildSourceInput,
  bytes: Uint8Array = MINIMAL_WASM,
): Promise<void> {
  const generated = join(input.roots.stateRoot, "evidence", "generated");
  await mkdir(generated, { recursive: true });
  await Promise.all([
    writeFile(join(generated, "clang-extractor.wasm"), bytes),
    writeFile(join(generated, "clang-extractor.mjs"), "export default 'factory';\n"),
  ]);
}

function destination(input: PrepareCppCuteClangWasmBuildSourceInput): string {
  return join(input.roots.outputRoot, "browsergrad-cpp-cute", "clang-extractor.wasm");
}

describe("bounded Clang-WASM build source executor", () => {
  it("snapshots the exact locked source set into an owned read-only staging root", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);

    expect(prepared).toMatchObject({
      authority: "build-source-snapshot-only",
      lockId: lock.lockId,
      sourceSetSha256: lock.extractorSourceSetSha256,
      fileCount: 37,
      sourceVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
    });
    expect((await readdir(input.roots.extractorSourceRoot)).sort()).toEqual([
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
    const stagedCpp = join(input.roots.extractorSourceRoot, "BrowserGradCppCuteExtractor.cpp");
    const before = await readFile(stagedCpp);
    await writeFile(join(input.extractorSourceInputRoot, "BrowserGradCppCuteExtractor.cpp"), "mutated after snapshot");
    expect(await readFile(stagedCpp)).toEqual(before);
    expect((await lstat(stagedCpp)).mode & 0o222).toBe(0);
  });

  it.each([
    ["extra entry", async (input: PrepareCppCuteClangWasmBuildSourceInput) => {
      await writeFile(join(input.extractorSourceInputRoot, "extra.cpp"), "extra");
    }, "extractorSourceInputRoot"],
    ["missing entry", async (input: PrepareCppCuteClangWasmBuildSourceInput) => {
      await rm(join(input.extractorSourceInputRoot, "CMakeLists.txt"));
    }, "extractorSourceInputRoot"],
    ["symlink entry", async (input: PrepareCppCuteClangWasmBuildSourceInput) => {
      const path = join(input.extractorSourceInputRoot, "CMakeLists.txt");
      await rm(path);
      await symlink(join(checkedInExtractorRoot, "CMakeLists.txt"), path);
    }, "CMakeLists.txt"],
  ])("rejects %s without leaving staged bytes", async (_name, mutate, expectedPath) => {
    const { input } = await fixture();
    await mutate(input);

    await expect(prepareCppCuteClangWasmBuildSource(input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: expect.stringContaining(expectedPath),
    });
    await expect(lstat(input.roots.extractorSourceRoot)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rejects wrong length and same-length wrong content before staging", async () => {
    const first = await fixture();
    await writeFile(join(first.input.extractorSourceInputRoot, "CMakeLists.txt"), "short");
    await expect(prepareCppCuteClangWasmBuildSource(first.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-HASH-MISMATCH",
      path: expect.stringContaining("byteLength"),
    });

    const second = await fixture();
    const cppPath = join(second.input.extractorSourceInputRoot, "BrowserGradCppCuteExtractor.cpp");
    const cpp = await readFile(cppPath);
    cpp[0] = cpp[0] === 0 ? 1 : 0;
    await writeFile(cppPath, cpp);
    await expect(prepareCppCuteClangWasmBuildSource(second.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-HASH-MISMATCH",
      path: expect.stringContaining("sha256"),
    });
  });

  it("rejects an existing staging root and an already-aborted operation", async () => {
    const conflict = await fixture();
    await mkdir(conflict.input.roots.extractorSourceRoot);
    await expect(prepareCppCuteClangWasmBuildSource(conflict.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.roots.extractorSourceRoot",
    });

    const aborted = await fixture();
    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteClangWasmBuildSource(aborted.input, {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });
  });

  it("requires a private staging parent and binds the staged root inode", async () => {
    const unsafeParent = await fixture();
    await chmod(unsafeParent.root, 0o777);
    await expect(prepareCppCuteClangWasmBuildSource(unsafeParent.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.roots.extractorSourceRoot.parent",
    });

    const replaced = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(replaced.input);
    await writeGeneratedSidecar(replaced.input);
    const displaced = join(replaced.root, "displaced-staged-source");
    await chmod(replaced.input.roots.extractorSourceRoot, 0o700);
    await rename(replaced.input.roots.extractorSourceRoot, displaced);
    await mkdir(replaced.input.roots.extractorSourceRoot, { mode: 0o700 });
    const sentinel = join(replaced.input.roots.extractorSourceRoot, "sentinel.txt");
    await writeFile(sentinel, "replacement must survive");

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.stagedSourceRoot",
    });
    expect(await readFile(sentinel, "utf8")).toBe("replacement must survive");
  });
});

describe("exact Clang-Wasm build executor", () => {
  it("runs the four lock-derived steps and seals bounded build evidence", async () => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    const expectedPlan = planCppCuteClangWasmBuild({
      lock: input.lock,
      tools: input.tools,
      roots: input.roots,
    });

    const executed = await executeCppCuteClangWasmBuild(prepared);

    expect(executed).toMatchObject({
      authority: "clang-wasm-build-execution-observation-only",
      lockId: lock.lockId,
      sourceSetSha256: lock.extractorSourceSetSha256,
      stepCount: 4,
      sourceVerified: true,
      buildExecuted: true,
      factoryModuleUtf8Validated: true,
      webAssemblyValidated: true,
      abiConformanceVerified: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
      factoryModuleDistributed: false,
    });
    expect(executorProcessState.calls).toHaveLength(4);
    expect(executorProcessState.calls.map((call) => ({
      executable: call.executable,
      arguments: call.arguments,
      cwd: call.cwd,
      environment: call.environment,
    }))).toEqual(expectedPlan.steps.map((step) => ({
      executable: step.executable,
      arguments: step.arguments,
      cwd: step.cwd,
      environment: step.environment,
    })));
    expect(executorProcessState.calls.every((call) => (
      call.maximumOutputByteLength === 16 * 1024 * 1024 &&
      call.maximumDurationMs === 4 * 60 * 60 * 1_000
    ))).toBe(true);
    expect(executed.steps.map((step) => step.id)).toEqual(expectedPlan.steps.map((step) => step.id));
    expect(executed.steps.map((step) => ({
      executable: step.executable,
      arguments: step.arguments,
      cwd: step.cwd,
      environment: step.environment,
    }))).toEqual(expectedPlan.steps.map((step) => ({
      executable: step.executable,
      arguments: step.arguments,
      cwd: step.cwd,
      environment: step.environment,
    })));
    expect(executed.paths).toEqual(input.roots);
    for (const [index, step] of executed.steps.entries()) {
      expect(await readFile(step.stdoutPath, "utf8")).toBe(`stdout ${index}\n`);
      expect(await readFile(step.stderrPath, "utf8")).toBe(`stderr ${index}\n`);
      expect((await lstat(step.stdoutPath)).mode & 0o222).toBe(0);
      expect((await lstat(step.stderrPath)).mode & 0o222).toBe(0);
    }
    expect((await lstat(executed.factoryModulePath)).mode & 0o222).toBe(0);
    expect((await lstat(executed.wasmSidecarPath)).mode & 0o222).toBe(0);
    expect((await lstat(executed.linkMapPath)).mode & 0o222).toBe(0);
    expect(executed.factoryModuleSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(executed.wasmSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(executed.linkMapSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(executed.nativeTools).toEqual({
      clangTablegen: {
        path: expectedPlan.nativeTools.clangTablegen,
        sha256: "ca0dae392791b04184cde603e2b45357bb46005bab79a1411768e6626b292825",
        byteLength: 22,
      },
      llvmTablegen: {
        path: expectedPlan.nativeTools.llvmTablegen,
        sha256: "d408810ec59188c5ef1e7ca809630c6fbdc46ac87a37d5b6f816008e2fa2acaf",
        byteLength: 21,
      },
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("persists failed-step logs and halts before later build steps", async () => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    executorProcessState.failureIndex = 1;

    await expect(executeCppCuteClangWasmBuild(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-BUILD-FAILED",
      path: "$.steps[1]",
    });
    expect(executorProcessState.calls).toHaveLength(2);
    const logRoot = join(input.roots.stateRoot, "evidence", "build-logs");
    expect((await readdir(logRoot)).sort()).toEqual([
      "native-tablegen-build.stderr.log",
      "native-tablegen-build.stdout.log",
      "native-tablegen-configure.stderr.log",
      "native-tablegen-configure.stdout.log",
    ]);
  });

  it("rejects target-level exception drift before the expensive Wasm build", async () => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    executorProcessState.invalidConfiguredTarget = true;

    await expect(executeCppCuteClangWasmBuild(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.configuredTarget",
    });
    expect(executorProcessState.calls).toHaveLength(3);
    const logs = await readdir(join(input.roots.stateRoot, "evidence", "build-logs"));
    expect(logs).not.toContain("clang-extractor-wasm-build.stdout.log");
    expect(logs).not.toContain("clang-extractor-wasm-build.stderr.log");
  });

  it("rejects missing Clang include wiring before the expensive Wasm build", async () => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    executorProcessState.missingClangInclude = true;

    await expect(executeCppCuteClangWasmBuild(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.configuredTarget",
    });
    expect(executorProcessState.calls).toHaveLength(3);
    const logs = await readdir(join(input.roots.stateRoot, "evidence", "build-logs"));
    expect(logs).not.toContain("clang-extractor-wasm-build.stdout.log");
    expect(logs).not.toContain("clang-extractor-wasm-build.stderr.log");
  });

  it.each([
    ["output-limit", "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-RESOURCE-LIMIT", "output"],
    ["output-sink", "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-IO", "output"],
    ["timeout", "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-RESOURCE-LIMIT", "duration"],
    ["spawn", "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-IO", "steps[0]"],
    ["cancelled", "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED", "signal"],
  ] as const)("maps %s process-boundary failures into typed executor failures", async (
    reason,
    code,
    expectedPath,
  ) => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    executorProcessState.thrownReason = reason;

    await expect(executeCppCuteClangWasmBuild(prepared)).rejects.toMatchObject({
      code,
      path: expect.stringContaining(expectedPath),
    });
    expect(executorProcessState.calls).toHaveLength(1);
    const logRoot = join(input.roots.stateRoot, "evidence", "build-logs");
    expect(await readFile(join(logRoot, "native-tablegen-configure.stdout.log"), "utf8"))
      .toBe("stdout 0\n");
    expect(await readFile(join(logRoot, "native-tablegen-configure.stderr.log"), "utf8"))
      .toBe("stderr 0\n");
    expect((await lstat(join(logRoot, "native-tablegen-configure.stdout.log"))).mode & 0o222)
      .toBe(0);
  });

  it("detects admitted build-tool mutation between exact steps", async () => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    executorProcessState.afterRun = async (index) => {
      if (index !== 0) return;
      await chmod(input.tools.cmakeExecutable, 0o755);
      await writeFile(input.tools.cmakeExecutable, "mutated cmake\n");
      await chmod(input.tools.cmakeExecutable, 0o555);
    };

    await expect(executeCppCuteClangWasmBuild(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.tools.cmakeExecutable",
    });
    expect(executorProcessState.calls).toHaveLength(1);
  });

  it("rejects writable build inputs when no read-only mount protects them", async () => {
    const { input } = await fixture(true);
    const writableSearchDirectory = input.tools.searchPath[2];
    if (writableSearchDirectory === undefined) throw new Error("missing search fixture");
    await chmod(writableSearchDirectory, 0o777);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);

    await expect(executeCppCuteClangWasmBuild(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.tools.emscriptenToolchainFile",
    });
    expect(executorProcessState.calls).toHaveLength(0);
  });

  it("rejects malformed factory and Wasm outputs after all exact steps", async () => {
    const malformedFactory = await fixture(true);
    const preparedFactory = await prepareCppCuteClangWasmBuildSource(malformedFactory.input);
    executorProcessState.malformedFactory = true;
    await expect(executeCppCuteClangWasmBuild(preparedFactory)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.generatedExtractor.factoryModulePath",
    });

    executorProcessState.calls.splice(0);
    executorProcessState.factoryPath = undefined;
    executorProcessState.linkMapPath = undefined;
    executorProcessState.malformedFactory = false;
    executorProcessState.malformedWasm = true;
    const malformedWasm = await fixture(true);
    const preparedWasm = await prepareCppCuteClangWasmBuildSource(malformedWasm.input);
    await expect(executeCppCuteClangWasmBuild(preparedWasm)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.generatedExtractor.wasmSidecarPath",
    });
  });

  it("requires opaque prepared authority and honors pre-spawn cancellation", async () => {
    const { input } = await fixture(true);
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await expect(executeCppCuteClangWasmBuild({ ...prepared })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-UNVERIFIED",
    });

    const controller = new AbortController();
    controller.abort();
    await expect(executeCppCuteClangWasmBuild(prepared, {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });
    expect(executorProcessState.calls).toHaveLength(0);
  });
});

describe("deterministic Clang-WASM sidecar materializer", () => {
  it("atomically installs exact Wasm bytes, is idempotent, and never distributes the factory", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);

    const first = await materializeCppCuteClangWasmSidecar(prepared);
    const second = await materializeCppCuteClangWasmSidecar(prepared);

    expect(second).toEqual(first);
    expect(first).toMatchObject({
      authority: "wasm-sidecar-byte-materialization-observation-only",
      lockId: lock.lockId,
      sourceSetSha256: lock.extractorSourceSetSha256,
      wasmByteLength: MINIMAL_WASM.byteLength,
      distributedWasmPath: destination(input),
      sidecarBytesMaterialized: true,
      webAssemblyValidated: false,
      abiConformanceVerified: false,
      sourceVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
      factoryModuleDistributed: false,
    });
    expect(first.generatedWasmSha256).toBe(first.distributedWasmSha256);
    expect([...(await readFile(destination(input)))]).toEqual([...MINIMAL_WASM]);
    await expect(lstat(join(
      input.roots.outputRoot,
      "browsergrad-cpp-cute",
      "clang-extractor.mjs",
    ))).rejects.toMatchObject({ code: "ENOENT" });
    expect((await lstat(destination(input))).mode & 0o222).toBe(0);
  });

  it("rejects conflicting or symlink destinations without replacing them", async () => {
    const conflict = await fixture();
    const preparedConflict = await prepareCppCuteClangWasmBuildSource(conflict.input);
    await writeGeneratedSidecar(conflict.input);
    await mkdir(dirname(destination(conflict.input)), { recursive: true });
    await writeFile(destination(conflict.input), Uint8Array.of(1, 2, 3));
    await chmod(destination(conflict.input), 0o444);
    await expect(materializeCppCuteClangWasmSidecar(preparedConflict)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.distributedWasmPath",
    });
    expect([...(await readFile(destination(conflict.input)))]).toEqual([1, 2, 3]);

    const linked = await fixture();
    const preparedLinked = await prepareCppCuteClangWasmBuildSource(linked.input);
    await writeGeneratedSidecar(linked.input);
    await mkdir(dirname(destination(linked.input)), { recursive: true });
    const linkTarget = join(linked.root, "link-target.wasm");
    await writeFile(linkTarget, MINIMAL_WASM);
    await symlink(linkTarget, destination(linked.input));
    await expect(materializeCppCuteClangWasmSidecar(preparedLinked)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.distributedWasmPath",
    });
    expect([...(await readFile(linkTarget))]).toEqual([...MINIMAL_WASM]);
  });

  it("rejects writable identical destinations and unsafe output roots", async () => {
    const writable = await fixture();
    const preparedWritable = await prepareCppCuteClangWasmBuildSource(writable.input);
    await writeGeneratedSidecar(writable.input);
    await mkdir(dirname(destination(writable.input)), { recursive: true });
    await writeFile(destination(writable.input), MINIMAL_WASM, { mode: 0o644 });
    await expect(materializeCppCuteClangWasmSidecar(preparedWritable)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.distributedWasmPath",
    });
    expect([...(await readFile(destination(writable.input)))]).toEqual([...MINIMAL_WASM]);

    const unsafe = await fixture();
    const preparedUnsafe = await prepareCppCuteClangWasmBuildSource(unsafe.input);
    await writeGeneratedSidecar(unsafe.input);
    await chmod(unsafe.input.roots.outputRoot, 0o777);
    await expect(materializeCppCuteClangWasmSidecar(preparedUnsafe)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.outputRoot",
    });
    await expect(lstat(destination(unsafe.input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rejects a private output root beneath an untrusted writable ancestor", async () => {
    const { root, input } = await fixture();
    const unsafeAncestor = join(root, "unsafe-ancestor");
    const privateOutputRoot = join(unsafeAncestor, "private-output");
    await mkdir(privateOutputRoot, { recursive: true, mode: 0o700 });
    await chmod(unsafeAncestor, 0o777);
    (input.roots as { outputRoot: string }).outputRoot = privateOutputRoot;
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.outputRoot",
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rechecks the exact staged closure after source-byte verification", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    await chmod(input.roots.extractorSourceRoot, 0o700);
    await writeFile(join(input.roots.extractorSourceRoot, "undeclared.cpp"), "undeclared");
    await chmod(input.roots.extractorSourceRoot, 0o555);

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: expect.stringContaining("$.stagedSourceRoot"),
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("binds every staged source file to the inode captured during preparation", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const stagedFile = join(input.roots.extractorSourceRoot, "CMakeLists.txt");
    const identicalBytes = await readFile(stagedFile);
    await chmod(input.roots.extractorSourceRoot, 0o700);
    await rm(stagedFile);
    await writeFile(stagedFile, identicalBytes, { mode: 0o444 });
    await chmod(input.roots.extractorSourceRoot, 0o555);

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: expect.stringContaining("CMakeLists.txt"),
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rejects malformed generated bytes, forged source authority, and cancellation", async () => {
    const malformed = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(malformed.input);
    await writeGeneratedSidecar(malformed.input, Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8));
    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.generatedWasmPath",
    });
    const outputDirectory = dirname(destination(malformed.input));
    await expect(lstat(outputDirectory)).rejects.toMatchObject({ code: "ENOENT" });

    await expect(materializeCppCuteClangWasmSidecar({ ...prepared })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-UNVERIFIED",
    });

    await writeGeneratedSidecar(malformed.input);
    const controller = new AbortController();
    controller.abort();
    await expect(materializeCppCuteClangWasmSidecar(prepared, {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });

    const inFlight = await fixture();
    const inFlightPrepared = await prepareCppCuteClangWasmBuildSource(inFlight.input);
    await writeGeneratedSidecar(inFlight.input);
    const inFlightController = new AbortController();
    const pending = materializeCppCuteClangWasmSidecar(inFlightPrepared, {
      signal: inFlightController.signal,
    });
    inFlightController.abort();
    await expect(pending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });
    await expect(lstat(destination(inFlight.input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("leaves no temporary output after a failed destination admission", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const outputDirectory = dirname(destination(input));
    await mkdir(outputDirectory, { recursive: true });
    await writeFile(destination(input), Uint8Array.of(9));

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toBeInstanceOf(
      CppCuteBrowserBuildExecutorError,
    );
    expect((await readdir(outputDirectory)).filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });

  it("rolls back the exact linked inode when post-link durability fails", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const outputDirectory = dirname(destination(input));
    executorFsFaults.syncDirectoryFailure = {
      path: outputDirectory,
      remaining: 1,
      error: new Error("injected post-link directory sync failure"),
    };

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-IO",
      path: "$.distributedWasmPath",
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
    expect((await readdir(outputDirectory)).filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });

  it("refuses destructive rollback after a post-link destination identity swap", async () => {
    const { root, input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const displaced = join(root, "displaced-linked-sidecar.wasm");
    let replacementIdentity: { dev: bigint; ino: bigint } | undefined;
    executorFsFaults.afterLink = async (_sourcePath, destinationPath) => {
      await rename(destinationPath, displaced);
      await writeFile(destinationPath, MINIMAL_WASM, { mode: 0o444 });
      const stat = await lstat(destinationPath, { bigint: true });
      replacementIdentity = { dev: stat.dev, ino: stat.ino };
    };

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CLEANUP",
      path: "$.distributedWasmPath",
    });
    const surviving = await lstat(destination(input), { bigint: true });
    expect({ dev: surviving.dev, ino: surviving.ino }).toEqual(replacementIdentity);
    expect([...(await readFile(destination(input)))]).toEqual([...MINIMAL_WASM]);
    expect((await readdir(dirname(destination(input))))
      .filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });

  it("settles temp removal and reports a typed composite close failure", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const outputDirectory = dirname(destination(input));
    executorFsFaults.syncDirectoryFailure = {
      path: outputDirectory,
      remaining: 1,
      error: new Error("injected post-link directory sync failure"),
    };
    executorFsFaults.closeTemporaryFailure = new Error("injected temporary close failure");

    const observedError: unknown = await materializeCppCuteClangWasmSidecar(prepared)
      .then(() => undefined, (error: unknown) => error);
    expect(observedError).toBeInstanceOf(CppCuteBrowserBuildExecutorError);
    if (!(observedError instanceof CppCuteBrowserBuildExecutorError)) {
      throw new Error("expected typed executor cleanup failure");
    }
    expect(observedError).toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CLEANUP",
      path: "$.distributedWasmPath",
    });
    expect(observedError.cause).toBeInstanceOf(AggregateError);
    if (!(observedError.cause instanceof AggregateError)) {
      throw new Error("expected aggregate operation and cleanup causes");
    }
    expect(observedError.cause.errors).toHaveLength(2);
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
    expect((await readdir(outputDirectory)).filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });
});
