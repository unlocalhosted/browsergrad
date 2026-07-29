import { existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { runNativeTestProcess } from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const extractorRoot = join(scriptRoot, "extractor");
const nativeSource = join(scriptRoot, "cpp_cute_browser_clang_pass_native_test.cpp");
const llvmConfigCandidates = [
  process.env["BROWSERGRAD_LLVM_CONFIG"],
  "/opt/homebrew/opt/llvm/bin/llvm-config",
  "/usr/local/opt/llvm/bin/llvm-config",
  "/usr/bin/llvm-config",
].filter((candidate): candidate is string => candidate !== undefined);
const llvmConfig = llvmConfigCandidates.find((candidate) => existsSync(candidate));
const requireNativeClangPass = process.env["BROWSERGRAD_REQUIRE_NATIVE_CLANG_PASS"] === "1";
const expectedClangVersion = "22.1.8";

interface NativeClangToolchain {
  readonly compiler: string;
  readonly includeDirectory: string;
  readonly libraryDirectory: string;
  readonly llvmLibrary: string;
  readonly sdkRoot?: string;
}

async function discoverNativeClangToolchain(): Promise<NativeClangToolchain | undefined> {
  if (llvmConfig === undefined) return undefined;
  const query = async (argument: string): Promise<string | undefined> => {
    const result = await runNativeTestProcess(llvmConfig, [argument], {
      encoding: "utf8",
      timeout: 10_000,
    });
    return result.status === 0 ? result.stdout.trim() : undefined;
  };
  const [bindir, includeDirectory, libraryDirectory, version] = await Promise.all([
    query("--bindir"),
    query("--includedir"),
    query("--libdir"),
    query("--version"),
  ]);
  if (bindir === undefined || includeDirectory === undefined ||
      libraryDirectory === undefined || version !== expectedClangVersion) {
    return undefined;
  }
  const compiler = join(bindir, "clang++");
  const clangLibrary = join(libraryDirectory, "libclang-cpp.dylib");
  const linuxClangLibrary = join(libraryDirectory, "libclang-cpp.so");
  if (!existsSync(compiler) ||
      (!existsSync(clangLibrary) && !existsSync(linuxClangLibrary))) {
    return undefined;
  }
  const sdkRoot = process.platform === "darwin"
    ? await (async () => {
        const result = await runNativeTestProcess(
          "/usr/bin/xcrun",
          ["--show-sdk-path"],
          { encoding: "utf8", timeout: 10_000 },
        );
        const candidate = result.status === 0
          ? result.stdout.trim()
          : "";
        try {
          return isAbsolute(candidate) && statSync(candidate).isDirectory()
            ? candidate
            : undefined;
        } catch {
          return undefined;
        }
      })()
    : undefined;
  if (process.platform === "darwin" && sdkRoot === undefined) {
    return undefined;
  }
  return {
    compiler,
    includeDirectory,
    libraryDirectory,
    llvmLibrary: `-lLLVM-${version.split(".")[0]}`,
    ...(sdkRoot === undefined ? {} : { sdkRoot }),
  };
}

const toolchain = await discoverNativeClangToolchain();

async function compileAndRun(): Promise<void> {
  if (toolchain === undefined) throw new Error("Clang development toolchain unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-clang-pass-"));
  const executable = join(workingDirectory, "clang-pass-native-test");
  try {
    const compilation = await runNativeTestProcess(toolchain.compiler, [
      "-std=c++20",
      "-O1",
      "-Wall",
      "-Wextra",
      "-Wpedantic",
      "-Werror",
      "-DBG_CPP_CUTE_METRICS_TESTING=1",
      "-I", scriptRoot,
      "-I", extractorRoot,
      "-isystem", toolchain.includeDirectory,
      ...(toolchain.sdkRoot === undefined
        ? []
        : ["-isysroot", toolchain.sdkRoot]),
      nativeSource,
      join(extractorRoot, "BrowserGradCppCuteClangAction.cpp"),
      join(extractorRoot, "BrowserGradCppCuteMetrics.cpp"),
      join(extractorRoot, "BrowserGradCppCutePreprocessorPolicy.cpp"),
      join(extractorRoot, "BrowserGradCppCuteVirtualPath.cpp"),
      `-L${toolchain.libraryDirectory}`,
      `-Wl,-rpath,${toolchain.libraryDirectory}`,
      "-lclang-cpp",
      toolchain.llvmLibrary,
      "-o", executable,
    ], { encoding: "utf8", timeout: 120_000 });
    expect(compilation.error).toBeUndefined();
    expect(compilation.status, compilation.stderr).toBe(0);

    const execution = await runNativeTestProcess(executable, [], {
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(execution.error).toBeUndefined();
    expect(
      execution.status,
      `signal=${execution.signal ?? "none"}\n${execution.stdout}\n${execution.stderr}`,
    ).toBe(0);
  } finally {
    rmSync(workingDirectory, { recursive: true, force: true });
  }
}

describe("native Clang C++/CuTe semantic passes", () => {
  it.skipIf(toolchain === undefined && !requireNativeClangPass)(
    "runs isolated CUDA device and host actions and resets temporal policy",
    compileAndRun,
    180_000,
  );
});
