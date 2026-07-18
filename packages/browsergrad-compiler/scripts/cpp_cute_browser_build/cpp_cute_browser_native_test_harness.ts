import {
  accessSync,
  constants,
  realpathSync,
  statSync,
} from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const configuredCompiler = process.env["BROWSERGRAD_NATIVE_CXX"];
const compilerCandidates = configuredCompiler === undefined
  ? [
      "/opt/homebrew/opt/llvm/bin/clang++",
      "/usr/local/opt/llvm/bin/clang++",
      "/usr/bin/clang++",
      "/usr/bin/c++",
      "/usr/bin/g++",
    ]
  : [configuredCompiler];

export interface NativeCompilerDiscovery {
  /**
   * The validated absolute invocation path. Keep its basename intact because
   * compiler drivers such as Clang select C++ link behavior from argv[0].
   */
  readonly path: string;
  /** The canonical executable identity behind the invocation path. */
  readonly canonicalPath: string;
  readonly isClang: boolean;
}

export function resolveNativeCompiler(
  candidates: readonly string[] = compilerCandidates,
): NativeCompilerDiscovery | undefined {
  for (const candidate of candidates) {
    if (!isAbsolute(candidate)) continue;
    try {
      const invocationPath = resolve(candidate);
      const canonicalPath = realpathSync(invocationPath);
      const stat = statSync(canonicalPath);
      accessSync(canonicalPath, constants.X_OK);
      if (!stat.isFile()) continue;
      const version = spawnSync(invocationPath, ["--version"], {
        encoding: "utf8",
        timeout: 5_000,
      });
      if (version.status !== 0 || version.error !== undefined) continue;
      const predefinedMacros = spawnSync(
        invocationPath,
        ["-dM", "-E", "-x", "c++", "/dev/null"],
        { encoding: "utf8", timeout: 5_000 },
      );
      return {
        path: invocationPath,
        canonicalPath,
        isClang: predefinedMacros.status === 0 &&
          predefinedMacros.error === undefined &&
          /^#define __clang__ 1$/mu.test(predefinedMacros.stdout),
      };
    } catch {
      // Try the next closed candidate. A required lane fails through the
      // unskipped behavioral test instead of trusting a partial discovery.
    }
  }
  return undefined;
}

const discovery = resolveNativeCompiler();

export const nativeCompiler = discovery?.path;
export const nativeCompilerCanonicalPath = discovery?.canonicalPath;
export const nativeCompilerRequired =
  process.env["BROWSERGRAD_REQUIRE_NATIVE_CPP_TESTS"] === "1";
if (nativeCompilerRequired && discovery !== undefined && !discovery.isClang) {
  throw new Error(
    "BROWSERGRAD_REQUIRE_NATIVE_CPP_TESTS requires a compiler that defines __clang__",
  );
}
export const nativeCompilerUnavailableUnlessOptional =
  nativeCompiler === undefined && !nativeCompilerRequired;
export const nativeCompilerIsClang = discovery?.isClang === true;
