import {
  accessSync,
  constants,
  realpathSync,
  statSync,
} from "node:fs";
import { isAbsolute } from "node:path";
import { spawnSync } from "node:child_process";

const configuredCompiler = process.env["BROWSERGRAD_NATIVE_CXX"];
const compilerCandidates = configuredCompiler === undefined
  ? ["/usr/bin/clang++", "/usr/bin/c++", "/usr/bin/g++"]
  : [configuredCompiler];

interface NativeCompilerDiscovery {
  readonly path: string;
  readonly isClang: boolean;
}

function resolveNativeCompiler(): NativeCompilerDiscovery | undefined {
  for (const candidate of compilerCandidates) {
    if (!isAbsolute(candidate)) continue;
    try {
      const resolved = realpathSync(candidate);
      const stat = statSync(resolved);
      accessSync(resolved, constants.X_OK);
      if (!stat.isFile()) continue;
      const version = spawnSync(resolved, ["--version"], {
        encoding: "utf8",
        timeout: 5_000,
      });
      if (version.status !== 0 || version.error !== undefined) continue;
      return {
        path: resolved,
        isClang: /(?:^|\n)(?:Apple )?clang version /u.test(version.stdout),
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
export const nativeCompilerRequired =
  process.env["BROWSERGRAD_REQUIRE_NATIVE_CPP_TESTS"] === "1";
export const nativeCompilerUnavailableUnlessOptional =
  nativeCompiler === undefined && !nativeCompilerRequired;
export const nativeCompilerIsClang = discovery?.isClang === true;
