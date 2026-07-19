import { pathToFileURL } from "node:url";

import {
  parseCppCuteClangWasmCleanBuildArguments,
  verifyCppCuteClangWasmCleanBuild,
  writeVerifiedCppCuteClangWasmFactoryCandidate,
} from "./cpp_cute_browser_build_reproducibility.mjs";

/** @param {readonly string[]} argv */
export async function admitCppCuteClangWasmCleanBuild(argv) {
  const input = parseCppCuteClangWasmCleanBuildArguments(argv);
  const cleanBuild = await verifyCppCuteClangWasmCleanBuild({ root: input.root });
  const factoryCandidate = await writeVerifiedCppCuteClangWasmFactoryCandidate(
    input["factory-output"],
    cleanBuild,
  );
  return Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.clean-build-package-candidate-observation",
    version: 1,
    authority: "clean-build-admission-and-package-candidate-observation-only",
    cleanBuild,
    factoryCandidate,
  });
}

async function main() {
  try {
    const observation = await admitCppCuteClangWasmCleanBuild(process.argv.slice(2));
    process.stdout.write(`${JSON.stringify(observation)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown clean-build admission failure");
    const path = typeof cause === "object" && cause !== null && "path" in cause &&
      typeof cause.path === "string"
      ? ` at ${cause.path}`
      : "";
    process.stderr.write(`${error.name}${path}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
