import { spawn } from "node:child_process";
import { chmod, lstat, mkdtemp, opendir, readFile, realpath, rm, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, normalize, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME,
  CPP_CUTE_BROWSER_BUILD_RUNTIME_SOURCE_PATHS,
  parseCppCuteBrowserBuildRuntimeClosureArguments,
  stageCppCuteBrowserBuildRuntimeClosure,
  verifyCppCuteBrowserBuildRuntimeClosure,
} from "./cpp_cute_browser_build_runtime_closure.mjs";

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../..",
);
const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(async (root) => {
    await makeTreeWritable(root).catch(() => {});
    await rm(root, { force: true, recursive: true });
  }));
});

describe("Clang-Wasm exact build runtime closure", () => {
  it("parses only exact absolute source and output roots", () => {
    expect(parseCppCuteBrowserBuildRuntimeClosureArguments([
      "--output-root=/private/runtime",
      "--source-root=/workspace",
    ])).toEqual({
      outputRoot: "/private/runtime",
      sourceRoot: "/workspace",
    });
    for (const argv of [
      ["--source-root=/workspace"],
      ["--source-root=workspace", "--output-root=/private/runtime"],
      ["--source-root=/workspace", "--unknown=/private/runtime"],
      ["--source-root=/workspace", "--source-root=/second"],
    ]) {
      expect(() => parseCppCuteBrowserBuildRuntimeClosureArguments(argv)).toThrow();
    }
  });

  it("stages, seals, imports, and re-verifies the exact file closure", async () => {
    const parent = await temporaryParent();
    const outputRoot = join(parent, "runtime-workspace");
    const staged = await stageCppCuteBrowserBuildRuntimeClosure({
      outputRoot,
      sourceRoot: repositoryRoot,
    });
    const verified = await verifyCppCuteBrowserBuildRuntimeClosure({ workspaceRoot: outputRoot });

    expect(verified.observationSha256).toBe(staged.observationSha256);
    expect(verified.observation.closureSha256).toBe(staged.closureSha256);
    expect(verified.observation.fileCount).toBe(staged.fileCount);
    expect(verified.observation.files.some((file) => file.kind === "extractor")).toBe(true);
    expect(verified.observation.files.some((file) => file.kind === "runtime")).toBe(true);
    expect(await readFile(join(
      outputRoot,
      CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME,
    ))).toHaveLength(staged.observationByteLength);

    const schemaUrl = pathToFileURL(join(
      outputRoot,
      "packages/browsergrad-compiler/node_modules/@unlocalhosted/browsergrad-semantic-core/dist/schema.js",
    ));
    schemaUrl.searchParams.set("closure-test", staged.closureSha256);
    const schema = await import(schemaUrl.href) as { canonicalJsonBytes?: unknown };
    expect(typeof schema.canonicalJsonBytes).toBe("function");

    const runnerRoot = join(outputRoot, "packages/browsergrad-compiler");
    const runner = await runProcess(process.execPath, [
      join(runnerRoot, "scripts/cpp_cute_browser_build/cpp_cute_browser_build_runner.mjs"),
    ], runnerRoot);
    expect(runner.exitCode).toBe(1);
    expect(runner.stderr).toContain("expected exactly 5 named arguments");
    expect(runner.stderr).not.toContain("ERR_MODULE_NOT_FOUND");
  });

  it("rejects changed, missing, and extra staged files", async () => {
    const parent = await temporaryParent();
    const outputRoot = join(parent, "runtime-workspace");
    const staged = await stageCppCuteBrowserBuildRuntimeClosure({
      outputRoot,
      sourceRoot: repositoryRoot,
    });
    const filePath = join(outputRoot, "packages/browsergrad-compiler/package.json");
    await chmod(filePath, 0o600);
    await writeFile(filePath, "{}\n");
    await expect(verifyCppCuteBrowserBuildRuntimeClosure({ workspaceRoot: outputRoot })).rejects.toThrow();

    await chmod(dirname(filePath), 0o700);
    await unlink(filePath);
    await expect(verifyCppCuteBrowserBuildRuntimeClosure({ workspaceRoot: outputRoot })).rejects.toThrow();

    await chmod(outputRoot, 0o700);
    await writeFile(join(outputRoot, "ambient.txt"), "ambient");
    await expect(verifyCppCuteBrowserBuildRuntimeClosure({ workspaceRoot: outputRoot })).rejects.toThrow(
      /undeclared node/u,
    );
    expect(staged.fileCount).toBeGreaterThan(CPP_CUTE_BROWSER_BUILD_RUNTIME_SOURCE_PATHS.length);
  });

  it("declares every local runtime-module dependency in the staged closure", async () => {
    const declared = new Set(CPP_CUTE_BROWSER_BUILD_RUNTIME_SOURCE_PATHS);
    for (const sourcePath of declared) {
      if (!sourcePath.endsWith(".js") && !sourcePath.endsWith(".mjs")) continue;
      const source = await readFile(join(repositoryRoot, sourcePath), "utf8");
      for (const specifier of moduleSpecifiers(source)) {
        if (specifier.startsWith("node:")) continue;
        if (specifier === "@unlocalhosted/browsergrad-semantic-core/schema") {
          expect(declared.has("packages/browsergrad-semantic-core/dist/schema.js")).toBe(true);
          expect(declared.has("packages/browsergrad-semantic-core/package.json")).toBe(true);
          continue;
        }
        expect(specifier.startsWith("."), `${sourcePath} imports undeclared package ${specifier}`).toBe(true);
        const dependency = normalize(join(dirname(sourcePath), specifier)).replaceAll("\\", "/");
        expect(declared.has(dependency), `${sourcePath} imports undeclared file ${dependency}`).toBe(true);
      }
    }
  });
});

async function temporaryParent(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-runtime-closure."));
  temporaryRoots.push(root);
  await chmod(root, 0o700);
  return realpath(root);
}

function moduleSpecifiers(source: string): string[] {
  const specifiers = new Set<string>();
  for (const match of source.matchAll(
    /(?:import|export)\s+(?:[^'"]*?\s+from\s+)?["']([^"']+)["']/gu,
  )) {
    if (match[1] !== undefined) specifiers.add(match[1]);
  }
  for (const match of source.matchAll(/import\(\s*["']([^"']+)["']\s*\)/gu)) {
    if (match[1] !== undefined) specifiers.add(match[1]);
  }
  return [...specifiers];
}

async function makeTreeWritable(path: string): Promise<void> {
  const metadata = await lstat(path);
  if (!metadata.isDirectory()) return;
  await chmod(path, 0o700);
  const directory = await opendir(path);
  try {
    for await (const entry of directory) {
      if (entry.isDirectory()) await makeTreeWritable(join(path, entry.name));
    }
  } finally {
    await directory.close().catch(() => {});
  }
}

async function runProcess(
  executable: string,
  arguments_: readonly string[],
  cwd: string,
): Promise<{ exitCode: number | null; stderr: string }> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(executable, arguments_, {
      cwd,
      env: { PATH: process.env.PATH ?? "" },
      shell: false,
      stdio: ["ignore", "ignore", "pipe"],
    });
    const chunks: Buffer[] = [];
    child.stderr.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.once("error", reject);
    child.once("close", (exitCode) => resolvePromise({
      exitCode,
      stderr: Buffer.concat(chunks).toString("utf8"),
    }));
  });
}
