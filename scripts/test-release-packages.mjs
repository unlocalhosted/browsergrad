import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const root = resolve(new URL("..", import.meta.url).pathname);
const tmp = mkdtempSync(join(tmpdir(), "browsergrad-release-pack-"));

try {
  const semanticCore = packAndExtract("browsergrad-semantic-core");
  const kernels = packAndExtract("browsergrad-kernels");
  const compiler = packAndExtract("browsergrad-compiler");

  const semanticCorePkg = readPackage(semanticCore);
  const workspaceSemanticCoreVersion = readPackage(join(root, "packages/browsergrad-semantic-core")).version;
  assert(semanticCorePkg.version === workspaceSemanticCoreVersion, `semantic-core version mismatch: ${semanticCorePkg.version}`);
  assert(semanticCorePkg.private !== true, "semantic-core tarball must be publishable");
  assert(semanticCorePkg.exports?.["./schema"], "semantic-core package missing ./schema export");
  assert(semanticCorePkg.exports?.["./layout"], "semantic-core package missing ./layout export");
  assert(!semanticCorePkg.exports?.["."], "semantic-core package must not add a root barrel");
  assert(Object.keys(semanticCorePkg.dependencies ?? {}).length === 0, "semantic-core package must remain dependency-free");
  for (const file of [
    "dist/schema.js",
    "dist/schema.d.ts",
    "dist/layout.js",
    "dist/layout.d.ts",
    "python/browsergrad_semantic_core.py",
    "fixtures/layout-v1/row-major-rank2.input.json",
    "fixtures/layout-v1/symbolic-byte-rank3.input.json",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
  ]) {
    assert(existsSync(join(semanticCore, file)), `semantic-core tarball missing ${file}`);
  }
  const semanticSchema = await import(pathToFileURL(join(semanticCore, "dist/schema.js")));
  const semanticLayout = await import(pathToFileURL(join(semanticCore, "dist/layout.js")));
  for (const exportName of ["canonicalizeJson", "hashSemanticArtifact", "validateWireEnvelope"]) {
    assert(exportName in semanticSchema, `semantic-core schema export missing ${exportName}`);
  }
  for (const exportName of ["normalizeLayoutExpr", "traceViewCoordinate", "verifyLayoutArtifact"]) {
    assert(exportName in semanticLayout, `semantic-core layout export missing ${exportName}`);
  }

  const kernelsPkg = readPackage(kernels);
  const workspaceKernelsVersion = readPackage(join(root, "packages/browsergrad-kernels")).version;
  assert(kernelsPkg.version === workspaceKernelsVersion, `kernels version mismatch: ${kernelsPkg.version}`);
  for (const subpath of ["./wgsl_program", "./float16", "./cuda_concepts", "./cuda_program", "./rubric"]) {
    assert(kernelsPkg.exports?.[subpath], `kernels package missing export ${subpath}`);
  }

  const kernelsRoot = await import(pathToFileURL(join(kernels, "dist/index.js")));
  for (const exportName of [
    "createWgslFloat16Array",
    "float16BitsToFloat32",
    "defineWgslKernelProgram",
    "prepareWgslKernelProgramSequence",
    "createWgslStorageBuffer",
    "defineCuda1DProgram",
    "runKernel1DProgramReference",
    "simulateCuda1DGrid",
    "runThreadGrid",
    "createKernelRubric",
    "createBrowsergradKernelRubric",
  ]) {
    assert(exportName in kernelsRoot, `kernels root export missing ${exportName}`);
  }

  for (const [subpath, exportName] of [
    ["wgsl_program", "prepareWgslKernelProgramSequence"],
    ["float16", "createWgslFloat16Array"],
    ["cuda_concepts", "runThreadGrid"],
    ["cuda_program", "defineCuda1DProgram"],
    ["rubric", "createKernelRubric"],
  ]) {
    const mod = await import(pathToFileURL(join(kernels, `dist/${subpath}.js`)));
    assert(exportName in mod, `kernels ${subpath} export missing ${exportName}`);
  }

  const compilerPkg = readPackage(compiler);
  const kernelsRange = compilerPkg.dependencies?.["@unlocalhosted/browsergrad-kernels"];
  assert(kernelsRange, "compiler package missing kernels dependency");
  assert(!kernelsRange.includes("workspace:"), `compiler package leaked workspace dependency: ${kernelsRange}`);
  assert(
    kernelsRange === workspaceKernelsVersion,
    `compiler package kernels dependency should be ${workspaceKernelsVersion}, got ${kernelsRange}`,
  );

  console.log("release package tests ok");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

function packAndExtract(packageDirName) {
  const cwd = join(root, "packages", packageDirName);
  const pack = run("pnpm", ["pack", "--pack-destination", tmp], cwd);
  const tarball = pack.stdout
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .at(-1);
  assert(tarball, `pnpm pack did not print tarball for ${packageDirName}`);
  const tarballPath = resolve(cwd, tarball);
  const extractDir = join(tmp, packageDirName);
  run("mkdir", ["-p", extractDir], root);
  run("tar", ["-xzf", tarballPath, "-C", extractDir], root);
  return join(extractDir, "package");
}

function readPackage(packageDir) {
  return JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
}

function run(cmd, args, cwd) {
  const result = spawnSync(cmd, args, {
    cwd,
    encoding: "utf8",
    stdio: "pipe",
    env: process.env,
  });
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} failed\n${result.stdout}\n${result.stderr}`);
  }
  return result;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}
