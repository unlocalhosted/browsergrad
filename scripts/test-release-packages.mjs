import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const root = resolve(new URL("..", import.meta.url).pathname);
const tmp = mkdtempSync(join(tmpdir(), "browsergrad-release-pack-"));

try {
  const kernels = packAndExtract("browsergrad-kernels");
  const compiler = packAndExtract("browsergrad-compiler");

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
