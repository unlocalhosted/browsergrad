import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const root = resolve(new URL("..", import.meta.url).pathname);
const tmp = mkdtempSync(join(tmpdir(), "browsergrad-release-pack-"));
const packedTarballs = new Map();

try {
  const semanticCore = packAndExtract("browsergrad-semantic-core");
  const kernels = packAndExtract("browsergrad-kernels");
  const compiler = packAndExtract("browsergrad-compiler");
  linkPackedDependency(kernels, "@unlocalhosted/browsergrad-semantic-core", semanticCore);
  linkPackedDependency(compiler, "@unlocalhosted/browsergrad-kernels", kernels);
  linkPackedDependency(compiler, "@unlocalhosted/browsergrad-semantic-core", semanticCore);

  const semanticCorePkg = readPackage(semanticCore);
  const workspaceSemanticCoreVersion = readPackage(join(root, "packages/browsergrad-semantic-core")).version;
  assert(semanticCorePkg.version === workspaceSemanticCoreVersion, `semantic-core version mismatch: ${semanticCorePkg.version}`);
  assert(semanticCorePkg.private !== true, "semantic-core tarball must be publishable");
  assert(semanticCorePkg.exports?.["./schema"], "semantic-core package missing ./schema export");
  assert(semanticCorePkg.exports?.["./layout"], "semantic-core package missing ./layout export");
  assert(semanticCorePkg.exports?.["./kernel"], "semantic-core package missing ./kernel export");
  const densePermutationFixtureExport = "./fixtures/kernel-v1/dense-permutation-view-copy.cases.json";
  assert(
    semanticCorePkg.exports?.[densePermutationFixtureExport] === densePermutationFixtureExport,
    `semantic-core package missing exact ${densePermutationFixtureExport} export`,
  );
  assert(!semanticCorePkg.exports?.["."], "semantic-core package must not add a root barrel");
  assert(Object.keys(semanticCorePkg.dependencies ?? {}).length === 0, "semantic-core package must remain dependency-free");
  for (const file of [
    "dist/schema.js",
    "dist/schema.d.ts",
    "dist/layout.js",
    "dist/layout.d.ts",
    "dist/kernel.js",
    "dist/kernel.d.ts",
    "python/browsergrad_semantic_core.py",
    "fixtures/layout-v1/row-major-rank2.input.json",
    "fixtures/layout-v1/symbolic-byte-rank3.input.json",
    "fixtures/kernel-v1/dense-permutation-view-copy.cases.json",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
  ]) {
    assert(existsSync(join(semanticCore, file)), `semantic-core tarball missing ${file}`);
  }
  const fixtureContracts = JSON.parse(readFileSync(join(root, "architecture/semantic-fixture-contracts.json"), "utf8"));
  const densePermutationFixtureContract = fixtureContracts.contracts.find(
    (contract) => contract.id === "semantic-core.dense-permutation-view-copy.kernel-v1",
  );
  assert(densePermutationFixtureContract, "dense-permutation fixture contract is missing");
  const packedDensePermutationFixtureBytes = readFileSync(
    join(semanticCore, "fixtures/kernel-v1/dense-permutation-view-copy.cases.json"),
  );
  assert(
    createHash("sha256").update(packedDensePermutationFixtureBytes).digest("hex") === densePermutationFixtureContract.contentSha256,
    "packed dense-permutation fixture differs from its architecture contract",
  );
  const packedDensePermutationFixture = JSON.parse(packedDensePermutationFixtureBytes);
  assert(
    JSON.stringify(packedDensePermutationFixture.cases.map(({ id }) => id)) === JSON.stringify(densePermutationFixtureContract.caseIds),
    "packed dense-permutation fixture case coverage differs from its architecture contract",
  );
  const semanticSchema = await import(pathToFileURL(join(semanticCore, "dist/schema.js")));
  const semanticLayout = await import(pathToFileURL(join(semanticCore, "dist/layout.js")));
  const semanticKernel = await import(pathToFileURL(join(semanticCore, "dist/kernel.js")));
  for (const exportName of ["canonicalizeJson", "hashSemanticArtifact", "validateWireEnvelope"]) {
    assert(exportName in semanticSchema, `semantic-core schema export missing ${exportName}`);
  }
  for (const exportName of ["normalizeLayoutExpr", "traceViewCoordinate", "verifyLayoutArtifact"]) {
    assert(exportName in semanticLayout, `semantic-core layout export missing ${exportName}`);
  }
  for (const exportName of [
    "createVerifiedDensePermutationViewCopyArtifacts",
    "createVerifiedViewCopyArtifacts",
    "verifyKernelArtifact",
    "verifyInitialPortableViewCopyProfile",
    "prepareViewCopyCpu",
  ]) {
    assert(exportName in semanticKernel, `semantic-core kernel export missing ${exportName}`);
  }
  const packedArtifacts = await semanticKernel.createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: ["2", "2"],
    axes: [1, 0],
    dtype: "f32",
  }, {
    producer: { id: "release-test", version: "1" },
    layoutArtifactId: "packed-layout",
    kernelArtifactId: "packed-kernel",
  });
  const packedLayout = packedArtifacts.layout;
  const packedLayoutPayload = semanticLayout.layoutArtifactPayload(packedLayout);
  const packedKernel = packedArtifacts.kernel;
  const packedOperationId = packedArtifacts.operationId;
  const packedCopy = await semanticKernel.prepareViewCopyCpu(packedLayout, packedKernel, { operationId: packedOperationId });
  const packedSource = new Uint8Array(16);
  const packedDestination = new Uint8Array(16);
  const packedSourceView = new DataView(packedSource.buffer);
  [1, 2, 3, 4].forEach((value, index) => packedSourceView.setFloat32(index * 4, value, true));
  packedCopy.execute({ source: packedSource, destination: packedDestination });
  const packedDestinationView = new DataView(packedDestination.buffer);
  assert(
    [1, 3, 2, 4].every((value, index) => packedDestinationView.getFloat32(index * 4, true) === value),
    "semantic-core packed /kernel view-copy execution produced the wrong transpose",
  );

  const kernelsPkg = readPackage(kernels);
  const workspaceKernelsPkg = readPackage(join(root, "packages/browsergrad-kernels"));
  const workspaceKernelsVersion = workspaceKernelsPkg.version;
  assert(kernelsPkg.version === workspaceKernelsVersion, `kernels version mismatch: ${kernelsPkg.version}`);
  const semanticCoreRange = kernelsPkg.dependencies?.["@unlocalhosted/browsergrad-semantic-core"];
  assert(semanticCoreRange, "kernels package missing semantic-core dependency");
  assert(!semanticCoreRange.includes("workspace:"), `kernels package leaked workspace dependency: ${semanticCoreRange}`);
  assert(
    semanticCoreRange === workspaceSemanticCoreVersion,
    `kernels package semantic-core dependency should be ${workspaceSemanticCoreVersion}, got ${semanticCoreRange}`,
  );
  for (const subpath of ["./wgsl_program", "./float16", "./cuda_concepts", "./cuda_program", "./rubric", "./semantic_view_copy"]) {
    assert(kernelsPkg.exports?.[subpath], `kernels package missing export ${subpath}`);
  }
  for (const file of [
    "dist/semantic_view_copy.js",
    "dist/semantic_view_copy.d.ts",
  ]) {
    assert(existsSync(join(kernels, file)), `kernels tarball missing ${file}`);
  }
  assert(
    workspaceKernelsPkg.scripts?.prepublishOnly?.includes("require_view_copy_publish_gate.mjs"),
    "kernels workspace package must retain the exact-commit publish gate",
  );
  assert(
    existsSync(join(root, "packages/browsergrad-kernels/scripts/require_view_copy_publish_gate.mjs")),
    "kernels workspace package is missing the exact-commit publish gate implementation",
  );

  const kernelsRoot = await import(pathToFileURL(join(kernels, "dist/index.js")));
  for (const exportName of [
    "createWgslFloat16Array",
    "float16BitsToFloat32",
    "defineWgslKernelProgram",
    "WgslShaderCreationError",
    "WgslPipelineCreationError",
    "prepareWgslKernelProgramSequence",
    "createWgslStorageBuffer",
    "defineCuda1DProgram",
    "runKernel1DProgramReference",
    "simulateCuda1DGrid",
    "runThreadGrid",
    "createKernelRubric",
    "createBrowsergradKernelRubric",
    "prepareSemanticViewCopyWgsl",
    "runSemanticViewCopyWebGpu",
  ]) {
    assert(exportName in kernelsRoot, `kernels root export missing ${exportName}`);
  }

  for (const [subpath, exportName] of [
    ["wgsl_program", "prepareWgslKernelProgramSequence"],
    ["float16", "createWgslFloat16Array"],
    ["cuda_concepts", "runThreadGrid"],
    ["cuda_program", "defineCuda1DProgram"],
    ["rubric", "createKernelRubric"],
    ["semantic_view_copy", "prepareSemanticViewCopyWgsl"],
  ]) {
    const mod = await import(pathToFileURL(join(kernels, `dist/${subpath}.js`)));
    assert(exportName in mod, `kernels ${subpath} export missing ${exportName}`);
  }
  const kernelsSemanticViewCopy = await import(pathToFileURL(join(kernels, "dist/semantic_view_copy.js")));
  const packedWgsl = await kernelsSemanticViewCopy.prepareSemanticViewCopyWgsl(
    packedLayout,
    packedKernel,
    { operationId: packedOperationId },
  );
  assert(packedWgsl.semantic.specializationHash === packedCopy.specializationHash, "packed CPU/WGSL specializations diverged");
  assert(packedWgsl.program.wgsl.includes("var<storage, read> source_words: array<u32>"), "packed WGSL lowering lost bit-exact source words");
  assert(packedWgsl.program.wgsl.includes("destination_words[destination_word] = copied_bits"), "packed WGSL lowering lost destination copy");
  assert(!packedWgsl.program.wgsl.includes("select("), "packed WGSL lowering used eager select for guarded copy");
  const installedConsumer = installPackedConsumer(["browsergrad-semantic-core", "browsergrad-kernels", "browsergrad-compiler"]);
  verifyInstalledSemanticViewCopyConsumer(installedConsumer);

  const compilerPkg = readPackage(compiler);
  const workspaceCompilerPkg = readPackage(join(root, "packages/browsergrad-compiler"));
  const kernelsRange = compilerPkg.dependencies?.["@unlocalhosted/browsergrad-kernels"];
  assert(kernelsRange, "compiler package missing kernels dependency");
  assert(!kernelsRange.includes("workspace:"), `compiler package leaked workspace dependency: ${kernelsRange}`);
  assert(
    kernelsRange === workspaceKernelsVersion,
    `compiler package kernels dependency should be ${workspaceKernelsVersion}, got ${kernelsRange}`,
  );
  const compilerSemanticCoreRange = compilerPkg.dependencies?.["@unlocalhosted/browsergrad-semantic-core"];
  assert(compilerSemanticCoreRange, "compiler package missing semantic-core dependency");
  assert(!compilerSemanticCoreRange.includes("workspace:"), `compiler package leaked semantic-core workspace dependency: ${compilerSemanticCoreRange}`);
  assert(
    compilerSemanticCoreRange === workspaceSemanticCoreVersion,
    `compiler package semantic-core dependency should be ${workspaceSemanticCoreVersion}, got ${compilerSemanticCoreRange}`,
  );
  assert(
    workspaceCompilerPkg.scripts?.prepublishOnly?.includes("require_layout_bindings_publish_gate.mjs"),
    "compiler workspace package must retain the exact-commit layout-binding publish gate",
  );
  assert(
    existsSync(join(root, "packages/browsergrad-compiler/scripts/require_layout_bindings_publish_gate.mjs")),
    "compiler workspace package is missing the exact-commit layout-binding publish gate implementation",
  );
  const releaseWorkflow = readFileSync(join(root, ".github/workflows/release.yml"), "utf8");
  assert(
    releaseWorkflow.includes("Verify compiler dependencies are published (compiler release)"),
    "compiler release workflow must verify its published dependency chain",
  );
  for (const dependency of ["browsergrad-semantic-core", "browsergrad-kernels"]) {
    assert(
      releaseWorkflow.includes(`npm view "@unlocalhosted/${dependency}@$`),
      `compiler release workflow must query the published ${dependency} version`,
    );
  }
  const compilerRoot = await import(pathToFileURL(join(compiler, "dist/index.js")));
  for (const exportName of [
    "prepareCudaLiteLayoutBindings",
    "createCudaLiteLayoutBindingCompileCacheKey",
  ]) {
    assert(exportName in compilerRoot, `compiler root export missing ${exportName}`);
  }
  const packedCompilerBinding = await compilerRoot.prepareCudaLiteLayoutBindings(packedLayout, [{
    parameter: "input",
    viewId: packedLayoutPayload.views[0].viewId,
    access: "read",
    indexing: "row-major-flat",
  }]);
  assert(
    packedCompilerBinding.layoutSemanticHash === await semanticSchema.hashSemanticArtifact(packedLayout),
    "packed compiler binding lost semantic layout identity",
  );
  assert(
    compilerRoot.createCudaLiteLayoutBindingCompileCacheKey("source", packedCompilerBinding).includes(packedCompilerBinding.bindingProjectionHash),
    "packed compiler cache key lost binding projection identity",
  );
  const packedCompilerSource = `
__global__ void packed_layout_copy(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 4u) output[i] = input[i];
}`;
  const packedCompiledView = compilerRoot.compileCudaLiteKernelWithLayoutBindings(
    packedCompilerSource,
    packedCompilerBinding,
    { workgroupSize: [4, 1, 1] },
  );
  const packedCompilerResult = compilerRoot.runCompiledKernelSemanticReference(
    packedCompiledView,
    { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
    { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
  );
  assert(
    [1, 3, 2, 4].every((value, index) => packedCompilerResult.buffers.output[index] === value),
    "packed compiler layout lowering produced the wrong transpose",
  );
  assert(
    packedCompiledView.preparedLayoutBindings === packedCompilerBinding &&
      packedCompiledView.layoutBindingCompileCacheKey.includes(packedCompilerBinding.bindingProjectionHash),
    "packed compiler result lost prepared layout proof identity",
  );
  assert(
    packedCompiledView.wgslProgram.name.includes(packedCompilerBinding.layoutSemanticHash) &&
      packedCompiledView.wgslProgram.name.includes(packedCompilerBinding.bindingProjectionHash),
    "packed compiler WGSL program identity lost layout proof hashes",
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
  packedTarballs.set(packageDirName, tarballPath);
  const extractDir = join(tmp, packageDirName);
  run("mkdir", ["-p", extractDir], root);
  run("tar", ["-xzf", tarballPath, "-C", extractDir], root);
  return join(extractDir, "package");
}

function installPackedConsumer(packageNames) {
  const consumer = join(tmp, "consumer");
  mkdirSync(consumer, { recursive: true });
  const dependencies = {};
  for (const name of packageNames) {
    const tarball = packedTarballs.get(name);
    assert(tarball, `missing packed tarball for ${name}`);
    dependencies[`@unlocalhosted/${name}`] = `file:${tarball}`;
  }
  writeFileSync(join(consumer, "package.json"), JSON.stringify({
    name: "browsergrad-release-consumer",
    private: true,
    type: "module",
    dependencies,
    pnpm: {
      overrides: {
        "@unlocalhosted/browsergrad-semantic-core": dependencies["@unlocalhosted/browsergrad-semantic-core"],
        ...(dependencies["@unlocalhosted/browsergrad-kernels"] === undefined
          ? {}
          : { "@unlocalhosted/browsergrad-kernels": dependencies["@unlocalhosted/browsergrad-kernels"] }),
      },
    },
  }));
  run("pnpm", ["install", "--offline", "--ignore-scripts"], consumer);
  return consumer;
}

function verifyInstalledSemanticViewCopyConsumer(consumer) {
  const source = `
import { layoutArtifactPayload } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { createVerifiedDensePermutationViewCopyArtifacts, prepareViewCopyCpu } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { hashSemanticArtifact, parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import densePermutationFixtures from "@unlocalhosted/browsergrad-semantic-core/fixtures/kernel-v1/dense-permutation-view-copy.cases.json" with { type: "json" };
import { prepareSemanticViewCopyWgsl } from "@unlocalhosted/browsergrad-kernels/semantic_view_copy";
import {
  compileCudaLiteKernelWithLayoutBindings,
  createCudaLiteLayoutBindingCompileCacheKey,
  prepareCudaLiteLayoutBindings,
  runCompiledKernelSemanticReference,
} from "@unlocalhosted/browsergrad-compiler";

const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
  inputShape: [parseWireI64("2"), parseWireI64("2")],
  axes: [1, 0],
  dtype: "f32",
}, {
  producer: { id: "fresh-consumer", version: "1" },
  layoutArtifactId: "transpose-layout",
  kernelArtifactId: "transpose-kernel",
});
if (densePermutationFixtures.schema !== "browsergrad.semantic-core.dense-permutation-view-copy-fixtures" || densePermutationFixtures.cases.length !== 2) throw new Error("fresh consumer lost the versioned dense-permutation fixture export");
const { layout, kernel, operationId } = artifacts;
const payload = layoutArtifactPayload(layout);
const cpu = await prepareViewCopyCpu(layout, kernel, { operationId });
const wgsl = await prepareSemanticViewCopyWgsl(layout, kernel, { operationId });
const compilerBinding = await prepareCudaLiteLayoutBindings(layout, [{
  parameter: "input",
  viewId: payload.views[0].viewId,
  access: "read",
  indexing: "row-major-flat",
}]);
const compilerSource = \`
__global__ void consumer_layout_copy(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 4u) output[i] = input[i];
}\`;
const compiledView = compileCudaLiteKernelWithLayoutBindings(compilerSource, compilerBinding, { workgroupSize: [4, 1, 1] });
const compiledResult = runCompiledKernelSemanticReference(
  compiledView,
  { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
  { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
);
if (cpu.specializationHash !== wgsl.semantic.specializationHash) throw new Error("fresh consumer CPU/WGSL specialization mismatch");
if (!wgsl.program.wgsl.includes("destination_words[destination_word] = copied_bits")) throw new Error("fresh consumer lowering missing copy");
if (compilerBinding.layoutSemanticHash !== await hashSemanticArtifact(layout)) throw new Error("fresh consumer compiler binding lost semantic layout identity");
if (!createCudaLiteLayoutBindingCompileCacheKey("source", compilerBinding).includes(compilerBinding.bindingProjectionHash)) throw new Error("fresh consumer compiler cache key lost binding identity");
if (![1, 3, 2, 4].every((value, index) => compiledResult.buffers.output[index] === value)) throw new Error("fresh consumer compiler layout lowering produced the wrong transpose");
if (compiledView.preparedLayoutBindings !== compilerBinding || !compiledView.layoutBindingCompileCacheKey.includes(compilerBinding.bindingProjectionHash)) throw new Error("fresh consumer compiled result lost layout proof identity");
if (!compiledView.wgslProgram?.name.includes(compilerBinding.layoutSemanticHash) || !compiledView.wgslProgram.name.includes(compilerBinding.bindingProjectionHash)) throw new Error("fresh consumer compiler WGSL identity lost layout proof hashes");
`;
  writeFileSync(join(consumer, "consumer.mjs"), source);
  writeFileSync(join(consumer, "consumer.ts"), source);
  run("node", ["consumer.mjs"], consumer);
  run(join(root, "packages/browsergrad-kernels/node_modules/typescript/bin/tsc"), [
    "--noEmit",
    "--strict",
    "--target", "ES2022",
    "--module", "NodeNext",
    "--moduleResolution", "NodeNext",
    "--resolveJsonModule",
    "--skipLibCheck",
    join(consumer, "consumer.ts"),
  ], root);
}

function readPackage(packageDir) {
  return JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
}

function linkPackedDependency(packageDir, specifier, dependencyDir) {
  const [scope, name] = specifier.split("/");
  assert(scope?.startsWith("@") && name, `expected scoped package specifier, got ${specifier}`);
  const scopeDir = join(packageDir, "node_modules", scope);
  mkdirSync(scopeDir, { recursive: true });
  symlinkSync(dependencyDir, join(scopeDir, name), "dir");
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
