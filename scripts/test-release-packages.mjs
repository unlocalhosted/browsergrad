import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const root = resolve(new URL("..", import.meta.url).pathname);
const tmp = mkdtempSync(join(tmpdir(), "browsergrad-release-pack-"));
const packedTarballs = new Map();
const packedPackageDirectories = new Map();

try {
  const primitives = packAndExtract("browsergrad-primitives");
  const runtime = packAndExtract("browsergrad-runtime");
  const semanticCore = packAndExtract("browsergrad-semantic-core");
  const kernels = packAndExtract("browsergrad-kernels");
  const grad = packAndExtract("browsergrad-grad");
  const jit = packAndExtract("browsergrad-jit");
  const compiler = packAndExtract("browsergrad-compiler");
  linkPackedDependency(runtime, "@unlocalhosted/browsergrad-semantic-core", semanticCore);
  linkPackedDependency(kernels, "@unlocalhosted/browsergrad-semantic-core", semanticCore);
  linkPackedDependency(grad, "@unlocalhosted/browsergrad-kernels", kernels);
  linkPackedDependency(compiler, "@unlocalhosted/browsergrad-kernels", kernels);
  linkPackedDependency(compiler, "@unlocalhosted/browsergrad-semantic-core", semanticCore);
  const workspaceSemanticCoreVersion = readPackage(
    join(root, "packages/browsergrad-semantic-core"),
  ).version;

  const workspacePrimitivesPkg = readPackage(join(root, "packages/browsergrad-primitives"));
  const primitivesPkg = readPackage(primitives);
  assert(
    primitivesPkg.version === workspacePrimitivesPkg.version,
    `primitives version mismatch: ${primitivesPkg.version}`,
  );
  assertRepositoryMetadata(primitivesPkg, "browsergrad-primitives");
  assert(primitivesPkg.private !== true, "primitives tarball must be publishable");
  assertNoWorkspaceProtocol(primitivesPkg, "primitives packed manifest");
  assert(
    Object.keys(primitivesPkg.dependencies ?? {}).length === 0
      && Object.keys(primitivesPkg.optionalDependencies ?? {}).length === 0
      && Object.keys(primitivesPkg.peerDependencies ?? {}).length === 0,
    "primitives packed package must remain dependency-free",
  );
  for (const subpath of [
    ".",
    "./data",
    "./evaluation",
    "./rl",
    "./scaling",
    "./simulation",
    "./text",
    "./package.json",
  ]) {
    assert(primitivesPkg.exports?.[subpath], `primitives package missing ${subpath} export`);
  }
  for (const file of [
    "dist/index.js",
    "dist/index.d.ts",
    "dist/data.js",
    "dist/data.d.ts",
    "dist/evaluation.js",
    "dist/evaluation.d.ts",
    "dist/rl.js",
    "dist/rl.d.ts",
    "dist/scaling.js",
    "dist/scaling.d.ts",
    "dist/simulation.js",
    "dist/simulation.d.ts",
    "dist/text.js",
    "dist/text.d.ts",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
  ]) {
    assert(existsSync(join(primitives, file)), `primitives tarball missing ${file}`);
  }
  const primitivesRoot = await import(pathToFileURL(join(primitives, "dist/index.js")));
  for (const namespace of ["data", "evaluation", "rl", "scaling", "simulation", "text"]) {
    assert(typeof primitivesRoot[namespace] === "object", `primitives packed root missing ${namespace} namespace`);
  }
  const npmPrimitivesConsumer = installPackedNpmConsumer("primitives", ["browsergrad-primitives"]);
  verifyInstalledPrimitivesConsumer(npmPrimitivesConsumer, workspacePrimitivesPkg.version);

  const workspaceRuntimePkg = readPackage(join(root, "packages/browsergrad-runtime"));
  const runtimePkg = readPackage(runtime);
  assert(runtimePkg.version === workspaceRuntimePkg.version, `runtime version mismatch: ${runtimePkg.version}`);
  assertRepositoryMetadata(runtimePkg, "browsergrad-runtime");
  assert(runtimePkg.private !== true, "runtime tarball must be publishable");
  assertNoWorkspaceProtocol(runtimePkg, "runtime packed manifest");
  const runtimeSemanticCoreRange =
    runtimePkg.dependencies?.["@unlocalhosted/browsergrad-semantic-core"];
  assert(
    runtimeSemanticCoreRange === workspaceSemanticCoreVersion
      && Object.keys(runtimePkg.dependencies ?? {}).length === 1
      && Object.keys(runtimePkg.optionalDependencies ?? {}).length === 0,
    `runtime packed package must depend only on semantic-core ${workspaceSemanticCoreVersion}`,
  );
  assert(
    runtimePkg.peerDependencies?.pyodide === "^0.26.0"
      && runtimePkg.peerDependenciesMeta?.pyodide?.optional === false,
    "runtime packed package must retain its required Pyodide peer contract",
  );
  for (const subpath of [".", "./worker", "./package.json"]) {
    assert(runtimePkg.exports?.[subpath], `runtime package missing ${subpath} export`);
  }
  for (const file of [
    "dist/index.js",
    "dist/index.d.ts",
    "dist/worker/index.js",
    "dist/worker/index.d.ts",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
  ]) {
    assert(existsSync(join(runtime, file)), `runtime tarball missing ${file}`);
  }
  const runtimeRoot = await import(pathToFileURL(join(runtime, "dist/index.js")));
  for (const exportName of [
    "BrowsergradError",
    "assignmentRequirementDefinitions",
    "createFrameworkPlatformSupportView",
    "createProgramCapabilitySupportView",
    "createAssignmentRequirementResolutionEnvironment",
    "createSession",
    "isSemverCompatible",
    "parseManifest",
  ]) {
    assert(exportName in runtimeRoot, `runtime packed root missing ${exportName}`);
  }
  const pyodidePeer = join(root, "packages/browsergrad-runtime/node_modules/pyodide");
  assert(existsSync(pyodidePeer), "runtime fresh npm consumer requires the installed Pyodide peer fixture");
  const npmRuntimeConsumer = installPackedNpmConsumer(
    "runtime",
    ["browsergrad-runtime", "browsergrad-semantic-core"],
    { pyodide: `file:${pyodidePeer}` },
  );
  verifyInstalledRuntimeConsumer(npmRuntimeConsumer, workspaceRuntimePkg.version);

  const semanticCorePkg = readPackage(semanticCore);
  assert(semanticCorePkg.version === workspaceSemanticCoreVersion, `semantic-core version mismatch: ${semanticCorePkg.version}`);
  assertRepositoryMetadata(semanticCorePkg, "browsergrad-semantic-core");
  assert(semanticCorePkg.private !== true, "semantic-core tarball must be publishable");
  assert(semanticCorePkg.exports?.["./schema"], "semantic-core package missing ./schema export");
  assert(semanticCorePkg.exports?.["./layout"], "semantic-core package missing ./layout export");
  assert(semanticCorePkg.exports?.["./kernel"], "semantic-core package missing ./kernel export");
  assert(semanticCorePkg.exports?.["./schedule"], "semantic-core package missing ./schedule export");
  assert(semanticCorePkg.exports?.["./requirement"], "semantic-core package missing ./requirement export");
  assert(semanticCorePkg.exports?.["./capability"], "semantic-core package missing ./capability export");
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
    "dist/schedule.js",
    "dist/schedule.d.ts",
    "dist/requirement.js",
    "dist/requirement.d.ts",
    "dist/capability.js",
    "dist/capability.d.ts",
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
  const semanticSchedule = await import(pathToFileURL(join(semanticCore, "dist/schedule.js")));
  const semanticRequirement = await import(pathToFileURL(join(semanticCore, "dist/requirement.js")));
  const semanticCapability = await import(pathToFileURL(join(semanticCore, "dist/capability.js")));
  for (const exportName of ["canonicalizeJson", "hashSemanticArtifact", "SCHEDULE_DIAGNOSTIC_CODES", "validateWireEnvelope"]) {
    assert(exportName in semanticSchema, `semantic-core schema export missing ${exportName}`);
  }
  for (const exportName of ["normalizeLayoutExpr", "traceViewCoordinate", "verifyLayoutArtifact"]) {
    assert(exportName in semanticLayout, `semantic-core layout export missing ${exportName}`);
  }
  for (const exportName of [
    "INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY",
    "attentionForwardArtifactPayload",
    "copyCertifiedLogicalGemmExactF32Inputs",
    "createVerifiedDenseAttentionForwardArtifacts",
    "createVerifiedLogicalGemmExactF32InputCertificate",
    "createVerifiedDensePermutationViewCopyArtifacts",
    "createVerifiedViewCopyArtifacts",
    "verifyKernelArtifact",
    "verifyInitialPortableViewCopyProfile",
    "prepareViewCopyCpu",
    "prepareAttentionForwardCpu",
    "prepareAttentionForwardSpecialization",
    "verifyAttentionForwardArtifact",
  ]) {
    assert(exportName in semanticKernel, `semantic-core kernel export missing ${exportName}`);
  }
  for (const exportName of [
    "attentionOnlineKvTileScheduleArtifactPayload",
    "createVerifiedAttentionOnlineKvTileSchedule",
    "createVerifiedLogicalGemmTileSchedule",
    "logicalGemmTileScheduleArtifactPayload",
    "prepareAttentionOnlineKvTileSchedule",
    "prepareLogicalGemmTileSchedule",
    "verifyLogicalGemmTileScheduleArtifact",
    "verifyAttentionOnlineKvTileScheduleArtifact",
  ]) {
    assert(exportName in semanticSchedule, `semantic-core schedule export missing ${exportName}`);
  }
  for (const exportName of [
    "createAssignmentRequirementDefinition",
    "createAssignmentRequirementResolution",
  ]) {
    assert(exportName in semanticRequirement, `semantic-core requirement export missing ${exportName}`);
  }
  for (const exportName of [
    "createLoweringDecision",
    "createSemanticBackendDefinition",
    "createSemanticCapabilityDefinition",
  ]) {
    assert(exportName in semanticCapability, `semantic-core capability export missing ${exportName}`);
  }
  const packedLogicalGemm = await semanticKernel.createVerifiedDenseLogicalGemmTileArtifacts({
    m: "16",
    n: "16",
    k: "16",
    logicalTile: { m: "16", n: "16", k: "16" },
  });
  const packedSchedule8 = await semanticSchedule.createVerifiedLogicalGemmTileSchedule(
    packedLogicalGemm.kernel,
    { physicalTile: { m: "8", n: "8", k: "8" } },
  );
  const packedSchedule16 = await semanticSchedule.createVerifiedLogicalGemmTileSchedule(
    packedLogicalGemm.kernel,
    { physicalTile: { m: "16", n: "16", k: "16" } },
  );
  assert(
    packedSchedule8.logicalGemmSemanticHash === await semanticSchema.hashSemanticArtifact(packedLogicalGemm.kernel),
    "semantic-core packed /schedule lost the exact logical GEMM semantic reference",
  );
  assert(
    packedSchedule8.scheduleSemanticHash !== packedSchedule16.scheduleSemanticHash,
    "semantic-core packed /schedule collapsed distinct physical GEMM schedules",
  );
  const packedLogicalGemmSpecialization = await semanticKernel.prepareLogicalGemmTileSpecialization(
    packedLogicalGemm.layout,
    packedLogicalGemm.kernel,
    { operationId: packedLogicalGemm.operationId },
  );
  const packedPreparedSchedule8 = await semanticSchedule.prepareLogicalGemmTileSchedule(
    packedLogicalGemmSpecialization,
    packedLogicalGemm.kernel,
    packedSchedule8.artifact,
  );
  const packedPreparedSchedule16 = await semanticSchedule.prepareLogicalGemmTileSchedule(
    packedLogicalGemmSpecialization,
    packedLogicalGemm.kernel,
    packedSchedule16.artifact,
  );
  assert(
    packedPreparedSchedule8.logical === packedLogicalGemmSpecialization
      && packedPreparedSchedule16.logical === packedLogicalGemmSpecialization,
    "semantic-core packed /schedule repeated or reconstructed the authorized logical specialization",
  );
  assert(
    packedPreparedSchedule8.scheduleSpecializationHash
      !== packedPreparedSchedule16.scheduleSpecializationHash,
    "semantic-core packed /schedule collapsed distinct specialized schedule geometry",
  );
  const packedExactLogicalGemm = await semanticKernel.createVerifiedDenseLogicalGemmTileArtifacts({
    m: "1",
    n: "1",
    k: "1",
    logicalTile: { m: "1", n: "1", k: "1" },
  });
  const packedExactLhs = new Uint8Array(4);
  const packedExactRhs = new Uint8Array(4);
  new DataView(packedExactLhs.buffer).setFloat32(0, 3, true);
  new DataView(packedExactRhs.buffer).setFloat32(0, 5, true);
  const packedExactInput = await semanticKernel.createVerifiedLogicalGemmExactF32InputCertificate(
    packedExactLogicalGemm.layout,
    packedExactLogicalGemm.kernel,
    {
      operationId: packedExactLogicalGemm.operationId,
      inputs: { lhs: packedExactLhs, rhs: packedExactRhs },
    },
  );
  packedExactLhs.fill(0xff);
  packedExactRhs.fill(0xff);
  const packedCertifiedCopies = semanticKernel.copyCertifiedLogicalGemmExactF32Inputs(
    packedExactInput.certificate,
  );
  assert(
    new DataView(packedCertifiedCopies.lhs.buffer).getFloat32(0, true) === 3
      && new DataView(packedCertifiedCopies.rhs.buffer).getFloat32(0, true) === 5,
    "semantic-core packed exact-input certificate did not retain authoritative input snapshots",
  );
  assert(
    semanticKernel.logicalGemmExactF32InputCertificatePayload(
      packedExactInput.certificate,
    ).proof.maximumOutputSum === "15",
    "semantic-core packed exact-input certificate lost its exact arithmetic proof",
  );
  const packedAttention = await semanticKernel.createVerifiedDenseAttentionForwardArtifacts({
    batch: "1",
    heads: "2",
    queryLength: "3",
    keyLength: "5",
    queryDepth: "4",
    valueDepth: "6",
    causal: true,
  });
  const packedAttentionPayload = semanticKernel.attentionForwardArtifactPayload(
    packedAttention.kernel,
  );
  assert(
    packedAttentionPayload.layoutSemanticHash
      === await semanticSchema.hashSemanticArtifact(packedAttention.layout)
      && packedAttentionPayload.operation.mask.kind === "causal"
      && packedAttentionPayload.operation.scale.value.bits === "3f000000",
    "semantic-core packed attention-forward artifact lost layout, mask, or exact-scale meaning",
  );
  assert(
    semanticKernel.INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.absoluteTolerance === 0.0001
      && semanticKernel.INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.relativeTolerance === 0.0001
      && !JSON.stringify(packedAttentionPayload).match(/workgroup|staging|barrier|wgsl|webgpu|cuda|flash/iu),
    "semantic-core packed attention-forward contract gained schedule/backend meaning or lost its comparison policy",
  );
  const packedAttentionSchedule8 = await semanticSchedule.createVerifiedAttentionOnlineKvTileSchedule(
    packedAttention.kernel,
    { physicalTile: { queryRows: "8", keyRows: "8" } },
  );
  const packedAttentionSchedule16 = await semanticSchedule.createVerifiedAttentionOnlineKvTileSchedule(
    packedAttention.kernel,
    { physicalTile: { queryRows: "8", keyRows: "16" } },
  );
  const packedAttentionSchedulePayload = semanticSchedule.attentionOnlineKvTileScheduleArtifactPayload(
    packedAttentionSchedule16.artifact,
  );
  assert(
    packedAttentionSchedule16.attentionForwardSemanticHash
      === await semanticSchema.hashSemanticArtifact(packedAttention.kernel)
      && packedAttentionSchedule8.scheduleSemanticHash
        !== packedAttentionSchedule16.scheduleSemanticHash
      && packedAttentionSchedulePayload.schedule.staging.key === "cooperative"
      && packedAttentionSchedulePayload.schedule.staging.value === "cooperative"
      && packedAttentionSchedulePayload.schedule.masks.invalidKeyScore
        === "exclude-before-online-state-update"
      && packedAttentionSchedulePayload.schedule.masks.logicalMask
        === "exclude-before-online-state-update",
    "semantic-core packed attention schedule lost logical binding, staging, masks, or distinct tiling",
  );
  const packedAttentionLogical = await semanticKernel.prepareAttentionForwardSpecialization(
    packedAttention.layout,
    packedAttention.kernel,
    { operationId: packedAttention.operationId },
  );
  const packedPreparedAttentionSchedule8 = await semanticSchedule.prepareAttentionOnlineKvTileSchedule(
    packedAttentionLogical,
    packedAttention.kernel,
    packedAttentionSchedule8.artifact,
  );
  const packedPreparedAttentionSchedule16 = await semanticSchedule.prepareAttentionOnlineKvTileSchedule(
    packedAttentionLogical,
    packedAttention.kernel,
    packedAttentionSchedule16.artifact,
  );
  assert(
    packedPreparedAttentionSchedule8.logical === packedAttentionLogical
      && packedPreparedAttentionSchedule16.logical === packedAttentionLogical
      && packedPreparedAttentionSchedule8.aggregateStagingBytes === 320n
      && packedPreparedAttentionSchedule16.aggregateStagingBytes === 640n
      && packedPreparedAttentionSchedule8.scheduleSpecializationHash
        !== packedPreparedAttentionSchedule16.scheduleSpecializationHash,
    "semantic-core packed attention schedule specialization lost logical authority or geometry",
  );
  const packedAttentionCpu = await semanticKernel.prepareAttentionForwardCpu(
    packedAttention.layout,
    packedAttention.kernel,
    { operationId: packedAttention.operationId },
  );
  const packedAttentionBuffers = {
    query: new Uint8Array(96),
    key: new Uint8Array(160),
    value: new Uint8Array(240),
    destination: new Uint8Array(144).fill(0xff),
  };
  const packedAttentionTrace = await packedAttentionCpu.execute(packedAttentionBuffers);
  assert(
    packedAttentionTrace.mask === "causal-upper-left"
      && packedAttentionTrace.validScoreElements === "12"
      && packedAttentionTrace.scoreMultiplyAdds === "48"
      && packedAttentionTrace.weightedValueMultiplyAdds === "72"
      && packedAttentionCpu.compare(
        packedAttentionBuffers.destination,
        new Uint8Array(144),
      ).passed,
    "semantic-core packed attention CPU reference lost causal work accounting or f32 output meaning",
  );
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
  for (const subpath of ["./wgsl_program", "./float16", "./cuda_concepts", "./cuda_program", "./rubric", "./semantic_view_copy", "./semantic_gemm", "./semantic_attention"]) {
    assert(kernelsPkg.exports?.[subpath], `kernels package missing export ${subpath}`);
  }
  for (const file of [
    "dist/semantic_view_copy.js",
    "dist/semantic_view_copy.d.ts",
    "dist/semantic_gemm.js",
    "dist/semantic_gemm.d.ts",
    "dist/semantic_attention.js",
    "dist/semantic_attention.d.ts",
  ]) {
    assert(existsSync(join(kernels, file)), `kernels tarball missing ${file}`);
  }
  const kernelsPrepublishCommands = workspaceKernelsPkg.scripts?.prepublishOnly
    ?.split("&&")
    .map((command) => command.trim());
  assert(
    Array.isArray(kernelsPrepublishCommands)
      && kernelsPrepublishCommands.at(-1) === "node scripts/require_view_copy_publish_gate.mjs",
    "kernels exact-commit evidence gate must be the final prepublish command",
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
    "prepareSemanticGemmWgsl",
    "runSemanticGemmWebGpu",
    "prepareSemanticAttentionWgsl",
    "runSemanticAttentionWebGpu",
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
    ["semantic_gemm", "prepareSemanticGemmWgsl"],
    ["semantic_attention", "prepareSemanticAttentionWgsl"],
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
  const kernelsSemanticGemm = await import(pathToFileURL(join(kernels, "dist/semantic_gemm.js")));
  const packedExactSchedule = await semanticSchedule.createVerifiedLogicalGemmTileSchedule(
    packedExactLogicalGemm.kernel,
    { physicalTile: { m: "1", n: "1", k: "1" } },
  );
  const packedSemanticGemm = await kernelsSemanticGemm.prepareSemanticGemmWgsl(
    packedExactLogicalGemm.layout,
    packedExactLogicalGemm.kernel,
    packedExactSchedule.artifact,
    { operationId: packedExactLogicalGemm.operationId },
  );
  assert(
    packedSemanticGemm.semantic.specializationHash === packedExactInput.specializationHash,
    "packed semantic GEMM lost the certificate-bound logical specialization",
  );
  assert(
    packedSemanticGemm.program.wgsl.match(/workgroupBarrier\(\);/gu)?.length === 2,
    "packed semantic GEMM lost its two uniform staging barriers",
  );
  const kernelsSemanticAttention = await import(pathToFileURL(join(kernels, "dist/semantic_attention.js")));
  const packedSemanticAttention = await kernelsSemanticAttention.prepareSemanticAttentionWgsl(
    packedAttention.layout,
    packedAttention.kernel,
    packedAttentionSchedule8.artifact,
    { operationId: packedAttention.operationId },
  );
  assert(
    packedSemanticAttention.semantic.specializationHash
      === packedAttentionLogical.specializationHash
      && packedSemanticAttention.algorithmProfile
        === "block-tiled-kv-online-softmax-forward"
      && packedSemanticAttention.program.wgsl.includes("var<workgroup> key_tile")
      && packedSemanticAttention.program.wgsl.includes("var<workgroup> value_tile")
      && packedSemanticAttention.program.wgsl.match(/workgroupBarrier\(\);/gu)?.length === 2,
    "packed semantic attention lost logical authority, K/V staging, or uniform barriers",
  );
  const installedConsumer = installPackedConsumer(
    "semantic-view-copy",
    ["browsergrad-semantic-core", "browsergrad-kernels", "browsergrad-compiler"],
  );
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
  const compilerPrepublishCommands = workspaceCompilerPkg.scripts?.prepublishOnly
    ?.split("&&")
    .map((command) => command.trim());
  assert(
    Array.isArray(compilerPrepublishCommands)
      && compilerPrepublishCommands.at(-1) === "node scripts/require_compiler_evidence_publish_gate.mjs",
    "compiler combined exact-commit evidence gate must be the final prepublish command",
  );
  assert(
    existsSync(join(root, "packages/browsergrad-compiler/scripts/require_compiler_evidence_publish_gate.mjs")),
    "compiler workspace package is missing the combined exact-commit publish gate implementation",
  );
  for (const [script, command] of [
    ["test:browser:view-copy-bindings", "semantic_view_copy_bindings_webgpu.test.ts"],
    ["test:browser:view-copy-bindings:required", "--mode webgpu-required"],
  ]) {
    assert(
      workspaceCompilerPkg.scripts?.[script]?.includes(command),
      `compiler workspace package is missing ${script}`,
    );
  }
  for (const relativePath of [
    "packages/browsergrad-compiler/scripts/verify_view_copy_bindings_evidence_log.mjs",
    "packages/browsergrad-compiler/scripts/verify_view_copy_bindings_evidence_log.d.mts",
    "scripts/compiler-view-copy-evidence-source.mjs",
    "scripts/compiler-view-copy-evidence-source.d.mts",
  ]) {
    assert(existsSync(join(root, relativePath)), `compiler L2 evidence surface missing ${relativePath}`);
  }
  const compilerBrowserConfig = readFileSync(
    join(root, "packages/browsergrad-compiler/vitest.browser.config.ts"),
    "utf8",
  );
  assert(
    compilerBrowserConfig.includes('execFileSync("git", ["rev-parse", "HEAD"]')
      && compilerBrowserConfig.includes("__BG_SOURCE_REVISION__"),
    "compiler browser evidence config must inject exact git HEAD",
  );
  const compilerLayoutEvidence = readFileSync(
    join(root, "packages/browsergrad-compiler/tests-browser/semantic_layout_bindings_webgpu.test.ts"),
    "utf8",
  );
  assert(
    compilerLayoutEvidence.includes("layout-bindings.webgpu-conformance@2")
      && compilerLayoutEvidence.includes("plannedWorkgroupCount")
      && compilerLayoutEvidence.includes("plannedPipelineCount")
      && !compilerLayoutEvidence.includes("submittedWorkgroupCount")
      && !/\bpipelineCount\b/u.test(compilerLayoutEvidence),
    "compiler layout evidence must label plan-derived topology as planned",
  );
  const releaseWorkflow = readFileSync(join(root, ".github/workflows/release.yml"), "utf8");
  const publishWorkflow = readFileSync(join(root, ".github/workflows/publish-npm.yml"), "utf8");
  const ciWorkflow = readFileSync(join(root, ".github/workflows/ci.yml"), "utf8");
  const workspaceRootPkg = readPackage(root);
  assert(
    workspaceRootPkg.packageManager === "pnpm@10.34.5",
    `release tooling requires pinned pnpm@10.34.5, got ${workspaceRootPkg.packageManager}`,
  );
  for (const [workflowName, workflow] of [["release", releaseWorkflow], ["publish", publishWorkflow]]) {
    assert(workflow.includes("environment: npm-production"), `${workflowName} workflow must use protected npm environment`);
    assert(
      workflow.includes('git merge-base --is-ancestor "$GITHUB_SHA" origin/main'),
      `${workflowName} workflow must reject commits outside main`,
    );
    assert(
      workflow.includes("concurrency:\n  group: browsergrad-npm-production\n  queue: max"),
      `${workflowName} workflow must serialize every production publication without dropping queued releases`,
    );
    assert(workflow.includes("version: 10.34.5"), `${workflowName} workflow pnpm must match packageManager`);
    const actionRefs = [...workflow.matchAll(/uses: [^@\s]+@([^\s#]+)/gu)].map((match) => match[1]);
    assert(actionRefs.length > 0, `${workflowName} workflow must use pinned actions`);
    assert(
      actionRefs.every((ref) => /^[0-9a-f]{40}$/u.test(ref)),
      `${workflowName} workflow actions must use immutable full commit SHAs`,
    );
    assert(
      workflow.includes("--stage-dir npm-release-artifacts")
        && workflow.includes("actions/upload-artifact@")
        && workflow.includes("actions/download-artifact@")
        && workflow.includes("--publish-staged npm-release-artifacts")
        && workflow.includes("--provenance"),
      `${workflowName} workflow must transfer one immutable staged artifact into protected publication`,
    );
    const publishJob = workflow.slice(workflow.indexOf("\n  publish:\n"));
    const validateJob = workflow.slice(
      workflow.indexOf("\n  validate:\n"),
      workflow.indexOf("\n  publish:\n"),
    );
    assert(publishJob.length > 0, `${workflowName} workflow must define a distinct publish job`);
    assert(
      !validateJob.includes("id-token: write") && !validateJob.includes("NODE_AUTH_TOKEN"),
      `${workflowName} validation and lifecycle job cannot receive npm credentials or OIDC authority`,
    );
    assert(!publishJob.includes("pnpm install"), `${workflowName} protected publish job cannot install workspace dependencies`);
    assert(!publishJob.includes("prepublishOnly"), `${workflowName} protected publish job cannot run package lifecycle scripts`);
    assert(
      publishJob.includes("id-token: write")
        && publishJob.includes("npm install --global npm@11.12.1 --ignore-scripts")
        && publishJob.includes("node scripts/require-npm-version.mjs 11.12.0")
        && publishJob.includes("ACTIONS_ID_TOKEN_REQUEST_URL: ''")
        && publishJob.includes("ACTIONS_ID_TOKEN_REQUEST_TOKEN: ''"),
      `${workflowName} protected publish job must install a pinned attestation-capable npm CLI without OIDC authority`,
    );
    assert(
      (workflow.match(/NODE_AUTH_TOKEN:/gu) ?? []).length === (workflowName === "release" ? 0 : 1),
      `${workflowName} workflow must keep trusted publishing tokenless and scope fallback token to one protected step`,
    );
  }
  assert(
    releaseWorkflow.includes('--preflight --package "${{ steps.parse.outputs.npmname }}"')
      && releaseWorkflow.includes('--publish-staged npm-release-artifacts --package "${{ needs.validate.outputs.npmname }}" --provenance'),
    "release workflow must preflight and publish the exact selected package closure",
  );
  assert(
    releaseWorkflow.includes('gh release view "$GITHUB_REF_NAME"')
      && releaseWorkflow.includes('gh release edit "$GITHUB_REF_NAME"'),
    "release workflow must resume after an already-created npm version or GitHub release",
  );
  const releaseNpmJob = releaseWorkflow.slice(
    releaseWorkflow.indexOf("\n  publish:\n"),
    releaseWorkflow.indexOf("\n  github-release:\n"),
  );
  const githubReleaseJob = releaseWorkflow.slice(releaseWorkflow.indexOf("\n  github-release:\n"));
  assert(
    releaseNpmJob.includes("contents: read") && !releaseNpmJob.includes("contents: write"),
    "npm publication job must not receive repository write authority",
  );
  assert(
    githubReleaseJob.includes("contents: write")
      && !githubReleaseJob.includes("id-token: write")
      && !githubReleaseJob.includes("--publish-staged"),
    "GitHub Release job must hold only repository authority after npm publication",
  );
  assert(
    publishWorkflow.includes("Run Grad integration suite")
      && publishWorkflow.includes("Run JIT integration suite")
      && publishWorkflow.includes("publish-missing-npm.mjs --preflight")
      && publishWorkflow.includes('if [ "$GITHUB_REF" != "refs/heads/main" ]')
      && !publishWorkflow.includes("NPM_CONFIG_PROVENANCE: false"),
    "manual publish validation must cover integration, main-ref enforcement, generic dependency preflight, and provenance",
  );
  assert(
    !releaseWorkflow.includes("Verify JIT dependencies are published")
      && !releaseWorkflow.includes("Verify compiler dependencies are published")
      && !releaseWorkflow.includes("Verify semantic-core dependency is published"),
    "release workflow must not regress to package-specific dependency checks",
  );
  const publicPackageDirectories = readdirSync(join(root, "packages"))
    .filter((directory) => existsSync(join(root, "packages", directory, "package.json")))
    .map((directory) => ({ directory, pkg: readPackage(join(root, "packages", directory)) }))
    .filter(({ pkg }) => pkg.private !== true && pkg.name?.startsWith("@unlocalhosted/"));
  const expectedPackedPackages = publicPackageDirectories
    .map(({ directory }) => directory)
    .sort();
  const actualPackedPackages = [...packedTarballs.keys()].sort();
  assert(
    JSON.stringify(actualPackedPackages) === JSON.stringify(expectedPackedPackages),
    `packed release coverage must exactly match every public workspace package; expected ${expectedPackedPackages.join(", ")}, got ${actualPackedPackages.join(", ")}`,
  );
  for (const { directory, pkg } of publicPackageDirectories) {
    const shortname = directory.slice("browsergrad-".length);
    assert(
      releaseWorkflow.includes(`- '${shortname}-v*'`),
      `${pkg.name} is missing its release tag trigger`,
    );
    assertRepositoryMetadata(pkg, directory);
    assert(
      typeof pkg.scripts?.prepublishOnly === "string" && pkg.scripts.prepublishOnly.length > 0,
      `${pkg.name} must define its complete validation gate in prepublishOnly`,
    );
    for (const lifecycle of ["prepublish", "prepare", "prepack", "postpack", "publish", "postpublish", "dependencies"]) {
      assert(pkg.scripts?.[lifecycle] === undefined, `${pkg.name} cannot mutate the frozen release artifact through ${lifecycle}`);
    }
  }
  assertCommandFails(
    process.execPath,
    [join(root, "scripts/publish-missing-npm.mjs")],
    root,
    "exactly one of --dry-run, --preflight, --stage-dir, or --publish-staged is required",
  );
  const publisherSource = readFileSync(join(root, "scripts/publish-missing-npm.mjs"), "utf8");
  assert(
    publisherSource.indexOf("const baselineIdentityByName = new Map()")
      < publisherSource.indexOf('console.log(`prepare ${exactSpec(manifest)}`)'),
    "publisher must capture every target baseline before running the first mutating prepublish gate",
  );
  assert(
    publisherSource.includes('["ls-files", "--others", "--exclude-standard", "-z"]')
      && publisherSource.includes("Untracked source files cannot enter staged release artifacts"),
    "publisher must reject untracked source while allowing only its declared staging directory",
  );
  assertCommandFails(
    process.execPath,
    [join(root, "scripts/publish-missing-npm.mjs"), "--preflight", "--provenance"],
    root,
    "--provenance is valid only with --publish-staged",
  );

  const workspaceGradPkg = readPackage(join(root, "packages/browsergrad-grad"));
  const gradPkg = readPackage(grad);
  assert(gradPkg.version === workspaceGradPkg.version, `Grad version mismatch: ${gradPkg.version}`);
  assert(gradPkg.optionalPeerDependencies === undefined, "Grad tarball must not use nonstandard optionalPeerDependencies");
  assert(
    gradPkg.dependencies?.["@unlocalhosted/browsergrad-kernels"] === workspaceKernelsVersion,
    `Grad kernels dependency should pack as ${workspaceKernelsVersion}`,
  );
  assert(
    gradPkg.peerDependencies?.pyodide === "^0.26.4"
      && gradPkg.peerDependenciesMeta?.pyodide?.optional === true,
    "Grad Pyodide peer must use standard optional peer metadata",
  );
  assert(
    gradPkg.dependencies?.["@unlocalhosted/browsergrad-semantic-core"] === undefined
      && gradPkg.peerDependencies?.["@unlocalhosted/browsergrad-semantic-core"] === undefined,
    "Grad must depend on semantic-core only through kernels",
  );
  assertNoWorkspaceProtocol(gradPkg, "Grad packed manifest");
  for (const file of [
    "dist/index.js",
    "dist/index.d.ts",
    "dist/python/index.js",
    "dist/python/index.d.ts",
    "dist/node-adapter.js",
    "dist/node-adapter.d.ts",
    "dist/kernel-device.js",
    "dist/kernel-device.d.ts",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
  ]) {
    assert(existsSync(join(grad, file)), `Grad tarball missing ${file}`);
  }
  const npmGradConsumer = installPackedNpmConsumer(
    "grad-kernels-semantic-core",
    ["browsergrad-semantic-core", "browsergrad-kernels", "browsergrad-grad"],
  );
  verifyInstalledGradConsumer(npmGradConsumer, workspaceGradPkg.version);

  const workspaceJitPkg = readPackage(join(root, "packages/browsergrad-jit"));
  const jitPkg = readPackage(jit);
  assert(jitPkg.version === workspaceJitPkg.version, `JIT version mismatch: ${jitPkg.version}`);
  assert(jitPkg.optionalPeerDependencies === undefined, "JIT tarball must not use nonstandard optionalPeerDependencies");
  assert(Object.keys(jitPkg.dependencies ?? {}).length === 0, "JIT tarball must have no install-time dependencies");
  assert(Object.keys(jitPkg.optionalDependencies ?? {}).length === 0, "JIT tarball must have no optionalDependencies");
  assert(
    JSON.stringify(Object.keys(jitPkg.peerDependencies ?? {}).sort())
      === JSON.stringify(["@unlocalhosted/browsergrad-kernels", "pyodide"].sort()),
    "JIT tarball must expose exactly kernels and Pyodide as peers",
  );
  assert(
    jitPkg.peerDependencies?.["@unlocalhosted/browsergrad-kernels"] === `^${workspaceKernelsVersion}`,
    `JIT kernels peer must pack as ^${workspaceKernelsVersion}`,
  );
  assert(
    jitPkg.peerDependencies?.pyodide === "^0.26.4",
    `JIT Pyodide peer must remain ^0.26.4, got ${jitPkg.peerDependencies?.pyodide}`,
  );
  for (const peerName of ["@unlocalhosted/browsergrad-kernels", "pyodide"]) {
    assert(
      jitPkg.peerDependenciesMeta?.[peerName]?.optional === true,
      `JIT peer ${peerName} must be explicitly optional`,
    );
  }
  assert(
    jitPkg.dependencies?.["@unlocalhosted/browsergrad-semantic-core"] === undefined
      && jitPkg.peerDependencies?.["@unlocalhosted/browsergrad-semantic-core"] === undefined,
    "JIT package must remain semantic-core bridge-neutral; kernels owns semantic-core",
  );
  assertNoWorkspaceProtocol(jitPkg, "JIT packed manifest");
  for (const file of [
    "dist/index.js",
    "dist/index.d.ts",
    "dist/python/index.js",
    "dist/python/index.d.ts",
    "dist/node-adapter.js",
    "dist/node-adapter.d.ts",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
  ]) {
    assert(existsSync(join(jit, file)), `JIT tarball missing ${file}`);
  }
  const jitRoot = await import(pathToFileURL(join(jit, "dist/index.js")));
  const jitSource = await import(pathToFileURL(join(jit, "dist/python/index.js")));
  const jitNodeAdapter = await import(pathToFileURL(join(jit, "dist/node-adapter.js")));
  assert(typeof jitRoot.installJit === "function", "JIT packed root missing installJit");
  assert(typeof jitRoot.JitInstallError === "function", "JIT packed root missing JitInstallError");
  assert(
    typeof jitRoot.frameworkOperationSupport === "function"
      && typeof jitRoot.frameworkPlatformSupportSource === "function",
    "JIT packed root missing generated framework support exports",
  );
  assert(
    jitSource.MOUNT_ROOT === "/lib/browsergrad_jit_src"
      && Array.isArray(jitSource.SOURCE_FILES)
      && jitSource.SOURCE_FILES.some(({ path }) => path === "browsergrad_jit/__init__.py"),
    "JIT packed source export lost mount root or Python package sources",
  );
  assert(
    typeof jitNodeAdapter.createNodePyodideTarget === "function",
    "JIT packed Node adapter missing createNodePyodideTarget",
  );
  const jitOnlyConsumer = installPackedConsumer("jit-only", ["browsergrad-jit"]);
  assert(
    !existsSync(join(jitOnlyConsumer, "node_modules/@unlocalhosted/browsergrad-kernels")),
    "JIT-only consumer must not auto-install optional kernels peer",
  );
  assert(
    !existsSync(join(jitOnlyConsumer, "node_modules/pyodide")),
    "JIT-only consumer must not auto-install optional Pyodide peer",
  );
  verifyInstalledJitConsumer(jitOnlyConsumer, false, workspaceJitPkg.version);
  const npmJitOnlyConsumer = installPackedNpmConsumer("jit-only", ["browsergrad-jit"]);
  assert(
    !existsSync(join(npmJitOnlyConsumer, "node_modules/@unlocalhosted/browsergrad-kernels"))
      && !existsSync(join(npmJitOnlyConsumer, "node_modules/pyodide")),
    "fresh npm JIT-only consumer must not auto-install optional peers",
  );
  verifyInstalledJitConsumer(npmJitOnlyConsumer, false, workspaceJitPkg.version);
  const npmFrameworkPlatformConsumer = installPackedNpmConsumer(
    "framework-platform-support",
    [
      "browsergrad-runtime",
      "browsergrad-semantic-core",
      "browsergrad-jit",
      "browsergrad-kernels",
      "browsergrad-grad",
    ],
    { pyodide: `file:${pyodidePeer}` },
  );
  verifyInstalledFrameworkPlatformConsumer(
    npmFrameworkPlatformConsumer,
    workspaceRuntimePkg.version,
    workspaceJitPkg.version,
  );
  const integratedJitConsumer = installPackedConsumer(
    "jit-kernels-semantic-core",
    ["browsergrad-semantic-core", "browsergrad-kernels", "browsergrad-jit"],
  );
  verifyInstalledJitConsumer(integratedJitConsumer, true, workspaceJitPkg.version);
  const jitPrepublishCommands = workspaceJitPkg.scripts?.prepublishOnly
    ?.split("&&")
    .map((command) => command.trim());
  assert(
    Array.isArray(jitPrepublishCommands)
      && jitPrepublishCommands.at(-1) === "node scripts/require_semantic_permute_publish_gate.mjs",
    "JIT exact-commit semantic-permute publish gate must be the final prepublish command",
  );
  assert(
    workspaceJitPkg.scripts?.prepublishOnly?.includes("typecheck:browser:semantic-permute"),
    "JIT prepublish must typecheck the semantic-permute browser evidence lane",
  );
  assert(
    existsSync(join(root, "packages/browsergrad-jit/scripts/require_semantic_permute_publish_gate.mjs")),
    "JIT workspace package is missing the exact-commit semantic-permute publish gate implementation",
  );
  assert(
    existsSync(join(root, "packages/browsergrad-jit/scripts/verify_semantic_permute_evidence_log.mjs")),
    "JIT workspace package is missing the retained terminal-record verifier",
  );
  const jitBrowserConfig = readFileSync(
    join(root, "packages/browsergrad-jit/vitest.browser.config.ts"),
    "utf8",
  );
  const jitBrowserEvidence = readFileSync(
    join(root, "packages/browsergrad-jit/tests-browser/semantic_permute_webgpu.test.ts"),
    "utf8",
  );
  const jitEvidenceValidator = readFileSync(
    join(root, "packages/browsergrad-jit/tests-browser/semantic_permute_evidence.ts"),
    "utf8",
  );
  assert(
    jitBrowserConfig.includes("captureJitSemanticPermuteSubmissions")
      && jitBrowserConfig.includes("__BG_JIT_SEMANTIC_PERMUTE_CAPTURE_JSON__"),
    "JIT browser evidence config must capture the production GpuExecutionSubmission",
  );
  assert(
    jitBrowserEvidence.includes("__BG_JIT_SEMANTIC_PERMUTE_CAPTURE_JSON__")
      && jitBrowserEvidence.includes("preparedCase.semanticRequestsJson")
      && !jitBrowserEvidence.includes("function createPlan("),
    "JIT browser evidence must execute the exact captured plan/wire without a synthetic plan",
  );
  assert(
    jitBrowserEvidence.includes("semanticTensorPlanExecutionTrace")
      && jitBrowserEvidence.includes("prepared-case-set"),
    "JIT browser evidence must bind actual preparation/topology and the complete case set",
  );
  assert(
    jitBrowserEvidence.includes("finalizeTerminalEvidence")
      && jitEvidenceValidator.includes("terminalManifestHash")
      && jitEvidenceValidator.includes("TERMINAL_MANIFEST_HASH_DOMAIN"),
    "JIT retained evidence must carry a validated domain-separated whole-record digest",
  );
  for (const [workflowName, workflow] of [["release", releaseWorkflow], ["publish", publishWorkflow]]) {
    assertSemanticGemmEvidenceWorkflowOrder(workflowName, workflow);
    assertSemanticAttentionEvidenceWorkflowOrder(workflowName, workflow);
    assertSemanticAttentionPerformanceEvidenceWorkflowOrder(workflowName, workflow);
    assertCompilerViewCopyEvidenceWorkflowOrder(workflowName, workflow);
    assertJitEvidenceWorkflowOrder(workflowName, workflow);
  }
  const performanceCiStart = ciWorkflow.indexOf("\n  semantic-attention-performance-webgpu:\n");
  const performanceCiEnd = ciWorkflow.indexOf("\n  real-world-audits:\n", performanceCiStart);
  assert(
    performanceCiStart >= 0
      && performanceCiEnd > performanceCiStart
      && ciWorkflow.includes("--exclude tests-browser/semantic_attention_performance_webgpu.test.ts"),
    "CI must isolate semantic attention performance from the broad browser lane",
  );
  const performanceCiJob = ciWorkflow.slice(performanceCiStart, performanceCiEnd);
  assert(
    performanceCiJob.includes("test:browser:semantic-attention:performance:required")
      && performanceCiJob.includes("semantic-attention-performance-webgpu-evidence-${{ github.sha }}")
      && !/^    needs:/mu.test(performanceCiJob),
    "CI semantic attention performance evidence must be required, retained, and independently parallel",
  );
  const compilerRoot = await import(pathToFileURL(join(compiler, "dist/index.js")));
  const packedCompilerEntries = listRelativeEntries(compiler);
  assert(
    !existsSync(join(compiler, "scripts"))
      && !packedCompilerEntries.some((entry) => /(?:^|\/)cpp_cute_aot_docker_[^/]+$/u.test(entry)),
    `packed compiler must exclude all Node-only Docker shell entries: ${packedCompilerEntries.join(", ")}`,
  );
  const leakedDockerExports = Object.keys(compilerRoot)
    .filter((name) => /docker|boundedchildprocess/iu.test(name));
  assert(
    leakedDockerExports.length === 0,
    `compiler root leaked Node-only Docker/process/test exports: ${leakedDockerExports.join(", ")}`,
  );
  const leakedDockerSubpaths = Object.keys(compilerPkg.exports ?? {})
    .filter((subpath) => /docker|bounded[_-]child[_-]process/iu.test(subpath));
  assert(
    leakedDockerSubpaths.length === 0,
    `compiler package leaked Node-only Docker subpath exports: ${leakedDockerSubpaths.join(", ")}`,
  );
  for (const exportName of [
    "prepareCudaLiteLayoutBindings",
    "createCudaLiteLayoutBindingCompileCacheKey",
    "prepareCudaLiteViewCopyBinding",
    "compileCudaLiteKernelWithViewCopyBinding",
    "prepareVerifiedCppCuteLogicalGemmTileSemantics",
    "lowerAuthorizedCppCuteLogicalGemmTileEntry",
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
  const packedViewCopyBinding = await compilerRoot.prepareCudaLiteViewCopyBinding(
    packedLayout,
    packedKernel,
    {
      operationId: packedOperationId,
      sourceParameter: "input",
      destinationParameter: "output",
      indexing: "row-major-flat",
    },
  );
  const packedCompiledViewCopy = compilerRoot.compileCudaLiteKernelWithViewCopyBinding(
    packedCompilerSource,
    packedViewCopyBinding,
    { workgroupSize: [4, 1, 1] },
  );
  const packedViewCopyResult = compilerRoot.runCompiledKernelSemanticReference(
    packedCompiledViewCopy,
    {
      buffers: {
        input: new Uint32Array([0x3f800000, 0x40000000, 0x40400000, 0x40800000]),
        output: new Uint32Array(4),
      },
    },
    { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
  );
  assert(
    [0x3f800000, 0x40400000, 0x40000000, 0x40800000]
      .every((value, index) => packedViewCopyResult.buffers.output[index] === value),
    "packed compiler L2 view-copy lowering produced the wrong raw-word transpose",
  );
  assert(
    packedCompiledViewCopy.preparedViewCopyBinding === packedViewCopyBinding
      && packedCompiledViewCopy.viewCopyBindingCompileCacheKey.includes(
        packedViewCopyBinding.bindingProjectionHash,
      ),
    "packed compiler L2 result lost prepared binding authority",
  );
  assert(
    packedCompiledViewCopy.wgslProgram.name.includes(packedViewCopyBinding.layoutSemanticHash)
      && packedCompiledViewCopy.wgslProgram.name.includes(packedViewCopyBinding.kernelSemanticHash)
      && packedCompiledViewCopy.wgslProgram.name.includes(packedViewCopyBinding.specializationHash)
      && packedCompiledViewCopy.wgslProgram.name.includes(packedViewCopyBinding.bindingProjectionHash),
    "packed compiler L2 WGSL identity lost semantic binding hashes",
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
  const packageDirectory = join(extractDir, "package");
  packedPackageDirectories.set(packageDirName, packageDirectory);
  return packageDirectory;
}

function installPackedConsumer(consumerId, packageNames) {
  const consumer = join(tmp, `consumer-${consumerId}`);
  mkdirSync(consumer, { recursive: true });
  const dependencies = {};
  for (const name of packageNames) {
    const tarball = packedTarballs.get(name);
    assert(tarball, `missing packed tarball for ${name}`);
    dependencies[`@unlocalhosted/${name}`] = `file:${tarball}`;
  }
  const overrides = packedRuntimeDependencyOverrides(packageNames);
  writeFileSync(join(consumer, "package.json"), JSON.stringify({
    name: "browsergrad-release-consumer",
    private: true,
    type: "module",
    dependencies,
    pnpm: { overrides },
  }));
  run("pnpm", [
    "install",
    "--offline",
    "--ignore-scripts",
    "--strict-peer-dependencies",
    "--config.auto-install-peers=false",
  ], consumer);
  return consumer;
}

function packedRuntimeDependencyOverrides(packageNames) {
  const selected = new Set(packageNames);
  const overrides = {};
  for (const parentName of packageNames) {
    const packageDirectory = packedPackageDirectories.get(parentName);
    assert(packageDirectory, `missing extracted packed package for ${parentName}`);
    const parent = readPackage(packageDirectory);
    for (const field of ["dependencies", "optionalDependencies"]) {
      for (const dependencyName of Object.keys(parent[field] ?? {})) {
        const dependencyDirectoryName = dependencyName.startsWith("@unlocalhosted/")
          ? dependencyName.slice("@unlocalhosted/".length)
          : undefined;
        if (dependencyDirectoryName === undefined || !selected.has(dependencyDirectoryName)) continue;
        const dependencyTarball = packedTarballs.get(dependencyDirectoryName);
        assert(dependencyTarball, `missing packed tarball for ${dependencyDirectoryName}`);
        overrides[`${parent.name}@${parent.version}>${dependencyName}`] = `file:${dependencyTarball}`;
      }
    }
  }
  return overrides;
}

function installPackedNpmConsumer(consumerId, packageNames, additionalDependencies = {}) {
  const consumer = join(tmp, `npm-consumer-${consumerId}`);
  mkdirSync(consumer, { recursive: true });
  const dependencies = { ...additionalDependencies };
  for (const name of packageNames) {
    const tarball = packedTarballs.get(name);
    assert(tarball, `missing packed tarball for ${name}`);
    dependencies[`@unlocalhosted/${name}`] = `file:${tarball}`;
  }
  writeFileSync(join(consumer, "package.json"), JSON.stringify({
    name: "browsergrad-npm-release-consumer",
    private: true,
    type: "module",
    dependencies,
  }));
  run("npm", [
    "install",
    "--offline",
    "--ignore-scripts",
    "--omit=optional",
    "--no-audit",
    "--no-fund",
  ], consumer);
  return consumer;
}

function verifyInstalledPrimitivesConsumer(consumer, expectedVersion) {
  const source = `
import { data, evaluation, rl, scaling, simulation, text } from "@unlocalhosted/browsergrad-primitives";
import { createDataCleaningReference } from "@unlocalhosted/browsergrad-primitives/data";
import { compareSnapshot } from "@unlocalhosted/browsergrad-primitives/evaluation";
import { computePerInstanceDpoLoss } from "@unlocalhosted/browsergrad-primitives/rl";
import { fitPowerLawScalingLaw } from "@unlocalhosted/browsergrad-primitives/scaling";
import { partitionStaticWork } from "@unlocalhosted/browsergrad-primitives/simulation";
import { createStreamingGate } from "@unlocalhosted/browsergrad-primitives/text";
import primitivesPackage from "@unlocalhosted/browsergrad-primitives/package.json" with { type: "json" };

if (primitivesPackage.version !== ${JSON.stringify(expectedVersion)}) throw new Error("unexpected packed primitives version");
if (typeof data.createDataCleaningReference !== "function" || typeof evaluation.compareSnapshot !== "function") throw new Error("packed primitives root namespaces invalid");
if (typeof rl.computePerInstanceDpoLoss !== "function" || typeof scaling.fitPowerLawScalingLaw !== "function") throw new Error("packed primitives root math namespaces invalid");
if (typeof simulation.partitionStaticWork !== "function" || typeof text.createStreamingGate !== "function") throw new Error("packed primitives root platform namespaces invalid");
if (typeof createDataCleaningReference().extractVisibleTextFromHtml !== "function") throw new Error("packed primitives data export invalid");
if (!compareSnapshot({ value: 3 }, { value: 3 }).ok) throw new Error("packed primitives evaluation export invalid");
const dpoLoss = computePerInstanceDpoLoss({
  beta: 0.1,
  policyChosenLogProbability: -0.2,
  policyRejectedLogProbability: -1.2,
  referenceChosenLogProbability: -0.4,
  referenceRejectedLogProbability: -0.9,
});
if (!Number.isFinite(dpoLoss)) throw new Error("packed primitives RL export invalid");
const fit = fitPowerLawScalingLaw([{ size: 1, loss: 4 }, { size: 2, loss: 2 }], { x: "size", y: "loss" });
if (!Number.isFinite(fit.predict(4))) throw new Error("packed primitives scaling export invalid");
const partitions = partitionStaticWork({ items: 5, workers: 2 });
if (partitions.length !== 2 || partitions.flatMap(({ ranges }) => ranges).length !== 2) throw new Error("packed primitives simulation export invalid");
const gate = createStreamingGate({ chunkCount: 1, maxChunksBeforeFirstYield: 1 });
gate.noteChunkConsumed();
gate.noteFirstYield();
gate.assertAllowed();
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

function verifyInstalledRuntimeConsumer(consumer, expectedVersion) {
  const source = `
import {
  BrowsergradError,
  assignmentCapabilityEnvironmentFromRequirementResolutions,
  assignmentRequirementDefinitions,
  createAssignmentRequirementResolutionEnvironment,
  createAssignmentRunPlan,
  createProgramCapabilitySupportView,
  createSession,
  isSemverCompatible,
  parseManifest,
} from "@unlocalhosted/browsergrad-runtime";
import runtimePackage from "@unlocalhosted/browsergrad-runtime/package.json" with { type: "json" };

if (runtimePackage.version !== ${JSON.stringify(expectedVersion)}) throw new Error("unexpected packed runtime version");
if (typeof BrowsergradError !== "function" || typeof createSession !== "function") throw new Error("packed runtime root exports invalid");
const requirementEnvironment = createAssignmentRequirementResolutionEnvironment({
  environmentId: "fresh-consumer",
  providers: [{
    requirementId: "pyodide",
    providerId: "worker.pyodide",
    mode: "browser",
  }],
});
const capabilities = assignmentCapabilityEnvironmentFromRequirementResolutions(requirementEnvironment);
if (assignmentRequirementDefinitions().length !== 53 || capabilities.capabilities.join(",") !== "pyodide") throw new Error("packed runtime requirement resolution invalid");
const plan = createAssignmentRunPlan({
  id: "release-requirement-consumer",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  runtime_packages: [],
  files: { root: "/assignment", rubric_path: "rubric.py" },
  timeouts: {},
  allowed_tests: [],
  oracles: [],
  gates: [{
    name: "runtime",
    kind: "capability",
    options: { requires: ["pyodide"] },
  }],
  datasets: [],
}, requirementEnvironment);
if (!plan.ok || plan.requirementResolutions?.[0]?.requirementId !== "pyodide" || plan.requirementResolutions[0].status !== "available") throw new Error("packed runtime direct requirement consumer invalid");
const support = createProgramCapabilitySupportView({
  viewId: "browsergrad.support.release-consumer",
  subject: {
    kind: "program",
    programId: "browsergrad.program.release-consumer",
  },
  decisions: [{
    capabilityId: "browsergrad.layout.index-map",
    backendId: "browsergrad.compiler.semantic-reference",
    executionTier: "semantic-reference",
    state: "supported",
    preservationLevel: "observable-equivalent",
  }],
});
if (support.decisions.length !== 1 || support.decisions[0].state !== "supported") throw new Error("packed runtime program capability view invalid");
if (!isSemverCompatible("^0.1.0", runtimePackage.version)) throw new Error("packed runtime semver export invalid");
const parsed = parseManifest({
  id: "release-consumer",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  required_ops: [],
  rubric_path: "rubric.py",
  starter_path: "starter.py",
  reference_path: "reference.py",
});
if (!parsed.ok || parsed.manifest.id !== "release-consumer") throw new Error("packed runtime manifest export invalid");
Object.defineProperty(globalThis, "self", {
  configurable: true,
  value: {
    addEventListener() {},
    postMessage() {},
  },
});
await import("@unlocalhosted/browsergrad-runtime/worker");
`;
  writeFileSync(join(consumer, "consumer.mjs"), source);
  writeFileSync(join(consumer, "consumer.ts"), source);
  run("node", ["consumer.mjs"], consumer);
  typecheckConsumer(consumer);
}

function verifyInstalledGradConsumer(consumer, expectedVersion) {
  const source = `
import {
  GradInstallError,
  installGrad,
  frameworkPlatformSupportSource,
} from "@unlocalhosted/browsergrad-grad";
import { MOUNT_ROOT, SOURCE_FILES } from "@unlocalhosted/browsergrad-grad/source";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-grad/node-adapter";
import { createGradKernelDeviceBridge } from "@unlocalhosted/browsergrad-grad/kernel-device";
import gradPackage from "@unlocalhosted/browsergrad-grad/package.json" with { type: "json" };

if (gradPackage.version !== ${JSON.stringify(expectedVersion)}) throw new Error("unexpected packed Grad version");
if (typeof installGrad !== "function" || !(new GradInstallError("test") instanceof Error)) throw new Error("packed Grad root exports invalid");
if (frameworkPlatformSupportSource().operations.length !== 22) throw new Error("packed Grad generated framework support invalid");
if (MOUNT_ROOT !== "/lib/browsergrad_grad_src" || !SOURCE_FILES.some(({ path }) => path === "browsergrad_grad/__init__.py")) throw new Error("packed Grad source export invalid");
if (typeof createNodePyodideTarget !== "function" || typeof createGradKernelDeviceBridge !== "function") throw new Error("packed Grad adapters missing");
`;
  writeFileSync(join(consumer, "consumer.mjs"), source);
  writeFileSync(join(consumer, "consumer.ts"), source);
  run("node", ["consumer.mjs"], consumer);
  typecheckConsumer(consumer);
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
  compileCudaLiteKernelWithViewCopyBinding,
  createCudaLiteLayoutBindingCompileCacheKey,
  prepareCudaLiteLayoutBindings,
  prepareCudaLiteViewCopyBinding,
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
const viewCopyBinding = await prepareCudaLiteViewCopyBinding(layout, kernel, {
  operationId,
  sourceParameter: "input",
  destinationParameter: "output",
  indexing: "row-major-flat",
});
const compiledViewCopy = compileCudaLiteKernelWithViewCopyBinding(
  compilerSource,
  viewCopyBinding,
  { workgroupSize: [4, 1, 1] },
);
const compiledViewCopyResult = runCompiledKernelSemanticReference(
  compiledViewCopy,
  {
    buffers: {
      input: new Uint32Array([0x3f800000, 0x40000000, 0x40400000, 0x40800000]),
      output: new Uint32Array(4),
    },
  },
  { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
);
if (cpu.specializationHash !== wgsl.semantic.specializationHash) throw new Error("fresh consumer CPU/WGSL specialization mismatch");
if (!wgsl.program.wgsl.includes("destination_words[destination_word] = copied_bits")) throw new Error("fresh consumer lowering missing copy");
if (compilerBinding.layoutSemanticHash !== await hashSemanticArtifact(layout)) throw new Error("fresh consumer compiler binding lost semantic layout identity");
if (!createCudaLiteLayoutBindingCompileCacheKey("source", compilerBinding).includes(compilerBinding.bindingProjectionHash)) throw new Error("fresh consumer compiler cache key lost binding identity");
if (![1, 3, 2, 4].every((value, index) => compiledResult.buffers.output[index] === value)) throw new Error("fresh consumer compiler layout lowering produced the wrong transpose");
if (compiledView.preparedLayoutBindings !== compilerBinding || !compiledView.layoutBindingCompileCacheKey.includes(compilerBinding.bindingProjectionHash)) throw new Error("fresh consumer compiled result lost layout proof identity");
if (!compiledView.wgslProgram?.name.includes(compilerBinding.layoutSemanticHash) || !compiledView.wgslProgram.name.includes(compilerBinding.bindingProjectionHash)) throw new Error("fresh consumer compiler WGSL identity lost layout proof hashes");
if (![0x3f800000, 0x40400000, 0x40000000, 0x40800000].every((value, index) => compiledViewCopyResult.buffers.output[index] === value)) throw new Error("fresh consumer compiler L2 view-copy produced wrong raw-word transpose");
if (compiledViewCopy.preparedViewCopyBinding !== viewCopyBinding || !compiledViewCopy.viewCopyBindingCompileCacheKey.includes(viewCopyBinding.bindingProjectionHash)) throw new Error("fresh consumer compiler L2 result lost prepared binding authority");
if (!compiledViewCopy.wgslProgram?.name.includes(viewCopyBinding.layoutSemanticHash) || !compiledViewCopy.wgslProgram.name.includes(viewCopyBinding.kernelSemanticHash) || !compiledViewCopy.wgslProgram.name.includes(viewCopyBinding.specializationHash) || !compiledViewCopy.wgslProgram.name.includes(viewCopyBinding.bindingProjectionHash)) throw new Error("fresh consumer compiler L2 WGSL identity lost semantic binding hashes");
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

function verifyInstalledJitConsumer(consumer, integrated, expectedVersion) {
  const integrationImports = integrated
    ? `
import { createVerifiedDensePermutationViewCopyArtifacts } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { prepareSemanticViewCopyWgsl } from "@unlocalhosted/browsergrad-kernels/semantic_view_copy";
`
    : "";
  const integrationChecks = integrated
    ? `
const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
  inputShape: [parseWireI64("2"), parseWireI64("3")],
  axes: [1, 0],
  dtype: "f32",
}, {
  producer: { id: "packed-jit-consumer", version: "1" },
  layoutArtifactId: "packed-jit-layout",
  kernelArtifactId: "packed-jit-kernel",
});
const prepared = await prepareSemanticViewCopyWgsl(artifacts.layout, artifacts.kernel, {
  operationId: artifacts.operationId,
});
if (prepared.semantic.operation.operationId !== artifacts.operationId) throw new Error("integrated packed JIT consumer lost semantic operation identity");
`
    : "";
  const source = `
import {
  installJit,
  JitInstallError,
  frameworkOperationSupport,
  frameworkPlatformSupportSource,
} from "@unlocalhosted/browsergrad-jit";
import { MOUNT_ROOT, SOURCE_FILES } from "@unlocalhosted/browsergrad-jit/source";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-jit/node-adapter";
import jitPackage from "@unlocalhosted/browsergrad-jit/package.json" with { type: "json" };
${integrationImports}

if (jitPackage.version !== ${JSON.stringify(expectedVersion)}) throw new Error(\`unexpected packed JIT version: \${jitPackage.version}\`);
if (typeof installJit !== "function" || !(new JitInstallError("test") instanceof Error)) throw new Error("packed JIT root exports invalid");
if (frameworkOperationSupport().operations.length !== 36 || frameworkPlatformSupportSource().operations.length !== 36) throw new Error("packed JIT generated framework support invalid");
if (MOUNT_ROOT !== "/lib/browsergrad_jit_src" || !SOURCE_FILES.some(({ path }) => path === "browsergrad_jit/__init__.py")) throw new Error("packed JIT source export invalid");

const observation = { writeCount: 0, executionCount: 0, lastCode: "" };
const target = createNodePyodideTarget({
  runPythonAsync: async (code) => {
    observation.executionCount += 1;
    observation.lastCode = code;
  },
  FS: {
    mkdirTree: () => {},
    writeFile: () => { observation.writeCount += 1; },
  },
});
await installJit(target, { skipImportCheck: true });
if (observation.writeCount !== SOURCE_FILES.length) throw new Error("packed JIT install did not write complete source set");
if (observation.executionCount !== 1 || !observation.lastCode.includes(MOUNT_ROOT)) throw new Error("packed JIT install did not mount source root");
${integrationChecks}
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

function verifyInstalledFrameworkPlatformConsumer(
  consumer,
  expectedRuntimeVersion,
  expectedJitVersion,
) {
  const source = `
import {
  createAssignmentRequirementResolutionEnvironment,
  createFrameworkPlatformSupportView,
} from "@unlocalhosted/browsergrad-runtime";
import {
  JIT_FRAMEWORK_VERSION,
  frameworkPlatformSupportSource as jitFrameworkPlatformSupportSource,
} from "@unlocalhosted/browsergrad-jit";
import {
  frameworkPlatformSupportSource as gradFrameworkPlatformSupportSource,
} from "@unlocalhosted/browsergrad-grad";
import runtimePackage from "@unlocalhosted/browsergrad-runtime/package.json" with { type: "json" };
import jitPackage from "@unlocalhosted/browsergrad-jit/package.json" with { type: "json" };

if (runtimePackage.version !== ${JSON.stringify(expectedRuntimeVersion)}) throw new Error("unexpected packed runtime version");
if (jitPackage.version !== ${JSON.stringify(expectedJitVersion)} || JIT_FRAMEWORK_VERSION !== jitPackage.version) throw new Error("unexpected packed JIT framework version");
const requirements = createAssignmentRequirementResolutionEnvironment({
  environmentId: "browser.release-consumer",
  providers: [{
    requirementId: "pyodide",
    providerId: "worker.pyodide",
    mode: "browser",
  }],
});
const view = createFrameworkPlatformSupportView({
  viewId: "browsergrad.support.release-platform",
  requirements,
  program: {
    viewId: "browsergrad.support.release-program",
    subject: {
      kind: "program",
      programId: "browsergrad.program.release-platform",
    },
    decisions: [{
      capabilityId: "browsergrad.layout.index-map",
      backendId: "browsergrad.compiler.semantic-reference",
      executionTier: "semantic-reference",
      state: "supported",
      preservationLevel: "observable-equivalent",
    }],
  },
  frameworks: [
    jitFrameworkPlatformSupportSource(),
    gradFrameworkPlatformSupportSource(),
  ],
});
if (
  view.environmentId !== "browser.release-consumer"
  || view.programSupport.decisions[0]?.state !== "supported"
  || view.frameworks[0]?.frameworkId !== "browsergrad.grad"
  || view.frameworks[0]?.operations.length !== 22
  || view.frameworks[1]?.frameworkId !== "browsergrad.jit"
  || view.frameworks[1]?.operations.length !== 36
) throw new Error("packed cross-package framework platform view invalid");
`;
  writeFileSync(join(consumer, "consumer.mjs"), source);
  writeFileSync(join(consumer, "consumer.ts"), source);
  run("node", ["consumer.mjs"], consumer);
  typecheckConsumer(consumer);
}

function typecheckConsumer(consumer) {
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

function listRelativeEntries(directory, prefix = "") {
  return readdirSync(directory, { withFileTypes: true })
    .flatMap((entry) => {
      const relativeEntry = prefix.length === 0 ? entry.name : `${prefix}/${entry.name}`;
      return entry.isDirectory()
        ? [relativeEntry, ...listRelativeEntries(join(directory, entry.name), relativeEntry)]
        : [relativeEntry];
    })
    .sort();
}

function assertRepositoryMetadata(pkg, packageDirectory) {
  assert(
    pkg.repository?.type === "git"
      && pkg.repository?.url === "https://github.com/unlocalhosted/browsergrad.git"
      && pkg.repository?.directory === `packages/${packageDirectory}`,
    `${pkg.name} repository metadata must bind npm provenance to its exact monorepo directory`,
  );
}

function assertNoWorkspaceProtocol(pkg, label) {
  for (const field of ["dependencies", "optionalDependencies", "peerDependencies"]) {
    for (const [name, range] of Object.entries(pkg[field] ?? {})) {
      assert(
        typeof range === "string" && !range.includes("workspace:"),
        `${label} leaked ${field}.${name}: ${String(range)}`,
      );
    }
  }
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

function assertCommandFails(cmd, args, cwd, expectedMessage) {
  const result = spawnSync(cmd, args, {
    cwd,
    encoding: "utf8",
    stdio: "pipe",
    env: process.env,
  });
  assert(result.error === undefined, `${cmd} ${args.join(" ")} could not execute`);
  assert(result.status !== 0, `${cmd} ${args.join(" ")} unexpectedly succeeded`);
  const output = `${result.stdout ?? ""}\n${result.stderr ?? ""}`;
  assert(
    output.includes(expectedMessage),
    `${cmd} ${args.join(" ")} failed without expected diagnostic ${JSON.stringify(expectedMessage)}:\n${output}`,
  );
}

function assertCompilerViewCopyEvidenceWorkflowOrder(workflowName, workflow) {
  const steps = workflowSteps(workflow);
  const names = workflowName === "release"
    ? {
      dependency: "Verify selected package dependency closure is published and equivalent",
      evidence: "Required compiler view-copy-binding WebGPU gate (compiler release)",
      verify: "Verify compiler view-copy-binding retained terminal record (compiler release)",
      upload: "Retain compiler view-copy-binding WebGPU evidence (compiler release)",
      stage: "Stage immutable npm artifact",
      publish: "Publish immutable artifact to npm",
    }
    : {
      dependency: "Verify publication plan and existing dependency artifacts",
      evidence: "Run required compiler view-copy-binding WebGPU gate",
      verify: "Verify compiler view-copy-binding retained terminal record",
      upload: "Retain compiler view-copy-binding WebGPU evidence",
      stage: "Stage immutable npm artifacts",
      publish: "Publish immutable artifacts",
    };
  const indexes = Object.fromEntries(Object.entries(names).map(([role, name]) => {
    const matches = steps
      .map((step, index) => ({ step, index }))
      .filter(({ step }) => step.name === name);
    assert(matches.length === 1, `${workflowName} workflow must contain exactly one compiler L2 ${role} step`);
    return [role, matches[0].index];
  }));
  assert(
    indexes.dependency < indexes.evidence
      && indexes.evidence < indexes.verify
      && indexes.verify < indexes.upload
      && indexes.upload < indexes.stage
      && indexes.stage < indexes.publish,
    `${workflowName} workflow must order dependency preflight, compiler L2 evidence, immutable staging, then publish`,
  );
  assert(
    steps[indexes.evidence].body.includes("set -o pipefail")
      && steps[indexes.evidence].body.includes("test:browser:view-copy-bindings:required")
      && steps[indexes.evidence].body.includes("compiler-view-copy-bindings-webgpu-evidence.log"),
    `${workflowName} compiler L2 evidence step must run and retain the required browser lane`,
  );
  assert(
    steps[indexes.verify].body.includes(
      'verify_view_copy_bindings_evidence_log.mjs compiler-view-copy-bindings-webgpu-evidence.log "${GITHUB_SHA}"',
    ),
    `${workflowName} compiler L2 verifier must bind the retained log to GITHUB_SHA`,
  );
  assert(
    steps[indexes.upload].body.includes("compiler-view-copy-bindings-webgpu-evidence-${{ github.sha }}"),
    `${workflowName} compiler L2 upload must retain evidence by exact SHA`,
  );
  assert(
    steps[indexes.stage].body.includes(
      "BG_REQUIRED_COMPILER_VIEW_COPY_BINDINGS_WEBGPU_EVIDENCE_COMMIT: ${{ github.sha }}",
    ),
    `${workflowName} staging must authorize only the compiler L2 evidenced SHA`,
  );
  for (const role of ["evidence", "verify"]) {
    assert(
      !/^        continue-on-error:/mu.test(steps[indexes[role]].body),
      `${workflowName} compiler L2 ${role} step cannot continue on error`,
    );
  }
  const evidenceCondition = stepField(steps[indexes.evidence], "if");
  const verifierCondition = stepField(steps[indexes.verify], "if");
  const uploadCondition = stepField(steps[indexes.upload], "if");
  if (workflowName === "release") {
    const releaseCondition = "steps.parse.outputs.shortname == 'compiler'";
    assert(
      evidenceCondition === releaseCondition && verifierCondition === releaseCondition,
      "release compiler L2 evidence and verifier conditions must be compiler-only and identical",
    );
    assert(
      uploadCondition === `always() && ${releaseCondition}`,
      "release compiler L2 upload must always run for compiler releases",
    );
  } else {
    assert(
      evidenceCondition === undefined && verifierCondition === undefined,
      "publish compiler L2 evidence and verifier steps must be unconditional",
    );
    assert(uploadCondition === "always()", "publish compiler L2 upload must always run");
  }
}

function assertSemanticGemmEvidenceWorkflowOrder(workflowName, workflow) {
  assertKernelsSemanticEvidenceWorkflowOrder(workflowName, workflow, {
    label: "semantic GEMM",
    releaseEvidence: "Required semantic GEMM WebGPU gate (kernels release)",
    releaseUpload: "Retain semantic GEMM WebGPU evidence (kernels release)",
    publishEvidence: "Run required semantic GEMM WebGPU gate",
    publishUpload: "Retain semantic GEMM WebGPU evidence",
    command: "test:browser:semantic-gemm:required",
    log: "semantic-gemm-webgpu-evidence.log",
    artifact: "semantic-gemm-webgpu-evidence-${{ github.sha }}",
    evidenceEnvironment: "BG_REQUIRED_SEMANTIC_GEMM_WEBGPU_EVIDENCE_COMMIT: ${{ github.sha }}",
  });
}

function assertSemanticAttentionEvidenceWorkflowOrder(workflowName, workflow) {
  assertKernelsSemanticEvidenceWorkflowOrder(workflowName, workflow, {
    label: "semantic attention",
    releaseEvidence: "Required semantic attention WebGPU gate (kernels release)",
    releaseUpload: "Retain semantic attention WebGPU evidence (kernels release)",
    publishEvidence: "Run required semantic attention WebGPU gate",
    publishUpload: "Retain semantic attention WebGPU evidence",
    command: "test:browser:semantic-attention:required",
    log: "semantic-attention-webgpu-evidence.log",
    artifact: "semantic-attention-webgpu-evidence-${{ github.sha }}",
    evidenceEnvironment: "BG_REQUIRED_SEMANTIC_ATTENTION_WEBGPU_EVIDENCE_COMMIT: ${{ github.sha }}",
  });
}

function assertSemanticAttentionPerformanceEvidenceWorkflowOrder(workflowName, workflow) {
  assertKernelsSemanticEvidenceWorkflowOrder(workflowName, workflow, {
    label: "semantic attention performance",
    releaseEvidence: "Required semantic attention WebGPU performance gate (kernels release)",
    releaseUpload: "Retain semantic attention WebGPU performance evidence (kernels release)",
    publishEvidence: "Run required semantic attention WebGPU performance gate",
    publishUpload: "Retain semantic attention WebGPU performance evidence",
    command: "test:browser:semantic-attention:performance:required",
    log: "semantic-attention-performance-webgpu-evidence.log",
    artifact: "semantic-attention-performance-webgpu-evidence-${{ github.sha }}",
    evidenceEnvironment:
      "BG_REQUIRED_SEMANTIC_ATTENTION_WEBGPU_PERFORMANCE_EVIDENCE_COMMIT: ${{ github.sha }}",
  });
}

function assertKernelsSemanticEvidenceWorkflowOrder(workflowName, workflow, config) {
  const steps = workflowSteps(workflow);
  const names = workflowName === "release"
    ? {
      dependency: "Verify selected package dependency closure is published and equivalent",
      evidence: config.releaseEvidence,
      upload: config.releaseUpload,
      stage: "Stage immutable npm artifact",
    }
    : {
      dependency: "Verify publication plan and existing dependency artifacts",
      evidence: config.publishEvidence,
      upload: config.publishUpload,
      stage: "Stage immutable npm artifacts",
    };
  const indexes = Object.fromEntries(Object.entries(names).map(([role, name]) => {
    const matches = steps
      .map((step, index) => ({ step, index }))
      .filter(({ step }) => step.name === name);
    assert(
      matches.length === 1,
      `${workflowName} workflow must contain exactly one ${config.label} ${role} step`,
    );
    return [role, matches[0].index];
  }));
  assert(
    indexes.dependency < indexes.evidence
      && indexes.evidence < indexes.upload
      && indexes.upload < indexes.stage,
    `${workflowName} workflow must order dependency preflight, ${config.label} evidence, then immutable staging`,
  );
  assert(
    steps[indexes.evidence].body.includes("set -o pipefail")
      && steps[indexes.evidence].body.includes(config.command)
      && steps[indexes.evidence].body.includes(config.log),
    `${workflowName} ${config.label} evidence step must run and retain the required browser lane`,
  );
  assert(
    steps[indexes.upload].body.includes(config.artifact),
    `${workflowName} ${config.label} upload must retain evidence by exact SHA`,
  );
  assert(
    steps[indexes.stage].body.includes(config.evidenceEnvironment),
    `${workflowName} staging must authorize only the ${config.label} evidenced SHA`,
  );
  assert(
    !/^        continue-on-error:/mu.test(steps[indexes.evidence].body),
    `${workflowName} ${config.label} evidence cannot continue on error`,
  );
  const evidenceCondition = stepField(steps[indexes.evidence], "if");
  const uploadCondition = stepField(steps[indexes.upload], "if");
  if (workflowName === "release") {
    const releaseCondition = "steps.parse.outputs.shortname == 'kernels'";
    assert(
      evidenceCondition === releaseCondition,
      `release ${config.label} evidence must be kernels-only`,
    );
    assert(
      uploadCondition === `always() && ${releaseCondition}`,
      `release ${config.label} upload must always run for kernels releases`,
    );
  } else {
    assert(
      evidenceCondition === undefined,
      `publish ${config.label} evidence must be unconditional`,
    );
    assert(
      uploadCondition === "always()",
      `publish ${config.label} upload must always run`,
    );
  }
}

function assertJitEvidenceWorkflowOrder(workflowName, workflow) {
  const steps = workflowSteps(workflow);
  const names = workflowName === "release"
    ? {
      dependency: "Verify selected package dependency closure is published and equivalent",
      evidence: "Required JIT-emitted semantic-permute WebGPU gate (JIT or kernels release)",
      verify: "Verify JIT semantic-permute retained terminal record (JIT or kernels release)",
      upload: "Retain JIT semantic-permute WebGPU evidence (JIT or kernels release)",
      stage: "Stage immutable npm artifact",
      publish: "Publish immutable artifact to npm",
    }
    : {
      dependency: "Verify publication plan and existing dependency artifacts",
      evidence: "Run required JIT-emitted semantic-permute WebGPU gate",
      verify: "Verify JIT semantic-permute retained terminal record",
      upload: "Retain JIT semantic-permute WebGPU evidence",
      stage: "Stage immutable npm artifacts",
      publish: "Publish immutable artifacts",
    };
  const indexes = Object.fromEntries(Object.entries(names).map(([role, name]) => {
    const matches = steps
      .map((step, index) => ({ step, index }))
      .filter(({ step }) => step.name === name);
    assert(matches.length === 1, `${workflowName} workflow must contain exactly one ${role} step`);
    return [role, matches[0].index];
  }));
  assert(
    indexes.dependency < indexes.evidence
      && indexes.evidence < indexes.verify
      && indexes.verify < indexes.upload
      && indexes.upload < indexes.stage
      && indexes.stage < indexes.publish,
    `${workflowName} workflow must order dependency preflight, JIT evidence, immutable staging, then publish`,
  );
  assert(
    steps[indexes.evidence].body.includes("set -o pipefail")
      && steps[indexes.evidence].body.includes("test:browser:semantic-permute:required"),
    `${workflowName} JIT evidence step must run the required browser lane`,
  );
  assert(
    steps[indexes.verify].body.includes(
      "verify_semantic_permute_evidence_log.mjs jit-semantic-permute-webgpu-evidence.log \"${GITHUB_SHA}\"",
    ),
    `${workflowName} JIT verifier step must validate the retained log against GITHUB_SHA`,
  );
  assert(
    steps[indexes.upload].body.includes("jit-semantic-permute-webgpu-evidence-${{ github.sha }}"),
    `${workflowName} JIT upload step must retain evidence by exact SHA`,
  );
  assert(
    steps[indexes.stage].body.includes(
      "BG_REQUIRED_JIT_SEMANTIC_PERMUTE_WEBGPU_EVIDENCE_COMMIT: ${{ github.sha }}",
    ),
    `${workflowName} staging step must authorize only the evidenced SHA`,
  );
  assert(
    steps[indexes.publish].body.includes("--publish-staged npm-release-artifacts")
      && steps[indexes.publish].body.includes("--provenance"),
    `${workflowName} publish step must publish only staged immutable artifacts with provenance`,
  );
  for (const role of ["evidence", "verify"]) {
    assert(
      !/^        continue-on-error:/mu.test(steps[indexes[role]].body),
      `${workflowName} JIT ${role} step cannot continue on error`,
    );
  }
  const evidenceCondition = stepField(steps[indexes.evidence], "if");
  const verifierCondition = stepField(steps[indexes.verify], "if");
  const uploadCondition = stepField(steps[indexes.upload], "if");
  if (workflowName === "release") {
    const releaseCondition = "contains(fromJSON('[\"jit\",\"kernels\"]'), steps.parse.outputs.shortname)";
    assert(
      stepField(steps[indexes.dependency], "if") === undefined
        && steps[indexes.dependency].body.includes('--preflight --package "${{ steps.parse.outputs.npmname }}"'),
      "release dependency prerequisite must generically verify every selected package closure",
    );
    assert(
      evidenceCondition === releaseCondition && verifierCondition === releaseCondition,
      "release JIT evidence and verifier conditions must cover identical JIT/kernels package set",
    );
    assert(
      uploadCondition === `always() && ${releaseCondition}`,
      "release JIT evidence upload must always run for identical JIT/kernels package set",
    );
  } else {
    assert(
      stepField(steps[indexes.dependency], "if") === undefined
        && steps[indexes.dependency].body.includes("publish-missing-npm.mjs --preflight")
        && evidenceCondition === undefined
        && verifierCondition === undefined,
      "publish dependency preflight, JIT evidence, and verifier steps must be unconditional",
    );
    assert(uploadCondition === "always()", "publish JIT evidence upload must always run");
  }
}

function stepField(step, field) {
  const match = new RegExp(`^        ${field}: (.+)$`, "mu").exec(step.body);
  return match?.[1];
}

function workflowSteps(workflow) {
  const steps = [];
  let current;
  for (const line of workflow.split(/\r?\n/)) {
    const match = /^      - name: (.+)$/u.exec(line);
    if (match) {
      current = { name: match[1], body: line };
      steps.push(current);
    } else if (current) {
      current.body += `\n${line}`;
    }
  }
  return steps;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}
