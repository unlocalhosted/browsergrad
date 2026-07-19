import { createHash } from "node:crypto";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  parseCppCuteClangWasmReproducibilityArguments,
  verifyCppCuteClangWasmReproducibility,
  writeCppCuteClangWasmReproducibilityEvidence,
} from "./cpp_cute_browser_build_reproducibility.mjs";

const temporaryRoots: string[] = [];
const factoryBytes = new TextEncoder().encode(
  "const createBrowserGradCppCuteExtractor = () => {}; export default createBrowserGradCppCuteExtractor;\n",
);
const wasmBytes = Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);
const linkMapBytes = new TextEncoder().encode("browsergrad deterministic link map\n");
const clangTablegenBytes = new TextEncoder().encode("native clang tablegen\n");
const llvmTablegenBytes = new TextEncoder().encode("native llvm tablegen\n");

let lock: PreparedCppCuteBrowserBuildInputLock;

beforeAll(async () => {
  lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
});

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

interface BuildFixtureOptions {
  readonly commandSuffix?: string;
  readonly evidenceWasmBytes?: Uint8Array;
  readonly observedWasmBytes?: Uint8Array;
  readonly runtimeClosureContent?: string;
  readonly runtimeAbiReviewTag?: string;
  readonly extraEvidenceField?: boolean;
  readonly nonCanonicalEvidence?: boolean;
}

async function fixture(options: Readonly<{
  first?: BuildFixtureOptions;
  second?: BuildFixtureOptions;
}> = {}) {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-clang-wasm-repro-"));
  temporaryRoots.push(root);
  const firstRoot = join(root, "first");
  const secondRoot = join(root, "second");
  await Promise.all([mkdir(firstRoot), mkdir(secondRoot)]);
  const first = await writeBuild(firstRoot, 1, options.first);
  const second = await writeBuild(secondRoot, 2, options.second);
  return { root, firstRoot, secondRoot, first, second };
}

async function writeBuild(
  localRoot: string,
  ordinal: 1 | 2,
  options: BuildFixtureOptions = {},
) {
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
  const llvm = body.sources.find((source) => source.sourceId === "llvm-project");
  if (llvm === undefined) throw new Error("missing LLVM lock fixture");
  const recordedWorkRoot = `/browsergrad/work/build-${ordinal}`;
  const paths = {
    llvmProjectSourceRoot: `/browsergrad/inputs/build-${ordinal}/llvm-project`,
    extractorSourceRoot: `${recordedWorkRoot}/staged-extractor-source`,
    nativeBuildRoot: `${recordedWorkRoot}/native-tablegen`,
    wasmBuildRoot: `${recordedWorkRoot}/clang-extractor-wasm`,
    outputRoot: `${recordedWorkRoot}/output`,
    stateRoot: `${recordedWorkRoot}/state`,
  };
  const evidenceRoot = join(localRoot, "state", "evidence");
  const generatedRoot = join(evidenceRoot, "generated");
  const logRoot = join(evidenceRoot, "build-logs");
  const outputRoot = join(localRoot, "output");
  const distributedRoot = join(outputRoot, "browsergrad-cpp-cute");
  await Promise.all([
    mkdir(generatedRoot, { recursive: true }),
    mkdir(logRoot, { recursive: true }),
    mkdir(distributedRoot, { recursive: true }),
  ]);
  const evidenceWasm = options.evidenceWasmBytes ?? wasmBytes;
  const observedWasm = options.observedWasmBytes ?? evidenceWasm;
  await Promise.all([
    writeFile(join(generatedRoot, "clang-extractor.mjs"), factoryBytes),
    writeFile(join(generatedRoot, "clang-extractor.wasm"), observedWasm),
    writeFile(join(distributedRoot, "clang-extractor.wasm"), observedWasm),
    writeFile(join(evidenceRoot, "clang-extractor.link.map"), linkMapBytes),
  ]);

  const profiles = [
    ["native-tablegen-configure", "native-tablegen", "configure", paths.nativeBuildRoot],
    ["native-tablegen-build", "native-tablegen", "build", paths.nativeBuildRoot],
    ["clang-extractor-wasm-configure", "clang-extractor-wasm", "configure", paths.wasmBuildRoot],
    ["clang-extractor-wasm-build", "clang-extractor-wasm", "build", paths.wasmBuildRoot],
  ] as const;
  const steps = [];
  for (const [id, stageId, kind, cwd] of profiles) {
    const stdout = new TextEncoder().encode(`${recordedWorkRoot}: ${id} stdout\n`);
    const stderr = new TextEncoder().encode(`${recordedWorkRoot}: ${id} stderr\n`);
    await Promise.all([
      writeFile(join(logRoot, `${id}.stdout.log`), stdout),
      writeFile(join(logRoot, `${id}.stderr.log`), stderr),
    ]);
    steps.push({
      id,
      stageId,
      kind,
      executable: "/usr/local/bin/cmake",
      arguments: [
        kind === "configure" ? "-S" : "--build",
        kind === "configure" ? `${paths.llvmProjectSourceRoot}/llvm` : cwd,
        ...(options.commandSuffix === undefined ? [] : [options.commandSuffix]),
      ],
      cwd,
      environment: {
        HOME: `${paths.stateRoot}/home`,
        LANG: "C",
        PATH: "/usr/local/bin:/emsdk/upstream/bin:/emsdk/upstream/emscripten:/usr/bin",
        TMPDIR: `${paths.stateRoot}/tmp`,
      },
      exitCode: 0,
      terminationSignal: null,
      stdoutPath: `${paths.stateRoot}/evidence/build-logs/${id}.stdout.log`,
      stdoutSha256: sha256(stdout),
      stdoutByteLength: stdout.byteLength,
      stderrPath: `${paths.stateRoot}/evidence/build-logs/${id}.stderr.log`,
      stderrSha256: sha256(stderr),
      stderrByteLength: stderr.byteLength,
    });
  }

  const execution = {
    authority: "clang-wasm-build-execution-observation-only",
    lockId: lock.lockId,
    sourceSetSha256: lock.extractorSourceSetSha256,
    paths,
    nativeTools: {
      clangTablegen: {
        path: `${paths.nativeBuildRoot}/bin/clang-tblgen`,
        sha256: sha256(clangTablegenBytes),
        byteLength: clangTablegenBytes.byteLength,
      },
      llvmTablegen: {
        path: `${paths.nativeBuildRoot}/bin/llvm-tblgen`,
        sha256: sha256(llvmTablegenBytes),
        byteLength: llvmTablegenBytes.byteLength,
      },
    },
    stepCount: 4,
    steps,
    factoryModulePath: `${paths.stateRoot}/evidence/generated/clang-extractor.mjs`,
    factoryModuleSha256: sha256(factoryBytes),
    factoryModuleByteLength: factoryBytes.byteLength,
    wasmSidecarPath: `${paths.stateRoot}/evidence/generated/clang-extractor.wasm`,
    wasmSha256: sha256(evidenceWasm),
    wasmByteLength: evidenceWasm.byteLength,
    linkMapPath: `${paths.stateRoot}/evidence/clang-extractor.link.map`,
    linkMapSha256: sha256(linkMapBytes),
    linkMapByteLength: linkMapBytes.byteLength,
    sourceVerified: true,
    buildExecuted: true,
    factoryModuleUtf8Validated: true,
    webAssemblyValidated: true,
    abiConformanceVerified: false,
    outputIdentityAuthorized: false,
    reproducibilityVerified: false,
    releaseReady: false,
    factoryModuleDistributed: false,
  };
  const runtimeAbiProjection = {
    fixtureTag: options.runtimeAbiReviewTag ?? "matching-review",
  };
  const runtimeAbiReview = {
    authority: "review-observation-only",
    wasmSha256: execution.wasmSha256,
    wasmByteLength: execution.wasmByteLength,
    observedProjectionSha256: sha256(canonicalJsonBytes(runtimeAbiProjection)),
    runtimeAbiManifestId: body.runtimeAbiResource.manifestId,
    runtimeAbiContractSha256: "1".repeat(64),
    exactInterfaceConformance: false,
    mismatches: ["fixture module intentionally lacks the production ABI"],
    projection: runtimeAbiProjection,
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
  };
  await writeFile(
    join(outputRoot, "clang-wasm-runtime-abi-review.v1.json"),
    canonicalJsonBytes(runtimeAbiReview),
  );
  const runtimeClosureBytes = new TextEncoder().encode(
    options.runtimeClosureContent ?? "runtime closure fixture",
  );
  const runtimeClosureFiles = [{
    kind: "runtime",
    path: "packages/browsergrad-compiler/package.json",
    sha256: sha256(runtimeClosureBytes),
    byteLength: runtimeClosureBytes.byteLength,
  }];
  const runtimeClosureObservation = {
    schema: "browsergrad.compiler.cpp-cute.build-runtime-closure",
    version: 1,
    authority: "staged-build-runtime-closure-observation-only",
    lockId: lock.lockId,
    extractorSourceSetSha256: lock.extractorSourceSetSha256,
    closureSha256: sha256(canonicalJsonBytes({
      domain: "browsergrad.compiler.cpp-cute.build-runtime-closure.v1",
      files: runtimeClosureFiles,
    })),
    fileCount: runtimeClosureFiles.length,
    files: runtimeClosureFiles,
    claims: {
      exactReadableWorkspaceClosureVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
    },
  };
  const runtimeClosureObservationBytes = canonicalJsonBytes(runtimeClosureObservation);
  const evidence = {
    schema: "browsergrad.compiler.cpp-cute.clang-wasm-build-execution-observation",
    version: 2,
    authority: "build-execution-observation-only",
    lockId: lock.lockId,
    builder: {
      schema: "browsergrad.compiler.cpp-cute.builder-container-observation",
      version: 1,
      platform: "linux/amd64",
      platformManifestDigest: body.builder.platformManifestDigest,
      imageConfigDigest: body.builder.imageConfigDigest,
    },
    runtimeClosure: {
      observationSha256: sha256(runtimeClosureObservationBytes),
      observationByteLength: runtimeClosureObservationBytes.byteLength,
      observation: runtimeClosureObservation,
    },
    isolation: {
      networkInterfaces: ["lo"],
      effectiveCapabilities: "0000000000000000",
      noNewPrivileges: true,
      rootFilesystemReadOnly: true,
      inputMountsReadOnly: true,
      workMountReadWrite: true,
    },
    llvmSourceArchive: {
      sourceId: "llvm-project",
      sha256: llvm.archiveSha256,
      byteLength: llvm.archiveByteLength,
      verified: true,
    },
    execution,
    sidecarMaterialization: {
      authority: "wasm-sidecar-byte-materialization-observation-only",
      lockId: lock.lockId,
      sourceSetSha256: lock.extractorSourceSetSha256,
      generatedWasmSha256: execution.wasmSha256,
      distributedWasmSha256: execution.wasmSha256,
      wasmByteLength: execution.wasmByteLength,
      distributedWasmPath: `${paths.outputRoot}/browsergrad-cpp-cute/clang-extractor.wasm`,
      sidecarBytesMaterialized: true,
      webAssemblyValidated: false,
      abiConformanceVerified: false,
      sourceVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
      factoryModuleDistributed: false,
    },
    claims: {
      sourceArchiveVerified: true,
      buildExecuted: true,
      networkDuringBuildObservedDisabled: true,
      exactReadableWorkspaceClosureVerified: true,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      producerAttested: false,
      releaseReady: false,
    },
    ...(options.extraEvidenceField === true ? { unexpected: true } : {}),
  };
  const evidenceBytes = options.nonCanonicalEvidence === true
    ? new TextEncoder().encode(JSON.stringify(evidence, null, 2))
    : canonicalJsonBytes(evidence);
  await writeFile(join(outputRoot, "build-execution-observation.v2.json"), evidenceBytes);
  return { evidence, evidenceBytes };
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

describe("Clang-Wasm clean-build reproducibility authority", () => {
  it("verifies two distinct clean paths and matching extractor byte identities", async () => {
    const { firstRoot, secondRoot } = await fixture();
    const evidence = await verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot });

    expect(evidence).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility",
      version: 2,
      authority: "clang-wasm-extractor-reproducibility-observation-only",
      lockId: lock.lockId,
      cleanBuildCount: 2,
      comparison: {
        sourceAndBuildPathsDistinct: true,
        runtimeClosureMatched: true,
        canonicalCommandsAndEnvironmentMatched: true,
        nativeTablegenIdentitiesMatched: true,
        factoryModuleBytesMatched: true,
        wasmBytesMatched: true,
        runtimeAbiReviewBytesMatched: true,
        linkMapBytesMatched: true,
      },
      claims: {
        extractorOutputsReproducible: true,
        fullDistributedOutputSetReproducible: false,
        abiConformanceVerified: false,
        outputIdentityAuthorized: false,
        producerAttested: false,
        releaseReady: false,
      },
    });
    expect(evidence.builds).toHaveLength(2);
    expect(evidence.builds[0]?.wasmSha256).toBe(evidence.builds[1]?.wasmSha256);
    expect(evidence.builds[0]?.runtimeAbiReviewSha256).toBe(
      evidence.builds[1]?.runtimeAbiReviewSha256,
    );
  });

  it("rejects reused or overlapping clean-build roots", async () => {
    const { firstRoot } = await fixture();
    await expect(verifyCppCuteClangWasmReproducibility({
      firstRoot,
      secondRoot: firstRoot,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-CONFLICT",
      path: "$input",
    });
  });

  it("rejects observed bytes that do not match their build evidence", async () => {
    const { firstRoot, secondRoot } = await fixture({
      second: { observedWasmBytes: Uint8Array.of(...wasmBytes, 0x00) },
    });
    await expect(verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH",
      path: "$builds[1].generatedWasm.byteLength",
    });
  });

  it("rejects byte differences between individually self-consistent builds", async () => {
    const differentWasm = Uint8Array.of(...wasmBytes, 0x00);
    const { firstRoot, secondRoot } = await fixture({
      second: { evidenceWasmBytes: differentWasm },
    });
    await expect(verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH",
      path: "$comparison.wasm",
    });
  });

  it("rejects different staged runtime closures across clean builds", async () => {
    const { firstRoot, secondRoot } = await fixture({
      second: { runtimeClosureContent: "different runtime closure" },
    });
    await expect(verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH",
      path: "$comparison.runtimeClosure",
    });
  });

  it("rejects different runtime-ABI review sidecars across clean builds", async () => {
    const { firstRoot, secondRoot } = await fixture({
      second: { runtimeAbiReviewTag: "drifted-review" },
    });
    await expect(verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH",
      path: "$comparison.runtimeAbiReview",
    });
  });

  it("rejects command drift after canonical path substitution", async () => {
    const { firstRoot, secondRoot } = await fixture({ second: { commandSuffix: "--drift" } });
    await expect(verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH",
      path: "$comparison.commands",
    });
  });

  it.each([
    ["noncanonical JSON", { nonCanonicalEvidence: true }],
    ["unknown evidence fields", { extraEvidenceField: true }],
  ])("rejects %s", async (_name, second) => {
    const { firstRoot, secondRoot } = await fixture({ second });
    await expect(verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-INVALID",
    });
  });

  it("writes canonical no-clobber read-only evidence", async () => {
    const { root, firstRoot, secondRoot } = await fixture();
    const evidence = await verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot });
    const outputPath = join(root, "reproducibility.v2.json");
    const result = await writeCppCuteClangWasmReproducibilityEvidence(outputPath, evidence);

    const bytes = await readFile(outputPath);
    expect(result).toEqual({
      outputPath,
      sha256: sha256(bytes),
      byteLength: bytes.byteLength,
    });
    expect([...bytes]).toEqual([...canonicalJsonBytes(evidence)]);
    expect((await stat(outputPath)).mode & 0o222).toBe(0);
    await expect(writeCppCuteClangWasmReproducibilityEvidence(outputPath, evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-CONFLICT",
      path: "$outputPath",
    });
    await chmod(outputPath, 0o600);
  });

  it("refuses to serialize forged reproducibility objects", async () => {
    const { root, firstRoot, secondRoot } = await fixture();
    const evidence = await verifyCppCuteClangWasmReproducibility({ firstRoot, secondRoot });
    await expect(writeCppCuteClangWasmReproducibilityEvidence(
      join(root, "forged.json"),
      { ...evidence },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-INVALID",
      path: "$evidence",
    });
  });
});

describe("Clang-Wasm reproducibility CLI arguments", () => {
  it("accepts only the exact named normalized absolute paths", () => {
    expect(parseCppCuteClangWasmReproducibilityArguments([
      "--second-root=/work/second",
      "--output=/work/reproducibility.json",
      "--first-root=/work/first",
    ])).toEqual({
      "first-root": "/work/first",
      output: "/work/reproducibility.json",
      "second-root": "/work/second",
    });
    expect(() => parseCppCuteClangWasmReproducibilityArguments([
      "--first-root=/work/first",
    ])).toThrow();
    expect(() => parseCppCuteClangWasmReproducibilityArguments([
      "--first-root=relative",
      "--second-root=/work/second",
      "--output=/work/reproducibility.json",
    ])).toThrow();
  });
});
