import {
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { expect, it } from "vitest";
import {
  decodeAcquiredCppCuteBrowserDiagnosticNormalizationAsset,
  decodeAcquiredCppCuteBrowserRuntimeAbiAsset,
  decodeAcquiredCppCuteBrowserSemanticAdapterAsset,
  installCppCuteBrowserVfs,
  verifyTransferredCppCuteBrowserAssetSet,
  type CppCuteBrowserTransferredAssetInput,
} from "../src/cpp_cute_browser_asset_installation.js";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetManifestId,
  deriveCppCuteBrowserAssetSetSha256,
  prepareCppCuteBrowserAssetManifest,
  unwrapPreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "../src/cpp_cute_browser_assets.js";
import {
  prepareObservedCppCuteBrowserLayoutCandidate,
  unwrapObservedCppCuteBrowserLayoutCandidate,
} from "../src/cpp_cute_browser_layout_candidate.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "../src/cpp_cute_browser_reproducibility.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  cppCuteDiagnosticNormalizationResourceBytes,
} from "../src/cpp_cute_diagnostic_normalization.js";
import {
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestSourceFileV1,
  type CppCuteFrontendRequestV1,
  type PreparedCppCuteFrontendRequest,
} from "../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../src/cpp_cute_frontend_types.js";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
} from "../src/cpp_cute_frontend_artifact.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
  cppCuteSemanticAdapterManifestResourceBytes,
} from "../src/cpp_cute_semantic_adapter_manifest.js";
import {
  inspectCppCuteBrowserVfsPack,
} from "../src/cpp_cute_browser_vfs_pack.js";
import {
  CppCuteBrowserWorkerReportedFailureError,
  executeCppCuteBrowserWorker,
  unwrapObservedCppCuteBrowserWorkerExecution,
} from "../src/cpp_cute_browser_worker_controller.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
} from "../src/cpp_cute_browser_worker_bundle.js";
import {
  unwrapValidatedCppCuteBrowserWorkerResultFrame,
} from "../src/cpp_cute_browser_worker_protocol.js";
import {
  executeCppCuteBrowserPackageWasmVerifier,
  inspectObservedCppCuteBrowserPackageWasmConformance,
} from "../src/cpp_cute_browser_wasm_verifier_controller.js";
import {
  createCppCuteBrowserProfileInput,
} from "../tests/compiler/support/cpp_cute_frontend_fixtures.js";

interface ExternalAssetInput {
  readonly assetId: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: number;
}

interface BrowserExternalInputs {
  readonly schema: "browsergrad.compiler.cpp-cute.browser-real-compile-inputs";
  readonly version: 2;
  readonly authority: "local-exact-byte-preflight-only";
  readonly assets: readonly ExternalAssetInput[];
  readonly wasmAuthority:
    | "package-pinned-two-clean-build-output"
    | "untrusted-diagnostic-local-byte-observation-only";
  readonly pinnedReproducibleWasmMatched: boolean;
  readonly untrustedDiagnosticWasm: boolean;
  readonly headerDistributionLicenseApproved: false;
  readonly producerTrusted: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

declare const __BG_CPP_CUTE_REAL_COMPILE_INPUTS__: BrowserExternalInputs;
declare const __BG_CPP_CUTE_REAL_COMPILE_ROUTE_PREFIX__: string;

const BUILD_PROVENANCE_POLICY_BUILDER =
  "https://browsergrad.dev/local-observation/untrusted-builder";
const EVIDENCE_MARKER = "BROWSERGRAD_CPP_CUTE_REAL_COMPILE_EVIDENCE=";
const MAIN_PATH = "/workspace/src/real-layout.cu";
const MAIN_SOURCE = [
  "#include <cute/layout.hpp>",
  "using namespace cute;",
  "auto layout = make_layout(",
  "  make_shape(Int<4>{}, Int<2>{}),",
  "  make_stride(Int<1>{}, Int<4>{})",
  ");",
  "",
].join("\n");
const MAIN_BYTES = new TextEncoder().encode(MAIN_SOURCE);
const PACK_ROOT_IDS = Object.freeze([
  "clang-resource",
  "cuda",
  "cutlass",
  "cxx-stdlib",
  "linux-sysroot",
] as const);

interface FetchedExternalAsset extends ExternalAssetInput {
  readonly bytes: Uint8Array;
}

interface RealCompileEnvironment {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly manifest: PreparedCppCuteBrowserAssetManifest;
  readonly transferredAssets: readonly CppCuteBrowserTransferredAssetInput[];
  readonly totalExternalByteLength: number;
}

it("observes unchanged CuTe layout source in the exact package Worker", async () => {
  const started = performance.now();
  const externalAssets = await fetchExternalAssets();
  const environment = await prepareRealCompileEnvironment(externalAssets);
  const request = await prepareRealLayoutRequest(environment.profile);
  const transferable = environment.transferredAssets.map((entry) => ({
    assetId: entry.assetId,
    bytes: new Uint8Array(entry.bytes),
  }));
  const assetSet = await verifyTransferredCppCuteBrowserAssetSet(
    environment.manifest,
    transferable,
  );
  for (const entry of transferable) entry.bytes.fill(0);

  const [runtimeAbiAsset, semanticAdapterAsset, diagnosticNormalizationAsset] =
    await Promise.all([
      decodeAcquiredCppCuteBrowserRuntimeAbiAsset(assetSet),
      decodeAcquiredCppCuteBrowserSemanticAdapterAsset(assetSet),
      decodeAcquiredCppCuteBrowserDiagnosticNormalizationAsset(assetSet),
    ]);
  expect(runtimeAbiAsset).toMatchObject({
    observedWasmVerified: false,
    workerExecutionReady: false,
    releaseReady: false,
  });
  expect(semanticAdapterAsset).toMatchObject({
    designAuthority: true,
    clangInvocationAuthorized: false,
    releaseReady: false,
  });
  expect(diagnosticNormalizationAsset).toMatchObject({
    designAuthority: true,
    diagnosticNormalizationPerformed: false,
    releaseReady: false,
  });

  const vfsInstallation = await installCppCuteBrowserVfs(assetSet);
  expect(vfsInstallation).toMatchObject({
    packCount: 5,
    fileCount: 5_768,
  });

  const independentWasmConformance =
    inspectObservedCppCuteBrowserPackageWasmConformance(
      await executeCppCuteBrowserPackageWasmVerifier({
        assetSet,
        runtimeAbiAsset,
      }),
    );
  expect(independentWasmConformance).toMatchObject({
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    verifierWorkerExecutionObserved: true,
    compilerWorkerExecutionObserved: false,
    releaseReady: false,
  });

  const compileStarted = performance.now();
  let execution: Awaited<ReturnType<typeof executeCppCuteBrowserWorker>>;
  try {
    execution = await executeCppCuteBrowserWorker({
      profile: environment.profile,
      assetManifest: environment.manifest,
      vfsInstallation,
      request,
      runtimeAbiAsset,
    });
  } catch (cause) {
    const compileElapsedMilliseconds = performance.now() - compileStarted;
    if (!(cause instanceof CppCuteBrowserWorkerReportedFailureError)) throw cause;
    expect(cause.workerFailure).toMatchObject({
      kind: "browsergrad-cpp-cute-worker-failure",
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    const evidence = Object.freeze({
      schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
      version: 1,
      outcome: "blocked",
      authority: "local-real-browser-worker-terminal-observation-only",
      source: {
        virtualPath: MAIN_PATH,
        sourceSha256: await sha256Hex(MAIN_BYTES),
        syntax: "unchanged-cpp17-cute",
        selectedDeclaration: "layout",
      },
    inputs: {
      externalAssetCount: externalAssets.length,
      totalExternalByteLength: environment.totalExternalByteLength,
      wasmSha256: externalAssets.find((asset) => asset.assetId === "clang-wasm")?.sha256,
      wasmAuthority: __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.wasmAuthority,
      pinnedReproducibleWasmMatched:
        __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.pinnedReproducibleWasmMatched,
      untrustedDiagnosticWasm:
        __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.untrustedDiagnosticWasm,
        assetSetSha256: environment.manifest.assetSetSha256,
        packCount: vfsInstallation.packCount,
        installedFileCount: vfsInstallation.fileCount,
      },
      execution: {
        invocationId: cause.workerFailure.invocationId,
        requestId: request.requestId,
        compilerWorkerTerminalObserved: true,
        phase: cause.workerFailure.phase,
        failureCode: cause.workerFailure.failureCode,
        failurePath: cause.workerFailure.failurePath,
        failureDetail: cause.workerFailure.failureDetail,
        compileElapsedMilliseconds: Math.round(compileElapsedMilliseconds),
        totalElapsedMilliseconds: Math.round(performance.now() - started),
        verifierEvidenceId: independentWasmConformance.evidenceId,
        rawWasmVerified: true,
        exactInterfaceConformanceObserved: true,
        verifierWorkerExecutionObserved: true,
        workerExecutionObserved: false,
      },
      blocker: {
        seam: "producer-or-artifact-v3",
        status: "internal-error",
        code: cause.workerFailure.failureCode,
        path: cause.workerFailure.failurePath,
        detail: cause.workerFailure.failureDetail,
      },
      headerDistributionLicenseApproved: false,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
      workerExecutionObserved: false,
    });
    console.warn(`${EVIDENCE_MARKER}${JSON.stringify(evidence)}`);
    return;
  }
  const compileElapsedMilliseconds = performance.now() - compileStarted;
  expect(execution).toMatchObject({
    authority: "host-owned-browser-worker-execution",
    acceptedTerminalMessages: "1",
    workerExecutionObserved: true,
    loweringAuthorityMinted: false,
    releaseReady: false,
  });

  const executionRecord = unwrapObservedCppCuteBrowserWorkerExecution(execution);
  const frame = unwrapValidatedCppCuteBrowserWorkerResultFrame(
    executionRecord.validatedResultFrame,
  );
  const artifact = unwrapVerifiedCppCuteFrontendArtifact(frame.artifact);
  const payload = artifact.envelope.payload;
  if (payload.outcome.kind === "rejected") {
    const evidence = Object.freeze({
      schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
      version: 1,
      outcome: "rejected",
      authority: "local-real-browser-worker-execution-observation-only",
      source: {
        virtualPath: MAIN_PATH,
        sourceSha256: await sha256Hex(MAIN_BYTES),
        syntax: "unchanged-cpp17-cute",
        selectedDeclaration: "layout",
      },
      inputs: {
        externalAssetCount: externalAssets.length,
        totalExternalByteLength: environment.totalExternalByteLength,
        wasmSha256: externalAssets.find((asset) => asset.assetId === "clang-wasm")?.sha256,
        wasmAuthority: __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.wasmAuthority,
        pinnedReproducibleWasmMatched:
          __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.pinnedReproducibleWasmMatched,
        untrustedDiagnosticWasm:
          __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.untrustedDiagnosticWasm,
        assetSetSha256: environment.manifest.assetSetSha256,
        packCount: vfsInstallation.packCount,
        installedFileCount: vfsInstallation.fileCount,
      },
      execution: {
        evidenceId: execution.evidenceId,
        invocationId: execution.invocationId,
        requestId: execution.requestId,
        artifactId: frame.artifact.artifactId,
        artifactHash: frame.artifact.artifactHash,
        artifactOutcome: frame.artifact.outcome,
        acceptedTerminalMessages: execution.acceptedTerminalMessages,
        hostElapsedMicroseconds: execution.hostElapsedMicroseconds,
        compileElapsedMilliseconds: Math.round(compileElapsedMilliseconds),
        totalElapsedMilliseconds: Math.round(performance.now() - started),
        openedSourceFiles: frame.result.openedInputs.openedSourceFiles,
        openedHeaderFiles: frame.result.openedInputs.openedHeaderFiles,
        verifierEvidenceId: independentWasmConformance.evidenceId,
        rawWasmVerified: true,
        exactInterfaceConformanceObserved: true,
        verifierWorkerExecutionObserved: true,
        workerExecutionObserved: true,
      },
      blocker: {
        seam: "clang-frontend-diagnostics",
        status: "canonical-rejected-artifact",
        blockingDiagnosticIds: payload.outcome.blockingDiagnosticIds,
        diagnostics: payload.diagnostics.map((diagnostic) => ({
          diagnosticId: diagnostic.diagnosticId,
          phase: diagnostic.phase,
          severity: diagnostic.severity,
          code: diagnostic.code,
          renderedMessage: diagnostic.renderedMessage,
        })),
      },
      headerDistributionLicenseApproved: false,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
      workerExecutionObserved: true,
    });
    console.warn(`${EVIDENCE_MARKER}${JSON.stringify(evidence)}`);
    return;
  }
  if (payload.outcome.selectedEntryIds.length !== 1 ||
      payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("real browser Clang-Wasm compile did not accept exactly one selected layout");
  }
  const entryId = payload.outcome.selectedEntryIds[0];
  const candidate = await prepareObservedCppCuteBrowserLayoutCandidate(
    execution,
    { entryId },
  );
  const candidateRecord = unwrapObservedCppCuteBrowserLayoutCandidate(candidate);
  expect(candidate).toMatchObject({
    coordinateRank: 2,
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    sharedLayoutSemanticsPrepared: true,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  });
  expect(candidateRecord.semantics.fact).toMatchObject({
    kind: "affine-layout",
    rank: 2,
    leafRank: 2,
  });
  const wasmConformance = inspectObservedCppCuteBrowserPackageWasmConformance(
    executionRecord.packageInvocationLineage.observedWasmConformance,
  );
  expect(wasmConformance).toMatchObject({
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    verifierWorkerExecutionObserved: true,
    compilerWorkerExecutionObserved: false,
    releaseReady: false,
  });

  const evidence = Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
    version: 1,
    outcome: "compiled",
    authority: "local-real-browser-worker-execution-observation-only",
    source: {
      virtualPath: MAIN_PATH,
      sourceSha256: await sha256Hex(MAIN_BYTES),
      syntax: "unchanged-cpp17-cute",
      selectedDeclaration: "layout",
    },
      inputs: {
        externalAssetCount: externalAssets.length,
        totalExternalByteLength: environment.totalExternalByteLength,
        wasmSha256: externalAssets.find((asset) => asset.assetId === "clang-wasm")?.sha256,
        wasmAuthority: __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.wasmAuthority,
        pinnedReproducibleWasmMatched:
          __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.pinnedReproducibleWasmMatched,
        untrustedDiagnosticWasm:
          __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.untrustedDiagnosticWasm,
      assetSetSha256: environment.manifest.assetSetSha256,
      packCount: vfsInstallation.packCount,
      installedFileCount: vfsInstallation.fileCount,
    },
    execution: {
      evidenceId: execution.evidenceId,
      invocationId: execution.invocationId,
      requestId: execution.requestId,
      artifactId: frame.artifact.artifactId,
      artifactHash: frame.artifact.artifactHash,
      artifactOutcome: frame.artifact.outcome,
      acceptedTerminalMessages: execution.acceptedTerminalMessages,
      hostElapsedMicroseconds: execution.hostElapsedMicroseconds,
      compileElapsedMilliseconds: Math.round(compileElapsedMilliseconds),
      totalElapsedMilliseconds: Math.round(performance.now() - started),
      openedSourceFiles: frame.result.openedInputs.openedSourceFiles,
      openedHeaderFiles: frame.result.openedInputs.openedHeaderFiles,
      verifierEvidenceId: wasmConformance.evidenceId,
      rawWasmVerified: wasmConformance.rawWasmVerified,
      exactInterfaceConformanceObserved:
        wasmConformance.exactInterfaceConformanceObserved,
      verifierWorkerExecutionObserved:
        wasmConformance.verifierWorkerExecutionObserved,
      workerExecutionObserved: true,
    },
    semanticCandidate: {
      candidateId: candidate.candidateId,
      entryId: candidate.entryId,
      layoutSemanticHash: candidate.layoutSemanticHash,
      indexMapId: candidate.indexMapId,
      coordinateRank: candidate.coordinateRank,
      sharedLayoutSemanticsPrepared: true,
    },
    headerDistributionLicenseApproved: false,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
    workerExecutionObserved: true,
  });
  console.warn(`${EVIDENCE_MARKER}${JSON.stringify(evidence)}`);
}, 240_000);

async function fetchExternalAssets(): Promise<readonly FetchedExternalAsset[]> {
  const observed: FetchedExternalAsset[] = [];
  for (const expected of __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.assets) {
    const response = await fetch(
      `${__BG_CPP_CUTE_REAL_COMPILE_ROUTE_PREFIX__}${encodeURIComponent(expected.assetId)}`,
      {
        method: "GET",
        credentials: "same-origin",
        redirect: "error",
        cache: "no-store",
      },
    );
    if (!response.ok || response.redirected ||
        response.headers.get("content-length") !== String(expected.byteLength)) {
      throw new Error(`external asset ${expected.assetId} failed exact HTTP acquisition`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    const sha256 = await sha256Hex(bytes);
    if (bytes.byteLength !== expected.byteLength || sha256 !== expected.sha256) {
      throw new Error(`external asset ${expected.assetId} changed after host preflight`);
    }
    observed.push(Object.freeze({ ...expected, bytes }));
  }
  return Object.freeze(observed);
}

async function prepareRealCompileEnvironment(
  externalAssets: readonly FetchedExternalAsset[],
): Promise<RealCompileEnvironment> {
  const reproducibility = await verifyCppCuteBrowserReproducibilityResource(
    cppCuteBrowserReproducibilityResourceBytes(),
  );
  const wasm = requireExternalAsset(externalAssets, "clang-wasm");
  const matchesReproducibility =
    wasm.sha256 === reproducibility.wasmSha256 &&
    wasm.byteLength === reproducibility.wasmByteLength;
  if (matchesReproducibility !==
      __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.pinnedReproducibleWasmMatched) {
    throw new Error("external Clang-Wasm differs from package reproducibility evidence");
  }
  const packInspections = new Map<string, Awaited<ReturnType<typeof inspectCppCuteBrowserVfsPack>>>();
  for (const includeRootId of PACK_ROOT_IDS) {
    const asset = requireExternalAsset(externalAssets, includeRootId);
    const inspection = await inspectCppCuteBrowserVfsPack(asset.bytes);
    if (inspection.packSha256 !== asset.sha256 ||
        inspection.packByteLength !== String(asset.byteLength)) {
      throw new Error(`external ${includeRootId} pack differs from its host preflight`);
    }
    packInspections.set(includeRootId, inspection);
  }

  const workerBundle = inspectVerifiedCppCuteBrowserWorkerBundle(
    await verifyCppCuteBrowserWorkerBundle(),
  );
  const buildProvenanceLockSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.local-real-compile-input-lock.v1",
    assets: externalAssets.map(({ bytes: _bytes, ...asset }) => asset),
    workerBundle: {
      sha256: workerBundle.sha256,
      byteLength: workerBundle.byteLength,
    },
  });
  const profileInput = structuredClone(createCppCuteBrowserProfileInput({
    workerModuleSha256: workerBundle.sha256,
    workerModuleByteLength: workerBundle.byteLength,
    runtimeAbiManifestSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
    buildProvenanceLockSha256,
  }));
  (profileInput.language as {
    options: typeof profileInput.language.options;
  }).options = profileInput.language.options.filter(
    (option) => option.kind !== "forced-include",
  );
  (profileInput.toolchain.compiler as {
    binarySha256: string;
    resourceDirectorySha256: string;
  }).binarySha256 = wasm.sha256;
  (profileInput.deployment.extractor as {
    binarySha256: string;
    semanticAdapterManifestSha256: string;
  }).binarySha256 = wasm.sha256;
  (profileInput.deployment.extractor as {
    binarySha256: string;
    semanticAdapterManifestSha256: string;
  }).semanticAdapterManifestSha256 =
      CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256;
  for (const root of profileInput.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const inspection = packInspections.get(root.includeRootId);
    if (inspection === undefined) {
      throw new Error(`prepared profile has no exact ${root.includeRootId} pack`);
    }
    (root as { manifestSha256: string }).manifestSha256 =
      inspection.contentSetSha256;
    if (root.owner.kind === "compiler-resource-directory") {
      (profileInput.toolchain.compiler as {
        resourceDirectorySha256: string;
      }).resourceDirectorySha256 = inspection.contentSetSha256;
    } else {
      const dependency = profileInput.toolchain.dependencies.find(
        (entry) => entry.dependencyId === root.owner.dependencyId,
      );
      if (dependency === undefined) {
        throw new Error(`prepared profile lost ${root.owner.dependencyId}`);
      }
      (dependency as { headerSetSha256: string }).headerSetSha256 =
        inspection.contentSetSha256;
      if (dependency.dependencyId === "cxx-stdlib") {
        (dependency as {
          version: string;
          revision: string;
        }).version = "22.1.8";
        (dependency as {
          version: string;
          revision: string;
        }).revision = "llvmorg-22.1.8";
      }
    }
  }

  const provisional = await prepareCppCuteFrontendProfile(profileInput);
  const provisionalProfile = unwrapPreparedCppCuteBrowserFrontendProfile(provisional).profile;
  const sourceAbi = cppCuteBrowserSourceAbi(provisional);
  const sourceAbiSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi,
  });
  const buildSubjectSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.local-real-compile-build-subject.v1",
    sourceAbiSha256,
    assets: externalAssets.map(({ bytes: _bytes, ...asset }) => asset),
    workerBundle: {
      sha256: workerBundle.sha256,
      byteLength: workerBundle.byteLength,
      factorySha256: workerBundle.factorySha256,
      factoryByteLength: workerBundle.factoryByteLength,
    },
  });
  const buildSubjectId = `bg.cpp.browser-build-subject.sha256.${buildSubjectSha256}`;
  const buildProvenancePolicy = {
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: buildProvenanceLockSha256,
    builderIds: [BUILD_PROVENANCE_POLICY_BUILDER],
  } as const;
  const inlineAssets = new Map<string, Uint8Array>([
    ["adapter", cppCuteSemanticAdapterManifestResourceBytes()],
    ["diagnostic-normalization", cppCuteDiagnosticNormalizationResourceBytes()],
    ["runtime-abi", cppCuteBrowserRuntimeAbiManifestResourceBytes()],
  ]);
  const assets: CppCuteBrowserAssetV1[] = [
    {
      assetId: "adapter",
      kind: "semantic-adapter-manifest",
      url: "/__browsergrad_cpp_cute_real_compile_inline__/adapter",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
      byteLength: wire(inlineAssets.get("adapter")?.byteLength ?? 0),
      unpackedByteLength: wire(inlineAssets.get("adapter")?.byteLength ?? 0),
      mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
      buildSubjectId,
    },
    {
      assetId: "clang-wasm",
      kind: "clang-extractor-wasm",
      url: `${__BG_CPP_CUTE_REAL_COMPILE_ROUTE_PREFIX__}clang-wasm`,
      urlPolicy: "same-origin-root-relative",
      sha256: wasm.sha256,
      byteLength: wire(wasm.byteLength),
      unpackedByteLength: wire(wasm.byteLength),
      mediaType: "application/wasm",
      compression: "identity",
      buildSubjectId,
      sourceAbiSha256,
    },
    {
      assetId: "diagnostic-normalization",
      kind: "diagnostic-normalization-manifest",
      url: "/__browsergrad_cpp_cute_real_compile_inline__/diagnostic-normalization",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
      byteLength: wire(inlineAssets.get("diagnostic-normalization")?.byteLength ?? 0),
      unpackedByteLength: wire(inlineAssets.get("diagnostic-normalization")?.byteLength ?? 0),
      mediaType: "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      compression: "identity",
      buildSubjectId,
    },
    {
      assetId: "runtime-abi",
      kind: "runtime-abi-manifest",
      url: "/__browsergrad_cpp_cute_real_compile_inline__/runtime-abi",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      byteLength: wire(inlineAssets.get("runtime-abi")?.byteLength ?? 0),
      unpackedByteLength: wire(inlineAssets.get("runtime-abi")?.byteLength ?? 0),
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      compression: "identity",
      buildSubjectId,
      runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
      runtimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    },
  ];
  for (const root of provisionalProfile.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const pack = requireExternalAsset(externalAssets, root.includeRootId);
    const inspection = packInspections.get(root.includeRootId);
    if (inspection === undefined) throw new Error(`missing ${root.includeRootId} pack inspection`);
    const common = {
      assetId: root.includeRootId,
      url: `${__BG_CPP_CUTE_REAL_COMPILE_ROUTE_PREFIX__}${encodeURIComponent(root.includeRootId)}`,
      urlPolicy: "same-origin-root-relative" as const,
      sha256: pack.sha256,
      byteLength: wire(pack.byteLength),
      unpackedByteLength: wire(pack.byteLength),
      fileContentByteLength: inspection.fileContentByteLength,
      mediaType: "application/vnd.browsergrad.vfs-pack.v1" as const,
      compression: "identity" as const,
      buildSubjectId,
      includeRootId: root.includeRootId,
      mountedVirtualRoot: root.virtualPath,
      contentSetSha256: inspection.contentSetSha256,
    };
    assets.push(root.owner.kind === "compiler-resource-directory"
      ? { ...common, kind: "compiler-resource-pack" }
      : {
          ...common,
          kind: "dependency-header-pack",
          dependencyId: root.owner.dependencyId,
        });
  }
  assets.sort((left, right) => left.assetId < right.assetId ? -1 : left.assetId > right.assetId ? 1 : 0);
  const dependencyIds = provisionalProfile.toolchain.dependencies.map(
    (entry) => entry.dependencyId,
  );
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256,
    dependencyIds,
    buildSubjectIds: [buildSubjectId],
    buildProvenancePolicy,
    mountedVirtualRoots,
    assets,
  });
  (profileInput.deployment as { assetSetSha256: string }).assetSetSha256 =
    assetSetSha256;
  const profile = await prepareCppCuteFrontendProfile(profileInput);
  const compressedByteLength = assets.reduce(
    (total, asset) => total + BigInt(asset.byteLength),
    0n,
  );
  const fileContentByteLength = assets.reduce(
    (total, asset) => total + (
      asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
        ? BigInt(asset.fileContentByteLength)
        : 0n
    ),
    0n,
  );
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    profileHash: profile.profileHash,
    sourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds,
    buildSubjectIds: [buildSubjectId],
    buildProvenancePolicy,
    mountedVirtualRoots,
    assets,
    totals: {
      compressedByteLength: compressedByteLength.toString() as WireU64,
      unpackedByteLength: compressedByteLength.toString() as WireU64,
      fileContentByteLength: fileContentByteLength.toString() as WireU64,
    },
  };
  const manifestInput: CppCuteBrowserAssetManifestV1 = {
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
      minor: CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
    },
    manifestId: await deriveCppCuteBrowserAssetManifestId(body),
    body,
  };
  const manifest = await prepareCppCuteBrowserAssetManifest(manifestInput, profile);
  const externalById = new Map(externalAssets.map((asset) => [asset.assetId, asset.bytes]));
  const transferredAssets = unwrapPreparedCppCuteBrowserAssetManifest(
    manifest,
  ).manifest.body.assets.map((asset) => {
    const bytes = externalById.get(asset.assetId) ?? inlineAssets.get(asset.assetId);
    if (bytes === undefined) throw new Error(`missing transferred bytes for ${asset.assetId}`);
    return Object.freeze({ assetId: asset.assetId, bytes: new Uint8Array(bytes) });
  });
  return Object.freeze({
    profile,
    manifest,
    transferredAssets: Object.freeze(transferredAssets),
    totalExternalByteLength: externalAssets.reduce(
      (total, asset) => total + asset.byteLength,
      0,
    ),
  });
}

async function prepareRealLayoutRequest(
  profile: PreparedCppCuteFrontendProfile,
): Promise<PreparedCppCuteFrontendRequest> {
  const source = await sourceFile();
  const tokenBegin = MAIN_SOURCE.indexOf("layout =");
  if (tokenBegin < 0) throw new Error("pinned real source lost its layout declaration");
  const anchor = {
    virtualPath: MAIN_PATH,
    beginByte: encodeWireU64(BigInt(tokenBegin)),
    endByte: encodeWireU64(BigInt(tokenBegin + "layout".length)),
    tokenSha256: await sha256Hex(
      MAIN_BYTES.subarray(tokenBegin, tokenBegin + "layout".length),
    ),
  };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor,
  };
  const entryRequest: CppCuteFrontendEntryRequestV1 = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: { major: 1, minor: 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: MAIN_PATH,
    files: [source],
    entryRequests: [entryRequest],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: requestLimits(profile),
  };
  const request: CppCuteFrontendRequestV1 = {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
  return prepareCppCuteFrontendRequest(profile, request, [{
    virtualPath: MAIN_PATH,
    bytes: MAIN_BYTES,
  }]);
}

async function sourceFile(): Promise<CppCuteFrontendRequestSourceFileV1> {
  const body = {
    role: "main-source" as const,
    virtualPath: MAIN_PATH,
    contentSha256: await sha256Hex(MAIN_BYTES),
    byteLength: wire(MAIN_BYTES.byteLength),
    includeRootId: null,
  };
  return { ...body, fileId: await deriveCppCuteFrontendSourceFileId(body) };
}

function requestLimits(
  profile: PreparedCppCuteFrontendProfile,
): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    // The unchanged CuTe source opens 520 distinct headers. Keep bounded
    // headroom while remaining well below Artifact V3's 4,096-file ceiling.
    maxHeaderFiles: 1_024,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
    maxTemplateDepth: limits.maxTemplateDepth,
    maxDeclarations: 16_384,
    maxTypes: 16_384,
    maxConstants: 16_384,
    maxLayouts: 1_024,
    maxTensors: 1_024,
    maxOperations: 4_096,
    maxTargetIntrinsics: 1_024,
    maxDiagnostics: 4_096,
    maxOutputBytes: limits.maxOutputBytes,
  };
}

function requireExternalAsset(
  assets: readonly FetchedExternalAsset[],
  assetId: string,
): FetchedExternalAsset {
  const asset = assets.find((candidate) => candidate.assetId === assetId);
  if (asset === undefined) throw new Error(`missing preflight asset ${assetId}`);
  return asset;
}

function wire(value: number): WireU64 {
  return encodeWireU64(BigInt(value));
}
