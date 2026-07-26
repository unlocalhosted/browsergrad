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
  prepareObservedCppCuteBrowserViewCopyCandidate,
  unwrapObservedCppCuteBrowserViewCopyCandidate,
} from "../src/cpp_cute_browser_view_copy_candidate.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "../src/cpp_cute_browser_reproducibility.js";
import {
  createCppCuteBrowserCompileProfileInput,
  deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256,
} from "../src/cpp_cute_browser_compile_profile.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../src/cpp_cute_browser_runtime_abi.js";
import {
  cppCuteBrowserRealCompileCase,
  type CppCuteBrowserRealCompileCaseId,
  type CppCuteBrowserRealCompileDType,
} from "../src/cpp_cute_browser_real_compile_cases.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  cppCuteDiagnosticNormalizationResourceBytes,
} from "../src/cpp_cute_diagnostic_normalization.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../src/cpp_cute_frontend_profile.js";
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
  cppCuteBrowserRealCompileStaticLayoutProjection,
  prepareCppCuteBrowserRealCompileViewCopyRequest,
} from "./support/cpp_cute_browser_real_compile_request.js";

interface ExternalAssetInput {
  readonly assetId: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: number;
}

type PackInspection =
  Awaited<ReturnType<typeof inspectCppCuteBrowserVfsPack>>;

interface BrowserExternalInputs {
  readonly schema: "browsergrad.compiler.cpp-cute.browser-real-compile-inputs";
  readonly version: 3;
  readonly authority: "local-exact-byte-preflight-only";
  readonly caseId: CppCuteBrowserRealCompileCaseId;
  readonly assets: readonly ExternalAssetInput[];
  readonly wasmAuthority:
    | "package-pinned-two-clean-build-output"
    | "untrusted-diagnostic-local-byte-observation-only";
  readonly pinnedReproducibleWasmMatched: boolean;
  readonly untrustedDiagnosticWasm: boolean;
  readonly headerDistributionReproducibilityId: string;
  readonly headerDistributionOutputVerificationId: string;
  readonly packagePinnedHeaderPacksMatched: true;
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
const REAL_COMPILE_CASE =
  cppCuteBrowserRealCompileCase(__BG_CPP_CUTE_REAL_COMPILE_INPUTS__.caseId);
const MAIN_PATH = REAL_COMPILE_CASE.virtualPath;
const MAIN_SOURCE = REAL_COMPILE_CASE.source;
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

it("observes unchanged CuTe view-copy source in the exact package Worker", async () => {
  const started = performance.now();
  expect(await sha256Hex(MAIN_BYTES)).toBe(REAL_COMPILE_CASE.sourceSha256);
  const externalAssets = await fetchExternalAssets();
  const environment = await prepareRealCompileEnvironment(externalAssets);
  const request = await prepareCppCuteBrowserRealCompileViewCopyRequest(
    environment.profile,
    REAL_COMPILE_CASE,
  );
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
    fileCount: 5_788,
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
  const evidenceInputs = Object.freeze({
    externalAssetCount: externalAssets.length,
    totalExternalByteLength: environment.totalExternalByteLength,
    wasmSha256: externalAssets.find((asset) => asset.assetId === "clang-wasm")?.sha256,
    wasmAuthority: __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.wasmAuthority,
    pinnedReproducibleWasmMatched:
      __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.pinnedReproducibleWasmMatched,
    untrustedDiagnosticWasm:
      __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.untrustedDiagnosticWasm,
    headerDistributionReproducibilityId:
      __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.headerDistributionReproducibilityId,
    headerDistributionOutputVerificationId:
      __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.headerDistributionOutputVerificationId,
    packagePinnedHeaderPacksMatched:
      __BG_CPP_CUTE_REAL_COMPILE_INPUTS__.packagePinnedHeaderPacksMatched,
    assetSetSha256: environment.manifest.assetSetSha256,
    packCount: vfsInstallation.packCount,
    installedFileCount: vfsInstallation.fileCount,
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
      version: 2,
      outcome: "blocked",
      authority: "local-real-browser-worker-terminal-observation-only",
      source: {
        caseId: REAL_COMPILE_CASE.caseId,
        virtualPath: MAIN_PATH,
        sourceSha256: await sha256Hex(MAIN_BYTES),
        syntax: "unchanged-cpp17-cute",
        selectedDeclaration: "copy_views",
      },
      inputs: evidenceInputs,
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
      version: 2,
      outcome: "rejected",
      authority: "local-real-browser-worker-execution-observation-only",
      source: {
        caseId: REAL_COMPILE_CASE.caseId,
        virtualPath: MAIN_PATH,
        sourceSha256: await sha256Hex(MAIN_BYTES),
        syntax: "unchanged-cpp17-cute",
        selectedDeclaration: "copy_views",
      },
      inputs: evidenceInputs,
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
    throw new Error(
      "real browser Clang-Wasm compile did not accept exactly one selected view-copy",
    );
  }
  const entryId = payload.outcome.selectedEntryIds[0];
  const candidate = await prepareObservedCppCuteBrowserViewCopyCandidate(
    execution,
    { entryId },
  );
  const candidateRecord = unwrapObservedCppCuteBrowserViewCopyCandidate(candidate);
  expect(candidate).toMatchObject({
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    sharedViewCopySemanticsPrepared: true,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  });
  expect(candidateRecord.semantics.sourceLayoutFact).toMatchObject({
    kind: "affine-layout",
    rank: REAL_COMPILE_CASE.coordinateRank,
    leafRank: REAL_COMPILE_CASE.coordinateRank,
  });
  expect(candidateRecord.semantics.destinationLayoutFact).toMatchObject({
    kind: "affine-layout",
    rank: REAL_COMPILE_CASE.coordinateRank,
    leafRank: REAL_COMPILE_CASE.coordinateRank,
  });
  expect(candidateRecord.semantics.sourceSpanElements).toBe(
    REAL_COMPILE_CASE.sourceSpanElements,
  );
  expect(candidateRecord.semantics.destinationSpanElements).toBe(
    REAL_COMPILE_CASE.destinationSpanElements,
  );
  expect(candidateRecord.semantics.dtype).toBe(
    REAL_COMPILE_CASE.dtype satisfies CppCuteBrowserRealCompileDType,
  );
  expect(cppCuteBrowserRealCompileStaticLayoutProjection(
    candidateRecord.semantics.sourceLayoutFact,
  )).toEqual(REAL_COMPILE_CASE.sourceLayout);
  expect(cppCuteBrowserRealCompileStaticLayoutProjection(
    candidateRecord.semantics.destinationLayoutFact,
  )).toEqual(REAL_COMPILE_CASE.destinationLayout);
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
    version: 2,
    outcome: "compiled",
    authority: "local-real-browser-worker-execution-observation-only",
    source: {
      caseId: REAL_COMPILE_CASE.caseId,
      virtualPath: MAIN_PATH,
      sourceSha256: await sha256Hex(MAIN_BYTES),
      syntax: "unchanged-cpp17-cute",
      selectedDeclaration: "copy_views",
    },
    inputs: evidenceInputs,
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
      entrySubjectHash: candidate.entrySubjectHash,
      sourceLayoutFactId: candidateRecord.semantics.sourceLayoutFact.factId,
      destinationLayoutFactId:
        candidateRecord.semantics.destinationLayoutFact.factId,
      sourceCoordinateRank: candidateRecord.semantics.sourceLayoutFact.rank,
      destinationCoordinateRank:
        candidateRecord.semantics.destinationLayoutFact.rank,
      sourceSpanElements:
        candidateRecord.semantics.sourceSpanElements.toString(10),
      destinationSpanElements:
        candidateRecord.semantics.destinationSpanElements.toString(10),
      dtype: candidateRecord.semantics.dtype,
      sharedViewCopySemanticsPrepared: true,
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
  const packInspections = new Map<string, PackInspection>();
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
  const sourceRootManifestSha256 =
    await deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256();
  const provisionalAssetSetSha256 = await hashCanonicalJson({
    domain:
      "browsergrad.compiler.cpp-cute.real-compile-provisional-asset-set.v1",
    buildProvenanceLockSha256,
    sourceRootManifestSha256,
  });
  const profileInput = structuredClone(createCppCuteBrowserCompileProfileInput({
    assetSetSha256: provisionalAssetSetSha256,
    buildProvenanceLockSha256,
    extractorWasmSha256: wasm.sha256,
    runtimeAbiManifestSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
    semanticAdapterManifestSha256:
      CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
    sourceRootManifestSha256,
    workerModuleSha256: workerBundle.sha256,
    workerModuleByteLength: workerBundle.byteLength,
    headerContentSets: {
      clangResource: requirePackInspection(
        packInspections,
        "clang-resource",
      ).contentSetSha256,
      cuda: requirePackInspection(packInspections, "cuda").contentSetSha256,
      cutlass:
        requirePackInspection(packInspections, "cutlass").contentSetSha256,
      cxxStdlib:
        requirePackInspection(packInspections, "cxx-stdlib").contentSetSha256,
      linuxSysroot:
        requirePackInspection(
          packInspections,
          "linux-sysroot",
        ).contentSetSha256,
    },
  }));

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

function requireExternalAsset(
  assets: readonly FetchedExternalAsset[],
  assetId: string,
): FetchedExternalAsset {
  const asset = assets.find((candidate) => candidate.assetId === assetId);
  if (asset === undefined) throw new Error(`missing preflight asset ${assetId}`);
  return asset;
}

function requirePackInspection(
  inspections: ReadonlyMap<string, PackInspection>,
  includeRootId: string,
): PackInspection {
  const inspection = inspections.get(includeRootId);
  if (inspection === undefined) {
    throw new Error(`missing ${includeRootId} pack inspection`);
  }
  return inspection;
}

function wire(value: number): WireU64 {
  return encodeWireU64(BigInt(value));
}
