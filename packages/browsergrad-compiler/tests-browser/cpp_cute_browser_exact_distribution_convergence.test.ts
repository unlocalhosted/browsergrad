import {
  prepareViewCopyCpu,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  canonicalJsonBytes,
  decodeWireJson,
  encodeWireU64,
  hashNamedComponents,
  sha256Hex,
  type DecodeLimits,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  createDevice,
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
  type KernelDevice,
} from "@unlocalhosted/browsergrad-kernels";
import { expect, it } from "vitest";
import {
  acquireWebGpuEvidenceDevice,
  createWebGpuExecutionEnvironmentRecord,
  nextWebGpuEvidenceTask,
  requiredEvidenceFailure,
  webGpuSemanticDeviceLimits,
  withWebGpuEvidenceTimeout,
} from "../../../test-support/webgpu-evidence.js";
import {
  decodeAcquiredCppCuteBrowserDiagnosticNormalizationAsset,
  decodeAcquiredCppCuteBrowserRuntimeAbiAsset,
  decodeAcquiredCppCuteBrowserSemanticAdapterAsset,
  installCppCuteBrowserVfs,
  verifyTransferredCppCuteBrowserAssetSet,
  type CppCuteBrowserTransferredAssetInput,
} from "../src/cpp_cute_browser_asset_installation.js";
import {
  decodeCppCuteBrowserAssetManifest,
} from "../src/cpp_cute_browser_assets.js";
import {
  authorizeCppCuteBrowserViewCopyArtifact,
  unwrapAuthorizedCppCuteBrowserViewCopyArtifact,
} from "../src/cpp_cute_browser_view_copy_authorization.js";
import {
  prepareObservedCppCuteBrowserViewCopyCandidate,
  unwrapObservedCppCuteBrowserViewCopyCandidate,
} from "../src/cpp_cute_browser_view_copy_candidate.js";
import {
  decodeCppCuteBrowserBuildInputLock,
} from "../src/cpp_cute_browser_build_lock.js";
import {
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../src/cpp_cute_browser_build_provenance.js";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
} from "../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  verifyCppCuteBrowserBuildProducer,
} from "../src/cpp_cute_browser_producer_trust.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
} from "../src/cpp_cute_browser_producer_trust_policy.js";
import {
  cppCuteBrowserRealCompileCase,
  type CppCuteBrowserRealCompileCase,
  type CppCuteBrowserRealCompileCaseId,
} from "../src/cpp_cute_browser_real_compile_cases.js";
import {
  decodeCppCuteFrontendProfile,
} from "../src/cpp_cute_frontend_profile.js";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
} from "../src/cpp_cute_frontend_artifact.js";
import {
  prepareCppCuteAttestationTrustStore,
} from "../src/cpp_cute_frontend_provenance.js";
import {
  lowerAuthorizedCppCuteViewCopyEntry,
} from "../src/cpp_cute_view_copy_lowering.js";
import {
  executeCppCuteBrowserWorker,
  unwrapObservedCppCuteBrowserWorkerExecution,
} from "../src/cpp_cute_browser_worker_controller.js";
import {
  unwrapValidatedCppCuteBrowserWorkerResultFrame,
} from "../src/cpp_cute_browser_worker_protocol.js";
import {
  verifyCppCuteBrowserWorkerBundle,
} from "../src/cpp_cute_browser_worker_bundle.js";
import {
  inspectObservedCppCuteBrowserPackageWasmConformance,
} from "../src/cpp_cute_browser_wasm_verifier_controller.js";
import {
  cppCuteBrowserRealCompileStaticLayoutProjection,
  prepareCppCuteBrowserRealCompileViewCopyRequest,
} from "./support/cpp_cute_browser_real_compile_request.js";

interface BrowserServedInput {
  readonly route: string;
  readonly mediaType: string;
  readonly sha256: string;
  readonly byteLength: number;
}

interface BrowserAssetInput extends BrowserServedInput {
  readonly assetId: string;
}

interface BrowserInputs {
  readonly schema:
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-inputs";
  readonly version: 1;
  readonly authority: "host-preflight-exact-private-distribution-only";
  readonly caseId: CppCuteBrowserRealCompileCaseId;
  readonly sourceRevision: string;
  readonly controls: Readonly<{
    readonly profile: BrowserServedInput;
    readonly assetManifest: BrowserServedInput;
    readonly buildInputLock: BrowserServedInput;
    readonly envelope: BrowserServedInput;
    readonly producerPolicy: BrowserServedInput;
    readonly producerTrustStore: BrowserServedInput;
  }>;
  readonly assets: readonly BrowserAssetInput[];
  readonly distribution: Readonly<{
    readonly reproducibilityId: string;
    readonly resourceSha256: string;
    readonly buildInputLockId: string;
    readonly buildInputLockResourceSha256: string;
    readonly profileHash: string;
    readonly profileSha256: string;
    readonly profileByteLength: string;
    readonly assetManifestId: string;
    readonly assetManifestSha256: string;
    readonly assetSetSha256: string;
    readonly buildSubjectId: string;
    readonly buildSubjectSha256: string;
    readonly workerBundleSha256: string;
    readonly exactRootVerificationId: string;
    readonly exactOutputCount: number;
    readonly exactOutputByteLength: string;
  }>;
  readonly producer: Readonly<{
    readonly producerEvidenceId: string;
    readonly policyId: string;
    readonly policySha256: string;
    readonly builderId: string;
    readonly keyId: string;
    readonly trustStoreSha256: string;
    readonly statementSha256: string;
    readonly signatureEvidenceSha256: string;
  }>;
  readonly claims: Readonly<Record<string, boolean>>;
}

interface FetchedInput extends BrowserServedInput {
  readonly bytes: Uint8Array;
}

interface FetchedAsset extends BrowserAssetInput {
  readonly bytes: Uint8Array;
}

interface RuntimeStorage {
  readonly sourceWords: Uint32Array;
  readonly initialDestinationWords: Uint32Array;
  readonly expectedDestinationWords: Uint32Array;
  readonly sourceAllocationByteLength: string;
  readonly destinationAllocationByteLength: string;
  readonly sourceByteOffset: "4";
  readonly destinationByteOffset: "4";
}

declare const
  __BG_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_INPUTS__: BrowserInputs;

const INPUTS =
  __BG_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_INPUTS__;
const COMPILE_CASE = cppCuteBrowserRealCompileCase(INPUTS.caseId);
const EVIDENCE_MARKER =
  "BROWSERGRAD_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_EVIDENCE=";
const LOCAL_ENGINEERING_BUILDER_ID =
  "https://builders.browsergrad.dev/local-engineering-reproducibility";
const WEBGPU_TIMEOUT_MS = 15_000;
const TRUST_STORE_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: 256 * 1024,
  maxDepth: 8,
  maxNodes: 2_048,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 256,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 32,
  maxArithmeticOperations: 4_096,
});

class ConvergenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "ConvergenceLaneError";
  }
}

it("converges one exact browser-produced candidate on CPU and required WebGPU", async () => {
  const started = performance.now();
  const sourceBytes = new TextEncoder().encode(COMPILE_CASE.source);
  expect(await sha256Hex(sourceBytes)).toBe(COMPILE_CASE.sourceSha256);
  expect(INPUTS.claims).toMatchObject({
    exactPrivateDistributionTreeVerified: true,
    packagePinnedFullDistributionReproducibilityMatched: true,
    localEngineeringProducerAuthenticated: true,
    externalProducerTrusted: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
    backendExecutionObserved: false,
    releaseReady: false,
  });

  const [controls, fetchedAssets] = await Promise.all([
    fetchControls(),
    fetchAssets(),
  ]);
  const profile = await decodeCppCuteFrontendProfile(controls.profile.bytes);
  const assetManifest = await decodeCppCuteBrowserAssetManifest(
    controls.assetManifest.bytes,
    profile,
  );
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    controls.buildInputLock.bytes,
  );
  expect(profile).toMatchObject({
    profileHash: INPUTS.distribution.profileHash,
    compilationContractHash: expect.stringMatching(/^[0-9a-f]{64}$/u),
    deploymentMode: "browser-local",
  });
  expect(assetManifest).toMatchObject({
    manifestId: INPUTS.distribution.assetManifestId,
    manifestSha256: INPUTS.distribution.assetManifestSha256,
    assetSetSha256: INPUTS.distribution.assetSetSha256,
  });
  expect(buildInputLock).toMatchObject({
    lockId: INPUTS.distribution.buildInputLockId,
    resourceSha256:
      INPUTS.distribution.buildInputLockResourceSha256,
  });

  const workerBundle = await verifyCppCuteBrowserWorkerBundle();
  const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(
    controls.producerPolicy.bytes,
  );
  const trustStore = await prepareCppCuteAttestationTrustStore(
    decodeCanonicalJson(
      controls.producerTrustStore.bytes,
      TRUST_STORE_DECODE_LIMITS,
    ),
    { limits: TRUST_STORE_DECODE_LIMITS },
  );
  const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
    decodeCanonicalJson(
      controls.envelope.bytes,
      CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
    ),
    {
      assetManifest,
      buildInputLock,
      workerBundle,
      trustStore,
    },
  );
  const producer = await verifyCppCuteBrowserBuildProducer(
    signatureBinding,
    trustPolicy,
  );
  expect(producer).toMatchObject({
    authority: "independently-admitted-browser-build-producer",
    producerEvidenceId: INPUTS.producer.producerEvidenceId,
    policyId: INPUTS.producer.policyId,
    policySha256: INPUTS.producer.policySha256,
    builderId: LOCAL_ENGINEERING_BUILDER_ID,
    keyId: INPUTS.producer.keyId,
    trustStoreSha256: INPUTS.producer.trustStoreSha256,
    buildSubjectId: INPUTS.distribution.buildSubjectId,
    profileHash: INPUTS.distribution.profileHash,
    manifestId: INPUTS.distribution.assetManifestId,
    signatureVerified: true,
    independentTrustPolicyMatched: true,
    producerTrusted: true,
    exactAssetBytesVerified: false,
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    releaseReady: false,
  });

  const transferred: CppCuteBrowserTransferredAssetInput[] =
    fetchedAssets.map((asset) => ({
      assetId: asset.assetId,
      bytes: new Uint8Array(asset.bytes),
    }));
  const assetSet = await verifyTransferredCppCuteBrowserAssetSet(
    assetManifest,
    transferred,
  );
  for (const entry of transferred) entry.bytes.fill(0);
  for (const asset of fetchedAssets) asset.bytes.fill(0);
  const [
    runtimeAbiAsset,
    semanticAdapterAsset,
    diagnosticNormalizationAsset,
  ] = await Promise.all([
    decodeAcquiredCppCuteBrowserRuntimeAbiAsset(assetSet),
    decodeAcquiredCppCuteBrowserSemanticAdapterAsset(assetSet),
    decodeAcquiredCppCuteBrowserDiagnosticNormalizationAsset(assetSet),
  ]);
  expect(semanticAdapterAsset.designAuthority).toBe(true);
  expect(diagnosticNormalizationAsset.designAuthority).toBe(true);
  const vfsInstallation = await installCppCuteBrowserVfs(assetSet);
  expect(vfsInstallation).toMatchObject({
    packCount: 5,
    fileCount: 5_788,
  });

  const request = await prepareCppCuteBrowserRealCompileViewCopyRequest(
    profile,
    COMPILE_CASE,
  );
  const compileStarted = performance.now();
  const execution = await executeCppCuteBrowserWorker({
    profile,
    assetManifest,
    vfsInstallation,
    request,
    runtimeAbiAsset,
  });
  const compileElapsedMilliseconds = performance.now() - compileStarted;
  const executionRecord =
    unwrapObservedCppCuteBrowserWorkerExecution(execution);
  const frame = unwrapValidatedCppCuteBrowserWorkerResultFrame(
    executionRecord.validatedResultFrame,
  );
  const artifact = unwrapVerifiedCppCuteFrontendArtifact(frame.artifact);
  if (artifact.envelope.payload.outcome.kind !== "accepted" ||
      artifact.envelope.payload.outcome.selectedEntryIds.length !== 1 ||
      artifact.envelope.payload.outcome.selectedEntryIds[0] === undefined) {
    throw new ConvergenceLaneError(
      "BG-COMPILER-CPP-CUTE-EXACT-DISTRIBUTION-COMPILE",
      "exact browser Worker did not produce one accepted view-copy entry",
    );
  }
  const entryId =
    artifact.envelope.payload.outcome.selectedEntryIds[0];
  const candidate = await prepareObservedCppCuteBrowserViewCopyCandidate(
    execution,
    { entryId },
  );
  const candidateRecord =
    unwrapObservedCppCuteBrowserViewCopyCandidate(candidate);
  expect(candidate).toMatchObject({
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    sharedViewCopySemanticsPrepared: true,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  });
  expect(candidateRecord.semantics.dtype).toBe(COMPILE_CASE.dtype);
  expect(candidateRecord.semantics.sourceSpanElements)
    .toBe(COMPILE_CASE.sourceSpanElements);
  expect(candidateRecord.semantics.destinationSpanElements)
    .toBe(COMPILE_CASE.destinationSpanElements);
  expect(cppCuteBrowserRealCompileStaticLayoutProjection(
    candidateRecord.semantics.sourceLayoutFact,
  )).toEqual(COMPILE_CASE.sourceLayout);
  expect(cppCuteBrowserRealCompileStaticLayoutProjection(
    candidateRecord.semantics.destinationLayoutFact,
  )).toEqual(COMPILE_CASE.destinationLayout);
  const wasmConformance =
    inspectObservedCppCuteBrowserPackageWasmConformance(
      executionRecord.packageInvocationLineage.observedWasmConformance,
    );
  expect(wasmConformance).toMatchObject({
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    verifierWorkerExecutionObserved: true,
    releaseReady: false,
  });

  const authorization = await authorizeCppCuteBrowserViewCopyArtifact(
    candidate,
    producer,
  );
  const authorizationRecord =
    unwrapAuthorizedCppCuteBrowserViewCopyArtifact(authorization);
  expect(authorization).toMatchObject({
    authority: "browser-worker-view-copy-local-semantic-authorization",
    workerExecutionObserved: true,
    producerTrusted: true,
    localSemanticLoweringAuthorized: true,
    backendExecutionAuthorized: false,
    distributionAuthorized: false,
    releaseReady: false,
  });
  const storage = createRuntimeStorage(COMPILE_CASE);
  const artifacts = await lowerAuthorizedCppCuteViewCopyEntry(
    authorizationRecord.authorization,
    {
      entryId,
      sourceAllocationByteLength:
        encodeWireU64(BigInt(storage.sourceAllocationByteLength)),
      destinationAllocationByteLength:
        encodeWireU64(BigInt(storage.destinationAllocationByteLength)),
      sourceByteOffset: encodeWireU64(BigInt(storage.sourceByteOffset)),
      destinationByteOffset:
        encodeWireU64(BigInt(storage.destinationByteOffset)),
    },
  );

  const cpuSourceWords = new Uint32Array(storage.sourceWords);
  const cpuDestinationWords =
    new Uint32Array(storage.initialDestinationWords);
  const cpu = await prepareViewCopyCpu(
    artifacts.layout,
    artifacts.kernel,
    { operationId: artifacts.operationId },
  );
  const cpuTrace = cpu.execute({
    source: bytes(cpuSourceWords),
    destination: bytes(cpuDestinationWords),
  });
  expect([...cpuSourceWords]).toEqual([...storage.sourceWords]);
  expect([...cpuDestinationWords])
    .toEqual([...storage.expectedDestinationWords]);
  expect(cpuTrace).toMatchObject({
    elementCount: logicalElementCount(COMPILE_CASE).toString(10),
    readElements: logicalElementCount(COMPILE_CASE).toString(10),
    filledElements: "0",
  });

  const acquisition = await withWebGpuEvidenceTimeout(
    acquireWebGpuEvidenceDevice(),
    WEBGPU_TIMEOUT_MS,
    "WebGPU device acquisition",
    timeoutError,
  );
  if (acquisition.kind === "unavailable") {
    throw requiredEvidenceFailure(acquisition.reason);
  }
  const {
    adapter,
    device,
    adapterInfo,
  } = acquisition.value;
  let kernelDevice: KernelDevice | undefined;
  const uncapturedErrors: string[] = [];
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => {
    uncapturedErrors.push(event.error.message);
  };
  device.addEventListener("uncapturederror", uncapturedHandler);
  const deviceLoss = device.lost;
  try {
    const environment = createWebGpuExecutionEnvironmentRecord({
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      adapter: adapterInfo,
      adapterSupportedFeatures:
        Object.freeze([...adapter.features].map(String).sort()),
      negotiatedDeviceFeatures:
        Object.freeze([...device.features].map(String).sort()),
      negotiatedDeviceLimits: webGpuSemanticDeviceLimits(device),
    });
    const deviceProfileHash = await hashNamedComponents({
      environment,
    });
    kernelDevice = await raceDeviceLoss(
      withWebGpuEvidenceTimeout(
        createDevice({ device }),
        WEBGPU_TIMEOUT_MS,
        "kernel device construction",
        timeoutError,
      ),
      deviceLoss,
    );
    const prepared = await raceDeviceLoss(
      withWebGpuEvidenceTimeout(
        prepareSemanticViewCopyWgsl(
          artifacts.layout,
          artifacts.kernel,
          { operationId: artifacts.operationId },
        ),
        WEBGPU_TIMEOUT_MS,
        "WebGPU lowering",
        timeoutError,
      ),
      deviceLoss,
    );
    const preparedBackendArtifactHash = await hashNamedComponents({
      backendProfile: prepared.backendProfile,
      backendVersion: prepared.backendVersion,
      semanticSpecializationHash: prepared.semantic.specializationHash,
      wgslModuleHash: prepared.wgslModuleHash,
      programName: prepared.program.name,
      programWgsl: prepared.program.wgsl,
      workgroupSize: prepared.program.workgroupSize,
      dispatchCount: prepared.launch.dispatchCount,
    });
    const gpuSourceWords = new Uint32Array(storage.sourceWords);
    const result = await raceDeviceLoss(
      withWebGpuEvidenceTimeout(
        runSemanticViewCopyWebGpu(kernelDevice, prepared, {
          sourceWords: gpuSourceWords,
          destinationWords:
            new Uint32Array(storage.initialDestinationWords),
        }),
        WEBGPU_TIMEOUT_MS,
        "WebGPU execution",
        timeoutError,
      ),
      deviceLoss,
    );
    await raceDeviceLoss(
      withWebGpuEvidenceTimeout(
        device.queue.onSubmittedWorkDone(),
        WEBGPU_TIMEOUT_MS,
        "WebGPU queue drain",
        timeoutError,
      ),
      deviceLoss,
    );
    await raceDeviceLoss(
      withWebGpuEvidenceTimeout(
        nextWebGpuEvidenceTask(),
        1_000,
        "late WebGPU error yield",
        timeoutError,
      ),
      deviceLoss,
    );
    if (uncapturedErrors.length > 0) {
      throw new ConvergenceLaneError(
        "BG-COMPILER-CPP-CUTE-EXACT-DISTRIBUTION-WEBGPU-ERROR",
        uncapturedErrors.join("; "),
      );
    }
    expect(result.trace.submitted).toBe(true);
    expect([...gpuSourceWords]).toEqual([...storage.sourceWords]);
    expect([...result.destinationWords])
      .toEqual([...storage.expectedDestinationWords]);
    expect([...result.destinationWords]).toEqual([...cpuDestinationWords]);
    expect(result.destinationWords[0])
      .toBe(storage.initialDestinationWords[0]);
    expect(result.destinationWords.at(-1))
      .toBe(storage.initialDestinationWords.at(-1));
    expect(result.trace.semanticSpecializationHash)
      .toBe(cpu.specializationHash);

    const [
      inputHash,
      expectedDestinationHash,
      cpuDestinationHash,
      webGpuDestinationHash,
    ] = await Promise.all([
      hashWordsPair(
        storage.sourceWords,
        storage.initialDestinationWords,
      ),
      hashWords(storage.expectedDestinationWords),
      hashWords(cpuDestinationWords),
      hashWords(result.destinationWords),
    ]);
    expect(cpuDestinationHash).toBe(expectedDestinationHash);
    expect(webGpuDestinationHash).toBe(expectedDestinationHash);
    const evidenceId =
      `bg.cpp.browser-exact-distribution-case-convergence.sha256.${
        await hashNamedComponents({
          caseId: COMPILE_CASE.caseId,
          sourceSha256: COMPILE_CASE.sourceSha256,
          distributionReproducibilityId:
            INPUTS.distribution.reproducibilityId,
          buildSubjectId: INPUTS.distribution.buildSubjectId,
          producerEvidenceId: producer.producerEvidenceId,
          executionEvidenceId: execution.evidenceId,
          candidateId: candidate.candidateId,
          authorizationId: authorization.authorizationId,
          layoutSemanticHash: artifacts.layoutSemanticHash,
          kernelSemanticHash: artifacts.kernelSemanticHash,
          inputHash,
          expectedDestinationHash,
          deviceProfileHash,
          preparedBackendArtifactHash,
        })
      }`;
    const evidence = Object.freeze({
      schema:
        "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-observation",
      version: 1,
      authority:
        "local-engineering-exact-payload-cpu-webgpu-observation-only",
      evidenceId,
      caseId: COMPILE_CASE.caseId,
      sourceRevision: INPUTS.sourceRevision,
      recordedAt: new Date().toISOString(),
      source: Object.freeze({
        virtualPath: COMPILE_CASE.virtualPath,
        sourceSha256: COMPILE_CASE.sourceSha256,
        syntax: "unchanged-cpp17-cute",
        selectedDeclaration: "copy_views",
        dtype: COMPILE_CASE.dtype,
        coordinateRank: COMPILE_CASE.coordinateRank,
        sourceLayout: COMPILE_CASE.sourceLayout,
        destinationLayout: COMPILE_CASE.destinationLayout,
      }),
      distribution: INPUTS.distribution,
      producer: INPUTS.producer,
      execution: Object.freeze({
        executionEvidenceId: execution.evidenceId,
        invocationId: execution.invocationId,
        requestId: request.requestId,
        artifactId: frame.artifact.artifactId,
        candidateId: candidate.candidateId,
        authorizationId: authorization.authorizationId,
        entryId,
        entrySubjectHash: candidate.entrySubjectHash,
        layoutSemanticHash: artifacts.layoutSemanticHash,
        kernelSemanticHash: artifacts.kernelSemanticHash,
        operationId: artifacts.operationId,
        cpuSemanticSpecializationHash: cpu.specializationHash,
        gpuSemanticSpecializationHash:
          result.trace.semanticSpecializationHash,
        backendSpecializationHash:
          result.trace.backendSpecializationHash,
        preparedBackendArtifactHash,
        wgslModuleHash: prepared.wgslModuleHash,
        inputHash,
        expectedDestinationHash,
        cpuDestinationHash,
        webGpuDestinationHash,
        browserWorkerCompiled: true,
        localSemanticAuthorizationMinted: true,
        cpuReferenceExecuted: true,
        actualWebGpuExecuted: true,
        completeDestinationBitComparisonPassed: true,
        nonzeroOffsetCanariesPreserved: true,
      }),
      webgpu: Object.freeze({
        required: true,
        actualExecutionObserved: true,
        environment,
        deviceProfileHash,
        backendProfile: prepared.backendProfile,
        backendVersion: prepared.backendVersion,
        uncapturedErrors: Object.freeze([...uncapturedErrors]),
      }),
      timings: Object.freeze({
        compileElapsedMilliseconds:
          Math.round(compileElapsedMilliseconds),
        totalElapsedMilliseconds:
          Math.round(performance.now() - started),
      }),
      claims: Object.freeze({
        exactPrivateDistributionTreeVerified: true,
        packagePinnedFullDistributionReproducibilityMatched: true,
        localEngineeringProducerAuthenticated: true,
        browserWorkerCompilationObserved: true,
        exactCandidateAuthorizedThroughSharedSeam: true,
        cpuReferenceConvergenceObserved: true,
        requiredRealWebGpuConvergenceObserved: true,
        externalProducerTrusted: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        backendExecutionAuthorityMinted: false,
        releaseReady: false,
      }),
    });
    console.warn(`${EVIDENCE_MARKER}${JSON.stringify(evidence)}`);
  } finally {
    device.removeEventListener("uncapturederror", uncapturedHandler);
    kernelDevice?.clearCache();
    device.destroy();
  }
}, 300_000);

async function fetchControls(): Promise<Readonly<{
  profile: FetchedInput;
  assetManifest: FetchedInput;
  buildInputLock: FetchedInput;
  envelope: FetchedInput;
  producerPolicy: FetchedInput;
  producerTrustStore: FetchedInput;
}>> {
  const entries = await Promise.all(
    Object.entries(INPUTS.controls).map(async ([name, input]) =>
      [name, await fetchInput(input)] as const),
  );
  return Object.freeze(Object.fromEntries(entries)) as Readonly<{
    profile: FetchedInput;
    assetManifest: FetchedInput;
    buildInputLock: FetchedInput;
    envelope: FetchedInput;
    producerPolicy: FetchedInput;
    producerTrustStore: FetchedInput;
  }>;
}

async function fetchAssets(): Promise<readonly FetchedAsset[]> {
  const assets: FetchedAsset[] = [];
  for (const asset of INPUTS.assets) {
    assets.push(Object.freeze({
      ...asset,
      bytes: (await fetchInput(asset)).bytes,
    }));
  }
  return Object.freeze(assets);
}

async function fetchInput(
  input: BrowserServedInput,
): Promise<FetchedInput> {
  const response = await fetch(input.route, {
    method: "GET",
    credentials: "same-origin",
    redirect: "error",
    cache: "no-store",
  });
  if (!response.ok || response.redirected ||
      response.headers.get("content-length") !==
        String(input.byteLength) ||
      response.headers.get("x-content-type-options") !== "nosniff") {
    throw new ConvergenceLaneError(
      "BG-COMPILER-CPP-CUTE-EXACT-DISTRIBUTION-ACQUISITION",
      `${input.route} failed exact same-origin acquisition`,
    );
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== input.byteLength ||
      await sha256Hex(bytes) !== input.sha256) {
    throw new ConvergenceLaneError(
      "BG-COMPILER-CPP-CUTE-EXACT-DISTRIBUTION-ACQUISITION",
      `${input.route} changed after host preflight`,
    );
  }
  return Object.freeze({ ...input, bytes });
}

function createRuntimeStorage(
  compileCase: CppCuteBrowserRealCompileCase,
): RuntimeStorage {
  const sourceSpan = Number(compileCase.sourceSpanElements);
  const destinationSpan = Number(compileCase.destinationSpanElements);
  if (!Number.isSafeInteger(sourceSpan) ||
      !Number.isSafeInteger(destinationSpan) ||
      sourceSpan <= 0 || destinationSpan <= 0) {
    throw new Error("compile case storage spans exceed browser evidence bounds");
  }
  const sourceWords = new Uint32Array(sourceSpan + 2);
  sourceWords[0] = 0xa1b2c3d4;
  sourceWords[sourceWords.length - 1] = 0xd4c3b2a1;
  for (let index = 0; index < sourceSpan; index += 1) {
    sourceWords[index + 1] = compileCase.dtype === "f32"
      ? (0x3f000000 + index * 0x00040000) >>> 0
      : (0x80000000 ^ ((index + 1) * 0x010203)) >>> 0;
  }
  const initialDestinationWords = new Uint32Array(destinationSpan + 2);
  initialDestinationWords.fill(0xdeadbeef);
  initialDestinationWords[0] = 0x13579bdf;
  initialDestinationWords[initialDestinationWords.length - 1] =
    0x2468ace0;
  const expectedDestinationWords =
    new Uint32Array(initialDestinationWords);
  applyIndependentViewCopy(
    compileCase,
    sourceWords,
    expectedDestinationWords,
  );
  return Object.freeze({
    sourceWords,
    initialDestinationWords,
    expectedDestinationWords,
    sourceAllocationByteLength: String(sourceWords.byteLength),
    destinationAllocationByteLength:
      String(initialDestinationWords.byteLength),
    sourceByteOffset: "4",
    destinationByteOffset: "4",
  });
}

function applyIndependentViewCopy(
  compileCase: CppCuteBrowserRealCompileCase,
  sourceWords: Uint32Array,
  destinationWords: Uint32Array,
): void {
  const extents = compileCase.sourceLayout.shape.map(Number);
  const sourceStrides = compileCase.sourceLayout.strides.map(Number);
  const destinationStrides =
    compileCase.destinationLayout.strides.map(Number);
  const count = logicalElementCount(compileCase);
  for (let ordinal = 0; ordinal < count; ordinal += 1) {
    let remaining = ordinal;
    let sourceIndex = 0;
    let destinationIndex = 0;
    for (let axis = extents.length - 1; axis >= 0; axis -= 1) {
      const extent = extents[axis]!;
      const coordinate = remaining % extent;
      remaining = Math.floor(remaining / extent);
      sourceIndex += coordinate * sourceStrides[axis]!;
      destinationIndex += coordinate * destinationStrides[axis]!;
    }
    const source = sourceWords[sourceIndex + 1];
    if (source === undefined ||
        destinationWords[destinationIndex + 1] === undefined) {
      throw new Error("independent view-copy geometry exceeded storage");
    }
    destinationWords[destinationIndex + 1] = source;
  }
}

function logicalElementCount(
  compileCase: CppCuteBrowserRealCompileCase,
): number {
  return compileCase.sourceLayout.shape
    .map(Number)
    .reduce((product, extent) => product * extent, 1);
}

function decodeCanonicalJson(
  bytes_: Uint8Array,
  limits: DecodeLimits,
): JsonValue {
  const value = decodeWireJson(bytes_, { limits });
  const canonical = canonicalJsonBytes(value, { limits });
  if (!equalBytes(bytes_, canonical)) {
    throw new Error("control-plane JSON bytes are not canonical");
  }
  return value;
}

async function hashWords(words: Uint32Array): Promise<string> {
  return hashNamedComponents({
    byteLength: words.byteLength,
    words: [...words],
  });
}

async function hashWordsPair(
  source: Uint32Array,
  destination: Uint32Array,
): Promise<string> {
  return hashNamedComponents({
    source: { byteLength: source.byteLength, words: [...source] },
    destination: {
      byteLength: destination.byteLength,
      words: [...destination],
    },
  });
}

function bytes(words: Uint32Array): Uint8Array {
  return new Uint8Array(
    words.buffer,
    words.byteOffset,
    words.byteLength,
  );
}

async function raceDeviceLoss<T>(
  promise: Promise<T>,
  loss: Promise<GPUDeviceLostInfo>,
): Promise<T> {
  const result = await Promise.race([
    promise.then((value) => ({ kind: "value" as const, value })),
    loss.then((info) => ({ kind: "lost" as const, info })),
  ]);
  if (result.kind === "lost") {
    throw new ConvergenceLaneError(
      "BG-COMPILER-CPP-CUTE-EXACT-DISTRIBUTION-WEBGPU-DEVICE-LOST",
      `${result.info.reason}: ${result.info.message}`,
    );
  }
  return result.value;
}

function timeoutError(message: string): Error {
  return new ConvergenceLaneError(
    "BG-COMPILER-CPP-CUTE-EXACT-DISTRIBUTION-WEBGPU-TIMEOUT",
    message,
  );
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}
