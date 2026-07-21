import { expect, it } from "vitest";
import { layoutArtifactPayload } from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  createVerifiedViewCopyArtifacts,
  kernelArtifactPayload,
  prepareViewCopyCpu,
  type CreateVerifiedViewCopyArtifactsRequest,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  hashNamedComponents,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  createDevice,
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
  type KernelDevice,
} from "@unlocalhosted/browsergrad-kernels";
import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createTerminalEvidenceEmitter,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
} from "../../../test-support/webgpu-evidence";
import {
  DEVICE_UNAVAILABLE_DIAGNOSTIC,
  UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
  deviceProfileHashFor,
  environmentIdFor,
  type EvidenceEnvironment,
} from "./semantic_view_copy_bindings_evidence";
import {
  CPP_CUTE_VIEW_COPY_WEBGPU_BACKEND_ID as BACKEND_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_CAPABILITY_ID as CAPABILITY_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_CASE_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_COMPARISON_POLICY_ID as COMPARISON_POLICY_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_SUITE_ID as SUITE_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_TERMINAL_EXPECTATION,
  finalizeCppCuteViewCopyWebGpuEvidence,
  type CppCuteViewCopyWebGpuCaseEvidence,
  type CppCuteViewCopyWebGpuTerminalEvidence,
} from "./cpp_cute_view_copy_webgpu_evidence";
import { CPP_CUTE_BROWSER_VIEW_COPY_CONVERGENCE_FIXTURE as convergence } from
  "../tests/fixtures/cpp_cute_browser_view_copy_convergence";

declare const __BG_COMPILER_VERSION__: string;
declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;
declare const __BG_SOURCE_REVISION__: string;

const EVIDENCE_PREFIX = "[browsergrad-cpp-cute-view-copy-fixture-webgpu-evidence]";
const EXECUTION_FAILURE_DIAGNOSTIC =
  "BG-COMPILER-CPP-CUTE-VIEW-COPY-FIXTURE-WEBGPU-EXECUTION";
const DEVICE_LOST_DIAGNOSTIC =
  "BG-COMPILER-CPP-CUTE-VIEW-COPY-WEBGPU-DEVICE-LOST";
const EVIDENCE_TIMEOUT_DIAGNOSTIC =
  "BG-COMPILER-CPP-CUTE-VIEW-COPY-WEBGPU-TIMEOUT";
const EVIDENCE_TIMEOUT_MS = 10_000;
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-compiler": __BG_COMPILER_VERSION__,
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
}) as JsonObject;
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(
  EVIDENCE_PREFIX,
  CPP_CUTE_VIEW_COPY_WEBGPU_TERMINAL_EXPECTATION,
);

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}

it("converges the exact canonical CuTe view-copy fixture payload on required actual WebGPU", async (context) => {
  const required = requiresWebGpuEvidence();
  const sourceWords = Uint32Array.from(convergence.expected.sourceWords);
  const initialDestinationWords = Uint32Array.from(
    convergence.expected.initialDestinationWords,
  );
  const [fixtureArtifactHash, inputHash, expectedDestinationHash] = await Promise.all([
    fixtureArtifactHashFor(),
    hashWordsPair(sourceWords, initialDestinationWords),
    hashWords(Uint32Array.from(convergence.expected.destinationWords)),
  ]);
  let environment = freezeEnvironment({ acquisition: "not-attempted" });
  let environmentId = await environmentIdFor(environment);
  let deviceProfileHash: string | undefined;
  let terminalEmitted = false;
  let actualWebGpuExecution = false;
  let device: GPUDevice | undefined;
  let kernelDevice: KernelDevice | undefined;
  let preparedBackendArtifactHash: string | undefined;
  let cpuSemanticSpecializationHash: string | undefined;
  let caseEvidence: CppCuteViewCopyWebGpuCaseEvidence | undefined;
  let deviceLostBeforeTerminal: GPUDeviceLostInfo | undefined;
  let deviceLoss: Promise<GPUDeviceLostInfo> | undefined;
  const uncapturedErrors: string[] = [];
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => {
    uncapturedErrors.push(event.error.message);
  };

  try {
    const artifacts = await createVerifiedViewCopyArtifacts(
      convergence.construction as unknown as CreateVerifiedViewCopyArtifactsRequest,
      convergence.constructionOptions,
    );
    expect(artifacts.layoutSemanticHash).toBe(convergence.expected.layoutSemanticHash);
    expect(artifacts.kernelSemanticHash).toBe(convergence.expected.kernelSemanticHash);
    expect(artifacts.source).toEqual(convergence.expected.source);
    expect(artifacts.destination).toEqual(convergence.expected.destination);
    expect(artifacts.operationId).toBe(convergence.expected.operationId);
    expect(layoutArtifactPayload(artifacts.layout)).toEqual(convergence.expected.layoutPayload);
    expect(kernelArtifactPayload(artifacts.kernel)).toEqual(convergence.expected.kernelPayload);

    const cpuDestinationWords = new Uint32Array(initialDestinationWords);
    const cpu = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    cpuSemanticSpecializationHash = cpu.specializationHash;
    const cpuTrace = cpu.execute({
      source: bytes(sourceWords),
      destination: bytes(cpuDestinationWords),
    });
    expect([...sourceWords]).toEqual(convergence.expected.sourceWords);
    expect([...cpuDestinationWords]).toEqual(convergence.expected.destinationWords);
    expect(cpuTrace).toMatchObject(convergence.expected.cpuTrace);

    const acquisition = await withEvidenceTimeout(
      acquireWebGpuEvidenceDevice(),
      EVIDENCE_TIMEOUT_MS,
      "device-acquisition",
    );
    if (acquisition.kind === "unavailable") {
      environment = freezeEnvironment({
        acquisition: "navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: acquisition.reason,
      });
      environmentId = await environmentIdFor(environment);
      emitTerminal({
        required,
        fixtureArtifactHash,
        inputHash,
        expectedDestinationHash,
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: [DEVICE_UNAVAILABLE_DIAGNOSTIC],
        actualWebGpuExecution,
        uncapturedErrors,
        error: errorRecord(
          new Error(acquisition.reason),
          DEVICE_UNAVAILABLE_DIAGNOSTIC,
        ),
      });
      terminalEmitted = true;
      if (required) throw requiredEvidenceFailure(acquisition.reason);
      context.skip(acquisition.reason);
      return;
    }

    const { adapter, device: acquiredDevice, adapterInfo } = acquisition.value;
    device = acquiredDevice;
    device.addEventListener("uncapturederror", uncapturedHandler);
    deviceLoss = device.lost.then((info) => {
      deviceLostBeforeTerminal = info;
      return info;
    });
    environment = freezeEnvironment({
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      adapter: adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures: Object.freeze([...adapter.features].map(String).sort()),
      negotiatedDeviceFeatures: Object.freeze([...device.features].map(String).sort()),
      negotiatedDeviceLimits: deviceLimits(device),
    });
    [environmentId, deviceProfileHash] = await Promise.all([
      environmentIdFor(environment),
      deviceProfileHashFor(environment),
    ]);

    kernelDevice = await raceDeviceLoss(
      withEvidenceTimeout(
        createDevice({ device }),
        EVIDENCE_TIMEOUT_MS,
        "kernel-device-construction",
      ),
      deviceLoss,
    );
    const prepared = await raceDeviceLoss(
      withEvidenceTimeout(
        prepareSemanticViewCopyWgsl(
          artifacts.layout,
          artifacts.kernel,
          { operationId: artifacts.operationId },
        ),
        EVIDENCE_TIMEOUT_MS,
        "semantic-view-copy-preparation",
      ),
      deviceLoss,
    );
    preparedBackendArtifactHash = await hashNamedComponents({
      backendProfile: prepared.backendProfile,
      backendVersion: prepared.backendVersion,
      semanticSpecializationHash: prepared.semantic.specializationHash,
      wgslModuleHash: prepared.wgslModuleHash,
      programName: prepared.program.name,
      programWgsl: prepared.program.wgsl,
      workgroupSize: prepared.program.workgroupSize,
      dispatchCount: prepared.launch.dispatchCount,
    });
    const gpuSourceWords = new Uint32Array(sourceWords);
    const result = await raceDeviceLoss(
      withEvidenceTimeout(
        runSemanticViewCopyWebGpu(kernelDevice, prepared, {
          sourceWords: gpuSourceWords,
          destinationWords: new Uint32Array(initialDestinationWords),
        }),
        EVIDENCE_TIMEOUT_MS,
        "semantic-view-copy-execution",
      ),
      deviceLoss,
    );
    actualWebGpuExecution = result.trace.submitted;
    await raceDeviceLoss(
      withEvidenceTimeout(
        device.queue.onSubmittedWorkDone(),
        EVIDENCE_TIMEOUT_MS,
        "execution-queue-drain",
      ),
      deviceLoss,
    );

    expect(actualWebGpuExecution).toBe(true);
    expect(result.trace.layoutSemanticHash).toBe(convergence.expected.layoutSemanticHash);
    expect(result.trace.kernelSemanticHash).toBe(convergence.expected.kernelSemanticHash);
    expect(result.trace.semanticSpecializationHash).toBe(cpuSemanticSpecializationHash);
    expect(result.trace.logicalShape).toEqual(convergence.expected.cpuTrace.logicalShape);
    expect([...gpuSourceWords]).toEqual(convergence.expected.sourceWords);
    expect([...result.destinationWords]).toEqual(convergence.expected.destinationWords);
    expect([...result.destinationWords]).toEqual([...cpuDestinationWords]);
    expect(convergence.storage.sourceByteOffset).not.toBe("0");
    expect(convergence.storage.destinationByteOffset).not.toBe("0");
    expect(result.destinationWords[0]).toBe(convergence.expected.initialDestinationWords[0]);
    expect(result.destinationWords.at(-1)).toBe(
      convergence.expected.initialDestinationWords.at(-1),
    );
    const actualDestinationHash = await hashWords(result.destinationWords);
    expect(actualDestinationHash).toBe(expectedDestinationHash);
    caseEvidence = Object.freeze({
      caseId: CPP_CUTE_VIEW_COPY_WEBGPU_CASE_ID,
      fixtureArtifactHash,
      inputHash,
      expectedDestinationHash,
      actualDestinationHash,
      layoutSemanticHash: artifacts.layoutSemanticHash,
      kernelSemanticHash: artifacts.kernelSemanticHash,
      operationId: artifacts.operationId,
      cpuSemanticSpecializationHash,
      gpuSemanticSpecializationHash: result.trace.semanticSpecializationHash,
      preparedBackendArtifactHash,
      wgslModuleHash: prepared.wgslModuleHash,
      backendSpecializationHash: result.trace.backendSpecializationHash,
      backendProfile: prepared.backendProfile,
      backendVersion: prepared.backendVersion,
      deviceProfileHash,
      completeDestinationBitComparisonPassed: true,
      nonzeroOffsetCanariesPreserved: true,
    });
    await raceDeviceLoss(
      withEvidenceTimeout(
        device.queue.onSubmittedWorkDone(),
        EVIDENCE_TIMEOUT_MS,
        "final-queue-drain",
      ),
      deviceLoss,
    );
    await raceDeviceLoss(
      withEvidenceTimeout(nextTask(), 1_000, "late-error-task-yield"),
      deviceLoss,
    );
    if (deviceLostBeforeTerminal !== undefined) {
      throw new EvidenceLaneError(
        DEVICE_LOST_DIAGNOSTIC,
        `${deviceLostBeforeTerminal.reason}: ${deviceLostBeforeTerminal.message}`,
      );
    }
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError(
        UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
        uncapturedErrors.join("; "),
      );
    }
    emitTerminal({
      required,
      fixtureArtifactHash,
      inputHash,
      expectedDestinationHash,
      preparedBackendArtifactHash,
      environment,
      environmentId,
      deviceProfileHash,
      outcome: "passed",
      diagnosticCodes: [],
      actualWebGpuExecution,
      caseEvidence,
      uncapturedErrors,
    });
    terminalEmitted = true;
  } catch (error) {
    if (!terminalEmitted) {
      const diagnostic = diagnosticCode(error, uncapturedErrors);
      emitTerminal({
        required,
        fixtureArtifactHash,
        inputHash,
        expectedDestinationHash,
        ...(preparedBackendArtifactHash === undefined ? {} : { preparedBackendArtifactHash }),
        environment,
        environmentId,
        ...(deviceProfileHash === undefined ? {} : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [diagnostic],
        actualWebGpuExecution,
        ...(caseEvidence === undefined ? {} : { caseEvidence }),
        uncapturedErrors,
        error: errorRecord(error, diagnostic),
      });
    }
    throw error;
  } finally {
    device?.removeEventListener("uncapturederror", uncapturedHandler);
    kernelDevice?.clearCache();
    device?.destroy();
  }
});

async function fixtureArtifactHashFor(): Promise<string> {
  return hashNamedComponents({
    schema: convergence.schema,
    entryId: convergence.entryId,
    claims: convergence.claims,
    storage: convergence.storage,
    layoutSemanticHash: convergence.expected.layoutSemanticHash,
    kernelSemanticHash: convergence.expected.kernelSemanticHash,
    operationId: convergence.expected.operationId,
    sourceWords: convergence.expected.sourceWords,
    initialDestinationWords: convergence.expected.initialDestinationWords,
    expectedDestinationWords: convergence.expected.destinationWords,
  });
}

function emitTerminal(input: Readonly<{
  required: boolean;
  fixtureArtifactHash: string;
  inputHash: string;
  expectedDestinationHash: string;
  preparedBackendArtifactHash?: string;
  environment: EvidenceEnvironment;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  actualWebGpuExecution: boolean;
  caseEvidence?: CppCuteViewCopyWebGpuCaseEvidence;
  uncapturedErrors: readonly string[];
  error?: JsonObject;
}>): void {
  const unsigned = Object.freeze({
    schema: "browsergrad.execution-evidence@1",
    kind: "terminal",
    suiteId: SUITE_ID,
    required: input.required,
    evidence: Object.freeze({
      capabilityId: CAPABILITY_ID,
      artifactHash: input.fixtureArtifactHash,
      backendId: BACKEND_ID,
      environmentId: input.environmentId,
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: __BG_SOURCE_REVISION__,
      ...(input.deviceProfileHash === undefined
        ? {}
        : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    }),
    environment: input.environment,
    artifactHashKind: "pinned-cpp-cute-view-copy-convergence-fixture",
    fixtureSchema: convergence.schema,
    inputHash: input.inputHash,
    expectedDestinationHash: input.expectedDestinationHash,
    ...(input.preparedBackendArtifactHash === undefined
      ? {}
      : { preparedBackendArtifactHash: input.preparedBackendArtifactHash }),
    productionBrowserCompileObserved:
      convergence.claims.productionBrowserCompileObserved,
    actualWebGpuExecution: input.actualWebGpuExecution,
    backendExecutionAuthorizationMinted:
      convergence.claims.backendExecutionAuthorizationMinted,
    cudaLiteRunnerUsed: convergence.claims.cudaLiteRunnerUsed,
    ...(input.caseEvidence === undefined ? {} : { case: input.caseEvidence }),
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.error === undefined ? {} : { error: input.error }),
  }) as CppCuteViewCopyWebGpuTerminalEvidence;
  TERMINAL_EMITTER.emit(finalizeCppCuteViewCopyWebGpuEvidence(unsigned, {
    expectedRequired: input.required,
    expectedFixtureSchema: convergence.schema,
    expectedFixtureArtifactHash: input.fixtureArtifactHash,
    expectedInputHash: input.inputHash,
    expectedDestinationHash: input.expectedDestinationHash,
    expectedEnvironmentId: input.environmentId,
    expectedSourceRevision: __BG_SOURCE_REVISION__,
    expectedProducerVersions: PRODUCER_VERSIONS,
  }));
}

function freezeEnvironment(input: Readonly<{
  acquisition: string;
  adapter?: JsonObject;
  adapterSupportedFeatures?: readonly string[];
  negotiatedDeviceFeatures?: readonly string[];
  negotiatedDeviceLimits?: JsonObject;
  unavailableReason?: string;
}>): EvidenceEnvironment {
  return Object.freeze({
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
    acquisition: input.acquisition,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    ...(input.adapter === undefined ? {} : { adapter: Object.freeze({ ...input.adapter }) }),
    ...(input.adapterSupportedFeatures === undefined ? {} : {
      adapterSupportedFeatures: Object.freeze([...input.adapterSupportedFeatures]),
    }),
    ...(input.negotiatedDeviceFeatures === undefined ? {} : {
      negotiatedDeviceFeatures: Object.freeze([...input.negotiatedDeviceFeatures]),
    }),
    ...(input.negotiatedDeviceLimits === undefined ? {} : {
      negotiatedDeviceLimits: Object.freeze({ ...input.negotiatedDeviceLimits }),
    }),
    ...(input.unavailableReason === undefined ? {} : {
      unavailableReason: input.unavailableReason,
    }),
  });
}

function deviceLimits(device: GPUDevice): JsonObject {
  return Object.freeze({
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxBindingsPerBindGroup: device.limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
  });
}

async function hashWords(words: Uint32Array): Promise<string> {
  return hashNamedComponents({ byteLength: words.byteLength, words: [...words] });
}

async function hashWordsPair(
  sourceWords: Uint32Array,
  destinationWords: Uint32Array,
): Promise<string> {
  return hashNamedComponents({
    source: { byteLength: sourceWords.byteLength, words: [...sourceWords] },
    destination: { byteLength: destinationWords.byteLength, words: [...destinationWords] },
  });
}

function bytes(words: Uint32Array): Uint8Array {
  return new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
}

async function raceDeviceLoss<T>(
  promise: Promise<T>,
  loss: Promise<GPUDeviceLostInfo> | undefined,
): Promise<T> {
  if (loss === undefined) return promise;
  const result = await Promise.race([
    promise.then((value) => ({ kind: "value" as const, value })),
    loss.then((info) => ({ kind: "lost" as const, info })),
  ]);
  if (result.kind === "lost") {
    throw new EvidenceLaneError(
      DEVICE_LOST_DIAGNOSTIC,
      `${result.info.reason}: ${result.info.message}`,
    );
  }
  return result.value;
}

async function withEvidenceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  name: string,
): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => reject(new EvidenceLaneError(
          EVIDENCE_TIMEOUT_DIAGNOSTIC,
          `${name} did not settle within ${timeoutMs}ms`,
        )), timeoutMs);
      }),
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function diagnosticCode(
  error: unknown,
  uncapturedErrors: readonly string[],
): string {
  if (uncapturedErrors.length > 0) return UNCAPTURED_GPU_ERROR_DIAGNOSTIC;
  if (error instanceof EvidenceLaneError) return error.code;
  return EXECUTION_FAILURE_DIAGNOSTIC;
}

function errorRecord(error: unknown, code: string): JsonObject {
  return Object.freeze({
    name: error instanceof Error ? error.name : "UnknownError",
    message: error instanceof Error ? error.message : String(error),
    code,
  });
}
