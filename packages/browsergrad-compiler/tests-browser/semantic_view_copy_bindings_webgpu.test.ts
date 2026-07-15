import { it } from "vitest";
import {
  prepareViewCopyCpu,
  type PreparedViewCopyCpu,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { hashNamedComponents, type JsonObject } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice, type KernelDevice } from "@unlocalhosted/browsergrad-kernels";

import {
  CudaLiteCompilerError,
  compileCudaLiteKernelWithViewCopyBinding,
  createCudaWebGpuExecutionPlan,
  prepareCompiledKernelWebGpu,
  prepareCudaLiteViewCopyBinding,
  runCompiledKernelSemanticReference,
  type CompiledCudaLiteViewCopyBoundKernel,
  type PreparedCompiledKernelWebGpu,
  type PreparedCudaLiteViewCopyBinding,
} from "../src/index";
import {
  cloneViewCopyConformanceWords,
  createViewCopyConformanceCases,
  type ViewCopyConformanceCase,
} from "../../../test-support/view-copy-conformance-fixtures";
import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createTerminalEvidenceEmitter,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
} from "../../../test-support/webgpu-evidence";
import {
  BACKEND_ID,
  CAPABILITY_ID,
  COMPARISON_POLICY_ID,
  DEVICE_UNAVAILABLE_DIAGNOSTIC,
  PLANNED_CASE_IDS,
  SUITE_ID,
  TERMINAL_EXPECTATION,
  UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
  caseArtifactHashFor,
  caseSetHashFor,
  deviceProfileHashFor,
  environmentIdFor,
  finalizeTerminalEvidence,
  plannedSuiteArtifactHashFor,
  preparedBackendArtifactHashFor,
  preparedSuiteArtifactHashFor,
  type CaseObservation,
  type EvidenceEnvironment,
  type PreparedCaseManifest,
  type PreparedCaseManifestInput,
  type TerminalEvidenceRecord,
  type TerminalStage,
  type UnsignedTerminalEvidenceRecord,
} from "./semantic_view_copy_bindings_evidence";

declare const __BG_COMPILER_VERSION__: string;
declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;
declare const __BG_SOURCE_REVISION__: string;

const EVIDENCE_PREFIX = "[browsergrad-webgpu-evidence]";
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-compiler": __BG_COMPILER_VERSION__,
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
}) as JsonObject;
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, TERMINAL_EXPECTATION);

interface PreparedExecutionCase {
  readonly fixture: ViewCopyConformanceCase;
  readonly binding: PreparedCudaLiteViewCopyBinding;
  readonly compiled: CompiledCudaLiteViewCopyBoundKernel;
  readonly cpu: PreparedViewCopyCpu;
  readonly manifest: PreparedCaseManifest;
  readonly launch: Readonly<{
    gridDim: readonly [number, number, number];
    blockDim: readonly [number, number, number];
  }>;
}

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}

it("executes compiler-authorized view-copy bindings on a required real GPUDevice", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage: TerminalStage = "suite-manifest";
  let currentCaseId: string | undefined;
  let artifactHash = await plannedSuiteArtifactHashFor(__BG_SOURCE_REVISION__);
  let artifactHashKind: TerminalEvidenceRecord["artifactHashKind"] = "planned-suite-manifest";
  let preparedCases: readonly PreparedCaseManifest[] | undefined;
  let preparedBackendArtifactHash: string | undefined;
  let caseSetHash: string | undefined;
  let environment = freezeEnvironment({ acquisition: "not-attempted" });
  let deviceProfileHash: string | undefined;
  let terminalEmitted = false;
  const completedCases: CaseObservation[] = [];
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let kernelDevice: KernelDevice | undefined;
  let deviceLostBeforeTerminal: GPUDeviceLostInfo | undefined;
  let deviceLoss: Promise<GPUDeviceLostInfo> | undefined;
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => uncapturedErrors.push(event.error.message);

  try {
    stage = "case-preparation";
    const executionCases = await Promise.all((await createViewCopyConformanceCases()).map(prepareExecutionCase));
    const nextPreparedCases = Object.freeze(executionCases.map(({ manifest }) => manifest));
    assertPreparedOrder(nextPreparedCases);
    const nextPreparedBackendArtifactHash = await preparedBackendArtifactHashFor(nextPreparedCases);
    const nextCaseSetHash = await caseSetHashFor(nextPreparedCases, __BG_SOURCE_REVISION__, PRODUCER_VERSIONS);
    const nextArtifactHash = await preparedSuiteArtifactHashFor(
      nextPreparedBackendArtifactHash,
      nextCaseSetHash,
      __BG_SOURCE_REVISION__,
      PRODUCER_VERSIONS,
    );
    preparedCases = nextPreparedCases;
    preparedBackendArtifactHash = nextPreparedBackendArtifactHash;
    caseSetHash = nextCaseSetHash;
    artifactHash = nextArtifactHash;
    artifactHashKind = "prepared-case-set";

    stage = "device-acquisition";
    const acquisition = await acquireWebGpuEvidenceDevice();
    if (acquisition.kind === "unavailable") {
      environment = freezeEnvironment({
        acquisition: "navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: acquisition.reason,
      });
      await emitTerminal({
        required,
        artifactHash,
        artifactHashKind,
        preparedBackendArtifactHash,
        caseSetHash,
        preparedCases,
        environment,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: [DEVICE_UNAVAILABLE_DIAGNOSTIC],
        completedCases,
        stage,
        uncapturedErrors,
        error: { name: "WebGpuEvidenceUnavailable", message: acquisition.reason, code: DEVICE_UNAVAILABLE_DIAGNOSTIC },
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
    deviceProfileHash = await deviceProfileHashFor(environment);

    stage = "kernel-device-construction";
    kernelDevice = await createDevice({ device });

    for (const executionCase of executionCases) {
      currentCaseId = executionCase.fixture.id;
      const gpuInput = {
        buffers: {
          input: cloneViewCopyConformanceWords(executionCase.fixture.sourceWords),
          output: cloneViewCopyConformanceWords(executionCase.fixture.initialDestinationWords),
        },
        readback: ["input", "output"],
      };
      stage = "plan-validation";
      const plan = createCudaWebGpuExecutionPlan(executionCase.compiled, gpuInput, executionCase.launch);
      if (!plan.supported) throw new EvidenceLaneError("BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-PLAN", plan.reason);
      if (plan.kind !== "single-dispatch" || plan.steps.length !== 1) {
        throw new EvidenceLaneError(
          "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-PLAN",
          `${executionCase.fixture.id} expected single-dispatch/1, received ${plan.kind}/${plan.steps.length}`,
        );
      }
      const step = plan.steps[0]!;
      if (!equalNumbers(step.launch.dispatchCount, executionCase.manifest.logicalInvocationCount)) {
        throw new EvidenceLaneError(
          "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-TOPOLOGY",
          `${executionCase.fixture.id} plan invocation count drifted`,
        );
      }
      const plannedWorkgroups = step.launch.dispatchCount.map((count, axis) => (
        Math.max(Math.ceil(count / step.program.workgroupSize[axis]!), 1)
      ));
      if (!equalNumbers(plannedWorkgroups, executionCase.manifest.plannedWorkgroupCount)) {
        throw new EvidenceLaneError(
          "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-TOPOLOGY",
          `${executionCase.fixture.id} planned workgroup count drifted`,
        );
      }
      const plannedPipelineCount = new Set(plan.steps.map(({ program }) => `${program.name}:${program.wgsl}`)).size;
      if (plannedPipelineCount !== 1) {
        throw new EvidenceLaneError(
          "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-PLAN",
          `${executionCase.fixture.id} expected one unique pipeline`,
        );
      }

      stage = "webgpu-preparation";
      let prepared: PreparedCompiledKernelWebGpu | undefined;
      try {
        prepared = await raceDeviceLoss(
          withEvidenceTimeout(
            prepareCompiledKernelWebGpu(kernelDevice, executionCase.compiled, gpuInput, executionCase.launch),
            10_000,
            "prepare-compiled-view-copy",
          ),
          deviceLoss,
        );
        if (prepared.kind !== plan.kind || prepared.stepCount !== plan.steps.length) {
          throw new EvidenceLaneError(
            "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-PLAN",
            `${executionCase.fixture.id} prepared topology differs from validated plan`,
          );
        }
        stage = "case-execution";
        const gpuResult = await raceDeviceLoss(
          withEvidenceTimeout(
            prepared.run({ awaitCompletion: true, readback: ["input", "output"] }),
            10_000,
            "run-compiled-view-copy",
          ),
          deviceLoss,
        );
        stage = "queue-drain";
        await raceDeviceLoss(withEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "case-queue-drain"), deviceLoss);
        const actualSource = requireWords(gpuResult.buffers.input, `${executionCase.fixture.id} source`);
        const actualDestination = requireWords(gpuResult.buffers.output, `${executionCase.fixture.id} destination`);
        assertWords(actualSource, executionCase.fixture.expectedSourceWords, `${executionCase.fixture.id} source mutated`);
        assertWords(
          actualDestination,
          executionCase.fixture.expectedDestinationWords,
          `${executionCase.fixture.id} destination differs`,
        );
        const [actualSourceHash, actualDestinationHash] = await Promise.all([
          hashWords(actualSource),
          hashWords(actualDestination),
        ]);
        completedCases.push(Object.freeze({
          ...executionCase.manifest,
          actualSourceHash,
          actualDestinationHash,
          planKind: plan.kind,
          stepCount: prepared.stepCount,
          plannedPipelineCount,
          comparisonPolicyId: COMPARISON_POLICY_ID,
        }));
      } finally {
        prepared?.destroy();
      }
    }

    stage = "late-error-drain";
    currentCaseId = undefined;
    await raceDeviceLoss(withEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "final-queue-drain"), deviceLoss);
    await nextTask();
    if (deviceLostBeforeTerminal !== undefined) {
      throw new EvidenceLaneError(
        "BG-COMPILER-VIEW-COPY-BINDING-DEVICE-LOST",
        `${deviceLostBeforeTerminal.reason}: ${deviceLostBeforeTerminal.message}`,
      );
    }
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError(UNCAPTURED_GPU_ERROR_DIAGNOSTIC, uncapturedErrors.join("; "));
    }

    stage = "terminal-summary";
    await emitTerminal({
      required,
      artifactHash,
      artifactHashKind,
      preparedBackendArtifactHash,
      caseSetHash,
      preparedCases,
      environment,
      deviceProfileHash,
      outcome: "passed",
      diagnosticCodes: [],
      completedCases,
      stage,
      uncapturedErrors,
    });
    terminalEmitted = true;
  } catch (error) {
    if (!terminalEmitted) {
      const diagnostic = diagnosticCode(error, uncapturedErrors);
      await emitTerminal({
        required,
        artifactHash,
        artifactHashKind,
        ...(preparedBackendArtifactHash === undefined ? {} : { preparedBackendArtifactHash }),
        ...(caseSetHash === undefined ? {} : { caseSetHash }),
        ...(preparedCases === undefined ? {} : { preparedCases }),
        environment,
        ...(deviceProfileHash === undefined ? {} : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [diagnostic],
        completedCases,
        stage,
        ...(currentCaseId === undefined ? {} : { currentCaseId }),
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

async function prepareExecutionCase(fixture: ViewCopyConformanceCase): Promise<PreparedExecutionCase> {
  const binding = await prepareCudaLiteViewCopyBinding(fixture.artifacts.layout, fixture.artifacts.kernel, {
    operationId: fixture.artifacts.operationId,
    sourceParameter: "input",
    destinationParameter: "output",
    indexing: "row-major-flat",
  });
  const elementCount = fixture.logicalShape.reduce((product, extent) => product * extent, 1);
  const workgroupSize = elementCount;
  const source = directViewCopySource(elementCount);
  const compiled = compileCudaLiteKernelWithViewCopyBinding(source, binding, {
    workgroupSize: [workgroupSize, 1, 1],
  });
  if (compiled.wgsl === undefined || compiled.wgslProgram === undefined) {
    throw new EvidenceLaneError(
      "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-PREPARATION",
      `${fixture.id} produced no WGSL program`,
    );
  }
  const entry = compiled.wgsl.slice(compiled.wgsl.lastIndexOf("@compute"));
  if (!compiled.wgsl.includes("array<u32>") || entry.includes("select(") || entry.indexOf("input[") < entry.indexOf("if (")) {
    throw new EvidenceLaneError(
      "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-GUARD",
      `${fixture.id} lost structured raw-word source guarding`,
    );
  }

  const launch = Object.freeze({
    gridDim: [1, 1, 1] as const,
    blockDim: [workgroupSize, 1, 1] as const,
  });
  const cpu = await prepareViewCopyCpu(fixture.artifacts.layout, fixture.artifacts.kernel, {
    operationId: fixture.artifacts.operationId,
  });
  const canonicalSource = cloneViewCopyConformanceWords(fixture.sourceWords);
  const canonicalDestination = cloneViewCopyConformanceWords(fixture.initialDestinationWords);
  const canonicalTrace = cpu.execute({
    source: bytes(canonicalSource),
    destination: bytes(canonicalDestination),
  });
  assertWords(canonicalSource, fixture.expectedSourceWords, `${fixture.id} canonical CPU mutated source`);
  assertWords(canonicalDestination, fixture.expectedDestinationWords, `${fixture.id} canonical CPU destination differs`);
  if (
    canonicalTrace.readElements !== String(fixture.expectedReadElements)
    || canonicalTrace.filledElements !== String(fixture.expectedFilledElements)
  ) {
    throw new EvidenceLaneError(
      "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-CPU",
      `${fixture.id} canonical CPU read/fill counts differ`,
    );
  }

  const compilerResult = runCompiledKernelSemanticReference(
    compiled,
    {
      buffers: {
        input: cloneViewCopyConformanceWords(fixture.sourceWords),
        output: cloneViewCopyConformanceWords(fixture.initialDestinationWords),
      },
      readback: ["input", "output"],
    },
    launch,
    { trace: "full" },
  );
  const compilerSource = requireWords(compilerResult.buffers.input, `${fixture.id} compiler source`);
  const compilerDestination = requireWords(compilerResult.buffers.output, `${fixture.id} compiler destination`);
  assertWords(compilerSource, fixture.expectedSourceWords, `${fixture.id} compiler CPU mutated source`);
  assertWords(compilerDestination, fixture.expectedDestinationWords, `${fixture.id} compiler CPU destination differs`);
  const sourceReads = compilerResult.trace.flatMap((thread) => (
    thread.reads.filter(({ name }) => name === "input").map(({ index }) => index)
  ));
  const destinationWrites = compilerResult.trace.flatMap((thread) => (
    thread.writes.filter(({ name }) => name === "output").map(({ index }) => index)
  ));
  if (
    !equalNumbers(sourceReads, fixture.expectedSourcePhysicalIndices)
    || !equalNumbers(destinationWrites, fixture.expectedDestinationPhysicalIndices)
  ) {
    throw new EvidenceLaneError(
      "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-CPU",
      `${fixture.id} compiler CPU physical trace differs from shared fixture`,
    );
  }

  const logicalInvocationCount = Object.freeze([elementCount, 1, 1]);
  const plannedWorkgroupCount = Object.freeze([1, 1, 1]);
  const [compileIdentityHash, wgslModuleHash, sourceHash, initialDestinationHash, expectedSourceHash, expectedDestinationHash] = await Promise.all([
    hashNamedComponents({ compileCacheKey: compiled.viewCopyBindingCompileCacheKey }),
    hashNamedComponents({ wgsl: compiled.wgsl }),
    hashWords(fixture.sourceWords),
    hashWords(fixture.initialDestinationWords),
    hashWords(fixture.expectedSourceWords),
    hashWords(fixture.expectedDestinationWords),
  ]);
  const manifestInput: PreparedCaseManifestInput = {
    caseId: fixture.id,
    layoutSemanticHash: binding.layoutSemanticHash,
    kernelSemanticHash: binding.kernelSemanticHash,
    specializationHash: binding.specializationHash,
    bindingProjectionHash: binding.bindingProjectionHash,
    compileIdentityHash,
    wgslModuleHash,
    programName: compiled.wgslProgram.name,
    sourceHash,
    initialDestinationHash,
    expectedSourceHash,
    expectedDestinationHash,
    logicalShape: Object.freeze([...fixture.logicalShape]),
    logicalInvocationCount,
    plannedWorkgroupCount,
    expectedReadElements: fixture.expectedReadElements,
    expectedFilledElements: fixture.expectedFilledElements,
  };
  const manifest = Object.freeze({
    ...manifestInput,
    caseArtifactHash: await caseArtifactHashFor(manifestInput),
  });
  return Object.freeze({ fixture, binding, compiled, cpu, manifest, launch });
}

async function emitTerminal(input: Readonly<{
  required: boolean;
  artifactHash: string;
  artifactHashKind: TerminalEvidenceRecord["artifactHashKind"];
  preparedBackendArtifactHash?: string;
  caseSetHash?: string;
  preparedCases?: readonly PreparedCaseManifest[];
  environment: EvidenceEnvironment;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  completedCases: readonly CaseObservation[];
  stage: TerminalStage;
  currentCaseId?: string;
  uncapturedErrors: readonly string[];
  error?: JsonObject;
}>): Promise<void> {
  const unsigned = {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: input.required,
    evidence: {
      capabilityId: CAPABILITY_ID,
      artifactHash: input.artifactHash,
      backendId: BACKEND_ID,
      environmentId: await environmentIdFor(input.environment),
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: __BG_SOURCE_REVISION__,
      ...(input.deviceProfileHash === undefined ? {} : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    },
    environment: input.environment,
    artifactHashKind: input.artifactHashKind,
    ...(input.preparedBackendArtifactHash === undefined ? {} : {
      preparedBackendArtifactHash: input.preparedBackendArtifactHash,
    }),
    ...(input.caseSetHash === undefined ? {} : { caseSetHash: input.caseSetHash }),
    ...(input.preparedCases === undefined ? {} : { preparedCases: input.preparedCases }),
    plannedCaseIds: PLANNED_CASE_IDS,
    completedCases: Object.freeze([...input.completedCases]),
    stage: input.stage,
    ...(input.currentCaseId === undefined ? {} : { currentCaseId: input.currentCaseId }),
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.error === undefined ? {} : { error: input.error }),
  } as UnsignedTerminalEvidenceRecord;
  const record = await finalizeTerminalEvidence(unsigned, {
    expectedRequired: input.required,
    expectedSourceRevision: __BG_SOURCE_REVISION__,
    producerVersions: PRODUCER_VERSIONS,
  });
  TERMINAL_EMITTER.emit(record);
}

function assertPreparedOrder(cases: readonly PreparedCaseManifest[]): void {
  if (
    cases.length !== PLANNED_CASE_IDS.length
    || cases.some(({ caseId }, index) => caseId !== PLANNED_CASE_IDS[index])
  ) {
    throw new EvidenceLaneError(
      "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-CASE-SET",
      `prepared case order drifted: ${cases.map(({ caseId }) => caseId).join(",")}`,
    );
  }
}

function directViewCopySource(elementCount: number): string {
  return `
__global__ void copy_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${elementCount}u) output[i] = input[i];
}`;
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
    ...(input.unavailableReason === undefined ? {} : { unavailableReason: input.unavailableReason }),
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

function requireWords(value: unknown, name: string): Uint32Array {
  if (!(value instanceof Uint32Array)) {
    throw new EvidenceLaneError("BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-READBACK", `${name} is not Uint32Array`);
  }
  return value;
}

function bytes(words: Uint32Array): Uint8Array {
  return new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
}

async function hashWords(words: Uint32Array): Promise<string> {
  return hashNamedComponents({
    byteLength: words.byteLength,
    words: [...words],
  });
}

function assertWords(actual: Uint32Array, expected: Uint32Array, message: string): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new EvidenceLaneError("BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-COMPARISON", message);
  }
}

function equalNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

async function raceDeviceLoss<T>(promise: Promise<T>, loss: Promise<GPUDeviceLostInfo> | undefined): Promise<T> {
  if (loss === undefined) return promise;
  const result = await Promise.race([
    promise.then((value) => ({ kind: "value" as const, value })),
    loss.then((info) => ({ kind: "lost" as const, info })),
  ]);
  if (result.kind === "lost") {
    throw new EvidenceLaneError(
      "BG-COMPILER-VIEW-COPY-BINDING-DEVICE-LOST",
      `${result.info.reason}: ${result.info.message}`,
    );
  }
  return result.value;
}

async function withEvidenceTimeout<T>(promise: Promise<T>, timeoutMs: number, name: string): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => reject(new EvidenceLaneError(
          "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-TIMEOUT",
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

function diagnosticCode(error: unknown, uncapturedErrors: readonly string[]): string {
  if (uncapturedErrors.length > 0) return UNCAPTURED_GPU_ERROR_DIAGNOSTIC;
  if (error instanceof EvidenceLaneError) return error.code;
  if (error instanceof CudaLiteCompilerError) {
    return error.diagnostics[0]?.code ?? "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-COMPILER";
  }
  return "BG-COMPILER-VIEW-COPY-BINDING-EVIDENCE-INTERNAL";
}

function errorRecord(error: unknown, code: string): JsonObject {
  if (error instanceof CudaLiteCompilerError) {
    return {
      name: error.name,
      message: error.message,
      code,
      diagnosticCodes: error.diagnostics.map((diagnostic) => diagnostic.code),
    };
  }
  if (error instanceof Error) return { name: error.name, message: error.message, code };
  return { name: "UnknownError", message: String(error), code };
}
