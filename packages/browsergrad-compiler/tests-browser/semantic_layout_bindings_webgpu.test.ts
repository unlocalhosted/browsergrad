import { it } from "vitest";
import {
  layoutArtifactPayload,
  traceViewCoordinate,
  verifyLayoutArtifact,
  type DimExpr,
  type IndexExpr,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  encodeWireI64,
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireI64,
  type JsonObject,
  type WireI64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice, type KernelDevice } from "@unlocalhosted/browsergrad-kernels";
import {
  CudaLiteCompilerError,
  compileCudaLiteKernelWithLayoutBindings,
  createCudaWebGpuExecutionPlan,
  prepareCompiledKernelWebGpu,
  prepareCudaLiteLayoutBindings,
  runCompiledKernelSemanticReference,
  type CompiledCudaLiteLayoutBoundKernel,
  type PreparedCompiledKernelWebGpu,
} from "../src/index";
import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createTerminalEvidenceEmitter,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";

declare const __BG_COMPILER_VERSION__: string;
declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const EVIDENCE_PREFIX = "[browsergrad-webgpu-evidence]";
const SUITE_ID = "browsergrad.compiler.layout-bindings.webgpu-conformance@2";
const CAPABILITY_ID = "browsergrad.compiler.verified-layout-read";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID = "browsergrad.comparison.bit-exact-f32-finite-complete-buffers.v1";
const SOURCE_ROOT_WORDS = 64;
const OUTPUT_WORDS = 16;
const LOGICAL_ELEMENTS = 6;
const WORKGROUP_SIZE = 8;
const PLANNED_CASE_IDS = Object.freeze([
  "rank2-transpose-nonzero-offset",
  "positive-strided-slice",
  "read-only-broadcast",
  "byte-map-nonzero-offset",
  "rank3-permutation",
  "dynamic-rank2-specialization",
]);
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-compiler": __BG_COMPILER_VERSION__,
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
});
const TERMINAL_EXPECTATION = Object.freeze({
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, TERMINAL_EXPECTATION);
const LAUNCH = Object.freeze({
  gridDim: [1, 1, 1] as const,
  blockDim: [WORKGROUP_SIZE, 1, 1] as const,
});
const COMPILE_OPTIONS = Object.freeze({ workgroupSize: [WORKGROUP_SIZE, 1, 1] as const });
const COPY_SOURCE = `
__global__ void copy_verified_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${LOGICAL_ELEMENTS}u) output[i] = input[i];
}`;

interface CaseSpec {
  readonly id: string;
  readonly wireShape: readonly DimExpr[];
  readonly resolvedShape: readonly number[];
  readonly location: IndexExpr;
  readonly locationUnit?: "element" | "byte";
  readonly byteOffset?: DimExpr;
  readonly symbols?: readonly { readonly id: string; readonly domain: { readonly min: string; readonly max: string } }[];
  readonly dimensionValues?: Readonly<Record<string, WireI64>>;
}

interface PreparedEvidenceCase {
  readonly id: string;
  readonly layout: VerifiedLayoutArtifact;
  readonly compiled: CompiledCudaLiteLayoutBoundKernel;
  readonly source: Float32Array;
  readonly initialOutput: Float32Array;
  readonly expectedSource: Uint32Array;
  readonly expectedOutput: Uint32Array;
  readonly expectedPhysicalIndices: readonly number[];
  readonly layoutSemanticHash: string;
  readonly compileIdentityHash: string;
  readonly wgslModuleHash: string;
  readonly physicalIndexHash: string;
  readonly inputHash: string;
  readonly artifactHash: string;
}

interface CaseObservation extends JsonObject {
  readonly caseId: string;
  readonly artifactHash: string;
  readonly inputHash: string;
  readonly layoutSemanticHash: string;
  readonly bindingProjectionHash: string;
  readonly compileIdentityHash: string;
  readonly wgslModuleHash: string;
  readonly physicalIndexHash: string;
  readonly programName: string;
  readonly planKind: string;
  readonly stepCount: number;
  readonly plannedPipelineCount: number;
  readonly logicalInvocationCount: readonly number[];
  readonly plannedWorkgroupCount: readonly number[];
  readonly comparisonPolicyId: typeof COMPARISON_POLICY_ID;
}

interface EvidenceEnvironment extends JsonObject {
  readonly schema: typeof EXECUTION_ENVIRONMENT_SCHEMA;
  readonly acquisition: string;
  readonly userAgent: string;
  readonly platform: string;
  readonly adapter?: JsonObject;
  readonly adapterSupportedFeatures?: readonly string[];
  readonly negotiatedDeviceFeatures?: readonly string[];
  readonly negotiatedDeviceLimits?: JsonObject;
  readonly unavailableReason?: string;
}

interface TerminalEvidenceRecord extends JsonObject {
  readonly schema: typeof EXECUTION_EVIDENCE_SCHEMA;
  readonly kind: "terminal";
  readonly suiteId: typeof SUITE_ID;
  readonly required: boolean;
  readonly evidence: JsonObject;
  readonly environment: EvidenceEnvironment;
  readonly artifactHashKind: "planned-suite-manifest" | "prepared-suite";
  readonly caseSetHash?: string;
  readonly plannedCaseIds: readonly string[];
  readonly completedCases: readonly CaseObservation[];
  readonly stage: string;
  readonly currentCaseId?: string;
  readonly uncapturedErrors: readonly string[];
  readonly error?: JsonObject;
}

it("executes verified compiler layout bindings on a required real GPUDevice", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage = "suite-manifest";
  let currentCaseId: string | undefined;
  let artifactHash = await hashNamedComponents({ suiteId: SUITE_ID, plannedCaseIds: PLANNED_CASE_IDS });
  let artifactHashKind: TerminalEvidenceRecord["artifactHashKind"] = "planned-suite-manifest";
  let caseSetHash: string | undefined;
  let environment = freezeEnvironment({ acquisition: "not-attempted" });
  let environmentId = await hashNamedComponents({ environment });
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
    stage = "fixture-and-compiler-preparation";
    const preparedCases = await Promise.all(createCaseSpecs().map(prepareEvidenceCase));
    assertPlannedCases(preparedCases);
    artifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifacts: preparedCases.map((entry) => ({ caseId: entry.id, artifactHash: entry.artifactHash })),
    });
    artifactHashKind = "prepared-suite";
    caseSetHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifactHash,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      launch: LAUNCH,
      cases: preparedCases.map((entry) => ({
        caseId: entry.id,
        artifactHash: entry.artifactHash,
        inputHash: entry.inputHash,
      })),
    });

    stage = "device-acquisition";
    const acquisition = await acquireWebGpuEvidenceDevice();
    if (acquisition.kind === "unavailable") {
      environment = freezeEnvironment({
        acquisition: "navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: acquisition.reason,
      });
      environmentId = await hashNamedComponents({ environment });
      emitTerminalEvidence({
        required,
        artifactHash,
        artifactHashKind,
        caseSetHash,
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: ["BG-WEBGPU-EVIDENCE-DEVICE-UNAVAILABLE"],
        completedCases,
        stage,
        uncapturedErrors,
        error: { name: "WebGpuEvidenceUnavailable", message: acquisition.reason },
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
    const adapterSupportedFeatures = Object.freeze([...adapter.features].map(String).sort());
    const negotiatedDeviceFeatures = Object.freeze([...device.features].map(String).sort());
    const negotiatedDeviceLimits = deviceLimits(device);
    environment = freezeEnvironment({
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      adapter: adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures,
      negotiatedDeviceFeatures,
      negotiatedDeviceLimits,
    });
    environmentId = await hashNamedComponents({ environment });
    deviceProfileHash = await hashNamedComponents({
      backendId: BACKEND_ID,
      adapter: adapterInfo as unknown as JsonObject,
      selectedFeatures: [],
      adapterSupportedFeatures,
      negotiatedDeviceFeatures,
      negotiatedDeviceLimits,
    });
    stage = "kernel-device-construction";
    kernelDevice = await createDevice({ device });

    for (const preparedCase of preparedCases) {
      currentCaseId = preparedCase.id;
      stage = "plan-validation";
      const gpuInput = {
        buffers: {
          input: cloneFloat32(preparedCase.source),
          output: cloneFloat32(preparedCase.initialOutput),
        },
        readback: ["input", "output"],
      };
      const plan = createCudaWebGpuExecutionPlan(preparedCase.compiled, gpuInput, LAUNCH);
      if (!plan.supported) throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-PLAN", plan.reason);
      if (plan.kind !== "single-dispatch" || plan.steps.length !== 1) {
        throw new EvidenceLaneError(
          "BG-WEBGPU-EVIDENCE-PLAN",
          `${preparedCase.id} expected one single-dispatch step; got ${plan.kind}/${plan.steps.length}`,
        );
      }
      const plannedPipelineCount = new Set(plan.steps.map((step) => `${step.program.name}:${step.program.wgsl}`)).size;
      if (plannedPipelineCount !== 1) throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-PLAN", `${preparedCase.id} expected one unique pipeline`);

      stage = "webgpu-preparation";
      let prepared: PreparedCompiledKernelWebGpu | undefined;
      try {
        prepared = await raceDeviceLoss(
          withEvidenceTimeout(
            prepareCompiledKernelWebGpu(kernelDevice, preparedCase.compiled, gpuInput, LAUNCH),
            10_000,
            "prepare-compiled-kernel",
          ),
          deviceLoss,
        );
        if (prepared.kind !== plan.kind || prepared.stepCount !== plan.steps.length) {
          throw new EvidenceLaneError(
            "BG-WEBGPU-EVIDENCE-PLAN",
            `${preparedCase.id} prepared topology differs from validated plan`,
          );
        }
        stage = "case-execution";
        const gpuResult = await raceDeviceLoss(
          withEvidenceTimeout(
            prepared.run({ awaitCompletion: true, readback: ["input", "output"] }),
            10_000,
            "prepared-run",
          ),
          deviceLoss,
        );
        await raceDeviceLoss(withEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "queue-drain"), deviceLoss);
        assertCompleteBufferBits(preparedCase, gpuResult.buffers);
        const logicalInvocationCount = Object.freeze([...plan.steps[0]!.launch.dispatchCount]);
        const plannedWorkgroupCount = Object.freeze(logicalInvocationCount.map((count, axis) => (
          Math.max(Math.ceil(count / plan.steps[0]!.program.workgroupSize[axis]!), 1)
        )));
        completedCases.push(Object.freeze({
          caseId: preparedCase.id,
          artifactHash: preparedCase.artifactHash,
          inputHash: preparedCase.inputHash,
          layoutSemanticHash: preparedCase.layoutSemanticHash,
          bindingProjectionHash: preparedCase.compiled.preparedLayoutBindings.bindingProjectionHash,
          compileIdentityHash: preparedCase.compileIdentityHash,
          wgslModuleHash: preparedCase.wgslModuleHash,
          physicalIndexHash: preparedCase.physicalIndexHash,
          programName: preparedCase.compiled.wgslProgram!.name,
          planKind: plan.kind,
          stepCount: prepared.stepCount,
          plannedPipelineCount,
          logicalInvocationCount,
          plannedWorkgroupCount,
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
        "BG-WEBGPU-EVIDENCE-DEVICE-LOST",
        `${deviceLostBeforeTerminal.reason}: ${deviceLostBeforeTerminal.message}`,
      );
    }
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-UNCAUGHT-GPU-ERROR", uncapturedErrors.join("; "));
    }
    stage = "terminal-summary";
    emitTerminalEvidence({
      required,
      artifactHash,
      artifactHashKind,
      caseSetHash,
      environment,
      environmentId,
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
      environmentId = await hashNamedComponents({ environment });
      emitTerminalEvidence({
        required,
        artifactHash,
        artifactHashKind,
        ...(caseSetHash === undefined ? {} : { caseSetHash }),
        environment,
        environmentId,
        ...(deviceProfileHash === undefined ? {} : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [diagnosticCode(error, stage)],
        completedCases,
        stage,
        ...(currentCaseId === undefined ? {} : { currentCaseId }),
        uncapturedErrors,
        error: errorRecord(error),
      });
    }
    throw error;
  } finally {
    device?.removeEventListener("uncapturederror", uncapturedHandler);
    kernelDevice?.clearCache();
    device?.destroy();
  }
});

function createCaseSpecs(): readonly CaseSpec[] {
  const n: DimExpr = { kind: "symbol", id: "n" };
  return Object.freeze([
    {
      id: "rank2-transpose-nonzero-offset",
      wireShape: dims("2", "3"),
      resolvedShape: [2, 3],
      location: add(multiply(coordinate(1), indexConstant("2")), coordinate(0)),
      byteOffset: dimConstant("8"),
    },
    {
      id: "positive-strided-slice",
      wireShape: dims("2", "3"),
      resolvedShape: [2, 3],
      location: add(
        multiply(add(coordinate(0), indexConstant("1")), indexConstant("4")),
        multiply(coordinate(1), indexConstant("2")),
      ),
    },
    {
      id: "read-only-broadcast",
      wireShape: dims("2", "3"),
      resolvedShape: [2, 3],
      location: coordinate(1),
    },
    {
      id: "byte-map-nonzero-offset",
      wireShape: dims("2", "3"),
      resolvedShape: [2, 3],
      locationUnit: "byte",
      byteOffset: dimConstant("4"),
      location: add(
        multiply(coordinate(0), indexConstant("24")),
        multiply(coordinate(1), indexConstant("8")),
      ),
    },
    {
      id: "rank3-permutation",
      wireShape: dims("1", "2", "3"),
      resolvedShape: [1, 2, 3],
      location: add(
        multiply(coordinate(2), indexConstant("2")),
        coordinate(1),
        multiply(coordinate(0), indexConstant("6")),
      ),
    },
    {
      id: "dynamic-rank2-specialization",
      wireShape: [n, dimConstant("3")],
      resolvedShape: [2, 3],
      location: add(multiply(coordinate(1), indexConstant("2")), coordinate(0)),
      symbols: [{ id: "n", domain: { min: "1", max: "8" } }],
      dimensionValues: { n: parseWireI64("2") },
    },
  ]);
}

async function prepareEvidenceCase(spec: CaseSpec): Promise<PreparedEvidenceCase> {
  const layout = await verifiedLayout(spec);
  const payload = layoutArtifactPayload(layout);
  const view = payload.views[0];
  if (view === undefined) throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-FIXTURE", `${spec.id} view missing`);
  const dimensionBindings = spec.dimensionValues === undefined
    ? undefined
    : Object.fromEntries(payload.symbols.map((symbol) => {
        const original = spec.symbols?.find((candidate) => candidate.domain.min === symbol.domain.min && candidate.domain.max === symbol.domain.max);
        const value = original === undefined ? undefined : spec.dimensionValues?.[original.id];
        if (value === undefined) throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-FIXTURE", `${spec.id} dimension binding missing`);
        return [symbol.id, value];
      }));
  const preparedBindings = await prepareCudaLiteLayoutBindings(layout, [{
    parameter: "input",
    viewId: view.viewId,
    access: "read",
    indexing: "row-major-flat",
    ...(dimensionBindings === undefined ? {} : { dimensionBindings }),
  }]);
  const compiled = compileCudaLiteKernelWithLayoutBindings(COPY_SOURCE, preparedBindings, COMPILE_OPTIONS);
  if (compiled.wgsl === undefined || compiled.wgslProgram === undefined) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-PREPARATION", `${spec.id} produced no WGSL program`);
  }
  if (
    !compiled.wgslProgram.name.includes(preparedBindings.layoutSemanticHash) ||
    !compiled.wgslProgram.name.includes(preparedBindings.bindingProjectionHash)
  ) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-PROGRAM-IDENTITY", `${spec.id} program name omitted layout proof hashes`);
  }

  const source = canonicalSource();
  const initialOutput = canonicalOutput();
  const cpuResult = runCompiledKernelSemanticReference(
    compiled,
    { buffers: { input: cloneFloat32(source), output: cloneFloat32(initialOutput) }, readback: ["input", "output"] },
    LAUNCH,
    { trace: "full" },
  );
  const expectedPhysicalIndices = logicalCoordinates(spec.resolvedShape).map((coordinates) => {
    const trace = traceViewCoordinate(layout, {
      viewId: view.viewId,
      coordinates: coordinates.map((value) => encodeWireI64(BigInt(value))),
      ...(dimensionBindings === undefined ? {} : { bindings: dimensionBindings }),
    });
    if (!trace.accessInBounds) throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-FIXTURE", `${spec.id} expected an in-bounds access`);
    return Number(BigInt(trace.rootByteStart) / 4n);
  });
  const cpuReadIndices = cpuResult.trace.flatMap((thread) => (
    thread.reads.filter((read) => read.name === "input").map((read) => read.index)
  ));
  if (!equalNumbers(cpuReadIndices, expectedPhysicalIndices)) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-PHYSICAL-INDEX", `${spec.id} CPU trace differs from semantic-core trace`);
  }
  const expectedSource = words(cpuResult.buffers.input);
  const expectedOutput = words(cpuResult.buffers.output);
  if (!equalWords(expectedSource, words(source))) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-READONLY", `${spec.id} CPU execution mutated the verified source root`);
  }

  const [layoutSemanticHash, compileIdentityHash, wgslModuleHash, physicalIndexHash, inputHash] = await Promise.all([
    hashSemanticArtifact(layout),
    hashNamedComponents({ layoutBindingCompileCacheKey: compiled.layoutBindingCompileCacheKey }),
    hashNamedComponents({ wgsl: compiled.wgsl }),
    hashNamedComponents({ expectedPhysicalIndices }),
    hashNamedComponents({ sourceWords: [...words(source)], initialOutputWords: [...words(initialOutput)] }),
  ]);
  const artifactHash = await hashNamedComponents({
    caseId: spec.id,
    layoutSemanticHash,
    bindingProjectionHash: preparedBindings.bindingProjectionHash,
    compileIdentityHash,
    wgslModuleHash,
    physicalIndexHash,
    programName: compiled.wgslProgram.name,
  });
  return Object.freeze({
    id: spec.id,
    layout,
    compiled,
    source,
    initialOutput,
    expectedSource,
    expectedOutput,
    expectedPhysicalIndices: Object.freeze(expectedPhysicalIndices),
    layoutSemanticHash,
    compileIdentityHash,
    wgslModuleHash,
    physicalIndexHash,
    inputHash,
    artifactHash,
  });
}

async function verifiedLayout(spec: CaseSpec): Promise<VerifiedLayoutArtifact> {
  return verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "compiler-layout-webgpu-evidence", version: "1" },
    artifactId: spec.id,
    requiredExtensions: [],
    payload: {
      symbols: spec.symbols ?? [],
      constraints: [],
      allocations: [{
        allocationId: "source",
        byteLength: dimConstant(String(SOURCE_ROOT_WORDS * 4)),
        memorySpace: { kind: "global" },
        alignmentBytes: 16,
        aliasSetId: "sourceAlias",
      }],
      indexMaps: [{
        indexMapId: "sourceMap",
        coordinateRank: spec.wireShape.length,
        locationUnit: spec.locationUnit ?? "element",
        location: spec.location,
        inBounds: { kind: "bool", value: true },
      }],
      views: [{
        viewId: "sourceView",
        allocationId: "source",
        dtype: "f32",
        byteOffset: spec.byteOffset ?? dimConstant("0"),
        shape: spec.wireShape,
        indexMapId: "sourceMap",
        requiredAlignmentBytes: 4,
      }],
    },
  });
}

function assertCompleteBufferBits(
  preparedCase: PreparedEvidenceCase,
  buffers: Readonly<Record<string, unknown>>,
): void {
  const source = buffers.input;
  const output = buffers.output;
  if (!(source instanceof Float32Array) || !(output instanceof Float32Array)) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-READBACK", `${preparedCase.id} returned wrong buffer types`);
  }
  if (!equalWords(words(source), preparedCase.expectedSource)) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-READONLY", `${preparedCase.id} WebGPU mutated the complete source root`);
  }
  if (!equalWords(words(output), preparedCase.expectedOutput)) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-EVIDENCE-COMPARISON",
      `${preparedCase.id} complete output differs under ${COMPARISON_POLICY_ID}`,
    );
  }
}

function assertPlannedCases(cases: readonly PreparedEvidenceCase[]): void {
  const actual = cases.map((entry) => entry.id);
  if (actual.length !== PLANNED_CASE_IDS.length || actual.some((id, index) => id !== PLANNED_CASE_IDS[index])) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-CASE-SET", `case set drifted: ${actual.join(",")}`);
  }
}

function validateTerminalEvidence(record: TerminalEvidenceRecord): void {
  validateTerminalExecutionEvidence(record, TERMINAL_EXPECTATION);
  const completedIds = record.completedCases.map((entry) => entry.caseId);
  if (new Set(completedIds).size !== completedIds.length || completedIds.some((id) => !PLANNED_CASE_IDS.includes(id))) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-CASE-SET", "completed cases are duplicated or unknown");
  }
  if (record.artifactHashKind === "prepared-suite" && record.caseSetHash === undefined) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "prepared-suite evidence requires caseSetHash");
  }
  if (record.evidence.outcome === "passed" && (
    completedIds.length !== PLANNED_CASE_IDS.length ||
    completedIds.some((id, index) => id !== PLANNED_CASE_IDS[index])
  )) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-CASE-SET", "passed evidence requires the complete ordered case set");
  }
}

function emitTerminalEvidence(input: Readonly<{
  required: boolean;
  artifactHash: string;
  artifactHashKind: TerminalEvidenceRecord["artifactHashKind"];
  caseSetHash?: string;
  environment: EvidenceEnvironment;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  completedCases: readonly CaseObservation[];
  stage: string;
  currentCaseId?: string;
  uncapturedErrors: readonly string[];
  error?: JsonObject;
}>): void {
  const record = Object.freeze({
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: input.required,
    evidence: Object.freeze({
      capabilityId: CAPABILITY_ID,
      artifactHash: input.artifactHash,
      backendId: BACKEND_ID,
      environmentId: input.environmentId,
      producerVersions: PRODUCER_VERSIONS,
      ...(input.deviceProfileHash === undefined ? {} : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    }),
    environment: input.environment,
    artifactHashKind: input.artifactHashKind,
    ...(input.caseSetHash === undefined ? {} : { caseSetHash: input.caseSetHash }),
    plannedCaseIds: PLANNED_CASE_IDS,
    completedCases: Object.freeze([...input.completedCases]),
    stage: input.stage,
    ...(input.currentCaseId === undefined ? {} : { currentCaseId: input.currentCaseId }),
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.error === undefined ? {} : { error: input.error }),
  }) as TerminalEvidenceRecord;
  validateTerminalEvidence(record);
  TERMINAL_EMITTER.emit(record);
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
    ...(input.adapterSupportedFeatures === undefined ? {} : { adapterSupportedFeatures: Object.freeze([...input.adapterSupportedFeatures]) }),
    ...(input.negotiatedDeviceFeatures === undefined ? {} : { negotiatedDeviceFeatures: Object.freeze([...input.negotiatedDeviceFeatures]) }),
    ...(input.negotiatedDeviceLimits === undefined ? {} : { negotiatedDeviceLimits: Object.freeze({ ...input.negotiatedDeviceLimits }) }),
    ...(input.unavailableReason === undefined ? {} : { unavailableReason: input.unavailableReason }),
  }) as EvidenceEnvironment;
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

function canonicalSource(): Float32Array {
  const source = Float32Array.from({ length: SOURCE_ROOT_WORDS }, (_, index) => index + 0.25);
  const sourceWords = words(source);
  sourceWords[0] = 0x00000000;
  sourceWords[1] = 0x80000000;
  return source;
}

function canonicalOutput(): Float32Array {
  const output = new Float32Array(OUTPUT_WORDS);
  const outputWords = words(output);
  for (let index = 0; index < outputWords.length; index += 1) outputWords[index] = 0x4f000001 + index;
  return output;
}

function logicalCoordinates(shape: readonly number[]): readonly number[][] {
  const count = shape.reduce((product, extent) => product * extent, 1);
  return Array.from({ length: count }, (_, flat) => shape.map((extent, axis) => {
    const stride = shape.slice(axis + 1).reduce((product, value) => product * value, 1);
    return Math.floor(flat / stride) % extent;
  }));
}

function dims(...values: readonly string[]): readonly DimExpr[] {
  return values.map(dimConstant);
}

function dimConstant(value: string): DimExpr {
  return { kind: "const", value: parseWireI64(value) };
}

function indexConstant(value: string): IndexExpr {
  return { kind: "const", value: parseWireI64(value) };
}

function coordinate(axis: number): IndexExpr {
  return { kind: "coordinate", axis };
}

function add(...terms: readonly IndexExpr[]): IndexExpr {
  return { kind: "add", terms };
}

function multiply(lhs: IndexExpr, rhs: IndexExpr): IndexExpr {
  return { kind: "mul", lhs, rhs };
}

function words(value: unknown): Uint32Array {
  if (!(value instanceof Float32Array)) throw new Error("expected Float32Array");
  return new Uint32Array(value.buffer, value.byteOffset, value.byteLength / 4);
}

function cloneFloat32(value: Float32Array): Float32Array {
  const copy = new Float32Array(value.length);
  words(copy).set(words(value));
  return copy;
}

function equalWords(left: Uint32Array, right: Uint32Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function equalNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

async function raceDeviceLoss<T>(promise: Promise<T>, loss: Promise<GPUDeviceLostInfo>): Promise<T> {
  const result = await Promise.race([
    promise.then((value) => ({ kind: "value" as const, value })),
    loss.then((info) => ({ kind: "lost" as const, info })),
  ]);
  if (result.kind === "lost") {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-DEVICE-LOST", `${result.info.reason}: ${result.info.message}`);
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
          "BG-WEBGPU-EVIDENCE-TIMEOUT",
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

function diagnosticCode(error: unknown, stage: string): string {
  if (error instanceof EvidenceLaneError) return error.code;
  if (error instanceof CudaLiteCompilerError) return error.diagnostics[0]?.code ?? "BG-WEBGPU-EVIDENCE-COMPILER";
  if (stage === "fixture-and-compiler-preparation") return "BG-WEBGPU-EVIDENCE-PREPARATION";
  if (stage === "device-acquisition") return "BG-WEBGPU-EVIDENCE-DEVICE-UNAVAILABLE";
  if (stage === "kernel-device-construction") return "BG-WEBGPU-EVIDENCE-DEVICE-WRAP";
  return "BG-WEBGPU-EVIDENCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof EvidenceLaneError) return { name: error.name, message: error.message, code: error.code };
  if (error instanceof CudaLiteCompilerError) {
    return {
      name: error.name,
      message: error.message,
      diagnosticCodes: error.diagnostics.map((diagnostic) => diagnostic.code),
    };
  }
  if (error instanceof Error) return { name: error.name, message: error.message };
  return { name: "UnknownError", message: String(error) };
}

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}
