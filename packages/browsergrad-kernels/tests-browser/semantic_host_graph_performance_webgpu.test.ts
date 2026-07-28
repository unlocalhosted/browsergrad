import { expect, it } from "vitest";
import {
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createWebGpuExecutionEnvironmentRecord,
  createTerminalEvidenceEmitter,
  nextWebGpuEvidenceTask,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
  webGpuSemanticDeviceLimits,
  withWebGpuEvidenceTimeout,
} from "../../../test-support/webgpu-evidence";

import {
  createVerifiedHostGraphArtifact,
  prepareHostGraphCpu,
  type HostGraphProgram,
  type VerifiedHostGraphArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  hashNamedComponents,
  parseWireU64,
  sha256Hex,
  type JsonObject,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
  SemanticHostGraphWebGpuError,
  destroySemanticHostGraphWebGpuPipeline,
  prepareSemanticHostGraphWebGpuPipeline,
  prepareSemanticHostGraphWebGpu,
  runSemanticHostGraphWebGpuPipeline,
  type PreparedSemanticHostGraphWebGpuPipeline,
  type PreparedSemanticHostGraphWebGpu,
  type SemanticHostGraphWebGpuInputBinding,
  type SemanticHostGraphWebGpuOutputBinding,
} from "../src/semantic_host_graph";
import type { KernelDevice } from "../src/types";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const EVIDENCE_PREFIX =
  "[browsergrad-semantic-host-graph-performance-evidence]";
const SUITE_ID =
  "browsergrad.kernels.semantic-host-graph.webgpu-performance@1";
const CORRECTNESS_SUITE_ID =
  "browsergrad.kernels.semantic-host-graph.webgpu-conformance@1";
const CAPABILITY_ID = "browsergrad.host-graph";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID =
  "browsergrad.comparison.cpu-reference-bit-exact-complete-outputs.v1";
const PERFORMANCE_RECORD_SCHEMA = "browsergrad.performance-observation@1";
const CANDIDATE_ID = "fixed-count-repeat-control";
const BASELINE_ID = "statically-unrolled-equivalent";
const WARMUP_ITERATIONS = 8;
const MEASURED_ITERATIONS = 12;
const WORKLOAD = Object.freeze({
  dtype: "f32",
  rankCount: 2,
  elementsPerRank: 65_536,
  repeatedAllReduceCount: 3,
  expandedWebGpuStepCount: 8,
});
const MEASUREMENT_METHOD = Object.freeze({
  boundary: "production-host-api-end-to-end",
  clock: "performance.now",
  completion: "complete-output-readback-plus-queue-drain",
  warmupIterations: WARMUP_ITERATIONS,
  measuredIterationsPerImplementation: MEASURED_ITERATIONS,
  ordering: "paired-alternating-first-implementation",
  candidateLifecycle:
    "prewarmed-pipeline-authority-captured-input-fixed-repeat-dispatch-readback-validation-cleanup",
  baselineLifecycle:
    "prewarmed-pipeline-authority-captured-input-unrolled-dispatch-readback-validation-cleanup",
  semanticRelation: "bit-exact-equivalent-expanded-work",
  comparisonClaim: "observational-only-no-superiority-or-regression-threshold",
});
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.semantic-host-graph":
    SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, {
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});
const wire = (value: number): WireU64 => parseWireU64(String(value));

interface BenchmarkFixture {
  readonly repeatedGraph: VerifiedHostGraphArtifact;
  readonly unrolledGraph: VerifiedHostGraphArtifact;
  readonly repeatedPrepared: PreparedSemanticHostGraphWebGpu;
  readonly unrolledPrepared: PreparedSemanticHostGraphWebGpu;
  readonly inputs: readonly SemanticHostGraphWebGpuInputBinding[];
  readonly expectedOutputs: readonly SemanticHostGraphWebGpuOutputBinding[];
  readonly artifactHash: string;
}

interface TimingStatistics extends JsonObject {
  readonly samplesMs: readonly number[];
  readonly minimumMs: number;
  readonly medianMs: number;
  readonly p95Ms: number;
  readonly maximumMs: number;
  readonly meanMs: number;
}

interface PerformanceObservation extends JsonObject {
  readonly schema: typeof PERFORMANCE_RECORD_SCHEMA;
  readonly workload: JsonObject;
  readonly method: JsonObject;
  readonly candidate: JsonObject;
  readonly baseline: JsonObject;
  readonly candidateMedianOverBaselineMedian: number;
  readonly comparisonClaim:
    "observational-only-no-superiority-or-regression-threshold";
}

it("records separate named semantic host-graph performance evidence", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage = "suite-manifest";
  let artifactHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    workload: WORKLOAD as unknown as JsonObject,
    method: MEASUREMENT_METHOD as unknown as JsonObject,
  });
  let environment = createWebGpuExecutionEnvironmentRecord({
    acquisition: "not-attempted",
  });
  let environmentId = await hashNamedComponents({ environment });
  let deviceProfileHash: string | undefined;
  let observation: PerformanceObservation | undefined;
  let terminalEmitted = false;
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let kernelDevice: KernelDevice | undefined;
  let repeatedPipeline:
    PreparedSemanticHostGraphWebGpuPipeline | undefined;
  let unrolledPipeline:
    PreparedSemanticHostGraphWebGpuPipeline | undefined;
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => {
    uncapturedErrors.push(event.error.message);
  };

  try {
    stage = "fixture-construction";
    const fixture = await prepareFixture();
    artifactHash = fixture.artifactHash;

    stage = "device-acquisition";
    const acquisition = await acquireWebGpuEvidenceDevice();
    if (acquisition.kind === "unavailable") {
      environment = createWebGpuExecutionEnvironmentRecord({
        acquisition: "navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: acquisition.reason,
      });
      environmentId = await hashNamedComponents({ environment });
      emitTerminal({
        required,
        artifactHash,
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: [
          "BG-WEBGPU-GRAPH-PERFORMANCE-DEVICE-UNAVAILABLE",
        ],
        stage,
        uncapturedErrors,
        error: {
          name: "WebGpuEvidenceUnavailable",
          message: acquisition.reason,
        },
      });
      terminalEmitted = true;
      if (required) throw requiredEvidenceFailure(acquisition.reason);
      context.skip(acquisition.reason);
      return;
    }

    const acquired = acquisition.value;
    device = acquired.device;
    const adapterFeatures = Object.freeze(
      [...acquired.adapter.features].map(String).sort(),
    );
    const deviceFeatures = Object.freeze(
      [...device.features].map(String).sort(),
    );
    const negotiatedLimits = webGpuSemanticDeviceLimits(device);
    environment = createWebGpuExecutionEnvironmentRecord({
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      adapter: acquired.adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures: adapterFeatures,
      negotiatedDeviceFeatures: deviceFeatures,
      negotiatedDeviceLimits: negotiatedLimits,
    });
    environmentId = await hashNamedComponents({ environment });
    deviceProfileHash = await hashNamedComponents({
      backendId: BACKEND_ID,
      adapter: acquired.adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures: adapterFeatures,
      negotiatedDeviceFeatures: deviceFeatures,
      negotiatedDeviceLimits: negotiatedLimits,
    });

    stage = "kernel-device-construction";
    kernelDevice = await createDevice({ device });
    device.addEventListener("uncapturederror", uncapturedHandler);

    stage = "pipeline-authority-prewarm";
    [repeatedPipeline, unrolledPipeline] = await Promise.all([
      prepareSemanticHostGraphWebGpuPipeline(
        kernelDevice,
        fixture.repeatedPrepared,
      ),
      prepareSemanticHostGraphWebGpuPipeline(
        kernelDevice,
        fixture.unrolledPrepared,
      ),
    ]);

    stage = "separate-correctness-preflight";
    const candidate = await runGraph(
      repeatedPipeline,
      fixture.inputs,
    );
    const baseline = await runGraph(
      unrolledPipeline,
      fixture.inputs,
    );
    requireOutputEquality(
      candidate,
      fixture.expectedOutputs,
      CANDIDATE_ID,
    );
    requireOutputEquality(
      baseline,
      fixture.expectedOutputs,
      BASELINE_ID,
    );

    stage = "warmup";
    for (let index = 0; index < WARMUP_ITERATIONS; index += 1) {
      await runGraph(
        repeatedPipeline,
        fixture.inputs,
      );
      await runGraph(
        unrolledPipeline,
        fixture.inputs,
      );
    }

    stage = "measurement";
    const candidateSamples: number[] = [];
    const baselineSamples: number[] = [];
    for (let index = 0; index < MEASURED_ITERATIONS; index += 1) {
      const candidateFirst = index % 2 === 0;
      if (candidateFirst) {
        candidateSamples.push(await measure(device, () =>
          runGraph(
            repeatedPipeline!,
            fixture.inputs,
          )));
        baselineSamples.push(await measure(device, () =>
          runGraph(
            unrolledPipeline!,
            fixture.inputs,
          )));
      } else {
        baselineSamples.push(await measure(device, () =>
          runGraph(
            unrolledPipeline!,
            fixture.inputs,
          )));
        candidateSamples.push(await measure(device, () =>
          runGraph(
            repeatedPipeline!,
            fixture.inputs,
          )));
      }
    }
    const candidateStats = statistics(candidateSamples);
    const baselineStats = statistics(baselineSamples);
    observation = Object.freeze({
      schema: PERFORMANCE_RECORD_SCHEMA,
      workload: WORKLOAD as unknown as JsonObject,
      method: MEASUREMENT_METHOD as unknown as JsonObject,
      candidate: Object.freeze({
        implementationId: CANDIDATE_ID,
        pipelineIdentityHash: repeatedPipeline.pipelineIdentityHash,
        ...candidateStats,
      }),
      baseline: Object.freeze({
        implementationId: BASELINE_ID,
        pipelineIdentityHash: unrolledPipeline.pipelineIdentityHash,
        ...baselineStats,
      }),
      candidateMedianOverBaselineMedian:
        candidateStats.medianMs / baselineStats.medianMs,
      comparisonClaim:
        "observational-only-no-superiority-or-regression-threshold",
    });

    stage = "late-error-drain";
    await withPerformanceTimeout(
      device.queue.onSubmittedWorkDone(),
      10_000,
      "final-queue-drain",
    );
    await nextWebGpuEvidenceTask();
    if (uncapturedErrors.length > 0) {
      throw new PerformanceEvidenceError(
        "BG-WEBGPU-GRAPH-PERFORMANCE-UNCAUGHT-GPU-ERROR",
        uncapturedErrors.join("; "),
      );
    }

    stage = "terminal-summary";
    emitTerminal({
      required,
      artifactHash,
      environment,
      environmentId,
      deviceProfileHash,
      outcome: "passed",
      diagnosticCodes: [],
      stage,
      uncapturedErrors,
      observation,
    });
    terminalEmitted = true;
  } catch (error) {
    if (!terminalEmitted) {
      emitTerminal({
        required,
        artifactHash,
        environment,
        environmentId,
        ...(deviceProfileHash === undefined
          ? {}
          : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [diagnosticCode(error, stage)],
        stage,
        uncapturedErrors,
        ...(observation === undefined ? {} : { observation }),
        error: errorRecord(error),
      });
    }
    throw error;
  } finally {
    device?.removeEventListener("uncapturederror", uncapturedHandler);
    if (repeatedPipeline !== undefined) {
      destroySemanticHostGraphWebGpuPipeline(repeatedPipeline);
    }
    if (unrolledPipeline !== undefined) {
      destroySemanticHostGraphWebGpuPipeline(unrolledPipeline);
    }
    kernelDevice?.clearCache();
    device?.destroy();
  }
});

async function prepareFixture(): Promise<BenchmarkFixture> {
  const repeatedGraph = (await createVerifiedHostGraphArtifact(
    repeatedProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const unrolledGraph = (await createVerifiedHostGraphArtifact(
    unrolledProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const [repeatedPrepared, unrolledPrepared, repeatedCpu, unrolledCpu] =
    await Promise.all([
      prepareSemanticHostGraphWebGpu(repeatedGraph, {
        kernelArtifacts: [],
        layoutArtifacts: [],
      }),
      prepareSemanticHostGraphWebGpu(unrolledGraph, {
        kernelArtifacts: [],
        layoutArtifacts: [],
      }),
      prepareHostGraphCpu(repeatedGraph, {
        kernelArtifacts: [],
        layoutArtifacts: [],
      }),
      prepareHostGraphCpu(unrolledGraph, {
        kernelArtifacts: [],
        layoutArtifacts: [],
      }),
    ]);
  if (
    repeatedPrepared.expandedStepCount !==
      WORKLOAD.expandedWebGpuStepCount ||
    unrolledPrepared.expandedStepCount !==
      WORKLOAD.expandedWebGpuStepCount ||
    repeatedCpu.elementOperations !== unrolledCpu.elementOperations
  ) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-GRAPH-PERFORMANCE-WORK-MISMATCH",
      "candidate and baseline do not preserve exact expanded work",
    );
  }
  const inputs = Object.freeze([
    input(0, patternedF32Bytes(WORKLOAD.elementsPerRank, 31)),
    input(1, patternedF32Bytes(WORKLOAD.elementsPerRank, 37)),
  ]);
  const [repeatedExpected, unrolledExpected] = await Promise.all([
    repeatedCpu.execute({ inputs }),
    unrolledCpu.execute({ inputs }),
  ]);
  requireOutputEquality(
    repeatedExpected.outputs,
    unrolledExpected.outputs,
    "cpu-fixed-repeat-vs-unrolled",
  );
  const inputHashes = await Promise.all(
    inputs.map(({ bytes }) => sha256Hex(bytes)),
  );
  const artifactHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    workload: WORKLOAD as unknown as JsonObject,
    method: MEASUREMENT_METHOD as unknown as JsonObject,
    candidate: {
      graphSemanticHash: repeatedPrepared.graphSemanticHash,
      wgslModuleHashes: repeatedPrepared.wgslModuleHashes,
      expandedStepCount: repeatedPrepared.expandedStepCount,
    },
    baseline: {
      graphSemanticHash: unrolledPrepared.graphSemanticHash,
      wgslModuleHashes: unrolledPrepared.wgslModuleHashes,
      expandedStepCount: unrolledPrepared.expandedStepCount,
    },
    inputHashes,
  });
  return Object.freeze({
    repeatedGraph,
    unrolledGraph,
    repeatedPrepared,
    unrolledPrepared,
    inputs,
    expectedOutputs: repeatedExpected.outputs,
    artifactHash,
  });
}

function repeatedProgram(): HostGraphProgram {
  return {
    ...programFoundation({ major: 1, minor: 4 }),
    nodes: [
      initializeNode(),
      {
        nodeId: "repeat-reduction",
        kind: "repeat",
        dependsOn: ["initialize"],
        iterationCount: wire(WORKLOAD.repeatedAllReduceCount),
        body: [allReduceNode("reduction-body", [])],
        mode: "fixed-count-sequential",
      },
      materializeNode("repeat-reduction"),
    ],
  };
}

function unrolledProgram(): HostGraphProgram {
  const reductions = Array.from(
    { length: WORKLOAD.repeatedAllReduceCount },
    (_, index) => allReduceNode(
      `reduction-${index}`,
      [index === 0 ? "initialize" : `reduction-${index - 1}`],
    ),
  );
  return {
    ...programFoundation({ major: 1, minor: 2 }),
    nodes: [
      initializeNode(),
      ...reductions,
      materializeNode(`reduction-${reductions.length - 1}`),
    ],
  };
}

function programFoundation(
  version: HostGraphProgram["version"],
): Omit<HostGraphProgram, "nodes"> {
  const byteLength = wire(
    WORKLOAD.elementsPerRank * Float32Array.BYTES_PER_ELEMENT,
  );
  return {
    kind: "host-graph",
    version,
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(WORKLOAD.rankCount),
    resources: [
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "f32",
        byteLength,
        alignmentBytes: 4,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength,
        alignmentBytes: 4,
      },
    ],
  };
}

function initializeNode() {
  return {
    nodeId: "initialize",
    kind: "copy" as const,
    dependsOn: [] as const,
    sourceResourceId: "input",
    destinationResourceId: "output",
    mode: "whole-allocation-bytes-per-rank" as const,
  };
}

function allReduceNode(
  nodeId: string,
  dependsOn: readonly string[],
) {
  return {
    nodeId,
    kind: "all-reduce" as const,
    dependsOn,
    resourceId: "output",
    reduction: "sum" as const,
    dtype: "f32" as const,
    numericalPolicy: "rank-order-f32" as const,
    participants: [wire(0), wire(1)],
    result: "replicated-to-all-participants" as const,
  };
}

function materializeNode(dependency: string) {
  return {
    nodeId: "materialize-output",
    kind: "materialize" as const,
    dependsOn: [dependency],
    resourceId: "output",
    mode: "host-readback-after-graph-success" as const,
  };
}

function input(
  rank: number,
  bytes: Uint8Array,
): SemanticHostGraphWebGpuInputBinding {
  return { rank: wire(rank), resourceId: "input", bytes };
}

function patternedF32Bytes(length: number, modulus: number): Uint8Array {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) {
    values[index] =
      (((index * 17) % modulus) - Math.floor(modulus / 2)) / modulus;
  }
  const bytes = new Uint8Array(values.byteLength);
  bytes.set(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
  return bytes;
}

async function runGraph(
  preparedPipeline: PreparedSemanticHostGraphWebGpuPipeline,
  inputs: readonly SemanticHostGraphWebGpuInputBinding[],
): Promise<readonly SemanticHostGraphWebGpuOutputBinding[]> {
  const result = await runSemanticHostGraphWebGpuPipeline(
    preparedPipeline,
    { inputs },
  );
  return result.outputs;
}

function requireOutputEquality(
  actual: readonly SemanticHostGraphWebGpuOutputBinding[],
  expected: readonly SemanticHostGraphWebGpuOutputBinding[],
  implementationId: string,
): void {
  if (actual.length !== expected.length) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-GRAPH-PERFORMANCE-CORRECTNESS",
      `${implementationId} returned ${actual.length} outputs; expected ${expected.length}`,
    );
  }
  for (const [index, output] of actual.entries()) {
    const expectedOutput = expected[index];
    if (
      expectedOutput === undefined ||
      output.rank !== expectedOutput.rank ||
      output.resourceId !== expectedOutput.resourceId ||
      output.bytes.length !== expectedOutput.bytes.length
    ) {
      throw new PerformanceEvidenceError(
        "BG-WEBGPU-GRAPH-PERFORMANCE-CORRECTNESS",
        `${implementationId} output ${index} identity/length differs`,
      );
    }
    for (let byteIndex = 0; byteIndex < output.bytes.length; byteIndex += 1) {
      if (output.bytes[byteIndex] !== expectedOutput.bytes[byteIndex]) {
        throw new PerformanceEvidenceError(
          "BG-WEBGPU-GRAPH-PERFORMANCE-CORRECTNESS",
          `${implementationId} output ${index} differs at byte ${byteIndex}`,
        );
      }
    }
  }
}

async function measure(
  device: GPUDevice,
  operation: () => Promise<unknown>,
): Promise<number> {
  await withPerformanceTimeout(
    device.queue.onSubmittedWorkDone(),
    10_000,
    "pre-sample-queue-drain",
  );
  const startedAt = performance.now();
  await operation();
  await withPerformanceTimeout(
    device.queue.onSubmittedWorkDone(),
    10_000,
    "post-sample-queue-drain",
  );
  const elapsedMs = performance.now() - startedAt;
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-GRAPH-PERFORMANCE-CLOCK",
      `performance.now produced invalid elapsed time ${elapsedMs}`,
    );
  }
  return elapsedMs;
}

function statistics(samples: readonly number[]): TimingStatistics {
  if (samples.length !== MEASURED_ITERATIONS) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-GRAPH-PERFORMANCE-SAMPLE-COUNT",
      `expected ${MEASURED_ITERATIONS} samples, received ${samples.length}`,
    );
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const meanMs = sorted.reduce(
    (total, sample) => total + sample,
    0,
  ) / sorted.length;
  return Object.freeze({
    samplesMs: Object.freeze([...samples]),
    minimumMs: sorted[0]!,
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    maximumMs: sorted[sorted.length - 1]!,
    meanMs,
  });
}

function percentile(
  sorted: readonly number[],
  proportion: number,
): number {
  return sorted[Math.floor((sorted.length - 1) * proportion)]!;
}

function emitTerminal(input: Readonly<{
  required: boolean;
  artifactHash: string;
  environment: JsonObject;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  stage: string;
  uncapturedErrors: readonly string[];
  observation?: PerformanceObservation;
  error?: JsonObject;
}>): void {
  TERMINAL_EMITTER.emit(Object.freeze({
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
      ...(input.deviceProfileHash === undefined
        ? {}
        : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    }),
    environment: input.environment,
    artifactHashKind: input.observation === undefined
      ? "planned-benchmark"
      : "prepared-benchmark",
    correctnessEvidence: Object.freeze({
      suiteId: CORRECTNESS_SUITE_ID,
      policyId: COMPARISON_POLICY_ID,
      relation: "separate-required-lane-plus-untimed-benchmark-preflight",
    }),
    stage: input.stage,
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.observation === undefined
      ? {}
      : { observation: input.observation }),
    ...(input.error === undefined ? {} : { error: input.error }),
  }));
}

function diagnosticCode(error: unknown, stage: string): string {
  if (error instanceof SemanticHostGraphWebGpuError) return error.code;
  if (error instanceof PerformanceEvidenceError) return error.code;
  if (stage === "fixture-construction") {
    return "BG-WEBGPU-GRAPH-PERFORMANCE-FIXTURE";
  }
  if (stage === "device-acquisition") {
    return "BG-WEBGPU-GRAPH-PERFORMANCE-DEVICE-UNAVAILABLE";
  }
  if (stage === "kernel-device-construction") {
    return "BG-WEBGPU-GRAPH-PERFORMANCE-DEVICE-WRAP";
  }
  return "BG-WEBGPU-GRAPH-PERFORMANCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof SemanticHostGraphWebGpuError) {
    return {
      name: error.name,
      message: error.message,
      code: error.code,
      path: error.path,
    };
  }
  if (error instanceof PerformanceEvidenceError) {
    return {
      name: error.name,
      message: error.message,
      code: error.code,
    };
  }
  if (error instanceof Error) {
    return { name: error.name, message: error.message };
  }
  return { name: "UnknownError", message: String(error) };
}

function withPerformanceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  label: string,
): Promise<T> {
  return withWebGpuEvidenceTimeout(
    promise,
    timeoutMs,
    label,
    (message) => new PerformanceEvidenceError(
      "BG-WEBGPU-GRAPH-PERFORMANCE-TIMEOUT",
      message,
    ),
  );
}

class PerformanceEvidenceError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "PerformanceEvidenceError";
  }
}
