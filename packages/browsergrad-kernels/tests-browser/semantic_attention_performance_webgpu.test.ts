import { it } from "vitest";
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
  createVerifiedDenseAttentionForwardArtifacts,
  prepareAttentionForwardCpu,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  createVerifiedAttentionOnlineKvTileSchedule,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
import {
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireU64,
  sha256Hex,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import { rowWiseOnlineAttentionDirect } from "../src/kernels/flash_attention";
import {
  materializeFloat32,
  releaseDirectBuffer,
  uploadFloat32,
} from "../src/runner";
import {
  SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION,
  SemanticAttentionWebGpuError,
  prepareSemanticAttentionWgsl,
  runSemanticAttentionWebGpu,
  type PreparedSemanticAttentionWgsl,
} from "../src/semantic_attention";
import type { KernelDevice } from "../src/types";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const wire = (value: number) => parseWireU64(String(value));
const EVIDENCE_PREFIX = "[browsergrad-semantic-attention-performance-evidence]";
const SUITE_ID = "browsergrad.kernels.semantic-attention.webgpu-performance@1";
const CORRECTNESS_SUITE_ID = "browsergrad.kernels.semantic-attention.webgpu-conformance@1";
const CAPABILITY_ID = "browsergrad.kernel.attention-forward.block-tiled-online-kv";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID = "browsergrad.attention-forward.f32-abs-relative@1";
const PERFORMANCE_RECORD_SCHEMA = "browsergrad.performance-observation@1";
const CANDIDATE_ID = "block-tiled-kv-online-softmax-forward";
const BASELINE_ID = "row-wise-online-softmax-baseline";
const WARMUP_ITERATIONS = 16;
const MEASURED_ITERATIONS = 20;
const WORKLOAD = Object.freeze({
  batch: 1,
  heads: 2,
  queryLength: 256,
  keyLength: 256,
  queryDepth: 32,
  valueDepth: 32,
  mask: "none",
  queryRows: 8,
  keyRows: 16,
});
const MEASUREMENT_METHOD = Object.freeze({
  boundary: "production-host-api-end-to-end",
  clock: "performance.now",
  completion: "complete-output-readback-plus-queue-drain",
  warmupIterations: WARMUP_ITERATIONS,
  measuredIterationsPerImplementation: MEASURED_ITERATIONS,
  ordering: "paired-alternating-first-implementation",
  candidateLifecycle:
    "finite-input-snapshot-validation-upload-cached-pipeline-dispatch-readback-output-validation-cleanup",
  baselineLifecycle:
    "fresh-input-upload-cached-pipeline-dispatch-readback-output-pool-release-cleanup",
  comparisonClaim: "observational-only-no-superiority-or-regression-threshold",
});
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.semantic-attention": SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, {
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});

interface BenchmarkFixture {
  readonly prepared: PreparedSemanticAttentionWgsl;
  readonly cpu: Awaited<ReturnType<typeof prepareAttentionForwardCpu>>;
  readonly expected: Uint8Array;
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly inputs: Readonly<{
    query: Uint8Array;
    key: Uint8Array;
    value: Uint8Array;
  }>;
  readonly scale: number;
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
  readonly comparisonClaim: "observational-only-no-superiority-or-regression-threshold";
}

it("records separate named semantic attention performance evidence", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage = "suite-manifest";
  let artifactHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    workload: WORKLOAD as unknown as JsonObject,
    method: MEASUREMENT_METHOD as unknown as JsonObject,
  });
  let environment = createWebGpuExecutionEnvironmentRecord({ acquisition: "not-attempted" });
  let environmentId = await hashNamedComponents({ environment });
  let deviceProfileHash: string | undefined;
  let observation: PerformanceObservation | undefined;
  let terminalEmitted = false;
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let kernelDevice: KernelDevice | undefined;
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
        diagnosticCodes: ["BG-WEBGPU-ATTENTION-PERFORMANCE-DEVICE-UNAVAILABLE"],
        stage,
        uncapturedErrors,
        error: { name: "WebGpuEvidenceUnavailable", message: acquisition.reason },
      });
      terminalEmitted = true;
      if (required) throw requiredEvidenceFailure(acquisition.reason);
      context.skip(acquisition.reason);
      return;
    }

    const acquired = acquisition.value;
    device = acquired.device;
    const adapterFeatures = Object.freeze([...acquired.adapter.features].map(String).sort());
    const deviceFeatures = Object.freeze([...device.features].map(String).sort());
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

    stage = "separate-correctness-preflight";
    const candidateOutput = await runCandidate(kernelDevice, fixture);
    const baselineOutput = await runBaseline(kernelDevice, fixture);
    requireCpuComparison(fixture, candidateOutput, CANDIDATE_ID);
    requireCpuComparison(fixture, baselineOutput, BASELINE_ID);

    stage = "warmup";
    for (let index = 0; index < WARMUP_ITERATIONS; index += 1) {
      await runCandidate(kernelDevice, fixture);
      await runBaseline(kernelDevice, fixture);
    }

    stage = "measurement";
    const candidateSamples: number[] = [];
    const baselineSamples: number[] = [];
    for (let index = 0; index < MEASURED_ITERATIONS; index += 1) {
      const candidateFirst = index % 2 === 0;
      if (candidateFirst) {
        candidateSamples.push(await measure(device, () => runCandidate(kernelDevice!, fixture)));
        baselineSamples.push(await measure(device, () => runBaseline(kernelDevice!, fixture)));
      } else {
        baselineSamples.push(await measure(device, () => runBaseline(kernelDevice!, fixture)));
        candidateSamples.push(await measure(device, () => runCandidate(kernelDevice!, fixture)));
      }
    }
    const candidate = statistics(candidateSamples);
    const baseline = statistics(baselineSamples);
    observation = Object.freeze({
      schema: PERFORMANCE_RECORD_SCHEMA,
      workload: WORKLOAD as unknown as JsonObject,
      method: MEASUREMENT_METHOD as unknown as JsonObject,
      candidate: Object.freeze({ implementationId: CANDIDATE_ID, ...candidate }),
      baseline: Object.freeze({ implementationId: BASELINE_ID, ...baseline }),
      candidateMedianOverBaselineMedian: candidate.medianMs / baseline.medianMs,
      comparisonClaim: "observational-only-no-superiority-or-regression-threshold",
    });

    stage = "late-error-drain";
    await withPerformanceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "final-queue-drain");
    await nextWebGpuEvidenceTask();
    if (uncapturedErrors.length > 0) {
      throw new PerformanceEvidenceError(
        "BG-WEBGPU-ATTENTION-PERFORMANCE-UNCAUGHT-GPU-ERROR",
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
        ...(deviceProfileHash === undefined ? {} : { deviceProfileHash }),
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
    kernelDevice?.clearCache();
    device?.destroy();
  }
});

async function prepareFixture(): Promise<BenchmarkFixture> {
  const semantics = await createVerifiedDenseAttentionForwardArtifacts({
    batch: wire(WORKLOAD.batch),
    heads: wire(WORKLOAD.heads),
    queryLength: wire(WORKLOAD.queryLength),
    keyLength: wire(WORKLOAD.keyLength),
    queryDepth: wire(WORKLOAD.queryDepth),
    valueDepth: wire(WORKLOAD.valueDepth),
    causal: false,
  });
  const schedule = await createVerifiedAttentionOnlineKvTileSchedule(semantics.kernel, {
    physicalTile: {
      queryRows: wire(WORKLOAD.queryRows),
      keyRows: wire(WORKLOAD.keyRows),
    },
  });
  const prepared = await prepareSemanticAttentionWgsl(
    semantics.layout,
    semantics.kernel,
    schedule.artifact,
    { operationId: semantics.operationId },
  );
  const query = patternedValues(
    WORKLOAD.batch * WORKLOAD.heads * WORKLOAD.queryLength * WORKLOAD.queryDepth,
    37,
  );
  const key = patternedValues(
    WORKLOAD.batch * WORKLOAD.heads * WORKLOAD.keyLength * WORKLOAD.queryDepth,
    41,
  );
  const value = patternedValues(
    WORKLOAD.batch * WORKLOAD.heads * WORKLOAD.keyLength * WORKLOAD.valueDepth,
    43,
  );
  const inputs = Object.freeze({
    query: bytesOf(query),
    key: bytesOf(key),
    value: bytesOf(value),
  });
  const expected = new Uint8Array(
    WORKLOAD.batch
      * WORKLOAD.heads
      * WORKLOAD.queryLength
      * WORKLOAD.valueDepth
      * Float32Array.BYTES_PER_ELEMENT,
  );
  const cpu = await prepareAttentionForwardCpu(
    semantics.layout,
    semantics.kernel,
    { operationId: semantics.operationId },
  );
  await cpu.execute({ ...inputs, destination: expected });
  const [layoutHash, kernelHash, queryHash, keyHash, valueHash] = await Promise.all([
    hashSemanticArtifact(semantics.layout),
    hashSemanticArtifact(semantics.kernel),
    sha256Hex(inputs.query),
    sha256Hex(inputs.key),
    sha256Hex(inputs.value),
  ]);
  const artifactHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    workload: WORKLOAD as unknown as JsonObject,
    method: MEASUREMENT_METHOD as unknown as JsonObject,
    layoutHash,
    kernelHash,
    scheduleHash: prepared.scheduled.scheduleSemanticHash,
    backendPreparationHash: prepared.backendPreparationHash,
    wgslModuleHash: prepared.wgslModuleHash,
    inputs: { queryHash, keyHash, valueHash },
  });
  return Object.freeze({
    prepared,
    cpu,
    expected,
    query,
    key,
    value,
    inputs,
    scale: decodeF32(prepared.semantic.operation.scale.value.bits),
    artifactHash,
  });
}

async function runCandidate(
  device: KernelDevice,
  fixture: BenchmarkFixture,
): Promise<Uint8Array> {
  const result = await runSemanticAttentionWebGpu(device, fixture.prepared, fixture.inputs);
  return result.destination;
}

async function runBaseline(
  device: KernelDevice,
  fixture: BenchmarkFixture,
): Promise<Uint8Array> {
  const query = uploadFloat32(device, fixture.query);
  const key = uploadFloat32(device, fixture.key);
  const value = uploadFloat32(device, fixture.value);
  let result: ReturnType<typeof rowWiseOnlineAttentionDirect> | undefined;
  try {
    result = rowWiseOnlineAttentionDirect(
      device,
      query,
      key,
      value,
      null,
      {
        B: WORKLOAD.batch,
        H: WORKLOAD.heads,
        Sq: WORKLOAD.queryLength,
        Sk: WORKLOAD.keyLength,
        D: WORKLOAD.queryDepth,
      },
      fixture.scale,
      { enabled: false },
    );
    const output = await materializeFloat32(device, result.buffer, result.byteLength);
    return bytesOf(output);
  } finally {
    if (result !== undefined) releaseDirectBuffer(device, result.buffer, result.byteLength);
    query.destroy();
    key.destroy();
    value.destroy();
  }
}

function requireCpuComparison(
  fixture: BenchmarkFixture,
  actual: Uint8Array,
  implementationId: string,
): void {
  const comparison = fixture.cpu.compare(actual, fixture.expected);
  if (!comparison.passed) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-ATTENTION-PERFORMANCE-CORRECTNESS",
      `${implementationId} differs from the CPU oracle at ${comparison.firstMismatchIndex}`,
    );
  }
}

async function measure(
  device: GPUDevice,
  operation: () => Promise<Uint8Array>,
): Promise<number> {
  await withPerformanceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "pre-sample-queue-drain");
  const startedAt = performance.now();
  await operation();
  await withPerformanceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "post-sample-queue-drain");
  const elapsedMs = performance.now() - startedAt;
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-ATTENTION-PERFORMANCE-CLOCK",
      `performance.now produced invalid elapsed time ${elapsedMs}`,
    );
  }
  return elapsedMs;
}

function statistics(samples: readonly number[]): TimingStatistics {
  if (samples.length !== MEASURED_ITERATIONS) {
    throw new PerformanceEvidenceError(
      "BG-WEBGPU-ATTENTION-PERFORMANCE-SAMPLE-COUNT",
      `expected ${MEASURED_ITERATIONS} samples, received ${samples.length}`,
    );
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const meanMs = sorted.reduce((total, sample) => total + sample, 0) / sorted.length;
  return Object.freeze({
    samplesMs: Object.freeze([...samples]),
    minimumMs: sorted[0]!,
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    maximumMs: sorted[sorted.length - 1]!,
    meanMs,
  });
}

function percentile(sorted: readonly number[], proportion: number): number {
  return sorted[Math.floor((sorted.length - 1) * proportion)]!;
}

function patternedValues(length: number, modulus: number): Float32Array {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = (((index * 17) % modulus) - Math.floor(modulus / 2)) / modulus;
  }
  return values;
}

function bytesOf(values: Float32Array): Uint8Array {
  const bytes = new Uint8Array(values.byteLength);
  bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  return bytes;
}

function decodeF32(bits: string): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setUint32(0, Number.parseInt(bits, 16), false);
  return view.getFloat32(0, false);
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
    artifactHashKind: input.observation === undefined ? "planned-benchmark" : "prepared-benchmark",
    correctnessEvidence: Object.freeze({
      suiteId: CORRECTNESS_SUITE_ID,
      policyId: COMPARISON_POLICY_ID,
      relation: "separate-required-lane-plus-untimed-benchmark-preflight",
    }),
    stage: input.stage,
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.observation === undefined ? {} : { observation: input.observation }),
    ...(input.error === undefined ? {} : { error: input.error }),
  }));
}

function diagnosticCode(error: unknown, stage: string): string {
  if (error instanceof SemanticAttentionWebGpuError) return error.code;
  if (error instanceof PerformanceEvidenceError) return error.code;
  if (stage === "fixture-construction") return "BG-WEBGPU-ATTENTION-PERFORMANCE-FIXTURE";
  if (stage === "device-acquisition") {
    return "BG-WEBGPU-ATTENTION-PERFORMANCE-DEVICE-UNAVAILABLE";
  }
  if (stage === "kernel-device-construction") {
    return "BG-WEBGPU-ATTENTION-PERFORMANCE-DEVICE-WRAP";
  }
  return "BG-WEBGPU-ATTENTION-PERFORMANCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof SemanticAttentionWebGpuError) {
    return { name: error.name, message: error.message, code: error.code, path: error.path };
  }
  if (error instanceof PerformanceEvidenceError) {
    return { name: error.name, message: error.message, code: error.code };
  }
  if (error instanceof Error) return { name: error.name, message: error.message };
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
      "BG-WEBGPU-ATTENTION-PERFORMANCE-TIMEOUT",
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
