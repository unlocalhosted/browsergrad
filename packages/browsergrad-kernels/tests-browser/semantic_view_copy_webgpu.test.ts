import { it } from "vitest";
import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createTerminalEvidenceEmitter,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";

import {
  layoutArtifactPayload,
  verifyLayoutArtifact,
  type DimExpr,
  type IndexExpr,
  type PredicateExpr,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  kernelArtifactPayload,
  prepareViewCopyCpu,
  verifyKernelArtifact,
  type InvalidSourcePolicy,
  type VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireI64,
  wireIntegerToBigInt,
  type JsonObject,
  type WireI64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
  SemanticViewCopyWebGpuError,
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
  type PreparedSemanticViewCopyWgsl,
  type SemanticViewCopyWebGpuTrace,
} from "../src/semantic_view_copy";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const TRUE: PredicateExpr = { kind: "bool", value: true };
const EVIDENCE_PREFIX = "[browsergrad-webgpu-evidence]";
const EVIDENCE_SCHEMA = EXECUTION_EVIDENCE_SCHEMA;
const ENVIRONMENT_SCHEMA = EXECUTION_ENVIRONMENT_SCHEMA;
const SUITE_ID = "browsergrad.kernels.view-copy.webgpu-conformance@2";
const CAPABILITY_ID = "browsergrad.kernel.view-copy";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID =
  "browsergrad.comparison.bit-exact-complete-destination-words.v2";
const PLANNED_CASE_IDS = Object.freeze([
  "rank2-transpose",
  "rank3-permutation",
  "rank1-positive-stride",
  "rank4-permutation",
  "rank5-permutation",
  "rank6-permutation",
  "rank7-permutation",
  "rank1-u32-negative-stride",
  "rank2-negative-stride",
  "rank3-u32-negative-stride",
  "rank4-negative-stride",
  "rank5-i32-negative-stride",
  "rank6-u32-negative-stride",
  "rank7-u32-negative-stride",
  "rank2-negative-predicate-padding",
  "positive-strided-slice",
  "read-only-broadcast",
  "byte-map-nonzero-offsets",
  "rank2-padding-exact-nan",
  "rank3-padding-exact-nan",
  "dynamic-rank2-specialization",
  "i32-rank2-transpose",
  "u32-read-only-broadcast",
  "bool-rank2-odd-transpose",
  "i8-rank2-transpose",
  "i8-rank1-positive-stride",
  "i8-rank1-negative-stride",
  "i8-rank2-negative-stride",
  "i8-rank4-permutation",
  "i8-rank4-negative-stride",
  "u8-read-only-broadcast",
  "i16-rank2-transpose",
  "u16-read-only-broadcast",
  "f16-rank2-odd-transpose",
  "f16-rank1-positive-stride",
  "f16-rank1-negative-stride",
  "f16-rank2-negative-stride",
  "f16-rank4-permutation",
  "f16-rank4-negative-stride",
  "bf16-rank3-permutation",
  "f64-rank2-transpose",
  "f64-rank1-positive-stride",
  "f64-rank1-negative-stride",
  "f64-rank2-negative-stride",
  "f64-rank4-permutation",
  "f64-rank4-negative-stride",
  "i64-byte-map-broadcast",
  "u64-rank3-permutation",
  "zero-extent-no-submit",
]);
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.view-copy": SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
});
const TERMINAL_EXPECTATION = Object.freeze({
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, TERMINAL_EXPECTATION);

interface LayoutInput {
  readonly shape: readonly DimExpr[];
  readonly sourceLocation: IndexExpr;
  readonly sourcePredicate?: PredicateExpr;
  readonly destinationLocation?: IndexExpr;
  readonly sourceLocationUnit?: "element" | "byte";
  readonly destinationLocationUnit?: "element" | "byte";
  readonly sourceByteOffset?: DimExpr;
  readonly destinationByteOffset?: DimExpr;
  readonly sourceBytes: DimExpr;
  readonly destinationBytes: DimExpr;
  readonly symbols?: readonly { readonly id: string; readonly domain: { readonly min: string; readonly max: string } }[];
  readonly dtype?:
    | "bool"
    | "i8"
    | "u8"
    | "i16"
    | "u16"
    | "f16"
    | "bf16"
    | "f32"
    | "i32"
    | "u32"
    | "f64"
    | "i64"
    | "u64";
}

interface EvidenceCase {
  readonly id: string;
  readonly layout: VerifiedLayoutArtifact;
  readonly kernel: VerifiedKernelArtifact;
  readonly sourceWords: Uint32Array;
  readonly initialDestinationWords: Uint32Array;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly expectedSubmitted: boolean;
}

interface PreparedEvidenceCase {
  readonly evidenceCase: EvidenceCase;
  readonly preparedWgsl: PreparedSemanticViewCopyWgsl;
  readonly preparedCpu: ReturnType<typeof prepareViewCopyCpu> extends Promise<infer T> ? T : never;
  readonly inputHash: string;
  readonly artifactHash: string;
}

interface ExecutionEvidence {
  readonly capabilityId: string;
  readonly artifactHash: string;
  readonly backendId: string;
  readonly environmentId: string;
  readonly producerVersions: Readonly<Record<string, string>>;
  readonly deviceProfileHash?: string;
  readonly recordedAt: string;
  readonly outcome: "not-run" | "passed" | "failed";
  readonly comparisonPolicyId?: string;
  readonly diagnosticCodes: readonly string[];
}

interface EvidenceEnvironment extends JsonObject {
  readonly schema: typeof ENVIRONMENT_SCHEMA;
  readonly acquisition: string;
  readonly userAgent: string;
  readonly platform: string;
  readonly adapter?: JsonObject;
  readonly adapterSupportedFeatures?: readonly string[];
  readonly negotiatedDeviceFeatures?: readonly string[];
  readonly negotiatedDeviceLimits?: JsonObject;
  readonly unavailableReason?: string;
}

interface CaseObservation extends JsonObject {
  readonly caseId: string;
  readonly artifactHash: string;
  readonly inputHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly semanticSpecializationHash: string;
  readonly backendSpecializationHash: string;
  readonly wgslModuleHash: string;
  readonly workgroupSize: number;
  readonly logicalInvocationCount: readonly number[];
  readonly submittedWorkgroupCount: readonly number[];
  readonly pipelineCount: number;
  readonly submitted: boolean;
  readonly comparisonPolicyId: typeof COMPARISON_POLICY_ID;
}

interface TerminalEvidenceRecord extends JsonObject {
  readonly schema: typeof EVIDENCE_SCHEMA;
  readonly kind: "terminal";
  readonly suiteId: typeof SUITE_ID;
  readonly required: boolean;
  readonly evidence: ExecutionEvidence & JsonObject;
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

it("executes the Gate 2 view-copy matrix on a required real GPUDevice", async (context) => {
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
  let preparedCases: readonly PreparedEvidenceCase[] = [];
  const completedCases: CaseObservation[] = [];
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let kernelDevice: Awaited<ReturnType<typeof createDevice>> | undefined;
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => {
    uncapturedErrors.push(event.error.message);
  };
  try {
    stage = "fixture-construction";
    const evidenceCases = await createEvidenceCases();
    assertPlannedCases(evidenceCases);
    stage = "semantic-and-wgsl-preparation";
    preparedCases = await Promise.all(evidenceCases.map(prepareEvidenceCase));
    artifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifacts: preparedCases.map(({ evidenceCase, artifactHash: caseArtifactHash }) => ({
        caseId: evidenceCase.id,
        artifactHash: caseArtifactHash,
      })),
    });
    artifactHashKind = "prepared-suite";
    caseSetHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifactHash,
      cases: preparedCases.map(({ evidenceCase, inputHash, artifactHash: caseArtifactHash }) => ({
        caseId: evidenceCase.id,
        artifactHash: caseArtifactHash,
        inputHash,
        expectedSubmitted: evidenceCase.expectedSubmitted,
      })),
      comparisonPolicyId: COMPARISON_POLICY_ID,
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
      negotiatedDeviceFeatures,
      negotiatedDeviceLimits,
    });
    stage = "kernel-device-construction";
    kernelDevice = await createDevice({ device });
    device.addEventListener("uncapturederror", uncapturedHandler);

    for (const preparedCase of preparedCases) {
      const { evidenceCase, preparedWgsl, preparedCpu } = preparedCase;
      stage = "case-execution";
      currentCaseId = evidenceCase.id;
      const cpuDestination = cloneWords(evidenceCase.initialDestinationWords);
      const cpuTrace = preparedCpu.execute({
        source: wordsAsBytes(cloneWords(evidenceCase.sourceWords)),
        destination: wordsAsBytes(cpuDestination),
      });
      const gpuResult = await runSemanticViewCopyWebGpu(kernelDevice, preparedWgsl, {
        sourceWords: cloneWords(evidenceCase.sourceWords),
        destinationWords: cloneWords(evidenceCase.initialDestinationWords),
      });
      await withEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "queue-drain");
      assertCaseResult(evidenceCase, preparedWgsl, cpuTrace.specializationHash, cpuDestination, gpuResult);
      completedCases.push(caseObservation(preparedCase, gpuResult.trace));
    }
    stage = "late-error-drain";
    currentCaseId = undefined;
    await withEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "final-queue-drain");
    await nextTask();
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

async function createEvidenceCases(): Promise<readonly EvidenceCase[]> {
  const transpose = await makeCase(
    "rank2-transpose",
    {
      shape: dims("2", "3"),
      sourceLocation: add(multiply(coordinate(1), indexConstant("2")), coordinate(0)),
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("24"),
    },
    { kind: "reject" },
    words(0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000),
  );

  const rank3Permutation = await makeCase(
    "rank3-permutation",
    {
      shape: dims("2", "3", "4"),
      sourceLocation: add(
        multiply(coordinate(2), indexConstant("6")),
        multiply(coordinate(0), indexConstant("3")),
        coordinate(1),
      ),
      sourceBytes: dimConstant("96"),
      destinationBytes: dimConstant("96"),
    },
    { kind: "reject" },
    sequenceWords(24, 0x3f000000),
  );

  const rank1PositiveStride = await makeCase(
    "rank1-positive-stride",
    {
      shape: dims("4"),
      sourceLocation: multiply(coordinate(0), indexConstant("2")),
      sourceBytes: dimConstant("28"),
      destinationBytes: dimConstant("16"),
    },
    { kind: "reject" },
    sequenceWords(7, 0x40000000),
  );

  const rank4Permutation = await makeCase(
    "rank4-permutation",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
      ),
      sourceBytes: dimConstant("64"),
      destinationBytes: dimConstant("64"),
    },
    { kind: "reject" },
    sequenceWords(16, 0x41000000),
  );

  const rank5Permutation = await makeCase(
    "rank5-permutation",
    {
      shape: dims("2", "2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
        multiply(coordinate(4), indexConstant("16")),
      ),
      sourceBytes: dimConstant("128"),
      destinationBytes: dimConstant("128"),
    },
    { kind: "reject" },
    sequenceWords(32, 0x41800000),
  );

  const rank6Permutation = await makeCase(
    "rank6-permutation",
    {
      shape: dims("2", "2", "2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
        multiply(coordinate(4), indexConstant("16")),
        multiply(coordinate(5), indexConstant("32")),
      ),
      sourceBytes: dimConstant("256"),
      destinationBytes: dimConstant("256"),
    },
    { kind: "reject" },
    sequenceWords(64, 0x42000000),
  );

  const rank7Permutation = await makeCase(
    "rank7-permutation",
    {
      shape: dims("2", "2", "2", "2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
        multiply(coordinate(4), indexConstant("16")),
        multiply(coordinate(5), indexConstant("32")),
        multiply(coordinate(6), indexConstant("64")),
      ),
      sourceBytes: dimConstant("512"),
      destinationBytes: dimConstant("512"),
    },
    { kind: "reject" },
    sequenceWords(128, 0x42800000),
  );

  const rank1U32NegativeStride = await makeCase(
    "rank1-u32-negative-stride",
    {
      shape: dims("4"),
      sourceLocation: multiply(coordinate(0), indexConstant("-1")),
      sourceByteOffset: dimConstant("12"),
      sourceBytes: dimConstant("16"),
      destinationBytes: dimConstant("16"),
      dtype: "u32",
    },
    { kind: "reject" },
    sequenceWords(4, 0x80000000),
  );

  const rank2NegativeStride = await makeCase(
    "rank2-negative-stride",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-3")),
        multiply(coordinate(1), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("20"),
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("24"),
    },
    { kind: "reject" },
    sequenceWords(6, 0x41c00000),
  );

  const rank3U32NegativeStride = await makeCase(
    "rank3-u32-negative-stride",
    {
      shape: dims("2", "2", "3"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-6")),
        multiply(coordinate(1), indexConstant("-3")),
        multiply(coordinate(2), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("44"),
      sourceBytes: dimConstant("48"),
      destinationBytes: dimConstant("48"),
      dtype: "u32",
    },
    { kind: "reject" },
    sequenceWords(12, 0x80000000),
  );

  const rank4NegativeStride = await makeCase(
    "rank4-negative-stride",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-8")),
        multiply(coordinate(1), indexConstant("-4")),
        multiply(coordinate(2), indexConstant("-2")),
        multiply(coordinate(3), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("60"),
      sourceBytes: dimConstant("64"),
      destinationBytes: dimConstant("64"),
    },
    { kind: "reject" },
    sequenceWords(16, 0x43000000),
  );

  const rank5I32NegativeStride = await makeCase(
    "rank5-i32-negative-stride",
    {
      shape: dims("2", "2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-16")),
        multiply(coordinate(1), indexConstant("-8")),
        multiply(coordinate(2), indexConstant("-4")),
        multiply(coordinate(3), indexConstant("-2")),
        multiply(coordinate(4), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("124"),
      sourceBytes: dimConstant("128"),
      destinationBytes: dimConstant("128"),
      dtype: "i32",
    },
    { kind: "reject" },
    sequenceWords(32, 0x80000000),
  );

  const rank6U32NegativeStride = await makeCase(
    "rank6-u32-negative-stride",
    {
      shape: dims("2", "2", "2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-32")),
        multiply(coordinate(1), indexConstant("-16")),
        multiply(coordinate(2), indexConstant("-8")),
        multiply(coordinate(3), indexConstant("-4")),
        multiply(coordinate(4), indexConstant("-2")),
        multiply(coordinate(5), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("252"),
      sourceBytes: dimConstant("256"),
      destinationBytes: dimConstant("256"),
      dtype: "u32",
    },
    { kind: "reject" },
    sequenceWords(64, 0x80000000),
  );

  const rank7U32NegativeStride = await makeCase(
    "rank7-u32-negative-stride",
    {
      shape: dims("2", "2", "2", "2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-64")),
        multiply(coordinate(1), indexConstant("-32")),
        multiply(coordinate(2), indexConstant("-16")),
        multiply(coordinate(3), indexConstant("-8")),
        multiply(coordinate(4), indexConstant("-4")),
        multiply(coordinate(5), indexConstant("-2")),
        multiply(coordinate(6), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("508"),
      sourceBytes: dimConstant("512"),
      destinationBytes: dimConstant("512"),
      dtype: "u32",
    },
    { kind: "reject" },
    sequenceWords(128, 0x80000000),
  );

  const rank2NegativePredicatePadding = await makeCase(
    "rank2-negative-predicate-padding",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("3")),
        coordinate(1),
      ),
      sourcePredicate: {
        kind: "lessEqual",
        lhs: indexConstant("-1"),
        rhs: multiply(coordinate(1), indexConstant("-1")),
      },
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("24"),
    },
    exactNanFill(),
    sequenceWords(6, 0x42000000),
  );

  const sliceShape = dims("2", "2");
  const stridedSlice = await makeCase(
    "positive-strided-slice",
    {
      shape: sliceShape,
      sourceLocation: multiply(rowMajor(sliceShape), indexConstant("2")),
      sourceBytes: dimConstant("28"),
      destinationBytes: dimConstant("16"),
    },
    { kind: "reject" },
    sequenceWords(7, 0x41000000),
  );

  const broadcastShape = dims("2", "2");
  const broadcast = await makeCase(
    "read-only-broadcast",
    {
      shape: broadcastShape,
      sourceLocation: coordinate(1),
      sourceBytes: dimConstant("8"),
      destinationBytes: dimConstant("16"),
    },
    { kind: "reject" },
    words(0x3f800000, 0x40000000),
  );

  const byteShape = dims("2", "2");
  const byteOffsets = await makeCase(
    "byte-map-nonzero-offsets",
    {
      shape: byteShape,
      sourceLocation: multiply(rowMajor(byteShape), indexConstant("4")),
      destinationLocation: multiply(rowMajor(byteShape), indexConstant("4")),
      sourceLocationUnit: "byte",
      destinationLocationUnit: "byte",
      sourceByteOffset: dimConstant("4"),
      destinationByteOffset: dimConstant("4"),
      sourceBytes: dimConstant("20"),
      destinationBytes: dimConstant("20"),
    },
    { kind: "reject" },
    words(0xaaaaaaa1, 0x3f800000, 0x40000000, 0x40400000, 0x40800000),
  );

  const padding2Shape = dims("4", "5");
  const padding2Predicate = rectangularPaddingPredicate([1, 1], [2, 3]);
  const padding2 = await makeCase(
    "rank2-padding-exact-nan",
    {
      shape: padding2Shape,
      sourceLocation: add(
        multiply(add(coordinate(0), indexConstant("-1")), indexConstant("3")),
        add(coordinate(1), indexConstant("-1")),
      ),
      sourcePredicate: padding2Predicate,
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("80"),
    },
    exactNanFill(),
    sequenceWords(6, 0x42000000),
  );

  const padding3Shape = dims("4", "4", "4");
  const padding3 = await makeCase(
    "rank3-padding-exact-nan",
    {
      shape: padding3Shape,
      sourceLocation: add(
        multiply(add(coordinate(0), indexConstant("-1")), indexConstant("4")),
        multiply(add(coordinate(1), indexConstant("-1")), indexConstant("2")),
        add(coordinate(2), indexConstant("-1")),
      ),
      sourcePredicate: rectangularPaddingPredicate([1, 1, 1], [2, 2, 2]),
      sourceBytes: dimConstant("32"),
      destinationBytes: dimConstant("256"),
    },
    exactNanFill(),
    sequenceWords(8, 0x43000000),
  );

  const dynamicN: DimExpr = { kind: "symbol", id: "n" };
  const dynamicShape = [dynamicN, dimConstant("2")] as const;
  const dynamicBytes: DimExpr = { kind: "mul", lhs: dynamicN, rhs: dimConstant("8") };
  const dynamic = await makeCase(
    "dynamic-rank2-specialization",
    {
      shape: dynamicShape,
      sourceLocation: add(multiply(coordinate(0), indexConstant("2")), coordinate(1)),
      sourceBytes: dynamicBytes,
      destinationBytes: dynamicBytes,
      symbols: [{ id: "n", domain: { min: "0", max: "8" } }],
    },
    { kind: "reject" },
    words(0x3f800000, 0x40000000, 0x40400000, 0x40800000),
    { bindings: { n: parseWireI64("2") }, destinationWordLength: 4 },
  );

  const i32Transpose = await makeCase(
    "i32-rank2-transpose",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(1), indexConstant("2")),
        coordinate(0),
      ),
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("24"),
      dtype: "i32",
    },
    { kind: "reject" },
    words(0x80000000, 0xffffffff, 0, 1, 0x7fffffff, 0xdeadbeef),
  );

  const u32Broadcast = await makeCase(
    "u32-read-only-broadcast",
    {
      shape: dims("2", "3"),
      sourceLocation: coordinate(1),
      sourceBytes: dimConstant("12"),
      destinationBytes: dimConstant("24"),
      dtype: "u32",
    },
    { kind: "reject" },
    words(0, 0x80000000, 0xffffffff),
  );

  const boolOddTranspose = await makeCase(
    "bool-rank2-odd-transpose",
    {
      shape: dims("3", "3"),
      sourceLocation: add(
        multiply(coordinate(1), indexConstant("3")),
        coordinate(0),
      ),
      sourceBytes: dimConstant("12"),
      destinationBytes: dimConstant("12"),
      dtype: "bool",
    },
    { kind: "reject" },
    packed8Words(
      0x00,
      0x01,
      0x7f,
      0x80,
      0xff,
      0x55,
      0xaa,
      0x02,
      0xfe,
      0xa5,
      0x5a,
      0xc3,
    ),
  );

  const i8Transpose = await makeCase(
    "i8-rank2-transpose",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(1), indexConstant("2")),
        coordinate(0),
      ),
      sourceByteOffset: dimConstant("1"),
      destinationByteOffset: dimConstant("4"),
      sourceBytes: dimConstant("8"),
      destinationBytes: dimConstant("12"),
      dtype: "i8",
    },
    { kind: "reject" },
    packed8Words(0xa5, 0x80, 0xff, 0x00, 0x01, 0x7f, 0x5a, 0xc3),
  );

  const u8Broadcast = await makeCase(
    "u8-read-only-broadcast",
    {
      shape: dims("2", "3"),
      sourceLocation: coordinate(1),
      sourceBytes: dimConstant("4"),
      destinationBytes: dimConstant("8"),
      dtype: "u8",
    },
    { kind: "reject" },
    packed8Words(0x00, 0x80, 0xff, 0xa5),
  );

  const i8Rank1PositiveStride = await makeCase(
    "i8-rank1-positive-stride",
    {
      shape: dims("3"),
      sourceLocation: multiply(coordinate(0), indexConstant("2")),
      sourceBytes: dimConstant("8"),
      destinationBytes: dimConstant("4"),
      dtype: "i8",
    },
    { kind: "reject" },
    packed8Words(0x00, 0x01, 0x7f, 0x80, 0xff, 0x5a, 0xa5, 0xc3),
  );

  const i8Rank1NegativeStride = await makeCase(
    "i8-rank1-negative-stride",
    {
      shape: dims("3"),
      sourceLocation: multiply(coordinate(0), indexConstant("-1")),
      sourceByteOffset: dimConstant("2"),
      sourceBytes: dimConstant("4"),
      destinationBytes: dimConstant("4"),
      dtype: "i8",
    },
    { kind: "reject" },
    packed8Words(0x00, 0x80, 0xff, 0xa5),
  );

  const i8NegativeStride = await makeCase(
    "i8-rank2-negative-stride",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-3")),
        multiply(coordinate(1), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("5"),
      sourceBytes: dimConstant("8"),
      destinationBytes: dimConstant("8"),
      dtype: "i8",
    },
    { kind: "reject" },
    packed8Words(0x00, 0x01, 0x7f, 0x80, 0xff, 0x5a, 0xa5, 0xc3),
  );

  const i8Rank4Permutation = await makeCase(
    "i8-rank4-permutation",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
      ),
      sourceBytes: dimConstant("16"),
      destinationBytes: dimConstant("16"),
      dtype: "i8",
    },
    { kind: "reject" },
    sequenceWords(4, 0x10203040),
  );

  const i8Rank4NegativeStride = await makeCase(
    "i8-rank4-negative-stride",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-8")),
        multiply(coordinate(1), indexConstant("-4")),
        multiply(coordinate(2), indexConstant("-2")),
        multiply(coordinate(3), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("15"),
      sourceBytes: dimConstant("16"),
      destinationBytes: dimConstant("16"),
      dtype: "i8",
    },
    { kind: "reject" },
    sequenceWords(4, 0x50607080),
  );

  const i16Transpose = await makeCase(
    "i16-rank2-transpose",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(1), indexConstant("2")),
        coordinate(0),
      ),
      sourceByteOffset: dimConstant("2"),
      destinationByteOffset: dimConstant("4"),
      sourceBytes: dimConstant("16"),
      destinationBytes: dimConstant("20"),
      dtype: "i16",
    },
    { kind: "reject" },
    packed16Words(
      0xa55a,
      0x8000,
      0xffff,
      0,
      1,
      0x7fff,
      0x1234,
      0x5aa5,
    ),
  );

  const u16Broadcast = await makeCase(
    "u16-read-only-broadcast",
    {
      shape: dims("2", "3"),
      sourceLocation: coordinate(1),
      sourceBytes: dimConstant("8"),
      destinationBytes: dimConstant("12"),
      dtype: "u16",
    },
    { kind: "reject" },
    packed16Words(0, 0x8000, 0xffff, 0xa55a),
  );

  const f16OddTranspose = await makeCase(
    "f16-rank2-odd-transpose",
    {
      shape: dims("3", "3"),
      sourceLocation: add(
        multiply(coordinate(1), indexConstant("3")),
        coordinate(0),
      ),
      sourceBytes: dimConstant("20"),
      destinationBytes: dimConstant("20"),
      dtype: "f16",
    },
    { kind: "reject" },
    packed16Words(
      0x0000,
      0x8000,
      0x3c00,
      0x7c00,
      0xfc00,
      0x7e01,
      0x0001,
      0x03ff,
      0x7bff,
      0xa55a,
    ),
  );

  const bf16Rank3Permutation = await makeCase(
    "bf16-rank3-permutation",
    {
      shape: dims("2", "2", "3"),
      sourceLocation: add(
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(0), indexConstant("2")),
        coordinate(1),
      ),
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("24"),
      dtype: "bf16",
    },
    { kind: "reject" },
    packed16Words(
      0x0000,
      0x8000,
      0x3f80,
      0x7f80,
      0xff80,
      0x7fc1,
      0x0001,
      0x007f,
      0x0080,
      0x7f7f,
      0x1234,
      0xffff,
    ),
  );

  const f16Rank1PositiveStride = await makeCase(
    "f16-rank1-positive-stride",
    {
      shape: dims("3"),
      sourceLocation: multiply(coordinate(0), indexConstant("2")),
      sourceBytes: dimConstant("12"),
      destinationBytes: dimConstant("8"),
      dtype: "f16",
    },
    { kind: "reject" },
    packed16Words(0x0000, 0x8000, 0x3c00, 0x7c00, 0x7e01, 0xffff),
  );

  const f16Rank1NegativeStride = await makeCase(
    "f16-rank1-negative-stride",
    {
      shape: dims("3"),
      sourceLocation: multiply(coordinate(0), indexConstant("-1")),
      sourceByteOffset: dimConstant("4"),
      sourceBytes: dimConstant("8"),
      destinationBytes: dimConstant("8"),
      dtype: "f16",
    },
    { kind: "reject" },
    packed16Words(0x0000, 0x8000, 0x7e01, 0xa55a),
  );

  const f16NegativeStride = await makeCase(
    "f16-rank2-negative-stride",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-3")),
        multiply(coordinate(1), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("10"),
      sourceBytes: dimConstant("12"),
      destinationBytes: dimConstant("12"),
      dtype: "f16",
    },
    { kind: "reject" },
    packed16Words(0x0000, 0x8000, 0x3c00, 0x7c00, 0x7e01, 0xffff),
  );

  const f16Rank4Permutation = await makeCase(
    "f16-rank4-permutation",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
      ),
      sourceBytes: dimConstant("32"),
      destinationBytes: dimConstant("32"),
      dtype: "f16",
    },
    { kind: "reject" },
    sequenceWords(8, 0x11110000),
  );

  const f16Rank4NegativeStride = await makeCase(
    "f16-rank4-negative-stride",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-8")),
        multiply(coordinate(1), indexConstant("-4")),
        multiply(coordinate(2), indexConstant("-2")),
        multiply(coordinate(3), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("30"),
      sourceBytes: dimConstant("32"),
      destinationBytes: dimConstant("32"),
      dtype: "f16",
    },
    { kind: "reject" },
    sequenceWords(8, 0x22220000),
  );

  const f64Transpose = await makeCase(
    "f64-rank2-transpose",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(1), indexConstant("2")),
        coordinate(0),
      ),
      sourceByteOffset: dimConstant("8"),
      destinationByteOffset: dimConstant("8"),
      sourceBytes: dimConstant("56"),
      destinationBytes: dimConstant("56"),
      dtype: "f64",
    },
    { kind: "reject" },
    words(
      0xa5a5a5a5, 0x5a5a5a5a,
      0x00000000, 0x80000000,
      0x00000001, 0x7ff00000,
      0x00000001, 0x7ff80000,
      0xffffffff, 0xffffffff,
      0x89abcdef, 0x01234567,
      0x76543210, 0xfedcba98,
    ),
  );

  const i64ByteMapBroadcast = await makeCase(
    "i64-byte-map-broadcast",
    {
      shape: dims("2", "3"),
      sourceLocation: multiply(coordinate(1), indexConstant("8")),
      sourceLocationUnit: "byte",
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("48"),
      dtype: "i64",
    },
    { kind: "reject" },
    words(
      0x00000000, 0x80000000,
      0xffffffff, 0xffffffff,
      0x00000001, 0x7fffffff,
    ),
  );

  const f64Rank1PositiveStride = await makeCase(
    "f64-rank1-positive-stride",
    {
      shape: dims("3"),
      sourceLocation: multiply(coordinate(0), indexConstant("2")),
      sourceBytes: dimConstant("40"),
      destinationBytes: dimConstant("24"),
      dtype: "f64",
    },
    { kind: "reject" },
    words(
      0x00000000, 0x80000000,
      0x00000001, 0x7ff00000,
      0x00000001, 0x7ff80000,
      0xffffffff, 0xffffffff,
      0x89abcdef, 0x01234567,
    ),
  );

  const f64Rank1NegativeStride = await makeCase(
    "f64-rank1-negative-stride",
    {
      shape: dims("3"),
      sourceLocation: multiply(coordinate(0), indexConstant("-1")),
      sourceByteOffset: dimConstant("16"),
      sourceBytes: dimConstant("24"),
      destinationBytes: dimConstant("24"),
      dtype: "f64",
    },
    { kind: "reject" },
    words(
      0x00000000, 0x80000000,
      0x00000001, 0x7ff00000,
      0x00000001, 0x7ff80000,
    ),
  );

  const f64NegativeStride = await makeCase(
    "f64-rank2-negative-stride",
    {
      shape: dims("2", "3"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-3")),
        multiply(coordinate(1), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("40"),
      sourceBytes: dimConstant("48"),
      destinationBytes: dimConstant("48"),
      dtype: "f64",
    },
    { kind: "reject" },
    words(
      0x00000000, 0x80000000,
      0x00000001, 0x7ff00000,
      0x00000001, 0x7ff80000,
      0xffffffff, 0xffffffff,
      0x89abcdef, 0x01234567,
      0x76543210, 0xfedcba98,
    ),
  );

  const f64Rank4Permutation = await makeCase(
    "f64-rank4-permutation",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), indexConstant("2")),
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(3), indexConstant("8")),
      ),
      sourceBytes: dimConstant("128"),
      destinationBytes: dimConstant("128"),
      dtype: "f64",
    },
    { kind: "reject" },
    sequenceWords(32, 0x70000000),
  );

  const f64Rank4NegativeStride = await makeCase(
    "f64-rank4-negative-stride",
    {
      shape: dims("2", "2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(0), indexConstant("-8")),
        multiply(coordinate(1), indexConstant("-4")),
        multiply(coordinate(2), indexConstant("-2")),
        multiply(coordinate(3), indexConstant("-1")),
      ),
      sourceByteOffset: dimConstant("120"),
      sourceBytes: dimConstant("128"),
      destinationBytes: dimConstant("128"),
      dtype: "f64",
    },
    { kind: "reject" },
    sequenceWords(32, 0x71000000),
  );

  const u64Rank3Permutation = await makeCase(
    "u64-rank3-permutation",
    {
      shape: dims("2", "2", "2"),
      sourceLocation: add(
        multiply(coordinate(2), indexConstant("4")),
        multiply(coordinate(0), indexConstant("2")),
        coordinate(1),
      ),
      sourceBytes: dimConstant("64"),
      destinationBytes: dimConstant("64"),
      dtype: "u64",
    },
    { kind: "reject" },
    words(
      0x00000000, 0x00000000,
      0x00000001, 0x00000000,
      0xffffffff, 0x00000000,
      0x00000000, 0x00000001,
      0xffffffff, 0xffffffff,
      0x89abcdef, 0x01234567,
      0x76543210, 0xfedcba98,
      0xaaaaaaaa, 0x55555555,
    ),
  );

  const zero = await makeCase(
    "zero-extent-no-submit",
    {
      shape: dynamicShape,
      sourceLocation: add(multiply(coordinate(0), indexConstant("2")), coordinate(1)),
      sourceBytes: dynamicBytes,
      destinationBytes: dynamicBytes,
      symbols: [{ id: "n", domain: { min: "0", max: "8" } }],
    },
    { kind: "reject" },
    words(),
    {
      bindings: { n: parseWireI64("0") },
      destinationWordLength: 0,
      expectedSubmitted: false,
    },
  );

  return Object.freeze([
    transpose,
    rank3Permutation,
    rank1PositiveStride,
    rank4Permutation,
    rank5Permutation,
    rank6Permutation,
    rank7Permutation,
    rank1U32NegativeStride,
    rank2NegativeStride,
    rank3U32NegativeStride,
    rank4NegativeStride,
    rank5I32NegativeStride,
    rank6U32NegativeStride,
    rank7U32NegativeStride,
    rank2NegativePredicatePadding,
    stridedSlice,
    broadcast,
    byteOffsets,
    padding2,
    padding3,
    dynamic,
    i32Transpose,
    u32Broadcast,
    boolOddTranspose,
    i8Transpose,
    i8Rank1PositiveStride,
    i8Rank1NegativeStride,
    i8NegativeStride,
    i8Rank4Permutation,
    i8Rank4NegativeStride,
    u8Broadcast,
    i16Transpose,
    u16Broadcast,
    f16OddTranspose,
    f16Rank1PositiveStride,
    f16Rank1NegativeStride,
    f16NegativeStride,
    f16Rank4Permutation,
    f16Rank4NegativeStride,
    bf16Rank3Permutation,
    f64Transpose,
    f64Rank1PositiveStride,
    f64Rank1NegativeStride,
    f64NegativeStride,
    f64Rank4Permutation,
    f64Rank4NegativeStride,
    i64ByteMapBroadcast,
    u64Rank3Permutation,
    zero,
  ]);
}

async function makeCase(
  id: string,
  input: LayoutInput,
  invalidSource: InvalidSourcePolicy,
  sourceWords: Uint32Array,
  options: Readonly<{
    bindings?: Readonly<Record<string, WireI64>>;
    destinationWordLength?: number;
    expectedSubmitted?: boolean;
  }> = {},
): Promise<EvidenceCase> {
  const layout = await verifiedLayout(input);
  const kernel = await verifiedKernel(layout, invalidSource);
  const destinationByteLength = layoutArtifactPayload(layout).allocations[1]?.byteLength;
  if (destinationByteLength === undefined) throw new Error("evidence fixture destination allocation missing");
  const destinationWords = options.destinationWordLength ?? (
    destinationByteLength.kind === "const"
      ? Number(wireIntegerToBigInt(destinationByteLength.value) / 4n)
      : undefined
  );
  if (destinationWords === undefined) throw new Error("symbolic evidence fixtures require destinationWordLength");
  return Object.freeze({
    id,
    layout,
    kernel,
    sourceWords,
    initialDestinationWords: filledWords(destinationWords, 0xdeadbeef),
    ...(options.bindings === undefined ? {} : { bindings: options.bindings }),
    expectedSubmitted: options.expectedSubmitted ?? true,
  });
}

async function verifiedLayout(input: LayoutInput): Promise<VerifiedLayoutArtifact> {
  const destinationLocation = input.destinationLocation ?? rowMajor(input.shape);
  const dtype = input.dtype ?? "f32";
  const requiredAlignmentBytes =
    dtype === "bool" ||
    dtype === "i8" ||
    dtype === "u8"
      ? 1
      : dtype === "i16" ||
    dtype === "u16" ||
    dtype === "f16" ||
    dtype === "bf16"
      ? 2
      : dtype === "f64" ||
          dtype === "i64" ||
          dtype === "u64"
        ? 8
      : 4;
  return verifyLayoutArtifact(JSON.parse(JSON.stringify({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "webgpu-evidence", version: "1" },
    artifactId: "layout",
    requiredExtensions: [],
    payload: {
      symbols: input.symbols ?? [],
      constraints: [],
      allocations: [
        { allocationId: "source", byteLength: input.sourceBytes, memorySpace: { kind: "global" }, alignmentBytes: Math.max(4, requiredAlignmentBytes), aliasSetId: "sourceAlias" },
        { allocationId: "destination", byteLength: input.destinationBytes, memorySpace: { kind: "global" }, alignmentBytes: Math.max(4, requiredAlignmentBytes), aliasSetId: "destinationAlias" },
      ],
      indexMaps: [
        { indexMapId: "sourceMap", coordinateRank: input.shape.length, locationUnit: input.sourceLocationUnit ?? "element", location: input.sourceLocation, inBounds: input.sourcePredicate ?? TRUE },
        { indexMapId: "destinationMap", coordinateRank: input.shape.length, locationUnit: input.destinationLocationUnit ?? "element", location: destinationLocation, inBounds: TRUE },
      ],
      views: [
        { viewId: "sourceView", allocationId: "source", dtype, byteOffset: input.sourceByteOffset ?? dimConstant("0"), shape: input.shape, indexMapId: "sourceMap", requiredAlignmentBytes },
        { viewId: "destinationView", allocationId: "destination", dtype, byteOffset: input.destinationByteOffset ?? dimConstant("0"), shape: input.shape, indexMapId: "destinationMap", requiredAlignmentBytes },
      ],
    },
  })));
}

async function verifiedKernel(layout: VerifiedLayoutArtifact, invalidSource: InvalidSourcePolicy): Promise<VerifiedKernelArtifact> {
  const payload = layoutArtifactPayload(layout);
  return verifyKernelArtifact({
    schema: "browsergrad.kernel",
    version: { major: 1, minor: 0 },
    producer: { id: "webgpu-evidence", version: "1" },
    artifactId: "kernel",
    requiredExtensions: [],
    payload: {
      layoutSemanticHash: await hashSemanticArtifact(layout),
      operations: [{
        operationId: "copy",
        kind: "view-copy",
        version: { major: 1, minor: 0 },
        dtype: payload.views[0]!.dtype,
        source: { viewId: payload.views[0]!.viewId, access: "read", invalidSource },
        destination: { viewId: payload.views[1]!.viewId, access: "write" },
        overlap: { kind: "forbid" },
      }],
    },
  }, { layout });
}

function rectangularPaddingPredicate(minima: readonly number[], maxima: readonly number[]): PredicateExpr {
  return {
    kind: "and",
    values: minima.flatMap((minimum, axis) => [
      { kind: "lessEqual", lhs: indexConstant(String(minimum)), rhs: coordinate(axis) } as const,
      { kind: "lessEqual", lhs: coordinate(axis), rhs: indexConstant(String(maxima[axis])) } as const,
    ]),
  };
}

function exactNanFill(): InvalidSourcePolicy {
  return { kind: "fill", value: { kind: "float-bits", dtype: "f32", bits: "7fc01234" } };
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

function rowMajor(shape: readonly DimExpr[]): IndexExpr {
  let result: IndexExpr = coordinate(0);
  for (let axis = 1; axis < shape.length; axis += 1) {
    const dimension = shape[axis];
    if (dimension?.kind !== "const") throw new Error("evidence fixtures require static shapes");
    result = add(multiply(result, indexConstant(dimension.value)), coordinate(axis));
  }
  return result;
}

function words(...values: readonly number[]): Uint32Array {
  return new Uint32Array(values);
}

function packed16Words(...values: readonly number[]): Uint32Array {
  if (values.length % 2 !== 0) {
    throw new Error("packed16Words requires complete u32 words");
  }
  return Uint32Array.from(
    Array.from(
      { length: values.length / 2 },
      (_, index) =>
        ((values[index * 2] as number) & 0xffff) |
        (((values[(index * 2) + 1] as number) & 0xffff) << 16),
    ),
  );
}

function packed8Words(...values: readonly number[]): Uint32Array {
  if (values.length % 4 !== 0) {
    throw new Error("packed8Words requires complete u32 words");
  }
  return Uint32Array.from(
    Array.from(
      { length: values.length / 4 },
      (_, index) =>
        ((values[index * 4] as number) & 0xff) |
        (((values[(index * 4) + 1] as number) & 0xff) << 8) |
        (((values[(index * 4) + 2] as number) & 0xff) << 16) |
        (((values[(index * 4) + 3] as number) & 0xff) << 24),
    ),
  );
}

function sequenceWords(length: number, start: number): Uint32Array {
  const result = new Uint32Array(length);
  for (let index = 0; index < length; index += 1) result[index] = start + index;
  return result;
}

function filledWords(length: number, value: number): Uint32Array {
  const result = new Uint32Array(length);
  result.fill(value);
  return result;
}

function cloneWords(source: Uint32Array): Uint32Array {
  const result = new Uint32Array(source.length);
  for (let index = 0; index < source.length; index += 1) result[index] = source[index]!;
  return result;
}

function wordsAsBytes(source: Uint32Array): Uint8Array {
  return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
}

async function prepareEvidenceCase(evidenceCase: EvidenceCase): Promise<PreparedEvidenceCase> {
  const operation = kernelArtifactPayload(evidenceCase.kernel).operations[0];
  if (operation === undefined) throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-FIXTURE", "evidence fixture operation missing");
  const request = {
    operationId: operation.operationId,
    ...(evidenceCase.bindings === undefined ? {} : { bindings: evidenceCase.bindings }),
  };
  const [preparedWgsl, preparedCpu, layoutSemanticHash, kernelSemanticHash, inputHash] = await Promise.all([
    prepareSemanticViewCopyWgsl(evidenceCase.layout, evidenceCase.kernel, request),
    prepareViewCopyCpu(evidenceCase.layout, evidenceCase.kernel, request),
    hashSemanticArtifact(evidenceCase.layout),
    hashSemanticArtifact(evidenceCase.kernel),
    hashNamedComponents({
      sourceWords: [...evidenceCase.sourceWords],
      initialDestinationWords: [...evidenceCase.initialDestinationWords],
    }),
  ]);
  const artifactHash = await hashNamedComponents({
    caseId: evidenceCase.id,
    layoutSemanticHash,
    kernelSemanticHash,
    semanticSpecializationHash: preparedWgsl.semantic.specializationHash,
    wgslModuleHash: preparedWgsl.wgslModuleHash,
    backendProfile: preparedWgsl.backendProfile,
    backendVersion: preparedWgsl.backendVersion,
  });
  return Object.freeze({ evidenceCase, preparedWgsl, preparedCpu, inputHash, artifactHash });
}

function assertPlannedCases(cases: readonly EvidenceCase[]): void {
  const actual = cases.map((entry) => entry.id);
  if (actual.length !== PLANNED_CASE_IDS.length || actual.some((id, index) => id !== PLANNED_CASE_IDS[index])) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-EVIDENCE-CASE-SET",
      `case set drifted: expected ${PLANNED_CASE_IDS.join(",")}; got ${actual.join(",")}`,
    );
  }
}

function assertCaseResult(
  evidenceCase: EvidenceCase,
  preparedWgsl: PreparedSemanticViewCopyWgsl,
  cpuSpecializationHash: string,
  cpuDestination: Uint32Array,
  gpuResult: Awaited<ReturnType<typeof runSemanticViewCopyWebGpu>>,
): void {
  if (!equalWords(gpuResult.destinationWords, cpuDestination)) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-EVIDENCE-COMPARISON",
      `${evidenceCase.id} destination differs under ${COMPARISON_POLICY_ID}`,
    );
  }
  if (gpuResult.trace.semanticSpecializationHash !== cpuSpecializationHash) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-EVIDENCE-SEMANTIC-HASH",
      `${evidenceCase.id} CPU and WebGPU semantic specialization hashes differ`,
    );
  }
  if (gpuResult.trace.semanticSpecializationHash !== preparedWgsl.semantic.specializationHash) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-EVIDENCE-SEMANTIC-HASH",
      `${evidenceCase.id} executed a different semantic specialization than the prepared WGSL`,
    );
  }
  if (gpuResult.trace.submitted !== evidenceCase.expectedSubmitted) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-EVIDENCE-SUBMISSION",
      `${evidenceCase.id} submitted=${gpuResult.trace.submitted}; expected ${evidenceCase.expectedSubmitted}`,
    );
  }
}

function caseObservation(
  preparedCase: PreparedEvidenceCase,
  trace: SemanticViewCopyWebGpuTrace,
): CaseObservation {
  const logicalInvocationCount = Object.freeze([...preparedCase.preparedWgsl.launch.dispatchCount]);
  const submittedWorkgroupCount = trace.submitted
    ? Object.freeze([
        Math.max(Math.ceil(logicalInvocationCount[0]! / preparedCase.preparedWgsl.workgroupSize), 1),
        Math.max(logicalInvocationCount[1]!, 1),
        Math.max(logicalInvocationCount[2]!, 1),
      ])
    : Object.freeze([0, 0, 0]);
  return Object.freeze({
    caseId: preparedCase.evidenceCase.id,
    artifactHash: preparedCase.artifactHash,
    inputHash: preparedCase.inputHash,
    layoutSemanticHash: trace.layoutSemanticHash,
    kernelSemanticHash: trace.kernelSemanticHash,
    semanticSpecializationHash: trace.semanticSpecializationHash,
    backendSpecializationHash: trace.backendSpecializationHash,
    wgslModuleHash: trace.wgslModuleHash,
    workgroupSize: preparedCase.preparedWgsl.workgroupSize,
    logicalInvocationCount,
    submittedWorkgroupCount,
    pipelineCount: trace.submitted ? 1 : 0,
    submitted: trace.submitted,
    comparisonPolicyId: COMPARISON_POLICY_ID,
  });
}

function equalWords(left: Uint32Array, right: Uint32Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function freezeEnvironment(
  input: Readonly<{
    acquisition: string;
    adapter?: JsonObject;
    adapterSupportedFeatures?: readonly string[];
    negotiatedDeviceFeatures?: readonly string[];
    negotiatedDeviceLimits?: JsonObject;
    unavailableReason?: string;
  }>,
): EvidenceEnvironment {
  return Object.freeze({
    schema: ENVIRONMENT_SCHEMA,
    acquisition: input.acquisition,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    ...(input.adapter === undefined ? {} : { adapter: Object.freeze({ ...input.adapter }) }),
    ...(input.adapterSupportedFeatures === undefined
      ? {}
      : { adapterSupportedFeatures: Object.freeze([...input.adapterSupportedFeatures]) }),
    ...(input.negotiatedDeviceFeatures === undefined
      ? {}
      : { negotiatedDeviceFeatures: Object.freeze([...input.negotiatedDeviceFeatures]) }),
    ...(input.negotiatedDeviceLimits === undefined
      ? {}
      : { negotiatedDeviceLimits: Object.freeze({ ...input.negotiatedDeviceLimits }) }),
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

function emitTerminalEvidence(input: Readonly<{
  required: boolean;
  artifactHash: string;
  artifactHashKind: TerminalEvidenceRecord["artifactHashKind"];
  caseSetHash?: string;
  environment: EvidenceEnvironment;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: ExecutionEvidence["outcome"];
  diagnosticCodes: readonly string[];
  completedCases: readonly CaseObservation[];
  stage: string;
  currentCaseId?: string;
  uncapturedErrors: readonly string[];
  error?: JsonObject;
}>): void {
  const evidence = Object.freeze({
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
  }) as ExecutionEvidence & JsonObject;
  const record = Object.freeze({
    schema: EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: input.required,
    evidence,
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
  emitEvidence(record);
}

function validateTerminalEvidence(record: TerminalEvidenceRecord): void {
  validateTerminalExecutionEvidence(record, TERMINAL_EXPECTATION);
  if (record.schema !== EVIDENCE_SCHEMA || record.kind !== "terminal" || record.suiteId !== SUITE_ID) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "terminal evidence envelope identity is invalid");
  }
  if (record.evidence.capabilityId !== CAPABILITY_ID || record.evidence.backendId !== BACKEND_ID) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "terminal evidence capability/backend identity is invalid");
  }
  if (record.evidence.comparisonPolicyId !== COMPARISON_POLICY_ID) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "terminal evidence comparison policy is invalid");
  }
  for (const [name, value] of [
    ["artifactHash", record.evidence.artifactHash],
    ["environmentId", record.evidence.environmentId],
    ...(record.evidence.deviceProfileHash === undefined
      ? []
      : [["deviceProfileHash", record.evidence.deviceProfileHash]]),
  ] as const) {
    if (!/^[0-9a-f]{64}$/u.test(value)) {
      throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", `${name} is not a full SHA-256 digest`);
    }
  }
  if (Number.isNaN(Date.parse(record.evidence.recordedAt))) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "recordedAt is not an ISO timestamp");
  }
  if (Object.values(record.evidence.producerVersions).some((version) => typeof version !== "string" || version.length === 0)) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "producerVersions contains an empty version");
  }
  const completedIds = record.completedCases.map((entry) => entry.caseId);
  if (new Set(completedIds).size !== completedIds.length || completedIds.some((id) => !PLANNED_CASE_IDS.includes(id))) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-CASE-SET", "completed case observations are duplicated or unknown");
  }
  if (record.artifactHashKind === "prepared-suite" && record.caseSetHash === undefined) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "prepared-suite evidence requires caseSetHash");
  }
  if (record.evidence.outcome === "passed") {
    if (record.evidence.diagnosticCodes.length !== 0) {
      throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "passed evidence cannot contain diagnostics");
    }
    if (record.evidence.deviceProfileHash === undefined) {
      throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "passed device evidence requires deviceProfileHash");
    }
    if (completedIds.length !== PLANNED_CASE_IDS.length || completedIds.some((id, index) => id !== PLANNED_CASE_IDS[index])) {
      throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-CASE-SET", "passed evidence requires the complete ordered case set");
    }
  } else if (record.evidence.diagnosticCodes.length === 0) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "failed/not-run evidence requires a diagnostic code");
  }
  if (record.evidence.outcome === "not-run" && record.required) {
    throw new EvidenceLaneError("BG-WEBGPU-EVIDENCE-SCHEMA", "required evidence cannot report not-run");
  }
}

function diagnosticCode(error: unknown, stage: string): string {
  if (error instanceof SemanticViewCopyWebGpuError) return error.code;
  if (error instanceof EvidenceLaneError) return error.code;
  if (stage === "fixture-construction") return "BG-WEBGPU-EVIDENCE-FIXTURE";
  if (stage === "semantic-and-wgsl-preparation") return "BG-WEBGPU-EVIDENCE-PREPARATION";
  if (stage === "device-acquisition") return "BG-WEBGPU-EVIDENCE-DEVICE-UNAVAILABLE";
  if (stage === "kernel-device-construction") return "BG-WEBGPU-EVIDENCE-DEVICE-WRAP";
  return "BG-WEBGPU-EVIDENCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof SemanticViewCopyWebGpuError) {
    return { name: error.name, message: error.message, code: error.code, path: error.path };
  }
  if (error instanceof EvidenceLaneError) {
    return { name: error.name, message: error.message, code: error.code };
  }
  if (error instanceof Error) return { name: error.name, message: error.message };
  return { name: "UnknownError", message: String(error) };
}

async function withEvidenceTimeout<T>(promise: Promise<T>, timeoutMs: number, stage: string): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => reject(new EvidenceLaneError(
          "BG-WEBGPU-EVIDENCE-TIMEOUT",
          `${stage} did not settle within ${timeoutMs}ms`,
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

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}

function emitEvidence(record: TerminalEvidenceRecord): void {
  TERMINAL_EMITTER.emit(record);
}
