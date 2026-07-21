export const EXECUTION_EVIDENCE_SCHEMA = "browsergrad.execution-evidence@1";
export const EXECUTION_ENVIRONMENT_SCHEMA = "browsergrad.execution-environment@1";

declare const __BG_REQUIRE_WEBGPU__: boolean;

export interface WebGpuEvidenceDevice {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly adapterInfo: Readonly<{
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  }>;
}

export type WebGpuEvidenceAcquisition =
  | { readonly kind: "available"; readonly value: WebGpuEvidenceDevice }
  | { readonly kind: "unavailable"; readonly reason: string };

export interface TerminalEvidenceExpectation {
  readonly suiteId: string;
  readonly capabilityId: string;
  readonly backendId: string;
  readonly comparisonPolicyId?: string;
  readonly requireDeviceProfile?: boolean;
}

export interface TerminalEvidenceEmitter {
  readonly emitted: boolean;
  emit(record: unknown): void;
}

export type EvidenceJsonPrimitive = null | boolean | string | number;
export type EvidenceJsonValue = EvidenceJsonPrimitive | EvidenceJsonObject | readonly EvidenceJsonValue[];
export interface EvidenceJsonObject {
  readonly [key: string]: EvidenceJsonValue;
}

export interface WebGpuExecutionEnvironmentInput {
  readonly acquisition: string;
  readonly adapter?: EvidenceJsonObject;
  readonly adapterSupportedFeatures?: readonly string[];
  readonly negotiatedDeviceFeatures?: readonly string[];
  readonly negotiatedDeviceLimits?: EvidenceJsonObject;
  readonly unavailableReason?: string;
}

/** Canonical named browser/device record shared by required WebGPU evidence lanes. */
export function createWebGpuExecutionEnvironmentRecord(
  input: WebGpuExecutionEnvironmentInput,
): EvidenceJsonObject {
  return Object.freeze({
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
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
    ...(input.unavailableReason === undefined
      ? {}
      : { unavailableReason: input.unavailableReason }),
  });
}

/** Device limits that can change semantic compute admission for current lanes. */
export function webGpuSemanticDeviceLimits(device: GPUDevice): EvidenceJsonObject {
  return Object.freeze({
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
    maxBindingsPerBindGroup: device.limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
  });
}

export async function withWebGpuEvidenceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  label: string,
  timeoutError: (message: string) => Error,
): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(
          () => reject(timeoutError(`${label} did not settle within ${timeoutMs}ms`)),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

export function nextWebGpuEvidenceTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export function requiresWebGpuEvidence(): boolean {
  return __BG_REQUIRE_WEBGPU__;
}

export async function acquireWebGpuEvidenceDevice(): Promise<WebGpuEvidenceAcquisition> {
  if (typeof navigator === "undefined" || navigator.gpu === undefined) {
    return unavailable("navigator.gpu is unavailable");
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (error) {
    return unavailable(`requestAdapter failed: ${message(error)}`);
  }
  if (adapter === null) return unavailable("requestAdapter returned no adapter");
  try {
    const device = await adapter.requestDevice();
    return {
      kind: "available",
      value: {
        adapter,
        device,
        adapterInfo: Object.freeze({
          vendor: adapter.info?.vendor ?? "",
          architecture: adapter.info?.architecture ?? "",
          device: adapter.info?.device ?? "",
          description: adapter.info?.description ?? "",
        }),
      },
    };
  } catch (error) {
    return unavailable(`requestDevice failed: ${message(error)}`);
  }
}

export function requiredEvidenceFailure(reason: string): Error {
  return new Error(`required WebGPU evidence unavailable: ${reason}`);
}

export function validateTerminalExecutionEvidence(
  input: unknown,
  expected: TerminalEvidenceExpectation,
): void {
  const record = object(input, "$record");
  requireEqual(record.schema, EXECUTION_EVIDENCE_SCHEMA, "$record.schema");
  requireEqual(record.kind, "terminal", "$record.kind");
  requireEqual(record.suiteId, expected.suiteId, "$record.suiteId");
  if (typeof record.required !== "boolean") fail("$record.required", "must be boolean");
  const evidence = object(record.evidence, "$record.evidence");
  requireEqual(evidence.capabilityId, expected.capabilityId, "$record.evidence.capabilityId");
  requireEqual(evidence.backendId, expected.backendId, "$record.evidence.backendId");
  if (expected.comparisonPolicyId !== undefined) {
    requireEqual(evidence.comparisonPolicyId, expected.comparisonPolicyId, "$record.evidence.comparisonPolicyId");
  }
  requireDigest(evidence.artifactHash, "$record.evidence.artifactHash");
  requireDigest(evidence.environmentId, "$record.evidence.environmentId");
  if (evidence.deviceProfileHash !== undefined) {
    requireDigest(evidence.deviceProfileHash, "$record.evidence.deviceProfileHash");
  }
  if (typeof evidence.recordedAt !== "string" || Number.isNaN(Date.parse(evidence.recordedAt))) {
    fail("$record.evidence.recordedAt", "must be an ISO timestamp");
  }
  const outcome = evidence.outcome;
  if (outcome !== "not-run" && outcome !== "passed" && outcome !== "failed") {
    fail("$record.evidence.outcome", "must be not-run, passed, or failed");
  }
  const diagnosticCodes = stringArray(evidence.diagnosticCodes, "$record.evidence.diagnosticCodes");
  const versions = object(evidence.producerVersions, "$record.evidence.producerVersions");
  if (Object.keys(versions).length === 0) fail("$record.evidence.producerVersions", "must not be empty");
  for (const [name, version] of Object.entries(versions)) {
    if (typeof version !== "string" || version.length === 0) {
      fail(`$record.evidence.producerVersions.${name}`, "must be a nonempty string");
    }
  }
  if (outcome === "passed") {
    if (diagnosticCodes.length !== 0) fail("$record.evidence.diagnosticCodes", "passed evidence cannot contain diagnostics");
    if (expected.requireDeviceProfile === true && evidence.deviceProfileHash === undefined) {
      fail("$record.evidence.deviceProfileHash", "passed device evidence requires a profile hash");
    }
  } else if (diagnosticCodes.length === 0) {
    fail("$record.evidence.diagnosticCodes", "failed/not-run evidence requires a diagnostic code");
  }
  if (outcome === "not-run" && record.required === true) {
    fail("$record.evidence.outcome", "required evidence cannot report not-run");
  }
}

export function createTerminalEvidenceEmitter(
  prefix: string,
  expected: TerminalEvidenceExpectation,
): TerminalEvidenceEmitter {
  let emitted = false;
  return {
    get emitted() { return emitted; },
    emit(record: unknown): void {
      if (emitted) throw new Error("terminal execution evidence was emitted more than once");
      validateTerminalExecutionEvidence(record, expected);
      emitted = true;
      console.warn(`${prefix} ${JSON.stringify(record)}`);
    },
  };
}

function unavailable(reason: string): WebGpuEvidenceAcquisition {
  return { kind: "unavailable", reason };
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(path, "must be an object");
  return value as Record<string, unknown>;
}

function stringArray(value: unknown, path: string): readonly string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.length === 0)) {
    fail(path, "must be an array of nonempty strings");
  }
  return value as string[];
}

function requireDigest(value: unknown, path: string): void {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/u.test(value)) fail(path, "must be a full SHA-256 digest");
}

function requireEqual(value: unknown, expected: string, path: string): void {
  if (value !== expected) fail(path, `must equal ${expected}`);
}

function fail(path: string, message: string): never {
  throw new Error(`invalid terminal execution evidence at ${path}: ${message}`);
}
