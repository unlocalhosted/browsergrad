import type { PreparedViewAccessor } from "../layout/prepare.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { parseWireI64, type WireI64 } from "../schema/integers.js";

const YIELD_INTERVAL_MS = 16;

export interface KernelPreparationControl {
  readonly startedAt: number;
  readonly maxPreparationMs: number;
  readonly signal?: AbortSignal;
}

export function normalizeKernelBindings(
  bindings: Readonly<Record<string, WireI64>>,
): Readonly<Record<string, WireI64>> {
  if (typeof bindings !== "object" || bindings === null || Array.isArray(bindings)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "bindings must be a plain data object");
  }
  const prototype = Object.getPrototypeOf(bindings);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "bindings must be a plain data object");
  }
  const descriptors = Object.getOwnPropertyDescriptors(bindings);
  const result = Object.create(null) as Record<string, WireI64>;
  for (const key of Reflect.ownKeys(bindings)) {
    if (typeof key !== "string") {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "binding keys must be strings");
    }
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.invalidBinding,
        `$.bindings.${key}`,
        "bindings must use enumerable data properties without accessors",
      );
    }
    result[key] = parseWireI64(descriptor.value, `$.bindings.${key}`);
  }
  return Object.freeze(result);
}

export function resolveKernelBudget(
  value: number | undefined,
  fallback: number,
  maximum: number,
  name: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.resourceLimit,
      `$.${name}`,
      `${name} must be a positive safe integer no greater than ${maximum}`,
    );
  }
  return resolved;
}

export function kernelMonotonicNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

export function ensureKernelPreparationActive(control: KernelPreparationControl): void {
  if (control.signal?.aborted === true) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.signal", "kernel preparation was aborted");
  }
  if (kernelMonotonicNow() - control.startedAt > control.maxPreparationMs) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.resourceLimit,
      "$.maxPreparationMs",
      `kernel preparation exceeded ${control.maxPreparationMs} ms`,
    );
  }
}

export async function proveDenseRowMajorAccessor(
  accessor: PreparedViewAccessor,
  shape: readonly bigint[],
  control: KernelPreparationControl,
  role: string,
): Promise<void> {
  let yieldAt = control.startedAt + YIELD_INTERVAL_MS;
  const coordinates = shape.map(() => 0n);
  const elements = shape.reduce((product, extent) => product * extent, 1n);
  for (let linear = 0n; linear < elements; linear += 1n) {
    if ((linear & 1023n) === 0n) {
      ensureKernelPreparationActive(control);
      const now = kernelMonotonicNow();
      if (now >= yieldAt) {
        await yieldToMainThread();
        ensureKernelPreparationActive(control);
        yieldAt = kernelMonotonicNow() + YIELD_INTERVAL_MS;
      }
    }
    linearToCoordinates(linear, shape, coordinates);
    const access = accessor.access(coordinates);
    if (!access.accessInBounds) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.invalidAccess,
        `$.${role}[${coordinates.join(",")}]`,
        `dense ${role} coordinate is not a valid allocation access`,
      );
    }
    const expected = accessor.viewByteOffset + (linear * BigInt(accessor.dtypeBytes));
    if (access.rootByteStart !== expected) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.${role}[${coordinates.join(",")}]`,
        `initial ${role} profile requires a dense row-major view`,
      );
    }
    if (access.rootByteStart > BigInt(Number.MAX_SAFE_INTEGER)) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.${role}[${coordinates.join(",")}]`,
        `CPU-addressable ${role} offsets must fit exact JavaScript indexes`,
      );
    }
  }
  ensureKernelPreparationActive(control);
}

function linearToCoordinates(
  linear: bigint,
  shape: readonly bigint[],
  output: bigint[],
): void {
  let remainder = linear;
  for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
    const extent = shape[axis];
    if (extent === undefined) throw new Error("internal: dense shape axis disappeared");
    output[axis] = remainder % extent;
    remainder /= extent;
  }
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
