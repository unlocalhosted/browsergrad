const OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const OBJECT_CREATE = Object.create;
const OBJECT_FREEZE = Object.freeze;
const REFLECT_OWN_KEYS = Reflect.ownKeys;
const REFLECT_APPLY = Reflect.apply;
const UINT8_ARRAY_CONSTRUCTOR = Uint8Array;
const UINT8_ARRAY_PROTOTYPE = Uint8Array.prototype;
const ARRAY_BUFFER_PROTOTYPE = ArrayBuffer.prototype;
const TYPED_ARRAY_PROTOTYPE = OBJECT_GET_PROTOTYPE_OF(UINT8_ARRAY_PROTOTYPE) as object;
const BUFFER_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "buffer");
const BYTE_OFFSET_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteOffset");
const BYTE_LENGTH_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteLength");
const ARRAY_BUFFER_RESIZABLE_GETTER = OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(
  ARRAY_BUFFER_PROTOTYPE,
  "resizable",
)?.get;
const ARRAY_BUFFER_SLICE = ARRAY_BUFFER_PROTOTYPE.slice;
const UINT8_ARRAY_SET = UINT8_ARRAY_PROTOTYPE.set;
const DATA_VIEW_CONSTRUCTOR = DataView;
const DATA_VIEW_GET_FLOAT32 = DataView.prototype.getFloat32;
const NUMBER_IS_FINITE = Number.isFinite;
const PERFORMANCE_NOW = typeof globalThis.performance?.now === "function"
  ? globalThis.performance.now.bind(globalThis.performance)
  : Date.now;
const ABORT_SIGNAL_PROTOTYPE = typeof globalThis.AbortSignal === "undefined"
  ? undefined
  : globalThis.AbortSignal.prototype;
const ABORT_SIGNAL_ABORTED_GETTER = ABORT_SIGNAL_PROTOTYPE === undefined
  ? undefined
  : OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(ABORT_SIGNAL_PROTOTYPE, "aborted")?.get;
const YIELD_INTERVAL_MS = 16;

export interface NativeF32SnapshotSpec<Name extends string> {
  readonly name: Name;
  readonly expectedByteLength: bigint;
  readonly alignmentBytes: number;
}

export interface NativeF32SnapshotControl {
  readonly maxValidationMs: number;
  readonly signal?: AbortSignal;
  readonly fail: (
    issue: "invalid-binding" | "numerical-domain" | "resource-limit" | "cancelled" | "internal",
    path: string,
    message: string,
    cause?: unknown,
  ) => never;
}

interface NativeUint8Slots {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
}

/**
 * Captures exact fixed unshared f32 allocation snapshots before the first
 * yield, then validates their complete finite domain under time/abort bounds.
 */
export async function captureFiniteF32Snapshots<Name extends string>(
  value: unknown,
  specs: readonly NativeF32SnapshotSpec<Name>[],
  path: string,
  control: NativeF32SnapshotControl,
): Promise<Readonly<Record<Name, Uint8Array>>> {
  const captured = captureBindings(value, specs, path, control.fail);
  const slots = OBJECT_CREATE(null) as Record<Name, NativeUint8Slots>;
  for (const spec of specs) {
    const rolePath = `${path}.${spec.name}`;
    const roleSlots = nativeSlots(captured[spec.name], rolePath, control.fail);
    if (BigInt(roleSlots.byteLength) !== spec.expectedByteLength) {
      control.fail(
        "invalid-binding",
        rolePath,
        `binding length ${roleSlots.byteLength} does not equal declared allocation length ${spec.expectedByteLength}`,
      );
    }
    if (roleSlots.byteOffset % spec.alignmentBytes !== 0) {
      control.fail(
        "invalid-binding",
        rolePath,
        `binding byte offset does not satisfy ${spec.alignmentBytes}-byte alignment`,
      );
    }
    if (roleSlots.byteLength % 4 !== 0) {
      control.fail("invalid-binding", rolePath, "f32 binding byte length must be a multiple of four");
    }
    slots[spec.name] = roleSlots;
  }
  for (let left = 0; left < specs.length; left += 1) {
    for (let right = left + 1; right < specs.length; right += 1) {
      const leftSpec = specs[left];
      const rightSpec = specs[right];
      if (leftSpec === undefined || rightSpec === undefined) {
        control.fail("internal", path, "internal snapshot role disappeared");
      }
      if (rangesOverlap(slots[leftSpec.name], slots[rightSpec.name])) {
        control.fail(
          "invalid-binding",
          path,
          `bindings ${leftSpec.name} and ${rightSpec.name} must not overlap`,
        );
      }
    }
  }

  const snapshots = OBJECT_CREATE(null) as Record<Name, Uint8Array>;
  for (const spec of specs) {
    const roleSlots = slots[spec.name];
    if (roleSlots === undefined) {
      control.fail("internal", path, "internal snapshot slots disappeared");
    }
    const copy = new UINT8_ARRAY_CONSTRUCTOR(roleSlots.byteLength);
    try {
      REFLECT_APPLY(UINT8_ARRAY_SET, copy, [captured[spec.name]]);
    } catch (error) {
      control.fail(
        "invalid-binding",
        `${path}.${spec.name}`,
        "binding could not be copied atomically",
        error,
      );
    }
    snapshots[spec.name] = copy;
  }
  const frozen = OBJECT_FREEZE(snapshots);

  const startedAt = PERFORMANCE_NOW();
  let yieldAt = startedAt + YIELD_INTERVAL_MS;
  for (const spec of specs) {
    const snapshot = frozen[spec.name];
    const view = new DATA_VIEW_CONSTRUCTOR(snapshot.buffer);
    for (let offset = 0; offset < snapshot.byteLength; offset += 4) {
      if ((offset & 16_383) === 0) {
        ensureActive(startedAt, control);
        const now = PERFORMANCE_NOW();
        if (now >= yieldAt) {
          await yieldToMainThread();
          ensureActive(startedAt, control);
          yieldAt = PERFORMANCE_NOW() + YIELD_INTERVAL_MS;
        }
      }
      const item = REFLECT_APPLY(DATA_VIEW_GET_FLOAT32, view, [offset, true]) as number;
      if (!NUMBER_IS_FINITE(item)) {
        control.fail(
          "numerical-domain",
          `${path}.${spec.name}`,
          `${spec.name} f32 element ${offset / 4} is not finite`,
        );
      }
    }
  }
  ensureActive(startedAt, control);
  return frozen;
}

function captureBindings<Name extends string>(
  value: unknown,
  specs: readonly NativeF32SnapshotSpec<Name>[],
  path: string,
  fail: NativeF32SnapshotControl["fail"],
): Readonly<Record<Name, Uint8Array>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail("invalid-binding", path, "bindings must be a plain data object");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = OBJECT_GET_PROTOTYPE_OF(value);
    descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (error) {
    fail(
      "invalid-binding",
      path,
      "bindings could not be captured without invoking properties",
      error,
    );
  }
  if (prototype !== Object.prototype && prototype !== null) {
    fail("invalid-binding", path, "bindings must be a plain data object");
  }
  const names = new Set(specs.map(({ name }) => name));
  const keys = REFLECT_OWN_KEYS(descriptors);
  if (keys.length !== specs.length
    || keys.some((key) => typeof key !== "string" || !names.has(key as Name))) {
    fail(
      "invalid-binding",
      path,
      `bindings require exactly ${[...names].join(", ")} own properties`,
    );
  }
  const captured = OBJECT_CREATE(null) as Record<Name, Uint8Array>;
  for (const spec of specs) {
    const descriptor = descriptors[spec.name];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      fail(
        "invalid-binding",
        `${path}.${spec.name}`,
        "bindings must use enumerable own data properties without accessors",
      );
    }
    captured[spec.name] = descriptor.value as Uint8Array;
  }
  return OBJECT_FREEZE(captured);
}

function nativeSlots(
  value: Uint8Array,
  path: string,
  fail: NativeF32SnapshotControl["fail"],
): NativeUint8Slots {
  let prototype: object | null;
  let rawBuffer: ArrayBufferLike;
  let byteOffset: number;
  let byteLength: number;
  try {
    prototype = OBJECT_GET_PROTOTYPE_OF(value);
    rawBuffer = REFLECT_APPLY(BUFFER_GETTER, value, []) as ArrayBufferLike;
    byteOffset = REFLECT_APPLY(BYTE_OFFSET_GETTER, value, []) as number;
    byteLength = REFLECT_APPLY(BYTE_LENGTH_GETTER, value, []) as number;
  } catch (error) {
    fail(
      "invalid-binding",
      path,
      "binding does not expose native typed-array internal slots",
      error,
    );
  }
  if (prototype !== UINT8_ARRAY_PROTOTYPE) {
    fail(
      "invalid-binding",
      path,
      "bindings must be direct Uint8Array values without subclass or proxy behavior",
    );
  }
  if (OBJECT_GET_PROTOTYPE_OF(rawBuffer) !== ARRAY_BUFFER_PROTOTYPE) {
    fail("invalid-binding", path, "bindings must use unshared ArrayBuffer storage");
  }
  const buffer = rawBuffer as ArrayBuffer;
  if (ARRAY_BUFFER_RESIZABLE_GETTER !== undefined
    && REFLECT_APPLY(ARRAY_BUFFER_RESIZABLE_GETTER, buffer, []) === true) {
    fail("invalid-binding", path, "bindings must use fixed-length ArrayBuffer storage");
  }
  try {
    REFLECT_APPLY(ARRAY_BUFFER_SLICE, buffer, [0, 0]);
  } catch (error) {
    fail("invalid-binding", path, "binding storage must not be detached", error);
  }
  return OBJECT_FREEZE({ buffer, byteOffset, byteLength });
}

function rangesOverlap(left: NativeUint8Slots, right: NativeUint8Slots): boolean {
  if (left.buffer !== right.buffer) return false;
  return left.byteOffset < right.byteOffset + right.byteLength
    && right.byteOffset < left.byteOffset + left.byteLength;
}

function ensureActive(startedAt: number, control: NativeF32SnapshotControl): void {
  if (control.signal !== undefined && abortSignalAborted(control.signal, control.fail)) {
    control.fail("cancelled", "$.signal", "input snapshot validation was cancelled");
  }
  if (PERFORMANCE_NOW() - startedAt > control.maxValidationMs) {
    control.fail(
      "resource-limit",
      "$.maxInputValidationMs",
      `input snapshot validation exceeded ${control.maxValidationMs} ms`,
    );
  }
}

function abortSignalAborted(
  signal: AbortSignal,
  fail: NativeF32SnapshotControl["fail"],
): boolean {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined
    || OBJECT_GET_PROTOTYPE_OF(signal) !== ABORT_SIGNAL_PROTOTYPE) {
    fail("invalid-binding", "$.signal", "signal must be a native AbortSignal");
  }
  try {
    return REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []) as boolean;
  } catch (error) {
    fail(
      "invalid-binding",
      "$.signal",
      "signal does not expose native AbortSignal state",
      error,
    );
  }
}

function requiredGetter(target: object, name: string): (this: unknown) => unknown {
  const getter = OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(target, name)?.get;
  if (getter === undefined) throw new Error(`internal: missing typed-array ${name} getter`);
  return getter;
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
