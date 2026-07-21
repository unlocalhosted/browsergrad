import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";

export interface NativeUint8Slots {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
}

const OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const OBJECT_CREATE = Object.create;
const OBJECT_FREEZE = Object.freeze;
const REFLECT_OWN_KEYS = Reflect.ownKeys;
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

export function captureExactUint8Bindings<Name extends string>(
  value: unknown,
  names: readonly Name[],
  path: string,
): Readonly<Record<Name, Uint8Array>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(path, "CPU bindings must be a plain data object");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = OBJECT_GET_PROTOTYPE_OF(value);
    descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch {
    invalid(path, "CPU bindings must expose ordinary own data properties");
  }
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(path, "CPU bindings must be a plain data object");
  }
  const nameSet = new Set<string>(names);
  const capturedKeys = REFLECT_OWN_KEYS(descriptors);
  if (capturedKeys.length !== names.length
    || capturedKeys.some((key) => typeof key !== "string" || !nameSet.has(key))) {
    invalid(path, `CPU bindings require exactly ${names.join(", ")} own properties`);
  }
  const captured = OBJECT_CREATE(null) as Record<Name, Uint8Array>;
  for (const name of names) {
    const descriptor = descriptors[name];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(
        `${path}.${name}`,
        "CPU bindings must use enumerable own data properties without accessors",
      );
    }
    captured[name] = descriptor.value as Uint8Array;
  }
  return OBJECT_FREEZE(captured);
}

export function nativeUint8Slots(value: Uint8Array, path: string): NativeUint8Slots {
  try {
    if (OBJECT_GET_PROTOTYPE_OF(value) !== UINT8_ARRAY_PROTOTYPE) {
      invalid(path, "CPU bindings must be direct Uint8Array values without subclass or proxy behavior");
    }
    const buffer = BUFFER_GETTER.call(value) as ArrayBufferLike;
    if (OBJECT_GET_PROTOTYPE_OF(buffer) !== ARRAY_BUFFER_PROTOTYPE) {
      invalid(path, "CPU bindings must use unshared ArrayBuffer storage");
    }
    const arrayBuffer = buffer as ArrayBuffer;
    if (ARRAY_BUFFER_RESIZABLE_GETTER?.call(arrayBuffer) === true) {
      invalid(path, "CPU bindings must use fixed-length ArrayBuffer storage");
    }
    try {
      ARRAY_BUFFER_SLICE.call(arrayBuffer, 0, 0);
    } catch {
      invalid(path, "CPU binding storage must not be detached");
    }
    return OBJECT_FREEZE({
      buffer: arrayBuffer,
      byteOffset: BYTE_OFFSET_GETTER.call(value) as number,
      byteLength: BYTE_LENGTH_GETTER.call(value) as number,
    });
  } catch (error) {
    if (error instanceof SemanticSchemaError) throw error;
    invalid(path, "CPU binding does not expose native typed-array internal slots");
  }
}

export function requireExactNativeByteLength(
  slots: NativeUint8Slots,
  expected: bigint,
  path: string,
): void {
  if (BigInt(slots.byteLength) !== expected) {
    invalid(
      path,
      `binding length ${slots.byteLength} does not equal declared allocation length ${expected}`,
    );
  }
}

export function requireNativeAlignment(
  slots: NativeUint8Slots,
  alignment: number,
  path: string,
): void {
  if (slots.byteOffset % alignment !== 0) {
    invalid(path, `binding byte offset does not satisfy ${alignment}-byte alignment`);
  }
}

export function nativeRangesOverlap(left: NativeUint8Slots, right: NativeUint8Slots): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftEnd = left.byteOffset + left.byteLength;
  const rightEnd = right.byteOffset + right.byteLength;
  return left.byteOffset < rightEnd && right.byteOffset < leftEnd;
}

export function copyNativeUint8(value: Uint8Array, slots: NativeUint8Slots): Uint8Array {
  const copy = new UINT8_ARRAY_CONSTRUCTOR(slots.byteLength);
  UINT8_ARRAY_SET.call(copy, value);
  return copy;
}

function requiredGetter(target: object, name: string): (this: unknown) => unknown {
  const getter = OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(target, name)?.get;
  if (getter === undefined) throw new Error(`internal: missing typed-array ${name} getter`);
  return getter;
}

function invalid(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: KERNEL_DIAGNOSTIC_CODES.invalidBinding,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}
