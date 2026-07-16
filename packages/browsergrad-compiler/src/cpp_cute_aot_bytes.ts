const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype) as object;
const TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "buffer",
)?.get;
const TYPED_ARRAY_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "byteLength",
)?.get;
const TYPED_ARRAY_TAG_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  Symbol.toStringTag,
)?.get;
const UINT8_ARRAY_SET = Uint8Array.prototype.set;
const INSPECTIONS = new WeakMap<object, StoredUint8ArrayInspection>();

interface StoredUint8ArrayInspection {
  readonly value: object;
  readonly byteLength: number;
}

declare const inspectedUnsharedUint8ArrayBrand: unique symbol;

/** Opaque synchronous inspection of one exact, unshared, plain byte view. */
export interface InspectedUnsharedUint8Array {
  readonly [inspectedUnsharedUint8ArrayBrand]: true;
  readonly byteLength: number;
}

/**
 * Uses typed-array intrinsics rather than caller properties. It rejects
 * proxies, detached/shared buffers, subclasses, clamped bytes, and
 * prototype-disguised non-byte typed arrays.
 */
export function inspectUnsharedPlainUint8Array(value: unknown): InspectedUnsharedUint8Array {
  if (
    TYPED_ARRAY_BUFFER_GETTER === undefined
    || TYPED_ARRAY_BYTE_LENGTH_GETTER === undefined
    || TYPED_ARRAY_TAG_GETTER === undefined
  ) {
    throw new TypeError("Uint8Array intrinsic accessors are unavailable");
  }
  if (typeof value !== "object" || value === null) throw new TypeError("value is not an object");
  const prototype = Object.getPrototypeOf(value);
  const buffer = TYPED_ARRAY_BUFFER_GETTER.call(value) as ArrayBufferLike;
  const byteLength = TYPED_ARRAY_BYTE_LENGTH_GETTER.call(value) as unknown;
  const tag = TYPED_ARRAY_TAG_GETTER.call(value) as unknown;
  if (prototype !== Uint8Array.prototype) throw new TypeError("value is not a plain Uint8Array");
  if (tag !== "Uint8Array") throw new TypeError("value does not have Uint8Array element semantics");
  if (typeof byteLength !== "number" || !Number.isSafeInteger(byteLength) || byteLength < 0) {
    throw new TypeError("Uint8Array byte length is invalid");
  }
  if (typeof SharedArrayBuffer !== "undefined" && buffer instanceof SharedArrayBuffer) {
    throw new TypeError("SharedArrayBuffer-backed bytes are forbidden");
  }
  const inspection = Object.freeze({ byteLength }) as InspectedUnsharedUint8Array;
  INSPECTIONS.set(inspection, Object.freeze({ value, byteLength }));
  return inspection;
}

/** Copies the exact view bound to one prior inspection without species access. */
export function copyInspectedUnsharedUint8Array(
  value: unknown,
  inspection: InspectedUnsharedUint8Array,
): Uint8Array {
  if (typeof inspection !== "object" || inspection === null) {
    throw new TypeError("expected opaque Uint8Array inspection");
  }
  const record = INSPECTIONS.get(inspection as object);
  if (record === undefined || value !== record.value || inspection.byteLength !== record.byteLength) {
    throw new TypeError("Uint8Array inspection does not own this exact view");
  }
  const copy = new Uint8Array(record.byteLength);
  UINT8_ARRAY_SET.call(copy, value as Uint8Array);
  return copy;
}
