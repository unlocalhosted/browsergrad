import type { WgslFloat16Array } from "./wgsl_program.js";

export interface WgslFloat16ArrayConstructor {
  readonly prototype: WgslFloat16Array;
  new(): WgslFloat16Array;
  new(length: number): WgslFloat16Array;
  new(array: ArrayLike<number> | Iterable<number>): WgslFloat16Array;
  new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): WgslFloat16Array;
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

export function float32ToFloat16Bits(value: number): number {
  f32Scratch[0] = value;
  const bits = u32Scratch[0] ?? 0;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;

  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);

  const unbiasedExponent = exponent - 127;
  if (unbiasedExponent > 15) return sign | 0x7c00;

  if (unbiasedExponent >= -14) {
    let halfExponent = unbiasedExponent + 15;
    let halfMantissa = roundShiftRightToEven(mantissa, 13);
    if (halfMantissa === 0x400) {
      halfMantissa = 0;
      halfExponent++;
      if (halfExponent >= 0x1f) return sign | 0x7c00;
    }
    return sign | (halfExponent << 10) | halfMantissa;
  }

  const subnormalMantissa = roundShiftRightToEven(mantissa | 0x800000, -unbiasedExponent - 1);
  return sign | subnormalMantissa;
}

export function float16BitsToFloat32(bits: number): number {
  const half = bits & 0xffff;
  const sign = (half & 0x8000) === 0 ? 1 : -1;
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x03ff;
  if (exponent === 0) return sign * (mantissa === 0 ? 0 : 2 ** -14 * (mantissa / 1024));
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

export function getWgslFloat16ArrayConstructor(): WgslFloat16ArrayConstructor {
  const existing = nativeFloat16ArrayConstructor();
  return existing ?? (BrowserGradFloat16Array as unknown as WgslFloat16ArrayConstructor);
}

export function installWgslFloat16ArrayPolyfill(): WgslFloat16ArrayConstructor {
  const existing = nativeFloat16ArrayConstructor();
  if (existing) return existing;
  Object.defineProperty(globalThis, "Float16Array", {
    value: BrowserGradFloat16Array,
    writable: true,
    configurable: true,
  });
  return BrowserGradFloat16Array as unknown as WgslFloat16ArrayConstructor;
}

export function createWgslFloat16Array(
  input?: number | ArrayBufferLike | ArrayLike<number> | Iterable<number>,
  byteOffset?: number,
  length?: number,
): WgslFloat16Array {
  const Constructor = getWgslFloat16ArrayConstructor();
  if (input === undefined) return new Constructor();
  if (typeof input === "number") return new Constructor(input);
  if (isArrayBufferLike(input)) return new Constructor(input, byteOffset, length);
  return new Constructor(input);
}

export function isWgslFloat16Array(value: unknown): value is WgslFloat16Array {
  return value instanceof getWgslFloat16ArrayConstructor();
}

function nativeFloat16ArrayConstructor(): WgslFloat16ArrayConstructor | undefined {
  return (globalThis as typeof globalThis & { readonly Float16Array?: WgslFloat16ArrayConstructor }).Float16Array;
}

function isArrayBufferLike(value: unknown): value is ArrayBufferLike {
  return typeof value === "object"
    && value !== null
    && "byteLength" in value
    && typeof (value as { readonly byteLength: unknown }).byteLength === "number";
}

class BrowserGradFloat16Array implements WgslFloat16Array {
  readonly BYTES_PER_ELEMENT = 2;
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly length: number;
  private readonly view: DataView;
  [index: number]: number;

  constructor(input?: number | ArrayBufferLike | ArrayLike<number> | Iterable<number>, byteOffset = 0, length?: number) {
    if (input === undefined || typeof input === "number") {
      const elementCount = input ?? 0;
      assertNonNegativeInteger(elementCount, "Float16Array length");
      this.buffer = new ArrayBuffer(elementCount * this.BYTES_PER_ELEMENT);
      this.byteOffset = 0;
      this.length = elementCount;
    } else if (isArrayBufferLike(input)) {
      assertNonNegativeInteger(byteOffset, "Float16Array byteOffset");
      if (byteOffset % this.BYTES_PER_ELEMENT !== 0) throw new RangeError("Float16Array byteOffset must be aligned to 2 bytes");
      const availableBytes = input.byteLength - byteOffset;
      if (availableBytes < 0) throw new RangeError("Float16Array byteOffset is outside the buffer");
      const inferredLength = Math.floor(availableBytes / this.BYTES_PER_ELEMENT);
      this.length = length ?? inferredLength;
      assertNonNegativeInteger(this.length, "Float16Array length");
      if (this.length * this.BYTES_PER_ELEMENT > availableBytes) {
        throw new RangeError("Float16Array length is outside the buffer");
      }
      this.buffer = input;
      this.byteOffset = byteOffset;
    } else {
      const values = Array.from(input);
      this.buffer = new ArrayBuffer(values.length * this.BYTES_PER_ELEMENT);
      this.byteOffset = 0;
      this.length = values.length;
      this.view = new DataView(this.buffer as ArrayBuffer, this.byteOffset, this.length * this.BYTES_PER_ELEMENT);
      this.byteLength = this.length * this.BYTES_PER_ELEMENT;
      this.defineIndices();
      values.forEach((value, index) => {
        this[index] = Number(value);
      });
      return;
    }
    this.byteLength = this.length * this.BYTES_PER_ELEMENT;
    this.view = new DataView(this.buffer as ArrayBuffer, this.byteOffset, this.byteLength);
    this.defineIndices();
  }

  get [Symbol.toStringTag](): string {
    return "Float16Array";
  }

  *[Symbol.iterator](): IterableIterator<number> {
    for (let index = 0; index < this.length; index++) yield this[index] ?? 0;
  }

  slice(start?: number, end?: number): WgslFloat16Array {
    const normalizedStart = normalizeSliceIndex(start ?? 0, this.length);
    const normalizedEnd = normalizeSliceIndex(end ?? this.length, this.length);
    const values: number[] = [];
    for (let index = normalizedStart; index < normalizedEnd; index++) values.push(this[index] ?? 0);
    return new BrowserGradFloat16Array(values);
  }

  private defineIndices(): void {
    for (let index = 0; index < this.length; index++) {
      Object.defineProperty(this, index, {
        enumerable: true,
        configurable: false,
        get: () => float16BitsToFloat32(this.view.getUint16(index * this.BYTES_PER_ELEMENT, true)),
        set: (value: number) => {
          this.view.setUint16(index * this.BYTES_PER_ELEMENT, float32ToFloat16Bits(Number(value)), true);
        },
      });
    }
  }
}

function normalizeSliceIndex(index: number, length: number): number {
  const integer = Math.trunc(index);
  return integer < 0 ? Math.max(length + integer, 0) : Math.min(integer, length);
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`${name} must be a non-negative integer`);
}

function roundShiftRightToEven(value: number, shift: number): number {
  if (shift <= 0) return value;
  if (shift > 24) return 0;
  const quotient = value >>> shift;
  const remainderMask = (1 << shift) - 1;
  const remainder = value & remainderMask;
  const halfway = 1 << (shift - 1);
  return quotient + (remainder > halfway || (remainder === halfway && (quotient & 1) === 1) ? 1 : 0);
}
