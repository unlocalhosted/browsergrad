interface Float16Array extends Uint16Array {}

interface Float16ArrayConstructor {
  readonly prototype: Float16Array;
  readonly BYTES_PER_ELEMENT: 2;
  new(): Float16Array;
  new(length: number): Float16Array;
  new(array: ArrayLike<number> | Iterable<number>): Float16Array;
  new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float16Array;
}

declare const Float16Array: Float16ArrayConstructor;
