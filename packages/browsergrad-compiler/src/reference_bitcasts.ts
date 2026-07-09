const REFERENCE_BITCAST_BUFFER = new ArrayBuffer(4);
const REFERENCE_BITCAST_FLOAT = new Float32Array(REFERENCE_BITCAST_BUFFER);
const REFERENCE_BITCAST_UINT = new Uint32Array(REFERENCE_BITCAST_BUFFER);
const REFERENCE_BITCAST_INT = new Int32Array(REFERENCE_BITCAST_BUFFER);

export function referenceFloat32ToUintBits(value: number): number {
  REFERENCE_BITCAST_FLOAT[0] = value;
  return REFERENCE_BITCAST_UINT[0] ?? 0;
}

export function referenceUintBitsToFloat32(value: number): number {
  REFERENCE_BITCAST_UINT[0] = Math.trunc(value) >>> 0;
  return REFERENCE_BITCAST_FLOAT[0] ?? 0;
}

export function referenceUintBitsToInt32(value: number): number {
  REFERENCE_BITCAST_UINT[0] = Math.trunc(value) >>> 0;
  return REFERENCE_BITCAST_INT[0] ?? 0;
}
