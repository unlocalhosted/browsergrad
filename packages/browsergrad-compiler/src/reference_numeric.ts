import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from "@unlocalhosted/browsergrad-kernels";

export function referenceSignedAverage(xValue: number, yValue: number): number {
  const x = BigInt(Math.trunc(xValue) | 0);
  const y = BigInt(Math.trunc(yValue) | 0);
  return Number((x + y) >> 1n) | 0;
}

export function referenceEvalI8x4DotAdd(aValue: number, bValue: number, addValue = 0): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = Math.trunc(addValue) | 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    out = (out + Math.imul(signExtend8((a >>> shift) & 0xff), signExtend8((b >>> shift) & 0xff))) | 0;
  }
  return out;
}

export function referenceEvalU8x4DotAdd(aValue: number, bValue: number, addValue = 0): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  let out = Math.trunc(addValue) >>> 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    out = (out + (((a >>> shift) & 0xff) * ((b >>> shift) & 0xff))) >>> 0;
  }
  return out;
}

export function referenceEvalI16x2I8x2DotAdd(aValue: number, bValue: number, addValue = 0, byteShift: 0 | 16): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const left0 = signExtend16(a & 0xffff);
  const left1 = signExtend16((a >>> 16) & 0xffff);
  const right0 = signExtend8((b >>> byteShift) & 0xff);
  const right1 = signExtend8((b >>> (byteShift + 8)) & 0xff);
  return ((Math.trunc(addValue) | 0) + Math.imul(left0, right0) + Math.imul(left1, right1)) | 0;
}

export function referenceEvalU16x2U8x2DotAdd(aValue: number, bValue: number, addValue = 0, byteShift: 0 | 16): number {
  const a = Math.trunc(aValue) >>> 0;
  const b = Math.trunc(bValue) >>> 0;
  const left0 = a & 0xffff;
  const left1 = (a >>> 16) & 0xffff;
  const right0 = (b >>> byteShift) & 0xff;
  const right1 = (b >>> (byteShift + 8)) & 0xff;
  return ((Math.trunc(addValue) >>> 0) + (left0 * right0) + (left1 * right1)) >>> 0;
}

export function referenceRoundHalf(value: number): number {
  return float16BitsToFloat32(float32ToFloat16Bits(value));
}

export function referenceSaturateHalf(value: number): number {
  return Number.isNaN(value) ? 0 : Math.min(1, Math.max(0, value));
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

export function referenceRoundBfloat16(value: number): number {
  f32Scratch[0] = value;
  const bits = u32Scratch[0] ?? 0;
  u32Scratch[0] = (bits + 0x8000) & 0xffff0000;
  return f32Scratch[0] ?? 0;
}

export function referenceSaturateBfloat16(value: number): number {
  return Number.isNaN(value) ? 0 : referenceRoundBfloat16(Math.min(1, Math.max(0, value)));
}

export function referenceReluBfloat16(value: number): number {
  return Number.isNaN(value) ? referenceRoundBfloat16(Number.NaN) : referenceRoundBfloat16(Math.max(0, value));
}

function signExtend8(value: number): number {
  return (value << 24) >> 24;
}

function signExtend16(value: number): number {
  return (value << 16) >> 16;
}
