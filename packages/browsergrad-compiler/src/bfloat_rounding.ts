export type BfloatRoundingMode = "rn" | "rz" | "ru" | "rd";

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

export function roundFloat32ToBfloat16(value: number, mode: BfloatRoundingMode = "rn"): number {
  u32Scratch[0] = roundFloat32ToBfloat16Bits(value, mode);
  u32Scratch[0] <<= 16;
  return f32Scratch[0] ?? 0;
}

export function roundFloat32ToBfloat16Bits(value: number, mode: BfloatRoundingMode = "rn"): number {
  f32Scratch[0] = value;
  const bits = u32Scratch[0] ?? 0;
  const exponent = bits & 0x7f800000;
  if (exponent === 0x7f800000) return (bits >>> 16) & 0xffff;

  const lower = bits & 0xffff;
  if (mode === "rn") return (((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff) >>> 0;

  let high = (bits >>> 16) & 0xffff;
  if (lower !== 0) {
    const negative = (bits & 0x80000000) !== 0;
    if ((mode === "ru" && !negative) || (mode === "rd" && negative)) high += 1;
  }
  return high & 0xffff;
}

export function bfloat16BitsToFloat32(bits: number): number {
  u32Scratch[0] = (Math.trunc(bits) & 0xffff) << 16;
  return f32Scratch[0] ?? 0;
}
