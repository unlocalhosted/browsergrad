import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from "@unlocalhosted/browsergrad-kernels";

export type HalfRoundingMode = "rn" | "rz" | "ru" | "rd";

export function roundFloat32ToFloat16(value: number, mode: HalfRoundingMode): number {
  return float16BitsToFloat32(roundFloat32ToFloat16Bits(value, mode));
}

export function roundFloat32ToFloat16Bits(value: number, mode: HalfRoundingMode): number {
  const nearest = float32ToFloat16Bits(value) & 0xffff;
  if (mode === "rn" || Number.isNaN(value)) return nearest;
  const rounded = float16BitsToFloat32(nearest);
  if (mode === "rz") {
    if (value > 0 && rounded > value) return previousHalfBits(nearest);
    if (value < 0 && rounded < value) return nextHalfBits(nearest);
    return nearest;
  }
  if (mode === "ru") return rounded < value ? nextHalfBits(nearest) : nearest;
  return rounded > value ? previousHalfBits(nearest) : nearest;
}

function nextHalfBits(bits: number): number {
  const value = bits & 0xffff;
  if ((value & 0x7fff) > 0x7c00) return value;
  if (value === 0x7c00) return value;
  if (value === 0xfc00) return 0xfbff;
  if ((value & 0x8000) !== 0) return value === 0x8000 ? 0 : value - 1;
  return value + 1;
}

function previousHalfBits(bits: number): number {
  const value = bits & 0xffff;
  if ((value & 0x7fff) > 0x7c00) return value;
  if (value === 0xfc00) return value;
  if (value === 0x7c00) return 0x7bff;
  if ((value & 0x8000) !== 0) return value + 1;
  return value === 0 ? 0x8000 : value - 1;
}
