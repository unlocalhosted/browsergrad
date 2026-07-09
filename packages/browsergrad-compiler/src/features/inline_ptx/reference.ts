import type { InlineAsmFloatToIntRounding } from "./model.js";

export function evalInlineAsmBytePerm(x: number, y: number, selector: number): number {
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const control = (selector >>> (lane * 4)) & 0xf;
    const source = control & 0x7;
    const input = source < 4 ? x : y;
    const byte = (input >>> ((source & 3) * 8)) & 0xff;
    const value = (control & 0x8) === 0 ? byte : (byte & 0x80) === 0 ? 0 : 0xff;
    out |= value << (lane * 8);
  }
  return out >>> 0;
}

export function evalInlineAsmLop3(a: number, b: number, c: number, immLut: number): number {
  let out = 0;
  for (let row = 0; row < 8; row++) {
    if (((immLut >>> row) & 1) === 0) continue;
    const aMask = (row & 0x4) === 0 ? ~a : a;
    const bMask = (row & 0x2) === 0 ? ~b : b;
    const cMask = (row & 0x1) === 0 ? ~c : c;
    out |= aMask & bMask & cMask;
  }
  return out >>> 0;
}

export function evalInlineAsmBitwise(op: "and" | "or" | "xor" | "not", left: number, right: number): number {
  if (op === "and") return (left & right) >>> 0;
  if (op === "or") return (left | right) >>> 0;
  if (op === "xor") return (left ^ right) >>> 0;
  return (~left) >>> 0;
}

export function evalInlineAsmShift(op: "shl" | "shr", value: number, amount: number, signed: boolean): number {
  if (amount >= 32) {
    if (op === "shr" && signed) return (value & 0x80000000) === 0 ? 0 : 0xffffffff;
    return 0;
  }
  if (op === "shl") return (value << amount) >>> 0;
  if (signed) return (value >> amount) >>> 0;
  return value >>> amount;
}

export function evalInlineAsmArithmetic(op: "add" | "sub" | "mul-lo" | "mad-lo", left: number, right: number, addend = 0): number {
  if (op === "add") return (left + right) >>> 0;
  if (op === "sub") return (left - right) >>> 0;
  const product = Math.imul(left, right) >>> 0;
  return op === "mad-lo" ? (product + addend) >>> 0 : product;
}

export function evalInlineAsmMinMax(op: "min" | "max", left: number, right: number, signed: boolean): number {
  const leftValue = signed ? left | 0 : left >>> 0;
  const rightValue = signed ? right | 0 : right >>> 0;
  const takeLeft = op === "min" ? leftValue <= rightValue : leftValue >= rightValue;
  return (takeLeft ? left : right) >>> 0;
}

export function evalInlineAsmUnaryInt(op: "neg" | "abs", value: number): number {
  if (op === "neg") return (0 - value) >>> 0;
  const mask = (value & 0x80000000) === 0 ? 0 : 0xffffffff;
  return ((value ^ mask) - mask) >>> 0;
}

export function evalInlineAsmCompare(op: "eq" | "ne" | "lt" | "le" | "gt" | "ge", left: number, right: number, signed: boolean): boolean {
  const leftValue = signed ? left | 0 : left >>> 0;
  const rightValue = signed ? right | 0 : right >>> 0;
  if (op === "eq") return leftValue === rightValue;
  if (op === "ne") return leftValue !== rightValue;
  if (op === "lt") return leftValue < rightValue;
  if (op === "le") return leftValue <= rightValue;
  if (op === "gt") return leftValue > rightValue;
  return leftValue >= rightValue;
}

export function evalInlineAsmFloatToInt(value: number, rounding: InlineAsmFloatToIntRounding, signed: boolean): number {
  const rounded = rounding === "rn"
    ? roundTiesToEvenNumber(value)
    : rounding === "rz"
    ? Math.trunc(value)
    : rounding === "rm"
    ? Math.floor(value)
    : Math.ceil(value);
  if (signed) return Math.max(-2147483648, Math.min(2147483520, rounded)) | 0;
  return Math.max(0, Math.min(4294967040, rounded)) >>> 0;
}

function roundTiesToEvenNumber(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
