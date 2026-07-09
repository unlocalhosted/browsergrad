import type { InlineAsmFloatToIntRounding } from "./model.js";

export function emitInlineBytePermWgsl(x: string, y: string, selector: string): string {
  const lanes = [0, 1, 2, 3].map((lane) => {
    const control = `((${selector} >> ${lane * 4}u) & 0xfu)`;
    const source = `(${control} & 0x7u)`;
    const shift = `((${source} & 0x3u) * 8u)`;
    const byte = `(select((${x} >> ${shift}), (${y} >> ${shift}), (${source} >= 4u)) & 0xffu)`;
    const sign = `select(0u, 0xffu, ((${byte} & 0x80u) != 0u))`;
    return `(select(${byte}, ${sign}, ((${control} & 0x8u) != 0u)) << ${lane * 8}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

export function emitInlineLop3Wgsl(a: string, b: string, c: string, lut: string): string {
  const rows = Array.from({ length: 8 }, (_, row) => {
    const aMask = (row & 0x4) === 0 ? `(~${a})` : a;
    const bMask = (row & 0x2) === 0 ? `(~${b})` : b;
    const cMask = (row & 0x1) === 0 ? `(~${c})` : c;
    const rowMask = `(${aMask} & ${bMask} & ${cMask})`;
    return `select(0u, ${rowMask}, ((${lut} & ${1 << row}u) != 0u))`;
  });
  return `(${rows.join(" | ")})`;
}

export function emitInlineBitwiseWgsl(op: "and" | "or" | "xor" | "not", left: string, right: string): string {
  if (op === "not") return `(~${left})`;
  const operator = op === "and" ? "&" : op === "or" ? "|" : "^";
  return `(${left} ${operator} ${right})`;
}

export function emitInlineShiftWgsl(op: "shl" | "shr", value: string, amount: string, signed: boolean): string {
  const clamped = `min(${amount}, 31u)`;
  if (op === "shl") return `select((${value} << ${clamped}), 0u, (${amount} >= 32u))`;
  if (!signed) return `select((${value} >> ${clamped}), 0u, (${amount} >= 32u))`;
  return `u32(i32(${value}) >> ${clamped})`;
}

export function emitInlineArithmeticWgsl(op: "add" | "sub" | "mul-lo" | "mad-lo", left: string, right: string, addend = "0u"): string {
  if (op === "add") return `(${left} + ${right})`;
  if (op === "sub") return `(${left} - ${right})`;
  if (op === "mad-lo") return `((${left} * ${right}) + ${addend})`;
  return `(${left} * ${right})`;
}

export function emitInlineMinMaxWgsl(op: "min" | "max", left: string, right: string, signed: boolean): string {
  if (!signed) return `${op}(${left}, ${right})`;
  return `bitcast<u32>(${op}(bitcast<i32>(${left}), bitcast<i32>(${right})))`;
}

export function emitInlineUnaryIntWgsl(op: "neg" | "abs", value: string): string {
  if (op === "neg") return `(0u - ${value})`;
  const mask = `select(0u, 0xffffffffu, ((${value} & 0x80000000u) != 0u))`;
  return `((${value} ^ ${mask}) - ${mask})`;
}

export function emitInlineSelectWgsl(trueValue: string, falseValue: string, predicate: string): string {
  return `select(${falseValue}, ${trueValue}, (${predicate} != 0u))`;
}

export function emitInlineCompareWgsl(op: "eq" | "ne" | "lt" | "le" | "gt" | "ge", left: string, right: string): string {
  const operator = op === "eq" ? "==" : op === "ne" ? "!=" : op === "lt" ? "<" : op === "le" ? "<=" : op === "gt" ? ">" : ">=";
  return `select(0u, 1u, (${left} ${operator} ${right}))`;
}

export function emitInlineF32ToIntConvertWgsl(source: string, rounding: InlineAsmFloatToIntRounding, toSigned: boolean, targetIsUint: boolean): string {
  const rounded = rounding === "rn"
    ? `bg_round_even_f32(${source})`
    : rounding === "rz"
    ? `trunc(${source})`
    : rounding === "rm"
    ? `floor(${source})`
    : `ceil(${source})`;
  if (toSigned && !targetIsUint) return `i32(clamp(${rounded}, -2147483648.0, 2147483520.0))`;
  if (toSigned) return `bitcast<u32>(i32(clamp(${rounded}, -2147483648.0, 2147483520.0)))`;
  return `u32(clamp(${rounded}, 0.0, 4294967040.0))`;
}

export function emitU8x4SadAddWgsl(a: string, b: string, c: string): string {
  const lanes = [0, 8, 16, 24].map((shift) => {
    const left = `((${a} >> ${shift}u) & 0xffu)`;
    const right = `((${b} >> ${shift}u) & 0xffu)`;
    return `(max(${left}, ${right}) - min(${left}, ${right}))`;
  });
  return `(${c} + ${lanes.join(" + ")})`;
}
