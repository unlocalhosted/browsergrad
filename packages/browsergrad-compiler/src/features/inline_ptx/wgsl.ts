import type { InlineAsmFloatToIntRounding } from "./model.js";

export interface InlineAsmWgslExpressionCallbacks<TExpression, TContext> {
  readonly emitExpression: (expression: TExpression, context: TContext) => string;
  readonly emitNumberLiteral: (literal: string) => string;
  readonly expressionValueTypeForEmit: (expression: TExpression, context: TContext) => string | undefined;
}

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

export function emitInlineBytePermExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  selectorImmediate?: number,
): string {
  const x = `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const y = `u32(${callbacks.emitExpression(inputs[1]!, context)})`;
  const selector = selectorImmediate === undefined ? `u32(${callbacks.emitExpression(inputs[2]!, context)})` : `${selectorImmediate >>> 0}u`;
  return emitInlineBytePermWgsl(x, y, selector);
}

export function emitInlineLop3ExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  immLut: number | undefined,
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  dataImmediates: readonly [number | undefined, number | undefined, number | undefined] = [undefined, undefined, undefined],
): string {
  let inputIndex = 0;
  const a = dataImmediates[0] === undefined ? `u32(${callbacks.emitExpression(inputs[inputIndex++]!, context)})` : `${dataImmediates[0] >>> 0}u`;
  const b = dataImmediates[1] === undefined ? `u32(${callbacks.emitExpression(inputs[inputIndex++]!, context)})` : `${dataImmediates[1] >>> 0}u`;
  const c = dataImmediates[2] === undefined ? `u32(${callbacks.emitExpression(inputs[inputIndex++]!, context)})` : `${dataImmediates[2] >>> 0}u`;
  const lut = immLut === undefined ? `(u32(${callbacks.emitExpression(inputs[inputIndex]!, context)}) & 0xffu)` : `${immLut & 0xff}u`;
  return emitInlineLop3Wgsl(a, b, c, lut);
}

export function emitInlineBitwiseExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  op: "and" | "or" | "xor" | "not",
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  immediate?: number,
): string {
  const left = op === "not" && immediate !== undefined ? `${immediate >>> 0}u` : `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const right = op === "not" ? "0u" : immediate === undefined ? `u32(${callbacks.emitExpression(inputs[1]!, context)})` : `${immediate >>> 0}u`;
  return emitInlineBitwiseWgsl(op, left, right);
}

export function emitInlineShiftExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  op: "shl" | "shr",
  signed: boolean,
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  immediate?: number,
): string {
  const value = `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const amount = immediate === undefined ? `u32(${callbacks.emitExpression(inputs[1]!, context)})` : `${immediate >>> 0}u`;
  return emitInlineShiftWgsl(op, value, amount, signed);
}

export function emitInlineArithmeticExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  op: "add" | "sub" | "mul-lo" | "mad-lo",
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  immediate?: number,
): string {
  const left = `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const right = immediate !== undefined && op !== "mad-lo" ? `${immediate >>> 0}u` : `u32(${callbacks.emitExpression(inputs[1]!, context)})`;
  const addend = op === "mad-lo"
    ? immediate === undefined ? `u32(${callbacks.emitExpression(inputs[2]!, context)})` : `${immediate >>> 0}u`
    : "0u";
  return emitInlineArithmeticWgsl(op, left, right, addend);
}

export function emitInlineMinMaxExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  op: "min" | "max",
  signed: boolean,
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  immediate?: number,
): string {
  const left = `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const right = immediate === undefined ? `u32(${callbacks.emitExpression(inputs[1]!, context)})` : `${immediate >>> 0}u`;
  return emitInlineMinMaxWgsl(op, left, right, signed);
}

export function emitInlineUnaryIntExpressionWgsl<TExpression, TContext>(
  input: TExpression | undefined,
  op: "neg" | "abs",
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  immediate?: number,
): string {
  const value = immediate === undefined ? `u32(${callbacks.emitExpression(input!, context)})` : `${immediate >>> 0}u`;
  return emitInlineUnaryIntWgsl(op, value);
}

export function emitInlineSelectExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  trueImmediate?: number,
  falseImmediate?: number,
): string {
  let inputIndex = 0;
  const trueValue = trueImmediate === undefined ? `u32(${callbacks.emitExpression(inputs[inputIndex++]!, context)})` : `${trueImmediate >>> 0}u`;
  const falseValue = falseImmediate === undefined ? `u32(${callbacks.emitExpression(inputs[inputIndex++]!, context)})` : `${falseImmediate >>> 0}u`;
  const predicate = `u32(${callbacks.emitExpression(inputs[inputIndex]!, context)})`;
  return emitInlineSelectWgsl(trueValue, falseValue, predicate);
}

export function emitInlineCompareExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  op: "eq" | "ne" | "lt" | "le" | "gt" | "ge",
  signed: boolean,
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
  immediate?: number,
): string {
  const left = signed ? `bitcast<i32>(u32(${callbacks.emitExpression(inputs[0]!, context)}))` : `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const rightU32 = immediate === undefined ? `u32(${callbacks.emitExpression(inputs[1]!, context)})` : `${immediate >>> 0}u`;
  const right = signed ? `bitcast<i32>(${rightU32})` : rightU32;
  return emitInlineCompareWgsl(op, left, right);
}

export function emitInlineF32ToIntConvertExpressionWgsl<TExpression, TContext>(
  source: string,
  rounding: InlineAsmFloatToIntRounding,
  toSigned: boolean,
  target: TExpression,
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
): string {
  const targetType = callbacks.expressionValueTypeForEmit(target, context);
  return emitInlineF32ToIntConvertWgsl(source, rounding, toSigned, targetType === "uint");
}

export function emitU8x4SadAddExpressionWgsl<TExpression, TContext>(
  inputs: readonly TExpression[],
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
): string {
  const a = `u32(${callbacks.emitExpression(inputs[0]!, context)})`;
  const b = `u32(${callbacks.emitExpression(inputs[1]!, context)})`;
  const c = `u32(${callbacks.emitExpression(inputs[2]!, context)})`;
  return emitU8x4SadAddWgsl(a, b, c);
}
