import {
  classifyInlineAsm,
  inlineAsmSupportedList,
  type InlineAsmF32Source,
  type InlineAsmFloatToIntRounding,
  type InlineAsmIntSource,
} from "./model.js";
import { inlineAsmExpectedInputCount, inlineAsmInputCountMatches, inlineAsmOutputCountMatches } from "./validation.js";

export interface InlineAsmWgslExpressionCallbacks<TExpression, TContext> {
  readonly emitExpression: (expression: TExpression, context: TContext) => string;
  readonly emitNumberLiteral: (literal: string) => string;
  readonly expressionValueTypeForEmit: (expression: TExpression, context: TContext) => string | undefined;
}

export interface InlineAsmWgslStatement<TExpression, TSpan> {
  readonly template: string;
  readonly output?: TExpression;
  readonly outputs?: readonly TExpression[];
  readonly inputs: readonly TExpression[];
  readonly span: TSpan;
}

export interface InlineAsmWgslStatementCallbacks<TExpression, TContext, TSpan> extends InlineAsmWgslExpressionCallbacks<TExpression, TContext> {
  readonly emitExpressionAsValueType: (expression: TExpression, valueType: "float", context: TContext) => string;
  readonly emitLocalLinearRank: (context: TContext) => string;
  readonly emitU32Output: (target: TExpression, expression: string, context: TContext) => string;
  readonly emitSpecialRegister: (register: string, context: TContext) => string;
  readonly emitAddressPredicate: (space: "global" | "shared" | "const" | "local", expression: TExpression, context: TContext) => string;
  readonly emitGlobalTimerTick: (context: TContext) => string;
  readonly wgslScalarTypeForExpression: (expression: TExpression, context: TContext) => string | undefined;
  readonly featureError: (code: string, message: string, span: TSpan) => unknown;
}

export function emitInlineAsmStatementWgsl<TExpression, TContext, TSpan>(
  statement: InlineAsmWgslStatement<TExpression, TSpan>,
  context: TContext,
  callbacks: InlineAsmWgslStatementCallbacks<TExpression, TContext, TSpan>,
): string {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  const inputCountMatches = op === undefined ? false : inlineAsmInputCountMatches(op, outputs.length, statement.inputs.length);
  const outputCountMatches = op === undefined ? false : inlineAsmOutputCountMatches(op, outputs.length);
  if (op?.kind === "laneid" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `u32(${callbacks.emitLocalLinearRank(context)} % 32)`, context)}`;
  }
  if (op?.kind === "warpid" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `u32(${callbacks.emitLocalLinearRank(context)} / 32)`, context)}`;
  }
  if (op?.kind === "lanemask-lt" && inputCountMatches && outputCountMatches) {
    const lane = `u32(${callbacks.emitLocalLinearRank(context)} & 31)`;
    const mask = `select(0u, ((1u << ${lane}) - 1u), ${lane} > 0u)`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, mask, context)}`;
  }
  if (op?.kind === "special-register-u32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, callbacks.emitSpecialRegister(op.register, context), context)}`;
  }
  if (op?.kind === "globaltimer-u64" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, callbacks.emitGlobalTimerTick(context), context)}`;
  }
  if (op?.kind === "isspacep" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, callbacks.emitAddressPredicate(op.space, statement.inputs[0]!, context), context)}`;
  }
  if (op?.kind === "bfind-u32" && inputCountMatches && outputCountMatches) {
    const value = op.immediate === undefined ? `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})` : `${op.immediate >>> 0}u`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = (31u - countLeadingZeros(${value}))`;
  }
  if (op?.kind === "ffs-b32" && inputCountMatches && outputCountMatches) {
    const value = op.immediate === undefined ? `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})` : `${op.immediate >>> 0}u`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `select(0u, (countTrailingZeros(${value}) + 1u), (${value} != 0u))`, context)}`;
  }
  if (op?.kind === "popc-b32" && inputCountMatches && outputCountMatches) {
    const value = op.immediate === undefined ? `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})` : `${op.immediate >>> 0}u`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `countOneBits(${value})`, context)}`;
  }
  if (op?.kind === "clz-b32" && inputCountMatches && outputCountMatches) {
    const value = op.immediate === undefined ? `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})` : `${op.immediate >>> 0}u`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `countLeadingZeros(${value})`, context)}`;
  }
  if (op?.kind === "brev-b32" && inputCountMatches && outputCountMatches) {
    const value = op.immediate === undefined ? `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})` : `${op.immediate >>> 0}u`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `reverseBits(${value})`, context)}`;
  }
  if (op?.kind === "prmt-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineBytePermExpressionWgsl(statement.inputs, context, callbacks, op.selectorImmediate), context)}`;
  }
  if (op?.kind === "lop3-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineLop3ExpressionWgsl(statement.inputs, op.immLut, context, callbacks, op.dataImmediates), context)}`;
  }
  if (op?.kind === "bitwise-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineBitwiseExpressionWgsl(statement.inputs, op.op, context, callbacks, op.immediate), context)}`;
  }
  if (op?.kind === "shift-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineShiftExpressionWgsl(statement.inputs, op.op, op.signed, context, callbacks, op.immediate), context)}`;
  }
  if (op?.kind === "arithmetic-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineArithmeticExpressionWgsl(statement.inputs, op.op, context, callbacks, op.immediate), context)}`;
  }
  if (op?.kind === "minmax-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineMinMaxExpressionWgsl(statement.inputs, op.op, op.signed, context, callbacks, op.immediate), context)}`;
  }
  if (op?.kind === "unary-int-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineUnaryIntExpressionWgsl(statement.inputs[0], op.op, context, callbacks, op.immediate), context)}`;
  }
  if (op?.kind === "select-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineSelectExpressionWgsl(statement.inputs, context, callbacks, op.trueImmediate, op.falseImmediate), context)}`;
  }
  if (op?.kind === "compare-b32" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitInlineCompareExpressionWgsl(statement.inputs, op.op, op.signed, context, callbacks, op.immediate), context)}`;
  }
  if (op?.kind === "move-b32" && inputCountMatches && op.immediate === undefined && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})`, context)}`;
  }
  if (op?.kind === "move-b32" && inputCountMatches && op.immediate !== undefined && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, `${op.immediate >>> 0}u`, context)}`;
  }
  if (op?.kind === "convert-b32" && inputCountMatches && outputCountMatches) {
    const value = op.immediate === undefined ? `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})` : `${op.immediate >>> 0}u`;
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, value, context)}`;
  }
  if (op?.kind === "convert-f32-to-int" && inputCountMatches && outputCountMatches) {
    const source = op.source === undefined
      ? callbacks.emitExpressionAsValueType(statement.inputs[0]!, "float", context)
      : emitInlineAsmF32SourceWgsl(op.source, statement, outputs, context, callbacks);
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${emitInlineF32ToIntConvertExpressionWgsl(source, op.rounding, op.toSigned, outputs[0]!, context, callbacks)}`;
  }
  if (op?.kind === "convert-int-to-f32" && inputCountMatches && outputCountMatches) {
    const source = op.source === undefined
      ? callbacks.emitExpression(statement.inputs[0]!, context)
      : emitInlineAsmIntSourceWgsl(op.source, statement, outputs, context, callbacks);
    return `${callbacks.emitExpression(outputs[0]!, context)} = f32(${op.fromSigned ? `i32(${source})` : `u32(${source})`})`;
  }
  if (op?.kind === "float-binary-rn-f32") {
    const sources = op.sources ?? [
      { kind: "operand", index: outputs.length },
      { kind: "operand", index: outputs.length + 1 },
    ] satisfies readonly [InlineAsmF32Source, InlineAsmF32Source];
    if (inputCountMatches && outputCountMatches) {
      const left = emitInlineAsmF32SourceWgsl(sources[0]!, statement, outputs, context, callbacks);
      const right = emitInlineAsmF32SourceWgsl(sources[1]!, statement, outputs, context, callbacks);
      const operator = op.op === "add" ? "+" : op.op === "sub" ? "-" : op.op === "mul" ? "*" : "/";
      return `${callbacks.emitExpression(outputs[0]!, context)} = (${left} ${operator} ${right})`;
    }
  }
  if (op?.kind === "u8x4-sad-add" && inputCountMatches && outputCountMatches) {
    return `${callbacks.emitExpression(outputs[0]!, context)} = ${callbacks.emitU32Output(outputs[0]!, emitU8x4SadAddExpressionWgsl(statement.inputs, context, callbacks), context)}`;
  }
  if (op?.kind === "ldmatrix" && inputCountMatches && outputCountMatches) {
    const base = `u32(${callbacks.emitExpression(statement.inputs[0]!, context)})`;
    const tag = op.transposed ? "0x80000000u" : "0u";
    return outputs.map((output, index) => {
      const carrier = `(${tag} + ${base} + ${index * 2}u)`;
      return `${callbacks.emitExpression(output, context)} = ${callbacks.emitU32Output(output, carrier, context)}`;
    }).join("\n");
  }
  if (op?.kind === "cp-async-fence" && inputCountMatches && outputCountMatches) return `// cp.async inline asm fence omitted`;
  if (op?.kind === "membar" && inputCountMatches && outputCountMatches) return `storageBarrier();`;
  if (op?.kind === "bar-sync" && inputCountMatches && outputCountMatches) return `workgroupBarrier();`;
  if (op?.kind === "mma-m16n8k16") return emitMmaM16N8K16StatementWgsl(statement, outputs, op.accumulator, context, callbacks);
  const fmaSources = op?.kind === "fma-rn-f32"
    ? op.sources ?? [
      { kind: "operand", index: outputs.length },
      { kind: "operand", index: outputs.length + 1 },
      { kind: "operand", index: 0 },
    ] satisfies readonly [InlineAsmF32Source, InlineAsmF32Source, InlineAsmF32Source]
    : undefined;
  const expectedFmaInputs = op?.kind === "fma-rn-f32" ? inlineAsmExpectedInputCount(op, outputs.length) : undefined;
  if (op?.kind !== "fma-rn-f32" || fmaSources === undefined || expectedFmaInputs === undefined || statement.inputs.length !== expectedFmaInputs || !outputCountMatches) {
    throw callbacks.featureError("unsupported-inline-asm", `only ${inlineAsmSupportedList()} inline PTX are supported in WGSL output`, statement.span);
  }
  const target = callbacks.emitExpression(outputs[0]!, context);
  const [a, b, c] = fmaSources.map((source) => emitInlineAsmF32SourceWgsl(source, statement, outputs, context, callbacks));
  return `${target} = fma(${a}, ${b}, ${c})`;
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

function emitInlineAsmIntSourceWgsl<TExpression, TContext, TSpan>(
  source: InlineAsmIntSource,
  statement: InlineAsmWgslStatement<TExpression, TSpan>,
  outputs: readonly TExpression[],
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
): string {
  if (source.kind === "immediate") return source.raw.startsWith("-") ? `${source.value | 0}` : `${source.value >>> 0}u`;
  if (source.index < outputs.length) return callbacks.emitExpression(outputs[source.index]!, context);
  return callbacks.emitExpression(statement.inputs[source.index - outputs.length]!, context);
}

function emitInlineAsmF32SourceWgsl<TExpression, TContext, TSpan>(
  source: InlineAsmF32Source,
  statement: InlineAsmWgslStatement<TExpression, TSpan>,
  outputs: readonly TExpression[],
  context: TContext,
  callbacks: InlineAsmWgslExpressionCallbacks<TExpression, TContext>,
): string {
  if (source.kind === "immediate") return callbacks.emitNumberLiteral(source.raw);
  if (source.index < outputs.length) return callbacks.emitExpression(outputs[source.index]!, context);
  return callbacks.emitExpression(statement.inputs[source.index - outputs.length]!, context);
}

function emitMmaM16N8K16StatementWgsl<TExpression, TContext, TSpan>(
  statement: InlineAsmWgslStatement<TExpression, TSpan>,
  outputs: readonly TExpression[],
  accumulator: "f16" | "f32",
  context: TContext,
  callbacks: InlineAsmWgslStatementCallbacks<TExpression, TContext, TSpan>,
): string {
  if (accumulator === "f16") {
    if (outputs.length !== 2 || statement.inputs.length !== 8) {
      throw callbacks.featureError("invalid-inline-asm-operands", "mma.m16n8k16 f16 inline PTX operand mismatch", statement.span);
    }
    return outputs.map((output, index) => {
      const a = `u32(${callbacks.emitExpression(statement.inputs[index % 4]!, context)})`;
      const b = `u32(${callbacks.emitExpression(statement.inputs[4 + (index % 2)]!, context)})`;
      const c = `u32(${callbacks.emitExpression(statement.inputs[6 + index]!, context)})`;
      const value = `pack2x16float(unpack2x16float(${c}) + (unpack2x16float(${a}) * unpack2x16float(${b})))`;
      return `${callbacks.emitExpression(output, context)} = ${callbacks.emitU32Output(output, value, context)}`;
    }).join("\n");
  }
  if (outputs.length !== 4 || statement.inputs.length !== 10) {
    throw callbacks.featureError("invalid-inline-asm-operands", "mma.m16n8k16 f32 inline PTX operand mismatch", statement.span);
  }
  return outputs.map((output, index) => {
    const a = `u32(${callbacks.emitExpression(statement.inputs[index % 4]!, context)})`;
    const b = `u32(${callbacks.emitExpression(statement.inputs[4 + (index % 2)]!, context)})`;
    const c = emitMmaF32AccumulatorInputWgsl(statement.inputs[6 + index]!, context, callbacks);
    const value = `(${c} + dot(unpack2x16float(${a}), unpack2x16float(${b})))`;
    return `${callbacks.emitExpression(output, context)} = ${emitMmaF32AccumulatorOutputWgsl(output, value, context, callbacks)}`;
  }).join("\n");
}

function emitMmaF32AccumulatorInputWgsl<TExpression, TContext, TSpan>(
  expression: TExpression,
  context: TContext,
  callbacks: InlineAsmWgslStatementCallbacks<TExpression, TContext, TSpan>,
): string {
  const value = callbacks.emitExpression(expression, context);
  const scalar = callbacks.wgslScalarTypeForExpression(expression, context);
  if (scalar === "u32") return `bitcast<f32>(${value})`;
  if (scalar === "i32") return `bitcast<f32>(u32(${value}))`;
  if (scalar === "f16") return `f32(${value})`;
  return value;
}

function emitMmaF32AccumulatorOutputWgsl<TExpression, TContext, TSpan>(
  target: TExpression,
  value: string,
  context: TContext,
  callbacks: InlineAsmWgslStatementCallbacks<TExpression, TContext, TSpan>,
): string {
  const scalar = callbacks.wgslScalarTypeForExpression(target, context);
  if (scalar === "u32") return `bitcast<u32>(${value})`;
  if (scalar === "i32") return `bitcast<i32>(${value})`;
  if (scalar === "f16") return `f16(${value})`;
  return value;
}
