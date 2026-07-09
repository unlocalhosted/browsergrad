export type InlineAsmF32Source =
  | { readonly kind: "operand"; readonly index: number }
  | { readonly kind: "immediate"; readonly value: number; readonly raw: string };

export type InlineAsmIntSource =
  | { readonly kind: "operand"; readonly index: number }
  | { readonly kind: "immediate"; readonly value: number; readonly raw: string };

export type InlineAsmFloatToIntRounding = "rn" | "rz" | "rm" | "rp";

export type InlineAsmOp =
  | { readonly kind: "fma-rn-f32"; readonly sources?: readonly [InlineAsmF32Source, InlineAsmF32Source, InlineAsmF32Source] }
  | { readonly kind: "laneid" }
  | { readonly kind: "warpid" }
  | { readonly kind: "lanemask-lt" }
  | { readonly kind: "special-register-u32"; readonly register: PtxSpecialU32Register }
  | { readonly kind: "globaltimer-u64" }
  | { readonly kind: "isspacep"; readonly space: "global" | "shared" | "const" | "local" }
  | { readonly kind: "bfind-u32"; readonly immediate?: number }
  | { readonly kind: "ffs-b32"; readonly immediate?: number }
  | { readonly kind: "popc-b32"; readonly immediate?: number }
  | { readonly kind: "clz-b32"; readonly immediate?: number }
  | { readonly kind: "brev-b32"; readonly immediate?: number }
  | { readonly kind: "prmt-b32"; readonly selectorImmediate?: number }
  | { readonly kind: "lop3-b32"; readonly immLut?: number; readonly dataImmediates?: readonly [number | undefined, number | undefined, number | undefined] }
  | { readonly kind: "bitwise-b32"; readonly op: "and" | "or" | "xor" | "not"; readonly immediate?: number }
  | { readonly kind: "shift-b32"; readonly op: "shl" | "shr"; readonly signed: boolean; readonly immediate?: number }
  | { readonly kind: "arithmetic-b32"; readonly op: "add" | "sub" | "mul-lo" | "mad-lo"; readonly signed: boolean; readonly immediate?: number }
  | { readonly kind: "minmax-b32"; readonly op: "min" | "max"; readonly signed: boolean; readonly immediate?: number }
  | { readonly kind: "unary-int-b32"; readonly op: "neg" | "abs"; readonly signed: boolean; readonly immediate?: number }
  | { readonly kind: "select-b32"; readonly signed: boolean; readonly trueImmediate?: number; readonly falseImmediate?: number }
  | { readonly kind: "compare-b32"; readonly op: "eq" | "ne" | "lt" | "le" | "gt" | "ge"; readonly signed: boolean; readonly immediate?: number }
  | { readonly kind: "move-b32"; readonly signed: boolean; readonly immediate?: number }
  | { readonly kind: "convert-b32"; readonly fromSigned: boolean; readonly toSigned: boolean; readonly immediate?: number }
  | { readonly kind: "convert-f32-to-int"; readonly rounding: InlineAsmFloatToIntRounding; readonly toSigned: boolean; readonly source?: InlineAsmF32Source }
  | { readonly kind: "convert-int-to-f32"; readonly fromSigned: boolean; readonly source?: InlineAsmIntSource }
  | { readonly kind: "float-binary-rn-f32"; readonly op: "add" | "sub" | "mul" | "div"; readonly sources?: readonly [InlineAsmF32Source, InlineAsmF32Source] }
  | { readonly kind: "u8x4-sad-add" }
  | { readonly kind: "cp-async-fence"; readonly fence: "commit_group" | "wait_group" | "wait_all" }
  | { readonly kind: "membar"; readonly scope: "cta" | "gl" | "sys" }
  | { readonly kind: "bar-sync"; readonly operand: "literal0" | "input0" }
  | {
    readonly kind: "ldmatrix";
    readonly matrices: 1 | 2 | 4;
    readonly transposed: boolean;
  }
  | {
    readonly kind: "mma-m16n8k16";
    readonly accumulator: "f16" | "f32";
  };

export type PtxSpecialU32Register =
  | "tid.x"
  | "tid.y"
  | "tid.z"
  | "ctaid.x"
  | "ctaid.y"
  | "ctaid.z"
  | "ntid.x"
  | "ntid.y"
  | "ntid.z"
  | "nctaid.x"
  | "nctaid.y"
  | "nctaid.z";

export function classifyInlineAsm(template: string): InlineAsmOp | undefined {
  if (/\bmov\.u32\b/u.test(template) && /%{1,2}laneid\b/u.test(template)) return { kind: "laneid" };
  if (/\bmov\.u32\b/u.test(template) && /%{1,2}warpid\b/u.test(template)) return { kind: "warpid" };
  if (/\bmov\.u32\b/u.test(template) && /%{1,2}lanemask_lt\b/u.test(template)) return { kind: "lanemask-lt" };
  const special = /\bmov\.u32\b[\s\S]*%{1,2}((?:tid|ctaid|ntid|nctaid)\.[xyz])\b/u.exec(template);
  if (special) return { kind: "special-register-u32", register: special[1] as PtxSpecialU32Register };
  if (/\bmov\.u64\b/u.test(template) && /%globaltimer\b/u.test(template)) return { kind: "globaltimer-u64" };
  const isspacep = /\bisspacep\.(global|shared|const|local)\b/u.exec(template);
  if (isspacep) return { kind: "isspacep", space: isspacep[1] as "global" | "shared" | "const" | "local" };
  const bfindImmediate = /\bbfind\.u32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (bfindImmediate) return { kind: "bfind-u32", immediate: parseInlineAsmImmediate(bfindImmediate[1]!) >>> 0 };
  if (/\bbfind\.u32\b/u.test(template)) return { kind: "bfind-u32" };
  const ffsImmediate = /\bffs\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (ffsImmediate) return { kind: "ffs-b32", immediate: parseInlineAsmImmediate(ffsImmediate[1]!) >>> 0 };
  if (/\bffs\.b32\b/u.test(template)) return { kind: "ffs-b32" };
  const popcImmediate = /\bpopc\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (popcImmediate) return { kind: "popc-b32", immediate: parseInlineAsmImmediate(popcImmediate[1]!) >>> 0 };
  if (/\bpopc\.b32\b/u.test(template)) return { kind: "popc-b32" };
  const clzImmediate = /\bclz\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (clzImmediate) return { kind: "clz-b32", immediate: parseInlineAsmImmediate(clzImmediate[1]!) >>> 0 };
  if (/\bclz\.b32\b/u.test(template)) return { kind: "clz-b32" };
  const brevImmediate = /\bbrev\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (brevImmediate) return { kind: "brev-b32", immediate: parseInlineAsmImmediate(brevImmediate[1]!) >>> 0 };
  if (/\bbrev\.b32\b/u.test(template)) return { kind: "brev-b32" };
  const prmtImmediate = /\bprmt\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (prmtImmediate) return { kind: "prmt-b32", selectorImmediate: parseInlineAsmImmediate(prmtImmediate[1]!) >>> 0 };
  if (/\bprmt\.b32\b/u.test(template)) return { kind: "prmt-b32" };
  const lop3 = /\blop3\.b32\b\s+([^;]+)/u.exec(template);
  if (lop3) {
    const operands = lop3[1]!.split(",").map((operand) => operand.trim());
    const dataImmediates = [
      parseInlineAsmImmediateOperand(operands[1]),
      parseInlineAsmImmediateOperand(operands[2]),
      parseInlineAsmImmediateOperand(operands[3]),
    ] as const;
    const immLut = parseInlineAsmImmediateOperand(operands[4]);
    return {
      kind: "lop3-b32",
      ...(immLut === undefined ? {} : { immLut: immLut & 0xff }),
      ...(dataImmediates.some((value) => value !== undefined) ? { dataImmediates } : {}),
    };
  }
  if (/\blop3\.b32\b/u.test(template)) return { kind: "lop3-b32" };
  const bitwiseImmediate = /\b(and|or|xor)\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (bitwiseImmediate) return { kind: "bitwise-b32", op: bitwiseImmediate[1] as "and" | "or" | "xor", immediate: parseInlineAsmImmediate(bitwiseImmediate[2]!) >>> 0 };
  const notImmediate = /\bnot\.b32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (notImmediate) return { kind: "bitwise-b32", op: "not", immediate: parseInlineAsmImmediate(notImmediate[1]!) >>> 0 };
  const bitwise = /\b(and|or|xor|not)\.b32\b/u.exec(template);
  if (bitwise) return { kind: "bitwise-b32", op: bitwise[1] as "and" | "or" | "xor" | "not" };
  const shiftImmediate = /\b(shl|shr)\.(b32|u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (shiftImmediate) return { kind: "shift-b32", op: shiftImmediate[1] as "shl" | "shr", signed: shiftImmediate[2] === "s32", immediate: parseInlineAsmImmediate(shiftImmediate[3]!) >>> 0 };
  const shift = /\b(shl|shr)\.(b32|u32|s32)\b/u.exec(template);
  if (shift) return { kind: "shift-b32", op: shift[1] as "shl" | "shr", signed: shift[2] === "s32" };
  const arithmeticImmediate = /\b(add|sub)\.(b32|u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (arithmeticImmediate) return { kind: "arithmetic-b32", op: arithmeticImmediate[1] as "add" | "sub", signed: arithmeticImmediate[2] === "s32", immediate: parseInlineAsmImmediate(arithmeticImmediate[3]!) >>> 0 };
  const arithmetic = /\b(add|sub)\.(b32|u32|s32)\b/u.exec(template);
  if (arithmetic) return { kind: "arithmetic-b32", op: arithmetic[1] as "add" | "sub", signed: arithmetic[2] === "s32" };
  const mulLoImmediate = /\bmul\.lo\.(b32|u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (mulLoImmediate) return { kind: "arithmetic-b32", op: "mul-lo", signed: mulLoImmediate[1] === "s32", immediate: parseInlineAsmImmediate(mulLoImmediate[2]!) >>> 0 };
  const mulLo = /\bmul\.lo\.(b32|u32|s32)\b/u.exec(template);
  if (mulLo) return { kind: "arithmetic-b32", op: "mul-lo", signed: mulLo[1] === "s32" };
  const madLoImmediate = /\bmad\.lo\.(b32|u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (madLoImmediate) return { kind: "arithmetic-b32", op: "mad-lo", signed: madLoImmediate[1] === "s32", immediate: parseInlineAsmImmediate(madLoImmediate[2]!) >>> 0 };
  const madLo = /\bmad\.lo\.(b32|u32|s32)\b/u.exec(template);
  if (madLo) return { kind: "arithmetic-b32", op: "mad-lo", signed: madLo[1] === "s32" };
  const minmaxImmediate = /\b(min|max)\.(u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (minmaxImmediate) return { kind: "minmax-b32", op: minmaxImmediate[1] as "min" | "max", signed: minmaxImmediate[2] === "s32", immediate: parseInlineAsmImmediate(minmaxImmediate[3]!) >>> 0 };
  const minmax = /\b(min|max)\.(u32|s32)\b/u.exec(template);
  if (minmax) return { kind: "minmax-b32", op: minmax[1] as "min" | "max", signed: minmax[2] === "s32" };
  const negImmediate = /\bneg\.(b32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (negImmediate) return { kind: "unary-int-b32", op: "neg", signed: negImmediate[1] === "s32", immediate: parseInlineAsmImmediate(negImmediate[2]!) >>> 0 };
  const neg = /\bneg\.(b32|s32)\b/u.exec(template);
  if (neg) return { kind: "unary-int-b32", op: "neg", signed: neg[1] === "s32" };
  const absImmediate = /\babs\.s32\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (absImmediate) return { kind: "unary-int-b32", op: "abs", signed: true, immediate: parseInlineAsmImmediate(absImmediate[1]!) >>> 0 };
  if (/\babs\.s32\b/u.test(template)) return { kind: "unary-int-b32", op: "abs", signed: true };
  const selp = /\bselp\.(b32|u32|s32)\b\s+([^;]+)/u.exec(template);
  if (selp) {
    const operands = selp[2]!.split(",").map((operand) => operand.trim());
    const trueImmediate = parseInlineAsmImmediateOperand(operands[1]);
    const falseImmediate = parseInlineAsmImmediateOperand(operands[2]);
    return {
      kind: "select-b32",
      signed: selp[1] === "s32",
      ...(trueImmediate === undefined ? {} : { trueImmediate }),
      ...(falseImmediate === undefined ? {} : { falseImmediate }),
    };
  }
  const setpImmediate = /\bsetp\.(eq|ne|lt|le|gt|ge)\.(u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (setpImmediate) return { kind: "compare-b32", op: setpImmediate[1] as "eq" | "ne" | "lt" | "le" | "gt" | "ge", signed: setpImmediate[2] === "s32", immediate: parseInlineAsmImmediate(setpImmediate[3]!) >>> 0 };
  const setp = /\bsetp\.(eq|ne|lt|le|gt|ge)\.(u32|s32)\b/u.exec(template);
  if (setp) return { kind: "compare-b32", op: setp[1] as "eq" | "ne" | "lt" | "le" | "gt" | "ge", signed: setp[2] === "s32" };
  const movImmediate = /\bmov\.(b32|u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?/u.exec(template);
  if (movImmediate) return { kind: "move-b32", signed: movImmediate[1] === "s32", immediate: parseInlineAsmImmediate(movImmediate[2]!) >>> 0 };
  const mov = /\bmov\.(b32|u32|s32)\b/u.exec(template);
  if (mov) return { kind: "move-b32", signed: mov[1] === "s32" };
  const cvtImmediate = /\bcvt\.(u32|s32)\.(u32|s32)\b[\s\S]*,\s*(0x[0-9a-fA-F]+|-?\d+)\s*;?\s*$/u.exec(template);
  if (cvtImmediate) return { kind: "convert-b32", toSigned: cvtImmediate[1] === "s32", fromSigned: cvtImmediate[2] === "s32", immediate: parseInlineAsmImmediate(cvtImmediate[3]!) >>> 0 };
  const cvt = /\bcvt\.(u32|s32)\.(u32|s32)\b/u.exec(template);
  if (cvt) return { kind: "convert-b32", toSigned: cvt[1] === "s32", fromSigned: cvt[2] === "s32" };
  const cvtFloat = /\bcvt\.(rni|rzi|rmi|rpi|rn|rz|rm|rp)\.(u32|s32)\.f32\b\s+([^;]+)/u.exec(template);
  if (cvtFloat) {
    const operands = cvtFloat[3]!.split(",").map((operand) => operand.trim());
    const source = parseInlineAsmF32SourceOperand(operands[1]);
    const rawRounding = cvtFloat[1]!;
    return {
      kind: "convert-f32-to-int",
      rounding: rawRounding.slice(0, 2) as InlineAsmFloatToIntRounding,
      toSigned: cvtFloat[2] === "s32",
      ...(source === undefined ? {} : { source }),
    };
  }
  const cvtIntToFloat = /\bcvt\.rn\.f32\.(u32|s32)\b\s+([^;]+)/u.exec(template);
  if (cvtIntToFloat) {
    const operands = cvtIntToFloat[2]!.split(",").map((operand) => operand.trim());
    const source = parseInlineAsmIntSourceOperand(operands[1]);
    return {
      kind: "convert-int-to-f32",
      fromSigned: cvtIntToFloat[1] === "s32",
      ...(source === undefined ? {} : { source }),
    };
  }
  const floatBinary = /\b(add|sub|mul|div)\.rn\.f32\b\s+([^;]+)/u.exec(template);
  if (floatBinary) {
    const operands = floatBinary[2]!.split(",").map((operand) => operand.trim());
    const sources = [
      parseInlineAsmF32SourceOperand(operands[1]),
      parseInlineAsmF32SourceOperand(operands[2]),
    ] as const;
    return sources.every((source) => source !== undefined)
      ? { kind: "float-binary-rn-f32", op: floatBinary[1] as "add" | "sub" | "mul" | "div", sources: sources as readonly [InlineAsmF32Source, InlineAsmF32Source] }
      : { kind: "float-binary-rn-f32", op: floatBinary[1] as "add" | "sub" | "mul" | "div" };
  }
  if (/\bvabsdiff4\.u32\.u32\.u32\.add\b/u.test(template)) return { kind: "u8x4-sad-add" };
  const cpAsyncFence = /\bcp\.async\.(commit_group|wait_group|wait_all)\b/u.exec(template);
  if (cpAsyncFence) return { kind: "cp-async-fence", fence: cpAsyncFence[1] as "commit_group" | "wait_group" | "wait_all" };
  const membar = /\bmembar\.(cta|gl|sys)\b/u.exec(template);
  if (membar) return { kind: "membar", scope: membar[1] as "cta" | "gl" | "sys" };
  const barSync = /\bbar\.sync\s+(0|%0)\s*;/u.exec(template);
  if (barSync) return { kind: "bar-sync", operand: barSync[1] === "%0" ? "input0" : "literal0" };
  const fma = /\bfma\.rn\.f32\b\s+([^;]+)/u.exec(template);
  if (fma) {
    const operands = fma[1]!.split(",").map((operand) => operand.trim());
    const sources = [
      parseInlineAsmF32SourceOperand(operands[1]),
      parseInlineAsmF32SourceOperand(operands[2]),
      parseInlineAsmF32SourceOperand(operands[3]),
    ] as const;
    return sources.every((source) => source !== undefined)
      ? { kind: "fma-rn-f32", sources: sources as readonly [InlineAsmF32Source, InlineAsmF32Source, InlineAsmF32Source] }
      : { kind: "fma-rn-f32" };
  }
  if (/\bfma\.rn\.f32\b/u.test(template)) return { kind: "fma-rn-f32" };
  const ldmatrix = /\bldmatrix\.sync\.aligned\.x([124])(\.trans)?\.m8n8\.shared\.b16\b/u.exec(template);
  if (ldmatrix) {
    return {
      kind: "ldmatrix",
      matrices: Number(ldmatrix[1]) as 1 | 2 | 4,
      transposed: ldmatrix[2] !== undefined,
    };
  }
  const mma = /\bmma\.sync\.aligned\.m16n8k16\.row\.col\.(f16|f32)\.f16\.f16\.(f16|f32)\b/u.exec(template);
  if (mma) {
    return {
      kind: "mma-m16n8k16",
      accumulator: mma[1] === "f32" || mma[2] === "f32" ? "f32" : "f16",
    };
  }
  return undefined;
}

export function inlineAsmSupportedList(): string {
  return [
    "fma.rn.f32",
    "laneid",
    "warpid",
    "lanemask_lt",
    "{tid,ctaid,ntid,nctaid}.{x,y,z}",
    "globaltimer",
    "isspacep.{global,shared,const,local}",
    "bfind.u32",
    "ffs.b32",
    "popc.b32",
    "clz.b32",
    "brev.b32",
    "prmt.b32",
    "lop3.b32",
    "{and,or,xor,not}.b32",
    "shl.{b32,u32,s32}",
    "shr.{b32,u32,s32}",
    "add.{b32,u32,s32}",
    "sub.{b32,u32,s32}",
    "mul.lo.{b32,u32,s32}",
    "mad.lo.{b32,u32,s32}",
    "{min,max}.{u32,s32}",
    "neg.{b32,s32}",
    "abs.s32",
    "selp.{b32,u32,s32}",
    "setp.{eq,ne,lt,le,gt,ge}.{u32,s32}",
    "mov.{b32,u32,s32}",
    "cvt.{u32,s32}.{u32,s32}",
    "cvt.{rni,rzi,rmi,rpi}.{u32,s32}.f32",
    "cvt.rn.f32.{u32,s32}",
    "{add,sub,mul,div}.rn.f32",
    "vabsdiff4.u32.u32.u32.add",
    "cp.async.{commit_group,wait_group,wait_all}",
    "membar.{cta,gl,sys}",
    "bar.sync 0",
    "ldmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16",
    "mma.sync.aligned.m16n8k16.row.col.{f16,f32}.f16.f16.{f16,f32}",
  ].join(", ");
}

function parseInlineAsmImmediate(value: string): number {
  return value.startsWith("0x") || value.startsWith("0X") ? Number.parseInt(value, 16) : Number.parseInt(value, 10);
}

function parseInlineAsmImmediateOperand(value: string | undefined): number | undefined {
  if (value === undefined || !/^(?:0x[0-9a-fA-F]+|-?\d+)$/u.test(value)) return undefined;
  return parseInlineAsmImmediate(value) >>> 0;
}

function parseInlineAsmIntSourceOperand(value: string | undefined): InlineAsmIntSource | undefined {
  if (value === undefined) return undefined;
  const operand = /^%(\d+)$/u.exec(value);
  if (operand) return { kind: "operand", index: Number.parseInt(operand[1]!, 10) };
  if (!/^(?:0x[0-9a-fA-F]+|-?\d+)$/u.test(value)) return undefined;
  return { kind: "immediate", value: parseInlineAsmImmediate(value) >>> 0, raw: value };
}

function parseInlineAsmF32SourceOperand(value: string | undefined): InlineAsmF32Source | undefined {
  if (value === undefined) return undefined;
  const operand = /^%(\d+)$/u.exec(value);
  if (operand) return { kind: "operand", index: Number.parseInt(operand[1]!, 10) };
  if (!/^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?[fF]?$/u.test(value)) return undefined;
  const parsed = Number.parseFloat(value.replace(/[fF]$/u, ""));
  return Number.isFinite(parsed) ? { kind: "immediate", value: parsed, raw: value } : undefined;
}
