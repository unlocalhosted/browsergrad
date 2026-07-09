import {
  expectedInlineAsmF32SourceInputs,
  expectedInlineAsmSourceInputs,
  type InlineAsmF32Source,
  type InlineAsmIntSource,
  type InlineAsmOp,
} from "./model.js";

export type InlineAsmOutputValueType = "bool" | "float" | "int" | "uint";
export type InlineAsmInputValueType = "float" | "int" | "uint";

export interface InlineAsmOutputValueContract {
  readonly label: string;
  readonly description: string;
  readonly allowed: readonly InlineAsmOutputValueType[];
  readonly allOutputs?: boolean;
}

export interface InlineAsmInputValueContract {
  readonly inputIndex: number;
  readonly label: string;
  readonly description: string;
  readonly allowed: readonly InlineAsmInputValueType[];
}

export function inlineAsmExpectedInputCount(op: InlineAsmOp, outputCount: number): number | undefined {
  switch (op.kind) {
    case "fma-rn-f32":
      return expectedInlineAsmF32SourceInputs(op.sources, outputCount);
    case "float-binary-rn-f32":
      return expectedInlineAsmF32SourceInputs(op.sources, outputCount);
    case "laneid":
    case "warpid":
    case "lanemask-lt":
    case "special-register-u32":
    case "globaltimer-u64":
    case "membar":
      return 0;
    case "isspacep":
      return 1;
    case "bfind-u32":
    case "ffs-b32":
    case "popc-b32":
    case "clz-b32":
    case "brev-b32":
    case "unary-int-b32":
    case "move-b32":
    case "convert-b32":
      return op.immediate === undefined ? 1 : 0;
    case "prmt-b32":
      return op.selectorImmediate === undefined ? 3 : 2;
    case "lop3-b32": {
      const dataImmediateCount = op.dataImmediates?.filter((value) => value !== undefined).length ?? 0;
      return 3 - dataImmediateCount + (op.immLut === undefined ? 1 : 0);
    }
    case "bitwise-b32":
      return op.op === "not" ? (op.immediate === undefined ? 1 : 0) : (op.immediate === undefined ? 2 : 1);
    case "shift-b32":
    case "minmax-b32":
    case "compare-b32":
      return op.immediate === undefined ? 2 : 1;
    case "arithmetic-b32":
      return (op.op === "mad-lo" ? 3 : 2) - (op.immediate === undefined ? 0 : 1);
    case "select-b32":
      return 3 - (op.trueImmediate === undefined ? 0 : 1) - (op.falseImmediate === undefined ? 0 : 1);
    case "convert-f32-to-int":
      return op.source === undefined ? 1 : expectedInlineAsmF32SourceInputs([op.source], outputCount);
    case "convert-int-to-f32":
      return op.source === undefined ? 1 : expectedInlineAsmSourceInputs([op.source], outputCount);
    case "u8x4-sad-add":
      return 3;
    case "cp-async-fence":
      return op.fence === "wait_group" ? 1 : 0;
    case "bar-sync":
      return op.operand === "input0" ? 1 : 0;
    case "ldmatrix":
      return 1;
    case "mma-m16n8k16":
      return op.accumulator === "f32" ? 10 : 8;
  }
}

export function inlineAsmInputCountMatches(op: InlineAsmOp, outputCount: number, actualInputCount: number): boolean {
  if (op.kind === "cp-async-fence" && op.fence === "wait_group") return actualInputCount <= 1;
  const expected = inlineAsmExpectedInputCount(op, outputCount);
  return expected !== undefined && actualInputCount === expected;
}

export function inlineAsmExpectedOutputCount(op: InlineAsmOp): number {
  switch (op.kind) {
    case "cp-async-fence":
    case "membar":
    case "bar-sync":
      return 0;
    case "ldmatrix":
      return op.matrices;
    case "mma-m16n8k16":
      return op.accumulator === "f32" ? 4 : 2;
    default:
      return 1;
  }
}

export function inlineAsmOutputCountMatches(op: InlineAsmOp, actualOutputCount: number): boolean {
  return actualOutputCount === inlineAsmExpectedOutputCount(op);
}

export function inlineAsmOperandShapeDiagnostic(
  op: InlineAsmOp,
  actualOutputCount: number,
  actualInputCount: number,
): string | undefined {
  const expectedInputs = inlineAsmExpectedInputCount(op, actualOutputCount);
  switch (op.kind) {
    case "fma-rn-f32":
      return actualOutputCount === 1 && expectedInputs !== undefined && actualInputCount === expectedInputs
        ? undefined
        : `fma.rn.f32 inline PTX expects one output operand and ${expectedInputs ?? 2} input operands`;
    case "float-binary-rn-f32":
      return actualOutputCount === 1 && expectedInputs !== undefined && actualInputCount === expectedInputs
        ? undefined
        : `${op.op}.rn.f32 inline PTX expects one output operand and ${expectedInputs ?? 2} input operands`;
    case "laneid":
      return actualOutputCount === 1 && actualInputCount === 0 ? undefined : "laneid inline PTX expects one output operand and no input operands";
    case "warpid":
      return actualOutputCount === 1 && actualInputCount === 0 ? undefined : "warpid inline PTX expects one output operand and no input operands";
    case "lanemask-lt":
      return actualOutputCount === 1 && actualInputCount === 0 ? undefined : "lanemask_lt inline PTX expects one output operand and no input operands";
    case "special-register-u32":
      return actualOutputCount === 1 && actualInputCount === 0 ? undefined : `${op.register} inline PTX expects one output operand and no input operands`;
    case "globaltimer-u64":
      return actualOutputCount === 1 && actualInputCount === 0 ? undefined : "globaltimer inline PTX expects one output operand and no input operands";
    case "isspacep":
      return actualOutputCount === 1 && actualInputCount === 1 ? undefined : `isspacep.${op.space} inline PTX expects one output operand and one pointer input operand`;
    case "bfind-u32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `bfind.u32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "ffs-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `ffs.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "popc-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `popc.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "clz-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `clz.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "brev-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `brev.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "prmt-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `prmt.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "lop3-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `lop3.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "bitwise-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `${op.op}.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "shift-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `${op.op}.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "arithmetic-b32": {
      const opLabel = op.op === "mul-lo" ? "mul.lo" : op.op === "mad-lo" ? "mad.lo" : op.op;
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `${opLabel}.b32 inline PTX expects one output operand and ${expectedInputs} input operands`;
    }
    case "minmax-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `${op.op}.${op.signed ? "s32" : "u32"} inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "unary-int-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `${op.op}.${op.op === "abs" || op.signed ? "s32" : "b32"} inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "select-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `selp.${op.signed ? "s32" : "b32"} inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "compare-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `setp.${op.op}.${op.signed ? "s32" : "u32"} inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "move-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs
        ? undefined
        : `mov.${op.signed ? "s32" : "b32"} inline PTX expects one output operand and ${expectedInputs === 0 ? "no input operands" : "one input operand"}`;
    case "convert-b32":
      return actualOutputCount === 1 && actualInputCount === expectedInputs ? undefined : `cvt.${op.toSigned ? "s32" : "u32"}.${op.fromSigned ? "s32" : "u32"} inline PTX expects one output operand and ${expectedInputs} input operands`;
    case "convert-f32-to-int":
      return actualOutputCount === 1 && expectedInputs !== undefined && actualInputCount === expectedInputs
        ? undefined
        : `cvt.${op.rounding}i.${op.toSigned ? "s32" : "u32"}.f32 inline PTX expects one output operand and ${expectedInputs ?? 1} input operands`;
    case "convert-int-to-f32":
      return actualOutputCount === 1 && expectedInputs !== undefined && actualInputCount === expectedInputs
        ? undefined
        : `cvt.rn.f32.${op.fromSigned ? "s32" : "u32"} inline PTX expects one output operand and ${expectedInputs ?? 1} input operands`;
    case "u8x4-sad-add":
      return actualOutputCount === 1 && actualInputCount === 3 ? undefined : "vabsdiff4.u32.u32.u32.add inline PTX expects one output operand and three input operands";
    case "cp-async-fence": {
      const maxInputs = op.fence === "wait_group" ? 1 : 0;
      return actualOutputCount === 0 && actualInputCount <= maxInputs
        ? undefined
        : `${op.fence === "wait_group" ? "cp.async.wait_group" : "cp.async fence"} inline PTX expects no output operands${maxInputs === 0 ? " and no input operands" : " and at most one input operand"}`;
    }
    case "membar":
      return actualOutputCount === 0 && actualInputCount === 0 ? undefined : `membar.${op.scope} inline PTX expects no output or input operands`;
    case "bar-sync":
      return actualOutputCount === 0 && actualInputCount === expectedInputs ? undefined : `bar.sync inline PTX expects no output operands and ${expectedInputs} input operand(s)`;
    case "ldmatrix":
      return actualOutputCount === op.matrices && actualInputCount === 1 ? undefined : `ldmatrix.x${op.matrices} inline PTX expects ${op.matrices} output operand(s) and one shared-address input operand`;
    case "mma-m16n8k16": {
      const expectedOutputs = op.accumulator === "f32" ? 4 : 2;
      const expectedMmaInputs = op.accumulator === "f32" ? 10 : 8;
      return actualOutputCount === expectedOutputs && actualInputCount === expectedMmaInputs
        ? undefined
        : `mma.m16n8k16 ${op.accumulator} inline PTX expects ${expectedOutputs} output operand(s) and ${expectedMmaInputs} input operands`;
    }
  }
}

export function inlineAsmOutputValueContract(op: InlineAsmOp): InlineAsmOutputValueContract | undefined {
  switch (op.kind) {
    case "fma-rn-f32":
      return { label: "fma.rn.f32", description: "an f32 output operand", allowed: ["float"] };
    case "float-binary-rn-f32":
      return { label: `${op.op}.rn.f32`, description: "an f32 output operand", allowed: ["float"] };
    case "laneid":
      return { label: "laneid", description: "an integer output operand", allowed: ["uint", "int"] };
    case "warpid":
      return { label: "warpid", description: "an integer output operand", allowed: ["uint", "int"] };
    case "lanemask-lt":
      return { label: "lanemask_lt", description: "an integer output operand", allowed: ["uint", "int"] };
    case "special-register-u32":
      return { label: op.register, description: "an integer output operand", allowed: ["uint", "int"] };
    case "globaltimer-u64":
      return { label: "globaltimer", description: "an integer output operand", allowed: ["uint", "int"] };
    case "isspacep":
      return { label: `isspacep.${op.space}`, description: "an integer predicate output operand", allowed: ["uint", "int", "bool"] };
    case "bfind-u32":
      return { label: "bfind.u32", description: "a uint output operand", allowed: ["uint"] };
    case "ffs-b32":
      return { label: "ffs.b32", description: "an integer output operand", allowed: ["uint", "int"] };
    case "popc-b32":
      return { label: "popc.b32", description: "an integer output operand", allowed: ["uint", "int"] };
    case "clz-b32":
      return { label: "clz.b32", description: "an integer output operand", allowed: ["uint", "int"] };
    case "brev-b32":
      return { label: "brev.b32", description: "an integer output operand", allowed: ["uint", "int"] };
    case "prmt-b32":
      return { label: "prmt.b32", description: "an integer output operand", allowed: ["uint", "int"] };
    case "lop3-b32":
      return { label: "lop3.b32", description: "an integer output operand", allowed: ["uint", "int"] };
    case "bitwise-b32":
      return { label: `${op.op}.b32`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "shift-b32":
      return { label: `${op.op}.b32`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "arithmetic-b32": {
      const label = op.op === "mul-lo" ? "mul.lo.b32" : op.op === "mad-lo" ? "mad.lo.b32" : `${op.op}.b32`;
      return { label, description: "an integer output operand", allowed: ["uint", "int"] };
    }
    case "minmax-b32":
      return { label: `${op.op}.${op.signed ? "s32" : "u32"}`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "unary-int-b32":
      return { label: `${op.op}.${op.op === "abs" || op.signed ? "s32" : "b32"}`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "select-b32":
      return { label: `selp.${op.signed ? "s32" : "b32"}`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "compare-b32":
      return { label: `setp.${op.op}.${op.signed ? "s32" : "u32"}`, description: "an integer predicate output operand", allowed: ["uint", "int", "bool"] };
    case "move-b32":
      return { label: `mov.${op.signed ? "s32" : "b32"}`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "convert-b32":
      return { label: `cvt.${op.toSigned ? "s32" : "u32"}.${op.fromSigned ? "s32" : "u32"}`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "convert-f32-to-int":
      return { label: `cvt.${op.rounding}i.${op.toSigned ? "s32" : "u32"}.f32`, description: "an integer output operand", allowed: ["uint", "int"] };
    case "convert-int-to-f32":
      return { label: `cvt.rn.f32.${op.fromSigned ? "s32" : "u32"}`, description: "an f32 output operand", allowed: ["float"] };
    case "u8x4-sad-add":
      return { label: "vabsdiff4.u32.u32.u32.add", description: "an integer output operand", allowed: ["uint", "int"] };
    case "ldmatrix":
      return { label: "ldmatrix", description: "integer register carrier operands", allowed: ["uint", "int"], allOutputs: true };
    default:
      return undefined;
  }
}

export function inlineAsmOutputValueTypeMatches(contract: InlineAsmOutputValueContract, valueType: string | undefined): boolean {
  return valueType === undefined || contract.allowed.includes(valueType as InlineAsmOutputValueType);
}

export function inlineAsmInputValueContracts(op: InlineAsmOp, outputCount: number): readonly InlineAsmInputValueContract[] {
  switch (op.kind) {
    case "float-binary-rn-f32": {
      const sources = op.sources ?? [
        { kind: "operand", index: outputCount },
        { kind: "operand", index: outputCount + 1 },
      ] satisfies readonly [InlineAsmF32Source, InlineAsmF32Source];
      return inputContractsForSources(sources, outputCount, {
        label: `${op.op}.rn.f32`,
        description: "f32 input operands",
        allowed: ["float"],
      });
    }
    case "convert-f32-to-int":
      return op.source === undefined
        ? [{ inputIndex: 0, label: `cvt.${op.rounding}i.${op.toSigned ? "s32" : "u32"}.f32`, description: "an f32 input operand", allowed: ["float"] }]
        : inputContractsForSources([op.source], outputCount, {
          label: `cvt.${op.rounding}i.${op.toSigned ? "s32" : "u32"}.f32`,
          description: "an f32 input operand",
          allowed: ["float"],
        });
    case "convert-int-to-f32":
      return op.source === undefined
        ? [{ inputIndex: 0, label: `cvt.rn.f32.${op.fromSigned ? "s32" : "u32"}`, description: "an integer input operand", allowed: ["uint", "int"] }]
        : inputContractsForSources([op.source], outputCount, {
          label: `cvt.rn.f32.${op.fromSigned ? "s32" : "u32"}`,
          description: "an integer input operand",
          allowed: ["uint", "int"],
        });
    default:
      return [];
  }
}

export function inlineAsmInputValueTypeMatches(contract: InlineAsmInputValueContract, valueType: string | undefined): boolean {
  return valueType === undefined || contract.allowed.includes(valueType as InlineAsmInputValueType);
}

function inputContractsForSources(
  sources: readonly (InlineAsmF32Source | InlineAsmIntSource)[],
  outputCount: number,
  contract: Omit<InlineAsmInputValueContract, "inputIndex">,
): readonly InlineAsmInputValueContract[] {
  return sources.flatMap((source) => {
    if (source.kind !== "operand" || source.index < outputCount) return [];
    return [{ ...contract, inputIndex: source.index - outputCount }];
  });
}
