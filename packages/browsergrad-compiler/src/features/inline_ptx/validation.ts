import {
  expectedInlineAsmF32SourceInputs,
  expectedInlineAsmSourceInputs,
  type InlineAsmOp,
} from "./model.js";

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
