export type InlineAsmOp =
  | { readonly kind: "fma-rn-f32" }
  | { readonly kind: "laneid" }
  | { readonly kind: "lanemask-lt" }
  | { readonly kind: "bfind-u32" }
  | { readonly kind: "u8x4-sad-add" }
  | {
    readonly kind: "ldmatrix";
    readonly matrices: 1 | 2 | 4;
    readonly transposed: boolean;
  }
  | {
    readonly kind: "mma-m16n8k16";
    readonly accumulator: "f16" | "f32";
  };

export function classifyInlineAsm(template: string): InlineAsmOp | undefined {
  if (/\bmov\.u32\b/u.test(template) && /%%laneid\b/u.test(template)) return { kind: "laneid" };
  if (/\bmov\.u32\b/u.test(template) && /%%lanemask_lt\b/u.test(template)) return { kind: "lanemask-lt" };
  if (/\bbfind\.u32\b/u.test(template)) return { kind: "bfind-u32" };
  if (/\bvabsdiff4\.u32\.u32\.u32\.add\b/u.test(template)) return { kind: "u8x4-sad-add" };
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
    "lanemask_lt",
    "bfind.u32",
    "vabsdiff4.u32.u32.u32.add",
    "ldmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16",
    "mma.sync.aligned.m16n8k16.row.col.{f16,f32}.f16.f16.{f16,f32}",
  ].join(", ");
}
