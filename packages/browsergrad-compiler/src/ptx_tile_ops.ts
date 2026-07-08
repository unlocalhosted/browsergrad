export type InlineAsmOp =
  | { readonly kind: "fma-rn-f32" }
  | { readonly kind: "laneid" }
  | { readonly kind: "warpid" }
  | { readonly kind: "lanemask-lt" }
  | { readonly kind: "globaltimer-u64" }
  | { readonly kind: "isspacep"; readonly space: "global" | "shared" | "const" | "local" }
  | { readonly kind: "bfind-u32" }
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

export function classifyInlineAsm(template: string): InlineAsmOp | undefined {
  if (/\bmov\.u32\b/u.test(template) && /%{1,2}laneid\b/u.test(template)) return { kind: "laneid" };
  if (/\bmov\.u32\b/u.test(template) && /%{1,2}warpid\b/u.test(template)) return { kind: "warpid" };
  if (/\bmov\.u32\b/u.test(template) && /%{1,2}lanemask_lt\b/u.test(template)) return { kind: "lanemask-lt" };
  if (/\bmov\.u64\b/u.test(template) && /%globaltimer\b/u.test(template)) return { kind: "globaltimer-u64" };
  const isspacep = /\bisspacep\.(global|shared|const|local)\b/u.exec(template);
  if (isspacep) return { kind: "isspacep", space: isspacep[1] as "global" | "shared" | "const" | "local" };
  if (/\bbfind\.u32\b/u.test(template)) return { kind: "bfind-u32" };
  if (/\bvabsdiff4\.u32\.u32\.u32\.add\b/u.test(template)) return { kind: "u8x4-sad-add" };
  const cpAsyncFence = /\bcp\.async\.(commit_group|wait_group|wait_all)\b/u.exec(template);
  if (cpAsyncFence) return { kind: "cp-async-fence", fence: cpAsyncFence[1] as "commit_group" | "wait_group" | "wait_all" };
  const membar = /\bmembar\.(cta|gl|sys)\b/u.exec(template);
  if (membar) return { kind: "membar", scope: membar[1] as "cta" | "gl" | "sys" };
  const barSync = /\bbar\.sync\s+(0|%0)\s*;/u.exec(template);
  if (barSync) return { kind: "bar-sync", operand: barSync[1] === "%0" ? "input0" : "literal0" };
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
    "globaltimer",
    "isspacep.{global,shared,const,local}",
    "bfind.u32",
    "vabsdiff4.u32.u32.u32.add",
    "cp.async.{commit_group,wait_group,wait_all}",
    "membar.{cta,gl,sys}",
    "bar.sync 0",
    "ldmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16",
    "mma.sync.aligned.m16n8k16.row.col.{f16,f32}.f16.f16.{f16,f32}",
  ].join(", ");
}
