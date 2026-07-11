export type SemanticPtxIntegerCallInfo =
  | { readonly family: "arithmetic"; readonly op: "add" | "sub" | "mul-lo" | "mad-lo" }
  | { readonly family: "shift"; readonly op: "shl" | "shr"; readonly signed: boolean }
  | { readonly family: "minmax"; readonly op: "min" | "max"; readonly signed: boolean }
  | { readonly family: "unary"; readonly op: "neg" | "abs" };

export function semanticPtxIntegerCallInfo(name: string): SemanticPtxIntegerCallInfo | undefined {
  const arithmetic = /^__bg_ptx_arithmetic_(add|sub|mul_lo|mad_lo)$/u.exec(name)?.[1];
  if (arithmetic) return { family: "arithmetic", op: arithmetic.replace("_", "-") as "add" | "sub" | "mul-lo" | "mad-lo" };
  const shift = /^__bg_ptx_shift_(shl|shr)_([su])$/u.exec(name);
  if (shift) return { family: "shift", op: shift[1] as "shl" | "shr", signed: shift[2] === "s" };
  const minmax = /^__bg_ptx_(min|max)_([su])$/u.exec(name);
  if (minmax) return { family: "minmax", op: minmax[1] as "min" | "max", signed: minmax[2] === "s" };
  const unary = /^__bg_ptx_(neg|abs)$/u.exec(name)?.[1];
  return unary ? { family: "unary", op: unary as "neg" | "abs" } : undefined;
}
