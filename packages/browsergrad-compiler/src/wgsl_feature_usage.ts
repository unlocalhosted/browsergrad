export function rewriteF16WgslToF32(wgsl: string): string {
  return wgsl.replace(/\bf16\b/gu, "f32");
}

export function rewriteF16BindingsToF32<T extends { readonly kind: string; readonly valueType?: string }>(
  bindings: readonly T[],
): readonly T[] {
  return bindings.map((binding) => {
    if (binding.kind !== "storage" || binding.valueType !== "f16") return binding;
    return { ...binding, valueType: "f32" } as T;
  });
}
