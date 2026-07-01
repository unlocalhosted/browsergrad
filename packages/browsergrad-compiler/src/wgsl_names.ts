const WGSL_RESERVED_IDENTIFIERS = new Set([
  "active",
  "alias",
  "array",
  "atomic",
  "bitcast",
  "bool",
  "break",
  "case",
  "const",
  "continue",
  "default",
  "discard",
  "else",
  "enable",
  "false",
  "fn",
  "for",
  "f16",
  "f32",
  "i32",
  "if",
  "function",
  "handle",
  "let",
  "loop",
  "override",
  "precision",
  "private",
  "read",
  "read_write",
  "ref",
  "return",
  "shared",
  "storage",
  "struct",
  "switch",
  "true",
  "u32",
  "uniform",
  "var",
  "while",
  "workgroup",
  "main",
  "mod",
  "params",
  "global_id",
  "local_id",
  "workgroup_id",
  "num_workgroups",
]);

export function createWgslNameMap(names: readonly string[], extraReserved: Iterable<string> = []): ReadonlyMap<string, string> {
  const used = new Set([...WGSL_RESERVED_IDENTIFIERS, ...extraReserved]);
  const out = new Map<string, string>();
  for (const name of names) {
    if (out.has(name)) continue;
    const candidate = safeWgslIdentifier(name);
    if (!used.has(candidate)) {
      used.add(candidate);
      out.set(name, candidate);
      continue;
    }
    let index = 0;
    let renamed = `bg_${candidate}`;
    while (used.has(renamed)) renamed = `bg_${candidate}_${++index}`;
    used.add(renamed);
    out.set(name, renamed);
  }
  return out;
}

export function safeWgslIdentifier(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/gu, "_");
  const prefixed = /^[A-Za-z_]/u.test(cleaned) ? cleaned : `bg_${cleaned}`;
  return prefixed.startsWith("__") ? `bg${prefixed}` : prefixed;
}
