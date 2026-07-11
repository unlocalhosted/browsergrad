const WGSL_RESERVED_IDENTIFIERS = new Set([
  "NULL", "Self", "abstract", "alignas", "alignof", "as", "asm", "asm_fragment", "async",
  "attribute", "auto", "await", "become", "cast", "catch", "class", "co_await", "co_return",
  "co_yield", "coherent", "column_major", "common", "compile", "compile_fragment", "concept",
  "const_cast", "consteval", "constexpr", "constinit", "crate", "debugger", "decltype", "delete",
  "demote", "demote_to_helper", "do", "dynamic_cast", "enum", "explicit", "export", "extends",
  "extern", "external", "fallthrough", "filter", "final", "finally", "friend", "from", "fxgroup",
  "get", "goto", "groupshared", "highp", "impl", "implements", "import", "inline", "instanceof",
  "interface", "layout", "lowp", "macro", "macro_rules", "match", "mediump", "meta", "module",
  "move", "mut", "mutable", "namespace", "new", "nil", "noexcept", "noinline", "nointerpolation",
  "non_coherent", "noncoherent", "noperspective", "null", "nullptr", "of", "operator", "package",
  "packoffset", "partition", "patch", "pixelfragment", "precise", "premerge", "priv", "protected",
  "pub", "public", "readonly", "regardless", "register", "reinterpret_cast", "require", "resource",
  "restrict", "self", "set", "sizeof", "smooth", "snorm", "static", "static_assert", "static_cast",
  "std", "subroutine", "super", "target", "template", "this", "thread_local", "throw", "trait", "try",
  "type", "typedef", "typeid", "typename", "typeof", "union", "unless", "unorm", "unsafe", "unsized",
  "use", "using", "varying", "virtual", "volatile", "wgsl", "where", "with", "writeonly", "yield",
  "active",
  "alias",
  "array",
  "atomic",
  "bitcast",
  "bool",
  "break",
  "case",
  "const",
  "const_assert",
  "continue",
  "continuing",
  "default",
  "diagnostic",
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
  "requires",
  "pass",
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

// User declarations share WGSL's function namespace with these predeclared operations.
const WGSL_PREDECLARED_FUNCTION_IDENTIFIERS = new Set([
  "abs", "acos", "acosh", "all", "any", "arrayLength", "asin", "asinh", "atan", "atan2",
  "ceil", "clamp", "cos", "cosh", "countLeadingZeros", "countOneBits", "countTrailingZeros",
  "cross", "degrees", "determinant", "distance", "dot", "dpdx", "dpdxCoarse", "dpdxFine",
  "dpdy", "dpdyCoarse", "dpdyFine", "exp", "exp2", "extractBits", "faceForward",
  "firstLeadingBit", "firstTrailingBit", "floor", "fma", "fract", "frexp", "fwidth",
  "fwidthCoarse", "fwidthFine", "insertBits", "inverseSqrt", "ldexp", "length", "log", "log2",
  "max", "min", "mix", "modf", "normalize", "pack2x16float", "pack2x16snorm", "pack2x16unorm",
  "pack4x8snorm", "pack4x8unorm", "pow", "quantizeToF16", "radians", "reflect", "refract",
  "reverseBits", "round", "sign", "sin", "sinh", "smoothstep", "sqrt", "step", "tan", "tanh",
  "textureDimensions", "textureGather", "textureGatherCompare", "textureLoad", "textureNumLayers",
  "textureNumLevels", "textureNumSamples", "textureSample", "textureSampleBaseClampToEdge",
  "textureSampleBias", "textureSampleCompare", "textureSampleCompareLevel", "textureSampleGrad",
  "textureSampleLevel", "transpose", "trunc", "unpack2x16float", "unpack2x16snorm", "unpack2x16unorm",
  "unpack4x8snorm", "unpack4x8unorm",
]);

export function createWgslNameMap(
  names: readonly string[],
  extraReserved: Iterable<string> = [],
  functionNames: Iterable<string> = [],
): ReadonlyMap<string, string> {
  const used = new Set([...WGSL_RESERVED_IDENTIFIERS, ...extraReserved]);
  const functions = new Set(functionNames);
  const out = new Map<string, string>();
  for (const name of names) {
    if (out.has(name)) continue;
    const candidate = safeWgslIdentifier(name);
    const shadowsPredeclaredFunction = functions.has(name) && WGSL_PREDECLARED_FUNCTION_IDENTIFIERS.has(candidate);
    if (!used.has(candidate) && !shadowsPredeclaredFunction) {
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
