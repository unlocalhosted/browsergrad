export type WgslAtomicAddressSpace = "storage" | "workgroup";
export type WgslAtomicIntegerScalar = "i32" | "u32";

export interface WgslIntegerAtomicLoopTarget {
  readonly valueType: string;
  readonly storageValueType: string;
  readonly storageScalar: WgslAtomicIntegerScalar;
  readonly addressSpace: WgslAtomicAddressSpace;
}

export function floatAtomicHelperName(kind: "Add" | "Sub" | "Min" | "Max", addressSpace: WgslAtomicAddressSpace): string {
  return addressSpace === "storage" ? `bg_atomic${kind}_f32` : `bg_atomic${kind}_f32_workgroup`;
}

export function integerAtomicLoopHelperName(kind: "Inc" | "Dec", target: WgslIntegerAtomicLoopTarget): string {
  if ((target.storageValueType === "float" || target.storageValueType === "double") && target.valueType === "uint") {
    return `bg_atomic${kind}_${target.addressSpace}_f32_as_u32`;
  }
  return `bg_atomic${kind}_${target.addressSpace}_${target.storageScalar}`;
}

export function isAtomicCasCallName(name: string | undefined): boolean {
  return name === "atomicCAS" || name === "atomicCAS_system";
}

export function isAtomicExchangeCallName(name: string | undefined): boolean {
  return name === "atomicExch" || name === "atomicExch_system";
}

export function isAtomicReturnCallName(name: string | undefined): boolean {
  return name === "atomicAdd" ||
    name === "atomicAdd_system" ||
    name === "atomicSub" ||
    name === "atomicSub_system" ||
    name === "atomicMin" ||
    name === "atomicMin_system" ||
    name === "atomicMax" ||
    name === "atomicMax_system" ||
    name === "atomicMaxFloat" ||
    name === "atomicAnd" ||
    name === "atomicAnd_system" ||
    name === "atomicOr" ||
    name === "atomicOr_system" ||
    name === "atomicXor" ||
    name === "atomicXor_system" ||
    name === "atomicInc" ||
    name === "atomicInc_system" ||
    name === "atomicDec" ||
    name === "atomicDec_system" ||
    isAtomicExchangeCallName(name);
}

export function emitFloatAtomicAddHelper(addressSpace: WgslAtomicAddressSpace): string[] {
  const name = floatAtomicHelperName("Add", addressSpace);
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, "u32")}, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_bits = bitcast<u32>(old_value + value);",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

export function emitFloatAtomicSubHelper(addressSpace: WgslAtomicAddressSpace): string[] {
  const name = floatAtomicHelperName("Sub", addressSpace);
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, "u32")}, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_bits = bitcast<u32>(old_value - value);",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

export function emitFloatAtomicMinHelper(addressSpace: WgslAtomicAddressSpace): string[] {
  const name = floatAtomicHelperName("Min", addressSpace);
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, "u32")}, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_value = min(old_value, value);",
    "    let new_bits = bitcast<u32>(new_value);",
    "    if (new_bits == old_bits) {",
    "      return old_value;",
    "    }",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

export function emitFloatAtomicMaxHelper(addressSpace: WgslAtomicAddressSpace): string[] {
  const name = floatAtomicHelperName("Max", addressSpace);
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, "u32")}, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_value = max(old_value, value);",
    "    let new_bits = bitcast<u32>(new_value);",
    "    if (new_bits == old_bits) {",
    "      return old_value;",
    "    }",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

export function emitIntegerAtomicLoopHelpers(): string[] {
  return [
    ...emitIntegerAtomicIncHelper("storage", "u32"),
    "",
    ...emitFloatBackedIntegerAtomicIncHelper("storage"),
    "",
    ...emitIntegerAtomicIncHelper("storage", "i32"),
    "",
    ...emitIntegerAtomicIncHelper("workgroup", "u32"),
    "",
    ...emitFloatBackedIntegerAtomicIncHelper("workgroup"),
    "",
    ...emitIntegerAtomicIncHelper("workgroup", "i32"),
    "",
    ...emitIntegerAtomicDecHelper("storage", "u32"),
    "",
    ...emitFloatBackedIntegerAtomicDecHelper("storage"),
    "",
    ...emitIntegerAtomicDecHelper("storage", "i32"),
    "",
    ...emitIntegerAtomicDecHelper("workgroup", "u32"),
    "",
    ...emitFloatBackedIntegerAtomicDecHelper("workgroup"),
    "",
    ...emitIntegerAtomicDecHelper("workgroup", "i32"),
  ];
}

function atomicPointerType(addressSpace: WgslAtomicAddressSpace, scalar: WgslAtomicIntegerScalar): string {
  return addressSpace === "workgroup"
    ? `ptr<workgroup, atomic<${scalar}>>`
    : `ptr<storage, atomic<${scalar}>, read_write>`;
}

function emitIntegerAtomicIncHelper(addressSpace: WgslAtomicAddressSpace, scalar: WgslAtomicIntegerScalar): string[] {
  const name = `bg_atomicInc_${addressSpace}_${scalar}`;
  const load = scalar === "u32" ? "atomicLoad(ptr_value)" : "bitcast<u32>(atomicLoad(ptr_value))";
  const compare = scalar === "u32" ? "old_bits" : "bitcast<i32>(old_bits)";
  const store = scalar === "u32" ? "next_bits" : "bitcast<i32>(next_bits)";
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, scalar)}, limit: u32) -> u32 {`,
    `  var old_bits = ${load};`,
    "  loop {",
    "    let next_bits = select(old_bits + 1u, 0u, old_bits >= limit);",
    `    let result = atomicCompareExchangeWeak(ptr_value, ${compare}, ${store});`,
    "    if (result.exchanged) {",
    "      return old_bits;",
    "    }",
    `    old_bits = ${scalar === "u32" ? "result.old_value" : "bitcast<u32>(result.old_value)"};`,
    "  }",
    "}",
  ];
}

function emitIntegerAtomicDecHelper(addressSpace: WgslAtomicAddressSpace, scalar: WgslAtomicIntegerScalar): string[] {
  const name = `bg_atomicDec_${addressSpace}_${scalar}`;
  const load = scalar === "u32" ? "atomicLoad(ptr_value)" : "bitcast<u32>(atomicLoad(ptr_value))";
  const compare = scalar === "u32" ? "old_bits" : "bitcast<i32>(old_bits)";
  const store = scalar === "u32" ? "next_bits" : "bitcast<i32>(next_bits)";
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, scalar)}, limit: u32) -> u32 {`,
    `  var old_bits = ${load};`,
    "  loop {",
    "    let next_bits = select(old_bits - 1u, limit, old_bits == 0u || old_bits > limit);",
    `    let result = atomicCompareExchangeWeak(ptr_value, ${compare}, ${store});`,
    "    if (result.exchanged) {",
    "      return old_bits;",
    "    }",
    `    old_bits = ${scalar === "u32" ? "result.old_value" : "bitcast<u32>(result.old_value)"};`,
    "  }",
    "}",
  ];
}

function emitFloatBackedIntegerAtomicIncHelper(addressSpace: WgslAtomicAddressSpace): string[] {
  const name = `bg_atomicInc_${addressSpace}_f32_as_u32`;
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, "u32")}, limit: u32) -> u32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let next_bits = select(old_bits + 1u, 0u, old_bits >= limit);",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, next_bits);",
    "    if (result.exchanged) {",
    "      return old_bits;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

function emitFloatBackedIntegerAtomicDecHelper(addressSpace: WgslAtomicAddressSpace): string[] {
  const name = `bg_atomicDec_${addressSpace}_f32_as_u32`;
  return [
    `fn ${name}(ptr_value: ${atomicPointerType(addressSpace, "u32")}, limit: u32) -> u32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let next_bits = select(old_bits - 1u, limit, old_bits == 0u || old_bits > limit);",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, next_bits);",
    "    if (result.exchanged) {",
    "      return old_bits;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}
