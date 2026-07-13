import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import {
  walkSemanticOperations,
} from "./semantic_ir.js";
import { semanticIdsEqual } from "./semantic_ids.js";
import {
  isSemanticKernelIrOperation,
  semanticAtomicMemoryRootNames,
} from "./semantic_ir_walk.js";
import type { CudaLiteScalarType } from "./types.js";
import { semanticPointerArgumentMemoryRef as semanticPointerArgMemoryRef } from "./semantic_pointer_arguments.js";
import {
  semanticPointerDeclarationNeedsRuntimeState,
  semanticRuntimePointerDeclarations as semanticLocalPointerDeclarations,
} from "./semantic_runtime_pointers.js";
import {
  semanticAtomicOperation,
  semanticAtomicUsesF32Storage,
} from "./semantic_atomic_intrinsics.js";
import { sizeofCudaType } from "./type_layout.js";
import {
  floatAtomicHelperName,
  integerAtomicLoopHelperName,
  wgslAtomicCalleeForCudaAtomic,
  wgslIntegerLoopAtomicKindForCudaAtomic,
} from "./wgsl_atomic_helpers.js";
import {
  wgslValueScalar,
  wgslValueType,
  wgslVectorScalar,
  zeroForType,
} from "./semantic_wgsl_types.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";

export function semanticWgslFunctionStoragePointerParam(
  ir: SemanticKernelIrModule,
  base: string,
  owner?: string | null,
): SemanticKernelIrModule["functions"][number]["params"][number] | undefined {
  if (owner !== undefined) {
    if (owner === null) return undefined;
    return ir.functions.find((fn) => fn.name === owner)?.params.find((item) =>
      item.name === base && item.pointer && item.addressSpace === "storage"
    );
  }
  for (const fn of ir.functions) {
    const param = fn.params.find((item) => item.name === base && item.pointer && item.addressSpace === "storage");
    if (param) return param;
  }
  return undefined;
}

export function semanticWgslFunctionSharedPointerParam(
  ir: SemanticKernelIrModule,
  base: string,
  owner?: string | null,
): SemanticKernelIrModule["functions"][number]["params"][number] | undefined {
  if (owner !== undefined) {
    if (owner === null) return undefined;
    return ir.functions.find((fn) => fn.name === owner)?.params.find((item) =>
      item.name === base && item.pointer && item.addressSpace === "shared"
    );
  }
  for (const fn of ir.functions) {
    const param = fn.params.find((item) => item.name === base && item.pointer && item.addressSpace === "shared");
    if (param) return param;
  }
  return undefined;
}

export function semanticStoragePointerBufferId(base: string, ir: SemanticKernelIrModule): number | undefined {
  const index = ir.params.findIndex((param) => param.name === base && param.addressSpace === "storage");
  if (index >= 0) return index;
  const globalIndex = ir.memory.filter((symbol) => symbol.kind === "device-global").findIndex((symbol) => symbol.name === base);
  return globalIndex < 0 ? undefined : ir.params.length + globalIndex;
}

export function semanticPointerStorageCompatible(pointerType: CudaLiteScalarType, storageType: CudaLiteScalarType | undefined): boolean {
  if (storageType === undefined) return false;
  return pointerType === storageType ||
    isCudaVectorType(storageType) && cudaVectorScalarType(storageType) === pointerType ||
    isCudaVectorType(pointerType) && cudaVectorScalarType(pointerType) === storageType ||
    isCudaVectorType(pointerType) && cudaVectorScalarType(pointerType) === cudaVectorScalarType(storageType);
}

export function semanticPointerReadHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_read_${semanticPointerHelperTypeName(valueType)}`;
}

export function semanticPointerWriteHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_write_${semanticPointerHelperTypeName(valueType)}`;
}

export function semanticPointerAtomicCasHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_atomicCompareExchange_${semanticPointerHelperTypeName(valueType)}`;
}

export function semanticPointerBufferParamName(base: string): string {
  return `${base}_buffer`;
}

export function semanticPointerBaseParamName(base: string): string {
  return `${base}_base`;
}

export function semanticStorageOffsetSymbol(base: string): string {
  return `${base}__bg_ptr_offset`;
}

export function semanticStorageOffsetBaseNames(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  pointerBaseOffsets: Readonly<Record<string, number>> | undefined,
): Set<string> {
  const out = new Set(ir.params
    .filter((param) =>
      param.addressSpace === "storage" &&
      param.pointer &&
      pointerBaseOffsets?.[param.name] !== undefined
    )
    .map((param) => param.name));
  collectSemanticStorageOffsetBaseNames(operations, out);
  return out;
}

function collectSemanticStorageOffsetBaseNames(
  operations: readonly SemanticKernelIrOperation[],
  out: Set<string>,
): void {
  for (const operation of operations) {
    if (
      operation.kind === "store" &&
      operation.target.addressSpace === "storage" &&
      operation.target.indices.length === 0 &&
      operation.target.fields.length === 0 &&
      (operation.operator === "+=" || operation.operator === "-=")
    ) out.add(operation.target.base);
    if (operation.kind === "branch") collectSemanticStorageOffsetBaseNames([...operation.consequent, ...operation.alternate], out);
    if (operation.kind === "loop") collectSemanticStorageOffsetBaseNames(operation.body, out);
  }
}

function semanticPointerHelperTypeName(valueType: CudaLiteScalarType): string {
  const scalar = semanticPointerHelperScalarName(valueType);
  return isCudaVectorType(valueType) ? `${scalar}x${cudaVectorLaneCount(valueType)}` : scalar;
}

function semanticPointerHelperScalarName(valueType: CudaLiteScalarType | undefined): string {
  if (valueType === "half" || valueType === "half2") return "f16";
  const scalarType = isCudaVectorType(valueType) ? cudaVectorScalarType(valueType) : valueType;
  if (scalarType === "int") return "i32";
  if (scalarType === "uint") return "u32";
  if (scalarType === "half") return "f16";
  if (valueType === "bool") return "u32";
  return "f32";
}


export interface SemanticRuntimePointerWgslHost {
  readonly memoryRefFromIndexExpression: (expression: SemanticExpression) => SemanticMemoryRef | undefined;
  readonly nameFor: (name: string, names: ReadonlyMap<string, string>) => string;
  readonly semanticWgslFloatAtomicCallKind: (callee: string) => "Add" | "Sub" | "Min" | "Max" | "Exchange" | "CompareExchange" | undefined;
}

export function createSemanticRuntimePointerWgsl(host: SemanticRuntimePointerWgslHost) {
  const { memoryRefFromIndexExpression, nameFor, semanticWgslFloatAtomicCallKind } = host;

function emitSemanticStoragePointerHelpers(
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly (readonly string[])[] {
  const types = semanticStoragePointerValueTypes(ir);
  return [...types].flatMap((type) => [
    emitSemanticStoragePointerReadHelper(type, ir, names),
    emitSemanticStoragePointerWriteHelper(type, ir, names),
    ...SEMANTIC_POINTER_ATOMIC_CALLS.flatMap((callee) => {
      const helper = emitSemanticStoragePointerAtomicHelper(callee, type, ir, names);
      return helper.length === 0 ? [] : [helper];
    }),
  ]);
}

function semanticStoragePointerValueTypes(ir: SemanticKernelIrModule): ReadonlySet<CudaLiteScalarType> {
  const types = new Set<CudaLiteScalarType>();
  for (const declaration of semanticLocalPointerDeclarations(ir)) {
    if (declaration.target.dimensions.length === 0 && declaration.target.valueType !== undefined &&
      (semanticLocalPointerStorageRef(declaration) !== undefined || declaration.target.pointerRuntimeState === true)) {
      types.add(declaration.target.valueType);
    }
  }
  for (const fn of ir.functions) {
    const pointerNames = new Set(fn.params
      .filter((param) => param.pointer && param.addressSpace === "storage")
      .map((param) => param.name));
    if (pointerNames.size === 0) continue;
    const add = (ref: SemanticMemoryRef): void => {
      if (pointerNames.has(ref.base) && ref.valueType !== undefined) types.add(ref.valueType);
    };
    for (const param of fn.params) {
      if (param.pointer && param.addressSpace === "storage" && param.valueType !== undefined) types.add(param.valueType);
    }
    walkSemanticOperations(fn.body, (expression) => {
      const ref = memoryRefFromIndexExpression(expression);
      if (ref) add(ref);
    });
    collectSemanticStoragePointerOperationRefs(fn.body, add);
  }
  return types;
}

function semanticLocalPointerStorageRef(
  declaration: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
): SemanticMemoryRef | undefined {
  const ref = declaration.init ? semanticPointerArgMemoryRef(declaration.init) : undefined;
  return ref?.addressSpace === "storage" || ref?.addressSpace === "device-global" ? ref : undefined;
}

function semanticLocalStoragePointerDeclaration(
  ir: SemanticKernelIrModule,
  expression: SemanticExpression,
): Extract<SemanticKernelIrOperation, { readonly kind: "declare" }> | undefined {
  if (expression.kind !== "symbol" || expression.addressSpace !== "local") return undefined;
  return semanticLocalPointerDeclarations(ir).find((operation) =>
    semanticIdsEqual(operation.target.id, expression.id) &&
    operation.target.dimensions.length === 0 &&
    semanticPointerDeclarationNeedsRuntimeState(operation) &&
    (semanticLocalPointerStorageRef(operation) !== undefined || operation.target.pointerRuntimeState === true)
  );
}

function collectSemanticStoragePointerOperationRefs(
  operations: readonly SemanticKernelIrOperation[],
  add: (ref: SemanticMemoryRef) => void,
): void {
  for (const operation of operations) {
    switch (operation.kind) {
      case "load":
        add(operation.source);
        break;
      case "store":
        add(operation.target);
        operation.reads.forEach(add);
        break;
      case "copy":
        add(operation.source);
        add(operation.target);
        break;
      case "atomic":
        if (operation.target) add(operation.target);
        break;
      case "call":
        operation.reads.forEach(add);
        break;
      case "pointer-rebind":
        add(operation.source);
        break;
      case "branch":
        collectSemanticStoragePointerOperationRefs(operation.consequent, add);
        collectSemanticStoragePointerOperationRefs(operation.alternate, add);
        break;
      case "loop":
        if (operation.init && isSemanticKernelIrOperation(operation.init)) {
          collectSemanticStoragePointerOperationRefs([operation.init], add);
        }
        collectSemanticStoragePointerOperationRefs(operation.body, add);
        if (operation.continuing) collectSemanticStoragePointerOperationRefs(operation.continuing, add);
        break;
      case "block":
        collectSemanticStoragePointerOperationRefs(operation.body, add);
        break;
      case "declare":
      case "dim3-declare":
      case "surface-write":
      case "surface-read-store":
      case "cooperative-group-declare":
      case "barrier":
      case "fence":
      case "inline-asm":
      case "expression":
      case "device-launch":
      case "return":
      case "continue":
      case "break":
        break;
    }
  }
}

const SEMANTIC_POINTER_ATOMIC_CALLS = [
  "atomicAdd", "atomicSub", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExch", "atomicCAS",
  "atomicInc", "atomicInc_system", "atomicDec", "atomicDec_system",
] as const;

function emitSemanticStoragePointerReadHelper(
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  return [
    `fn ${semanticPointerReadHelperName(valueType)}(buffer: u32, index: u32) -> ${wgslType} {`,
    "  switch buffer {",
    ...semanticStoragePointerBindings(ir).flatMap((binding) => {
      if (semanticAtomicBytePointerBindingCompatible(valueType, binding, atomicStorage)) {
        return [`    case ${binding.id}u: { return ${emitSemanticAtomicByteStorageReadValue(valueType, nameFor(binding.name, names), "index")}; }`];
      }
      return semanticPointerStorageCompatible(valueType, binding.valueType)
        ? [`    case ${binding.id}u: { return ${emitSemanticStoragePointerReadValue(valueType, nameFor(binding.name, names), "index", atomicStorage.has(binding.name))}; }`]
        : [];
    }),
    "    default: { return " + zeroForType(wgslType) + "; }",
    "  }",
    "}",
  ];
}

function emitSemanticStoragePointerWriteHelper(
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  return [
    `fn ${semanticPointerWriteHelperName(valueType)}(buffer: u32, index: u32, value: ${wgslType}) {`,
    "  switch buffer {",
    ...semanticStoragePointerBindings(ir).flatMap((binding) => {
      if (!binding.constant && semanticAtomicBytePointerBindingCompatible(valueType, binding, atomicStorage)) {
        return [`    case ${binding.id}u: { ${emitSemanticAtomicByteStorageWriteValue(valueType, nameFor(binding.name, names), "index", "value")} return; }`];
      }
      return !binding.constant && semanticPointerStorageCompatible(valueType, binding.valueType)
        ? [`    case ${binding.id}u: { ${emitSemanticStoragePointerWriteValue(valueType, nameFor(binding.name, names), "index", "value", atomicStorage.has(binding.name))} return; }`]
        : [];
    }),
    "    default: { return; }",
    "  }",
    "}",
  ];
}

function emitSemanticStoragePointerAtomicHelper(
  callee: typeof SEMANTIC_POINTER_ATOMIC_CALLS[number],
  valueType: CudaLiteScalarType,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): readonly string[] {
  if (!semanticWgslPointerAtomicCallSupported(callee, valueType)) return [];
  const wgslType = wgslValueType(valueType);
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  const op = semanticAtomicOperation(callee);
  const cas = op === "cas";
  return [
    `fn ${semanticPointerAtomicHelperName(callee, valueType)}(buffer: u32, index: u32, ${cas ? `compare: ${wgslType}, ` : ""}value: ${wgslType}) -> ${wgslType} {`,
    "  switch buffer {",
    ...semanticStoragePointerBindings(ir).flatMap((binding) => {
      if (!binding.constant && semanticAtomicBytePointerBindingCompatible(valueType, binding, atomicStorage)) {
        return [`    case ${binding.id}u: { return ${emitSemanticAtomicByteStorageAtomicValue(callee, valueType, nameFor(binding.name, names), "index", "compare", "value")}; }`];
      }
      return !binding.constant && atomicStorage.has(binding.name) && semanticPointerStorageCompatible(valueType, binding.valueType)
        ? [`    case ${binding.id}u: { return ${emitSemanticStoragePointerAtomicValue(callee, valueType, nameFor(binding.name, names), "index", "compare", "value")}; }`]
        : [];
    }),
    "    default: { return " + zeroForType(wgslType) + "; }",
    "  }",
    "}",
  ];
}

function semanticStoragePointerBindings(ir: SemanticKernelIrModule): readonly {
  readonly id: number;
  readonly name: string;
  readonly valueType?: CudaLiteScalarType;
  readonly constant: boolean;
}[] {
  return [
    ...ir.params.flatMap((param, index) => param.addressSpace === "storage"
      ? [{ id: index, name: param.name, ...(param.valueType === undefined ? {} : { valueType: param.valueType }), constant: param.constant ?? false }]
      : []),
    ...ir.memory.filter((symbol) => symbol.kind === "device-global").map((symbol, index) => ({
      id: ir.params.length + index,
      name: symbol.name,
      ...(symbol.valueType === undefined ? {} : { valueType: symbol.valueType }),
      constant: false,
    })),
  ];
}

function semanticHasAtomicByteStorage(ir: SemanticKernelIrModule): boolean {
  const atomicStorage = semanticAtomicMemoryRootNames(ir);
  return semanticStoragePointerBindings(ir).some((binding) =>
    binding.valueType === "uchar" && atomicStorage.has(binding.name)
  );
}

function semanticAtomicBytePointerBindingCompatible(
  valueType: CudaLiteScalarType,
  binding: { readonly name: string; readonly valueType?: CudaLiteScalarType },
  atomicStorage: ReadonlySet<string>,
): boolean {
  return binding.valueType === "uchar" && atomicStorage.has(binding.name) &&
    !isCudaVectorType(valueType) && sizeofCudaType(valueType) === 4;
}

function emitSemanticAtomicByteStorageReadValue(
  valueType: CudaLiteScalarType,
  storage: string,
  byteIndex: string,
): string {
  const loaded = `atomicLoad(&${storage}[(${byteIndex} >> 2u)])`;
  if (valueType === "int") return `bitcast<i32>(${loaded})`;
  if (valueType === "float" || valueType === "double") return `bitcast<f32>(${loaded})`;
  return loaded;
}

function emitSemanticAtomicByteStorageWriteValue(
  valueType: CudaLiteScalarType,
  storage: string,
  byteIndex: string,
  value: string,
): string {
  const bits = valueType === "uint" ? value : `bitcast<u32>(${value})`;
  return `atomicStore(&${storage}[(${byteIndex} >> 2u)], ${bits});`;
}

function emitSemanticAtomicByteStorageAtomicValue(
  callee: string,
  valueType: CudaLiteScalarType,
  storage: string,
  byteIndex: string,
  compare: string,
  value: string,
): string {
  const access = `${storage}[(${byteIndex} >> 2u)]`;
  if (valueType === "uint") return emitSemanticStoragePointerAtomicValue(callee, valueType, storage, `(${byteIndex} >> 2u)`, compare, value);
  if (valueType === "float" || valueType === "double") {
    return emitSemanticStoragePointerAtomicValue(callee, valueType, storage, `(${byteIndex} >> 2u)`, compare, value);
  }
  const op = semanticAtomicOperation(callee);
  if (op === "min" || op === "max") return `bg_atomic${op === "min" ? "Min" : "Max"}_storage_u32_as_i32(&${access}, ${value})`;
  if (op === "cas") return `bitcast<i32>(atomicCompareExchangeWeak(&${access}, bitcast<u32>(${compare}), bitcast<u32>(${value})).old_value)`;
  const wgslCallee = wgslAtomicCalleeForCudaAtomic(callee);
  return wgslCallee === undefined ? "0" : `bitcast<i32>(${wgslCallee}(&${access}, bitcast<u32>(${value})))`;
}

function emitSemanticSignedByteAtomicHelpers(): readonly string[] {
  const helper = (op: "Min" | "Max", compare: "min" | "max"): readonly string[] => [
    `fn bg_atomic${op}_storage_u32_as_i32(word: ptr<storage, atomic<u32>, read_write>, value: i32) -> i32 {`,
    "  var old_bits = atomicLoad(word);",
    "  loop {",
    "    let old_value = bitcast<i32>(old_bits);",
    `    let next_value = ${compare}(old_value, value);`,
    "    let result = atomicCompareExchangeWeak(word, old_bits, bitcast<u32>(next_value));",
    "    if (result.exchanged) { return old_value; }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
  return [...helper("Min", "min"), "", ...helper("Max", "max")];
}

function semanticPointerAtomicHelperName(callee: string, valueType: CudaLiteScalarType): string {
  if (semanticAtomicOperation(callee) === "cas") return `bg_ptr_atomicCompareExchange_${wgslValueScalar(valueType)}`;
  return `bg_ptr_${callee}_${wgslValueScalar(valueType)}`;
}

function semanticWgslPointerAtomicCallSupported(callee: string, valueType: CudaLiteScalarType): boolean {
  const op = semanticAtomicOperation(callee);
  if (semanticAtomicUsesF32Storage(valueType)) return op === "add" || op === "sub" || op === "min" || op === "max" || op === "exchange" || op === "cas";
  if (valueType === "int" || valueType === "uint") return op === "add" || op === "sub" || op === "min" || op === "max" || op === "and" || op === "or" || op === "xor" || op === "exchange" || op === "cas" || op === "inc" || op === "dec";
  return false;
}

function emitSemanticStoragePointerReadValue(valueType: CudaLiteScalarType, storage: string, index: string, atomic: boolean): string {
  if (!isCudaVectorType(valueType)) return emitSemanticStoragePointerReadScalarValue(valueType, `${storage}[${index}]`, atomic);
  const laneCount = cudaVectorLaneCount(valueType);
  const scalar = wgslVectorScalar(valueType);
  return `${wgslValueType(valueType)}(${Array.from({ length: laneCount }, (_, lane) =>
    `${scalar}(${emitSemanticStoragePointerReadScalarValue(cudaVectorScalarType(valueType) ?? valueType, `${storage}[(${index} + ${lane}u)]`, atomic)})`
  ).join(", ")})`;
}

function emitSemanticStoragePointerReadScalarValue(valueType: CudaLiteScalarType, access: string, atomic: boolean): string {
  if (!atomic) return access;
  const loaded = `atomicLoad(&${access})`;
  return semanticAtomicUsesF32Storage(valueType) ? `bitcast<f32>(${loaded})` : loaded;
}

function emitSemanticStoragePointerWriteValue(valueType: CudaLiteScalarType, storage: string, index: string, value: string, atomic: boolean): string {
  if (!isCudaVectorType(valueType)) return emitSemanticStoragePointerWriteScalarValue(valueType, `${storage}[${index}]`, value, atomic);
  return Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) =>
    emitSemanticStoragePointerWriteScalarValue(cudaVectorScalarType(valueType) ?? valueType, `${storage}[(${index} + ${lane}u)]`, `(${value}).${["x", "y", "z", "w"][lane]}`, atomic)
  ).join(" ");
}

function emitSemanticStoragePointerWriteScalarValue(valueType: CudaLiteScalarType, access: string, value: string, atomic: boolean): string {
  if (!atomic) return `${access} = ${value};`;
  const stored = semanticAtomicUsesF32Storage(valueType) ? `bitcast<u32>(${value})` : value;
  return `atomicStore(&${access}, ${stored});`;
}

function emitSemanticStoragePointerAtomicValue(
  callee: string,
  valueType: CudaLiteScalarType,
  storage: string,
  index: string,
  compare: string,
  value: string,
): string {
  const op = semanticAtomicOperation(callee);
  if (valueType === "float" || valueType === "double") {
    if (op === "exchange") return `bitcast<f32>(atomicExchange(&${storage}[${index}], bitcast<u32>(${value})))`;
    if (op === "cas") return `bitcast<f32>(atomicCompareExchangeWeak(&${storage}[${index}], bitcast<u32>(${compare}), bitcast<u32>(${value})).old_value)`;
    const kind = semanticWgslFloatAtomicCallKind(callee);
    return kind === "Add" || kind === "Sub" || kind === "Min" || kind === "Max"
      ? `${floatAtomicHelperName(kind, "storage")}(&${storage}[${index}], ${value})`
      : "0.0";
  }
  const loopKind = wgslIntegerLoopAtomicKindForCudaAtomic(callee);
  if (loopKind && (valueType === "uint" || valueType === "int")) {
    const scalar = valueType === "uint" ? "u32" : "i32";
    const helper = integerAtomicLoopHelperName(loopKind, {
      valueType,
      storageValueType: valueType,
      storageScalar: scalar,
      addressSpace: "storage",
    });
    return `${helper}(&${storage}[${index}], ${value})`;
  }
  const wgslCallee = wgslAtomicCalleeForCudaAtomic(callee);
  if (wgslCallee === "atomicCompareExchangeWeak") return `atomicCompareExchangeWeak(&${storage}[${index}], ${compare}, ${value}).old_value`;
  return wgslCallee === undefined ? "0" : `${wgslCallee}(&${storage}[${index}], ${value})`;
}

  return {
    emitSemanticStoragePointerHelpers,
    semanticLocalPointerDeclarations,
    semanticLocalPointerStorageRef,
    semanticPointerDeclarationNeedsRuntimeState,
    semanticLocalStoragePointerDeclaration,
    semanticHasAtomicByteStorage,
    emitSemanticAtomicByteStorageReadValue,
    emitSemanticAtomicByteStorageWriteValue,
    emitSemanticAtomicByteStorageAtomicValue,
    emitSemanticSignedByteAtomicHelpers,
    semanticPointerAtomicHelperName,
    semanticWgslPointerAtomicCallSupported,
  };
}
