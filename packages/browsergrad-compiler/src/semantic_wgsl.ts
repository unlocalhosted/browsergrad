import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslValueType,
} from "@unlocalhosted/browsergrad-kernels";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import type {
  CudaLiteDiagnostic,
  CudaLiteScalarType,
  SourceSpan,
} from "./types.js";
import { CudaLiteCompilerError } from "./types.js";
import { createWgslNameMap, safeWgslIdentifier } from "./wgsl_names.js";

export interface SemanticKernelIrWgslOutput {
  readonly wgsl: string;
  readonly program: ReturnType<typeof defineWgslKernelProgram>;
}

const UNIFORM_PARAMS_NAME = "bg_uniforms";
const BUILTIN_VECTOR_NAMES = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const LOGICAL_OPERATORS = new Set(["&&", "||"]);
const SEMANTIC_MATH_CALLS = new Map([
  ["sqrt", "sqrt"],
  ["sqrtf", "sqrt"],
  ["exp", "exp"],
  ["expf", "exp"],
  ["log", "log"],
  ["logf", "log"],
  ["fabs", "abs"],
  ["fabsf", "abs"],
  ["abs", "abs"],
  ["floor", "floor"],
  ["floorf", "floor"],
  ["ceil", "ceil"],
  ["ceilf", "ceil"],
  ["sin", "sin"],
  ["sinf", "sin"],
  ["cos", "cos"],
  ["cosf", "cos"],
  ["fmin", "min"],
  ["fminf", "min"],
  ["min", "min"],
  ["fmax", "max"],
  ["fmaxf", "max"],
  ["max", "max"],
  ["pow", "pow"],
  ["powf", "pow"],
]);
const WGSL_ATOMIC_CALLEES = new Map([
  ["atomicAdd", "atomicAdd"],
  ["atomicAdd_system", "atomicAdd"],
  ["atomicSub", "atomicSub"],
  ["atomicSub_system", "atomicSub"],
  ["atomicMin", "atomicMin"],
  ["atomicMin_system", "atomicMin"],
  ["atomicMax", "atomicMax"],
  ["atomicMax_system", "atomicMax"],
  ["atomicAnd", "atomicAnd"],
  ["atomicAnd_system", "atomicAnd"],
  ["atomicOr", "atomicOr"],
  ["atomicOr_system", "atomicOr"],
  ["atomicXor", "atomicXor"],
  ["atomicXor_system", "atomicXor"],
  ["atomicExch", "atomicExchange"],
  ["atomicExch_system", "atomicExchange"],
  ["atomicCAS", "atomicCompareExchangeWeak"],
  ["atomicCAS_system", "atomicCompareExchangeWeak"],
]);

export function canEmitSemanticKernelIrWgsl(ir: SemanticKernelIrModule): boolean {
  return unsupportedSemanticWgslOperation(ir.operations, ir) === undefined &&
    ir.requiredFeatures.length === 0 &&
    ir.params.every(semanticWgslParamSupported) &&
    semanticWgslSharedBarrierShapeSupported(ir) &&
    ir.memory.every(semanticWgslMemorySymbolSupported);
}

export function emitSemanticKernelIrWgsl(ir: SemanticKernelIrModule): SemanticKernelIrWgslOutput {
  const unsupported = unsupportedSemanticWgslOperation(ir.operations, ir);
  if (unsupported) throw semanticWgslError(`semantic WGSL does not support ${unsupported.kind}`, unsupported.span);
  if (ir.requiredFeatures.length > 0) throw semanticWgslError("semantic WGSL does not support required WebGPU features yet", ir.span);
  const unsupportedParam = ir.params.find((param) => !semanticWgslParamSupported(param));
  if (unsupportedParam) throw semanticWgslError(`semantic WGSL does not support parameter '${unsupportedParam.name}'`, unsupportedParam.span);

  const rawNames = new Set(ir.params.map((param) => param.name));
  for (const operation of ir.operations) collectOperationNames(operation, rawNames);
  const names = createWgslNameMap([...rawNames]);
  const uniformParams = [
    ...ir.params.filter((param) => param.addressSpace === "uniform"),
    ...constantMemorySymbols(ir).filter((symbol) => symbol.dimensions.length === 0),
  ];
  const constantBuffers = constantMemorySymbols(ir).filter((symbol) => symbol.dimensions.length > 0);
  const deviceGlobalBuffers = deviceGlobalMemorySymbols(ir);
  const atomicStorage = semanticAtomicStorageNames(ir.operations);
  const atomicDeviceGlobals = semanticAtomicDeviceGlobalNames(ir.operations);
  const bindings: WgslKernelBindingInput[] = ir.params
    .filter((param) => param.addressSpace === "storage")
    .map((param, binding) => ({
      kind: "storage",
      name: param.name,
      valueType: wgslBindingType(param.valueType),
      access: param.constant ? "read" : "read_write",
      binding,
    }));
  for (const constant of constantBuffers) {
    bindings.push({
      kind: "storage",
      name: constant.name,
      valueType: wgslBindingType(constant.valueType),
      access: "read",
      binding: bindings.length,
    });
  }
  for (const global of deviceGlobalBuffers) {
    bindings.push({
      kind: "storage",
      name: global.name,
      valueType: wgslBindingType(global.valueType),
      access: "read_write",
      binding: bindings.length,
    });
  }
  if (uniformParams.length > 0) {
    bindings.push({
      kind: "uniform",
      name: UNIFORM_PARAMS_NAME,
      byteLength: Math.max(16, uniformParams.length * 4),
      binding: bindings.length,
    });
  }

  const lines: string[] = ["// browsergrad-semantic-wgsl: direct semantic IR emission"];
  for (const param of ir.params.filter((item) => item.addressSpace === "storage")) {
    const access = param.constant ? "read" : "read_write";
    const elementType = atomicStorage.has(param.name)
      ? `atomic<${wgslAtomicScalar(param.valueType)}>`
      : wgslScalar(param.valueType);
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, param.name)}) var<storage, ${access}> ${nameFor(param.name, names)}: array<${elementType}>;`);
  }
  for (const constant of constantBuffers) {
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, constant.name)}) var<storage, read> ${nameFor(constant.name, names)}: array<${wgslScalar(constant.valueType)}>;`);
  }
  for (const global of deviceGlobalBuffers) {
    const elementType = atomicDeviceGlobals.has(global.name)
      ? `atomic<${wgslAtomicScalar(global.valueType)}>`
      : wgslScalar(global.valueType);
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, global.name)}) var<storage, read_write> ${nameFor(global.name, names)}: array<${elementType}>;`);
  }
  for (const shared of sharedMemorySymbols(ir)) {
    lines.push(`var<workgroup> ${nameFor(shared.name, names)}: ${emitSharedType(shared)};`);
  }
  if (uniformParams.length > 0) {
    lines.push("struct Params {");
    for (const param of uniformParams) lines.push(`  ${nameFor(param.name, names)}: ${wgslUniformScalar(param.valueType)},`);
    lines.push("};");
    lines.push(`@group(0) @binding(${bindings.length - 1}) var<uniform> ${UNIFORM_PARAMS_NAME}: Params;`);
  }
  lines.push(
    "",
    `@compute @workgroup_size(${ir.workgroupSize.join(", ")})`,
    "fn main(",
    "  @builtin(global_invocation_id) global_id: vec3<u32>,",
    "  @builtin(local_invocation_id) local_id: vec3<u32>,",
    "  @builtin(workgroup_id) workgroup_id: vec3<u32>,",
    "  @builtin(num_workgroups) num_workgroups: vec3<u32>",
    ") {",
    ...emitSemanticOperations(ir.operations, ir, names, 1),
    "}",
  );
  const wgsl = lines.join("\n");
  return {
    wgsl,
    program: defineWgslKernelProgram({
      name: ir.name,
      wgsl,
      bindings,
      workgroupSize: ir.workgroupSize,
    }),
  };
}

function unsupportedSemanticWgslOperation(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
): SemanticKernelIrOperation | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        if (operation.target.addressSpace === "shared") {
          if (operation.target.pointer || !semanticWgslScalarTypeSupported(operation.target.valueType)) return operation;
          break;
        }
        if (operation.target.addressSpace !== "local" || operation.target.pointer) return operation;
        if (!semanticWgslScalarTypeSupported(operation.target.valueType)) return operation;
        if (operation.target.dimensions.length > 0 && operation.init && !semanticWgslLocalArrayInitSupported(operation.init)) return operation;
        if (operation.target.dimensions.length === 0 && operation.init && !semanticWgslExpressionSupported(operation.init, "scalar")) return operation;
        break;
      case "store":
        if (!semanticWgslAssignmentOperatorSupported(operation.operator)) return operation;
        if (!semanticWgslMemoryRefSupported(operation.target)) return operation;
        if (
          operation.target.addressSpace === "storage" &&
          !ir.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage")
        ) return operation;
        if (!semanticWgslValueExpressionSupported(operation.value, ir)) return operation;
        break;
      case "atomic":
        if (!semanticWgslAtomicSupported(operation, ir)) return operation;
        break;
      case "expression":
        if (!semanticWgslExpressionSupported(operation.expression, "scalar")) return operation;
        break;
      case "branch":
        if (!semanticWgslExpressionSupported(operation.condition, "scalar")) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.consequent, ir) ??
          unsupportedSemanticWgslOperation(operation.alternate, ir);
          if (unsupported) return unsupported;
        }
        break;
      case "loop":
        if (operation.init && !semanticWgslLoopInitSupported(operation.init, ir)) return operation;
        if (operation.condition && !semanticWgslExpressionSupported(operation.condition, "scalar")) return operation;
        if (operation.update && !semanticWgslExpressionSupported(operation.update, "scalar")) return operation;
        {
          const unsupported = unsupportedSemanticWgslOperation(operation.body, ir);
          if (unsupported) return unsupported;
        }
        break;
      case "barrier":
        if (operation.callee !== "__syncthreads") return operation;
        break;
      default:
        return operation;
    }
  }
  return undefined;
}

function semanticWgslParamSupported(param: SemanticKernelIrModule["params"][number]): boolean {
  if (param.addressSpace === "storage") return Boolean(param.pointer) && semanticWgslScalarTypeSupported(param.valueType);
  if (param.addressSpace === "uniform") return semanticWgslScalarTypeSupported(param.valueType);
  return false;
}

function semanticWgslMemorySymbolSupported(symbol: SemanticKernelIrModule["memory"][number]): boolean {
  if (symbol.kind === "local" || symbol.kind === "shared") return true;
  if (symbol.kind === "constant") return !symbol.initialized && semanticWgslScalarTypeSupported(symbol.valueType);
  if (symbol.kind === "device-global") return semanticWgslScalarTypeSupported(symbol.valueType);
  return false;
}

function semanticWgslSharedBarrierShapeSupported(ir: SemanticKernelIrModule): boolean {
  const shared = sharedMemorySymbols(ir);
  if (shared.length === 0 && !operationsContainBarrier(ir.operations)) return true;
  if (!shared.every((symbol) => symbol.dimensions.length === 1 && (symbol.dimensions[0] ?? 0) > 0)) return false;
  return operationsHaveOnlyTopLevelBarriers(ir.operations);
}

function operationsContainBarrier(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) =>
    operation.kind === "barrier" ||
    operation.kind === "branch" && (operationsContainBarrier(operation.consequent) || operationsContainBarrier(operation.alternate)) ||
    operation.kind === "loop" && operationsContainBarrier(operation.body) ||
    operation.kind === "block" && operationsContainBarrier(operation.body)
  );
}

function operationsHaveOnlyTopLevelBarriers(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.every((operation) =>
    operation.kind !== "branch" &&
    operation.kind !== "loop" &&
    operation.kind !== "block"
  );
}

function semanticWgslScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" || valueType === "int" || valueType === "uint";
}

function semanticWgslAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
}

function semanticWgslLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  ir: SemanticKernelIrModule,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? unsupportedSemanticWgslOperation([init], ir) === undefined
    : semanticWgslExpressionSupported(init, "scalar");
}

function semanticWgslMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  if (ref.addressSpace !== "storage" && ref.addressSpace !== "shared" && ref.addressSpace !== "constant" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local") return false;
  if (ref.fields.length > 0) return false;
  if ((ref.addressSpace === "storage" || ref.addressSpace === "constant") && ref.indices.length !== 1) return false;
  if (ref.addressSpace === "device-global" && ref.indices.length > 1) return false;
  if (ref.addressSpace === "local" && ref.indices.length === 0) return false;
  return ref.indices.every((index) => semanticWgslExpressionSupported(index, "scalar"));
}

function semanticWgslAtomicSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (!WGSL_ATOMIC_CALLEES.has(operation.callee)) return false;
  if (!operation.target || (operation.target.addressSpace !== "storage" && operation.target.addressSpace !== "device-global")) return false;
  if (!semanticWgslMemoryRefSupported(operation.target)) return false;
  if (operation.target.addressSpace === "storage" && operation.target.indices.length !== 1) return false;
  if (operation.target.addressSpace === "device-global" && operation.target.indices.length > 1) return false;
  if (operation.target.fields.length > 0) return false;
  if (operation.target.valueType !== "uint" && operation.target.valueType !== "int") return false;
  if (!semanticWgslAtomicTargetRootSupported(operation.target, ir)) {
    return false;
  }
  const expectedArgs = operation.callee === "atomicCAS" || operation.callee === "atomicCAS_system" ? 3 : 2;
  return operation.args.length >= expectedArgs &&
    operation.args.slice(1, expectedArgs).every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslValueExpressionSupported(expression: SemanticExpression, ir: SemanticKernelIrModule): boolean {
  return semanticWgslExpressionSupported(expression, "scalar") ||
    expression.kind === "call" && (semanticWgslAtomicCallSupported(expression, ir) || semanticWgslMathCallSupported(expression));
}

function semanticWgslLocalArrayInitSupported(expression: SemanticExpression): boolean {
  return expression.kind === "initializer" &&
    flattenInitializerExpressions(expression).every((item) => semanticWgslExpressionSupported(item, "scalar"));
}

function semanticWgslMathCallSupported(expression: Extract<SemanticExpression, { readonly kind: "call" }>): boolean {
  if (expression.callee.kind !== "symbol" || !SEMANTIC_MATH_CALLS.has(expression.callee.name)) return false;
  const arity = semanticMathCallArity(expression.callee.name);
  return expression.args.length === arity && expression.args.every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslAtomicCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "symbol" || !WGSL_ATOMIC_CALLEES.has(expression.callee.name)) return false;
  const target = semanticAtomicCallTarget(expression);
  if (!target || (target.addressSpace !== "storage" && target.addressSpace !== "device-global")) return false;
  if (!semanticWgslMemoryRefSupported(target)) return false;
  if (target.addressSpace === "storage" && target.indices.length !== 1) return false;
  if (target.addressSpace === "device-global" && target.indices.length > 1) return false;
  if (target.fields.length > 0) return false;
  if (target.valueType !== "uint" && target.valueType !== "int") return false;
  if (!semanticWgslAtomicTargetRootSupported(target, ir)) return false;
  const expectedArgs = expression.callee.name === "atomicCAS" || expression.callee.name === "atomicCAS_system" ? 3 : 2;
  return expression.args.length >= expectedArgs &&
    expression.args.slice(1, expectedArgs).every((arg) => semanticWgslExpressionSupported(arg, "scalar"));
}

function semanticWgslAtomicTargetRootSupported(ref: SemanticMemoryRef, ir: SemanticKernelIrModule): boolean {
  if (ref.addressSpace === "storage") {
    return ir.params.some((param) => param.name === ref.base && param.addressSpace === "storage" && !param.constant);
  }
  if (ref.addressSpace === "device-global") {
    return ir.memory.some((symbol) => symbol.name === ref.base && symbol.kind === "device-global");
  }
  return false;
}

function semanticWgslExpressionSupported(expression: SemanticExpression, expected: "scalar" | "any"): boolean {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number";
    case "symbol":
      return expression.addressSpace === "uniform" ||
        expression.addressSpace === "local" ||
        expression.addressSpace === "constant" ||
        expression.addressSpace === "device-global" ||
        expression.addressSpace === "shared" ||
        BUILTIN_VECTOR_NAMES.has(expression.name);
    case "member":
      return expression.object.kind === "symbol" &&
        BUILTIN_VECTOR_NAMES.has(expression.object.name) &&
        (expression.property === "x" || expression.property === "y" || expression.property === "z");
    case "index":
      return expected === "scalar" && semanticWgslMemoryRefSupported(memoryRefFromIndexExpression(expression) ?? unsupportedMemoryRef(expression.span));
    case "cast":
      return !expression.pointer && semanticWgslExpressionSupported(expression.expression, "scalar");
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" && semanticWgslExpressionSupported(expression.argument, "scalar");
    case "binary":
      return semanticWgslExpressionSupported(expression.left, "scalar") &&
        semanticWgslExpressionSupported(expression.right, "scalar");
    case "conditional":
      return semanticWgslExpressionSupported(expression.condition, "scalar") &&
        semanticWgslExpressionSupported(expression.consequent, expected) &&
        semanticWgslExpressionSupported(expression.alternate, expected);
    case "assignment":
      return expression.operator === "=" &&
        expression.target.kind === "symbol" &&
        expression.target.addressSpace === "local" &&
        semanticWgslExpressionSupported(expression.value, "scalar");
    case "update":
      return expression.argument.kind === "symbol" &&
        expression.argument.addressSpace === "local" &&
        (expression.operator === "++" || expression.operator === "--");
    case "sequence":
      return expression.expressions.every((item) => semanticWgslExpressionSupported(item, "scalar"));
    case "call":
      return semanticWgslMathCallSupported(expression);
    case "initializer":
      return false;
  }
}

function emitSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
): readonly string[] {
  return operations.flatMap((operation) => emitSemanticOperation(operation, ir, names, indentLevel));
}

function emitSemanticOperation(
  operation: SemanticKernelIrOperation,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  switch (operation.kind) {
    case "declare": {
      if (operation.target.addressSpace === "shared") return [];
      if (operation.target.dimensions.length > 0) {
        return [
          `${prefix}var ${nameFor(operation.target.name, names)}: ${emitLocalArrayType(operation.target)};`,
          ...emitLocalArrayInit(operation, ir, names, indentLevel),
        ];
      }
      const type = wgslScalar(operation.target.valueType);
      const init = operation.init ? ` = ${emitSemanticExpressionAs(operation.init, ir, names, wgslValueScalar(operation.target.valueType))}` : "";
      return [`${prefix}var ${nameFor(operation.target.name, names)}: ${type}${init};`];
    }
    case "store":
      return [`${prefix}${emitSemanticStore(operation, ir, names)};`];
    case "atomic":
      return [`${prefix}${emitSemanticAtomic(operation, ir, names)};`];
    case "expression":
      return [`${prefix}${emitSemanticExpression(operation.expression, ir, names)};`];
    case "branch": {
      const lines = [`${prefix}if (${emitTruthiness(operation.condition, ir, names)}) {`];
      lines.push(...emitSemanticOperations(operation.consequent, ir, names, indentLevel + 1));
      if (operation.alternate.length > 0) {
        lines.push(`${prefix}} else {`);
        lines.push(...emitSemanticOperations(operation.alternate, ir, names, indentLevel + 1));
      }
      lines.push(`${prefix}}`);
      return lines;
    }
    case "loop":
      return emitSemanticLoop(operation, ir, names, indentLevel);
    case "barrier":
      return [`${prefix}workgroupBarrier();`];
    default:
      throw semanticWgslError(`semantic WGSL does not support ${operation.kind}`, operation.span);
  }
}

function emitSemanticStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  const target = emitSemanticMemoryRef(operation.target, ir, names);
  if (semanticAtomicStorageNames(ir.operations).has(operation.target.base) || semanticAtomicDeviceGlobalNames(ir.operations).has(operation.target.base)) {
    if (operation.operator !== "=") {
      throw semanticWgslError(`semantic WGSL does not support atomic storage assignment '${operation.operator}'`, operation.span);
    }
    const atomicValue = emitSemanticExpressionAs(operation.value, ir, names, wgslAtomicScalar(operation.target.valueType));
    return `atomicStore(&${target}, ${atomicValue})`;
  }
  const value = emitSemanticExpressionAs(operation.value, ir, names, wgslValueScalar(operation.target.valueType));
  if (operation.operator === "=") return `${target} = ${value}`;
  if (operation.operator === "+=") return `${target} = (${target} + ${value})`;
  if (operation.operator === "-=") return `${target} = (${target} - ${value})`;
  throw semanticWgslError(`semantic WGSL does not support assignment '${operation.operator}'`, operation.span);
}

function emitLocalArrayInit(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
): readonly string[] {
  if (!operation.init || operation.init.kind !== "initializer") return [];
  const prefix = "  ".repeat(indentLevel);
  return flattenInitializerExpressions(operation.init)
    .slice(0, totalElements(operation.target.dimensions))
    .map((value, index) => {
      const indices = flatIndicesForDimensions(operation.target.dimensions, index)
        .map((item) => `[${item}u]`)
        .join("");
      return `${prefix}${nameFor(operation.target.name, names)}${indices} = ${emitSemanticExpressionAs(value, ir, names, wgslValueScalar(operation.target.valueType))};`;
    });
}

function emitSemanticAtomic(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "atomic" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  const wgslCallee = WGSL_ATOMIC_CALLEES.get(operation.callee);
  if (!operation.target || !wgslCallee) {
    throw semanticWgslError(`semantic WGSL does not support atomic '${operation.callee}'`, operation.span);
  }
  const target = emitSemanticMemoryRef(operation.target, ir, names);
  const operands = operation.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  if (operands.length === 0 || operands.some((operand) => operand === undefined)) {
    throw semanticWgslError(`semantic WGSL atomic '${operation.callee}' missing operand`, operation.span);
  }
  const emitted = operands.map((operand) =>
    emitSemanticExpressionAs(operand!, ir, names, wgslAtomicScalar(operation.target!.valueType))
  );
  return `_ = ${wgslCallee}(&${target}, ${emitted.join(", ")})`;
}

function emitSemanticLoop(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  if (operation.loopKind === "for") {
    const init = operation.init ? emitSemanticLoopInit(operation.init, ir, names) : "";
    const condition = operation.condition ? emitTruthiness(operation.condition, ir, names) : "true";
    const update = operation.update ? emitSemanticExpression(operation.update, ir, names) : "";
    return [
      `${prefix}for (${init}; ${condition}; ${update}) {`,
      ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1),
      `${prefix}}`,
    ];
  }
  if (operation.loopKind === "while") {
    return [
      `${prefix}while (${operation.condition ? emitTruthiness(operation.condition, ir, names) : "true"}) {`,
      ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1),
      `${prefix}}`,
    ];
  }
  return [
    `${prefix}loop {`,
    ...emitSemanticOperations(operation.body, ir, names, indentLevel + 1),
    `${"  ".repeat(indentLevel + 1)}if (!(${operation.condition ? emitTruthiness(operation.condition, ir, names) : "false"})) { break; }`,
    `${prefix}}`,
  ];
}

function emitSemanticLoopInit(
  init: SemanticKernelIrOperation | SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (!isSemanticKernelIrOperation(init)) return emitSemanticExpression(init, ir, names);
  if (init.kind === "declare") {
    const type = wgslScalar(init.target.valueType);
    const value = init.init ? emitSemanticExpressionAs(init.init, ir, names, wgslValueScalar(init.target.valueType)) : zeroForType(type);
    return `var ${nameFor(init.target.name, names)}: ${type} = ${value}`;
  }
  if (init.kind === "expression") return emitSemanticExpression(init.expression, ir, names);
  throw semanticWgslError(`semantic WGSL does not support ${init.kind} loop initializer`, init.span);
}

function emitSemanticExpression(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  switch (expression.kind) {
    case "literal":
      if (typeof expression.value !== "number") throw semanticWgslError("semantic WGSL supports numeric literals only", expression.span);
      return emitNumberLiteral(expression.value, expression.valueType);
    case "symbol":
      if (expression.addressSpace === "uniform") return `${UNIFORM_PARAMS_NAME}.${nameFor(expression.name, names)}`;
      if (expression.addressSpace === "constant") return `${UNIFORM_PARAMS_NAME}.${nameFor(expression.name, names)}`;
      if (expression.addressSpace === "device-global") {
        const ref = `${nameFor(expression.name, names)}[0u]`;
        return semanticAtomicDeviceGlobalNames(ir.operations).has(expression.name) ? `atomicLoad(&${ref})` : ref;
      }
      return nameFor(expression.name, names);
    case "member":
      return emitSemanticMember(expression, ir, names);
    case "index": {
      const ref = memoryRefFromIndexExpression(expression);
      if (ref) {
        const memoryRef = emitSemanticMemoryRef(ref, ir, names);
        if (semanticAtomicStorageNames(ir.operations).has(ref.base) || semanticAtomicDeviceGlobalNames(ir.operations).has(ref.base)) return `atomicLoad(&${memoryRef})`;
        return memoryRef;
      }
      throw semanticWgslError("semantic WGSL does not support index target", expression.span);
    }
    case "cast":
      return `${wgslScalar(expression.valueType)}(${emitSemanticExpression(expression.expression, ir, names)})`;
    case "unary":
      return emitSemanticUnary(expression, ir, names);
    case "binary":
      return emitSemanticBinary(expression, ir, names);
    case "conditional":
      return `select(${emitSemanticExpression(expression.alternate, ir, names)}, ${emitSemanticExpression(expression.consequent, ir, names)}, ${emitTruthiness(expression.condition, ir, names)})`;
    case "assignment":
      if (expression.target.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar assignment targets only", expression.target.span);
      return `(${nameFor(expression.target.name, names)} = ${emitSemanticExpressionAs(expression.value, ir, names, wgslValueScalar(expression.valueType))})`;
    case "update":
      return emitSemanticUpdate(expression, names);
    case "sequence":
      return emitSemanticExpression(expression.expressions.at(-1) ?? zeroExpression(expression.span), ir, names);
    case "call":
      if (semanticWgslAtomicCallSupported(expression, ir)) return emitSemanticAtomicCall(expression, ir, names);
      if (semanticWgslMathCallSupported(expression)) return emitSemanticMathCall(expression, ir, names);
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
    case "initializer":
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
  }
}

function emitSemanticUpdate(
  expression: Extract<SemanticExpression, { readonly kind: "update" }>,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.argument.kind !== "symbol") throw semanticWgslError("semantic WGSL supports local scalar updates only", expression.span);
  const name = nameFor(expression.argument.name, names);
  if (expression.operator === "++") return `${name} += ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))}`;
  if (expression.operator === "--") return `${name} -= ${emitNumberLiteral(1, expression.valueType, wgslValueScalar(expression.valueType))}`;
  throw semanticWgslError(`semantic WGSL does not support update '${expression.operator}'`, expression.span);
}

function emitSemanticExpressionAs(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  targetType: WgslValueType,
): string {
  if (expression.kind === "literal" && typeof expression.value === "number") {
    return emitNumberLiteral(expression.value, expression.valueType, targetType);
  }
  const emitted = emitSemanticExpression(expression, ir, names);
  const atomicValueType = semanticAtomicCallValueType(expression);
  if (atomicValueType) {
    const sourceType = wgslAtomicScalar(atomicValueType);
    if (sourceType === targetType) return emitted;
    return `${targetType}(${emitted})`;
  }
  if (expression.kind === "call" && semanticWgslMathCallSupported(expression)) {
    if (targetType === "f32") return emitted;
    return `${targetType}(${emitted})`;
  }
  const sourceType = semanticExpressionWgslScalar(expression);
  if (sourceType === targetType) return emitted;
  return `${targetType}(${emitted})`;
}

function emitSemanticAtomicCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL atomic call requires symbol callee", expression.span);
  const wgslCallee = WGSL_ATOMIC_CALLEES.get(expression.callee.name);
  const target = semanticAtomicCallTarget(expression);
  if (!wgslCallee || !target) throw semanticWgslError(`semantic WGSL does not support atomic '${expression.callee.name}'`, expression.span);
  const memoryRef = emitSemanticMemoryRef(target, ir, names);
  const operands = expression.args.slice(1, wgslCallee === "atomicCompareExchangeWeak" ? 3 : 2);
  const emitted = operands.map((operand) => emitSemanticExpressionAs(operand, ir, names, wgslAtomicScalar(target.valueType)));
  const call = `${wgslCallee}(&${memoryRef}, ${emitted.join(", ")})`;
  return wgslCallee === "atomicCompareExchangeWeak" ? `${call}.old_value` : call;
}

function emitSemanticMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.callee.kind !== "symbol") throw semanticWgslError("semantic WGSL math call requires symbol callee", expression.span);
  const wgslCallee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  if (!wgslCallee) throw semanticWgslError(`semantic WGSL does not support math call '${expression.callee.name}'`, expression.span);
  return `${wgslCallee}(${expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32")).join(", ")})`;
}

function semanticMathCallArity(name: string): number {
  return name === "fmin" ||
    name === "fminf" ||
    name === "min" ||
    name === "fmax" ||
    name === "fmaxf" ||
    name === "max" ||
    name === "pow" ||
    name === "powf"
    ? 2
    : 1;
}

function emitSemanticMember(
  expression: Extract<SemanticExpression, { readonly kind: "member" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.object.kind !== "symbol") throw semanticWgslError("semantic WGSL supports builtin vector members only", expression.span);
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  switch (expression.object.name) {
    case "threadIdx":
      return ir.workgroupSize[axisIndex] === 1 ? "0u" : `local_id.${expression.property}`;
    case "blockIdx":
      return `workgroup_id.${expression.property}`;
    case "blockDim":
      return `${ir.workgroupSize[axisIndex]}u`;
    case "gridDim":
      return `num_workgroups.${expression.property}`;
    default:
      return `${emitSemanticExpression(expression.object, ir, names)}.${expression.property}`;
  }
}

function emitSemanticUnary(
  expression: Extract<SemanticExpression, { readonly kind: "unary" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.operator === "!") return `!(${emitTruthiness(expression.argument, ir, names)})`;
  if (expression.operator === "~") return `~(${emitSemanticExpression(expression.argument, ir, names)})`;
  if (expression.operator === "+") return emitSemanticExpression(expression.argument, ir, names);
  if (expression.operator === "-") return `-(${emitSemanticExpression(expression.argument, ir, names)})`;
  throw semanticWgslError(`semantic WGSL does not support unary '${expression.operator}'`, expression.span);
}

function emitSemanticBinary(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (LOGICAL_OPERATORS.has(expression.operator)) {
    return `(${emitTruthiness(expression.left, ir, names)} ${expression.operator} ${emitTruthiness(expression.right, ir, names)})`;
  }
  const operandType = semanticBinaryOperandType(expression);
  const left = emitSemanticExpressionAs(expression.left, ir, names, operandType);
  const right = emitSemanticExpressionAs(expression.right, ir, names, operandType);
  return `(${left} ${expression.operator} ${right})`;
}

function semanticBinaryOperandType(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): WgslValueType {
  const left = semanticExpressionWgslScalar(expression.left);
  const right = semanticExpressionWgslScalar(expression.right);
  const result = wgslValueScalar(expression.valueType);
  if (left === "f32" || right === "f32" || result === "f32") return "f32";
  if (left === "u32" || right === "u32" || result === "u32") return "u32";
  return "i32";
}

function emitTruthiness(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (expression.kind === "binary" && (COMPARISON_OPERATORS.has(expression.operator) || LOGICAL_OPERATORS.has(expression.operator))) {
    return emitSemanticBinary(expression, ir, names);
  }
  return `(${emitSemanticExpression(expression, ir, names)} != 0)`;
}

function emitSemanticMemoryRef(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (ref.fields.length > 0) throw semanticWgslError("semantic WGSL supports scalar memory refs only", ref.span);
  if (ref.addressSpace === "storage") {
    if (ref.indices.length !== 1) throw semanticWgslError("semantic WGSL supports 1D storage refs only", ref.span);
    return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32")}]`;
  }
  if (ref.addressSpace === "constant") {
    if (ref.indices.length !== 1) throw semanticWgslError("semantic WGSL supports 1D constant refs only", ref.span);
    return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32")}]`;
  }
  if (ref.addressSpace === "device-global") {
    if (ref.indices.length > 1) throw semanticWgslError("semantic WGSL supports scalar/1D device globals only", ref.span);
    const index = ref.indices[0] ? emitSemanticExpressionAs(ref.indices[0], ir, names, "u32") : "0u";
    return `${nameFor(ref.base, names)}[${index}]`;
  }
  if (ref.addressSpace === "local") {
    const local = localMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
    if (!local) throw semanticWgslError(`unknown local memory '${ref.base}'`, ref.span);
    if (ref.indices.length !== local.dimensions.length) throw semanticWgslError(`local memory '${ref.base}' index rank mismatch`, ref.span);
    return `${nameFor(ref.base, names)}${ref.indices.map((index) => `[${emitSemanticExpressionAs(index, ir, names, "u32")}]`).join("")}`;
  }
  if (ref.addressSpace === "shared") {
    const shared = sharedMemorySymbols(ir).find((symbol) => symbol.name === ref.base);
    if (!shared) throw semanticWgslError(`unknown shared memory '${ref.base}'`, ref.span);
    return `${nameFor(ref.base, names)}[${emitFlatSharedIndex(shared, ref.indices, ir, names)}]`;
  }
  throw semanticWgslError(`semantic WGSL does not support ${ref.addressSpace} memory refs`, ref.span);
}

function memoryRefFromIndexExpression(expression: Extract<SemanticExpression, { readonly kind: "index" }>): SemanticMemoryRef | undefined {
  const flattened = flattenMemoryRef(expression);
  if (!flattened || (flattened.base.addressSpace !== "storage" && flattened.base.addressSpace !== "shared" && flattened.base.addressSpace !== "constant" && flattened.base.addressSpace !== "device-global" && flattened.base.addressSpace !== "local")) return undefined;
  return {
    base: flattened.base.name,
    addressSpace: flattened.base.addressSpace,
    ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
    indices: flattened.indices,
    fields: [],
    span: expression.span,
  };
}

function flattenMemoryRef(expression: SemanticExpression): {
  readonly base: Extract<SemanticExpression, { readonly kind: "symbol" }>;
  readonly indices: readonly SemanticExpression[];
} | undefined {
  if (expression.kind === "symbol") return { base: expression, indices: [] };
  if (expression.kind !== "index") return undefined;
  const target = flattenMemoryRef(expression.target);
  if (!target) return undefined;
  return { base: target.base, indices: [...target.indices, expression.index] };
}

function unsupportedMemoryRef(span: SourceSpan): SemanticMemoryRef {
  return { base: "", addressSpace: "unknown", indices: [], fields: [], span };
}

function sharedMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "shared");
}

function constantMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "constant");
}

function deviceGlobalMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "device-global");
}

function localMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "local" && symbol.dimensions.length > 0);
}

function emitLocalArrayType(symbol: SemanticKernelIrModule["memory"][number]): string {
  return symbol.dimensions.reduceRight<string>(
    (element, dimension) => `array<${element}, ${Math.max(1, dimension)}>`,
    wgslScalar(symbol.valueType),
  );
}

function emitSharedType(symbol: SemanticKernelIrModule["memory"][number]): string {
  return `array<${wgslScalar(symbol.valueType)}, ${Math.max(1, totalElements(symbol.dimensions))}>`;
}

function totalElements(dimensions: readonly number[]): number {
  return dimensions.length === 0 ? 1 : dimensions.reduce((product, dimension) => product * dimension, 1);
}

function flattenInitializerExpressions(expression: SemanticExpression): readonly SemanticExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

function flatIndicesForDimensions(dimensions: readonly number[], flatIndex: number): readonly number[] {
  return dimensions.map((_, offset) => {
    const stride = dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
    return Math.floor(flatIndex / stride) % Math.max(1, dimensions[offset] ?? 1);
  });
}

function emitFlatSharedIndex(
  symbol: SemanticKernelIrModule["memory"][number],
  indices: readonly SemanticExpression[],
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
): string {
  if (indices.length === 0) return "0u";
  if (indices.length === 1) return emitSemanticExpressionAs(indices[0]!, ir, names, "u32");
  if (indices.length !== symbol.dimensions.length) {
    throw semanticWgslError(`shared memory '${symbol.name}' index rank mismatch`, symbol.span);
  }
  const terms = indices.map((index, offset) => {
    const stride = symbol.dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
    const emitted = emitSemanticExpressionAs(index, ir, names, "u32");
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function collectOperationNames(
  operation: SemanticKernelIrOperation,
  names: Set<string>,
): void {
  if (operation.kind === "declare") names.add(operation.target.name);
  if (operation.kind === "branch") {
    for (const child of [...operation.consequent, ...operation.alternate]) collectOperationNames(child, names);
  }
  if (operation.kind === "loop") {
    if (operation.init && isSemanticKernelIrOperation(operation.init)) collectOperationNames(operation.init, names);
    for (const child of operation.body) collectOperationNames(child, names);
  }
}

function wgslBindingType(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  if (scalar !== "bool") return scalar;
  return "u32";
}

function wgslScalar(valueType: CudaLiteScalarType | undefined): WgslValueType | "bool" {
  if (valueType === "int") return "i32";
  if (valueType === "uint") return "u32";
  if (valueType === "bool") return "bool";
  return "f32";
}

function wgslValueScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  return scalar === "bool" ? "u32" : scalar;
}

function wgslAtomicScalar(valueType: CudaLiteScalarType | undefined): Extract<WgslValueType, "i32" | "u32"> {
  return valueType === "int" ? "i32" : "u32";
}

function wgslUniformScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "int") return "i32";
  if (valueType === "uint" || valueType === "bool") return "u32";
  return "f32";
}

function semanticExpressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

function semanticExpressionWgslScalar(expression: SemanticExpression): WgslValueType {
  switch (expression.kind) {
    case "call": {
      if (semanticWgslMathCallSupported(expression)) return "f32";
      const atomicType = semanticAtomicCallValueType(expression);
      return atomicType ? wgslAtomicScalar(atomicType) : wgslValueScalar(expression.valueType);
    }
    case "binary": {
      const left = semanticExpressionWgslScalar(expression.left);
      const right = semanticExpressionWgslScalar(expression.right);
      const result = wgslValueScalar(expression.valueType);
      if (left === "f32" || right === "f32" || result === "f32") return "f32";
      if (left === "u32" || right === "u32" || result === "u32") return "u32";
      return "i32";
    }
    case "conditional": {
      const consequent = semanticExpressionWgslScalar(expression.consequent);
      const alternate = semanticExpressionWgslScalar(expression.alternate);
      const result = wgslValueScalar(expression.valueType);
      if (consequent === "f32" || alternate === "f32" || result === "f32") return "f32";
      if (consequent === "u32" || alternate === "u32" || result === "u32") return "u32";
      return "i32";
    }
    case "sequence":
      return expression.expressions.length > 0
        ? semanticExpressionWgslScalar(expression.expressions.at(-1)!)
        : wgslValueScalar(expression.valueType);
    default:
      return wgslValueScalar(semanticExpressionValueType(expression));
  }
}

function emitNumberLiteral(value: number, valueType: CudaLiteScalarType | undefined, expectedType?: WgslValueType): string {
  const type = expectedType ?? wgslScalar(valueType);
  if (type === "u32") return `${Math.trunc(value) >>> 0}u`;
  if (type === "i32" && value > 2147483647) return `bitcast<i32>(${Math.trunc(value) >>> 0}u)`;
  if (type === "i32") return String(Math.trunc(value));
  if (Number.isInteger(value)) return `${value}.0`;
  return String(value);
}

function zeroExpression(span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: 0, valueType: "int", span };
}

function zeroForType(valueType: WgslValueType | "bool"): string {
  if (valueType === "u32") return "0u";
  if (valueType === "i32") return "0";
  if (valueType === "bool") return "false";
  return "0.0";
}

function bindingIndexFor(bindings: readonly WgslKernelBindingInput[], name: string): number {
  const binding = bindings.find((item) => item.name === name)?.binding;
  return binding ?? 0;
}

function semanticAtomicStorageNames(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "storage") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicStorageNamesFromOperation(operation)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicStorageNames(operation.consequent)) names.add(name);
      for (const name of semanticAtomicStorageNames(operation.alternate)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicStorageNames([operation.init])) names.add(name);
      }
      for (const name of semanticAtomicStorageNames(operation.body)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicStorageNames(operation.body)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicDeviceGlobalNames(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "device-global") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicDeviceGlobalNamesFromOperation(operation)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.consequent)) names.add(name);
      for (const name of semanticAtomicDeviceGlobalNames(operation.alternate)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicDeviceGlobalNames([operation.init])) names.add(name);
      }
      for (const name of semanticAtomicDeviceGlobalNames(operation.body)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.body)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicStorageNamesFromOperation(operation: SemanticKernelIrOperation): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  for (const expression of expressions) {
    for (const name of semanticAtomicStorageNamesFromExpression(expression)) names.add(name);
  }
  return names;
}

function semanticAtomicStorageNamesFromExpression(expression: SemanticExpression): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === "storage") names.add(target.base);
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicStorageNamesFromExpression(child)) names.add(name);
  }
  return names;
}

function semanticAtomicDeviceGlobalNamesFromOperation(operation: SemanticKernelIrOperation): ReadonlySet<string> {
  const names = new Set<string>();
  for (const name of semanticAtomicNamesFromOperation(operation, "device-global")) names.add(name);
  return names;
}

function semanticAtomicNamesFromOperation(
  operation: SemanticKernelIrOperation,
  addressSpace: "storage" | "device-global",
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  for (const expression of expressions) {
    for (const name of semanticAtomicNamesFromExpression(expression, addressSpace)) names.add(name);
  }
  return names;
}

function semanticAtomicNamesFromExpression(
  expression: SemanticExpression,
  addressSpace: "storage" | "device-global",
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === addressSpace) names.add(target.base);
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicNamesFromExpression(child, addressSpace)) names.add(name);
  }
  return names;
}

function semanticExpressionChildren(expression: SemanticExpression): readonly SemanticExpression[] {
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return [];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "call":
      return [expression.callee, ...expression.args];
    case "cast":
      return [expression.expression];
    case "unary":
    case "update":
      return [expression.argument];
    case "binary":
      return [expression.left, expression.right];
    case "conditional":
      return [expression.condition, expression.consequent, expression.alternate];
    case "assignment":
      return [expression.target, expression.value];
    case "initializer":
      return expression.elements;
    case "sequence":
      return expression.expressions;
  }
}

function semanticAtomicCallTarget(expression: Extract<SemanticExpression, { readonly kind: "call" }>): SemanticMemoryRef | undefined {
  const firstArg = expression.args[0];
  if (!firstArg) return undefined;
  if (firstArg.kind === "unary" && firstArg.operator === "&" && firstArg.argument.kind === "index") {
    return memoryRefFromIndexExpression(firstArg.argument);
  }
  if (firstArg.kind === "unary" && firstArg.operator === "&" && firstArg.argument.kind === "symbol" && firstArg.argument.addressSpace === "device-global") {
    return {
      base: firstArg.argument.name,
      addressSpace: "device-global",
      ...(firstArg.argument.valueType === undefined ? {} : { valueType: firstArg.argument.valueType }),
      indices: [],
      fields: [],
      span: firstArg.argument.span,
    };
  }
  if (firstArg.kind === "index") return memoryRefFromIndexExpression(firstArg);
  return undefined;
}

function semanticAtomicCallValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  if (expression.kind !== "call") return undefined;
  return semanticAtomicCallTarget(expression)?.valueType;
}

function nameFor(name: string, names: ReadonlyMap<string, string>): string {
  if (BUILTIN_VECTOR_NAMES.has(name)) return name;
  return names.get(name) ?? safeWgslIdentifier(name);
}

function semanticWgslError(message: string, span: SourceSpan): CudaLiteCompilerError {
  const diagnostic: CudaLiteDiagnostic = {
    code: "semantic-wgsl-unsupported",
    severity: "error",
    message,
    span,
  };
  return new CudaLiteCompilerError(message, [diagnostic]);
}

function isSemanticKernelIrOperation(
  value: SemanticKernelIrOperation | SemanticExpression,
): value is SemanticKernelIrOperation {
  switch (value.kind) {
    case "declare":
    case "dim3-declare":
    case "cooperative-group-declare":
    case "load":
    case "store":
    case "atomic":
    case "expression":
    case "branch":
    case "loop":
    case "barrier":
    case "device-launch":
    case "inline-asm":
    case "return":
    case "continue":
    case "break":
    case "block":
      return true;
    case "call":
      return typeof value.callee === "string";
    default:
      return false;
  }
}
