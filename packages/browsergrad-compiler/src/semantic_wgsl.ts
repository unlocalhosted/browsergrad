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

export function canEmitSemanticKernelIrWgsl(ir: SemanticKernelIrModule): boolean {
  return unsupportedSemanticWgslOperation(ir.operations, ir) === undefined &&
    ir.requiredFeatures.length === 0 &&
    ir.params.every(semanticWgslParamSupported) &&
    ir.memory.every((symbol) => symbol.kind === "local" || symbol.kind === "shared");
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
  const uniformParams = ir.params.filter((param) => param.addressSpace === "uniform");
  const bindings: WgslKernelBindingInput[] = ir.params
    .filter((param) => param.addressSpace === "storage")
    .map((param, binding) => ({
      kind: "storage",
      name: param.name,
      valueType: wgslBindingType(param.valueType),
      access: param.constant ? "read" : "read_write",
      binding,
    }));
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
    lines.push(`@group(0) @binding(${bindingIndexFor(bindings, param.name)}) var<storage, ${access}> ${nameFor(param.name, names)}: array<${wgslScalar(param.valueType)}>;`);
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
        if (operation.target.addressSpace !== "local" || operation.target.pointer || operation.target.dimensions.length > 0) return operation;
        if (!semanticWgslScalarTypeSupported(operation.target.valueType)) return operation;
        if (operation.init && !semanticWgslExpressionSupported(operation.init, "scalar")) return operation;
        break;
      case "store":
        if (operation.operator !== "=") return operation;
        if (!semanticWgslMemoryRefSupported(operation.target)) return operation;
        if (!ir.params.some((param) => param.name === operation.target.base && param.addressSpace === "storage")) return operation;
        if (!semanticWgslExpressionSupported(operation.value, "scalar")) return operation;
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

function semanticWgslScalarTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" || valueType === "int" || valueType === "uint";
}

function semanticWgslMemoryRefSupported(ref: SemanticMemoryRef): boolean {
  return ref.addressSpace === "storage" &&
    ref.indices.length === 1 &&
    ref.fields.length === 0 &&
    semanticWgslExpressionSupported(ref.indices[0]!, "scalar");
}

function semanticWgslExpressionSupported(expression: SemanticExpression, expected: "scalar" | "any"): boolean {
  switch (expression.kind) {
    case "literal":
      return typeof expression.value === "number";
    case "symbol":
      return expression.addressSpace === "uniform" || expression.addressSpace === "local" || BUILTIN_VECTOR_NAMES.has(expression.name);
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
    case "sequence":
      return expression.expressions.every((item) => semanticWgslExpressionSupported(item, "scalar"));
    case "call":
    case "update":
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
      const type = wgslScalar(operation.target.valueType);
      const init = operation.init ? ` = ${emitSemanticExpressionAs(operation.init, ir, names, wgslValueScalar(operation.target.valueType))}` : "";
      return [`${prefix}var ${nameFor(operation.target.name, names)}: ${type}${init};`];
    }
    case "store":
      return [`${prefix}${emitSemanticMemoryRef(operation.target, ir, names)} = ${emitSemanticExpressionAs(operation.value, ir, names, wgslValueScalar(operation.target.valueType))};`];
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
    default:
      throw semanticWgslError(`semantic WGSL does not support ${operation.kind}`, operation.span);
  }
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
      return nameFor(expression.name, names);
    case "member":
      return emitSemanticMember(expression, ir, names);
    case "index": {
      const ref = memoryRefFromIndexExpression(expression);
      if (ref) return emitSemanticMemoryRef(ref, ir, names);
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
    case "sequence":
      return emitSemanticExpression(expression.expressions.at(-1) ?? zeroExpression(expression.span), ir, names);
    case "call":
    case "update":
    case "initializer":
      throw semanticWgslError(`semantic WGSL does not support ${expression.kind} expression`, expression.span);
  }
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
  const sourceType = wgslValueScalar(semanticExpressionValueType(expression));
  if (sourceType === targetType) return emitted;
  return `${targetType}(${emitted})`;
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
  const left = wgslValueScalar(semanticExpressionValueType(expression.left));
  const right = wgslValueScalar(semanticExpressionValueType(expression.right));
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
  if (ref.indices.length !== 1 || ref.fields.length > 0) throw semanticWgslError("semantic WGSL supports 1D storage refs only", ref.span);
  return `${nameFor(ref.base, names)}[${emitSemanticExpressionAs(ref.indices[0]!, ir, names, "u32")}]`;
}

function memoryRefFromIndexExpression(expression: Extract<SemanticExpression, { readonly kind: "index" }>): SemanticMemoryRef | undefined {
  const target = expression.target;
  if (target.kind !== "symbol" || target.addressSpace !== "storage") return undefined;
  return {
    base: target.name,
    addressSpace: target.addressSpace,
    ...(target.valueType === undefined ? {} : { valueType: target.valueType }),
    indices: [expression.index],
    fields: [],
    span: expression.span,
  };
}

function unsupportedMemoryRef(span: SourceSpan): SemanticMemoryRef {
  return { base: "", addressSpace: "unknown", indices: [], fields: [], span };
}

function collectOperationNames(
  operation: SemanticKernelIrOperation,
  names: Set<string>,
): void {
  if (operation.kind === "declare") names.add(operation.target.name);
  if (operation.kind === "branch") {
    for (const child of [...operation.consequent, ...operation.alternate]) collectOperationNames(child, names);
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

function wgslUniformScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "int") return "i32";
  if (valueType === "uint" || valueType === "bool") return "u32";
  return "f32";
}

function semanticExpressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
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

function bindingIndexFor(bindings: readonly WgslKernelBindingInput[], name: string): number {
  const binding = bindings.find((item) => item.name === name)?.binding;
  return binding ?? 0;
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
