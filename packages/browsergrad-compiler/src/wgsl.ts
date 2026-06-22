import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslKernelProgram,
} from "@unlocalhosted/browsergrad-kernels";
import { expressionName, rootIdentifier } from "./analyzer.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteCallExpression,
  type CudaLiteDeviceFunction,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface EmitKernelIrWgslOptions {
  readonly features?: Partial<Record<"shader-f16" | "subgroups" | "compatibility", boolean>>;
}

export interface KernelIrWgslOutput {
  readonly wgsl: string;
  readonly program: WgslKernelProgram;
}

export function emitKernelIrWgsl(
  ir: KernelIrModule,
  options: EmitKernelIrWgslOptions = {},
): KernelIrWgslOutput {
  if (ir.requiredFeatures.includes("shader-f16") && !options.features?.["shader-f16"]) {
    throw featureError("missing-feature-shader-f16", "half requires WebGPU shader-f16 support");
  }
  if (ir.requiredFeatures.includes("subgroups") && !options.features?.subgroups) {
    throw featureError("missing-feature-subgroups", "bg_subgroup_add requires WebGPU subgroups support");
  }

  const context = createEmitContext(ir);
  const lines: string[] = [];
  if (ir.requiredFeatures.includes("shader-f16")) lines.push("enable f16;");
  if (ir.requiredFeatures.includes("subgroups")) lines.push("enable subgroups;");
  if (lines.length > 0) lines.push("");
  lines.push(`// BrowserGrad CUDA-lite kernel: ${ir.name}`);

  for (const param of ir.params.filter((param) => param.pointer)) {
    const access = param.constant ? "read" : "read_write";
    const element = storageElementType(param, ir);
    lines.push(
      `@group(0) @binding(${context.bindingFor(param.name)}) var<storage, ${access}> ${param.name}: array<${element}>;`,
    );
  }

  for (const constant of ir.constants.filter((constant) => constant.dimensions.length > 0)) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(constant.name)}) var<storage, read> ${constant.name}: ${emitConstantArrayType(constant)};`,
    );
  }

  for (const texture of ir.textures) {
    lines.push(`@group(0) @binding(${context.bindingFor(texture.name)}) var ${texture.name}: texture_2d<f32>;`);
  }

  const uniformScalars = context.uniformScalars;
  if (uniformScalars.length > 0) {
    lines.push("struct Params {");
    for (const scalar of uniformScalars) {
      const align = scalar.valueType === "half" ? "@align(4) " : "";
      lines.push(`  ${align}${scalar.name}: ${wgslScalar(scalar.valueType)},`);
    }
    lines.push("};");
    lines.push(`@group(0) @binding(${context.paramsBinding}) var<uniform> params: Params;`);
  }

  for (const shared of ir.sharedDeclarations) {
    lines.push(`var<workgroup> ${shared.name}: ${emitSharedType(shared)};`);
  }

  if (ir.textures.length > 0) {
    lines.push("");
    for (const texture of ir.textures) lines.push(...emitTextureHelper(texture.name));
  }
  if (usesFloatAtomicAdd(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicAddHelper());
  }

  for (const fn of ir.functions) {
    lines.push("");
    lines.push(...emitDeviceFunction(fn, context));
  }

  lines.push("");
  lines.push(`@compute @workgroup_size(${ir.workgroupSize.join(", ")})`);
  lines.push("fn main(");
  lines.push("  @builtin(global_invocation_id) global_id: vec3<u32>,");
  lines.push("  @builtin(local_invocation_id) local_id: vec3<u32>,");
  lines.push("  @builtin(workgroup_id) workgroup_id: vec3<u32>,");
  lines.push("  @builtin(num_workgroups) num_workgroups: vec3<u32>");
  lines.push(") {");
  lines.push(...ir.body.flatMap((statement) => emitStatement(statement, context, 1)));
  lines.push("}");

  return {
    wgsl: lines.join("\n"),
    program: defineWgslKernelProgram({
      name: ir.name,
      wgsl: lines.join("\n"),
      bindings: context.bindings,
      workgroupSize: ir.workgroupSize,
    }),
  };
}

interface EmitContext {
  readonly ir: KernelIrModule;
  readonly bindings: readonly WgslKernelBindingInput[];
  readonly paramsBinding?: number;
  readonly uniformScalars: readonly { readonly name: string; readonly valueType: CudaLiteScalarType }[];
  readonly deviceFunctionNames: ReadonlySet<string>;
  bindingFor(name: string): number;
  paramFor(name: string): CudaLiteParam | undefined;
  isUniformScalar(name: string): boolean;
  pointerAliasFor(name: string): PointerAlias | undefined;
}

interface PointerAlias {
  readonly rootName: string;
  readonly baseIndex: CudaLiteExpression;
}

type EmitMode = "value" | "lvalue";

function createEmitContext(ir: KernelIrModule): EmitContext {
  const bindings: WgslKernelBindingInput[] = [];
  const bindingByName = new Map<string, number>();
  for (const param of ir.params.filter((param) => param.pointer)) {
    const binding = bindings.length;
    bindingByName.set(param.name, binding);
    bindings.push({
      kind: "storage",
      name: param.name,
      valueType: wgslBindingType(param.valueType),
      access: param.constant ? "read" : "read_write",
      binding,
    });
  }
  for (const constant of ir.constants.filter((constant) => constant.dimensions.length > 0)) {
    const binding = bindings.length;
    bindingByName.set(constant.name, binding);
    bindings.push({
      kind: "storage",
      name: constant.name,
      valueType: wgslBindingType(constant.valueType),
      access: "read",
      binding,
    });
  }
  for (const texture of ir.textures) {
    const binding = bindings.length;
    bindingByName.set(texture.name, binding);
    bindings.push({
      kind: "texture2d",
      name: texture.name,
      valueType: "f32",
      binding,
    });
  }
  const uniformScalars = [
    ...ir.params.filter((param) => !param.pointer).map((param) => ({ name: param.name, valueType: param.valueType })),
    ...ir.constants.filter((constant) => constant.dimensions.length === 0).map((constant) => ({ name: constant.name, valueType: constant.valueType })),
  ];
  const uniformScalarNames = new Set(uniformScalars.map((scalar) => scalar.name));
  const pointerAliases = collectPointerAliases(ir.body);
  const paramsBinding = uniformScalars.length > 0 ? bindings.length : undefined;
  if (paramsBinding !== undefined) {
    bindings.push({
      kind: "uniform",
      name: "params",
      byteLength: Math.max(16, uniformScalars.length * 4),
      binding: paramsBinding,
    });
  }
  return {
    ir,
    bindings,
    ...(paramsBinding === undefined ? {} : { paramsBinding }),
    uniformScalars,
    deviceFunctionNames: new Set(ir.functions.map((fn) => fn.name)),
    bindingFor(name) {
      const binding = bindingByName.get(name);
      if (binding === undefined) throw featureError("missing-wgsl-binding", `missing WGSL binding for '${name}'`);
      return binding;
    },
    paramFor(name) {
      return ir.params.find((param) => param.name === name);
    },
    isUniformScalar(name) {
      return uniformScalarNames.has(name);
    },
    pointerAliasFor(name) {
      return pointerAliases.get(name);
    },
  };
}

function emitStatement(
  statement: CudaLiteStatement,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  switch (statement.kind) {
    case "var":
      if (statement.storage === "shared") return [];
      if (statement.pointer) return [];
      return [`${prefix}var ${statement.name}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitExpression(statement.init, context)}` : ""};`];
    case "expr":
      if (isNoopCall(statement.expression)) return [`${prefix}// printf omitted: WebGPU has no device stdout`];
      if (isBarrierCall(statement.expression)) return [`${prefix}workgroupBarrier();`];
      return [`${prefix}${emitExpression(statement.expression, context)};`];
    case "if": {
      const lines = [`${prefix}if (${emitExpression(statement.condition, context)}) {`];
      lines.push(...statement.consequent.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      if (statement.alternate) {
        lines.push(`${prefix}} else {`);
        lines.push(...statement.alternate.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      }
      lines.push(`${prefix}}`);
      return lines;
    }
    case "for": {
      const init = statement.init?.kind === "var"
        ? emitForVar(statement.init, context)
        : statement.init
          ? emitExpression(statement.init, context)
          : "";
      const condition = statement.condition ? emitExpression(statement.condition, context) : "true";
      const update = statement.update ? emitExpression(statement.update, context) : "";
      const lines = [`${prefix}for (${init}; ${condition}; ${update}) {`];
      lines.push(...statement.body.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "return":
      return [`${prefix}${statement.value ? `return ${emitExpression(statement.value, context)};` : "return;"}`];
    case "continue":
      return [`${prefix}continue;`];
  }
}

function emitForVar(statement: CudaLiteVarDecl, context: EmitContext): string {
  return `var ${statement.name}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitExpression(statement.init, context)}` : ""}`;
}

function emitDeviceFunction(fn: CudaLiteDeviceFunction, context: EmitContext): string[] {
  const params = [
    ...fn.params.map((param) => `${param.name}_arg: ${wgslScalar(param.valueType)}`),
    "local_id: vec3<u32>",
    "workgroup_id: vec3<u32>",
    "num_workgroups: vec3<u32>",
  ];
  const returnType = fn.returnType === "void" ? "" : ` -> ${wgslScalar(fn.returnType)}`;
  const lines = [`fn ${fn.name}(${params.join(", ")})${returnType} {`];
  for (const param of fn.params) {
    lines.push(`  var ${param.name}: ${wgslScalar(param.valueType)} = ${param.name}_arg;`);
  }
  lines.push(...fn.body.flatMap((statement) => emitStatement(statement, context, 1)));
  if (fn.returnType !== "void" && !functionBodyHasReturn(fn.body)) {
    lines.push(`  return ${zeroValue(fn.returnType)};`);
  }
  lines.push("}");
  return lines;
}

function emitExpression(expression: CudaLiteExpression, context: EmitContext, mode: EmitMode = "value"): string {
  switch (expression.kind) {
    case "number":
      return expression.raw.includes(".") ? expression.raw : `${expression.raw}`;
    case "string":
      return expression.raw;
    case "identifier":
      return emitIdentifier(expression.name, context);
    case "cast":
      return `${wgslScalar(expression.valueType)}(${emitExpression(expression.expression, context)})`;
    case "member":
      return emitMember(expression, context);
    case "index": {
      if (expression.target.kind === "identifier") {
        const alias = context.pointerAliasFor(expression.target.name);
        if (alias) {
          return `${alias.rootName}[(${emitExpression(alias.baseIndex, context)}) + (${emitExpression(expression.index, context)})]`;
        }
      }
      const access = `${emitExpression(expression.target, context)}[${emitExpression(expression.index, context)}]`;
      const root = rootIdentifier(expression);
      const param = root ? context.paramFor(root) : undefined;
      if (mode === "value" && param && context.ir.atomicParams.includes(param.name)) {
        const loaded = `atomicLoad(&${access})`;
        return param.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
      }
      return access;
    }
    case "call":
      return emitCall(expression, context);
    case "unary":
      if (expression.operator === "&") return `&${emitExpression(expression.argument, context, "lvalue")}`;
      return `(${expression.operator}${emitExpression(expression.argument, context)})`;
    case "binary":
      return `(${emitExpression(expression.left, context)} ${expression.operator} ${emitExpression(expression.right, context)})`;
    case "conditional":
      return `select(${emitExpression(expression.alternate, context)}, ${emitExpression(expression.consequent, context)}, ${emitExpression(expression.condition, context)})`;
    case "assignment":
      return emitAssignment(expression, context);
    case "update":
      return expression.prefix
        ? `${expression.operator}${emitExpression(expression.argument, context, "lvalue")}`
        : `${emitExpression(expression.argument, context, "lvalue")}${expression.operator}`;
  }
}

function collectPointerAliases(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, PointerAlias> {
  const aliases = new Map<string, PointerAlias>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer) {
        const alias = pointerAliasForVar(item);
        if (alias) aliases.set(item.name, alias);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") walk(item.body);
    }
  };
  walk(statements);
  return aliases;
}

function pointerAliasForVar(statement: CudaLiteVarDecl): PointerAlias | undefined {
  const init = statement.init;
  if (init?.kind !== "unary" || init.operator !== "&") return undefined;
  const target = init.argument;
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  return {
    rootName: target.target.name,
    baseIndex: target.index,
  };
}

function emitIdentifier(name: string, context: EmitContext): string {
  if (name === "threadIdx" || name === "blockIdx" || name === "blockDim" || name === "gridDim") return name;
  const param = context.paramFor(name);
  if ((param && !param.pointer) || context.isUniformScalar(name)) return `params.${name}`;
  return name;
}

function emitMember(expression: Extract<CudaLiteExpression, { kind: "member" }>, context: EmitContext): string {
  const objectName = expressionName(expression.object);
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  switch (objectName) {
    case "threadIdx":
      return `i32(local_id.${expression.property})`;
    case "blockIdx":
      return `i32(workgroup_id.${expression.property})`;
    case "blockDim":
      return String(context.ir.workgroupSize[axisIndex]);
    case "gridDim":
      return `i32(num_workgroups.${expression.property})`;
    default:
      return `${emitExpression(expression.object, context)}.${expression.property}`;
  }
}

function emitCall(expression: CudaLiteCallExpression, context: EmitContext): string {
  const name = expressionName(expression.callee);
  const args = expression.args.map((arg) => emitExpression(arg, context));
  if (name && context.deviceFunctionNames.has(name)) {
    return `${name}(${[...args, "local_id", "workgroup_id", "num_workgroups"].join(", ")})`;
  }
  switch (name) {
    case "__syncthreads":
      return "workgroupBarrier()";
    case "sqrtf":
      return `sqrt(${args.join(", ")})`;
    case "expf":
      return `exp(${args.join(", ")})`;
    case "logf":
      return `log(${args.join(", ")})`;
    case "__half2float":
      return `f32(${args.join(", ")})`;
    case "__float2half":
      return `f16(${args.join(", ")})`;
    case "min":
    case "max":
      return `${name}(${args.join(", ")})`;
    case "bg_subgroup_add":
      return `subgroupAdd(${args.join(", ")})`;
    case "__shfl_down_sync":
      return `subgroupShuffleDown(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "__shfl_up_sync":
      return `subgroupShuffleUp(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "__shfl_xor_sync":
      return `subgroupShuffleXor(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "tex2D":
      if (expression.args.length === 3 && expression.args[0]?.kind === "identifier") {
        return `bg_tex2d_${expression.args[0].name}(${emitExpression(expression.args[1]!, context)}, ${emitExpression(expression.args[2]!, context)})`;
      }
      return `tex2D(${args.join(", ")})`;
    case "atomicAdd":
      return emitAtomicCall("atomicAdd", expression, context, args);
    case "atomicSub":
      return emitAtomicCall("atomicSub", expression, context, args);
    case "atomicMin":
      return emitAtomicCall("atomicMin", expression, context, args);
    case "atomicMax":
      return emitAtomicCall("atomicMax", expression, context, args);
    case "atomicExch":
      return emitAtomicCall("atomicExchange", expression, context, args);
    case "atomicCAS": {
      const target = expression.args[0];
      const compare = expression.args[1];
      const value = expression.args[2];
      if (target?.kind === "unary" && target.operator === "&" && compare && value) {
        return `atomicCompareExchangeWeak(&${emitExpression(target.argument, context, "lvalue")}, ${emitExpression(compare, context)}, ${emitExpression(value, context)}).old_value`;
      }
      if (target?.kind === "identifier" && compare && value) {
        return `atomicCompareExchangeWeak(&${target.name}[0], ${emitExpression(compare, context)}, ${emitExpression(value, context)}).old_value`;
      }
      return `atomicCompareExchangeWeak(${args.join(", ")}).old_value`;
    }
    default:
      return `${emitExpression(expression.callee, context)}(${args.join(", ")})`;
  }
}

function emitAtomicCall(
  wgslName: string,
  expression: CudaLiteCallExpression,
  context: EmitContext,
  args: readonly string[],
): string {
  const target = expression.args[0];
  const value = expression.args[1];
  if (target?.kind === "unary" && target.operator === "&" && value) {
    const targetParam = atomicTargetParam(target.argument, context);
    const targetExpression = emitExpression(target.argument, context, "lvalue");
    const valueExpression = emitExpression(value, context);
    if (wgslName === "atomicAdd" && targetParam?.valueType === "float") {
      return `bg_atomicAdd_f32(&${targetExpression}, ${valueExpression})`;
    }
    if (wgslName === "atomicExchange" && targetParam?.valueType === "float") {
      return `bitcast<f32>(atomicExchange(&${targetExpression}, bitcast<u32>(${valueExpression})))`;
    }
    return `${wgslName}(&${targetExpression}, ${valueExpression})`;
  }
  if (target?.kind === "identifier" && value) {
    const targetParam = context.paramFor(target.name);
    const valueExpression = emitExpression(value, context);
    if (wgslName === "atomicAdd" && targetParam?.valueType === "float") {
      return `bg_atomicAdd_f32(&${target.name}[0], ${valueExpression})`;
    }
    if (wgslName === "atomicExchange" && targetParam?.valueType === "float") {
      return `bitcast<f32>(atomicExchange(&${target.name}[0], bitcast<u32>(${valueExpression})))`;
    }
    return `${wgslName}(&${target.name}[0], ${valueExpression})`;
  }
  return `${wgslName}(${args.join(", ")})`;
}

function emitAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string {
  const root = rootIdentifier(expression.left);
  const param = root ? context.paramFor(root) : undefined;
  if (param && context.ir.atomicParams.includes(param.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (param.valueType === "float") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
    }
  }
  const left = emitExpression(expression.left, context, "lvalue");
  const right = emitExpression(expression.right, context);
  if (expression.operator === "<<=") return `${left} = (${left} << ${right})`;
  if (expression.operator === ">>=") return `${left} = (${left} >> ${right})`;
  return `${left} ${expression.operator} ${right}`;
}

function isBarrierCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" && expressionName(expression.callee) === "__syncthreads";
}

function isNoopCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" && expressionName(expression.callee) === "printf";
}

function emitSharedType(statement: CudaLiteVarDecl): string {
  let type = wgslScalar(statement.valueType);
  for (let i = statement.dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${statement.dimensions[i]!}>`;
  }
  return type;
}

function emitConstantArrayType(constant: CudaLiteGlobalConstant): string {
  let type = wgslScalar(constant.valueType);
  for (let i = constant.dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${constant.dimensions[i]!}>`;
  }
  return type;
}

function emitTextureHelper(name: string): string[] {
  return [
    `fn bg_tex2d_${name}(x: f32, y: f32) -> f32 {`,
    `  let dims = textureDimensions(${name});`,
    "  let max_coord = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);",
    "  let coord = clamp(vec2<i32>(i32(floor(x)), i32(floor(y))), vec2<i32>(0, 0), max_coord);",
    `  return textureLoad(${name}, coord, 0).r;`,
    "}",
  ];
}

function emitFloatAtomicAddHelper(): string[] {
  return [
    "fn bg_atomicAdd_f32(ptr_value: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32 {",
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

function storageElementType(param: CudaLiteParam, ir: KernelIrModule): string {
  if (!ir.atomicParams.includes(param.name)) return wgslScalar(param.valueType);
  if (param.valueType === "float") return "atomic<u32>";
  return `atomic<${wgslScalar(param.valueType)}>`;
}

function usesFloatAtomicAdd(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && param.valueType === "float" && ir.atomicParams.includes(param.name));
}

function atomicTargetParam(expression: CudaLiteExpression, context: EmitContext): CudaLiteParam | undefined {
  const root = rootIdentifier(expression);
  return root ? context.paramFor(root) : undefined;
}

function functionBodyHasReturn(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "return") return true;
    if (statement.kind === "if" && (functionBodyHasReturn(statement.consequent) || (statement.alternate && functionBodyHasReturn(statement.alternate)))) {
      return true;
    }
  }
  return false;
}

function zeroValue(type: CudaLiteScalarType): string {
  if (type === "float") return "0.0";
  if (type === "half") return "f16(0.0)";
  if (type === "uint") return "0u";
  return "0";
}

function wgslScalar(type: CudaLiteScalarType): string {
  switch (type) {
    case "float":
      return "f32";
    case "int":
      return "i32";
    case "uint":
      return "u32";
    case "half":
      return "f16";
    case "void":
      return "void";
  }
}

function wgslBindingType(type: CudaLiteScalarType): "f16" | "f32" | "i32" | "u32" {
  if (type === "half") return "f16";
  if (type === "int") return "i32";
  if (type === "uint") return "u32";
  return "f32";
}

function indent(level: number): string {
  return "  ".repeat(level);
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}
