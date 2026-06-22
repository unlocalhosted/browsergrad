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
  type CudaLiteExpression,
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
    const atomic = ir.atomicParams.includes(param.name);
    const element = atomic ? `atomic<${wgslScalar(param.valueType)}>` : wgslScalar(param.valueType);
    lines.push(
      `@group(0) @binding(${context.bindingFor(param.name)}) var<storage, ${access}> ${param.name}: array<${element}>;`,
    );
  }

  const scalarParams = ir.params.filter((param) => !param.pointer);
  if (scalarParams.length > 0) {
    lines.push("struct Params {");
    for (const param of scalarParams) {
      lines.push(`  ${param.name}: ${wgslScalar(param.valueType)},`);
    }
    lines.push("};");
    lines.push(`@group(0) @binding(${context.paramsBinding}) var<uniform> params: Params;`);
  }

  for (const shared of ir.sharedDeclarations) {
    lines.push(`var<workgroup> ${shared.name}: ${emitSharedType(shared)};`);
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
  bindingFor(name: string): number;
  paramFor(name: string): CudaLiteParam | undefined;
}

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
  const scalarParams = ir.params.filter((param) => !param.pointer);
  const paramsBinding = scalarParams.length > 0 ? bindings.length : undefined;
  if (paramsBinding !== undefined) {
    bindings.push({
      kind: "uniform",
      name: "params",
      byteLength: Math.max(16, scalarParams.length * 4),
      binding: paramsBinding,
    });
  }
  return {
    ir,
    bindings,
    ...(paramsBinding === undefined ? {} : { paramsBinding }),
    bindingFor(name) {
      const binding = bindingByName.get(name);
      if (binding === undefined) throw new Error(`missing binding for ${name}`);
      return binding;
    },
    paramFor(name) {
      return ir.params.find((param) => param.name === name);
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
      return [`${prefix}var ${statement.name}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitExpression(statement.init, context)}` : ""};`];
    case "expr":
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
  }
}

function emitForVar(statement: CudaLiteVarDecl, context: EmitContext): string {
  return `var ${statement.name}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitExpression(statement.init, context)}` : ""}`;
}

function emitExpression(expression: CudaLiteExpression, context: EmitContext): string {
  switch (expression.kind) {
    case "number":
      return expression.raw.includes(".") ? expression.raw : `${expression.raw}`;
    case "identifier":
      return emitIdentifier(expression.name, context);
    case "member":
      return emitMember(expression, context);
    case "index":
      return `${emitExpression(expression.target, context)}[${emitExpression(expression.index, context)}]`;
    case "call":
      return emitCall(expression, context);
    case "unary":
      if (expression.operator === "&") return `&${emitExpression(expression.argument, context)}`;
      return `(${expression.operator}${emitExpression(expression.argument, context)})`;
    case "binary":
      return `(${emitExpression(expression.left, context)} ${expression.operator} ${emitExpression(expression.right, context)})`;
    case "assignment":
      return emitAssignment(expression, context);
    case "update":
      return expression.prefix
        ? `${expression.operator}${emitExpression(expression.argument, context)}`
        : `${emitExpression(expression.argument, context)}${expression.operator}`;
  }
}

function emitIdentifier(name: string, context: EmitContext): string {
  if (name === "threadIdx" || name === "blockIdx" || name === "blockDim" || name === "gridDim") return name;
  const param = context.paramFor(name);
  if (param && !param.pointer) return `params.${name}`;
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
  switch (name) {
    case "__syncthreads":
      return "workgroupBarrier()";
    case "sqrtf":
      return `sqrt(${args.join(", ")})`;
    case "expf":
      return `exp(${args.join(", ")})`;
    case "logf":
      return `log(${args.join(", ")})`;
    case "min":
    case "max":
      return `${name}(${args.join(", ")})`;
    case "bg_subgroup_add":
      return `subgroupAdd(${args.join(", ")})`;
    case "atomicAdd":
      return `atomicAdd(${args.join(", ")})`;
    default:
      return `${emitExpression(expression.callee, context)}(${args.join(", ")})`;
  }
}

function emitAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string {
  const root = rootIdentifier(expression.left);
  const param = root ? context.paramFor(root) : undefined;
  if (param && context.ir.atomicParams.includes(param.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context);
    if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
    if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
  }
  return `${emitExpression(expression.left, context)} ${expression.operator} ${emitExpression(expression.right, context)}`;
}

function isBarrierCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" && expressionName(expression.callee) === "__syncthreads";
}

function emitSharedType(statement: CudaLiteVarDecl): string {
  let type = wgslScalar(statement.valueType);
  for (let i = statement.dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${statement.dimensions[i]!}>`;
  }
  return type;
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

function wgslBindingType(type: CudaLiteScalarType): "f32" | "i32" | "u32" {
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
