import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslKernelProgram,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames, collectKernelLaunchCallees } from "./ast_queries.js";
import { expressionName, rootIdentifier } from "./analyzer.js";
import { CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import {
  cudaVectorConstructorType,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteCallExpression,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteDeviceFunction,
  type CudaLiteExpression,
  type CudaLiteFeatureOptions,
  type CudaLiteGlobalConstant,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface EmitKernelIrWgslOptions {
  readonly features?: CudaLiteFeatureOptions;
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
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

  const context = createEmitContext(ir, options);
  const textures = textureBindings(ir);
  const lines: string[] = [];
  if (ir.requiredFeatures.includes("shader-f16")) lines.push("enable f16;");
  if (ir.requiredFeatures.includes("subgroups")) lines.push("enable subgroups;");
  if (lines.length > 0) lines.push("");
  lines.push(`// BrowserGrad CUDA-lite kernel: ${ir.name}`);

  for (const param of ir.params.filter((param) => param.pointer && !isDevicePoolParam(param))) {
    const access = param.constant ? "read" : "read_write";
    const element = storageElementType(param, ir);
    lines.push(
      `@group(0) @binding(${context.bindingFor(param.name)}) var<storage, ${access}> ${param.name}: array<${element}>;`,
    );
  }
  for (const surface of ir.params.filter(isSurfaceParam)) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(surface.name)}) var<storage, read_write> ${surface.name}: array<f32>;`,
    );
  }
  for (const pool of ir.params.filter(isDevicePoolParam)) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(poolDataName(pool.name))}) var<storage, read_write> ${poolDataName(pool.name)}: array<u32>;`,
    );
    lines.push(
      `@group(0) @binding(${context.bindingFor(poolOffsetName(pool.name))}) var<storage, read_write> ${poolOffsetName(pool.name)}: atomic<u32>;`,
    );
  }
  for (const poolName of context.externalPoolNames) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(poolDataName(poolName))}) var<storage, read_write> ${poolDataName(poolName)}: array<u32>;`,
    );
    lines.push(
      `@group(0) @binding(${context.bindingFor(poolOffsetName(poolName))}) var<storage, read_write> ${poolOffsetName(poolName)}: atomic<u32>;`,
    );
  }

  for (const constant of ir.constants.filter((constant) => constant.dimensions.length > 0)) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(constant.name)}) var<storage, read> ${constant.name}: ${emitConstantArrayType(constant)};`,
    );
  }

  for (const texture of textures) {
    lines.push(`@group(0) @binding(${context.bindingFor(texture.name)}) var ${texture.name}: texture_2d<f32>;`);
  }

  const uniformScalars = context.uniformScalars;
  if (uniformScalars.length > 0) {
    lines.push("struct Params {");
    for (const scalar of uniformScalars) {
      const align = scalar.valueType === "half" ? "@align(4) " : "";
      lines.push(`  ${align}${scalar.name}: ${wgslUniformScalar(scalar.valueType)},`);
    }
    lines.push("};");
    lines.push(`@group(0) @binding(${context.paramsBinding}) var<uniform> params: Params;`);
  }

  for (const shared of ir.sharedDeclarations) {
    lines.push(`var<workgroup> ${shared.name}: ${emitSharedType(shared, ir)};`);
  }

  if (textures.length > 0) {
    lines.push("");
    for (const texture of textures) lines.push(...emitTextureHelper(texture.name));
  }
  for (const surface of ir.params.filter(isSurfaceParam)) {
    lines.push("");
    lines.push(...emitSurfaceHelper(surface.name));
  }
  for (const pool of ir.params.filter(isDevicePoolParam)) {
    lines.push("");
    lines.push(...emitPoolHelper(pool.name));
  }
  for (const poolName of context.externalPoolNames) {
    lines.push("");
    lines.push(...emitPoolHelper(poolName));
  }
  for (const allocator of context.rawPoolAllocators) {
    lines.push("");
    lines.push(...emitRawPoolHelper(allocator));
  }
  if (usesFloatAtomicAdd(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicAddHelper());
  }
  if (usesFloatAtomicSub(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicSubHelper());
  }
  if (usesFloatAtomicMin(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicMinHelper());
  }
  if (usesFloatAtomicMax(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicMaxHelper());
  }
  if (usesCurand(ir)) {
    lines.push("");
    lines.push(...emitCurandHelpers());
  }
  if (usesSpecialFloatNamedConstants(ir)) {
    lines.push("");
    lines.push(...emitSpecialFloatConstantHelpers());
  }
  if (usesDevicePointerParams(ir)) {
    lines.push("");
    lines.push(...emitDevicePointerHelpers(ir, context));
  }

  const emittedFunctions = functionsToEmit(ir);
  for (const fn of emittedFunctions) {
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
  deviceFunctionFor(name: string): CudaLiteDeviceFunction | undefined;
  devicePointerParamFor(name: string): CudaLiteParam | undefined;
  storagePointerIdFor(name: string): number | undefined;
  sharedPointerIdFor(name: string): number | undefined;
  bindingFor(name: string): number;
  paramFor(name: string): CudaLiteParam | undefined;
  isUniformScalar(name: string): boolean;
  uniformScalarTypeFor(name: string): CudaLiteScalarType | undefined;
  isAtomicShared(name: string): boolean;
  pointerAliasFor(name: string): PointerAlias | undefined;
  poolPointerFor(name: string): PoolPointerAlias | undefined;
  pointerBaseOffsetFieldFor(name: string): string | undefined;
  readonly rawPoolAllocators: readonly RawPoolAllocator[];
  readonly externalPoolNames: readonly string[];
  isLocalName(name: string): boolean;
  cooperativeGroupFor(name: string): CudaLiteCooperativeGroupDecl | undefined;
  surfaceWidthField(name: string): string;
}

interface PointerAlias {
  readonly rootName: string;
  readonly baseIndex: CudaLiteExpression;
  readonly valueType?: CudaLiteScalarType;
}

interface PoolPointerAlias {
  readonly poolName: string;
  readonly offsetName?: string;
  readonly rawBuffer?: boolean;
}

interface RawPoolAllocator {
  readonly baseName: string;
  readonly offsetName: string;
}

type EmitMode = "value" | "lvalue";

function createEmitContext(ir: KernelIrModule, options: EmitKernelIrWgslOptions = {}): EmitContext {
  const bindings: WgslKernelBindingInput[] = [];
  const bindingByName = new Map<string, number>();
  const storagePointerParams = ir.params.filter((param) => param.pointer && !isDevicePoolParam(param));
  const textures = textureBindings(ir);
  const storagePointerIds = new Map(storagePointerParams.map((param, index) => [param.name, index] as const));
  const sharedPointerIds = new Map(ir.sharedDeclarations
    .filter((shared) => shared.dimensions.length === 1)
    .map((shared, index) => [shared.name, storagePointerParams.length + index] as const));
  for (const param of storagePointerParams) {
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
  for (const param of ir.params.filter(isSurfaceParam)) {
    const binding = bindings.length;
    bindingByName.set(param.name, binding);
    bindings.push({
      kind: "storage",
      name: param.name,
      valueType: "f32",
      access: "read_write",
      binding,
    });
  }
  for (const param of ir.params.filter(isDevicePoolParam)) {
    const dataBinding = bindings.length;
    bindingByName.set(poolDataName(param.name), dataBinding);
    bindings.push({
      kind: "storage",
      name: poolDataName(param.name),
      valueType: "u32",
      access: "read_write",
      binding: dataBinding,
    });
    const offsetBinding = bindings.length;
    bindingByName.set(poolOffsetName(param.name), offsetBinding);
    bindings.push({
      kind: "storage",
      name: poolOffsetName(param.name),
      valueType: "u32",
      access: "read_write",
      binding: offsetBinding,
    });
  }
  const externalPoolNames = collectExternalDevicePoolNames(
    ir.body,
    new Set(ir.params.filter(isDevicePoolParam).map((param) => param.name)),
  );
  for (const poolName of externalPoolNames) {
    const dataBinding = bindings.length;
    bindingByName.set(poolDataName(poolName), dataBinding);
    bindings.push({
      kind: "storage",
      name: poolDataName(poolName),
      valueType: "u32",
      access: "read_write",
      binding: dataBinding,
    });
    const offsetBinding = bindings.length;
    bindingByName.set(poolOffsetName(poolName), offsetBinding);
    bindings.push({
      kind: "storage",
      name: poolOffsetName(poolName),
      valueType: "u32",
      access: "read_write",
      binding: offsetBinding,
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
  for (const texture of textures) {
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
    ...ir.params.filter((param) => !param.pointer && !isSurfaceParam(param) && !isTextureParam(param)).map((param) => ({ name: param.name, valueType: param.valueType })),
    ...ir.constants.filter((constant) => constant.dimensions.length === 0).map((constant) => ({ name: constant.name, valueType: constant.valueType })),
    ...ir.params.filter(isSurfaceParam).flatMap((param) => [
      { name: surfaceWidthField(param.name), valueType: "uint" as const },
      { name: surfaceHeightField(param.name), valueType: "uint" as const },
    ]),
    ...ir.params
      .filter((param) => param.pointer && options.pointerBaseOffsets?.[param.name] !== undefined)
      .map((param) => ({ name: pointerBaseOffsetUniformName(param.name), valueType: "uint" as const })),
  ];
  const uniformScalarNames = new Set(uniformScalars.map((scalar) => scalar.name));
  const uniformScalarTypes = new Map(uniformScalars.map((scalar) => [scalar.name, scalar.valueType] as const));
  const pointerAliases = collectPointerAliases(ir.body);
  const poolPointers = collectPoolPointers(ir.body);
  const rawPoolAllocators = collectRawPoolAllocators(ir.body);
  const cooperativeGroups = collectCooperativeGroups(ir.body);
  const localNames = collectLocalNames(ir.body);
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
    deviceFunctionFor(name) {
      return ir.functions.find((fn) => fn.name === name);
    },
    devicePointerParamFor() {
      return undefined;
    },
    storagePointerIdFor(name) {
      return storagePointerIds.get(name);
    },
    sharedPointerIdFor(name) {
      return sharedPointerIds.get(name);
    },
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
    uniformScalarTypeFor(name) {
      return uniformScalarTypes.get(name);
    },
    isAtomicShared(name) {
      return ir.atomicShared.includes(name);
    },
    pointerAliasFor(name) {
      return pointerAliases.get(name);
    },
    poolPointerFor(name) {
      return poolPointers.get(name);
    },
    pointerBaseOffsetFieldFor(name) {
      return options.pointerBaseOffsets?.[name] === undefined ? undefined : pointerBaseOffsetUniformName(name);
    },
    rawPoolAllocators,
    externalPoolNames,
    isLocalName(name) {
      return localNames.has(name);
    },
    cooperativeGroupFor(name) {
      return cooperativeGroups.get(name);
    },
    surfaceWidthField(name) {
      return surfaceWidthField(name);
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
      if (statement.pointer) {
        if (!isEmittedPointerVar(statement, context)) return [];
        return [`${prefix}var ${statement.name}: u32${statement.init ? ` = ${emitExpression(statement.init, context)}` : " = 0u"};`];
      }
      if (statement.dimensions.length > 0) return [`${prefix}var ${statement.name}: ${emitLocalArrayType(statement)};`];
      return [`${prefix}var ${statement.name}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitLocalInit(statement, context)}` : ""};`];
    case "dim3":
      return [];
    case "cooperative-group":
      return [];
    case "kernel-launch":
      return [`${prefix}// device-side launch omitted: ${statement.callee}<<<...>>>`];
    case "asm":
      return [`${prefix}${emitInlineAsmStatement(statement, context)};`];
    case "expr":
      {
        const noopComment = noopCallComment(statement.expression);
        if (noopComment) return [`${prefix}// ${noopComment}`];
      }
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

function functionsToEmit(ir: KernelIrModule): readonly CudaLiteDeviceFunction[] {
  const launchCallees = new Set(collectKernelLaunchCallees(ir.body));
  return ir.functions.filter((fn) => fn.name !== ir.name && !launchCallees.has(fn.name));
}

function emitInlineAsmStatement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  context: EmitContext,
): string {
  if (!/\bfma\.rn\.f32\b/u.test(statement.template) || statement.inputs.length !== 2) {
    throw featureError("unsupported-inline-asm", "only fma.rn.f32 inline PTX is supported in WGSL output");
  }
  const target = emitExpression(statement.output, context);
  return `${target} = fma(${emitExpression(statement.inputs[0]!, context)}, ${emitExpression(statement.inputs[1]!, context)}, ${target})`;
}

function emitForVar(statement: CudaLiteVarDecl, context: EmitContext): string {
  if (statement.pointer) return `var ${statement.name}: u32${statement.init ? ` = ${emitExpression(statement.init, context)}` : " = 0u"}`;
  if (statement.dimensions.length > 0) return `var ${statement.name}: ${emitLocalArrayType(statement)}`;
  return `var ${statement.name}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitLocalInit(statement, context)}` : ""}`;
}

function emitLocalInit(statement: CudaLiteVarDecl, context: EmitContext): string {
  const value = statement.init ? emitExpression(statement.init, context) : zeroValue(statement.valueType);
  if (statement.valueType === "uint") return `u32(${value})`;
  return value;
}

function emitDeviceFunction(fn: CudaLiteDeviceFunction, context: EmitContext): string[] {
  const functionContext = withDevicePointerParams(
    context,
    fn.params.filter((param) => param.pointer),
    new Set([...fn.params.map((param) => param.name), ...collectLocalNames(fn.body)]),
  );
  const params = [
    ...fn.params.flatMap((param) => param.pointer
      ? [`${param.name}_buffer_arg: u32`, `${param.name}_base_arg: u32`]
      : [`${param.name}_arg: ${wgslScalar(param.valueType)}`]),
    "local_id: vec3<u32>",
    "workgroup_id: vec3<u32>",
    "num_workgroups: vec3<u32>",
  ];
  const returnType = fn.returnType === "void" ? "" : ` -> ${wgslScalar(fn.returnType)}`;
  const lines = [`fn ${fn.name}(${params.join(", ")})${returnType} {`];
  for (const param of fn.params) {
    if (param.pointer) {
      lines.push(`  var ${param.name}_buffer: u32 = ${param.name}_buffer_arg;`);
      lines.push(`  var ${param.name}_base: u32 = ${param.name}_base_arg;`);
    } else {
      lines.push(`  var ${param.name}: ${wgslScalar(param.valueType)} = ${param.name}_arg;`);
    }
  }
  lines.push(...fn.body.flatMap((statement) => emitStatement(statement, functionContext, 1)));
  if (fn.returnType !== "void" && !functionBodyHasReturn(fn.body)) {
    lines.push(`  return ${zeroValue(fn.returnType)};`);
  }
  lines.push("}");
  return lines;
}

function withDevicePointerParams(
  context: EmitContext,
  params: readonly CudaLiteParam[],
  localNames: ReadonlySet<string> = new Set(),
): EmitContext {
  const pointerParams = new Map(params.map((param) => [param.name, param] as const));
  return {
    ...context,
    isLocalName(name) {
      return localNames.has(name) || context.isLocalName(name);
    },
    devicePointerParamFor(name) {
      return pointerParams.get(name) ?? context.devicePointerParamFor(name);
    },
  };
}

function usesDevicePointerParams(ir: KernelIrModule): boolean {
  return ir.functions.some((fn) => fn.params.some((param) => param.pointer)) ||
    statementsUseCall(ir.body, new Set(["__ldcs", "__stcs"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["__ldcs", "__stcs"])));
}

function emitDevicePointerHelpers(ir: KernelIrModule, context: EmitContext): string[] {
  const types = [...new Set([
    ...ir.params.filter((param) => param.pointer && !isDevicePoolParam(param)).map((param) => param.valueType),
    ...ir.functions.flatMap((fn) => fn.params.filter((param) => param.pointer).map((param) => param.valueType)),
  ])];
  return types.flatMap((type) => emitDevicePointerHelper(type, ir, context));
}

function emitDevicePointerHelper(type: CudaLiteScalarType, ir: KernelIrModule, context: EmitContext): string[] {
  if (!isDevicePointerHelperType(type)) {
    throw featureError("unsupported-device-pointer-param", `device pointer helpers do not support ${type} pointers yet`);
  }
  const storageParams = ir.params.filter((param) =>
    param.pointer &&
    !isDevicePoolParam(param) &&
    param.valueType === type
  );
  const sharedDeclarations = ir.sharedDeclarations.filter((shared) =>
    shared.valueType === type &&
    shared.dimensions.length === 1
  );
  const scalar = wgslScalar(type);
  const lines = [
    `fn ${pointerReadHelperName(type)}(buffer: u32, index: u32) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of storageParams) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { return ${emitPointerStorageRead(param, "index", ir)}; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { return ${emitSharedPointerRead(shared, "index", ir)}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  lines.push("");
  lines.push(`fn ${pointerWriteHelperName(type)}(buffer: u32, index: u32, value: ${scalar}) {`);
  lines.push("  switch buffer {");
  for (const param of storageParams.filter((param) => !param.constant)) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { ${emitPointerStorageWrite(param, "index", "value", ir)}; return; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { ${emitSharedPointerWrite(shared, "index", "value", ir)}; return; }`);
  }
  lines.push("    default: { return; }");
  lines.push("  }");
  lines.push("}");
  return lines;
}

function isDevicePointerHelperType(type: CudaLiteScalarType): boolean {
  return type === "float" || type === "int" || type === "uint" || type === "half" || isCudaVectorType(type);
}

function emitPointerStorageRead(param: CudaLiteParam, index: string, ir: KernelIrModule): string {
  if (isCudaVectorType(param.valueType)) return emitVectorStorageRead(param.name, param.valueType, index);
  const access = `${param.name}[${index}]`;
  if (!ir.atomicParams.includes(param.name)) return access;
  const loaded = `atomicLoad(&${access})`;
  return param.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitPointerStorageWrite(param: CudaLiteParam, index: string, value: string, ir: KernelIrModule): string {
  if (isCudaVectorType(param.valueType)) return emitVectorStorageWrite(param.name, param.valueType, index, value);
  const access = `${param.name}[${index}]`;
  if (!ir.atomicParams.includes(param.name)) return `${access} = ${value}`;
  if (param.valueType === "float") return `atomicStore(&${access}, bitcast<u32>(${value}))`;
  return `atomicStore(&${access}, ${value})`;
}

function emitVectorStorageRead(name: string, type: CudaLiteScalarType, index: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  const values = Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u]`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

function emitVectorStorageWrite(name: string, type: CudaLiteScalarType, index: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

function emitVectorStorageFieldWrite(name: string, type: CudaLiteScalarType, index: string, field: string, value: string): string | undefined {
  const fieldIndex = cudaVectorFieldIndex(type, field);
  if (fieldIndex === undefined) return undefined;
  const base = vectorStorageBase(index, cudaVectorLaneCount(type));
  return `${name}[${base} + ${fieldIndex}u] = ${value}`;
}

function vectorStorageBase(index: string, lanes: number): string {
  return `(u32(${index}) * ${lanes}u)`;
}

function vectorFieldName(index: number): string {
  return index === 0 ? "x" : index === 1 ? "y" : index === 2 ? "z" : "w";
}

function emitSharedPointerRead(shared: CudaLiteVarDecl, index: string, ir: KernelIrModule): string {
  const access = `${shared.name}[${index}]`;
  return ir.atomicShared.includes(shared.name) ? `atomicLoad(&${access})` : access;
}

function emitSharedPointerWrite(shared: CudaLiteVarDecl, index: string, value: string, ir: KernelIrModule): string {
  const access = `${shared.name}[${index}]`;
  return ir.atomicShared.includes(shared.name) ? `atomicStore(&${access}, ${value})` : `${access} = ${value}`;
}

interface DevicePointerParts {
  readonly buffer: string;
  readonly base: string;
}

interface DevicePointerLValue {
  readonly buffer: string;
  readonly index: string;
  readonly valueType: CudaLiteScalarType;
}

function emitDevicePointerArgument(expression: CudaLiteExpression, context: EmitContext): readonly [string, string] {
  const parts = devicePointerArgumentParts(expression, context);
  if (!parts) {
    throw featureError("unsupported-device-pointer-param", "device pointer argument must be a storage pointer or derived storage address");
  }
  return [parts.buffer, parts.base];
}

function devicePointerArgumentParts(expression: CudaLiteExpression, context: EmitContext): DevicePointerParts | undefined {
  if (expression.kind === "identifier") {
    const pointerParam = context.devicePointerParamFor(expression.name);
    if (pointerParam) return { buffer: `${expression.name}_buffer`, base: `${expression.name}_base` };
    const storageId = context.storagePointerIdFor(expression.name);
    const storageParam = context.paramFor(expression.name);
    if (storageId !== undefined && storageParam?.pointer) {
      const baseField = context.pointerBaseOffsetFieldFor(expression.name);
      return { buffer: `${storageId}u`, base: baseField ? `params.${baseField}` : "0u" };
    }
    const sharedId = context.sharedPointerIdFor(expression.name);
    if (sharedId !== undefined) return { buffer: `${sharedId}u`, base: "0u" };
  }
  if (expression.kind === "cast" && expression.pointer) {
    return devicePointerArgumentParts(expression.expression, context);
  }
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "index") {
    const target = devicePointerArgumentParts(expression.argument.target, context);
    if (!target) return undefined;
    return {
      buffer: target.buffer,
      base: `(${target.base} + u32(${emitExpression(expression.argument.index, context)}))`,
    };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const target = devicePointerArgumentParts(expression.left, context);
    if (!target) return undefined;
    const delta = `u32(${emitExpression(expression.right, context)})`;
    return {
      buffer: target.buffer,
      base: expression.operator === "+"
        ? `(${target.base} + ${delta})`
        : `(${target.base} - ${delta})`,
    };
  }
  return undefined;
}

function devicePointerValueTypeForExpression(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType {
  const root = rootIdentifier(expression);
  if (root) {
    const param = context.devicePointerParamFor(root) ?? context.paramFor(root);
    if (param?.pointer) return param.valueType;
  }
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  return "float";
}

function devicePointerParamForIndex(
  expression: Extract<CudaLiteExpression, { kind: "index" }>,
  context: EmitContext,
): CudaLiteParam | undefined {
  return expression.target.kind === "identifier"
    ? context.devicePointerParamFor(expression.target.name)
    : undefined;
}

function devicePointerIndexExpression(
  name: string,
  index: CudaLiteExpression,
  context: EmitContext,
): string {
  return `(${name}_base + u32(${emitExpression(index, context)}))`;
}

function devicePointerLValue(expression: CudaLiteExpression, context: EmitContext): DevicePointerLValue | undefined {
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    const param = context.devicePointerParamFor(expression.target.name);
    if (!param) return undefined;
    return {
      buffer: `${expression.target.name}_buffer`,
      index: devicePointerIndexExpression(expression.target.name, expression.index, context),
      valueType: param.valueType,
    };
  }
  if (expression.kind === "unary" && expression.operator === "*" && expression.argument.kind === "identifier") {
    const param = context.devicePointerParamFor(expression.argument.name);
    if (!param) return undefined;
    return {
      buffer: `${expression.argument.name}_buffer`,
      index: `${expression.argument.name}_base`,
      valueType: param.valueType,
    };
  }
  return undefined;
}

function pointerReadHelperName(type: CudaLiteScalarType): string {
  return `bg_ptr_read_${pointerHelperTypeName(type)}`;
}

function pointerWriteHelperName(type: CudaLiteScalarType): string {
  return `bg_ptr_write_${pointerHelperTypeName(type)}`;
}

function pointerHelperTypeName(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type) ?? "float";
    return `${scalar === "float" ? "f32" : scalar === "int" ? "i32" : "u32"}x${cudaVectorLaneCount(type)}`;
  }
  if (type === "float") return "f32";
  if (type === "half") return "f16";
  if (type === "int") return "i32";
  if (type === "uint") return "u32";
  return type;
}

function emitExpression(expression: CudaLiteExpression, context: EmitContext, mode: EmitMode = "value"): string {
  switch (expression.kind) {
    case "number":
      return expression.raw.includes(".") ? expression.raw : `${expression.raw}`;
    case "string":
      return expression.raw;
    case "identifier":
      return emitIdentifier(expression.name, context, mode);
    case "cast":
      if (expression.pointer) return emitExpression(expression.expression, context);
      return `${wgslScalar(expression.valueType)}(${emitExpression(expression.expression, context)})`;
    case "member":
      return emitMember(expression, context);
    case "index": {
      const storageView = storageViewLValue(expression, context);
      if (storageView && isCudaVectorType(storageView.valueType)) {
        return emitVectorStorageReadAt(storageView.name, storageView.valueType, storageView.index);
      }
      const poolAccess = poolAccessForIndex(expression, context);
      if (poolAccess) return emitPoolRead(poolAccess, context);
      const pointerParam = devicePointerParamForIndex(expression, context);
      if (pointerParam) {
        return `${pointerReadHelperName(pointerParam.valueType)}(${pointerParam.name}_buffer, ${devicePointerIndexExpression(pointerParam.name, expression.index, context)})`;
      }
      if (expression.target.kind === "identifier") {
        const alias = context.pointerAliasFor(expression.target.name);
        if (alias) {
          if (alias.valueType && isCudaVectorType(alias.valueType)) {
            const index = emitStorageViewIndex(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
            return emitVectorStorageReadAt(alias.rootName, alias.valueType, index);
          }
          return `${alias.rootName}[${emitPointerAliasIndex(alias, expression.index, context)}]`;
        }
      }
      const root = rootIdentifier(expression);
      const index = root && expression.target.kind === "identifier"
        ? emitPointerIndex(root, expression.index, context)
        : emitExpression(expression.index, context);
      const param = root ? context.paramFor(root) : undefined;
      if (mode === "value" && param && isCudaVectorType(param.valueType) && expression.target.kind === "identifier") {
        return emitVectorStorageRead(root!, param.valueType, index);
      }
      const access = `${emitExpression(expression.target, context, "lvalue")}[${index}]`;
      if (mode === "value" && param && context.ir.atomicParams.includes(param.name)) {
        const loaded = `atomicLoad(&${access})`;
        return param.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
      }
      if (mode === "value" && root && context.isAtomicShared(root)) return `atomicLoad(&${access})`;
      return access;
    }
    case "call":
      return emitCall(expression, context);
    case "unary":
      if (expression.operator === "&") return `&${emitExpression(expression.argument, context, "lvalue")}`;
      if (expression.operator === "*") return emitDeref(expression.argument, context);
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

function collectLocalNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" || item.kind === "dim3" || item.kind === "cooperative-group") names.add(item.name);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var") names.add(item.init.name);
        walk(item.body);
      }
    }
  };
  walk(statements);
  return names;
}

function collectPointerAliases(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, PointerAlias> {
  const aliases = new Map<string, PointerAlias>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer) {
        const alias = pointerAliasForVar(item);
        if (alias) aliases.set(item.name, alias);
      }
      if (item.kind === "dim3" || item.kind === "cooperative-group" || item.kind === "kernel-launch") continue;
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

function collectPoolPointers(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, PoolPointerAlias> {
  const aliases = new Map<string, PoolPointerAlias>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer) {
        const alias = poolPointerForInitializer(item.init, aliases);
        if (alias) aliases.set(item.name, alias);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.pointer) {
          const alias = poolPointerForInitializer(item.init.init, aliases);
          if (alias) aliases.set(item.init.name, alias);
        }
        walk(item.body);
      }
    }
  };
  walk(statements);
  return aliases;
}

function poolPointerForInitializer(
  init: CudaLiteExpression | undefined,
  aliases: ReadonlyMap<string, PoolPointerAlias>,
): PoolPointerAlias | undefined {
  if (!init) return undefined;
  if (init.kind === "call") return poolPointerForAllocationCall(init);
  if (init.kind === "cast" && init.pointer) return poolPointerForInitializer(init.expression, aliases);
  if (init.kind === "identifier") return aliases.get(init.name);
  return undefined;
}

function poolPointerForAllocationCall(call: CudaLiteCallExpression): PoolPointerAlias | undefined {
  const name = expressionName(call.callee);
  if (name !== "deviceAllocate" && name !== "streamOrderedAllocate") return undefined;
  if (call.args.length === 4) {
    const base = call.args[0];
    const offset = call.args[1];
    return base?.kind === "identifier" && offset?.kind === "identifier"
      ? { poolName: base.name, offsetName: offset.name, rawBuffer: true }
      : undefined;
  }
  const pool = call.args[0];
  if (pool?.kind === "unary" && pool.operator === "&" && pool.argument.kind === "identifier") {
    return { poolName: pool.argument.name };
  }
  return pool?.kind === "identifier" ? { poolName: pool.name } : undefined;
}

function collectRawPoolAllocators(statements: readonly CudaLiteStatement[]): readonly RawPoolAllocator[] {
  const allocators = new Map<string, RawPoolAllocator>();
  const visitExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "call") {
      const alias = poolPointerForAllocationCall(expression);
      if (alias?.rawBuffer && alias.offsetName) {
        const key = `${alias.poolName}\0${alias.offsetName}`;
        allocators.set(key, { baseName: alias.poolName, offsetName: alias.offsetName });
      }
      visitExpression(expression.callee);
      for (const arg of expression.args) visitExpression(arg);
      return;
    }
    if (expression.kind === "cast") visitExpression(expression.expression);
    else if (expression.kind === "member") visitExpression(expression.object);
    else if (expression.kind === "index") {
      visitExpression(expression.target);
      visitExpression(expression.index);
    } else if (expression.kind === "unary" || expression.kind === "update") visitExpression(expression.argument);
    else if (expression.kind === "binary") {
      visitExpression(expression.left);
      visitExpression(expression.right);
    } else if (expression.kind === "conditional") {
      visitExpression(expression.condition);
      visitExpression(expression.consequent);
      visitExpression(expression.alternate);
    } else if (expression.kind === "assignment") {
      visitExpression(expression.left);
      visitExpression(expression.right);
    }
  };
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.init) visitExpression(item.init);
      if (item.kind === "expr") visitExpression(item.expression);
      if (item.kind === "if") {
        visitExpression(item.condition);
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.init) visitExpression(item.init.init);
        else if (item.init && item.init.kind !== "var") visitExpression(item.init);
        if (item.condition) visitExpression(item.condition);
        if (item.update) visitExpression(item.update);
        walk(item.body);
      }
      if (item.kind === "return" && item.value) visitExpression(item.value);
    }
  };
  walk(statements);
  return [...allocators.values()];
}

function collectCooperativeGroups(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteCooperativeGroupDecl> {
  const groups = new Map<string, CudaLiteCooperativeGroupDecl>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "cooperative-group") groups.set(item.name, item);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") walk(item.body);
    }
  };
  walk(statements);
  return groups;
}

function pointerAliasForVar(statement: CudaLiteVarDecl): PointerAlias | undefined {
  const init = statement.init;
  const view = pointerAliasForPointerExpression(init, statement.valueType);
  if (view) return view;
  if (init?.kind !== "unary" || init.operator !== "&") return undefined;
  const target = init.argument;
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  return {
    rootName: target.target.name,
    baseIndex: target.index,
  };
}

function pointerAliasForPointerExpression(
  expression: CudaLiteExpression | undefined,
  valueType: CudaLiteScalarType,
): PointerAlias | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return pointerAliasForPointerExpression(expression.expression, valueType);
  if (expression.kind === "unary" && expression.operator === "&") {
    const target = expression.argument;
    if (target.kind === "index" && target.target.kind === "identifier") {
      return { rootName: target.target.name, baseIndex: target.index, valueType };
    }
    if (target.kind === "identifier") {
      return { rootName: target.name, baseIndex: zeroExpression(target.span), valueType };
    }
    return undefined;
  }
  if (expression.kind === "identifier") {
    return { rootName: expression.name, baseIndex: zeroExpression(expression.span), valueType };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const base = pointerAliasForPointerExpression(expression.left, valueType);
    if (!base) return undefined;
    return {
      ...base,
      baseIndex: expression.operator === "+"
        ? { kind: "binary", operator: "+", left: base.baseIndex, right: expression.right, span: expression.span }
        : { kind: "binary", operator: "-", left: base.baseIndex, right: expression.right, span: expression.span },
    };
  }
  return undefined;
}

function emitPointerAliasIndex(alias: PointerAlias, index: CudaLiteExpression, context: EmitContext): string {
  const base = context.pointerBaseOffsetFieldFor(alias.rootName);
  if (!base) return `(${emitExpression(alias.baseIndex, context)}) + (${emitExpression(index, context)})`;
  return `(params.${base} + u32(${emitExpression(alias.baseIndex, context)}) + u32(${emitExpression(index, context)}))`;
}

function emitStorageViewIndex(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  const base = context.pointerBaseOffsetFieldFor(rootName);
  const prefix = base
    ? `params.${base} + u32(${emitExpression(baseIndex, context)})`
    : `u32(${emitExpression(baseIndex, context)})`;
  const lanes = cudaVectorLaneCount(valueType);
  return `(${prefix} + (u32(${emitExpression(index, context)}) * ${lanes}u))`;
}

function emitVectorStorageReadAt(name: string, type: CudaLiteScalarType, storageIndex: string): string {
  const lanes = cudaVectorLaneCount(type);
  const values = Array.from({ length: lanes }, (_, lane) => `${name}[${storageIndex} + ${lane}u]`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

function emitVectorStorageWriteAt(name: string, type: CudaLiteScalarType, storageIndex: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${storageIndex} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

function emitVectorStorageFieldWriteAt(name: string, storageIndex: string, fieldIndex: number, value: string): string {
  return `${name}[${storageIndex} + ${fieldIndex}u] = ${value}`;
}

function zeroExpression(span: CudaLiteExpression["span"]): CudaLiteExpression {
  return { kind: "number", value: 0, raw: "0", span };
}

function emitPointerIndex(rootName: string, index: CudaLiteExpression, context: EmitContext): string {
  const base = context.pointerBaseOffsetFieldFor(rootName);
  if (!base) return emitExpression(index, context);
  return `(params.${base} + u32(${emitExpression(index, context)}))`;
}

function emitIdentifier(name: string, context: EmitContext, mode: EmitMode = "value"): string {
  if (name === "nullptr") return "0u";
  const namedConstant = CUDA_NAMED_CONSTANTS.get(name);
  if (namedConstant) return namedConstant.wgsl;
  if (name === "threadIdx" || name === "blockIdx" || name === "blockDim" || name === "gridDim") return name;
  if (context.isAtomicShared(name) && mode === "value") return `atomicLoad(&${name})`;
  if (context.isLocalName(name)) return name;
  const param = context.paramFor(name);
  const uniformType = context.uniformScalarTypeFor(name);
  if ((param && !param.pointer && !isSurfaceParam(param)) || context.isUniformScalar(name)) {
    return uniformType === "bool" ? `(params.${name} != 0u)` : `params.${name}`;
  }
  return name;
}

function emitDeref(expression: CudaLiteExpression, context: EmitContext): string {
  if (expression.kind === "identifier") {
    const pointerParam = context.devicePointerParamFor(expression.name);
    if (pointerParam) {
      return `${pointerReadHelperName(pointerParam.valueType)}(${expression.name}_buffer, ${expression.name}_base)`;
    }
    const param = context.paramFor(expression.name);
    const access = `${expression.name}[0]`;
    if (param && context.ir.atomicParams.includes(param.name)) {
      const loaded = `atomicLoad(&${access})`;
      return param.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
    }
    return access;
  }
  if (expression.kind === "cast" && expression.pointer) {
    return emitPoolRead({
      poolName: poolPointerExpressionInfo(expression.expression, context)?.poolName ?? "",
      pointerExpression: expression.expression,
      indexExpression: { kind: "number", value: 0, raw: "0", span: expression.span },
      valueType: expression.valueType,
      ...(poolPointerExpressionInfo(expression.expression, context)?.rawBuffer ? { rawBuffer: true } : {}),
    }, context);
  }
  return `*${emitExpression(expression, context)}`;
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
  const cooperativeGroupCall = emitCooperativeGroupCall(expression, context);
  if (cooperativeGroupCall !== undefined) return cooperativeGroupCall;
  const deviceFunction = name ? context.deviceFunctionFor(name) : undefined;
  if (deviceFunction) {
    const args = deviceFunction.params.flatMap((param, index) => {
      const arg = expression.args[index];
      if (!arg) return param.pointer ? ["0u", "0u"] : [zeroValue(param.valueType)];
      return param.pointer
        ? emitDevicePointerArgument(arg, context)
        : [emitExpression(arg, context)];
    });
    return `${name}(${[...args, "local_id", "workgroup_id", "num_workgroups"].join(", ")})`;
  }
  const args = expression.args.map((arg) => emitExpression(arg, context));
  const intrinsic = name ? CUDA_INTRINSICS_BY_NAME.get(name) : undefined;
  if (intrinsic?.emitWgsl) return intrinsic.emitWgsl(args);
  const vectorConstructor = name ? cudaVectorConstructorType(name) : undefined;
  if (vectorConstructor) return `${wgslScalar(vectorConstructor)}(${args.join(", ")})`;
  if (name === "__ldcs") {
    const target = expression.args[0];
    if (!target) return "0";
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", "__ldcs expects a storage pointer or derived storage address");
    const valueType = devicePointerValueTypeForExpression(target, context);
    return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  if (name === "__stcs") {
    const target = expression.args[0];
    const value = expression.args[1];
    if (!target || !value) return "0";
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", "__stcs expects a storage pointer or derived storage address");
    const valueType = devicePointerValueTypeForExpression(target, context);
    return `${pointerWriteHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${emitExpression(value, context)})`;
  }
  switch (name) {
    case "__syncthreads":
      return "workgroupBarrier()";
    case "cudaDeviceSynchronize":
    case "cudaStreamCreate":
    case "cudaStreamCreateWithFlags":
    case "cudaStreamDestroy":
    case "cudaStreamSynchronize":
    case "cudaEventCreate":
    case "cudaEventCreateWithFlags":
    case "cudaEventDestroy":
    case "cudaEventRecord":
    case "cudaEventSynchronize":
      return "0";
    case "cudaMemcpy":
      return "0";
    case "cudaMemcpyAsync":
      return "0";
    case "cudaMemcpyPeerAsync":
      return "0";
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
    case "surf2Dwrite":
      if (expression.args.length >= 4 && expression.args[1]?.kind === "identifier") {
        return `bg_surf2dwrite_${expression.args[1].name}(${emitExpression(expression.args[0]!, context)}, ${emitExpression(expression.args[2]!, context)}, ${emitExpression(expression.args[3]!, context)})`;
      }
      return `surf2Dwrite(${args.join(", ")})`;
    case "deviceAllocate":
    case "streamOrderedAllocate":
      if (expression.args.length === 4 && expression.args[0]?.kind === "identifier" && expression.args[1]?.kind === "identifier") {
        return `${rawPoolHelperName(expression.args[0].name, expression.args[1].name)}(u32(${emitExpression(expression.args[2]!, context)}), u32(${emitExpression(expression.args[3]!, context)}))`;
      }
      if (expression.args.length >= 2 && expression.args[0]?.kind === "identifier") {
        return `bg_pool_alloc_${expression.args[0].name}(u32(${emitExpression(expression.args[1]!, context)}))`;
      }
      if (expression.args.length >= 2 && expression.args[0]?.kind === "unary" && expression.args[0].operator === "&" && expression.args[0].argument.kind === "identifier") {
        return `bg_pool_alloc_${expression.args[0].argument.name}(u32(${emitExpression(expression.args[1]!, context)}))`;
      }
      return "0u";
    case "sizeof":
      if (expression.args[0]?.kind === "identifier") return String(sizeofType(expression.args[0].name));
      return "4";
    case "vec_at":
      return `(${args[0] ?? "vec4<f32>()"}[u32(${args[1] ?? "0"})])`;
    case "curand_init":
      return `bg_curand_init(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}), ${args[3] ?? "&state"})`;
    case "curand_uniform":
      return `bg_curand_uniform(${args[0] ?? "&state"})`;
    case "atomicAdd":
      return emitAtomicCall("atomicAdd", expression, context, args);
    case "atomicSub":
      return emitAtomicCall("atomicSub", expression, context, args);
    case "atomicMin":
      return emitAtomicCall("atomicMin", expression, context, args);
    case "atomicMax":
      return emitAtomicCall("atomicMax", expression, context, args);
    case "atomicMaxFloat":
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

function emitCooperativeGroupCall(expression: CudaLiteCallExpression, context: EmitContext): string | undefined {
  const namespaceCall = emitCooperativeNamespaceCall(expression, context);
  if (namespaceCall !== undefined) return namespaceCall;
  const callee = expression.callee;
  if (callee.kind !== "member" || callee.object.kind !== "identifier") return undefined;
  const group = context.cooperativeGroupFor(callee.object.name);
  if (!group) return undefined;
  if (callee.property === "sync") {
    return group.groupKind === "grid" ? "0" : "workgroupBarrier()";
  }
  if (callee.property === "size") {
    if (group.groupKind === "tile") return String(group.tileSize ?? 32);
    return String(context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2]);
  }
  if (callee.property === "thread_rank") {
    const localRank = emitLocalLinearRank(context);
    if (group.groupKind === "tile") return `(${localRank} % ${group.tileSize ?? 32})`;
    return localRank;
  }
  if (callee.property === "meta_group_size") {
    if (group.groupKind !== "tile") return "1";
    const blockSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
    return String(Math.ceil(blockSize / (group.tileSize ?? 32)));
  }
  if (callee.property === "meta_group_rank") {
    if (group.groupKind !== "tile") return "0";
    return `(${emitLocalLinearRank(context)} / ${group.tileSize ?? 32})`;
  }
  if (callee.property === "shfl_down" || callee.property === "shfl_up" || callee.property === "shfl_xor") {
    const value = expression.args[0] ? emitExpression(expression.args[0], context) : "0";
    const offset = expression.args[1] ? emitExpression(expression.args[1], context) : "0";
    const intrinsic = callee.property === "shfl_up"
      ? "subgroupShuffleUp"
      : callee.property === "shfl_xor"
        ? "subgroupShuffleXor"
        : "subgroupShuffleDown";
    return `${intrinsic}(${value}, u32(${offset}))`;
  }
  return undefined;
}

function emitCooperativeNamespaceCall(expression: CudaLiteCallExpression, context: EmitContext): string | undefined {
  const name = expressionName(expression.callee);
  if (!name?.endsWith("::sync") && !name?.endsWith("::reduce")) return undefined;
  const groupArg = expression.args[0];
  if (groupArg?.kind !== "identifier") return undefined;
  const group = context.cooperativeGroupFor(groupArg.name);
  if (!group) return undefined;
  if (name.endsWith("::sync")) return group.groupKind === "grid" ? "0" : "workgroupBarrier()";
  const value = expression.args[1] ? emitExpression(expression.args[1], context) : "0";
  const op = expression.args[2] ? expressionName(expression.args[2]) : undefined;
  if (op?.endsWith("::greater")) return `subgroupMax(${value})`;
  return `subgroupAdd(${value})`;
}

function emitLocalLinearRank(context: EmitContext): string {
  const [, y, z] = context.ir.workgroupSize;
  if (y === 1 && z === 1) return "i32(local_id.x)";
  if (z === 1) return `i32(local_id.x + local_id.y * ${context.ir.workgroupSize[0]}u)`;
  return `i32(local_id.x + local_id.y * ${context.ir.workgroupSize[0]}u + local_id.z * ${context.ir.workgroupSize[0] * context.ir.workgroupSize[1]}u)`;
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
    const sharedRoot = rootIdentifier(target.argument);
    if (sharedRoot && context.isAtomicShared(sharedRoot)) {
      return `${wgslName}(&${targetExpression}, ${valueExpression})`;
    }
    if (wgslName === "atomicAdd" && targetParam?.valueType === "float") {
      return `bg_atomicAdd_f32(&${targetExpression}, ${valueExpression})`;
    }
    if (wgslName === "atomicSub" && targetParam?.valueType === "float") {
      return `bg_atomicSub_f32(&${targetExpression}, ${valueExpression})`;
    }
    if (wgslName === "atomicMin" && targetParam?.valueType === "float") {
      return `bg_atomicMin_f32(&${targetExpression}, ${valueExpression})`;
    }
    if (wgslName === "atomicMax" && targetParam?.valueType === "float") {
      return `bg_atomicMax_f32(&${targetExpression}, ${valueExpression})`;
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
    if (wgslName === "atomicSub" && targetParam?.valueType === "float") {
      return `bg_atomicSub_f32(&${target.name}[0], ${valueExpression})`;
    }
    if (wgslName === "atomicMin" && targetParam?.valueType === "float") {
      return `bg_atomicMin_f32(&${target.name}[0], ${valueExpression})`;
    }
    if (wgslName === "atomicMax" && targetParam?.valueType === "float") {
      return `bg_atomicMax_f32(&${target.name}[0], ${valueExpression})`;
    }
    if (wgslName === "atomicExchange" && targetParam?.valueType === "float") {
      return `bitcast<f32>(atomicExchange(&${target.name}[0], bitcast<u32>(${valueExpression})))`;
    }
    return `${wgslName}(&${target.name}[0], ${valueExpression})`;
  }
  return `${wgslName}(${args.join(", ")})`;
}

function emitAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string {
  const storageView = storageViewLValue(expression.left, context);
  if (storageView && isCudaVectorType(storageView.valueType)) {
    const right = emitExpression(expression.right, context);
    if (storageView.field) {
      const current = `${storageView.name}[${storageView.index} + ${storageView.fieldIndex ?? 0}u]`;
      if (expression.operator !== "=") {
        const op = expression.operator.slice(0, -1);
        return `${current} = (${current} ${op} ${right})`;
      }
      return emitVectorStorageFieldWriteAt(storageView.name, storageView.index, storageView.fieldIndex ?? 0, right);
    }
    if (expression.operator !== "=") {
      const current = emitVectorStorageReadAt(storageView.name, storageView.valueType, storageView.index);
      const op = expression.operator.slice(0, -1);
      return emitVectorStorageWriteAt(storageView.name, storageView.valueType, storageView.index, `(${current} ${op} ${right})`);
    }
    return emitVectorStorageWriteAt(storageView.name, storageView.valueType, storageView.index, right);
  }
  const poolAccess = poolAccessForIndex(expression.left, context);
  if (poolAccess) return emitPoolAssignment(poolAccess, expression.operator, expression.right, context);
  const vectorAssignment = emitVectorAssignment(expression, context);
  if (vectorAssignment) return vectorAssignment;
  const pointerLvalue = devicePointerLValue(expression.left, context);
  if (pointerLvalue) {
    const right = emitExpression(expression.right, context);
    const read = `${pointerReadHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index})`;
    const value = expression.operator === "="
      ? right
      : expression.operator === "+="
        ? `(${read} + ${right})`
        : expression.operator === "-="
          ? `(${read} - ${right})`
          : expression.operator === "*="
            ? `(${read} * ${right})`
            : expression.operator === "/="
              ? `(${read} / ${right})`
              : expression.operator === "<<="
                ? `(${read} << ${right})`
                : `(${read} >> ${right})`;
    return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${value})`;
  }
  const root = rootIdentifier(expression.left);
  const param = root ? context.paramFor(root) : undefined;
  if (root && context.isAtomicShared(root)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
    if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
    if (expression.operator === "-=") return `atomicSub(&${target}, ${value})`;
  }
  if (param && context.ir.atomicParams.includes(param.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (param.valueType === "float") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32(&${target}, ${value})`;
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

function emitVectorAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  const direct = vectorStorageLValue(expression.left, context);
  if (!direct) return undefined;
  const right = emitExpression(expression.right, context);
  if (direct.field) {
    if (expression.operator !== "=") {
      const current = `${direct.name}[${vectorStorageBase(direct.index, direct.lanes)} + ${direct.fieldIndex ?? 0}u]`;
      const op = expression.operator.slice(0, -1);
      return `${current} = (${current} ${op} ${right})`;
    }
    return emitVectorStorageFieldWrite(direct.name, direct.valueType, direct.index, direct.field, right);
  }
  if (expression.operator !== "=") {
    const current = emitVectorStorageRead(direct.name, direct.valueType, direct.index);
    const op = expression.operator.slice(0, -1);
    return emitVectorStorageWrite(direct.name, direct.valueType, direct.index, `(${current} ${op} ${right})`);
  }
  return emitVectorStorageWrite(direct.name, direct.valueType, direct.index, right);
}

interface VectorStorageLValue {
  readonly name: string;
  readonly valueType: CudaLiteScalarType;
  readonly index: string;
  readonly lanes: number;
  readonly field?: string;
  readonly fieldIndex?: number;
}

function vectorStorageLValue(expression: CudaLiteExpression, context: EmitContext): VectorStorageLValue | undefined {
  let target = expression;
  let field: string | undefined;
  if (expression.kind === "member") {
    field = expression.property;
    target = expression.object;
  }
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  const param = context.paramFor(target.target.name);
  if (!param || !isCudaVectorType(param.valueType)) return undefined;
  const index = emitPointerIndex(param.name, target.index, context);
  const base = {
    name: param.name,
    valueType: param.valueType,
    index,
    lanes: cudaVectorLaneCount(param.valueType),
  };
  if (!field) return base;
  const fieldIndex = cudaVectorFieldIndex(param.valueType, field);
  return fieldIndex === undefined ? undefined : { ...base, field, fieldIndex };
}

function storageViewLValue(expression: CudaLiteExpression, context: EmitContext): VectorStorageLValue | undefined {
  let target = expression;
  let field: string | undefined;
  if (expression.kind === "member") {
    field = expression.property;
    target = expression.object;
  }
  if (target.kind !== "index") return undefined;
  const view = storageViewForPointerExpression(target.target, target.index, context);
  if (!view || !isCudaVectorType(view.valueType)) return undefined;
  const base = {
    name: view.rootName,
    valueType: view.valueType,
    index: view.index,
    lanes: cudaVectorLaneCount(view.valueType),
  };
  if (!field) return base;
  const fieldIndex = cudaVectorFieldIndex(view.valueType, field);
  return fieldIndex === undefined ? undefined : { ...base, field, fieldIndex };
}

function storageViewForPointerExpression(
  pointer: CudaLiteExpression,
  index: CudaLiteExpression,
  context: EmitContext,
): { readonly rootName: string; readonly valueType: CudaLiteScalarType; readonly index: string } | undefined {
  if (pointer.kind === "cast" && pointer.pointer) {
    const alias = pointerAliasForPointerExpression(pointer.expression, pointer.valueType);
    if (!alias) return undefined;
    return {
      rootName: alias.rootName,
      valueType: pointer.valueType,
      index: emitStorageViewIndex(alias.rootName, alias.baseIndex, index, pointer.valueType, context),
    };
  }
  if (pointer.kind === "identifier") {
    const alias = context.pointerAliasFor(pointer.name);
    if (!alias?.valueType) return undefined;
    return {
      rootName: alias.rootName,
      valueType: alias.valueType,
      index: emitStorageViewIndex(alias.rootName, alias.baseIndex, index, alias.valueType, context),
    };
  }
  return undefined;
}

function isBarrierCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" && expressionName(expression.callee) === "__syncthreads";
}

function noopCallComment(expression: CudaLiteExpression): string | undefined {
  if (expression.kind !== "call") return undefined;
  switch (expressionName(expression.callee)) {
    case "printf":
      return "printf omitted: WebGPU has no device stdout";
    case "cudaDeviceSynchronize":
      return "cudaDeviceSynchronize omitted: WebGPU dispatch completion is host-managed";
    case "cudaStreamCreate":
    case "cudaStreamCreateWithFlags":
    case "cudaStreamDestroy":
    case "cudaStreamSynchronize":
    case "cudaEventCreate":
    case "cudaEventCreateWithFlags":
    case "cudaEventDestroy":
    case "cudaEventRecord":
    case "cudaEventSynchronize":
      return `${expressionName(expression.callee)} omitted: WebGPU stream/event orchestration is host-managed`;
    case "cudaMemcpy":
      return "cudaMemcpy omitted: WebGPU copy orchestration is host-managed";
    case "cudaMemcpyAsync":
      return "cudaMemcpyAsync omitted: WebGPU copy orchestration is host-managed";
    case "cudaMemcpyPeerAsync":
      return "cudaMemcpyPeerAsync omitted: WebGPU copy orchestration is host-managed";
    default:
      return undefined;
  }
}

function emitSharedType(statement: CudaLiteVarDecl, ir: KernelIrModule): string {
  let type = ir.atomicShared.includes(statement.name) ? `atomic<${wgslScalar(statement.valueType)}>` : wgslScalar(statement.valueType);
  for (let i = statement.dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${statement.dimensions[i]!}>`;
  }
  return type;
}

function emitLocalArrayType(statement: CudaLiteVarDecl): string {
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

function emitSurfaceHelper(name: string): string[] {
  return [
    `fn bg_surf2dwrite_${name}(value: f32, x_bytes: i32, y: i32) {`,
    "  let x = x_bytes / 4;",
    `  let index = y * i32(params.${surfaceWidthField(name)}) + x;`,
    `  if (index >= 0 && index < i32(arrayLength(&${name}))) {`,
    `    ${name}[index] = value;`,
    "  }",
    "}",
  ];
}

interface PoolAccess {
  readonly poolName: string;
  readonly pointerExpression: CudaLiteExpression;
  readonly indexExpression: CudaLiteExpression;
  readonly valueType: CudaLiteScalarType;
  readonly rawBuffer?: boolean;
}

function poolAccessForIndex(expression: CudaLiteExpression, context: EmitContext): PoolAccess | undefined {
  if (expression.kind !== "index") return undefined;
  const target = expression.target;
  if (target.kind === "cast" && target.pointer) {
    const pointer = poolPointerExpressionInfo(target.expression, context);
    if (!pointer) return undefined;
    return {
      poolName: pointer.poolName,
      pointerExpression: target.expression,
      indexExpression: expression.index,
      valueType: target.valueType,
      ...(pointer.rawBuffer ? { rawBuffer: true } : {}),
    };
  }
  if (target.kind === "identifier") {
    const pointer = context.poolPointerFor(target.name);
    if (!pointer) return undefined;
    return {
      poolName: pointer.poolName,
      pointerExpression: target,
      indexExpression: expression.index,
      valueType: "float",
      ...(pointer.rawBuffer ? { rawBuffer: true } : {}),
    };
  }
  return undefined;
}

function poolPointerExpressionInfo(
  expression: CudaLiteExpression,
  context: EmitContext,
): PoolPointerAlias | undefined {
  if (expression.kind === "identifier") return context.poolPointerFor(expression.name);
  if (expression.kind === "call") return poolPointerForAllocationCall(expression);
  if (expression.kind === "cast" && expression.pointer) return poolPointerExpressionInfo(expression.expression, context);
  return undefined;
}

function emitPoolRead(access: PoolAccess, context: EmitContext): string {
  const raw = poolRawAccess(access, context);
  if (access.rawBuffer) return raw;
  return decodePoolWord(access.valueType, raw);
}

function emitPoolAssignment(
  access: PoolAccess,
  operator: string,
  right: CudaLiteExpression,
  context: EmitContext,
): string {
  const raw = poolRawAccess(access, context);
  const rightValue = emitExpression(right, context);
  if (access.rawBuffer) {
    if (operator === "=") return `${raw} = ${rightValue}`;
    const op = operator.slice(0, -1);
    return `${raw} = (${raw} ${op} ${rightValue})`;
  }
  if (operator === "=") return `${raw} = ${encodePoolWord(access.valueType, rightValue)}`;
  const current = decodePoolWord(access.valueType, raw);
  const op = operator.slice(0, -1);
  return `${raw} = ${encodePoolWord(access.valueType, `(${current} ${op} ${rightValue})`)}`;
}

function poolRawAccess(access: PoolAccess, context: EmitContext): string {
  const name = access.rawBuffer ? access.poolName : poolDataName(access.poolName);
  return `${name}[${poolWordIndex(access, context)}]`;
}

function poolWordIndex(access: PoolAccess, context: EmitContext): string {
  const pointer = emitExpression(access.pointerExpression, context);
  const index = emitExpression(access.indexExpression, context);
  const stride = access.valueType === "complex64" ? "2u" : "1u";
  return `(((${pointer} - 1u) / 4u) + (u32(${index}) * ${stride}))`;
}

function decodePoolWord(type: CudaLiteScalarType, raw: string): string {
  if (type === "float") return `bitcast<f32>(${raw})`;
  if (type === "int") return `bitcast<i32>(${raw})`;
  if (type === "uint" || type === "voidptr" || type === "devicepool" || type === "surface2d") return raw;
  if (type === "bool") return `(${raw} != 0u)`;
  if (type === "half") return `f16(0.0)`;
  if (type === "complex64") return `vec2<f32>(bitcast<f32>(${raw}), bitcast<f32>(${raw}))`;
  return raw;
}

function encodePoolWord(type: CudaLiteScalarType, value: string): string {
  if (type === "float") return `bitcast<u32>(${value})`;
  if (type === "int") return `bitcast<u32>(${value})`;
  if (type === "uint" || type === "voidptr" || type === "devicepool" || type === "surface2d") return `u32(${value})`;
  if (type === "bool") return `select(0u, 1u, ${value})`;
  if (type === "half") return "0u";
  if (type === "complex64") return `bitcast<u32>(${value}.x)`;
  return `u32(${value})`;
}

function emitPoolHelper(name: string): string[] {
  return [
    `fn bg_pool_alloc_${name}(size_bytes: u32) -> u32 {`,
    `  let old = atomicAdd(&${poolOffsetName(name)}, size_bytes);`,
    `  let capacity = arrayLength(&${poolDataName(name)}) * 4u;`,
    "  if ((old + size_bytes) > capacity) {",
    "    return 0u;",
    "  }",
    "  return old + 1u;",
    "}",
  ];
}

function emitRawPoolHelper(allocator: RawPoolAllocator): string[] {
  return [
    `fn ${rawPoolHelperName(allocator.baseName, allocator.offsetName)}(pool_size_bytes: u32, size_bytes: u32) -> u32 {`,
    `  let old = atomicAdd(&${allocator.offsetName}[0], size_bytes);`,
    "  if ((old + size_bytes) > pool_size_bytes) {",
    "    return 0u;",
    "  }",
    "  return old + 1u;",
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

function emitFloatAtomicSubHelper(): string[] {
  return [
    "fn bg_atomicSub_f32(ptr_value: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32 {",
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

function emitFloatAtomicMinHelper(): string[] {
  return [
    "fn bg_atomicMin_f32(ptr_value: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32 {",
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

function emitFloatAtomicMaxHelper(): string[] {
  return [
    "fn bg_atomicMax_f32(ptr_value: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32 {",
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

function emitCurandHelpers(): string[] {
  return [
    "fn bg_curand_next(state: ptr<function, u32>) -> u32 {",
    "  var x = *state;",
    "  x = (x * 1664525u) + 1013904223u;",
    "  *state = x;",
    "  return x;",
    "}",
    "fn bg_curand_init(seed: u32, sequence: u32, offset: u32, state: ptr<function, u32>) {",
    "  *state = seed ^ (sequence * 747796405u) ^ offset ^ 2891336453u;",
    "  _ = bg_curand_next(state);",
    "}",
    "fn bg_curand_uniform(state: ptr<function, u32>) -> f32 {",
    "  let bits = bg_curand_next(state);",
    "  return (f32(bits) + 1.0) * 2.3283064365386963e-10;",
    "}",
  ];
}

function storageElementType(param: CudaLiteParam, ir: KernelIrModule): string {
  if (isCudaVectorType(param.valueType)) return wgslScalar(cudaVectorScalarType(param.valueType) ?? "float");
  if (!ir.atomicParams.includes(param.name)) return wgslScalar(param.valueType);
  if (param.valueType === "float") return "atomic<u32>";
  return `atomic<${wgslScalar(param.valueType)}>`;
}

function usesFloatAtomicAdd(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && param.valueType === "float" && ir.atomicParams.includes(param.name)) &&
    (statementsUseCall(ir.body, new Set(["atomicAdd"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAdd"]))));
}

function usesFloatAtomicSub(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && param.valueType === "float" && ir.atomicParams.includes(param.name)) &&
    (statementsUseCall(ir.body, new Set(["atomicSub"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicSub"]))));
}

function usesFloatAtomicMin(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && param.valueType === "float" && ir.atomicParams.includes(param.name)) &&
    (statementsUseCall(ir.body, new Set(["atomicMin"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMin"]))));
}

function usesFloatAtomicMax(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && param.valueType === "float" && ir.atomicParams.includes(param.name)) &&
    (statementsUseCall(ir.body, new Set(["atomicMax", "atomicMaxFloat"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMax", "atomicMaxFloat"]))));
}

function usesCurand(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["curand_init", "curand_uniform"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["curand_init", "curand_uniform"])));
}

function usesSpecialFloatNamedConstants(ir: KernelIrModule): boolean {
  const names = new Set(["INFINITY", "NAN"]);
  return statementsUseIdentifier(ir.body, names) ||
    ir.functions.some((fn) => statementsUseIdentifier(fn.body, names));
}

function statementsUseCall(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && expressionUsesCall(statement.expression, names)) return true;
    if (statement.kind === "var" && statement.init && expressionUsesCall(statement.init, names)) return true;
    if (statement.kind === "if" && (
      expressionUsesCall(statement.condition, names) ||
      statementsUseCall(statement.consequent, names) ||
      (statement.alternate ? statementsUseCall(statement.alternate, names) : false)
    )) return true;
    if (statement.kind === "for" && (
      (statement.init?.kind === "var" && statement.init.init ? expressionUsesCall(statement.init.init, names) : false) ||
      (statement.init && statement.init.kind !== "var" ? expressionUsesCall(statement.init, names) : false) ||
      (statement.condition ? expressionUsesCall(statement.condition, names) : false) ||
      (statement.update ? expressionUsesCall(statement.update, names) : false) ||
      statementsUseCall(statement.body, names)
    )) return true;
    if (statement.kind === "return" && statement.value && expressionUsesCall(statement.value, names)) return true;
  }
  return false;
}

function statementsUseIdentifier(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && expressionUsesIdentifier(statement.expression, names)) return true;
    if (statement.kind === "var" && statement.init && expressionUsesIdentifier(statement.init, names)) return true;
    if (statement.kind === "if" && (
      expressionUsesIdentifier(statement.condition, names) ||
      statementsUseIdentifier(statement.consequent, names) ||
      (statement.alternate ? statementsUseIdentifier(statement.alternate, names) : false)
    )) return true;
    if (statement.kind === "for" && (
      (statement.init?.kind === "var" && statement.init.init ? expressionUsesIdentifier(statement.init.init, names) : false) ||
      (statement.init && statement.init.kind !== "var" ? expressionUsesIdentifier(statement.init, names) : false) ||
      (statement.condition ? expressionUsesIdentifier(statement.condition, names) : false) ||
      (statement.update ? expressionUsesIdentifier(statement.update, names) : false) ||
      statementsUseIdentifier(statement.body, names)
    )) return true;
    if (statement.kind === "return" && statement.value && expressionUsesIdentifier(statement.value, names)) return true;
    if (statement.kind === "kernel-launch" && (
      statement.grid.some((expression) => expressionUsesIdentifier(expression, names)) ||
      statement.block.some((expression) => expressionUsesIdentifier(expression, names)) ||
      statement.args.some((expression) => expressionUsesIdentifier(expression, names))
    )) return true;
    if (statement.kind === "dim3" && statement.args.some((expression) => expressionUsesIdentifier(expression, names))) return true;
    if (statement.kind === "asm" && (
      expressionUsesIdentifier(statement.output, names) ||
      statement.inputs.some((expression) => expressionUsesIdentifier(expression, names))
    )) return true;
  }
  return false;
}

function expressionUsesCall(expression: CudaLiteExpression, names: ReadonlySet<string>): boolean {
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    if (name && names.has(name)) return true;
    return expression.args.some((arg) => expressionUsesCall(arg, names)) || expressionUsesCall(expression.callee, names);
  }
  if (expression.kind === "cast") return expressionUsesCall(expression.expression, names);
  if (expression.kind === "member") return expressionUsesCall(expression.object, names);
  if (expression.kind === "index") return expressionUsesCall(expression.target, names) || expressionUsesCall(expression.index, names);
  if (expression.kind === "unary" || expression.kind === "update") return expressionUsesCall(expression.argument, names);
  if (expression.kind === "binary") return expressionUsesCall(expression.left, names) || expressionUsesCall(expression.right, names);
  if (expression.kind === "conditional") {
    return expressionUsesCall(expression.condition, names) ||
      expressionUsesCall(expression.consequent, names) ||
      expressionUsesCall(expression.alternate, names);
  }
  if (expression.kind === "assignment") return expressionUsesCall(expression.left, names) || expressionUsesCall(expression.right, names);
  return false;
}

function expressionUsesIdentifier(expression: CudaLiteExpression, names: ReadonlySet<string>): boolean {
  if (expression.kind === "identifier") return names.has(expression.name);
  if (expression.kind === "cast") return expressionUsesIdentifier(expression.expression, names);
  if (expression.kind === "member") return expressionUsesIdentifier(expression.object, names);
  if (expression.kind === "index") return expressionUsesIdentifier(expression.target, names) || expressionUsesIdentifier(expression.index, names);
  if (expression.kind === "unary" || expression.kind === "update") return expressionUsesIdentifier(expression.argument, names);
  if (expression.kind === "binary") return expressionUsesIdentifier(expression.left, names) || expressionUsesIdentifier(expression.right, names);
  if (expression.kind === "conditional") {
    return expressionUsesIdentifier(expression.condition, names) ||
      expressionUsesIdentifier(expression.consequent, names) ||
      expressionUsesIdentifier(expression.alternate, names);
  }
  if (expression.kind === "assignment") return expressionUsesIdentifier(expression.left, names) || expressionUsesIdentifier(expression.right, names);
  if (expression.kind === "call") {
    return expressionUsesIdentifier(expression.callee, names) ||
      expression.args.some((arg) => expressionUsesIdentifier(arg, names));
  }
  return false;
}

function emitSpecialFloatConstantHelpers(): readonly string[] {
  return [
    "fn bg_f32_inf() -> f32 {",
    "  var bits: u32 = 0x7f800000u;",
    "  return bitcast<f32>(bits);",
    "}",
    "fn bg_f32_nan() -> f32 {",
    "  var bits: u32 = 0x7fc00000u;",
    "  return bitcast<f32>(bits);",
    "}",
  ];
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
  if (isCudaVectorType(type)) return `${wgslScalar(type)}(${Array.from({ length: cudaVectorLaneCount(type) }, () => zeroValue(cudaVectorScalarType(type) ?? "float")).join(", ")})`;
  if (type === "float") return "0.0";
  if (type === "half") return "f16(0.0)";
  if (type === "uint") return "0u";
  if (type === "bool") return "false";
  if (type === "complex64") return "vec2<f32>(0.0, 0.0)";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "0u";
  return "0";
}

function wgslScalar(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type) ?? "float";
    return `vec${cudaVectorLaneCount(type)}<${wgslScalar(scalar)}>`;
  }
  switch (type) {
    case "float":
      return "f32";
    case "int":
      return "i32";
    case "uint":
      return "u32";
    case "half":
      return "f16";
    case "bool":
      return "bool";
    case "complex64":
      return "vec2<f32>";
    case "texture2d":
    case "surface2d":
    case "devicepool":
    case "voidptr":
      return "u32";
    case "void":
      return "void";
  }
}

function wgslUniformScalar(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) return wgslScalar(type);
  if (type === "complex64") return "vec2<f32>";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "u32";
  return type === "bool" ? "u32" : wgslScalar(type);
}

function wgslBindingType(type: CudaLiteScalarType): "f16" | "f32" | "i32" | "u32" {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type);
    return scalar === "int" ? "i32" : scalar === "uint" ? "u32" : scalar === "half" ? "f16" : "f32";
  }
  if (type === "half") return "f16";
  if (type === "int") return "i32";
  if (type === "uint") return "u32";
  if (type === "bool") return "u32";
  if (type === "complex64") return "f32";
  if (type === "texture2d") return "f32";
  if (type === "surface2d") return "f32";
  if (type === "devicepool" || type === "voidptr") return "u32";
  return "f32";
}

function isSurfaceParam(param: CudaLiteParam): boolean {
  return param.valueType === "surface2d";
}

function isTextureParam(param: CudaLiteParam): boolean {
  return param.valueType === "texture2d";
}

function textureBindings(ir: KernelIrModule): readonly { readonly name: string }[] {
  return [
    ...ir.textures.map((texture) => ({ name: texture.name })),
    ...ir.params.filter(isTextureParam).map((param) => ({ name: param.name })),
  ];
}

function isDevicePoolParam(param: CudaLiteParam): boolean {
  return param.pointer && param.valueType === "devicepool";
}

function isEmittedPointerVar(statement: CudaLiteVarDecl, context: EmitContext): boolean {
  return statement.valueType === "voidptr" || context.poolPointerFor(statement.name) !== undefined;
}

function rawPoolHelperName(baseName: string, offsetName: string): string {
  return `bg_raw_pool_alloc_${baseName}_${offsetName}`;
}

function surfaceWidthField(name: string): string {
  return `${name}_width`;
}

function surfaceHeightField(name: string): string {
  return `${name}_height`;
}

function sizeofType(typeName: string): number {
  switch (typeName) {
    case "half":
    case "__half":
      return 2;
    case "cufftComplex":
      return 8;
    default:
      return 4;
  }
}

function indent(level: number): string {
  return "  ".repeat(level);
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}
