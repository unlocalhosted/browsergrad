import { matrixTileStorageDimensions } from "./matrix_tiles.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";
import {
  externalConstantDimensions,
  wgslScalar,
  zeroValue,
} from "./wgsl_storage.js";
import {
  CudaLiteCompilerError,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type CudaLiteParam,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface WgslDeclarationEmitContext {
  readonly ir: KernelIrModule;
  nameFor(name: string): string;
  isAtomicShared(name: string): boolean;
}

export type WgslExpressionEmitter = (expression: CudaLiteExpression) => string;

export function emitSharedType(statement: CudaLiteVarDecl, ir: KernelIrModule): string {
  if (ir.atomicShared.includes(statement.name) && isCudaVectorType(statement.valueType)) {
    const scalar = cudaVectorScalarType(statement.valueType) ?? "uint";
    const element = scalar === "float" ? "atomic<u32>" : `atomic<${wgslScalar(scalar)}>`;
    const lanes = cudaVectorLaneCount(statement.valueType);
    const total = statement.dimensions.reduce((product, dimension) => product * dimension, 1) * lanes;
    return `array<${element}, ${total}>`;
  }
  let type = ir.atomicShared.includes(statement.name)
    ? `atomic<${statement.valueType === "float" ? "u32" : wgslScalar(statement.valueType)}>`
    : wgslScalar(statement.valueType);
  if (statement.valueType === "uchar" && statement.dimensions.length > 0) {
    const totalBytes = statement.dimensions.reduce((product, item) => product * item, 1);
    return `array<u32, ${Math.ceil(totalBytes / 4)}>`;
  }
  for (let i = statement.dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${statement.dimensions[i]}>`;
  }
  return type;
}

export function emitLocalArrayType(statement: CudaLiteVarDecl): string {
  const dimensions = matrixTileStorageDimensions(statement);
  let type = wgslScalar(statement.valueType);
  for (let i = dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${dimensions[i]}>`;
  }
  return type;
}

export function emitLocalArrayInitializer(
  statement: CudaLiteVarDecl,
  context: WgslDeclarationEmitContext,
  indentLevel: number,
  indent: (level: number) => string,
  emitExpression: WgslExpressionEmitter,
): string[] {
  if (!statement.init) return [];
  const elements = flattenInitializerExpressions(statement.init);
  if (elements.length === 0) return [];
  const prefix = indent(indentLevel);
  const totalElements = statement.dimensions.reduce((product, item) => product * item, 1);
  return elements.map((element, flatIndex) =>
    `${prefix}${emitLocalArrayElementAccess(context.nameFor(statement.name), statement.dimensions, flatIndex)} = ${emitExpression(element)};`,
  ).slice(0, totalElements);
}

export function flattenInitializerExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

export function emitLocalArrayElementAccess(name: string, dimensions: readonly number[], flatIndex: number): string {
  let remainder = flatIndex;
  const indices: number[] = [];
  for (let i = dimensions.length - 1; i >= 0; i--) {
    const size = dimensions[i]!;
    indices.unshift(remainder % size);
    remainder = Math.trunc(remainder / size);
  }
  return indices.reduce((expr, index) => `${expr}[${index}]`, name);
}

export function emitConstantArrayType(constant: CudaLiteGlobalConstant): string {
  let type = wgslScalar(cudaVectorScalarType(constant.valueType) ?? constant.valueType);
  const dimensions = externalConstantDimensions(constant);
  for (let i = dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${dimensions[i]}>`;
  }
  return type;
}

export function isExternalConstantBinding(constant: CudaLiteGlobalConstant): boolean {
  return constant.init === undefined && (constant.dimensions.length > 0 || isCudaVectorType(constant.valueType));
}

export function emitInitializedConstant(
  constant: CudaLiteGlobalConstant,
  context: WgslDeclarationEmitContext,
  emitExpression: WgslExpressionEmitter,
): string {
  if (!constant.init) {
    throw featureError("invalid-constant-initializer", `constant '${constant.name}' has no initializer`);
  }
  if (constant.dimensions.length === 0) {
    if (isCudaVectorType(constant.valueType)) {
      if (constant.init.kind !== "initializer") {
        return `const ${context.nameFor(constant.name)}: ${wgslScalar(constant.valueType)} = ${emitConstantInitializerExpression(constant.init, emitExpression)};`;
      }
      const lanes = cudaVectorLaneCount(constant.valueType);
      const values = flattenInitializerExpressions(constant.init)
        .slice(0, lanes)
        .map((expression) => emitConstantInitializerExpression(expression, emitExpression));
      while (values.length < lanes) values.push(zeroValue(cudaVectorScalarType(constant.valueType) ?? "float"));
      return `const ${context.nameFor(constant.name)}: ${wgslScalar(constant.valueType)} = ${wgslScalar(constant.valueType)}(${values.join(", ")});`;
    }
    return `const ${context.nameFor(constant.name)}: ${wgslScalar(constant.valueType)} = ${emitConstantInitializerExpression(constant.init, emitExpression)};`;
  }
  const type = emitConstantArrayType(constant);
  const total = constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  const values = flattenInitializerExpressions(constant.init)
    .slice(0, total)
    .map((expression) => emitConstantInitializerExpression(expression, emitExpression));
  while (values.length < total) values.push(zeroValue(constant.valueType));
  return `const ${context.nameFor(constant.name)}: ${type} = ${type}(${values.join(", ")});`;
}

export function emitConstantInitializerExpression(expression: CudaLiteExpression, emitExpression: WgslExpressionEmitter): string {
  if (expression.kind === "initializer") {
    throw featureError("invalid-constant-initializer", "nested constant initializer must be flattened before WGSL emission");
  }
  return emitExpression(expression);
}

export function functionBodyHasReturn(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "return") return true;
    if (statement.kind === "if" && (functionBodyHasReturn(statement.consequent) || (statement.alternate && functionBodyHasReturn(statement.alternate)))) {
      return true;
    }
  }
  return false;
}

export function isSurfaceParam(param: CudaLiteParam): boolean {
  return param.valueType === "surface2d";
}

export function isTextureParam(param: CudaLiteParam): boolean {
  return param.valueType === "texture2d";
}

export function textureBindings(ir: KernelIrModule): readonly { readonly name: string }[] {
  return [
    ...ir.textures.map((texture) => ({ name: texture.name })),
    ...ir.params.filter(isTextureParam).map((param) => ({ name: param.name })),
  ];
}

export function isDevicePoolParam(param: CudaLiteParam): boolean {
  return param.pointer && param.valueType === "devicepool";
}

export function surfaceWidthField(name: string): string {
  return `${name}_width`;
}

export function surfaceHeightField(name: string): string {
  return `${name}_height`;
}

export function isEmittedPointerVar(statement: CudaLiteVarDecl, poolPointerFor: (name: string) => unknown): boolean {
  return statement.valueType === "voidptr" || poolPointerFor(statement.name) !== undefined;
}

export function emitSharedAddressIndex(
  expression: CudaLiteExpression,
  ir: KernelIrModule,
  emitExpression: WgslExpressionEmitter,
): string | undefined {
  const target = expression.kind === "unary" && expression.operator === "&" ? expression.argument : expression;
  const indexes: CudaLiteExpression[] = [];
  let cursor = target;
  while (cursor.kind === "index") {
    indexes.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const declaration = ir.sharedDeclarations.find((item) => item.name === cursor.name);
  if (!declaration) return undefined;
  if (indexes.length === 0) return "0u";
  const terms: string[] = [];
  for (let i = 0; i < indexes.length; i++) {
    const stride = declaration.dimensions.slice(i + 1).reduce((product, dimension) => product * dimension, 1);
    const value = `u32(${emitExpression(indexes[i]!)})`;
    terms.push(stride === 1 ? value : `(${value} * ${stride}u)`);
  }
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}
