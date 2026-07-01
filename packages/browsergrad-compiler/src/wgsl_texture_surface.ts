import { expressionName } from "./analyzer.js";
import type { CudaLiteCallExpression, CudaLiteExpression, CudaLiteScalarType } from "./types.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
  type CudaLiteVectorType,
} from "./vector_types.js";
import { wgslScalar } from "./wgsl_storage.js";

type EmitMode = "value" | "lvalue";

export interface WgslTextureSurfaceEmitContext {
  readonly requiredFeatures: readonly string[];
  readonly textureNames: readonly string[];
  readonly surfaceNames: readonly string[];
  readonly uniformParamsName: string;
  nameFor(name: string): string;
  surfaceWidthField(name: string): string;
  surfaceHeightField(name: string): string;
  emitExpression(expression: CudaLiteExpression, mode?: EmitMode): string;
  emitExpressionAsValueType(expression: CudaLiteExpression, valueType: CudaLiteScalarType): string;
  expressionValueType(expression: CudaLiteExpression): CudaLiteScalarType | undefined;
}

export function emitTextureHelper(name: string, context: WgslTextureSurfaceEmitContext): string[] {
  const safeName = context.nameFor(name);
  const lines = [
    `fn bg_tex2d_coord_${name}(x: f32, y: f32) -> vec2<i32> {`,
    `  let dims = textureDimensions(${safeName});`,
    "  let max_coord = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);",
    "  return clamp(vec2<i32>(i32(floor(x)), i32(floor(y))), vec2<i32>(0, 0), max_coord);",
    "}",
    `fn bg_tex2d_f32_${name}(x: f32, y: f32) -> f32 {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r;`,
    "}",
    `fn bg_tex2d_float2_${name}(x: f32, y: f32) -> vec2<f32> {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).xy;`,
    "}",
    `fn bg_tex2d_float3_${name}(x: f32, y: f32) -> vec3<f32> {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).xyz;`,
    "}",
    `fn bg_tex2d_float4_${name}(x: f32, y: f32) -> vec4<f32> {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0);`,
    "}",
    `fn bg_tex2d_int_${name}(x: f32, y: f32) -> i32 {`,
    `  return i32(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
    "}",
    `fn bg_tex2d_uint_${name}(x: f32, y: f32) -> u32 {`,
    `  return u32(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
    "}",
    `fn bg_tex2d_uchar_${name}(x: f32, y: f32) -> u32 {`,
    `  return u32(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
    "}",
    ...emitTextureVectorCastHelpers(name, safeName, ["int2", "int3", "int4", "uint2", "uint3", "uint4"]),
  ];
  if (context.requiredFeatures.includes("shader-f16")) {
    lines.push(
      `fn bg_tex2d_half_${name}(x: f32, y: f32) -> f16 {`,
      `  return f16(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
      "}",
      `fn bg_tex2d_half2_${name}(x: f32, y: f32) -> vec2<f16> {`,
      `  return vec2<f16>(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).xy);`,
      "}",
    );
  }
  return lines;
}

export function emitCubeTextureAtlasHelpers(): string[] {
  return [
    "fn bg_cube_face(x: f32, y: f32, z: f32) -> f32 {",
    "  let ax = abs(x);",
    "  let ay = abs(y);",
    "  let az = abs(z);",
    "  if (ax >= ay && ax >= az) {",
    "    return select(1.0, 0.0, x >= 0.0);",
    "  }",
    "  if (ay >= az) {",
    "    return select(3.0, 2.0, y >= 0.0);",
    "  }",
    "  return select(5.0, 4.0, z >= 0.0);",
    "}",
    "fn bg_cube_u(x: f32, y: f32, z: f32) -> f32 {",
    "  let ax = max(abs(x), 0.000001);",
    "  let ay = max(abs(y), 0.000001);",
    "  let az = max(abs(z), 0.000001);",
    "  if (ax >= ay && ax >= az) { return z / ax; }",
    "  return x / max(ay, az);",
    "}",
    "fn bg_cube_v(x: f32, y: f32, z: f32) -> f32 {",
    "  let ax = max(abs(x), 0.000001);",
    "  let ay = max(abs(y), 0.000001);",
    "  let az = max(abs(z), 0.000001);",
    "  if (ax >= ay && ax >= az) { return y / ax; }",
    "  if (ay >= az) { return z / ay; }",
    "  return y / az;",
    "}",
  ];
}

export function textureReadHelperSuffix(valueType: CudaLiteScalarType | undefined): string {
  if (valueType === "int" || valueType === "uint" || valueType === "uchar") return valueType;
  if (valueType === "half" || valueType === "half2") return valueType;
  if (isCudaVectorType(valueType)) return valueType;
  return "f32";
}

export function isTextureBindingName(name: string, context: WgslTextureSurfaceEmitContext): boolean {
  return context.textureNames.includes(name);
}

export function isSurfaceBindingName(name: string, context: WgslTextureSurfaceEmitContext): boolean {
  return context.surfaceNames.includes(name);
}

export function isTextureReadCall(name: string): boolean {
  return name === "tex1D" ||
    name === "tex1Dfetch" ||
    name === "tex2D" ||
    name === "tex2DLod" ||
    name === "tex2DLayered" ||
    name === "tex3D" ||
    name === "texCubemap";
}

export function textureReadArgsForEmit(
  expression: CudaLiteCallExpression,
  context: WgslTextureSurfaceEmitContext,
): readonly [string, string] {
  const name = expressionName(expression.callee);
  const x = textureCoordForEmit(expression.args[1], context);
  if (name === "tex1D" || name === "tex1Dfetch") return [x, "0.0"];
  const y = textureCoordForEmit(expression.args[2], context);
  if (name === "tex2D" || name === "tex2DLod") return [x, y];
  if (name === "tex2DLayered" || name === "tex3D") {
    return [x, `(${y} + ${textureCoordForEmit(expression.args[3], context)})`];
  }
  if (name === "texCubemap") {
    const z = textureCoordForEmit(expression.args[3], context);
    const texture = emitTextureArgument(expression.args[0], context);
    const width = `f32(textureDimensions(${texture}).x)`;
    const localX = `((bg_cube_u(${x}, ${y}, ${z}) + 1.0) * 0.5 * (${width} - 1.0))`;
    const localY = `((bg_cube_v(${x}, ${y}, ${z}) + 1.0) * 0.5 * (${width} - 1.0) + bg_cube_face(${x}, ${y}, ${z}) * ${width})`;
    return [localX, localY];
  }
  return [x, y];
}

function textureCoordForEmit(expression: CudaLiteExpression | undefined, context: WgslTextureSurfaceEmitContext): string {
  return expression ? `f32(${context.emitExpression(expression)})` : "0.0";
}

export function emitTextureArgument(
  expression: CudaLiteExpression | undefined,
  context: WgslTextureSurfaceEmitContext,
): string {
  if (expression?.kind === "identifier") return context.nameFor(expression.name);
  if (expression) return context.emitExpression(expression);
  return "bg_missing_texture";
}

export function emitSurfaceArgument(
  expression: CudaLiteExpression | undefined,
  context: WgslTextureSurfaceEmitContext,
): string {
  if (expression?.kind === "identifier") {
    const index = context.surfaceNames.indexOf(expression.name);
    if (index >= 0) return `${index}u`;
  }
  if (expression) return context.emitExpressionAsValueType(expression, "surface2d");
  return "4294967295u";
}

export function emitTextureReadExpression(
  texture: string,
  coordArgs: readonly [string, string],
  valueType: CudaLiteScalarType | undefined,
): string {
  const [x, y] = coordArgs;
  const coord = `clamp(vec2<i32>(i32(floor(${x})), i32(floor(${y}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${texture})) - vec2<i32>(1, 1))`;
  const value = `textureLoad(${texture}, ${coord}, 0)`;
  if (valueType === "float2") return `${value}.xy`;
  if (valueType === "float3") return `${value}.xyz`;
  if (valueType === "float4") return value;
  if (valueType === "int") return `i32(${value}.r)`;
  if (valueType === "uint") return `u32(${value}.r)`;
  if (valueType === "uchar") return `u32(${value}.r)`;
  if (valueType === "half") return `f16(${value}.r)`;
  if (valueType === "half2") return `vec2<f16>(${value}.xy)`;
  if (isCudaVectorType(valueType)) {
    const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(valueType));
    const scalarType = wgslScalar(cudaVectorScalarType(valueType) ?? "float");
    return `${wgslScalar(valueType)}(${fields.map((field) => `${scalarType}(${value}.${field})`).join(", ")})`;
  }
  return `${value}.r`;
}

export function emitSurfaceReadExpression(
  surfaceExpression: CudaLiteExpression,
  xBytesExpression: CudaLiteExpression,
  yExpression: CudaLiteExpression,
  zExpression: CudaLiteExpression | undefined,
  valueType: CudaLiteScalarType,
  context: WgslTextureSurfaceEmitContext,
): string {
  const xBytes = context.emitExpressionAsValueType(xBytesExpression, "int");
  const y = context.emitExpressionAsValueType(yExpression, "int");
  const z = zExpression ? context.emitExpressionAsValueType(zExpression, "int") : "0";
  const directSurfaceName = surfaceExpression.kind === "identifier" && isSurfaceBindingName(surfaceExpression.name, context)
    ? surfaceExpression.name
    : undefined;
  const surfaceHandle = directSurfaceName ? undefined : emitSurfaceArgument(surfaceExpression, context);
  const readCall = (x: string): string => directSurfaceName
    ? `bg_surf2dread_${directSurfaceName}(${x}, ${y}, ${z})`
    : `bg_surf2dread(${surfaceHandle}, ${x}, ${y}, ${z})`;
  if (isCudaVectorType(valueType)) {
    const laneCount = cudaVectorLaneCount(valueType);
    const scalarType = wgslScalar(cudaVectorScalarType(valueType) ?? "float");
    const values = Array.from({ length: laneCount }, (_, index) => `${scalarType}(${readCall(`(${xBytes} + ${index * 4})`)})`);
    return `${wgslScalar(valueType)}(${values.join(", ")})`;
  }
  const read = readCall(xBytes);
  if (valueType === "float" || valueType === "double" || valueType === "bf16") return read;
  if (valueType === "int") return `i32(${read})`;
  if (valueType === "uint" || valueType === "uchar") return `u32(${read})`;
  if (valueType === "half") return `f16(${read})`;
  if (valueType === "bool") return `(${read} != 0.0)`;
  if (valueType === "complex64") return `vec2<f32>(${read}, 0.0)`;
  return `${wgslScalar(valueType)}(${read})`;
}

export function emitSurfaceWriteExpression(
  surfaceExpression: CudaLiteExpression,
  valueExpression: CudaLiteExpression,
  xBytesExpression: CudaLiteExpression,
  y: string,
  z: string,
  context: WgslTextureSurfaceEmitContext,
): string {
  const valueType = context.expressionValueType(valueExpression);
  const directSurfaceName = surfaceExpression.kind === "identifier" && isSurfaceBindingName(surfaceExpression.name, context)
    ? surfaceExpression.name
    : undefined;
  const surfaceHandle = directSurfaceName ? undefined : emitSurfaceArgument(surfaceExpression, context);
  const surfaceY = `i32(${y})`;
  const surfaceZ = `i32(${z})`;
  const xBytes = context.emitExpressionAsValueType(xBytesExpression, "int");
  const writeCall = (value: string, x: string): string => directSurfaceName
    ? `bg_surf2dwrite_${directSurfaceName}(${value}, ${x}, ${surfaceY}, ${surfaceZ})`
    : `bg_surf2dwrite(${surfaceHandle}, ${value}, ${x}, ${surfaceY}, ${surfaceZ})`;
  if (isCudaVectorType(valueType)) {
    const value = context.emitExpression(valueExpression);
    const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(valueType));
    return fields.map((field, index) => writeCall(`f32(${value}.${field})`, `(${xBytes} + ${index * 4})`)).join("; ");
  }
  return writeCall(context.emitExpressionAsValueType(valueExpression, "float"), xBytes);
}

function emitTextureVectorCastHelpers(
  name: string,
  safeName: string,
  valueTypes: readonly CudaLiteVectorType[],
): string[] {
  return valueTypes.flatMap((valueType) => {
    const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(valueType));
    const scalarType = wgslScalar(cudaVectorScalarType(valueType) ?? "float");
    const values = fields.map((field) => `${scalarType}(value.${field})`).join(", ");
    return [
      `fn bg_tex2d_${valueType}_${name}(x: f32, y: f32) -> ${wgslScalar(valueType)} {`,
      `  let value = textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0);`,
      `  return ${wgslScalar(valueType)}(${values});`,
      "}",
    ];
  });
}

export function emitSurfaceHelper(name: string, context: WgslTextureSurfaceEmitContext): string[] {
  const safeName = context.nameFor(name);
  return [
    `fn bg_surf2dread_${name}(x_bytes: i32, y: i32, z: i32) -> f32 {`,
    "  let x = x_bytes / 4;",
    `  let index = ((z * i32(${context.uniformParamsName}.${context.nameFor(context.surfaceHeightField(name))})) + y) * i32(${context.uniformParamsName}.${context.nameFor(context.surfaceWidthField(name))}) + x;`,
    `  if (index >= 0 && index < i32(arrayLength(&${safeName}))) {`,
    `    return ${safeName}[index];`,
    "  }",
    "  return 0.0;",
    "}",
    `fn bg_surf2dwrite_${name}(value: f32, x_bytes: i32, y: i32, z: i32) {`,
    "  let x = x_bytes / 4;",
    `  let index = ((z * i32(${context.uniformParamsName}.${context.nameFor(context.surfaceHeightField(name))})) + y) * i32(${context.uniformParamsName}.${context.nameFor(context.surfaceWidthField(name))}) + x;`,
    `  if (index >= 0 && index < i32(arrayLength(&${safeName}))) {`,
    `    ${safeName}[index] = value;`,
    "  }",
    "}",
  ];
}

export function emitSurfaceDispatchHelpers(context: WgslTextureSurfaceEmitContext): string[] {
  if (context.surfaceNames.length === 0) return [];
  const readCases = context.surfaceNames.flatMap((name, index) => [
    `    case ${index}u: {`,
    `      return bg_surf2dread_${name}(x_bytes, y, z);`,
    "    }",
  ]);
  const writeCases = context.surfaceNames.flatMap((name, index) => [
    `    case ${index}u: {`,
    `      bg_surf2dwrite_${name}(value, x_bytes, y, z);`,
    "      return;",
    "    }",
  ]);
  return [
    "fn bg_surf2dread(surface: u32, x_bytes: i32, y: i32, z: i32) -> f32 {",
    "  switch surface {",
    ...readCases,
    "    default: {}",
    "  }",
    "  return 0.0;",
    "}",
    "fn bg_surf2dwrite(surface: u32, value: f32, x_bytes: i32, y: i32, z: i32) {",
    "  switch surface {",
    ...writeCases,
    "    default: {}",
    "  }",
    "}",
  ];
}
