import {
  CUDA_VECTOR_TYPES,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";
import {
  cudaLiteDimensionStride as dimensionStride,
} from "./cuda_lite_values.js";
import type { CudaLiteScalarType } from "./types.js";

export function zeroValue(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    return `${wgslScalar(type)}(${Array.from({ length: cudaVectorLaneCount(type) }, () => zeroValue(cudaVectorScalarType(type) ?? "float")).join(", ")})`;
  }
  if (type === "float") return "0.0";
  if (type === "half") return "f16(0.0)";
  if (type === "bf16") return "0.0";
  if (type === "uint" || type === "uchar") return "0u";
  if (type === "bool") return "false";
  if (type === "complex64") return "vec2<f32>(0.0, 0.0)";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "0u";
  return "0";
}

export function wgslScalar(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type) ?? "float";
    return `vec${cudaVectorLaneCount(type)}<${wgslScalar(scalar)}>`;
  }
  switch (type) {
    case "float":
    case "double":
      return "f32";
    case "int":
      return "i32";
    case "uint":
    case "uchar":
      return "u32";
    case "half":
      return "f16";
    case "bf16":
      return "f32";
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

export function cudaScalarWgslType(type: CudaLiteScalarType): "f32" | "f16" | "i32" | "u32" | "bool" | undefined {
  if (type === "float" || type === "double" || type === "bf16") return "f32";
  if (type === "half") return "f16";
  if (type === "int") return "i32";
  if (type === "uint" || type === "uchar") return "u32";
  if (type === "bool") return "bool";
  return undefined;
}

export function bitcastStorageViewType(from: CudaLiteScalarType, to: CudaLiteScalarType): "f32" | "i32" | "u32" | undefined {
  const source = cudaScalarWgslType(from);
  const target = cudaScalarWgslType(to);
  if (!source || !target || source === "bool" || target === "bool") return undefined;
  if ((source === "f32" || source === "i32" || source === "u32") &&
    (target === "f32" || target === "i32" || target === "u32")) {
    return target;
  }
  return undefined;
}

export function emitStorageCarrierAsU32(access: string, storageType: CudaLiteScalarType): string {
  const storageScalar = cudaScalarWgslType(storageType);
  if (storageScalar === "u32") return access;
  if (storageScalar === "i32" || storageScalar === "f32") return `bitcast<u32>(${access})`;
  return `u32(${access})`;
}

export function emitU32AsStorageCarrier(value: string, storageType: CudaLiteScalarType): string {
  const storageScalar = cudaScalarWgslType(storageType);
  if (storageScalar === "u32") return value;
  if (storageScalar === "i32" || storageScalar === "f32") return `bitcast<${storageScalar}>(${value})`;
  return `${wgslScalar(storageType)}(${value})`;
}

export function emitSharedFlatAccess(name: string, dimensions: readonly number[], index: string): string {
  if (dimensions.length === 0) return name;
  if (dimensions.length <= 1) return `${name}[${index}]`;
  return dimensions.reduce((expr, dimension, axis) => {
    const stride = dimensionStride(dimensions, axis);
    const rawAxisIndex = stride === 1
      ? `(${index} % ${dimension}u)`
      : `(((${index}) / ${stride}u) % ${dimension}u)`;
    const axisIndex = dimension <= 1 ? "0u" : `min(${rawAxisIndex}, ${dimension - 1}u)`;
    return `${expr}[${axisIndex}]`;
  }, name);
}

export function vectorStorageBase(index: string, lanes: number): string {
  return `(u32(${index}) * ${lanes}u)`;
}

export function vectorFieldName(index: number): string {
  return index === 0 ? "x" : index === 1 ? "y" : index === 2 ? "z" : "w";
}

export function emitVectorLaneSetExpression(
  base: string,
  type: CudaLiteScalarType,
  index: string | number,
  value: string,
): string {
  const scalar = wgslScalar(cudaVectorScalarType(type) ?? "float");
  const indexExpression = typeof index === "number" ? `${index}u` : index;
  const values = Array.from({ length: cudaVectorLaneCount(type) }, (_unused, lane) => {
    const current = `(${base}).${vectorFieldName(lane)}`;
    return `select(${current}, ${scalar}(${value}), ${indexExpression} == ${lane}u)`;
  });
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

export function emitVectorStorageRead(name: string, type: CudaLiteScalarType, index: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  const values = Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u]`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

export function emitVectorStorageWrite(name: string, type: CudaLiteScalarType, index: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

export function emitVectorStorageFieldWrite(name: string, type: CudaLiteScalarType, index: string, field: string, value: string): string | undefined {
  const fieldIndex = cudaVectorFieldIndex(type, field);
  if (fieldIndex === undefined) return undefined;
  const base = vectorStorageBase(index, cudaVectorLaneCount(type));
  return `${name}[${base} + ${fieldIndex}u] = ${value}`;
}

export function emitVectorStorageReadAt(name: string, type: CudaLiteScalarType, storageIndex: string): string {
  const lanes = cudaVectorLaneCount(type);
  const scalar = cudaVectorScalarType(type) ?? "float";
  const values = Array.from({ length: lanes }, (_, lane) => `${wgslScalar(scalar)}(${name}[${storageIndex} + ${lane}u])`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

export function emitVectorStorageWriteAt(name: string, type: CudaLiteScalarType, storageIndex: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${storageIndex} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

export function emitVectorStorageFieldWriteAt(name: string, storageIndex: string, fieldIndex: number, value: string): string {
  return `${name}[${storageIndex} + ${fieldIndex}u] = ${value}`;
}

export function emitVectorConstructor(vectorType: CudaLiteScalarType, args: readonly string[]): string {
  const info = CUDA_VECTOR_TYPES.get(vectorType as never);
  if (!info) return `${wgslScalar(vectorType)}(${args.join(", ")})`;
  if (args.length === 1) return `${wgslScalar(vectorType)}(${Array(info.lanes).fill(args[0] ?? "0").join(", ")})`;
  return `${wgslScalar(vectorType)}(${args.join(", ")})`;
}

export function castExpressionToVectorScalar(value: string, vectorType: CudaLiteScalarType): string {
  return `${wgslScalar(cudaVectorScalarType(vectorType) ?? "float")}(${value})`;
}

export function emitVectorSplat(vectorType: CudaLiteScalarType, value: string): string {
  return `${wgslScalar(vectorType)}(${Array.from({ length: cudaVectorLaneCount(vectorType) }, () => value).join(", ")})`;
}

export function wgslElementByteSize(valueType: CudaLiteScalarType): number {
  if (valueType === "uchar") return 1;
  if (valueType === "half") return 2;
  if (valueType === "bf16") return 2;
  if (isCudaVectorType(valueType)) return cudaVectorLaneCount(valueType) * wgslElementByteSize(cudaVectorScalarType(valueType) ?? "float");
  return 4;
}
