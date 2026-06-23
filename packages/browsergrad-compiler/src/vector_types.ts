import type { CudaLiteScalarType } from "./types.js";

export type CudaLiteVectorType =
  | "float2"
  | "float3"
  | "float4"
  | "half2"
  | "bf162"
  | "int2"
  | "int3"
  | "int4"
  | "uint2"
  | "uint3"
  | "uint4";

export interface CudaVectorTypeInfo {
  readonly scalarType: "float" | "half" | "bf16" | "int" | "uint";
  readonly lanes: 2 | 3 | 4;
  readonly fields: readonly string[];
}

export const CUDA_VECTOR_TYPES: ReadonlyMap<CudaLiteVectorType, CudaVectorTypeInfo> = new Map([
  ["float2", { scalarType: "float", lanes: 2, fields: ["x", "y"] }],
  ["float3", { scalarType: "float", lanes: 3, fields: ["x", "y", "z"] }],
  ["float4", { scalarType: "float", lanes: 4, fields: ["x", "y", "z", "w"] }],
  ["half2", { scalarType: "half", lanes: 2, fields: ["x", "y"] }],
  ["bf162", { scalarType: "bf16", lanes: 2, fields: ["x", "y"] }],
  ["int2", { scalarType: "int", lanes: 2, fields: ["x", "y"] }],
  ["int3", { scalarType: "int", lanes: 3, fields: ["x", "y", "z"] }],
  ["int4", { scalarType: "int", lanes: 4, fields: ["x", "y", "z", "w"] }],
  ["uint2", { scalarType: "uint", lanes: 2, fields: ["x", "y"] }],
  ["uint3", { scalarType: "uint", lanes: 3, fields: ["x", "y", "z"] }],
  ["uint4", { scalarType: "uint", lanes: 4, fields: ["x", "y", "z", "w"] }],
]);

export const CUDA_VECTOR_TYPE_ALIASES: ReadonlyMap<string, CudaLiteVectorType> = new Map([
  ["char2", "int2"],
  ["char3", "int3"],
  ["char4", "int4"],
  ["uchar2", "uint2"],
  ["uchar3", "uint3"],
  ["uchar4", "uint4"],
  ["uint8_t2", "uint2"],
  ["uint8_t3", "uint3"],
  ["uint8_t4", "uint4"],
  ["int8_t2", "int2"],
  ["int8_t3", "int3"],
  ["int8_t4", "int4"],
]);

export const CUDA_VECTOR_CONSTRUCTORS: ReadonlyMap<string, CudaLiteVectorType> = new Map([
  ...[...CUDA_VECTOR_TYPES.keys()].map((type) => [`make_${type}`, type] as const),
  ...[...CUDA_VECTOR_TYPE_ALIASES].map(([alias, type]) => [`make_${alias}`, type] as const),
]);

export function isCudaVectorType(type: CudaLiteScalarType | string | undefined): type is CudaLiteVectorType {
  return type !== undefined && CUDA_VECTOR_TYPES.has(type as CudaLiteVectorType);
}

export function cudaVectorTypeAlias(type: string | undefined): CudaLiteVectorType | undefined {
  return type === undefined ? undefined : CUDA_VECTOR_TYPE_ALIASES.get(type);
}

export function cudaVectorTypeInfo(type: CudaLiteScalarType): CudaVectorTypeInfo | undefined {
  return CUDA_VECTOR_TYPES.get(type as CudaLiteVectorType);
}

export function cudaVectorLaneCount(type: CudaLiteScalarType | undefined): number {
  return cudaVectorTypeInfo(type as CudaLiteScalarType)?.lanes ?? 1;
}

export function cudaVectorScalarType(type: CudaLiteScalarType): "float" | "half" | "bf16" | "int" | "uint" | undefined {
  return cudaVectorTypeInfo(type)?.scalarType;
}

export function cudaVectorFieldIndex(type: CudaLiteScalarType, field: string): number | undefined {
  const info = cudaVectorTypeInfo(type);
  if (!info) return undefined;
  const index = info.fields.indexOf(field);
  return index < 0 ? undefined : index;
}

export function cudaVectorConstructorType(name: string): CudaLiteVectorType | undefined {
  return CUDA_VECTOR_CONSTRUCTORS.get(name);
}
