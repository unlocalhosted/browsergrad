import type { CudaLiteScalarType } from "./types.js";

export type CudaLiteVectorType =
  | "float2"
  | "float3"
  | "float4"
  | "half2"
  | "int2"
  | "int3"
  | "int4"
  | "uint2"
  | "uint3"
  | "uint4";

export interface CudaVectorTypeInfo {
  readonly scalarType: "float" | "half" | "int" | "uint";
  readonly lanes: 2 | 3 | 4;
  readonly fields: readonly string[];
}

export const CUDA_VECTOR_TYPES: ReadonlyMap<CudaLiteVectorType, CudaVectorTypeInfo> = new Map([
  ["float2", { scalarType: "float", lanes: 2, fields: ["x", "y"] }],
  ["float3", { scalarType: "float", lanes: 3, fields: ["x", "y", "z"] }],
  ["float4", { scalarType: "float", lanes: 4, fields: ["x", "y", "z", "w"] }],
  ["half2", { scalarType: "half", lanes: 2, fields: ["x", "y"] }],
  ["int2", { scalarType: "int", lanes: 2, fields: ["x", "y"] }],
  ["int3", { scalarType: "int", lanes: 3, fields: ["x", "y", "z"] }],
  ["int4", { scalarType: "int", lanes: 4, fields: ["x", "y", "z", "w"] }],
  ["uint2", { scalarType: "uint", lanes: 2, fields: ["x", "y"] }],
  ["uint3", { scalarType: "uint", lanes: 3, fields: ["x", "y", "z"] }],
  ["uint4", { scalarType: "uint", lanes: 4, fields: ["x", "y", "z", "w"] }],
]);

export function isCudaVectorType(type: CudaLiteScalarType | string | undefined): type is CudaLiteVectorType {
  return type !== undefined && CUDA_VECTOR_TYPES.has(type as CudaLiteVectorType);
}

export function cudaVectorTypeInfo(type: CudaLiteScalarType): CudaVectorTypeInfo | undefined {
  return CUDA_VECTOR_TYPES.get(type as CudaLiteVectorType);
}

export function cudaVectorLaneCount(type: CudaLiteScalarType | undefined): number {
  return cudaVectorTypeInfo(type as CudaLiteScalarType)?.lanes ?? 1;
}

export function cudaVectorScalarType(type: CudaLiteScalarType): "float" | "half" | "int" | "uint" | undefined {
  return cudaVectorTypeInfo(type)?.scalarType;
}

export function cudaVectorFieldIndex(type: CudaLiteScalarType, field: string): number | undefined {
  const info = cudaVectorTypeInfo(type);
  if (!info) return undefined;
  const index = info.fields.indexOf(field);
  return index < 0 ? undefined : index;
}

export function cudaVectorConstructorType(name: string): CudaLiteVectorType | undefined {
  const match = /^make_(float|int|uint)([234])$/u.exec(name) ?? /^make_(half)(2)$/u.exec(name);
  if (!match) return undefined;
  const type = `${match[1]}${match[2]}` as CudaLiteVectorType;
  return CUDA_VECTOR_TYPES.has(type) ? type : undefined;
}
