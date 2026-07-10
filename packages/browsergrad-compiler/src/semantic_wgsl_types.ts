import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import type { CudaLiteScalarType } from "./types.js";
import {
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";

export type SemanticWgslValueType =
  | WgslValueType
  | "bool"
  | "vec2<f32>"
  | "vec3<f32>"
  | "vec4<f32>"
  | "vec2<f16>"
  | "vec2<i32>"
  | "vec3<i32>"
  | "vec4<i32>"
  | "vec2<u32>"
  | "vec3<u32>"
  | "vec4<u32>";

export function wgslBindingType(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  if (scalar !== "bool") return scalar;
  return "u32";
}

export function wgslScalar(valueType: CudaLiteScalarType | undefined): WgslValueType | "bool" {
  if (valueType === "half" || valueType === "half2") return "f16";
  const scalarType = isCudaVectorType(valueType) ? cudaVectorScalarType(valueType) : valueType;
  if (scalarType === "int") return "i32";
  if (scalarType === "uint" || scalarType === "uchar") return "u32";
  if (scalarType === "half") return "f16";
  if (valueType === "bool") return "bool";
  return "f32";
}

export function wgslValueType(valueType: CudaLiteScalarType | undefined): SemanticWgslValueType {
  if (valueType === "float2") return "vec2<f32>";
  if (valueType === "float3") return "vec3<f32>";
  if (valueType === "float4") return "vec4<f32>";
  if (valueType === "half2") return "vec2<f16>";
  if (valueType === "bf162") return "vec2<f32>";
  if (valueType === "int2") return "vec2<i32>";
  if (valueType === "int3") return "vec3<i32>";
  if (valueType === "int4") return "vec4<i32>";
  if (valueType === "uint2") return "vec2<u32>";
  if (valueType === "uint3") return "vec3<u32>";
  if (valueType === "uint4") return "vec4<u32>";
  return wgslScalar(valueType);
}

export function wgslVectorScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "half2") return "f16";
  if (valueType === "int2" || valueType === "int3" || valueType === "int4") return "i32";
  if (valueType === "uint2" || valueType === "uint3" || valueType === "uint4") return "u32";
  return "f32";
}

export function wgslValueScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  const scalar = wgslScalar(valueType);
  return scalar === "bool" ? "u32" : scalar;
}

export function wgslAtomicScalar(valueType: CudaLiteScalarType | undefined): Extract<WgslValueType, "i32" | "u32"> {
  return valueType === "int" ? "i32" : "u32";
}

export function wgslUniformScalar(valueType: CudaLiteScalarType | undefined): WgslValueType {
  if (valueType === "half") return "f16";
  if (valueType === "int") return "i32";
  if (valueType === "uint" || valueType === "uchar" || valueType === "bool") return "u32";
  return "f32";
}

export function zeroForType(valueType: SemanticWgslValueType): string {
  if (valueType === "u32") return "0u";
  if (valueType === "i32") return "0";
  if (valueType === "bool") return "false";
  if (valueType === "f16") return "f16(0.0)";
  if (valueType === "vec2<f16>") return "vec2<f16>(f16(0.0))";
  if (valueType === "vec2<f32>") return "vec2<f32>(0.0)";
  if (valueType === "vec3<f32>") return "vec3<f32>(0.0)";
  if (valueType === "vec4<f32>") return "vec4<f32>(0.0)";
  if (valueType === "vec2<i32>") return "vec2<i32>(0)";
  if (valueType === "vec3<i32>") return "vec3<i32>(0)";
  if (valueType === "vec4<i32>") return "vec4<i32>(0)";
  if (valueType === "vec2<u32>") return "vec2<u32>(0u)";
  if (valueType === "vec3<u32>") return "vec3<u32>(0u)";
  if (valueType === "vec4<u32>") return "vec4<u32>(0u)";
  return "0.0";
}
