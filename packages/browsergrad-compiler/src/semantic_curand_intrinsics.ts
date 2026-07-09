import type { CudaLiteScalarType } from "./types.js";

export const SEMANTIC_CURAND_CALLS = new Set([
  "curand_init",
  "curand",
  "curand_uniform",
  "curand_uniform4",
  "curand_uniform_double",
  "curand_normal",
  "curand_normal2",
  "curand_normal4",
  "curand_normal_double",
  "curand_log_normal",
  "curand_log_normal2",
  "curand_log_normal4",
  "curand_log_normal_double",
  "curand_poisson",
  "curand_poisson4",
  "skipahead",
]);

export const SEMANTIC_CURAND_VECTOR_RETURN_TYPES = new Map<string, CudaLiteScalarType>([
  ["curand_uniform4", "float4"],
  ["curand_normal2", "float2"],
  ["curand_normal4", "float4"],
  ["curand_log_normal2", "float2"],
  ["curand_log_normal4", "float4"],
  ["curand_poisson4", "uint4"],
]);

export const SEMANTIC_CURAND_VECTOR_CALLS = new Set(SEMANTIC_CURAND_VECTOR_RETURN_TYPES.keys());

export const SEMANTIC_CURAND_STATE_ONLY_CALLS = new Set([
  "curand",
  "curand_uniform",
  "curand_uniform4",
  "curand_uniform_double",
  "curand_normal",
  "curand_normal2",
  "curand_normal4",
  "curand_normal_double",
]);

export const SEMANTIC_CURAND_DISTRIBUTION_CALLS = new Set([
  "curand_log_normal",
  "curand_log_normal2",
  "curand_log_normal4",
  "curand_log_normal_double",
  "curand_poisson",
  "curand_poisson4",
]);
