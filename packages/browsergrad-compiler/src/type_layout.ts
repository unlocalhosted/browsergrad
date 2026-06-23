const TYPE_SIZE_BYTES = new Map<string, number>([
  ["bool", 4],
  ["char", 1],
  ["signed char", 1],
  ["unsigned char", 1],
  ["uchar", 1],
  ["int8_t", 1],
  ["uint8_t", 1],
  ["half", 2],
  ["__half", 2],
  ["bf16", 2],
  ["__nv_bfloat16", 2],
  ["nv_bfloat16", 2],
  ["short", 2],
  ["short int", 2],
  ["unsigned short", 2],
  ["half2", 4],
  ["bf162", 4],
  ["__nv_bfloat162", 4],
  ["float", 4],
  ["int", 4],
  ["uint", 4],
  ["unsigned", 4],
  ["unsigned int", 4],
  ["signed", 4],
  ["signed int", 4],
  ["long", 4],
  ["long int", 4],
  ["long long", 4],
  ["long long int", 4],
  ["size_t", 4],
  ["int32_t", 4],
  ["uint32_t", 4],
  ["int64_t", 4],
  ["uint64_t", 4],
  ["uintptr_t", 4],
  ["clock_t", 4],
  ["curandState", 4],
  ["curandState_t", 4],
  ["CUtexObject", 4],
  ["CUtensorMap", 4],
  ["cudaTextureObject_t", 4],
  ["cudaSurfaceObject_t", 4],
  ["cudaEvent_t", 4],
  ["cudaStream_t", 4],
  ["void", 4],
  ["voidptr", 4],
  ["char2", 2],
  ["uchar2", 2],
  ["int2", 8],
  ["uint2", 8],
  ["float2", 8],
  ["char3", 3],
  ["uchar3", 3],
  ["int3", 12],
  ["uint3", 12],
  ["float3", 12],
  ["char4", 4],
  ["uchar4", 4],
  ["int4", 16],
  ["uint4", 16],
  ["float4", 16],
  ["cufftComplex", 8],
]);

const TYPE_ALIGN_BYTES = new Map<string, number>([
  ...TYPE_SIZE_BYTES,
  ["float3", 4],
  ["int3", 4],
  ["uint3", 4],
  ["char3", 1],
  ["uchar3", 1],
]);

export function sizeofCudaType(typeName: string): number | undefined {
  return TYPE_SIZE_BYTES.get(normalizeLayoutTypeName(typeName));
}

export function alignofCudaType(typeName: string): number | undefined {
  return TYPE_ALIGN_BYTES.get(normalizeLayoutTypeName(typeName));
}

export function normalizeLayoutTypeName(typeName: string): string {
  return typeName
    .replace(/\b(?:const|volatile|typename|class|struct|__restrict__|__restrict|restrict)\b/gu, " ")
    .replace(/[&*]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
}
