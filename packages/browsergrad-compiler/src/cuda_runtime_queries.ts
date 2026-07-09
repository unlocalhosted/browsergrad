import { cudaDeviceAttributeValue } from "./cuda_device_attributes.js";
import { cudaDeviceLimitValue } from "./cuda_device_limits.js";
import type { CudaLiteScalarType } from "./types.js";

const CUDA_INTEGER_RUNTIME_QUERY_CALLS: ReadonlySet<string> = new Set([
  "cudaDeviceCanAccessPeer",
  "cudaDeviceGetAttribute",
  "cudaDeviceGetCacheConfig",
  "cudaDeviceGetLimit",
  "cudaDeviceGetSharedMemConfig",
  "cudaDeviceGetStreamPriorityRange",
  "cudaDriverGetVersion",
  "cudaEventCreate",
  "cudaEventCreateWithFlags",
  "cudaGetDevice",
  "cudaGetDeviceCount",
  "cudaGetDeviceFlags",
  "cudaGraphCreate",
  "cudaGraphExecUpdate",
  "cudaGraphInstantiate",
  "cudaGraphInstantiateWithFlags",
  "cudaMemGetInfo",
  "cudaOccupancyAvailableDynamicSMemPerBlock",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cudaOccupancyMaxPotentialBlockSize",
  "cudaOccupancyMaxPotentialBlockSizeWithFlags",
  "cudaRuntimeGetVersion",
  "cudaStreamCreate",
  "cudaStreamCreateWithFlags",
  "cudaStreamCreateWithPriority",
  "cudaStreamEndCapture",
  "cudaStreamGetCaptureInfo",
  "cudaStreamGetCaptureInfo_v2",
  "cudaStreamGetDevice",
  "cudaStreamGetFlags",
  "cudaStreamGetId",
  "cudaStreamGetPriority",
  "cudaStreamIsCapturing",
  "cudaThreadExchangeStreamCaptureMode",
  "cudaThreadGetCacheConfig",
  "cudaThreadGetLimit",
] as const);

export function isCudaIntegerRuntimeQueryCall(name: string | undefined): boolean {
  return name !== undefined && CUDA_INTEGER_RUNTIME_QUERY_CALLS.has(name);
}

export function isCudaRuntimeQueryWriteCall(name: string | undefined): boolean {
  return isCudaIntegerRuntimeQueryCall(name) || name === "cudaEventElapsedTime";
}

export function cudaIntegerRuntimeQueryTargetValueType(name: string | undefined): Exclude<CudaLiteScalarType, "void"> {
  return name === "cudaDeviceGetLimit" ||
    name === "cudaThreadGetLimit" ||
    name === "cudaGetDeviceFlags" ||
    name === "cudaMemGetInfo" ||
    name === "cudaStreamCreate" ||
    name === "cudaStreamCreateWithFlags" ||
    name === "cudaStreamCreateWithPriority" ||
    name === "cudaStreamGetFlags" ||
    name === "cudaStreamGetId" ||
    name === "cudaStreamEndCapture" ||
    name === "cudaGraphCreate" ||
    name === "cudaGraphInstantiate" ||
    name === "cudaGraphInstantiateWithFlags" ||
    name === "cudaGraphExecUpdate" ||
    name === "cudaOccupancyAvailableDynamicSMemPerBlock" ||
    name === "cudaEventCreate" ||
    name === "cudaEventCreateWithFlags"
    ? "uint"
    : "int";
}

export function cudaStreamGetCaptureInfoTargetValueType(index: number): Exclude<CudaLiteScalarType, "void"> {
  return index === 1 ? "int" : "uint";
}

export function cudaIntegerRuntimeQueryValue(
  name: string | undefined,
  constantArg: (index: number) => number | undefined,
): number {
  if (name === "cudaGetDeviceCount") return 1;
  if (name === "cudaDeviceGetAttribute") return cudaDeviceAttributeValue(constantArg(1) ?? 0);
  if (name === "cudaDeviceGetLimit" || name === "cudaThreadGetLimit") return cudaDeviceLimitValue(constantArg(1) ?? 0);
  if (name === "cudaDeviceCanAccessPeer") return 1;
  if (name === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" || name === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") return 1;
  if (name === "cudaOccupancyAvailableDynamicSMemPerBlock") return 49152;
  if (name === "cudaRuntimeGetVersion" || name === "cudaDriverGetVersion") return 12000;
  return 0;
}

export function cudaRuntimeMemInfoBytes(): number {
  return 268435456;
}

export function cudaRuntimeAvailableDynamicSmemBytes(): number {
  return 49152;
}
