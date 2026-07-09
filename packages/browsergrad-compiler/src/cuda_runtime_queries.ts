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
