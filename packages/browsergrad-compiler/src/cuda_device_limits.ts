export const CUDA_DEVICE_LIMIT_ENUM_VALUES = new Map<string, number>([
  ["cudaLimitStackSize", 0],
  ["cudaLimitPrintfFifoSize", 1],
  ["cudaLimitMallocHeapSize", 2],
  ["cudaLimitDevRuntimeSyncDepth", 3],
  ["cudaLimitDevRuntimePendingLaunchCount", 4],
  ["cudaLimitMaxL2FetchGranularity", 5],
  ["cudaLimitPersistingL2CacheSize", 6],
]);

export function cudaDeviceLimitValue(limit: number): number {
  switch (Math.trunc(limit)) {
    case 0: return 1024;
    case 1: return 1048576;
    case 2: return 8388608;
    case 3: return 2;
    case 4: return 2048;
    case 5: return 128;
    case 6: return 0;
    default: return 0;
  }
}
