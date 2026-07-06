export const CUDA_DEVICE_ATTRIBUTE_ENUM_VALUES = new Map<string, number>([
  ["cudaDevAttrMaxThreadsPerBlock", 1],
  ["cudaDevAttrMaxBlockDimX", 2],
  ["cudaDevAttrMaxBlockDimY", 3],
  ["cudaDevAttrMaxBlockDimZ", 4],
  ["cudaDevAttrMaxGridDimX", 5],
  ["cudaDevAttrMaxGridDimY", 6],
  ["cudaDevAttrMaxGridDimZ", 7],
  ["cudaDevAttrMaxSharedMemoryPerBlock", 8],
  ["cudaDevAttrTotalConstantMemory", 9],
  ["cudaDevAttrWarpSize", 10],
  ["cudaDevAttrMaxPitch", 11],
  ["cudaDevAttrMaxRegistersPerBlock", 12],
  ["cudaDevAttrClockRate", 13],
  ["cudaDevAttrTextureAlignment", 14],
  ["cudaDevAttrGpuOverlap", 15],
  ["cudaDevAttrMultiProcessorCount", 16],
  ["cudaDevAttrKernelExecTimeout", 17],
  ["cudaDevAttrIntegrated", 18],
  ["cudaDevAttrCanMapHostMemory", 19],
  ["cudaDevAttrComputeMode", 20],
  ["cudaDevAttrConcurrentKernels", 31],
  ["cudaDevAttrMaxThreadsPerMultiProcessor", 39],
  ["cudaDevAttrAsyncEngineCount", 40],
  ["cudaDevAttrUnifiedAddressing", 41],
]);

export function cudaDeviceAttributeValue(attr: number): number {
  switch (Math.trunc(attr)) {
    case 1: return 1024;
    case 2: return 1024;
    case 3: return 1024;
    case 4: return 64;
    case 5: return 2147483647;
    case 6: return 65535;
    case 7: return 65535;
    case 8: return 49152;
    case 9: return 65536;
    case 10: return 32;
    case 11: return 2147483647;
    case 12: return 65536;
    case 13: return 1000000;
    case 14: return 256;
    case 15: return 1;
    case 16: return 1;
    case 17: return 0;
    case 18: return 1;
    case 19: return 1;
    case 20: return 0;
    case 31: return 1;
    case 39: return 1024;
    case 40: return 1;
    case 41: return 1;
    default: return 0;
  }
}
