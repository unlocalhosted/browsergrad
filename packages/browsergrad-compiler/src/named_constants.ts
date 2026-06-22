import type { CudaLiteScalarType } from "./types.js";

export interface CudaNamedConstant {
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly value: number;
  readonly wgsl: string;
}

export const CUDA_NAMED_CONSTANTS = new Map<string, CudaNamedConstant>([
  ["INFINITY", floatConstant(Number.POSITIVE_INFINITY, "bg_f32_inf()")],
  ["NAN", floatConstant(Number.NaN, "bg_f32_nan()")],
  ["FLT_MAX", floatConstant(3.4028234663852886e38, "3.4028234663852886e38")],
  ["M_PI", floatConstant(Math.PI, "3.141592653589793")],
  ["cudaEventDefault", uintConstant(0)],
  ["cudaEventDisableTiming", uintConstant(2)],
  ["cudaEventInterprocess", uintConstant(4)],
  ["cudaEventBlockingSync", uintConstant(1)],
  ["cudaStreamDefault", uintConstant(0)],
  ["cudaStreamNonBlocking", uintConstant(1)],
  ["cudaMemcpyDeviceToDevice", uintConstant(3)],
  ["cudaMemcpyDefault", uintConstant(4)],
]);

function floatConstant(value: number, wgsl: string): CudaNamedConstant {
  return { valueType: "float", value, wgsl };
}

function uintConstant(value: number): CudaNamedConstant {
  return { valueType: "uint", value, wgsl: `${value}u` };
}
