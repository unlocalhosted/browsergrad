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
  ["M_E", floatConstant(Math.E, "2.718281828459045")],
  ["M_LOG2E", floatConstant(Math.LOG2E, "1.4426950408889634")],
  ["M_LOG10E", floatConstant(Math.LOG10E, "0.4342944819032518")],
  ["M_LN2", floatConstant(Math.LN2, "0.6931471805599453")],
  ["M_LN10", floatConstant(Math.LN10, "2.302585092994046")],
  ["M_PI", floatConstant(Math.PI, "3.141592653589793")],
  ["M_PI_2", floatConstant(Math.PI / 2, "1.5707963267948966")],
  ["M_PI_4", floatConstant(Math.PI / 4, "0.7853981633974483")],
  ["M_1_PI", floatConstant(1 / Math.PI, "0.3183098861837907")],
  ["M_2_PI", floatConstant(2 / Math.PI, "0.6366197723675814")],
  ["M_2_SQRTPI", floatConstant(2 / Math.sqrt(Math.PI), "1.1283791670955126")],
  ["M_SQRT2", floatConstant(Math.SQRT2, "1.4142135623730951")],
  ["M_SQRT1_2", floatConstant(Math.SQRT1_2, "0.7071067811865476")],
  ["warpSize", intConstant(32)],
  ["WARP_SIZE", intConstant(32)],
  ["NULL", { valueType: "voidptr", value: 0, wgsl: "0u" }],
  ["cudaEventDefault", uintConstant(0)],
  ["cudaEventDisableTiming", uintConstant(2)],
  ["cudaEventInterprocess", uintConstant(4)],
  ["cudaEventBlockingSync", uintConstant(1)],
  ["cudaStreamDefault", uintConstant(0)],
  ["cudaStreamNonBlocking", uintConstant(1)],
  ["cudaMemcpyDeviceToDevice", uintConstant(3)],
  ["cudaMemcpyDefault", uintConstant(4)],
  ["float2::size", intConstant(2)],
  ["float3::size", intConstant(3)],
  ["float4::size", intConstant(4)],
  ["int2::size", intConstant(2)],
  ["int3::size", intConstant(3)],
  ["int4::size", intConstant(4)],
  ["uint2::size", intConstant(2)],
  ["uint3::size", intConstant(3)],
  ["uint4::size", intConstant(4)],
  ["half2::size", intConstant(2)],
]);

function floatConstant(value: number, wgsl: string): CudaNamedConstant {
  return { valueType: "float", value, wgsl };
}

function intConstant(value: number): CudaNamedConstant {
  return { valueType: "int", value, wgsl: `${value}` };
}

function uintConstant(value: number): CudaNamedConstant {
  return { valueType: "uint", value, wgsl: `${value}u` };
}
