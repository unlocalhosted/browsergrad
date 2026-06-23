#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  collectCudaLiteContextDefines,
  collectKernelTemplateArguments,
  createKernelCompilationUnit,
  kernelDefinitionName,
  pruneCudaPreprocessorBranches,
} from "./cuda-lite-source-normalizer.mjs";

const scalarKernel = `
__global__ void add(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}`;

const vectorSibling = `
__global__ void add4(float *a, float *b, float *c, int N) {
  int idx = threadIdx.x * 4;
  float4 av = FLOAT4(a[idx]);
  c[idx] = av.x + b[idx];
}`;

{
  const source = `
#define ENABLE_FAST 1
#if ENABLE_FAST && defined(ENABLE_FAST)
__device__ float helper(float x) { return x + 1.0f; }
#else
__device__ broken_t helper(float x) { return x; }
#endif
#ifndef ENABLE_SLOW
__device__ float keep(float x) { return helper(x); }
#endif
__global__ void kernel(float *out) { out[0] = keep(1.0f); }`;
  const pruned = pruneCudaPreprocessorBranches(source);
  assert.match(pruned, /float helper\(float x\)/u);
  assert.match(pruned, /float keep\(float x\)/u);
  assert.doesNotMatch(pruned, /broken_t/u);
}

{
  const source = `
#if UNKNOWN_FEATURE
__device__ unknown_t maybe(float x) { return x; }
#else
__device__ float fallback(float x) { return x; }
#endif`;
  const pruned = pruneCudaPreprocessorBranches(source);
  assert.doesNotMatch(pruned, /unknown_t maybe/u);
  assert.match(pruned, /float fallback/u);
}

{
  const source = `
#if MODE
__device__ unknown_t maybe(float x) { return x; }
#else
__device__ float fallback(float x) { return x; }
#endif`;
  const pruned = pruneCudaPreprocessorBranches(source, new Map([["MODE", "runtime_flag()"]]));
  assert.match(pruned, /unknown_t maybe/u);
  assert.match(pruned, /float fallback/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: scalarKernel,
    siblingKernels: [vectorSibling],
    functionDeclarations: ["#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])"],
  });
  assert.match(source, /__global__ void add\(/u);
  assert.doesNotMatch(source, /__global__ void add4\(/u);
  assert.doesNotMatch(source, /FLOAT4/u);
}

{
  const parent = `
__global__ void parent(float *out) {
  child<<<1, 1>>>(out);
}`;
  const child = `
__global__ void child(float *out) {
  out[0] = 1.0f;
}`;
  const unused = `
__global__ void unused(float *out) {
  out[0] = broken_type();
}`;
  const source = createKernelCompilationUnit({
    kernel: parent,
    siblingKernels: [child, unused],
  });
  assert.match(source, /__global__ void child\(/u);
  assert.doesNotMatch(source, /__global__ void unused\(/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out) {
  out[0] = foo(1.0f);
}`,
    deviceFunctions: [
      {
        name: "foo",
        source: "__device__ float foo(float x) { return bar(x) + 1.0f; }",
      },
      {
        name: "bar",
        source: "__device__ float bar(float x) { return x * 2.0f; }",
      },
      {
        name: "baz",
        source: "__device__ float baz(float x) { return x * 4.0f; }",
      },
    ],
  });
  assert.match(source, /float foo/u);
  assert.match(source, /float bar/u);
  assert.doesNotMatch(source, /float baz/u);
}

{
  const source = `
template <uint sortDir>
__global__ void sample(uint *out) { out[0] = sortDir; }
void launch(uint *out) {
  sample<1U>
      <<<1, 1>>>(out);
  sample<0U>
      <<<1, 1>>>(out);
}`;
  const launches = collectKernelTemplateArguments(source);
  assert.deepEqual(launches.get("sample"), ["1U"]);
}

{
  const kernel = `
template <uint sortDir>
__global__ void ranks(uint *data, uint *out) {
  out[0] = binarySearchExclusive<sortDir>(data[0], data, 16, 8);
}`;
  const source = createKernelCompilationUnit({
    kernel,
    templateArgumentsByKernelName: new Map([["ranks", ["1U"]]]),
    deviceFunctions: [
      {
        name: "binarySearchExclusive",
        source: `
template <uint sortDir>
__device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride) {
  uint pos = 0;
  for (; stride > 0; stride >>= 1) {
    uint newPos = umin(pos + stride, L);
    if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val))) {
      pos = newPos;
    }
  }
  return pos;
}`,
      },
      {
        name: "unusedPointerHelper",
        source: "__device__ uint unusedPointerHelper(uint *data) { return data[0]; }",
      },
    ],
  });
  assert.match(source, /template <uint sortDir = 1>/u);
  assert.match(source, /uint binarySearchExclusive\(uint val, uint \*data/u);
  assert.match(source, /binarySearchExclusive\(data\[0\], data, 16, 8\)/u);
  assert.doesNotMatch(source, /unusedPointerHelper/u);
  assert.doesNotMatch(source, /binarySearchExclusive<sortDir>/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(int *out) {
  out[0] = templated_helper(7);
}`,
    deviceFunctions: [
      {
        name: "templated_helper",
        source: `
template <const int kStep = 8>
static __device__ __forceinline__ int templated_helper(int value) {
  return value + kStep;
}`,
      },
    ],
  });
  assert.match(source, /template <const int kStep = 8>/u);
  assert.match(source, /static __device__ __forceinline__ int templated_helper/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out) {
  out[0] = warp_reduce_sum<WARP_SIZE>(1.0f);
}`,
    definesByName: new Map([["WARP_SIZE", "32"]]),
    deviceFunctions: [
      {
        name: "warp_reduce_sum",
        source: `
template <const int kWarpSize = WARP_SIZE, typename T = float>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) val += mask;
  return val;
}`,
      },
    ],
  });
  assert.match(source, /template <const int kWarpSize = 32, typename T = float>/u);
  assert.match(source, /__device__ __forceinline__ float warp_reduce_sum\(float val\)/u);
  assert.match(source, /warp_reduce_sum\(1\.0f\)/u);
  assert.doesNotMatch(source, /\bT\s+(?:val|tmp|value)\b/u);
  assert.doesNotMatch(source, /warp_reduce_sum<WARP_SIZE>/u);
}

{
  const launchBoundsKernel = `
__global__ void __launch_bounds__(WARP_SIZE * kTiles)
    bounded(float *out, const float *in, int N) {
  int idx = threadIdx.x;
  if (idx < N) out[idx] = in[idx];
}`;
  assert.equal(kernelDefinitionName(launchBoundsKernel), "bounded");
  const source = createKernelCompilationUnit({
    kernel: launchBoundsKernel,
    definesByName: new Map([
      ["WARP_SIZE", "32"],
      ["N", "999"],
    ]),
  });
  assert.match(source, /#define WARP_SIZE 32/u);
  assert.doesNotMatch(source, /#define N 999/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
[] __global__ (float *data, int N) {
  int idx = threadIdx.x;
  if (idx < N) data[idx] = 0.0f;
}`,
    definesByName: new Map([
      ["N", "1024"],
      ["BLOCK", "256"],
    ]),
  });
  assert.doesNotMatch(source, /#define N 1024/u);
  assert.match(source, /#define BLOCK 256/u);
}

{
  const child = `
template <const int K, const bool UseBias>
__global__ void child(float *out) {
  __shared__ float tile[K];
  int idx = threadIdx.x;
  if (idx < K) out[idx] = UseBias ? tile[idx] + 1.0f : tile[idx];
}`;
  const parent = `
__global__ void parent(float *out) {
  child<16, true><<<1, 16>>>(out);
}`;
  const launches = collectKernelTemplateArguments(`${child}\n${parent}`);
  assert.deepEqual(launches.get("child"), ["16", "true"]);
  const source = createKernelCompilationUnit({
    kernel: parent,
    siblingKernels: [child],
    definesByName: new Map([
      ["K", "999"],
      ["UseBias", "0"],
    ]),
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /template <const int K = 16, const bool UseBias = 1>/u);
  assert.match(source, /child<<<1, 16>>>\(out\);/u);
  assert.doesNotMatch(source, /#define K 999/u);
  assert.doesNotMatch(source, /#define UseBias 0/u);
}

{
  const kernel = `
template <const int HeadDim, const int Tile, const int Stage>
__global__ void templated_kernel(float *out) {
  float tmp[Tile][Stage];
  out[0] = tmp[0][0] + HeadDim;
}`;
  const wrapper = `
template <const int HeadDim, const int Stage>
void launch_templated_kernel(float *out) {
  constexpr int Tile = (HeadDim < 128) ? 4 : 8;
  templated_kernel<HeadDim, Tile, Stage><<<1, 1>>>(out);
}
void host(float *out) {
  launch_templated_kernel<64, 2>(out);
}`;
  const launches = collectKernelTemplateArguments(`${kernel}\n${wrapper}`);
  assert.deepEqual(launches.get("templated_kernel"), ["64", "4", "2"]);
  const source = createKernelCompilationUnit({
    kernel,
    definesByName: new Map([
      ["HeadDim", "999"],
      ["Tile", "999"],
      ["Stage", "999"],
    ]),
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /template <const int HeadDim = 64, const int Tile = 4, const int Stage = 2>/u);
  assert.doesNotMatch(source, /#define HeadDim 999/u);
  assert.doesNotMatch(source, /#define Tile 999/u);
  assert.doesNotMatch(source, /#define Stage 999/u);
}

{
  const kernel = `
template <typename T, const int Width>
__global__ void typed_kernel(T *out) {
  T value = (T)Width;
  out[0] = value;
}`;
  const launcher = `
void host(float *out) {
  typed_kernel<float, 4><<<1, 1>>>(out);
}`;
  const launches = collectKernelTemplateArguments(`${kernel}\n${launcher}`);
  assert.deepEqual(launches.get("typed_kernel"), ["float", "4"]);
  const source = createKernelCompilationUnit({
    kernel,
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /template <typename T = float, const int Width = 4>/u);
  assert.match(source, /__global__ void typed_kernel\(float \*out\)/u);
  assert.match(source, /float value = \(float\)Width;/u);
}

{
  const kernel = `
template <class T>
__global__ void typed_from_wrapper(T *out) {
  out[0] = (T)1;
}`;
  const wrapper = `
template <class T>
void launch_typed_from_wrapper(T *out) {
  typed_from_wrapper<T><<<1, 1>>>(out);
}
void run(float *out) {
  launch_typed_from_wrapper<float>(out);
}`;
  const launches = collectKernelTemplateArguments(`${kernel}\n${wrapper}`);
  assert.deepEqual(launches.get("typed_from_wrapper"), ["float"]);
  const source = createKernelCompilationUnit({
    kernel,
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /__global__ void typed_from_wrapper\(float \*out\)/u);
  assert.match(source, /template <class T = float>/u);
  assert.doesNotMatch(source, /\(T\)1/u);
}

{
  const kernel = `
template <typename OutFloat, bool Atomic>
__global__ void inferred_kernel(OutFloat *out, const float *in, std::bool_constant<Atomic>) {
  if constexpr (!Atomic) {
    out[0] = (OutFloat)in[0];
  }
}`;
  const wrapper = `
void host(float *out, const float *in) {
  inferred_kernel<<<1, 1>>>(out, in, std::bool_constant<false>{});
}`;
  const launches = collectKernelTemplateArguments(`${kernel}\n${wrapper}`);
  assert.deepEqual(launches.get("inferred_kernel"), ["float", "false"]);
  const source = createKernelCompilationUnit({
    kernel,
    definesByName: new Map([["OutFloat", "999"]]),
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /template <typename OutFloat = float, bool Atomic = 0>/u);
  assert.match(source, /__global__ void inferred_kernel\(float \*out, const float \*in, bool __bg_bool_constant_Atomic\)/u);
  assert.match(source, /out\[0\] = \(float\)in\[0\];/u);
  assert.doesNotMatch(source, /#define OutFloat 999/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *in, float *out, int ld) {
  int row = threadIdx.x;
  out[0] = in[IDX2C(row, 0, ld)];
}`,
    functionDeclarations: [
      "#define IDX2C(i,j,ld) (((j)*(ld))+(i))",
      "#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])",
    ],
  });
  assert.match(source, /#define IDX2C/u);
  assert.doesNotMatch(source, /#define FLOAT4/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <class T>
__global__ void reduce(T *input, T *output) {
  T *scratch = SharedMemory<T>();
  scratch[threadIdx.x] = input[threadIdx.x];
  output[0] = scratch[0];
}`,
    templateArgumentsByKernelName: new Map([["reduce", ["float"]]]),
  });
  assert.match(source, /extern __shared__ float scratch\[\];/u);
  assert.doesNotMatch(source, /SharedMemory/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <class T>
__global__ void sharedmem_get_pointer(T *input, T *output) {
  SharedMemory<T> smem;
  T *scratch = smem.getPointer();
  scratch[threadIdx.x] = input[threadIdx.x];
  output[0] = scratch[0];
}`,
    templateArgumentsByKernelName: new Map([["sharedmem_get_pointer", ["float"]]]),
  });
  assert.match(source, /extern __shared__ float scratch\[\];/u);
  assert.doesNotMatch(source, /getPointer/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out, float phase) {
  float2 value = make_float2(0.0f, 0.0f);
  twiddle(value, phase);
  out[0] = value.x + value.y;
}`,
    deviceFunctions: [
      {
        name: "twiddle",
        source: "__device__ void twiddle(float2 &value, float phase) { __sincosf(phase, &value.y, &value.x); }",
      },
    ],
  });
  assert.match(source, /value\.y = sinf\(phase\); value\.x = cosf\(phase\)/u);
  assert.doesNotMatch(source, /__sincosf/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *in, float *out) {
  out[0] = GELU_OPS(in[0]);
}`,
    definesByName: new Map([
      ["GELU_OPS", "gelu_tanh_approximate"],
    ]),
    deviceFunctions: [
      {
        name: "gelu_tanh_approximate",
        source: "__device__ float gelu_tanh_approximate(float x) { return x * 0.5f; }",
      },
      {
        name: "unused_activation",
        source: "__device__ float unused_activation(float x) { return x; }",
      },
    ],
  });
  assert.match(source, /gelu_tanh_approximate\(float x\)/u);
  assert.match(source, /#define GELU_OPS gelu_tanh_approximate/u);
  assert.doesNotMatch(source, /unused_activation/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out, float value) {
  out[0] = blockReduce<warpReduceMax>(value, false, -INFINITY);
}`,
    deviceFunctions: [
      {
        name: "blockReduce",
        source: `
template<class Reducer>
__device__ inline float blockReduce(float value, bool final_sync, float out_of_bounds) {
  return value;
}`,
      },
    ],
  });
  assert.match(source, /out\[0\] = warpReduceMax\(value\);/u);
  assert.doesNotMatch(source, /__device__ inline float blockReduce/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(int *out, int n) {
  out[0] = div_ceil(n, 4);
}`,
    deviceFunctions: [
      {
        name: "div_ceil",
        source: "__device__ __host__ inline int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }",
      },
    ],
  });
  assert.match(source, /div_ceil\(n, 4\)/u);
  assert.doesNotMatch(source, /int div_ceil\(int a/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out) {
  Scratch[threadIdx.x] = out[threadIdx.x];
  __syncthreads();
  out[threadIdx.x] = Scratch[threadIdx.x];
}`,
    definesByName: new Map([["BLOCK_SIZE", "8"]]),
    sharedDeclarations: ["__shared__ float Scratch[BLOCK_SIZE];"],
  });
  assert.match(source, /__shared__ float Scratch\[BLOCK_SIZE\];/u);
  assert.match(source, /Scratch\[threadIdx\.x\] = out\[threadIdx\.x\]/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float4 *input, float *out) {
  float4 value = input[0];
  out[0] = vec_at(value, 2);
}`,
    deviceFunctions: [
      {
        name: "vec_at",
        source: "__device__ float vec_at(const float4& vec, int index) { return reinterpret_cast<const float*>(&vec)[index]; }",
      },
    ],
  });
  assert.match(source, /vec_at\(value, 2\)/u);
  assert.doesNotMatch(source, /reinterpret_cast/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void packed(float *out, const float *in) {
  int idx = threadIdx.x * x128::size;
  x128 value = load128cs(in + idx);
  store128cs(out + idx, value);
}`,
    definesByName: new Map([["x128", "float4"]]),
    deviceFunctions: [
      { name: "load128cs", source: "__device__ Packed128<ElementType> load128cs(const ElementType* address) { return Packed128<ElementType>{}; }" },
      { name: "store128cs", source: "__device__ void store128cs(ElementType* target, Packed128<ElementType> value) { }" },
    ],
  });
  assert.match(source, /#define x128 float4/u);
  assert.match(source, /reinterpret_cast<float4 \*>\(in \+ idx\)\[0\]/u);
  assert.match(source, /\(reinterpret_cast<float4 \*>\(out \+ idx\)\[0\] = value\)/u);
  assert.doesNotMatch(source, /__device__ Packed128/u);
  assert.doesNotMatch(source, /\bload128cs\s*\(/u);
  assert.doesNotMatch(source, /\bstore128cs\s*\(/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
typedef float floatX;
typedef Packed128<floatX> x128;
__global__ void packed_alias(floatX *out, const floatX *in) {
  constexpr int lanes = sizeof(float4) / sizeof(float);
  int idx = threadIdx.x * x128::size;
  x128 value = load128cs(in + idx);
  store128cs(out + idx, value);
}`,
    deviceFunctions: [
      { name: "load128cs", source: "__device__ Packed128<ElementType> load128cs(const ElementType* address) { return Packed128<ElementType>{}; }" },
      { name: "store128cs", source: "__device__ void store128cs(ElementType* target, Packed128<ElementType> value) { }" },
    ],
  });
  assert.match(source, /#define floatX float/u);
  assert.match(source, /#define x128 float4/u);
  assert.match(source, /reinterpret_cast<float4 \*>\(in \+ idx\)\[0\]/u);
  assert.doesNotMatch(source, /Packed128/u);
}

{
  const kernel = `
template <typename T, int Block>
__global__ void alias_template(T *out) {
  __shared__ T tile[Block / (sizeof(float4) / sizeof(float))];
  out[threadIdx.x] = (T)alignof(float4);
}`;
  const wrapper = `
using scalar_t = float;
template <typename T, int Block>
void launch_alias_template(T *out) {
  alias_template<T, Block><<<1, Block>>>(out);
}
void run(float *out) {
  launch_alias_template<scalar_t, 128>(out);
}`;
  const launches = collectKernelTemplateArguments(`${kernel}\n${wrapper}`);
  assert.deepEqual(launches.get("alias_template"), ["float", "128"]);
  const source = createKernelCompilationUnit({
    kernel,
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /template <typename T = float, int Block = 128>/u);
  assert.match(source, /__global__ void alias_template\(float \*out\)/u);
  assert.match(source, /__shared__ float tile\[Block \/ \(sizeof\(float4\) \/ sizeof\(float\)\)\];/u);
  assert.match(source, /\(float\)alignof\(float4\)/u);
}

{
  const carrierSource = `
template <typename T_, int Warps>
struct alignas(16) HgemvConfig {
  using T = T_;
  static constexpr int NumThreads = Warps * 32;
  static constexpr int BlockM = 16 * Warps;
};
using config = HgemvConfig<half, 4>;
template <typename Config_>
__global__ void carrier_kernel(typename Config_::T *out) {
  constexpr int block_m = Config_::BlockM;
  __shared__ typename Config_::T tile[Config_::NumThreads];
  out[threadIdx.x] = tile[threadIdx.x] + (typename Config_::T)block_m;
}
void host(half *out) {
  carrier_kernel<config><<<1, config::NumThreads>>>(out);
}`;
  const defines = collectCudaLiteContextDefines(carrierSource);
  assert.equal(defines.get("config::T"), "half");
  assert.equal(defines.get("config::NumThreads"), "128");
  assert.equal(defines.get("config::BlockM"), "64");
  const launches = collectKernelTemplateArguments(carrierSource);
  assert.deepEqual(launches.get("carrier_kernel"), ["config"]);
  const kernelStart = carrierSource.indexOf("template <typename Config_>");
  const kernelEnd = carrierSource.indexOf("\n}", carrierSource.indexOf("out[threadIdx.x]"));
  const kernel = kernelStart >= 0 && kernelEnd >= 0 ? carrierSource.slice(kernelStart, kernelEnd + 2) : undefined;
  assert.ok(kernel);
  const source = createKernelCompilationUnit({
    kernel,
    definesByName: defines,
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /__global__ void carrier_kernel\(half \*out\)/u);
  assert.match(source, /constexpr int block_m = 64;/u);
  assert.match(source, /__shared__ half tile\[128\];/u);
  assert.doesNotMatch(source, /Config_::/u);
  assert.doesNotMatch(source, /config::/u);
}

{
  const packedSource = `
template<class ElementType>
struct alignas(16) Packed128 {
  static constexpr const int size = sizeof(int4) / sizeof(ElementType);
  ElementType payload[size];
};
typedef Packed128<float> f128;
__global__ void packed_size(float *out) {
  float accum[f128::size];
  out[threadIdx.x] = accum[0] + f128::size;
}`;
  const defines = collectCudaLiteContextDefines(packedSource);
  assert.equal(defines.get("f128"), "float4");
  assert.equal(defines.get("f128::size"), "4");
  const source = createKernelCompilationUnit({
    kernel: /__global__ void packed_size[\s\S]*$/u.exec(packedSource)?.[0],
    definesByName: defines,
  });
  assert.match(source, /float accum\[4\];/u);
  assert.match(source, /out\[threadIdx\.x\] = accum\[0\] \+ 4;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void async_copy(float *out, float *in) {
  __shared__ float tile[4];
  CP_ASYNC_CG(tile, in, 16);
  CP_ASYNC_COMMIT_GROUP();
  CP_ASYNC_WAIT_GROUP(0);
  out[threadIdx.x] = tile[threadIdx.x];
}`,
    functionDeclarations: [
      '#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\\n" ::"r"(dst), "l"(src), "n"(bytes))',
      '#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\\n" ::)',
      '#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\\n" ::"n"(n))',
    ],
  });
  assert.match(source, /#define CP_ASYNC_CG\(dst, src, bytes\) CP_ASYNC_CG\(dst, src, bytes\)/u);
  assert.match(source, /#define CP_ASYNC_COMMIT_GROUP\(\) CP_ASYNC_COMMIT_GROUP\(\)/u);
  assert.match(source, /#define CP_ASYNC_WAIT_GROUP\(n\) CP_ASYNC_WAIT_GROUP\(n\)/u);
  assert.doesNotMatch(source, /asm volatile/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void local_constexpr(float *out) {
  constexpr int BM = 32;
  __shared__ float tile[BM];
  out[threadIdx.x] = tile[threadIdx.x];
}`,
  });
  assert.doesNotMatch(source, /#define BM 32/u);
  assert.match(source, /constexpr int BM = 32;/u);
}

console.log("cuda-lite source normalizer tests ok");
