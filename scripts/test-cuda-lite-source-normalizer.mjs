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
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(uint *out) {
  out[0] = numErrors;
}`,
    deviceGlobalDeclarations: ["__device__ static unsigned int numErrors = 3;"],
  });
  assert.match(source, /__device__ static unsigned int numErrors = 3;/u);
  assert.match(source, /out\[0\] = numErrors/u);
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
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void packed_shared(bf16 *out, const bf16 *input, int C) {
  extern __shared__ char params[];
  x128 *tile = reinterpret_cast<x128*>(params);
  x128 *tail = reinterpret_cast<x128*>(params + C * sizeof(floatX));
  x128 value = load128(input);
  tile[threadIdx.x] = value;
  const x128 cached = tile[threadIdx.x];
  x128 zero = x128::zeros();
  tail[0] = zero;
  out[threadIdx.x] = cached[0];
}`,
    definesByName: new Map([["floatX", "bf16"]]),
  });
  assert.match(source, /extern __shared__ bf16 params\[\];/u);
  assert.match(source, /bf16\* tile = params;/u);
  assert.match(source, /bf16\* tail = params \+ \(\(C \* 2\) \/ 2\);/u);
  assert.match(source, /tile\[\(\(threadIdx\.x\) \* 8\) \+ 0\] = value\[0\]/u);
  assert.match(source, /bf16 cached\[8\]; cached\[0\] = tile\[\(\(threadIdx\.x\) \* 8\) \+ 0\]/u);
  assert.match(source, /bf16 zero\[8\]; zero\[0\] = 0\.0f/u);
  assert.doesNotMatch(source, /\bx128\b/u);
}

{
  const source = `
using fn_ptr = void(*)(float*);
template<fn_ptr fn>
__global__ void symbol_kernel(float* out) { fn(out); }
template<fn_ptr fn>
void launch_symbol_kernel(float* out) {
  symbol_kernel<fn><<<1, 1>>>(out);
}
__device__ void concrete_symbol(float* out) { out[0] = 1.0f; }
void host(float* out) { launch_symbol_kernel<concrete_symbol>(out); }`;
  const launches = collectKernelTemplateArguments(source);
  assert.deepEqual(launches.get("symbol_kernel"), ["concrete_symbol"]);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void std_math(float *out) {
  float x = std::numeric_limits<float>::infinity();
  out[0] = std::isinf(x) ? -x : x;
}`,
  });
  assert.match(source, /float x = INFINITY;/u);
  assert.match(source, /out\[0\] = isinf\(x\) \? -x : x;/u);
  assert.doesNotMatch(source, /std::/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void scalar_pack_load(bf16 *out, const bf16 *in, int i) {
  bf16 *slot = out;
  slot[i / 8] = load128(in + i);
}`,
  });
  assert.match(source, /slot\[\(\(i \/ 8\) \* 8\) \+ 0\] = in\[\(i\) \+ 0\]/u);
  assert.match(source, /slot\[\(\(i \/ 8\) \* 8\) \+ 7\] = in\[\(i\) \+ 7\]/u);
  assert.doesNotMatch(source, /\bload128\s*\(/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
namespace cg = cooperative_groups;
__device__ float2 merge_pair(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__global__ void vector_reduce(float2 *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  float2 value = make_float2(out[0].x, out[0].y);
  float2 total = cg::reduce(warp, value, merge_pair);
  out[0] = total;
}`,
  });
  assert.match(source, /float2 total = value;/u);
  assert.match(source, /__bg_for___bg_cg_reduce_offset_0_0 = 16/u);
  assert.match(source, /total = merge_pair\(total, make_float2\(__shfl_xor_sync\(0xffffffff, total\.x/u);
  assert.doesNotMatch(source, /float2 total = cg::reduce/u);
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
  assert.match(source, /float value = \(float\)4;/u);
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
  assert.match(source, /__global__ void inferred_kernel\(float \*out, const float \*in, bool __bg_bool_constant_0\)/u);
  assert.match(source, /out\[0\] = \(float\)in\[0\];/u);
  assert.doesNotMatch(source, /#define OutFloat 999/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void helper_reduce(int *out) {
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  out[0] = cg_reduce_n(1, tile);
}`,
    deviceFunctions: [
      {
        name: "cg_reduce_n",
        source: `template <typename T, typename Group> __device__ T cg_reduce_n(T in, Group &threads) {
  return cg::reduce(threads, in, cg::plus<T>());
}`,
      },
    ],
  });
  assert.match(source, /__device__ int cg_reduce_n\(int in, cooperative_groups::thread_group threads\)/u);
  assert.doesNotMatch(source, /float &threads/u);
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
  assert.match(source, /__shared__ float Scratch\[8\];/u);
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
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void restrict_packed(float* __restrict__ out, const float* __restrict__ in) {
  int idx = threadIdx.x * 4;
  float4 value = load128(in + idx);
  store128(out + idx, value);
}`,
    deviceFunctions: [
      { name: "load128", source: "__device__ Packed128<ElementType> load128(const ElementType* address) { return Packed128<ElementType>{}; }" },
      { name: "store128", source: "__device__ void store128(ElementType* target, Packed128<ElementType> value) { }" },
    ],
  });
  assert.match(source, /reinterpret_cast<float4 \*>\(in \+ idx\)\[0\]/u);
  assert.match(source, /\(reinterpret_cast<float4 \*>\(out \+ idx\)\[0\] = value\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void nested_packed(float* out, const float* in) {
  int idx = threadIdx.x * 4;
  store128(out + idx, load128(in + idx));
}`,
    deviceFunctions: [
      { name: "load128", source: "__device__ Packed128<ElementType> load128(const ElementType* address) { return Packed128<ElementType>{}; }" },
      { name: "store128", source: "__device__ void store128(ElementType* target, Packed128<ElementType> value) { }" },
    ],
  });
  assert.match(source, /\(reinterpret_cast<float4 \*>\(out \+ idx\)\[0\] = reinterpret_cast<float4 \*>\(in \+ idx\)\[0\]\)/u);
  assert.doesNotMatch(source, /\b(?:load128|store128)\s*\(/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
typedef float floatX;
typedef Packed128<floatX> x128;
__global__ void packed_static(floatX *out) {
  x128 zero = x128::zeros();
  x128 one = x128::constant(1.0f);
  out[threadIdx.x] = zero[threadIdx.x] + one[threadIdx.x];
}`,
    deviceFunctions: [
      { name: "zeros", source: "__device__ static Packed128 zeros() { return constant(0); }" },
      { name: "constant", source: "__device__ static Packed128 constant(ElementType value) { Packed128 result; return result; }" },
    ],
  });
  assert.match(source, /x128 zero = make_float4\(0\.0f, 0\.0f, 0\.0f, 0\.0f\);/u);
  assert.match(source, /x128 one = make_float4\(1\.0f, 1\.0f, 1\.0f, 1\.0f\);/u);
  assert.doesNotMatch(source, /__device__ static Packed128/u);
  assert.doesNotMatch(source, /::zeros|::constant/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
#define SET_PAIR(dst, src) (dst).x = (src).x + 1.0f; (dst).y = (src).y + 2.0f;
__global__ void statement_macro(float2 *out, float2 *in) {
  float2 x = in[threadIdx.x];
  float2 y;
  SET_PAIR(y, x);
  out[threadIdx.x] = y;
}`,
  });
  assert.match(source, /\(y\)\.x = \(x\)\.x \+ 1\.0f; \(y\)\.y = \(x\)\.y \+ 2\.0f;/u);
  assert.doesNotMatch(source, /\bSET_PAIR\s*\(y, x\)/u);
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
  assert.match(source, /__shared__ float tile\[128 \/ \(sizeof\(float4\) \/ sizeof\(float\)\)\];/u);
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
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T = float>
__global__ void vector_carrier_kernel(typename vec4<T>::Type *out, typename vec3<T>::Type value) {
  out[threadIdx.x] = { value.x, value.y, value.z, 1.0f };
}`,
  });
  assert.match(source, /__global__ void vector_carrier_kernel\(float4 \*out, float3 value\)/u);
  assert.doesNotMatch(source, /typename vec/u);
}

{
  const wrapperSource = `
typedef float floatX;
template<class T>
__global__ void typed_kernel(float *out, const T *values) {
  out[threadIdx.x] = (float)values[threadIdx.x];
}
template<typename T>
void launch_typed(float *out, const T *values) {
  typed_kernel<<<1, 256>>>(out, values);
}
void host(float *out, const floatX *values) {
  launch_typed(out, values);
}`;
  const defines = new Map([["floatX", "float"]]);
  const launches = collectKernelTemplateArguments(wrapperSource);
  assert.deepEqual(launches.get("typed_kernel"), ["float"]);
  const kernel = /template<class T>[\s\S]*?^\}/mu.exec(wrapperSource)?.[0];
  assert.ok(kernel);
  const source = createKernelCompilationUnit({
    kernel,
    definesByName: defines,
    templateArgumentsByKernelName: launches,
  });
  assert.match(source, /__global__ void typed_kernel\(float \*out, const float \*values\)/u);
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

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void packed_local_alias(const floatX *input, floatX *output) {
  using x128 = Packed128<floatX>;
  int idx = threadIdx.x * x128::size;
  x128 value = load128cs(input + idx);
  store128cs(output + idx, value);
}`,
    definesByName: new Map([["floatX", "float"]]),
  });
  assert.match(source, /#define x128 float4/u);
  assert.match(source, /reinterpret_cast<float4 \*>\(input \+ idx\)\[0\]/u);
  assert.doesNotMatch(source, /using x128/u);
  assert.doesNotMatch(source, /Packed128/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void macro_calls_helper(float *out) {
  BG_APPLY(out[threadIdx.x]);
}`,
    functionDeclarations: ["#define BG_APPLY(value) helper(value)"],
    deviceFunctions: [
      { name: "helper", source: "__device__ void helper(float &value) { value = value + 1.0f; }" },
    ],
  });
  assert.match(source, /#define BG_APPLY\(value\) helper\(value\)/u);
  assert.match(source, /__device__ void helper/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void line_continuation_and_shadow(float *out, int n) {
  int idx = \\
    (blockIdx.x * blockDim.x)
    + threadIdx.x;
  int block_size = blockDim.x;
  if (idx < n) { out[idx] = (float)block_size; }
}`,
    definesByName: new Map([["block_size", "512"]]),
  });
  assert.doesNotMatch(source, /#define block_size/u);
  assert.doesNotMatch(source, /\\/u);
  assert.match(source, /int idx =\s+\(blockIdx\.x \* blockDim\.x\)/u);
  assert.match(source, /int block_size = blockDim\.x/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <const int NUM_THREADS = 256>
__global__ void numeric_define_const_dim(float *out, int N) {
  const int WARP_NUM = NUM_THREADS / WARP_SIZE;
  __shared__ float scratch[WARP_NUM];
  int WARP_SIZE = threadIdx.x;
  if (WARP_SIZE < N) out[WARP_SIZE] = scratch[0];
}`,
    definesByName: new Map([
      ["WARP_SIZE", "32"],
      ["N", "999"],
    ]),
  });
  assert.doesNotMatch(source, /#define WARP_SIZE 32/u);
  assert.doesNotMatch(source, /#define N 999/u);
  assert.match(source, /const int WARP_NUM = 256 \/ 32;/u);
  assert.match(source, /int WARP_SIZE = threadIdx\.x;/u);
  assert.match(source, /if \(WARP_SIZE < N\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void lambda_inline(float *out, int n) {
  auto apply = [&](int j){
    out[j] = out[j] + 1.0f;
  };
  int idx = threadIdx.x;
  if (idx < n) { apply(idx); }
}`,
  });
  assert.doesNotMatch(source, /\bauto\s+apply/u);
  assert.doesNotMatch(source, /\bapply\s*\(/u);
  assert.match(source, /\{\s*int j = idx;\s*out\[j\] = out\[j\] \+ 1\.0f;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void templated_scalar_helper(float *out, int n) {
  int blocks = ceil_div(n + 1, 4);
  out[threadIdx.x] = (float)blocks;
}`,
    deviceFunctions: [
      { name: "ceil_div", source: "template<class T> __device__ T ceil_div(T dividend, T divisor) { return (dividend + divisor - 1) / divisor; }" },
    ],
  });
  assert.match(source, /template<class T = int>/u);
  assert.match(source, /__device__ int ceil_div\(int dividend, int divisor\)/u);
  assert.doesNotMatch(source, /\bT\s+dividend/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename Value, typename Layout>
__global__ void canonical_type_kernel(Value *out, const Value *input, Layout layout) {
  int idx = threadIdx.x;
  out[idx] = input[idx] + (Value)1;
}`,
  });
  assert.match(source, /template <typename Value = float, typename Layout>/u);
  assert.match(source, /__global__ void canonical_type_kernel\(float \*out, const float \*input, Layout layout\)/u);
  assert.doesNotMatch(source, /Layout = float/u);
}

{
  const source = createKernelCompilationUnit({
    functionDeclarations: [
      "typedef unsigned int TColor;",
      "typedef unsigned char (*pointFunction_t)(unsigned char, float);",
    ],
    kernel: `
__global__ void alias_intake_kernel(TColor *out, pointFunction_t fn) {
  out[threadIdx.x] = (TColor)threadIdx.x;
}`,
  });
  assert.match(source, /__global__ void alias_intake_kernel\(uint \*out, uint fn\)/u);
  assert.match(source, /out\[threadIdx\.x\] = \(uint\)threadIdx\.x;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void inferred_scalar_template(uint *out, unsigned short *scratch) {
  unsigned int count = threadIdx.x;
  sink(out, scratch, count);
}`,
    deviceFunctions: [
      {
        name: "sink",
        source: `
template <class S, class T>
__device__ void sink(T *out, unsigned short *scratch, const S count) {
  out[0] = (T)(count + scratch[0]);
}`,
      },
    ],
  });
  assert.match(source, /template <class S = uint, class T = uint>/u);
  assert.match(source, /__device__ void sink\(uint \*out, unsigned short \*scratch, const uint count\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void semantic_min_kernel(float *out) {
  out[0] = min(1.0f, 2.0f);
}`,
    deviceFunctions: [
      {
        name: "min",
        source: `
__host__ __device__
T
min(const T &lhs, const T &rhs) {
  return (lhs < rhs) ? lhs : rhs;
}`,
      },
    ],
  });
  assert.match(source, /out\[0\] = min\(1\.0f, 2\.0f\);/u);
  assert.doesNotMatch(source, /\bT\s+min/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void for_scope_shadow(uint *out, uint stride) {
  for (uint stride = 4; stride > 0; stride >>= 1) {
    out[stride] = stride;
  }
  out[0] = stride;
}`,
  });
  assert.match(source, /for \(uint __bg_for_stride_0 = 4;__bg_for_stride_0 > 0;__bg_for_stride_0 >>= 1\)/u);
  assert.match(source, /out\[__bg_for_stride_0\] = __bg_for_stride_0;/u);
  assert.match(source, /out\[0\] = stride;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void side_effect_canonical(unsigned int *data, int *out, int max_depth, int depth) {
  unsigned int *lptr = data;
  unsigned int *rptr = data + 3;
  unsigned int rval = *rptr;
  unsigned int lval = *lptr;
  if (++depth >= max_depth) {
    out[0] = depth;
  }
  *lptr++ = rval;
  *rptr-- = lval;
}`,
  });
  assert.match(source, /depth\+\+;\s*if \(depth >= max_depth\)/u);
  assert.match(source, /\*lptr = rval;\s*lptr\+\+;/u);
  assert.match(source, /\*rptr = lval;\s*rptr--;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void deref_prefix(int *ptr, int *out) {
  if (--(*ptr) == 0) {
    out[0] = *ptr;
  }
}`,
  });
  assert.match(source, /\*ptr -= 1;\s*if \(\*ptr == 0\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
struct __align__(8) MD {
  float m;
  float d;
};

__device__ MD reduce_md(MD value, MD other) {
  bool pick = value.m > other.m;
  MD bigger = pick ? value : other;
  MD smaller = pick ? other : value;
  MD result;
  result.d = bigger.d + smaller.d * __expf(smaller.m - bigger.m);
  result.m = bigger.m;
  return result;
}

__global__ void pod_record(float *out) {
  MD value = {out[0], 1.0f};
  MD other = MD{-1.0f, 2.0f};
  __shared__ MD shared[1];
  shared[0] = reduce_md(value, other);
  out[0] = shared[0].m + shared[0].d;
}`,
  });
  assert.doesNotMatch(source, /\bstruct\s+__align__\s*\([^)]*\)\s+MD/u);
  assert.doesNotMatch(source, /\bMD\b/u);
  assert.match(source, /__device__ float2 reduce_md\(float2 value, float2 other\)/u);
  assert.match(source, /float2 bigger = pick \? value : other;/u);
  assert.match(source, /result\.y = bigger\.y \+ smaller\.y \* __expf\(smaller\.x - bigger\.x\);/u);
  assert.match(source, /__shared__ float2 shared\[1\];/u);
  assert.match(source, /make_float2\(-1\.0f, 2\.0f\)/u);
  assert.match(source, /out\[0\] = shared\[0\]\.x \+ shared\[0\]\.y;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
struct Ray {
  float3 o;
  float3 d;
};

__device__ int intersectBox(Ray r, float3 boxmin, float *tnear) {
  float3 invR = make_float3(1.0f) / r.d;
  float3 tbot = invR * (boxmin - r.o);
  *tnear = tbot.x;
  return tbot.x > 0.0f;
}

__global__ void ray_record(float *out) {
  Ray eyeRay;
  eyeRay.o = make_float3(out[0], 0.0f, 0.0f);
  eyeRay.d = make_float3(1.0f, 0.0f, 0.0f);
  float tnear;
  int hit = intersectBox(eyeRay, eyeRay.d, &tnear);
  out[0] = tnear + hit;
}`,
    recordDeclarations: [
      `struct Ray {
  float3 o;
  float3 d;
};`,
    ],
    deviceFunctions: [
      {
        name: "intersectBox",
        source: `__device__ int intersectBox(Ray r, float3 boxmin, float *tnear) {
  float3 invR = make_float3(1.0f) / r.d;
  float3 tbot = invR * (boxmin - r.o);
  *tnear = tbot.x;
  return tbot.x > 0.0f;
}`,
      },
    ],
  });
  assert.doesNotMatch(source, /\bstruct\s+Ray\b/u);
  assert.doesNotMatch(source, /\bRay\b/u);
  assert.match(source, /__device__ int intersectBox\(float3 r__o, float3 r__d, float3 boxmin, float \*tnear\)/u);
  assert.match(source, /float3 invR = make_float3\(1\.0f\) \/ r__d;/u);
  assert.match(source, /float3 eyeRay__o;\s*float3 eyeRay__d;/u);
  assert.match(source, /eyeRay__o = make_float3/u);
  assert.match(source, /int hit = intersectBox\(eyeRay__o, eyeRay__d, eyeRay__d, &tnear\);/u);
}

{
  const enumTemplateSource = `
enum Mode {
  MODE_A = 0,
  MODE_B = 1,
};
template <int MODE>
__device__ float pick(float x, float y) {
  return MODE == MODE_A ? x : y;
}
__global__ void enum_template(float *out) {
  out[0] = pick<MODE_B>(1.0f, 2.0f);
}`;
  const source = createKernelCompilationUnit({
    kernel: /__global__ void enum_template[\s\S]*$/u.exec(enumTemplateSource)?.[0],
    definesByName: collectCudaLiteContextDefines(enumTemplateSource),
    deviceFunctions: [
      {
        name: "pick",
        source: `template <int MODE>
__device__ float pick(float x, float y) {
  return MODE == MODE_A ? x : y;
}`,
      },
    ],
  });
  assert.match(source, /#define MODE_A 0/u);
  assert.match(source, /#define MODE_B 1/u);
  assert.match(source, /template <int MODE = 1>/u);
  assert.match(source, /return 1 == 0 \? x : y;/u);
  assert.doesNotMatch(source, /\benum\s+Mode\b/u);
  assert.match(source, /out\[0\] = pick\(1\.0f, 2\.0f\);/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
typedef struct {
  float4 m[3];
} float3x4;
__constant__ float3x4 c_invViewMatrix;

__device__ float4 mul(const float3x4 &M, const float4 &v) {
  float4 r;
  r.x = dot(v, M.m[0]);
  r.y = dot(v, M.m[1]);
  r.z = dot(v, M.m[2]);
  r.w = 1.0f;
  return r;
}

__global__ void matrix_record(float4 *out) {
  out[0] = mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
}`,
    recordDeclarations: [
      `typedef struct {
  float4 m[3];
} float3x4;`,
    ],
    constantDeclarations: ["__constant__ float3x4 c_invViewMatrix;"],
    deviceFunctions: [
      {
        name: "mul",
        source: `__device__ float4 mul(const float3x4 &M, const float4 &v) {
  float4 r;
  r.x = dot(v, M.m[0]);
  r.y = dot(v, M.m[1]);
  r.z = dot(v, M.m[2]);
  r.w = 1.0f;
  return r;
}`,
      },
    ],
  });
  assert.doesNotMatch(source, /\bfloat3x4\b/u);
  assert.match(source, /__constant__ float4 c_invViewMatrix\[3\];/u);
  assert.match(source, /__device__ float4 mul\(const float4 \*M__m, const float4 &v\)/u);
  assert.match(source, /r\.x = dot\(v, M__m\[0\]\);/u);
  assert.match(source, /out\[0\] = mul\(c_invViewMatrix, make_float4/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
typedef half floatX;
typedef Packed128<floatX> x128;
__global__ void packed_half(floatX *out, const floatX *inp) {
  int idx = threadIdx.x * x128::size;
  x128 packed_out;
  x128 packed_inp = load128cs(inp + idx);
  for (int k = 0; k < packed_inp.size; ++k) {
    packed_out[k] = (floatX)((float)packed_inp[k] + 1.0f);
  }
  store128(out + idx, packed_out);
}`,
    definesByName: new Map([
      ["floatX", "half"],
      ["x128", "__bg_pack128_half8"],
      ["x128::size", "8"],
    ]),
  });
  assert.doesNotMatch(source, /\bx128\b/u);
  assert.match(source, /half packed_out\[8\];/u);
  assert.match(source, /half packed_inp\[8\]; packed_inp\[0\] = inp\[\(idx\) \+ 0\]/u);
  assert.match(source, /packed_inp\[7\] = inp\[\(idx\) \+ 7\]/u);
  assert.match(source, /out\[\(idx\) \+ 7\] = packed_out\[7\]/u);
  assert.match(source, /for \(int k = 0; k < 8; \+\+k\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void async_pipeline(float* out, const float* in) {
  __shared__ alignas(alignof(float4)) float tile[16];
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  pipe.producer_acquire();
  cuda::memcpy_async(&tile[threadIdx.x * 4], &in[threadIdx.x * 4], shape4, pipe);
  pipe.producer_commit();
  pipe.consumer_wait();
  __syncthreads();
  out[threadIdx.x] = tile[threadIdx.x];
  pipe.consumer_release();
}`,
  });
  assert.match(source, /const int shape4 = 16;/u);
  assert.doesNotMatch(source, /cuda::pipeline/u);
  assert.doesNotMatch(source, /cuda::memcpy_async/u);
  assert.match(source, /CP_ASYNC_CG\(&tile\[threadIdx\.x \* 4\], &in\[threadIdx\.x \* 4\], 16\);/u);
  assert.match(source, /CP_ASYNC_COMMIT_GROUP\(\);/u);
  assert.match(source, /CP_ASYNC_WAIT_GROUP\(0\);/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T = float, bool UseAuxBuffer>
__global__ void bool_template_carrier(float* out, const float* in, bool __bg_bool_constant_UseAuxBuffer) {
  if constexpr (!UseAuxBuffer) {
    out[threadIdx.x] = in[threadIdx.x];
  }
}`,
  });
  assert.match(source, /if constexpr \(!__bg_bool_constant_UseAuxBuffer\)/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: ["struct Ray { float length; float3 dir; };"],
    kernel: `
__global__ void pod_value_param(float* out, const Ray ray) {
  out[0] = ray.length + ray.dir.x;
}`,
  });
  assert.doesNotMatch(source, /\bRay\b/u);
  assert.match(source, /float ray__length/u);
  assert.match(source, /float3 ray__dir/u);
  assert.match(source, /out\[0\] = ray__length \+ ray__dir\.x;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void extent_param(uint* out, cudaExtent volumeSize) {
  out[0] = volumeSize.width + volumeSize.height + volumeSize.depth;
}`,
  });
  assert.doesNotMatch(source, /\bcudaExtent\b/u);
  assert.match(source, /uint3 volumeSize/u);
  assert.match(source, /volumeSize\.x \+ volumeSize\.y \+ volumeSize\.z/u);
}

{
  const source = createKernelCompilationUnit({
    definesByName: new Map([["KERNEL_PARAM_LIMIT", "(1024)"]]),
    recordDeclarations: ["typedef struct { int param[KERNEL_PARAM_LIMIT]; } param_t;"],
    kernel: `
__global__ void macro_array_record(__grid_constant__ const param_t p, int* out) {
  out[0] = p.param[threadIdx.x];
}`,
  });
  assert.doesNotMatch(source, /\bparam_t\b/u);
  assert.match(source, /const int \*p__param/u);
  assert.match(source, /out\[0\] = p__param\[threadIdx\.x\];/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
typedef float vec2[2];
__global__ void array_alias(vec2* points, float* out) {
  out[0] = points[threadIdx.x].x + points[threadIdx.x].y;
}`,
  });
  assert.doesNotMatch(source.replace(/^#.*$/gmu, ""), /\bvec2\b/u);
  assert.match(source, /float2\* points/u);
  assert.match(source, /points\[threadIdx\.x\]\.x \+ points\[threadIdx\.x\]\.y/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void vector_length(float3* input, float* out) {
  float3 rel = input[threadIdx.x] - make_float3(1.0f, 2.0f, 3.0f);
  out[threadIdx.x] = length(rel);
}`,
  });
  assert.doesNotMatch(source, /\blength\s*\(/u);
  assert.match(source, /sqrtf\(rel\.x \* rel\.x \+ rel\.y \* rel\.y \+ rel\.z \* rel\.z\)/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: ["struct Vertex { float4 position; float2 uv; };"],
    kernel: `
__global__ void record_storage_param(Vertex* vertices, const Vertex* src) {
  int i = threadIdx.x;
  vertices[i].position.x = src[i].position.x + vertices[i].uv.y;
}`,
  });
  assert.doesNotMatch(source, /\bVertex\b/u);
  assert.match(source, /float4 \*vertices__position/u);
  assert.match(source, /float2 \*vertices__uv/u);
  assert.match(source, /const float4 \*src__position/u);
  assert.match(source, /const float2 \*src__uv/u);
  assert.match(source, /vertices__position\[i\]\.x = src__position\[i\]\.x \+ vertices__uv\[i\]\.y;/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: ["struct Vertex { XMFLOAT3 position; XMFLOAT4 color; };"],
    kernel: `
__global__ void external_vector_record(Vertex* vertices, unsigned int width) {
  int i = threadIdx.x;
  vertices[i].position.x = 1.0f;
  vertices[i].color.w = (float)width;
}`,
  });
  assert.doesNotMatch(source, /\bVertex\b/u);
  assert.doesNotMatch(source, /\bXMFLOAT[34]\b/u);
  assert.match(source, /float3 \*vertices__position/u);
  assert.match(source, /float4 \*vertices__color/u);
  assert.match(source, /vertices__position\[i\]\.x = 1\.0f;/u);
  assert.match(source, /vertices__color\[i\]\.w = \(float\)width;/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: ["struct SimParams { float3 gravity; uint count; };"],
    constantDeclarations: ["__constant__ SimParams cudaParams;"],
    definesByName: new Map([["uint", "uint"]]),
    deviceFunctions: [
      {
        name: "load_gravity",
        source: `__device__ float load_gravity() {
  return cudaParams.gravity.x + (float)cudaParams.count;
}`,
      },
    ],
    kernel: `
__global__ void constant_record(float* out) {
  out[0] = load_gravity();
}`,
  });
  assert.doesNotMatch(source, /\bSimParams\b/u);
  assert.match(source, /__constant__ float3 cudaParams__gravity;/u);
  assert.match(source, /__constant__ uint cudaParams__count;/u);
  assert.match(source, /return cudaParams__gravity\.x \+ \(float\)cudaParams__count;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template<int BlockSize>
__global__ void __launch_bounds__(BlockSize) block_sized(float* out) {
  __shared__ float scratch[BlockSize];
  scratch[threadIdx.x] = 1.0f;
  out[threadIdx.x] = scratch[threadIdx.x];
}`,
  });
  assert.match(source, /template\s*<int BlockSize = 256>/u);
  assert.match(source, /__shared__ float scratch\[256\]/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <const int K_STAGE, const int A_PAD, const bool WARP_SWIZZLE>
__global__ void staged_scratch(float* out) {
  extern __shared__ half smem[];
  half* next = smem + K_STAGE * (64 + A_PAD);
  out[0] = (float)K_STAGE + (float)A_PAD + (float)WARP_SWIZZLE + (float)(next != nullptr);
}`,
  });
  assert.match(source, /template\s*<const int K_STAGE = 2, const int A_PAD = 0, const bool WARP_SWIZZLE = 0>/u);
  assert.match(source, /half\* next = smem \+ 2 \* \(64 \+ 0\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void atomic_forward(float* out) {
  addX(&out[threadIdx.x], 1.0f);
}`,
    deviceFunctions: [
      {
        name: "addX",
        source: "__device__ void addX(float* addr, float value) { atomicAdd(addr, value); }",
      },
    ],
  });
  assert.doesNotMatch(source, /\bvoid addX\b/u);
  assert.match(source, /atomicAdd\(&out\[threadIdx\.x\], 1\.0f\);/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void store_forward(float* out) {
  float tmp[1] = {0.0f};
  assignX(2.0f, &tmp[0], 123u);
  out[0] = tmp[0];
}`,
    deviceFunctions: [
      {
        name: "assignX",
        source: "__device__ __forceinline__ void assignX(float value, float* out, unsigned int seed) { *out = value; }",
      },
    ],
  });
  assert.doesNotMatch(source, /\bvoid assignX\b/u);
  assert.match(source, /\(tmp\[0\] = 2\.0f\);/u);
}

{
  const source = createKernelCompilationUnit({
    definesByName: new Map([["floatX", "bf16"]]),
    kernel: `
__global__ void implicit_x128(const floatX* x, floatX* y) {
  x128 value = load128(x);
  store128(y, value);
}`,
  });
  assert.doesNotMatch(source, /\bx128\b/u);
  assert.match(source, /bf16 value\[8\]/u);
  assert.match(source, /value\[7\] = x\[7\]/u);
  assert.match(source, /y\[7\] = value\[7\]/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void nested_template_value(uint* out, uint* data) {
  merge<1>(out, data);
}`,
    deviceFunctions: [
      {
        name: "merge",
        source: `
template <uint sortDir>
__device__ void merge(uint* out, uint* data) {
  out[0] = binarySearchInclusive<sortDir>(data[0], data, 4, 2);
}`,
      },
      {
        name: "binarySearchInclusive",
        source: `
template <uint sortDir>
__device__ uint binarySearchInclusive(uint val, uint* data, uint len, uint stride) {
  return sortDir ? val : data[0];
}`,
      },
    ],
  });
  assert.match(source, /template <uint sortDir = 1>\s*__device__ uint binarySearchInclusive/u);
  assert.doesNotMatch(source, /template <uint sortDir>\s*__device__ uint binarySearchInclusive/u);
}

{
  const source = createKernelCompilationUnit({
    sharedDeclarations: ["extern __shared__ unsigned char LocalBlock[];"],
    kernel: `
__global__ void uses_translation_unit_shared(uint* out) {
  out[threadIdx.x] = LocalBlock[threadIdx.x];
}`,
  });
  assert.match(source, /extern __shared__ unsigned char LocalBlock\[\];/u);
}

{
  const context = `
constexpr int BATCH = 1;
constexpr int Q_HEADS = 8;
constexpr int SEQ_LEN = 64;
constexpr int HEAD_DIM = 64;
constexpr int HALF_ROPE_DIM = HEAD_DIM / 2;
constexpr int COS_BS = 1;
constexpr std::size_t Q_SIZE = (std::size_t)BATCH * Q_HEADS * SEQ_LEN * HEAD_DIM;
constexpr std::size_t COS_SIZE = (std::size_t)COS_BS * SEQ_LEN * HALF_ROPE_DIM;
constexpr std::size_t cmax(std::size_t a, std::size_t b) { return a > b ? a : b; }
constexpr std::size_t INIT_N = cmax(Q_SIZE, COS_SIZE);`;
  const source = createKernelCompilationUnit({
    definesByName: collectCudaLiteContextDefines(context),
    kernel: `
__global__ void uses_constexpr_sizes(uint* out) {
  std::size_t tid = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < Q_SIZE) out[0] = INIT_N;
}`,
  });
  assert.match(source, /if \(tid < 32768\)/u);
  assert.match(source, /out\[0\] = 32768;/u);
  assert.doesNotMatch(source.slice(source.indexOf("__global__")), /\bQ_SIZE\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void while_prefix_update(uint* out, int limit) {
  int i = limit;
  while (--i && (out[0] < 4u)) {
    out[0] += 1u;
  }
}`,
  });
  assert.match(source, /while \(true\) \{\n\s+i--;\n\s+if \(!\(i && \(out\[0\] < 4u\)\)\) break;/u);
  assert.doesNotMatch(source, /while \(--i/u);
}

{
  const source = createKernelCompilationUnit({
    definesByName: new Map([["size", "4"]]),
    kernel: `
__global__ void keep_member_size(uint* out) {
  if (warp.size() < size) out[0] = size;
}`,
  });
  assert.match(source, /warp\.size\(\) < 4/u);
  assert.match(source, /out\[0\] = 4;/u);
  assert.doesNotMatch(source, /^#define size\b/mu);
  assert.doesNotMatch(source, /warp\.4/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T = float>
__global__ void use_vec_carrier(float4* oldPos, float3* out) {
  float4 position = oldPos[threadIdx.x];
  out[threadIdx.x] = computeBodyAccel<float>(position, oldPos, 1);
}`,
    deviceFunctions: [
      {
        name: "getSofteningSquared",
        source: `
template <typename T>
__device__ T getSofteningSquared() { return T(1); }`,
      },
      {
        name: "getSofteningSquared",
        source: `
template <> __device__ double getSofteningSquared<double>() { return 2.0; }`,
      },
      {
        name: "bodyBodyInteraction",
        source: `
template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai, typename vec4<T>::Type bi, typename vec4<T>::Type bj) {
  typename vec3<T>::Type r;
  T distSqr = getSofteningSquared<T>();
  r.x = ai.x + bi.x + bj.x + distSqr;
  return r;
}`,
      },
      {
        name: "computeBodyAccel",
        source: `
template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos, typename vec4<T>::Type *positions, int numTiles) {
  typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};
  acc = bodyBodyInteraction<T>(acc, bodyPos, positions[0]);
  acc.x += float(numTiles);
  return acc;
}`,
      },
    ],
  });
  assert.match(source, /__device__ float\s+getSofteningSquared\(\)/u);
  assert.match(source, /__device__ float3\s+bodyBodyInteraction\(float3 ai, float4 bi, float4 bj\)/u);
  assert.match(source, /__device__ float3\s+computeBodyAccel\(float4 bodyPos, float4 \*positions, int numTiles\)/u);
  assert.match(source, /float3 acc = \{0\.0f, 0\.0f, 0\.0f\};/u);
  assert.match(source, /computeBodyAccel\(position, oldPos, 1\)/u);
  assert.doesNotMatch(source, /typename vec/u);
  assert.doesNotMatch(source, /template\s*<\s*>/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void inline_pointer_helper(float* x, bf16* out, uint seed) {
  bf16 tmp[1];
  stochastic_rounding(x[0], &tmp[0], seed);
  out[0] = tmp[0];
}`,
    deviceFunctions: [
      {
        name: "stochastic_rounding",
        source: `
__device__ __forceinline__ void stochastic_rounding(float in, bf16 *out, unsigned int seed) {
  unsigned int bits = __float_as_uint(in);
  bits += seed;
  *out = __float2bfloat16_rn(__uint_as_float(bits));
}`,
      },
    ],
  });
  assert.doesNotMatch(source, /__device__ __forceinline__ void stochastic_rounding/u);
  assert.match(source, /float __bg_in = x\[0\];/u);
  assert.match(source, /unsigned int __bg_seed = seed;/u);
  assert.match(source, /tmp\[0\] = __float2bfloat16_rn/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void ref_atomic(uint* out) {
  __shared__ unsigned int flag;
  if (threadIdx.x == 0) flag = 0u;
  mark(flag, 1);
  out[0] = flag;
}`,
    deviceFunctions: [
      {
        name: "mark",
        source: `
__device__ void mark(unsigned int &flag, int value) {
  if (value > 0) atomicExch(&flag, 1u);
}`,
      },
    ],
  });
  assert.match(source, /void mark\(unsigned int \*flag, int value\)/u);
  assert.match(source, /mark\(&flag, 1\)/u);
  assert.match(source, /atomicExch\(flag, 1u\)/u);
  assert.doesNotMatch(source, /unsigned int &flag/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void skip_float_ref(float2* out) {
  float2 twiddle;
  fill(twiddle);
  out[0] = twiddle;
}`,
    deviceFunctions: [
      {
        name: "fill",
        source: `
__device__ void fill(float2 &twiddle) {
  __sincosf(1.0f, &twiddle.x, &twiddle.y);
}`,
      },
    ],
  });
  assert.match(source, /float2 &twiddle/u);
  assert.match(source, /twiddle\.x = sinf\(1\.0f\); twiddle\.y = cosf\(1\.0f\);/u);
  assert.doesNotMatch(source, /\(\*twiddle\)/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void reachable_device_global(uint* out) {
  out[0] = activeGlobal;
}`,
    deviceGlobalDeclarations: [
      "__device__ blockFunction_t unusedFunctionTable[2];",
      "__device__ unsigned int activeGlobal = 7u;",
    ],
  });
  assert.match(source, /__device__ unsigned int activeGlobal = 7u;/u);
  assert.doesNotMatch(source, /unusedFunctionTable/u);
  assert.doesNotMatch(source, /blockFunction_t/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void ignore_unused_context(uint* out) {
  out[0] = 1u;
}`,
    constantDeclarations: [
      "__constant__ short unusedTable[] = {1, 2, 3};",
    ],
    deviceGlobalDeclarations: [
      "__device__ virtualTable_t unusedVirtualDispatch;",
    ],
    textureDeclarations: [
      "texture<float, 2, cudaReadModeElementType> unusedTexture;",
    ],
  });
  assert.doesNotMatch(source, /unusedTable/u);
  assert.doesNotMatch(source, /unusedVirtualDispatch/u);
  assert.doesNotMatch(source, /unusedTexture/u);
}

console.log("cuda-lite source normalizer tests ok");
