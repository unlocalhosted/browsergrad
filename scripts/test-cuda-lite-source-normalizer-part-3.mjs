import {
  assert,
  collectCudaLiteContextDefines,
  collectKernelTemplateArguments,
  createKernelCompilationUnit,
  kernelDefinitionName,
  pruneCudaPreprocessorBranches,
  scalarKernel,
  vectorSibling,
} from "./test-cuda-lite-source-normalizer-support.mjs";

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
  assert.doesNotMatch(source, /#define BG_APPLY/u);
  assert.match(source, /helper\(\(out\[threadIdx\.x\]\)\);/u);
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
template <class T = float>
__global__ void defaulted_kernel_template(float *out) {
  out[0] = project<T>(1.0f, 2.0f);
}`,
    deviceFunctions: [
      {
        name: "project",
        source: `
template <class T>
__device__ T project(const T x, const T y) {
  T sum = x + y;
  return sum;
}`,
      },
    ],
  });
  assert.match(source, /template <class T = float>/u);
  assert.match(source, /__device__ float project\(const float x, const float y\)/u);
  assert.doesNotMatch(source, /\bT\s+sum/u);
  assert.match(source, /out\[0\] = project\(1\.0f, 2\.0f\);/u);
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
    definesByName: new Map([["VolumeType", "uint"]]),
    kernel: `
template <typename T> struct VolumeTypeInfo
{
};
__global__ void static_template_converter(uint *out, float *input) {
  out[threadIdx.x] = VolumeTypeInfo<VolumeType>::convert(input[threadIdx.x]);
}`,
  });
  assert.doesNotMatch(source, /template <typename T> struct VolumeTypeInfo/u);
  assert.match(source, /out\[threadIdx\.x\] = uint\(input\[threadIdx\.x\]\);/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: [
      "typedef struct { double Expected; double Confidence; } __TOptionValue;",
    ],
    kernel: `
__global__ void double_record_kernel(float *out, __TOptionValue *values) {
  __TOptionValue sum = {0, 0};
  sum.Expected += 1.0;
  sum.Confidence += 2.0;
  values[threadIdx.x] = sum;
  out[threadIdx.x] = sum.Confidence;
}`,
  });
  assert.doesNotMatch(source, /__TOptionValue/u);
  assert.match(source, /double sum__Expected = 0;/u);
  assert.match(source, /double \*values__Expected, double \*values__Confidence/u);
  assert.match(source, /values__Confidence\[threadIdx\.x\] = sum__Confidence;/u);
  assert.match(source, /out\[threadIdx\.x\] = sum__Confidence;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void f64_helper_intake(float *out) {
  extern double __shared__ scratch[];
  scratch[threadIdx.x] = 1.0;
  reduceDouble(scratch);
  out[threadIdx.x] = scratch[threadIdx.x];
}`,
    deviceFunctions: [
      { name: "reduceDouble", source: "__device__ void reduceDouble(double *scratch) { scratch[0] = scratch[0] + 1.0; }" },
    ],
  });
  assert.match(source, /__device__ void reduceDouble\(double \*scratch\)/u);
  assert.match(source, /reduceDouble\(scratch\);/u);
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
__global__ void cooperative_scratch_kernel(int *out) {
  __shared__ cg::block_tile_memory<1024> scratch;
  auto cta = cg::this_thread_block(scratch);
  auto tile = cg::tiled_partition<32>(cta);
  if (tile.thread_rank() == 0) out[0] = tile.meta_group_size();
}`,
  });
  assert.doesNotMatch(source, /block_tile_memory/u);
  assert.match(source, /cg::thread_block cta = cg::this_thread_block\(\);/u);
  assert.match(source, /auto tile = cg::tiled_partition<32>\(cta\);/u);
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
  assert.match(source, /for \(uint bg_for_stride_0 = 4;bg_for_stride_0 > 0;bg_for_stride_0 >>= 1\)/u);
  assert.match(source, /out\[bg_for_stride_0\] = bg_for_stride_0;/u);
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
      ["x128", "bg_pack128_half8"],
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
__global__ void bool_template_carrier(float* out, const float* in, bool bg_bool_constant_UseAuxBuffer) {
  if constexpr (!UseAuxBuffer) {
    out[threadIdx.x] = in[threadIdx.x];
  }
}`,
  });
  assert.match(source, /if constexpr \(!bg_bool_constant_UseAuxBuffer\)/u);
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
