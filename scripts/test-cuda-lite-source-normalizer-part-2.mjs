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
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void vector_record(uint4 *dst, uint4 src) {
  packed_result value = pack(src);
  dst[0] = value.x;
  dst[1] = value.y;
}`,
    deviceFunctions: [{
      name: "pack",
      source: `
__device__ packed_result pack(uint4 value) {
  packed_result out;
  out.x = value;
  out.y = make_uint4(1u, 2u, 3u, 4u);
  return out;
}`,
    }],
    recordDeclarations: [`
struct packed_result {
  uint4 x, y, z, w;
};`],
  });
  assert.doesNotMatch(source, /\bstruct\s+packed_result\b/u);
  assert.match(source, /__device__ void pack\(uint4 value, uint4 \*pack__bg_return__x/u);
  assert.match(source, /uint4 value__x; uint4 value__y; uint4 value__z; uint4 value__w; pack\(src, &value__x, &value__y, &value__z, &value__w\);/u);
  assert.match(source, /dst\[0\] = value__x;/u);
  assert.match(source, /dst\[1\] = value__y;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void record_memcpy(uint4 *dst) {
  packed_result out;
  unsigned int raw[8];
  memcpy(&out, raw, sizeof(out));
  dst[0] = out.x;
  dst[1] = out.y;
}`,
    recordDeclarations: [`
struct packed_result {
  uint4 x, y;
};`],
  });
  assert.doesNotMatch(source, /\bmemcpy\b/u);
  assert.match(source, /out__x = make_uint4\(raw\[0\], raw\[1\], raw\[2\], raw\[3\]\);/u);
  assert.match(source, /out__y = make_uint4\(raw\[4\], raw\[5\], raw\[6\], raw\[7\]\);/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void vector_shuffle(uint4 *out) {
  cg::thread_block block = cg::this_thread_block();
  auto tile = cg::tiled_partition<4>(block);
  uint4 value = out[threadIdx.x];
  value = tile.shfl_xor(value, 1);
  out[threadIdx.x] = value;
}`,
  });
  assert.match(source, /value = make_uint4\(tile\.shfl_xor\(value\.x, 1\), tile\.shfl_xor\(value\.y, 1\), tile\.shfl_xor\(value\.z, 1\), tile\.shfl_xor\(value\.w, 1\)\);/u);
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
  assert.match(source, /float2 total = cg::reduce\(warp, value, merge_pair\);/u);
  assert.doesNotMatch(source, /bg_for_bg_cg_reduce_offset_0_0/u);
  assert.doesNotMatch(source, /__shfl_xor_sync\(0xffffffff, total\.x/u);
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
  assert.deepEqual(launches.get("child"), ["16", "1"]);
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
template <typename T, int BM, int BN, int BK, int kStage>
__global__ void cute_template_values(T *out) {
  out[0] = (T)(BM + BN + BK + kStage);
}`;
  const wrapper = `
template <typename T, const int Stages = 2>
void launch_cute_template_values(T *out) {
  auto BM = Int<128>{};
  auto BN = Int<256>{};
  auto BK = Int<32>{};
  auto KStage = Int<Stages>{};
  cute_template_values<T, BM, BN, BK, KStage><<<1, 1>>>(out);
}
void host(half *out) {
  launch_cute_template_values<half, 2>(out);
}`;
  const launches = collectKernelTemplateArguments(`${kernel}\n${wrapper}`);
  assert.deepEqual(launches.get("cute_template_values"), ["half", "128", "256", "32", "2"]);
  const source = createKernelCompilationUnit({ kernel, templateArgumentsByKernelName: launches });
  assert.match(source, /template <typename T = half, int BM = 128, int BN = 256, int BK = 32, int kStage = 2>/u);
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
  assert.match(source, /__global__ void inferred_kernel\(float \*out, const float \*in, bool bg_bool_constant_0\)/u);
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
__global__ void record_helper_call(OutPair *out, float *sum, float *sum2, int i) {
  store_pair(sum, sum2, &out[i]);
}`,
    deviceFunctions: [
      {
        name: "store_pair",
        source: `__device__ void store_pair(float *sum, float *sum2, OutPair *out) {
  OutPair value = {sum[0], sum2[0]};
  *out = value;
}`,
      },
    ],
    recordDeclarations: [
      `typedef struct {
  double expected;
  double confidence;
} OutPair;`,
    ],
  });
  assert.match(source, /__global__ void record_helper_call\(double \*out__expected, double \*out__confidence, float \*sum, float \*sum2, int i\)/u);
  assert.match(source, /__device__ void store_pair\(float \*sum, float \*sum2, double \*out__expected, double \*out__confidence\)/u);
  assert.match(source, /store_pair\(sum, sum2, &out__expected\[i\], &out__confidence\[i\]\)/u);
  assert.match(source, /out__expected\[0\] = value__expected; out__confidence\[0\] = value__confidence;/u);
  assert.doesNotMatch(source, /\bOutPair\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
typedef int (*op_t)(int, int);
__global__ void function_pointer_dispatch(int *out, int selector, op_t pointOp) {
  op_t localOp;
  localOp = op_table[selector];
  int value = (*localOp)(out[0], 3);
  if (pointOp != NULL) {
    value = (*pointOp)(value, 1);
  }
  out[0] = value;
}`,
    deviceFunctions: [
      { name: "add_op", source: "__device__ int add_op(int a, int b) { return a + b; }" },
      { name: "mul_op", source: "__device__ int mul_op(int a, int b) { return a * b; }" },
    ],
    deviceGlobalDeclarations: [
      "__device__ op_t op_table[2];",
      "__device__ op_t localOp;",
    ],
    functionPointerTables: [
      {
        tableName: "op_table",
        aliasName: "op_t",
        entries: [
          { index: 0, target: "add_op" },
          { index: 1, target: "mul_op" },
        ],
      },
    ],
  });
  assert.match(source, /__device__ int add_op/u);
  assert.match(source, /__device__ int mul_op/u);
  assert.match(source, /uint localOp = selector;/u);
  assert.match(source, /int value = \(\(localOp\) == 0 \? add_op\(out\[0\], 3\) : mul_op\(out\[0\], 3\)\);/u);
  assert.match(source, /if \(pointOp != 0xffffffffu\)/u);
  assert.match(source, /value = \(\(pointOp\) == 0 \? add_op\(value, 1\) : mul_op\(value, 1\)\);/u);
  assert.doesNotMatch(source, /__device__ op_t/u);
  assert.doesNotMatch(source, /op_table\[/u);
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
  assert.doesNotMatch(source, /#define IDX2C/u);
  assert.match(source, /out\[0\] = in\[\(\(\(\(0\)\)\*\(\(ld\)\)\)\+\(\(row\)\)\)\];/u);
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
template <class T>
__global__ void sharedmem_index(T *input, T *output) {
  SharedMemory<T> scratch;
  scratch[threadIdx.x] = input[threadIdx.x];
  output[0] = scratch[0];
}`,
    templateArgumentsByKernelName: new Map([["sharedmem_index", ["float"]]]),
  });
  assert.match(source, /extern __shared__ float scratch\[\];/u);
  assert.doesNotMatch(source, /SharedMemory/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename rngState_t, typename rngDirectionVectors_t>
__global__ void rng_init(rngState_t *state, rngDirectionVectors_t *direction) {
  curand_init(direction[0], threadIdx.x, &state[threadIdx.x]);
}`,
    templateArgumentsByKernelName: new Map([["rng_init", ["curandStateSobol_sz", "curandDirectionVectors_sz"]]]),
  });
  assert.match(source, /template <typename rngState_t = uint, typename rngDirectionVectors_t = uint>/u);
  assert.match(source, /curand_init\(direction\[0\], threadIdx\.x, 0, &state\[threadIdx\.x\]\);/u);
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
#define FMUL(x, y) (__mul24(x, y))
#define FMUL(x, y) ((x) * (y))
#define IMAD(a, b, c) (((a) * (b)) + (c))
__global__ void expression_macro(float *src, int stride) {
  src += IMAD(blockIdx.y, stride, threadIdx.x);
  src[0] = FMUL(src[0], 2.0f);
}`,
  });
  assert.doesNotMatch(source, /^\s*#define\s+FMUL/mu);
  assert.doesNotMatch(source, /\b(?:FMUL|IMAD)\s*\(/u);
  assert.match(source, /src \+= \(\(\(\(blockIdx\.y\)\) \* \(\(stride\)\)\) \+ \(\(threadIdx\.x\)\)\);/u);
  assert.match(source, /src\[0\] = \(\(\(src\[0\]\)\) \* \(\(2\.0f\)\)\);/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void missing_legacy_macro(float *dst, int width) {
  dst[IMAD(blockIdx.y, width, threadIdx.x)] = FMUL(dst[0], 2.0f);
}`,
  });
  assert.doesNotMatch(source, /\b(?:FMUL|IMAD)\s*\(/u);
  assert.match(source, /dst\[\(\(\(blockIdx\.y\) \* \(width\)\) \+ \(threadIdx\.x\)\)\]/u);
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
