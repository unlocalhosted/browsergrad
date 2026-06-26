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
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out) {
  out[0] = helper(1.0f);
}`,
    definesByName: new Map([["USE_FAST", "1"]]),
    deviceFunctions: [{ name: "helper", source: `
__device__ float helper(float x) {
#if USE_FAST
  float value = x + 1.0f;
#else
  broken_t value = x;
#endif
  return value;
}` }],
  });
  assert.match(source, /float value = x \+ 1.0f;/u);
  assert.doesNotMatch(source, /broken_t/u);
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
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void macro_member_props(int *out) {
  int row = FMUL(threadIdx.y, BLOCK_SIZE) + threadIdx.x;
  int col = IMAD(blockIdx.x, BLOCK_SIZE, threadIdx.z);
  out[0] = row + col;
}`,
    definesByName: new Map([["BLOCK_SIZE", "8"]]),
    functionDeclarations: [
      "#define FMUL(x, y) (__mul24(x, y))",
      "#define IMAD(a, b, c) (((a) * (b)) + (c))",
    ],
  });
  assert.match(source, /threadIdx\.y/u);
  assert.match(source, /threadIdx\.x/u);
  assert.match(source, /blockIdx\.x/u);
  assert.match(source, /threadIdx\.z/u);
  assert.doesNotMatch(source, /threadIdx\.\(/u);
  assert.doesNotMatch(source, /blockIdx\.\(/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void bitpack(uint *input, uint *out) {
  use_pack(input, out);
}`,
    deviceFunctions: [
      {
        name: "use_pack",
        source: `
__device__ void use_pack(uint *input, uint *out) {
  PackedShorts a, b;
  a.hInt = input[0];
  b.hShort1 = a.hShort2;
  b.hShort2 = a.hShort1;
  out[0] = b.hInt;
}`,
      },
    ],
    recordDeclarations: [
      `
union PackedShorts {
  struct __align__(8) {
    short hShort1;
    short hShort2;
  };
  unsigned int hInt;
};`,
    ],
  });
  assert.doesNotMatch(source, /\bunion\s+PackedShorts/u);
  assert.match(source, /uint a, b;/u);
  assert.match(source, /a = input\[0\];/u);
  assert.match(source, /b = \(b & 0xffff0000u\) \| \(uint\(\(int\(a\) >> 16\)\) & 0xffffu\);/u);
  assert.match(source, /b = \(b & 0x0000ffffu\) \| \(\(uint\(\(\(int\(a << 16\)\) >> 16\)\) & 0xffffu\) << 16\);/u);
  assert.match(source, /out\[0\] = b;/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename Real>
__global__ void option_kernel(float *out, const AsianOption<Real> *const option) {
  Real payoff = option->spot - option->strike;
  if (option->type == AsianOption<Real>::Put) payoff = -payoff;
  out[0] = payoff;
}`,
    templateArgumentsByKernelName: new Map([["option_kernel", ["float"]]]),
    recordDeclarations: [
      `
template <typename Real> struct AsianOption {
  enum CallPut { Call, Put };
  Real spot;
  Real strike;
  CallPut type;
};`,
    ],
  });
  assert.doesNotMatch(source, /AsianOption\s*</u);
  assert.doesNotMatch(source, /\benum\s+CallPut/u);
  assert.match(source, /float \*out, const float \*option__spot, const float \*option__strike, const int \*option__type/u);
  assert.match(source, /float payoff = option__spot\[0\] - option__strike\[0\];/u);
  assert.match(source, /if \(option__type\[0\] == 1\)/u);
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
  assert.deepEqual(launches.get("sample"), ["1"]);
}

{
  const source = `
template <int A, int B, int C>
__global__ void dependent(int *out) { out[0] = A + B + C; }

template <int HeadDim>
void launch_dependent(int *out) {
  constexpr int LocalB = HeadDim / 8;
  constexpr int LocalC = 2 * LocalB;
  dependent<HeadDim, LocalB, LocalC><<<1, 32>>>(out);
}

void launch(int *out) {
  launch_dependent<64>(out);
}`;
  const launches = collectKernelTemplateArguments(source);
  assert.deepEqual(launches.get("dependent"), ["64", "8", "16"]);
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
  assert.match(source, /warp_reduce_sum\(1\.0f\)/u);
  assert.doesNotMatch(source, /__device__ __forceinline__.*warp_reduce_sum/u);
  assert.doesNotMatch(source, /\bT\s+(?:val|tmp|value)\b/u);
  assert.doesNotMatch(source, /warp_reduce_sum<WARP_SIZE>/u);
  assert.doesNotMatch(source, /warp_reduce_sum<32>/u);
}

{
  const fullSource = `
template <typename T_, int Block_>
struct KernelConfig {
  using T = T_;
  static constexpr int Block = Block_;
};

template <typename Config_>
__global__ void configured_kernel(typename Config_::T *out) {
  using T = typename Config_::T;
  constexpr int Block = Config_::Block;
  out[threadIdx.x] = static_cast<T>(Block);
}

template <int Block>
void launch_configured(half_t *out) {
  using config = KernelConfig<half_t, Block>;
  configured_kernel<config><<<1, Block>>>(out);
}

void launch(half_t *out) {
  launch_configured<64>(out);
}`;
  const kernel = fullSource.slice(fullSource.indexOf("template <typename Config_>"), fullSource.indexOf("template <int Block>")).trim();
  const source = createKernelCompilationUnit({
    kernel,
    functionDeclarations: [fullSource],
    templateArgumentsByKernelName: collectKernelTemplateArguments(fullSource),
  });
  assert.match(source, /template <typename Config_>/u);
  assert.match(source, /__global__ void configured_kernel\(half \*out\)/u);
  assert.match(source, /constexpr int Block = 64;/u);
  assert.match(source, /static_cast<half>\(Block\)/u);
  assert.doesNotMatch(source, /\bT\s+[A-Za-z_]/u);
  assert.doesNotMatch(source, /Config_::/u);
}

{
  const source = `
template <int A, int B>
__global__ void direct_dependent(int *out) { out[0] = A + B; }

void launch(int *out) {
  constexpr int Base = 64;
  constexpr int Derived = Base / 8;
  direct_dependent<Base, Derived><<<1, 32>>>(out);
}`;
  const launches = collectKernelTemplateArguments(source);
  assert.deepEqual(launches.get("direct_dependent"), ["64", "8"]);
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
  const source = createKernelCompilationUnit({
    kernel: `
template <int kNumElemPerThread = 8>
__global__ void cute_affine(half *z, int num, const half *x, const half *y, half a, half b, half c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) return;
  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));
  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);
  copy(txr, txR);
  copy(tyr, tyR);
  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};
  auto tzR2 = recast<half2>(tzR);
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);
  for (int i = 0; i < size(tzR2); ++i) {
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }
  auto tzRx = recast<half>(tzR2);
  copy(tzRx, tzr);
}`,
    templateArgumentsByKernelName: new Map([["cute_affine", ["8"]]]),
  });
  assert.match(source, /for \(int bg_for_bg_cute_i_0 = 0;bg_for_bg_cute_i_0 < 8;\+\+bg_for_bg_cute_i_0\)/u);
  assert.match(source, /z\[bg_cute_pos\] = x\[bg_cute_pos\] \* a \+ \(y\[bg_cute_pos\] \* b \+ c\);/u);
  assert.doesNotMatch(source, /\bTensor\b/u);
  assert.doesNotMatch(source, /\brecast\s*</u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA, typename ThreadLayoutB>
__global__ void cute_transpose_direct(const T *pA, T *pB, int M, int N, ThreadLayoutA tA, ThreadLayoutB tB) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  auto mA = make_tensor(make_gmem_ptr(pA), make_layout(make_shape(M, N), GenRowMajor{}));
  auto mB = make_tensor(make_gmem_ptr(pB), make_layout(make_shape(N, M), GenRowMajor{}));
  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by));
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx));
  auto cA = local_tile(make_identity_tensor(mA.shape()), make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by));
  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tBgB = local_partition(gB, tB, tx);
  Tensor tAcA = local_partition(cA, tA, tx);
  Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
  copy_if(tApA, tAgA, tBgB);
}`,
    templateArgumentsByKernelName: new Map([["cute_transpose_direct", ["float", "8", "16"]]]),
  });
  assert.match(source, /__global__ void cute_transpose_direct\(const float \*pA, float \*pB, int M, int N\)/u);
  assert.match(source, /for \(int bg_for_bg_cute_linear_0 = threadIdx\.x;bg_for_bg_cute_linear_0 < \(8 \* 16\);bg_for_bg_cute_linear_0 = bg_for_bg_cute_linear_0 \+ blockDim\.x\)/u);
  assert.match(source, /pB\[\(bg_cute_n \* M\) \+ bg_cute_m\] = pA\[\(bg_cute_m \* N\) \+ bg_cute_n\];/u);
  assert.doesNotMatch(source, /\bThreadLayoutA\b/u);
  assert.doesNotMatch(source, /\bTensor\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA, typename ThreadLayoutB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void cute_transpose_smem(const T *pA, T *pB, int M, int N, ThreadLayoutA tA, ThreadLayoutB tB, SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  auto mA = make_tensor(make_gmem_ptr(pA), make_layout(make_shape(M, N), GenRowMajor{}));
  auto mB = make_tensor(make_gmem_ptr(pB), make_layout(make_shape(N, M), GenRowMajor{}));
  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by));
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx));
  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(make_smem_ptr(smem), sA_layout);
  auto sB = make_tensor(make_smem_ptr(smem), sB_layout);
  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tAsA = local_partition(sA, tA, tx);
  Tensor tBsB = local_partition(sB, tB, tx);
  Tensor tBgB = local_partition(gB, tB, tx);
  copy_if(tAgA, tAgA, tAsA);
  __syncthreads();
  copy_if(tBgB, tBsB, tBgB);
}`,
    templateArgumentsByKernelName: new Map([["cute_transpose_smem", ["float", "8", "16"]]]),
  });
  assert.match(source, /__global__ void cute_transpose_smem\(const float \*pA, float \*pB, int M, int N\)/u);
  assert.match(source, /bg_cute_m = \(\(blockIdx\.x\) \* 8\) \+ bg_cute_row/u);
  assert.doesNotMatch(source, /\bSmemLayoutA\b/u);
  assert.doesNotMatch(source, /\bmake_tensor\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T, int BLK_M, int BLK_N, typename TiledCopyA, typename TiledCopyTrans, typename TiledCopyB, typename SmemLayoutB>
__global__ void cute_transpose_retile(const T *pA, T *pB, int M, int N, TiledCopyA copy_a, TiledCopyTrans copy_trans, TiledCopyB copy_b, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  auto mA = make_tensor(make_gmem_ptr(pA), make_layout(make_shape(M, N), GenRowMajor{}));
  auto mB = make_tensor(make_gmem_ptr(pB), make_layout(make_shape(N, M), GenRowMajor{}));
  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by));
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx));
  __shared__ T smem[BLK_M * BLK_N];
  auto sB = make_tensor(make_smem_ptr(smem), sB_layout);
  auto thr_copy_a = copy_a.get_slice(tx);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  auto tAsA = make_tensor_like(tAgA);
  Tensor tAsA_view = thr_copy_a.retile_D(tAsA);
  copy(copy_a, tAgA, tAsA_view);
  auto thr_copy_trans = copy_trans.get_slice(tx);
  auto tAsB = thr_copy_trans.retile_S(tAsA);
  auto tBsB_trans = thr_copy_trans.partition_D(sB);
  copy(copy_trans, tAsB, tBsB_trans);
  auto thr_copy_b = copy_b.get_slice(tx);
  Tensor tBsB = thr_copy_b.partition_S(sB);
  Tensor tBgB = thr_copy_b.partition_D(gB);
  copy(copy_b, tBsB, tBgB);
}`,
    templateArgumentsByKernelName: new Map([["cute_transpose_retile", ["float", "8", "16"]]]),
  });
  assert.match(source, /__global__ void cute_transpose_retile\(const float \*pA, float \*pB, int M, int N\)/u);
  assert.match(source, /pB\[\(bg_cute_n \* M\) \+ bg_cute_m\] = pA\[\(bg_cute_m \* N\) \+ bg_cute_n\];/u);
  assert.doesNotMatch(source, /\bTiledCopyA\b/u);
  assert.doesNotMatch(source, /\bretile_[SD]\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename TiledCopy, int BlockM, int BlockK, int WARP_SIZE = 32>
__global__ void cute_hgemv(half *Aptr, half *Bptr, half *Cptr, const int M, const int K) {
  using namespace cute;
  int thrid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockid = blockIdx.x;
  int laneid = threadIdx.x % WARP_SIZE;
  auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
  auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
  auto C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));
  auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
  auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
  auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, 0));
  TiledCopy tiled_copy;
  auto thr_copy = tiled_copy.get_slice(thrid);
  auto tAgA = thr_copy.partition_S(gA);
  auto tBgB = thr_copy.partition_S(gB);
  auto sum = make_tensor_like(gC(0, _));
  clear(sum);
  for (int iter_k = 0; iter_k < size<2>(gA); iter_k++) {
    copy_if(tiled_copy, tAgA(_, _, _, iter_k), sum);
    sum(0) += tAgA(0) * tBgB(0);
  }
  sum(0) = warp_reduce_sum_f16<WARP_SIZE>(sum(0));
  copy_if(laneid == 0, sum, gC(0, _));
}`,
    templateArgumentsByKernelName: new Map([["cute_hgemv", ["TiledCopy", "4", "16"]]]),
  });
  assert.match(source, /__global__ void cute_hgemv\(half \*Aptr, half \*Bptr, half \*Cptr, const int M, const int K\)/u);
  assert.match(source, /half bg_cute_sum = 0\.0f;/u);
  assert.match(source, /Cptr\[bg_cute_row\] = bg_cute_sum;/u);
  assert.doesNotMatch(source, /\bTiledCopy\b/u);
  assert.doesNotMatch(source, /\bmake_tensor\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA>
__global__ void cute_hgemm_tn(T *Aptr, T *Bptr, T *Dptr, int m, int n, int k) {
  using namespace cute;
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD);
  cute::gemm(tiled_mma, tCrD, tCrA, tCrB, tCrD);
}`,
    templateArgumentsByKernelName: new Map([["cute_hgemm_tn", ["half", "8", "16", "4", "2", "MMA"]]]),
  });
  assert.match(source, /__global__ void cute_hgemm_tn\(half \*Aptr, half \*Bptr, half \*Dptr, int m, int n, int k\)/u);
  assert.match(source, /for \(int bg_for_bg_cute_linear_0 = bg_cute_tid;bg_for_bg_cute_linear_0 < \(8 \* 16\);bg_for_bg_cute_linear_0 = bg_for_bg_cute_linear_0 \+ \(\(blockDim\.x \* blockDim\.y\) \* blockDim\.z\)\)/u);
  assert.match(source, /Dptr\[\(bg_cute_row \* n\) \+ bg_cute_col\] = \(half\)bg_cute_acc;/u);
  assert.doesNotMatch(source, /\bTiledMMA\b/u);
  assert.doesNotMatch(source, /\bmake_tensor\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void device_runtime_noops(float *ptr) {
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaFree(ptr);
  cudaStreamDestroy(s);
}`,
  });
  assert.match(source, /cudaStream_t s;/u);
  assert.match(source, /s = cudaStreamNonBlocking;/u);
  assert.doesNotMatch(source, /\bcudaFree\b/u);
  assert.doesNotMatch(source, /\bcudaStreamCreateWithFlags\b/u);
  assert.doesNotMatch(source, /\bcudaStreamDestroy\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename WSHGEMMTraits>
__global__ void cute_carrier_gemm(typename WSHGEMMTraits::Arguments args) {
  constexpr int kCTAM = WSHGEMMTraits::kCTAM;
  constexpr int kCTAN = WSHGEMMTraits::kCTAN;
  constexpr int kCTAK = WSHGEMMTraits::kCTAK;
  auto tile_id_m = blockIdx.x;
  auto tile_id_n = blockIdx.y;
  auto A = make_tensor(make_gmem_ptr<typename WSHGEMMTraits::MatrixTypeAB>(args.a_ptr),
                       select<0, 2>(args.problem_shape), GenRowMajor{});
  auto B = make_tensor(make_gmem_ptr<typename WSHGEMMTraits::MatrixTypeAB>(args.b_ptr),
                       select<1, 2>(args.problem_shape), GenRowMajor{});
  auto C = make_tensor(make_gmem_ptr<typename WSHGEMMTraits::AccType>(args.c_ptr),
                       select<0, 1>(args.problem_shape), GenRowMajor{});
  auto gA = local_tile(A, make_tile(Int<kCTAM>{}, Int<kCTAK>{}), make_coord(tile_id_m, _));
  auto gB = local_tile(B, make_tile(Int<kCTAN>{}, Int<kCTAK>{}), make_coord(tile_id_n, _));
  auto gC = local_tile(C, make_tile(Int<kCTAM>{}, Int<kCTAN>{}), make_coord(tile_id_m, tile_id_n));
  WSHGEMMTraits::consumer(args, gC);
}`,
    recordDeclarations: [`
template <class CTATile, int Stage> struct WSTraits {
  using MatrixTypeAB = half;
  using AccType = half;
  constexpr static int kCTAM = get<0>(CTATile{});
  constexpr static int kCTAN = get<1>(CTATile{});
  constexpr static int kCTAK = get<2>(CTATile{});
  constexpr static int kStage = Stage;
  struct Arguments {
    void *a_ptr;
    void *b_ptr;
    void *c_ptr;
    Shape<int, int, int> problem_shape;
  };
};
using GEMM_Traits = WSTraits<decltype(make_shape(_8{}, _16{}, _4{})), 2>;`],
    templateArgumentsByKernelName: new Map([["cute_carrier_gemm", ["GEMM_Traits"]]]),
  });
  assert.match(source, /__global__ void cute_carrier_gemm\(half \*args__a_ptr, half \*args__b_ptr, half \*args__c_ptr, int args__problem_shape0, int args__problem_shape1, int args__problem_shape2\)/u);
  assert.match(source, /bg_for_bg_cute_linear_0 < \(8 \* 16\)/u);
  assert.match(source, /args__c_ptr\[\(bg_cute_row \* args__problem_shape1\) \+ bg_cute_col\] = \(half\)bg_cute_acc;/u);
  assert.doesNotMatch(source, /\btypename\s+GEMM_Traits::Arguments\b/u);
  assert.doesNotMatch(source, /\bmake_tensor\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <const int BM = 128, const int BN = 128, const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
wgmma_tma_kernel(int M, int N, int K, half *C, const CUtensorMap *__restrict__ tensorMapA, const CUtensorMap *__restrict__ tensorMapB) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  cde::cp_async_bulk_tensor_2d_global_to_shared(smem, tensorMapA, 0, by * BM, full);
  cde::cp_async_bulk_tensor_2d_global_to_shared(smem, tensorMapB, 0, bx * BN, full);
  WGMMA_M64N128K16_F32F16F16(d, sA, sB, 1, 1, 1, 0, 0);
}`,
  });
  assert.match(source, /__global__ void __launch_bounds__\(256\)\s*wgmma_tma_kernel\(int M, int N, int K, half \*C, half \*tensorMapA__base, half \*tensorMapB__base\)/u);
  assert.match(source, /bg_for_bg_wgmma_linear_0 < \(128 \* 128\)/u);
  assert.match(source, /tensorMapA__base\[\(bg_wgmma_row \* K\) \+ bg_wgmma_k\]/u);
  assert.match(source, /tensorMapB__base\[\(bg_wgmma_col \* K\) \+ bg_wgmma_k\]/u);
  assert.doesNotMatch(source, /\bCUtensorMap\b/u);
  assert.doesNotMatch(source, /\bcp_async_bulk_tensor_2d_global_to_shared\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void cute_flash_attn(half *pQ, half *pK, half *pV, half *pO, int B, int H, int N_QO, int N_KV, int D, float scale) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  auto Q = make_tensor(make_gmem_ptr(pQ), make_layout(make_shape(B, H, N_QO, D), GenRowMajor{}));
  auto K = make_tensor(make_gmem_ptr(pK), make_layout(make_shape(B, H, N_KV, D), GenRowMajor{}));
  auto V = make_tensor(make_gmem_ptr(pV), make_layout(make_shape(B, H, N_KV, D), GenRowMajor{}));
  auto O = make_tensor(make_gmem_ptr(pO), make_layout(make_shape(B, H, N_QO, D), GenRowMajor{}));
  auto gQ = local_tile(Q, make_shape(_1{}, _1{}, Int<4>{}, Int<16>{}), make_coord(bx, by, bz, 0))(0,0,_,_);
  auto global_row_denominator = make_tensor<float>(make_shape(Int<4>{}, Int<1>{}));
  gemm(tiled_mma, gQ, gQ, global_row_denominator);
  global_row_denominator(0, 0) = exp(scale);
}`,
  });
  assert.match(source, /__global__ void cute_flash_attn\(half \*pQ, half \*pK, half \*pV, half \*pO, int B, int H, int N_QO, int N_KV, int D, float scale\)/u);
  assert.match(source, /bg_for_bg_attn_linear_0 < \(4 \* D\)/u);
  assert.match(source, /float bg_attn_weight = expf\(\(bg_attn_score \* scale\) - bg_attn_max\);/u);
  assert.match(source, /pO\[\(\(\(\(\(bg_attn_b \* H\) \+ bg_attn_h\) \* N_QO\) \+ bg_attn_q\) \* D\) \+ bg_attn_dim\] = \(half\)\(bg_attn_acc \/ bg_attn_denom\);/u);
  assert.doesNotMatch(source, /\bmake_tensor\b/u);
  assert.doesNotMatch(source, /\bglobal_row_denominator\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
__global__ void vec4_recover(float *x, float *y, int N) {
  int idx = threadIdx.x * 4;
  float4 reg_x = FLOAT4(x[idx]) float value = idx < N ? reg_x.x : 0.0f;
  if (idx < N) y[idx] = value;
}`,
  });
  assert.match(source, /float4 reg_x = FLOAT4\(x\[idx\]\); float value = idx < N/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void lambda_switch(uint4 *out, uint4 value) {
  auto idxToElem = [&value](unsigned int idx) -> const uint4 {
    switch (idx) {
    case 0:
      return value;
    case 1:
      return make_uint4(1u, 2u, 3u, 4u);
    }
    return {};
  };
  out[threadIdx.x] = idxToElem(threadIdx.x);
}`,
  });
  assert.doesNotMatch(source, /\bauto\s+idxToElem\b/u);
  assert.doesNotMatch(source, /\bswitch\s*\(/u);
  assert.match(source, /out\[threadIdx\.x\] = \(\(\(\(threadIdx\.x\)\) == 0 \? value : \(\(\(threadIdx\.x\)\) == 1 \? make_uint4\(1u, 2u, 3u, 4u\) : make_uint4\(0u, 0u, 0u, 0u\)\)\)\);/u);
}
