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
    recordDeclarations: ["struct RefArrays { float *values; int n; size_t bytes; };"],
    kernel: `
__global__ void pointer_record(float *out, RefArrays ref) {
  int idx = threadIdx.x;
  if (idx < ref.n) {
    out[idx] = ref.values[idx] + (float)ref.bytes;
  }
}`,
  });
  assert.doesNotMatch(source, /struct RefArrays/u);
  assert.match(source, /float \*ref__values, int ref__n, uint ref__bytes/u);
  assert.match(source, /idx < ref__n/u);
  assert.match(source, /out\[idx\] = ref__values\[idx\] \+ \(float\)ref__bytes;/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: ["struct RefArrays { float *values; int n; size_t bytes; };"],
    kernel: `
__global__ void record_child(int i, RefArrays ref, float *out) {
  if (i < ref.n) out[i] = ref.values[i] + (float)ref.bytes;
}
__global__ void record_parent(RefArrays ref, float *out) {
  record_child<<<1, 1>>>(0, ref, out);
}`,
  });
  assert.match(source, /__global__ void record_child\(int i, float \*ref__values, int ref__n, uint ref__bytes, float \*out\)/u);
  assert.match(source, /__global__ void record_parent\(float \*ref__values, int ref__n, uint ref__bytes, float \*out\)/u);
  assert.match(source, /record_child<<<1, 1>>>\(0, ref__values, ref__n, ref__bytes, out\);/u);
  assert.doesNotMatch(source, /\bRefArrays\b/u);
}

{
  const source = createKernelCompilationUnit({
    recordDeclarations: ["struct DevicePool { int* base; unsigned int poolSize; unsigned int offset; };"],
    kernel: `
__global__ void allocationKernel(DevicePool* dp, float* output, int N) {
  void* ptr = deviceAllocate(dp, sizeof(float));
  output[threadIdx.x] = ptr != nullptr ? 1.0f : 0.0f;
}`,
  });
  assert.match(source, /DevicePool\* dp/u);
  assert.match(source, /deviceAllocate\(dp, sizeof\(float\)\)/u);
  assert.doesNotMatch(source, /\bdp__base\b/u);
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
    kernel: `
__global__ void postfix_index_load(float* out, const float* weights) {
  int idx = 0;
  float weight = weights[idx++];
  out[0] = weight + float(idx);
}`,
  });
  assert.match(source, /float weight = weights\[idx\];\n\s+idx\+\+;/u);
  assert.doesNotMatch(source, /idx\+\+\]/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void postfix_index_reuse_guard(float* out, const float* weights) {
  int idx = 0;
  out[idx] = weights[idx++];
}`,
  });
  assert.match(source, /out\[idx\] = weights\[idx\+\+\];/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void local_reference_alias(const uint* startpoints, const uint* verticesMapping, uint* edges, uint edgesCount) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < edgesCount) {
    uint startpoint = startpoints[tid];
    uint &endpoint = edges[tid];
    uint newStartpoint = verticesMapping[startpoint];
    uint newEndpoint = verticesMapping[endpoint];
    if (newStartpoint == newEndpoint) {
      endpoint = 0xffffffff;
    }
  }
}`,
  });
  assert.match(source, /uint endpoint = edges\[tid\];/u);
  assert.match(source, /edges\[tid\] = 0xffffffff;\n\s+endpoint = 0xffffffff;/u);
  assert.doesNotMatch(source, /uint\s*&\s*endpoint/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void local_reference_alias_scope(uint* edges) {
  {
    uint &endpoint = edges[0];
    endpoint = 1u;
  }
  uint endpoint = 0u;
  endpoint = 2u;
}`,
  });
  assert.match(source, /edges\[0\] = 1u;\n\s+endpoint = 1u;/u);
  assert.match(source, /uint endpoint = 0u;\n\s+endpoint = 2u;/u);
  assert.doesNotMatch(source, /edges\[0\] = 2u/u);
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
  assert.match(source, /float bg_in = x\[0\];/u);
  assert.match(source, /unsigned int bg_seed = seed;/u);
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
    kernel: String.raw`
__device__ float helper(float x) { \n return x + 1.0f; \n }
__global__ \n void escaped_newline_kernel(float* out) { \n out[0] = helper(1.0f); \n }`,
  });
  assert.match(source, /__global__\s+void escaped_newline_kernel/u);
  assert.match(source, /return x \+ 1\.0f;/u);
  assert.doesNotMatch(source, /\\n void escaped_newline_kernel/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: String.raw`
__global__ void keep_asm_newline(uint* out) {
  asm volatile("mov.u32 %0, %laneid;\n" : "=r"(out[0]));
}`,
  });
  assert.match(source, /"mov\.u32 %0, %laneid;\\n"/u);
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

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void reachable_constant_table(uint* out) {
  uint *row = &c_Table[threadIdx.y][0];
  out[0] = row[threadIdx.x];
}`,
    constantDeclarations: [
      "__constant__ unsigned int c_Table[3][31];",
      "__constant__ unsigned int unusedTable[4][4];",
    ],
  });
  assert.match(source, /__constant__ unsigned int c_Table\[3\]\[31\];/u);
  assert.doesNotMatch(source, /unusedTable/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void record_pointer_with_fixed_array_fields(PackedLine *lines, float2 *out, int line, int lane) {
  float2 value = lines[line].CP[lane];
  lines[line].target[lane] = value;
  out[lane] = value;
}`,
    recordDeclarations: [
      `struct PackedLine {
  float2 CP[3];
  float2 *target;
  int n;
};`,
    ],
  });
  assert.match(source, /__global__ void record_pointer_with_fixed_array_fields\(float2 \*lines__CP, float2 \*lines__target, int \*lines__n, float2 \*out, int line, int lane\)/u);
  assert.match(source, /float2 value = lines__CP\[\(line\) \* 3 \+ \(lane\)\];/u);
  assert.match(source, /lines__target\[lane\] = value;/u);
  assert.doesNotMatch(source, /\bPackedLine\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename Tensor>
__global__ void scale_rows_kernel(Tensor tensor) {
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < static_cast<int>(tensor.extent(0)) && c < static_cast<int>(tensor.extent(1))) {
    tensor(r, c) *= static_cast<float>(r + 1);
  }
}`,
  });
  assert.match(source, /__global__ void scale_rows_kernel\(float \*tensor, int tensor_extent0, int tensor_extent1, int tensor_stride0, int tensor_stride1\)/u);
  assert.match(source, /tensor\[\(r\) \* tensor_stride0 \+ \(c\) \* tensor_stride1\] \*= static_cast<float>\(r \+ 1\);/u);
  assert.doesNotMatch(source, /\btemplate\s*</u);
  assert.doesNotMatch(source, /tensor\.extent/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <typename InTensor, typename OutTensor>
__global__ void shared_tile_transpose_kernel(InTensor in, OutTensor out) {
  __shared__ float smem_storage[8 * 8];
  cuda::shared_memory_mdspan smem(smem_storage, cuda::std::dextents<cuda::std::size_t, 2>{8, 8});
  const int tr = threadIdx.y;
  const int tc = threadIdx.x;
  const int r = blockIdx.y * 8 + tr;
  const int c = blockIdx.x * 8 + tc;
  if (r < static_cast<int>(in.extent(0)) && c < static_cast<int>(in.extent(1))) {
    smem(tr, tc) = in(r, c);
  }
  __syncthreads();
  if (r < static_cast<int>(out.extent(0)) && c < static_cast<int>(out.extent(1))) {
    out(r, c) = smem(tc, tr);
  }
}`,
  });
  assert.match(source, /__global__ void shared_tile_transpose_kernel\(const float \*in, int in_extent0, int in_extent1, int in_stride0, int in_stride1, float \*out, int out_extent0, int out_extent1, int out_stride0, int out_stride1\)/u);
  assert.match(source, /smem_storage\[\(tr\) \* 8 \+ \(tc\) \* 1\] = in\[\(r\) \* in_stride0 \+ \(c\) \* in_stride1\];/u);
  assert.match(source, /out\[\(r\) \* out_stride0 \+ \(c\) \* out_stride1\] = smem_storage\[\(tc\) \* 8 \+ \(tr\) \* 1\];/u);
  assert.doesNotMatch(source, /shared_memory_mdspan/u);
  assert.doesNotMatch(source, /\bInTensor\b|\bOutTensor\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
template <const int NUM_THREADS = 128>
__global__ void softmax(float *x, float *y, float *total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  constexpr int NUM_WARPS = (128 + 32 - 1) / 32;
  __shared__ float reduce_smem[NUM_WARPS];

  float sum = (idx < N) ? expf(x[idx]) : 0.0f;
  int warp = tid / 32;
  int lane = tid % 32;
  sum = warp_reduce_sum(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();

  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  sum = warp_reduce_sum(sum);

  if (tid == 0)
    atomicAdd(total, sum);
  __threadfence();

  if (idx < N)
    y[idx] = block_smem[tid] / (*total);
}`,
  });
  assert.match(source, /float bg_thread_numerator = \(idx < N\) \? expf\(x\[idx\]\) : 0\.0f;/u);
  assert.match(source, /float sum = bg_thread_numerator;/u);
  assert.match(source, /y\[idx\] = bg_thread_numerator \/ \(\*total\);/u);
  assert.doesNotMatch(source, /\bblock_smem\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void runtime_alloc_child(float2 *points, int *counts, int n) {
  int idx = threadIdx.x;
  int localN = min(max(n, 4), 32);
  if (points == NULL) {
    counts[idx] = localN;
    cudaMalloc((void **)&points, localN * sizeof(float2));
  }
  child<<<ceilf((float)counts[idx] / 32.0f), 32>>>(points, localN);
}`,
    siblingKernels: [`
__global__ void child(float2 *points, int n) {
  if (threadIdx.x < n) points[threadIdx.x] = make_float2(0.0f, 0.0f);
}`],
  });
  assert.match(source, /counts\[idx\] = localN;/u);
  assert.match(source, /__device__ void bg_inline_child/u);
  assert.match(source, /bg_inline_child\(points, localN,/u);
  assert.doesNotMatch(source, /child<<<ceilf\(\(float\)localN \/ 32\.0f\), 32>>>/u);
  assert.doesNotMatch(source, /\bcudaMalloc\b/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void placement_vector(int *out) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ unsigned char __align__(8) s_buffer[sizeof(Vector<int>)];
  __shared__ int __align__(8) s_data[1024];
  __shared__ Vector<int> *s_vector;
  if (threadIdx.x == 0) s_vector = new (s_buffer) Vector<int>(1024, s_data);
  cg::sync(cta);
  s_vector->push(threadIdx.x);
  int v;
  if (s_vector->pop(v)) out[threadIdx.x] = v;
}`,
  });
  assert.match(source, /__device__ void bg_vector_push_int/u);
  assert.match(source, /__shared__ int s_vector_top\[1\];/u);
  assert.match(source, /bg_vector_push_int\(s_vector_top, s_data, int\(threadIdx\.x\)\);/u);
  assert.match(source, /bg_vector_pop_int\(s_vector_top, s_data, &v\)/u);
  assert.doesNotMatch(source, /\bVector\s*</u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void random_kernel(unsigned long long seed, float *uniform_out, int *poisson_out) {
  cuda::pcg64 rng(seed + threadIdx.x);
  cuda::std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
  cuda::std::poisson_distribution<int> poisson_dist(4.0);
  uniform_out[threadIdx.x] = uniform_dist(rng);
  poisson_out[threadIdx.x] = poisson_dist(rng);
}`,
  });
  assert.match(source, /__device__ float bg_random_uniform/u);
  assert.match(source, /uint rng = uint\(seed \+ threadIdx\.x\);/u);
  assert.match(source, /uniform_out\[threadIdx\.x\] = bg_random_uniform\(&rng\);/u);
  assert.match(source, /poisson_out\[threadIdx\.x\] = bg_random_poisson4\(&rng\);/u);
  assert.doesNotMatch(source, /cuda::std::uniform_real_distribution/u);
}
