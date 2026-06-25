import path from "node:path";

const corpusRoot = process.env.BROWSERGRAD_CUDA_CORPUS_ROOT ??
  "/tmp/browsergrad-corpora";

export const cudaLiteCorpora = [
  {
    id: "cuda-120",
    name: "CUDA-120-DAYS--CHALLENGE",
    repo: "https://github.com/AdepojuJeremy/CUDA-120-DAYS--CHALLENGE.git",
    commit: "fd2987a2b1a4e506629ae9beb22ee6434da2d414",
    path: "/tmp/CUDA-120-DAYS--CHALLENGE",
    expectations: {
      total: 240,
      okMin: 225,
      compileCodegenMin: 240,
      referenceOnlyMax: 0,
      hardFailMax: 0,
    },
  },
  {
    id: "cuda-samples",
    name: "NVIDIA/cuda-samples",
    repo: "https://github.com/NVIDIA/cuda-samples.git",
    commit: "b7c5481c556c3fe98db060207ecaa41a4b9a9abc",
    path: path.join(corpusRoot, "cuda-samples"),
    expectations: {
      total: 357,
      compileCodegenMin: 308,
      hardFailMax: 48,
    },
  },
  {
    id: "llm.c",
    name: "karpathy/llm.c",
    repo: "https://github.com/karpathy/llm.c.git",
    commit: "f1e2ace651495b74ae22d45d1723443fd00ecd3a",
    path: path.join(corpusRoot, "llm.c"),
    expectations: {
      total: 148,
      compileCodegenMin: 148,
      hardFailMax: 0,
    },
  },
  {
    id: "leetcuda",
    name: "xlite-dev/LeetCUDA",
    repo: "https://github.com/xlite-dev/LeetCUDA.git",
    commit: "c5dde9a653d077d71445bcbf822d4bf13672a69e",
    path: path.join(corpusRoot, "LeetCUDA"),
    expectations: {
      total: 293,
      compileCodegenMin: 286,
      hardFailMax: 7,
    },
  },
];

export const cudaLiteCorpusExecutionFixtures = [
  {
    sourceKey: "corpusCuda120VectorAddKernel",
    caseName: "corpus:cuda-120:vectorAddKernel",
    corpusId: "cuda-120",
    relativePath: "daily-updates/day-23-Asynchronous-Memory-Copy.md",
    kernelName: "vectorAddKernel",
    workgroupSize: [8, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    input: {
      buffers: {
        A: { type: "Float32Array", data: [1, 2, 3, 4] },
        B: { type: "Float32Array", data: [10, 20, 30, 40] },
        C: { type: "Float32Array", length: 4 },
      },
      scalars: { N: 4 },
    },
    output: "C",
  },
  {
    sourceKey: "corpusCuda120ScaleVector",
    caseName: "corpus:cuda-120:scaleVector",
    corpusId: "cuda-120",
    relativePath: "daily-updates/day-7-Memory-Model-(Global-Memory-Basics).md",
    kernelName: "scaleVector",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        input: { type: "Float32Array", data: [1, 2, 3, 4] },
        output: { type: "Float32Array", length: 4 },
      },
      scalars: { scale: 2, N: 4 },
    },
    output: "output",
  },
  {
    sourceKey: "corpusCudaSamplesVectorAdd",
    caseName: "corpus:cuda-samples:vectorAdd",
    corpusId: "cuda-samples",
    relativePath: "cpp/0_Introduction/vectorAdd/vectorAdd.cu",
    kernelName: "vectorAdd",
    workgroupSize: [8, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    input: {
      buffers: {
        A: { type: "Float32Array", data: [1, 2, 3, 4] },
        B: { type: "Float32Array", data: [10, 20, 30, 40] },
        C: { type: "Float32Array", length: 4 },
      },
      scalars: { numElements: 4 },
    },
    output: "C",
  },
  {
    sourceKey: "corpusCudaSamplesIncKernel",
    caseName: "corpus:cuda-samples:incKernel",
    corpusId: "cuda-samples",
    relativePath: "cpp/0_Introduction/simpleCallback/simpleCallback.cu",
    kernelName: "incKernel",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        data: { type: "Int32Array", data: [0, 1, 2, 3] },
      },
      scalars: { N: 4 },
    },
    output: "data",
  },
  {
    sourceKey: "corpusCudaSamplesKernelAddConstant",
    caseName: "corpus:cuda-samples:kernelAddConstant",
    corpusId: "cuda-samples",
    relativePath: "cpp/0_Introduction/cudaOpenMP/cudaOpenMP.cu",
    kernelName: "kernelAddConstant",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        g_a: { type: "Int32Array", data: [0, 1, 2, 3] },
      },
      scalars: { b: 5 },
    },
    output: "g_a",
  },
  {
    sourceKey: "corpusLlmAddBias",
    caseName: "corpus:llm.c:add_bias",
    corpusId: "llm.c",
    relativePath: "dev/cuda/matmul_forward.cu",
    kernelName: "add_bias",
    workgroupSize: [8, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    input: {
      buffers: {
        out: { type: "Float32Array", data: [1, 2, 3, 4, 5, 6] },
        bias: { type: "Float32Array", data: [0.5, 1.5, -2] },
      },
      scalars: { B: 1, T: 2, OC: 3 },
    },
    output: "out",
  },
  {
    sourceKey: "corpusLlmSetVector",
    caseName: "corpus:llm.c:set_vector",
    corpusId: "llm.c",
    relativePath: "dev/cuda/nccl_all_reduce.cu",
    kernelName: "set_vector",
    workgroupSize: [8, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    input: {
      buffers: {
        data: { type: "Float32Array", length: 4 },
      },
      scalars: { N: 4, value: 7 },
    },
    output: "data",
  },
  {
    sourceKey: "corpusLeetCudaElementwiseAddF32",
    caseName: "corpus:LeetCUDA:elementwise_add_f32_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/elementwise/elementwise.cu",
    kernelName: "elementwise_add_f32_kernel",
    workgroupSize: [8, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    input: {
      buffers: {
        a: { type: "Float32Array", data: [1, 2, 3, 4] },
        b: { type: "Float32Array", data: [10, 20, 30, 40] },
        c: { type: "Float32Array", length: 4 },
      },
      scalars: { N: 4 },
    },
    output: "c",
  },
  {
    sourceKey: "corpusLeetCudaReluF32",
    caseName: "corpus:LeetCUDA:relu_f32_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/relu/relu.cu",
    kernelName: "relu_f32_kernel",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        x: { type: "Float32Array", data: [-2, 0, 3, -4] },
        y: { type: "Float32Array", length: 4 },
      },
      scalars: { N: 4 },
    },
    output: "y",
  },
  {
    sourceKey: "corpusLeetCudaSigmoidF32",
    caseName: "corpus:LeetCUDA:sigmoid_f32_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/sigmoid/sigmoid.cu",
    kernelName: "sigmoid_f32_kernel",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        x: { type: "Float32Array", data: [-2, 0, 2, 4] },
        y: { type: "Float32Array", length: 4 },
      },
      scalars: { N: 4 },
    },
    output: "y",
  },
  {
    sourceKey: "corpusLeetCudaEmbeddingF32",
    caseName: "corpus:LeetCUDA:embedding_f32_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/embedding/embedding.cu",
    kernelName: "embedding_f32_kernel",
    workgroupSize: [3, 1, 1],
    launch: { gridDim: [2, 1, 1], blockDim: [3, 1, 1] },
    input: {
      buffers: {
        idx: { type: "Int32Array", data: [1, 0] },
        weight: { type: "Float32Array", data: [10, 11, 12, 20, 21, 22] },
        output: { type: "Float32Array", length: 6 },
      },
      scalars: { n: 2, emb_size: 3 },
    },
    output: "output",
  },
  {
    sourceKey: "corpusLlmGeluForward1",
    caseName: "corpus:llm.c:gelu_forward_kernel1",
    corpusId: "llm.c",
    relativePath: "dev/cuda/gelu_forward.cu",
    kernelName: "gelu_forward_kernel1",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        out: { type: "Float32Array", length: 4 },
        inp: { type: "Float32Array", data: [-1, 0, 1, 2] },
      },
      scalars: { N: 4 },
    },
    output: "out",
  },
  {
    sourceKey: "corpusLlmResidualForward1",
    caseName: "corpus:llm.c:residual_forward_kernel1",
    corpusId: "llm.c",
    relativePath: "dev/cuda/residual_forward.cu",
    kernelName: "residual_forward_kernel1",
    workgroupSize: [4, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    input: {
      buffers: {
        out: { type: "Float32Array", length: 4 },
        inp1: { type: "Float32Array", data: [1, 2, 3, 4] },
        inp2: { type: "Float32Array", data: [10, 20, 30, 40] },
      },
      scalars: { N: 4 },
    },
    output: "out",
  },
  {
    sourceKey: "corpusLlmLayernormNormalization",
    caseName: "corpus:llm.c:normalization_kernel",
    corpusId: "llm.c",
    relativePath: "dev/cuda/layernorm_forward.cu",
    kernelName: "normalization_kernel",
    workgroupSize: [6, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [6, 1, 1] },
    input: {
      buffers: {
        out: { type: "Float32Array", length: 6 },
        inp: { type: "Float32Array", data: [1, 2, 3, 4, 5, 6] },
        mean: { type: "Float32Array", data: [2, 5] },
        rstd: { type: "Float32Array", data: [1, 0.5] },
        weight: { type: "Float32Array", data: [1, 2, 3] },
        bias: { type: "Float32Array", data: [0, 10, 100] },
      },
      scalars: { B: 1, T: 2, C: 3 },
    },
    output: "out",
  },
  {
    sourceKey: "corpusLeetCudaCuteTransposeReg",
    caseName: "corpus:LeetCUDA:mat_transpose_cute_reg_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/mat-transpose/mat_transpose_cute.cu",
    kernelName: "mat_transpose_cute_reg_kernel",
    workgroupSize: [64, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
    input: {
      buffers: {
        pA: { type: "Float32Array", data: [1, 2, 3, 4, 5, 6] },
        pB: { type: "Float32Array", length: 6 },
      },
      scalars: { M: 2, N: 3 },
    },
    output: "pB",
  },
  {
    sourceKey: "corpusLeetCudaCuteTransposeSmem",
    caseName: "corpus:LeetCUDA:mat_transpose_cute_smem_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/mat-transpose/mat_transpose_cute.cu",
    kernelName: "mat_transpose_cute_smem_kernel",
    workgroupSize: [64, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
    input: {
      buffers: {
        pA: { type: "Float32Array", data: [1, 2, 3, 4, 5, 6] },
        pB: { type: "Float32Array", length: 6 },
      },
      scalars: { M: 2, N: 3 },
    },
    output: "pB",
  },
  {
    sourceKey: "corpusLeetCudaCuteTransposeSmemVectorized",
    caseName: "corpus:LeetCUDA:mat_transpose_cute_smem_vectorized_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/mat-transpose/mat_transpose_cute.cu",
    kernelName: "mat_transpose_cute_smem_vectorized_kernel",
    workgroupSize: [64, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
    input: {
      buffers: {
        pA: { type: "Float32Array", data: [1, 2, 3, 4, 5, 6] },
        pB: { type: "Float32Array", length: 6 },
      },
      scalars: { M: 2, N: 3 },
    },
    output: "pB",
  },
  {
    sourceKey: "corpusLeetCudaCuteTransposeSmemVectorizedOptimized",
    caseName: "corpus:LeetCUDA:mat_transpose_cute_smem_vectorized_optimized_kernel",
    corpusId: "leetcuda",
    relativePath: "kernels/mat-transpose/mat_transpose_cute.cu",
    kernelName: "mat_transpose_cute_smem_vectorized_optimized_kernel",
    workgroupSize: [64, 1, 1],
    launch: { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
    input: {
      buffers: {
        pA: { type: "Float32Array", data: [1, 2, 3, 4, 5, 6] },
        pB: { type: "Float32Array", length: 6 },
      },
      scalars: { M: 2, N: 3 },
    },
    output: "pB",
  },
];

export function corpusById(id) {
  const corpus = cudaLiteCorpora.find((candidate) => candidate.id === id);
  if (corpus === undefined) throw new Error(`unknown CUDA-lite corpus id: ${id}`);
  return corpus;
}

export function corpusFixturePath(fixture) {
  return path.join(corpusById(fixture.corpusId).path, fixture.relativePath);
}
