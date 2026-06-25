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
      compileCodegenMin: 303,
      hardFailMax: 53,
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
      compileCodegenMin: 278,
      hardFailMax: 15,
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
];

export function corpusById(id) {
  const corpus = cudaLiteCorpora.find((candidate) => candidate.id === id);
  if (corpus === undefined) throw new Error(`unknown CUDA-lite corpus id: ${id}`);
  return corpus;
}

export function corpusFixturePath(fixture) {
  return path.join(corpusById(fixture.corpusId).path, fixture.relativePath);
}
