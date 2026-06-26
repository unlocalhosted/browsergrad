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
      compileCodegenMin: 342,
      hardFailMax: 12,
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
      compileCodegenMin: 292,
      hardFailMax: 1,
    },
  },
];

export { cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-fixtures.mjs";
export const cudaLiteCorpusExecutionFixtureBaseline = {
  totalMin: 51,
  expectedOutputMin: 22,
  byCorpusMin: {
    "cuda-120": 2,
    "cuda-samples": 8,
    "llm.c": 12,
    leetcuda: 29,
  },
};

export function corpusById(id) {
  const corpus = cudaLiteCorpora.find((candidate) => candidate.id === id);
  if (corpus === undefined) throw new Error(`unknown CUDA-lite corpus id: ${id}`);
  return corpus;
}

export function corpusFixturePath(fixture) {
  return path.join(corpusById(fixture.corpusId).path, fixture.relativePath);
}
