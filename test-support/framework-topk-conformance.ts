export const FRAMEWORK_TOPK_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework.axis-topk-conformance.v1",
  valid: {
    inputValues: [
      [3, 1, 6, 2, 5, 4],
      [-2, 8, 0, 7, 1, 9],
    ],
    k: 3,
    dim: -1,
    largestValues: [
      [6, 5, 4],
      [9, 8, 7],
    ],
    largestIndices: [
      [2, 4, 5],
      [5, 1, 3],
    ],
    smallestValues: [
      [1, 2],
      [-2, 0],
    ],
    smallestIndices: [
      [1, 3],
      [0, 2],
    ],
    cotangent: [
      [10, 20, 30],
      [40, 50, 60],
    ],
    expectedGradient: [
      [0, 0, 10, 0, 20, 30],
      [0, 50, 0, 60, 0, 40],
    ],
  },
  dtypeCases: [
    { dtype: "float16", input: [2, -1, 4, 0], values: [4, 2, 0, -1] },
    { dtype: "int64", input: [2, -1, 4, 0], values: [4, 2, 0, -1] },
    { dtype: "uint8", input: [2, 1, 4, 0], values: [4, 2, 1, 0] },
    { dtype: "bool", input: [true, false, true, false], values: [true, true, false, false] },
  ],
  empty: { inputShape: [2, 0], k: 0, expectedShape: [2, 0] },
  zeroK: { inputShape: [2, 4], expectedShape: [2, 0] },
  unsorted: {
    input: [9, 1, 7, 3, 5],
    k: 3,
    expectedValueSet: [1, 3, 5],
    expectedIndexSet: [1, 3, 4],
  },
  vmap: {
    inputValues: [
      [3, 1, 6, 2],
      [0, -1, 4, 2],
    ],
    k: 2,
    expectedValues: [
      [6, 3],
      [4, 2],
    ],
    expectedIndices: [
      [2, 0],
      [2, 3],
    ],
  },
  onnx: {
    valuesOpTypes: ["TopK", "GatherElements", "Identity"],
    indicesOpTypes: ["TopK", "Identity"],
    k: 3,
    axis: 1,
    largest: 1,
    sorted: 1,
    topkOutputCount: 2,
  },
  invalid: [
    { id: "non-tensor" },
    { id: "bool-k" },
    { id: "float-k" },
    { id: "negative-k" },
    { id: "oversized-k" },
    { id: "bool-dim" },
    { id: "float-dim" },
    { id: "out-of-range-dim" },
    { id: "largest-type" },
    { id: "sorted-type" },
    { id: "out-mutation" },
    { id: "unsupported-dtype" },
    { id: "scalar-input" },
    { id: "hostile-k" },
    { id: "hostile-dim" },
    { id: "hostile-largest" },
    { id: "hostile-sorted" },
    { id: "zero-size-oversized-axis" },
    { id: "zero-size-oversized-extent" },
  ],
} as const);
