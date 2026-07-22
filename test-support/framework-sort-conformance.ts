export const FRAMEWORK_SORT_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework.axis-sort-conformance.v1",
  valid: {
    inputValues: [
      [3, 1, 1, 2],
      [-1, 4, 4, 0],
    ],
    dim: -1,
    ascendingValues: [
      [1, 1, 2, 3],
      [-1, 0, 4, 4],
    ],
    ascendingIndices: [
      [1, 2, 3, 0],
      [0, 3, 1, 2],
    ],
    descendingValues: [
      [3, 2, 1, 1],
      [4, 4, 0, -1],
    ],
    descendingIndices: [
      [0, 3, 1, 2],
      [1, 2, 3, 0],
    ],
    cotangent: [
      [10, 20, 30, 40],
      [50, 60, 70, 80],
    ],
    expectedGradient: [
      [40, 10, 20, 30],
      [50, 70, 80, 60],
    ],
  },
  dtypeCases: [
    {
      dtype: "float16",
      input: [2, -1, 0],
      values: [-1, 0, 2],
      indices: [1, 2, 0],
    },
    {
      dtype: "int64",
      input: [2, -1, 0],
      values: [-1, 0, 2],
      indices: [1, 2, 0],
    },
    {
      dtype: "bool",
      input: [true, false, true],
      values: [false, true, true],
      indices: [1, 0, 2],
    },
    {
      dtype: "uint8",
      input: [2, 1, 0],
      values: [0, 1, 2],
      indices: [2, 1, 0],
    },
  ],
  scalar: { value: 7, expectedIndex: 0 },
  empty: { inputShape: [2, 0], expectedShape: [2, 0] },
  vmap: {
    inputValues: [
      [3, 1, 2],
      [0, -1, 4],
    ],
    expectedValues: [
      [1, 2, 3],
      [-1, 0, 4],
    ],
    expectedIndices: [
      [1, 2, 0],
      [1, 0, 2],
    ],
  },
  onnx: {
    valuesOpTypes: ["TopK", "GatherElements", "Identity"],
    indicesOpTypes: ["TopK", "Identity"],
    k: 4,
    axis: 1,
    largest: 0,
    sorted: 1,
    topkOutputCount: 2,
  },
  invalid: [
    { id: "non-tensor" },
    { id: "bool-dim" },
    { id: "float-dim" },
    { id: "out-of-range-dim" },
    { id: "descending-type" },
    { id: "stable-type" },
    { id: "out-mutation" },
    { id: "unsupported-dtype" },
    { id: "hostile-dim" },
    { id: "hostile-descending" },
    { id: "hostile-stable" },
    { id: "zero-size-oversized-axis" },
  ],
} as const);
