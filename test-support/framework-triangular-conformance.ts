const INPUT_SHAPE = Object.freeze([2, 2, 3]);
const INPUT_VALUES = Object.freeze([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const DTYPE_CASES = Object.freeze([
  Object.freeze({ dtype: "float16", expectedDtype: "float16" }),
  Object.freeze({ dtype: "float32", expectedDtype: "float32" }),
  Object.freeze({ dtype: "float64", expectedDtype: "float64" }),
  Object.freeze({ dtype: "int32", expectedDtype: "int32" }),
  Object.freeze({ dtype: "int64", expectedDtype: "int64" }),
  Object.freeze({ dtype: "bool", expectedDtype: "bool" }),
]);

function invalidCases(operation: "tril" | "triu") {
  return Object.freeze([
    Object.freeze({
      id: "rank-one-input",
      kind: "rank",
      value: Object.freeze([3]),
      message: "rank at least two",
    }),
    Object.freeze({
      id: "boolean-diagonal",
      kind: "diagonal",
      value: true,
      message: "diagonal must be a built-in or NumPy integer scalar",
    }),
    Object.freeze({
      id: "floating-diagonal",
      kind: "diagonal",
      value: 1.5,
      message: "diagonal must be a built-in or NumPy integer scalar",
    }),
    Object.freeze({
      id: "string-diagonal",
      kind: "diagonal",
      value: "1",
      message: "diagonal must be a built-in or NumPy integer scalar",
    }),
    Object.freeze({
      id: "unsupported-dtype",
      kind: "dtype",
      value: "uint16",
      message: `${operation}: dtype 'uint16' is not supported`,
    }),
  ]);
}

export const FRAMEWORK_TRIL_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework-tril-conformance@1",
  valid: Object.freeze({
    inputShape: INPUT_SHAPE,
    inputValues: INPUT_VALUES,
    diagonal: -1,
    outputValues: Object.freeze([0, 0, 0, 4, 0, 0, 0, 0, 0, 10, 0, 0]),
    sourceGradient: Object.freeze([0, 0, 0, 4, 0, 0, 0, 0, 0, 10, 0, 0]),
    mainDiagonalValues: Object.freeze([1, 0, 0, 4, 5, 0, 7, 0, 0, 10, 11, 0]),
    upperDiagonalValues: Object.freeze([1, 2, 0, 4, 5, 6, 7, 8, 0, 10, 11, 12]),
  }),
  dtypeCases: DTYPE_CASES,
  invalid: invalidCases("tril"),
});

export const FRAMEWORK_TRIU_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework-triu-conformance@1",
  valid: Object.freeze({
    inputShape: INPUT_SHAPE,
    inputValues: INPUT_VALUES,
    diagonal: 1,
    outputValues: Object.freeze([0, 2, 3, 0, 0, 6, 0, 8, 9, 0, 0, 12]),
    sourceGradient: Object.freeze([0, 2, 3, 0, 0, 6, 0, 8, 9, 0, 0, 12]),
    mainDiagonalValues: Object.freeze([1, 2, 3, 0, 5, 6, 7, 8, 9, 0, 11, 12]),
    lowerDiagonalValues: INPUT_VALUES,
  }),
  dtypeCases: DTYPE_CASES,
  invalid: invalidCases("triu"),
});
