export const FRAMEWORK_PROD_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework-prod-conformance@1",
  valid: Object.freeze({
    inputShape: Object.freeze([2, 3]),
    inputValues: Object.freeze([2, 3, 4, 5, 6, 7]),
    axis: 1,
    outputShape: Object.freeze([2]),
    outputValues: Object.freeze([24, 210]),
    keepdimsShape: Object.freeze([2, 1]),
    keepdimsValues: Object.freeze([Object.freeze([24]), Object.freeze([210])]),
    fullValue: 5040,
  }),
  dtypeCases: Object.freeze([
    Object.freeze({ dtype: "float16", expectedDtype: "float16" }),
    Object.freeze({ dtype: "int32", expectedDtype: "int32" }),
    Object.freeze({ dtype: "bool", expectedDtype: "bool" }),
  ]),
  invalid: Object.freeze([
    Object.freeze({
      id: "non-integral-axis",
      kind: "axis",
      value: 0.5,
      message: "axis must be None, an integer scalar, or a plain tuple/list",
    }),
    Object.freeze({
      id: "boolean-axis",
      kind: "axis",
      value: true,
      message: "axis must be None, an integer scalar, or a plain tuple/list",
    }),
    Object.freeze({
      id: "empty-axes",
      kind: "axis",
      value: Object.freeze([]),
      message: "axis sequence must be non-empty",
    }),
    Object.freeze({
      id: "duplicate-axes",
      kind: "axis",
      value: Object.freeze([0, -2]),
      message: "axes must be unique after normalization",
    }),
    Object.freeze({
      id: "axis-out-of-range",
      kind: "axis",
      value: 2,
      message: "axis 2 out of range for rank 2",
    }),
    Object.freeze({
      id: "both-aliases",
      kind: "both",
      value: 0,
      message: "specify only one of axis or dim",
    }),
    Object.freeze({
      id: "non-boolean-keepdims",
      kind: "keepdims",
      value: 1,
      message: "keepdim and keepdims must be booleans",
    }),
    Object.freeze({
      id: "scalar-axis",
      kind: "scalar",
      value: 0,
      message: "axis 0 out of range for rank 0",
    }),
  ]),
});
