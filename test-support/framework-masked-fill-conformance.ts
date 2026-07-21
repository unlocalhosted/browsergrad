export const FRAMEWORK_MASKED_FILL_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework-masked-fill-conformance@1",
  valid: Object.freeze({
    inputShape: Object.freeze([2, 3]),
    inputValues: Object.freeze([1, 2, 3, 4, 5, 6]),
    maskShape: Object.freeze([3]),
    maskValues: Object.freeze([true, false, true]),
    fillValue: -1,
    outputValues: Object.freeze([-1, 2, -1, -1, 5, -1]),
    sourceGradient: Object.freeze([0, 1, 0, 0, 1, 0]),
  }),
  dtypeCases: Object.freeze([
    Object.freeze({ dtype: "float16", expectedDtype: "float16" }),
    Object.freeze({ dtype: "float32", expectedDtype: "float32" }),
    Object.freeze({ dtype: "float64", expectedDtype: "float64" }),
    Object.freeze({ dtype: "int32", expectedDtype: "int32" }),
    Object.freeze({ dtype: "int64", expectedDtype: "int64" }),
    Object.freeze({ dtype: "bool", expectedDtype: "bool", fillValue: false }),
  ]),
  invalid: Object.freeze([
    Object.freeze({
      id: "non-tensor-mask",
      kind: "mask-value",
      value: Object.freeze([true, false, true]),
      message: "mask must be a Tensor",
    }),
    Object.freeze({
      id: "non-bool-mask",
      kind: "mask-dtype",
      value: "int32",
      message: "mask dtype must be bool",
    }),
    Object.freeze({
      id: "non-broadcast-mask",
      kind: "mask-shape",
      value: Object.freeze([2, 2]),
      message: "cannot broadcast to source shape",
    }),
    Object.freeze({
      id: "expanding-mask",
      kind: "mask-shape",
      value: Object.freeze([2, 2, 3]),
      message: "cannot broadcast to source shape",
    }),
    Object.freeze({
      id: "boolean-numeric-fill",
      kind: "fill",
      value: true,
      message: "value must be a built-in or NumPy real scalar",
    }),
    Object.freeze({
      id: "fractional-integer-fill",
      kind: "integer-fill",
      value: 1.5,
      message: "not an exact finite integer",
    }),
    Object.freeze({
      id: "overflow-integer-fill",
      kind: "integer-fill",
      value: 2147483648,
      message: "out of range for int32",
    }),
  ]),
});
