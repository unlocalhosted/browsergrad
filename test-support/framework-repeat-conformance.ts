export const FRAMEWORK_REPEAT_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework-repeat-conformance@1",
  valid: Object.freeze({
    inputShape: Object.freeze([2, 1]),
    inputValues: Object.freeze([1, 2]),
    requestedRepeats: Object.freeze([2, 3]),
    outputShape: Object.freeze([4, 3]),
    outputValues: Object.freeze([
      Object.freeze([1, 1, 1]),
      Object.freeze([2, 2, 2]),
      Object.freeze([1, 1, 1]),
      Object.freeze([2, 2, 2]),
    ]),
  }),
  dtypeCases: Object.freeze([
    Object.freeze({ dtype: "float16", expectedDtype: "float16" }),
    Object.freeze({ dtype: "int32", expectedDtype: "int32" }),
    Object.freeze({ dtype: "bool", expectedDtype: "bool" }),
  ]),
  invalid: Object.freeze([
    Object.freeze({
      id: "empty",
      inputShape: Object.freeze([2, 1]),
      requestedRepeats: Object.freeze([]),
      message: "expected at least one repeat size",
    }),
    Object.freeze({
      id: "non-integral",
      inputShape: Object.freeze([2, 1]),
      requestedRepeats: Object.freeze([2, 3.5]),
      message: "must be a built-in or NumPy integer scalar",
    }),
    Object.freeze({
      id: "boolean",
      inputShape: Object.freeze([2, 1]),
      requestedRepeats: Object.freeze([2, true]),
      message: "must be a built-in or NumPy integer scalar",
    }),
    Object.freeze({
      id: "negative",
      inputShape: Object.freeze([2, 1]),
      requestedRepeats: Object.freeze([2, -1]),
      message: "must be in [0, 1073741824]",
    }),
    Object.freeze({
      id: "rank-reduction",
      inputShape: Object.freeze([2, 1]),
      requestedRepeats: Object.freeze([2]),
      message: "cannot be shorter than input rank",
    }),
    Object.freeze({
      id: "factor-ceiling",
      inputShape: Object.freeze([0]),
      requestedRepeats: Object.freeze([1073741825]),
      message: "must be in [0, 1073741824]",
    }),
  ]),
});
