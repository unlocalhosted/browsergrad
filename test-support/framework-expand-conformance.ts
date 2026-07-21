export const FRAMEWORK_EXPAND_CONFORMANCE = Object.freeze({
  schema: "browsergrad.framework-expand-conformance@1",
  valid: Object.freeze({
    inputShape: Object.freeze([2, 1]),
    inputValues: Object.freeze([1, 2]),
    requestedShape: Object.freeze([-1, 3]),
    outputShape: Object.freeze([2, 3]),
    outputValues: Object.freeze([[1, 1, 1], [2, 2, 2]]),
  }),
  dtypeCases: Object.freeze([
    Object.freeze({ dtype: "float16", expectedDtype: "float16" }),
    Object.freeze({ dtype: "int32", expectedDtype: "int32" }),
  ]),
  invalid: Object.freeze([
    Object.freeze({
      id: "non-integral",
      inputShape: Object.freeze([2, 1]),
      requestedShape: Object.freeze([2, 3.5]),
      message: "must be an integer",
    }),
    Object.freeze({
      id: "boolean",
      inputShape: Object.freeze([2, 1]),
      requestedShape: Object.freeze([2, true]),
      message: "must be an integer",
    }),
    Object.freeze({
      id: "leading-minus-one",
      inputShape: Object.freeze([2]),
      requestedShape: Object.freeze([-1, 2]),
      message: "-1 is not allowed in a new leading dimension",
    }),
    Object.freeze({
      id: "negative",
      inputShape: Object.freeze([2, 1]),
      requestedShape: Object.freeze([2, -2]),
      message: "must be non-negative or -1",
    }),
    Object.freeze({
      id: "incompatible",
      inputShape: Object.freeze([2, 1]),
      requestedShape: Object.freeze([3, 3]),
      message: "cannot expand to 3",
    }),
    Object.freeze({
      id: "rank-reduction",
      inputShape: Object.freeze([2, 1]),
      requestedShape: Object.freeze([2]),
      message: "has fewer dims than input",
    }),
  ]),
});
