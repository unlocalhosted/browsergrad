export const FRAMEWORK_BATCH_NORM_1D_CONFORMANCE = {
  schema: "browsergrad.framework.batch-norm-1d-conformance.v1",
  eps: 1e-5,
  momentum: 1,
  input2d: [
    [1, 3],
    [5, 7],
  ],
  input3d: [
    [
      [1, 2],
      [3, 4],
    ],
    [
      [5, 6],
      [7, 8],
    ],
  ],
  affineWeight: [1.5, -2],
  affineBias: [0.25, 3],
  upstream2d: [
    [1, 2],
    [3, 5],
  ],
  expectedBatchMean2d: [3, 5],
  expectedBiasedVariance2d: [4, 4],
  expectedUnbiasedRunningVariance2d: [8, 8],
  invalidNumFeatures: [0, -1, true, 1.5],
  invalidEps: [-1, "nan", "inf", true],
  invalidMomentum: [-0.1, 1.1, "nan", "inf", true],
  limits: {
    outputBytes: 1 << 28,
    outputExtent: 1 << 28,
    workElements: 1 << 28,
    workspaceBytes: 1 << 28,
    workVisitFactor: 32,
  },
} as const;
