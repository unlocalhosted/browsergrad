export const FRAMEWORK_INTERPOLATE_2D_CONFORMANCE = {
  schema: "browsergrad.framework.interpolate-2d-conformance.v1",
  input: [[[[1, 3], [5, 7]]]],
  nearest: {
    size: [3, 5],
    cotangent: [[[
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.6, 0.7, 0.8, 0.9, 1],
      [1.1, 1.2, 1.3, 1.4, 1.5],
    ]]],
    expected: [[[
      [1, 1, 1, 3, 3],
      [1, 1, 1, 3, 3],
      [5, 5, 5, 7, 7],
    ]]],
    gradient: [[[[2.7, 2.8], [3.6, 2.9]]]],
  },
  bilinear: {
    scaleFactor: 2,
    cotangent: [[[
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1, 1.1, 1.2],
      [1.3, 1.4, 1.5, 1.6],
    ]]],
    expected: [[[
      [1, 1.5, 2.5, 3],
      [2, 2.5, 3.5, 4],
      [4, 4.5, 5.5, 6],
      [5, 5.5, 6.5, 7],
    ]]],
    gradient: [[[[1.65, 2.35], [4.45, 5.15]]]],
  },
  alignCorners: {
    size: [3, 3],
    cotangent: [[[
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
      [0.7, 0.8, 0.9],
    ]]],
    expected: [[[
      [1, 2, 3],
      [3, 4, 5],
      [5, 6, 7],
    ]]],
    gradient: [[[[0.525, 0.825], [1.425, 1.725]]]],
  },
  nonIntegralScale: {
    input: [[[
      [0, 1, 2, 3, 4],
      [5, 6, 7, 8, 9],
      [10, 11, 12, 13, 14],
    ]]],
    scaleFactor: [1.6, 1.5],
    expectedShape: [1, 1, 4, 7],
    withoutRecomputeFirstRow: [
      0, 0.5, 1.1666666667, 1.8333333333, 2.5, 3.1666666667, 3.8333333333,
    ],
    withoutRecomputeLastRow: [
      8.4375, 8.9375, 9.6041666667, 10.2708333333, 10.9375,
      11.6041666667, 12.2708333333,
    ],
    withRecomputeFirstRow: [
      0, 0.5714285714, 1.2857142857, 2, 2.7142857143, 3.4285714286, 4,
    ],
    withRecomputeLastRow: [
      10, 10.5714285714, 11.2857142857, 12, 12.7142857143,
      13.4285714286, 14,
    ],
  },
  limits: {
    outputBytes: 1 << 28,
    outputExtent: 1 << 20,
    workElements: 1 << 28,
    workspaceBytes: 1 << 28,
    nearestWorkVisitFactor: 4,
    bilinearWorkVisitFactor: 32,
  },
} as const;
