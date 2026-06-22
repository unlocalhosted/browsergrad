import { ScalingApiError } from "./scaling-error.js";
import type { PowerLawFitOptions, PowerLawScalingFit } from "./scaling-types.js";

export function fitPowerLawScalingLaw<T extends Record<string, number>>(
  samples: readonly T[],
  options: PowerLawFitOptions,
): PowerLawScalingFit {
  if (samples.length < 2) {
    throw new ScalingApiError(400, "at least two samples are required");
  }

  const points = samples.map((sample, index) => {
    const x = positiveSampleValue(sample, options.x, index);
    const y = positiveSampleValue(sample, options.y, index);
    return {
      logX: Math.log(x),
      logY: Math.log(y),
    };
  });
  const meanX = mean(points.map((point) => point.logX));
  const meanY = mean(points.map((point) => point.logY));
  let covariance = 0;
  let varianceX = 0;
  for (const point of points) {
    const dx = point.logX - meanX;
    covariance += dx * (point.logY - meanY);
    varianceX += dx * dx;
  }
  if (varianceX === 0) {
    throw new ScalingApiError(400, `${options.x} values must not all match`);
  }
  const slope = covariance / varianceX;
  const intercept = meanY - slope * meanX;
  const predictions = points.map((point) => intercept + slope * point.logX);
  const residualSumSquares = points.reduce((sum, point, index) => {
    const prediction = predictions[index];
    if (prediction === undefined) return sum;
    const residual = point.logY - prediction;
    return sum + residual * residual;
  }, 0);
  const totalSumSquares = points.reduce((sum, point) => {
    const centered = point.logY - meanY;
    return sum + centered * centered;
  }, 0);
  const rSquared =
    totalSumSquares === 0 ? 1 : 1 - residualSumSquares / totalSumSquares;
  const multiplier = Math.exp(intercept);

  return {
    x: options.x,
    y: options.y,
    slope,
    intercept,
    exponent: slope,
    multiplier,
    rSquared,
    predict(x) {
      if (!Number.isFinite(x) || x <= 0) {
        throw new ScalingApiError(400, `${options.x} must be positive`);
      }
      return multiplier * x ** slope;
    },
  };
}

function positiveSampleValue<T extends Record<string, number>>(
  sample: T,
  field: string,
  index: number,
): number {
  const value = sample[field];
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new ScalingApiError(400, `samples[${index}].${field} must be positive`);
  }
  return value;
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
