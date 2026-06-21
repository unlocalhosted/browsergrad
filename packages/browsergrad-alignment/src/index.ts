export interface DpoLossInput {
  readonly beta: number;
  readonly policyChosenLogProbability: number;
  readonly policyRejectedLogProbability: number;
  readonly referenceChosenLogProbability: number;
  readonly referenceRejectedLogProbability: number;
}

export interface MmluExample {
  readonly subject?: string;
  readonly question?: string;
  readonly options: readonly string[];
  readonly answer?: string;
}

export interface RolloutRewardComponents {
  readonly reward: number;
  readonly formatReward: number;
  readonly answerReward: number;
}

export interface ComputeRolloutRewardsInput {
  readonly rolloutResponses: readonly string[];
  readonly repeatedGroundTruths: readonly string[];
  readonly rewardFn: (
    response: string,
    groundTruth: string,
  ) => RolloutRewardComponents;
}

export interface RolloutRewardsResult {
  readonly rawRewards: readonly number[];
  readonly metadata: {
    readonly meanReward: number;
    readonly meanFormatReward: number;
    readonly meanAnswerReward: number;
  };
}

export type RewardBaseline = "mean" | "none";
export type AdvantageNormalizer = "std" | "none" | "mean";

export interface GroupNormalizedRewardsInput {
  readonly rawRewards: readonly number[];
  readonly groupSize: number;
  readonly baseline?: RewardBaseline;
  readonly advantageEps?: number;
  readonly advantageNormalizer?: AdvantageNormalizer;
}

export interface GroupNormalizedRewardsResult {
  readonly advantages: readonly number[];
  readonly metadata: {
    readonly meanReward: number;
    readonly meanAdvantage: number;
    readonly rewardStd: number;
    readonly groupSize: number;
  };
}

export type ImportanceReweightingMethod = "none" | "noclip" | "grpo" | "gspo";
export type LossNormalization = "sequence" | "constant";

export interface PolicyGradientLossInput {
  readonly advantages: readonly number[];
  readonly policyLogProbs: readonly (readonly number[])[];
  readonly importanceReweightingMethod?: ImportanceReweightingMethod;
  readonly oldLogProbs?: readonly (readonly number[])[];
  readonly cliprange?: number;
  readonly responseMask?: readonly (readonly number[])[];
}

export interface PolicyGradientLossResult {
  readonly perTokenLoss: readonly (readonly number[])[];
  readonly metadata: {
    readonly clipFraction: number;
    readonly meanImportanceRatio: number;
  };
}

export interface AggregateLossInput {
  readonly perTokenLoss: readonly (readonly number[])[];
  readonly mask: readonly (readonly number[])[];
  readonly lossNormalization?: LossNormalization;
  readonly normalizationConstant?: number;
}

export function computePerInstanceDpoLoss(input: DpoLossInput): number {
  const beta = finiteNumber(input.beta, "beta");
  const policyLogRatio =
    finiteNumber(input.policyChosenLogProbability, "policyChosenLogProbability") -
    finiteNumber(input.policyRejectedLogProbability, "policyRejectedLogProbability");
  const referenceLogRatio =
    finiteNumber(input.referenceChosenLogProbability, "referenceChosenLogProbability") -
    finiteNumber(input.referenceRejectedLogProbability, "referenceRejectedLogProbability");
  const logit = beta * (policyLogRatio - referenceLogRatio);
  return softplus(-logit);
}

export function parseMmluResponse(
  mmluExample: MmluExample,
  modelOutput: string,
): string | null {
  const validLetters = mmluExample.options.map((_, index) =>
    String.fromCharCode("A".charCodeAt(0) + index),
  );
  const valid = new Set(validLetters);
  const patterns = [
    /\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\)?\b/iu,
    /\b([A-Z])\b/u,
  ];

  for (const pattern of patterns) {
    for (const match of modelOutput.matchAll(new RegExp(pattern, "giu"))) {
      const letter = match[1]?.toUpperCase();
      if (letter && valid.has(letter)) return letter;
    }
  }
  return null;
}

export function computeRolloutRewards(
  input: ComputeRolloutRewardsInput,
): RolloutRewardsResult {
  if (input.rolloutResponses.length !== input.repeatedGroundTruths.length) {
    throw new Error("rolloutResponses and repeatedGroundTruths must have equal length");
  }

  const rawRewards: number[] = [];
  const formatRewards: number[] = [];
  const answerRewards: number[] = [];
  for (let index = 0; index < input.rolloutResponses.length; index++) {
    const response = input.rolloutResponses[index];
    const groundTruth = input.repeatedGroundTruths[index];
    if (response === undefined || groundTruth === undefined) continue;
    const components = input.rewardFn(response, groundTruth);
    rawRewards.push(finiteNumber(components.reward, "reward"));
    formatRewards.push(finiteNumber(components.formatReward, "formatReward"));
    answerRewards.push(finiteNumber(components.answerReward, "answerReward"));
  }

  return {
    rawRewards,
    metadata: {
      meanReward: mean(rawRewards),
      meanFormatReward: mean(formatRewards),
      meanAnswerReward: mean(answerRewards),
    },
  };
}

export function computeGroupNormalizedRewards(
  input: GroupNormalizedRewardsInput,
): GroupNormalizedRewardsResult {
  const groupSize = validatePositiveInteger(input.groupSize, "groupSize");
  if (input.rawRewards.length % groupSize !== 0) {
    throw new Error("rawRewards length must be divisible by groupSize");
  }
  const baseline = input.baseline ?? "mean";
  const advantageNormalizer = input.advantageNormalizer ?? "std";
  const advantageEps = input.advantageEps ?? 1e-6;
  const advantages: number[] = [];

  for (let offset = 0; offset < input.rawRewards.length; offset += groupSize) {
    const group = input.rawRewards
      .slice(offset, offset + groupSize)
      .map((value, index) => finiteNumber(value, `rawRewards[${offset + index}]`));
    const groupMean = mean(group);
    const centered = group.map((value) =>
      baseline === "mean" ? value - groupMean : value,
    );
    const denominator = groupNormalizer(group, advantageNormalizer, advantageEps);
    advantages.push(...centered.map((value) => value / denominator));
  }

  return {
    advantages,
    metadata: {
      meanReward: mean(input.rawRewards),
      meanAdvantage: mean(advantages),
      rewardStd: standardDeviation(input.rawRewards),
      groupSize,
    },
  };
}

export function parseGsm8kResponse(modelOutput: string): string | null {
  const matches = [
    ...modelOutput.matchAll(/[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?/gu),
  ];
  const last = matches.at(-1)?.[0];
  return last ? last.replaceAll(",", "") : null;
}

export function computePolicyGradientLoss(
  input: PolicyGradientLossInput,
): PolicyGradientLossResult {
  const method = input.importanceReweightingMethod ?? "none";
  const policyLogProbs = validateMatrix(input.policyLogProbs, "policyLogProbs");
  if (input.advantages.length !== policyLogProbs.length) {
    throw new Error("advantages length must match policyLogProbs batch size");
  }
  const oldLogProbs =
    method === "none" ? undefined : validateRequiredMatrix(input.oldLogProbs, "oldLogProbs");
  if (oldLogProbs) assertSameShape(policyLogProbs, oldLogProbs, "oldLogProbs");
  const responseMask = input.responseMask
    ? validateMatrix(input.responseMask, "responseMask")
    : undefined;
  if (responseMask) assertSameShape(policyLogProbs, responseMask, "responseMask");
  const cliprange =
    method === "grpo" || method === "gspo"
      ? finiteNumber(input.cliprange ?? Number.NaN, "cliprange")
      : undefined;
  const perTokenLoss: number[][] = [];
  let clipped = 0;
  let ratioTotal = 0;
  let ratioCount = 0;

  for (let row = 0; row < policyLogProbs.length; row++) {
    const advantage = finiteNumber(input.advantages[row] ?? Number.NaN, `advantages[${row}]`);
    const rowLoss: number[] = [];
    const sequenceRatio =
      method === "gspo" && oldLogProbs
        ? Math.exp(maskedMeanDelta(policyLogProbs[row] ?? [], oldLogProbs[row] ?? [], responseMask?.[row]))
        : undefined;
    for (let column = 0; column < (policyLogProbs[row]?.length ?? 0); column++) {
      const policyLogProb = policyLogProbs[row]?.[column] ?? 0;
      const ratio = importanceRatio({
        method,
        policyLogProb,
        oldLogProb: oldLogProbs?.[row]?.[column],
        sequenceRatio,
      });
      const { loss, wasClipped } = policyGradientTokenLoss(
        advantage,
        ratio,
        cliprange,
      );
      rowLoss.push(loss);
      ratioTotal += ratio;
      ratioCount++;
      if (wasClipped) clipped++;
    }
    perTokenLoss.push(rowLoss);
  }

  return {
    perTokenLoss,
    metadata: {
      clipFraction: ratioCount === 0 ? 0 : clipped / ratioCount,
      meanImportanceRatio: ratioCount === 0 ? 0 : ratioTotal / ratioCount,
    },
  };
}

export function aggregateLossAcrossMicrobatch(input: AggregateLossInput): number {
  const perTokenLoss = validateMatrix(input.perTokenLoss, "perTokenLoss");
  const mask = validateMatrix(input.mask, "mask");
  assertSameShape(perTokenLoss, mask, "mask");
  const normalization = input.lossNormalization ?? "sequence";

  if (normalization === "constant") {
    const normalizationConstant = finiteNumber(
      input.normalizationConstant ?? Number.NaN,
      "normalizationConstant",
    );
    return maskedSum(perTokenLoss, mask) / normalizationConstant;
  }

  const perSequence: number[] = [];
  for (let row = 0; row < perTokenLoss.length; row++) {
    const rowMask = mask[row] ?? [];
    const rowLoss = perTokenLoss[row] ?? [];
    const rowWeight = rowMask.reduce((sum, value) => sum + value, 0);
    if (rowWeight === 0) continue;
    perSequence.push(
      rowLoss.reduce((sum, value, column) => sum + value * (rowMask[column] ?? 0), 0) /
        rowWeight,
    );
  }
  return mean(perSequence);
}

function groupNormalizer(
  group: readonly number[],
  normalizer: AdvantageNormalizer,
  eps: number,
): number {
  if (normalizer === "none") return 1;
  if (normalizer === "mean") return Math.abs(mean(group)) + eps;
  return standardDeviation(group) + eps;
}

function importanceRatio(input: {
  method: ImportanceReweightingMethod;
  policyLogProb: number;
  oldLogProb: number | undefined;
  sequenceRatio: number | undefined;
}): number {
  if (input.method === "none") return input.policyLogProb;
  if (input.method === "gspo") return finiteNumber(input.sequenceRatio ?? Number.NaN, "sequenceRatio");
  return Math.exp(
    input.policyLogProb - finiteNumber(input.oldLogProb ?? Number.NaN, "oldLogProb"),
  );
}

function policyGradientTokenLoss(
  advantage: number,
  ratioOrLogProb: number,
  cliprange: number | undefined,
): { loss: number; wasClipped: boolean } {
  if (cliprange === undefined) {
    return { loss: -advantage * ratioOrLogProb, wasClipped: false };
  }
  const lower = 1 - cliprange;
  const upper = 1 + cliprange;
  const clippedRatio = Math.min(Math.max(ratioOrLogProb, lower), upper);
  const objective = ratioOrLogProb * advantage;
  const clippedObjective = clippedRatio * advantage;
  const selectedObjective =
    advantage >= 0
      ? Math.min(objective, clippedObjective)
      : Math.max(objective, clippedObjective);
  return {
    loss: -selectedObjective,
    wasClipped: selectedObjective !== objective,
  };
}

function maskedMeanDelta(
  policyRow: readonly number[],
  oldRow: readonly number[],
  maskRow: readonly number[] | undefined,
): number {
  let total = 0;
  let count = 0;
  for (let index = 0; index < policyRow.length; index++) {
    const weight = maskRow?.[index] ?? 1;
    if (weight === 0) continue;
    total += (policyRow[index] ?? 0) - (oldRow[index] ?? 0);
    count += 1;
  }
  return count === 0 ? 0 : total / count;
}

function maskedSum(
  values: readonly (readonly number[])[],
  mask: readonly (readonly number[])[],
): number {
  let total = 0;
  for (let row = 0; row < values.length; row++) {
    const valueRow = values[row] ?? [];
    const maskRow = mask[row] ?? [];
    for (let column = 0; column < valueRow.length; column++) {
      total += (valueRow[column] ?? 0) * (maskRow[column] ?? 0);
    }
  }
  return total;
}

function validateRequiredMatrix(
  matrix: readonly (readonly number[])[] | undefined,
  name: string,
): readonly (readonly number[])[] {
  if (!matrix) throw new Error(`${name} is required`);
  return validateMatrix(matrix, name);
}

function validateMatrix(
  matrix: readonly (readonly number[])[],
  name: string,
): readonly (readonly number[])[] {
  const width = matrix[0]?.length;
  if (width === undefined) return matrix;
  for (let row = 0; row < matrix.length; row++) {
    const current = matrix[row];
    if (!current || current.length !== width) {
      throw new Error(`${name} must be rectangular`);
    }
    for (let column = 0; column < current.length; column++) {
      finiteNumber(current[column] ?? Number.NaN, `${name}[${row}][${column}]`);
    }
  }
  return matrix;
}

function assertSameShape(
  expected: readonly (readonly number[])[],
  actual: readonly (readonly number[])[],
  name: string,
): void {
  if (expected.length !== actual.length) {
    throw new Error(`${name} must match policyLogProbs shape`);
  }
  for (let row = 0; row < expected.length; row++) {
    if ((expected[row]?.length ?? 0) !== (actual[row]?.length ?? 0)) {
      throw new Error(`${name} must match policyLogProbs shape`);
    }
  }
}

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const average = mean(values);
  const variance =
    values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function softplus(value: number): number {
  if (value > 30) return value;
  if (value < -30) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

function finiteNumber(value: number, name: string): number {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite`);
  }
  return value;
}
