import { SimulatorError } from "./simulation-error.js";

export function validateRankCount(ranks: number): number {
  if (!Number.isInteger(ranks) || ranks <= 0) {
    throw new SimulatorError("ranks must be a positive integer");
  }
  return ranks;
}

export function allRanks(rankCount: number): number[] {
  return Array.from({ length: rankCount }, (_, rank) => rank);
}
