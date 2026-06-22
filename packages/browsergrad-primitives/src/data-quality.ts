import type { GopherQualityOptions, GopherQualityReport, GopherQualityRule } from "./data-types.js";

export function evaluateGopherQuality(
  text: string,
  options: GopherQualityOptions = {},
): GopherQualityReport {
  const minNonSymbolWords = options.minNonSymbolWords ?? 50;
  const maxNonSymbolWords = options.maxNonSymbolWords ?? 100_000;
  const minAverageWordLength = options.minAverageWordLength ?? 3;
  const maxAverageWordLength = options.maxAverageWordLength ?? 10;
  const maxEllipsisLineRatio = options.maxEllipsisLineRatio ?? 0.3;
  const minAlphabeticWordRatio = options.minAlphabeticWordRatio ?? 0.8;
  const words = text.match(/\S+/gu) ?? [];
  const nonSymbolWords = words
    .map((word) => word.match(/[\p{L}\p{N}]+/gu)?.join("") ?? "")
    .filter((word) => word.length > 0);
  const alphabeticWords = nonSymbolWords.filter((word) => /\p{L}/u.test(word));
  const lines = text.split(/\r?\n/u).filter((line) => line.length > 0);
  const ellipsisLines = lines.filter((line) => {
    const trimmed = line.trimEnd();
    return trimmed.endsWith("...") || trimmed.endsWith("…");
  });
  const averageWordLength =
    nonSymbolWords.length === 0
      ? 0
      : nonSymbolWords.reduce((sum, word) => sum + word.length, 0) /
        nonSymbolWords.length;
  const ellipsisLineRatio = lines.length === 0 ? 0 : ellipsisLines.length / lines.length;
  const alphabeticWordRatio =
    nonSymbolWords.length === 0 ? 0 : alphabeticWords.length / nonSymbolWords.length;
  const failedRules: GopherQualityRule[] = [];

  if (
    nonSymbolWords.length < minNonSymbolWords ||
    nonSymbolWords.length > maxNonSymbolWords
  ) {
    failedRules.push("non_symbol_word_count");
  }
  if (
    averageWordLength < minAverageWordLength ||
    averageWordLength > maxAverageWordLength
  ) {
    failedRules.push("average_word_length");
  }
  if (ellipsisLineRatio > maxEllipsisLineRatio) {
    failedRules.push("ellipsis_line_ratio");
  }
  if (alphabeticWordRatio < minAlphabeticWordRatio) {
    failedRules.push("alphabetic_word_ratio");
  }

  return {
    passed: failedRules.length === 0,
    failedRules,
    metrics: {
      nonSymbolWords: nonSymbolWords.length,
      averageWordLength,
      ellipsisLineRatio,
      alphabeticWordRatio,
    },
  };
}

export function gopherQualityFilter(
  text: string,
  options: GopherQualityOptions = {},
): boolean {
  return evaluateGopherQuality(text, options).passed;
}
