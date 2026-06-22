import { extractVisibleTextFromHtml } from "./data-html.js";
import { exactLineDeduplicate, minhashDeduplicateDocuments } from "./data-dedupe.js";
import { evaluateGopherQuality, gopherQualityFilter } from "./data-quality.js";
import { maskPii } from "./data-pii.js";

export type * from "./data-types.js";
export { exactLineDeduplicate, minhashDeduplicateDocuments } from "./data-dedupe.js";
export { extractVisibleTextFromHtml } from "./data-html.js";
export { evaluateGopherQuality, gopherQualityFilter } from "./data-quality.js";
export { maskPii } from "./data-pii.js";

export interface DataCleaningReference {
  readonly extractVisibleTextFromHtml: typeof extractVisibleTextFromHtml;
  readonly exactLineDeduplicate: typeof exactLineDeduplicate;
  readonly minhashDeduplicateDocuments: typeof minhashDeduplicateDocuments;
  readonly maskPii: typeof maskPii;
  readonly evaluateGopherQuality: typeof evaluateGopherQuality;
  readonly gopherQualityFilter: typeof gopherQualityFilter;
}

const defaultDataCleaningReference: DataCleaningReference = Object.freeze({
  extractVisibleTextFromHtml,
  exactLineDeduplicate,
  minhashDeduplicateDocuments,
  maskPii,
  evaluateGopherQuality,
  gopherQualityFilter,
});

export function createDataCleaningReference(): DataCleaningReference {
  return defaultDataCleaningReference;
}
