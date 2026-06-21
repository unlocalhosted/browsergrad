export {
  evaluateGopherQuality,
  exactLineDeduplicate,
  extractVisibleTextFromHtml,
  gopherQualityFilter,
  maskPii,
  minhashDeduplicateDocuments,
  type DeduplicationOptions,
  type DeduplicationResult,
  type DocumentInput,
  type DuplicateLine,
  type GopherQualityOptions,
  type GopherQualityReport,
  type GopherQualityRule,
  type MinHashDeduplicationOptions,
  type MinHashDeduplicationResult,
  type NearDuplicateDocument,
  type PiiMaskResult,
  type PiiKind,
  type PiiSpan,
} from "@unlocalhosted/browsergrad-data";

import {
  evaluateGopherQuality,
  exactLineDeduplicate,
  extractVisibleTextFromHtml,
  maskPii,
  minhashDeduplicateDocuments,
  gopherQualityFilter,
} from "@unlocalhosted/browsergrad-data";

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
