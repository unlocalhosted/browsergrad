# @unlocalhosted/browsergrad-data

Browser-safe data processing primitives for BrowserGrad.

This package targets small fixture slices of browser-side data processing. It
is not a Common Crawl replacement; it provides deterministic helpers for PII
masking, exact deduplication, near-duplicate document deduplication,
Gopher-style quality rules, and HTML text extraction without native classifiers
or WARC tooling.

## Public Surface

```ts
import {
  evaluateGopherQuality,
  exactLineDeduplicate,
  extractVisibleTextFromHtml,
  maskPii,
  minhashDeduplicateDocuments,
} from "@unlocalhosted/browsergrad-data";

const pii = maskPii("Email jane@stanford.edu or call 415-555-1212");
console.log(pii.text);

const deduped = exactLineDeduplicate(["alpha", "beta", "alpha"]);
console.log(deduped.keptLines);

const nearDeduped = minhashDeduplicateDocuments([
  { id: "a", text: "permission is hereby granted free of charge" },
  { id: "b", text: "permission is hereby granted free of charge" },
]);
console.log(nearDeduped.duplicates);

console.log(evaluateGopherQuality("high quality words ".repeat(80)).passed);
console.log(extractVisibleTextFromHtml("<p>Hello&nbsp;data</p>"));
```
