# @unlocalhosted/browsergrad-data

Browser-safe data processing oracles for BrowserGrad assignment rubrics.

This package targets small fixture slices of data-processing labs such as
CS336 Assignment 4. It is not a Common Crawl replacement; it gives rubrics
deterministic helpers for PII masking, exact deduplication, and HTML text
extraction without native classifiers or WARC tooling.

## Public Surface

```ts
import {
  exactLineDeduplicate,
  extractVisibleTextFromHtml,
  maskPii,
} from "@unlocalhosted/browsergrad-data";

const pii = maskPii("Email jane@stanford.edu or call 415-555-1212");
console.log(pii.text);

const deduped = exactLineDeduplicate(["alpha", "beta", "alpha"]);
console.log(deduped.keptLines);

console.log(extractVisibleTextFromHtml("<p>Hello&nbsp;data</p>"));
```
