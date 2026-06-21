# CS336 Assignment 4 Data Handoff

This note captures the browser-safe path for
`stanford-cs336/assignment4-data`. Keep it as an assignment profile record, not
root agent behavior.

## Upstream Shape

- CS336 Spring 2026 describes Assignment 4 as converting raw Common Crawl dumps
  into usable pretraining data, then filtering and deduplicating to improve LM
  performance.
- The upstream repository keeps student work in `cs336_data`, uses `uv`, and
  downloads offline/shared data through scripts. Full data and Modal training
  paths are platform/external concerns, not first browser fixture targets.

## Browser-Safe First Slice

- Use `docs/internal/cs336-assignment4-data.profile.json` as source profile.
- Use small checked-in HTML/text fixtures declared as datasets.
- Keep `large-file-streaming` as behavior under test; do not replace it with
  Linux RSS or process-resource checks.
- Register a JS oracle module from `@unlocalhosted/browsergrad-data` for:
  - `extractVisibleTextFromHtml()` for HTML fixture text extraction.
  - `exactLineDeduplicate()` for exact line dedupe checks.
  - `minhashDeduplicateDocuments()` for fixture-scale exact/fuzzy document
    dedupe checks.
  - `maskPii()` for email, phone, and IP masking checks.
  - `evaluateGopherQuality()` / `gopherQualityFilter()` for deterministic
    Gopher-style quality rules.
- Use `@unlocalhosted/browsergrad-snapshots` when expected data-cleaning output
  is easiest to store as JSON.

## Non-Portable Upstream Assumptions

- Full Common Crawl / WET / WARC-scale data access.
- Modal/shared-data paths.
- Native or large external classifiers such as fastText and transformers.
- Training the final language model on full filtered data inside the browser.

## Platform Work

- Fetch/cache tiny fixtures, then call
  `createVerifiedAssignmentBenchmarkPreflightMatrix()` before mounting.
- Route `pii-oracle`, `dedupe-oracle`, `near-dedupe-oracle`, and
  `quality-rule-oracle` to `@unlocalhosted/browsergrad-data`.
- Route `classifier-oracle` to a later deterministic fixture classifier or to
  explicit external classifier capabilities.
- Render failures as data-specific rubric messages, for example:
  - `email was not masked`
  - `duplicate line survived`
  - `near duplicate document survived`
  - `Gopher quality rule failed`
  - `script/style text leaked from HTML extraction`
  - `student consumed corpus eagerly`

## Later Slices

- Add fixture classifier oracle for language/quality/toxicity labels.
- Add WARC/WET fixture parser only after exact fixture tests stabilize.
