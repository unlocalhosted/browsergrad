# CS336 Assignment 1 Handoff

This note captures the browser-safe path for running
`stanford-cs336/assignment1-basics` on BrowserGrad. Keep it as an assignment
profile record, not root agent behavior.

## Portable As-Is

- Pure Python correctness tests that depend only on stdlib, `numpy`, `regex`,
  `pytest`, and BrowserGrad's PyTorch-shaped package surface.
- Fixture comparisons for tokenizer vocabularies, merges, and encode/decode
  output once expected values are mounted locally.
- Small model/math tests that avoid native PyTorch-only features outside the
  supported `grad` or `jit` surface.

## Needs Browser-Safe Rubrics

- `tiktoken` comparisons: replace native Python dependency with a JS tokenizer
  oracle or checked-in expected outputs.
- `psutil` and Linux `resource` checks: replace with behavior gates and worker
  timeouts.
- Multiprocessing guidance: treat as an upstream optimization hint, not a
  BrowserGrad requirement, because Pyodide does not provide process forking.
- Large-corpus timing: use calibrated benchmarks and clear timeout errors
  instead of upstream machine-specific absolute limits.

## Tokenizer/BPE Contract

- Use the GPT-2 pre-tokenizer pattern:
  `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
- Represent pre-tokens as UTF-8 byte sequences.
- Initialize byte-level vocabulary.
- Treat special tokens such as `<|endoftext|>` as hard boundaries.
- Exclude special tokens from BPE merge counts.
- Never merge across pre-token boundaries.
- For equal-frequency byte pairs, pick the lexicographically greater pair.

## Required Fixtures

- Tiny BPE training corpus with hand-verifiable merges.
- Special-token corpus proving no merge crosses document boundaries.
- Medium CS336 fixture with checked-in expected `vocab` and `merges`.
- Encode/decode cases covering ASCII, Unicode, whitespace, punctuation, and
  special tokens.
- Streaming iterable fixture that catches eager full-input consumption.

## Platform Wiring

- Use `@unlocalhosted/browsergrad-tokenizers` as the TS source of truth.
- Register a small JS oracle module into Pyodide for Python rubrics.
- Mount assignment files under `/assignments/cs336-assignment1/`.
- Load only browser-available Pyodide packages in the profile.
- Report rubric failures as assignment failures, for example:
  - `wrong BPE tie-break`
  - `merge crossed special-token boundary`
  - `encode_iterable consumed input eagerly`
  - `student code timed out`

## Non-Portable Upstream Assumptions

- Python `tiktoken` native package availability.
- `psutil` process RSS.
- Linux `resource.setrlimit` behavior.
- `multiprocessing` process workers.
- Machine-specific BPE training time and memory budgets.
