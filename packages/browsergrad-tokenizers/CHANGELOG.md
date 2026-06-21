# @unlocalhosted/browsergrad-tokenizers

## 0.1.0

- Initial browser-safe tokenizer/BPE reference package scaffold.
- Public oracle helpers use generic byte-BPE names:
  `createByteBpeOracleModule()` and `train_byte_bpe()`.
- `createStreamingGate()` exposes `wrapInput()` and `wrapOutput()` helpers for
  JS/TS rubrics that need to catch eager iterable consumption.
