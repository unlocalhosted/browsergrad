# @unlocalhosted/browsergrad-tokenizers

Browser-safe tokenizer and byte-level BPE reference helpers for BrowserGrad
rubrics.

This package is intentionally independent of Pyodide. Platform code can call it
directly in a worker and expose narrow oracle functions to Python rubrics.

## Public Surface

```ts
import {
  trainByteBpe,
  encodeByteBpe,
  decodeByteBpe,
  createCs336TokenizerOracle,
  createStreamingGate,
} from "@unlocalhosted/browsergrad-tokenizers";
```

The CS336 oracle follows assignment 1 tokenizer rules: GPT-2 pre-tokenization,
byte-level BPE, special-token hard boundaries, no cross-pretoken merges, and
lexicographically greater tie-breaks for equal-frequency pairs.
