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

`createStreamingGate()` is a browser-safe behavior gate for JS/TS rubrics. Use
`gate.wrapInput(chunks)` as the iterable passed to student code and
`gate.wrapOutput(output)` around the student iterable. Eager consumers fail when
they consume more than `maxChunksBeforeFirstYield` chunks before first output:

```ts
const gate = createStreamingGate({
  maxChunksBeforeFirstYield: 2,
  chunkCount: chunks.length,
});

const output = gate.wrapOutput(student.encodeIterable(gate.wrapInput(chunks)));
const first = output[Symbol.iterator]().next();
```
