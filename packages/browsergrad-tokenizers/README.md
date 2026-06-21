# @unlocalhosted/browsergrad-tokenizers

Browser-safe tokenizer and byte-level BPE primitives for BrowserGrad.

This package is intentionally independent of Pyodide. Call it directly from
JS/TS workers, or expose narrow adapters to Python runtimes when needed.

## Public Surface

```ts
import {
  trainByteBpe,
  encodeByteBpe,
  decodeByteBpe,
  createByteBpeOracle,
  createStreamingGate,
} from "@unlocalhosted/browsergrad-tokenizers";
```

The default byte-BPE oracle uses GPT-2 pre-tokenization, byte-level initial
vocabulary, special-token hard boundaries, no cross-pretoken merges, and
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
