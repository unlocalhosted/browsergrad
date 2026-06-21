# @unlocalhosted/browsergrad-snapshots

Browser-safe snapshot comparison oracles for BrowserGrad rubrics.

Use this package when a lab has small expected outputs checked in as JSON:
loss scalars, logits, masks, event traces, dedupe decisions, or alignment math
snapshots. It is dependency-free and works in JS rubrics or as a narrow oracle
registered into Pyodide.

## Public Surface

```ts
import { compareSnapshot, createSnapshotOracle } from "@unlocalhosted/browsergrad-snapshots";

const oracle = createSnapshotOracle(
  { loss: 1.25, logits: [0.1, 0.2] },
  { absoluteTolerance: 1e-4 },
);

const result = oracle.compare({ loss: 1.25001, logits: [0.1, 0.20002] });
console.log(result.ok, result.mismatches);

const direct = compareSnapshot({ score: 0.9 }, { score: 1.0 }, {
  absoluteTolerance: 0.05,
});
console.log(direct.mismatches);
```
