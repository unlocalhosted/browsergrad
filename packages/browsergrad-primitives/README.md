# @unlocalhosted/browsergrad-primitives

Generic browser-safe ML primitives for BrowserGrad.

This is the canonical package for small reusable helpers that are useful beyond
one lab, course, or platform: byte-BPE text processing, data cleaning,
evaluation comparators, deterministic simulations, hosted-training fixtures,
and RL/alignment math. Course profiles may wrap these helpers as rubrics or
oracles, but this package keeps the public interface assignment-agnostic.

## Public Surface

```ts
import {
  createByteBpeReference,
  createHostedTrainingApiFixture,
  createSnapshotComparator,
  data,
  rl,
  simulation,
} from "@unlocalhosted/browsergrad-primitives";
```

Subpath imports are available when bundle policy wants one domain:

```ts
import { createStreamingGate } from "@unlocalhosted/browsergrad-primitives/text";
import { compareSnapshot } from "@unlocalhosted/browsergrad-primitives/evaluation";
import { createDeterministicMesh } from "@unlocalhosted/browsergrad-primitives/simulation";
```

Use `reference`, `comparator`, `fixture`, and `simulator` language in package
code. Assignment profiles may expose these helpers as profile-local oracles.
