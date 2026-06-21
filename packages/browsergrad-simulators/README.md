# @unlocalhosted/browsergrad-simulators

Deterministic browser-safe simulators for BrowserGrad systems labs.

This package is intentionally small and dependency-free. It models event traces
and collective behavior for course rubrics that need distributed or parallel
systems concepts without depending on native threads, CUDA, MPI, or Linux
process behavior.

## Public Surface

```ts
import { createDeterministicMesh } from "@unlocalhosted/browsergrad-simulators";

const mesh = createDeterministicMesh({ ranks: 4 });
const reduced = mesh.allReduce({
  tag: "grad-bucket-0",
  values: [1, 2, 3, 4],
  op: "sum",
});

console.log(reduced); // [10, 10, 10, 10]
console.log(mesh.trace());
```

Use it as a rubric oracle for DDP/FSDP/task-system labs when the learning goal
is ordering, participation, and reduction semantics rather than native
performance.
